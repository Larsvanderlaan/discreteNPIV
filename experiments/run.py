#!/usr/bin/env python3
"""Run the simulation configurations reported in the paper."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

from experiments.paper import (  # noqa: E402
    LongTermSurrogateConfig,
    config_hash,
    run_provenance,
    run_replication,
    summarize_rows,
    write_rows_csv,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "main": ROOT / "config" / "main_loo.json",
    "large_n": ROOT / "config" / "large_n_loo.json",
}
REPORTED_METHODS = {
    "main": {
        "2sls_dml",
        "loo_npjive",
        "loo_npjive_star",
    },
    "large_n": {
        "loo_npjive",
        "loo_npjive_star",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(CONFIG_PATHS), default="main")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Override the frozen replication count; intended only for smoke tests.",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=None,
        help="Run only the named cell. May be supplied more than once.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", choices=("validation", "confirmatory"), default="confirmatory")
    return parser.parse_args()


def load_configs(path: Path) -> list[LongTermSurrogateConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["configs"] if isinstance(payload, dict) else payload
    field_names = {field.name for field in fields(LongTermSurrogateConfig)}
    configs = [
        LongTermSurrogateConfig(
            **{name: value for name, value in item.items() if name in field_names}
        )
        for item in items
    ]
    for config, item in zip(configs, items, strict=True):
        expected_hash = item.get("config_hash")
        if expected_hash is not None and config_hash(config) != expected_hash:
            raise ValueError(f"Frozen configuration hash mismatch for {config.cell_id}.")
    return configs


def _run_one(task: tuple[LongTermSurrogateConfig, int, dict[str, Any]]) -> list[dict[str, Any]]:
    config, replication, provenance = task
    return run_replication(
        config,
        stage=str(provenance["stage"]),
        replication=replication,
        provenance=provenance,
    )


def run_config(
    config: LongTermSurrogateConfig,
    *,
    output_dir: Path,
    workers: int,
    provenance: dict[str, Any],
    reported_methods: set[str],
    resume: bool,
) -> list[dict[str, Any]]:
    part_path = output_dir / "parts" / f"{config.cell_id}.csv"
    checkpoint_dir = output_dir / "checkpoints" / config.cell_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    for path in sorted(checkpoint_dir.glob("*.json")):
        if not resume:
            raise ValueError(f"Existing checkpoints in {checkpoint_dir}; use --resume or a new output directory.")
        payload = json.loads(path.read_text())
        if payload["config_hash"] != config_hash(config) or payload["source_snapshot_id"] != provenance["source_snapshot_id"]:
            raise ValueError(f"Checkpoint does not match the current configuration and source: {path}.")
        replication = int(payload["replication"])
        if not 0 <= replication < config.n_replications:
            raise ValueError(f"Checkpoint replication is outside the configured range: {path}.")
        completed.add(replication)
        rows.extend(payload["rows"])

    pending = [replication for replication in range(config.n_replications) if replication not in completed]
    tasks = [(config, replication, provenance) for replication in pending]
    if workers == 1:
        nested_rows = map(_run_one, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        nested_rows = executor.map(_run_one, tasks)

    try:
        for position, (replication, replication_rows) in enumerate(zip(pending, nested_rows, strict=True), 1):
            kept = [row for row in replication_rows if str(row.get("row_type")) == "failure" or str(row.get("method")) in reported_methods]
            payload = {
                "config_hash": config_hash(config),
                "source_snapshot_id": provenance["source_snapshot_id"],
                "replication": replication,
                "rows": kept,
            }
            path = checkpoint_dir / f"{replication:05d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, default=lambda value: value.item()) + "\n")
            temporary.replace(path)
            rows.extend(kept)
            done = len(completed) + position
            if done == 1 or done % 25 == 0 or done == config.n_replications:
                print(f"{config.cell_id}: {done}/{config.n_replications}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()

    rows.sort(key=lambda row: (int(row["replication"]), str(row["method"])))
    write_rows_csv(part_path, rows)
    return rows


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    if args.replications is not None and args.replications < 1:
        raise SystemExit("--replications must be positive.")

    configs = load_configs(CONFIG_PATHS[args.study] if args.config is None else args.config)
    if args.cell:
        requested = set(args.cell)
        configs = [config for config in configs if config.cell_id in requested]
        missing = requested - {config.cell_id for config in configs}
        if missing:
            raise SystemExit(f"Unknown cell(s): {', '.join(sorted(missing))}")
    if args.replications is not None:
        configs = [replace(config, n_replications=args.replications) for config in configs]

    output_dir = args.output_dir / args.study
    (output_dir / "parts").mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(command=" ".join(sys.argv))
    provenance["study"] = args.study
    provenance["stage"] = args.stage

    rows: list[dict[str, Any]] = []
    for config in configs:
        rows.extend(
            run_config(
                config,
                output_dir=output_dir,
                workers=args.workers,
                provenance=provenance,
                reported_methods=REPORTED_METHODS[args.study],
                resume=args.resume,
            )
        )

    rows = [
        row
        for row in rows
        if str(row.get("row_type")) == "failure"
        or str(row.get("method")) in REPORTED_METHODS[args.study]
    ]

    write_rows_csv(output_dir / "replications.csv", rows)
    write_rows_csv(output_dir / "summary.csv", summarize_rows(rows))
    failures = sum(str(row.get("row_type")) == "failure" for row in rows)
    print(
        json.dumps(
            {
                "cells": len(configs),
                "failures": failures,
                "output_dir": str(output_dir),
                "replications": sum(config.n_replications for config in configs),
                "study": args.study,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
