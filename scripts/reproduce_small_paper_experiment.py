#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from discreteNPIV.reproduction import run_small_paper_experiment, summarize_legacy_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact paper-style simulation and optionally compare it to an archived legacy result file."
    )
    parser.add_argument("--replications", type=int, default=20, help="Number of simulation replications.")
    parser.add_argument("--n-per-instrument", type=int, default=30, help="Observations per instrument level.")
    parser.add_argument("--n-instruments", type=int, default=50, help="Number of discrete instrument levels.")
    parser.add_argument("--n-features", type=int, default=18, help="Number of basis features.")
    parser.add_argument("--n-target-samples", type=int, default=4000, help="Number of target-distribution samples.")
    parser.add_argument("--design-seed", type=int, default=123, help="Seed for the fixed design parameters.")
    parser.add_argument("--noise-seed-start", type=int, default=0, help="First seed used for replication noise.")
    parser.add_argument("--n-splits", type=int, default=2, help="Number of tuning folds.")
    parser.add_argument("--selection", type=str, default="adaptive", choices=["adaptive", "npjive", "2sls"])
    parser.add_argument(
        "--legacy-path",
        type=Path,
        default=None,
        help="Optional archived legacy .pkl file to summarize alongside the new simulation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_small_paper_experiment(
        n_replications=args.replications,
        n_per_instrument=args.n_per_instrument,
        n_instruments=args.n_instruments,
        n_features=args.n_features,
        n_target_samples=args.n_target_samples,
        design_seed=args.design_seed,
        noise_seed_start=args.noise_seed_start,
        n_splits=args.n_splits,
        selection=args.selection,
        lambda_grid=np.array([1e-1, 1e-3, 1e-5]),
        gamma_grid=np.array([1e-2, 1e-4, 0.0]),
    )

    payload: dict[str, object] = {"new_experiment": summary.to_dict()}
    if args.legacy_path is not None:
        payload["legacy_archive"] = summarize_legacy_archive(args.legacy_path).to_dict()

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

