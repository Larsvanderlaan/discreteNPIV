"""Compact row serialization and paper-level simulation summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .model import _target_estimand_for_method


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (tuple, list, dict)):
        return repr(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous replication rows with a deterministic schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else np.nan


def _mc_se(coverage: float, count: int) -> float:
    if count == 0 or not np.isfinite(coverage):
        return np.nan
    return float(np.sqrt(max(coverage * (1.0 - coverage), 0.0) / count))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the bias, RMSE, coverage, and SE calibration reported in the paper."""

    estimate_rows = [row for row in rows if row.get("row_type") == "estimate"]
    failure_rows = [row for row in rows if row.get("row_type") == "failure"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("stage", "cell_id", "config_hash", "regime", "method")
    for row in estimate_rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        errors = _finite_values(group_rows, "error")
        projection_errors = _finite_values(group_rows, "projection_error")
        estimand_errors = _finite_values(group_rows, "estimand_error")
        ses = _finite_values(group_rows, "se")
        covered = _finite_values(group_rows, "covered")
        projection_covered = _finite_values(group_rows, "projection_covered")
        estimand_covered = _finite_values(group_rows, "estimand_covered")
        ci_lengths = _finite_values(group_rows, "ci_length")
        identification_gaps = _finite_values(group_rows, "identification_gap")
        normalized_gaps = _finite_values(group_rows, "normalized_identification_gap")
        runtimes = _finite_values(group_rows, "runtime_seconds")

        target_estimand = str(
            first.get("target_estimand")
            or _target_estimand_for_method(str(first.get("method", "")))
        )
        empirical_sd = float(np.std(errors, ddof=1)) if errors.size > 1 else np.nan
        estimand_sd = (
            float(np.std(estimand_errors, ddof=1))
            if estimand_errors.size > 1
            else np.nan
        )
        mean_se = _mean(ses)
        coverage = _mean(covered)
        projection_coverage = _mean(projection_covered)
        estimand_coverage = _mean(estimand_covered)

        stage, cell_id, config_hash, regime, method = group_key
        failed = sum(
            row.get("stage") == stage
            and row.get("cell_id") == cell_id
            and row.get("config_hash") == config_hash
            and row.get("regime") == regime
            for row in failure_rows
        )
        completed = len({int(row["replication"]) for row in group_rows})
        summary = {
            "cell_id": cell_id,
            "config_hash": config_hash,
            "growth_path": first.get("growth_path"),
            "regime": regime,
            "method": method,
            "target_estimand": target_estimand,
            "n_instruments": first.get("n_instruments"),
            "n_per_instrument": first.get("n_per_instrument"),
            "n_replications": completed,
            "failed_replications": failed,
            "failure_rate": failed / max(completed + failed, 1),
            "nonfinite_rate": 1.0 - errors.size / max(len(group_rows), 1),
            "bias": _mean(errors),
            "rmse": _rmse(errors),
            "coverage": coverage,
            "coverage_mc_se": _mc_se(coverage, covered.size),
            "empirical_sd": empirical_sd,
            "mean_se": mean_se,
            "mean_ci_length": _mean(ci_lengths),
            "se_over_empirical_sd": (
                np.nan if not np.isfinite(empirical_sd) or empirical_sd == 0 else mean_se / empirical_sd
            ),
            "projection_bias": _mean(projection_errors),
            "projection_rmse": _rmse(projection_errors),
            "projection_coverage": projection_coverage,
            "estimand_bias": _mean(estimand_errors),
            "estimand_rmse": _rmse(estimand_errors),
            "estimand_coverage": estimand_coverage,
            "estimand_coverage_mc_se": _mc_se(estimand_coverage, estimand_covered.size),
            "estimand_empirical_sd": estimand_sd,
            "se_over_estimand_empirical_sd": (
                np.nan if not np.isfinite(estimand_sd) or estimand_sd == 0 else mean_se / estimand_sd
            ),
            "mean_identification_gap": _mean(identification_gaps),
            "mean_normalized_identification_gap": _mean(normalized_gaps),
            "mean_runtime_seconds": _mean(runtimes),
            "selection_rule": first.get("selection_rule"),
            "tuning_stability_rule": first.get("tuning_stability_rule"),
            "projection_estimation_mode": first.get("projection_estimation_mode"),
        }
        summaries.append(summary)
    return summaries
