#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from discreteNPIV import estimate_long_term_effect_from_surrogates


THETA = np.array([0.6, -0.5, 0.35, 0.15, -0.10, 0.25, 0.20], dtype=float)


def _basis_features(raw_surrogates: np.ndarray) -> np.ndarray:
    s1 = raw_surrogates[:, 0]
    s2 = raw_surrogates[:, 1]
    return np.column_stack(
        [
            s1,
            s2,
            s1 * s2,
            s1**2,
            s2**2,
            np.sin(s1),
            np.cos(s2),
        ]
    )


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    truth = np.asarray([row["truth"] for row in rows], dtype=float)
    estimate = np.asarray([row["estimate"] for row in rows], dtype=float)
    se = np.asarray([row["se"] for row in rows], dtype=float)
    coverage = np.asarray([row["covered"] for row in rows], dtype=float)
    errors = estimate - truth
    empirical_sd = float(np.std(errors, ddof=1)) if errors.shape[0] > 1 else 0.0
    mean_se = float(np.mean(se))
    ratio = None if empirical_sd == 0.0 else float(mean_se / empirical_sd)
    method_counts: dict[str, int] = {}
    for row in rows:
        method = str(row["method"])
        method_counts[method] = method_counts.get(method, 0) + 1
    return {
        "n_replications": len(rows),
        "coverage": float(np.mean(coverage)),
        "mean_bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "empirical_sd": empirical_sd,
        "mean_se": mean_se,
        "mean_se_over_empirical_sd": ratio,
        "method_counts": dict(sorted(method_counts.items())),
    }


def _run_single_replication(
    *,
    seed: int,
    n_experiments: int,
    n_per_arm: int,
    shift_loc: float,
    target_shift_scale: float,
    base_scale: float,
    confounder_scale: float,
    raw_noise_scale: float,
    outcome_noise_scale: float,
    confounder_loading: float,
    n_target_samples: int,
    selection: str,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    base_shifts = rng.normal(scale=base_scale, size=(n_experiments, 2))
    treatment_shifts = rng.normal(loc=shift_loc, scale=0.025, size=(n_experiments, 2))

    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    historical_experiment_ids: list[str] = []
    historical_arm_labels: list[str] = []
    for experiment_idx in range(n_experiments):
        for arm_label, delta in (("control", np.zeros(2)), ("treatment", treatment_shifts[experiment_idx])):
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=confounder_scale)
                raw = (
                    base_shifts[experiment_idx]
                    + delta
                    + np.array([confounder, 0.55 * confounder])
                    + rng.normal(scale=raw_noise_scale, size=2)
                )
                features = _basis_features(raw[None, :])[0]
                outcome = float(features @ THETA + confounder_loading * confounder + rng.normal(scale=outcome_noise_scale))
                X_hist_rows.append(features)
                Y_hist_rows.append(outcome)
                historical_experiment_ids.append(f"exp_{experiment_idx}")
                historical_arm_labels.append(arm_label)

    X_hist = np.asarray(X_hist_rows, dtype=float)
    Y_hist = np.asarray(Y_hist_rows, dtype=float)
    X_new_control_raw = rng.normal(
        loc=np.mean(base_shifts, axis=0),
        scale=raw_noise_scale + 0.02,
        size=(n_target_samples, 2),
    )
    X_new_treated_raw = rng.normal(
        loc=np.mean(base_shifts + target_shift_scale * treatment_shifts, axis=0),
        scale=raw_noise_scale + 0.02,
        size=(n_target_samples, 2),
    )
    X_new_control = _basis_features(X_new_control_raw)
    X_new_treated = _basis_features(X_new_treated_raw)
    truth = float(np.mean(X_new_treated @ THETA) - np.mean(X_new_control @ THETA))

    result = estimate_long_term_effect_from_surrogates(
        X_hist=X_hist,
        Y_hist=Y_hist,
        historical_arms=np.asarray(historical_arm_labels, dtype=object),
        historical_experiment_ids=np.asarray(historical_experiment_ids, dtype=object),
        X_new_treated=X_new_treated,
        X_new_control=X_new_control,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2, 1e-1]),
        gamma_grid=np.array([0.0, 1e-6]),
        selection=selection,
        random_state=seed,
    )
    return {
        "truth": truth,
        "estimate": float(result.selected.estimate),
        "se": float(result.selected.se),
        "covered": bool(result.selected.ci_lower <= truth <= result.selected.ci_upper),
        "method": result.selected.method_name,
    }


def _run_overlap_replication(
    *,
    seed: int,
    n_overlap_experiments: int,
    n_per_cell: int,
    shift_loc: float,
    target_shift_scale: float,
    confounder_scale: float,
    raw_noise_scale: float,
    outcome_noise_scale: float,
    confounder_loading: float,
    n_target_samples: int,
    selection: str,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    experiment_names = [f"exp_{idx + 1}" for idx in range(n_overlap_experiments)]
    treatment_shifts = {
        name: rng.normal(loc=shift_loc, scale=0.015, size=2) for name in experiment_names
    }

    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    historical_overlaps: list[list[tuple[str, str]]] = []
    for bits in itertools.product([0, 1], repeat=n_overlap_experiments):
        overlap_level: list[tuple[str, str]] = []
        mean_shift = np.zeros(2)
        for experiment_name, bit in zip(experiment_names, bits, strict=True):
            arm_label = "treatment" if bit else "control"
            overlap_level.append((experiment_name, arm_label))
            if bit:
                mean_shift = mean_shift + treatment_shifts[experiment_name]

        for _ in range(n_per_cell):
            confounder = rng.normal(scale=confounder_scale)
            raw = mean_shift + np.array([confounder, 0.55 * confounder]) + rng.normal(scale=raw_noise_scale, size=2)
            features = _basis_features(raw[None, :])[0]
            outcome = float(features @ THETA + confounder_loading * confounder + rng.normal(scale=outcome_noise_scale))
            X_hist_rows.append(features)
            Y_hist_rows.append(outcome)
            historical_overlaps.append(overlap_level)

    X_hist = np.asarray(X_hist_rows, dtype=float)
    Y_hist = np.asarray(Y_hist_rows, dtype=float)
    X_new_control_raw = rng.normal(loc=np.zeros(2), scale=raw_noise_scale + 0.02, size=(n_target_samples, 2))
    X_new_treated_raw = rng.normal(
        loc=target_shift_scale * np.sum(np.vstack([treatment_shifts[name] for name in experiment_names]), axis=0),
        scale=raw_noise_scale + 0.02,
        size=(n_target_samples, 2),
    )
    X_new_control = _basis_features(X_new_control_raw)
    X_new_treated = _basis_features(X_new_treated_raw)
    truth = float(np.mean(X_new_treated @ THETA) - np.mean(X_new_control @ THETA))

    result = estimate_long_term_effect_from_surrogates(
        X_hist=X_hist,
        Y_hist=Y_hist,
        historical_arms=historical_overlaps,
        X_new_treated=X_new_treated,
        X_new_control=X_new_control,
        encoding_mode="overlap",
        allow_empty=False,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2, 1e-1]),
        gamma_grid=np.array([0.0, 1e-6]),
        selection=selection,
        random_state=seed,
    )
    return {
        "truth": truth,
        "estimate": float(result.selected.estimate),
        "se": float(result.selected.se),
        "covered": bool(result.selected.ci_lower <= truth <= result.selected.ci_upper),
        "method": result.selected.method_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate nonlinear surrogate coverage under basis-feature DGPs.")
    parser.add_argument("--replications", type=int, default=20)
    args = parser.parse_args()

    scenarios = [
        {
            "name": "single_tight_coverage",
            "kind": "single",
            "params": {
                "n_experiments": 14,
                "n_per_arm": 280,
                "shift_loc": 0.22,
                "target_shift_scale": 0.72,
                "base_scale": 0.23,
                "confounder_scale": 0.26,
                "raw_noise_scale": 0.08,
                "outcome_noise_scale": 0.06,
                "confounder_loading": 0.16,
                "n_target_samples": 10_000,
                "selection": "2sls",
            },
        },
        {
            "name": "single_conservative_large_sample",
            "kind": "single",
            "params": {
                "n_experiments": 16,
                "n_per_arm": 500,
                "shift_loc": 0.24,
                "target_shift_scale": 0.75,
                "base_scale": 0.22,
                "confounder_scale": 0.25,
                "raw_noise_scale": 0.08,
                "outcome_noise_scale": 0.06,
                "confounder_loading": 0.15,
                "n_target_samples": 12_000,
                "selection": "2sls",
            },
        },
        {
            "name": "overlap_undercovered",
            "kind": "overlap",
            "params": {
                "n_overlap_experiments": 4,
                "n_per_cell": 320,
                "shift_loc": 0.18,
                "target_shift_scale": 0.72,
                "confounder_scale": 0.25,
                "raw_noise_scale": 0.08,
                "outcome_noise_scale": 0.06,
                "confounder_loading": 0.16,
                "n_target_samples": 10_000,
                "selection": "2sls",
            },
        },
    ]

    results: list[dict[str, object]] = []
    for scenario in scenarios:
        rows = []
        for seed in range(args.replications):
            if scenario["kind"] == "single":
                rows.append(_run_single_replication(seed=seed, **scenario["params"]))
            else:
                rows.append(_run_overlap_replication(seed=seed, **scenario["params"]))
        results.append(
            {
                "name": scenario["name"],
                "kind": scenario["kind"],
                "params": scenario["params"],
                "summary": _summarize(rows),
            }
        )

    print(json.dumps({"replications": args.replications, "scenarios": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
