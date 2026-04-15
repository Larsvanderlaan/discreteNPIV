#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from discreteNPIV import estimate_average_functional, estimate_long_term_effect_from_surrogates, fit_structural_nuisance
from discreteNPIV.simulation import generate_synthetic_data


NONLINEAR_THETA = np.array([0.6, -0.5, 0.35, 0.15, -0.10, 0.25, 0.20], dtype=float)


def _basis_features(raw: np.ndarray) -> np.ndarray:
    s1 = raw[:, 0]
    s2 = raw[:, 1]
    return np.column_stack([s1, s2, s1 * s2, s1**2, s2**2, np.sin(s1), np.cos(s2)])


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    truth = np.asarray([row["truth"] for row in rows], dtype=float)
    estimate = np.asarray([row["estimate"] for row in rows], dtype=float)
    structural_rmse = np.asarray([row["structural_rmse"] for row in rows], dtype=float)
    se = np.asarray([row["se"] for row in rows], dtype=float)
    covered = np.asarray([row["covered"] for row in rows], dtype=float)
    errors = estimate - truth
    empirical_sd = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0
    mean_se = float(np.mean(se))
    ratio = None if empirical_sd == 0.0 else float(mean_se / empirical_sd)
    return {
        "mean_structural_rmse": float(np.mean(structural_rmse)),
        "mean_bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "coverage": float(np.mean(covered)),
        "empirical_sd": empirical_sd,
        "mean_se": mean_se,
        "mean_se_over_empirical_sd": ratio,
    }


def _run_linear_replication(
    *,
    seed: int,
    n_per_instrument: int,
    n_instruments: int,
) -> dict[str, object]:
    data = generate_synthetic_data(
        n_per_instrument=n_per_instrument,
        n_instruments=n_instruments,
        n_features=8,
        n_target_samples=3_000,
        sigma_x=0.1,
        sigma_y=0.1,
        sigma_c=0.1,
        sparsity_structural=3.0,
        sparsity_target=3.5,
        random_state=seed,
    )
    fit = fit_structural_nuisance(
        data["X"],
        data["Z"],
        data["Y"],
        n_splits=2,
        lambda_grid=np.array([1e-1, 1e-3, 1e-5]),
        gamma_grid=np.array([1e-2, 1e-4, 0.0]),
        selection="npjive",
        random_state=seed,
    )
    structural_rmse = float(np.sqrt(np.mean((data["X_new"] @ fit.coef_selected - data["X_new"] @ data["theta"]) ** 2)))
    result = estimate_average_functional(
        data["X"],
        data["Z"],
        data["Y"],
        data["X_new"],
        n_splits=2,
        lambda_grid=np.array([1e-1, 1e-3, 1e-5]),
        gamma_grid=np.array([1e-2, 1e-4, 0.0]),
        selection="npjive",
        random_state=seed,
    )
    truth = float(np.mean(data["X_new"] @ data["theta"]))
    return {
        "truth": truth,
        "estimate": float(result.selected.estimate),
        "structural_rmse": structural_rmse,
        "se": float(result.selected.se),
        "covered": bool(result.selected.ci_lower <= truth <= result.selected.ci_upper),
    }


def _run_nonlinear_replication(
    *,
    seed: int,
    n_experiments: int,
    n_per_arm: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    base_shifts = rng.normal(scale=0.25, size=(n_experiments, 2))
    treatment_shifts = rng.normal(loc=0.20, scale=0.03, size=(n_experiments, 2))

    X_rows = []
    Y_rows = []
    Z_rows = []
    for experiment_idx in range(n_experiments):
        for arm_idx, arm_shift in enumerate((np.zeros(2), treatment_shifts[experiment_idx])):
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=0.28)
                raw = base_shifts[experiment_idx] + arm_shift + np.array([confounder, 0.55 * confounder]) + rng.normal(
                    scale=0.08, size=2
                )
                features = _basis_features(raw[None, :])[0]
                outcome = float(features @ NONLINEAR_THETA + 0.16 * confounder + rng.normal(scale=0.06))
                X_rows.append(features)
                Y_rows.append(outcome)
                Z_rows.append(2 * experiment_idx + arm_idx)

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)
    Z = np.asarray(Z_rows, dtype=int)
    X_new_raw = rng.normal(
        loc=np.mean(base_shifts + 0.72 * treatment_shifts, axis=0),
        scale=0.10,
        size=(6_000, 2),
    )
    X_new = _basis_features(X_new_raw)

    fit = fit_structural_nuisance(
        X,
        Z,
        Y,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2, 1e-1]),
        gamma_grid=np.array([0.0, 1e-6]),
        selection="npjive",
        random_state=seed,
    )
    structural_rmse = float(np.sqrt(np.mean((X_new @ fit.coef_selected - X_new @ NONLINEAR_THETA) ** 2)))
    result = estimate_average_functional(
        X,
        Z,
        Y,
        X_new,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2, 1e-1]),
        gamma_grid=np.array([0.0, 1e-6]),
        selection="npjive",
        random_state=seed,
    )
    truth = float(np.mean(X_new @ NONLINEAR_THETA))
    return {
        "truth": truth,
        "estimate": float(result.selected.estimate),
        "structural_rmse": structural_rmse,
        "se": float(result.selected.se),
        "covered": bool(result.selected.ci_lower <= truth <= result.selected.ci_upper),
    }


def _run_overlap_replication(*, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    experiment_names = [f"exp_{idx + 1}" for idx in range(4)]
    treatment_shifts = {name: rng.normal(loc=0.16, scale=0.015, size=2) for name in experiment_names}

    X_hist_rows = []
    Y_hist_rows = []
    historical_overlaps = []
    for bits in itertools.product([0, 1], repeat=4):
        overlap_level = []
        mean_shift = np.zeros(2)
        for experiment_name, bit in zip(experiment_names, bits, strict=True):
            arm_label = "treatment" if bit else "control"
            overlap_level.append((experiment_name, arm_label))
            if bit:
                mean_shift = mean_shift + treatment_shifts[experiment_name]
        for _ in range(220):
            confounder = rng.normal(scale=0.25)
            raw = mean_shift + np.array([confounder, 0.55 * confounder]) + rng.normal(scale=0.08, size=2)
            features = _basis_features(raw[None, :])[0]
            outcome = float(features @ NONLINEAR_THETA + 0.16 * confounder + rng.normal(scale=0.06))
            X_hist_rows.append(features)
            Y_hist_rows.append(outcome)
            historical_overlaps.append(overlap_level)

    X_hist = np.asarray(X_hist_rows, dtype=float)
    Y_hist = np.asarray(Y_hist_rows, dtype=float)
    X_new_control = _basis_features(rng.normal(loc=np.zeros(2), scale=0.10, size=(6_000, 2)))
    X_new_treated = _basis_features(
        rng.normal(
            loc=0.72 * np.sum(np.vstack([treatment_shifts[name] for name in experiment_names]), axis=0),
            scale=0.10,
            size=(6_000, 2),
        )
    )

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
        selection="npjive",
        random_state=seed,
    )
    truth = float(np.mean(X_new_treated @ NONLINEAR_THETA) - np.mean(X_new_control @ NONLINEAR_THETA))
    return {
        "truth": truth,
        "estimate": float(result.selected.estimate),
        "structural_rmse": float("nan"),
        "se": float(result.selected.se),
        "covered": bool(result.selected.ci_lower <= truth <= result.selected.ci_upper),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forced npJIVE recovery and Wald coverage in validation scenarios.")
    parser.add_argument("--replications", type=int, default=6)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help=(
            "Optional subset of scenario names to run. Choices: "
            "linear_weak_disjoint, linear_medium_disjoint, linear_strong_disjoint, "
            "nonlinear_weak_disjoint, nonlinear_medium_disjoint, nonlinear_strong_disjoint."
        ),
    )
    parser.add_argument("--include-overlap", action="store_true")
    args = parser.parse_args()

    scenarios = [
        ("linear_weak_disjoint", lambda seed: _run_linear_replication(seed=seed, n_per_instrument=8, n_instruments=20)),
        ("linear_medium_disjoint", lambda seed: _run_linear_replication(seed=seed, n_per_instrument=8, n_instruments=60)),
        ("linear_strong_disjoint", lambda seed: _run_linear_replication(seed=seed, n_per_instrument=20, n_instruments=60)),
        ("nonlinear_weak_disjoint", lambda seed: _run_nonlinear_replication(seed=seed, n_experiments=8, n_per_arm=80)),
        ("nonlinear_medium_disjoint", lambda seed: _run_nonlinear_replication(seed=seed, n_experiments=12, n_per_arm=180)),
        ("nonlinear_strong_disjoint", lambda seed: _run_nonlinear_replication(seed=seed, n_experiments=14, n_per_arm=280)),
    ]
    if args.scenarios:
        available = {name for name, _ in scenarios}
        requested = set(args.scenarios)
        unknown = requested - available
        if unknown:
            raise SystemExit(f"Unknown scenario names: {sorted(unknown)}")
        scenarios = [(name, runner) for name, runner in scenarios if name in requested]

    payload: dict[str, object] = {"ci_method": "wald", "replications": args.replications, "scenarios": {}}
    for name, runner in scenarios:
        rows = [runner(seed) for seed in range(args.replications)]
        payload["scenarios"][name] = _summarize(rows)

    if args.include_overlap:
        overlap_rows = [_run_overlap_replication(seed=seed) for seed in range(args.replications)]
        payload["overlap_secondary"] = {
            "note": "Exact overlap interaction groups remain a secondary stress test and are not yet validated to the same standard as disjoint-arm designs.",
            "summary": _summarize(overlap_rows),
        }

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
