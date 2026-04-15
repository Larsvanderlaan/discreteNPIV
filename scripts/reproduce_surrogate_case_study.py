#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from discreteNPIV import (
    encode_experiment_arms,
    estimate_long_term_effect_from_surrogates,
    estimate_long_term_mean_from_surrogates,
)


def _make_case_study_data(
    *,
    n_experiments: int,
    n_per_arm: int,
    n_features: int,
    n_target_samples: int,
    random_state: int,
) -> dict[str, object]:
    rng = np.random.default_rng(random_state)

    experiment_ids = [f"exp_{idx + 1}" for idx in range(n_experiments)]
    theta = rng.normal(size=n_features)
    theta /= np.linalg.norm(theta)

    arm_shift: dict[str, np.ndarray] = {}
    for experiment_id in experiment_ids:
        baseline = rng.normal(scale=0.35, size=n_features)
        treatment_lift = rng.normal(loc=0.10, scale=0.12, size=n_features)
        arm_shift[f"{experiment_id}:control"] = baseline
        arm_shift[f"{experiment_id}:treatment"] = baseline + treatment_lift

    historical_experiment_ids: list[str] = []
    historical_arm_labels: list[str] = []
    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    for experiment_id in experiment_ids:
        for arm_label in ("control", "treatment"):
            key = f"{experiment_id}:{arm_label}"
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=0.40)
                surrogate = arm_shift[key] + confounder + rng.normal(scale=0.20, size=n_features)
                outcome = float(surrogate @ theta + 0.45 * confounder + rng.normal(scale=0.20))
                historical_experiment_ids.append(experiment_id)
                historical_arm_labels.append(arm_label)
                X_hist_rows.append(surrogate)
                Y_hist_rows.append(outcome)

    control_centers = np.vstack([arm_shift[f"{experiment_id}:control"] for experiment_id in experiment_ids])
    treatment_centers = np.vstack([arm_shift[f"{experiment_id}:treatment"] for experiment_id in experiment_ids])
    treatment_deltas = treatment_centers - control_centers

    control_mean = np.mean(control_centers, axis=0)
    treated_mean = control_mean + 0.75 * np.mean(treatment_deltas, axis=0)

    X_new_control = rng.normal(loc=control_mean, scale=0.20, size=(n_target_samples, n_features))
    X_new_treated = rng.normal(loc=treated_mean, scale=0.20, size=(n_target_samples, n_features))

    return {
        "theta": theta,
        "historical_experiment_ids": np.asarray(historical_experiment_ids, dtype=object),
        "historical_arm_labels": np.asarray(historical_arm_labels, dtype=object),
        "X_hist": np.asarray(X_hist_rows, dtype=float),
        "Y_hist": np.asarray(Y_hist_rows, dtype=float),
        "X_new_control": X_new_control,
        "X_new_treated": X_new_treated,
        "true_control_mean": float(np.mean(X_new_control @ theta)),
        "true_treated_mean": float(np.mean(X_new_treated @ theta)),
    }


def _format_interval(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a surrogate-based long-term causal inference case study.")
    parser.add_argument("--n-experiments", type=int, default=4)
    parser.add_argument("--n-per-arm", type=int, default=50)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--n-target-samples", type=int, default=800)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    data = _make_case_study_data(
        n_experiments=args.n_experiments,
        n_per_arm=args.n_per_arm,
        n_features=args.n_features,
        n_target_samples=args.n_target_samples,
        random_state=args.seed,
    )

    encoding = encode_experiment_arms(
        assignments=data["historical_arm_labels"],
        experiment_ids=data["historical_experiment_ids"],
        mode="single",
    )

    treated_mean = estimate_long_term_mean_from_surrogates(
        X_hist=data["X_hist"],
        Y_hist=data["Y_hist"],
        historical_arms=data["historical_arm_labels"],
        historical_experiment_ids=data["historical_experiment_ids"],
        X_new=data["X_new_treated"],
        target_name="novel_treated",
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    control_mean = estimate_long_term_mean_from_surrogates(
        X_hist=data["X_hist"],
        Y_hist=data["Y_hist"],
        historical_arms=data["historical_arm_labels"],
        historical_experiment_ids=data["historical_experiment_ids"],
        X_new=data["X_new_control"],
        target_name="novel_control",
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    effect = estimate_long_term_effect_from_surrogates(
        X_hist=data["X_hist"],
        Y_hist=data["Y_hist"],
        historical_arms=data["historical_arm_labels"],
        historical_experiment_ids=data["historical_experiment_ids"],
        X_new_treated=data["X_new_treated"],
        X_new_control=data["X_new_control"],
        treated_name="novel_treated",
        control_name="novel_control",
        effect_name="novel_treated_minus_control",
        n_splits=args.n_splits,
        random_state=args.seed,
    )

    true_treated = float(data["true_treated_mean"])
    true_control = float(data["true_control_mean"])
    true_effect = true_treated - true_control

    print("discreteNPIV surrogate case study")
    print()
    print("Historical design")
    print(f"  rows: {data['X_hist'].shape[0]}")
    print(f"  encoded levels: {encoding.n_levels}")
    print(f"  minimum level count: {encoding.min_count}")
    print("  encoded level counts:")
    for row in encoding.level_table():
        print(f"    code {row['code']}: {row['level']} (count={row['count']})")
    print()
    print("Novel treated arm long-term mean")
    print(
        f"  estimate: {treated_mean.selected.estimate:.3f} "
        f"(SE {treated_mean.selected.se:.3f}, CI {_format_interval(treated_mean.selected.ci_lower, treated_mean.selected.ci_upper)})"
    )
    print(f"  selected method: {treated_mean.selected.method_name}")
    print(f"  latent simulation truth: {true_treated:.3f}")
    print()
    print("Novel control arm long-term mean")
    print(
        f"  estimate: {control_mean.selected.estimate:.3f} "
        f"(SE {control_mean.selected.se:.3f}, CI {_format_interval(control_mean.selected.ci_lower, control_mean.selected.ci_upper)})"
    )
    print(f"  selected method: {control_mean.selected.method_name}")
    print(f"  latent simulation truth: {true_control:.3f}")
    print()
    print("Long-term treatment effect")
    print(
        f"  estimate: {effect.selected.estimate:.3f} "
        f"(SE {effect.selected.se:.3f}, CI {_format_interval(effect.selected.ci_lower, effect.selected.ci_upper)})"
    )
    print(f"  selected method: {effect.selected.method_name}")
    print(f"  latent simulation truth: {true_effect:.3f}")


if __name__ == "__main__":
    main()
