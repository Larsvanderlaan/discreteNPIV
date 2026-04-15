import unittest

import numpy as np

from discreteNPIV import estimate_average_functional, fit_structural_nuisance
from discreteNPIV.simulation import generate_synthetic_data


def _structural_metrics(*, seed: int, n_per_instrument: int, n_instruments: int) -> tuple[float, float]:
    data = generate_synthetic_data(
        n_per_instrument=n_per_instrument,
        n_instruments=n_instruments,
        n_features=8,
        n_target_samples=3000,
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
    return structural_rmse, abs(float(result.selected.estimate - truth))


def _basis_features(raw: np.ndarray) -> np.ndarray:
    s1 = raw[:, 0]
    s2 = raw[:, 1]
    return np.column_stack([s1, s2, s1 * s2, s1**2, s2**2, np.sin(s1), np.cos(s2)])


def _nonlinear_metrics(*, seed: int, n_experiments: int, n_per_arm: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    theta = np.array([0.6, -0.5, 0.35, 0.15, -0.10, 0.25, 0.20])
    base_shifts = rng.normal(scale=0.25, size=(n_experiments, 2))
    treatment_shifts = rng.normal(loc=0.20, scale=0.03, size=(n_experiments, 2))

    X_rows = []
    Y_rows = []
    Z_rows = []
    for experiment_idx in range(n_experiments):
        for arm_shift in (np.zeros(2), treatment_shifts[experiment_idx]):
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=0.28)
                raw = base_shifts[experiment_idx] + arm_shift + np.array([confounder, 0.55 * confounder]) + rng.normal(
                    scale=0.08, size=2
                )
                features = _basis_features(raw[None, :])[0]
                outcome = float(features @ theta + 0.16 * confounder + rng.normal(scale=0.06))
                X_rows.append(features)
                Y_rows.append(outcome)
                Z_rows.append(2 * experiment_idx + int(np.any(arm_shift)))

    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)
    Z = np.asarray(Z_rows)
    X_new_control_raw = rng.normal(loc=np.mean(base_shifts, axis=0), scale=0.10, size=(6_000, 2))
    X_new_treated_raw = rng.normal(loc=np.mean(base_shifts + 0.72 * treatment_shifts, axis=0), scale=0.10, size=(6_000, 2))
    X_new = _basis_features(X_new_treated_raw)
    X_new_control = _basis_features(X_new_control_raw)

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
    structural_rmse = float(np.sqrt(np.mean((X_new @ fit.coef_selected - X_new @ theta) ** 2)))
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
    truth = float(np.mean(X_new @ theta))
    _ = X_new_control
    return structural_rmse, abs(float(result.selected.estimate - truth))


class NPJiveValidationTests(unittest.TestCase):
    def test_linear_npjive_recovery_improves_with_stronger_disjoint_design(self) -> None:
        weak = np.asarray([_structural_metrics(seed=seed, n_per_instrument=8, n_instruments=20) for seed in range(3)])
        strong = np.asarray([_structural_metrics(seed=seed, n_per_instrument=20, n_instruments=60) for seed in range(3)])
        self.assertLess(float(np.mean(strong[:, 0])), float(np.mean(weak[:, 0])))
        self.assertLess(float(np.mean(strong[:, 1])), float(np.mean(weak[:, 1])))

    def test_nonlinear_npjive_recovery_improves_with_stronger_disjoint_design(self) -> None:
        weak = np.asarray([_nonlinear_metrics(seed=seed, n_experiments=8, n_per_arm=80) for seed in range(3)])
        strong = np.asarray([_nonlinear_metrics(seed=seed, n_experiments=14, n_per_arm=280) for seed in range(3)])
        self.assertLess(float(np.mean(strong[:, 0])), float(np.mean(weak[:, 0])))
        self.assertLess(float(np.mean(strong[:, 1])), float(np.mean(weak[:, 1])))


if __name__ == "__main__":
    unittest.main()
