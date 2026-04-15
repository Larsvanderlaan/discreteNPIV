import unittest
from pathlib import Path

import numpy as np

from discreteNPIV import (
    estimate_average_functional,
    fit_dual_nuisance,
    fit_structural_nuisance,
    summarize_legacy_archive,
)
from discreteNPIV.grouping import group_means
from discreteNPIV.reproduction import run_small_paper_experiment
from discreteNPIV.simulation import generate_synthetic_data


def _legacy_2sls_structural_coef(X: np.ndarray, Z: np.ndarray, Y: np.ndarray, lambda_value: float, gamma: float) -> np.ndarray:
    counts = np.bincount(Z)
    weights = counts / counts.sum()
    group_X = group_means(X, Z, n_groups=counts.shape[0])
    group_Y = group_means(Y, Z, n_groups=counts.shape[0])
    quadratic = (group_X.T * weights) @ group_X
    linear = (group_X.T * weights) @ group_Y
    penalty = X.T @ X / X.shape[0]
    system = quadratic + lambda_value * penalty + gamma * np.eye(X.shape[1])
    return np.linalg.solve(0.5 * (system + system.T), linear)


def _legacy_2sls_dual_coef(X: np.ndarray, Z: np.ndarray, X_new: np.ndarray, lambda_value: float, gamma: float) -> np.ndarray:
    counts = np.bincount(Z)
    weights = counts / counts.sum()
    group_X = group_means(X, Z, n_groups=counts.shape[0])
    quadratic = (group_X.T * weights) @ group_X
    linear = np.mean(X_new, axis=0)
    penalty = X.T @ X / X.shape[0]
    system = quadratic + lambda_value * penalty + gamma * np.eye(X.shape[1])
    return np.linalg.solve(0.5 * (system + system.T), linear)


class LegacyRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = generate_synthetic_data(
            n_per_instrument=10,
            n_instruments=4,
            n_features=3,
            n_target_samples=200,
            random_state=11,
        )

    def test_structural_2sls_matches_legacy_formula(self) -> None:
        lambda_value = 1e-3
        gamma = 1e-4
        fit = fit_structural_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            n_splits=2,
            lambda_grid=np.array([lambda_value]),
            gamma_grid=np.array([gamma]),
            selection="2sls",
            random_state=0,
        )
        expected = _legacy_2sls_structural_coef(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            lambda_value=lambda_value,
            gamma=gamma,
        )
        np.testing.assert_allclose(fit.coef_selected, expected)

    def test_dual_2sls_matches_legacy_formula(self) -> None:
        lambda_value = 1e-3
        gamma = 1e-4
        fit = fit_dual_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["X_new"],
            n_splits=2,
            lambda_grid=np.array([lambda_value]),
            gamma_grid=np.array([gamma]),
            selection="2sls",
            random_state=0,
        )
        expected = _legacy_2sls_dual_coef(
            self.data["X"],
            self.data["Z"],
            self.data["X_new"],
            lambda_value=lambda_value,
            gamma=gamma,
        )
        np.testing.assert_allclose(fit.coef_selected, expected)

    def test_legacy_archive_summary_is_stable(self) -> None:
        archive = summarize_legacy_archive(
            Path("main-depreciated/legacy-results/results_n30_K500_18_ab2_3.5_123_LOO.pkl")
        )
        self.assertAlmostEqual(archive.npjive.mean_estimate, 0.37995778711773054)
        self.assertAlmostEqual(archive.npjive.coverage, 0.873)
        self.assertAlmostEqual(archive.baseline_2sls.mean_estimate, 0.32322606340343013)
        self.assertAlmostEqual(archive.baseline_2sls.coverage, 0.584)
        self.assertIsNotNone(archive.single_split)
        assert archive.single_split is not None
        self.assertAlmostEqual(archive.single_split.mean_estimate, 0.3649762525560808)

    def test_small_paper_experiment_is_reproducible(self) -> None:
        summary_a = run_small_paper_experiment(
            n_replications=4,
            n_per_instrument=8,
            n_instruments=6,
            n_features=4,
            n_target_samples=300,
            design_seed=5,
            noise_seed_start=20,
            lambda_grid=np.array([1e-1, 1e-3]),
            gamma_grid=np.array([1e-2, 0.0]),
        )
        summary_b = run_small_paper_experiment(
            n_replications=4,
            n_per_instrument=8,
            n_instruments=6,
            n_features=4,
            n_target_samples=300,
            design_seed=5,
            noise_seed_start=20,
            lambda_grid=np.array([1e-1, 1e-3]),
            gamma_grid=np.array([1e-2, 0.0]),
        )
        self.assertEqual(summary_a.to_dict(), summary_b.to_dict())
        self.assertEqual(sum(summary_a.selected_method_counts.values()), 4)
        self.assertTrue(np.isfinite(summary_a.selected.mean_estimate))

    def test_selected_2sls_path_returns_baseline_result(self) -> None:
        result = estimate_average_functional(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            self.data["X_new"],
            n_splits=2,
            lambda_grid=np.array([1e-3]),
            gamma_grid=np.array([1e-4]),
            selection="2sls",
            random_state=0,
        )
        self.assertEqual(result.selected.method_name, "2sls+2sls")
        self.assertAlmostEqual(result.selected.estimate, result.baseline_2sls.estimate)
        self.assertAlmostEqual(result.selected.se, result.baseline_2sls.se)
        self.assertAlmostEqual(result.selected.ci_lower, result.baseline_2sls.ci_lower)
        self.assertAlmostEqual(result.selected.ci_upper, result.baseline_2sls.ci_upper)

    def test_selected_npjive_path_returns_npjive_result(self) -> None:
        result = estimate_average_functional(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            self.data["X_new"],
            n_splits=2,
            lambda_grid=np.array([1e-3]),
            gamma_grid=np.array([1e-4]),
            selection="npjive",
            random_state=0,
        )
        self.assertEqual(result.selected.method_name, "npjive+npjive")
        self.assertAlmostEqual(result.selected.estimate, result.npjive.estimate)
        self.assertAlmostEqual(result.selected.se, result.npjive.se)
        self.assertAlmostEqual(result.selected.ci_lower, result.npjive.ci_lower)
        self.assertAlmostEqual(result.selected.ci_upper, result.npjive.ci_upper)


if __name__ == "__main__":
    unittest.main()
