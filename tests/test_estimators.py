import unittest

import numpy as np

from discreteNPIV import estimate_average_functional, fit_dual_nuisance, fit_structural_nuisance
from discreteNPIV.simulation import generate_synthetic_data


class EstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = generate_synthetic_data(
            n_per_instrument=12,
            n_instruments=4,
            n_features=3,
            n_target_samples=500,
            random_state=4,
        )

    def test_fit_structural_nuisance_shapes(self) -> None:
        fit = fit_structural_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            n_splits=2,
            lambda_grid=np.array([1e-1, 1e-3]),
            gamma_grid=np.array([0.0, 1e-4]),
            random_state=0,
        )
        self.assertEqual(fit.coef_selected.shape, (3,))
        self.assertIn(fit.selected_method, {"npjive", "2sls"})

    def test_fit_dual_nuisance_accepts_different_x_new_length(self) -> None:
        fit = fit_dual_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["X_new"][:123],
            n_splits=2,
            lambda_grid=np.array([1e-1, 1e-3]),
            gamma_grid=np.array([0.0]),
            random_state=0,
        )
        self.assertEqual(fit.coef_selected.shape, (3,))

    def test_end_to_end_inference_returns_finite_outputs(self) -> None:
        result = estimate_average_functional(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            self.data["X_new"],
            n_splits=2,
            lambda_grid=np.array([1e-1, 1e-3]),
            gamma_grid=np.array([0.0, 1e-4]),
            random_state=0,
        )
        self.assertTrue(np.isfinite(result.selected.estimate))
        self.assertTrue(np.isfinite(result.selected.se))
        self.assertEqual(result.selected.influence_function.shape[0], self.data["X"].shape[0])

    def test_large_penalty_shrinks_coefficients(self) -> None:
        low_penalty = fit_structural_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            n_splits=2,
            lambda_grid=np.array([1e-8]),
            gamma_grid=np.array([0.0]),
            selection="npjive",
            random_state=0,
        )
        high_penalty = fit_structural_nuisance(
            self.data["X"],
            self.data["Z"],
            self.data["Y"],
            n_splits=2,
            lambda_grid=np.array([1e2]),
            gamma_grid=np.array([0.0]),
            selection="npjive",
            random_state=0,
        )
        self.assertLess(np.linalg.norm(high_penalty.coef_selected), np.linalg.norm(low_penalty.coef_selected))


if __name__ == "__main__":
    unittest.main()

