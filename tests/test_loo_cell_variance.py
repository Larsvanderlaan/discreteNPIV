import itertools
import unittest

import numpy as np

from experiments.paper.cell_variance import _loo_kernel_variance, loo_cell_variance


class LOOCellVarianceTests(unittest.TestCase):
    def test_disjoint_four_observation_formula(self) -> None:
        rng = np.random.default_rng(702)
        for n in (4, 5, 7):
            raw = rng.normal(size=(n, n))
            kernel = (raw + raw.T) / 2
            np.fill_diagonal(kernel, 0.0)
            c = np.sum(kernel) / (n * (n - 1))
            products = [kernel[i, j] * kernel[l, m] for i, j, l, m in itertools.permutations(range(n), 4)]
            expected = n * n * (c * c - np.mean(products))
            actual = _loo_kernel_variance(kernel, small_cell_product=np.zeros(n))
            self.assertAlmostEqual(actual, expected, places=12)
            shifted = kernel + 100_000.0
            np.fill_diagonal(shifted, 0.0)
            self.assertAlmostEqual(
                actual,
                _loo_kernel_variance(shifted, small_cell_product=np.zeros(n)),
                places=8,
            )

    def test_pair_kernel_reproduces_cross_fitted_point_estimate(self) -> None:
        rng = np.random.default_rng(317)
        z = np.repeat(np.arange(3), [5, 7, 4])
        folds = np.arange(z.size) % 3
        beta = rng.normal(size=(z.size, 3))
        rho = rng.normal(size=beta.shape)
        errors, a, h = rng.normal(size=(3, z.size))
        point = 0.0
        for i in range(z.size):
            other = (z == z[i]) & (np.arange(z.size) != i)
            point += errors[i] * np.mean(beta[other, folds[i]])
            point += a[i] * (np.mean(rho[other, folds[i]]) - h[i])
        result = loo_cell_variance(
            Z=z, nuisance_folds=folds, residuals=errors,
            beta_predictions=beta, rho_predictions=rho, a=a, h=h,
        )
        self.assertAlmostEqual(result["loo_cell_correction"], point / z.size, places=12)

    def test_small_cells_use_only_distinct_observations(self) -> None:
        for n in (2, 3):
            kernel = np.ones((n, n)) * 2
            np.fill_diagonal(kernel, 0.0)
            z = np.arange(1.0, n + 1)
            expected_u = np.mean([z[i] * z[j] for i in range(n) for j in range(n) if i != j])
            self.assertAlmostEqual(
                _loo_kernel_variance(kernel, small_cell_product=z),
                n * n * (4 - expected_u),
            )

    def test_degenerate_component_has_the_correct_variance(self) -> None:
        rng = np.random.default_rng(9931)
        n, replications = 6, 12_000
        corrections, variance_estimates = [], []
        for _ in range(replications):
            beta, errors = rng.normal(size=(2, n))
            kernel = (np.outer(beta, errors) + np.outer(errors, beta)) / 2
            np.fill_diagonal(kernel, 0.0)
            corrections.append(np.sum(kernel) / (n * (n - 1)))
            variance_estimates.append(_loo_kernel_variance(kernel, small_cell_product=np.zeros(n)) / n**2)
        exact_variance = 1 / (n * (n - 1))
        self.assertLess(abs(np.var(corrections, ddof=1) / exact_variance - 1), 0.06)
        self.assertLess(abs(np.mean(variance_estimates) / exact_variance - 1), 0.06)


if __name__ == "__main__":
    unittest.main()
