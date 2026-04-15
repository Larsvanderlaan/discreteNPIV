import unittest

import numpy as np

from discreteNPIV.grouping import group_means, leave_one_out_group_means, make_stratified_folds


class GroupingTests(unittest.TestCase):
    def test_group_means_1d(self) -> None:
        values = np.array([1.0, 3.0, 5.0, 7.0])
        groups = np.array([0, 0, 1, 1])
        means = group_means(values, groups)
        np.testing.assert_allclose(means, np.array([2.0, 6.0]))

    def test_leave_one_out_handles_singletons(self) -> None:
        values = np.array([10.0, 20.0, 7.0])
        groups = np.array([0, 0, 1])
        loo = leave_one_out_group_means(values, groups)
        np.testing.assert_allclose(loo, np.array([20.0, 10.0, 7.0]))

    def test_leave_one_out_matrix(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]])
        groups = np.array([0, 0, 1, 1])
        loo = leave_one_out_group_means(values, groups)
        expected = np.array([[3.0, 6.0], [1.0, 2.0], [7.0, 14.0], [5.0, 10.0]])
        np.testing.assert_allclose(loo, expected)

    def test_make_stratified_folds_is_reproducible(self) -> None:
        groups = np.repeat(np.arange(3), 6)
        folds_a = make_stratified_folds(groups, n_splits=3, random_state=1)
        folds_b = make_stratified_folds(groups, n_splits=3, random_state=1)
        np.testing.assert_array_equal(folds_a, folds_b)
        for group in range(3):
            counts = np.bincount(folds_a[groups == group], minlength=3)
            np.testing.assert_array_equal(counts, np.array([2, 2, 2]))


if __name__ == "__main__":
    unittest.main()

