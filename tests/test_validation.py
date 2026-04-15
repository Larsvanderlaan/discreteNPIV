import unittest

import numpy as np

from discreteNPIV.validation import encode_instruments, validate_n_splits, validate_regularization_grid


class ValidationTests(unittest.TestCase):
    def test_encode_instruments_maps_to_zero_based_codes(self) -> None:
        encoded = encode_instruments(np.array(["b", "a", "b", "c"]))
        np.testing.assert_array_equal(encoded.codes, np.array([1, 0, 1, 2]))
        np.testing.assert_array_equal(encoded.levels, np.array(["a", "b", "c"]))

    def test_validate_regularization_grid_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            validate_regularization_grid("lambda_grid", np.array([1.0, -1.0]), np.array([1.0]))

    def test_validate_n_splits_checks_smallest_group(self) -> None:
        with self.assertRaises(ValueError):
            validate_n_splits(4, 3)


if __name__ == "__main__":
    unittest.main()

