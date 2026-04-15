from __future__ import annotations

import numpy as np


def solve_regularized_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    sym_matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.solve(sym_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(sym_matrix, rhs, rcond=None)[0]


def prediction_variance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.shape[0] <= 1:
        return 0.0
    return float(np.var(values, ddof=1))

