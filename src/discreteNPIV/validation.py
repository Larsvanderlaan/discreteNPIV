from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedInstruments:
    codes: np.ndarray
    levels: np.ndarray
    counts: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(self.counts.shape[0])


def _ensure_finite(name: str, array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def as_2d_float(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return _ensure_finite(name, array)


def as_1d_float(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return _ensure_finite(name, array)


def encode_instruments(Z: object) -> EncodedInstruments:
    raw = np.asarray(Z)
    if raw.ndim != 1:
        raise ValueError("Z must be a 1D array.")
    levels, codes = np.unique(raw, return_inverse=True)
    counts = np.bincount(codes, minlength=levels.shape[0]).astype(int)
    return EncodedInstruments(codes=codes.astype(int), levels=levels, counts=counts)


def validate_training_data(
    X: object,
    Z: object,
    Y: object,
) -> tuple[np.ndarray, EncodedInstruments, np.ndarray]:
    X_arr = as_2d_float("X", X)
    Y_arr = as_1d_float("Y", Y)
    encoded = encode_instruments(Z)
    n_obs = X_arr.shape[0]
    if Y_arr.shape[0] != n_obs or encoded.codes.shape[0] != n_obs:
        raise ValueError("X, Z, and Y must have the same number of rows.")
    return X_arr, encoded, Y_arr


def validate_target_features(X_new: object, n_features: int) -> np.ndarray:
    X_new_arr = as_2d_float("X_new", X_new)
    if X_new_arr.shape[1] != n_features:
        raise ValueError("X_new must have the same number of columns as X.")
    return X_new_arr


def validate_regularization_grid(name: str, grid: object, default: np.ndarray) -> np.ndarray:
    if grid is None:
        return default.copy()
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return np.unique(np.sort(values)[::-1])


def validate_selection(selection: str) -> str:
    valid = {"adaptive", "npjive", "2sls"}
    if selection not in valid:
        raise ValueError(f"selection must be one of {sorted(valid)}.")
    return selection


def validate_n_splits(n_splits: int, min_group_count: int) -> int:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > min_group_count:
        raise ValueError(
            "n_splits cannot exceed the smallest instrument-group size. "
            f"Received n_splits={n_splits} and min_group_count={min_group_count}."
        )
    return int(n_splits)
