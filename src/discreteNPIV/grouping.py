from __future__ import annotations

import numpy as np


def group_counts(Z: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=int)
    if n_groups is None:
        n_groups = int(np.max(Z)) + 1
    return np.bincount(Z, minlength=n_groups).astype(int)


def group_means(values: np.ndarray, Z: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    Z = np.asarray(Z, dtype=int)
    counts = group_counts(Z, n_groups=n_groups)
    n_groups = counts.shape[0]

    if values.ndim == 1:
        sums = np.bincount(Z, weights=values, minlength=n_groups)
        means = np.zeros(n_groups, dtype=float)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero]
        return means

    if values.ndim != 2:
        raise ValueError("values must be a 1D or 2D array.")

    sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(sums, Z, values)
    means = np.zeros_like(sums)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero, None]
    return means


def leave_one_out_group_means(
    values: np.ndarray,
    Z: np.ndarray,
    n_groups: int | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    Z = np.asarray(Z, dtype=int)
    counts = group_counts(Z, n_groups=n_groups)
    n_groups = counts.shape[0]
    row_counts = counts[Z]

    if values.ndim == 1:
        sums = np.bincount(Z, weights=values, minlength=n_groups)
        loo = (sums[Z] - values) / np.maximum(row_counts - 1, 1)
        singleton = row_counts == 1
        loo[singleton] = values[singleton]
        return loo

    if values.ndim != 2:
        raise ValueError("values must be a 1D or 2D array.")

    sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(sums, Z, values)
    loo = (sums[Z] - values) / np.maximum(row_counts - 1, 1)[:, None]
    singleton = row_counts == 1
    loo[singleton] = values[singleton]
    return loo


def make_stratified_folds(Z: np.ndarray, n_splits: int, random_state: int | None = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=int)
    rng = np.random.default_rng(random_state)
    folds = np.empty(Z.shape[0], dtype=int)
    n_groups = int(np.max(Z)) + 1

    for group in range(n_groups):
        indices = np.flatnonzero(Z == group)
        shuffled = rng.permutation(indices)
        for fold_id, fold_indices in enumerate(np.array_split(shuffled, n_splits)):
            folds[fold_indices] = fold_id
    return folds

