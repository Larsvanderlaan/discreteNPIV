from __future__ import annotations

import numpy as np


def group_counts(Z: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=int)
    if n_groups is None:
        n_groups = int(np.max(Z)) + 1
    return np.bincount(Z, minlength=n_groups).astype(int)


def _contiguous_group_sums(values: np.ndarray, Z: np.ndarray, n_groups: int) -> np.ndarray | None:
    """Fast group sums when rows are already ordered by instrument level."""

    if Z.size == 0 or np.any(Z[1:] < Z[:-1]):
        return None
    starts = np.r_[0, np.flatnonzero(Z[1:] != Z[:-1]) + 1]
    labels = Z[starts]
    if values.ndim == 1:
        reduced = np.add.reduceat(values, starts)
        sums = np.zeros(n_groups, dtype=float)
        sums[labels] = reduced
        return sums
    reduced = np.add.reduceat(values, starts, axis=0)
    sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    sums[labels] = reduced
    return sums


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

    sums = _contiguous_group_sums(values, Z, n_groups)
    if sums is None:
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

    sums = _contiguous_group_sums(values, Z, n_groups)
    if sums is None:
        sums = np.zeros((n_groups, values.shape[1]), dtype=float)
        np.add.at(sums, Z, values)
    loo = (sums[Z] - values) / np.maximum(row_counts - 1, 1)[:, None]
    singleton = row_counts == 1
    loo[singleton] = values[singleton]
    return loo


def fold_excluded_group_means(
    values: np.ndarray,
    Z: np.ndarray,
    folds: np.ndarray,
    n_groups: int | None = None,
) -> np.ndarray:
    """Group means excluding the row's whole validation fold within its group."""

    values = np.asarray(values, dtype=float)
    Z = np.asarray(Z, dtype=int)
    folds = np.asarray(folds, dtype=int)
    if Z.shape[0] != folds.shape[0] or Z.shape[0] != values.shape[0]:
        raise ValueError("values, Z, and folds must have the same number of rows.")
    if values.ndim not in {1, 2}:
        raise ValueError("values must be a 1D or 2D array.")

    counts = group_counts(Z, n_groups=n_groups)
    n_groups = counts.shape[0]
    output = np.zeros_like(values, dtype=float)
    for fold_id in np.unique(folds):
        eval_mask = folds == fold_id
        train_mask = ~eval_mask
        train_counts = group_counts(Z[train_mask], n_groups=n_groups)
        eval_groups = np.unique(Z[eval_mask])
        empty_groups = eval_groups[train_counts[eval_groups] == 0]
        if empty_groups.size:
            raise ValueError("fold-excluded group means require at least one training observation per instrument/fold.")
        means = group_means(values[train_mask], Z[train_mask], n_groups=n_groups)
        output[eval_mask] = means[Z[eval_mask]]
    return output


def make_stratified_folds(Z: np.ndarray, n_splits: int, random_state: int | None = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=int)
    rng = np.random.default_rng(random_state)
    folds = np.empty(Z.shape[0], dtype=int)
    order = np.argsort(Z, kind="stable")
    sorted_Z = Z[order]
    starts = np.r_[0, np.flatnonzero(sorted_Z[1:] != sorted_Z[:-1]) + 1]
    stops = np.r_[starts[1:], sorted_Z.shape[0]]

    for start, stop in zip(starts, stops, strict=True):
        indices = order[start:stop]
        shuffled = rng.permutation(indices)
        for fold_id, fold_indices in enumerate(np.array_split(shuffled, n_splits)):
            folds[fold_indices] = fold_id
    return folds
