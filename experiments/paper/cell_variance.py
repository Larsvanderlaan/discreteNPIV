"""Variance of the leave-one-out cell corrections used in the paper."""

from __future__ import annotations

import numpy as np


def _loo_kernel_variance(kernel: np.ndarray, *, small_cell_product: np.ndarray) -> float:
    """Return n^2 times the disjoint-pair estimate of Var(C_k)."""

    n = kernel.shape[0]
    correction = float(np.sum(kernel) / (n * (n - 1)))
    if n < 4:
        z = np.asarray(small_cell_product, dtype=float)
        squared_mean = float((np.sum(z) ** 2 - z @ z) / (n * (n - 1)))
        return float(n * n * (correction**2 - squared_mean))

    # Centering avoids subtracting two estimates of the squared cell mean.
    centered = kernel - correction
    np.fill_diagonal(centered, 0.0)
    row_sums = np.sum(centered, axis=1)
    denominator = n * (n - 1) * (n - 2) * (n - 3)
    return float(
        n * n
        * (4.0 * (row_sums @ row_sums) - 2.0 * np.sum(centered**2))
        / denominator
    )


def loo_cell_variance(
    *,
    Z: np.ndarray,
    nuisance_folds: np.ndarray,
    residuals: np.ndarray,
    beta_predictions: np.ndarray,
    a: np.ndarray | None = None,
    rho_predictions: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> dict[str, float | str | bool]:
    """Estimate the historical variance of the full LOO estimator.

    Column f of beta_predictions and rho_predictions evaluates nuisance fit f
    on every historical row. The fit used at the scored row determines both
    factors of each directional pair. This retains the actual cross-fitted
    estimator when constructing its symmetric pair kernel.
    """

    z = np.asarray(Z)
    folds = np.asarray(nuisance_folds)
    errors = np.asarray(residuals, dtype=float)
    beta = np.asarray(beta_predictions, dtype=float)
    n_total = z.size
    if z.ndim != 1 or n_total < 2:
        raise ValueError("Z must contain at least two historical rows.")
    if errors.shape != z.shape or folds.shape != z.shape:
        raise ValueError("Residuals and nuisance folds must have one entry per row.")
    if beta.ndim != 2 or beta.shape[0] != n_total:
        raise ValueError("beta_predictions must have one row per historical observation.")
    if not np.issubdtype(folds.dtype, np.integer) or np.any(folds < 0) or np.any(folds >= beta.shape[1]):
        raise ValueError("Nuisance fold indices must identify prediction columns.")
    corrected = a is not None or rho_predictions is not None or h is not None
    if corrected and (a is None or rho_predictions is None or h is None):
        raise ValueError("The corrected estimator requires a, rho_predictions and h.")
    a_values = np.zeros(n_total) if a is None else np.asarray(a, dtype=float)
    h_values = np.zeros(n_total) if h is None else np.asarray(h, dtype=float)
    rho = np.zeros_like(beta) if rho_predictions is None else np.asarray(rho_predictions, dtype=float)
    if a_values.shape != z.shape or h_values.shape != z.shape or rho.shape != beta.shape:
        raise ValueError("Corrected nuisance predictions have incompatible shapes.")
    if not all(np.all(np.isfinite(values)) for values in (errors, beta, a_values, h_values, rho)):
        raise ValueError("LOO variance requires finite nuisance predictions and residuals.")

    variance_sum = 0.0
    correction_sum = 0.0
    negative_cells = 0
    for group in np.unique(z):
        rows = np.flatnonzero(z == group)
        n = rows.size
        if n < 2:
            raise ValueError("LOO inference requires at least two observations per instrument level.")
        columns = folds[rows]
        directional = errors[rows, None] * beta[rows][:, columns].T
        if corrected:
            directional += a_values[rows, None] * rho[rows][:, columns].T
            directional -= (a_values[rows] * h_values[rows])[:, None]
        kernel = 0.5 * (directional + directional.T)
        np.fill_diagonal(kernel, 0.0)
        correction_sum += float(np.sum(kernel) / (n - 1))
        contribution = _loo_kernel_variance(
            kernel, small_cell_product=a_values[rows] * h_values[rows]
        )
        variance_sum += contribution
        negative_cells += contribution < 0.0

    raw_variance = variance_sum / (n_total * n_total)
    floor = 1.0 / (n_total * n_total)
    return {
        "if_historical_variance": float(max(floor, raw_variance)),
        "loo_historical_variance_raw": float(raw_variance),
        "loo_variance_floor_applied": bool(raw_variance < floor),
        "loo_negative_cell_variance_count": int(negative_cells),
        "loo_cell_correction": float(correction_sum / n_total),
        "if_variance_method": "loo_disjoint_pairs",
    }
