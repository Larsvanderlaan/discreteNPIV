from __future__ import annotations

import numpy as np

from discreteNPIV._linear_solvers import prediction_variance, solve_regularized_system
from discreteNPIV.api import _primal_npjive_terms
from discreteNPIV.grouping import leave_one_out_group_means
from discreteNPIV.results import StructuralFitResult



def _star_component_diagnostics(base_influence: np.ndarray, correction: np.ndarray) -> dict[str, float]:
    base = np.asarray(base_influence, dtype=float)
    component = np.asarray(correction, dtype=float)
    star = base + component
    n_obs = float(base.shape[0])
    base_variance = float(prediction_variance(base))
    component_variance = float(prediction_variance(component))
    star_variance = float(prediction_variance(star))
    covariance = (
        0.0
        if base.shape[0] < 2
        else float(np.cov(base, component, ddof=1)[0, 1])
    )
    abs_component = np.abs(component)
    return {
        "correction_mean": float(np.mean(component)),
        "correction_abs_mean": float(np.mean(abs_component)),
        "correction_sd": float(np.std(component, ddof=1)) if component.shape[0] > 1 else 0.0,
        "correction_abs_p95": float(np.quantile(abs_component, 0.95)),
        "correction_abs_p99": float(np.quantile(abs_component, 0.99)),
        "base_if_variance": base_variance,
        "correction_if_variance": component_variance,
        "if_covariance": covariance,
        "star_if_variance": star_variance,
        "base_if_se_component": float(np.sqrt(base_variance / n_obs)),
        "correction_if_se_component": float(np.sqrt(component_variance / n_obs)),
        "star_if_se_component": float(np.sqrt(star_variance / n_obs)),
        "if_variance_increment": float(star_variance - base_variance),
        "if_variance_ratio": np.nan if base_variance <= 0.0 else float(star_variance / base_variance),
        "correction_variance_share": np.nan if star_variance <= 0.0 else float(component_variance / star_variance),
    }


def _triangular_if_variance(
    scores: np.ndarray,
    Z: np.ndarray,
    *,
    eval_mask: np.ndarray | None = None,
) -> float:
    """Estimate variance of ``N^{-1} sum phi_ki`` under the fixed-arm design."""

    score_arr = np.asarray(scores, dtype=float)
    z_arr = np.asarray(Z, dtype=int)
    if score_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("scores and Z must have the same number of rows.")
    if eval_mask is None:
        mask = np.ones(z_arr.shape[0], dtype=bool)
    else:
        mask = np.asarray(eval_mask, dtype=bool)
        if mask.shape[0] != z_arr.shape[0]:
            raise ValueError("eval_mask and Z must have the same number of rows.")

    n_total = float(z_arr.shape[0])
    if n_total <= 0.0:
        return np.nan
    variance_sum = 0.0
    for group in np.unique(z_arr):
        group_scores = score_arr[(z_arr == group) & mask]
        if group_scores.shape[0] <= 1:
            continue
        variance_sum += float(group_scores.shape[0]) * prediction_variance(group_scores)
    return float(variance_sum / (n_total * n_total))


def _target_sample_variance(*prediction_arrays: np.ndarray) -> float:
    variance = 0.0
    for predictions in prediction_arrays:
        arr = np.asarray(predictions, dtype=float)
        if arr.shape[0] > 0:
            variance += prediction_variance(arr) / arr.shape[0]
    return float(variance)


def _paper_if_se_diagnostics(
    *,
    scores: np.ndarray,
    Z: np.ndarray,
    target_predictions: tuple[np.ndarray, ...] = (),
    eval_mask: np.ndarray | None = None,
) -> dict[str, float | str]:
    historical_variance = _triangular_if_variance(scores, Z, eval_mask=eval_mask)
    target_variance = _target_sample_variance(*target_predictions)
    total_variance = historical_variance + target_variance
    return {
        "se_if_historical": float(np.sqrt(max(historical_variance, 0.0))),
        "se_target_sample": float(np.sqrt(max(target_variance, 0.0))),
        "se_if_total": float(np.sqrt(max(total_variance, 0.0))),
        "if_historical_variance": float(historical_variance),
        "target_sample_variance": float(target_variance),
        "if_total_variance": float(total_variance),
        "if_variance_method": "paper_triangular_eval_fold" if eval_mask is not None else "paper_triangular_all_rows",
    }


def _structural_linearized_mean_scores(
    *,
    structural_fit: StructuralFitResult,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    correction_weights: np.ndarray,
    fit_mask: np.ndarray | None = None,
    target_gradient_weight: float = 1.0,
) -> np.ndarray:
    """Linearized Y-scores including first-stage structural nuisance variation."""

    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    Y_arr = np.asarray(Y, dtype=float)
    weights_arr = np.asarray(correction_weights, dtype=float)
    if weights_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("correction_weights must have one entry per row.")
    if fit_mask is None:
        mask = np.ones(X_arr.shape[0], dtype=bool)
    else:
        mask = np.asarray(fit_mask, dtype=bool)
        if mask.shape[0] != X_arr.shape[0]:
            raise ValueError("fit_mask must have one entry per row.")
    X_fit = X_arr[mask]
    Z_fit = Z_arr[mask]
    Y_fit = Y_arr[mask]
    terms = _primal_npjive_terms(X_fit, Z_fit, Y_fit)
    system = (
        terms.quadratic
        + structural_fit.tuning_npjive.lambda_value * terms.penalty
        + structural_fit.tuning_npjive.gamma * np.eye(X_arr.shape[1])
    )
    # For the symmetric npJIVE linear term, the coefficient map is
    # n_fit^{-1} sum_i \bar X_{-i} Y_i.
    excluded_means = leave_one_out_group_means(X_fit, Z_fit)
    linear_map = excluded_means.T / X_fit.shape[0]
    direct_coef = weights_arr / X_arr.shape[0]
    derivative_direction = (
        float(target_gradient_weight) * np.mean(X_new, axis=0)
        - X_arr.T @ direct_coef
    )
    structural_coef_fit = linear_map.T @ solve_regularized_system(system.T, derivative_direction)
    coefficients = np.array(direct_coef, dtype=float, copy=True)
    coefficients[mask] += structural_coef_fit
    return X_arr.shape[0] * coefficients * Y_arr


def _loo_correction_weights(*, X: np.ndarray, Z: np.ndarray, beta_coef: np.ndarray) -> np.ndarray:
    return leave_one_out_group_means(X @ beta_coef, Z)


def _structural_linearized_effect_scores(
    *,
    structural_fit: StructuralFitResult,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_treated: np.ndarray,
    X_control: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
    fit_mask: np.ndarray | None = None,
    extra_scores: np.ndarray | None = None,
) -> np.ndarray:
    treated_scores = _structural_linearized_mean_scores(
        structural_fit=structural_fit,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_treated,
        correction_weights=treated_weights,
        fit_mask=fit_mask,
    )
    control_scores = _structural_linearized_mean_scores(
        structural_fit=structural_fit,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_control,
        correction_weights=control_weights,
        fit_mask=fit_mask,
    )
    scores = treated_scores - control_scores
    if extra_scores is not None:
        scores = scores + np.asarray(extra_scores, dtype=float)
    return scores


def _center_scores_within_instrument(
    scores: np.ndarray,
    Z: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Center score values within fixed instrument levels."""

    score_arr = np.asarray(scores, dtype=float)
    z_arr = np.asarray(Z, dtype=int)
    if score_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("scores and Z must have the same number of rows.")
    if mask is None:
        active = np.ones(score_arr.shape[0], dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.shape[0] != score_arr.shape[0]:
            raise ValueError("mask must have the same number of rows as scores.")
    centered = np.array(score_arr, dtype=float, copy=True)
    for group in np.unique(z_arr):
        group_mask = (z_arr == group) & active
        if np.any(group_mask):
            centered[group_mask] -= float(np.mean(centered[group_mask]))
    return centered
