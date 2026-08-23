from __future__ import annotations

import numpy as np

from discreteNPIV._linear_solvers import solve_regularized_system
from discreteNPIV._tuning import select_regularization
from discreteNPIV.api import (
    _build_fold_terms_dual,
    _dual_npjive_terms,
    fit_dual_nuisance,
    fit_structural_nuisance,
)
from discreteNPIV.grouping import make_stratified_folds
from discreteNPIV.results import DualFitResult, StructuralFitResult

from .model import (
    LongTermSurrogateConfig,
    NPJIVEInverseFit,
    _dual_gamma_grid,
    _dual_lambda_grid,
    _nuisance_n_splits_for_codes,
    _structural_gamma_grid,
    _structural_lambda_grid,
)
from .projection import _npjive_operator_stability_constraint


def _fit_structural_nuisance_for_config(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    *,
    config: LongTermSurrogateConfig,
    n_splits: int,
    selection: str,
    random_state: int,
    strict_stability: bool = False,
) -> StructuralFitResult:
    enforce_inverse_stability = (
        strict_stability
        or config.tuning_stability_rule == "generalized_positive_condition_le_10000"
    )
    max_coef_norm = (
        config.star_nuisance_max_fold_coef_norm
        if enforce_inverse_stability
        else None
    )
    operator_stability = _npjive_operator_stability_constraint(
        X=np.asarray(X, dtype=float),
        Z=np.asarray(Z, dtype=int),
        n_splits=n_splits,
        config=config,
        random_state=random_state,
    )
    return fit_structural_nuisance(
        X,
        Z,
        Y,
        n_splits=n_splits,
        lambda_grid=np.asarray(_structural_lambda_grid(config), dtype=float),
        gamma_grid=np.asarray(_structural_gamma_grid(config), dtype=float),
        selection=selection,
        random_state=random_state,
        max_fold_coef_norm=max_coef_norm,
        min_system_eigenvalue=(
            config.star_nuisance_min_system_eigenvalue if enforce_inverse_stability else None
        ),
        max_system_condition=(
            config.star_nuisance_max_system_condition if enforce_inverse_stability else None
        ),
        strict_regularization_constraints=enforce_inverse_stability,
        npjive_operator_stability=operator_stability,
    )


def _fit_dual_nuisance_for_config(
    X: np.ndarray,
    Z: np.ndarray,
    X_new: np.ndarray,
    *,
    config: LongTermSurrogateConfig,
    n_splits: int,
    selection: str,
    random_state: int,
    strict_stability: bool = False,
) -> DualFitResult:
    enforce_inverse_stability = (
        strict_stability
        or config.tuning_stability_rule == "generalized_positive_condition_le_10000"
    )
    max_coef_norm = (
        config.star_nuisance_max_fold_coef_norm
        if enforce_inverse_stability
        else None
    )
    operator_stability = _npjive_operator_stability_constraint(
        X=np.asarray(X, dtype=float),
        Z=np.asarray(Z, dtype=int),
        n_splits=n_splits,
        config=config,
        random_state=random_state,
    )
    return fit_dual_nuisance(
        X,
        Z,
        X_new,
        n_splits=n_splits,
        lambda_grid=np.asarray(_dual_lambda_grid(config), dtype=float),
        gamma_grid=np.asarray(_dual_gamma_grid(config), dtype=float),
        selection=selection,
        random_state=random_state,
        max_fold_coef_norm=max_coef_norm,
        min_system_eigenvalue=(
            config.star_nuisance_min_system_eigenvalue if enforce_inverse_stability else None
        ),
        max_system_condition=(
            config.star_nuisance_max_system_condition if enforce_inverse_stability else None
        ),
        strict_regularization_constraints=enforce_inverse_stability,
        npjive_operator_stability=operator_stability,
    )


def _fit_strict_npjive_inverse_for_linear_term(
    X: np.ndarray,
    Z: np.ndarray,
    target_feature_mean: np.ndarray,
    *,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> NPJIVEInverseFit:
    """Solve one npJIVE inverse problem with binding stability constraints."""

    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    target = np.asarray(target_feature_mean, dtype=float)
    if target.ndim != 1 or target.shape[0] != X_arr.shape[1]:
        raise ValueError("target_feature_mean must be a length-p vector.")
    folds = make_stratified_folds(
        Z_arr,
        n_splits=_nuisance_n_splits_for_codes(Z_arr, config.n_splits),
        random_state=random_state,
    )
    fold_terms = _build_fold_terms_dual(X_arr, Z_arr, target[None, :], folds)
    terms = _dual_npjive_terms(X_arr, Z_arr, target[None, :])
    operator_stability = _npjive_operator_stability_constraint(
        X=X_arr,
        Z=Z_arr,
        n_splits=_nuisance_n_splits_for_codes(Z_arr, config.n_splits),
        config=config,
        random_state=random_state,
    )
    tuning, _ = select_regularization(
        fold_terms,
        method_name="npjive",
        lambda_grid=np.asarray(_dual_lambda_grid(config), dtype=float),
        gamma_grid=np.asarray(_dual_gamma_grid(config), dtype=float),
        max_fold_coef_norm=config.star_nuisance_max_fold_coef_norm,
        min_system_eigenvalue=config.star_nuisance_min_system_eigenvalue,
        max_system_condition=config.star_nuisance_max_system_condition,
        strict_constraints=True,
        operator_stability=operator_stability,
        full_terms=terms,
    )
    system = (
        terms.quadratic
        + tuning.lambda_value * terms.penalty
        + tuning.gamma * np.eye(X_arr.shape[1])
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (system + system.T))
    eigen_min = float(np.min(eigenvalues))
    eigen_max = float(np.max(eigenvalues))
    condition = np.inf if eigen_min <= 0.0 else float(eigen_max / eigen_min)
    if not np.isfinite(eigen_min) or eigen_min <= 0.0 or not np.isfinite(condition):
        raise RuntimeError("Selected npJIVE inverse system violates the stability constraints.")
    inverse_operator = solve_regularized_system(system, np.eye(X_arr.shape[1]))
    coef = inverse_operator @ terms.linear
    coef_norm = float(np.linalg.norm(coef))
    if not np.all(np.isfinite(coef)) or coef_norm > config.star_nuisance_max_fold_coef_norm:
        raise RuntimeError("Selected npJIVE inverse coefficient violates the stability constraints.")
    return NPJIVEInverseFit(
        coef=coef,
        lambda_value=float(tuning.lambda_value),
        gamma=float(tuning.gamma),
        cv_risk=float(tuning.cv_risk),
        coef_norm=coef_norm,
        system_eigen_min=eigen_min,
        system_condition=condition,
        inverse_operator=inverse_operator,
        operator_noise_floor=float(tuning.operator_noise_floor),
        operator_noise_margin=float(tuning.operator_noise_margin),
        generalized_system_eigen_min=float(tuning.generalized_system_eigen_min),
        generalized_system_condition=float(tuning.generalized_system_condition),
    )
