from __future__ import annotations

import numpy as np

from ._ci import normal_ci
from ._linear_solvers import prediction_variance, solve_regularized_system
from ._tuning import EstimationTerms, FoldTerms, select_regularization
from .grouping import group_counts, group_means, leave_one_out_group_means, make_stratified_folds
from .results import DualFitResult, FunctionalEstimate, NPIVInferenceResult, StructuralFitResult
from .validation import (
    encode_instruments,
    validate_n_splits,
    validate_regularization_grid,
    validate_selection,
    validate_target_features,
    validate_training_data,
)


DEFAULT_LAMBDA_GRID = np.logspace(-1, -8, 8)
DEFAULT_GAMMA_GRID = np.array([1e-2, 1e-4, 1e-6, 1e-8, 0.0], dtype=float)


def _penalty_matrix(X: np.ndarray) -> np.ndarray:
    return X.T @ X / X.shape[0]


def _primal_npjive_terms(X: np.ndarray, Z: np.ndarray, Y: np.ndarray) -> EstimationTerms:
    loo_X = leave_one_out_group_means(X, Z)
    loo_Y = leave_one_out_group_means(Y, Z)
    n_obs = X.shape[0]
    quadratic = (loo_X.T @ X + X.T @ loo_X) / (2.0 * n_obs)
    linear = (loo_X.T @ Y + X.T @ loo_Y) / (2.0 * n_obs)
    return EstimationTerms(quadratic=quadratic, linear=linear, penalty=_penalty_matrix(X))


def _primal_2sls_terms(X: np.ndarray, Z: np.ndarray, Y: np.ndarray) -> EstimationTerms:
    counts = group_counts(Z)
    weights = counts / counts.sum()
    group_X = group_means(X, Z, n_groups=counts.shape[0])
    group_Y = group_means(Y, Z, n_groups=counts.shape[0])
    quadratic = (group_X.T * weights) @ group_X
    linear = (group_X.T * weights) @ group_Y
    return EstimationTerms(quadratic=quadratic, linear=linear, penalty=_penalty_matrix(X))


def _dual_npjive_terms(X: np.ndarray, Z: np.ndarray, X_new: np.ndarray) -> EstimationTerms:
    loo_X = leave_one_out_group_means(X, Z)
    n_obs = X.shape[0]
    quadratic = (loo_X.T @ X + X.T @ loo_X) / (2.0 * n_obs)
    linear = np.mean(X_new, axis=0)
    return EstimationTerms(quadratic=quadratic, linear=linear, penalty=_penalty_matrix(X))


def _dual_2sls_terms(X: np.ndarray, Z: np.ndarray, X_new: np.ndarray) -> EstimationTerms:
    counts = group_counts(Z)
    weights = counts / counts.sum()
    group_X = group_means(X, Z, n_groups=counts.shape[0])
    quadratic = (group_X.T * weights) @ group_X
    linear = np.mean(X_new, axis=0)
    return EstimationTerms(quadratic=quadratic, linear=linear, penalty=_penalty_matrix(X))


def _build_fold_terms_primal(X: np.ndarray, Z: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> list[FoldTerms]:
    fold_terms: list[FoldTerms] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        valid_mask = folds == fold_id
        fold_terms.append(
            FoldTerms(
                train=_primal_npjive_terms(X[train_mask], Z[train_mask], Y[train_mask]),
                valid=_primal_npjive_terms(X[valid_mask], Z[valid_mask], Y[valid_mask]),
            )
        )
    return fold_terms


def _build_fold_terms_primal_2sls(X: np.ndarray, Z: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> list[FoldTerms]:
    fold_terms: list[FoldTerms] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        valid_mask = folds == fold_id
        fold_terms.append(
            FoldTerms(
                train=_primal_2sls_terms(X[train_mask], Z[train_mask], Y[train_mask]),
                valid=_primal_2sls_terms(X[valid_mask], Z[valid_mask], Y[valid_mask]),
            )
        )
    return fold_terms


def _build_fold_terms_dual(X: np.ndarray, Z: np.ndarray, X_new: np.ndarray, folds: np.ndarray) -> list[FoldTerms]:
    fold_terms: list[FoldTerms] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        valid_mask = folds == fold_id
        fold_terms.append(
            FoldTerms(
                train=_dual_npjive_terms(X[train_mask], Z[train_mask], X_new),
                valid=_dual_npjive_terms(X[valid_mask], Z[valid_mask], X_new),
            )
        )
    return fold_terms


def _build_fold_terms_dual_2sls(X: np.ndarray, Z: np.ndarray, X_new: np.ndarray, folds: np.ndarray) -> list[FoldTerms]:
    fold_terms: list[FoldTerms] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        valid_mask = folds == fold_id
        fold_terms.append(
            FoldTerms(
                train=_dual_2sls_terms(X[train_mask], Z[train_mask], X_new),
                valid=_dual_2sls_terms(X[valid_mask], Z[valid_mask], X_new),
            )
        )
    return fold_terms


def _select_method(selection: str, npjive_risk: float, baseline_risk: float) -> str:
    if selection == "npjive":
        return "npjive"
    if selection == "2sls":
        return "2sls"
    return "npjive" if npjive_risk <= baseline_risk else "2sls"


def _use_leave_one_out_for_dual(dual_method: str) -> bool:
    return dual_method == "npjive"


def fit_structural_nuisance(
    X: object,
    Z: object,
    Y: object,
    *,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> StructuralFitResult:
    """
    Fit the structural nuisance for the minimum-norm NPIV solution.

    Parameters
    ----------
    X, Z, Y:
        Training covariates, discrete instruments, and outcomes.
    n_splits:
        Number of stratified folds used for regularization tuning.
    lambda_grid, gamma_grid:
        Optional regularization grids. When omitted, defaults tuned for a small
        research workflow are used.
    selection:
        ``"adaptive"`` chooses between npJIVE and 2SLS using cross-validation.
        ``"npjive"`` and ``"2sls"`` force the chosen nuisance family.
    random_state:
        Seed for fold construction.
    """

    X_arr, encoded, Y_arr = validate_training_data(X, Z, Y)
    selection = validate_selection(selection)
    n_splits = validate_n_splits(n_splits, int(encoded.counts.min()))
    lambda_values = validate_regularization_grid("lambda_grid", lambda_grid, DEFAULT_LAMBDA_GRID)
    gamma_values = validate_regularization_grid("gamma_grid", gamma_grid, DEFAULT_GAMMA_GRID)

    folds = make_stratified_folds(encoded.codes, n_splits=n_splits, random_state=random_state)
    fold_terms_npjive = _build_fold_terms_primal(X_arr, encoded.codes, Y_arr, folds)
    fold_terms_2sls = _build_fold_terms_primal_2sls(X_arr, encoded.codes, Y_arr, folds)

    tune_npjive, _ = select_regularization(
        fold_terms_npjive,
        method_name="npjive",
        lambda_grid=lambda_values,
        gamma_grid=gamma_values,
    )
    tune_2sls, _ = select_regularization(
        fold_terms_2sls,
        method_name="2sls",
        lambda_grid=lambda_values,
        gamma_grid=gamma_values,
    )

    terms_npjive = _primal_npjive_terms(X_arr, encoded.codes, Y_arr)
    system_npjive = terms_npjive.quadratic + tune_npjive.lambda_value * terms_npjive.penalty + tune_npjive.gamma * np.eye(X_arr.shape[1])
    coef_npjive = solve_regularized_system(system_npjive, terms_npjive.linear)

    terms_2sls = _primal_2sls_terms(X_arr, encoded.codes, Y_arr)
    system_2sls = terms_2sls.quadratic + tune_2sls.lambda_value * terms_2sls.penalty + tune_2sls.gamma * np.eye(X_arr.shape[1])
    coef_2sls = solve_regularized_system(system_2sls, terms_2sls.linear)

    selected_method = _select_method(selection, tune_npjive.cv_risk, tune_2sls.cv_risk)
    coef_selected = coef_npjive if selected_method == "npjive" else coef_2sls

    return StructuralFitResult(
        coef_selected=coef_selected,
        coef_npjive=coef_npjive,
        coef_2sls=coef_2sls,
        selected_method=selected_method,
        tuning_npjive=tune_npjive,
        tuning_2sls=tune_2sls,
        instrument_levels=encoded.levels,
    )


def fit_dual_nuisance(
    X: object,
    Z: object,
    X_new: object,
    *,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> DualFitResult:
    """
    Fit the dual / Riesz nuisance associated with the target covariate law.

    ``X_new`` may have a different number of rows than the training sample. It
    only needs to share the same feature dimension as ``X``.
    """

    X_arr = np.asarray(X, dtype=float)
    encoded = encode_instruments(Z)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite values.")
    if encoded.codes.shape[0] != X_arr.shape[0]:
        raise ValueError("X and Z must have the same number of rows.")
    X_new_arr = validate_target_features(X_new, X_arr.shape[1])
    selection = validate_selection(selection)
    n_splits = validate_n_splits(n_splits, int(encoded.counts.min()))
    lambda_values = validate_regularization_grid("lambda_grid", lambda_grid, DEFAULT_LAMBDA_GRID)
    gamma_values = validate_regularization_grid("gamma_grid", gamma_grid, DEFAULT_GAMMA_GRID)

    folds = make_stratified_folds(encoded.codes, n_splits=n_splits, random_state=random_state)
    fold_terms_npjive = _build_fold_terms_dual(X_arr, encoded.codes, X_new_arr, folds)
    fold_terms_2sls = _build_fold_terms_dual_2sls(X_arr, encoded.codes, X_new_arr, folds)

    tune_npjive, _ = select_regularization(
        fold_terms_npjive,
        method_name="npjive",
        lambda_grid=lambda_values,
        gamma_grid=gamma_values,
    )
    tune_2sls, _ = select_regularization(
        fold_terms_2sls,
        method_name="2sls",
        lambda_grid=lambda_values,
        gamma_grid=gamma_values,
    )

    terms_npjive = _dual_npjive_terms(X_arr, encoded.codes, X_new_arr)
    system_npjive = terms_npjive.quadratic + tune_npjive.lambda_value * terms_npjive.penalty + tune_npjive.gamma * np.eye(X_arr.shape[1])
    coef_npjive = solve_regularized_system(system_npjive, terms_npjive.linear)

    terms_2sls = _dual_2sls_terms(X_arr, encoded.codes, X_new_arr)
    system_2sls = terms_2sls.quadratic + tune_2sls.lambda_value * terms_2sls.penalty + tune_2sls.gamma * np.eye(X_arr.shape[1])
    coef_2sls = solve_regularized_system(system_2sls, terms_2sls.linear)

    selected_method = _select_method(selection, tune_npjive.cv_risk, tune_2sls.cv_risk)
    coef_selected = coef_npjive if selected_method == "npjive" else coef_2sls

    return DualFitResult(
        coef_selected=coef_selected,
        coef_npjive=coef_npjive,
        coef_2sls=coef_2sls,
        selected_method=selected_method,
        tuning_npjive=tune_npjive,
        tuning_2sls=tune_2sls,
        instrument_levels=encoded.levels,
    )

def _build_functional_estimate(
    *,
    method_name: str,
    theta_coef: np.ndarray,
    beta_coef: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    use_leave_one_out: bool,
) -> FunctionalEstimate:
    h_hat = X @ theta_coef
    h_hat_new = X_new @ theta_coef
    beta_scores = X @ beta_coef
    counts = group_counts(Z)
    weights = counts / counts.sum()

    if use_leave_one_out:
        score_weights = leave_one_out_group_means(beta_scores, Z, n_groups=counts.shape[0])
        influence = score_weights * (Y - h_hat)
        plugin = float(np.mean(h_hat_new))
        ipw = float(np.mean(score_weights * Y))
        estimate = float(plugin + np.mean(influence))
    else:
        score_weights = group_means(beta_scores, Z, n_groups=counts.shape[0])
        group_Y = group_means(Y, Z, n_groups=counts.shape[0])
        group_h = group_means(h_hat, Z, n_groups=counts.shape[0])
        influence = score_weights[Z] * (Y - h_hat)
        plugin = float(np.mean(h_hat_new))
        ipw = float(np.sum(weights * score_weights * group_Y))
        estimate = float(plugin + np.sum(weights * score_weights * (group_Y - group_h)))

    se = float(
        np.sqrt(
            prediction_variance(influence) / X.shape[0]
            + prediction_variance(h_hat_new) / X_new.shape[0]
        )
    )
    ci_lower, ci_upper = normal_ci(estimate, se)
    return FunctionalEstimate(
        method_name=method_name,
        estimate=estimate,
        plugin_estimate=plugin,
        ipw_estimate=ipw,
        se=se,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        influence_function=influence,
    )


def _estimate_average_functional_from_arrays(
    X_arr: np.ndarray,
    Z_arr: np.ndarray,
    Y_arr: np.ndarray,
    X_new_arr: np.ndarray,
    *,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> NPIVInferenceResult:
    encoded = encode_instruments(Z_arr)

    structural = fit_structural_nuisance(
        X_arr,
        encoded.codes,
        Y_arr,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )
    dual = fit_dual_nuisance(
        X_arr,
        encoded.codes,
        X_new_arr,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )

    npjive = _build_functional_estimate(
        method_name="npjive",
        theta_coef=structural.coef_npjive,
        beta_coef=dual.coef_npjive,
        X=X_arr,
        Z=encoded.codes,
        Y=Y_arr,
        X_new=X_new_arr,
        use_leave_one_out=True,
    )
    baseline = _build_functional_estimate(
        method_name="2sls",
        theta_coef=structural.coef_2sls,
        beta_coef=dual.coef_2sls,
        X=X_arr,
        Z=encoded.codes,
        Y=Y_arr,
        X_new=X_new_arr,
        use_leave_one_out=False,
    )

    selected_name = f"{structural.selected_method}+{dual.selected_method}"
    selected = _build_functional_estimate(
        method_name=selected_name,
        theta_coef=structural.coef_selected,
        beta_coef=dual.coef_selected,
        X=X_arr,
        Z=encoded.codes,
        Y=Y_arr,
        X_new=X_new_arr,
        use_leave_one_out=_use_leave_one_out_for_dual(dual.selected_method),
    )

    return NPIVInferenceResult(
        selected=selected,
        npjive=npjive,
        baseline_2sls=baseline,
        structural_fit=structural,
        dual_fit=dual,
    )


def estimate_average_functional(
    X: object,
    Z: object,
    Y: object,
    X_new: object,
    *,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> NPIVInferenceResult:
    """
    Estimate an average linear functional of the structural function.

    The returned object contains:

    - ``selected``: the estimator induced by the chosen nuisance fits
    - ``npjive``: the npJIVE-based functional estimate
    - ``baseline_2sls``: the 2SLS-style baseline estimate
    - ``structural_fit`` and ``dual_fit``: nuisance fit details
    """

    X_arr, encoded, Y_arr = validate_training_data(X, Z, Y)
    X_new_arr = validate_target_features(X_new, X_arr.shape[1])
    return _estimate_average_functional_from_arrays(
        X_arr,
        encoded.codes,
        Y_arr,
        X_new_arr,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )
