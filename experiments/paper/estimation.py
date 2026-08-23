from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from discreteNPIV._ci import normal_ci
from discreteNPIV._linear_solvers import prediction_variance
from discreteNPIV.api import _build_functional_estimate
from discreteNPIV.grouping import group_means, leave_one_out_group_means, make_stratified_folds
from discreteNPIV.results import DualFitResult, FunctionalEstimate, StructuralFitResult

from .design import _fit_riesz_representer_for_feature_mean
from .inference import (
    _center_scores_within_instrument,
    _cross_fitted_single_split_structural_effect_scores,
    _loo_correction_weights,
    _paper_if_se_diagnostics,
    _star_component_diagnostics,
    _structural_linearized_effect_scores,
    _target_sample_variance,
)
from .model import (
    CrossFittedStructuralScores,
    EffectFitBundle,
    LongTermSurrogateConfig,
    SingleSplitMeanFit,
    StarMeanFit,
    _nuisance_n_splits_for_codes,
)
from .nuisance import (
    _fit_dual_nuisance_for_config,
    _fit_strict_npjive_inverse_for_linear_term,
    _fit_structural_nuisance_for_config,
)
from .projection import _fit_riesz_projection


def _finite_statistic(values: list[float], statistic: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "median":
        return float(np.median(finite))
    if statistic == "min":
        return float(np.min(finite))
    if statistic == "max":
        return float(np.max(finite))
    raise ValueError(f"Unknown finite statistic {statistic!r}.")


def _contrast_estimate(
    *,
    method_name: str,
    treated: FunctionalEstimate,
    control: FunctionalEstimate,
    treated_predictions: np.ndarray,
    control_predictions: np.ndarray,
) -> FunctionalEstimate:
    influence = treated.influence_function - control.influence_function
    estimate = float(treated.estimate - control.estimate)
    plugin = float(treated.plugin_estimate - control.plugin_estimate)
    ipw = float(treated.ipw_estimate - control.ipw_estimate)
    se = float(
        np.sqrt(
            prediction_variance(influence) / influence.shape[0]
            + prediction_variance(treated_predictions) / treated_predictions.shape[0]
            + prediction_variance(control_predictions) / control_predictions.shape[0]
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


def _contrast_estimate_with_paper_if_se(
    *,
    method_name: str,
    treated: FunctionalEstimate,
    control: FunctionalEstimate,
    treated_predictions: np.ndarray,
    control_predictions: np.ndarray,
    Z: np.ndarray,
    eval_mask: np.ndarray | None = None,
    structural_linearized_scores: np.ndarray | None = None,
) -> tuple[FunctionalEstimate, dict[str, float | str]]:
    pooled = _contrast_estimate(
        method_name=method_name,
        treated=treated,
        control=control,
        treated_predictions=treated_predictions,
        control_predictions=control_predictions,
    )
    return _effect_estimate_with_paper_if_se(
        method_name=method_name,
        effect=pooled,
        target_predictions=(treated_predictions, control_predictions),
        Z=Z,
        eval_mask=eval_mask,
        structural_linearized_scores=structural_linearized_scores,
    )


def _effect_estimate_with_paper_if_se(
    *,
    method_name: str,
    effect: FunctionalEstimate,
    target_predictions: tuple[np.ndarray, ...],
    Z: np.ndarray,
    eval_mask: np.ndarray | None = None,
    structural_linearized_scores: np.ndarray | None = None,
) -> tuple[FunctionalEstimate, dict[str, float | str]]:
    diagnostics = _paper_if_se_diagnostics(
        scores=effect.influence_function,
        Z=Z,
        target_predictions=target_predictions,
        eval_mask=eval_mask,
    )
    se_if_total = float(diagnostics["se_if_total"])
    ci_lower, ci_upper = normal_ci(effect.estimate, se_if_total)
    pooled_length = (
        np.nan
        if not np.isfinite(effect.ci_lower) or not np.isfinite(effect.ci_upper)
        else float(effect.ci_upper - effect.ci_lower)
    )
    diagnostics.update(
        {
            "se_pooled_row": float(effect.se),
            "ci_pooled_lower": float(effect.ci_lower),
            "ci_pooled_upper": float(effect.ci_upper),
            "ci_pooled_length": pooled_length,
            "pooled_row_total_variance": float(effect.se**2),
            "pooled_row_historical_variance": float(
                max(
                    effect.se**2
                    - _target_sample_variance(*target_predictions),
                    0.0,
                )
            ),
        }
    )
    reported_se = se_if_total
    reported_method = str(diagnostics["if_variance_method"])
    if structural_linearized_scores is not None:
        structural_diagnostics = _paper_if_se_diagnostics(
            scores=np.asarray(structural_linearized_scores, dtype=float),
            Z=Z,
            target_predictions=target_predictions,
        )
        structural_se = float(structural_diagnostics["se_if_total"])
        structural_ci_lower, structural_ci_upper = normal_ci(effect.estimate, structural_se)
        diagnostics.update(
            {
                "se_structural_linearized": structural_se,
                "se_structural_linearized_historical": float(structural_diagnostics["se_if_historical"]),
                "se_structural_linearized_target_sample": float(structural_diagnostics["se_target_sample"]),
                "ci_structural_linearized_lower": float(structural_ci_lower),
                "ci_structural_linearized_upper": float(structural_ci_upper),
                "ci_structural_linearized_length": float(structural_ci_upper - structural_ci_lower),
            }
        )
    diagnostics["reported_if_variance_method"] = reported_method
    ci_lower, ci_upper = normal_ci(effect.estimate, reported_se)
    return (
        replace(
            effect,
            method_name=method_name,
            se=reported_se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        ),
        diagnostics,
    )


def _single_split_mean(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> FunctionalEstimate:
    return _single_split_mean_fit(
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_new,
        config=config,
        random_state=random_state,
    ).estimate


def _single_split_mean_fit(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
    structural_scores: CrossFittedStructuralScores | None = None,
    outer_folds: np.ndarray | None = None,
) -> SingleSplitMeanFit:
    folds = (
        make_stratified_folds(Z, n_splits=2, random_state=random_state)
        if outer_folds is None
        else np.asarray(outer_folds, dtype=int)
    )
    if folds.shape[0] != X.shape[0]:
        raise ValueError("outer_folds must have one entry per historical row.")
    train_mask = folds == 0
    eval_mask = folds == 1
    eval_fraction = float(np.mean(eval_mask))
    if eval_fraction <= 0.0:
        raise ValueError("single-split npJIVE requires a non-empty evaluation fold.")
    if structural_scores is None:
        structural_scores = _cross_fitted_structural_scores(
            X=X,
            Z=Z,
            Y=Y,
            config=config,
            random_state=random_state + 101,
        )

    correction_weights = np.zeros(X.shape[0], dtype=float)
    target_predictions = np.zeros(X_new.shape[0], dtype=float)
    fold_duals: dict[int, DualFitResult] = {}
    n_scored = int(np.sum(eval_mask))
    for fold_id in np.unique(structural_scores.folds):
        nuisance_eval = structural_scores.folds == fold_id
        nuisance_train = ~nuisance_eval
        active = eval_mask & nuisance_eval
        dual = _fit_dual_nuisance_for_config(
            X[nuisance_train],
            Z[nuisance_train],
            X_new,
            config=config,
            n_splits=_nuisance_n_splits_for_codes(
                Z[nuisance_train],
                config.n_splits,
            ),
            selection="npjive",
            random_state=random_state + 2_000 + int(fold_id),
            strict_stability=True,
        )
        fold_duals[int(fold_id)] = dual
        train_group_scores = group_means(
            X[train_mask] @ dual.coef_npjive,
            Z[train_mask],
            n_groups=int(np.max(Z)) + 1,
        )
        correction_weights[active] = (
            train_group_scores[Z[active]] / eval_fraction
        )
        target_predictions += (
            float(np.sum(active) / n_scored)
            * (X_new @ structural_scores.fold_coefs[int(fold_id)])
        )

    influence = correction_weights * (Y - structural_scores.h_hat)
    plugin = float(np.mean(target_predictions))
    ipw = float(np.mean(correction_weights * Y))
    point_estimate = float(plugin + np.mean(influence))
    se = float(
        np.sqrt(
            prediction_variance(influence) / X.shape[0]
            + prediction_variance(target_predictions) / X_new.shape[0]
        )
    )
    ci_lower, ci_upper = normal_ci(point_estimate, se)
    estimate = FunctionalEstimate(
        method_name="single_split_npjive",
        estimate=point_estimate,
        plugin_estimate=plugin,
        ipw_estimate=ipw,
        se=se,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        influence_function=influence,
    )
    return SingleSplitMeanFit(
        estimate=estimate,
        structural_scores=structural_scores,
        fold_duals=fold_duals,
        correction_weights=correction_weights,
        target_predictions=target_predictions,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )


def _cross_fitted_structural_scores(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> CrossFittedStructuralScores:
    n_splits = _nuisance_n_splits_for_codes(Z, config.n_splits)
    folds = make_stratified_folds(Z, n_splits=n_splits, random_state=random_state)
    h_hat = np.zeros(X.shape[0], dtype=float)
    fold_coefs: dict[int, np.ndarray] = {}
    fold_fits: dict[int, StructuralFitResult] = {}
    for fold_id in np.unique(folds):
        eval_mask = folds == fold_id
        train_mask = ~eval_mask
        structural = _fit_structural_nuisance_for_config(
            X[train_mask],
            Z[train_mask],
            Y[train_mask],
            config=config,
            n_splits=_nuisance_n_splits_for_codes(Z[train_mask], config.n_splits),
            selection="adaptive",
            random_state=random_state + 1_000 + int(fold_id),
        )
        coef = structural.coef_npjive
        h_hat[eval_mask] = X[eval_mask] @ coef
        fold_coefs[int(fold_id)] = coef
        fold_fits[int(fold_id)] = structural
    return CrossFittedStructuralScores(
        folds=folds,
        h_hat=h_hat,
        fold_coefs=fold_coefs,
        fold_fits=fold_fits,
    )


def _cross_fitted_loo_mean(
    *,
    full_structural_fit: StructuralFitResult,
    structural_scores: CrossFittedStructuralScores,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
    method_name: str = "loo_npjive",
) -> FunctionalEstimate:
    del full_structural_fit  # Kept in the private signature for call-site compatibility.
    weights = np.zeros(Y.shape[0], dtype=float)
    fold_target_predictions: list[np.ndarray] = []
    for fold_id in np.unique(structural_scores.folds):
        eval_mask = structural_scores.folds == fold_id
        train_mask = ~eval_mask
        dual = _fit_dual_nuisance_for_config(
            X[train_mask],
            Z[train_mask],
            X_new,
            config=config,
            n_splits=_nuisance_n_splits_for_codes(Z[train_mask], config.n_splits),
            selection="adaptive",
            random_state=random_state + 2_000 + int(fold_id),
        )
        beta_scores = X @ dual.coef_npjive
        fold_weights = leave_one_out_group_means(beta_scores, Z)
        weights[eval_mask] = fold_weights[eval_mask]
        fold_target_predictions.append(X_new @ structural_scores.fold_coefs[int(fold_id)])

    h_hat_new = np.mean(np.stack(fold_target_predictions, axis=0), axis=0)
    influence = weights * (Y - structural_scores.h_hat)
    plugin = float(np.mean(h_hat_new))
    ipw = float(np.mean(weights * Y))
    estimate = float(plugin + np.mean(influence))
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


def _cross_fitted_projection_corrected_contrast(
    *,
    base: FunctionalEstimate,
    structural_scores: CrossFittedStructuralScores,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_treated: np.ndarray,
    X_control: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
    method_name: str,
    cell_mean_train_mask: np.ndarray | None = None,
    score_mask: np.ndarray | None = None,
) -> StarMeanFit:
    if (cell_mean_train_mask is None) != (score_mask is None):
        raise ValueError("cell-mean and score masks must be supplied together.")
    single_split_cell_means = score_mask is not None
    if single_split_cell_means:
        cell_mean_mask = np.asarray(cell_mean_train_mask, dtype=bool)
        active_score_mask = np.asarray(score_mask, dtype=bool)
        if cell_mean_mask.shape[0] != X.shape[0] or active_score_mask.shape[0] != X.shape[0]:
            raise ValueError("single-split masks must have one entry per row.")
        if np.any(cell_mean_mask & active_score_mask):
            raise ValueError("cell-mean and score rows must be disjoint.")
        eval_fraction = float(np.mean(active_score_mask))
        if eval_fraction <= 0.0:
            raise ValueError("single-split corrected estimator needs scored rows.")
        n_scored = int(np.sum(active_score_mask))
    else:
        cell_mean_mask = np.ones(X.shape[0], dtype=bool)
        active_score_mask = np.ones(X.shape[0], dtype=bool)
        eval_fraction = 1.0
        n_scored = X.shape[0]

    target_feature_mean = np.mean(X_treated, axis=0) - np.mean(X_control, axis=0)
    raw_primary = np.zeros(X.shape[0], dtype=float)
    raw_correction = np.zeros(X.shape[0], dtype=float)
    primary_ipw_scores = np.zeros(X.shape[0], dtype=float)
    fold_plugins: list[float] = []
    fold_plugin_weights: list[float] = []
    fold_theta_coefs: list[np.ndarray] = []
    alpha_norms: list[float] = []
    alpha_residuals: list[float] = []
    alpha_gammas: list[float] = []
    projection_lambdas: list[float] = []
    projection_gammas: list[float] = []
    projection_risks: list[float] = []
    projection_norms: list[float] = []
    projection_eigen_mins: list[float] = []
    projection_conditions: list[float] = []
    projection_negative_edges: list[float] = []
    projection_fold_deviations: list[float] = []
    projection_retained_ranks: list[float] = []
    projection_largest_discarded: list[float] = []
    alpha_perp_norms: list[float] = []
    alpha_perp_group_norms: list[float] = []
    beta_lambdas: list[float] = []
    beta_gammas: list[float] = []
    beta_norms: list[float] = []
    beta_eigen_mins: list[float] = []
    beta_conditions: list[float] = []
    beta_noise_floors: list[float] = []
    beta_generalized_eigen_mins: list[float] = []
    beta_generalized_conditions: list[float] = []
    rho_lambdas: list[float] = []
    rho_gammas: list[float] = []
    rho_norms: list[float] = []
    rho_eigen_mins: list[float] = []
    rho_conditions: list[float] = []
    rho_noise_floors: list[float] = []
    rho_generalized_eigen_mins: list[float] = []
    rho_generalized_conditions: list[float] = []
    projection_objectives: list[str] = []
    target_gradient_correction = np.zeros(X.shape[1], dtype=float)
    if single_split_cell_means:
        cell_features = group_means(
            X[cell_mean_mask],
            Z[cell_mean_mask],
            n_groups=int(np.max(Z)) + 1,
        )
        empirical_feature_scores = cell_features[Z]
    else:
        empirical_feature_scores = leave_one_out_group_means(X, Z)

    for fold_id in np.unique(structural_scores.folds):
        nuisance_eval_mask = structural_scores.folds == fold_id
        nuisance_train_mask = ~nuisance_eval_mask
        fold_score_mask = nuisance_eval_mask & active_score_mask
        X_fit = X[nuisance_train_mask]
        Z_fit = Z[nuisance_train_mask]
        theta_coef = structural_scores.fold_coefs[int(fold_id)]
        C = X_fit.T @ X_fit / X_fit.shape[0]
        alpha_fit = _fit_riesz_representer_for_feature_mean(
            X=X_fit,
            Z=Z_fit,
            target_feature_mean=target_feature_mean,
            config=config,
            random_state=random_state + 3_000 + int(fold_id),
        )
        projection_fit = _fit_riesz_projection(
            alpha_coef=alpha_fit.coef,
            X=X_fit,
            Z=Z_fit,
            config=config,
            random_state=random_state + 3_500 + int(fold_id),
            target_feature_mean=target_feature_mean,
            alpha_gamma=alpha_fit.gamma,
        )
        pi_alpha_coef = projection_fit.projected_coef
        alpha_perp_coef = projection_fit.residual_coef
        beta_fit = _fit_strict_npjive_inverse_for_linear_term(
            X_fit,
            Z_fit,
            C @ pi_alpha_coef,
            config=config,
            random_state=random_state + 3_750 + int(fold_id),
        )
        rho_fit = _fit_strict_npjive_inverse_for_linear_term(
            X_fit,
            Z_fit,
            C @ theta_coef,
            config=config,
            # The stability floor and tuning folds depend only on the observed
            # operator, so beta and rho intentionally share them.
            random_state=random_state + 3_750 + int(fold_id),
        )
        alpha_perp_scores = X @ alpha_perp_coef
        q_scores = empirical_feature_scores @ beta_fit.coef
        r_scores = empirical_feature_scores @ rho_fit.coef
        outcome_residual = (
            Y[fold_score_mask] - structural_scores.h_hat[fold_score_mask]
        )
        projected_inverse_map = (
            beta_fit.inverse_operator
            @ C
            @ projection_fit.projection_operator
            @ alpha_fit.linear_operator
        )
        residual_riesz_map = (
            np.eye(X.shape[1]) - projection_fit.projection_operator
        ) @ alpha_fit.linear_operator
        target_gradient_correction += projected_inverse_map.T @ (
            empirical_feature_scores[fold_score_mask].T
            @ outcome_residual
            / (X.shape[0] * eval_fraction)
        )
        target_gradient_correction += residual_riesz_map.T @ (
            X[fold_score_mask].T
            @ (
                r_scores[fold_score_mask]
                - structural_scores.h_hat[fold_score_mask]
            )
            / (X.shape[0] * eval_fraction)
        )
        raw_primary[fold_score_mask] = (
            q_scores[fold_score_mask]
            * outcome_residual
            / eval_fraction
        )
        primary_ipw_scores[fold_score_mask] = (
            q_scores[fold_score_mask] * Y[fold_score_mask] / eval_fraction
        )
        raw_correction[fold_score_mask] = (
            alpha_perp_scores[fold_score_mask]
            * (
                r_scores[fold_score_mask]
                - structural_scores.h_hat[fold_score_mask]
            )
            / eval_fraction
        )
        fold_plugins.append(float(target_feature_mean @ theta_coef))
        fold_plugin_weights.append(float(np.sum(fold_score_mask) / n_scored))
        fold_theta_coefs.append(theta_coef)
        alpha_norms.append(alpha_fit.coef_norm)
        alpha_residuals.append(alpha_fit.moment_residual_norm)
        alpha_gammas.append(alpha_fit.gamma)
        projection_lambdas.append(projection_fit.lambda_value)
        projection_gammas.append(projection_fit.gamma)
        projection_risks.append(projection_fit.cv_risk)
        projection_norms.append(projection_fit.projected_coef_norm)
        projection_eigen_mins.append(projection_fit.system_eigen_min)
        projection_conditions.append(projection_fit.system_condition)
        projection_negative_edges.append(projection_fit.negative_spectral_edge)
        projection_fold_deviations.append(projection_fit.fold_operator_deviation)
        projection_retained_ranks.append(projection_fit.retained_rank)
        projection_largest_discarded.append(projection_fit.largest_discarded_eigenvalue)
        alpha_perp_norms.append(projection_fit.residual_coef_norm)
        alpha_perp_group_norms.append(projection_fit.residual_group_mean_norm)
        beta_lambdas.append(beta_fit.lambda_value)
        beta_gammas.append(beta_fit.gamma)
        beta_norms.append(beta_fit.coef_norm)
        beta_eigen_mins.append(beta_fit.system_eigen_min)
        beta_conditions.append(beta_fit.system_condition)
        beta_noise_floors.append(beta_fit.operator_noise_floor)
        beta_generalized_eigen_mins.append(beta_fit.generalized_system_eigen_min)
        beta_generalized_conditions.append(beta_fit.generalized_system_condition)
        rho_lambdas.append(rho_fit.lambda_value)
        rho_gammas.append(rho_fit.gamma)
        rho_norms.append(rho_fit.coef_norm)
        rho_eigen_mins.append(rho_fit.system_eigen_min)
        rho_conditions.append(rho_fit.system_condition)
        rho_noise_floors.append(rho_fit.operator_noise_floor)
        rho_generalized_eigen_mins.append(rho_fit.generalized_system_eigen_min)
        rho_generalized_conditions.append(rho_fit.generalized_system_condition)
        projection_objectives.append(projection_fit.objective)

    correction = _center_scores_within_instrument(
        raw_correction,
        Z,
        mask=active_score_mask if single_split_cell_means else None,
    )
    influence = raw_primary + correction
    plugin = (
        float(np.dot(fold_plugin_weights, fold_plugins))
        if single_split_cell_means
        else float(np.mean(fold_plugins))
    )
    correction_shift = float(np.mean(raw_correction))
    projected_primary_estimate = float(plugin + np.mean(raw_primary))
    estimate = float(projected_primary_estimate + correction_shift)
    estimate_shift = float(estimate - base.estimate)
    ipw = float(np.mean(primary_ipw_scores) + correction_shift)
    theta_stack = np.stack(fold_theta_coefs, axis=0)
    mean_theta_coef = (
        np.asarray(fold_plugin_weights, dtype=float) @ theta_stack
        if single_split_cell_means
        else np.mean(theta_stack, axis=0)
    )
    structural_fold_fits = list(structural_scores.fold_fits.values())
    structural_fold_coef_norms = [
        float(np.linalg.norm(fit.coef_npjive)) for fit in structural_fold_fits
    ]
    structural_fold_choices = [fit.tuning_npjive for fit in structural_fold_fits]
    target_gradient_coef = mean_theta_coef + target_gradient_correction
    se = float(
        np.sqrt(
            prediction_variance(influence) / X.shape[0]
            + _target_sample_variance(
                X_treated @ target_gradient_coef,
                X_control @ target_gradient_coef,
            )
        )
    )
    ci_lower, ci_upper = normal_ci(estimate, se)
    diagnostics = {
        "estimate_shift": estimate_shift,
        "ipw_shift": float(ipw - base.ipw_estimate),
        "structural_npjive_coef_norm_max": float(np.max(structural_fold_coef_norms)),
        "structural_npjive_operator_noise_floor": float(
            np.max([choice.operator_noise_floor for choice in structural_fold_choices])
        ),
        "structural_npjive_generalized_system_eigen_min": float(
            np.min([choice.generalized_system_eigen_min for choice in structural_fold_choices])
        ),
        "structural_npjive_generalized_system_condition": float(
            np.max([choice.generalized_system_condition for choice in structural_fold_choices])
        ),
        "alpha_gamma": float(np.median(alpha_gammas)),
        "alpha_cv_risk": np.nan,
        "alpha_coef_norm": float(np.mean(alpha_norms)),
        "alpha_moment_residual_norm": float(np.mean(alpha_residuals)),
        "pi_alpha_lambda": _finite_statistic(projection_lambdas, "median"),
        "pi_alpha_gamma": _finite_statistic(projection_gammas, "median"),
        "pi_alpha_cv_risk": _finite_statistic(projection_risks, "mean"),
        "pi_alpha_coef_norm": float(np.mean(projection_norms)),
        "alpha_perp_coef_norm": float(np.mean(alpha_perp_norms)),
        "alpha_perp_group_mean_norm": float(np.mean(alpha_perp_group_norms)),
        "pi_alpha_system_eigen_min": _finite_statistic(projection_eigen_mins, "min"),
        "pi_alpha_system_condition": _finite_statistic(projection_conditions, "max"),
        "pi_alpha_negative_spectral_edge": _finite_statistic(
            projection_negative_edges,
            "max",
        ),
        "pi_alpha_fold_operator_deviation": _finite_statistic(
            projection_fold_deviations,
            "max",
        ),
        "pi_alpha_retained_rank": _finite_statistic(projection_retained_ranks, "median"),
        "pi_alpha_largest_discarded_eigenvalue": _finite_statistic(
            projection_largest_discarded,
            "max",
        ),
        "pi_alpha_spectral_threshold": _finite_statistic(projection_lambdas, "median"),
        "target_feature_mean_norm": float(np.linalg.norm(target_feature_mean)),
        "beta_lambda": float(np.median(beta_lambdas)),
        "beta_gamma": float(np.median(beta_gammas)),
        "beta_cv_risk": np.nan,
        "beta_coef_norm": float(np.mean(beta_norms)),
        "beta_coef_norm_max": float(np.max(beta_norms)),
        "beta_system_eigen_min": float(np.min(beta_eigen_mins)),
        "beta_system_condition": float(np.max(beta_conditions)),
        "beta_operator_noise_floor": float(np.max(beta_noise_floors)),
        "beta_generalized_system_eigen_min": float(np.min(beta_generalized_eigen_mins)),
        "beta_generalized_system_condition": float(np.max(beta_generalized_conditions)),
        "rho_lambda": float(np.median(rho_lambdas)),
        "rho_gamma": float(np.median(rho_gammas)),
        "rho_cv_risk": np.nan,
        "rho_coef_norm": float(np.mean(rho_norms)),
        "rho_coef_norm_max": float(np.max(rho_norms)),
        "rho_system_eigen_min": float(np.min(rho_eigen_mins)),
        "rho_system_condition": float(np.max(rho_conditions)),
        "rho_operator_noise_floor": float(np.max(rho_noise_floors)),
        "rho_generalized_system_eigen_min": float(np.min(rho_generalized_eigen_mins)),
        "rho_generalized_system_condition": float(np.max(rho_generalized_conditions)),
        "direct_contrast_star": True,
        "nuisance_crossfit": (
            "five_fold_nuisance_crossfit_plus_single_split_cell_means"
            if single_split_cell_means
            else "five_fold_out_of_fold"
        ),
        "alpha_objective": "crossfit_contrast_ridge_riesz_cv",
        "pi_alpha_objective": projection_objectives[0],
        "beta_objective": "crossfit_npjive_dual_cv_for_projected_riesz_representer",
        "rho_objective": "crossfit_npjive_dual_cv_for_structural_quadratic_functional",
        "projected_primary_shift": float(projected_primary_estimate - base.estimate),
        "correction_split": (
            "five_fold_nuisance_crossfit_plus_single_split_cell_means"
            if single_split_cell_means
            else "five_fold_crossfit_plus_leave_one_out_empirical_means"
        ),
        "cell_mean_source": (
            "outer_training_half"
            if single_split_cell_means
            else "leave_one_out_or_fold_excluded_all_rows"
        ),
        "score_fraction": float(eval_fraction),
        "nuisance_crossfit_folds": int(np.unique(structural_scores.folds).size),
        "target_gradient_coef_norm": float(np.linalg.norm(target_gradient_coef)),
        "target_gradient_structural_coef_norm": float(np.linalg.norm(mean_theta_coef)),
        "target_gradient_linearization_error": float(
            estimate - target_feature_mean @ target_gradient_coef
        ),
    }
    diagnostics.update(_star_component_diagnostics(raw_primary, correction))
    diagnostics["raw_correction_mean"] = correction_shift
    diagnostics["raw_correction_abs_mean"] = float(np.mean(np.abs(raw_correction)))
    diagnostics["raw_correction_sd"] = (
        float(np.std(raw_correction, ddof=1)) if raw_correction.shape[0] > 1 else 0.0
    )
    return StarMeanFit(
        estimate=FunctionalEstimate(
            method_name=method_name,
            estimate=estimate,
            plugin_estimate=plugin,
            ipw_estimate=ipw,
            se=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            influence_function=influence,
        ),
        diagnostics=diagnostics,
        target_gradient_coef=target_gradient_coef,
    )


def _fit_effect_bundle(
    config: LongTermSurrogateConfig,
    data: dict[str, Any],
    *,
    replication: int,
) -> EffectFitBundle:
    X = np.asarray(data["X"], dtype=float)
    Z = np.asarray(data["Z"], dtype=int)
    Y = np.asarray(data["Y"], dtype=float)
    X_treated = np.asarray(data["X_new_treated"], dtype=float)
    X_control = np.asarray(data["X_new_control"], dtype=float)
    random_state = config.seed_base + 20_000 + replication

    structural = _fit_structural_nuisance_for_config(
        X,
        Z,
        Y,
        config=config,
        n_splits=_nuisance_n_splits_for_codes(Z, config.n_splits),
        selection="adaptive",
        random_state=random_state,
    )
    treated_dual = _fit_dual_nuisance_for_config(
        X,
        Z,
        X_treated,
        config=config,
        n_splits=_nuisance_n_splits_for_codes(Z, config.n_splits),
        selection="adaptive",
        random_state=random_state,
    )
    control_dual = _fit_dual_nuisance_for_config(
        X,
        Z,
        X_control,
        config=config,
        n_splits=_nuisance_n_splits_for_codes(Z, config.n_splits),
        selection="adaptive",
        random_state=random_state,
    )
    loo_structural_crossfit = _cross_fitted_structural_scores(
        X=X,
        Z=Z,
        Y=Y,
        config=config,
        random_state=random_state + 700,
    )
    single_split_random_state = config.seed_base + 30_000 + replication
    single_outer_folds = make_stratified_folds(
        Z,
        n_splits=2,
        random_state=single_split_random_state,
    )
    single_structural_crossfit = _cross_fitted_structural_scores(
        X=X,
        Z=Z,
        Y=Y,
        config=config,
        random_state=single_split_random_state + 101,
    )
    treated_loo = _cross_fitted_loo_mean(
        method_name="loo_npjive",
        full_structural_fit=structural,
        structural_scores=loo_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_treated,
        config=config,
        random_state=random_state + 801,
    )
    control_loo = _cross_fitted_loo_mean(
        method_name="loo_npjive",
        full_structural_fit=structural,
        structural_scores=loo_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_control,
        config=config,
        random_state=random_state + 802,
    )
    treated_2sls = _build_functional_estimate(
        method_name="2sls_dml",
        theta_coef=structural.coef_2sls,
        beta_coef=treated_dual.coef_2sls,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_treated,
        use_leave_one_out=False,
    )
    control_2sls = _build_functional_estimate(
        method_name="2sls_dml",
        theta_coef=structural.coef_2sls,
        beta_coef=control_dual.coef_2sls,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_control,
        use_leave_one_out=False,
    )
    treated_single = _single_split_mean_fit(
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_treated,
        config=config,
        random_state=single_split_random_state,
        structural_scores=single_structural_crossfit,
        outer_folds=single_outer_folds,
    )
    control_single = _single_split_mean_fit(
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_control,
        config=config,
        random_state=single_split_random_state,
        structural_scores=single_structural_crossfit,
        outer_folds=single_outer_folds,
    )
    if not np.array_equal(treated_single.eval_mask, control_single.eval_mask):
        raise RuntimeError("treated and control single-split fits must use the same evaluation fold.")
    crossfit_theta_coef = np.mean(
        np.stack(list(loo_structural_crossfit.fold_coefs.values()), axis=0),
        axis=0,
    )
    treated_pred_npjive = X_treated @ crossfit_theta_coef
    control_pred_npjive = X_control @ crossfit_theta_coef
    treated_pred_single = treated_single.target_predictions
    control_pred_single = control_single.target_predictions
    treated_pred_2sls = X_treated @ structural.coef_2sls
    control_pred_2sls = X_control @ structural.coef_2sls
    loo_treated_weights = _loo_correction_weights(X=X, Z=Z, beta_coef=treated_dual.coef_npjive)
    loo_control_weights = _loo_correction_weights(X=X, Z=Z, beta_coef=control_dual.coef_npjive)
    loo_structural_scores = _structural_linearized_effect_scores(
        structural_fit=structural,
        X=X,
        Z=Z,
        Y=Y,
        X_treated=X_treated,
        X_control=X_control,
        treated_weights=loo_treated_weights,
        control_weights=loo_control_weights,
    )
    single_treated_weights = treated_single.correction_weights
    single_control_weights = control_single.correction_weights
    single_structural_scores = _cross_fitted_single_split_structural_effect_scores(
        structural_scores=single_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_treated=X_treated,
        X_control=X_control,
        treated_weights=single_treated_weights,
        control_weights=single_control_weights,
        score_mask=treated_single.eval_mask,
    )
    baseline_effect, baseline_inference = _contrast_estimate_with_paper_if_se(
        method_name="2sls_dml",
        treated=treated_2sls,
        control=control_2sls,
        treated_predictions=treated_pred_2sls,
        control_predictions=control_pred_2sls,
        Z=Z,
    )
    single_effect, single_inference = _contrast_estimate_with_paper_if_se(
        method_name="single_split_npjive",
        treated=treated_single.estimate,
        control=control_single.estimate,
        treated_predictions=treated_pred_single,
        control_predictions=control_pred_single,
        Z=Z,
        eval_mask=treated_single.eval_mask,
        structural_linearized_scores=single_structural_scores,
    )
    single_inference.update(
        {
            "nuisance_crossfit": (
                "five_fold_nuisance_crossfit_plus_single_split_cell_means"
            ),
            "cell_mean_source": "outer_training_half",
            "score_fraction": float(np.mean(treated_single.eval_mask)),
            "nuisance_crossfit_folds": int(
                np.unique(single_structural_crossfit.folds).size
            ),
        }
    )
    loo_effect, loo_inference = _contrast_estimate_with_paper_if_se(
        method_name="loo_npjive",
        treated=treated_loo,
        control=control_loo,
        treated_predictions=treated_pred_npjive,
        control_predictions=control_pred_npjive,
        Z=Z,
        structural_linearized_scores=loo_structural_scores,
    )
    loo_star_fit = _cross_fitted_projection_corrected_contrast(
        base=loo_effect,
        structural_scores=loo_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_treated=X_treated,
        X_control=X_control,
        config=config,
        random_state=random_state + 401,
        method_name="loo_npjive_star",
    )
    single_star_fit = _cross_fitted_projection_corrected_contrast(
        base=single_effect,
        structural_scores=single_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_treated=X_treated,
        X_control=X_control,
        config=config,
        random_state=random_state + 501,
        method_name="single_split_npjive_star",
        cell_mean_train_mask=treated_single.train_mask,
        score_mask=treated_single.eval_mask,
    )
    effect_star_correction = loo_star_fit.estimate.influence_function - loo_effect.influence_function
    effect_single_star_correction = single_star_fit.estimate.influence_function - single_effect.influence_function
    loo_star_structural_scores = loo_structural_scores + effect_star_correction
    single_star_structural_scores = single_structural_scores + effect_single_star_correction
    effect_star_diagnostics = dict(loo_star_fit.diagnostics)
    effect_single_star_diagnostics = dict(single_star_fit.diagnostics)
    single_star_effect, single_star_inference = _effect_estimate_with_paper_if_se(
        method_name="single_split_npjive_star",
        effect=single_star_fit.estimate,
        target_predictions=(
            X_treated @ single_star_fit.target_gradient_coef,
            X_control @ single_star_fit.target_gradient_coef,
        ),
        Z=Z,
        eval_mask=treated_single.eval_mask,
        structural_linearized_scores=single_star_structural_scores,
    )
    loo_star_effect, loo_star_inference = _effect_estimate_with_paper_if_se(
        method_name="loo_npjive_star",
        effect=loo_star_fit.estimate,
        target_predictions=(
            X_treated @ loo_star_fit.target_gradient_coef,
            X_control @ loo_star_fit.target_gradient_coef,
        ),
        Z=Z,
        structural_linearized_scores=loo_star_structural_scores,
    )
    estimates = {
        "2sls_dml": baseline_effect,
        "single_split_npjive": single_effect,
        "single_split_npjive_star": single_star_effect,
        "loo_npjive": loo_effect,
        "loo_npjive_star": loo_star_effect,
    }
    inference_diagnostics = {
        "2sls_dml": baseline_inference,
        "single_split_npjive": single_inference,
        "single_split_npjive_star": single_star_inference,
        "loo_npjive": loo_inference,
        "loo_npjive_star": loo_star_inference,
    }
    return EffectFitBundle(
        estimates=estimates,
        structural=structural,
        treated_dual=treated_dual,
        control_dual=control_dual,
        star_diagnostics={
            "treated": {},
            "control": {},
            "effect": effect_star_diagnostics,
            "single_split_treated": {},
            "single_split_control": {},
            "single_split_effect": effect_single_star_diagnostics,
        },
        star_target_gradients={
            "single_split_npjive_star": single_star_fit.target_gradient_coef,
            "loo_npjive_star": loo_star_fit.target_gradient_coef,
        },
        inference_diagnostics=inference_diagnostics,
    )
