from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from discreteNPIV._ci import normal_ci
from discreteNPIV._linear_solvers import prediction_variance
from discreteNPIV.api import _build_functional_estimate
from discreteNPIV.grouping import leave_one_out_group_means, make_stratified_folds
from discreteNPIV.results import DualFitResult, FunctionalEstimate, StructuralFitResult

from .design import _fit_riesz_representer_for_feature_mean
from .cell_variance import loo_cell_variance
from .inference import (
    _center_scores_within_instrument,
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
    LOOMeanFit,
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
    historical_variance: dict[str, Any] | None = None,
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
        historical_variance=historical_variance,
    )


def _effect_estimate_with_paper_if_se(
    *,
    method_name: str,
    effect: FunctionalEstimate,
    target_predictions: tuple[np.ndarray, ...],
    Z: np.ndarray,
    eval_mask: np.ndarray | None = None,
    structural_linearized_scores: np.ndarray | None = None,
    historical_variance: dict[str, Any] | None = None,
) -> tuple[FunctionalEstimate, dict[str, float | str]]:
    diagnostics = _paper_if_se_diagnostics(
        scores=effect.influence_function,
        Z=Z,
        target_predictions=target_predictions,
        eval_mask=eval_mask,
    )
    if historical_variance is not None:
        diagnostics.update(historical_variance)
        variance = float(diagnostics["if_historical_variance"])
        diagnostics["se_if_historical"] = float(np.sqrt(variance))
        diagnostics["if_total_variance"] = variance + float(diagnostics["target_sample_variance"])
        diagnostics["se_if_total"] = float(np.sqrt(diagnostics["if_total_variance"]))
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


def _cross_fitted_loo_mean_fit(
    *,
    structural_scores: CrossFittedStructuralScores,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    X_new: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
    method_name: str = "loo_npjive",
) -> LOOMeanFit:
    weights = np.zeros(Y.shape[0], dtype=float)
    target_predictions = np.zeros(X_new.shape[0], dtype=float)
    beta_predictions = np.zeros((X.shape[0], len(structural_scores.fold_coefs)))
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
        beta_predictions[:, int(fold_id)] = beta_scores
        fold_weights = leave_one_out_group_means(beta_scores, Z)
        weights[eval_mask] = fold_weights[eval_mask]
        target_predictions += float(np.mean(eval_mask)) * (X_new @ structural_scores.fold_coefs[int(fold_id)])

    h_hat_new = target_predictions
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
    estimate_result = FunctionalEstimate(
        method_name=method_name,
        estimate=estimate,
        plugin_estimate=plugin,
        ipw_estimate=ipw,
        se=se,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        influence_function=influence,
    )
    return LOOMeanFit(
        estimate=estimate_result,
        beta_predictions=beta_predictions,
        target_predictions=target_predictions,
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
) -> StarMeanFit:
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
    empirical_feature_scores = leave_one_out_group_means(X, Z)
    beta_predictions = np.zeros((X.shape[0], len(structural_scores.fold_coefs)))
    rho_predictions = np.zeros_like(beta_predictions)
    a_predictions = np.zeros(X.shape[0])

    for fold_id in np.unique(structural_scores.folds):
        nuisance_eval_mask = structural_scores.folds == fold_id
        nuisance_train_mask = ~nuisance_eval_mask
        fold_score_mask = nuisance_eval_mask
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
        beta_predictions[:, int(fold_id)] = X @ beta_fit.coef
        rho_predictions[:, int(fold_id)] = X @ rho_fit.coef
        a_predictions[fold_score_mask] = alpha_perp_scores[fold_score_mask]
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
            / X.shape[0]
        )
        target_gradient_correction += residual_riesz_map.T @ (
            X[fold_score_mask].T
            @ (
                r_scores[fold_score_mask]
                - structural_scores.h_hat[fold_score_mask]
            )
            / X.shape[0]
        )
        raw_primary[fold_score_mask] = (
            q_scores[fold_score_mask]
            * outcome_residual
        )
        primary_ipw_scores[fold_score_mask] = (
            q_scores[fold_score_mask] * Y[fold_score_mask]
        )
        raw_correction[fold_score_mask] = (
            alpha_perp_scores[fold_score_mask]
            * (
                r_scores[fold_score_mask]
                - structural_scores.h_hat[fold_score_mask]
            )
        )
        fold_plugins.append(float(target_feature_mean @ theta_coef))
        fold_plugin_weights.append(float(np.mean(fold_score_mask)))
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

    correction = _center_scores_within_instrument(raw_correction, Z)
    influence = raw_primary + correction
    plugin = float(np.dot(fold_plugin_weights, fold_plugins))
    correction_shift = float(np.mean(raw_correction))
    projected_primary_estimate = float(plugin + np.mean(raw_primary))
    estimate = float(projected_primary_estimate + correction_shift)
    estimate_shift = float(estimate - base.estimate)
    ipw = float(np.mean(primary_ipw_scores) + correction_shift)
    theta_stack = np.stack(fold_theta_coefs, axis=0)
    mean_theta_coef = np.asarray(fold_plugin_weights, dtype=float) @ theta_stack
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
        "nuisance_crossfit": "five_fold_out_of_fold",
        "alpha_objective": "crossfit_contrast_ridge_riesz_cv",
        "pi_alpha_objective": projection_objectives[0],
        "beta_objective": "crossfit_npjive_dual_cv_for_projected_riesz_representer",
        "rho_objective": "crossfit_npjive_dual_cv_for_structural_quadratic_functional",
        "projected_primary_shift": float(projected_primary_estimate - base.estimate),
        "correction_split": "five_fold_crossfit_plus_leave_one_out_empirical_means",
        "cell_mean_source": "leave_one_out_all_other_rows_in_cell",
        "score_fraction": 1.0,
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
    historical_variance = loo_cell_variance(
        Z=Z,
        nuisance_folds=structural_scores.folds,
        residuals=Y - structural_scores.h_hat,
        beta_predictions=beta_predictions,
        a=a_predictions,
        rho_predictions=rho_predictions,
        h=structural_scores.h_hat,
    )
    if not np.isclose(historical_variance["loo_cell_correction"], np.mean(raw_primary + raw_correction), rtol=1e-9, atol=1e-10):
        raise RuntimeError("LOO corrected point estimate and pair kernel disagree.")
    return StarMeanFit(
        historical_variance=historical_variance,
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
    treated_loo = _cross_fitted_loo_mean_fit(
        method_name="loo_npjive",
        structural_scores=loo_structural_crossfit,
        X=X,
        Z=Z,
        Y=Y,
        X_new=X_treated,
        config=config,
        random_state=random_state + 801,
    )
    control_loo = _cross_fitted_loo_mean_fit(
        method_name="loo_npjive",
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
    treated_pred_npjive = treated_loo.target_predictions
    control_pred_npjive = control_loo.target_predictions
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
    baseline_effect, baseline_inference = _contrast_estimate_with_paper_if_se(
        method_name="2sls_dml",
        treated=treated_2sls,
        control=control_2sls,
        treated_predictions=treated_pred_2sls,
        control_predictions=control_pred_2sls,
        Z=Z,
    )
    loo_historical_variance = loo_cell_variance(
        Z=Z,
        nuisance_folds=loo_structural_crossfit.folds,
        residuals=Y - loo_structural_crossfit.h_hat,
        beta_predictions=treated_loo.beta_predictions - control_loo.beta_predictions,
    )
    correction = (
        treated_loo.estimate.estimate - treated_loo.estimate.plugin_estimate
        - control_loo.estimate.estimate + control_loo.estimate.plugin_estimate
    )
    if not np.isclose(loo_historical_variance["loo_cell_correction"], correction, rtol=1e-9, atol=1e-10):
        raise RuntimeError("LOO uncorrected point estimate and pair kernel disagree.")
    loo_effect, loo_inference = _contrast_estimate_with_paper_if_se(
        method_name="loo_npjive",
        treated=treated_loo.estimate,
        control=control_loo.estimate,
        treated_predictions=treated_pred_npjive,
        control_predictions=control_pred_npjive,
        Z=Z,
        structural_linearized_scores=loo_structural_scores,
        historical_variance=loo_historical_variance,
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
    effect_star_correction = loo_star_fit.estimate.influence_function - loo_effect.influence_function
    loo_star_structural_scores = loo_structural_scores + effect_star_correction
    effect_star_diagnostics = dict(loo_star_fit.diagnostics)
    loo_star_effect, loo_star_inference = _effect_estimate_with_paper_if_se(
        method_name="loo_npjive_star",
        effect=loo_star_fit.estimate,
        target_predictions=(
            X_treated @ loo_star_fit.target_gradient_coef,
            X_control @ loo_star_fit.target_gradient_coef,
        ),
        Z=Z,
        structural_linearized_scores=loo_star_structural_scores,
        historical_variance=loo_star_fit.historical_variance,
    )
    estimates = {
        "2sls_dml": baseline_effect,
        "loo_npjive": loo_effect,
        "loo_npjive_star": loo_star_effect,
    }
    inference_diagnostics = {
        "2sls_dml": baseline_inference,
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
        },
        star_target_gradients={
            "loo_npjive_star": loo_star_fit.target_gradient_coef,
        },
        inference_diagnostics=inference_diagnostics,
    )
