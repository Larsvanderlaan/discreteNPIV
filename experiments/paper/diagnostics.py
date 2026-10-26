from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from discreteNPIV._linear_solvers import prediction_variance
from discreteNPIV._version import __version__
from discreteNPIV.grouping import group_means
from discreteNPIV.results import DualFitResult, FunctionalEstimate, StructuralFitResult

from .design import _design_diagnostics, generate_longterm_surrogate_data
from .estimation import _fit_effect_bundle
from .model import (
    EffectFitBundle,
    LongTermSurrogateConfig,
    REPORT_METHODS,
    _dual_gamma_grid,
    _dual_lambda_grid,
    _structural_gamma_grid,
    _structural_lambda_grid,
    _target_estimand_for_method,
    config_hash,
    config_to_dict,
)


def _influence_diagnostics(X: np.ndarray, Z: np.ndarray, effect: FunctionalEstimate) -> dict[str, float]:
    scores = np.asarray(effect.influence_function, dtype=float)
    denom = float(np.sum(scores**2))
    effective_support = np.nan if denom == 0.0 else float((np.sum(np.abs(scores)) ** 2) / denom / scores.shape[0])
    grouped = group_means(scores, Z)
    return {
        "influence_norm": float(np.sqrt(np.mean(scores**2))),
        "influence_group_norm": float(np.sqrt(np.mean(grouped**2))),
        "influence_abs_p99": float(np.quantile(np.abs(scores), 0.99)),
        "influence_abs_max": float(np.max(np.abs(scores))),
        "influence_effective_support": effective_support,
    }


def _choice_fields(prefix: str, result: StructuralFitResult | DualFitResult) -> dict[str, Any]:
    return {
        f"{prefix}_selected_method": result.selected_method,
        f"{prefix}_npjive_lambda": result.tuning_npjive.lambda_value,
        f"{prefix}_npjive_gamma": result.tuning_npjive.gamma,
        f"{prefix}_npjive_cv_risk": result.tuning_npjive.cv_risk,
        f"{prefix}_npjive_operator_noise_floor": result.tuning_npjive.operator_noise_floor,
        f"{prefix}_npjive_operator_noise_margin": result.tuning_npjive.operator_noise_margin,
        f"{prefix}_npjive_generalized_system_eigen_min": (
            result.tuning_npjive.generalized_system_eigen_min
        ),
        f"{prefix}_npjive_generalized_system_condition": (
            result.tuning_npjive.generalized_system_condition
        ),
        f"{prefix}_2sls_lambda": result.tuning_2sls.lambda_value,
        f"{prefix}_2sls_gamma": result.tuning_2sls.gamma,
        f"{prefix}_2sls_cv_risk": result.tuning_2sls.cv_risk,
    }


def _lower_boundary_fields(
    prefix: str,
    result: StructuralFitResult | DualFitResult,
    config: LongTermSurrogateConfig,
) -> dict[str, Any]:
    if prefix == "structural":
        lambda_values = _structural_lambda_grid(config)
        gamma_values = _structural_gamma_grid(config)
    else:
        lambda_values = _dual_lambda_grid(config)
        gamma_values = _dual_gamma_grid(config)
    lambda_min = float(np.min(np.asarray(lambda_values, dtype=float)))
    gamma_min = float(np.min(np.asarray(gamma_values, dtype=float)))
    return {
        f"{prefix}_npjive_lambda_at_lower_boundary": bool(np.isclose(result.tuning_npjive.lambda_value, lambda_min)),
        f"{prefix}_npjive_gamma_at_lower_boundary": bool(np.isclose(result.tuning_npjive.gamma, gamma_min)),
        f"{prefix}_2sls_lambda_at_lower_boundary": bool(np.isclose(result.tuning_2sls.lambda_value, lambda_min)),
        f"{prefix}_2sls_gamma_at_lower_boundary": bool(np.isclose(result.tuning_2sls.gamma, gamma_min)),
    }


def _fit_metadata(bundle: EffectFitBundle, config: LongTermSurrogateConfig | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload.update(_choice_fields("structural", bundle.structural))
    payload.update(_choice_fields("treated_dual", bundle.treated_dual))
    payload.update(_choice_fields("control_dual", bundle.control_dual))
    if config is not None:
        payload.update(_lower_boundary_fields("structural", bundle.structural, config))
        payload.update(_lower_boundary_fields("treated_dual", bundle.treated_dual, config))
        payload.update(_lower_boundary_fields("control_dual", bundle.control_dual, config))
    return payload


def _base_row(
    *,
    config: LongTermSurrogateConfig,
    stage: str,
    replication: int,
    data: dict[str, Any],
    runtime_seconds: float,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = config_to_dict(config)
    if provenance:
        payload.update(provenance)
    payload.update(data.get("metadata", {}))
    payload.update(_design_diagnostics(np.asarray(data["X"], dtype=float), np.asarray(data["Z"], dtype=int)))
    payload.update(
        {
            "row_type": "estimate",
            "stage": stage,
            "package_version": __version__,
            "config_hash": config_hash(config),
            "replication": int(replication),
            "replication_seed": int(config.seed_base + 10_000 + replication),
            "truth": float(data["truth"]),
            "projection_truth": float(data["projection_truth"]),
            "identification_gap": float(data["identification_gap"]),
            "selection_rule": "deployable_cv_risk",
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return payload


def _estimate_row(
    *,
    base: dict[str, Any],
    method: str,
    estimate: FunctionalEstimate,
    X: np.ndarray,
    Z: np.ndarray,
    fit_metadata: dict[str, Any] | None = None,
    inference_diagnostics: dict[str, Any] | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = float(base["truth"])
    projection_truth = float(base["projection_truth"])
    target_estimand = _target_estimand_for_method(method)
    estimand_truth = projection_truth if target_estimand == "projection" else truth
    reported_inference_method = (
        "paper_if"
        if inference_diagnostics is None
        else str(inference_diagnostics.get("reported_if_variance_method", "paper_if"))
    )
    has_ci = np.isfinite(estimate.ci_lower) and np.isfinite(estimate.ci_upper)
    row = dict(base)
    if fit_metadata:
        row.update(fit_metadata)
    row.update(_influence_diagnostics(X, Z, estimate))
    if inference_diagnostics:
        row.update(inference_diagnostics)
    row.update(
        {
            "method": method,
            "method_role": "baseline" if method == "2sls_dml" else "reported",
            "target_estimand": target_estimand,
            "estimand_truth": estimand_truth,
            "reported_inference_method": reported_inference_method,
            "estimate": float(estimate.estimate),
            "error": float(estimate.estimate - truth),
            "projection_error": float(estimate.estimate - projection_truth),
            "estimand_error": float(estimate.estimate - estimand_truth),
            "plugin_error": float(estimate.plugin_estimate - truth),
            "plugin_estimand_error": float(estimate.plugin_estimate - estimand_truth),
            "residual_correction": float(estimate.estimate - estimate.plugin_estimate),
            "influence_variance": prediction_variance(np.asarray(estimate.influence_function, dtype=float)),
            "influence_variance_over_n": prediction_variance(np.asarray(estimate.influence_function, dtype=float))
            / max(float(np.asarray(estimate.influence_function).shape[0]), 1.0),
            "se": float(estimate.se),
            "ci_lower": float(estimate.ci_lower),
            "ci_upper": float(estimate.ci_upper),
            "ci_length": np.nan if not has_ci else float(estimate.ci_upper - estimate.ci_lower),
            "covered": np.nan if not has_ci else bool(estimate.ci_lower <= truth <= estimate.ci_upper),
            "projection_covered": (
                np.nan if not has_ci else bool(estimate.ci_lower <= projection_truth <= estimate.ci_upper)
            ),
            "estimand_covered": (
                np.nan if not has_ci else bool(estimate.ci_lower <= estimand_truth <= estimate.ci_upper)
            ),
            "estimand_gap_to_true": float(truth - estimand_truth),
            "normalized_identification_gap": (
                np.nan if not np.isfinite(estimate.se) or estimate.se <= 0.0 else abs(float(base["identification_gap"])) / estimate.se
            ),
            "nonfinite": not bool(np.isfinite(estimate.estimate) and np.isfinite(estimate.se)),
            "error_message": "",
        }
    )
    pooled_lower = float(row.get("ci_pooled_lower", np.nan))
    pooled_upper = float(row.get("ci_pooled_upper", np.nan))
    structural_lower = float(row.get("ci_structural_linearized_lower", np.nan))
    structural_upper = float(row.get("ci_structural_linearized_upper", np.nan))
    has_structural_ci = np.isfinite(structural_lower) and np.isfinite(structural_upper)
    has_pooled_ci = np.isfinite(pooled_lower) and np.isfinite(pooled_upper)
    row.update(
        {
            "covered_pooled": np.nan if not has_pooled_ci else bool(pooled_lower <= truth <= pooled_upper),
            "projection_covered_pooled": (
                np.nan if not has_pooled_ci else bool(pooled_lower <= projection_truth <= pooled_upper)
            ),
            "estimand_covered_pooled": (
                np.nan if not has_pooled_ci else bool(pooled_lower <= estimand_truth <= pooled_upper)
            ),
            "normalized_identification_gap_pooled": (
                np.nan
                if not np.isfinite(float(row.get("se_pooled_row", np.nan))) or float(row.get("se_pooled_row", np.nan)) <= 0.0
                else abs(float(base["identification_gap"])) / float(row["se_pooled_row"])
            ),
            "covered_structural_linearized": (
                np.nan if not has_structural_ci else bool(structural_lower <= truth <= structural_upper)
            ),
            "projection_covered_structural_linearized": (
                np.nan if not has_structural_ci else bool(structural_lower <= projection_truth <= structural_upper)
            ),
            "estimand_covered_structural_linearized": (
                np.nan if not has_structural_ci else bool(structural_lower <= estimand_truth <= structural_upper)
            ),
        }
    )
    if extra_diagnostics:
        for key, value in extra_diagnostics.items():
            if isinstance(value, (bool, str)) or np.isscalar(value):
                field = key if key.startswith("star_") else f"star_{key}"
                row[field] = value
    return row


def run_replication(
    config: LongTermSurrogateConfig,
    *,
    stage: str,
    replication: int,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the estimators reported in the paper for one replication."""

    started = perf_counter()
    try:
        data = generate_longterm_surrogate_data(config, replication)
        bundle = _fit_effect_bundle(
            config,
            data,
            replication=replication,
        )
        runtime = perf_counter() - started
        X = np.asarray(data["X"], dtype=float)
        Z = np.asarray(data["Z"], dtype=int)
        base = _base_row(
            config=config,
            stage=stage,
            replication=replication,
            data=data,
            runtime_seconds=runtime,
            provenance=provenance,
        )
        base["projection_estimation_mode"] = "estimated"
        methods = list(REPORT_METHODS)
        metadata = _fit_metadata(bundle, config)
        extra_by_method = {
            "loo_npjive_star": bundle.star_diagnostics.get("effect", {}),
        }
        rows = [
            _estimate_row(
                base=base,
                method=method,
                estimate=bundle.estimates[method],
                X=X,
                Z=Z,
                fit_metadata=metadata,
                inference_diagnostics=bundle.inference_diagnostics.get(method, {}),
                extra_diagnostics=extra_by_method.get(method),
            )
            for method in methods
        ]
        return rows
    except Exception as exc:  # pragma: no cover - retained for long unattended runs
        runtime = perf_counter() - started
        payload = config_to_dict(config)
        if provenance:
            payload.update(provenance)
        payload.update(
            {
                "row_type": "failure",
                "stage": stage,
                "package_version": __version__,
                "config_hash": config_hash(config),
                "replication": int(replication),
                "replication_seed": int(config.seed_base + 10_000 + replication),
                "method": "all",
                "projection_estimation_mode": "estimated",
                "runtime_seconds": runtime,
                "estimate": np.nan,
                "truth": np.nan,
                "projection_truth": np.nan,
                "identification_gap": np.nan,
                "target_estimand": "",
                "estimand_truth": np.nan,
                "error": np.nan,
                "projection_error": np.nan,
                "estimand_error": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "covered": np.nan,
                "projection_covered": np.nan,
                "estimand_covered": np.nan,
                "estimand_gap_to_true": np.nan,
                "nonfinite": True,
                "selection_rule": "deployable_cv_risk",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        )
        return [payload]
