from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._ci import normal_ci
from ._linear_solvers import prediction_variance
from .api import (
    _build_functional_estimate,
    _use_leave_one_out_for_dual,
    estimate_average_functional,
    fit_dual_nuisance,
    fit_structural_nuisance,
)
from .results import (
    ExperimentEncodingResult,
    FunctionalEstimate,
    LongTermEffectResult,
    LongTermMeanResult,
    NPIVInferenceResult,
)
from .validation import as_1d_float, as_2d_float, validate_target_features


def _make_object_array(values: Sequence[object]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


def _normalize_arm_key(key: object, *, experiment_id: object | None = None) -> tuple[str, str]:
    if experiment_id is not None:
        return f"{experiment_id}:{key}", str(experiment_id)

    if isinstance(key, str):
        if ":" not in key:
            raise ValueError(
                "Experiment-arm keys must be globally unique. Use values like "
                "'experiment_id:arm_id' or pass experiment_ids separately."
            )
        experiment, arm = key.split(":", 1)
        return f"{experiment}:{arm}", experiment

    if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 2:
        experiment = str(key[0])
        arm = str(key[1])
        return f"{experiment}:{arm}", experiment

    raise ValueError(
        "Could not parse an experiment-arm key. Use strings like "
        "'experiment_id:arm_id' or 2-tuples like ('experiment_id', 'arm_id')."
    )


def _looks_like_single_arm_tuple_key(row: object) -> bool:
    if not isinstance(row, (tuple, list, np.ndarray)) or len(row) != 2:
        return False
    first, second = row[0], row[1]
    if isinstance(first, (tuple, list, set, np.ndarray)) or isinstance(second, (tuple, list, set, np.ndarray)):
        return False
    if isinstance(first, str) and ":" in first:
        return False
    if isinstance(second, str) and ":" in second:
        return False
    return True


def _canonicalize_overlap_row(row: object, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(row, (str, bytes)) or _looks_like_single_arm_tuple_key(row):
        active_keys = [row]
    else:
        active_keys = list(row)

    canonical: list[str] = []
    experiments_seen: set[str] = set()
    for key in active_keys:
        arm_key, experiment = _normalize_arm_key(key)
        if experiment in experiments_seen:
            raise ValueError(
                "A row cannot belong to two arms from the same experiment. "
                f"Found duplicate experiment '{experiment}'."
            )
        experiments_seen.add(experiment)
        canonical.append(arm_key)

    if not canonical and not allow_empty:
        raise ValueError("Encountered an empty overlap row while allow_empty=False.")

    return tuple(sorted(canonical))


def _format_level(level: object) -> str:
    if isinstance(level, tuple):
        if len(level) == 0:
            return "<empty>"
        return " + ".join(level)
    return str(level)


def _build_encoding_result(
    *,
    mode: str,
    row_levels: Sequence[object],
    low_support_threshold: int,
) -> ExperimentEncodingResult:
    if low_support_threshold < 1:
        raise ValueError("low_support_threshold must be at least 1.")

    unique_levels = tuple(sorted(set(row_levels)))
    level_mapping = {level: idx for idx, level in enumerate(unique_levels)}
    codes = np.array([level_mapping[level] for level in row_levels], dtype=int)
    counts = np.bincount(codes, minlength=len(unique_levels)).astype(int)
    singleton_levels = tuple(level for level, count in zip(unique_levels, counts, strict=True) if count == 1)
    low_support_levels = tuple(
        level for level, count in zip(unique_levels, counts, strict=True) if count < low_support_threshold
    )
    return ExperimentEncodingResult(
        mode=mode,
        instrument_values=_make_object_array(row_levels),
        codes=codes,
        levels=unique_levels,
        counts=counts,
        low_support_threshold=int(low_support_threshold),
        singleton_levels=singleton_levels,
        low_support_levels=low_support_levels,
    )


def encode_experiment_arms(
    assignments: object,
    *,
    mode: str = "single",
    experiment_ids: object | None = None,
    arm_keys: object | None = None,
    allow_empty: bool = True,
    low_support_threshold: int = 5,
) -> ExperimentEncodingResult:
    """
    Encode historical experiment-arm assignments as a discrete instrument.

    Parameters
    ----------
    assignments:
        Single-mode:
            one arm label per row. Use globally unique keys such as
            ``"exp_17:treatment"`` or pass ``experiment_ids`` separately.
        Overlap-mode:
            either a sequence of active arm-key sets/tuples per row, or a
            binary membership matrix with ``arm_keys`` supplied.
    mode:
        ``"single"`` for one active arm per row, ``"overlap"`` for multiple
        concurrent experiments encoded as the active set.
    experiment_ids:
        Optional experiment identifier per row for single-mode inputs whose
        ``assignments`` are only local arm labels.
    arm_keys:
        Required when overlap-mode input is a binary membership matrix. Each key
        must already identify the experiment and arm, e.g. ``"exp_1:treat"`` or
        ``("exp_1", "treat")``.
    allow_empty:
        Whether the empty active set is allowed in overlap mode.
    low_support_threshold:
        Counts below this threshold are flagged in the returned diagnostics.
    """

    if mode not in {"single", "overlap"}:
        raise ValueError("mode must be either 'single' or 'overlap'.")

    if mode == "single":
        row_assignments = np.asarray(assignments, dtype=object)
        if row_assignments.ndim != 1:
            raise ValueError("Single-mode assignments must be a 1D array-like input.")
        if experiment_ids is not None:
            experiment_arr = np.asarray(experiment_ids, dtype=object)
            if experiment_arr.ndim != 1 or experiment_arr.shape[0] != row_assignments.shape[0]:
                raise ValueError("experiment_ids must be 1D and aligned row-wise with assignments.")
        else:
            experiment_arr = None

        row_levels: list[str] = []
        for idx, assignment in enumerate(row_assignments):
            experiment_id = None if experiment_arr is None else experiment_arr[idx]
            if experiment_id is not None and (
                (isinstance(assignment, str) and ":" in assignment)
                or _looks_like_single_arm_tuple_key(assignment)
            ):
                raise ValueError(
                    "When experiment_ids are supplied, assignments must be local arm labels, "
                    "not globally encoded keys."
                )
            canonical, _ = _normalize_arm_key(assignment, experiment_id=experiment_id)
            row_levels.append(canonical)
        return _build_encoding_result(mode=mode, row_levels=row_levels, low_support_threshold=low_support_threshold)

    if arm_keys is not None:
        membership = np.asarray(assignments)
        if membership.ndim != 2:
            raise ValueError("Overlap-mode membership matrices must be 2D.")
        key_array = np.asarray(arm_keys, dtype=object)
        if key_array.ndim != 1 or key_array.shape[0] != membership.shape[1]:
            raise ValueError("arm_keys must be 1D and match the number of membership columns.")

        row_levels = []
        for row in membership:
            active = [key_array[col] for col, flag in enumerate(row) if bool(flag)]
            row_levels.append(_canonicalize_overlap_row(active, allow_empty=allow_empty))
        return _build_encoding_result(mode=mode, row_levels=row_levels, low_support_threshold=low_support_threshold)

    row_assignments = list(assignments)
    row_levels = [_canonicalize_overlap_row(row, allow_empty=allow_empty) for row in row_assignments]
    return _build_encoding_result(mode=mode, row_levels=row_levels, low_support_threshold=low_support_threshold)


def _validate_surrogate_support(encoding: ExperimentEncodingResult, *, n_splits: int) -> None:
    if encoding.min_count >= n_splits:
        return

    sparse = [
        f"{_format_level(level)} (count={int(count)})"
        for level, count in zip(encoding.levels, encoding.counts, strict=True)
        if int(count) < n_splits
    ]
    raise ValueError(
        "Encoded experiment support is too sparse for the requested cross-validation. "
        f"n_splits={n_splits}, but some encoded levels have fewer observations: {', '.join(sparse)}. "
        "Recommended remedies: restrict to a stable non-overlapping slice, coarsen experiments "
        "using domain knowledge before encoding, or exclude extremely sparse overlap combinations."
    )


def _build_npiv_result_from_fits(
    *,
    X_hist: np.ndarray,
    Z_codes: np.ndarray,
    Y_hist: np.ndarray,
    X_new: np.ndarray,
    structural_fit,
    dual_fit,
) -> NPIVInferenceResult:
    npjive = _build_functional_estimate(
        method_name="npjive",
        theta_coef=structural_fit.coef_npjive,
        beta_coef=dual_fit.coef_npjive,
        X=X_hist,
        Z=Z_codes,
        Y=Y_hist,
        X_new=X_new,
        use_leave_one_out=True,
    )
    baseline = _build_functional_estimate(
        method_name="2sls",
        theta_coef=structural_fit.coef_2sls,
        beta_coef=dual_fit.coef_2sls,
        X=X_hist,
        Z=Z_codes,
        Y=Y_hist,
        X_new=X_new,
        use_leave_one_out=False,
    )
    selected_name = f"{structural_fit.selected_method}+{dual_fit.selected_method}"
    selected = _build_functional_estimate(
        method_name=selected_name,
        theta_coef=structural_fit.coef_selected,
        beta_coef=dual_fit.coef_selected,
        X=X_hist,
        Z=Z_codes,
        Y=Y_hist,
        X_new=X_new,
        use_leave_one_out=_use_leave_one_out_for_dual(dual_fit.selected_method),
    )
    return NPIVInferenceResult(
        selected=selected,
        npjive=npjive,
        baseline_2sls=baseline,
        structural_fit=structural_fit,
        dual_fit=dual_fit,
    )


def _build_long_term_mean_result(
    *,
    X_hist: np.ndarray,
    Y_hist: np.ndarray,
    encoding: ExperimentEncodingResult,
    X_new: np.ndarray,
    target_name: str | None,
    n_splits: int,
    lambda_grid: object,
    gamma_grid: object,
    selection: str,
    random_state: int | None,
    structural_fit=None,
) -> LongTermMeanResult:
    if structural_fit is None:
        npiv_result = estimate_average_functional(
            X_hist,
            encoding.codes,
            Y_hist,
            X_new,
            n_splits=n_splits,
            lambda_grid=lambda_grid,
            gamma_grid=gamma_grid,
            selection=selection,
            random_state=random_state,
        )
    else:
        dual_fit = fit_dual_nuisance(
            X_hist,
            encoding.codes,
            X_new,
            n_splits=n_splits,
            lambda_grid=lambda_grid,
            gamma_grid=gamma_grid,
            selection=selection,
            random_state=random_state,
        )
        npiv_result = _build_npiv_result_from_fits(
            X_hist=X_hist,
            Z_codes=encoding.codes,
            Y_hist=Y_hist,
            X_new=X_new,
            structural_fit=structural_fit,
            dual_fit=dual_fit,
        )
    return LongTermMeanResult(target_name=target_name, encoding=encoding, npiv_result=npiv_result)


def estimate_long_term_mean_from_surrogates(
    X_hist: object,
    Y_hist: object,
    historical_arms: object,
    X_new: object,
    *,
    encoding_mode: str = "single",
    historical_experiment_ids: object | None = None,
    arm_keys: object | None = None,
    allow_empty: bool = True,
    low_support_threshold: int = 5,
    target_name: str | None = None,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> LongTermMeanResult:
    """
    Estimate the long-term mean outcome of a novel arm from surrogate data.

    Historical experiment-arm assignments are encoded as a discrete instrument
    and then passed through the core discrete NPIV estimators.
    """

    X_hist_arr = as_2d_float("X_hist", X_hist)
    Y_hist_arr = as_1d_float("Y_hist", Y_hist)
    if X_hist_arr.shape[0] != Y_hist_arr.shape[0]:
        raise ValueError("X_hist and Y_hist must have the same number of rows.")
    X_new_arr = validate_target_features(X_new, X_hist_arr.shape[1])
    encoding = encode_experiment_arms(
        historical_arms,
        mode=encoding_mode,
        experiment_ids=historical_experiment_ids,
        arm_keys=arm_keys,
        allow_empty=allow_empty,
        low_support_threshold=low_support_threshold,
    )
    if encoding.codes.shape[0] != X_hist_arr.shape[0]:
        raise ValueError("historical_arms must be aligned row-wise with X_hist and Y_hist.")
    _validate_surrogate_support(encoding, n_splits=n_splits)
    return _build_long_term_mean_result(
        X_hist=X_hist_arr,
        Y_hist=Y_hist_arr,
        encoding=encoding,
        X_new=X_new_arr,
        target_name=target_name,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )

def _contrast_estimate(
    *,
    method_name: str,
    treated_estimate: FunctionalEstimate,
    control_estimate: FunctionalEstimate,
    treated_predictions: np.ndarray,
    control_predictions: np.ndarray,
) -> FunctionalEstimate:
    influence = treated_estimate.influence_function - control_estimate.influence_function
    estimate = float(treated_estimate.estimate - control_estimate.estimate)
    plugin = float(treated_estimate.plugin_estimate - control_estimate.plugin_estimate)
    ipw = float(treated_estimate.ipw_estimate - control_estimate.ipw_estimate)
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


def _estimate_long_term_effect_base(
    X_hist_arr: np.ndarray,
    Y_hist_arr: np.ndarray,
    historical_arms: object,
    X_new_treated_arr: np.ndarray,
    X_new_control_arr: np.ndarray,
    *,
    encoding_mode: str = "single",
    historical_experiment_ids: object | None = None,
    arm_keys: object | None = None,
    allow_empty: bool = True,
    low_support_threshold: int = 5,
    treated_name: str | None = "treated",
    control_name: str | None = "control",
    effect_name: str | None = None,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> LongTermEffectResult:
    encoding = encode_experiment_arms(
        historical_arms,
        mode=encoding_mode,
        experiment_ids=historical_experiment_ids,
        arm_keys=arm_keys,
        allow_empty=allow_empty,
        low_support_threshold=low_support_threshold,
    )
    if encoding.codes.shape[0] != X_hist_arr.shape[0]:
        raise ValueError("historical_arms must be aligned row-wise with X_hist and Y_hist.")
    _validate_surrogate_support(encoding, n_splits=n_splits)

    structural_fit = fit_structural_nuisance(
        X_hist_arr,
        encoding.codes,
        Y_hist_arr,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )
    treated_mean = _build_long_term_mean_result(
        X_hist=X_hist_arr,
        Y_hist=Y_hist_arr,
        encoding=encoding,
        X_new=X_new_treated_arr,
        target_name=treated_name,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
        structural_fit=structural_fit,
    )
    control_mean = _build_long_term_mean_result(
        X_hist=X_hist_arr,
        Y_hist=Y_hist_arr,
        encoding=encoding,
        X_new=X_new_control_arr,
        target_name=control_name,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
        structural_fit=structural_fit,
    )

    selected = _contrast_estimate(
        method_name=f"contrast({treated_mean.selected.method_name}, {control_mean.selected.method_name})",
        treated_estimate=treated_mean.selected,
        control_estimate=control_mean.selected,
        treated_predictions=X_new_treated_arr @ treated_mean.structural_fit.coef_selected,
        control_predictions=X_new_control_arr @ control_mean.structural_fit.coef_selected,
    )
    npjive = _contrast_estimate(
        method_name="contrast(npjive, npjive)",
        treated_estimate=treated_mean.npjive,
        control_estimate=control_mean.npjive,
        treated_predictions=X_new_treated_arr @ treated_mean.structural_fit.coef_npjive,
        control_predictions=X_new_control_arr @ control_mean.structural_fit.coef_npjive,
    )
    baseline = _contrast_estimate(
        method_name="contrast(2sls, 2sls)",
        treated_estimate=treated_mean.baseline_2sls,
        control_estimate=control_mean.baseline_2sls,
        treated_predictions=X_new_treated_arr @ treated_mean.structural_fit.coef_2sls,
        control_predictions=X_new_control_arr @ control_mean.structural_fit.coef_2sls,
    )

    return LongTermEffectResult(
        effect_name=effect_name,
        encoding=encoding,
        treated_mean=treated_mean,
        control_mean=control_mean,
        selected=selected,
        npjive=npjive,
        baseline_2sls=baseline,
    )

def estimate_long_term_effect_from_surrogates(
    X_hist: object,
    Y_hist: object,
    historical_arms: object,
    X_new_treated: object,
    X_new_control: object,
    *,
    encoding_mode: str = "single",
    historical_experiment_ids: object | None = None,
    arm_keys: object | None = None,
    allow_empty: bool = True,
    low_support_threshold: int = 5,
    treated_name: str | None = "treated",
    control_name: str | None = "control",
    effect_name: str | None = None,
    n_splits: int = 2,
    lambda_grid: object = None,
    gamma_grid: object = None,
    selection: str = "adaptive",
    random_state: int | None = None,
) -> LongTermEffectResult:
    """
    Estimate a long-term treatment effect from surrogate-only novel-arm data.

    The effect is defined as the difference between two long-term mean estimates
    built from the same historical structural nuisance fit and two target-arm
    surrogate distributions.
    """

    X_hist_arr = as_2d_float("X_hist", X_hist)
    Y_hist_arr = as_1d_float("Y_hist", Y_hist)
    if X_hist_arr.shape[0] != Y_hist_arr.shape[0]:
        raise ValueError("X_hist and Y_hist must have the same number of rows.")

    X_new_treated_arr = validate_target_features(X_new_treated, X_hist_arr.shape[1])
    X_new_control_arr = validate_target_features(X_new_control, X_hist_arr.shape[1])
    return _estimate_long_term_effect_base(
        X_hist_arr,
        Y_hist_arr,
        historical_arms,
        X_new_treated_arr,
        X_new_control_arr,
        encoding_mode=encoding_mode,
        historical_experiment_ids=historical_experiment_ids,
        arm_keys=arm_keys,
        allow_empty=allow_empty,
        low_support_threshold=low_support_threshold,
        treated_name=treated_name,
        control_name=control_name,
        effect_name=effect_name,
        n_splits=n_splits,
        lambda_grid=lambda_grid,
        gamma_grid=gamma_grid,
        selection=selection,
        random_state=random_state,
    )
