# API Reference

This package exposes a core discrete NPIV interface plus a surrogate application layer built on top of it.

## Which Interface Should I Use?

- Use the core API if you already have a discrete instrument `Z` and want to estimate a linear functional defined by `X_new`.
- Use the surrogate API if your data are organized around historical experiments, surrogate outcomes, and a novel experiment with only short-term outcomes.

## Core Discrete NPIV API

### `fit_structural_nuisance`

Fits the primal nuisance for the structural function.

Inputs:

- `X`: `(n, d)` feature matrix
- `Z`: length-`n` discrete instrument vector
- `Y`: length-`n` outcome vector

Key options:

- `n_splits`
- `lambda_grid`
- `gamma_grid`
- `selection in {"adaptive", "npjive", "2sls"}`
- `random_state`

Returns a `StructuralFitResult`.

### `fit_dual_nuisance`

Fits the dual / Riesz nuisance for the target feature distribution.

Inputs:

- `X`: `(n, d)` training feature matrix
- `Z`: length-`n` discrete instrument vector
- `X_new`: `(m, d)` target feature matrix

`X_new` may have a different number of rows than `X`.

Returns a `DualFitResult`.

### `estimate_average_functional`

Fits both nuisances and computes the average linear functional with uncertainty.

Key options:

- `n_splits`
- `lambda_grid`
- `gamma_grid`
- `selection in {"adaptive", "npjive", "2sls"}`
- `random_state`

Returns an `NPIVInferenceResult` with:

- `selected`
- `npjive`
- `baseline_2sls`
- `structural_fit`
- `dual_fit`

### Grouped-Operator Utilities

- `group_means`
- `leave_one_out_group_means`
- `make_stratified_folds`

These are the public grouped-statistics helpers used by the estimator implementation.

## Surrogate Application API

These functions wrap the core NPIV estimators in application-native inputs.

### `encode_experiment_arms`

Encodes historical experiment-arm assignments as a discrete instrument.

Supported modes:

- `mode="single"`: one active historical arm per row
- `mode="overlap"`: multiple concurrent historical experiment arms per row

Key options:

- `experiment_ids`: optional experiment identifier per row when single-mode assignments are only local arm labels such as `control` and `treatment`
- `arm_keys`: required when overlap-mode input is a membership matrix
- `allow_empty`: whether the empty active set is allowed in overlap mode
- `low_support_threshold`: count threshold used for diagnostics

Returns an `ExperimentEncodingResult`.

### `estimate_long_term_mean_from_surrogates`

Estimates the long-term mean for one novel arm from surrogate-only novel data.

Inputs:

- `X_hist`: historical surrogate features
- `Y_hist`: historical long-term outcomes
- `historical_arms`: historical experiment-arm assignments
- `X_new`: surrogate features from one novel arm

Returns a `LongTermMeanResult`.

Key options mirror the core estimator, including:

- `selection in {"adaptive", "npjive", "2sls"}`

### `estimate_long_term_effect_from_surrogates`

Estimates a long-term treatment effect for a novel experiment by contrasting two novel-arm surrogate distributions.

Inputs:

- `X_hist`: historical surrogate features
- `Y_hist`: historical long-term outcomes
- `historical_arms`: historical experiment-arm assignments
- `X_new_treated`: surrogate features for the novel treated arm
- `X_new_control`: surrogate features for the novel control arm

Returns a `LongTermEffectResult`.

Key options mirror the core estimator, including:

- `selection in {"adaptive", "npjive", "2sls"}`

## Reproduction Helpers

### `run_small_paper_experiment`

Runs a compact paper-style simulation with:

- fixed design parameters controlled by `design_seed`
- varying noise seeds across replications
- summaries for selected, npJIVE, and 2SLS estimators

Returns a `PaperExperimentSummary`.

### `summarize_legacy_archive`

Loads one archived legacy pickle from `main-depreciated/legacy-results` and produces stable summary metrics for:

- legacy npJIVE
- legacy 2SLS
- legacy single-split estimator when present

## Result Objects

### `RegularizationChoice`

- `method`
- `lambda_value`
- `gamma`
- `cv_risk`

### `StructuralFitResult`

- `coef_selected`
- `coef_npjive`
- `coef_2sls`
- `selected_method`
- `tuning_npjive`
- `tuning_2sls`
- `instrument_levels`
- `.predict(X)`

### `DualFitResult`

- `coef_selected`
- `coef_npjive`
- `coef_2sls`
- `selected_method`
- `tuning_npjive`
- `tuning_2sls`
- `instrument_levels`
- `.score(X)`

### `FunctionalEstimate`

- `method_name`
- `estimate`
- `plugin_estimate`
- `ipw_estimate`
- `se`
- `ci_lower`
- `ci_upper`
- `influence_function`

### `NPIVInferenceResult`

- `selected`
- `npjive`
- `baseline_2sls`
- `structural_fit`
- `dual_fit`

### `ExperimentEncodingResult`

- `mode`
- `codes`
- `levels`
- `counts`
- `singleton_levels`
- `low_support_levels`
- `level_mapping()`
- `level_table()`

### `LongTermMeanResult`

- `target_name`
- `encoding`
- `npiv_result`
- `selected`
- `npjive`
- `baseline_2sls`
- `structural_fit`
- `dual_fit`

### `LongTermEffectResult`

- `effect_name`
- `encoding`
- `treated_mean`
- `control_mean`
- `selected`
- `npjive`
- `baseline_2sls`
- `structural_fit`
- `treated_dual_fit`
- `control_dual_fit`
