# discreteNPIV

`discreteNPIV` is a research-oriented Python package for discrete nonparametric instrumental variable inference in the many-weak-instruments regime. It is designed to accompany the paper "Nonparametric Instrumental Variable Inference with Many Weak Instruments."

The package has two clear entry paths:

- use it directly as a discrete NPIV package when you already have covariates `X`, a discrete instrument `Z`, outcomes `Y`, and a target covariate distribution `X_new`
- use it as a long-term causal inference package for surrogate problems, where historical experiment arms become the discrete instrument and a novel experiment contributes only surrogate outcomes

Long-term causal inference with surrogates is a flagship application of the package, not the whole package.

## Installation

```bash
python3 -m pip install -e .
```

## Which Interface Should I Use?

- Use the core NPIV API if you already think in terms of a discrete instrument and a linear functional of the structural map.
- Use the surrogate API if your data come from historical experiments with long-term outcomes plus a new experiment that only has short-term surrogate outcomes.

## Use As A Discrete NPIV Package

At the core level, the package solves three tasks:

- fit the structural nuisance for the minimum-norm NPIV solution
- fit the dual/Riesz nuisance for a target covariate law
- estimate a debiased average linear functional with standard errors and confidence intervals

The main low-level API is:

- `fit_structural_nuisance`
- `fit_dual_nuisance`
- `estimate_average_functional`

In this notation:

- `X` is the observed feature or basis representation
- `Z` is a discrete instrument
- `Y` is the observed outcome
- `X_new` is a sample from the target covariate distribution that defines the linear functional of interest

### Core NPIV Quickstart

```python
from discreteNPIV import (
    estimate_average_functional,
    fit_dual_nuisance,
    fit_structural_nuisance,
    generate_synthetic_data,
)

data = generate_synthetic_data(
    n_per_instrument=20,
    n_instruments=8,
    n_features=5,
    n_target_samples=4000,
    random_state=7,
)

structural = fit_structural_nuisance(
    data["X"],
    data["Z"],
    data["Y"],
    n_splits=2,
    random_state=7,
)

dual = fit_dual_nuisance(
    data["X"],
    data["Z"],
    data["X_new"],
    n_splits=2,
    random_state=7,
)

result = estimate_average_functional(
    data["X"],
    data["Z"],
    data["Y"],
    data["X_new"],
    n_splits=2,
    random_state=7,
)

print(structural.selected_method)
print(dual.selected_method)
print(result.selected.estimate, result.selected.se)
```

`estimate_average_functional` returns an `NPIVInferenceResult` with:

- `selected`
- `npjive`
- `baseline_2sls`
- `structural_fit`
- `dual_fit`

Inference uses the package's Wald-style confidence interval based on the influence-function approximation.

## Use For Long-Term Causal Inference With Surrogates

The package also provides an application layer for the paper's motivating use case: estimate long-term outcomes or treatment effects in a new experiment when you only observe short-term surrogate outcomes there, but you have historical experiments with both surrogates and long-term outcomes.

The surrogate API is:

- `encode_experiment_arms`
- `estimate_long_term_mean_from_surrogates`
- `estimate_long_term_effect_from_surrogates`

### Surrogate-To-NPIV Mapping

- historical experiment arms -> discrete instrument `Z`
- historical surrogate vector -> `X`
- historical long-term outcome -> `Y`
- surrogate data from a novel arm -> `X_new`
- long-term mean for a novel arm -> `E[h(X_new)]`
- long-term treatment effect for a novel experiment -> `E[h(X_new_treated)] - E[h(X_new_control)]`

Here `h` is the structural surrogate-to-long-term map estimated from historical experimental variation.

### Surrogate Quickstart

```python
import numpy as np

from discreteNPIV import estimate_long_term_effect_from_surrogates

rng = np.random.default_rng(12)

historical_experiment_ids = np.repeat(["exp_a", "exp_b", "exp_c"], 80)
historical_arm_labels = np.tile(np.repeat(["control", "treatment"], 40), 3)

arm_shift = {
    "exp_a:control": np.array([0.0, 0.1, -0.1]),
    "exp_a:treatment": np.array([0.4, 0.2, 0.0]),
    "exp_b:control": np.array([-0.1, 0.0, 0.1]),
    "exp_b:treatment": np.array([0.2, 0.3, 0.2]),
    "exp_c:control": np.array([0.1, -0.2, 0.0]),
    "exp_c:treatment": np.array([0.3, 0.1, 0.3]),
}
theta = np.array([0.8, -0.5, 0.4])

surrogates = []
outcomes = []
for experiment_id, arm_label in zip(historical_experiment_ids, historical_arm_labels, strict=True):
    key = f"{experiment_id}:{arm_label}"
    confounder = rng.normal(scale=0.35)
    x = arm_shift[key] + confounder + rng.normal(scale=0.2, size=3)
    y = x @ theta + 0.5 * confounder + rng.normal(scale=0.2)
    surrogates.append(x)
    outcomes.append(y)

X_hist = np.asarray(surrogates)
Y_hist = np.asarray(outcomes)

X_new_control = rng.normal(loc=[0.05, 0.0, 0.0], scale=0.2, size=(500, 3))
X_new_treated = rng.normal(loc=[0.25, 0.15, 0.1], scale=0.2, size=(500, 3))

effect = estimate_long_term_effect_from_surrogates(
    X_hist=X_hist,
    Y_hist=Y_hist,
    historical_arms=historical_arm_labels,
    historical_experiment_ids=historical_experiment_ids,
    X_new_treated=X_new_treated,
    X_new_control=X_new_control,
    n_splits=2,
    random_state=12,
)

print(effect.selected.estimate)
print(effect.selected.ci_lower, effect.selected.ci_upper)
print(effect.selected.method_name)
```

## How Past Experiments Become The Instrument

Historical experiments are used as the source of exogenous variation in surrogate outcomes:

- if each historical unit belongs to one experiment arm, encode each row with a globally unique arm key such as `pricing_test:treatment`
- if your raw arm labels are only local within an experiment, pass `historical_experiment_ids` together with arm labels like `control` and `treatment`
- if units can be in multiple concurrent experiments, use `encoding_mode="overlap"` and encode the full active set of experiment-arm keys for each row

The package does not silently merge sparse overlap categories. Instead it reports support diagnostics and raises a clear error when the encoded design is too sparse for the requested cross-validation split.

More detail is in:

- `docs/experiment_encoding.md`
- `docs/long_term_surrogate_case_study.md`
- `docs/paper_notation_map.md`

## LOO/Jackknife Versus Tuning CV

The package uses two separate ideas that should not be conflated.

For nuisance estimation:

- group-level leave-one-out or jackknife means are used inside the npJIVE estimating equations
- this is the bias-reduction device that mirrors the spirit of classical JIVE

For regularization tuning:

- regularization parameters are selected by a separate stratified K-fold cross-validation routine
- this routine evaluates empirical risk on held-out folds

The paper's theory uses two-fold splitting for analysis, while the implementation here uses leave-one-out/jackknife grouped means for nuisance construction and separate K-fold CV for tuning.

More detail is in `docs/loo_jackknife.md`.

## Public API Summary

Core NPIV:

- `fit_structural_nuisance`
- `fit_dual_nuisance`
- `estimate_average_functional`
- `group_means`
- `leave_one_out_group_means`
- `make_stratified_folds`
- `generate_synthetic_data`

Surrogate application layer:

- `encode_experiment_arms`
- `estimate_long_term_mean_from_surrogates`
- `estimate_long_term_effect_from_surrogates`

Reproduction helpers:

- `run_small_paper_experiment`
- `summarize_legacy_archive`

## Examples And Reproduction

Core paper-style reproduction:

- `scripts/reproduce_small_paper_experiment.py`

Surrogate case study:

- `scripts/reproduce_surrogate_case_study.py`

Nonlinear surrogate coverage stress test:

- `scripts/evaluate_nonlinear_surrogate_coverage.py`

npJIVE validation suite:

- `scripts/evaluate_npjive_validation.py`

Example targeted validation run:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_npjive_validation.py \
  --replications 2 \
  --scenarios linear_strong_disjoint nonlinear_medium_disjoint
```

The surrogate case-study script prints a human-readable summary of encoded historical arms, long-term means for novel treated and control arms, and the resulting long-term treatment effect estimate.

## Repository Layout

- `src/discreteNPIV`: supported package code
- `tests`: unit and integration tests
- `docs`: package documentation
- `scripts`: runnable examples and reproduction scripts
- `main-depreciated`: archived research workspace and legacy outputs

## Migration Note

This is a clean break from the original research workspace. The old module tree, notebooks, result dumps, and exploratory scripts were moved to `main-depreciated` and are not part of the supported package surface.
