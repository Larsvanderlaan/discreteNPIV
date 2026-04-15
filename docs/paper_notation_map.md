# Paper Notation Map

This note connects the paper's notation, the core NPIV API, and the surrogate application layer.

## Core NPIV Language

The package's core API uses:

- `X`: observed covariates or basis representation
- `Z`: discrete instrument
- `Y`: observed outcome
- `X_new`: sample from the target covariate distribution that defines the linear functional

The main estimators are:

- structural nuisance: `fit_structural_nuisance`
- dual/Riesz nuisance: `fit_dual_nuisance`
- debiased functional estimator: `estimate_average_functional`

## Paper-To-Code Map

| Paper object | Meaning | Package entry point |
| --- | --- | --- |
| `h_K` | structural nuisance / minimum-norm NPIV solution | `fit_structural_nuisance` |
| `beta_K` | dual or Riesz representer for the target functional | `fit_dual_nuisance` |
| target linear functional | average of `h_K(X_new)` under the target distribution | `estimate_average_functional` |
| npJIVE | leave-one-out/jackknife many-instrument correction | `selected`, `npjive`, and grouped utilities |
| 2SLS baseline | grouped baseline estimator fit alongside npJIVE | `baseline_2sls` |

## Surrogate Application Map

For the long-term causal inference application:

| Causal language | Core NPIV object |
| --- | --- |
| historical experiment arms | `Z` |
| historical surrogate outcomes | `X_hist` |
| historical long-term outcomes | `Y_hist` |
| surrogate data from a novel arm | `X_new` |
| long-term mean for a novel arm | `E[h(X_new)]` |
| long-term treatment effect for a novel experiment | `E[h(X_new_treated)] - E[h(X_new_control)]` |

This is why the package offers both:

- a core NPIV interface for users who work directly with `X, Z, Y, X_new`
- a surrogate wrapper interface for users who think in terms of experiments and surrogates

## Where The Surrogate Wrappers Live

- `encode_experiment_arms`
- `estimate_long_term_mean_from_surrogates`
- `estimate_long_term_effect_from_surrogates`

These wrappers translate application-specific inputs into the core NPIV estimand. They do not replace the core estimators.
