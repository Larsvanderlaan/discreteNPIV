# discreteNPIV

`discreteNPIV` implements regularized nonparametric instrumental-variable
estimation with a discrete instrument. It accompanies the paper
“Nonparametric Instrumental Variable Inference with Many Weak Instruments.”

The repository has two parts:

- `src/discreteNPIV/`: the installable package;
- `experiments/`: the exact simulation design, frozen configurations, summary
  results, and plotting code used in the paper.

## Installation

The package requires Python 3.10 or newer and NumPy.

```bash
python -m pip install -e .
```

## Core API

The main entry point estimates an average linear functional of the structural
NPIV regression:

```python
from discreteNPIV import estimate_average_functional, generate_synthetic_data

data = generate_synthetic_data(
    n_per_instrument=20,
    n_instruments=8,
    n_features=5,
    n_target_samples=1000,
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

print(result.selected.estimate, result.selected.se)
```

Lower-level functions are available when the structural and dual nuisances are
needed separately:

- `fit_structural_nuisance`
- `fit_dual_nuisance`
- `estimate_average_functional`

The package also provides a direct interface for the long-term surrogate
application:

- `encode_experiment_arms`
- `estimate_long_term_mean_from_surrogates`
- `estimate_long_term_effect_from_surrogates`

All public functions validate array shapes and support, use deterministic
stratified folds when a seed is supplied, and return the selected regularization
choices together with the estimate and influence-function standard error.

## Reproducing the paper experiments

See [`experiments/README.md`](experiments/README.md) for the frozen designs and
commands. A one-replication smoke run is:

```bash
python -m experiments.run \
  --study main \
  --cell projection_confirmatory_longterm_50_n30_covered \
  --replications 1
```

Generate every reported figure and table from the checked summary files with:

```bash
Rscript experiments/plot.R
```

The plotting script requires R and `ggplot2`.

## Tests

```bash
python -m unittest discover -s tests
```

## Repository layout

```text
src/discreteNPIV/       reusable estimator implementation
experiments/paper/      simulation design and paper-specific estimators
experiments/config/     frozen configurations and seeds
experiments/results/    compact summaries used by the figures and tables
tests/                  focused package and reproduction tests
```

The automatic regularization selectors use only observed-data cross-validation
criteria and numerical stability checks. Simulation truth is used only to
evaluate bias, RMSE, and coverage after estimation.
