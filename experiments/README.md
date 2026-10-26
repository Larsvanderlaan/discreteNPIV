# Paper experiments

This directory reproduces the simulations reported in the paper. It is kept
separate from the installable package so that the scientific design and the
general-purpose estimator API can be reviewed independently.

## Designs

`config/main_loo.json` contains the twelve fixed-cell designs:

- numbers of instrument levels (K\in\{50,100,200,400\});
- 30 historical observations per instrument level;
- point-identified, asymptotically identified, and persistent-gap regimes;
- 500 replications per design.

`config/large_n_loo.json` contains the large-cell design with (K=400), 400
historical observations per instrument level, a persistent identification gap,
and 200 replications.

Every run reports only the estimators shown in the paper:

- nonsplit 2SLS-DML (main designs only);
- leave-one-out DML-JIVE and DML-JIVE*.

Both JIVE estimators use every observation with leave-one-out cell averages. Their historical variance is calculated from the symmetric pair kernel and products involving four distinct observations. The target-sample variance is added separately. Nuisance tuning remains five-fold.

The starred estimators learn the projection from the observed samples. The
prespecified tuning rules use observed-data risks and system-stability checks;
simulation truth is not used for tuning. Every scheduled replication is retained,
including an explicit failure row if an estimator raises an exception.

## Run the simulations

Install the package from the repository root, then run:

```bash
python -m pip install -e .
python -m experiments.run --study main --workers 4
python -m experiments.run --study large_n --workers 4
```

The large-cell study is computationally intensive. For a fast end-to-end check:

```bash
python -m experiments.run \
  --study main \
  --config experiments/config/smoke_loo.json \
  --stage validation \
  --output-dir experiments/generated/smoke
```

Generated rows and summaries are written under `experiments/generated/`. The
runner accepts `--resume`, checkpoints every replication, and writes one part file per design. A resumed run must match the stored configuration and source snapshot.

## Make figures and tables

The checked summaries use the LOO configurations and the seeds frozen in `config/loo_run_plan.json`. The earlier configurations (`config/main.json` and `config/large_n.json`) remain available for provenance.

The checked summaries in `results/` are the sources for the paper artifacts.
Regenerate them from a new run by passing the corresponding `summary.csv`
paths.

```bash
Rscript experiments/plot.R
```

or

```bash
Rscript experiments/plot.R \
  --main-summary experiments/generated/main/summary.csv \
  --projection-summary experiments/generated/large_n/summary.csv
```

The script writes two figures, two CSV tables, their LaTeX versions, and a
source manifest to `experiments/generated/figures/`.

## Code map

- `paper/model.py`: frozen configuration and reproducibility metadata;
- `paper/design.py`: feature map and data-generating process;
- `paper/nuisance.py`: cross-fitted structural and dual inverse problems;
- `paper/projection.py`: data-driven npJIVE projection estimator;
- `paper/inference.py`: target-sample variance calculations;
- `paper/cell_variance.py`: symmetric LOO kernels and historical variance;
- `paper/estimation.py`: reported estimators;
- `paper/diagnostics.py`: replication records and failure handling;
- `paper/reporting.py`: aggregation into bias, RMSE, coverage, and SE/SD;
- `run.py`: deterministic execution and checkpointing;
- `plot.R`: all reported figures and tables.

The checked result files contain only the summary columns needed to verify the
paper and regenerate its artifacts. Full replication-level rows are regenerated
by `run.py` and are not stored in the source repository.
