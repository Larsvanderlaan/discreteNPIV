# Migration From The Legacy Workspace

The repository was rebuilt as a clean package.

## What Moved

The old materials now live under `main-depreciated`:

- `main-depreciated/legacy-workspace`: old package tree, notebooks, and exploratory scripts
- `main-depreciated/legacy-results`: old readme and root-level result dumps
- `main-depreciated/legacy-figures`: archived figures from the research workspace

## What Changed

- the supported code now lives in `src/discreteNPIV`
- the public API is redesigned around explicit fitting and inference functions
- the numerical core no longer depends on pandas or scikit-learn
- legacy notebooks are archival and are not a supported interface

## How To Update Usage

Old workflow:

- import internal files directly
- run notebooks against an in-place module tree

New workflow:

- install the package with `python3 -m pip install -e .`
- import from `discreteNPIV`
- use `fit_structural_nuisance`, `fit_dual_nuisance`, or `estimate_average_functional`

## Scope Of The Legacy Archive

The archive is preserved for paper-tracing and sanity checks, not as a supported package surface. If legacy behavior conflicts with the paper or with the new implementation's validation rules, the supported package follows the paper-oriented implementation.

