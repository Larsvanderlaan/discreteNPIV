# npJIVE Validation Notes

This package now includes a dedicated validation path for forced `npjive`, because that is the paper's main methodological contribution.

## What Was Audited

- the forced `npjive` structural nuisance path
- the forced `npjive` dual nuisance path
- the functional estimator construction when the selected nuisance family is `npjive`, `2sls`, or mixed
- Wald confidence intervals for the target functional

## Important Implementation Note

The package uses leave-one-out or jackknife grouped means inside the npJIVE nuisance construction.
That is separate from confidence intervals.

This validation pass also fixed a construction mismatch in the previously selected estimator path:
the package no longer forces leave-one-out functional logic when the selected nuisance family is actually `2sls`.
The selected path now follows the selected dual nuisance family, and the surrogate wrappers use the same rule.

The package now focuses on the Wald confidence interval path based on the influence-function-style normal approximation.

## What The Validation Suite Covers

Use `scripts/evaluate_npjive_validation.py` to evaluate:

- linear disjoint-arm designs with known truth
- nonlinear basis-feature designs with known truth
- how structural RMSE and functional bias change as support and instrument strength improve
- Wald coverage for forced `npjive`

The script supports targeted runs through:

- `--scenarios` to run a subset of disjoint-arm designs
- `--include-overlap` to add the secondary overlap stress test

Overlap-heavy interaction-group designs remain secondary in this pass. They are useful stress tests, but they are not yet treated as validated to the same standard as disjoint-arm designs.
