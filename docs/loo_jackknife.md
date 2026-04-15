# LOO And Jackknife Implementation Notes

This package uses two different resampling ideas.

## 1. Leave-One-Out / Jackknife Inside npJIVE

For the structural nuisance and the dual/Riesz nuisance, the package computes group-wise leave-one-out means. If observation `i` belongs to instrument group `z_i`, the leave-one-out mean replaces the group average by removing observation `i` from that group before forming the corresponding grouped statistic.

For a scalar quantity `u_i`, the leave-one-out mean is

`T_-i(u | Z = z_i) = (sum_{j: z_j = z_i} u_j - u_i) / (n_{z_i} - 1)`

with the singleton-group fallback

`T_-i(u | Z = z_i) = u_i`

when `n_{z_i} = 1`.

This fallback is explicit in the implementation so the estimator stays numerically defined even when some groups are tiny. The research interpretation is that singleton groups should usually be treated with caution, but the package avoids undefined arithmetic.

## 2. Cross-Validation For Regularization

Regularization is tuned by a separate stratified K-fold routine:

- folds are built within instrument groups
- each candidate pair `(lambda, gamma)` is fit on the training folds
- the fitted coefficients are scored on the held-out fold using the same quadratic objective family as the estimator

These folds are only for tuning. They are not the jackknife itself.

## Why The Distinction Matters

The manuscript analyzes two-fold cross-fitting for theory because it is easier to study formally. It also explicitly notes that leave-one-out or jackknife splitting is recommended in practice. The package follows that practical recommendation for nuisance construction while keeping tuning as a separate K-fold procedure.

## Code Locations

- grouped means and leave-one-out means: `src/discreteNPIV/grouping.py`
- tuning: `src/discreteNPIV/_tuning.py`
- primal and dual estimators: `src/discreteNPIV/api.py`

