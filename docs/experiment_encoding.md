# Experiment Encoding Guide

This guide explains how to turn historical experiments into the discrete instrument used by the core NPIV estimators.

## Core Idea

In the surrogate application, the instrument is not a single scalar treatment label from the novel experiment. The instrument comes from historical experimental variation:

- historical experiment arms shift the distribution of surrogate outcomes
- those shifts provide exogenous variation that helps identify the surrogate-to-long-term mapping
- the package encodes those historical experiment arms as a discrete instrument `Z`

## Single Active Arm Per Row

Use `mode="single"` when each historical unit belongs to exactly one experiment arm.

There are two supported input patterns.

### Pattern 1: Globally Unique Arm Keys

Pass one globally unique key per row, such as:

- `pricing_test:control`
- `pricing_test:treatment`
- `search_ranking:baseline`
- `search_ranking:variant_b`

Example:

```python
from discreteNPIV import encode_experiment_arms

encoding = encode_experiment_arms(
    [
        "pricing_test:control",
        "pricing_test:treatment",
        "search_ranking:baseline",
        "search_ranking:variant_b",
    ],
    mode="single",
)
```

### Pattern 2: Local Arm Labels Plus Experiment IDs

If your raw data store only local arm labels such as `control` and `treatment`, pass the experiment ID separately.

Example:

```python
encoding = encode_experiment_arms(
    assignments=["control", "treatment", "control", "treatment"],
    experiment_ids=["pricing_test", "pricing_test", "search_ranking", "search_ranking"],
    mode="single",
)
```

This is often the cleanest approach when you already have columns like:

- `experiment_id`
- `arm_label`

The package turns them into canonical keys such as `pricing_test:treatment`.

### What Not To Do

Do not pass bare labels like `control` or `treatment` without telling the package which experiment they belong to. Those labels are not globally unique across experiments.

## Overlapping Historical Experiments

Use `mode="overlap"` when the same historical unit can be exposed to multiple concurrent experiments.

The package encodes the full active set of experiment-arm keys for each row as one categorical instrument level.

### Sequence-Of-Sets Input

Pass one collection of active experiment-arm keys per row.

Example:

```python
encoding = encode_experiment_arms(
    [
        ["pricing_test:treatment", "ranking_test:baseline"],
        ["pricing_test:treatment", "ranking_test:baseline"],
        ["pricing_test:control"],
        [],
    ],
    mode="overlap",
    allow_empty=True,
)
```

This produces levels corresponding to:

- `("pricing_test:treatment", "ranking_test:baseline")`
- `("pricing_test:control",)`
- `()`

The empty tuple `()` is the encoded level for rows with no active experiment arms.

### Membership-Matrix Input

If your historical data are stored as a binary matrix of arm memberships, pass that matrix together with `arm_keys`.

Example:

```python
import numpy as np

membership = np.array(
    [
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ]
)

encoding = encode_experiment_arms(
    membership,
    mode="overlap",
    arm_keys=[
        "pricing_test:treatment",
        "pricing_test:control",
        "ranking_test:baseline",
    ],
    allow_empty=True,
)
```

## Validation Rules

The encoder applies a few intentionally strict rules.

- A row cannot belong to two arms from the same experiment.
- Single-mode arm labels must be globally unique unless `experiment_ids` are supplied.
- Overlap-mode encodings are canonicalized, so the order of active arms does not matter.
- Support diagnostics are always reported through `counts`, `singleton_levels`, and `low_support_levels`.

## Sparse Overlap Cells

The package does not silently pool or merge sparse overlap categories in v1.

If overlap produces many tiny categories, the encoder will still report them, and the surrogate estimators will raise a clear error when the smallest encoded level is too small for the requested `n_splits`.

Recommended remedies:

- restrict to a stable non-overlapping analysis slice
- coarsen experiments using domain knowledge before encoding
- exclude extremely sparse overlap combinations from the analysis design

These are design decisions, not implementation details to hide.

## Inspecting Diagnostics

`encode_experiment_arms` returns an `ExperimentEncodingResult`.

Useful fields and methods:

- `encoding.codes`
- `encoding.levels`
- `encoding.counts`
- `encoding.singleton_levels`
- `encoding.low_support_levels`
- `encoding.level_table()`

Example:

```python
encoding = encode_experiment_arms(
    assignments=["control", "treatment", "control", "treatment"],
    experiment_ids=["exp_a", "exp_a", "exp_b", "exp_b"],
    mode="single",
)

print(encoding.level_table())
```
