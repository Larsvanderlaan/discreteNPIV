from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .api import estimate_average_functional
from .simulation import find_alpha


@dataclass(frozen=True)
class MethodSimulationSummary:
    """Summary statistics for one estimator across repeated simulations."""

    name: str
    mean_estimate: float
    mean_truth: float
    bias: float
    rmse: float
    mean_se: float | None
    coverage: float | None

    def to_dict(self) -> dict[str, float | None | str]:
        return {
            "name": self.name,
            "mean_estimate": self.mean_estimate,
            "mean_truth": self.mean_truth,
            "bias": self.bias,
            "rmse": self.rmse,
            "mean_se": self.mean_se,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class PaperExperimentSummary:
    """Summary of a small paper-style simulation using the supported package."""

    n_replications: int
    design_seed: int
    selected_method_counts: dict[str, int]
    selected: MethodSimulationSummary
    npjive: MethodSimulationSummary
    baseline_2sls: MethodSimulationSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "n_replications": self.n_replications,
            "design_seed": self.design_seed,
            "selected_method_counts": self.selected_method_counts,
            "selected": self.selected.to_dict(),
            "npjive": self.npjive.to_dict(),
            "baseline_2sls": self.baseline_2sls.to_dict(),
        }


@dataclass(frozen=True)
class LegacyArchiveSummary:
    """Summary of an archived legacy simulation result file."""

    path: str
    npjive: MethodSimulationSummary
    baseline_2sls: MethodSimulationSummary
    single_split: MethodSimulationSummary | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "npjive": self.npjive.to_dict(),
            "baseline_2sls": self.baseline_2sls.to_dict(),
        }
        if self.single_split is not None:
            payload["single_split"] = self.single_split.to_dict()
        return payload


@dataclass(frozen=True)
class _FixedSyntheticDesign:
    theta: np.ndarray
    pi: np.ndarray
    pi_target: np.ndarray
    alpha_structural: float
    alpha_target: float


def _make_fixed_design(
    *,
    n_instruments: int,
    n_features: int,
    sparsity_structural: float,
    sparsity_target: float,
    design_seed: int,
) -> _FixedSyntheticDesign:
    alpha_structural = find_alpha(float(sparsity_structural), n_features)
    alpha_target = find_alpha(float(sparsity_target), n_features)
    target_decay = np.array([j ** (-alpha_target) for j in range(1, n_features + 1)], dtype=float)
    structural_decay = np.array([j ** (-alpha_structural) for j in range(1, n_features + 1)], dtype=float)

    rng = np.random.default_rng(design_seed)
    pi_target = rng.standard_normal(n_features) * target_decay
    pi = rng.standard_normal((n_instruments, n_features)) * target_decay[None, :]
    theta = (2.0 * (rng.standard_normal(n_features) >= 0.0) - 1.0) * structural_decay
    return _FixedSyntheticDesign(
        theta=theta,
        pi=pi,
        pi_target=pi_target,
        alpha_structural=alpha_structural,
        alpha_target=alpha_target,
    )


def _sample_from_fixed_design(
    design: _FixedSyntheticDesign,
    *,
    n_per_instrument: int,
    n_target_samples: int,
    sigma_x: float,
    sigma_y: float,
    sigma_c: float,
    noise_seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(noise_seed)
    n_instruments, n_features = design.pi.shape

    X_new = (
        np.repeat(design.pi_target[None, :], n_target_samples, axis=0)
        + rng.standard_normal((n_target_samples, 1)) * sigma_x
        + rng.standard_normal((n_target_samples, 1)) * sigma_c
    )

    u_x = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_x
    u_y = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_y
    u_c = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_c

    X = design.pi[None, :, :] + u_x[:, :, None] + u_c[:, :, None]
    X = X.reshape(n_per_instrument * n_instruments, n_features)
    Y = X @ design.theta + u_y.ravel() + u_c.ravel()
    Z = np.tile(np.arange(n_instruments), n_per_instrument)

    return {
        "theta": design.theta,
        "Pi": design.pi,
        "X": X,
        "Y": Y,
        "Z": Z,
        "X_new": X_new,
        "alpha_structural": design.alpha_structural,
        "alpha_target": design.alpha_target,
    }


def _summarize_method(
    *,
    name: str,
    estimates: list[float],
    truths: list[float],
    ses: list[float] | None = None,
    cis: list[tuple[float, float]] | None = None,
) -> MethodSimulationSummary:
    estimate_arr = np.asarray(estimates, dtype=float)
    truth_arr = np.asarray(truths, dtype=float)
    bias = estimate_arr - truth_arr
    mean_se = None if ses is None else float(np.mean(np.asarray(ses, dtype=float)))
    coverage = None
    if cis is not None:
        ci_arr = np.asarray(cis, dtype=float)
        coverage = float(np.mean((ci_arr[:, 0] <= truth_arr) & (truth_arr <= ci_arr[:, 1])))

    return MethodSimulationSummary(
        name=name,
        mean_estimate=float(np.mean(estimate_arr)),
        mean_truth=float(np.mean(truth_arr)),
        bias=float(np.mean(bias)),
        rmse=float(np.sqrt(np.mean(bias**2))),
        mean_se=mean_se,
        coverage=coverage,
    )


def run_small_paper_experiment(
    *,
    n_replications: int = 20,
    n_per_instrument: int = 30,
    n_instruments: int = 50,
    n_features: int = 18,
    n_target_samples: int = 4000,
    sparsity_structural: float = 2.5,
    sparsity_target: float = 3.5,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    sigma_c: float = 0.1,
    design_seed: int = 123,
    noise_seed_start: int = 0,
    n_splits: int = 2,
    lambda_grid: np.ndarray | None = None,
    gamma_grid: np.ndarray | None = None,
    selection: str = "adaptive",
) -> PaperExperimentSummary:
    """
    Run a compact paper-style simulation with fixed design and varying noise.

    The design parameters are held fixed using ``design_seed`` while each
    replication changes only the stochastic noise terms. This mirrors the style
    of the archived simulation workflow and makes summaries easier to compare
    against legacy result files.
    """

    design = _make_fixed_design(
        n_instruments=n_instruments,
        n_features=n_features,
        sparsity_structural=sparsity_structural,
        sparsity_target=sparsity_target,
        design_seed=design_seed,
    )

    truths: list[float] = []
    selected_estimates: list[float] = []
    selected_ses: list[float] = []
    selected_cis: list[tuple[float, float]] = []
    npjive_estimates: list[float] = []
    npjive_ses: list[float] = []
    npjive_cis: list[tuple[float, float]] = []
    baseline_estimates: list[float] = []
    baseline_ses: list[float] = []
    baseline_cis: list[tuple[float, float]] = []
    selected_counts: dict[str, int] = {}

    lambda_values = None if lambda_grid is None else np.asarray(lambda_grid, dtype=float)
    gamma_values = None if gamma_grid is None else np.asarray(gamma_grid, dtype=float)

    for replication in range(n_replications):
        data = _sample_from_fixed_design(
            design,
            n_per_instrument=n_per_instrument,
            n_target_samples=n_target_samples,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_c=sigma_c,
            noise_seed=noise_seed_start + replication,
        )
        result = estimate_average_functional(
            data["X"],
            data["Z"],
            data["Y"],
            data["X_new"],
            n_splits=n_splits,
            lambda_grid=lambda_values,
            gamma_grid=gamma_values,
            selection=selection,
            random_state=noise_seed_start + replication,
        )

        truth = float(np.mean(np.asarray(data["X_new"]) @ np.asarray(data["theta"])))
        truths.append(truth)

        selected_estimates.append(result.selected.estimate)
        selected_ses.append(result.selected.se)
        selected_cis.append((result.selected.ci_lower, result.selected.ci_upper))

        npjive_estimates.append(result.npjive.estimate)
        npjive_ses.append(result.npjive.se)
        npjive_cis.append((result.npjive.ci_lower, result.npjive.ci_upper))

        baseline_estimates.append(result.baseline_2sls.estimate)
        baseline_ses.append(result.baseline_2sls.se)
        baseline_cis.append((result.baseline_2sls.ci_lower, result.baseline_2sls.ci_upper))

        selected_counts[result.selected.method_name] = selected_counts.get(result.selected.method_name, 0) + 1

    return PaperExperimentSummary(
        n_replications=n_replications,
        design_seed=design_seed,
        selected_method_counts=dict(sorted(selected_counts.items())),
        selected=_summarize_method(
            name="selected",
            estimates=selected_estimates,
            truths=truths,
            ses=selected_ses,
            cis=selected_cis,
        ),
        npjive=_summarize_method(
            name="npjive",
            estimates=npjive_estimates,
            truths=truths,
            ses=npjive_ses,
            cis=npjive_cis,
        ),
        baseline_2sls=_summarize_method(
            name="2sls",
            estimates=baseline_estimates,
            truths=truths,
            ses=baseline_ses,
            cis=baseline_cis,
        ),
    )


def summarize_legacy_archive(path: str | Path) -> LegacyArchiveSummary:
    """Load one archived result pickle and return stable summary statistics."""

    archive_path = Path(path)
    with archive_path.open("rb") as handle:
        legacy = pickle.load(handle)

    truth = np.asarray(legacy["truth"], dtype=float).tolist()
    npjive = _summarize_method(
        name="legacy_npjive",
        estimates=np.asarray(legacy["dml_jive"], dtype=float).tolist(),
        truths=truth,
        cis=[tuple(row) for row in np.asarray(legacy["ci_jive"], dtype=float)],
    )
    baseline = _summarize_method(
        name="legacy_2sls",
        estimates=np.asarray(legacy["dml_tsls"], dtype=float).tolist(),
        truths=truth,
        cis=[tuple(row) for row in np.asarray(legacy["ci_tsls"], dtype=float)],
    )

    single_split = None
    if "dml_jsingle" in legacy and "ci_single" in legacy:
        single_split = _summarize_method(
            name="legacy_single_split",
            estimates=np.asarray(legacy["dml_jsingle"], dtype=float).tolist(),
            truths=truth,
            cis=[tuple(row) for row in np.asarray(legacy["ci_single"], dtype=float)],
        )

    return LegacyArchiveSummary(
        path=str(archive_path),
        npjive=npjive,
        baseline_2sls=baseline,
        single_split=single_split,
    )

