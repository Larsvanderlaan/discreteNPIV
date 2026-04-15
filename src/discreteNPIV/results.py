from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegularizationChoice:
    """Cross-validation choice for one estimator family."""

    method: str
    lambda_value: float
    gamma: float
    cv_risk: float


@dataclass(frozen=True)
class StructuralFitResult:
    """Result of fitting the structural nuisance."""

    coef_selected: np.ndarray
    coef_npjive: np.ndarray
    coef_2sls: np.ndarray
    selected_method: str
    tuning_npjive: RegularizationChoice
    tuning_2sls: RegularizationChoice
    instrument_levels: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_selected


@dataclass(frozen=True)
class DualFitResult:
    """Result of fitting the dual / Riesz nuisance."""

    coef_selected: np.ndarray
    coef_npjive: np.ndarray
    coef_2sls: np.ndarray
    selected_method: str
    tuning_npjive: RegularizationChoice
    tuning_2sls: RegularizationChoice
    instrument_levels: np.ndarray

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_selected


@dataclass(frozen=True)
class FunctionalEstimate:
    """One functional estimate together with uncertainty diagnostics."""

    method_name: str
    estimate: float
    plugin_estimate: float
    ipw_estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    influence_function: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "estimate": self.estimate,
            "plugin_estimate": self.plugin_estimate,
            "ipw_estimate": self.ipw_estimate,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "influence_function": self.influence_function,
        }


@dataclass(frozen=True)
class NPIVInferenceResult:
    """Full inference bundle with selected, npJIVE, and 2SLS outputs."""

    selected: FunctionalEstimate
    npjive: FunctionalEstimate
    baseline_2sls: FunctionalEstimate
    structural_fit: StructuralFitResult
    dual_fit: DualFitResult

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected.to_dict(),
            "npjive": self.npjive.to_dict(),
            "baseline_2sls": self.baseline_2sls.to_dict(),
            "structural_selected_method": self.structural_fit.selected_method,
            "dual_selected_method": self.dual_fit.selected_method,
        }


@dataclass(frozen=True)
class ExperimentEncodingResult:
    """Encoded instrument levels and support diagnostics for experiment-arm inputs."""

    mode: str
    instrument_values: np.ndarray
    codes: np.ndarray
    levels: tuple[object, ...]
    counts: np.ndarray
    low_support_threshold: int
    singleton_levels: tuple[object, ...]
    low_support_levels: tuple[object, ...]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def min_count(self) -> int:
        return int(np.min(self.counts))

    @property
    def max_count(self) -> int:
        return int(np.max(self.counts))

    def level_mapping(self) -> dict[object, int]:
        return {level: idx for idx, level in enumerate(self.levels)}

    def level_table(self) -> list[dict[str, object]]:
        return [
            {"level": level, "code": idx, "count": int(count)}
            for idx, (level, count) in enumerate(zip(self.levels, self.counts, strict=True))
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "levels": list(self.levels),
            "counts": self.counts.tolist(),
            "low_support_threshold": self.low_support_threshold,
            "singleton_levels": list(self.singleton_levels),
            "low_support_levels": list(self.low_support_levels),
        }


@dataclass(frozen=True)
class LongTermMeanResult:
    """User-facing result for a long-term mean estimated from surrogate data."""

    target_name: str | None
    encoding: ExperimentEncodingResult
    npiv_result: NPIVInferenceResult

    @property
    def selected(self) -> FunctionalEstimate:
        return self.npiv_result.selected

    @property
    def npjive(self) -> FunctionalEstimate:
        return self.npiv_result.npjive

    @property
    def baseline_2sls(self) -> FunctionalEstimate:
        return self.npiv_result.baseline_2sls

    @property
    def structural_fit(self) -> StructuralFitResult:
        return self.npiv_result.structural_fit

    @property
    def dual_fit(self) -> DualFitResult:
        return self.npiv_result.dual_fit

    def to_dict(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "encoding": self.encoding.to_dict(),
            "selected": self.selected.to_dict(),
            "npjive": self.npjive.to_dict(),
            "baseline_2sls": self.baseline_2sls.to_dict(),
        }


@dataclass(frozen=True)
class LongTermEffectResult:
    """User-facing result for a long-term treatment effect from surrogate data."""

    effect_name: str | None
    encoding: ExperimentEncodingResult
    treated_mean: LongTermMeanResult
    control_mean: LongTermMeanResult
    selected: FunctionalEstimate
    npjive: FunctionalEstimate
    baseline_2sls: FunctionalEstimate

    @property
    def structural_fit(self) -> StructuralFitResult:
        return self.treated_mean.structural_fit

    @property
    def treated_dual_fit(self) -> DualFitResult:
        return self.treated_mean.dual_fit

    @property
    def control_dual_fit(self) -> DualFitResult:
        return self.control_mean.dual_fit

    def to_dict(self) -> dict[str, object]:
        return {
            "effect_name": self.effect_name,
            "encoding": self.encoding.to_dict(),
            "treated_mean": self.treated_mean.to_dict(),
            "control_mean": self.control_mean.to_dict(),
            "selected": self.selected.to_dict(),
            "npjive": self.npjive.to_dict(),
            "baseline_2sls": self.baseline_2sls.to_dict(),
        }
