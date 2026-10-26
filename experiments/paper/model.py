from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from discreteNPIV._version import __version__
from discreteNPIV.grouping import group_counts
from discreteNPIV.results import DualFitResult, FunctionalEstimate, StructuralFitResult


DEFAULT_LAMBDA_GRID = (3.0, 1.0, 0.3, 0.1, 0.03)
DEFAULT_GAMMA_GRID = (3.0, 1.0, 0.3, 0.1, 0.03)
VALID_REGIMES = {"covered", "vanishing", "persistent", "boundary"}
REPORT_METHODS = (
    "2sls_dml",
    "loo_npjive",
    "loo_npjive_star",
)

_BASE_THETA = np.array([1.0, 0.8, -0.5, 0.25, 0.10, 0.20, 0.10], dtype=float)
_CONFOUNDING_DIRECTION = np.array([1.0, 0.5, -0.25, 0.75, 0.0], dtype=float)
_CONFOUNDING_DIRECTION /= np.linalg.norm(_CONFOUNDING_DIRECTION)
_NOVEL_DIRECTION = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
_NOVEL_DIRECTION /= np.linalg.norm(_NOVEL_DIRECTION)
_S5_FEATURE_INDEX = 4
def _target_estimand_for_method(method: str) -> str:
    """Return the estimand a reported method is intended to cover."""

    if method.endswith("_star"):
        return "projection"
    return "true"


@dataclass(frozen=True)
class FeatureSpec:
    """Deterministic finite feature map used by the paper simulation."""

    directions: np.ndarray
    phases: np.ndarray

    @property
    def n_features(self) -> int:
        return int(7 + 2 * self.directions.shape[0])


@dataclass(frozen=True)
class LongTermSurrogateConfig:
    """Configuration for the theory-mapped long-term surrogate simulation."""

    cell_id: str
    n_replications: int
    n_instruments: int
    n_per_instrument: int
    regime: str
    seed_base: int
    growth_path: str = "fixed_n30"
    n_new_per_arm: int | None = None
    n_splits: int = 5
    n_fourier_pairs: int = 0
    truth_fourier_pairs: int = 128
    truth_fourier_amplitude: float = 0.18
    truth_fourier_smoothness: float = 1.5
    feature_seed: int = 31_415
    signal_scale: float = 0.8
    kernel_bandwidth: float = 0.08
    control_location: float = 0.35
    treated_location: float = 0.65
    surrogate_noise_scale: float = 0.6
    confounding_scale: float = 0.7
    outcome_confounding_loading: float = 0.7
    outcome_noise_scale: float = 0.5
    novel_shift: float = 3.0
    novel_channel_effect: float = float(_BASE_THETA[_S5_FEATURE_INDEX])
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID
    structural_lambda_grid: tuple[float, ...] | None = None
    structural_gamma_grid: tuple[float, ...] | None = None
    dual_lambda_grid: tuple[float, ...] | None = None
    dual_gamma_grid: tuple[float, ...] | None = None
    projection_lambda_grid: tuple[float, ...] | None = None
    projection_gamma_grid: tuple[float, ...] | None = None
    projection_method: str = "spectral_npjive"
    projection_multiplier_replications: int = 199
    projection_multiplier_quantile: float = 0.95
    inverse_stability_multiplier_replications: int = 99
    inverse_stability_multiplier_quantile: float = 0.95
    inverse_stability_margin: float = 1.25
    projection_max_fold_coef_norm: float = 10_000.0
    projection_min_system_eigenvalue: float = 1e-10
    projection_max_system_condition: float = 1e10
    star_nuisance_max_fold_coef_norm: float = 10_000.0
    star_nuisance_min_system_eigenvalue: float = 1e-10
    star_nuisance_max_system_condition: float = 1e4
    tuning_stability_rule: str = "generalized_positive_condition_le_10000"

    @property
    def family(self) -> str:
        return "longterm_surrogate"

    @property
    def target_sample_size(self) -> int:
        return int(15 * self.n_instruments if self.n_new_per_arm is None else self.n_new_per_arm)

    @property
    def growth_label(self) -> str:
        if self.growth_path == "fixed_n30":
            return f"fixed per-experiment n={self.n_per_instrument}"
        if self.growth_path == "slow_log":
            return "slow per-experiment growth"
        if self.growth_path == "sqrt_n":
            return "sqrt per-experiment growth"
        if self.growth_path == "large_n120":
            return "large per-experiment n=120 calibration"
        if self.growth_path == "large_n400":
            return "large per-experiment n=400 calibration"
        raise ValueError(f"Unknown growth_path {self.growth_path!r}.")

    @property
    def historical_total_n(self) -> int:
        return int(self.n_instruments * self.n_per_instrument)

    @property
    def target_sample_rate(self) -> float:
        return float(self.target_sample_size / self.n_instruments)

    @property
    def triangular_array_label(self) -> str:
        if self.growth_path == "fixed_n30":
            return f"K grows; historical experiment size fixed at {self.n_per_instrument}"
        if self.growth_path == "slow_log":
            return "K grows; historical experiment size grows as ceil(8 log K)"
        if self.growth_path == "sqrt_n":
            return "K grows; historical experiment size grows as ceil(5 sqrt K)"
        if self.growth_path == "large_n120":
            return "large-n calibration; historical experiment size fixed at 120"
        if self.growth_path == "large_n400":
            return "large-n calibration; historical experiment size fixed at 400"
        raise ValueError(f"Unknown growth_path {self.growth_path!r}.")

    @property
    def eta(self) -> float:
        ratio = self.n_instruments / 50.0
        if self.regime == "covered":
            return 0.0
        if self.regime == "vanishing":
            return min(0.95, 0.20 * ratio ** -1.5)
        if self.regime == "persistent":
            return 0.10
        if self.regime == "boundary":
            return 0.12 * ratio ** -0.5
        raise ValueError(f"Unknown regime {self.regime!r}.")


@dataclass(frozen=True)
class EffectFitBundle:
    """Fitted nuisance bundle used by the long-term contrast estimators."""

    estimates: dict[str, FunctionalEstimate]
    structural: StructuralFitResult
    treated_dual: DualFitResult
    control_dual: DualFitResult
    star_diagnostics: dict[str, Any]
    star_target_gradients: dict[str, np.ndarray]
    inference_diagnostics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RieszRepresenterFit:
    """Deployable ridge fit for an observed target-law Riesz representer."""

    coef: np.ndarray
    gamma: float
    cv_risk: float
    coef_norm: float
    moment_residual_norm: float
    linear_operator: np.ndarray


@dataclass(frozen=True)
class RieszProjectionFit:
    """Observed-data npJIVE projection of a fitted Riesz representer."""

    projected_coef: np.ndarray
    residual_coef: np.ndarray
    lambda_value: float
    gamma: float
    cv_risk: float
    projected_coef_norm: float
    residual_coef_norm: float
    residual_group_mean_norm: float
    system_eigen_min: float
    system_condition: float
    negative_spectral_edge: float
    fold_operator_deviation: float
    retained_rank: float
    largest_discarded_eigenvalue: float
    projection_operator: np.ndarray
    objective: str


@dataclass(frozen=True)
class NPJIVEInverseFit:
    """Strict npJIVE inverse fit for a supplied observed linear term."""

    coef: np.ndarray
    lambda_value: float
    gamma: float
    cv_risk: float
    coef_norm: float
    system_eigen_min: float
    system_condition: float
    inverse_operator: np.ndarray
    operator_noise_floor: float
    operator_noise_margin: float
    generalized_system_eigen_min: float
    generalized_system_condition: float


@dataclass(frozen=True)
class StarMeanFit:
    """Projection-corrected mean estimate and its correction diagnostics."""

    estimate: FunctionalEstimate
    diagnostics: dict[str, Any]
    target_gradient_coef: np.ndarray
    historical_variance: dict[str, Any]


@dataclass(frozen=True)
class CrossFittedStructuralScores:
    """Out-of-fold structural nuisance predictions for historical rows."""

    folds: np.ndarray
    h_hat: np.ndarray
    fold_coefs: dict[int, np.ndarray]
    fold_fits: dict[int, StructuralFitResult]


@dataclass(frozen=True)
class LOOMeanFit:
    """LOO mean estimate and predictions for its symmetric pair kernel."""

    estimate: FunctionalEstimate
    beta_predictions: np.ndarray
    target_predictions: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
    return value


def _csv_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def config_to_dict(config: LongTermSurrogateConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["family"] = config.family
    payload["eta"] = config.eta
    payload["target_sample_size"] = config.target_sample_size
    payload["growth_label"] = config.growth_label
    payload["historical_experiment_count"] = config.n_instruments
    payload["historical_per_experiment_n"] = config.n_per_instrument
    payload["historical_total_n"] = config.historical_total_n
    payload["target_sample_rate"] = config.target_sample_rate
    payload["triangular_array_label"] = config.triangular_array_label
    payload["effective_structural_lambda_grid"] = _structural_lambda_grid(config)
    payload["effective_structural_gamma_grid"] = _structural_gamma_grid(config)
    payload["effective_dual_lambda_grid"] = _dual_lambda_grid(config)
    payload["effective_dual_gamma_grid"] = _dual_gamma_grid(config)
    payload["effective_projection_lambda_grid"] = _projection_lambda_grid(config)
    payload["effective_projection_gamma_grid"] = _projection_gamma_grid(config)
    return payload


def config_hash(config: LongTermSurrogateConfig) -> str:
    rendered = json.dumps(_jsonable(config_to_dict(config)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _source_snapshot_id() -> str:
    root = _repo_root()
    source_paths = [
        *sorted((root / "experiments" / "paper").glob("*.py")),
        root / "experiments" / "run.py",
        *sorted((root / "src" / "discreteNPIV").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in sorted(source_paths):
        if path.exists():
            digest.update(str(path.relative_to(_repo_root())).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:12]


def run_provenance(*, command: str | None = None) -> dict[str, Any]:
    """Return reproducibility metadata for a paper simulation run."""

    status = _git_value(["status", "--short"])
    return {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "run_command": "" if command is None else command,
        "package_version": __version__,
        "source_snapshot_id": _source_snapshot_id(),
        "git_head": _git_value(["rev-parse", "HEAD"]),
        "dirty_worktree": bool(status),
        "dirty_worktree_status": status,
    }


def _feature_spec(config: LongTermSurrogateConfig, n_fourier_pairs: int | None = None) -> FeatureSpec:
    n_pairs = config.n_fourier_pairs if n_fourier_pairs is None else int(n_fourier_pairs)
    rng_directions = np.random.default_rng(config.feature_seed)
    rng_scales = np.random.default_rng(config.feature_seed + 1)
    rng_phases = np.random.default_rng(config.feature_seed + 2)
    directions = rng_directions.normal(size=(n_pairs, 5))
    # The RKHS component is an ordinary-surrogate smooth effect. Keeping S5 out
    # of these random features makes the novel-channel identification gap
    # transparent and driven by the explicit S5 term.
    if n_pairs:
        directions[:, _S5_FEATURE_INDEX] = 0.0
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    directions *= rng_scales.uniform(0.5, 2.0, size=(n_pairs, 1))
    phases = rng_phases.uniform(-np.pi, np.pi, size=n_pairs)
    return FeatureSpec(directions=directions, phases=phases)


def _structural_lambda_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    return config.lambda_grid if config.structural_lambda_grid is None else config.structural_lambda_grid


def _structural_gamma_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    return config.gamma_grid if config.structural_gamma_grid is None else config.structural_gamma_grid


def _dual_lambda_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    return config.lambda_grid if config.dual_lambda_grid is None else config.dual_lambda_grid


def _dual_gamma_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    return config.gamma_grid if config.dual_gamma_grid is None else config.dual_gamma_grid


def _projection_lambda_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    if config.projection_lambda_grid is not None:
        return config.projection_lambda_grid
    if config.projection_method == "spectral_npjive":
        return ()
    return _structural_lambda_grid(config)


def _projection_gamma_grid(config: LongTermSurrogateConfig) -> tuple[float, ...]:
    return _structural_gamma_grid(config) if config.projection_gamma_grid is None else config.projection_gamma_grid


def _nuisance_n_splits_for_codes(Z: np.ndarray, requested: int) -> int:
    """Use up to the requested number of folds while keeping two units per arm/fold."""

    counts = group_counts(np.asarray(Z, dtype=int))
    supported = int(np.min(counts) // 2)
    if supported < 2:
        raise ValueError("nonparametric nuisance CV requires at least 4 units per instrument level.")
    return int(min(int(requested), supported))
