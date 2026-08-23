from __future__ import annotations

from typing import Any

import numpy as np

from discreteNPIV._linear_solvers import solve_regularized_system
from discreteNPIV.grouping import group_counts, group_means, make_stratified_folds

from .model import (
    FeatureSpec,
    LongTermSurrogateConfig,
    RieszRepresenterFit,
    VALID_REGIMES,
    _BASE_THETA,
    _CONFOUNDING_DIRECTION,
    _NOVEL_DIRECTION,
    _S5_FEATURE_INDEX,
    _dual_gamma_grid,
    _feature_spec,
    _nuisance_n_splits_for_codes,
)


def _theta(spec: FeatureSpec, config: LongTermSurrogateConfig) -> np.ndarray:
    theta = np.zeros(spec.n_features, dtype=float)
    theta[: _BASE_THETA.shape[0]] = _BASE_THETA
    theta[_S5_FEATURE_INDEX] = config.novel_channel_effect
    n_pairs = spec.directions.shape[0]
    if n_pairs:
        rng = np.random.default_rng(config.feature_seed + 3)
        raw = rng.normal(size=(max(n_pairs, config.truth_fourier_pairs), 2))
        orders = np.arange(1, raw.shape[0] + 1, dtype=float)
        scales = config.truth_fourier_amplitude / np.power(orders, config.truth_fourier_smoothness)
        coefficients = raw[:n_pairs] * scales[:n_pairs, None]
        theta[7 : 7 + n_pairs] = coefficients[:, 0]
        theta[7 + n_pairs : 7 + 2 * n_pairs] = coefficients[:, 1]
    return theta


def _mu(t: np.ndarray | float, *, signal_scale: float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return signal_scale * np.stack(
        [
            t_arr - 0.5,
            0.5 * np.sin(2.0 * np.pi * t_arr),
            0.5 * np.cos(2.0 * np.pi * t_arr),
            (t_arr - 0.5) ** 2 - 1.0 / 12.0,
            np.zeros_like(t_arr),
        ],
        axis=-1,
    )


def _surrogate_cov(config: LongTermSurrogateConfig) -> np.ndarray:
    return (
        config.confounding_scale**2 * np.outer(_CONFOUNDING_DIRECTION, _CONFOUNDING_DIRECTION)
        + config.surrogate_noise_scale**2 * np.eye(5)
    )


def surrogate_feature_map(raw_surrogates: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """
    Map raw five-dimensional surrogates to the finite feature representation.

    The first seven coordinates are interpretable low-order terms, and the
    remaining coordinates are a random Fourier sieve for the smooth RKHS part.
    The fifth raw surrogate is a novel-channel coordinate: historical
    experiments never move its arm-level mean, but the new experiment can shift
    it.
    """

    raw = np.asarray(raw_surrogates, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise ValueError("raw_surrogates must be a 2D array with five columns.")
    s1, s2, s3, s4, s5 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
    base = np.column_stack([np.ones(raw.shape[0]), s1, s2, s3, s5, s1**2 - 1.0, s2 * s4])
    projected = raw @ spec.directions.T + spec.phases[None, :]
    return np.column_stack([base, np.sin(projected), np.cos(projected)])


def _expected_features(mean: np.ndarray, cov: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    base = np.array(
        [
            1.0,
            mean[0],
            mean[1],
            mean[2],
            mean[4],
            cov[0, 0] + mean[0] ** 2 - 1.0,
            cov[1, 3] + mean[1] * mean[3],
        ],
        dtype=float,
    )
    linear_mean = spec.directions @ mean + spec.phases
    linear_var = np.einsum("ij,jk,ik->i", spec.directions, cov, spec.directions)
    attenuation = np.exp(-0.5 * linear_var)
    return np.concatenate([base, attenuation * np.sin(linear_mean), attenuation * np.cos(linear_mean)])


def _historical_locations(K: int) -> np.ndarray:
    if K < 2:
        raise ValueError("n_instruments must be at least 2.")
    return np.linspace(0.0, 1.0, K)


def _kernel_weights(locations: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    scaled = -0.5 * ((locations - center) / bandwidth) ** 2
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / np.sum(weights)


def _regime_label(regime: str) -> str:
    labels = {
        "covered": "covered target",
        "vanishing": "vanishing gap",
        "persistent": "persistent gap",
        "boundary": "boundary/local gap",
    }
    return labels[regime]


def _arm_population_features(
    config: LongTermSurrogateConfig,
    *,
    center: float,
    novel_shift: float,
) -> dict[str, np.ndarray | float]:
    spec = _feature_spec(config)
    truth_spec = _feature_spec(config, n_fourier_pairs=config.truth_fourier_pairs)
    theta = _theta(spec, config)
    truth_theta = _theta(truth_spec, config)
    locations = _historical_locations(config.n_instruments)
    cov = _surrogate_cov(config)
    historical_means = _mu(locations, signal_scale=config.signal_scale)
    historical_features = np.vstack([_expected_features(mean, cov, spec) for mean in historical_means])
    truth_historical_features = np.vstack([_expected_features(mean, cov, truth_spec) for mean in historical_means])
    weights = _kernel_weights(locations, center, config.kernel_bandwidth)
    mixture_features = weights @ historical_features
    truth_mixture_features = weights @ truth_historical_features
    shifted_historical_means = historical_means + novel_shift * _NOVEL_DIRECTION
    shifted_historical_features = np.vstack([_expected_features(mean, cov, spec) for mean in shifted_historical_means])
    truth_shifted_historical_features = np.vstack(
        [_expected_features(mean, cov, truth_spec) for mean in shifted_historical_means]
    )
    novel_features = weights @ shifted_historical_features
    truth_novel_features = weights @ truth_shifted_historical_features
    novel_raw_mean = weights @ shifted_historical_means
    target_features = (1.0 - config.eta) * mixture_features + config.eta * novel_features
    truth_target_features = (1.0 - config.eta) * truth_mixture_features + config.eta * truth_novel_features
    return {
        "historical_features": historical_features,
        "mixture_features": mixture_features,
        "novel_features": novel_features,
        "target_features": target_features,
        "mixture_truth": float(truth_mixture_features @ truth_theta),
        "target_truth": float(truth_target_features @ truth_theta),
        "novel_truth": float(truth_novel_features @ truth_theta),
        "mixture_sieve_truth": float(mixture_features @ theta),
        "target_sieve_truth": float(target_features @ theta),
        "novel_sieve_truth": float(novel_features @ theta),
        "novel_raw_mean": novel_raw_mean,
        "weights": weights,
    }


def generate_longterm_surrogate_data(config: LongTermSurrogateConfig, replication: int) -> dict[str, Any]:
    """Generate one deterministic replication of the long-term surrogate design."""

    if config.regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {sorted(VALID_REGIMES)}.")
    spec = _feature_spec(config)
    truth_spec = _feature_spec(config, n_fourier_pairs=config.truth_fourier_pairs)
    theta = _theta(spec, config)
    truth_theta = _theta(truth_spec, config)
    rng = np.random.default_rng(config.seed_base + 10_000 + replication)
    locations = _historical_locations(config.n_instruments)
    historical_means = _mu(locations, signal_scale=config.signal_scale)

    U = rng.normal(size=(config.n_instruments, config.n_per_instrument, 1))
    eps = rng.normal(size=(config.n_instruments, config.n_per_instrument, 5))
    raw_hist = (
        historical_means[:, None, :]
        + config.confounding_scale * U * _CONFOUNDING_DIRECTION[None, None, :]
        + config.surrogate_noise_scale * eps
    )
    raw_hist_flat = raw_hist.reshape(-1, 5)
    X = surrogate_feature_map(raw_hist_flat, spec)
    X_truth = surrogate_feature_map(raw_hist_flat, truth_spec)
    structural = X_truth @ truth_theta
    sieve_structural = X @ theta
    Y = (
        structural
        + config.outcome_confounding_loading * U.reshape(-1)
        + rng.normal(scale=config.outcome_noise_scale, size=structural.shape[0])
    )
    Z = np.repeat(np.arange(config.n_instruments), config.n_per_instrument)

    control = _arm_population_features(config, center=config.control_location, novel_shift=-config.novel_shift)
    treated = _arm_population_features(config, center=config.treated_location, novel_shift=config.novel_shift)
    X_new_control = _sample_new_arm(
        config,
        rng,
        locations=locations,
        historical_means=historical_means,
        weights=np.asarray(control["weights"], dtype=float),
        novel_shift=-config.novel_shift,
        spec=spec,
    )
    X_new_treated = _sample_new_arm(
        config,
        rng,
        locations=locations,
        historical_means=historical_means,
        weights=np.asarray(treated["weights"], dtype=float),
        novel_shift=config.novel_shift,
        spec=spec,
    )

    truth = float(treated["target_truth"] - control["target_truth"])
    projection_truth = float(treated["mixture_truth"] - control["mixture_truth"])
    sieve_truth = float(treated["target_sieve_truth"] - control["target_sieve_truth"])
    sieve_projection_truth = float(treated["mixture_sieve_truth"] - control["mixture_sieve_truth"])
    identification_gap = truth - projection_truth
    s5_feature_gap = float(
        (np.asarray(treated["target_features"], dtype=float)[_S5_FEATURE_INDEX]
         - np.asarray(control["target_features"], dtype=float)[_S5_FEATURE_INDEX])
        - (np.asarray(treated["mixture_features"], dtype=float)[_S5_FEATURE_INDEX]
           - np.asarray(control["mixture_features"], dtype=float)[_S5_FEATURE_INDEX])
    )
    s5_truth_gap = float(theta[_S5_FEATURE_INDEX] * s5_feature_gap)
    return {
        "X": X,
        "Z": Z,
        "Y": Y,
        "raw_surrogates": raw_hist_flat,
        "X_new_treated": X_new_treated,
        "X_new_control": X_new_control,
        "truth": truth,
        "projection_truth": projection_truth,
        "identification_gap": identification_gap,
        "theta": theta,
        "truth_theta": truth_theta,
        "target_feature_mean_treated": np.asarray(treated["target_features"], dtype=float),
        "target_feature_mean_control": np.asarray(control["target_features"], dtype=float),
        "projection_feature_mean_treated": np.asarray(treated["mixture_features"], dtype=float),
        "projection_feature_mean_control": np.asarray(control["mixture_features"], dtype=float),
        "projection_weights_treated": np.asarray(treated["weights"], dtype=float),
        "projection_weights_control": np.asarray(control["weights"], dtype=float),
        "historical_feature_means": np.asarray(control["historical_features"], dtype=float),
        "historical_raw_arm_means": historical_means,
        "novel_raw_mean_treated": np.asarray(treated["novel_raw_mean"], dtype=float),
        "novel_raw_mean_control": np.asarray(control["novel_raw_mean"], dtype=float),
        "metadata": {
            "eta": config.eta,
            "regime_label": _regime_label(config.regime),
            "target_sample_size": config.target_sample_size,
            "s5_historical_population_mean_abs_max": float(np.max(np.abs(historical_means[:, 4]))),
            "s5_identification_gap": s5_feature_gap,
            "s5_truth_gap": s5_truth_gap,
            "truth_treated": float(treated["target_truth"]),
            "truth_control": float(control["target_truth"]),
            "projection_truth_treated": float(treated["mixture_truth"]),
            "projection_truth_control": float(control["mixture_truth"]),
            "identification_gap_abs": abs(identification_gap),
            "novel_truth_treated": float(treated["novel_truth"]),
            "novel_truth_control": float(control["novel_truth"]),
            "sieve_truth": sieve_truth,
            "sieve_projection_truth": sieve_projection_truth,
            "rkhs_sieve_target_gap": float(truth - sieve_truth),
            "rkhs_sieve_projection_gap": float(projection_truth - sieve_projection_truth),
            "rkhs_sieve_historical_rmse": float(np.sqrt(np.mean((structural - sieve_structural) ** 2))),
            "rkhs_truth_fourier_pairs": int(config.truth_fourier_pairs),
            "rkhs_estimator_fourier_pairs": int(config.n_fourier_pairs),
        },
    }


def _sample_new_arm(
    config: LongTermSurrogateConfig,
    rng: np.random.Generator,
    *,
    locations: np.ndarray,
    historical_means: np.ndarray,
    weights: np.ndarray,
    novel_shift: float,
    spec: FeatureSpec,
) -> np.ndarray:
    m = config.target_sample_size
    use_novel = rng.random(m) < config.eta
    historical_index = rng.choice(locations.shape[0], size=m, p=weights)
    means = historical_means[historical_index].copy()
    if np.any(use_novel):
        means[use_novel] = historical_means[historical_index[use_novel]] + novel_shift * _NOVEL_DIRECTION
    U = rng.normal(size=(m, 1))
    eps = rng.normal(size=(m, 5))
    raw = means + config.confounding_scale * U * _CONFOUNDING_DIRECTION[None, :] + config.surrogate_noise_scale * eps
    return surrogate_feature_map(raw, spec)


def _design_diagnostics(X: np.ndarray, Z: np.ndarray) -> dict[str, float | int]:
    counts = group_counts(Z)
    grouped_X = group_means(X, Z, n_groups=counts.shape[0])
    singular_values = np.linalg.svd(grouped_X, compute_uv=False)
    smallest = float(np.min(singular_values)) if singular_values.size else np.nan
    largest = float(np.max(singular_values)) if singular_values.size else np.nan
    condition = np.inf if smallest <= 1e-12 else largest / smallest
    between = float(np.mean(np.var(grouped_X, axis=0)))
    within = float(np.mean(np.var(X - grouped_X[Z], axis=0)))
    return {
        "n_obs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_instrument_levels": int(counts.shape[0]),
        "min_instrument_count": int(np.min(counts)),
        "max_instrument_count": int(np.max(counts)),
        "grouped_design_smin": smallest,
        "grouped_design_smax": largest,
        "grouped_design_condition": float(condition),
        "instrument_strength_proxy": np.nan if within <= 0.0 else between / within,
    }


def _fit_riesz_representer(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    X_new: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> RieszRepresenterFit:
    """Fit alpha for the observed target law by deployable CV Riesz risk."""

    return _fit_riesz_representer_for_feature_mean(
        X=X,
        Z=Z,
        target_feature_mean=np.mean(X_new, axis=0),
        config=config,
        random_state=random_state,
    )


def _fit_riesz_representer_for_feature_mean(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    target_feature_mean: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> RieszRepresenterFit:
    """Fit alpha for a linear feature-mean functional by deployable CV risk."""

    mu_new = np.asarray(target_feature_mean, dtype=float)
    if mu_new.ndim != 1 or mu_new.shape[0] != X.shape[1]:
        raise ValueError("target_feature_mean must be a length-p feature vector.")
    n_splits = _nuisance_n_splits_for_codes(Z, config.n_splits)
    folds = make_stratified_folds(Z, n_splits=n_splits, random_state=random_state)
    gamma_grid = np.asarray(_dual_gamma_grid(config), dtype=float)
    eye = np.eye(X.shape[1])
    candidates: list[dict[str, float]] = []

    for gamma in gamma_grid:
        total_risk = 0.0
        for fold_id in np.unique(folds):
            train_mask = folds != fold_id
            valid_mask = folds == fold_id
            C_train = X[train_mask].T @ X[train_mask] / np.sum(train_mask)
            C_valid = X[valid_mask].T @ X[valid_mask] / np.sum(valid_mask)
            coef = solve_regularized_system(C_train + gamma * eye, mu_new)
            total_risk += float(coef.T @ C_valid @ coef - 2.0 * mu_new @ coef)
        candidates.append({"gamma": float(gamma), "cv_risk": float(total_risk / len(np.unique(folds)))})

    selected = min(candidates, key=lambda item: item["cv_risk"])
    C = X.T @ X / X.shape[0]
    linear_operator = solve_regularized_system(
        C + selected["gamma"] * eye,
        eye,
    )
    coef = linear_operator @ mu_new
    return RieszRepresenterFit(
        coef=coef,
        gamma=float(selected["gamma"]),
        cv_risk=float(selected["cv_risk"]),
        coef_norm=float(np.linalg.norm(coef)),
        moment_residual_norm=float(np.linalg.norm(C @ coef - mu_new)),
        linear_operator=linear_operator,
    )
