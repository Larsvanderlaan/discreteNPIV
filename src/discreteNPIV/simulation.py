from __future__ import annotations

import numpy as np


def effective_sparsity(alpha: float, indices: np.ndarray) -> float:
    """Return the effective sparsity implied by Sobolev-style decay."""

    decay = indices ** (-alpha)
    l1_sq = np.sum(decay) ** 2
    l2_sq = np.sum(decay**2)
    return float(l1_sq / l2_sq)


def find_alpha(target_sparsity: float, n_features: int, *, tol: float = 1e-6) -> float:
    """Solve for the decay exponent whose effective sparsity matches the target."""

    indices = np.arange(1, n_features + 1, dtype=float)
    lo = 0.0
    hi = 10.0

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        current = effective_sparsity(mid, indices)
        if abs(current - target_sparsity) <= tol:
            return mid
        if current > target_sparsity:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_synthetic_data(
    *,
    n_per_instrument: int,
    n_instruments: int,
    n_features: int = 5,
    n_target_samples: int = 10_000,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    sigma_c: float = 0.1,
    sparsity_structural: float | None = None,
    sparsity_target: float | None = None,
    random_state: int = 42,
) -> dict[str, np.ndarray | float]:
    """
    Generate a deterministic synthetic NPIV data set.

    Parameters
    ----------
    n_per_instrument:
        Number of observations per instrument level.
    n_instruments:
        Number of discrete instrument levels.
    n_features:
        Number of basis features in the linear representation.
    n_target_samples:
        Number of samples used to approximate the target covariate distribution.
    sigma_x, sigma_y, sigma_c:
        Noise scales for treatment, outcome, and shared confounding terms.
    sparsity_structural, sparsity_target:
        Effective sparsity controls for the structural parameter and target
        covariate distribution. If omitted, a mild default based on the feature
        dimension is used.
    random_state:
        Seed controlling the full synthetic draw.

    Returns
    -------
    dict
        Dictionary containing ``X``, ``Z``, ``Y``, ``X_new``, the structural
        coefficient vector ``theta``, and the instrument-level means ``Pi``.
    """

    if sparsity_structural is None:
        sparsity_structural = min(5.0, 0.1 * n_features)
    if sparsity_target is None:
        sparsity_target = sparsity_structural

    alpha_structural = find_alpha(float(sparsity_structural), n_features)
    alpha_target = find_alpha(float(sparsity_target), n_features)

    target_decay = np.array([j ** (-alpha_target) for j in range(1, n_features + 1)], dtype=float)
    structural_decay = np.array([j ** (-alpha_structural) for j in range(1, n_features + 1)], dtype=float)

    rng = np.random.default_rng(random_state)
    pi_target = rng.standard_normal(n_features) * target_decay
    X_new = (
        np.repeat(pi_target[None, :], n_target_samples, axis=0)
        + rng.standard_normal((n_target_samples, 1)) * sigma_x
        + rng.standard_normal((n_target_samples, 1)) * sigma_c
    )

    pi = rng.standard_normal((n_instruments, n_features)) * target_decay[None, :]
    theta = (2.0 * (rng.standard_normal(n_features) >= 0.0) - 1.0) * structural_decay

    u_x = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_x
    u_y = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_y
    u_c = rng.standard_normal((n_per_instrument, n_instruments)) * sigma_c

    X = pi[None, :, :] + u_x[:, :, None] + u_c[:, :, None]
    X = X.reshape(n_per_instrument * n_instruments, n_features)
    Y = X @ theta + u_y.ravel() + u_c.ravel()
    Z = np.tile(np.arange(n_instruments), n_per_instrument)

    return {
        "theta": theta,
        "Y": Y,
        "X": X,
        "Z": Z,
        "Pi": pi,
        "X_new": X_new,
        "alpha_structural": alpha_structural,
        "alpha_target": alpha_target,
    }
