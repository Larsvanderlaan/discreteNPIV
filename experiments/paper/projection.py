from __future__ import annotations

import hashlib

import numpy as np

from discreteNPIV._linear_solvers import solve_regularized_system
from discreteNPIV._tuning import (
    EstimationTerms,
    FoldTerms,
    OperatorStabilityConstraint,
    select_regularization,
)
from discreteNPIV.api import _primal_npjive_terms
from discreteNPIV.grouping import group_means, make_stratified_folds

from .model import (
    LongTermSurrogateConfig,
    RieszProjectionFit,
    _nuisance_n_splits_for_codes,
    _projection_gamma_grid,
    _projection_lambda_grid,
)


_OPERATOR_STABILITY_CACHE: dict[str, OperatorStabilityConstraint] = {}
_OPERATOR_STABILITY_CACHE_MAXSIZE = 128


def _spectral_npjive_projection(
    *,
    alpha_coef: np.ndarray,
    terms: EstimationTerms,
    threshold: float,
) -> tuple[np.ndarray, float, float]:
    """Project without shrinking retained generalized npJIVE eigendirections."""

    projection_operator, eigen_min, condition = _spectral_npjive_projection_operator(
        terms=terms,
        threshold=threshold,
    )
    projected = projection_operator @ np.asarray(alpha_coef, dtype=float)
    return projected, eigen_min, condition


def _spectral_npjive_projection_operator(
    *,
    terms: EstimationTerms,
    threshold: float,
) -> tuple[np.ndarray, float, float]:
    """Return the coefficient-space hard projector and retained spectrum."""

    whitened, C_sqrt, C_inverse_sqrt = _whitened_npjive_operator(terms)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    retained = eigenvalues > float(threshold)
    if np.any(retained):
        retained_vectors = eigenvectors[:, retained]
        projector = retained_vectors @ retained_vectors.T
        retained_eigenvalues = eigenvalues[retained]
        eigen_min = float(np.min(retained_eigenvalues))
        condition = float(np.max(retained_eigenvalues) / eigen_min)
    else:
        projector = np.zeros_like(whitened)
        eigen_min = np.nan
        condition = 1.0
    projection_operator = C_inverse_sqrt @ projector @ C_sqrt
    return projection_operator, eigen_min, condition


def _whitened_npjive_operator(
    terms: EstimationTerms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the symmetric npJIVE operator in the empirical ``C`` geometry."""

    C = 0.5 * (terms.penalty + terms.penalty.T)
    c_eigenvalues, c_eigenvectors = np.linalg.eigh(C)
    c_floor = max(float(np.max(c_eigenvalues)) * 1e-10, 1e-10)
    c_eigenvalues = np.maximum(c_eigenvalues, c_floor)
    C_sqrt = (c_eigenvectors * np.sqrt(c_eigenvalues)) @ c_eigenvectors.T
    C_inverse_sqrt = (c_eigenvectors * (1.0 / np.sqrt(c_eigenvalues))) @ c_eigenvectors.T
    whitened = C_inverse_sqrt @ (0.5 * (terms.quadratic + terms.quadratic.T)) @ C_inverse_sqrt
    return 0.5 * (whitened + whitened.T), C_sqrt, C_inverse_sqrt


def _multiplier_npjive_noise_floor(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    terms: EstimationTerms,
    n_replications: int,
    quantile: float,
    random_state: int,
    two_sided: bool = False,
) -> float:
    """Calibrate an observed-operator noise floor with centered multipliers."""

    if n_replications < 1:
        raise ValueError("n_replications must be positive.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    _, _, C_inverse_sqrt = _whitened_npjive_operator(terms)
    sorted_groups, first_indices, sorted_inverse = np.unique(
        Z_arr,
        return_index=True,
        return_inverse=True,
    )
    first_order = np.argsort(first_indices)
    groups = sorted_groups[first_order]
    sorted_to_first = np.empty_like(first_order)
    sorted_to_first[first_order] = np.arange(first_order.size)
    inverse = sorted_to_first[sorted_inverse]
    counts = np.bincount(inverse, minlength=groups.size)
    if np.any(counts < 2):
        raise ValueError("Multiplier npJIVE calibration requires at least two observations per group.")

    centered = np.array(X_arr, copy=True)
    for group_index in range(groups.size):
        mask = inverse == group_index
        centered[mask] -= np.mean(centered[mask], axis=0)
        centered[mask] *= np.sqrt(counts[group_index] / (counts[group_index] - 1.0))
    whitened_residuals = centered @ C_inverse_sqrt
    weights = counts.astype(float) / X_arr.shape[0]
    scales = weights / (counts * (counts - 1.0))
    rng = np.random.default_rng(random_state)

    if np.all(counts == counts[0]):
        order = np.argsort(inverse, kind="stable")
        grouped = whitened_residuals[order].reshape(groups.size, int(counts[0]), X_arr.shape[1])
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(n_replications, groups.size, int(counts[0])),
        )
        group_sums = np.einsum("bkn,knp->bkp", signs, grouped, optimize=True)
        null_operators = np.einsum(
            "bkp,bkq,k->bpq",
            group_sums,
            group_sums,
            scales,
            optimize=True,
        )
        diagonal = np.einsum("knp,knq,k->pq", grouped, grouped, scales, optimize=True)
        null_operators -= diagonal[None, :, :]
    else:
        dimension = X_arr.shape[1]
        null_operators = np.zeros((n_replications, dimension, dimension), dtype=float)
        for group_index in range(groups.size):
            group_residuals = whitened_residuals[inverse == group_index]
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(n_replications, group_residuals.shape[0]),
            )
            group_sums = signs @ group_residuals
            null_operators += scales[group_index] * np.einsum(
                "bp,bq->bpq",
                group_sums,
                group_sums,
                optimize=True,
            )
            null_operators -= scales[group_index] * (
                group_residuals.T @ group_residuals
            )[None, :, :]

    null_operators = 0.5 * (null_operators + np.swapaxes(null_operators, 1, 2))
    null_eigenvalues = np.linalg.eigvalsh(null_operators)
    if two_sided:
        upper_edges = np.max(np.abs(null_eigenvalues), axis=1)
    else:
        upper_edges = np.maximum(null_eigenvalues[:, -1], 0.0)
    order_index = min(
        int(np.ceil((n_replications + 1) * quantile)),
        n_replications,
    ) - 1
    return float(np.sort(upper_edges)[order_index])


def _npjive_operator_stability_constraint(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    n_splits: int,
    config: LongTermSurrogateConfig,
    random_state: int,
) -> OperatorStabilityConstraint | None:
    """Build a design-only inverse-stability constraint for one CV fit."""

    if config.tuning_stability_rule in {
        "none",
        "fold_coef_norm_le_10000",
        "generalized_positive_condition_le_10000",
    }:
        return None
    if config.tuning_stability_rule != "two_sided_multiplier_generalized_eigen_floor":
        raise ValueError(f"Unknown tuning_stability_rule {config.tuning_stability_rule!r}.")
    if config.inverse_stability_multiplier_replications < 1:
        raise ValueError("inverse stability multiplier replications must be positive.")
    if not 0.0 < config.inverse_stability_multiplier_quantile < 1.0:
        raise ValueError("inverse stability multiplier quantile must lie between zero and one.")
    if config.inverse_stability_margin <= 1.0:
        raise ValueError("inverse stability margin must exceed one.")

    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.ascontiguousarray(X_arr).view(np.uint8))
    digest.update(np.ascontiguousarray(Z_arr).view(np.uint8))
    digest.update(
        repr(
            (
                int(n_splits),
                int(random_state),
                int(config.inverse_stability_multiplier_replications),
                float(config.inverse_stability_multiplier_quantile),
                float(config.inverse_stability_margin),
            )
        ).encode("ascii")
    )
    cache_key = digest.hexdigest()
    cached = _OPERATOR_STABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    folds = make_stratified_folds(
        Z_arr,
        n_splits=n_splits,
        random_state=random_state,
    )
    full_terms = _primal_npjive_terms(
        X_arr,
        Z_arr,
        np.zeros(X_arr.shape[0], dtype=float),
    )
    full_floor = _multiplier_npjive_noise_floor(
        X=X_arr,
        Z=Z_arr,
        terms=full_terms,
        n_replications=config.inverse_stability_multiplier_replications,
        quantile=config.inverse_stability_multiplier_quantile,
        random_state=random_state + 910_000,
        two_sided=True,
    )
    fold_floors: list[float] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        train_terms = _primal_npjive_terms(
            X_arr[train_mask],
            Z_arr[train_mask],
            np.zeros(int(np.sum(train_mask)), dtype=float),
        )
        fold_floors.append(
            _multiplier_npjive_noise_floor(
                X=X_arr[train_mask],
                Z=Z_arr[train_mask],
                terms=train_terms,
                n_replications=config.inverse_stability_multiplier_replications,
                quantile=config.inverse_stability_multiplier_quantile,
                random_state=random_state + 911_000 + int(fold_id),
                two_sided=True,
            )
        )
    constraint = OperatorStabilityConstraint(
        fold_noise_floors=tuple(fold_floors),
        full_terms=full_terms,
        full_noise_floor=float(full_floor),
        margin=float(config.inverse_stability_margin),
    )
    if len(_OPERATOR_STABILITY_CACHE) >= _OPERATOR_STABILITY_CACHE_MAXSIZE:
        _OPERATOR_STABILITY_CACHE.pop(next(iter(_OPERATOR_STABILITY_CACHE)))
    _OPERATOR_STABILITY_CACHE[cache_key] = constraint
    return constraint


def _spectral_npjive_stability_edges(
    full_terms: EstimationTerms,
    fold_train_terms: list[EstimationTerms],
) -> tuple[float, float, float]:
    """Return heuristic negative/fold edges for multiplier-cutoff diagnostics."""

    full_operator, _, C_inverse_sqrt = _whitened_npjive_operator(full_terms)
    negative_edge = max(0.0, -float(np.min(np.linalg.eigvalsh(full_operator))))
    fold_deviation = max(
        (
            float(
                np.linalg.norm(
                    C_inverse_sqrt
                    @ (0.5 * (terms.quadratic + terms.quadratic.T))
                    @ C_inverse_sqrt
                    - full_operator,
                    ord=2,
                )
            )
            for terms in fold_train_terms
        ),
        default=0.0,
    )
    return max(negative_edge, fold_deviation), negative_edge, fold_deviation


def _fit_spectral_npjive_projection(
    *,
    alpha_coef: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    folds: np.ndarray,
    fold_alphas: dict[int, np.ndarray],
    config: LongTermSurrogateConfig,
    random_state: int,
) -> tuple[np.ndarray, float, float, float, float, float, float, float, float, np.ndarray]:
    full_terms = _primal_npjive_terms(X, Z, X @ alpha_coef)
    fold_terms: list[tuple[np.ndarray, EstimationTerms, EstimationTerms]] = []
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        valid_mask = folds == fold_id
        fold_alpha = fold_alphas[int(fold_id)]
        fold_terms.append(
            (
                fold_alpha,
                _primal_npjive_terms(
                    X[train_mask],
                    Z[train_mask],
                    X[train_mask] @ fold_alpha,
                ),
                _primal_npjive_terms(
                    X[valid_mask],
                    Z[valid_mask],
                    X[valid_mask] @ fold_alpha,
                ),
            )
        )
    _, negative_edge, fold_deviation = _spectral_npjive_stability_edges(
        full_terms,
        [train_terms for _, train_terms, _ in fold_terms],
    )
    selected_threshold = _multiplier_npjive_noise_floor(
        X=X,
        Z=Z,
        terms=full_terms,
        n_replications=config.projection_multiplier_replications,
        quantile=config.projection_multiplier_quantile,
        random_state=random_state + 4_721,
    )
    full_operator = _whitened_npjive_operator(full_terms)[0]
    full_eigenvalues = np.linalg.eigvalsh(full_operator)
    retained_rank = float(np.sum(full_eigenvalues > selected_threshold))
    discarded_eigenvalues = full_eigenvalues[full_eigenvalues <= selected_threshold]
    largest_discarded = (
        float(np.max(discarded_eigenvalues))
        if discarded_eigenvalues.size
        else np.nan
    )
    risks: list[float] = []
    for fold_alpha, train_terms, valid_terms in fold_terms:
        fold_projected, fold_eigen_min, fold_condition = _spectral_npjive_projection(
            alpha_coef=fold_alpha,
            terms=train_terms,
            threshold=selected_threshold,
        )
        if (
            (
                np.isfinite(fold_eigen_min)
                and fold_eigen_min < config.projection_min_system_eigenvalue
            )
            or fold_condition > config.projection_max_system_condition
            or np.linalg.norm(fold_projected) > config.projection_max_fold_coef_norm
        ):
            raise RuntimeError("Noise-floor spectral projection violates a fold stability constraint.")
        risk = float(
            fold_projected.T @ valid_terms.quadratic @ fold_projected
            - 2.0 * valid_terms.linear @ fold_projected
        )
        if not np.isfinite(risk):
            raise RuntimeError("Noise-floor spectral projection has nonfinite validation risk.")
        risks.append(risk)
    cv_risk = float(np.mean(risks))
    projection_operator, eigen_min, condition = _spectral_npjive_projection_operator(
        terms=full_terms,
        threshold=selected_threshold,
    )
    projected = projection_operator @ np.asarray(alpha_coef, dtype=float)
    if (
        (np.isfinite(eigen_min) and eigen_min < config.projection_min_system_eigenvalue)
        or condition > config.projection_max_system_condition
        or np.linalg.norm(projected) > config.projection_max_fold_coef_norm
    ):
        raise RuntimeError("Selected spectral projection violates the stability constraints.")
    return (
        projected,
        selected_threshold,
        cv_risk,
        eigen_min,
        condition,
        negative_edge,
        fold_deviation,
        retained_rank,
        largest_discarded,
        projection_operator,
    )


def _fit_riesz_projection(
    *,
    alpha_coef: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    config: LongTermSurrogateConfig,
    random_state: int,
    target_feature_mean: np.ndarray | None = None,
    alpha_gamma: float | None = None,
) -> RieszProjectionFit:
    """Project ``alpha`` onto the multiplier-detectable npJIVE subspace.

    With pseudo-outcome ``X @ alpha``, the npJIVE normal equations have
    quadratic ``A`` and linear term ``A @ alpha``.  The hard spectral
    projector retains generalized eigendirections of ``A`` above a centered
    Rademacher multiplier null edge, without shrinking retained directions.
    """

    alpha_arr = np.asarray(alpha_coef, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    if alpha_arr.ndim != 1 or alpha_arr.shape[0] != X_arr.shape[1]:
        raise ValueError("alpha_coef must be a length-p coefficient vector.")
    folds = make_stratified_folds(
        Z_arr,
        n_splits=_nuisance_n_splits_for_codes(Z_arr, config.n_splits),
        random_state=random_state,
    )
    fold_alphas: dict[int, np.ndarray] = {}
    for fold_id in np.unique(folds):
        train_mask = folds != fold_id
        fold_alpha = alpha_arr
        if target_feature_mean is not None:
            C_train = X_arr[train_mask].T @ X_arr[train_mask] / np.sum(train_mask)
            gamma_alpha = 0.0 if alpha_gamma is None else float(alpha_gamma)
            fold_alpha = solve_regularized_system(
                C_train + gamma_alpha * np.eye(X_arr.shape[1]),
                np.asarray(target_feature_mean, dtype=float),
            )
        fold_alphas[int(fold_id)] = fold_alpha
    if config.projection_method == "spectral_npjive":
        (
            projected,
            lambda_value,
            cv_risk,
            system_eigen_min,
            system_condition,
            negative_spectral_edge,
            fold_operator_deviation,
            retained_rank,
            largest_discarded_eigenvalue,
            projection_operator,
        ) = (
            _fit_spectral_npjive_projection(
                alpha_coef=alpha_arr,
                X=X_arr,
                Z=Z_arr,
                folds=folds,
                fold_alphas=fold_alphas,
                config=config,
                random_state=random_state,
            )
        )
        gamma = 0.0
        objective = "spectral_leave_one_out_npjive_projection_multiplier_noise_floor"
    elif config.projection_method == "ridge_npjive":
        projection_fold_terms: list[FoldTerms] = []
        for fold_id in np.unique(folds):
            train_mask = folds != fold_id
            valid_mask = folds == fold_id
            fold_alpha = fold_alphas[int(fold_id)]
            projection_fold_terms.append(
                FoldTerms(
                    train=_primal_npjive_terms(
                        X_arr[train_mask],
                        Z_arr[train_mask],
                        X_arr[train_mask] @ fold_alpha,
                    ),
                    valid=_primal_npjive_terms(
                        X_arr[valid_mask],
                        Z_arr[valid_mask],
                        X_arr[valid_mask] @ fold_alpha,
                    ),
                )
            )
        tuning, _ = select_regularization(
            projection_fold_terms,
            method_name="npjive_projection",
            lambda_grid=np.asarray(_projection_lambda_grid(config), dtype=float),
            gamma_grid=np.asarray(_projection_gamma_grid(config), dtype=float),
            max_fold_coef_norm=config.projection_max_fold_coef_norm,
            min_system_eigenvalue=config.projection_min_system_eigenvalue,
            max_system_condition=config.projection_max_system_condition,
            strict_constraints=True,
        )
        terms = _primal_npjive_terms(X_arr, Z_arr, X_arr @ alpha_arr)
        system = (
            terms.quadratic
            + tuning.lambda_value * terms.penalty
            + tuning.gamma * np.eye(X_arr.shape[1])
        )
        projection_operator = solve_regularized_system(
            system,
            terms.quadratic,
        )
        projected = projection_operator @ alpha_arr
        lambda_value = float(tuning.lambda_value)
        gamma = float(tuning.gamma)
        cv_risk = float(tuning.cv_risk)
        objective = "ridge_leave_one_out_npjive_projection_cv"
        eigenvalues = np.linalg.eigvalsh(0.5 * (system + system.T))
        system_eigen_min = float(np.min(eigenvalues))
        system_eigen_max = float(np.max(eigenvalues))
        system_condition = (
            np.inf
            if system_eigen_min <= 0.0
            else float(system_eigen_max / system_eigen_min)
        )
        if (
            system_eigen_min < config.projection_min_system_eigenvalue
            or system_condition > config.projection_max_system_condition
        ):
            raise RuntimeError("Selected projection system violates the stability constraints.")
        negative_spectral_edge = fold_operator_deviation = np.nan
        retained_rank = largest_discarded_eigenvalue = np.nan
    else:
        raise ValueError(f"Unknown projection_method {config.projection_method!r}.")
    residual = alpha_arr - projected
    residual_scores = X_arr @ residual
    return RieszProjectionFit(
        projected_coef=projected,
        residual_coef=residual,
        lambda_value=lambda_value,
        gamma=gamma,
        cv_risk=cv_risk,
        projected_coef_norm=float(np.linalg.norm(projected)),
        residual_coef_norm=float(np.linalg.norm(residual)),
        residual_group_mean_norm=float(np.sqrt(np.mean(group_means(residual_scores, Z_arr) ** 2))),
        system_eigen_min=system_eigen_min,
        system_condition=system_condition,
        negative_spectral_edge=negative_spectral_edge,
        fold_operator_deviation=fold_operator_deviation,
        retained_rank=retained_rank,
        largest_discarded_eigenvalue=largest_discarded_eigenvalue,
        projection_operator=projection_operator,
        objective=objective,
    )
