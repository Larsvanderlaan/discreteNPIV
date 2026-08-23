from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._linear_solvers import solve_regularized_system
from .results import RegularizationChoice


@dataclass(frozen=True)
class EstimationTerms:
    quadratic: np.ndarray
    linear: np.ndarray
    penalty: np.ndarray


@dataclass(frozen=True)
class FoldTerms:
    train: EstimationTerms
    valid: EstimationTerms


@dataclass(frozen=True)
class OperatorStabilityConstraint:
    """Observed-operator floor for a statistically stable inverse fit."""

    fold_noise_floors: tuple[float, ...]
    full_terms: EstimationTerms
    full_noise_floor: float
    margin: float = 1.25


TermBuilder = Callable[..., EstimationTerms]


def _generalized_system_diagnostics(
    terms: EstimationTerms,
    *,
    lambda_value: float,
    gamma: float,
) -> tuple[float, float]:
    """Return min eigenvalue and condition in the empirical penalty geometry."""

    penalty = 0.5 * (terms.penalty + terms.penalty.T)
    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-10)
    inverse_sqrt = (
        eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
    ) @ eigenvectors.T
    system = (
        terms.quadratic
        + float(lambda_value) * terms.penalty
        + float(gamma) * np.eye(terms.quadratic.shape[0])
    )
    generalized = inverse_sqrt @ (0.5 * (system + system.T)) @ inverse_sqrt
    system_eigenvalues = np.linalg.eigvalsh(0.5 * (generalized + generalized.T))
    smallest = float(np.min(system_eigenvalues))
    largest = float(np.max(system_eigenvalues))
    condition = np.inf if smallest <= 0.0 else float(largest / smallest)
    return smallest, condition


def select_regularization(
    fold_terms: list[FoldTerms],
    method_name: str,
    lambda_grid: np.ndarray,
    gamma_grid: np.ndarray,
    max_fold_coef_norm: float | None = None,
    min_system_eigenvalue: float | None = None,
    max_system_condition: float | None = None,
    strict_constraints: bool = False,
    operator_stability: OperatorStabilityConstraint | None = None,
    full_terms: EstimationTerms | None = None,
) -> tuple[RegularizationChoice, np.ndarray]:
    if not fold_terms:
        raise ValueError("fold_terms cannot be empty.")

    dimension = fold_terms[0].train.quadratic.shape[0]
    eye = np.eye(dimension)
    best_choice: RegularizationChoice | None = None
    best_coef: np.ndarray | None = None
    fallback_choice: RegularizationChoice | None = None
    fallback_coef: np.ndarray | None = None
    if operator_stability is not None:
        if len(operator_stability.fold_noise_floors) != len(fold_terms):
            raise ValueError("operator stability must supply one noise floor per fold.")
        if operator_stability.full_noise_floor < 0.0:
            raise ValueError("operator stability noise floors must be nonnegative.")
        if any(value < 0.0 for value in operator_stability.fold_noise_floors):
            raise ValueError("operator stability noise floors must be nonnegative.")
        if operator_stability.margin <= 1.0:
            raise ValueError("operator stability margin must exceed one.")

    for gamma in gamma_grid:
        for lambda_value in lambda_grid:
            total_risk = 0.0
            last_coef = None
            fold_coef_norms: list[float] = []
            candidate_is_stable = True
            full_generalized_eigen_min = np.nan
            full_generalized_condition = np.nan
            if operator_stability is not None:
                (
                    full_generalized_eigen_min,
                    full_generalized_condition,
                ) = _generalized_system_diagnostics(
                    operator_stability.full_terms,
                    lambda_value=float(lambda_value),
                    gamma=float(gamma),
                )
                if full_generalized_eigen_min < (
                    operator_stability.margin * operator_stability.full_noise_floor
                ):
                    continue
            elif full_terms is not None and (
                min_system_eigenvalue is not None or max_system_condition is not None
            ):
                (
                    full_generalized_eigen_min,
                    full_generalized_condition,
                ) = _generalized_system_diagnostics(
                    full_terms,
                    lambda_value=float(lambda_value),
                    gamma=float(gamma),
                )
                if (
                    min_system_eigenvalue is not None
                    and full_generalized_eigen_min < min_system_eigenvalue
                ):
                    continue
                if (
                    max_system_condition is not None
                    and full_generalized_condition > max_system_condition
                ):
                    continue
            for fold_index, fold in enumerate(fold_terms):
                system = fold.train.quadratic + lambda_value * fold.train.penalty + gamma * eye
                if operator_stability is not None:
                    fold_generalized_eigen_min, _ = _generalized_system_diagnostics(
                        fold.train,
                        lambda_value=float(lambda_value),
                        gamma=float(gamma),
                    )
                    if fold_generalized_eigen_min < (
                        operator_stability.margin
                        * operator_stability.fold_noise_floors[fold_index]
                    ):
                        candidate_is_stable = False
                        break
                if min_system_eigenvalue is not None or max_system_condition is not None:
                    smallest, condition = _generalized_system_diagnostics(
                        fold.train,
                        lambda_value=float(lambda_value),
                        gamma=float(gamma),
                    )
                    if min_system_eigenvalue is not None and smallest < min_system_eigenvalue:
                        candidate_is_stable = False
                        break
                    if max_system_condition is not None and condition > max_system_condition:
                        candidate_is_stable = False
                        break
                coef = solve_regularized_system(system, fold.train.linear)
                risk = float(coef.T @ fold.valid.quadratic @ coef - 2.0 * fold.valid.linear @ coef)
                if not np.isfinite(risk) or not np.all(np.isfinite(coef)):
                    candidate_is_stable = False
                    break
                total_risk += risk
                last_coef = coef
                fold_coef_norms.append(float(np.linalg.norm(coef)))

            if not candidate_is_stable or last_coef is None:
                continue

            mean_risk = total_risk / len(fold_terms)
            choice = RegularizationChoice(
                method=method_name,
                lambda_value=float(lambda_value),
                gamma=float(gamma),
                cv_risk=float(mean_risk),
                operator_noise_floor=(
                    np.nan
                    if operator_stability is None
                    else float(operator_stability.full_noise_floor)
                ),
                operator_noise_margin=(
                    np.nan
                    if operator_stability is None
                    else float(operator_stability.margin)
                ),
                generalized_system_eigen_min=float(full_generalized_eigen_min),
                generalized_system_condition=float(full_generalized_condition),
            )
            if fallback_choice is None or mean_risk < fallback_choice.cv_risk:
                fallback_choice = choice
                fallback_coef = last_coef
            if max_fold_coef_norm is not None and max(fold_coef_norms) > max_fold_coef_norm:
                continue
            if best_choice is None or mean_risk < best_choice.cv_risk:
                best_choice = choice
                best_coef = last_coef

    if best_choice is None and fallback_choice is not None and not strict_constraints:
        best_choice = RegularizationChoice(
            method=method_name,
            lambda_value=fallback_choice.lambda_value,
            gamma=fallback_choice.gamma,
            cv_risk=fallback_choice.cv_risk,
            operator_noise_floor=fallback_choice.operator_noise_floor,
            operator_noise_margin=fallback_choice.operator_noise_margin,
            generalized_system_eigen_min=fallback_choice.generalized_system_eigen_min,
            generalized_system_condition=fallback_choice.generalized_system_condition,
        )
        best_coef = fallback_coef

    if best_choice is None or best_coef is None:
        raise RuntimeError("Regularization search found no candidate satisfying the stability constraints.")

    return best_choice, best_coef
