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


TermBuilder = Callable[..., EstimationTerms]


def select_regularization(
    fold_terms: list[FoldTerms],
    method_name: str,
    lambda_grid: np.ndarray,
    gamma_grid: np.ndarray,
) -> tuple[RegularizationChoice, np.ndarray]:
    if not fold_terms:
        raise ValueError("fold_terms cannot be empty.")

    dimension = fold_terms[0].train.quadratic.shape[0]
    eye = np.eye(dimension)
    best_choice: RegularizationChoice | None = None
    best_coef: np.ndarray | None = None

    for gamma in gamma_grid:
        for lambda_value in lambda_grid:
            total_risk = 0.0
            last_coef = None
            for fold in fold_terms:
                system = fold.train.quadratic + lambda_value * fold.train.penalty + gamma * eye
                coef = solve_regularized_system(system, fold.train.linear)
                risk = float(coef.T @ fold.valid.quadratic @ coef - 2.0 * fold.valid.linear @ coef)
                total_risk += risk
                last_coef = coef

            mean_risk = total_risk / len(fold_terms)
            if best_choice is None or mean_risk < best_choice.cv_risk:
                best_choice = RegularizationChoice(
                    method=method_name,
                    lambda_value=float(lambda_value),
                    gamma=float(gamma),
                    cv_risk=float(mean_risk),
                )
                best_coef = last_coef

    if best_choice is None or best_coef is None:
        raise RuntimeError("Regularization search failed to select a solution.")

    return best_choice, best_coef

