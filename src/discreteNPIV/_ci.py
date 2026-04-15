from __future__ import annotations


def normal_ci(estimate: float, se: float) -> tuple[float, float]:
    radius = 1.96 * se
    return estimate - radius, estimate + radius
