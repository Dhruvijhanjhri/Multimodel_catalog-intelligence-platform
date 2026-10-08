from typing import Sequence


def calculate_psi(
    baseline: Sequence[float],
    current: Sequence[float],
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index (PSI) for two distributions.

    Both inputs must contain non-negative proportions or counts
    representing the same ordered categories/bins.
    """

    if len(baseline) != len(current):
        raise ValueError("Baseline and current distributions must have the same length.")

    baseline_total = sum(baseline)
    current_total = sum(current)

    if baseline_total <= 0 or current_total <= 0:
        raise ValueError("Both distributions must contain at least one observation.")

    baseline_pct = [max(value / baseline_total, epsilon) for value in baseline]
    current_pct = [max(value / current_total, epsilon) for value in current]

    return sum(
        (current_pct[i] - baseline_pct[i])
        * __import__("math").log(current_pct[i] / baseline_pct[i])
        for i in range(len(baseline_pct))
    )


def classify_psi(psi: float) -> str:
    """
    Classify PSI using operational monitoring bands.

    < 0.10  -> STABLE
    < 0.25  -> WARNING
    >= 0.25 -> DRIFT
    """

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "WARNING"

    return "DRIFT"

def check_sample_size(current_count: int, minimum_samples: int = 100) -> str:
    """
    Determine whether the current monitoring window has enough observations.

    Returns INSUFFICIENT_DATA when the current window is too small.
    """

    if current_count < minimum_samples:
        return "INSUFFICIENT_DATA"

    return "READY"

