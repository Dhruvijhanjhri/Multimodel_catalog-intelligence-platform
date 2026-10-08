from datetime import date

from services.monitoring.data import (
    get_prediction_distribution,
    get_current_prediction_distribution,
)
from services.monitoring.drift import calculate_psi, classify_psi, check_sample_size


TAXONOMY_CATEGORIES = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen",
]


def evaluate_category_drift(baseline_date: date):
    baseline_rows = get_prediction_distribution(baseline_date)
    current_rows = get_current_prediction_distribution(baseline_date)

    baseline_counts = dict(baseline_rows)
    current_counts = dict(current_rows)

    baseline_distribution = [
        baseline_counts.get(category, 0)
        for category in TAXONOMY_CATEGORIES
    ]

    current_distribution = [
        current_counts.get(category, 0)
        for category in TAXONOMY_CATEGORIES
    ]

    current_count = sum(current_distribution)
    sample_status = check_sample_size(current_count)

    result = {
        "signal": "category_distribution",
        "baseline_date": str(baseline_date),
        "baseline_count": sum(baseline_distribution),
        "current_count": current_count,
        "sample_status": sample_status,
        "baseline_distribution": dict(
            zip(TAXONOMY_CATEGORIES, baseline_distribution)
        ),
        "current_distribution": dict(
            zip(TAXONOMY_CATEGORIES, current_distribution)
        ),
    }

    if sample_status == "INSUFFICIENT_DATA":
        result["status"] = "INSUFFICIENT_DATA"
        result["psi"] = None
        return result

    psi = calculate_psi(
        baseline_distribution,
        current_distribution,
    )

    result["psi"] = round(psi, 4)
    result["status"] = classify_psi(psi)

    return result
