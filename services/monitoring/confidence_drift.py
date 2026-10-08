from datetime import date

from services.monitoring.confidence import build_confidence_distribution
from services.monitoring.data import (
    get_current_prediction_confidences,
    get_prediction_confidences,
)
from services.monitoring.drift import (
    calculate_psi,
    check_sample_size,
    classify_psi,
)


CONFIDENCE_BUCKETS = [
    "<0.70",
    "0.70-0.79",
    "0.80-0.89",
    "0.90-0.94",
    "0.95-1.00",
]


def evaluate_confidence_drift(baseline_date: date):
    baseline_confidences = get_prediction_confidences(baseline_date)
    current_confidences = get_current_prediction_confidences(baseline_date)

    baseline_distribution = build_confidence_distribution(
        baseline_confidences
    )
    current_distribution = build_confidence_distribution(
        current_confidences
    )

    current_count = len(current_confidences)
    sample_status = check_sample_size(current_count)

    result = {
        "signal": "confidence_distribution",
        "baseline_date": str(baseline_date),
        "baseline_count": len(baseline_confidences),
        "current_count": current_count,
        "sample_status": sample_status,
        "baseline_distribution": baseline_distribution,
        "current_distribution": current_distribution,
    }

    if sample_status == "INSUFFICIENT_DATA":
        result["status"] = "INSUFFICIENT_DATA"
        result["psi"] = None
        return result

    baseline_values = [
        baseline_distribution[bucket]
        for bucket in CONFIDENCE_BUCKETS
    ]

    current_values = [
        current_distribution[bucket]
        for bucket in CONFIDENCE_BUCKETS
    ]

    psi = calculate_psi(
        baseline_values,
        current_values,
    )

    result["psi"] = round(psi, 4)
    result["status"] = classify_psi(psi)

    return result
