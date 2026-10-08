from datetime import date

from services.monitoring.category_drift import evaluate_category_drift
from services.monitoring.confidence_drift import evaluate_confidence_drift


def evaluate_monitoring(baseline_date: date):
    category_result = evaluate_category_drift(baseline_date)
    confidence_result = evaluate_confidence_drift(baseline_date)

    signals = [category_result, confidence_result]
    statuses = [signal["status"] for signal in signals]

    if "DRIFT" in statuses:
        overall_status = "DRIFT"
    elif "WARNING" in statuses:
        overall_status = "WARNING"
    elif all(status == "STABLE" for status in statuses):
        overall_status = "STABLE"
    else:
        overall_status = "INSUFFICIENT_DATA"

    return {
        "baseline_date": str(baseline_date),
        "overall_status": overall_status,
        "category_distribution": category_result,
        "confidence_distribution": confidence_result,
    }
