from services.monitoring.data import get_human_feedback_quality


MIN_VALIDATED_SAMPLES = 30


def evaluate_human_feedback_quality():
    rows = get_human_feedback_quality()

    reviewed_predictions = len(rows)
    validated_predictions = 0
    validated_correct = 0
    corrections = 0
    unresolved_reviews = 0

    for row in rows:
        (
            prediction_id,
            predicted_category,
            model_version,
            decision,
            corrected_category,
            created_at,
        ) = row

        if decision == "Approved":
            validated_predictions += 1
            validated_correct += 1

        elif decision == "Rejected":
            if corrected_category is None:
                unresolved_reviews += 1
                continue

            validated_predictions += 1

            if corrected_category == predicted_category:
                validated_correct += 1
            else:
                corrections += 1

    if validated_predictions == 0:
        validated_accuracy = None
        correction_rate = None
    else:
        validated_accuracy = validated_correct / validated_predictions
        correction_rate = corrections / validated_predictions

    status = (
        "READY"
        if validated_predictions >= MIN_VALIDATED_SAMPLES
        else "INSUFFICIENT_DATA"
    )

    return {
        "status": status,
        "reviewed_predictions": reviewed_predictions,
        "validated_predictions": validated_predictions,
        "validated_correct": validated_correct,
        "validated_accuracy": validated_accuracy,
        "corrections": corrections,
        "correction_rate": correction_rate,
        "unresolved_reviews": unresolved_reviews,
        "minimum_validated_samples": MIN_VALIDATED_SAMPLES,
    }