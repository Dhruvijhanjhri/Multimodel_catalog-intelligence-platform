def evaluate_decision(
    confidence,
    mismatch,
    taxonomy_status,
    duplicate_score,
):
    """
    Central decision policy for catalog intelligence.

    Returns PASS when no review signal is triggered.
    Returns REVIEW when one or more signals require
    human inspection.
    """

    reasons = []

    if float(confidence) < 0.70:
        reasons.append("Low Confidence")

    if bool(mismatch):
        reasons.append("Image-Text Mismatch")

    if taxonomy_status == "REVIEW":
        reasons.append("Taxonomy Uncertainty")

    if float(duplicate_score) > 0.90:
        reasons.append("Possible Duplicate")

    return {
        "decision": "REVIEW" if reasons else "PASS",
        "reasons": reasons,
    }
