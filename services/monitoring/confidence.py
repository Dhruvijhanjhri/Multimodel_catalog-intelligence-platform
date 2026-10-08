from typing import Sequence


def build_confidence_distribution(confidences: Sequence[float]):
    buckets = {
        "<0.70": 0,
        "0.70-0.79": 0,
        "0.80-0.89": 0,
        "0.90-0.94": 0,
        "0.95-1.00": 0,
    }

    for confidence in confidences:
        value = float(confidence)

        if value < 0.70:
            buckets["<0.70"] += 1
        elif value < 0.80:
            buckets["0.70-0.79"] += 1
        elif value < 0.90:
            buckets["0.80-0.89"] += 1
        elif value < 0.95:
            buckets["0.90-0.94"] += 1
        else:
            buckets["0.95-1.00"] += 1

    return buckets
