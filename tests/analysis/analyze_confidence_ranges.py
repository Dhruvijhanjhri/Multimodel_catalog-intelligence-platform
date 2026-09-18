import pandas as pd

from services.database.postgres import get_connection


def main():
    conn = get_connection()

    query = """
        SELECT
            p.category AS actual_category,
            pp.predicted_category,
            pp.confidence
        FROM products p
        JOIN product_predictions pp
            ON pp.product_id = p.id
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["correct"] = (
        df["actual_category"] == df["predicted_category"]
    )

    ranges = [
        ("< 0.50", 0.00, 0.50),
        ("0.50 - 0.55", 0.50, 0.55),
        ("0.55 - 0.60", 0.55, 0.60),
        ("0.60 - 0.65", 0.60, 0.65),
        ("0.65 - 0.70", 0.65, 0.70),
        ("0.70 - 0.80", 0.70, 0.80),
        ("0.80 - 0.90", 0.80, 0.90),
        ("0.90 - 1.00", 0.90, 1.01),
    ]

    print("=" * 80)
    print("CONFIDENCE RANGE ANALYSIS")
    print("=" * 80)

    for label, lower, upper in ranges:
        subset = df[
            (df["confidence"] >= lower)
            & (df["confidence"] < upper)
        ]

        total = len(subset)

        if total == 0:
            continue

        correct = subset["correct"].sum()
        incorrect = total - correct
        accuracy = correct / total * 100

        print(
            f"{label:12} | "
            f"Records: {total:5} | "
            f"Correct: {correct:5} | "
            f"Incorrect: {incorrect:5} | "
            f"Accuracy: {accuracy:6.2f}%"
        )

    print("\nThreshold analysis:")

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        subset = df[df["confidence"] < threshold]

        total = len(subset)

        if total == 0:
            continue

        incorrect = (~subset["correct"]).sum()
        accuracy = subset["correct"].mean() * 100

        print(
            f"< {threshold:.2f} | "
            f"Review records: {total:5} | "
            f"Incorrect: {incorrect:5} | "
            f"Accuracy: {accuracy:6.2f}%"
        )


if __name__ == "__main__":
    main()