import pandas as pd

from services.database.postgres import get_connection


LOW_CONFIDENCE_THRESHOLD = 0.70


def main():
    conn = get_connection()

    query = """
        SELECT
            p.item_id,
            p.title,
            p.category AS actual_category,
            pp.predicted_category,
            pp.confidence
        FROM products p
        JOIN product_predictions pp
            ON pp.product_id = p.id
        WHERE pp.confidence < %s
        ORDER BY pp.confidence ASC
    """

    df = pd.read_sql(
        query,
        conn,
        params=(LOW_CONFIDENCE_THRESHOLD,),
    )

    conn.close()

    print("=" * 70)
    print("LOW-CONFIDENCE CANDIDATE ANALYSIS")
    print("=" * 70)

    print(f"Threshold: < {LOW_CONFIDENCE_THRESHOLD}")
    print(f"Total low-confidence records: {len(df)}")

    if df.empty:
        print("No low-confidence records found.")
        return

    print("\nConfidence statistics:")
    print(df["confidence"].describe())

    correct = (
        df["actual_category"] == df["predicted_category"]
    ).sum()

    accuracy = correct / len(df) * 100

    print("\nPrediction accuracy:")
    print(f"Correct: {correct}")
    print(f"Incorrect: {len(df) - correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nAll low-confidence records:")
    print(
        df[
            [
                "item_id",
                "title",
                "actual_category",
                "predicted_category",
                "confidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()