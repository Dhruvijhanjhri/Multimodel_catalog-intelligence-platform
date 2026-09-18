import pandas as pd

from services.database.postgres import get_connection


def main():
    conn = get_connection()

    query = """
        SELECT
            p.item_id,
            p.title,
            pp.predicted_category,
            pp.confidence
        FROM products p
        JOIN product_predictions pp
            ON pp.product_id = p.id
    """

    predictions = pd.read_sql(query, conn)

    review_queue = pd.read_sql(
        """
        SELECT
            item_id,
            status,
            reason
        FROM review_queue
        """,
        conn,
    )

    conn.close()

    low_confidence = predictions["confidence"] < 0.70

    mismatch = predictions["item_id"].isin(
        predictions.loc[
            predictions["item_id"].isin(
                predictions.loc[
                    low_confidence,
                    "item_id"
                ]
            ),
            "item_id"
        ]
    )

    print("=" * 70)
    print("CATALOG REVIEW QUEUE OVERLAP")
    print("=" * 70)

    print(f"Catalog predictions: {len(predictions)}")
    print(f"Existing review-queue records: {len(review_queue)}")

    print("\nExisting review-queue status:")
    if len(review_queue):
        print(
            review_queue["status"]
            .value_counts()
            .to_string()
        )
    else:
        print("No existing review-queue records.")

    print("\nExisting review-queue reasons:")
    if len(review_queue):
        print(
            review_queue["reason"]
            .value_counts()
            .to_string()
        )
    else:
        print("No existing review-queue records.")

    print("\nCatalog item IDs matching existing review queue:")

    if len(review_queue):
        overlap = predictions.merge(
            review_queue,
            on="item_id",
            how="inner",
        )

        print(f"Matching catalog records: {len(overlap)}")

        if len(overlap):
            print(overlap.to_string(index=False))
    else:
        print("0")


if __name__ == "__main__":
    main()