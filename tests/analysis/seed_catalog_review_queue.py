import numpy as np
import pandas as pd

from services.database.postgres import get_connection


LOW_CONFIDENCE_THRESHOLD = 0.70
MISMATCH_THRESHOLD = 0.175


def main():
    print("=" * 80)
    print("CATALOG REVIEW QUEUE SEED - DRY RUN")
    print("=" * 80)

    # --------------------------------------------------
    # Load catalog predictions from PostgreSQL
    # --------------------------------------------------

    conn = get_connection()

    predictions = pd.read_sql(
        """
        SELECT
            p.id AS product_id,
            p.item_id,
            p.title,
            pp.id AS prediction_id,
            pp.predicted_category,
            pp.confidence
        FROM products p
        JOIN product_predictions pp
            ON pp.product_id = p.id
        """,
        conn,
    )

    conn.close()

    print(f"Catalog predictions loaded: {len(predictions)}")

    # --------------------------------------------------
    # Load saved embeddings and metadata
    # --------------------------------------------------

    image_embeddings = np.load(
        "embeddings/train_image_embeddings.npy"
    )

    text_embeddings = np.load(
        "embeddings/train_text_embeddings.npy"
    )

    metadata = pd.read_parquet(
        "embeddings/embedding_metadata.parquet"
    )

    assert len(image_embeddings) == len(metadata)
    assert len(text_embeddings) == len(metadata)

    # --------------------------------------------------
    # Calculate image-text similarity
    # --------------------------------------------------

    similarities = np.sum(
        image_embeddings * text_embeddings,
        axis=1,
    )

    similarity_df = pd.DataFrame(
        {
            "item_id": metadata["item_id"],
            "image_text_similarity": similarities,
        }
    )

    df = predictions.merge(
        similarity_df,
        on="item_id",
        how="inner",
    )

    print(f"Records with embedding match: {len(df)}")

    # --------------------------------------------------
    # Strong duplicate detection
    # Same normalized title + same image
    # + identical attributes
    # --------------------------------------------------

    conn = get_connection()

    duplicate_df = pd.read_sql(
        """
        WITH catalog AS (
            SELECT
                p.item_id,
                LOWER(TRIM(p.title)) AS normalized_title,
                LOWER(TRIM(pi.image_path)) AS normalized_image,
                p.brand,
                p.product_type,
                p.color,
                p.category
            FROM products p
            JOIN product_images pi
                ON pi.product_id = p.id
        ),
        valid_groups AS (
            SELECT
                normalized_title,
                normalized_image
            FROM catalog
            GROUP BY
                normalized_title,
                normalized_image
            HAVING COUNT(*) > 1
                AND COUNT(
                    DISTINCT jsonb_build_array(
                        brand,
                        product_type,
                        color,
                        category
                    )
                ) = 1
        )
        SELECT DISTINCT
            c.item_id
        FROM catalog c
        JOIN valid_groups vg
            ON c.normalized_title = vg.normalized_title
            AND c.normalized_image = vg.normalized_image
        """,
        conn,
    )

    conn.close()

    duplicate_items = set(
        duplicate_df["item_id"]
    )

    # --------------------------------------------------
    # Review signals
    # --------------------------------------------------

    df["low_confidence"] = (
        df["confidence"] < LOW_CONFIDENCE_THRESHOLD
    )

    df["image_text_mismatch"] = (
        df["image_text_similarity"] < MISMATCH_THRESHOLD
    )

    df["strong_duplicate"] = (
        df["item_id"].isin(duplicate_items)
    )

    df["review_required"] = (
        df["low_confidence"]
        | df["image_text_mismatch"]
        | df["strong_duplicate"]
    )

    review_df = df[
        df["review_required"]
    ].copy()

    # --------------------------------------------------
    # Review reasons
    # --------------------------------------------------

    def get_reasons(row):
        reasons = []

        if row["low_confidence"]:
            reasons.append("Low Confidence")

        if row["image_text_mismatch"]:
            reasons.append("Image-Text Mismatch")

        if row["strong_duplicate"]:
            reasons.append("Possible Duplicate")

        return " + ".join(reasons)

    review_df["review_reason"] = review_df.apply(
        get_reasons,
        axis=1,
    )

    # --------------------------------------------------
    # Priority
    # --------------------------------------------------

    def get_priority(row):
        signal_count = sum(
            [
                row["low_confidence"],
                row["image_text_mismatch"],
                row["strong_duplicate"],
            ]
        )

        if signal_count >= 3:
            return "High"

        if signal_count == 2:
            return "Medium"

        return "Standard"

    review_df["priority"] = review_df.apply(
        get_priority,
        axis=1,
    )

    # --------------------------------------------------
    # Check existing PostgreSQL review queue
    # --------------------------------------------------

    conn = get_connection()

    existing_queue = pd.read_sql(
        """
        SELECT
            item_id,
            status
        FROM review_queue
        """,
        conn,
    )

    conn.close()

    existing_item_ids = set(
        existing_queue["item_id"]
    )

    new_candidates = review_df[
        ~review_df["item_id"].isin(
            existing_item_ids
        )
    ].copy()

    # --------------------------------------------------
    # Seed PostgreSQL review queue
    # --------------------------------------------------

    conn = get_connection()

    inserted = 0
    skipped = 0

    try:
        with conn.cursor() as cursor:

            for _, row in new_candidates.iterrows():

                cursor.execute(
                    """
                    INSERT INTO review_queue
                    (
                        item_id,
                        image_name,
                        title,
                        category,
                        confidence,
                        mismatch_score,
                        duplicate_score,
                        reason,
                        status,
                        created_at
                    )
                    VALUES
                    (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, CURRENT_TIMESTAMP
                    )
                    """,
                    (
                        row["item_id"],
                        None,
                        row["title"],
                        row["predicted_category"],
                        float(row["confidence"]),
                        float(row["image_text_similarity"]),
                        1.0 if row["strong_duplicate"] else 0.0,
                        row["review_reason"],
                        "Pending",
                    ),
                )

                inserted += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    skipped = len(review_df) - len(new_candidates)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("CATALOG REVIEW QUEUE SEED COMPLETE")
    print("=" * 80)

    print(f"Review candidates identified: {len(review_df)}")
    print(f"New candidates: {len(new_candidates)}")
    print(f"Records inserted: {inserted}")
    print(f"Records skipped: {skipped}")
    print()
    print("No existing review_queue records were modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()