import numpy as np
import pandas as pd

from services.database.postgres import get_connection


LOW_CONFIDENCE_THRESHOLD = 0.70
MISMATCH_THRESHOLD = 0.175
DUPLICATE_THRESHOLD = 0.90


def main():
    conn = get_connection()

    predictions = pd.read_sql(
        """
        SELECT
            p.item_id,
            p.title,
            p.category AS actual_category,
            pp.predicted_category,
            pp.confidence
        FROM products p
        JOIN product_predictions pp
            ON pp.product_id = p.id
        """,
        conn,
    )

    conn.close()

    # Load saved image/text embeddings.
    image_embeddings = np.load(
        "embeddings/train_image_embeddings.npy"
    )

    text_embeddings = np.load(
        "embeddings/train_text_embeddings.npy"
    )

    metadata = pd.read_parquet(
        "embeddings/embedding_metadata.parquet"
    )

    # Make sure embedding order matches metadata.
    assert len(image_embeddings) == len(metadata)
    assert len(text_embeddings) == len(metadata)

    # Cosine similarity because embeddings are normalized.
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

    # Strong duplicate candidates:
    # same normalized title + same image + identical attributes.
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

    duplicate_items = set(duplicate_df["item_id"])

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

    review_df = df[df["review_required"]].copy()

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

    print("=" * 80)
    print("CATALOG REVIEW REASON BREAKDOWN")
    print("=" * 80)

    print(f"Total catalog records: {len(df)}")
    print(f"Total review candidates: {len(review_df)}")

    print("\nIndividual signals:")
    print(
        f"Low Confidence:       "
        f"{df['low_confidence'].sum()}"
    )
    print(
        f"Image-Text Mismatch:   "
        f"{df['image_text_mismatch'].sum()}"
    )
    print(
        f"Strong Duplicate:      "
        f"{df['strong_duplicate'].sum()}"
    )

    print("\nReason combinations:")

    combination_counts = (
        review_df["review_reason"]
        .value_counts()
    )

    for reason, count in combination_counts.items():
        print(f"{reason}: {count}")

    print("\nReview workload:")
    print(
        f"Percentage requiring review: "
        f"{len(review_df) / len(df) * 100:.2f}%"
    )

    print("\nCandidate priority distribution:")

    priority = []

    for _, row in review_df.iterrows():

        signal_count = sum(
            [
                row["low_confidence"],
                row["image_text_mismatch"],
                row["strong_duplicate"],
            ]
        )

        if signal_count >= 3:
            priority.append("High")
        elif signal_count == 2:
            priority.append("Medium")
        else:
            priority.append("Standard")

    review_df["priority"] = priority

    print(
        review_df["priority"]
        .value_counts()
        .to_string()
    )

    print("\nFirst 20 review candidates:")

    print(
        review_df[
            [
                "item_id",
                "title",
                "predicted_category",
                "confidence",
                "image_text_similarity",
                "review_reason",
                "priority",
            ]
        ]
        .sort_values(
            ["priority", "confidence"],
            ascending=[True, True],
        )
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()