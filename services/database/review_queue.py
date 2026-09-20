from datetime import datetime, timezone
import os
from services.database.postgres import get_connection


def add_to_review_queue(
    item_id,
    image_name,
    title,
    predicted_category,
    confidence,
    image_similarity,
    duplicate_score,
    reason,
):
    conn = get_connection()

    with conn.cursor() as cursor:
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                item_id,
                image_name,
                title,
                predicted_category,
                confidence,
                image_similarity,
                duplicate_score,
                reason,
                "Pending",
                datetime.now(timezone.utc),
            ),
        )

    conn.commit()
    conn.close()


def get_review_queue():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                rq.id,
                rq.item_id,
                COALESCE(
                    rq.image_name,
                    (
                        SELECT pi.image_path
                        FROM product_images pi
                        JOIN products p2
                            ON p2.id = pi.product_id
                        WHERE p2.item_id = rq.item_id
                          AND pi.is_primary = TRUE
                        LIMIT 1
                    )
                ) AS image_name,
                rq.title,
                rq.category,
                rq.confidence,
                rq.mismatch_score,
                rq.duplicate_score,
                rq.reason,
                rq.status,
                rq.created_at
            FROM review_queue rq
            ORDER BY rq.created_at DESC
            """
        )

        rows = cursor.fetchall()

    conn.close()

    columns = [
        "id",
        "item_id",
        "image_name",
        "title",
        "category",
        "confidence",
        "mismatch_score",
        "duplicate_score",
        "reason",
        "status",
        "created_at",
    ]

    return [
        {
            **dict(zip(columns, row)),
            "image_name": os.path.basename(row[2]) if row[2] else None,
        }
        for row in rows
    ]

def update_review_status(record_id, status):
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE review_queue
            SET status = %s
            WHERE id = %s
            """,
            (status, record_id),
        )

    conn.commit()
    conn.close()