from datetime import datetime, timezone

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
                id,
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
            FROM review_queue
            ORDER BY created_at DESC
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

    return [dict(zip(columns, row)) for row in rows]


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