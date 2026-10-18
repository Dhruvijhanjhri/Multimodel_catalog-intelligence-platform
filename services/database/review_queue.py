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
    prediction_id=None,
    conn=None,
):
    """
    Add an item to the review queue.

    If a connection is supplied, the caller owns its transaction.
    Otherwise, this function manages its own connection and transaction.
    """
    owns_connection = conn is None

    if owns_connection:
        conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM review_queue
                WHERE item_id = %s
                  AND status = 'Pending'
                LIMIT 1
                """,
                (item_id,),
            )

            existing = cursor.fetchone()

            if existing:
                # Keep the existing pending row, but link it to the
                # latest prediction that needs review.
                if prediction_id is not None:
                    cursor.execute(
                        """
                        UPDATE review_queue
                        SET prediction_id = %s,
                            image_name = %s,
                            title = %s,
                            category = %s,
                            confidence = %s,
                            mismatch_score = %s,
                            duplicate_score = %s,
                            reason = %s
                        WHERE id = %s
                        """,
                        (
                            prediction_id,
                            image_name,
                            title,
                            predicted_category,
                            confidence,
                            image_similarity,
                            duplicate_score,
                            reason,
                            existing[0],
                        ),
                    )

                if owns_connection:
                    conn.commit()

                return

            cursor.execute(
                """
                INSERT INTO review_queue (
                    item_id,
                    image_name,
                    title,
                    category,
                    confidence,
                    mismatch_score,
                    duplicate_score,
                    reason,
                    status,
                    created_at,
                    prediction_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    prediction_id,
                ),
            )

        if owns_connection:
            conn.commit()

    except Exception:
        if owns_connection:
            conn.rollback()
        raise

    finally:
        if owns_connection:
            conn.close()


def get_review_queue():
    conn = get_connection()

    try:
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
    finally:
        conn.close()


def update_review_status(record_id, status):
    conn = get_connection()

    try:
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
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
