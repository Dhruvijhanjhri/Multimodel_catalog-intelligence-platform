from datetime import datetime, timezone

from services.database.postgres import get_connection


def add_product_review(
    product_id,
    prediction_id,
    reviewer,
    decision,
    corrected_category=None,
    feedback=None,
):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO product_reviews
                (
                    product_id,
                    prediction_id,
                    reviewer,
                    decision,
                    corrected_category,
                    feedback,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    product_id,
                    prediction_id,
                    reviewer,
                    decision,
                    corrected_category,
                    feedback,
                    datetime.now(timezone.utc),
                ),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()