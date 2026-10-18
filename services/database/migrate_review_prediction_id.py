from services.database.postgres import get_connection


def migrate_review_prediction_id():
    """Add an optional prediction reference to the PostgreSQL review queue."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                ALTER TABLE review_queue
                ADD COLUMN IF NOT EXISTS prediction_id BIGINT
                REFERENCES product_predictions(id)
                ON DELETE SET NULL
                """
            )

        conn.commit()
        print(
            "PostgreSQL review_queue.prediction_id column is ready."
        )

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    migrate_review_prediction_id()