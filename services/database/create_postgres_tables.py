from services.database.postgres import get_connection


def create_review_queue_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                id BIGSERIAL PRIMARY KEY,
                item_id TEXT,
                image_name TEXT,
                title TEXT,
                category TEXT,
                confidence DOUBLE PRECISION,
                mismatch_score DOUBLE PRECISION,
                duplicate_score DOUBLE PRECISION,
                reason TEXT,
                status TEXT DEFAULT 'Pending',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL review_queue table created successfully.")


if __name__ == "__main__":
    create_review_queue_table()