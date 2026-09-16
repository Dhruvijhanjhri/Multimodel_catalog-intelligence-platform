import sqlite3

from services.database.postgres import get_connection


SQLITE_DB = "services/review_queue.db"


def migrate_review_queue():
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row

    rows = sqlite_conn.execute(
        """
        SELECT
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
        ORDER BY id
        """
    ).fetchall()

    sqlite_conn.close()

    postgres_conn = get_connection()

    with postgres_conn.cursor() as cursor:
        for row in rows:
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
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    row["item_id"],
                    row["image_name"],
                    row["title"],
                    row["category"],
                    row["confidence"],
                    row["mismatch_score"],
                    row["duplicate_score"],
                    row["reason"],
                    row["status"],
                    row["created_at"],
                ),
            )

    postgres_conn.commit()
    postgres_conn.close()

    print(f"Migrated {len(rows)} review records to PostgreSQL.")


if __name__ == "__main__":
    migrate_review_queue()