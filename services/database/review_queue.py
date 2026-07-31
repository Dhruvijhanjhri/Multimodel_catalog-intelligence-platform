import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "services" / "review_queue.db"

print("\nReview Queue DB Path:")
print(DB_PATH.resolve())


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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()

def get_review_queue():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM review_queue
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


def update_review_status(item_id, status):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE review_queue
        SET status = ?
        WHERE id = ?
        """,
        (status, item_id)
    )

    conn.commit()
    conn.close()