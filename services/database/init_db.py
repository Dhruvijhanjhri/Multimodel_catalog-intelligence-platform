import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "services" / "review_queue.db"

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS review_queue(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    item_id TEXT,

    image_name TEXT,

    title TEXT,

    category TEXT,

    confidence REAL,

    mismatch_score REAL,

    duplicate_score REAL,

    reason TEXT,

    status TEXT DEFAULT 'Pending',

    created_at TEXT

)
""")

conn.commit()
conn.close()

print("Review Queue Database Created")
print(DB_PATH)