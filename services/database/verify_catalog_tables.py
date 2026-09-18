from services.database.postgres import get_connection


TABLES = [
    "products",
    "product_images",
    "product_predictions",
    "product_reviews",
    "ingestion_events",
    "model_versions",
]


conn = get_connection()

with conn.cursor() as cursor:
    cursor.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = ANY(%s)
        ORDER BY table_name
        """,
        (TABLES,),
    )

    rows = cursor.fetchall()

conn.close()

print("Catalog tables found:")
for row in rows:
    print("-", row[0])

print(f"Total: {len(rows)}/6")