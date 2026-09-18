from services.database.postgres import get_connection


conn = get_connection()

with conn.cursor() as cursor:
    cursor.execute(
        """
        SELECT
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_name = 'product_reviews'
        ORDER BY ordinal_position
        """
    )

    rows = cursor.fetchall()

conn.close()

print("product_reviews columns:")

for row in rows:
    print(row[0], "-", row[1], "-", row[2])