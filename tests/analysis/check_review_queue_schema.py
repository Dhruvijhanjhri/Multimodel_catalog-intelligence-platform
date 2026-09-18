from services.database.postgres import get_connection


conn = get_connection()

with conn.cursor() as cursor:
    cursor.execute(
        """
        SELECT
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_name = 'review_queue'
        ORDER BY ordinal_position
        """
    )

    rows = cursor.fetchall()

conn.close()

print("review_queue columns:")

for row in rows:
    print(row[0], "-", row[1])