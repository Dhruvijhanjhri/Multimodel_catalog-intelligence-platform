from services.database.postgres import get_connection


conn = get_connection()

with conn.cursor() as cursor:
    cursor.execute(
        """
        SELECT
            tc.constraint_name,
            tc.constraint_type
        FROM information_schema.table_constraints tc
        WHERE tc.table_name = 'review_queue'
        ORDER BY tc.constraint_name
        """
    )

    rows = cursor.fetchall()

conn.close()

print("review_queue constraints:")

for row in rows:
    print(row[0], "-", row[1])