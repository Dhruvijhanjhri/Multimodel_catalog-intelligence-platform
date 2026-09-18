from services.database.postgres import get_connection


conn = get_connection()

with conn.cursor() as cursor:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM products
        INNER JOIN product_images
            ON products.id = product_images.product_id
        """
    )

    count = cursor.fetchone()[0]

conn.close()

print("Products with images in PostgreSQL:", count)