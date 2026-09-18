import pandas as pd

from services.database.postgres import get_connection


SOURCE_FILE = "data/processed/image_manifest.parquet"


def ingest_product_images():
    df = pd.read_parquet(SOURCE_FILE)

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, item_id FROM products"
            )

            product_map = {
                item_id: product_id
                for product_id, item_id in cursor.fetchall()
            }

            records = []

            for row in df.itertuples(index=False):
                product_id = product_map.get(row.item_id)

                if product_id is None:
                    continue

                records.append(
                    (
                        product_id,
                        row.image_id,
                        row.image_path,
                        None,
                        row.image_id == row.main_image_id,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO product_images
                (
                    product_id,
                    image_name,
                    image_path,
                    embedding_path,
                    is_primary
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                records,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print(f"Ingested {len(records)} product images into PostgreSQL.")


if __name__ == "__main__":
    ingest_product_images()