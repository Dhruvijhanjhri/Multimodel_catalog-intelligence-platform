import pandas as pd
from psycopg import sql
from psycopg.rows import tuple_row

from services.database.postgres import get_connection


SOURCE_FILE = "data/splits/train.parquet"


def ingest_products():
    df = pd.read_parquet(SOURCE_FILE)

    records = [
        (
            row.item_id,
            row.title,
            row.brand,
            row.product_type,
            row.color,
            row.category,
            "train.parquet",
        )
        for row in df.itertuples(index=False)
    ]

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO products
                (
                    item_id,
                    title,
                    brand,
                    product_type,
                    color,
                    category,
                    source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print(f"Ingested {len(records)} products into PostgreSQL.")


if __name__ == "__main__":
    ingest_products()