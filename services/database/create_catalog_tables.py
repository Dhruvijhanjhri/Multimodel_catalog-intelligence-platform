from services.database.postgres import get_connection


def create_products_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id BIGSERIAL PRIMARY KEY,
                item_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                brand TEXT,
                product_type TEXT,
                color TEXT,
                category TEXT,
                source TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL products table created successfully.")

def create_product_images_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_images (
                id BIGSERIAL PRIMARY KEY,
                product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                image_name TEXT NOT NULL,
                image_path TEXT,
                embedding_path TEXT,
                is_primary BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL product_images table created successfully.")

def create_product_predictions_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_predictions (
                id BIGSERIAL PRIMARY KEY,
                product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                predicted_category TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                model_version TEXT,
                prediction_source TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL product_predictions table created successfully.")

def create_product_reviews_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_reviews (
                id BIGSERIAL PRIMARY KEY,
                product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                prediction_id BIGINT REFERENCES product_predictions(id) ON DELETE SET NULL,
                reviewer TEXT,
                decision TEXT NOT NULL,
                corrected_category TEXT,
                feedback TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL product_reviews table created successfully.")

def create_ingestion_events_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_events (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL,
                item_id TEXT,
                source TEXT,
                event_type TEXT NOT NULL,
                payload JSONB,
                status TEXT DEFAULT 'Received',
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMPTZ
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL ingestion_events table created successfully.")

def create_model_versions_table():
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id BIGSERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT,
                artifact_path TEXT,
                metrics JSONB,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.commit()
    conn.close()

    print("PostgreSQL model_versions table created successfully.")

if __name__ == "__main__":
    create_products_table()
    create_product_images_table()
    create_product_predictions_table()
    create_product_reviews_table()
    create_ingestion_events_table()
    create_model_versions_table()