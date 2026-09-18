from services.database.postgres import get_connection


MODEL_NAME = "multimodal_classifier"
MODEL_VERSION = "multimodal_classifier_v1"
MODEL_TYPE = "PyTorch multimodal classifier"
ARTIFACT_PATH = "models/multimodal_classifier.pt"


def register_model_version():
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_versions
                (
                    model_name,
                    version,
                    model_type,
                    artifact_path,
                    metrics,
                    is_active
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    MODEL_NAME,
                    MODEL_VERSION,
                    MODEL_TYPE,
                    ARTIFACT_PATH,
                    None,
                    True,
                ),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print("Model version registered successfully.")


if __name__ == "__main__":
    register_model_version()