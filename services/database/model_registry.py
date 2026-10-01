from services.database.postgres import get_connection
from psycopg.types.json import Json

def register_model_version(
    model_name,
    version,
    model_type,
    artifact_path,
    metrics
):
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
                VALUES (%s, %s, %s, %s, %s, FALSE)
                """,
                (
                    model_name,
                    version,
                    model_type,
                    artifact_path,
                    Json(metrics),
                ),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print(
        f"Model version registered: {model_name} / {version}"
    )

def promote_model_version(model_name, version):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT id, is_active, metrics
                FROM model_versions
                WHERE model_name = %s
                    AND version = %s
                """,
                (model_name, version),
            )

            candidate = cursor.fetchone()

            if candidate is None:
                raise ValueError(
                    f"Model version not found: {model_name} / {version}"
                )

            if candidate[1]:
                raise ValueError(
                    f"Model version is already active: {model_name} / {version}"
                )

            if not candidate[2].get("promotion_eligible", False):
                raise ValueError(
                    f"Model version is not eligible for promotion: {model_name} / {version}"
                )

            cursor.execute(
                """
                UPDATE model_versions
                SET is_active = FALSE
                WHERE model_name = %s
                  AND is_active = TRUE
                """,
                (model_name,),
            )

            cursor.execute(
                """
                UPDATE model_versions
                SET is_active = TRUE
                WHERE model_name = %s
                  AND version = %s
                """,
                (model_name, version),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Failed to activate model version: {model_name} / {version}"
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print(
        f"Model version promoted: {model_name} / {version}"
    )