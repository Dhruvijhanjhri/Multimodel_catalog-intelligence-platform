import uuid
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from services.database.postgres import get_connection


def create_ingestion_event(
    source,
    event_type,
    payload,
    item_id=None,
    status="Received",
):
    event_id = str(uuid.uuid4())

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_events
                (
                    event_id,
                    item_id,
                    source,
                    event_type,
                    payload,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    event_id,
                    item_id,
                    source,
                    event_type,
                    Jsonb(payload),
                    status,
                ),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return event_id


def update_ingestion_event(
    event_id,
    status,
    payload=None,
    error_message=None,
):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ingestion_events
                SET
                    status = %s,
                    payload = COALESCE(%s, payload),
                    error_message = %s,
                    processed_at = %s
                WHERE event_id = %s
                """,
                (
                    status,
                    Jsonb(payload) if payload is not None else None,
                    error_message,
                    datetime.now(timezone.utc),
                    event_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    f"Ingestion event not found: {event_id}"
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()