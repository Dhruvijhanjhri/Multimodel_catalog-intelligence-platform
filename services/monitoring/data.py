from datetime import date
from services.database.postgres import get_connection


def get_prediction_distribution(baseline_date: date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            predicted_category,
            COUNT(*) AS prediction_count
        FROM product_predictions
        WHERE prediction_source = 'saved_embeddings'
          AND created_at::date = %s
        GROUP BY predicted_category
        ORDER BY predicted_category
    """, (baseline_date,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def get_current_prediction_distribution(baseline_date: date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            predicted_category,
            COUNT(*) AS prediction_count
        FROM product_predictions
        WHERE prediction_source = 'seller_portal'
          AND created_at::date > %s
        GROUP BY predicted_category
        ORDER BY predicted_category
    """, (baseline_date,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def get_prediction_confidences(baseline_date: date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT confidence
        FROM product_predictions
        WHERE prediction_source = 'saved_embeddings'
          AND created_at::date = %s
        ORDER BY confidence
    """, (baseline_date,))

    rows = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return rows


def get_current_prediction_confidences(baseline_date: date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT confidence
        FROM product_predictions
        WHERE prediction_source = 'seller_portal'
          AND created_at::date > %s
        ORDER BY confidence
    """, (baseline_date,))

    rows = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return rows

def get_human_feedback_quality():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            pr.prediction_id,
            pp.predicted_category,
            pp.model_version,
            pr.decision,
            pr.corrected_category,
            pr.created_at
        FROM product_reviews pr
        JOIN product_predictions pp
            ON pp.id = pr.prediction_id
        WHERE pr.id = (
            SELECT pr2.id
            FROM product_reviews pr2
            WHERE pr2.prediction_id = pr.prediction_id
            ORDER BY pr2.created_at DESC, pr2.id DESC
            LIMIT 1
        )
        ORDER BY pr.created_at DESC
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows