from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import faiss
import pandas as pd
import open_clip
import torch
from services.inference.predict import predict
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from services.database.review_queue import add_to_review_queue, get_review_queue as get_postgres_review_queue
from services.database.postgres import get_connection
from services.database.product_reviews import add_product_review

def translate_to_english(text: str) -> str:
    """
    Translate product title to English.
    If translation fails, return original text.
    """

    if not text or text.strip() == "":
        return text

    try:
        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        return translated

    except Exception as e:
        print("Translation Error:", e)
        return text

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="AI Catalog Intelligence Platform",
    version="1.0.0",
    description="Production-style multimodal catalog intelligence API"
)

@app.get("/")
def root():
    return {
        "message": "AI Catalog Intelligence Platform API",
        "status": "running"
    }

# -----------------------------
# Load model
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "baseline" / "tfidf_logreg.pkl"

baseline_model = joblib.load(MODEL_PATH)

print(f"Loaded model from: {MODEL_PATH}")

# -----------------------------
# Load embedding assets
# -----------------------------
EMB_PATH = PROJECT_ROOT / "embeddings" / "text_embeddings.npy"
META_PATH = PROJECT_ROOT / "embeddings" / "embedding_metadata.parquet"
FAISS_PATH = PROJECT_ROOT / "embeddings" / "faiss.index"
# -----------------------------
# Review queue database
# -----------------------------
print("\nDashboard DB Path:")
text_embeddings = np.load(EMB_PATH)
metadata_df = pd.read_parquet(META_PATH)
print(metadata_df.columns.tolist())
faiss_index = faiss.read_index(str(FAISS_PATH))

print(f"Loaded {len(metadata_df)} embedding records")

# -----------------------------
# Load OpenCLIP
# -----------------------------
device = "cpu"

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model.eval()
clip_model.to(device)

print("OpenCLIP loaded for semantic search")

# -----------------------------
# Request schema
# -----------------------------
class PredictRequest(BaseModel):
    title: str
    image_path: str


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    title: str = Form(...)
):

    upload_dir = PROJECT_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    image_path = upload_dir / image.filename

    with open(image_path, "wb") as f:
        f.write(await image.read())

    # ---------- Translate Title ----------
    translated_title = translate_to_english(title)

    print("Original Title :", title)
    print("Translated Title:", translated_title)

    # ---------- Run AI ----------
    result = predict(
        image_path=str(image_path),
        title=translated_title
    )

    # -----------------------------------------
    # Automatic Review Queue Logic
    # -----------------------------------------

    duplicate_score = 0.0
    reason = []

    # Low confidence
    if result["confidence"] < 0.70:
        reason.append("Low Confidence")

    # Image/Text mismatch
    if result["mismatch"]:
        reason.append("Image-Text Mismatch")

    # Duplicate detection
    try:

        with torch.no_grad():

            tokens = clip_tokenizer([translated_title]).to(device)

            features = clip_model.encode_text(tokens)

            features = features / features.norm(dim=-1, keepdim=True)

            query_emb = features.cpu().numpy().astype(np.float32)

        scores, indices = faiss_index.search(query_emb, 1)

        duplicate_score = float(scores[0][0])

        if duplicate_score > 0.90:
            reason.append("Possible Duplicate")

    except Exception as e:

        print("Duplicate check failed:", e)

    # Save only if needed
    if reason:

        add_to_review_queue(

            item_id=image.filename,

            image_name=image.filename,

            title=title,

            predicted_category=result["category"],

            confidence=result["confidence"],

            image_similarity=result["image_title_similarity"],

            duplicate_score=duplicate_score,

            reason=", ".join(reason)

        )

    return result

class DuplicateRequest(BaseModel):
    query: str
    category: str | None = None
    top_k: int = 5

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "catalog-intelligence-api",
        "version": "1.0.0"
    }

# -----------------------------
# Predict
# -----------------------------
@app.post("/find-duplicates")
def find_duplicates(request: DuplicateRequest):

    with torch.no_grad():

        tokens = clip_tokenizer([request.query]).to(device)

        features = clip_model.encode_text(tokens)

        features = features / features.norm(dim=-1, keepdim=True)

        query_emb = features.cpu().numpy().astype(np.float32)

    # Search more candidates so filtering still leaves enough results
    scores, indices = faiss_index.search(query_emb, 50)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        row = metadata_df.iloc[idx]

        # Keep only predicted category (if provided)
        if (
            request.category is not None
            and row["target_category"] != request.category
        ):
            continue

        results.append({

            "item_id": row["item_id"],

            "title": row["title"],

            "category": row["target_category"],

            "similarity": round(float(score), 4),

            "image": Path(row["image_path"]).name

        })

        if len(results) == request.top_k:
            break

    return {

        "query": request.query,

        "results": results

    }

@app.get("/review-queue")
def get_review_queue():
    rows = get_postgres_review_queue()

    pending_rows = [
        row for row in rows
        if row["status"] == "Pending"
    ]

    return {
        "total_items": len(pending_rows),
        "items": pending_rows
    }

class ProductReviewRequest(BaseModel):
    product_id: int
    prediction_id: int
    reviewer: str
    decision: str
    corrected_category: str | None = None
    feedback: str | None = None


@app.post("/product-reviews")
def create_product_review(request: ProductReviewRequest):
    add_product_review(
        product_id=request.product_id,
        prediction_id=request.prediction_id,
        reviewer=request.reviewer,
        decision=request.decision,
        corrected_category=request.corrected_category,
        feedback=request.feedback,
    )

    return {
        "success": True,
        "message": "Product review recorded successfully"
    }

@app.put("/review-queue/{item_id}/approve")
def approve_review(item_id: int):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    rq.item_id,
                    rq.status,
                    p.id AS product_id,
                    pp.id AS prediction_id
                FROM review_queue rq
                JOIN products p
                    ON p.item_id = rq.item_id
                JOIN product_predictions pp
                    ON pp.product_id = p.id
                WHERE rq.id = %s
                """,
                (item_id,),
            )

            review = cursor.fetchone()

            if not review:
                conn.rollback()
                return {
                    "success": False,
                    "message": "Review not found"
                }

            queue_item_id, status, product_id, prediction_id = review

            if status != "Pending":
                conn.rollback()
                return {
                    "success": False,
                    "message": f"Review already processed with status: {status}"
                }
            cursor.execute(
                """
                INSERT INTO product_reviews
                (
                    product_id,
                    prediction_id,
                    reviewer,
                    decision,
                    corrected_category,
                    feedback,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    product_id,
                    prediction_id,
                    "admin",
                    "Approved",
                    None,
                    None,
                ),
            )

            cursor.execute(
                """
                UPDATE review_queue
                SET status = 'Approved'
                WHERE id = %s
                """,
                (item_id,),
            )

            updated = cursor.rowcount

        conn.commit()

        return {
            "success": updated > 0,
            "message": "Review Approved Successfully"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


@app.put("/review-queue/{item_id}/reject")
def reject_review(
    item_id: int,
    corrected_category: str | None = None,
    feedback: str | None = None,
):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    rq.item_id,
                    rq.status,
                    p.id AS product_id,
                    pp.id AS prediction_id
                FROM review_queue rq
                JOIN products p
                    ON p.item_id = rq.item_id
                JOIN product_predictions pp
                    ON pp.product_id = p.id
                WHERE rq.id = %s
                """,
                (item_id,),
            )

            review = cursor.fetchone()

            if not review:
                conn.rollback()
                return {
                    "success": False,
                    "message": "Review not found"
                }

            queue_item_id, status, product_id, prediction_id = review

            if status != "Pending":
                conn.rollback()
                return {
                    "success": False,
                    "message": f"Review already processed with status: {status}"
                }

            cursor.execute(
                """
                INSERT INTO product_reviews
                (
                    product_id,
                    prediction_id,
                    reviewer,
                    decision,
                    corrected_category,
                    feedback,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    product_id,
                    prediction_id,
                    "admin",
                    "Rejected",
                    corrected_category,
                    feedback,
                ),
            )

            cursor.execute(
                """
                UPDATE review_queue
                SET status = 'Rejected'
                WHERE id = %s
                """,
                (item_id,),
            )

            updated = cursor.rowcount

        conn.commit()

        return {
            "success": updated > 0,
            "message": "Review Rejected Successfully"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

@app.delete("/review-queue/{review_id}")
def delete_review(review_id: int):
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM review_queue
            WHERE id = %s
            """,
            (review_id,),
        )

        deleted = cursor.rowcount

    conn.commit()
    conn.close()

    return {
        "success": deleted > 0,
        "message": "Review deleted"
    }

@app.get("/metrics")
def get_metrics():
    category_counts = metadata_df["target_category"].value_counts().to_dict()

    return {
        "model": {
            "name": "OpenCLIP + Multimodal Classifier",
            "version": "2.0.0",
            "test_accuracy": 0.9765,
            "validation_accuracy": 0.9835
        },
        "embeddings": {
            "total_embeddings": int(text_embeddings.shape[0]),
            "dimension": int(text_embeddings.shape[1]),
            "faiss_vectors": int(faiss_index.ntotal)
        },
        "dataset": {
            "total_products": int(len(metadata_df)),
            "categories": category_counts
        },
        "thresholds": {
            "duplicate_threshold": 0.90,
            "review_threshold": 0.70,
            "mismatch_threshold": 0.175
        }
    }

@app.get("/dashboard-charts")
def dashboard_charts():

    category_counts = (
        metadata_df["target_category"]
        .value_counts()
        .to_dict()
    )

    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                reason,
                COUNT(*) AS total
            FROM review_queue
            GROUP BY reason
            """
        )

        reason_rows = cursor.fetchall()

    conn.close()

    reason_df = pd.DataFrame(
        reason_rows,
        columns=["reason", "total"]
    )

    return {
        "categories": category_counts,
        "reasons": reason_df.to_dict(
            orient="records"
        )
    }

print("\nRegistered Routes")
for route in app.routes:
    print(route.methods, route.path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

