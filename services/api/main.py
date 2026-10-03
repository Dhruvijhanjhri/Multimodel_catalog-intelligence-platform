from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import faiss
import pandas as pd
import open_clip
import torch
from services.inference.predict import predict, MODEL_VERSION
from services.inference.taxonomy_validator import (
    validate_taxonomy,
    determine_taxonomy_status,
)
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from services.database.review_queue import add_to_review_queue, get_review_queue as get_postgres_review_queue
from services.database.postgres import get_connection
from services.database.product_reviews import add_product_review
import uuid
from services.inference.decision_engine import evaluate_decision
from services.database.model_registry import get_model_versions

def translate_to_english(text: str) -> str:
    """
    Translate product title to English when needed.
    If the text already appears to be English, skip translation.
    If translation fails, return the original text.
    """

    if not text or text.strip() == "":
        return text

    # Skip external translation for titles that are already
    # composed mainly of English letters, numbers and punctuation.
    english_characters = sum(
        1 for char in text
        if char.isascii() and (char.isalpha() or char.isdigit())
    )

    total_characters = sum(
        1 for char in text
        if not char.isspace()
    )

    if total_characters > 0:
        english_ratio = english_characters / total_characters

        if english_ratio >= 0.80:
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
BULK_UPLOAD_DIR = PROJECT_ROOT / "bulk_uploads"
BULK_UPLOAD_DIR.mkdir(exist_ok=True)

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

    image_path = upload_dir / f"{uuid.uuid4()}_{image.filename}"

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

    # ---------- Taxonomy Validation ----------
    taxonomy_result = validate_taxonomy(
        image_embedding=result["image_embedding"],
        text_embedding=result["text_embedding"],
        predicted_category=result["category"],
    )

    taxonomy_status = determine_taxonomy_status(
        confidence=result["confidence"],
        support_margin=taxonomy_result["support_margin"],
        mismatch=result["mismatch"],
    )

    # -----------------------------------------
    # Automatic Review Queue Logic
    # -----------------------------------------

    duplicate_score = 0.0

    # Duplicate detection
    try:

        with torch.no_grad():

            tokens = clip_tokenizer([translated_title]).to(device)

            features = clip_model.encode_text(tokens)

            features = features / features.norm(dim=-1, keepdim=True)

            query_emb = features.cpu().numpy().astype(np.float32)

        scores, indices = faiss_index.search(query_emb, 1)

        duplicate_score = float(scores[0][0])


    except Exception as e:

        print("Duplicate check failed:", e)

    decision_result = evaluate_decision(
        confidence=result["confidence"],
        mismatch=result["mismatch"],
        taxonomy_status=taxonomy_status,
        duplicate_score=duplicate_score,
    )

    reason = decision_result["reasons"]
    result["decision"] = decision_result["decision"]
    result["review_reasons"] = reason
    
    # Save only if needed
    if reason:

        add_to_review_queue(

            item_id=image_path.name,

            image_name=image_path.name,

            title=title,

            predicted_category=result["category"],

            confidence=result["confidence"],

            image_similarity=result["image_title_similarity"],

            duplicate_score=duplicate_score,

            reason=", ".join(reason)

        )

    result["image_name"] = image_path.name
    result["duplicate_score"] = duplicate_score

    # Taxonomy validation results
    result["taxonomy_status"] = taxonomy_status
    result["taxonomy_support"] = taxonomy_result["catalog_support"]
    result["taxonomy_margin"] = taxonomy_result["support_margin"]
    result["best_alternative_category"] = (
        taxonomy_result["best_alternative_category"]
    )
    result["best_alternative_support"] = (
        taxonomy_result["best_alternative_support"]
    )

    # Keep embeddings internal
    result.pop("image_embedding", None)
    result.pop("text_embedding", None)

    return result

class SellerProductRequest(BaseModel):
    title: str
    brand: str | None = None
    description: str | None = None
    image_name: str
    category: str
    confidence: float
    image_title_similarity: float
    mismatch: bool
    duplicate_score: float
    taxonomy_status: str | None = None
    taxonomy_margin: float | None = None
    model_version: str | None = None
    item_id: str | None = None

@app.post("/seller/bulk-products")
async def create_bulk_seller_products(
    catalog: UploadFile = File(...),
    images: list[UploadFile] = File(...),
):
    batch_id = str(uuid.uuid4())
    batch_dir = BULK_UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = batch_dir / catalog.filename

    with open(catalog_path, "wb") as f:
        f.write(await catalog.read())

    image_map = {}

    for image in images:
        image_path = batch_dir / image.filename

        with open(image_path, "wb") as f:
            f.write(await image.read())

        image_map[image.filename] = image_path

    df = pd.read_csv(catalog_path)

    required_columns = {"product_id", "title", "brand", "image"}

    if not required_columns.issubset(df.columns):
        return {
            "success": False,
            "message": "CSV must contain product_id, title, brand and image columns.",
        }

    product_ids = df["product_id"].dropna().astype(str).str.strip()

    if product_ids.duplicated().any():
        return {
            "success": False,
            "message": "CSV contains duplicate product_id values.",
        }

    results = []

    for _, row in df.iterrows():

        product_id = str(row["product_id"]).strip()

        if not product_id or product_id.lower() == "nan":
            results.append({
                "product_id": None,
                "success": False,
                "message": "Product ID is required.",
            })
            continue

        csv_image_name = str(row["image"]).strip()

        title = str(row["title"]).strip()

        if not title or title.lower() == "nan":
            results.append({
                "product_id": product_id,
                "success": False,
                "message": "Product title is required.",
            })
            continue

        if not csv_image_name or csv_image_name.lower() == "nan":
            results.append({
                "product_id": product_id,
                "success": False,
                "message": "Image filename is required.",
            })
            continue

        if csv_image_name not in image_map:
            results.append({
                "product_id": str(row["product_id"]),
                "success": False,
                "message": f"Image not provided: {csv_image_name}",
            })
            continue

        source_image_path = image_map[csv_image_name]

        translated_title = translate_to_english(title)
        prediction = predict(
            image_path=str(source_image_path),
            title=translated_title,
        )

        taxonomy_result = validate_taxonomy(
            prediction["image_embedding"],
            prediction["text_embedding"],
            prediction["category"],
        )

        taxonomy_status = determine_taxonomy_status(
            confidence=prediction["confidence"],
            support_margin=taxonomy_result["support_margin"],
            mismatch=prediction["mismatch"],
        )

        duplicate_score = 0.0

        try:
            with torch.no_grad():

                tokens = clip_tokenizer([translated_title]).to(device)

                features = clip_model.encode_text(tokens)

                features = features / features.norm(dim=-1, keepdim=True)

                query_emb = features.cpu().numpy().astype(np.float32)

            scores, indices = faiss_index.search(query_emb, 1)

            duplicate_score = float(scores[0][0])

        except Exception as e:
            print("Bulk duplicate check failed:", e)

        decision_result = evaluate_decision(
            confidence=prediction["confidence"],
            mismatch=prediction["mismatch"],
            taxonomy_status=taxonomy_status,
            duplicate_score=duplicate_score,
        )

        decision = decision_result["decision"]
        review_reasons = decision_result["reasons"]

        stored_image_name = f"{uuid.uuid4()}_{csv_image_name}"
        stored_image_path = PROJECT_ROOT / "uploads" / stored_image_name

        with open(source_image_path, "rb") as source:
            with open(stored_image_path, "wb") as destination:
                destination.write(source.read())

        seller_request = SellerProductRequest(
            item_id=str(row["product_id"]),
            title=str(row["title"]),
            brand=None if pd.isna(row["brand"]) else str(row["brand"]),
            description=None,
            image_name=stored_image_name,
            category=prediction["category"],
            confidence=prediction["confidence"],
            image_title_similarity=prediction["image_title_similarity"],
            mismatch=prediction["mismatch"],
            duplicate_score=duplicate_score,
            taxonomy_status=taxonomy_status,
            taxonomy_margin=taxonomy_result["support_margin"],
            model_version=MODEL_VERSION,
        )

        existing_product = None

        with get_connection() as check_conn:
            with check_conn.cursor() as check_cursor:
                check_cursor.execute(
                    """
                    SELECT id
                    FROM products
                    WHERE item_id = %s
                    LIMIT 1
                    """,
                    (str(row["product_id"]),),
                )
                existing_product = check_cursor.fetchone()

        if existing_product:
            results.append({
                "product_id": str(row["product_id"]),
                "success": True,
                "skipped": True,
                "message": "Product already exists. Skipped duplicate ingestion.",
                "database_product_id": existing_product[0],
            })
            continue

        saved = create_seller_product(seller_request)

        results.append({
            "product_id": str(row["product_id"]),
            "success": True,
            "database_product_id": saved["product_id"],
            "category": prediction["category"],
            "confidence": prediction["confidence"],
            "image_title_similarity": prediction["image_title_similarity"],
            "mismatch": prediction["mismatch"],
            "duplicate_score": duplicate_score,
            "decision": decision,
            "review_reasons": review_reasons,
            "taxonomy_status": taxonomy_status,
            "taxonomy_margin": taxonomy_result["support_margin"],
            "review_reason": saved["reason"],
        })

    return {
        "success": True,
        "batch_id": batch_id,
        "total_products": len(df),
        "processed_products": len(results),
        "results": results,
    }

@app.get("/seller/products")
def get_seller_products():

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    p.id,
                    p.item_id,
                    p.title,
                    p.brand,
                    p.category,
                    pp.confidence,
                    pp.model_version,
                    pr.decision,
                    pr.corrected_category,
                    pr.feedback
                FROM products p
                LEFT JOIN product_predictions pp
                    ON pp.product_id = p.id
                LEFT JOIN product_reviews pr
                    ON pr.product_id = p.id
                WHERE p.source = 'seller_portal'
                ORDER BY p.created_at DESC
                """
            )

            rows = cursor.fetchall()

            columns = [
                "product_id",
                "item_id",
                "title",
                "brand",
                "category",
                "confidence",
                "model_version",
                "review_decision",
                "corrected_category",
                "review_feedback",
            ]

            return {
                "total_items": len(rows),
                "items": [
                    dict(zip(columns, row))
                    for row in rows
                ]
            }

    finally:
        conn.close()

@app.post("/seller/products")
def create_seller_product(request: SellerProductRequest):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            seller_item_id = request.item_id or f"SELLER-{uuid.uuid4()}"

            cursor.execute(
                """
                INSERT INTO products
                (
                    item_id,
                    title,
                    brand,
                    category,
                    source,
                    created_at,
                    updated_at
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP
                )
                RETURNING id
                """,
                (
                    seller_item_id,
                    request.title,
                    request.brand,
                    request.category,
                    "seller_portal",
                ),
            )
            product_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO product_images
                (
                    product_id,
                    image_name,
                    image_path,
                    embedding_path,
                    is_primary,
                    created_at
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    CURRENT_TIMESTAMP
                )
                """,
                (
                    product_id,
                    request.image_name,
                    f"uploads/{request.image_name}",
                    None,
                    True,
                ),
            )

            cursor.execute(
                """
                INSERT INTO product_predictions
                (
                    product_id,
                    predicted_category,
                    confidence,
                    model_version,
                    prediction_source,
                    created_at
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    CURRENT_TIMESTAMP
                )
                """,
                (
                    product_id,
                    request.category,
                    request.confidence,
                    request.model_version,
                    "seller_portal",
                ),
            )
        decision_result = evaluate_decision(
            confidence=request.confidence,
            mismatch=request.mismatch,
            taxonomy_status=request.taxonomy_status,
            duplicate_score=request.duplicate_score,
        )

        reason = decision_result["reasons"]

        if reason:
            add_to_review_queue(
                item_id=seller_item_id,
                image_name=request.image_name,
                title=request.title,
                predicted_category=request.category,
                confidence=request.confidence,
                image_similarity=request.image_title_similarity,
                duplicate_score=request.duplicate_score,
                reason=", ".join(reason),
            )

        conn.commit()

        return {
            "success": True,
            "product_id": product_id,
            "message": "Seller product saved successfully",
            "reason": reason
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

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

@app.get("/model-versions")
def get_model_versions_endpoint():
    return {
        "model_name": "multimodal_classifier",
        "versions": get_model_versions("multimodal_classifier"),
    }

@app.get("/metrics")
def get_metrics():
    category_counts = metadata_df["target_category"].value_counts().to_dict()

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,
                    AVG(confidence) AS average_confidence,
                    COUNT(*) FILTER (WHERE confidence < 0.70)
                        AS low_confidence_predictions
                FROM product_predictions
                """
            )

            prediction_stats = cursor.fetchone()

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_reviews,
                    COUNT(*) FILTER (WHERE decision = 'Approved')
                        AS approved_reviews,
                    COUNT(*) FILTER (WHERE decision = 'Rejected')
                        AS rejected_reviews,
                    COUNT(*) FILTER (
                        WHERE decision = 'Rejected'
                        AND corrected_category IS NOT NULL
                    ) AS corrected_reviews
                FROM product_reviews
                """
            )

            review_stats = cursor.fetchone()

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_queue_items,
                    COUNT(*) FILTER (WHERE status = 'Pending')
                        AS pending_reviews
                FROM review_queue
                """
            )

            queue_stats = cursor.fetchone()
            total_review_queue_items = int(queue_stats[0] or 0)
            pending_reviews = int(queue_stats[1] or 0)
            processed_reviews = total_review_queue_items - pending_reviews

            cursor.execute(
                """
                SELECT
                    model_version,
                    COUNT(*) AS total_predictions
                FROM product_predictions
                GROUP BY model_version
                ORDER BY total_predictions DESC
                """
            )

            model_version_rows = cursor.fetchall()

    finally:
        conn.close()

    total_predictions = int(prediction_stats[0] or 0)
    average_confidence = float(prediction_stats[1] or 0)
    low_confidence_predictions = int(prediction_stats[2] or 0)
    total_reviews = int(review_stats[0] or 0)
    approved_reviews = int(review_stats[1] or 0)
    rejected_reviews = int(review_stats[2] or 0)
    corrected_reviews = int(review_stats[3] or 0)

    approval_rate = (
        approved_reviews / total_reviews
        if total_reviews
        else 0
    )

    correction_rate = (
        corrected_reviews / rejected_reviews
        if rejected_reviews
        else 0
    )
    
    review_queue_processing_rate = (
        processed_reviews / total_review_queue_items
        if total_review_queue_items
        else 0
    )

    low_confidence_rate = (
        low_confidence_predictions / total_predictions
        if total_predictions
        else 0
    )

    predictions_by_model_version = {
        row[0]: int(row[1])
        for row in model_version_rows
    }

    return {
        "model": {
            "name": "OpenCLIP + Multimodal Classifier",
            "version": MODEL_VERSION,
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
        "prediction_monitoring": {
            "total_predictions": total_predictions,
            "average_confidence": average_confidence,
            "low_confidence_predictions": low_confidence_predictions,
            "low_confidence_rate": low_confidence_rate,
            "predictions_by_model_version": predictions_by_model_version
        },
        "feedback_monitoring": {
            "total_reviews": total_reviews,
            "approved_reviews": approved_reviews,
            "rejected_reviews": rejected_reviews,
            "corrected_reviews": corrected_reviews,
            "approval_rate": approval_rate,
            "correction_rate": correction_rate
        },
        "review_queue_monitoring": {
            "pending_reviews": pending_reviews,
            "processed_reviews": processed_reviews,
            "total_queue_items": total_review_queue_items,
            "processing_rate": review_queue_processing_rate
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

