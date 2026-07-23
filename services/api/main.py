from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import faiss
import pandas as pd
import open_clip
import torch
import sqlite3

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="AI Catalog Intelligence Platform",
    version="1.0.0",
    description="Production-style multimodal catalog intelligence API"
)

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
DB_PATH = PROJECT_ROOT / "services" / "review_queue.db"

print(f"Review DB: {DB_PATH}")

text_embeddings = np.load(EMB_PATH)
metadata_df = pd.read_parquet(META_PATH)
faiss_index = faiss.read_index(str(FAISS_PATH))

print(f"Loaded {len(metadata_df)} embedding records")

# -----------------------------
# Load OpenCLIP
# -----------------------------
device = "cpu"

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
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

class DuplicateRequest(BaseModel):
    query: str
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
@app.post("/predict")
def predict(request: PredictRequest):
    # Predict class
    prediction = baseline_model.predict([request.title])[0]
    
    # Predict probabilities
    probabilities = baseline_model.predict_proba([request.title])[0]
    
    # Confidence
    confidence = float(np.max(probabilities))
    
    # Review rule
    needs_review = confidence < 0.70
    
    return {
        "title": request.title,
        "category": prediction,
        "confidence": round(confidence, 4),
        "needs_review": needs_review
    }

@app.post("/find-duplicates")
def find_duplicates(request: DuplicateRequest):
    # Encode query
    with torch.no_grad():
        tokens = clip_tokenizer([request.query]).to(device)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        query_emb = features.cpu().numpy().astype(np.float32)
    
    # Search
    scores, indices = faiss_index.search(query_emb, request.top_k)
    
    results = []
    
    for score, idx in zip(scores[0], indices[0]):
        row = metadata_df.iloc[idx]
        
        results.append({
            "item_id": row["item_id"],
            "title": row["title"],
            "category": row["target_category"],
            "similarity": round(float(score), 4)
        })
    
    return {
        "query": request.query,
        "results": results
    }

@app.get("/review-queue")
def get_review_queue():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        item_id,
        title,
        category,
        confidence,
        mismatch_score,
        duplicate_score,
        reason,
        created_at
    FROM review_queue
    ORDER BY created_at DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return {
        "total_items": len(df),
        "items": df.to_dict(orient="records")
    }

@app.get("/metrics")
def get_metrics():
    category_counts = metadata_df["target_category"].value_counts().to_dict()

    return {
        "model": {
            "name": "TF-IDF + Logistic Regression",
            "version": "1.0.0",
            "test_accuracy": 0.9654,
            "validation_accuracy": 0.9595
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
            "mismatch_threshold": 0.80
        }
    }