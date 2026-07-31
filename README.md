# Enterprise Multimodal Catalog Intelligence Platform

An AI-powered multimodal catalog intelligence system that automatically classifies products, validates image-title consistency, detects duplicate listings, performs semantic similarity search, and manages a review workflow for catalog quality assurance.

---

## Project Overview

Large e-commerce platforms receive thousands of product listings from multiple sellers every day. Manual validation of these listings is time-consuming and error-prone.

This platform automates catalog quality checks using Artificial Intelligence by combining computer vision, natural language processing, and semantic search.

The system can:

- Classify products into predefined categories
- Validate image-title consistency
- Detect duplicate products using semantic similarity
- Recommend visually and semantically similar products
- Automatically flag suspicious products for manual review
- Provide an Admin Dashboard for review management

---

## Key Features

### AI Product Classification
Predicts the product category using a trained machine learning model.

### Multimodal Image–Text Validation
Uses OpenCLIP embeddings to verify whether the uploaded image matches the provided product title.

### Duplicate Product Detection
Uses FAISS similarity search to identify duplicate or highly similar catalog entries.

### Similar Product Recommendation
Retrieves the Top-5 most similar products from the catalog.

### Automated Review Queue
Automatically flags suspicious products based on configurable confidence and similarity thresholds.

### Admin Dashboard
Provides:

- Review Queue Management
- Approve / Reject / Delete workflow
- Dataset statistics
- Model metrics
- Category distribution visualization

---

## System Architecture

```
Flask Frontend
        │
        ▼
 FastAPI Backend
        │
 ┌──────┼─────────────┐
 │      │             │
 ▼      ▼             ▼
OpenCLIP   ML Model   FAISS
 │          │          │
 └──────┬───┴──────────┘
        ▼
 Review Queue (SQLite)
```

---

## Tech Stack

### Frontend

- Flask
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

### Backend

- FastAPI
- Python

### Machine Learning

- OpenCLIP
- Scikit-learn
- PyTorch
- NumPy
- Pandas

### Similarity Search

- FAISS

### Database

- SQLite

### Visualization

- Chart.js

---

## Dataset

**Amazon Berkeley Objects (ABO)**

Dataset has been cleaned and curated into six product categories.

### Categories

- Footwear
- Home & Kitchen
- Furniture
- Electronics Accessories
- Fashion & Travel
- Hardware & Home Improvement

### Dataset Size

- Total Products: **23,466**
- Text Embeddings: **7,900**
- Categories: **6**

---

## Machine Learning Pipeline

```
Product Image
        │
Product Title
        │
        ▼
OpenCLIP Embeddings
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Logistic Regression Classifier
        │
        ▼
Predicted Category
```

---

## Duplicate Detection Pipeline

```
Product Title
       │
OpenCLIP Embedding
       │
FAISS Index
       │
Top-5 Similar Products
```

---

## API Endpoints

### Prediction

```
POST /predict
```

Predict product category and validate image-title consistency.

---

### Duplicate Detection

```
POST /find-duplicates
```

Retrieve semantically similar products.

---

### Review Queue

```
GET /review-queue
PUT /review-queue/{id}/approve
PUT /review-queue/{id}/reject
DELETE /review-queue/{id}
```

---

### Dashboard

```
GET /metrics
GET /dashboard-charts
```

---

## Admin Dashboard Features

- Dataset Statistics
- Model Accuracy
- Review Queue
- Category Distribution
- Review Reason Distribution
- Approve Product
- Reject Product
- Delete Product

---

## Model Performance

| Metric | Value |
|---------|------:|
| Validation Accuracy | **95.95%** |
| Test Accuracy | **96.54%** |

---

## Project Structure

```
frontend/
    app.py
    templates/
    static/

services/
    api/
    review_queue.db

models/
data/
notebooks/
```

---

## Future Enhancements

- User Authentication
- Seller Dashboard
- OCR-based Catalog Validation
- Multi-language Product Support
- Docker Deployment
- PostgreSQL Integration
- Cloud Deployment (AWS/Azure/GCP)
- Real-time Catalog Monitoring

---

## Authors

**Dhruvi Jhanjhri**

M.Sc. Data Science

Christ University