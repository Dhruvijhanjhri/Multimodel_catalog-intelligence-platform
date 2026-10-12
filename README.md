# Enterprise Multimodal Catalog Intelligence Platform

An AI-powered catalog intelligence platform for e-commerce that combines computer vision, natural language processing, semantic similarity search, and human-in-the-loop review to improve product catalog quality.

The platform supports existing catalog analysis, seller product submissions, bulk catalog ingestion, image–text consistency checks, duplicate candidate retrieval, product taxonomy validation, administrative review, and system monitoring.

## Overview

E-commerce catalogs contain products from multiple sellers, often with inconsistent titles, incorrect categories, duplicate listings, and mismatched images. Manual validation becomes time-consuming as catalog size grows.

The Enterprise Multimodal Catalog Intelligence Platform aims to automate these quality checks while retaining human oversight for products that require additional review.

The system combines multimodal representations, category prediction, similarity search, validation rules, and review workflows to help identify potentially problematic listings.

## Key Features

### 1. AI Product Classification
Predicts a product category using the project's trained classification model.

### 2. Multimodal Image–Text Validation
Uses OpenCLIP representations to evaluate the semantic relationship between a product image and its title or description.

### 3. Similar Product Retrieval
Uses FAISS-based similarity search to retrieve the top five similar catalog products for comparison.

### 4. Duplicate Candidate Detection
Identifies potentially duplicate or highly similar listings using similarity scores. Retrieved matches are candidates for further inspection, not guaranteed duplicates.

### 5. Seller Product Intelligence
Allows sellers to analyze product information, view AI predictions and validation results, and submit products to the catalog.

### 6. Bulk Catalog Ingestion
Accepts a CSV catalog and associated product images for batch processing. In a verified demonstration run, all 10 submitted products were processed successfully.

### 7. Human-in-the-Loop Review
Maintains a review queue for products flagged for issues such as low confidence, image–text mismatch, taxonomy uncertainty, and possible duplication. Administrators can approve or reject flagged items.

### 8. Monitoring Dashboard
Displays catalog and embedding counts, prediction and confidence metrics, review activity, ingestion history, and visual summaries such as category distribution and review-queue reasons.

### 9. Automated Testing
Includes automated tests for prediction API validation, taxonomy validation, decision logic, human feedback, and drift-monitoring components.

## System Architecture

```text
                  Seller / Administrator
                           |
                           v
              Flask Frontend and Web UI
           HTML | CSS | Bootstrap | JavaScript
                           |
                           v
                    FastAPI Backend
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    AI Prediction     Catalog and       Review and
    and Validation    Ingestion Logic   Feedback Logic
          |                |                |
          v                v                v
       OpenCLIP       Database Layer    Review Queue
          |
          v
    Image/Text Features
          |
          v
    Classification Model
          |
          v
    Predicted Category

       Similarity and Retrieval
                 |
                 v
              FAISS
                 |
                 v
       Similar Product Candidates

       Monitoring and Evaluation
                 |
                 v
    Metrics | Ingestion History | Charts
    Drift Status | Human Feedback Quality
```

The frontend communicates with the FastAPI backend through HTTP requests. The backend coordinates prediction, validation, catalog operations, review workflows, and monitoring endpoints. OpenCLIP provides multimodal representations, while the classification model and FAISS retrieval component support category prediction and similar-product discovery.

The project includes PostgreSQL/Supabase integration and SQLite-based review-queue components. The exact storage responsibility depends on the relevant module.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | Flask, HTML5, CSS3, Bootstrap 5, JavaScript |
| Backend API | FastAPI, Python |
| Multimodal representation | OpenCLIP, ViT-B-32 |
| Machine learning | PyTorch, scikit-learn |
| Data processing | NumPy, Pandas |
| Similarity search | FAISS |
| Database integration | PostgreSQL/Supabase and SQLite components |
| Visualization | Dashboard charts and JavaScript-based visualizations |
| Testing | Python `unittest`, FastAPI test client |

## Dataset and Taxonomy

The project uses data derived from the **Amazon Berkeley Objects (ABO)** dataset and associated catalog-processing assets.

The verified training-embedding set contained 23,466 records, each represented by a 512-dimensional embedding. At the time of verification, the dashboard also reported 23,466 products and 23,466 embeddings. These are recorded system counts and do not independently establish the number of unique real-world products.

The platform uses six product taxonomy categories:

- `Electronics_Accessories`
- `Fashion_Travel`
- `Footwear`
- `Furniture`
- `Hardware_HomeImprovement`
- `Home_Kitchen`

The taxonomy supports category prediction, validation, and the identification of potentially inconsistent product listings.

## Machine Learning and Retrieval Pipeline

### Product Analysis

```text
Product Image + Title + Description
                  |
                  v
          Input Validation
                  |
                  v
       Image/Text Representation
                  |
                  v
       Category Classification
                  |
                  v
      Image–Text Consistency Check
                  |
                  v
       Taxonomy and Decision Rules
                  |
                  v
       Prediction and Review Result
```

The category prediction, image–text similarity, and review decision serve different purposes. Prediction confidence expresses the model's confidence in its predicted class; it is not the same as independently measured classification accuracy.

### Similar Product Retrieval

```text
Catalog Product
       |
       v
Feature / Embedding Representation
       |
       v
FAISS Similarity Search
       |
       v
Top-5 Similar Product Candidates
```

The retrieval component helps users compare potentially related listings. Similarity scores indicate relative similarity under the retrieval method and should not be interpreted as calibrated probabilities of duplication.

## Seller and Administrative Workflows

### Seller Workflow

1. Select an existing product or provide new product information.
2. Upload or select the relevant product image.
3. Request AI analysis.
4. Review the predicted category, confidence, and validation results.
5. Submit the product to the catalog.
6. For bulk ingestion, upload a catalog CSV and its associated product images.
7. View submitted products and their review status.

### Administrative Review Workflow

1. Retrieve products flagged by validation and review rules.
2. Inspect the product information, predicted category, confidence, and review reasons.
3. Approve or reject a product.
4. Persist the review decision.
5. Refresh the queue to verify that the reviewed item is no longer pending.

The review workflow provides human oversight rather than assuming that every AI prediction is correct.

## API Endpoints

The FastAPI backend includes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Product prediction and validation |
| GET | `/seller/products` | Retrieve seller products |
| POST | `/seller/products` | Submit a seller product |
| POST | `/seller/bulk-products` | Process bulk catalog uploads |
| GET | `/ingestion-events` | Retrieve ingestion events |
| POST | `/find-duplicates` | Retrieve similar product candidates |
| GET | `/review-queue` | Retrieve pending review items |
| POST | `/product-reviews` | Record product review information |
| PUT | `/review-queue/{item_id}/approve` | Approve a review item |
| PUT | `/review-queue/{item_id}/reject` | Reject a review item |
| DELETE | `/review-queue/{review_id}` | Delete a review item |
| GET | `/metrics` | Retrieve monitoring metrics |
| GET | `/drift-monitoring` | Retrieve drift-monitoring results |
| GET | `/human-feedback-quality` | Retrieve human-feedback quality information |
| GET | `/dashboard-charts` | Retrieve dashboard chart data |
| GET | `/model-versions` | Retrieve model-version information |
| GET | `/` | Root endpoint |


**Note:** Confirm HTTP methods and any additional request parameters against the current FastAPI implementation or its interactive API documentation before relying on this table as a complete API specification.

## Local Setup

### Prerequisites

- Python and a compatible virtual environment
- Git
- Project dependencies installed from the repository's dependency files
- Required dataset files, embeddings, model artifacts, and database configuration

### Start the Backend

From the project root, activate the project's virtual environment and run:

```powershell
uvicorn services.api.main:app --host 127.0.0.1 --port 8001
```

Backend health check:

```text
http://127.0.0.1:8001/health
```

### Start the Frontend

Open a separate terminal from the project root, activate the same virtual environment, and run:

```powershell
python frontend/app.py
```

Frontend:

```text
http://127.0.0.1:5000
```

Keep both services running while testing the integrated application.

### Configuration

Configure any required database credentials, model paths, dataset locations, and environment variables according to the project's existing configuration. Do not commit secrets, passwords, API keys, or database credentials to version control.

## Testing and Demonstration Results

The following workflows have been verified through manual functional checks and automated tests.

| Test area | Observed result |
|---|---|
| Existing catalog analysis | Product prediction and similar-product retrieval worked |
| Seller AI analysis | Product category and validation results displayed |
| Seller product submission | Success message displayed and product returned by the API |
| Seller product listing | Submitted products displayed correctly in the seller product list |
| Bulk catalog ingestion | 10 of 10 products processed in the demonstration batch |
| Admin rejection | Rejection succeeded and item disappeared from the pending queue after refresh |
| Admin approval | Approval succeeded and item disappeared from the pending queue after refresh |
| Monitoring dashboard | Metrics, ingestion history, and charts were visible |
| Automated test suite | 37 tests passed across five test modules |

Automated tests were run using:

```powershell
python -m unittest tests.test_decision_engine tests.test_drift_monitoring tests.test_human_feedback tests.test_taxonomy_validator tests.test_predict_api -v
```

Result: **37 tests ran, all OK.**

These results establish the tested behavior of the current implementation; they do not establish that every prediction, review decision, or ingestion scenario will be correct.

## Monitoring and Model Quality

The monitoring dashboard combines configured model metrics with live operational indicators covering catalog volume, predictions, confidence, human reviews, ingestion activity, and data drift.

### Configured Model Metrics

| Metric | Configured Value |
|---|---:|
| Test Accuracy | 97.65% |
| Validation Accuracy | 98.35% |

These accuracy values are currently configured in the backend metrics response. They are not calculated dynamically by the dashboard. Their validity should be confirmed against the corresponding model evaluation results before they are used for performance comparisons or production reporting.

### Operational Monitoring

The dashboard tracks the following indicators:

| Metric | Observed Value |
|---|---:|
| Total Products | 23,466 |
| Total Embeddings | 23,466 |
| Pending Reviews | 1,516 |
| Live Predictions | 23,503 |
| Average Prediction Confidence | 99.78% |
| Low-Confidence Prediction Rate | 0.19% |
| Human Reviews | 14 |
| Observed Validated Accuracy | 80.00% |
| Validated Predictions | 10 |
| Total Ingestion Events | 8 |
| Processed Ingestion Events | 4 |
| Failed Ingestion Events | 1 |
| Ingestion Failure Rate | 12.50% |

*Operational values represent a recorded system snapshot and may change as the catalog and review workflows are used. Prediction confidence is distinct from classification accuracy.*

### Current Limitations and Improvement Areas

- **Evaluation traceability:** Link reported accuracy metrics to reproducible evaluation runs, datasets, and model versions.
- **Validation coverage:** Expand the set of human-reviewed predictions before drawing conclusions about observed model quality.
- **Drift monitoring:** Collect sufficient representative observations to support meaningful drift analysis. Current drift indicators may report `INSUFFICIENT_DATA`.
- **Catalog quality:** Continue improving validation of category assignments, image–text consistency, and duplicate candidates.
- **Data integrity:** Strengthen duplicate-submission prevention and catalog input validation.
- **Ingestion reliability:** Improve error reporting and monitoring for failed or partially processed batch uploads.

These improvements prioritize measurable model quality, reliable data processing, and transparent operational monitoring.

## Project Status

The project has progressed beyond a standalone classification model into an integrated catalog-intelligence application with seller-facing workflows, batch ingestion, administrative review, and monitoring.

The platform integrates product intelligence, seller submissions, batch ingestion, administrative review, and operational monitoring. Ongoing development focuses on data integrity, evaluation traceability, ingestion reliability, and model-quality monitoring.

## Future Work

Planned areas for continued development include:

- Stronger catalog input validation and duplicate prevention
- More robust bulk-ingestion error handling and reporting
- Better coverage of taxonomy and image–text validation edge cases
- Improved human-feedback quality measurement
- Evidence-based drift and model-quality evaluation
- Broader automated integration tests
- Improved documentation of model evaluation methodology
- Deployment and operational-readiness improvements, where appropriate

The priority is to improve the reliability and verifiability of the existing workflows before expanding the feature set.

## License and Data Usage

Review the applicable licenses and terms for the source code, dependencies, pretrained models, and datasets before distributing or deploying the project. Add a project-specific license only after the repository owners have selected one.
