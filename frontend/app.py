import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Catalog Intelligence Platform",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 AI Catalog Intelligence Platform")
st.caption("Production-style multimodal catalog intelligence demo")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Predict", "Duplicate Search", "Review Queue", "Metrics"]
)

# -----------------------------
# Predict page
# -----------------------------
if page == "Predict":
    st.header("Product Category Prediction")
    
    title = st.text_input(
        "Enter product title",
        "women running sneakers with rubber sole"
    )
    
    if st.button("Predict Category"):
        response = requests.post(
            f"{API_BASE}/predict",
            json={"title": title}
        )
        
        result = response.json()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Category", result["category"])
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        
        if result["needs_review"]:
            st.warning("This product should be sent for manual review.")
        else:
            st.success("No manual review required.")

# -----------------------------
# Duplicate search page
# -----------------------------
elif page == "Duplicate Search":
    st.header("Duplicate / Similar Product Search")
    
    query = st.text_input(
        "Enter product description",
        "women trainers sneakers"
    )
    
    if st.button("Find Similar Products"):
        response = requests.post(
            f"{API_BASE}/find-duplicates",
            json={"query": query, "top_k": 5}
        )
        
        result = response.json()
        
        df = pd.DataFrame(result["results"])
        st.dataframe(df, use_container_width=True)

# -----------------------------
# Review queue page
# -----------------------------
elif page == "Review Queue":
    st.header("Manual Review Queue")
    
    response = requests.get(f"{API_BASE}/review-queue")
    result = response.json()
    
    st.metric("Total Flagged Items", result["total_items"])
    
    if result["items"]:
        df = pd.DataFrame(result["items"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Review queue is empty.")

# -----------------------------
# Metrics page
# -----------------------------
elif page == "Metrics":
    st.header("System Metrics")
    
    response = requests.get(f"{API_BASE}/metrics")
    metrics = response.json()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model")
        st.json(metrics["model"])
        
        st.subheader("Embeddings")
        st.json(metrics["embeddings"])
    
    with col2:
        st.subheader("Dataset")
        st.json(metrics["dataset"])
        
        st.subheader("Thresholds")
        st.json(metrics["thresholds"])