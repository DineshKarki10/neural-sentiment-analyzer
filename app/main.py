import os
import pandas as pd
import streamlit as st

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Neural Sentiment Analyzer & Drift Monitor",
    page_icon="📊",
    layout="wide"
)

# App Title and Description
st.title("🧠 Neural Sentiment Analyzer & Data Drift Dashboard")
st.markdown("""
This dashboard monitors customer review sentiments using a Hugging Face Transformer model 
and tracks real-time **Data Drift** to alert teams when customer feedback shifts.
""")

# Load Processed Data Function
@st.cache_data
def load_data():
    path = "data/processed/classified_reviews.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

if df is None:
    st.warning("No processed data found! Please run `python src/classifier.py` in your terminal first.")
else:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Dashboard Controls")
    product_filter = st.sidebar.selectbox(
        "Filter by Product", 
        options=["All Products"] + list(df["product_name"].unique())
    )
    
    # Filter DataFrame based on sidebar selection
    filtered_df = df if product_filter == "All Products" else df[df["product_name"] == product_filter]

    # --- TOP METRICS ROW ---
    col1, col2, col3 = st.columns(3)
    
    total_reviews = len(filtered_df)
    positive_count = (filtered_df["predicted_sentiment"] == "POSITIVE").sum()
    positive_ratio = (positive_count / total_reviews) if total_reviews > 0 else 0
    avg_confidence = filtered_df["confidence_score"].mean()

    col1.metric("Total Reviews Analyzed", total_reviews)
    col2.metric("Positive Sentiment Ratio", f"{positive_ratio * 100:.1f}%")
    col3.metric("Avg Model Confidence", f"{avg_confidence * 100:.1f}%")

    st.markdown("---")

    # --- DATA TABLE SECTION ---
    st.subheader(" Customer Review Classification Table")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    # --- DATA DRIFT MONITORING SECTION ---
    st.subheader("Data Drift & Sentiment Shift Monitor")
    
    # Simulate incoming batch comparison
    baseline_positive_ratio = (df["predicted_sentiment"] == "POSITIVE").mean()
    drift_threshold = 0.25
    
    # Let user simulate an incoming batch sentiment ratio via slider
    simulated_current_ratio = st.slider(
        "Simulate Incoming Batch Positive Ratio", 
        min_value=0.0, max_value=1.0, value=float(positive_ratio), step=0.05
    )
    
    shift_magnitude = abs(baseline_positive_ratio - simulated_current_ratio)
    
    col_a, col_b = st.columns(2)
    col_a.metric("Baseline Positive Ratio", f"{baseline_positive_ratio:.2f}")
    col_b.metric("Incoming Batch Ratio", f"{simulated_current_ratio:.2f}")

    if shift_magnitude > drift_threshold:
        st.error(f"**DATA DRIFT ALERT:** Sentiment shift magnitude ({shift_magnitude:.2f}) exceeds threshold ({drift_threshold}). Action required!")
    else:
        st.success(f"**Status:** Data stream is stable. Shift magnitude ({shift_magnitude:.2f}) is within acceptable limits.")