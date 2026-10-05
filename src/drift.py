import os
import pandas as pd

def check_data_drift():
    # Load our processed baseline data
    baseline_path = "data/processed/classified_reviews.csv"
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found. Please run classifier.py first.")
        return

    df_baseline = pd.read_csv(baseline_path)
    
    # Simulate a new incoming batch of reviews containing mostly negative sentiment to test drift
    incoming_data = {
        "review_id": [9, 10, 11, 12],
        "product_name": ["Wireless Mouse", "USB-C Hub", "Mechanical Keyboard", "Wireless Mouse"],
        "customer_review": [
            "Awful experience, stopped working immediately.",
            "Complete waste of money, ports broke.",
            "Terrible quality control, keys fell off.",
            "Disappointing product, will never buy again."
        ],
        "rating": [1, 1, 1, 1],
        "predicted_sentiment": ["NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE"],
        "confidence_score": [0.99, 0.98, 0.99, 0.97]
    }
    df_current = pd.DataFrame(incoming_data)

    # Calculate the ratio of positive sentiments in both datasets
    baseline_positive_ratio = (df_baseline["predicted_sentiment"] == "POSITIVE").mean()
    current_positive_ratio = (df_current["predicted_sentiment"] == "POSITIVE").mean()

    # Define a drift threshold (e.g., a 25% shift triggers an alert)
    drift_threshold = 0.25
    sentiment_shift = abs(baseline_positive_ratio - current_positive_ratio)

    print(f"Baseline Positive Ratio: {baseline_positive_ratio:.2f}")
    print(f"Current Incoming Positive Ratio: {current_positive_ratio:.2f}")
    print(f"Sentiment Shift Magnitude: {sentiment_shift:.2f}")

    # Evaluate drift condition
    if sentiment_shift > drift_threshold:
        print(" WARNING: Data Drift Detected! Customer sentiment has significantly shifted.")
    else:
        print(" Status: Data is stable. No significant drift detected.")

if __name__ == "__main__":
    check_data_drift()