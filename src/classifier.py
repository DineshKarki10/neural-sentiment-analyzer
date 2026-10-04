import os
import pandas as pd
from transformers import pipeline

def classify_reviews():
    # Define file paths for input and output data
    input_path = "data/raw/reviews.csv"
    output_path = "data/processed/classified_reviews.csv"
    
    # Ensure the raw data file exists before processing
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run collector.py first.")
        return

    # Load raw customer reviews into a Pandas DataFrame table
    df = pd.read_csv(input_path)
    
    # Initialize Hugging Face sentiment-analysis pipeline (downloads weights automatically on first run)
    print("Loading Hugging Face transformer model...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Loop through each customer review and run model inference
    print("Classifying reviews...")
    results = []
    for review in df["customer_review"]:
        prediction = sentiment_analyzer(review)[0]
        results.append(prediction)
        
    # Extract predicted labels (POSITIVE/NEGATIVE) and confidence scores (0 to 1)
    df["predicted_sentiment"] = [res["label"] for res in results]
    df["confidence_score"] = [res["score"] for res in results]
    
    # Ensure the processed data directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Save the updated dataframe with predictions into a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Successfully classified {len(df)} reviews and saved to {output_path}")

if __name__ == "__main__":
    classify_reviews()