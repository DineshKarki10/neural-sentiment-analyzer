# import necessary libraries
import os
import pandas as pd

# a function to generate sample reviews 
def generate_sample_reviews():
    """
    Generates a structured dataset of product reviews to simulate scraped e-commerce data.
    In a live project, we can replace this logic with BeautifulSoup or a public API scraper.
    """
    data = {
        "review_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "product_name": ["Wireless Mouse", "Mechanical Keyboard", "Wireless Mouse", "USB-C Hub", 
                         "Mechanical Keyboard", "USB-C Hub", "Wireless Mouse", "Mechanical Keyboard"],
        "customer_review": [
            "Absolute garbage. Stopped working after two days of light use.",
            "Fantastic keyboard! The clicky switches feel amazing for typing and gaming.",
            "Decent mouse for the price, but the scroll wheel feels a bit cheap.",
            "Terrible port replication. My laptop kept disconnecting when plugged in.",
            "Mediocre build quality. Keys started sticking after a week.",
            "Works perfectly right out of the box. Very sleek and portable!",
            "Best mouse I have ever owned. Ergonomics are top-notch.",
            "Worst purchase ever. Arrived broken and customer support was useless."
        ],
        "rating": [1, 5, 3, 1, 2, 5, 5, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure raw data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to CSV
    file_path = "data/raw/reviews.csv"
    df.to_csv(file_path, index=False)
    print(f"Successfully saved {len(df)} reviews to {file_path}")

if __name__ == "__main__":
    generate_sample_reviews()