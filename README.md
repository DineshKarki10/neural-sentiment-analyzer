# Neural Sentiment Analyzer & Data Drift Monitor

A  machine learning application that classifies customer product reviews using a pre-trained Hugging Face Transformer model and monitors real-time data drift via an interactive Streamlit dashboard.

---

## Features
* **Transformer-Based Sentiment Analysis:** Uses a fine-tuned DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) via Hugging Face to classify customer reviews into positive and negative sentiments with confidence scores.
* **Data Drift Monitoring:** Evaluates incoming batches of review data against baseline distributions to calculate sentiment shift magnitude and trigger automatic drift alerts.
* **Interactive Streamlit Dashboard:** A responsive web application featuring real-time metrics, product filters, interactive data tables, and a simulated drift threshold slider.

---

## Tech Stack
* **Python** (Core logic)
* **PyTorch & Hugging Face Transformers** (Deep learning / NLP model inference)
* **Pandas** (Data manipulation and structuring)
* **Streamlit & Plotly** (Interactive web dashboard and data visualization)
* **Scikit-learn** (Data utility metrics)

---

##  Project Structure

neural-sentiment-analyzer/
│
├── app/
│   └── main.py              # Streamlit web dashboard
├── data/                    # Raw and processed datasets (ignored in git)
├── src/
│   ├── collector.py         # E-commerce review dataset simulator
│   ├── classifier.py        # Hugging Face transformer pipeline
│   └── drift.py             # Data drift calculation and alert logic
├── .gitignore
├── README.md
└── requirements.txt