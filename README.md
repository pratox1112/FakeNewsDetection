# Fake News Detection Using DistilBERT & Random Forest

## Overview
This project classifies news articles as **REAL** or **FAKE** using two complementary approaches:
- A deep learning model based on **DistilBERT** (a lightweight, faster variant of BERT)
- A traditional machine learning model using **Random Forest** with **TF-IDF** features

The goal is to automate the detection of misinformation by leveraging state-of-the-art NLP techniques, and the models are served via a Flask API.

---

## Performance

### DistilBERT (Deep Learning)
- **Accuracy:** 99.0%
- **Metrics:**
  - **Precision:** REAL: 0.99, FAKE: 1.00
  - **Recall:** REAL: 1.00, FAKE: 0.99
  - **F1-Score:** REAL: 0.99, FAKE: 0.99

### Random Forest (TF-IDF)
- **Accuracy:** 93.0%
- **Metrics:**
  - **Precision:** REAL: 0.92, FAKE: 0.93
  - **Recall:** REAL: 0.93, FAKE: 0.92
  - **F1-Score:** REAL: 0.92, FAKE: 0.92

---

## Project Structure

fake-news-detection/ ├── data/ │ ├── WELFake_Dataset.csv # Contains the dataset (e.g., WELFake_Dataset.csv) ├── models/ │ ├── fake_news_rf_model.pkl # Trained Random Forest Model │ ├── tfidf_vectorizer.pkl # Saved TF-IDF Vectorizer │ ├── fake_news_distilbert_model.pth # Trained DistilBERT Model │ ├── distilbert_tokenizer.pkl # Saved DistilBERT Tokenizer ├── src/ │ ├── preprocess.py # Data cleaning and preprocessing │ ├── train_rf.py # Script to train the Random Forest model │ ├── train_bert.py # Script to fine-tune the DistilBERT model │ ├── predict_rf.py # Script for Random Forest predictions │ ├── predict_bert.py # Script for DistilBERT predictions ├── app/ │ ├── app.py # Flask API for serving predictions ├── tests/ │ ├── test_predictions.py # Automated tests for model predictions ├── requirements.txt # List of dependencies ├── README.md



