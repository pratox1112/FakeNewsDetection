project:
  name: "Fake News Detection Using DistilBERT & Random Forest"
  description: >
    This project classifies news articles as REAL or FAKE using two approaches:
    - A deep learning model based on DistilBERT (a lighter, faster variant of BERT)
    - A traditional machine learning model using Random Forest with TF-IDF features
    The goal is to automate the detection of misinformation by leveraging NLP techniques,
    and the models are served via a Flask API.
  
  performance:
    DistilBERT:
      accuracy: "99.0%"
      metrics:
        precision:
          REAL: "0.99"
          FAKE: "1.00"
        recall:
          REAL: "1.00"
          FAKE: "0.99"
        f1_score:
          REAL: "0.99"
          FAKE: "0.99"
    RandomForest:
      accuracy: "93.0%"
      metrics:
        precision:
          REAL: "0.92"
          FAKE: "0.93"
        recall:
          REAL: "0.93"
          FAKE: "0.92"
        f1_score:
          REAL: "0.92"
          FAKE: "0.92"

  structure:
    - data/: "Contains the dataset (e.g., WELFake_Dataset.csv)"
    - models/: 
        - "fake_news_rf_model.pkl (Random Forest Model)"
        - "tfidf_vectorizer.pkl (TF-IDF Vectorizer)"
        - "fake_news_distilbert_model.pth (DistilBERT Model)"
        - "distilbert_tokenizer.pkl (DistilBERT Tokenizer)"
    - src/:
        - preprocess.py: "Data cleaning and preprocessing"
        - train_rf.py: "Script to train the Random Forest model"
        - train_bert.py: "Script to fine-tune the DistilBERT model"
        - predict_rf.py: "Script for Random Forest predictions"
        - predict_bert.py: "Script for DistilBERT predictions"
    - app/:
        - app.py: "Flask API for serving predictions"
    - tests/:
        - test_predictions.py: "Automated tests for model predictions"
    - requirements.txt: "List of dependencies"
    - README.md: "Project documentation (this file)"
  
  installation:
    - "git clone https://github.com/yourusername/fake-news-detection.git"
    - "cd fake-news-detection"
    - "pip install -r requirements.txt"
  
  usage:
    training:
      - "python src/train_rf.py  # Train the Random Forest model"
      - "python src/train_bert.py  # Fine-tune the DistilBERT model"
    prediction:
      - "python src/predict_rf.py  # Run predictions using the Random Forest model"
      - "python src/predict_bert.py  # Run predictions using the DistilBERT model"
    flask_api:
      - "python app/app.py  # Launch the Flask API to serve predictions"
  
  deployment:
    description: >
      The Flask API can be deployed on various cloud platforms such as AWS, Google Cloud, or Render.
      This allows for scalable and accessible real-time predictions via RESTful endpoints.
  
  contact:
    email: "your.email@example.com"
  
  license: "MIT"
