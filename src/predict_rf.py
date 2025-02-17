import joblib

rf_model = joblib.load("../models/fake_news_rf_model.pkl")
tfidf_vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

def predict_fake_news(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = rf_model.predict(text_tfidf)[0]
    return "FAKE" if prediction == 1 else "REAL"

print(predict_fake_news("NASA confirms the discovery of a new habitable exoplanet."))
