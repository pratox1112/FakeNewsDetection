import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("../data/processed_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

y_pred = rf_model.predict(X_test_tfidf)

print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(rf_model, "../models/fake_news_rf_model.pkl")
joblib.dump(tfidf_vectorizer, "../models/tfidf_vectorizer.pkl")

print("✅ Random Forest Model Saved!")
