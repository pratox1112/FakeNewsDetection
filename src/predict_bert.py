import torch
import joblib
from transformers import DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("../models/fake_news_distilbert_model.pth", map_location=device))
model.to(device).eval()

tokenizer = joblib.load("../models/distilbert_tokenizer.pkl")

def predict_fake_news(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
    return "FAKE" if prediction == 1 else "REAL"

print(predict_fake_news("Government confirms UFO sightings."))
