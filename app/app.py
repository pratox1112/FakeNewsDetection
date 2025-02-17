from flask import Flask, request, jsonify
import torch
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("fake_news_distilbert_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = joblib.load("distilbert_tokenizer.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()

    result = "FAKE" if prediction == 1 else "REAL"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
