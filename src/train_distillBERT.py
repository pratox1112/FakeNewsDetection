import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("../data/processed_data.csv")

X_train, X_test, y_train, y_test = train_test_split(df["combined_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, max_length=512, return_tensors="pt")
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, max_length=512, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
train_loader = DataLoader(TensorDataset(X_train_tokens["input_ids"], torch.tensor(y_train.values)), batch_size=16, shuffle=True)

model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Completed!")

torch.save(model.state_dict(), "../models/fake_news_distilbert_model.pth")
joblib.dump(tokenizer, "../models/distilbert_tokenizer.pkl")

print("DistilBERT Model Saved!")
