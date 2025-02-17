import pandas as pd

df = pd.read_csv("../data/WELFake_Dataset.csv")

df = df.dropna(subset=["title", "text"])

df["combined_text"] = df["title"] + " " + df["text"]

df.to_csv("../data/processed_data.csv", index=False)

print("✅ Data Preprocessing Completed & Saved!")
