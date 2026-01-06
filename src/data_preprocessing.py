import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

os.makedirs("data/processed", exist_ok=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

def preprocess():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")

    train["content"] = train["content"].apply(clean_text)
    test["content"] = test["content"].apply(clean_text)

    # ✅ HANDLE EMPTY / NaN TEXT
    train["content"] = train["content"].replace("", pd.NA)
    test["content"] = test["content"].replace("", pd.NA)

    train.dropna(subset=["content"], inplace=True)
    test.dropna(subset=["content"], inplace=True)

    train.to_csv("data/processed/train_processed.csv", index=False)
    test.to_csv("data/processed/test_processed.csv", index=False)

if __name__ == "__main__":
    preprocess()