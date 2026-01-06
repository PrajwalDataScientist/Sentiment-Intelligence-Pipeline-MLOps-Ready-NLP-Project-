import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import joblib

# ✅ Create output directory
os.makedirs("data/features", exist_ok=True)

def feature_engineering():
    train = pd.read_csv("data/processed/train_processed.csv")
    test = pd.read_csv("data/processed/test_processed.csv")

    # ✅ COLUMN SAFETY CHECK
    if "content" not in train.columns:
        raise ValueError("Column 'content' not found in train_processed.csv")

    if "content" not in test.columns:
        raise ValueError("Column 'content' not found in test_processed.csv")

    # ✅ NaN SAFETY (extra guard)
    train = train.dropna(subset=["content"])
    test = test.dropna(subset=["content"])

    X_train_text = train["content"].astype(str)
    X_test_text = test["content"].astype(str)

    y_train = train["sentiment"].values
    y_test = test["sentiment"].values

    vectorizer = CountVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # ✅ SAVE EXACTLY AS dvc.yaml EXPECTS
    sparse.save_npz("data/features/X_train.npz", X_train)
    sparse.save_npz("data/features/X_test.npz", X_test)

    np.save("data/features/y_train.npy", y_train)
    np.save("data/features/y_test.npy", y_test)

    joblib.dump(vectorizer, "data/features/vectorizer.pkl")

if __name__ == "__main__":
    feature_engineering()
