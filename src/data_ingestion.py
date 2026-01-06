import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/tweet_emotions.csv"
RAW_DATA_DIR = "data/raw"

def ingest_data():
    # ✅ Create output directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["tweet_id"])

    df = df[df['sentiment'].isin(['sadness', 'neutral', 'happiness'])]

    df["sentiment"] = df["sentiment"].replace({
        "sadness": 0,
        "neutral": 1,
        "happiness": 2
    }).infer_objects(copy=False)

    train, test = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["sentiment"]
    )

    train.to_csv(os.path.join(RAW_DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(RAW_DATA_DIR, "test.csv"), index=False)

if __name__ == "__main__":
    ingest_data()
