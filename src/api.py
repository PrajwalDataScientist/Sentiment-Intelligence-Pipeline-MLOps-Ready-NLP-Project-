from fastapi import FastAPI, HTTPException
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost
# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "data/models/xgboost_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "data/features/vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run `dvc repro`.")
if not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("Vectorizer not found. Run `dvc repro`.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ---------------- NLP ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

label_map = {
    0: "sadness",
    1: "neutral",
    2: "happiness"
}

# ---------------- APP ----------------
app = FastAPI(title="Tweet Emotion API", version="1.0")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text.strip()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    cleaned = clean_text(text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid text after preprocessing")

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    return {
        "input": text,
        "prediction": label_map[int(prediction)]
    }
