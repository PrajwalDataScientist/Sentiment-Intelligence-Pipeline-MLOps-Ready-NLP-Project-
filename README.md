
# 🧠 Sentiment Intelligence Pipeline

**Author:** Prajwal (Data Scientist)  
**Role Target:** Data Scientist   

---

## 📌 Project Overview

This project is an **end-to-end NLP-based Sentiment Analysis system**

The goal of the project is not just to train a model, but to demonstrate:
- A **reproducible ETL pipeline**
- **Feature engineering & model lifecycle management**
- **Experiment reproducibility with DVC**
- **Production-ready inference using FastAPI**


The system classifies tweets into three sentiment classes:
- **Sadness (0)**
- **Neutral (1)**
- **Happiness (2)**

---

## 🏗️ Architecture & Workflow

```
Raw Data
   ↓
Data Ingestion (DVC Stage)
   ↓
Data Preprocessing (Text Cleaning, NaN Handling)
   ↓
Feature Engineering (Bag of Words)
   ↓
Model Training (XGBoost)
   ↓
Model Evaluation (Metrics + Confusion Matrix)
   ↓
FastAPI Inference API
```

All steps are orchestrated using **DVC**, ensuring full reproducibility.

---

## 📂 Project Structure

```
ml-project/
│
├── data/
│   ├── raw/              # Train/Test raw data
│   ├── processed/        # Cleaned datasets
│   ├── features/         # Feature matrices & vectorizer
│   ├── models/           # Trained ML model
│   └── reports/          # Evaluation metrics
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── api.py
│
├── dvc.yaml              # DVC pipeline definition
├── dvc.lock              # Pipeline lock file
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies & Skills Used

### 🧑‍💻 Programming & Data
- Python
- Pandas, NumPy
- Regular Expressions (Regex)

### 🤖 Machine Learning
- Scikit-learn
- XGBoost
- Feature Engineering (Bag of Words)
- Model Evaluation (Accuracy, Precision, Recall, F1-score)

### 📊 NLP
- Tokenization
- Stopword Removal
- Lemmatization
- Text Normalization


### 🌐 Deployment
- FastAPI


---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```

### 3️⃣ Initialize & Run DVC Pipeline

```bash
dvc init
dvc repro
```

This will:
- Generate processed data
- Train the model
- Save evaluation metrics

---

## 🚀 Run FastAPI Inference Server

```bash
python -m uvicorn src.api:app --reload
```

Open browser:
```
http://127.0.0.1:8000/docs
```

---

## 📈 Output Artifacts

- **Model:** `data/models/xgboost_model.pkl`
- **Vectorizer:** `data/features/vectorizer.pkl`
- **Metrics:** `data/reports/metrics.json`

---



---
## 📸 Project Output

<p align="center">
  <img src="./images/Screenshot%202026-01-06%20135437.png" width="700"/>
</p>

<p align="center">
  <img src="./images/Screenshot%202026-01-06%20135521.png" width="700"/>
</p>

<p align="center">
  <img src="./images/Screenshot%202026-01-06%20135554.png" width="700"/>
</p>

<p align="center">
  <img src="./images/Screenshot%202026-01-06%20145513.png" width="700"/>
</p>

## 🎯 Conclusion

This project demonstrates my ability to design, build, and deploy **production-ready machine learning systems** using modern **Data Science and MLOps workflows**.



---

### 👤 Author

**Prajwal**  
_Data Scientist_  
Passionate about building scalable ML systems and production-ready AI solutions.
