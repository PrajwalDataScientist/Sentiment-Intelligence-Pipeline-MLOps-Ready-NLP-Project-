
# 🧠 Sentiment Intelligence Pipeline (MLOps-Ready NLP Project)

**Author:** Prajwal (Data Scientist)  
**Role Target:** Data Scientist / MLOps Engineer  
**Experience Level Represented:** 3+ Years  

---

## 📌 Project Overview

This project is an **end-to-end NLP-based Sentiment Analysis system** built using **industry-grade Data Science and MLOps practices**.

The goal of the project is not just to train a model, but to demonstrate:
- A **reproducible ETL pipeline**
- **Feature engineering & model lifecycle management**
- **Experiment reproducibility with DVC**
- **Production-ready inference using FastAPI**
- **Cloud readiness (AWS EC2 / SageMaker compatible)**

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

### 🔁 MLOps
- DVC (Data Version Control)
- Reproducible Pipelines
- Artifact Versioning
- Dependency Tracking

### 🌐 Deployment
- FastAPI
- Uvicorn / Gunicorn
- AWS EC2 / SageMaker Ready

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

## 💡 Key Highlights

✔ End-to-end ETL pipeline  
✔ Fully reproducible ML workflow  
✔ Real-world NLP preprocessing  
✔ Production-grade API  
✔ Cloud-ready deployment  
✔ MLOps best practices applied  

---

## 🎯 Conclusion

This project demonstrates my ability to design, build, and deploy **production-ready machine learning systems** using modern **Data Science and MLOps workflows**.

It reflects **real-world problem solving**, **pipeline debugging**, and **system-level thinking** expected from a **Data Scientist with 3+ years of experience**.

---

### 👤 Author

**Prajwal**  
_Data Scientist_  
Passionate about building scalable ML systems and production-ready AI solutions.
