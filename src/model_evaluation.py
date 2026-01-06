import os
import json
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ✅ Create output directory
os.makedirs("data/reports", exist_ok=True)

def evaluate_model():
    # ✅ Load model (MATCH model_building.py)
    model = joblib.load("data/models/xgboost_model.pkl")

    # ✅ Load test data
    X_test = sparse.load_npz("data/features/X_test.npz")
    y_test = np.load("data/features/y_test.npy")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Save metrics
    with open("data/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model()
