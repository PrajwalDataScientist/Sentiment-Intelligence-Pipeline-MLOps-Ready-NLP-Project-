import os
import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
import joblib

# ✅ Create output directory
os.makedirs("data/models", exist_ok=True)

def train_model():
    # ✅ Load features (MATCH feature_engineering outputs)
    X_train = sparse.load_npz("data/features/X_train.npz")
    y_train = np.load("data/features/y_train.npy")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ✅ Save model
    joblib.dump(model, "data/models/xgboost_model.pkl")

if __name__ == "__main__":
    train_model()
