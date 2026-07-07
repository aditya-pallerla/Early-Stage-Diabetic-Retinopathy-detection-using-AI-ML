"""
Tabular-Only Model for Late Fusion
-----------------------------------
Trains a LightGBM classifier using ONLY clinical features.

Saves:
    outputs/tab_model.pkl
    outputs/tab_probs_val.npy
    outputs/tab_probs_test.npy
    outputs/tabular_val_results.csv
    outputs/tabular_test_results.csv
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# ==========================================================
# CONFIG
# ==========================================================
FEATURES = ["HbA1c", "fasting_glucose", "cholesterol", "duration_years", "age"]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values
    y = df["label"].values
    return X, y


def train_tabular():
    print("\n📌 Loading datasets...")
    X_train, y_train = load_dataset("data_filtered/train.csv")
    X_val, y_val = load_dataset("data_filtered/val.csv")
    X_test, y_test = load_dataset("data_filtered/test.csv")

    # ==========================================================
    # SCALE FEATURES
    # ==========================================================
    print("📌 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(scaler, "outputs/tabular_scaler.pkl")

    # ==========================================================
    # TRAIN LIGHTGBM CLASSIFIER
    # ==========================================================
    print("\n🚀 Training LightGBM Tabular Model...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],
              eval_metric="auc")

    joblib.dump(model, "outputs/tab_model.pkl")
    print("✅ Saved tab_model.pkl!")

    # ==========================================================
    # VALIDATION EVAL
    # ==========================================================
    print("\n📌 Evaluating on VAL set...")

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    np.save("outputs/tab_probs_val.npy", val_probs)

    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"VAL ACC: {val_acc:.4f}")
    print(f"VAL AUC: {val_auc:.4f}")

    pd.DataFrame({
        "label": y_val,
        "prob_stage1": val_probs,
        "pred": val_preds
    }).to_csv("outputs/tabular_val_results.csv", index=False)

    # ==========================================================
    # TEST EVAL
    # ==========================================================
    print("\n📌 Evaluating on TEST set...")

    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    np.save("outputs/tab_probs_test.npy", test_probs)

    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"TEST ACC: {test_acc:.4f}")
    print(f"TEST AUC: {test_auc:.4f}")

    pd.DataFrame({
        "label": y_test,
        "prob_stage1": test_probs,
        "pred": test_preds
    }).to_csv("outputs/tabular_test_results.csv", index=False)

    print("\n🎉 ALL DONE — Tabular-Only model ready for fusion!")
    print("Results saved in: outputs/")


if __name__ == "__main__":
    train_tabular()
