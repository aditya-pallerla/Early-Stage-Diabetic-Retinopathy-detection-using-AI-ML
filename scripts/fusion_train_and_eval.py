#!/usr/bin/env python3
"""
Fusion dataset creation, meta-learner training, and evaluation (Steps 10-12).

Usage:
    python scripts/fusion_train_and_eval.py
Outputs written to: outputs/
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, recall_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_DIR = "data_filtered"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Prob filenames to try in order (prefer calibrated)
IMG_VAL_FILES = [
    os.path.join(OUT_DIR, "img_probs_val_calibrated.npy"),
    os.path.join(OUT_DIR, "img_probs_val.npy")
]
IMG_TEST_FILES = [
    os.path.join(OUT_DIR, "img_probs_test_calibrated.npy"),
    os.path.join(OUT_DIR, "img_probs_test.npy")
]

TAB_VAL_FILES = [
    os.path.join(OUT_DIR, "tab_probs_val_calibrated.npy"),
    os.path.join(OUT_DIR, "tab_probs_val.npy")
]
TAB_TEST_FILES = [
    os.path.join(OUT_DIR, "tab_probs_test_calibrated.npy"),
    os.path.join(OUT_DIR, "tab_probs_test.npy")
]

# ---------- Helpers ----------
def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_probs(preferred_paths, name):
    pfile = find_first_existing(preferred_paths)
    if not pfile:
        raise FileNotFoundError(f"Could not find probability file for {name}. Tried: {preferred_paths}")
    probs = np.load(pfile, allow_pickle=True)
    probs = np.asarray(probs).ravel()
    return probs, pfile

def entropy_from_prob(p):
    # p is probability of positive class (stage1). Binary entropy.
    p = np.clip(p, 1e-8, 1 - 1e-8)
    e = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return e

# ---------- Step 10: Create fusion_train.csv ----------
print("🔎 Loading val/test CSVs ...")
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("🔎 Loading probabilities (prefers calibrated)...")
img_val_probs, img_val_src = load_probs(IMG_VAL_FILES, "img_val")
tab_val_probs, tab_val_src = load_probs(TAB_VAL_FILES, "tab_val")
img_test_probs, img_test_src = load_probs(IMG_TEST_FILES, "img_test")
tab_test_probs, tab_test_src = load_probs(TAB_TEST_FILES, "tab_test")

print("   img_val from:", img_val_src)
print("   tab_val from:", tab_val_src)
print("   img_test from:", img_test_src)
print("   tab_test from:", tab_test_src)

# Check lengths match CSVs
if len(img_val_probs) != len(val_df) or len(tab_val_probs) != len(val_df):
    raise ValueError("Validation probabilities length mismatch with val.csv")
if len(img_test_probs) != len(test_df) or len(tab_test_probs) != len(test_df):
    raise ValueError("Test probabilities length mismatch with test.csv")

# Build fusion train (validation set used to train meta-learner)
fusion_val = pd.DataFrame({
    "p_img": img_val_probs,
    "p_tab": tab_val_probs,
    "e_img": entropy_from_prob(img_val_probs),
    "e_tab": entropy_from_prob(tab_val_probs),
    "label": val_df["label"].values
})
fusion_val.to_csv(os.path.join(OUT_DIR, "fusion_train.csv"), index=False)
print("✅ Saved fusion_train.csv ->", os.path.join(OUT_DIR, "fusion_train.csv"))

# Also save fusion test features for final evaluation
fusion_test = pd.DataFrame({
    "p_img": img_test_probs,
    "p_tab": tab_test_probs,
    "e_img": entropy_from_prob(img_test_probs),
    "e_tab": entropy_from_prob(tab_test_probs),
    "label": test_df["label"].values
})
fusion_test.to_csv(os.path.join(OUT_DIR, "fusion_test.csv"), index=False)
print("✅ Saved fusion_test.csv ->", os.path.join(OUT_DIR, "fusion_test.csv"))

# ---------- Step 11: Train meta-learner ----------
X = fusion_val[["p_img", "p_tab", "e_img", "e_tab"]].values
y = fusion_val["label"].values

print("\n⚙️ Training meta-learner (Logistic Regression) ...")
# Use stratified CV to check generalization
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=2000, solver="lbfgs")

cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print("  CV AUC scores:", np.round(cv_scores, 4))
print("  CV AUC mean:", float(np.mean(cv_scores)))

# Fit on entire fusion_val
clf.fit(X, y)
joblib.dump(clf, os.path.join(OUT_DIR, "fusion_model.pkl"))
print("✅ Saved fusion_model.pkl")

# ---------- Step 12: Validate fusion on test ----------
print("\n🔬 Evaluating image-only, tabular-only and fusion on TEST set ...")
# Image-only metrics (use test probs)
y_true = fusion_test["label"].values
img_probs = fusion_test["p_img"].values
tab_probs = fusion_test["p_tab"].values

# Image-only predictions
img_preds = (img_probs >= 0.5).astype(int)
tab_preds = (tab_probs >= 0.5).astype(int)

# Fusion predictions (probability from meta-learner)
X_test = fusion_test[["p_img", "p_tab", "e_img", "e_tab"]].values
fusion_probs = clf.predict_proba(X_test)[:, 1]
fusion_preds = (fusion_probs >= 0.5).astype(int)

# Metrics helper
def summarize(name, y_true, probs, preds):
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    recall1 = recall_score(y_true, preds, pos_label=1)
    cm = confusion_matrix(y_true, preds)
    return {"name": name, "accuracy": acc, "auc": auc, "recall_stage1": recall1, "cm": cm}

img_met = summarize("Image-only", y_true, img_probs, img_preds)
tab_met = summarize("Tabular-only", y_true, tab_probs, tab_preds)
fus_met = summarize("Fusion", y_true, fusion_probs, fusion_preds)

# Print comparison
print("\n--- Comparison (TEST) ---")
for m in [img_met, tab_met, fus_met]:
    print(f"{m['name']}: ACC={m['accuracy']:.4f} | AUC={m['auc']:.4f} | Recall_stage1={m['recall_stage1']:.4f}")

# Save detailed test CSV
test_out_df = pd.DataFrame({
    "label": y_true,
    "p_img": img_probs,
    "p_tab": tab_probs,
    "p_fusion": fusion_probs,
    "pred_img": img_preds,
    "pred_tab": tab_preds,
    "pred_fusion": fusion_preds
})
test_out_df.to_csv(os.path.join(OUT_DIR, "fusion_test_results.csv"), index=False)
print("✅ Saved fusion_test_results.csv")

# Save metrics file
with open(os.path.join(OUT_DIR, "fusion_metrics.txt"), "w") as f:
    f.write("Comparison (TEST):\n")
    for m in [img_met, tab_met, fus_met]:
        f.write(f"{m['name']}: ACC={m['accuracy']:.4f} | AUC={m['auc']:.4f} | Recall_stage1={m['recall_stage1']:.4f}\n")
    f.write("\nConfusion matrix (fusion):\n")
    f.write(np.array2string(fus_met["cm"]))
print("✅ Saved fusion_metrics.txt")

# Save confusion matrix image for fusion
cm = fus_met["cm"]
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Fusion Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1], ["Stage0","Stage1"])
plt.yticks([0,1], ["Stage0","Stage1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_confusion_matrix_test.png"), dpi=150)
plt.close()
print("✅ Saved fusion_confusion_matrix_test.png")

print("\n🎉 Fusion training & evaluation complete. Outputs in", OUT_DIR)
