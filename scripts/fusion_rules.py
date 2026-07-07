"""
Fusion Rules Script (Steps 13 & 14)
-----------------------------------
Input:
    - outputs/img_probs_test_calibrated.npy
    - outputs/tab_probs_test_calibrated.npy
    - outputs/fusion_test.csv  (for labels)

Output:
    - outputs/fusion_rules_results.csv
    - outputs/fusion_rules_metrics.txt
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ================================
# Load data
# ================================
img = np.load("outputs/img_probs_test_calibrated.npy")
tab = np.load("outputs/tab_probs_test_calibrated.npy")
fusion_df = pd.read_csv("outputs/fusion_test.csv")   # contains true label

labels = fusion_df["label"].values

# ================================
# RULES
# ================================

def entropy(p):
    eps = 1e-9
    return -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))

ent_img = entropy(img)
ent_tab = entropy(tab)

pred = []
uncertainty_flag = []

for i in range(len(img)):
    p_img = img[i]
    p_tab = tab[i]

    # ---- Rule 1: Strong image confidence ----
    if p_img > 0.75:
        pred.append(1)
        uncertainty_flag.append(0)
        continue
    if p_img < 0.25:
        pred.append(0)
        uncertainty_flag.append(0)
        continue

    # ---- Rule 2: Both probs near 0.5 = Uncertain ----
    if abs(p_img - p_tab) < 0.15:
        pred.append(-1)  # -1 = abstain
        uncertainty_flag.append(1)
        continue

    # ---- Rule 3: Otherwise use meta-fusion probability ----
    p_fusion = 0.5 * img[i] + 0.5 * tab[i]
    pred.append(1 if p_fusion >= 0.5 else 0)
    uncertainty_flag.append(0)


# ================================
# Convert abstains
# ================================
pred_for_metrics = [p for p in pred if p != -1]
labels_for_metrics = [labels[i] for i in range(len(pred)) if pred[i] != -1]

# ================================
# Metrics
# ================================
acc = accuracy_score(labels_for_metrics, pred_for_metrics)
auc = roc_auc_score(labels_for_metrics, pred_for_metrics)
cm = confusion_matrix(labels_for_metrics, pred_for_metrics)

n_abstain = sum(1 for p in pred if p == -1)
abstain_rate = n_abstain / len(pred)

# ================================
# Save results
# ================================
out_df = pd.DataFrame({
    "label": labels,
    "p_img": img,
    "p_tab": tab,
    "pred_rule": pred,
    "abstain": uncertainty_flag
})
out_df.to_csv("outputs/fusion_rules_results.csv", index=False)

with open("outputs/fusion_rules_metrics.txt", "w") as f:
    f.write(f"Accuracy (excluding abstain): {acc:.4f}\n")
    f.write(f"AUC (excluding abstain): {auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"Abstain Rate: {abstain_rate:.4f} ({n_abstain} samples)\n")

print("\n🎉 Fusion rules complete!")
print("Saved: fusion_rules_results.csv, fusion_rules_metrics.txt in outputs/")
