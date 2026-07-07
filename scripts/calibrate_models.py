import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ================================
# 1. Load validation labels
# ================================
val = pd.read_csv("data_filtered/val.csv")
y_val = val["label"].values

# ================================
# 2. Load model probabilities
# ================================
img_val = np.load("outputs/img_probs_val.npy")
tab_val = np.load("outputs/tab_probs_val.npy")

# If shape is (N,2), convert to column for Stage 1
if img_val.ndim == 2:
    img_val = img_val[:, 1]
if tab_val.ndim == 2:
    tab_val = tab_val[:, 1]

# ================================
# 3. Fit Platt scaling (Logistic Regression)
# ================================
def fit_platt(probs, labels):
    lr = LogisticRegression(solver="liblinear")
    lr.fit(probs.reshape(-1,1), labels)
    return lr

img_platt = fit_platt(img_val, y_val)
tab_platt = fit_platt(tab_val, y_val)

# Save the calibration models
joblib.dump(img_platt, "outputs/img_platt.pkl")
joblib.dump(tab_platt, "outputs/tab_platt.pkl")

# ================================
# 4. Produce calibrated validation probabilities
# ================================
img_val_cal = img_platt.predict_proba(img_val.reshape(-1,1))[:,1]
tab_val_cal = tab_platt.predict_proba(tab_val.reshape(-1,1))[:,1]

np.save("outputs/img_probs_val_calibrated.npy", img_val_cal)
np.save("outputs/tab_probs_val_calibrated.npy", tab_val_cal)

print("\n✅ Calibration complete for validation set!")
print("Saved calibrated files in outputs/:")
print("  - img_platt.pkl")
print("  - tab_platt.pkl")
print("  - img_probs_val_calibrated.npy")
print("  - tab_probs_val_calibrated.npy")
# ================================
# 5. Calibrate TEST probabilities
# ================================
img_test = np.load("outputs/img_probs_test.npy")
tab_test = np.load("outputs/tab_probs_test.npy")

if img_test.ndim == 2:
    img_test = img_test[:,1]
if tab_test.ndim == 2:
    tab_test = tab_test[:,1]

img_test_cal = img_platt.predict_proba(img_test.reshape(-1,1))[:,1]
tab_test_cal = tab_platt.predict_proba(tab_test.reshape(-1,1))[:,1]

np.save("outputs/img_probs_test_calibrated.npy", img_test_cal)
np.save("outputs/tab_probs_test_calibrated.npy", tab_test_cal)

print("  - img_probs_test_calibrated.npy")
print("  - tab_probs_test_calibrated.npy")
