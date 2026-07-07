"""
Generate Clinical Scaler from Training Data
Run this ONCE before training to create the scaler file.

Usage:
    python scripts/generate_scaler.py
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Configuration
TRAIN_CSV = "data_filtered/train.csv"
OUTPUT_PATH = "models/clinical_scaler.pkl"
TABULAR_COLS = ["HbA1c", "fasting_glucose", "cholesterol", "duration_years", "age"]

print("=" * 70)
print("GENERATING CLINICAL SCALER")
print("=" * 70)

# 1. Load training data
print(f"\n📂 Loading training data from: {TRAIN_CSV}")
train_df = pd.read_csv(TRAIN_CSV)
print(f"✅ Loaded {len(train_df):,} training samples")

# 2. Extract tabular features
print(f"\n🔢 Extracting tabular features: {TABULAR_COLS}")
tabular_data = train_df[TABULAR_COLS].values

# Check for missing values
if pd.isnull(tabular_data).any():
    print("⚠️ WARNING: Missing values detected in tabular data!")
    print(train_df[TABULAR_COLS].isnull().sum())
    print("Filling missing values with column means...")
    tabular_data = train_df[TABULAR_COLS].fillna(train_df[TABULAR_COLS].mean()).values

# 3. Fit the scaler on TRAIN data only (critical to avoid data leakage)
print("\n📊 Fitting StandardScaler on training data...")
scaler = StandardScaler()
scaler.fit(tabular_data)

# 4. Display scaling parameters
print("\n✅ Scaler fitted successfully!")
print("\nScaling parameters:")
for i, col in enumerate(TABULAR_COLS):
    print(f"   {col}:")
    print(f"      Mean: {scaler.mean_[i]:.2f}")
    print(f"      Std:  {scaler.scale_[i]:.2f}")

# 5. Save the scaler
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
joblib.dump(scaler, OUTPUT_PATH)
print(f"\n💾 Scaler saved to: {OUTPUT_PATH}")

# 6. Verification test
print("\n🧪 Testing scaler with sample data...")
sample = train_df[TABULAR_COLS].iloc[0:1].values
scaled = scaler.transform(sample)
print(f"   Original: {sample[0]}")
print(f"   Scaled:   {scaled[0]}")

print("\n" + "=" * 70)
print("✅ SCALER GENERATION COMPLETE!")
print("   You can now run training with this scaler.")
print("=" * 70)