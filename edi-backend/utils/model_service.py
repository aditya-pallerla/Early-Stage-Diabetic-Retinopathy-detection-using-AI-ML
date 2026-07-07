# edi-backend/utils/model_service.py (FINAL — WITH CALIBRATION)

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import joblib
import io
import os
import pandas as pd
import datetime
from typing import Dict, Any
from .gradcam_service import get_gradcam_base64
from .shap_service import get_shap_values

# ============================================================================  
# CONFIGURATION  
# ============================================================================  
IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_image_model.pth')
TAB_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tab_model.pkl')
FUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fusion_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tabular_scaler.pkl')

# NEW — CALIBRATORS
IMG_CALIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'img_platt.pkl')
TAB_CALIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tab_platt.pkl')

DEVICE = torch.device("cpu")
IMAGE_SIZE = (512, 512)
TABULAR_FEATURES = ["HbA1c", "fasting_glucose", "cholesterol", "duration_years", "age"]

import timm

# ============================================================================  
# IMAGE MODEL DEFINITION  
# ============================================================================  
class ImageOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        self.classifier = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        f = self.backbone(x)
        out = self.classifier(f)
        return out

# ============================================================================  
# LOAD MODELS  
# ============================================================================  
def load_assets():
    try:
        scaler = joblib.load(SCALER_PATH)

        image_model = ImageOnlyNet().to(DEVICE)
        image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
        image_model.eval()

        tab_model = joblib.load(TAB_MODEL_PATH)
        fusion_model = joblib.load(FUSION_MODEL_PATH)

        # NEW — LOAD CALIBRATORS  
        img_calib = joblib.load(IMG_CALIB_PATH) if os.path.exists(IMG_CALIB_PATH) else None
        tab_calib = joblib.load(TAB_CALIB_PATH) if os.path.exists(TAB_CALIB_PATH) else None

        print("✅ All models loaded successfully!")
        return image_model, tab_model, fusion_model, scaler, img_calib, tab_calib

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None, None, None


IMAGE_MODEL, TAB_MODEL, FUSION_MODEL, SCALER, IMG_CALIB, TAB_CALIB = load_assets()

# ============================================================================  
# ENTROPY  
# ============================================================================  
def calculate_entropy(prob):
    prob = np.clip(prob, 1e-8, 1 - 1e-8)
    return -(prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))

# ============================================================================  
# MONTHS  
# ============================================================================  
def calculate_months_to_progression(clinical_data_raw, pred_class, risk_percent):
    hba1c = clinical_data_raw['HbA1c']
    glucose = clinical_data_raw['fasting_glucose']
    duration = clinical_data_raw['duration_years']

    base_months = 36 if pred_class == 0 else 12
    risk_multiplier = 1.0

    if hba1c < 6.5:
        risk_multiplier *= 1.3
    elif hba1c > 8.0:
        risk_multiplier *= 0.6

    if glucose < 126:
        risk_multiplier *= 1.2
    elif glucose > 180:
        risk_multiplier *= 0.7

    if duration > 10:
        risk_multiplier *= 0.8
    elif duration < 3:
        risk_multiplier *= 1.2

    months = int(round(base_months * risk_multiplier))
    return max(18, min(48, months)) if pred_class == 0 else max(3, min(18, months))


# ============================================================================  
# RECOMMENDATIONS  
# ============================================================================  
def get_detailed_recommendation(pred_class, risk_percent, months_display, hba1c):

    if pred_class == 0:
        if risk_percent < 30:
            return (
                f"✅ No diabetic retinopathy detected. Risk {risk_percent}%. "
                f"Next check in {months_display} months."
            )
        elif risk_percent < 60:
            return (
                f"⚠️ Moderate risk ({risk_percent}%). "
                f"Follow-up in {months_display} months. Improve HbA1c ({hba1c}%)."
            )
        else:
            return (
                f"⚠️ HIGH risk ({risk_percent}%). "
                f"Return in {months_display} months. HbA1c {hba1c}% requires attention."
            )

    else:
        if risk_percent < 60:
            return (
                f"⚠️ Mild DR detected. Risk {risk_percent}%. "
                f"Follow-up in {months_display} months."
            )
        elif risk_percent < 80:
            return (
                f"🚨 Mild DR (higher risk {risk_percent}%). "
                f"Retina specialist recommended."
            )
        else:
            return (
                f"🚨 URGENT: Stage 1 with very high risk ({risk_percent}%). "
                f"Immediate specialist and HbA1c control recommended."
            )

# ============================================================================  
# HISTORY  
# ============================================================================  
def save_history(record: Dict[str, Any]):
    HISTORY_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'history.csv')
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    df_new = pd.DataFrame([record])

    if os.path.exists(HISTORY_PATH) and os.path.getsize(HISTORY_PATH) > 0:
        df_old = pd.read_csv(HISTORY_PATH)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(HISTORY_PATH, index=False)
    else:
        df_new.to_csv(HISTORY_PATH, index=False)

    print("✅ Saved to history")

# ============================================================================  
# MAIN PREDICTION  
# ============================================================================  
def run_full_prediction(image_bytes: bytes, clinical_data: Dict[str, str]) -> Dict[str, Any]:

    if IMAGE_MODEL is None:
        raise RuntimeError("Models not loaded")

    # ---------------- TABULAR ----------------
    tab_raw = [float(clinical_data.get(k, 0)) for k in TABULAR_FEATURES]
    tab_scaled = SCALER.transform([tab_raw])

    # ---------------- IMAGE ----------------
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        img_out = IMAGE_MODEL(img_tensor)
        img_probs = torch.softmax(img_out, dim=1).numpy()[0]
        p_img = float(img_probs[1])

    # APPLY IMAGE CALIBRATION  
    if IMG_CALIB is not None:
        p_img = float(IMG_CALIB.predict_proba([[p_img]])[0][1])

    # ---------------- TABULAR PROB ----------------
    p_tab = float(TAB_MODEL.predict_proba(tab_scaled)[0][1])

    # APPLY TABULAR CALIBRATION  
    if TAB_CALIB is not None:
        p_tab = float(TAB_CALIB.predict_proba([[p_tab]])[0][1])

    # ---------------- ENTROPIES ----------------
    e_img = calculate_entropy(p_img)
    e_tab = calculate_entropy(p_tab)

    # ---------------- FUSION ----------------
    fusion_in = np.array([[p_img, p_tab, e_img, e_tab]])
    p_fusion = float(FUSION_MODEL.predict_proba(fusion_in)[0][1])

    # ---------------- FINAL STAGE ----------------
    pred_class = 1 if p_fusion >= 0.5 else 0
    risk_percent = int(round(p_fusion * 100))

    # ---------------- CLINICAL INFO ----------------
    clinical_raw = {
        "HbA1c": tab_raw[0],
        "fasting_glucose": tab_raw[1],
        "cholesterol": tab_raw[2],
        "duration_years": tab_raw[3],
        "age": tab_raw[4],
    }

    months_display = calculate_months_to_progression(clinical_raw, pred_class, risk_percent)

    result_label = "Stage 1 (Mild DR)" if pred_class == 1 else "Stage 0 (No DR)"
    recommendation = get_detailed_recommendation(pred_class, risk_percent, months_display, clinical_raw["HbA1c"])

    # ---------------- EXPLAINABILITY ----------------
    gradcam_url = get_gradcam_base64(
        IMAGE_MODEL, image_bytes, DEVICE, img_size=IMAGE_SIZE
    )

    shap_scores = get_shap_values(
        TAB_MODEL, tab_scaled, TABULAR_FEATURES
    )

    # ---------------- SAVE HISTORY ----------------
    record = {
        "date": datetime.datetime.now().isoformat(),
        "prediction": result_label,
        "risk_percent": risk_percent,
        "months_to_prog": months_display,
        "recommendation": recommendation,
        "p_img": round(p_img * 100, 1),
        "p_tab": round(p_tab * 100, 1),
        "p_fusion": round(p_fusion * 100, 1),
        "gradcam_preview": gradcam_url,
        "shap_scores": str(shap_scores),
        **clinical_raw,
    }

    save_history(record)

    # ---------------- RETURN ----------------
    return {
        "prediction": result_label,
        "risk": risk_percent,
        "months_to_prog": months_display,
        "recommendation": recommendation,
        "shap_scores": shap_scores,
        "gradcam_base64_url": gradcam_url,

        # useful visualization
        "image_model_says": f"{p_img * 100:.1f}%",
        "clinical_model_says": f"{p_tab * 100:.1f}%",
        "final_fusion_risk": f"{p_fusion * 100:.1f}%",

        "confidence": f"{p_fusion * 100:.1f}%"
    }
