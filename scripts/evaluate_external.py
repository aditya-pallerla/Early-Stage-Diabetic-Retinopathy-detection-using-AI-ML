# scripts/evaluate_external.py
"""
External evaluation for EDI project:
- Loads saved image-only model (EffNet-B4 + custom classifier)
- Loads tabular model
- Loads fusion model
- Applies optional Platt calibration
- Runs all 3 models on external dataset
- Saves CSV and metrics

Run:
    python scripts/evaluate_external.py --csv external_dataset/external.csv --img_root external_dataset
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  IMAGE MODEL (MATCHING TRAINING EXACTLY)
# ============================================================
class ImageOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,     # always False when loading weights
            num_classes=0,
            global_pool="avg"
        )
        self.classifier = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        f = self.backbone(x)
        out = self.classifier(f)
        return out


# ============================================================
#  HELPERS
# ============================================================
def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def entropy(p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def safe_platt(platt, probs):
    if platt is None:
        return probs
    try:
        X = probs.reshape(-1,1)
        return platt.predict_proba(X)[:,1]
    except:
        return probs


# ============================================================
#  MAIN
# ============================================================
def main(args):

    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(args.csv)

    if "label" not in df.columns or "image_path" not in df.columns:
        raise Exception("CSV must contain 'image_path' and 'label' columns.")

    has_tabular = all(c in df.columns for c in ["HbA1c","fasting_glucose","cholesterol","duration_years","age"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================================================
    # LOAD IMAGE MODEL
    # ============================================================
    print("\nLoading Image-Only model...")

    img_model = ImageOnlyNet().to(device)

    state = torch.load("outputs/best_image_model.pth", map_location=device)

    # PURE state_dict direct load (your training saved this way)
    img_model.load_state_dict(state, strict=True)

    img_model.eval()

    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img_probs = []
    print("Running image model on external set...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            path = os.path.join(args.img_root, str(row["image_path"]))
            if not os.path.exists(path):
                raise Exception(f"Missing image: {path}")
            inp = load_image(path, transform).to(device)
            out = img_model(inp)
            p = torch.softmax(out, dim=1).cpu().numpy()[0,1]
            img_probs.append(float(p))

    img_probs = np.array(img_probs)

    # load image calibrator if exists
    img_platt = joblib.load("outputs/img_platt.pkl") if os.path.exists("outputs/img_platt.pkl") else None
    img_probs = safe_platt(img_platt, img_probs)


    # ============================================================
    # TABULAR MODEL
    # ============================================================
    if has_tabular and os.path.exists("outputs/tab_model.pkl"):
        print("\nRunning Tabular-only model...")
        tab_model = joblib.load("outputs/tab_model.pkl")

        # load scaler
        scaler = None
        if os.path.exists("outputs/tabular_scaler.pkl"):
            scaler = joblib.load("outputs/tabular_scaler.pkl")

        X_tab = df[["HbA1c","fasting_glucose","cholesterol","duration_years","age"]].fillna(0).values

        if scaler:
            X_tab = scaler.transform(X_tab)

        try:
            tab_probs = tab_model.predict_proba(X_tab)[:,1]
        except:
            tab_probs = tab_model.predict(X_tab).astype(float)

        # calibrate
        tab_platt = joblib.load("outputs/tab_platt.pkl") if os.path.exists("outputs/tab_platt.pkl") else None
        tab_probs = safe_platt(tab_platt, tab_probs)

    else:
        print("Tabular-only not available or CSV missing columns.")
        tab_probs = np.zeros(len(df))


    # ============================================================
    # FUSION MODEL
    # ============================================================
    print("\nRunning Fusion model...")

    if os.path.exists("outputs/fusion_model.pkl"):
        fusion_model = joblib.load("outputs/fusion_model.pkl")

        ent_img = np.array([entropy(x) for x in img_probs])
        ent_tab = np.array([entropy(x) for x in tab_probs])

        X_fusion = np.vstack([img_probs, tab_probs, ent_img, ent_tab]).T

        fusion_probs = fusion_model.predict_proba(X_fusion)[:,1]

    else:
        print("Fusion model not found, defaulting to (img + tab) / 2")
        fusion_probs = (img_probs + tab_probs) / 2


    # ============================================================
    # SAVE RESULTS
    # ============================================================
    labels = df["label"].values.astype(int)

    out = pd.DataFrame({
        "image_path": df["image_path"],
        "label": labels,
        "p_img": img_probs,
        "p_tab": tab_probs,
        "p_fusion": fusion_probs,
        "pred_img": (img_probs >= 0.5).astype(int),
        "pred_tab": (tab_probs >= 0.5).astype(int),
        "pred_fusion": (fusion_probs >= 0.5).astype(int)
    })

    out.to_csv("outputs/external_results.csv", index=False)
    print("Saved outputs/external_results.csv")

    # compute metrics
    def metrics(y, p):
        pred = (p >= 0.5).astype(int)
        return {
            "acc": accuracy_score(y, pred),
            "auc": roc_auc_score(y, p),
            "recall": recall_score(y, pred),
            "cm": confusion_matrix(y, pred)
        }

    m_img = metrics(labels, img_probs)
    m_tab = metrics(labels, tab_probs)
    m_fus = metrics(labels, fusion_probs)

    with open("outputs/external_metrics.txt", "w") as f:
        f.write("External Evaluation\n\n")
        def bl(name, m):
            f.write(f"{name}:\n")
            f.write(f"  ACC: {m['acc']:.4f}\n")
            f.write(f"  AUC: {m['auc']:.4f}\n")
            f.write(f"  Recall: {m['recall']:.4f}\n")
            f.write(f"  Confusion:\n{m['cm']}\n\n")

        bl("Image-only", m_img)
        bl("Tabular-only", m_tab)
        bl("Fusion", m_fus)

    print("Saved outputs/external_metrics.txt")
    print("\n🎉 External evaluation complete.\n")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="external_dataset/external.csv")
    parser.add_argument("--img_root", type=str, default="external_dataset")
    args = parser.parse_args()
    main(args)
