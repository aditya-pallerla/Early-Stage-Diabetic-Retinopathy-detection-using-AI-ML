# scripts/ablation_image_vs_tab.py
import argparse
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.multimodal_v2 import MultiModalNetV2

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model(checkpoint, device):
    model = MultiModalNetV2(tab_in=5, num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            raise
    model.eval()
    return model

def str2tab(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 5:
        raise ValueError("Tabular must have 5 comma-separated values: HbA1c,fasting_glucose,cholesterol,duration_years,age")
    return np.array(parts, dtype=float).reshape(1,-1)

def run_single(img_path, model, device, scaler, tab_raw=None):
    # load image
    img = Image.open(img_path).convert("RGB")
    img_t = VAL_TRANSFORM(img).unsqueeze(0).to(device)

    # prepare tab: neutral = scaler.mean_ transformed (same scaling used at train)
    neutral_raw = getattr(scaler, "mean_", None)
    if neutral_raw is None:
        # fallback: zeros
        neutral_scaled = np.zeros((1,5), dtype=float)
    else:
        neutral_scaled = scaler.transform(np.array(neutral_raw).reshape(1,-1))

    # prepare provided tab if present
    if tab_raw is not None:
        provided_scaled = scaler.transform(tab_raw.reshape(1,-1))
    else:
        provided_scaled = None

    results = []

    with torch.no_grad():
        # image-only (neutral tab)
        tab_tensor = torch.tensor(neutral_scaled, dtype=torch.float32).to(device)
        class_out, reg_out = model(img_t, tab_tensor)
        probs = torch.softmax(class_out, dim=1).squeeze().cpu().numpy()
        reg = float(reg_out.squeeze().cpu().numpy())
        results.append(("image_only", probs[0], probs[1], reg))

        # provided tab (if any)
        if provided_scaled is not None:
            tab_tensor = torch.tensor(provided_scaled, dtype=torch.float32).to(device)
            class_out, reg_out = model(img_t, tab_tensor)
            probs = torch.softmax(class_out, dim=1).squeeze().cpu().numpy()
            reg = float(reg_out.squeeze().cpu().numpy())
            results.append(("with_tab", probs[0], probs[1], reg))

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="one or more image file paths")
    parser.add_argument("--checkpoint", type=str, default="outputs/multimodal_best.pth")
    parser.add_argument("--scaler", type=str, default="models/clinical_scaler.pkl")
    parser.add_argument("--tab", type=str, default=None,
                        help="optional comma-separated tabular values to test: HbA1c,fasting_glucose,cholesterol,duration_years,age")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_model(args.checkpoint, device)
    scaler = joblib.load(args.scaler)

    tab_raw = None
    if args.tab:
        tab_raw = str2tab(args.tab)

    for img in args.images:
        imgp = Path(img)
        if not imgp.exists():
            print(f"Image not found: {img}")
            continue
        print("\n---")
        print("Image:", img)
        results = run_single(str(imgp), model, device, scaler, tab_raw=tab_raw)
        for mode, prob0, prob1, reg in results:
            pred = 1 if prob1 > prob0 else 0
            print(f"{mode:10s} | pred={pred}  prob_stage0={prob0:.4f} prob_stage1={prob1:.4f}  reg_pred={reg:.2f}")

if __name__ == "__main__":
    main()
