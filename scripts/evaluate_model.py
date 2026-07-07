import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix

# Make sure we can import your model
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.multimodal_v2 import MultiModalNetV2
import joblib

# ---------------------------
# Minimal Dataset
# ---------------------------
class EvalDataset(Dataset):
    def __init__(self, csv_path, scaler_path):
        self.df = pd.read_csv(csv_path)

        self.img_root = os.path.dirname(csv_path)
        self.tab_cols = ["HbA1c", "fasting_glucose", "cholesterol", "duration_years", "age"]

        self.scaler = joblib.load(scaler_path)
        self.tabular = self.scaler.transform(self.df[self.tab_cols].values)

        self.labels = self.df["label"].values

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(self.img_root) / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        tab = torch.tensor(self.tabular[idx], dtype=torch.float32)
        label = int(self.labels[idx])

        return img, tab, label


# ---------------------------
# Evaluation
# ---------------------------
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n⭐ Using device: {device}")

    # Dataset + loader
    test_ds = EvalDataset(args.test_csv, args.scaler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = MultiModalNetV2(tab_in=5, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("\n🔍 Running inference on test set...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, tabs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            class_out, _ = model(imgs, tabs)
            preds = torch.argmax(torch.softmax(class_out, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n======================")
    print("📊 FINAL TEST METRICS")
    print("======================")
    print(f"✔ Accuracy: {acc:.4f}")
    print("\n✔ Confusion Matrix:")
    print(cm)
    print("======================\n")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, default="data_filtered/test.csv")
    parser.add_argument("--scaler", type=str, default="models/clinical_scaler.pkl")
    parser.add_argument("--checkpoint", type=str, default="outputs/multimodal_best.pth")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    run_eval(args)
