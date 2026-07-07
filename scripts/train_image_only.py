"""
Image-Only DR Classification Training Script (Plan B - Final Version)
--------------------------------------------------------------------
Model:
    - EfficientNet-B4 (pretrained, num_classes=0)
    - Custom classifier: Linear(1792 → 2)

Features:
    ✓ Strong color jitter (fixes orange, washed-out images)
    ✓ Balanced sampler
    ✓ CrossEntropyLoss
    ✓ AdamW + ReduceLROnPlateau
    ✓ Early stopping
    ✓ Saves:
         - best_image_model.pth
         - img_probs_val.npy
         - img_probs_test.npy
         - image_val_results.csv
         - image_test_results.csv
         - confusion_matrix_test.png

Run:
    python scripts/train_image_only.py --epochs 20 --batch_size 16
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


# =====================================================================
#                         DATASET (IMAGE ONLY)
# =====================================================================
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_file, img_root, use_aug=True):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.use_aug = use_aug

        self.paths = self.data["image_path"].values
        self.labels = self.data["label"].values

        # Strong augmentations + COLOR JITTER FIX
        if use_aug:
            self.albu = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),

                # FIX bright/orange/washed-out images
                A.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.05,
                    p=0.8
                ),

                A.RandomBrightnessContrast(
                    p=0.7, brightness_limit=0.2, contrast_limit=0.2
                ),

                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.25),

                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.val_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.img_root, self.paths[idx])
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        if self.use_aug:
            img = self.albu(image=img_np)["image"]
        else:
            img = self.val_transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# =====================================================================
#                        MODEL (EffNet-B4 + Head)
# =====================================================================
class ImageOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,         # <— We extract features only
            global_pool="avg"
        )
        self.classifier = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        f = self.backbone(x)
        out = self.classifier(f)
        return out


# =====================================================================
#                          TRAINING LOOP
# =====================================================================
def train_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================ DATA ============================
    train_ds = ImageOnlyDataset(args.train_csv, args.img_root, use_aug=True)
    val_ds   = ImageOnlyDataset(args.val_csv, args.img_root, use_aug=False)
    test_ds  = ImageOnlyDataset(args.test_csv, args.img_root, use_aug=False)

    print("\nLoaded dataset:")
    print("Train:", len(train_ds))
    print("Val:  ", len(val_ds))
    print("Test: ", len(test_ds))

    # Balanced sampler
    class_counts = np.bincount(train_ds.labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_ds.labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4)

    # ============================ MODEL ============================
    model = ImageOnlyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_auc = 0
    patience = 6
    no_improve = 0

    print("\n🚀 Starting training...")

    # ============================ TRAINING ============================
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # ============================ VALIDATION ============================
        model.eval()
        val_labels = []
        val_probs = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                probs = torch.softmax(out, dim=1)[:, 1]

                val_labels.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(out.argmax(1).cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")

        scheduler.step(val_auc)

        # Save validation probabilities
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/img_probs_val.npy", np.array(val_probs))

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), "outputs/best_image_model.pth")
            print("✔ Saved BEST model!")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{patience}")

        if no_improve >= patience:
            print("⏹ Early stopping!")
            break

    # =====================================================================
    #                            TEST EVALUATION
    # =====================================================================
    print("\nEvaluating on TEST set...")
    model.load_state_dict(torch.load("outputs/best_image_model.pth"))
    model.eval()

    test_labels, test_probs, test_preds = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            out = model(imgs)

            probs = torch.softmax(out, dim=1)[:, 1]
            pred = out.argmax(1).cpu().numpy()

            test_labels.extend(labels.numpy())
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(pred)

    # Save test probabilities
    np.save("outputs/img_probs_test.npy", np.array(test_probs))

    # ============================ CONFUSION MATRIX ============================
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_test.png", dpi=120)

    # ============================ CSV SAVE ============================
    val_df = pd.DataFrame({
        "label": val_labels,
        "prob_stage1": np.load("outputs/img_probs_val.npy")
    })
    val_df.to_csv("outputs/image_val_results.csv", index=False)

    test_df = pd.DataFrame({
        "label": test_labels,
        "prob_stage1": test_probs,
        "predicted": test_preds
    })
    test_df.to_csv("outputs/image_test_results.csv", index=False)

    print("\n✅ All outputs saved in /outputs/")
    print("DONE!")


# =====================================================================
#                                 MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="data_filtered/train.csv")
    parser.add_argument("--val_csv", type=str, default="data_filtered/val.csv")
    parser.add_argument("--test_csv", type=str, default="data_filtered/test.csv")
    parser.add_argument("--img_root", type=str, default="data_filtered")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train_model(args)
