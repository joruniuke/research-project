"""
Fine-tune ResNet18/50 for liquid-line Y regression.

Expects a dataset directory with the structure:
    <dataset>/images/train/*.jpg
    <dataset>/images/val/*.jpg
    <dataset>/labels/train/*.txt
    <dataset>/labels/val/*.txt

Label format (YOLO): class cx cy w h
The liquid line Y coordinate is stored in the cy field (index 2) and represents
the line's vertical position as a fraction of the image height, in [0, 1].

The model is trained to predict this normalized Y directly from the OBB-aligned
barrel crop. All geometric alignment is handled at dataset preparation time —
the network only has to learn where the liquid line sits within an already
upright barrel image.

Saves the best checkpoint and a training log to weights/<backbone>/.

Usage:
    python train_resnet.py --data /path/to/dataset
    python train_resnet.py --data /path/to/dataset --backbone resnet50 --epochs 100
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


INPUT_SIZE = (224, 224)

# ResNet18 uses V1 weights (standard pretraining) and ResNet50 uses V2 (improved
# training recipe). These are the torchvision defaults for each model version.
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class SyringeRegressionDataset(Dataset):
    """Loads OBB-aligned barrel crop images and their liquid-line Y labels."""

    def __init__(self, dataset_dir: Path, split: str, transform=None):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            parts = lbl_path.read_text().strip().split()
            # Label format: class cx cy w h (YOLO-style).
            # Index 2 (cy) holds the normalized liquid-line Y coordinate.
            line_y = float(parts[2])
            self.samples.append((img_path, line_y))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, line_y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(line_y, dtype=torch.float32)


def build_model(backbone: str = "resnet18") -> nn.Module:
    """Load a pretrained backbone and attach a regression head.

    The original ImageNet classification head is replaced with
    Dropout(0.3) + Linear → 1. Dropout regularizes the head on our
    relatively small training set (~1700 crops).
    """
    fn, weights = BACKBONES[backbone]
    model = fn(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    return model


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path,
                        help="Dataset directory (must contain images/ and labels/ subdirs)")
    parser.add_argument("--backbone", default="resnet18", choices=list(BACKBONES))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    device = best_device()
    OUT_DIR = Path(f"weights/{args.backbone}")
    print(f"Device: {device}  Backbone: {args.backbone}")

    # Augmentation is only applied during training to prevent overfitting.
    # The transforms simulate the variation seen in the test session:
    # slight rotation (syringes held at slightly different angles), color shifts
    # (different lighting between the two recording sessions), and blur (focus variation).
    # No augmentation at validation time — we want a clean, stable loss signal.
    train_tf = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SyringeRegressionDataset(args.data, "train", train_tf)
    val_ds = SyringeRegressionDataset(args.data, "val", val_tf)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(args.backbone).to(device)
    # AdamW with small weight decay prevents the regression head from overfitting
    # to the training distribution. Cosine annealing decays the learning rate
    # smoothly over the full run, which tends to find a better final minimum
    # than step decay for fine-tuning tasks like this.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_dl:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs).squeeze(1)
                val_loss += criterion(preds, targets).item() * len(imgs)
        val_loss /= len(val_ds)
        scheduler.step()

        epochs_log.append({"epoch": epoch, "train": round(train_loss, 4), "val": round(val_loss, 4)})
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        # Save only when we beat the best validation loss — not the last epoch,
        # since the model can slightly overfit toward the end of training.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best.pt")

    # Save a complete run record for reproducibility.
    # Includes config, final metrics, and a full per-epoch log so training
    # curves can be reproduced without re-running training.
    run_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch},
        "best_val_l1": round(best_val, 4),
        "final_train_l1": round(train_loss, 4),
        "epochs_log": epochs_log,
    }
    log_path = OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(run_record, indent=2))
    print(f"\nDone. Best val L1 = {best_val:.4f}  →  {OUT_DIR / 'best.pt'}")
    print(f"Run log saved → {log_path}")


if __name__ == "__main__":
    main()
