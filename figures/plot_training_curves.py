"""
Plot ResNet18 and ResNet50 training curves for the paper (Figure 4).

Loads the per-epoch train/val L1 loss from the saved run logs and plots them
side by side. ResNet18 shows healthy generalization; ResNet50 shows a small
but consistent gap between train and val loss, indicating slight overfitting.

Output: figures/output/training_curves.pdf and training_curves.png
Run from the repo root: python figures/plot_training_curves.py
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

with open("runs/resnet18/run_v2_augmented.json") as f:
    r18 = json.load(f)

with open("runs/resnet50/run_v1_augmented.json") as f:
    r50 = json.load(f)


def unpack(log):
    """Extract epoch, train, and val arrays from a run log."""
    epochs = [e["epoch"] for e in log]
    train = [e["train"] for e in log]
    val = [e["val"] for e in log]
    return np.array(epochs), np.array(train), np.array(val)


e18, tr18, vl18 = unpack(r18["epochs_log"])
e50, tr50, vl50 = unpack(r50["epochs_log"])

# Blue for train, orange for val — consistent with common ML paper conventions.
TRAIN_COLOR = "#2563EB"
VAL_COLOR = "#F97316"
FS = 10.0  # base font size for all text

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=False)
# wspace gives a bit of breathing room between the two subplots.
fig.subplots_adjust(wspace=0.35)


def style_ax(ax, title):
    """Apply consistent styling to both subplots."""
    ax.set_title(title, fontsize=FS + 0.5, fontweight="normal")
    ax.set_xlabel("Epoch", fontsize=FS)
    ax.set_ylabel("Mean loss", fontsize=FS)
    ax.tick_params(labelsize=FS - 0.5)
    ax.legend(fontsize=FS - 0.5, loc="upper right")
    ax.grid(False)
    # Remove top and right spines for a cleaner academic look.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# Left subplot: ResNet18 (augmented)
ax = axes[0]
ax.plot(e18, tr18, color=TRAIN_COLOR, lw=1.1, label="train")
ax.plot(e18, vl18, color=VAL_COLOR, lw=1.1, label="val")
style_ax(ax, "ResNet18 (augmented)")

# Right subplot: ResNet50 (augmented)
ax = axes[1]
ax.plot(e50, tr50, color=TRAIN_COLOR, lw=1.1, label="train")
ax.plot(e50, vl50, color=VAL_COLOR, lw=1.1, label="val")
style_ax(ax, "ResNet50 (augmented)")

OUT_DIR = Path("figures/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "training_curves"
plt.savefig(str(out) + ".pdf", bbox_inches="tight")
plt.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
print(f"Saved {out}.pdf and {out}.png")
