"""
Plot ResNet18 and ResNet50 training curves for the paper (Figure 4).

Loads the per-epoch train/val L1 loss from the saved run logs and saves each
model as a separate panel so they can be placed as subfigures (a)/(b) in LaTeX.

Output: figures/output/training_curves_a.pdf/.png  (ResNet18)
        figures/output/training_curves_b.pdf/.png  (ResNet50)
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

OUT_DIR = Path("figures/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_ax(ax, title):
    ax.set_title(title, fontsize=FS + 0.5, fontweight="normal")
    ax.set_xlabel("Epoch", fontsize=FS)
    ax.set_ylabel("Mean loss", fontsize=FS)
    ax.tick_params(labelsize=FS - 0.5)
    ax.legend(fontsize=FS - 0.5, loc="upper right")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


panels = [
    ("training_curves_a", "ResNet18 (augmented)", e18, tr18, vl18),
    ("training_curves_b", "ResNet50 (augmented)", e50, tr50, vl50),
]

for name, title, epochs, train, val in panels:
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(epochs, train, color=TRAIN_COLOR, lw=1.1, label="train")
    ax.plot(epochs, val,   color=VAL_COLOR,   lw=1.1, label="val")
    style_ax(ax, title)
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.pdf / .png")
