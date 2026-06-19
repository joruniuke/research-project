# Rubber Stopper Liquid-Level Detection

Code for the CSE3000 Research Project:
**"Automated Liquid-Line Detection in Handheld Syringe Videos: A Comparison of Edge-Based, Segmentation, and Learning-Based Methods"**

Four detection methods are implemented and evaluated on an annotated dataset of syringe videos. All methods share the same OBB-based barrel alignment preprocessing step.

---

## Methods

| Name | Description |
|------|-------------|
| `edge_scan` | Classical edge detection: dark pixel mask → Sobel gradient → morphological closing → row scan |
| `sam` | Zero-shot SAM2 prompted by a heuristic dark-blob detector |
| `yolo_sam` | SAM2 prompted by a fine-tuned YOLO11n-OBB stopper detector |
| `resnet` / `resnet50` | Fine-tuned ResNet18/50 that directly regresses the liquid-line Y coordinate |

---

## Results

Test set: 16 videos, 1894 annotated frames from a separate recording session.

| Method | Detection | Y_norm mean | Y_norm median | D_end mean | D_end median |
|--------|-----------|-------------|---------------|------------|--------------|
| OBB Edge Scan | 100% | 1.31% | **0.33%** | 19.1 px | 13.2 px |
| Blob SAM2 | 100% | 1.83% | 0.41% | 19.9 px | 13.2 px |
| YOLO SAM2 | 91% | 1.03% | 0.69% | 17.3 px | 15.0 px |
| ResNet18 | 100% | 8.96% | 3.26% | 82.8 px | 31.8 px |
| ResNet50 | 100% | 7.53% | 3.42% | 71.0 px | 36.2 px |

Y_norm is the vertical midpoint error as a fraction of barrel height. Full per-video breakdowns are in `results/` (`edge_scan_test.txt`, `blob_sam2_test.txt`, `yolo_sam2_test.txt`, `resnet18_test.txt`, `resnet50_test.txt`).

---

## Setup

```bash
pip install torch torchvision opencv-python-headless ultralytics
pip install git+https://github.com/facebookresearch/sam2.git
```

Pretrained weights are in `weights/`:
- `weights/yolo_obb.pt` — YOLO11n-OBB fine-tuned on rubber stopper crops (100 epochs)
- `weights/resnet18.pt` — ResNet18 regression head (best val L1: 0.0375)
- `weights/resnet50.pt` — ResNet50 regression head (best val L1: 0.0369)

SAM2 weights are downloaded automatically from Hugging Face on first run.

---

## Running Evaluation

Place your data in the repo root:
```
train-videos/   <id>.mp4  ...
train-labels/   <id>.zip  ...
test-videos/    <id>.mp4  ...
test-labels/    <id>.zip  ...
```

Label zips contain CVAT XML annotations. See `sample_data/` for the expected format.

```bash
# Evaluate on the test split
python run.py --method edge_scan --split test

# Evaluate on a single video
python run.py --method sam --split test --video <id>

# Save annotated frames for inspection
python run.py --method yolo_sam --split test --save-frames output/frames/

# Save annotated video
python run.py --method resnet --split test --save-video output/{video_id}.mp4
```

Available method names: `dummy`, `edge_scan`, `sam`, `yolo_sam`, `resnet`, `resnet50`.

The `dummy` baseline returns the bbox midline and is useful for verifying that the evaluation pipeline runs correctly before loading any real method.

---

## Training

ResNet regression training requires a dataset of OBB-aligned barrel crops with YOLO-format labels (the liquid-line Y stored as the `cy` field):

```bash
python training/train_resnet.py --data /path/to/dataset
python training/train_resnet.py --data /path/to/dataset --backbone resnet50
```

YOLO stopper detection training used Ultralytics with the config in `runs/obb/train/args.yaml`.

Training logs are saved in `runs/resnet18/` and `runs/resnet50/`.

---

## Generating Paper Figures

All figure scripts must be run from the repo root and require the test videos and labels to be present.

```bash
python figures/extract_obb_panels.py      # OBB preprocessing panels
python figures/generate_pipeline_fig.py   # edge detection pipeline (Figure 2)
python figures/generate_sam_fig.py        # SAM2 pipeline (Figure 3)
python figures/plot_training_curves.py    # ResNet training curves (Figure 4)
```

Output goes to `figures/output/`.

---

## Repository Structure

```
run.py                  evaluation entry point
methods/
  edge_scan.py          classical edge scan detector
  sam.py                blob-prompted SAM2 detector
  yolo_sam.py           YOLO-prompted SAM2 detector
  resnet_regression.py  ResNet regression detector
  base.py               abstract base class for all methods
  template.py           template for adding a new method
  dummy.py              dummy baseline for pipeline testing
data/
  annotations.py        CVAT XML parser
  dataset.py            video + label loader
evaluation/
  metrics.py            D_end and Y_norm computation
utils/
  obb.py                shared OBB alignment (obb_crop)
  viz.py                line/bbox drawing helpers
training/
  train_resnet.py       ResNet fine-tuning script
figures/
  generate_pipeline_fig.py
  generate_sam_fig.py
  plot_training_curves.py
  extract_obb_panels.py
weights/                pretrained model weights
results/                per-video evaluation outputs
runs/                   training logs and YOLO config
sample_data/            annotation format example
```

---

## Adding a New Method

1. Copy `methods/template.py` to `methods/your_method.py`
2. Implement `detect()` in your class
3. Register it in the `METHODS` dict at the top of `run.py`
4. Run: `python run.py --method your_method`
