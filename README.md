# Assignment 2 – Multi-Stage Visual Perception Pipeline

A VGG11-based multi-task perception system on the Oxford-IIIT Pet dataset (37 breeds, ~7.4k images). 
## W&B Report
W&B Report Link - https://wandb.ai/samwellthorson04-iit-madras/assignment2-report/reports/Assignment-2--VmlldzoxNjQ2MTM3OA?accessToken=q3vre8bl0elxp4qea53pzxk6trvp7pkdenelnsfs02ta65ktlno6a1s32yx0r8vo

Github Repo link- 
## Architecture

- **VGG11 Encoder** — 8 conv layers with BatchNorm
- **Classifier** — Encoder + FC head (4096→4096→37), Custom Dropout (p=0.5)
- **Localizer** — Encoder + regression head → [x_center, y_center, w, h] in pixel space, trained with SmoothL1 + custom IoU loss
- **U-Net Segmenter** — Encoder + symmetric decoder with transposed convolutions and skip connections, 3-class trimap output
- **Multi-Task Model** — Shared encoder with all three heads, loads pretrained checkpoints via Google Drive (`gdown`)

## Project Structure

```
├── models/
│   ├── vgg11.py           # VGG11 encoder (with VGG11 alias for autograder)
│   ├── layers.py          # Custom Dropout (inverted, no nn.Dropout)
│   ├── classification.py  # VGG11Classifier
│   ├── localization.py    # VGG11Localizer
│   ├── segmentation.py    # VGG11UNet
│   └── multitask.py       # MultiTaskPerceptionModel (gdown + shared backbone)
├── losses/
│   └── iou_loss.py        # Custom IoU loss (mean/sum/none reduction)
├── data/
│   └── pets_dataset.py    # Oxford-IIIT Pet dataset loader
├── multitask.py           # Root-level re-export for autograder
├── train.py               # Training entrypoint (classification/localization/segmentation)
├── inference.py            # Evaluation and visualization helpers
├── checkpoints/
│   └── checkpoints.md     # Weights hosted on Google Drive
├── requirements.txt
└── README.md
```

## Training

```bash
pip install -r requirements.txt

# 1. Classifier (trains encoder + FC head)
python train.py --task classification --data_dir ./data/pets --epochs 30 --lr 1e-4 --dropout_p 0.5

# 2. Localizer (2-phase: frozen encoder → unfreeze blocks 4-5)
python train.py --task localization --data_dir ./data/pets --epochs 50 --lr 5e-5

# 3. Segmentation (full fine-tune)
python train.py --task segmentation --data_dir ./data/pets --epochs 30 --lr 1e-4
```

## Checkpoints

Model weights are hosted on Google Drive and auto-downloaded via `gdown` in `MultiTaskPerceptionModel.__init__()`. No `.pth` files in this repo.

