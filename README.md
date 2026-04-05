# Assignment 2 – Multi-Stage Visual Perception Pipeline

VGG11-based multi-task perception system on the Oxford-IIIT Pet dataset.

## Tasks
1. **Classification** – 37-breed classifier with custom dropout & batch norm
2. **Localization** – Bounding box regression with MSE + custom IoU loss  
3. **Segmentation** – U-Net style decoder with transposed convolutions  
4. **Multi-task** – Shared backbone with all three heads

## Training (on Kaggle)
```bash
pip install -r requirements.txt

# Train in order:
python train.py --task classification --data_dir /kaggle/input/oxford-iiit-pet --epochs 30
python train.py --task localization --data_dir /kaggle/input/oxford-iiit-pet --epochs 30
python train.py --task segmentation --data_dir /kaggle/input/oxford-iiit-pet --epochs 30
```

## Checkpoints
Model weights are hosted on Google Drive and auto-downloaded via `gdown` in `MultiTaskPerceptionModel.__init__()`.

## W&B Report
[Link to W&B report goes here]
