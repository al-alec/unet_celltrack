import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from PIL import Image

mask_dir = Path("test_dataset/labeled_train_masks")
pred_dir = Path("test_dataset/labeled_masks")

f1_scores = []

for mpath in mask_dir.glob("*.png"):
    gt = np.array(Image.open(mpath).convert("L")) > 127
    pred_path = pred_dir / mpath.name
    pred = np.array(Image.open(pred_path).convert("L")) > 127

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    f1 = f1_score(gt_flat, pred_flat, average='binary', zero_division=1)
    f1_scores.append(f1)

print(f"F1-score moyen sur le test set: {np.mean(f1_scores):.4f}")
