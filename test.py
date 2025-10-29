import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2

def read_gray_01(p: Path) -> np.ndarray:
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def to_binary01(arr: np.ndarray) -> np.ndarray:
    return (arr > 0.5).astype(np.uint8)

def compute_iou_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    inter = np.logical_and(pred_bin == 1, gt_bin == 1).sum(dtype=np.float64)
    union = np.logical_or (pred_bin == 1, gt_bin == 1).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))

def overlay_rgb(base: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    base: HxWx3 uint8 (si absente, on génère un fond gris)
    Colors: pred-only=RED, gt-only=GREEN, intersection=YELLOW
    """
    ov = base.copy()
    red    = (pred_bin==1) & (gt_bin==0)
    green  = (gt_bin==1) & (pred_bin==0)
    yellow = (gt_bin==1) & (pred_bin==1)
    ov[red]    = [255,   0,   0]
    ov[green]  = [  0, 255,   0]
    ov[yellow] = [255, 255,   0]
    return ov

def compute_and_log_iou(
    root="test_dataset",
    masks_dir="masks",
    preds_dir="predictions",
    img_dir="img",
    run_name="runs/iou_eval",
    max_images=12,
):
    root = Path(root)
    dm = root / masks_dir
    dp = root / preds_dir

    mask_paths = sorted([p for p in dm.iterdir()])
    pred_index = {p.stem: p for p in dp.iterdir()}

    writer = SummaryWriter(log_dir=run_name)
    ious = []
    step = 0

    for mpath in mask_paths:
        stem = mpath.stem

        gt = to_binary01(read_gray_01(mpath))
        pr = to_binary01(read_gray_01(pred_index[stem]))

        # Calcul IoU
        iou = compute_iou_binary(pr, gt)
        ious.append(iou)
        writer.add_scalar("test/IoU_per_image", iou, step)

        step += 1

    writer.flush()
    writer.close()

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img

if __name__ == "__main__":
    # compute_and_log_iou(
    #     root="test_dataset",
    #     masks_dir="masks",
    #     preds_dir="predictions",
    #     img_dir="img",
    #     run_name="runs/iou_eval",
    #     max_images=12
    # )


    #composants connexes

    # predictions_masks_path = "test_dataset/predictions/sor"
    # os.makedirs("test_dataset/labeled_masks", exist_ok=True)

    predictions_masks_path = "test_dataset/masks"
    os.makedirs("test_dataset/labeled_train_masks", exist_ok=True)

    predictions_masks_img = sorted(glob.glob(predictions_masks_path + "/*.tif"))


    for img in predictions_masks_img:
        filename = Path(img).stem
        img = cv2.imread(img, 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        num_labels, labels_im = cv2.connectedComponents(img)

        labeled_img = imshow_components(labels_im)

        labeled_img = Image.fromarray(labeled_img)
        labeled_img.save(os.path.join("test_dataset/labeled_train_masks", f"{filename}_wl.png"))