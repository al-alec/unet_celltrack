import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from model import Unet
from dataset import CellTrackingDataset


def debug_model():
    """Debug model predictions vs ground truth"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = Unet(input_ch=1, n_class=1, base_ch=32)
    model.load_state_dict(torch.load('models/unet_celltrack_0.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Load one sample
    image_paths = sorted(glob.glob("data/tests/01/*.tif"))
    mask_paths = sorted(glob.glob("data/masks/*.tif"))

    idx = 0  # First image

    img = np.array(Image.open(image_paths[idx]))
    # msk = np.array(Image.open(mask_paths[idx]))

    # Preprocess exactly like training
    if len(img.shape) == 2:
        img_tensor = torch.from_numpy(img[..., np.newaxis]).float().permute(2, 0, 1) / 255.0
    else:
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

    # msk_binary = (msk > 0).astype(float)

    # Predict
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(device)
        pred = model(img_input)
        pred_sigmoid = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Print statistics
    print("="*60)
    print("DIAGNOSTIC")
    print("="*60)
    print(f"\nImage: {image_paths[idx]}")
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min()}, {img.max()}]")
    print(f"Image mean: {img.mean():.3f}")

    # print(f"\nMask shape: {msk.shape}")
    # print(f"Mask unique values: {np.unique(msk)[:10]}")
    # print(f"Mask binary ratio: {msk_binary.mean():.3f} (% of pixels that are cells)")

    print(f"\nPrediction shape: {pred_sigmoid.shape}")
    print(f"Prediction range: [{pred_sigmoid.min():.3f}, {pred_sigmoid.max():.3f}]")
    print(f"Prediction mean: {pred_sigmoid.mean():.3f}")
    print(f"Prediction std: {pred_sigmoid.std():.3f}")
    print(f"Pixels > 0.5: {(pred_sigmoid > 0.5).mean():.3f}")
    print(f"Pixels > 0.1: {(pred_sigmoid > 0.1).mean():.3f}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Inputs
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # axes[0, 1].imshow(msk, cmap='nipy_spectral')
    # axes[0, 1].set_title(f'Ground Truth IDs\n(unique: {len(np.unique(msk))} cells)', fontsize=14)
    # axes[0, 1].axis('off')

    # axes[0, 2].imshow(msk_binary, cmap='gray')
    # axes[0, 2].set_title(f'Binary Ground Truth\n({msk_binary.mean()*100:.1f}% cells)', fontsize=14)
    # axes[0, 2].axis('off')

    # Row 2: Predictions
    im1 = axes[1, 0].imshow(pred_sigmoid, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Prediction Heatmap\n(mean: {pred_sigmoid.mean():.3f})', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(pred_sigmoid > 0.5, cmap='gray')
    axes[1, 1].set_title(f'Pred Binary (>0.5)\n({(pred_sigmoid > 0.5).mean()*100:.1f}% cells)', fontsize=14)
    axes[1, 1].axis('off')

    # Overlay
    axes[1, 2].imshow(img, cmap='gray')
    axes[1, 2].imshow(pred_sigmoid > 0.5, cmap='Reds', alpha=0.5)
    # axes[1, 2].imshow(msk_binary, cmap='Greens', alpha=0.3)
    axes[1, 2].set_title('Overlay\n(Red=Pred, Green=GT)', fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('debug_prediction.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: debug_prediction.png")
    print("="*60)

    # Check for common issues
    print("\n🔍 POTENTIAL ISSUES:")
    if pred_sigmoid.mean() < 0.1:
        print("⚠️  Predictions are very low (model predicts mostly background)")
        print("   → Model may not be trained enough")
        print("   → Check if loss is decreasing")
    elif pred_sigmoid.mean() > 0.9:
        print("⚠️  Predictions are very high (model predicts mostly cells)")
        print("   → Model may be overfitting")
        print("   → Data imbalance issue")

    if pred_sigmoid.std() < 0.1:
        print("⚠️  Predictions have low variance (all similar values)")
        print("   → Model is not confident")
        print("   → May need more training")

    if (pred_sigmoid > 0.5).mean() < 0.01:
        print("⚠️  Almost no pixels predicted as cells")
        print("   → Model is very conservative")
        print("   → Lower threshold or train more")

    print("\n💡 RECOMMENDATIONS:")
    print("1. Check training loss curves in TensorBoard")
    print("2. Train for more epochs if loss was still decreasing")
    print("3. Verify data preprocessing matches between train and inference")
    print("4. Try lowering prediction threshold (e.g., 0.3 instead of 0.5)")
    print("="*60)


if __name__ == "__main__":
    debug_model()
