import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import Unet


def inference_on_images(model_path, input_dir, output_dir="predictions", threshold=0.9, visualize=True, log_tensorboard=True):
    """
    Run inference on images without ground truth masks

    Args:
        model_path: Path to trained model
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        threshold: Threshold for binary prediction (default: 0.5)
        visualize: Whether to create visualization images
        log_tensorboard: Whether to log results to TensorBoard
    """

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # TensorBoard writer
    writer = None
    if log_tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/inference_{timestamp}'
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")

    # Load model
    model = Unet(input_ch=1, n_class=1, base_ch=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Get all images
    image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    # binary_dir = os.path.join(output_dir, "binary_masks")
    binary_dir = "test_dataset/predictions"
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    if visualize:
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    print(f"\nProcessing images...")

    # Statistics for TensorBoard
    prediction_stats = {
        'mean_confidence': [],
        'positive_pixels_ratio': [],
    }

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            # Load image
            img = np.array(Image.open(img_path))
            filename = Path(img_path).stem

            print(f"[{idx+1}/{len(image_paths)}] Processing: {filename}")

            # Preprocess
            if len(img.shape) == 2:
                img_tensor = torch.from_numpy(img[..., np.newaxis]).float().permute(2, 0, 1) / 255.0
            else:
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Predict
            pred = model(img_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

            # Binary prediction
            binary_pred = (pred > threshold).astype(np.uint8) * 255

            # Calculate statistics
            mean_conf = pred.mean()
            pos_ratio = (pred > threshold).mean()
            prediction_stats['mean_confidence'].append(mean_conf)
            prediction_stats['positive_pixels_ratio'].append(pos_ratio)

            # Log to TensorBoard
            if writer is not None:
                # Add original image
                writer.add_image(f'Input/{filename}', img_tensor.squeeze(0), idx, dataformats='CHW')

                # Add prediction heatmap
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                writer.add_image(f'Prediction/{filename}', pred_tensor, idx, dataformats='CHW')

                # Add binary mask
                binary_tensor = torch.from_numpy(binary_pred / 255.0).unsqueeze(0)
                writer.add_image(f'Binary/{filename}', binary_tensor, idx, dataformats='CHW')

                # Add statistics
                writer.add_scalar('Stats/mean_confidence', mean_conf, idx)
                writer.add_scalar('Stats/positive_pixels_ratio', pos_ratio, idx)

            # Save predictions
            # Save probability map (0-255)
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img.save(os.path.join(masks_dir, f"{filename}_pred.png"))

            # Save binary mask
            binary_img = Image.fromarray(binary_pred)
            binary_img.save(os.path.join(binary_dir, f"{filename}.png"))

            if visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original Image', fontsize=14)
                axes[0].axis('off')

                # Probability map
                im1 = axes[1].imshow(pred, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title('Prediction Probability', fontsize=14)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                # Binary mask overlay
                axes[2].imshow(img, cmap='gray')
                axes[2].imshow(binary_pred, cmap='Reds', alpha=0.5)
                axes[2].set_title(f'Binary Mask (threshold={threshold})', fontsize=14)
                axes[2].axis('off')

                plt.suptitle(filename, fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{filename}_visualization.png"),
                           dpi=150, bbox_inches='tight')
                plt.close()

    # Log summary statistics to TensorBoard
    if writer is not None:
        avg_confidence = np.mean(prediction_stats['mean_confidence'])
        avg_pos_ratio = np.mean(prediction_stats['positive_pixels_ratio'])

        writer.add_scalar('Summary/avg_confidence', avg_confidence, 0)
        writer.add_scalar('Summary/avg_positive_ratio', avg_pos_ratio, 0)

        # Add histogram of confidence values
        writer.add_histogram('Distribution/confidence', np.array(prediction_stats['mean_confidence']), 0)
        writer.add_histogram('Distribution/positive_ratio', np.array(prediction_stats['positive_pixels_ratio']), 0)

        writer.close()
        print(f"\n  TensorBoard logs saved")

    print(f"\n{'='*60}")
    print(f"✓ Inference completed!")
    print(f"  Processed: {len(image_paths)} images")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"    - Probability masks: {masks_dir}/")
    print(f"    - Binary masks: {binary_dir}/")
    if visualize:
        print(f"    - Visualizations: {viz_dir}/")
    if writer is not None:
        print(f"    - TensorBoard logs: {log_dir}/")
    print(f"\n  Statistics:")
    print(f"    - Avg confidence: {avg_confidence:.4f}")
    print(f"    - Avg positive ratio: {avg_pos_ratio:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on images without ground truth')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Directory to save predictions (default: predictions)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction (default: 0.5)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization generation')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    args = parser.parse_args()

    inference_on_images(
        model_path=args.model,
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        visualize=not args.no_viz,
        log_tensorboard=not args.no_tensorboard
    )
