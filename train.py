import torch
import glob
from datetime import datetime
from torch.utils.data import DataLoader

from model import Unet
from dataset import CellTrackingDataset
from trainer import UnetTrainer
from pathlib import Path
import shutil


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # TensorBoard log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/unet_celltrack_{timestamp}'
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Data paths
    image_paths = sorted(glob.glob("data/images/*.tif"))
    mask_paths = sorted(glob.glob("data/masks/*.tif"))
    print(f"Images: {len(image_paths)}, Masks: {len(mask_paths)}")

    dataset = CellTrackingDataset(image_paths, mask_paths)

    longueur = len(dataset)
    train_size = int(0.8 * longueur)
    val_size = int(0.1 * longueur)
    test_size = longueur - train_size - val_size

    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # --- export du jeu de test vers test_dataset/img et test_dataset/masks ---
    out_img = Path("test_dataset/img");
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask = Path("test_dataset/masks");
    out_mask.mkdir(parents=True, exist_ok=True)

    for i in test_dataset.indices:
        src_img = Path(dataset.image_paths[i])  # ex: data/images/t087.tif
        src_mask = Path(dataset.mask_paths[i])  # ex: data/masks/t087.tif

        shutil.copy(src_img, out_img / src_img.name)
        shutil.copy(src_mask, out_mask / src_mask.name)

    print(f"Dataset size: {len(dataset)}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"Batches: {len(train_loader)}")



    # Model
    model = Unet(input_ch=1, n_class=1, base_ch=32)
    print(f"Model created with base_ch=32")

    trainer = UnetTrainer(model, train_loader, val_loader=val_loader, lr=1e-3, device=device, patience=5, log_dir=log_dir)

    trainer.train(epochs=25)

    trainer.save_model('models/unet_celltrack_6.pth')


if __name__ == "__main__":
    main()