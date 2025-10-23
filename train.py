import torch
import glob
from datetime import datetime
from torch.utils.data import DataLoader

from model import Unet
from dataset import CellTrackingDataset
from trainer import UnetTrainer

def main():
    # Configuration
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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-20, 20])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Dataset size: {len(dataset)}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"Batches: {len(train_loader)}")

    # Model
    model = Unet(input_ch=1, n_class=1, base_ch=32)
    print(f"Model created with base_ch=32")

    trainer = UnetTrainer(model, train_loader, val_loader=val_loader, lr=1e-3, device=device, patience=5, log_dir=log_dir)

    trainer.train(epochs=20)

    trainer.save_model('models/unet_celltrack_0.pth')


if __name__ == "__main__":
    main()