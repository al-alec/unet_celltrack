import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CellTrackingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]))
        msk = np.array(Image.open(self.mask_paths[idx]))

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        msk = torch.from_numpy(msk > 0).float().unsqueeze(0)  # Binariser: 0 = background, 1 = cellule

        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)

        return img, msk
