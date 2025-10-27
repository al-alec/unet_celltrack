import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

        # RESIZE AVANT tensorisation avec NEAREST pour le masque
        if img.shape[:2] != (512, 512):
            img_pil = Image.fromarray(img)
            msk_pil = Image.fromarray(msk)
            img = np.array(img_pil.resize((512, 512), Image.BILINEAR))
            msk = np.array(msk_pil.resize((512, 512), Image.NEAREST))  # NEAREST = pas d'interpolation!

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        msk = torch.from_numpy(msk > 127).float().unsqueeze(0)  # 127 = seuil au milieu de 0-255

        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)

        return img, msk
