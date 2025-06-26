import os
from PIL import Image

from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].
                                 replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask


def get_dataloader(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True):
    train_set = CarvanaDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = CarvanaDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader
