"""Dataset of undistorted photos and their masks, selected by a split file list."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch

DATASET = Path(__file__).resolve().parents[1] / "dataset"


def read_split(name):
    return (DATASET / "splits" / f"{name}.txt").read_text().split()


class MedBoxDataset(torch.utils.data.Dataset):
    def __init__(self, split, train=False, limit=None):
        self.ids = read_split(split)[:limit]
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.cvtColor(cv2.imread(str(DATASET / "undistorted/images" / f"{name}.jpg")),
                             cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(DATASET / "undistorted/masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if self.train:
            image, mask = augment(image, mask)
        ys, xs = np.where(mask)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),
            "masks": torch.from_numpy(mask[None].astype(np.uint8)),
        }
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        return tensor, target


def augment(image, mask):
    """Horizontal flip and brightness / contrast jitter. No rotation or scaling: those would
    change the apparent size of the object, which is the thing being measured."""
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    gain, bias = random.uniform(0.75, 1.25), random.uniform(-25, 25)
    image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def collate(batch):
    return tuple(zip(*batch))
