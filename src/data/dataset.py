"""Dataset that feeds images to PyTorch.

Design rule: CROP, NEVER RESIZE. Generator artefacts live in high
frequencies; resizing resamples them away.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(im):
    """PIL image -> normalised float tensor of shape (3, H, W)."""
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class ForensicDataset(Dataset):
    def __init__(self, manifest, root, split=None, crop=224, train=False):
        df = pd.read_csv(manifest)
        if split is not None:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.df)

    def _crop(self, im):
        w, h = im.size
        c = self.crop
        if self.train:
            left = random.randint(0, max(0, w - c))
            top = random.randint(0, max(0, h - c))
        else:
            left, top = (w - c) // 2, (h - c) // 2
        return im.crop((left, top, left + c, top + c))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self.root / row["path"]) as raw:
            im = raw.convert("RGB")
            im = self._crop(im)
            if self.train and random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            tensor = to_tensor(im)
        return tensor, torch.tensor(float(row["label"]))
