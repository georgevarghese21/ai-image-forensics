"""Dataset that feeds images to PyTorch.

Design rule: CROP, NEVER RESIZE. Generator artefacts live in high
frequencies; resizing resamples them away.
"""
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
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
    """Reads a manifest slice and yields (tensor, label).

    perturbation: eval-time only. Fixed degradation applied to every image,
        used by the robustness sweep. e.g. {"jpeg": 50}, {"scale": 0.5}
    augment: train-time only. Random blur, rescale and JPEG, so the model
        learns features that survive real-world handling.
    """

    def __init__(self, manifest, root, split=None, generators=None, crop=224,
                 train=False, perturbation=None, augment=False):
        df = pd.read_csv(manifest)
        if split is not None:
            df = df[df["split"] == split]
        # Filter which FAKE generators appear; always keep every real image,
        # since reals are the constant the fakes are measured against.
        if generators is not None:
            df = df[(df["label"] == 0) | (df["generator"].isin(generators))]
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.crop = crop
        self.train = train
        self.perturbation = perturbation or {}
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _perturb(self, im):
        """Fixed degradation before cropping, mirroring real-world handling."""
        p = self.perturbation
        if "scale" in p:
            w, h = im.size
            im = im.resize((max(1, int(w * p["scale"])),
                            max(1, int(h * p["scale"]))), Image.BICUBIC)
        if "blur" in p:
            im = im.filter(ImageFilter.GaussianBlur(p["blur"]))
        if "jpeg" in p:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(p["jpeg"]))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return im

    def _augment(self, im):
        """Random degradation during training (CNNSpot recipe, plus rescale).

        The model otherwise only ever sees clean quality-95 images and has no
        chance to learn features that survive real-world handling. Note the
        gain is specific to the degradations applied here - see the robustness
        table for what happens to transformations absent from this list.

        Order mirrors what a real image goes through: optical softness, then
        platform resizing, then final compression.
        """
        if random.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 3.0)))
        if random.random() < 0.5:
            # Downscale then restore. Augmentation runs after cropping, so the
            # tensor must stay crop x crop; scaling down and back up destroys
            # the same high-frequency detail a real resize would.
            w, h = im.size
            s = random.uniform(0.4, 0.9)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
            im = im.resize((w, h), Image.BICUBIC)
        if random.random() < 0.5:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=random.randint(30, 100))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return im

    def _crop(self, im):
        w, h = im.size
        c = self.crop
        # A downscaled image can end up smaller than the crop; pad by
        # reflection rather than upscaling, which would add its own artefacts.
        if w < c or h < c:
            im = ImageOps.expand(im, border=(max(0, (c - w + 1) // 2),
                                             max(0, (c - h + 1) // 2)))
            w, h = im.size
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
            # Degrade first, then crop - the order a real image goes through.
            if self.perturbation:
                im = self._perturb(im)
            im = self._crop(im)
            if self.train and random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if self.train and self.augment:
                im = self._augment(im)
            tensor = to_tensor(im)
        return tensor, torch.tensor(float(row["label"]))
