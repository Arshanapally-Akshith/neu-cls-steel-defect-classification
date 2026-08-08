"""PyTorch Dataset for the NEU-CLS transfer-learning model (Phase 3).

Reuses src.data.loader.load_split_images for image I/O — the same function
Phase 2's HOG baseline uses — so the frozen-manifest contract (only files
listed in a data/splits/*.csv manifest are ever loaded) and the RGB decode
path stay identical across phases. This module only adds the torchvision
transform + tensor/label conversion on top.
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.loader import load_split_images


class NEUClsDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, raw_dir: Path, class_to_idx: dict, transform=None):
        self.images, self.labels = load_split_images(manifest, raw_dir, grayscale=False)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[self.labels[idx]]
        return image, label
