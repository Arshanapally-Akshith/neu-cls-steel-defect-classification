"""Load images for a frozen split manifest (data/splits/{train,val,test}.csv).

This module is the ONLY place that turns a split manifest (filename, class)
back into actual image arrays — it never re-derives which file belongs to
which split. Callers must pass in a manifest already produced by
src.data.split / frozen under data/splits/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_split_images(manifest: pd.DataFrame, raw_dir: Path, grayscale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load images referenced by a split manifest.

    Returns (X, y):
      X: object array of length N, each element an (H, W) uint8 array (grayscale)
         or (H, W, 3) uint8 array (grayscale=False).
      y: array of length N of class labels (str), same order as X.
    """
    raw_dir = Path(raw_dir)
    images = []
    labels = []
    for filename, cls in zip(manifest["filename"], manifest["class"]):
        path = raw_dir / filename
        with Image.open(path) as img:
            img = img.convert("L") if grayscale else img.convert("RGB")
            images.append(np.array(img))
        labels.append(cls)

    X = np.empty(len(images), dtype=object)
    for i, im in enumerate(images):
        X[i] = im
    y = np.array(labels)
    return X, y
