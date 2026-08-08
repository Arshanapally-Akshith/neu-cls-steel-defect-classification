"""Generic image-grid plotting, shared by all three Phase 5 summary grids.

Takes already-rendered images (originals, overlays) — no model or file I/O
here beyond writing the final PNG, so this stays fast and easily testable.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(images: list[np.ndarray], titles: list[str], save_path: Path, n_cols: int = 3, suptitle: str = "") -> None:
    n = len(images)
    if n == 0:
        return
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=9)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
