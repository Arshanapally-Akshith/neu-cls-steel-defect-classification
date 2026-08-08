"""Error-analysis utilities shared by scripts/run_phase4_error_analysis.py.

Pure functions over already-computed predictions/confusion matrices — no
model loading, no I/O beyond the plotting helpers' save_path. Kept separate
and testable, matching the style of src/eval/metrics.py.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_split_images


def top_confusion_pairs(cm: list[list[int]] | np.ndarray, classes: list[str], top_n: int = 5) -> list[dict]:
    """Largest off-diagonal confusion-matrix cells, most confused first.

    Returns [{"true": cls, "pred": cls, "count": n}, ...], ties broken by
    (true, pred) alphabetical order for determinism, zero-count cells excluded.
    """
    cm = np.asarray(cm)
    pairs = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append({"true": true_cls, "pred": pred_cls, "count": count})
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["pred"]))
    return pairs[:top_n]


def confusion_concentration(cm: list[list[int]] | np.ndarray, classes: list[str]) -> dict:
    """Per true-class error concentration: how many total errors, and how many
    *distinct* predicted classes they're spread across.

    A class with many errors concentrated into 1 distinct wrong class is a
    systematic confusion; a class whose errors are spread across several
    distinct wrong classes (each with a low count) looks more like isolated
    noise than a specific learned confusion.
    """
    cm = np.asarray(cm)
    result = {}
    for i, true_cls in enumerate(classes):
        row = cm[i].copy()
        diag = int(row[i])
        row[i] = 0
        total_errors = int(row.sum())
        distinct_confused_with = int(np.count_nonzero(row))
        dominant_pair = None
        if distinct_confused_with > 0:
            j = int(row.argmax())
            dominant_pair = {"pred": classes[j], "count": int(row[j])}
        result[true_cls] = {
            "correct": diag,
            "total_errors": total_errors,
            "distinct_confused_with": distinct_confused_with,
            "dominant_confusion": dominant_pair,
        }
    return result


def compare_model_errors(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, filenames: np.ndarray | list,
    label_a: str = "model_a", label_b: str = "model_b",
) -> dict:
    """Per-sample overlap between two models' correctness on the SAME samples
    (same order/filenames assumed). Returns counts + filename lists for the
    4-way breakdown: both correct, both wrong, a-only-wrong, b-only-wrong.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true

    categories = {"both_correct": [], "both_wrong": [], f"{label_a}_only_wrong": [], f"{label_b}_only_wrong": []}
    for fname, ca, cb in zip(filenames, correct_a, correct_b):
        if ca and cb:
            categories["both_correct"].append(fname)
        elif not ca and not cb:
            categories["both_wrong"].append(fname)
        elif not ca and cb:
            categories[f"{label_a}_only_wrong"].append(fname)
        else:
            categories[f"{label_b}_only_wrong"].append(fname)

    return {"counts": {k: len(v) for k, v in categories.items()}, "filenames": categories}


def plot_per_class_f1_comparison(classes: list[str], f1_a: list[float], f1_b: list[float], label_a: str, label_b: str, save_path: Path) -> None:
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, f1_a, width, label=label_a)
    ax.bar(x + width / 2, f1_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1: Baseline vs. ResNet18 Transfer (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _format_misclassified_title(row: pd.Series, confidence_col: str) -> str:
    conf_str = f" ({row[confidence_col]:.2f})" if confidence_col in row.index else ""
    return f"{row['true_class']} -> {row['pred_class']}{conf_str}"


def plot_misclassified_grid(
    misclassified_df: pd.DataFrame, raw_dir: Path, save_path: Path, max_images: int = 9,
    title: str = "Misclassified Examples", confidence_col: str = "pred_confidence",
) -> None:
    """Grid of misclassified test images, titled "true -> pred (confidence)".

    `confidence_col` must be given explicitly (default "pred_confidence")
    rather than guessed from the DataFrame's columns — misclassified_df may
    legitimately carry OTHER models' "*_confidence" columns too (e.g. when
    comparing two models' errors on the same samples), and guessing risks
    silently plotting the wrong model's confidence.

    Reuses src.data.loader.load_split_images so image decoding stays
    identical to the rest of the pipeline.
    """
    sample = misclassified_df.head(max_images).reset_index(drop=True)
    if len(sample) == 0:
        return

    mini_manifest = sample.rename(columns={"true_class": "class"})[["filename", "class"]]
    images, _labels = load_split_images(mini_manifest, raw_dir, grayscale=False)

    n = len(sample)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        row = sample.iloc[i]
        ax.imshow(images[i])
        ax.set_title(_format_misclassified_title(row, confidence_col), fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
