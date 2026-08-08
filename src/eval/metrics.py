"""Classification metrics + confusion matrix reporting, shared across models.

Kept independent of any particular model/pipeline so Phase 3+ (transfer
learning) can reuse it unchanged.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class = {
        cls: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }

    return {
        "accuracy": accuracy,
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "f1_weighted": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
        "n_samples": int(len(y_true)),
    }


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, classes: list[str], save_path: Path, title: str = "Confusion Matrix") -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
