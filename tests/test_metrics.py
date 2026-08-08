import numpy as np

from src.eval.metrics import compute_metrics, plot_confusion_matrix

CLASSES = ["a", "b", "c"]


def test_compute_metrics_perfect_predictions():
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_pred = y_true.copy()
    result = compute_metrics(y_true, y_pred, CLASSES)
    assert result["accuracy"] == 1.0
    assert result["precision_macro"] == 1.0
    assert result["recall_macro"] == 1.0
    assert result["f1_macro"] == 1.0
    for cls in CLASSES:
        assert result["per_class"][cls]["precision"] == 1.0
        assert result["per_class"][cls]["recall"] == 1.0


def test_compute_metrics_known_values():
    # 3 classes, 2 samples each. All 'a' correct, all 'b' predicted as 'c',
    # all 'c' correct -> hand-computable precision/recall per class.
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "c", "c", "c", "c"])
    result = compute_metrics(y_true, y_pred, CLASSES)

    assert result["accuracy"] == 4 / 6
    assert result["per_class"]["a"]["precision"] == 1.0
    assert result["per_class"]["a"]["recall"] == 1.0
    assert result["per_class"]["b"]["recall"] == 0.0
    assert result["per_class"]["b"]["support"] == 2
    # 'c' predicted 4 times, 2 of which are true 'c' -> precision 0.5
    assert result["per_class"]["c"]["precision"] == 0.5
    assert result["per_class"]["c"]["recall"] == 1.0


def test_confusion_matrix_shape_and_totals():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "c", "c", "c", "c"])
    result = compute_metrics(y_true, y_pred, CLASSES)
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)


def test_plot_confusion_matrix_saves_file(tmp_path):
    cm = [[5, 1, 0], [0, 4, 1], [2, 0, 3]]
    save_path = tmp_path / "cm.png"
    plot_confusion_matrix(cm, CLASSES, save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
