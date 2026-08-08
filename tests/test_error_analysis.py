import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import resolve_path
from src.data.loader import load_split_images
from src.eval.error_analysis import (
    _format_misclassified_title,
    compare_model_errors,
    confusion_concentration,
    plot_misclassified_grid,
    plot_per_class_f1_comparison,
    top_confusion_pairs,
)
from src.eval.metrics import compute_metrics
from src.models.baseline import load_trained_pipeline, predict_with_confidence as baseline_predict_with_confidence
from src.models.transfer import build_transforms, load_trained_model, make_dataloader
from src.models.transfer import predict_with_confidence as resnet_predict_with_confidence

CLASSES = ["a", "b", "c"]
CM = [
    [5, 3, 0],   # true a: 5 correct, 3 -> b
    [1, 6, 1],   # true b: 1 -> a, 6 correct, 1 -> c
    [0, 0, 8],   # true c: all correct
]


def test_top_confusion_pairs_ranking_and_diagonal_excluded():
    pairs = top_confusion_pairs(CM, CLASSES, top_n=5)
    assert pairs == [
        {"true": "a", "pred": "b", "count": 3},
        {"true": "b", "pred": "a", "count": 1},
        {"true": "b", "pred": "c", "count": 1},
    ]
    # no diagonal entries anywhere
    assert all(p["true"] != p["pred"] for p in pairs)


def test_top_confusion_pairs_respects_top_n():
    pairs = top_confusion_pairs(CM, CLASSES, top_n=1)
    assert pairs == [{"true": "a", "pred": "b", "count": 3}]


def test_top_confusion_pairs_empty_when_no_errors():
    identity_cm = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    assert top_confusion_pairs(identity_cm, CLASSES) == []


def test_confusion_concentration_values():
    result = confusion_concentration(CM, CLASSES)

    assert result["a"]["correct"] == 5
    assert result["a"]["total_errors"] == 3
    assert result["a"]["distinct_confused_with"] == 1
    assert result["a"]["dominant_confusion"] == {"pred": "b", "count": 3}

    assert result["b"]["total_errors"] == 2
    assert result["b"]["distinct_confused_with"] == 2

    assert result["c"]["total_errors"] == 0
    assert result["c"]["distinct_confused_with"] == 0
    assert result["c"]["dominant_confusion"] is None


def test_compare_model_errors_four_way_breakdown():
    y_true = ["a", "a", "a", "a"]
    pred_baseline = ["a", "b", "a", "b"]  # correct: T, F, T, F
    pred_resnet = ["a", "a", "b", "b"]    # correct: T, T, F, F
    filenames = ["f1", "f2", "f3", "f4"]

    result = compare_model_errors(y_true, pred_baseline, pred_resnet, filenames, label_a="baseline", label_b="resnet")

    assert result["counts"] == {
        "both_correct": 1,
        "both_wrong": 1,
        "baseline_only_wrong": 1,
        "resnet_only_wrong": 1,
    }
    assert result["filenames"]["both_correct"] == ["f1"]
    assert result["filenames"]["baseline_only_wrong"] == ["f2"]
    assert result["filenames"]["resnet_only_wrong"] == ["f3"]
    assert result["filenames"]["both_wrong"] == ["f4"]


def test_compare_model_errors_all_categories_sum_to_total():
    rng = np.random.default_rng(0)
    y_true = rng.choice(CLASSES, size=50)
    pred_a = rng.choice(CLASSES, size=50)
    pred_b = rng.choice(CLASSES, size=50)
    filenames = [f"f{i}" for i in range(50)]

    result = compare_model_errors(y_true, pred_a, pred_b, filenames)
    assert sum(result["counts"].values()) == 50


def test_plot_per_class_f1_comparison_saves_file(tmp_path):
    save_path = tmp_path / "f1_comparison.png"
    plot_per_class_f1_comparison(CLASSES, [0.9, 0.8, 0.7], [0.95, 0.85, 0.75], "A", "B", save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_misclassified_grid_saves_file(split_manifests, raw_dir, tmp_path):
    sample = split_manifests["train"].head(2)
    df = pd.DataFrame({
        "filename": sample["filename"].tolist(),
        "true_class": sample["class"].tolist(),
        "pred_class": ["scratches", "patches"],
        # decoy column present alongside the real one, as in the Phase 4
        # per-sample table (baseline_confidence + resnet's pred_confidence)
        "baseline_confidence": [0.9999, 0.9999],
        "pred_confidence": [0.91, 0.42],
    })
    save_path = tmp_path / "grid.png"
    plot_misclassified_grid(df, raw_dir, save_path, max_images=9)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_misclassified_title_uses_named_confidence_column_not_any_confidence_column():
    """Regression test: a misclassified_df may legitimately carry more than
    one "*_confidence" column (e.g. both models' confidences, for
    cross-model comparison rows). The title must use exactly the column
    named by `confidence_col`, never just "whichever *_confidence column
    happens to come first" — that silently mislabels the plotted model's
    confidence with a different model's number."""
    row = pd.Series({
        "true_class": "a", "pred_class": "b",
        "baseline_confidence": 0.9445,  # decoy: a different model's confidence
        "pred_confidence": 0.7883,      # the one that must be used
    })
    title = _format_misclassified_title(row, confidence_col="pred_confidence")
    assert "0.79" in title
    assert "0.94" not in title


def test_plot_misclassified_grid_noop_on_empty_dataframe(raw_dir, tmp_path):
    empty_df = pd.DataFrame(columns=["filename", "true_class", "pred_class", "pred_confidence"])
    save_path = tmp_path / "grid_empty.png"
    plot_misclassified_grid(empty_df, raw_dir, save_path)
    assert not save_path.exists()


# ---------------------------------------------------------------------------
# Integration / consistency check: this is the core guarantee Phase 4 relies
# on — re-running INFERENCE ONLY on the full frozen test set, for both
# already-trained models, must reproduce Phase 2/3's saved aggregate metrics
# exactly (same models, same frozen test data, deterministic eval-time
# preprocessing => no reason for it to differ).
# ---------------------------------------------------------------------------

def test_full_test_set_reinference_matches_saved_phase2_and_phase3_metrics(
    config, split_manifests, raw_dir, baseline_model_path, resnet_checkpoint_path,
):
    reports_dir = resolve_path(config["error_analysis"]["output"]["results_dir"])
    baseline_results_path = reports_dir / "baseline_results.json"
    cv_results_path = reports_dir / "cv_results.json"
    if not baseline_results_path.exists() or not cv_results_path.exists():
        pytest.skip("Phase 2/3 results JSON not found; run scripts/run_phase2_baseline.py and run_phase3_transfer.py first.")

    classes = config["dataset"]["classes"]
    test_manifest = split_manifests["test"].sort_values("filename").reset_index(drop=True)
    y_true = test_manifest["class"].to_numpy()

    # Baseline
    pipeline = load_trained_pipeline(baseline_model_path)
    X_gray, _ = load_split_images(test_manifest, raw_dir, grayscale=True)
    baseline_preds, _conf = baseline_predict_with_confidence(pipeline, X_gray)
    baseline_metrics = compute_metrics(y_true, baseline_preds, classes)

    # ResNet18
    model, ckpt_classes, _ckpt = load_trained_model(resnet_checkpoint_path)
    tl_cfg = config["transfer_learning"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    _train_t, eval_t = build_transforms(tl_cfg)
    loader = make_dataloader(
        test_manifest, raw_dir, class_to_idx, eval_t,
        batch_size=tl_cfg["training"]["batch_size"], shuffle=False,
        num_workers=tl_cfg["training"]["num_workers"], seed=tl_cfg["seed"],
    )
    _y_true_r, resnet_preds, _conf_r = resnet_predict_with_confidence(model, loader, torch.device("cpu"), idx_to_class)
    resnet_metrics = compute_metrics(y_true, resnet_preds, classes)

    with open(baseline_results_path, "r", encoding="utf-8") as f:
        saved_baseline = json.load(f)["test_metrics"]
    with open(cv_results_path, "r", encoding="utf-8") as f:
        saved_resnet = json.load(f)["test_metrics"]

    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        assert baseline_metrics[key] == pytest.approx(saved_baseline[key], abs=1e-9)
        assert resnet_metrics[key] == pytest.approx(saved_resnet[key], abs=1e-9)
