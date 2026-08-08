"""Phase 4 entry point: error analysis on the already-trained Phase 2 and
Phase 3 models.

Re-runs INFERENCE ONLY (no training) for both models on the frozen test
manifest (data/splits/test.csv) to recover per-sample predictions and
confidences — Phase 2/3 only persisted aggregate metrics. Neither model,
the split, nor the test data are modified. As a consistency check, the
re-computed aggregate test metrics are compared against the values already
recorded in reports/baseline_results.json and reports/cv_results.json.

Usage: python -m scripts.run_phase4_error_analysis
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.loader import load_split_images
from src.data.split import SPLIT_NAMES, load_manifests
from src.eval.error_analysis import (
    compare_model_errors,
    confusion_concentration,
    plot_misclassified_grid,
    plot_per_class_f1_comparison,
    top_confusion_pairs,
)
from src.eval.metrics import compute_metrics
from src.eval.report_error_analysis import generate_error_analysis_report_markdown
from src.models.baseline import load_trained_pipeline, predict_with_confidence as baseline_predict_with_confidence
from src.models.transfer import build_transforms, load_trained_model, make_dataloader
from src.models.transfer import predict_with_confidence as resnet_predict_with_confidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_split_files(split_dir: Path) -> dict:
    hashes = {name: _sha256(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    hashes["split_summary.json"] = _sha256(split_dir / "split_summary.json")
    return hashes


def main() -> None:
    config = load_config()
    classes = config["dataset"]["classes"]
    ea_cfg = config["error_analysis"]

    split_dir = resolve_path(config["split"]["output_dir"])
    split_hashes_before = _hash_split_files(split_dir)
    manifests = load_manifests(split_dir)
    test_manifest = manifests["test"].sort_values("filename").reset_index(drop=True)
    print(f"[test] loaded frozen test manifest: {len(test_manifest)} images (untouched, read-only)")

    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    y_true = test_manifest["class"].to_numpy()
    filenames = test_manifest["filename"].to_numpy()

    baseline_model_path = resolve_path(config["baseline"]["output"]["model_path"])
    resnet_model_path = resolve_path(config["transfer_learning"]["output"]["model_path"])
    model_hashes_before = {"baseline": _sha256(baseline_model_path), "resnet": _sha256(resnet_model_path)}

    # --- Baseline inference (HOG + Logistic Regression) -------------------------
    pipeline = load_trained_pipeline(baseline_model_path)
    X_test_gray, y_test_baseline_order = load_split_images(test_manifest, raw_dir, grayscale=True)
    assert list(y_test_baseline_order) == list(y_true), "baseline image load order does not match test manifest"
    baseline_preds, baseline_conf = baseline_predict_with_confidence(pipeline, X_test_gray)
    baseline_metrics = compute_metrics(y_true, baseline_preds, classes)
    print(f"[baseline] re-inference accuracy={baseline_metrics['accuracy']:.4f} f1_macro={baseline_metrics['f1_macro']:.4f}")

    # --- ResNet18 inference -------------------------------------------------------
    model, ckpt_classes, _checkpoint = load_trained_model(resnet_model_path)
    assert ckpt_classes == classes, f"checkpoint classes {ckpt_classes} != config classes {classes}"
    tl_cfg = config["transfer_learning"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    _train_transform, eval_transform = build_transforms(tl_cfg)
    test_loader = make_dataloader(
        test_manifest, raw_dir, class_to_idx, eval_transform,
        batch_size=tl_cfg["training"]["batch_size"], shuffle=False,
        num_workers=tl_cfg["training"]["num_workers"], seed=tl_cfg["seed"],
    )
    y_test_resnet_order, resnet_preds, resnet_conf = resnet_predict_with_confidence(
        model, test_loader, torch.device("cpu"), idx_to_class
    )
    assert list(y_test_resnet_order) == list(y_true), "resnet image load order does not match test manifest"
    resnet_metrics = compute_metrics(y_true, resnet_preds, classes)
    print(f"[resnet18] re-inference accuracy={resnet_metrics['accuracy']:.4f} f1_macro={resnet_metrics['f1_macro']:.4f}")

    # --- Consistency check against Phase 2/3's already-reported results ---------
    reports_dir = resolve_path(ea_cfg["output"]["results_dir"])
    max_metric_diff = 0.0
    compare_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    with open(reports_dir / "baseline_results.json", "r", encoding="utf-8") as f:
        saved_baseline_metrics = json.load(f)["test_metrics"]
    with open(reports_dir / "cv_results.json", "r", encoding="utf-8") as f:
        saved_resnet_metrics = json.load(f)["test_metrics"]
    for key in compare_keys:
        max_metric_diff = max(max_metric_diff, abs(baseline_metrics[key] - saved_baseline_metrics[key]))
        max_metric_diff = max(max_metric_diff, abs(resnet_metrics[key] - saved_resnet_metrics[key]))
    print(f"[verify] max abs diff vs previously reported Phase 2/3 test metrics: {max_metric_diff:.2e}")
    if max_metric_diff > 1e-6:
        raise RuntimeError(
            "Re-computed test metrics diverge from Phase 2/3's saved results beyond floating-point "
            f"tolerance (max diff {max_metric_diff:.2e}) — inference must be deterministic. "
            "This likely means image loading order or preprocessing changed since Phase 2/3."
        )

    # --- Per-sample results table -------------------------------------------------
    predictions_df = pd.DataFrame({
        "filename": filenames,
        "true_class": y_true,
        "baseline_pred": baseline_preds,
        "baseline_confidence": baseline_conf,
        "baseline_correct": baseline_preds == y_true,
        "resnet_pred": resnet_preds,
        "resnet_confidence": resnet_conf,
        "resnet_correct": resnet_preds == y_true,
    })
    predictions_csv_path = resolve_path(ea_cfg["output"]["predictions_csv"])
    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"[save] per-sample predictions written to {predictions_csv_path}")

    # --- Confusion pairs, concentration, patches->crazing check -----------------
    top_n = ea_cfg["top_n_confusion_pairs"]
    resnet_top_pairs = top_confusion_pairs(resnet_metrics["confusion_matrix"], classes, top_n=top_n)
    baseline_top_pairs = top_confusion_pairs(baseline_metrics["confusion_matrix"], classes, top_n=top_n)
    resnet_concentration = confusion_concentration(resnet_metrics["confusion_matrix"], classes)

    patches_idx, crazing_idx = classes.index("patches"), classes.index("crazing")
    patches_crazing = {
        "baseline_count": baseline_metrics["confusion_matrix"][patches_idx][crazing_idx],
        "resnet_count": resnet_metrics["confusion_matrix"][patches_idx][crazing_idx],
        "patches_support": baseline_metrics["per_class"]["patches"]["support"],
    }
    print(f"[check] patches->crazing: baseline={patches_crazing['baseline_count']}, resnet18={patches_crazing['resnet_count']}")

    # --- Representative examples ---------------------------------------------------
    resnet_correct_df = predictions_df[predictions_df["resnet_correct"]]
    representative_correct = [
        {"filename": row["filename"], "true_class": row["true_class"], "confidence": row["resnet_confidence"]}
        for cls in classes
        for _, row in resnet_correct_df[resnet_correct_df["true_class"] == cls]
            .nlargest(1, "resnet_confidence").iterrows()
    ]
    resnet_incorrect_df = predictions_df[~predictions_df["resnet_correct"]].sort_values("resnet_confidence", ascending=False)
    representative_incorrect = [
        {"filename": r["filename"], "true_class": r["true_class"], "pred_class": r["resnet_pred"], "confidence": r["resnet_confidence"]}
        for _, r in resnet_incorrect_df.iterrows()
    ]

    baseline_patches_crazing_df = predictions_df[
        (predictions_df["true_class"] == "patches") & (predictions_df["baseline_pred"] == "crazing")
    ].sort_values("baseline_confidence", ascending=False)
    baseline_patches_crazing_examples = [
        {"filename": r["filename"], "true_class": r["true_class"], "pred_class": r["baseline_pred"], "confidence": r["baseline_confidence"]}
        for _, r in baseline_patches_crazing_df.iterrows()
    ]

    # --- Systematic vs isolated narrative --------------------------------------
    error_classes = [c for c, v in resnet_concentration.items() if v["total_errors"] > 0]
    concentrated = [c for c in error_classes if resnet_concentration[c]["distinct_confused_with"] == 1]
    scattered = [c for c in error_classes if resnet_concentration[c]["distinct_confused_with"] > 1]
    clean_classes = [c for c in classes if resnet_concentration[c]["total_errors"] == 0]
    systematic_narrative = (
        f"{len(clean_classes)} of {len(classes)} classes ({', '.join(clean_classes)}) have zero test errors. "
        + (f"{len(concentrated)} class(es) with errors ({', '.join(concentrated)}) concentrate ALL their errors into "
           "a single dominant wrong class — a systematic (not random) confusion. "
           if concentrated else "")
        + (f"{len(scattered)} class(es) ({', '.join(scattered)}) spread their errors across multiple different wrong "
           "classes, which looks more like isolated noise than one specific learned confusion."
           if scattered else "")
    )

    # --- Error overlap vs baseline -----------------------------------------------
    error_overlap = compare_model_errors(y_true, baseline_preds, resnet_preds, filenames, label_a="baseline", label_b="resnet")
    both_wrong_filenames = set(error_overlap["filenames"]["both_wrong"])
    both_wrong_examples = [
        {"filename": r["filename"], "true_class": r["true_class"], "baseline_pred": r["baseline_pred"], "resnet_pred": r["resnet_pred"]}
        for _, r in predictions_df[predictions_df["filename"].isin(both_wrong_filenames)].iterrows()
    ]

    # --- Plots ------------------------------------------------------------------
    f1_comparison_path = reports_dir / "error_analysis_f1_comparison.png"
    plot_per_class_f1_comparison(
        classes,
        [baseline_metrics["per_class"][c]["f1"] for c in classes],
        [resnet_metrics["per_class"][c]["f1"] for c in classes],
        "Baseline (HOG+LogReg)", "ResNet18 Transfer",
        f1_comparison_path,
    )
    print(f"[plot] per-class F1 comparison saved to {f1_comparison_path}")

    misclassified_grid_path = reports_dir / "error_analysis_misclassified_grid.png"
    if len(resnet_incorrect_df) > 0:
        grid_df = resnet_incorrect_df.rename(columns={"resnet_pred": "pred_class", "resnet_confidence": "pred_confidence"})
        plot_misclassified_grid(
            grid_df, raw_dir, misclassified_grid_path,
            max_images=ea_cfg["max_misclassified_grid_images"],
            title="ResNet18 Misclassified Test Examples",
        )
        print(f"[plot] misclassified examples grid saved to {misclassified_grid_path}")

    # --- Verify nothing touched ---------------------------------------------------
    split_hashes_after = _hash_split_files(split_dir)
    model_hashes_after = {"baseline": _sha256(baseline_model_path), "resnet": _sha256(resnet_model_path)}
    if split_hashes_before != split_hashes_after:
        raise RuntimeError("Frozen split manifests changed during Phase 4 — this must never happen.")
    if model_hashes_before != model_hashes_after:
        raise RuntimeError("Model files changed during Phase 4 — this must never happen.")
    print("[verify] frozen split manifests AND model files byte-identical before/after run (SHA-256 checked)")

    # --- Report -------------------------------------------------------------------
    ctx = {
        "classes": classes,
        "max_metric_diff": max_metric_diff,
        "resnet": {
            "confusion_matrix": resnet_metrics["confusion_matrix"],
            "per_class": resnet_metrics["per_class"],
            "top_pairs": resnet_top_pairs,
            "concentration": resnet_concentration,
            "representative_correct": representative_correct,
            "representative_incorrect": representative_incorrect,
            "n_test": len(test_manifest),
        },
        "baseline": {
            "confusion_matrix": baseline_metrics["confusion_matrix"],
            "per_class": baseline_metrics["per_class"],
            "top_pairs": baseline_top_pairs,
            "patches_crazing_examples": baseline_patches_crazing_examples,
        },
        "patches_crazing": patches_crazing,
        "systematic_narrative": systematic_narrative,
        "error_overlap": {**error_overlap, "both_wrong_examples": both_wrong_examples},
        "artifacts": {
            "predictions_csv": "error_analysis_predictions.csv",
            "resnet_cm": "transfer_confusion_matrix.png",
            "baseline_cm": "baseline_confusion_matrix.png",
            "f1_comparison": "error_analysis_f1_comparison.png",
            "misclassified_grid": "error_analysis_misclassified_grid.png",
        },
    }
    report_md = generate_error_analysis_report_markdown(ctx)
    report_path = reports_dir / "error_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[report] written to {report_path}")


if __name__ == "__main__":
    main()
