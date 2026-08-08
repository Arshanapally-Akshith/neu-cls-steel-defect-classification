"""Phase 2 entry point: HOG + Logistic Regression classical baseline.

Loads images strictly through the frozen Phase 1 split manifests
(data/splits/{train,val,test}.csv) — never recreates or modifies the split.

Usage: python -m scripts.run_phase2_baseline
"""
import json
import sys
import time
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.loader import load_split_images
from src.data.split import load_manifests
from src.eval.metrics import compute_metrics, plot_confusion_matrix
from src.eval.report import generate_baseline_report_markdown
from src.models.baseline import build_inference_pipeline, extract_features, select_best_model


def main() -> None:
    config = load_config()
    classes = config["dataset"]["classes"]
    baseline_cfg = config["baseline"]

    split_dir = resolve_path(config["split"]["output_dir"])
    manifests = load_manifests(split_dir)
    print(f"[splits] loaded frozen manifests from {split_dir}: "
          f"train={len(manifests['train'])}, val={len(manifests['val'])}, test={len(manifests['test'])}")

    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    grayscale = baseline_cfg["grayscale"]
    X_train_img, y_train = load_split_images(manifests["train"], raw_dir, grayscale=grayscale)
    X_val_img, y_val = load_split_images(manifests["val"], raw_dir, grayscale=grayscale)
    X_test_img, y_test = load_split_images(manifests["test"], raw_dir, grayscale=grayscale)
    print(f"[load] images loaded: train={len(X_train_img)}, val={len(X_val_img)}, test={len(X_test_img)}")

    hog_params = {
        "orientations": baseline_cfg["hog"]["orientations"],
        "pixels_per_cell": tuple(baseline_cfg["hog"]["pixels_per_cell"]),
        "cells_per_block": tuple(baseline_cfg["hog"]["cells_per_block"]),
        "block_norm": baseline_cfg["hog"]["block_norm"],
    }

    t0 = time.time()
    X_train_feat = extract_features(X_train_img, hog_params)
    X_val_feat = extract_features(X_val_img, hog_params)
    X_test_feat = extract_features(X_test_img, hog_params)
    print(f"[hog] feature dim={X_train_feat.shape[1]} extracted in {time.time() - t0:.1f}s "
          f"(train={X_train_feat.shape}, val={X_val_feat.shape}, test={X_test_feat.shape})")

    logreg_cfg = baseline_cfg["logistic_regression"]
    selection = select_best_model(
        X_train_feat, y_train, X_val_feat, y_val,
        classes=classes,
        C_grid=logreg_cfg["C_grid"],
        max_iter=logreg_cfg["max_iter"],
        solver=logreg_cfg["solver"],
        random_state=baseline_cfg["random_state"],
        selection_metric=baseline_cfg["selection_metric"],
    )
    print(f"[select] best C={selection.best_C} by {baseline_cfg['selection_metric']} on val")

    y_val_pred = selection.best_pipeline.predict(X_val_feat)
    val_metrics = compute_metrics(y_val, y_val_pred, classes)
    print(f"[val] accuracy={val_metrics['accuracy']:.4f} f1_macro={val_metrics['f1_macro']:.4f}")

    # Test set touched here for the first and only time.
    y_test_pred = selection.best_pipeline.predict(X_test_feat)
    test_metrics = compute_metrics(y_test, y_test_pred, classes)
    print(f"[test] accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")

    fitted_scaler = selection.best_pipeline.named_steps["scaler"]
    fitted_clf = selection.best_pipeline.named_steps["clf"]
    inference_pipeline = build_inference_pipeline(hog_params, fitted_scaler, fitted_clf)

    model_path = resolve_path(baseline_cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(inference_pipeline, model_path)
    print(f"[model] saved end-to-end (image -> prediction) pipeline to {model_path}")

    reports_dir = resolve_path(baseline_cfg["output"]["results_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_path = reports_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], classes, cm_path,
        title=f"HOG + LogisticRegression (C={selection.best_C}) — Test Confusion Matrix",
    )
    print(f"[report] confusion matrix saved to {cm_path}")

    split_sizes = {name: len(manifests[name]) for name in ("train", "val", "test")}

    results_json = {
        "config": {
            "grayscale": grayscale,
            "hog": hog_params,
            "logistic_regression": logreg_cfg,
            "selection_metric": baseline_cfg["selection_metric"],
            "random_state": baseline_cfg["random_state"],
        },
        "split_sizes": split_sizes,
        "selected_C": selection.best_C,
        "hyperparameter_search": selection.search_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    results_json_path = reports_dir / "baseline_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"[report] full results JSON saved to {results_json_path}")

    config_snapshot_path = reports_dir / "baseline_config.json"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(results_json["config"], f, indent=2, default=str)
    print(f"[report] config snapshot saved to {config_snapshot_path}")

    report_md = generate_baseline_report_markdown(
        config, split_sizes, selection.search_results, selection.best_C,
        val_metrics, test_metrics, confusion_matrix_path="baseline_confusion_matrix.png",
    )
    report_md_path = reports_dir / "baseline_results.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[report] markdown report saved to {report_md_path}")


if __name__ == "__main__":
    main()
