"""Phase 3 entry point: ResNet18 transfer learning.

Loads images strictly through the frozen Phase 1 split manifests
(data/splits/{train,val,test}.csv) — never recreates or modifies the split.
"Development" (dev) data = train + val pooled; k-fold CV runs on dev only
and selects the number of training epochs; the final model is retrained on
the full dev set and evaluated exactly once on the held-out test manifest.

Usage: python -m scripts.run_phase3_transfer
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.split import SPLIT_NAMES, load_manifests
from src.eval.metrics import plot_confusion_matrix
from src.eval.report_transfer import generate_transfer_learning_report_markdown
from src.models.transfer import evaluate_on_manifest, run_cross_validation, train_final_model


def _hash_split_files(split_dir: Path) -> dict:
    hashes = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.csv"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path = split_dir / "split_summary.json"
    hashes["split_summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    config = load_config()
    classes = config["dataset"]["classes"]
    tl_cfg = config["transfer_learning"]

    split_dir = resolve_path(config["split"]["output_dir"])
    hashes_before = _hash_split_files(split_dir)
    manifests = load_manifests(split_dir)
    split_sizes = {name: len(manifests[name]) for name in SPLIT_NAMES}
    print(f"[splits] loaded frozen manifests from {split_dir}: "
          f"train={split_sizes['train']}, val={split_sizes['val']}, test={split_sizes['test']}")

    dev_manifest = (
        pd.concat([manifests["train"], manifests["val"]], ignore_index=True)
        .sort_values("filename")
        .reset_index(drop=True)
    )
    print(f"[dev] pooled train+val = {len(dev_manifest)} images (test set NOT included, NOT loaded yet)")

    raw_dir = resolve_path(config["dataset"]["raw_dir"])

    t0 = time.time()
    cv_result = run_cross_validation(dev_manifest, raw_dir, classes, tl_cfg)
    print(f"[cv] {tl_cfg['cv']['n_splits']}-fold CV over {tl_cfg['training']['max_epochs']} epochs/fold "
          f"finished in {time.time() - t0:.1f}s")
    for epoch, metrics in sorted(cv_result.mean_std_by_epoch.items()):
        sel = tl_cfg["cv"]["selection_metric"]
        marker = " <- selected" if epoch == cv_result.selected_epoch else ""
        print(f"  epoch {epoch}: {sel}={metrics[sel]['mean']:.4f}±{metrics[sel]['std']:.4f} "
              f"accuracy={metrics['accuracy']['mean']:.4f}±{metrics['accuracy']['std']:.4f}{marker}")

    t0 = time.time()
    final_model = train_final_model(dev_manifest, raw_dir, classes, tl_cfg, epochs=cv_result.selected_epoch)
    print(f"[final-train] trained on full dev set ({len(dev_manifest)} images) "
          f"for {cv_result.selected_epoch} epochs in {time.time() - t0:.1f}s")

    # Test manifest loaded and predicted on here, for the first and only time.
    test_metrics = evaluate_on_manifest(final_model, manifests["test"], raw_dir, classes, tl_cfg)
    print(f"[test] accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")

    hashes_after = _hash_split_files(split_dir)
    if hashes_before != hashes_after:
        raise RuntimeError(
            "Frozen split manifests changed during the Phase 3 run — this must never happen. "
            f"before={hashes_before} after={hashes_after}"
        )
    print("[verify] frozen split manifests byte-identical before/after run (SHA-256 checked)")

    model_path = resolve_path(tl_cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_model.state_dict(),
        "classes": classes,
        "backbone": tl_cfg["backbone"],
        "input_size": tl_cfg["input_size"],
        "imagenet_mean": tl_cfg["imagenet_mean"],
        "imagenet_std": tl_cfg["imagenet_std"],
    }, model_path)
    print(f"[model] saved to {model_path}")

    reports_dir = resolve_path(tl_cfg["output"]["results_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_path = reports_dir / "transfer_confusion_matrix.png"
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], classes, cm_path,
        title=f"ResNet18 Transfer Learning (epochs={cv_result.selected_epoch}) — Test Confusion Matrix",
    )
    print(f"[report] confusion matrix saved to {cm_path}")

    baseline_results_path = reports_dir / "baseline_results.json"
    baseline_test_metrics = None
    if baseline_results_path.exists():
        with open(baseline_results_path, "r", encoding="utf-8") as f:
            baseline_test_metrics = json.load(f)["test_metrics"]

    cv_results_json = {
        "config": tl_cfg,
        "split_sizes": split_sizes,
        "selected_epoch": cv_result.selected_epoch,
        "cv_mean_std_by_epoch": cv_result.mean_std_by_epoch,
        "cv_per_fold_per_epoch": [
            {"fold": f.fold, "per_epoch_val_metrics": f.per_epoch_val_metrics}
            for f in cv_result.folds
        ],
        "test_metrics": test_metrics,
    }
    cv_results_json_path = reports_dir / "cv_results.json"
    with open(cv_results_json_path, "w", encoding="utf-8") as f:
        json.dump(cv_results_json, f, indent=2, default=str)
    print(f"[report] full CV + test results JSON saved to {cv_results_json_path}")

    config_snapshot_path = reports_dir / "transfer_config.json"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(tl_cfg, f, indent=2, default=str)
    print(f"[report] config snapshot saved to {config_snapshot_path}")

    report_md = generate_transfer_learning_report_markdown(
        config, split_sizes, cv_result.mean_std_by_epoch, cv_result.selected_epoch,
        test_metrics, confusion_matrix_path="transfer_confusion_matrix.png",
        baseline_test_metrics=baseline_test_metrics,
    )
    report_md_path = reports_dir / "cv_results.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[report] markdown report saved to {report_md_path}")


if __name__ == "__main__":
    main()
