"""Render reports/cv_results.md from actual computed Phase 3 results.

Kept separate from the model/CV code so those stay pure/testable and this
module is just string formatting over their output.
"""
from datetime import date


def generate_transfer_learning_report_markdown(
    config: dict,
    split_sizes: dict,
    cv_result_summary: dict,
    selected_epoch: int,
    test_metrics: dict,
    confusion_matrix_path: str,
    baseline_test_metrics: dict | None,
    generated_on: date | None = None,
) -> str:
    generated_on = generated_on or date.today()
    classes = config["dataset"]["classes"]
    tl_cfg = config["transfer_learning"]

    lines = []
    lines.append("# Phase 3 — ResNet18 Transfer Learning")
    lines.append("")
    lines.append(f"Generated: {generated_on.isoformat()}")
    lines.append("")
    lines.append(
        "Loads images strictly through the frozen Phase 1 manifests "
        "(`data/splits/{train,val,test}.csv`) — the split is not recreated "
        "or modified here. \"Development\" (dev) data = train + val pooled. "
        "Stratified k-fold cross-validation runs on dev only and is used to "
        "select the number of training epochs (best mean val f1_macro "
        "across folds). The final model is retrained from scratch on the "
        "full dev set for that many epochs, and the test manifest is loaded "
        "and predicted on exactly once, after the epoch count was already "
        "fixed by CV."
    )
    lines.append("")

    lines.append("## Split Sizes (from frozen Phase 1 manifests)")
    lines.append("")
    lines.append(f"- train: {split_sizes['train']}")
    lines.append(f"- val: {split_sizes['val']}")
    lines.append(f"- dev (train + val, used for CV): {split_sizes['train'] + split_sizes['val']}")
    lines.append(f"- test (untouched until final evaluation): {split_sizes['test']}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Backbone: `{tl_cfg['backbone']}` (ImageNet-pretrained: `{tl_cfg['pretrained']}`)")
    lines.append(f"- Freeze backbone: `{tl_cfg['freeze_backbone']}` (only the replaced FC head is trained)")
    lines.append(f"- Input size: `{tl_cfg['input_size']}`")
    aug = tl_cfg["augmentation"]
    lines.append(
        f"- Augmentation (train only): RandomResizedCrop(scale={aug['random_resized_crop_scale']}), "
        f"horizontal_flip={aug['horizontal_flip']}, vertical_flip={aug['vertical_flip']}, "
        f"rotation=±{aug['random_rotation_degrees']}° — no color jitter (grayscale content)"
    )
    train_cfg = tl_cfg["training"]
    lines.append(
        f"- Optimizer: `{train_cfg['optimizer']}`, lr={train_cfg['learning_rate']}, "
        f"weight_decay={train_cfg['weight_decay']}, batch_size={train_cfg['batch_size']}"
    )
    lines.append(f"- Max epochs per CV fold: {train_cfg['max_epochs']}")
    lines.append(f"- CV folds: {tl_cfg['cv']['n_splits']}, selection metric: `{tl_cfg['cv']['selection_metric']}`")
    lines.append(f"- Seed: {tl_cfg['seed']}")
    lines.append(f"- **Selected epoch count (via CV): {selected_epoch}**")
    lines.append("")

    lines.append("## k-Fold Cross-Validation Results (Development Set)")
    lines.append("")
    lines.append("Mean ± std across folds, per epoch (val metrics):")
    lines.append("")
    lines.append("| Epoch | val accuracy (mean±std) | val f1_macro (mean±std) | selected |")
    lines.append("|---|---|---|---|")
    for epoch in sorted(cv_result_summary.keys()):
        acc = cv_result_summary[epoch]["accuracy"]
        f1 = cv_result_summary[epoch][tl_cfg["cv"]["selection_metric"]]
        marker = "**yes**" if epoch == selected_epoch else ""
        lines.append(
            f"| {epoch} | {acc['mean']:.4f} ± {acc['std']:.4f} | "
            f"{f1['mean']:.4f} ± {f1['std']:.4f} | {marker} |"
        )
    lines.append("")

    lines.append("## Final Test Results (Touched Once)")
    lines.append("")
    lines.append(f"- Accuracy: **{test_metrics['accuracy']:.4f}**")
    lines.append(f"- Precision (macro): {test_metrics['precision_macro']:.4f}, (weighted): {test_metrics['precision_weighted']:.4f}")
    lines.append(f"- Recall (macro): {test_metrics['recall_macro']:.4f}, (weighted): {test_metrics['recall_weighted']:.4f}")
    lines.append(f"- F1 (macro): {test_metrics['f1_macro']:.4f}, (weighted): {test_metrics['f1_weighted']:.4f}")
    lines.append("")

    lines.append("### Per-Class Metrics (Test)")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for cls in classes:
        pc = test_metrics["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    lines.append("")

    lines.append("### Confusion Matrix (Test)")
    lines.append("")
    lines.append(f"![Confusion Matrix]({confusion_matrix_path})")
    lines.append("")
    header = "| True \\\\ Pred | " + " | ".join(classes) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(classes))
    for i, cls in enumerate(classes):
        row = test_metrics["confusion_matrix"][i]
        lines.append(f"| {cls} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    if baseline_test_metrics is not None:
        lines.append("## Comparison vs. Phase 2 HOG + Logistic Regression Baseline (Test Set)")
        lines.append("")
        lines.append("| Metric | Baseline (HOG+LogReg) | ResNet18 Transfer | Delta |")
        lines.append("|---|---|---|---|")
        for metric_key, label in [
            ("accuracy", "Accuracy"),
            ("precision_macro", "Precision (macro)"),
            ("recall_macro", "Recall (macro)"),
            ("f1_macro", "F1 (macro)"),
        ]:
            b = baseline_test_metrics[metric_key]
            t = test_metrics[metric_key]
            lines.append(f"| {label} | {b:.4f} | {t:.4f} | {t - b:+.4f} |")
        lines.append("")

    return "\n".join(lines)
