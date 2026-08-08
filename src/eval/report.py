"""Render reports/baseline_results.md from actual computed results.

Kept separate from the model/metrics code so those stay pure/testable and
this module is just string formatting over their output.
"""
from datetime import date


def generate_baseline_report_markdown(
    config: dict,
    split_sizes: dict,
    search_results: list,
    best_C: float,
    val_metrics: dict,
    test_metrics: dict,
    confusion_matrix_path: str,
    generated_on: date | None = None,
) -> str:
    generated_on = generated_on or date.today()
    classes = config["dataset"]["classes"]
    baseline_cfg = config["baseline"]

    lines = []
    lines.append("# Phase 2 — Classical Baseline: HOG + Logistic Regression")
    lines.append("")
    lines.append(f"Generated: {generated_on.isoformat()}")
    lines.append("")
    lines.append(
        "Loads images strictly through the frozen Phase 1 manifests "
        "(`data/splits/{train,val,test}.csv`) — the split is not recreated "
        "or modified here. HOG features are extracted per image; the "
        "feature scaler and Logistic Regression classifier are fit ONLY on "
        "the train split. The val split is used exclusively to pick the "
        "regularization strength `C`. The test split is touched exactly "
        "once, after `C` was already chosen, for final reporting."
    )
    lines.append("")

    lines.append("## Split Sizes (from frozen Phase 1 manifests)")
    lines.append("")
    lines.append(f"- train: {split_sizes['train']}")
    lines.append(f"- val: {split_sizes['val']}")
    lines.append(f"- test: {split_sizes['test']}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Grayscale input: `{baseline_cfg['grayscale']}`")
    lines.append(f"- HOG orientations: `{baseline_cfg['hog']['orientations']}`")
    lines.append(f"- HOG pixels_per_cell: `{baseline_cfg['hog']['pixels_per_cell']}`")
    lines.append(f"- HOG cells_per_block: `{baseline_cfg['hog']['cells_per_block']}`")
    lines.append(f"- HOG block_norm: `{baseline_cfg['hog']['block_norm']}`")
    lines.append(f"- Logistic Regression solver: `{baseline_cfg['logistic_regression']['solver']}`")
    lines.append(f"- Logistic Regression max_iter: `{baseline_cfg['logistic_regression']['max_iter']}`")
    lines.append(f"- C grid searched (on val): `{baseline_cfg['logistic_regression']['C_grid']}`")
    lines.append(f"- Selection metric (val): `{baseline_cfg['selection_metric']}`")
    lines.append(f"- Random state: `{baseline_cfg['random_state']}`")
    lines.append(f"- **Selected C: `{best_C}`**")
    lines.append("")

    lines.append("## Hyperparameter Search (Validation Set)")
    lines.append("")
    lines.append("| C | val accuracy | val precision (macro) | val recall (macro) | val F1 (macro) | selected |")
    lines.append("|---|---|---|---|---|---|")
    for r in search_results:
        m = r["val_metrics"]
        marker = "**yes**" if r["C"] == best_C else ""
        lines.append(
            f"| {r['C']} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
            f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {marker} |"
        )
    lines.append("")

    lines.append("## Validation Results (Selected Model, C=" + str(best_C) + ")")
    lines.append("")
    lines.append(f"- Accuracy: **{val_metrics['accuracy']:.4f}**")
    lines.append(f"- Precision (macro): {val_metrics['precision_macro']:.4f}, (weighted): {val_metrics['precision_weighted']:.4f}")
    lines.append(f"- Recall (macro): {val_metrics['recall_macro']:.4f}, (weighted): {val_metrics['recall_weighted']:.4f}")
    lines.append(f"- F1 (macro): {val_metrics['f1_macro']:.4f}, (weighted): {val_metrics['f1_weighted']:.4f}")
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

    return "\n".join(lines)
