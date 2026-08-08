"""Render reports/error_analysis.md from actual computed Phase 4 results.

Kept separate from the analysis code so those stay pure/testable and this
module is just string formatting over their output.
"""
from datetime import date


def _confusion_table(classes: list[str], cm: list[list[int]]) -> list[str]:
    lines = ["| True \\\\ Pred | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for i, cls in enumerate(classes):
        row = cm[i]
        lines.append(f"| {cls} | " + " | ".join(str(v) for v in row) + " |")
    return lines


def _per_class_table(classes: list[str], per_class: dict) -> list[str]:
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls in classes:
        pc = per_class[cls]
        lines.append(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    return lines


def _confusion_pairs_table(pairs: list[dict]) -> list[str]:
    if not pairs:
        return ["_No misclassifications found._"]
    lines = ["| True | Predicted | Count |", "|---|---|---|"]
    for p in pairs:
        lines.append(f"| {p['true']} | {p['pred']} | {p['count']} |")
    return lines


def generate_error_analysis_report_markdown(ctx: dict, generated_on: date | None = None) -> str:
    generated_on = generated_on or date.today()
    classes = ctx["classes"]

    lines = []
    lines.append("# Phase 4 — Error Analysis")
    lines.append("")
    lines.append(f"Generated: {generated_on.isoformat()}")
    lines.append("")
    lines.append(
        "This phase re-runs **inference only** (no training) for the already-trained "
        "Phase 2 (`models/baseline.joblib`) and Phase 3 (`models/resnet18_finetuned.pt`) "
        "models on the frozen test manifest (`data/splits/test.csv`), to recover the "
        "per-sample predictions and confidences that Phase 2/3 only summarized as "
        "aggregate metrics. Neither model, the split, nor the test data were modified. "
        "As a consistency check, the re-computed aggregate test metrics for both models "
        f"were verified to match the previously reported Phase 2/3 results "
        f"(max abs difference: {ctx['max_metric_diff']:.2e})."
    )
    lines.append("")

    # --- Full confusion matrices -------------------------------------------------
    lines.append("## Full Confusion Matrix — ResNet18 (Test)")
    lines.append("")
    lines.extend(_confusion_table(classes, ctx["resnet"]["confusion_matrix"]))
    lines.append("")
    lines.append(f"![ResNet18 Confusion Matrix]({ctx['artifacts']['resnet_cm']})")
    lines.append("")

    lines.append("## Full Confusion Matrix — HOG + Logistic Regression Baseline (Test)")
    lines.append("")
    lines.extend(_confusion_table(classes, ctx["baseline"]["confusion_matrix"]))
    lines.append("")
    lines.append(f"![Baseline Confusion Matrix]({ctx['artifacts']['baseline_cm']})")
    lines.append("")

    # --- Per-class metrics ---------------------------------------------------
    lines.append("## Per-Class Precision / Recall / F1 — ResNet18 (Test)")
    lines.append("")
    lines.extend(_per_class_table(classes, ctx["resnet"]["per_class"]))
    lines.append("")

    lines.append("## Per-Class Precision / Recall / F1 — Baseline (Test)")
    lines.append("")
    lines.extend(_per_class_table(classes, ctx["baseline"]["per_class"]))
    lines.append("")

    lines.append(f"![Per-Class F1 Comparison]({ctx['artifacts']['f1_comparison']})")
    lines.append("")

    # --- Most important confusion pairs --------------------------------------
    lines.append("## Most Important Confusion Pairs")
    lines.append("")
    lines.append(f"### ResNet18 (top {len(ctx['resnet']['top_pairs'])} by count)")
    lines.append("")
    lines.extend(_confusion_pairs_table(ctx["resnet"]["top_pairs"]))
    lines.append("")
    lines.append(f"### Baseline (top {len(ctx['baseline']['top_pairs'])} by count)")
    lines.append("")
    lines.extend(_confusion_pairs_table(ctx["baseline"]["top_pairs"]))
    lines.append("")

    # --- patches -> crazing explicit check -----------------------------------
    pc = ctx["patches_crazing"]
    lines.append("## Explicit Check: `patches` → `crazing` Confusion (Phase 2 Finding)")
    lines.append("")
    lines.append(
        f"Phase 2's error analysis found {pc['baseline_count']} of {pc['patches_support']} "
        "true `patches` test images misclassified as `crazing` by the HOG+LogReg "
        f"baseline ({pc['baseline_count'] / pc['patches_support']:.0%} of the class). "
        f"ResNet18 misclassifies {pc['resnet_count']} `patches` images as `crazing` on "
        f"the same test set — "
        + ("**fully resolved**." if pc["resnet_count"] == 0 else f"**reduced but not eliminated** ({pc['resnet_count']} remaining).")
    )
    lines.append("")

    # --- representative examples ----------------------------------------------
    lines.append("## Representative Predictions — ResNet18")
    lines.append("")
    lines.append("### Correct (highest confidence per class)")
    lines.append("")
    lines.append("| Filename | Class | Confidence |")
    lines.append("|---|---|---|")
    for r in ctx["resnet"]["representative_correct"]:
        lines.append(f"| {r['filename']} | {r['true_class']} | {r['confidence']:.4f} |")
    lines.append("")
    lines.append(f"### All Misclassified ({len(ctx['resnet']['representative_incorrect'])} of {ctx['resnet']['n_test']})")
    lines.append("")
    lines.append("| Filename | True | Predicted | Confidence |")
    lines.append("|---|---|---|---|")
    for r in ctx["resnet"]["representative_incorrect"]:
        lines.append(f"| {r['filename']} | {r['true_class']} | {r['pred_class']} | {r['confidence']:.4f} |")
    lines.append("")
    if ctx["resnet"]["representative_incorrect"]:
        lines.append(f"![Misclassified Examples]({ctx['artifacts']['misclassified_grid']})")
        lines.append("")

    lines.append("## Representative Predictions — Baseline (`patches` → `crazing` examples)")
    lines.append("")
    lines.append("| Filename | True | Predicted | Confidence |")
    lines.append("|---|---|---|---|")
    for r in ctx["baseline"]["patches_crazing_examples"]:
        lines.append(f"| {r['filename']} | {r['true_class']} | {r['pred_class']} | {r['confidence']:.4f} |")
    lines.append("")

    # --- systematic vs isolated -----------------------------------------------
    lines.append("## Systematic vs. Isolated Errors — ResNet18")
    lines.append("")
    lines.append(
        "For each true class: total test errors, how many *distinct* wrong "
        "classes those errors were spread across, and the single most common "
        "wrong prediction (if any). A class whose errors concentrate into one "
        "dominant wrong class is a systematic confusion; a class whose few "
        "errors are spread across several different wrong classes looks more "
        "like isolated noise than a learned confusion."
    )
    lines.append("")
    lines.append("| Class | Correct | Total Errors | Distinct Wrong Classes | Dominant Confusion |")
    lines.append("|---|---|---|---|---|")
    for cls in classes:
        c = ctx["resnet"]["concentration"][cls]
        dom = f"{c['dominant_confusion']['pred']} (x{c['dominant_confusion']['count']})" if c["dominant_confusion"] else "—"
        lines.append(f"| {cls} | {c['correct']} | {c['total_errors']} | {c['distinct_confused_with']} | {dom} |")
    lines.append("")
    lines.append(ctx["systematic_narrative"])
    lines.append("")

    # --- error overlap with baseline -------------------------------------------
    eo = ctx["error_overlap"]
    lines.append("## Error Overlap: ResNet18 vs. Baseline (Same 270 Test Images)")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for k, v in eo["counts"].items():
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines.append("")
    n_persistent = eo["counts"]["both_wrong"]
    n_new_errors = eo["counts"]["resnet_only_wrong"]
    new_errors_clause = (
        "no image the baseline got right is missed by ResNet18 — no new errors were introduced."
        if n_new_errors == 0
        else f"{n_new_errors} image(s) the baseline got right are missed by ResNet18 (new errors introduced by the switch)."
    )
    lines.append(
        f"{n_persistent} test image(s) fool **both** models — these are the hardest "
        "examples in the test set, independent of model architecture. "
        f"{eo['counts']['baseline_only_wrong']} image(s) that the baseline got wrong "
        "are correctly classified by ResNet18 (fixed by transfer learning); "
        + new_errors_clause
    )
    lines.append("")
    if eo["both_wrong_examples"]:
        lines.append("### Images Both Models Get Wrong")
        lines.append("")
        lines.append("| Filename | True | Baseline Pred | ResNet18 Pred |")
        lines.append("|---|---|---|---|")
        for r in eo["both_wrong_examples"]:
            lines.append(f"| {r['filename']} | {r['true_class']} | {r['baseline_pred']} | {r['resnet_pred']} |")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Per-sample predictions (both models, all {ctx['resnet']['n_test']} test images): `{ctx['artifacts']['predictions_csv']}`")
    lines.append(f"- ResNet18 confusion matrix: `{ctx['artifacts']['resnet_cm']}`")
    lines.append(f"- Baseline confusion matrix: `{ctx['artifacts']['baseline_cm']}`")
    lines.append(f"- Per-class F1 comparison chart: `{ctx['artifacts']['f1_comparison']}`")
    if ctx["resnet"]["representative_incorrect"]:
        lines.append(f"- ResNet18 misclassified-examples grid: `{ctx['artifacts']['misclassified_grid']}`")
    lines.append("")

    return "\n".join(lines)
