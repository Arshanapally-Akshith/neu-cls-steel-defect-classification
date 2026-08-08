"""Render reports/gradcam_analysis.md from actual computed Phase 5 results.

The quantitative sections (tables, border-energy numbers, image embeds) are
fully generated here. The QUALITATIVE findings — whether the highlighted
regions actually correspond to defect texture vs. borders/background/
lighting — require a human/model to actually LOOK at the generated heatmap
images; this module leaves clearly marked placeholder paragraphs for that,
filled in afterward by direct visual inspection (see scripts/run_phase5_gradcam.py
and the Phase 5 workflow notes). Never fabricate that content from numbers alone.
"""
from datetime import date

PLACEHOLDER = "**[VISUAL INSPECTION PENDING — see reports/gradcam/*.png; do not treat this placeholder as a finding.]**"


def _examples_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    """columns: list of (row_key, header_label)."""
    lines = ["| " + " | ".join(h for _, h in columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = []
        for key, _h in columns:
            v = r.get(key, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def generate_gradcam_report_markdown(ctx: dict, generated_on: date | None = None) -> str:
    generated_on = generated_on or date.today()

    lines = []
    lines.append("# Phase 5 — Grad-CAM Verification")
    lines.append("")
    lines.append(f"Generated: {generated_on.isoformat()}")
    lines.append("")
    lines.append(
        "**What this phase does and does not show.** Grad-CAM (Selvaraju et al., 2017) "
        "visualizes where the gradient of a predicted class's score is concentrated in "
        "the final convolutional feature map (`layer4[-1]` of the frozen ResNet18 "
        "backbone, a 7x7 spatial grid upsampled to 224x224 for display). This is "
        "**evidence about local gradient sensitivity, not proof of causal reasoning** — "
        "a high-activation region is a place the model's score is sensitive to, not "
        "necessarily the *reason* it chose that class, and Grad-CAM cannot see anything "
        "the frozen, generic-ImageNet backbone's 512-dim pooled features didn't already "
        "discard. Findings below are reported honestly, including if the model appears "
        "to attend to non-defect regions — that would be a legitimate finding about this "
        "specific model, not a failure of the analysis."
    )
    lines.append("")
    lines.append(
        f"Images were selected from Phase 4's per-sample predictions table "
        f"(`{ctx['predictions_csv']}`) — never re-derived. Neither the ResNet18 model nor "
        "the frozen test split were modified; both were re-hashed before and after this "
        f"run (see console log). {ctx['n_selected']} test images were explained "
        f"({ctx['n_correct_representative']} correct-representative, one per class; "
        f"{ctx['n_confusion_prone']} additional low-confidence-correct examples from "
        f"confusion-prone classes; {ctx['n_incorrect']} incorrect — all of ResNet18's "
        "test errors, per Phase 4)."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(f"- Target layer: `model.layer4[-1]` (last BasicBlock of ResNet18's final conv stage)")
    lines.append(f"- CAM: global-average-pooled gradient weights over channel activations, ReLU'd, normalized to [0,1] (standard Grad-CAM)")
    lines.append(f"- Overlay: `jet` colormap, alpha={ctx['overlay_alpha']}, blended over the same 224x224 resize the model sees")
    lines.append(
        "- Border-energy fraction: a quantitative proxy computed on the raw 7x7 CAM — "
        "the share of total activation energy in the outer ~15% border ring. High values "
        "are *consistent with* (not proof of) a border/background shortcut; this number "
        "supports the visual read below, it doesn't replace it."
    )
    lines.append("")

    # --- Correct representative -------------------------------------------------
    lines.append("## Correctly Classified — One Representative per Class")
    lines.append("")
    lines.append(f"![Correct Examples]({ctx['artifacts']['correct_grid']})")
    lines.append("")
    lines.extend(_examples_table(
        ctx["correct_representative"],
        [("filename", "Filename"), ("true_class", "Class"), ("resnet_confidence", "Confidence"), ("border_energy_fraction_pred", "Border Energy Frac.")],
    ))
    lines.append("")
    lines.append("### Visual assessment — correct predictions")
    lines.append("")
    lines.append(ctx.get("qualitative_correct", PLACEHOLDER))
    lines.append("")

    # --- Confusion-prone extra ---------------------------------------------------
    lines.append("## Confusion-Prone Classes — Additional Low-Confidence Correct Examples")
    lines.append("")
    lines.append(
        f"Classes Phase 4 flagged via top confusion pairs (`{', '.join(ctx['confusion_prone_classes'])}`), "
        "showing their least-confident still-correct predictions — the borderline cases "
        "most informative about whether attention degrades before the prediction does."
    )
    lines.append("")
    if ctx["confusion_prone_correct"]:
        lines.append(f"![Confusion-Prone Examples]({ctx['artifacts']['confusion_prone_grid']})")
        lines.append("")
        lines.extend(_examples_table(
            ctx["confusion_prone_correct"],
            [("filename", "Filename"), ("true_class", "Class"), ("resnet_confidence", "Confidence"), ("border_energy_fraction_pred", "Border Energy Frac.")],
        ))
    else:
        lines.append("_No additional examples (all confusion-prone classes' correct examples were already used above)._")
    lines.append("")
    lines.append("### Visual assessment — confusion-prone classes")
    lines.append("")
    lines.append(ctx.get("qualitative_confusion_prone", PLACEHOLDER))
    lines.append("")

    # --- Incorrect ----------------------------------------------------------------
    lines.append("## Incorrectly Classified — All ResNet18 Test Errors")
    lines.append("")
    lines.append(
        "Each misclassified image gets TWO CAMs: one for the class the model actually "
        "predicted (what evidence drove the wrong answer) and one for the true class "
        "(whether evidence for the right answer was present but was outweighed, or "
        "simply wasn't there)."
    )
    lines.append("")
    lines.append(f"![Incorrect Examples]({ctx['artifacts']['incorrect_grid']})")
    lines.append("")
    lines.extend(_examples_table(
        ctx["incorrect"],
        [
            ("filename", "Filename"), ("true_class", "True"), ("resnet_pred", "Predicted"), ("resnet_confidence", "Confidence"),
            ("border_energy_fraction_pred", "Border Frac. (pred CAM)"), ("border_energy_fraction_true", "Border Frac. (true CAM)"),
        ],
    ))
    lines.append("")
    lines.append("### Visual assessment — misclassified examples")
    lines.append("")
    lines.append(ctx.get("qualitative_incorrect", PLACEHOLDER))
    lines.append("")

    # --- Quantitative summary -----------------------------------------------------
    lines.append("## Quantitative Summary: Border-Energy Fraction by Group")
    lines.append("")
    lines.append("| Group | n | Mean | Median | Min | Max |")
    lines.append("|---|---|---|---|---|---|")
    for group_name, stats in ctx["border_energy_summary"].items():
        lines.append(
            f"| {group_name} | {stats['n']} | {stats['mean']:.3f} | {stats['median']:.3f} | "
            f"{stats['min']:.3f} | {stats['max']:.3f} |"
        )
    lines.append("")
    lines.append(
        "A uniform 7x7 CAM with all energy spread evenly would have a border fraction "
        "of about 0.49 (a 1-cell-wide ring at 15% width per side covers 24 of the grid's "
        "49 cells). Values below that indicate energy concentrated toward the center; "
        "values above it lean toward the edge — though this alone doesn't distinguish a "
        "genuinely edge-focused CAM from one that's simply diffuse. This number is a "
        "coarse signal, read alongside the images above, not instead of them."
    )
    lines.append("")

    # --- Overall assessment --------------------------------------------------------
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append(ctx.get("qualitative_overall", PLACEHOLDER))
    lines.append("")

    # --- Limitations ----------------------------------------------------------------
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Evidence, not proof.** Grad-CAM shows gradient sensitivity, not a causal "
        "explanation. A model can produce a plausible-looking heatmap while still "
        "reasoning in ways that don't generalize (or vice versa: a diffuse heatmap "
        "doesn't necessarily mean the model reasoned poorly)."
    )
    lines.append(
        "- **Coarse resolution.** The CAM's native resolution is 7x7 (image downsampled "
        "32x by ResNet18's stride); each cell corresponds to a 32x32 region of the 224x224 "
        "input. Fine-grained defect boundaries cannot be resolved at this granularity — "
        "\"the model attends to the defect\" can only be claimed at the scale of roughly "
        "a seventh of the image, not pixel-precise."
    )
    lines.append(
        "- **Frozen, generic-ImageNet backbone.** Only the final linear layer was trained "
        "(Phase 3); the conv features themselves were never adapted to steel-defect "
        "imagery specifically. Any attention pattern reflects what ImageNet-pretrained "
        "features happen to respond to, filtered through a linear readout — not features "
        "learned for this task."
    )
    lines.append(
        f"- **Small, non-random sample.** {ctx['n_selected']} images were deliberately "
        "selected (representative/confusion-prone/all-errors), not randomly sampled — "
        "appropriate for illustrating specific cases, but this is not a statistically "
        "powered claim about the model's attention in general."
    )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Per-sample Grad-CAM summary (all {ctx['n_selected']} explained images): `{ctx['artifacts']['summary_csv']}`")
    lines.append(f"- Correct-examples grid: `{ctx['artifacts']['correct_grid']}`")
    lines.append(f"- Confusion-prone examples grid: `{ctx['artifacts']['confusion_prone_grid']}`")
    lines.append(f"- Incorrect-examples grid: `{ctx['artifacts']['incorrect_grid']}`")
    lines.append(f"- Individual heatmap overlays (one/two PNGs per image): `{ctx['artifacts']['heatmaps_dir']}/`")
    lines.append("")

    return "\n".join(lines)
