"""Render reports/data_integrity.md from actual computed integrity + split results.

Kept separate from integrity.py/split.py so the checks stay pure/testable and
this module is just string formatting over their output.
"""
from datetime import date
from pathlib import Path


def _fmt_bool(b: bool) -> str:
    return "PASS" if b else "FAIL"


def _basename(p: str) -> str:
    return Path(p).name


def generate_markdown_report(
    config: dict,
    n_raw_files: int,
    integrity: dict,
    split_summary: dict,
    dropped_files: list[str] | None = None,
    generated_on: date | None = None,
) -> str:
    dropped_files = dropped_files or []
    generated_on = generated_on or date.today()
    classes = config["dataset"]["classes"]
    counts = integrity["counts"]
    corruption = integrity["corruption"]
    dimensions = integrity["dimensions"]
    duplicates = integrity["duplicates"]

    lines = []
    lines.append("# Data Integrity Report — NEU-CLS")
    lines.append("")
    lines.append(f"Generated: {generated_on.isoformat()}")
    lines.append("")
    lines.append(
        "Source: local `NEU-CLS.zip` (detection-annotation layout — "
        "`train/train/images/`, `valid/valid/images/`, plus YOLO bbox "
        "`.txt` labels). Only the 1,800 `.jpg` images were extracted; the "
        "zip's own train(295)/valid(5) per-class split is a detection split "
        "and was discarded. Label files were not used — this project treats "
        "class as a whole-image classification target, parsed from the "
        "filename prefix (e.g. `crazing_10.jpg` -> `crazing`)."
    )
    lines.append("")

    lines.append("## Class List")
    lines.append("")
    for c in classes:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Dataset Counts")
    lines.append("")
    lines.append(f"- Raw files extracted: **{n_raw_files}**")
    lines.append(
        f"- Total images indexed: **{counts['total_images']}** "
        f"(expected {counts['expected_total']}) — "
        f"**{_fmt_bool(counts['total_matches'])}**"
    )
    lines.append("")
    lines.append("| Class | Count | Expected |")
    lines.append("|---|---|---|")
    for c in classes:
        lines.append(f"| {c} | {counts['counts_per_class'][c]} | {counts['images_per_class_expected']} |")
    lines.append("")
    lines.append(f"- Class balance: **{_fmt_bool(counts['balanced'])}**")
    if counts["missing_classes"]:
        lines.append(f"  - Missing classes: {counts['missing_classes']}")
    if counts["unexpected_classes"]:
        lines.append(f"  - Unexpected classes: {counts['unexpected_classes']}")
    if counts["imbalanced_classes"]:
        lines.append(f"  - Imbalanced classes: {counts['imbalanced_classes']}")
    lines.append("")

    lines.append("## Corruption Check")
    lines.append("")
    lines.append(f"- Corrupted files found: **{corruption['n_corrupted']}**")
    if corruption["corrupted_files"]:
        for item in corruption["corrupted_files"]:
            lines.append(f"  - `{_basename(item['filepath'])}`: {item['error']}")
    lines.append("")

    lines.append("## Dimensions & Channel Mode")
    lines.append("")
    lines.append(f"- Expected size: {dimensions['expected_size']} (width x height)")
    lines.append(f"- Expected mode: `{dimensions['expected_mode']}`")
    lines.append(f"- Size distribution: {dimensions['size_distribution']}")
    lines.append(f"- Mode distribution: {dimensions['mode_distribution']}")
    lines.append(f"- All images uniform size: **{_fmt_bool(dimensions['all_uniform_size'])}**")
    lines.append(f"- All images uniform mode: **{_fmt_bool(dimensions['all_uniform_mode'])}**")
    lines.append("")

    lines.append("## Exact Duplicate Check (SHA-256)")
    lines.append("")
    lines.append(f"- Unique file hashes: {duplicates['n_unique_hashes']}")
    lines.append(f"- Duplicate groups found: {duplicates['n_duplicate_groups']}")
    lines.append(f"- Duplicate files (excess copies): {duplicates['n_duplicate_files']}")
    lines.append(f"- Has duplicates: **{_fmt_bool(not duplicates['has_duplicates'])}** (PASS = no duplicates)")
    if duplicates["duplicate_groups"]:
        for group in duplicates["duplicate_groups"]:
            lines.append(f"  - {[_basename(p) for p in group]}")
    lines.append("")

    lines.append("## Deduplication (Remediation)")
    lines.append("")
    if dropped_files:
        lines.append(
            f"The exact-duplicate check above found {integrity['duplicates']['n_duplicate_files']} "
            "duplicate file(s). Byte-identical images must never land in different "
            "splits (that would leak the same image between, e.g., train and val), "
            "so before building the train/val/test split, one copy of each "
            "duplicate group was dropped (keeping the alphabetically-first "
            "filename):"
        )
        for f in dropped_files:
            lines.append(f"  - dropped `{_basename(f)}`")
        n_after = counts["total_images"] - len(dropped_files)
        lines.append("")
        lines.append(
            f"Split is therefore built from **{n_after}** unique images, not "
            f"the raw {counts['total_images']}."
        )
    else:
        lines.append("No duplicates found — no deduplication needed before splitting.")
    lines.append("")

    lines.append("## Overall Integrity Result")
    lines.append("")
    lines.append(
        f"Raw data, before deduplication: **{_fmt_bool(integrity['all_checks_passed'])}** "
        "(counts/corruption/dimensions all pass; duplicate check is the one "
        "finding — see above, resolved via deduplication before splitting)."
        if integrity["duplicates"]["has_duplicates"]
        and integrity["counts"]["balanced"]
        and integrity["counts"]["total_matches"]
        and integrity["corruption"]["n_corrupted"] == 0
        and integrity["dimensions"]["all_uniform_size"]
        and integrity["dimensions"]["all_uniform_mode"]
        else f"**{_fmt_bool(integrity['all_checks_passed'])}**"
    )
    lines.append("")

    lines.append("## Train/Val/Test Split")
    lines.append("")
    lines.append(
        f"Fixed stratified split, seed=`{split_summary['seed']}`, fractions="
        f"train={split_summary['fractions']['train']}, "
        f"val={split_summary['fractions']['val']}, "
        f"test={split_summary['fractions']['test']}, built from the "
        f"post-deduplication image pool. "
        "Manifests are frozen CSV files under `data/splits/` (filename + class "
        "only, no absolute paths) — every later phase must load these same "
        "files rather than re-splitting."
    )
    lines.append("")
    lines.append("| Split | Total | " + " | ".join(classes) + " |")
    lines.append("|---|---|" + "---|" * len(classes))
    for split_name in ("train", "val", "test"):
        c = split_summary["counts"][split_name]
        per_class = c["per_class"]
        row = [str(per_class.get(cls, 0)) for cls in classes]
        lines.append(f"| {split_name} | {c['total']} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Early Visual Notes")
    lines.append("")
    lines.append(
        "- `rolled-in_scale` and `crazing` are the two classes most often "
        "flagged in the NEU-CLS literature as visually similar (both present "
        "as diffuse, low-contrast linear/mottled texture rather than a sharp "
        "localized defect like `scratches` or `patches`). This is a hypothesis "
        "to verify empirically in Phase 4 (confusion matrix), not a confirmed "
        "finding from Phase 1."
    )
    lines.append(
        "- All images are uniform 200x200 and stored as JPEG in RGB mode "
        "despite the underlying content being grayscale (see Dimensions & "
        "Channel Mode above) — worth keeping in mind for Phase 3 preprocessing "
        "(no channel-count mismatch to handle, but redundant channels)."
    )
    lines.append(
        "- The source zip is a detection-annotation release (YOLO bbox labels "
        "per image, 295/5 train/valid split). This project only needs "
        "classification labels, so bbox labels were ignored and a fresh "
        "70/15/15 stratified split was built from the pooled 1,800 images."
    )
    lines.append("")

    return "\n".join(lines)
