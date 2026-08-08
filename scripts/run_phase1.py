"""Phase 1 entry point: extract -> index -> integrity check -> split -> report.

Usage: python -m scripts.run_phase1
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.dedup import resolve_duplicates
from src.data.extract import extract_dataset
from src.data.index import build_image_index
from src.data.integrity import run_integrity_checks
from src.data.report import generate_markdown_report
from src.data.split import save_manifests, stratified_split


def main() -> None:
    config = load_config()

    zip_path = resolve_path(config["dataset"]["zip_path"])
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    extract_dataset(zip_path, raw_dir)
    n_raw_files = len(list(raw_dir.iterdir()))
    print(f"[extract] {n_raw_files} files in {raw_dir}")

    df = build_image_index(raw_dir, valid_extensions=tuple(config["dataset"]["valid_extensions"]))
    print(f"[index] {len(df)} images indexed")

    # Integrity check runs on the full raw set — it must describe the raw
    # data honestly, including any duplicates found.
    integrity = run_integrity_checks(df, config)
    print(f"[integrity] all_checks_passed={integrity['all_checks_passed']}")
    print(json.dumps(integrity["counts"], indent=2))

    # Deduplicate before splitting: exact-duplicate images must never end up
    # split across train/val/test (that would leak signal between splits).
    deduped_df, dropped_files = resolve_duplicates(df, integrity["duplicates"]["duplicate_groups"])
    if dropped_files:
        print(f"[dedup] dropped {len(dropped_files)} exact-duplicate file(s) before splitting: {dropped_files}")

    split_cfg = config["split"]
    splits = stratified_split(
        deduped_df, train=split_cfg["train"], val=split_cfg["val"], test=split_cfg["test"], seed=config["seed"]
    )
    split_output_dir = resolve_path(split_cfg["output_dir"])
    fractions = {"train": split_cfg["train"], "val": split_cfg["val"], "test": split_cfg["test"]}
    save_manifests(splits, split_output_dir, seed=config["seed"], fractions=fractions)
    print(f"[split] manifests written to {split_output_dir}")

    with open(split_output_dir / "split_summary.json", "r", encoding="utf-8") as f:
        split_summary = json.load(f)

    report_md = generate_markdown_report(config, n_raw_files, integrity, split_summary, dropped_files=dropped_files)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_integrity.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[report] written to {report_path}")

    integrity_json_path = reports_dir / "data_integrity.json"
    with open(integrity_json_path, "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2, default=str)
    print(f"[report] raw integrity JSON written to {integrity_json_path}")

    # The pipeline's pass/fail gate is distinct from integrity["all_checks_passed"]:
    # the latter is an honest description of the RAW data (duplicates included),
    # while unresolved-issue gating below accounts for the deduplication step
    # actually taken. Corruption/dimension problems have no remediation here and
    # must still fail the run; a duplicate finding does not, once resolved.
    unresolved_issues = (
        not integrity["counts"]["total_matches"]
        or not integrity["counts"]["balanced"]
        or integrity["corruption"]["n_corrupted"] > 0
        or not integrity["dimensions"]["all_uniform_size"]
        or not integrity["dimensions"]["all_uniform_mode"]
    )
    if unresolved_issues:
        print("\nWARNING: unresolved integrity issues found. See report for details.")
        sys.exit(1)
    elif integrity["duplicates"]["has_duplicates"]:
        print(
            f"\nNOTE: {integrity['duplicates']['n_duplicate_files']} duplicate file(s) found in raw data "
            "and resolved via deduplication before splitting. See report for details."
        )


if __name__ == "__main__":
    main()
