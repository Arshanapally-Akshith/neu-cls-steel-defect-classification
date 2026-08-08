"""Dataset integrity checks: counts, class balance, corruption, dimensions,
channel mode, and exact duplicates.

`run_integrity_checks` is the single entry point; it returns a plain dict
(JSON-serializable) so it can be inspected directly, written to disk, or
consumed by tests without re-running the (slower) file-reading checks.
"""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


def check_counts(df: pd.DataFrame, expected_total: int, expected_classes: list[str], images_per_class: int) -> dict:
    counts = df["class"].value_counts().to_dict()
    missing_classes = sorted(set(expected_classes) - set(counts))
    unexpected_classes = sorted(set(counts) - set(expected_classes))
    imbalanced = {
        cls: n for cls, n in counts.items() if cls in expected_classes and n != images_per_class
    }
    return {
        "total_images": len(df),
        "expected_total": expected_total,
        "total_matches": len(df) == expected_total,
        "counts_per_class": {cls: int(counts.get(cls, 0)) for cls in expected_classes},
        "images_per_class_expected": images_per_class,
        "missing_classes": missing_classes,
        "unexpected_classes": unexpected_classes,
        "imbalanced_classes": imbalanced,
        "balanced": not missing_classes and not unexpected_classes and not imbalanced,
    }


def check_corruption(df: pd.DataFrame) -> dict:
    corrupted = []
    for filepath in df["filepath"]:
        try:
            with Image.open(filepath) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupted.append({"filepath": filepath, "error": str(e)})
    return {"n_corrupted": len(corrupted), "corrupted_files": corrupted}


def check_dimensions(df: pd.DataFrame, expected_size: tuple[int, int], expected_mode: str) -> dict:
    mismatched_size = []
    mismatched_mode = []
    size_counts: dict = {}
    mode_counts: dict = {}

    for filepath in df["filepath"]:
        with Image.open(filepath) as img:
            size = img.size  # (width, height)
            mode = img.mode
        size_counts[size] = size_counts.get(size, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if tuple(size) != tuple(expected_size):
            mismatched_size.append({"filepath": filepath, "size": size})
        if mode != expected_mode:
            mismatched_mode.append({"filepath": filepath, "mode": mode})

    return {
        "expected_size": list(expected_size),
        "expected_mode": expected_mode,
        "size_distribution": {str(k): v for k, v in size_counts.items()},
        "mode_distribution": mode_counts,
        "n_mismatched_size": len(mismatched_size),
        "n_mismatched_mode": len(mismatched_mode),
        "mismatched_size_files": mismatched_size,
        "mismatched_mode_files": mismatched_mode,
        "all_uniform_size": len(mismatched_size) == 0,
        "all_uniform_mode": len(mismatched_mode) == 0,
    }


def _sha256(filepath: str, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicates(df: pd.DataFrame) -> dict:
    hashes: dict[str, list[str]] = {}
    for filepath in df["filepath"]:
        digest = _sha256(filepath)
        hashes.setdefault(digest, []).append(filepath)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    n_duplicate_files = sum(len(g) - 1 for g in duplicate_groups)

    return {
        "n_unique_hashes": len(hashes),
        "n_duplicate_groups": len(duplicate_groups),
        "n_duplicate_files": n_duplicate_files,
        "duplicate_groups": duplicate_groups,
        "has_duplicates": n_duplicate_files > 0,
    }


def run_integrity_checks(df: pd.DataFrame, config: dict) -> dict:
    dataset_cfg = config["dataset"]
    counts = check_counts(
        df,
        expected_total=dataset_cfg["total_images"],
        expected_classes=dataset_cfg["classes"],
        images_per_class=dataset_cfg["images_per_class"],
    )
    corruption = check_corruption(df)
    dimensions = check_dimensions(
        df,
        expected_size=tuple(dataset_cfg["expected_size"]),
        expected_mode=dataset_cfg["expected_mode"],
    )
    duplicates = check_duplicates(df)

    all_passed = (
        counts["balanced"]
        and counts["total_matches"]
        and corruption["n_corrupted"] == 0
        and dimensions["all_uniform_size"]
        and dimensions["all_uniform_mode"]
        and not duplicates["has_duplicates"]
    )

    return {
        "counts": counts,
        "corruption": corruption,
        "dimensions": dimensions,
        "duplicates": duplicates,
        "all_checks_passed": all_passed,
    }
