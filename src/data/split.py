"""Fixed, reproducible stratified train/val/test split.

The split is derived only from `filename` + `class` (not the machine-specific
absolute filepath), so the frozen manifests in data/splits/ are portable and
independent of where the raw images happen to live on disk. Every downstream
phase (baseline, transfer learning, error analysis) must load these same
manifests rather than re-splitting, so results stay comparable.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    df: pd.DataFrame, train: float, val: float, test: float, seed: int
) -> dict[str, pd.DataFrame]:
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {train + val + test}")

    # Sort first so the split is a pure function of (filename, class, seed) —
    # not of filesystem iteration order, which can vary across machines/OSes.
    df = df[["filename", "class"]].sort_values("filename").reset_index(drop=True)

    train_df, remainder_df = train_test_split(
        df, train_size=train, stratify=df["class"], random_state=seed
    )
    val_fraction_of_remainder = val / (val + test)
    val_df, test_df = train_test_split(
        remainder_df,
        train_size=val_fraction_of_remainder,
        stratify=remainder_df["class"],
        random_state=seed,
    )

    return {
        "train": train_df.sort_values("filename").reset_index(drop=True),
        "val": val_df.sort_values("filename").reset_index(drop=True),
        "test": test_df.sort_values("filename").reset_index(drop=True),
    }


def save_manifests(splits: dict[str, pd.DataFrame], output_dir: Path, seed: int, fractions: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"seed": seed, "fractions": fractions, "counts": {}}
    for name in SPLIT_NAMES:
        split_df = splits[name]
        split_df.to_csv(output_dir / f"{name}.csv", index=False)
        summary["counts"][name] = {
            "total": len(split_df),
            "per_class": split_df["class"].value_counts().sort_index().to_dict(),
        }

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=int)


def load_manifests(output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    return {name: pd.read_csv(output_dir / f"{name}.csv") for name in SPLIT_NAMES}


if __name__ == "__main__":
    from src.config import load_config, resolve_path
    from src.data.extract import extract_dataset
    from src.data.index import build_image_index

    config = load_config()
    zip_path = resolve_path(config["dataset"]["zip_path"])
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    extract_dataset(zip_path, raw_dir)

    df = build_image_index(raw_dir, valid_extensions=tuple(config["dataset"]["valid_extensions"]))
    split_cfg = config["split"]
    splits = stratified_split(
        df,
        train=split_cfg["train"],
        val=split_cfg["val"],
        test=split_cfg["test"],
        seed=config["seed"],
    )
    output_dir = resolve_path(split_cfg["output_dir"])
    save_manifests(
        splits,
        output_dir,
        seed=config["seed"],
        fractions={"train": split_cfg["train"], "val": split_cfg["val"], "test": split_cfg["test"]},
    )
    for name in SPLIT_NAMES:
        print(f"{name}: {len(splits[name])} images -> {output_dir / (name + '.csv')}")
