import pandas as pd
import pytest

from src.data.split import stratified_split

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# sklearn's stratified split rounds per class per split-stage; with an odd
# per-class count (e.g. 299 after deduplication) exact fractional counts
# aren't always integers, so allow a small tolerance rather than asserting
# exact equality.
COUNT_TOLERANCE = 2


@pytest.fixture(scope="module")
def splits(deduped_index, config):
    return stratified_split(deduped_index, seed=config["seed"], **FRACTIONS)


def test_split_sizes_match_fractions(splits, deduped_index, config):
    total = len(deduped_index)
    n_classes = len(config["dataset"]["classes"])
    for name, fraction in FRACTIONS.items():
        expected = total * fraction
        assert abs(len(splits[name]) - expected) <= COUNT_TOLERANCE * n_classes
    assert sum(len(s) for s in splits.values()) == total


def test_splits_are_disjoint(splits):
    train_files = set(splits["train"]["filename"])
    val_files = set(splits["val"]["filename"])
    test_files = set(splits["test"]["filename"])

    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)


def test_no_exact_duplicate_images_split_across_sets(splits, image_index):
    """Regression test for the patches_101/patches_105 leakage finding:
    byte-identical images must never land in different splits."""
    from src.data.integrity import check_duplicates

    duplicates = check_duplicates(image_index)
    filename_by_path = dict(zip(image_index["filepath"], image_index["filename"]))

    split_of = {}
    for name in ("train", "val", "test"):
        for fname in splits[name]["filename"]:
            split_of[fname] = name

    for group in duplicates["duplicate_groups"]:
        splits_hit = {split_of[filename_by_path[p]] for p in group if filename_by_path[p] in split_of}
        assert len(splits_hit) <= 1, f"Duplicate group {group} spans multiple splits: {splits_hit}"


def test_splits_cover_every_image(splits, deduped_index):
    all_split_files = (
        set(splits["train"]["filename"])
        | set(splits["val"]["filename"])
        | set(splits["test"]["filename"])
    )
    assert all_split_files == set(deduped_index["filename"])


def test_class_balance_preserved_per_split(splits, deduped_index, config):
    actual_class_counts = deduped_index["class"].value_counts()
    for split_name, fraction in FRACTIONS.items():
        counts = splits[split_name]["class"].value_counts()
        for cls in config["dataset"]["classes"]:
            expected = round(actual_class_counts[cls] * fraction)
            assert abs(counts.get(cls, 0) - expected) <= COUNT_TOLERANCE, (
                f"{split_name}/{cls}: got {counts.get(cls, 0)}, expected ~{expected}"
            )


def test_split_is_reproducible_with_same_seed(deduped_index, config):
    splits_a = stratified_split(deduped_index, seed=config["seed"], **FRACTIONS)
    splits_b = stratified_split(deduped_index, seed=config["seed"], **FRACTIONS)

    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(
            splits_a[name].reset_index(drop=True),
            splits_b[name].reset_index(drop=True),
        )


def test_different_seed_gives_different_split(deduped_index, config):
    splits_a = stratified_split(deduped_index, seed=config["seed"], **FRACTIONS)
    splits_b = stratified_split(deduped_index, seed=config["seed"] + 1, **FRACTIONS)

    assert set(splits_a["train"]["filename"]) != set(splits_b["train"]["filename"])


def test_fractions_must_sum_to_one():
    df = pd.DataFrame({"filename": ["a.jpg"], "class": ["crazing"]})
    with pytest.raises(ValueError):
        stratified_split(df, train=0.5, val=0.3, test=0.3, seed=0)
