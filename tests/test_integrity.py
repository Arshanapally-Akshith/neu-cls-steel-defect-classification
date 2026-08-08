from src.data.dedup import resolve_duplicates
from src.data.integrity import (
    check_corruption,
    check_counts,
    check_dimensions,
    check_duplicates,
    run_integrity_checks,
)


def test_total_image_count(image_index, config):
    assert len(image_index) == config["dataset"]["total_images"]


def test_six_classes_present(image_index, config):
    assert set(image_index["class"]) == set(config["dataset"]["classes"])


def test_class_balance_300_each(image_index, config):
    counts = image_index["class"].value_counts()
    for cls in config["dataset"]["classes"]:
        assert counts[cls] == config["dataset"]["images_per_class"], (
            f"{cls} has {counts.get(cls, 0)} images, "
            f"expected {config['dataset']['images_per_class']}"
        )


def test_no_corrupted_images(image_index):
    result = check_corruption(image_index)
    assert result["n_corrupted"] == 0, result["corrupted_files"]


def test_uniform_dimensions_and_mode(image_index, config):
    result = check_dimensions(
        image_index,
        expected_size=tuple(config["dataset"]["expected_size"]),
        expected_mode=config["dataset"]["expected_mode"],
    )
    assert result["all_uniform_size"], result["mismatched_size_files"]
    assert result["all_uniform_mode"], result["mismatched_mode_files"]


def test_known_duplicate_pair_is_detected(image_index):
    """Raw NEU-CLS.zip contains exactly one confirmed byte-identical pair:
    patches_101.jpg == patches_105.jpg (verified via SHA-256). This is a
    known characteristic of the source data (see reports/data_integrity.md),
    not a bug — resolved via deduplication before splitting, not here."""
    result = check_duplicates(image_index)
    assert result["has_duplicates"]
    assert result["n_duplicate_groups"] == 1
    flat = {name for group in result["duplicate_groups"] for name in group}
    assert any(p.endswith("patches_101.jpg") for p in flat)
    assert any(p.endswith("patches_105.jpg") for p in flat)


def test_deduplication_removes_all_duplicates(image_index):
    duplicates = check_duplicates(image_index)
    deduped_df, dropped = resolve_duplicates(image_index, duplicates["duplicate_groups"])
    assert len(dropped) == 1
    assert len(deduped_df) == len(image_index) - 1
    assert not check_duplicates(deduped_df)["has_duplicates"]


def test_counts_check_flags_missing_class(image_index, config):
    trimmed = image_index[image_index["class"] != "crazing"]
    result = check_counts(
        trimmed,
        expected_total=config["dataset"]["total_images"],
        expected_classes=config["dataset"]["classes"],
        images_per_class=config["dataset"]["images_per_class"],
    )
    assert not result["balanced"]
    assert "crazing" in result["missing_classes"]


def test_full_integrity_suite_result(image_index, config):
    """The raw data is not perfectly clean: counts/corruption/dimensions all
    pass, but the known duplicate pair (see above) makes all_checks_passed
    False overall — that's the honest, expected result for the raw zip."""
    result = run_integrity_checks(image_index, config)
    assert result["counts"]["balanced"]
    assert result["counts"]["total_matches"]
    assert result["corruption"]["n_corrupted"] == 0
    assert result["dimensions"]["all_uniform_size"]
    assert result["dimensions"]["all_uniform_mode"]
    assert result["duplicates"]["has_duplicates"]
    assert result["all_checks_passed"] is False
