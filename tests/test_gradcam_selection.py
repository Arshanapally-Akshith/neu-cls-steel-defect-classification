import pandas as pd

from src.gradcam.selection import (
    build_gradcam_selection,
    select_all_incorrect,
    select_highest_confidence_correct_per_class,
    select_lowest_confidence_correct_per_class,
)

CLASSES = ["a", "b", "c"]


def make_df():
    return pd.DataFrame([
        {"filename": "a1.jpg", "true_class": "a", "resnet_pred": "a", "resnet_confidence": 0.9, "resnet_correct": True},
        {"filename": "a2.jpg", "true_class": "a", "resnet_pred": "a", "resnet_confidence": 0.6, "resnet_correct": True},
        {"filename": "a3.jpg", "true_class": "a", "resnet_pred": "b", "resnet_confidence": 0.5, "resnet_correct": False},
        {"filename": "b1.jpg", "true_class": "b", "resnet_pred": "b", "resnet_confidence": 0.99, "resnet_correct": True},
        {"filename": "c1.jpg", "true_class": "c", "resnet_pred": "a", "resnet_confidence": 0.7, "resnet_correct": False},
    ])


def test_select_highest_confidence_correct_per_class():
    df = make_df()
    result = select_highest_confidence_correct_per_class(df, CLASSES)
    # class 'a': a1 (0.9) beats a2 (0.6); class 'b': b1; class 'c': no correct examples at all
    assert set(result["filename"]) == {"a1.jpg", "b1.jpg"}


def test_select_lowest_confidence_correct_per_class():
    df = make_df()
    result = select_lowest_confidence_correct_per_class(df, CLASSES, n=1)
    assert set(result["filename"]) == {"a2.jpg", "b1.jpg"}


def test_select_lowest_confidence_correct_per_class_n_greater_than_available():
    df = make_df()
    result = select_lowest_confidence_correct_per_class(df, CLASSES, n=5)
    # class 'a' only has 2 correct examples total -> both returned, not an error
    assert set(result[result["true_class"] == "a"]["filename"]) == {"a1.jpg", "a2.jpg"}


def test_select_all_incorrect():
    df = make_df()
    result = select_all_incorrect(df)
    assert set(result["filename"]) == {"a3.jpg", "c1.jpg"}
    assert (result["resnet_correct"] == False).all()  # noqa: E712


def test_build_gradcam_selection_categories_and_dedup():
    df = make_df()
    selection = build_gradcam_selection(df, CLASSES, confusion_prone_classes=["a"], n_extra_correct=1)

    by_filename = selection.set_index("filename")["category"].to_dict()

    # a1 is BOTH the highest-confidence correct for 'a' AND appears in
    # neither incorrect nor (since a2 is the lowest) confusion_prone_correct.
    assert by_filename["a1.jpg"] == "correct_representative"
    # a2 is the lowest-confidence correct for confusion-prone class 'a'.
    assert by_filename["a2.jpg"] == "confusion_prone_correct"
    assert by_filename["b1.jpg"] == "correct_representative"
    assert by_filename["a3.jpg"] == "incorrect"
    assert by_filename["c1.jpg"] == "incorrect"
    # no duplicate filenames
    assert selection["filename"].is_unique


def test_build_gradcam_selection_merges_categories_when_image_qualifies_for_both():
    # class 'b' has exactly one correct example -> it's simultaneously the
    # highest- AND lowest-confidence correct example for that class.
    df = make_df()
    selection = build_gradcam_selection(df, CLASSES, confusion_prone_classes=["b"], n_extra_correct=1)
    row = selection[selection["filename"] == "b1.jpg"].iloc[0]
    categories = set(row["category"].split(","))
    assert categories == {"correct_representative", "confusion_prone_correct"}


def test_build_gradcam_selection_empty_input():
    empty = pd.DataFrame(columns=["filename", "true_class", "resnet_pred", "resnet_confidence", "resnet_correct"])
    result = build_gradcam_selection(empty, CLASSES, confusion_prone_classes=CLASSES, n_extra_correct=1)
    assert len(result) == 0
