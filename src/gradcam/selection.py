"""Which test images to run Grad-CAM on.

Built purely from Phase 4's per-sample predictions table
(reports/error_analysis_predictions.csv) — this module never re-derives
correctness, re-runs inference, or touches the model/test images itself.
Kept pure/testable, matching src/eval/error_analysis.py's style.
"""
import pandas as pd

CATEGORY_CORRECT_REPRESENTATIVE = "correct_representative"
CATEGORY_CONFUSION_PRONE_CORRECT = "confusion_prone_correct"
CATEGORY_INCORRECT = "incorrect"


def select_highest_confidence_correct_per_class(
    df: pd.DataFrame, classes: list[str],
    class_col: str = "true_class", correct_col: str = "resnet_correct", confidence_col: str = "resnet_confidence",
) -> pd.DataFrame:
    """One row per class: its highest-confidence correctly classified example."""
    if len(df) == 0:
        return df.copy()
    correct = df[df[correct_col]]
    rows = [
        correct[correct[class_col] == cls].nlargest(1, confidence_col)
        for cls in classes
        if len(correct[correct[class_col] == cls]) > 0
    ]
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def select_lowest_confidence_correct_per_class(
    df: pd.DataFrame, classes: list[str], n: int = 1,
    class_col: str = "true_class", correct_col: str = "resnet_correct", confidence_col: str = "resnet_confidence",
) -> pd.DataFrame:
    """Up to `n` rows per class: its lowest-confidence correctly classified
    examples — the "least certain, still right" cases."""
    if len(df) == 0:
        return df.copy()
    correct = df[df[correct_col]]
    rows = [
        correct[correct[class_col] == cls].nsmallest(n, confidence_col)
        for cls in classes
        if len(correct[correct[class_col] == cls]) > 0
    ]
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def select_all_incorrect(df: pd.DataFrame, correct_col: str = "resnet_correct") -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    return df[~df[correct_col]].reset_index(drop=True)


def build_gradcam_selection(
    df: pd.DataFrame, classes: list[str], confusion_prone_classes: list[str], n_extra_correct: int,
) -> pd.DataFrame:
    """Combine the three selection strategies into one deduplicated table
    with a `category` column recording why each image was picked (joined
    by comma if more than one reason applies — e.g. an image could in
    principle be both the highest- and lowest-confidence correct example
    for its class if it's the only correct one)."""
    highest = select_highest_confidence_correct_per_class(df, classes).assign(category=CATEGORY_CORRECT_REPRESENTATIVE)
    extra = select_lowest_confidence_correct_per_class(df, confusion_prone_classes, n=n_extra_correct).assign(
        category=CATEGORY_CONFUSION_PRONE_CORRECT
    )
    incorrect = select_all_incorrect(df).assign(category=CATEGORY_INCORRECT)

    combined = pd.concat([highest, extra, incorrect], ignore_index=True)
    if len(combined) == 0:
        return combined

    category_by_filename = combined.groupby("filename")["category"].apply(lambda s: ",".join(sorted(set(s))))
    deduped = combined.drop_duplicates(subset="filename", keep="first").drop(columns="category").copy()
    deduped["category"] = deduped["filename"].map(category_by_filename)
    return deduped.reset_index(drop=True)
