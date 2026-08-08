"""HOG + Logistic Regression baseline (Phase 2).

Hyperparameter (C) selection uses the val set only; the test set is never
touched during selection. HOG is stateless (no fit), so computing it once
per split and reusing across the C grid is equivalent to — but far cheaper
than — refitting a [hog, scaler, clf] Pipeline from raw images for every
candidate C. The final Pipeline returned by `select_best_model` bundles the
already-fitted scaler + classifier with the (stateless) HOG step, so it can
run inference directly on raw images later (e.g. the Streamlit demo).
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import compute_metrics
from src.features.hog import HOGFeatureExtractor


@dataclass
class SelectionResult:
    best_C: float
    best_pipeline: Pipeline
    search_results: list = field(default_factory=list)


def extract_features(images: np.ndarray, hog_params: dict) -> np.ndarray:
    extractor = HOGFeatureExtractor(**hog_params)
    return extractor.transform(images)


def select_best_model(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_val_feat: np.ndarray,
    y_val: np.ndarray,
    classes: list[str],
    C_grid: list[float],
    max_iter: int,
    solver: str,
    random_state: int,
    selection_metric: str = "f1_macro",
) -> SelectionResult:
    """Fit a [scaler, LogisticRegression] pipeline on train features for each
    candidate C, score on val, and return the best one by `selection_metric`.

    The scaler and classifier are fit ONLY on X_train_feat/y_train in every
    iteration — val is used exclusively for scoring/selection, never fit.
    """
    search_results = []
    best = None

    for C in C_grid:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=random_state)),
        ])
        pipeline.fit(X_train_feat, y_train)

        y_val_pred = pipeline.predict(X_val_feat)
        val_metrics = compute_metrics(y_val, y_val_pred, classes)
        score = val_metrics[selection_metric]

        summary_metrics = {
            "accuracy": val_metrics["accuracy"],
            "precision_macro": val_metrics["precision_macro"],
            "recall_macro": val_metrics["recall_macro"],
            "f1_macro": val_metrics["f1_macro"],
            "precision_weighted": val_metrics["precision_weighted"],
            "recall_weighted": val_metrics["recall_weighted"],
            "f1_weighted": val_metrics["f1_weighted"],
        }
        search_results.append({"C": C, "val_metrics": summary_metrics, "selection_score": score})
        if best is None or score > best["selection_score"]:
            best = {"C": C, "selection_score": score, "pipeline": pipeline}

    return SelectionResult(best_C=best["C"], best_pipeline=best["pipeline"], search_results=search_results)


def build_inference_pipeline(hog_params: dict, fitted_scaler: StandardScaler, fitted_clf: LogisticRegression) -> Pipeline:
    """Bundle the stateless HOG step with an already-fitted scaler + classifier
    into a single Pipeline that accepts raw images end-to-end."""
    return Pipeline([
        ("hog", HOGFeatureExtractor(**hog_params)),
        ("scaler", fitted_scaler),
        ("clf", fitted_clf),
    ])
