import numpy as np
import pytest

from src.data.loader import load_split_images
from src.eval.metrics import compute_metrics
from src.models.baseline import build_inference_pipeline, extract_features, select_best_model

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")


# ---------------------------------------------------------------------------
# Loader: must read exactly what the frozen manifest says, nothing else.
# ---------------------------------------------------------------------------

def test_loader_reads_images_matching_manifest(split_manifests, raw_dir):
    sample = split_manifests["train"].head(6)
    X, y = load_split_images(sample, raw_dir, grayscale=True)

    assert len(X) == len(sample) == len(y)
    assert list(y) == list(sample["class"])
    for img in X:
        assert img.shape == (200, 200)
        assert img.dtype == np.uint8


def test_loader_grayscale_vs_rgb(split_manifests, raw_dir):
    sample = split_manifests["train"].head(2)
    X_gray, _ = load_split_images(sample, raw_dir, grayscale=True)
    X_rgb, _ = load_split_images(sample, raw_dir, grayscale=False)
    assert X_gray[0].ndim == 2
    assert X_rgb[0].ndim == 3
    assert X_rgb[0].shape[-1] == 3


# ---------------------------------------------------------------------------
# select_best_model: synthetic, well-separated classes so correctness is
# checkable, and so we can assert no leakage from val into the fitted scaler.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(42)
    classes = ["a", "b", "c"]
    n_per_class = 30

    def make_split(seed_offset):
        r = np.random.default_rng(42 + seed_offset)
        X, y = [], []
        centers = {"a": -5.0, "b": 0.0, "c": 5.0}
        for cls, center in centers.items():
            X.append(r.normal(loc=center, scale=0.5, size=(n_per_class, 4)))
            y.extend([cls] * n_per_class)
        return np.vstack(X), np.array(y)

    X_train, y_train = make_split(0)
    X_val, y_val = make_split(1)
    return classes, X_train, y_train, X_val, y_val


def test_select_best_model_picks_a_valid_C(synthetic_features):
    classes, X_train, y_train, X_val, y_val = synthetic_features
    result = select_best_model(
        X_train, y_train, X_val, y_val, classes,
        C_grid=[0.01, 1.0, 100.0], max_iter=1000, solver="lbfgs",
        random_state=42, selection_metric="f1_macro",
    )
    assert result.best_C in [0.01, 1.0, 100.0]
    assert len(result.search_results) == 3
    # well-separated synthetic classes should be near-perfectly classifiable
    assert result.search_results[[r["C"] for r in result.search_results].index(result.best_C)]["selection_score"] > 0.9


def test_select_best_model_scaler_fit_only_on_train(synthetic_features):
    classes, X_train, y_train, X_val, y_val = synthetic_features
    result = select_best_model(
        X_train, y_train, X_val, y_val, classes,
        C_grid=[1.0], max_iter=1000, solver="lbfgs",
        random_state=42, selection_metric="f1_macro",
    )
    scaler = result.best_pipeline.named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(X_train)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-10)


def test_select_best_model_val_never_mutated(synthetic_features):
    classes, X_train, y_train, X_val, y_val = synthetic_features
    X_val_before = X_val.copy()
    select_best_model(
        X_train, y_train, X_val, y_val, classes,
        C_grid=[0.01, 1.0], max_iter=1000, solver="lbfgs",
        random_state=42, selection_metric="f1_macro",
    )
    np.testing.assert_array_equal(X_val, X_val_before)


# ---------------------------------------------------------------------------
# End-to-end smoke test on a small real sample from the frozen manifests.
# Not a performance assertion — just checks the real pipeline runs and
# produces structurally valid output on real images.
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline_on_real_sample(split_manifests, raw_dir):
    classes = sorted(split_manifests["train"]["class"].unique())
    train_sample = split_manifests["train"].groupby("class", group_keys=False).head(4)
    val_sample = split_manifests["val"].groupby("class", group_keys=False).head(2)

    X_train_img, y_train = load_split_images(train_sample, raw_dir, grayscale=True)
    X_val_img, y_val = load_split_images(val_sample, raw_dir, grayscale=True)

    X_train_feat = extract_features(X_train_img, HOG_PARAMS)
    X_val_feat = extract_features(X_val_img, HOG_PARAMS)

    result = select_best_model(
        X_train_feat, y_train, X_val_feat, y_val, classes,
        C_grid=[0.1, 1.0], max_iter=2000, solver="lbfgs",
        random_state=42, selection_metric="f1_macro",
    )

    y_val_pred = result.best_pipeline.predict(X_val_feat)
    metrics = compute_metrics(y_val, y_val_pred, classes)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["per_class"].keys()) == set(classes)

    inference_pipeline = build_inference_pipeline(
        HOG_PARAMS,
        result.best_pipeline.named_steps["scaler"],
        result.best_pipeline.named_steps["clf"],
    )
    raw_preds = inference_pipeline.predict(X_val_img)
    np.testing.assert_array_equal(raw_preds, y_val_pred)
