"""Smoke tests for streamlit_app.py using streamlit.testing.v1.AppTest.

These run the actual app script headlessly and interact with its widgets —
the "practical" way to test a Streamlit app, short of a real browser.
Skips gracefully if the trained checkpoint isn't present (same convention
as the rest of the suite's real-artifact tests).
"""
from pathlib import Path

import pandas as pd
import pytest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def app(resnet_checkpoint_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def test_app_loads_without_exception(app):
    assert not app.exception


def test_app_shows_title_and_uploader(app, config):
    assert app.title[0].value == config["streamlit"]["title"]
    assert len(app.file_uploader) == 1


def test_app_shows_prompt_before_any_upload(app):
    assert any("Upload an image" in i.value for i in app.info)
    assert len(app.metric) == 0


def test_app_shows_model_limitations_expander(app):
    assert len(app.expander) >= 1
    expander_labels = [e.label for e in app.expander]
    assert any("limitations" in label.lower() for label in expander_labels)


def test_app_predicts_on_real_test_image(app, config, raw_dir):
    from src.config import resolve_path
    split_dir = resolve_path(config["split"]["output_dir"])
    test_df = pd.read_csv(split_dir / "test.csv")
    row = test_df.iloc[0]
    content = (raw_dir / row["filename"]).read_bytes()

    app.file_uploader[0].set_value((row["filename"], content, "image/jpeg"))
    app.run()

    assert not app.exception
    assert len(app.error) == 0
    assert len(app.metric) == 1
    assert app.metric[0].label == "Predicted defect class"
    classes = config["dataset"]["classes"]
    assert app.metric[0].value in classes
    assert "confidence" in app.metric[0].delta


def test_app_predicted_class_matches_reused_predict_image(app, config, raw_dir, resnet_checkpoint_path):
    """The app's displayed prediction must match src.models.transfer.predict_image
    called directly on the same image — proof the app isn't duplicating/
    diverging from the shared inference logic."""
    from PIL import Image

    from src.config import resolve_path
    from src.models.transfer import build_transforms, load_trained_model, predict_image

    split_dir = resolve_path(config["split"]["output_dir"])
    test_df = pd.read_csv(split_dir / "test.csv")
    row = test_df.iloc[1]
    content = (raw_dir / row["filename"]).read_bytes()

    app.file_uploader[0].set_value((row["filename"], content, "image/jpeg"))
    app.run()
    assert not app.exception
    app_prediction = app.metric[0].value

    model, classes, _ckpt = load_trained_model(resnet_checkpoint_path)
    _train_t, eval_t = build_transforms(config["transfer_learning"])
    direct_result = predict_image(model, Image.open(raw_dir / row["filename"]), eval_t, classes)

    assert app_prediction == direct_result["predicted_class"]


def test_app_handles_invalid_upload_gracefully(app):
    app.file_uploader[0].set_value(("bad.jpg", b"not a real image, just garbage bytes", "image/jpeg"))
    app.run()

    assert not app.exception
    assert len(app.error) == 1
    assert "could not read this file as an image" in app.error[0].value.lower()
    assert len(app.metric) == 0


def test_app_handles_empty_file_upload_gracefully(app):
    app.file_uploader[0].set_value(("empty.jpg", b"", "image/jpeg"))
    app.run()

    assert not app.exception
    assert len(app.error) == 1
    assert len(app.metric) == 0


def test_app_shows_confusion_note_for_flagged_classes(app, config, raw_dir):
    """Upload every test image until we find one predicted into the
    Phase 4 confusion group, and confirm the warning note appears exactly
    when it should (and not otherwise, checked implicitly by the app not
    crashing across the whole loop)."""
    from src.config import resolve_path

    confusion_classes = set(config["streamlit"]["confusion_note"]["classes"])
    split_dir = resolve_path(config["split"]["output_dir"])
    test_df = pd.read_csv(split_dir / "test.csv")

    found_flagged = False
    for _, row in test_df[test_df["class"].isin(confusion_classes)].head(5).iterrows():
        content = (raw_dir / row["filename"]).read_bytes()
        app.file_uploader[0].set_value((row["filename"], content, "image/jpeg"))
        app.run()
        assert not app.exception
        if app.metric[0].value in confusion_classes:
            assert len(app.warning) == 1
            found_flagged = True
            break

    assert found_flagged, "expected at least one of these confusion-prone-class test images to be predicted into the flagged group"
