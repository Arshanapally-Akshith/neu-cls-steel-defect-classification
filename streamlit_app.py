"""NEU-CLS Steel Surface Defect Classifier — Streamlit demo (Phase 6).

Thin UI layer only. All model loading, preprocessing, prediction, and
Grad-CAM logic is reused as-is from src/ (src.models.transfer,
src.gradcam.gradcam) — nothing here retrains, fine-tunes, or otherwise
modifies the frozen Phase 3 model, the frozen split, or any saved
evaluation results. See config/config.yaml's `streamlit:` section for the
copy/thresholds used below.
"""
import io

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import load_config, resolve_path
from src.gradcam.gradcam import GradCAM, overlay_heatmap, resize_cam
from src.models.transfer import build_transforms, load_trained_model, predict_image


@st.cache_resource
def load_app_resources():
    """Loaded once per server process and cached — never re-loaded per
    request. Returns (config, model, classes) or (config, None, None) if
    the checkpoint hasn't been trained yet."""
    config = load_config()
    tl_cfg = config["transfer_learning"]
    model_path = resolve_path(tl_cfg["output"]["model_path"])
    if not model_path.exists():
        return config, None, None
    model, classes, _checkpoint = load_trained_model(model_path)
    return config, model, classes


def render_prediction(config: dict, model, classes: list[str], pil_image: Image.Image) -> None:
    tl_cfg = config["transfer_learning"]
    gc_cfg = config["gradcam"]
    _train_transform, eval_transform = build_transforms(tl_cfg)

    rgb_image = pil_image.convert("RGB")
    result = predict_image(model, rgb_image, eval_transform, classes)

    input_size = tuple(tl_cfg["input_size"])
    display_image = np.array(rgb_image.resize(input_size, Image.BILINEAR))

    with GradCAM(model, model.layer4[-1]) as gc:
        cam_result = gc.generate(result["input_tensor"], class_idx=result["predicted_idx"])
    overlay = overlay_heatmap(display_image, resize_cam(cam_result["cam"], input_size), alpha=gc_cfg["overlay_alpha"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(display_image, width="stretch")
    with col2:
        st.subheader("Grad-CAM")
        st.image(overlay, width="stretch", caption="Where the model's prediction is most gradient-sensitive")

    st.subheader("Prediction")
    st.metric("Predicted defect class", result["predicted_class"], f"{result['confidence']:.1%} confidence")

    probs_df = (
        pd.Series(result["class_probabilities"], name="Probability")
        .sort_values(ascending=False)
        .to_frame()
    )
    st.bar_chart(probs_df)

    confusion_cfg = config["streamlit"]["confusion_note"]
    if result["predicted_class"] in confusion_cfg["classes"]:
        st.warning(confusion_cfg["text"])


def main() -> None:
    config, model, classes = load_app_resources()
    st_cfg = config["streamlit"]

    st.set_page_config(page_title=st_cfg["title"], page_icon=st_cfg["page_icon"], layout="centered")
    st.title(st_cfg["title"])
    st.write(
        "Upload a steel surface image to classify its defect type "
        "(`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) "
        "and see a Grad-CAM overlay showing where the model's prediction comes from."
    )

    if model is None:
        st.error(
            "No trained model checkpoint found at "
            f"`{config['transfer_learning']['output']['model_path']}`. "
            "Run `python -m scripts.run_phase3_transfer` first to train and save it."
        )
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=st_cfg["allowed_upload_types"])

    if uploaded_file is not None:
        try:
            pil_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            pil_image.load()  # force full decode now so truncated/corrupt files fail here, not later
        except (UnidentifiedImageError, OSError, ValueError) as e:
            st.error(f"Could not read this file as an image ({type(e).__name__}: {e}). Please upload a valid JPG/PNG/BMP image.")
        else:
            render_prediction(config, model, classes, pil_image)
    else:
        st.info("Upload an image above to get started.")

    with st.expander("About this model & its limitations"):
        st.markdown(st_cfg["model_limitations"])


main()
