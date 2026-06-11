"""Streamlit demo for the skin lesion classifier.

Run locally with:
    streamlit run app.py

Loads the trained checkpoints from results/models/. Train them first
(`python -m scripts.train`) or drop your checkpoints into that folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import Config, get_device
from src.inference import load_trained_model, predict_probs, preprocess
from src.interpretability import gradcam_overlay

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="wide")

CONFIG_PATH = Path("configs/default.yaml")
cfg = Config.from_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else Config()
device = get_device()
GRADCAM_MODELS = ("resnet50", "efficientnet_b3")


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models = {}
    for name in cfg.model_names:
        if (cfg.models_dir / f"{name}_main.pt").exists():
            models[name] = load_trained_model(name, cfg, device)
    return models


def probability_table(probs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Diagnosis": [cfg.class_full[c] for c in cfg.class_names],
        "Code": list(cfg.class_names),
        "Probability": probs,
    }).sort_values("Probability", ascending=False)


st.title("🔬 Skin Lesion Classifier")
st.markdown(
    "Upload a dermatoscopic image to classify it across the seven HAM10000 lesion "
    "types, and see a Grad-CAM heatmap of the regions driving the prediction."
)
st.warning(
    "**Research demo only — not a medical device.** This tool must not be used for "
    "diagnosis or any clinical decision. Always consult a qualified dermatologist.",
    icon="⚠️",
)

models = load_models()
if not models:
    st.error(
        f"No trained checkpoints found in `{cfg.models_dir}`.\n\n"
        "Train a model with `python -m scripts.train` (or copy a `<model>_main.pt` "
        "checkpoint into that folder), then reload this page."
    )
    st.stop()

# --- sidebar: model choice ---
display_to_name = {cfg.model_display[n]: n for n in models}
options = list(display_to_name)
if len(models) >= 2:
    options.append("Ensemble (soft vote)")

with st.sidebar:
    st.header("Model")
    choice = st.radio("Choose a model", options, index=len(options) - 1)
    st.caption(f"Running on **{device.type.upper()}**. "
               f"{len(models)} model(s) available.")
    st.divider()
    st.caption("Frozen ImageNet backbone with a trained classifier head. "
               "Grad-CAM is available for the CNN models.")

# --- main: upload and predict ---
uploaded = st.file_uploader("Dermatoscopic image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a JPG or PNG image to get a prediction.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
tensor, denorm = preprocess(image, cfg)

is_ensemble = choice == "Ensemble (soft vote)"
if is_ensemble:
    probs = np.mean([predict_probs(m, tensor, device) for m in models.values()], axis=0)
else:
    name = display_to_name[choice]
    probs = predict_probs(models[name], tensor, device)

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, use_column_width=True)

with right:
    st.subheader("Prediction")
    top = int(probs.argmax())
    st.metric(cfg.class_full[cfg.class_names[top]],
              f"{probs[top] * 100:.1f}% confidence")
    st.dataframe(
        probability_table(probs), hide_index=True, use_container_width=True,
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="%.3f")},
    )

st.divider()
st.subheader("Where the model looks (Grad-CAM)")
if is_ensemble:
    st.caption("Grad-CAM applies to a single model — pick a CNN model to see it.")
elif name in GRADCAM_MODELS:
    overlay = gradcam_overlay(models[name], name, tensor, denorm, device)
    gleft, gright = st.columns(2)
    gleft.image(denorm, caption="Preprocessed input", use_column_width=True)
    gright.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
else:
    st.caption("Grad-CAM is shown for the CNN models (ResNet-50, EfficientNet-B3).")
