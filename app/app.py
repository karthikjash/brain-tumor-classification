"""
Streamlit App for Brain Tumor Classification
"""

import os
import sys

import streamlit as st
import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CLASS_NAMES
from src.model import create_model
from src.preprocess import get_test_transforms


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model()

    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

transform = get_test_transforms()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Brain Tumor Classification")

st.write(
    "Upload an MRI image and the trained EfficientNet-B0 model "
    "will predict the tumour type."
)

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True,
    )

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    predicted_class = CLASS_NAMES[prediction.item()]

    st.success(
        f"Prediction: **{predicted_class.upper()}**"
    )

    st.write(
        f"Confidence: **{confidence.item()*100:.2f}%**"
    )

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):

        st.progress(
            float(probabilities[0][i])
        )

        st.write(
            f"{class_name}: "
            f"{probabilities[0][i].item()*100:.2f}%"
        )