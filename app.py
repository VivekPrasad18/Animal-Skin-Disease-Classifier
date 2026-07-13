import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog Skin Disease Classifier",
    page_icon="🐾",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Dermatitis', 'Fungal Infections', 'Healthy',
    'Hypersensitivity', 'Demodicosis', 'Ringworm'
]
CLASS_COLORS = {
    'Dermatitis':        '#e74c3c',
    'Fungal Infections': '#8e44ad',
    'Healthy':           '#27ae60',
    'Hypersensitivity':  '#e67e22',
    'Demodicosis':       '#2980b9',
    'Ringworm':          '#c0392b'
}
CLASS_INFO = {
    'Dermatitis':        'Skin inflammation — may cause redness, itching, and sores.',
    'Fungal Infections': 'Caused by fungi — look for scaly, crusty or bald patches.',
    'Healthy':           'No disease detected. Skin appears normal.',
    'Hypersensitivity':  'Allergic reaction — causes intense itching and redness.',
    'Demodicosis':       'Caused by Demodex mites — hair loss and skin lesions.',
    'Ringworm':          'Fungal infection — circular bald patches with red border.'
}

MODEL_DIR = 'models'

# ── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    resnet = ResNet50(weights='imagenet', include_top=False,
                      pooling='avg', input_shape=(224, 224, 3))
    svm    = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    return resnet, svm, scaler

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(img_array, resnet, svm, scaler):
    img = cv2.resize(img_array, IMG_SIZE)
    inp = np.expand_dims(preprocess_input(img.astype(np.float32)), 0)
    feats = resnet.predict(inp, verbose=0)
    feats = scaler.transform(feats)
    proba = svm.predict_proba(feats)[0]
    idx   = int(np.argmax(proba))
    return CLASS_NAMES[idx], float(np.max(proba)), proba

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐾 Dog Skin Disease Classifier")
st.markdown("Upload a dog skin image to detect the disease using AI.")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** ResNet50 + SVM (Hybrid)  
    **Accuracy:** 94.46%  
    **ROC-AUC:** 0.9934  
    **Classes:** 6 diseases  
    """)
    st.markdown("---")
    st.header("🦠 Diseases")
    for cls, color in CLASS_COLORS.items():
        st.markdown(f"<span style='color:{color}'>●</span> {cls}", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "Choose a dog skin image",
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded:
    # Load image
    pil_img  = Image.open(uploaded).convert('RGB')
    img_np   = np.array(pil_img)
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        resnet, svm, scaler = load_models()
        disease, confidence, all_probs = predict(img_bgr, resnet, svm, scaler)

    color = CLASS_COLORS[disease]

    with col2:
        st.markdown(f"""
        <div style='background:{color}; padding:20px; border-radius:12px; text-align:center; margin-top:10px'>
            <h2 style='color:white; margin:0'>🔬 {disease}</h2>
            <h3 style='color:white; margin:5px 0'>Confidence: {confidence*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<br><p style='color:gray'>{CLASS_INFO[disease]}</p>", unsafe_allow_html=True)

        if confidence < 0.55:
            st.warning("⚠️ Low confidence — please consult a veterinarian.")

    # Probability bars
    st.markdown("---")
    st.subheader("📊 Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, all_probs):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(float(prob), text=cls)
        with col_b:
            st.markdown(f"**{prob*100:.1f}%**")

else:
    st.info("👆 Upload a dog skin image above to get started.")
    st.image(
        "https://img.icons8.com/color/200/dog.png",
        width=150
    )
