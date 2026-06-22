"""
Dog Skin Disease Prediction Pipeline
=====================================
Flow:
  Input Image
    → Image Preprocessing (Resize, Normalize)
    → Pretrained CNN (MobileNetV2 + ResNet50) — Deep Feature Extraction
    → Feature Optimization (PCA)
    → Hybrid Classification (SVM + DNN)
    → Prediction Fusion (Weighted Voting)
    → Final Disease Class Output
    → Performance Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
"""

import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE      = (224, 224)
TEST_PATH     = "dataset/test"
MOBILENET_H5  = "models/mobilenet_best.h5"
SVM_PKL       = "models/svm_model.pkl"
SCALER_PKL    = "models/scaler.pkl"

# Exact folder names from dataset (order must match model training)
CLASS_NAMES = [
    "demodicosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "ringworm"
]
NUM_CLASSES = len(CLASS_NAMES)

# Weighted voting weights  (SVM=0.4, DNN/MobileNet=0.6)
W_SVM = 0.4
W_DNN = 0.6

# ============================================================
# STEP 1 — IMAGE PREPROCESSING
# ============================================================
def preprocess_for_resnet(img_bgr):
    """Resize + ResNet50 normalization"""
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = resnet_preprocess(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def preprocess_for_mobilenet(img_bgr):
    """Resize + MobileNetV2 normalization"""
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = mobilenet_preprocess(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

# ============================================================
# STEP 2 — LOAD PRETRAINED CNN BACKBONES
# ============================================================
_resnet_extractor  = None
_mobilenet_model   = None
_svm               = None
_scaler            = None

def get_resnet_extractor():
    global _resnet_extractor
    if _resnet_extractor is None:
        print("  Loading ResNet50 feature extractor...")
        _resnet_extractor = ResNet50(
            weights='imagenet', include_top=False,
            pooling='avg', input_shape=(224, 224, 3)
        )
    return _resnet_extractor

def get_mobilenet_model():
    global _mobilenet_model
    if _mobilenet_model is None:
        print("  Loading MobileNetV2 DNN model...")
        _mobilenet_model = tf.keras.models.load_model(MOBILENET_H5)
    return _mobilenet_model

def get_svm_and_scaler():
    global _svm, _scaler
    if _svm is None:
        print("  Loading SVM + Scaler...")
        _svm    = joblib.load(SVM_PKL)
        _scaler = joblib.load(SCALER_PKL)
    return _svm, _scaler

# ============================================================
# STEP 3 — DEEP FEATURE EXTRACTION
# ============================================================
def extract_resnet_features(img_bgr):
    """Extract 2048-d feature vector via ResNet50"""
    resnet = get_resnet_extractor()
    inp    = preprocess_for_resnet(img_bgr)
    feats  = resnet.predict(inp, verbose=0)   # (1, 2048)
    return feats

# ============================================================
# STEP 4 — FEATURE OPTIMIZATION (PCA applied at batch level)
# ============================================================
def apply_pca(features, n_components=None):
    """
    Lightweight PCA for dimensionality insight.
    For single-image inference we skip PCA (not enough samples).
    For batch evaluation we fit PCA on the batch.
    """
    if features.shape[0] < 2:
        return features   # can't fit PCA on 1 sample
    n = min(n_components or features.shape[1], features.shape[0], features.shape[1])
    pca = PCA(n_components=n, random_state=42)
    return pca.fit_transform(features)

# ============================================================
# STEP 5 — HYBRID CLASSIFICATION (SVM + DNN)
# ============================================================
def svm_predict_proba(features_scaled):
    svm, _ = get_svm_and_scaler()
    return svm.predict_proba(features_scaled)   # (N, 6)

def dnn_predict_proba(img_bgr):
    model = get_mobilenet_model()
    inp   = preprocess_for_mobilenet(img_bgr)
    return model.predict(inp, verbose=0)        # (1, 6)

# ============================================================
# STEP 6 — PREDICTION FUSION (Weighted Voting)
# ============================================================
def fuse_predictions(svm_proba, dnn_proba):
    """Weighted average: 40% SVM + 60% DNN"""
    return W_SVM * svm_proba + W_DNN * dnn_proba

# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================
def predict_single(img_path):
    print("\n[Pipeline] Starting prediction...")

    # --- Load & display ---
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    # --- Step 1: Preprocess ---
    print("  [1] Preprocessing image...")

    # --- Step 2+3: CNN Feature Extraction ---
    print("  [2] Extracting deep features (ResNet50)...")
    resnet_feats = extract_resnet_features(img_bgr)   # (1, 2048)

    # --- Step 4: Scale (trained scaler) + PCA note ---
    print("  [3] Scaling features...")
    _, scaler = get_svm_and_scaler()
    feats_scaled = scaler.transform(resnet_feats)

    # --- Step 5: SVM prediction ---
    print("  [4] SVM classification...")
    svm_proba = svm_predict_proba(feats_scaled)        # (1, 6)

    # --- Step 5: DNN (MobileNetV2) prediction ---
    print("  [5] DNN (MobileNetV2) classification...")
    dnn_proba = dnn_predict_proba(img_bgr)             # (1, 6)

    # --- Step 6: Fusion ---
    print("  [6] Fusing predictions (weighted voting)...")
    final_proba = fuse_predictions(svm_proba, dnn_proba)  # (1, 6)

    pred_idx   = int(np.argmax(final_proba[0]))
    confidence = float(np.max(final_proba[0]))
    label      = CLASS_NAMES[pred_idx]

    # --- Output ---
    print("\n========== FINAL PREDICTION ==========")
    print(f"  Disease    : {label}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print("  Per-class probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        bar = "█" * int(final_proba[0][i] * 30)
        print(f"    {cls:<20} {final_proba[0][i]*100:5.1f}%  {bar}")
    if confidence < 0.55:
        print("\n  ⚠ Low confidence — consider consulting a vet.")
    print("=======================================")

# ============================================================
# DATASET EVALUATION
# ============================================================
def evaluate_dataset():
    print(f"\n[Pipeline] Evaluating on test dataset: {TEST_PATH}\n")

    resnet   = get_resnet_extractor()
    model    = get_mobilenet_model()
    svm, scaler = get_svm_and_scaler()

    # --- Step 1+2: Extract ResNet features from all test images ---
    print("  [1] Extracting ResNet50 features from test set...")
    resnet_gen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
    test_gen_r = resnet_gen.flow_from_directory(
        TEST_PATH, target_size=IMG_SIZE,
        batch_size=16, class_mode='categorical', shuffle=False
    )
    resnet_feats = resnet.predict(test_gen_r, verbose=1)
    true_labels  = test_gen_r.classes

    # --- Step 3: Scale ---
    print("\n  [2] Scaling features...")
    feats_scaled = scaler.transform(resnet_feats)

    # --- Step 4: PCA (informational, 128 components) ---
    print("  [3] Applying PCA (128 components)...")
    pca = PCA(n_components=128, random_state=42)
    pca.fit(feats_scaled)
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"      PCA retains {explained:.1f}% variance with 128 components")

    # --- Step 5a: SVM probabilities ---
    print("  [4] SVM predictions...")
    svm_proba = svm.predict_proba(feats_scaled)   # (N, 6)

    # --- Step 5b: DNN (MobileNetV2) probabilities ---
    print("  [5] DNN (MobileNetV2) predictions...")
    mob_gen   = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    test_gen_m = mob_gen.flow_from_directory(
        TEST_PATH, target_size=IMG_SIZE,
        batch_size=16, class_mode='categorical', shuffle=False
    )
    dnn_proba = model.predict(test_gen_m, verbose=1)  # (N, 6)

    # --- Step 6: Fusion ---
    print("\n  [6] Fusing predictions (weighted voting)...")
    final_proba = fuse_predictions(svm_proba, dnn_proba)
    final_preds = np.argmax(final_proba, axis=1)

    # --- Step 7: Performance Evaluation ---
    acc       = accuracy_score(true_labels, final_preds)
    precision = precision_score(true_labels, final_preds, average='weighted', zero_division=0)
    recall    = recall_score(true_labels, final_preds, average='weighted', zero_division=0)
    f1        = f1_score(true_labels, final_preds, average='weighted', zero_division=0)

    # ROC-AUC (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(true_labels, classes=list(range(NUM_CLASSES)))
    roc_auc = roc_auc_score(y_bin, final_proba, multi_class='ovr', average='weighted')

    print("\n========== PERFORMANCE EVALUATION ==========")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1 Score  : {f1 * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("=============================================")
    print("\n--- Per-Class Report ---")
    print(classification_report(true_labels, final_preds, target_names=CLASS_NAMES, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, final_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix — Hybrid Model (SVM + DNN)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    # ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            y_bin[:, i], final_proba[:, i],
            name=cls, ax=ax
        )
    ax.set_title("ROC Curves — Hybrid Model (One-vs-Rest)")
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Animal Disease Classifier — Hybrid Pipeline")
    print("=" * 50)
    print(f"\nClasses ({NUM_CLASSES}):", CLASS_NAMES)
    print(f"Fusion weights: SVM={W_SVM}, DNN={W_DNN}\n")

    print("Select task:")
    print("  1 - Predict a single image")
    print("  2 - Evaluate on full test dataset")
    task = input("Enter choice [1/2]: ").strip()

    if task == "1":
        img_path = input("Enter image path: ").strip()
        predict_single(img_path)
    elif task == "2":
        evaluate_dataset()
    else:
        print("Invalid choice.")
