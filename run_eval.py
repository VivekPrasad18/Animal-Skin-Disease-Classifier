import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

TEST_PATH   = "dataset/test"
IMG_SIZE    = (224, 224)
# Alphabetical order as loaded by Keras flow_from_directory
CLASS_NAMES = ["Dermatitis","Fungal_infections","Healthy","Hypersensitivity","demodicosis","ringworm"]

print("Loading models...")
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
svm    = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("Models loaded.\n")

# ResNet features
print("Extracting ResNet50 features from test set...")
gen_r = ImageDataGenerator(preprocessing_function=resnet_pre).flow_from_directory(
    TEST_PATH, target_size=IMG_SIZE, batch_size=16, class_mode='categorical', shuffle=False)
resnet_feats = resnet.predict(gen_r, verbose=1)
true_labels  = gen_r.classes
feats_scaled = scaler.transform(resnet_feats)

# SVM predictions
print("\nSVM predictions...")
final_proba = svm.predict_proba(feats_scaled)   # (N, 6)
final_preds = np.argmax(final_proba, axis=1)

# Metrics
acc  = accuracy_score(true_labels, final_preds)
prec = precision_score(true_labels, final_preds, average='weighted', zero_division=0)
rec  = recall_score(true_labels, final_preds, average='weighted', zero_division=0)
f1   = f1_score(true_labels, final_preds, average='weighted', zero_division=0)
y_bin   = label_binarize(true_labels, classes=list(range(6)))
roc_auc = roc_auc_score(y_bin, final_proba, multi_class='ovr', average='weighted')

print("\n========== PERFORMANCE EVALUATION ==========")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print("=============================================")
print("\n--- Per-Class Report ---")
print(classification_report(true_labels, final_preds, target_names=CLASS_NAMES, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(true_labels, final_preds)
fig, ax = plt.subplots(figsize=(8,6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix — Hybrid (SVM + DNN)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved → confusion_matrix.png")

# ROC Curves
fig, ax = plt.subplots(figsize=(8,6))
for i, cls in enumerate(CLASS_NAMES):
    RocCurveDisplay.from_predictions(y_bin[:,i], final_proba[:,i], name=cls, ax=ax)
ax.plot([0,1],[0,1],'k--')
ax.set_title("ROC Curves — Hybrid Model (One-vs-Rest)")
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
print("ROC curves saved → roc_curves.png")
