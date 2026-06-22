"""
Anti-overfitting MobileNetV2 + ResNet50-SVM — 6 class dog skin disease
Techniques:
  - Heavy augmentation (rotation, zoom, flip, brightness, channel shift)
  - L2 regularization on Dense layers
  - Dropout 0.5 + 0.3
  - Label smoothing 0.1
  - EarlyStopping patience=5 on val_loss
  - ReduceLROnPlateau
  - Class weights (balanced) for imbalanced dataset
  - ModelCheckpoint saves best val_accuracy only
  - SVM with C=10, gamma=scale, class_weight=balanced
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "dataset", "train")
VAL_PATH   = os.path.join(ROOT, "dataset", "valid")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MOB_SAVE = os.path.join(MODELS_DIR, "mobilenet_best.h5")
SVM_SAVE = os.path.join(MODELS_DIR, "svm_model.pkl")
SCL_SAVE = os.path.join(MODELS_DIR, "scaler.pkl")

import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE    = (224, 224)
BATCH       = 16
NUM_CLASSES = 6

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — MobileNetV2
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 1: MobileNetV2 (anti-overfitting) ===")

aug = ImageDataGenerator(
    preprocessing_function=mob_pre,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    fill_mode="nearest"
)

train_data = aug.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=True)

val_data = ImageDataGenerator(preprocessing_function=mob_pre).flow_from_directory(
    VAL_PATH, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False)

print("Classes:", train_data.class_indices)

# Balanced class weights
cw = compute_class_weight("balanced",
                           classes=np.unique(train_data.classes),
                           y=train_data.classes)
class_weights = dict(enumerate(cw))
print("Class weights:", {k: round(v, 2) for k, v in class_weights.items()})

# Model with L2 + Dropout
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

cb1 = [
    ModelCheckpoint(MOB_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

print("\n--- Frozen base (20 epochs max) ---")
model.fit(train_data, validation_data=val_data, epochs=20,
          callbacks=cb1, class_weight=class_weights)

# Fine-tune top 40 layers only
print("\n--- Fine-tune top 40 layers (20 epochs max) ---")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

cb2 = [
    ModelCheckpoint(MOB_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

model.fit(train_data, validation_data=val_data, epochs=20,
          callbacks=cb2, class_weight=class_weights)

print(f"\nMobileNetV2 saved -> {MOB_SAVE}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — ResNet50 + SVM
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 2: ResNet50 + SVM ===")

resnet = ResNet50(weights="imagenet", include_top=False,
                  pooling="avg", input_shape=(224, 224, 3))

gen = ImageDataGenerator(preprocessing_function=res_pre)
tr  = gen.flow_from_directory(TRAIN_PATH, target_size=IMG_SIZE,
                               batch_size=BATCH, class_mode="categorical", shuffle=False)
vl  = gen.flow_from_directory(VAL_PATH, target_size=IMG_SIZE,
                               batch_size=BATCH, class_mode="categorical", shuffle=False)

print("Extracting features (train)...")
X_train = resnet.predict(tr, verbose=1)
y_train = tr.classes

print("Extracting features (val)...")
X_val = resnet.predict(vl, verbose=1)
y_val = vl.classes

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# C=10 (less than 30) + gamma=scale + balanced weights = less overfit
print("Training SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale",
          class_weight="balanced", probability=True)
svm.fit(X_train, y_train)

val_preds = svm.predict(X_val)
print(f"\nSVM Val Accuracy: {accuracy_score(y_val, val_preds)*100:.2f}%")
print(classification_report(y_val, val_preds, zero_division=0))

joblib.dump(svm, SVM_SAVE)
joblib.dump(scaler, SCL_SAVE)
print(f"\nAll models saved -> {MODELS_DIR}")
print("Done. Models are regularized and should generalize well.")
