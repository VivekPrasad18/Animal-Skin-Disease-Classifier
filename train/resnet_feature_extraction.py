import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import pandas as pd

# ============================
# CONFIG
# ============================
TRAIN_PATH = "../dataset/train"
VAL_PATH = "../dataset/valid"
TEST_PATH = "../dataset/test"

IMG_SIZE = (224,224)
BATCH_SIZE = 16

# 👉 Replace with your CNN test accuracy
cnn_acc = 0.9469   # ← from your ResNet output

# ============================
# DATA GENERATORS
# ============================

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_data = datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_data = datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ============================
# LOAD RESNET (FEATURE EXTRACTOR)
# ============================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ============================
# FEATURE EXTRACTION
# ============================

def extract_features(generator):
    features = base_model.predict(generator, verbose=1)
    labels = generator.classes
    return features, labels

print("\nExtracting features...")

X_train, y_train = extract_features(train_data)
X_val, y_val = extract_features(val_data)
X_test, y_test = extract_features(test_data)

# ============================
# SCALING
# ============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ============================
# SVM MODEL
# ============================

print("\nTraining SVM...")

svm = SVC(
    kernel='rbf',
    C=30,
    gamma='auto',
    class_weight='balanced',
    probability=True
)

svm.fit(X_train, y_train)



# ============================
# VALIDATION EVALUATION
# ============================

val_pred = svm.predict(X_val)

val_acc = accuracy_score(y_val, val_pred)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

print("\nValidation Report:")
print(classification_report(y_val, val_pred))

# ============================
# TEST EVALUATION + ENSEMBLE
# ============================

# 🔹 SVM probabilities
svm_prob = svm.predict_proba(X_test)

# 🔹 Load your trained CNN model
from tensorflow.keras.models import load_model

cnn_model = load_model("resnet_model.h5")  # make sure file exists

# 🔹 Get CNN predictions
cnn_prob = cnn_model.predict(test_data)

# 🔹 Combine (Ensemble)
final_prob = 0.6 * cnn_prob + 0.4 * svm_prob

# 🔹 Convert to class predictions
final_pred = np.argmax(final_prob, axis=1)

# 🔹 Accuracy
final_acc = accuracy_score(y_test, final_pred)

print(f"\nFinal Hybrid (CNN + SVM) Accuracy: {final_acc * 100:.2f}%")

print("\nFinal Classification Report:")
print(classification_report(y_test, final_pred))

test_precision = precision_score(y_test, final_pred, average='weighted')
test_recall = recall_score(y_test, final_pred, average='weighted')
test_f1 = f1_score(y_test, final_pred, average='weighted')

# ============================
# CONFUSION MATRIX
# ============================

cm = confusion_matrix(y_test, final_pred)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix - Hybrid Model")
plt.show()

# ============================
# MODEL COMPARISON TABLE
# ============================

results = {
    "Model": ["CNN (ResNet50)", "Hybrid (ResNet + SVM)"],
    "Accuracy": [cnn_acc,final_acc],
    "Precision": [None, test_precision],
    "Recall": [None, test_recall],
    "F1-Score": [None, test_f1]
}

df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(df)

# ============================
# BAR GRAPH
# ============================

models = ["CNN", "Hybrid"]
accuracy = [cnn_acc, final_acc]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()