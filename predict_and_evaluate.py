import numpy as np
import cv2
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================
# CONFIG
# ============================
IMG_SIZE = (224, 224)
TEST_PATH = "C:/Users/vivek/OneDrive/Documents/python/AnimalDisease/dataset/test"

class_names = [
    "demodicosis",
    "dermatitis",
    "fungal infection",
    "healthy",
    "hypersensitivity",
    "ringworm"
]

# ============================
# LOAD MODELS
# ============================
print("Loading models...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

svm = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Models loaded!")

# ============================
# PREPROCESS IMAGE
# ============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Invalid image path")

    h, w, _ = img.shape
    crop = img[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)]

    crop = cv2.resize(crop, (224, 224))
    crop = preprocess_input(crop)
    crop = np.expand_dims(crop, axis=0)

    return crop

def multi_crop_predict(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    crops = [
        img,  # full
        img[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)],
        img[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
    ]

    probs_list = []

    for c in crops:
        c = cv2.resize(c, (224,224))
        c = preprocess_input(c)
        c = np.expand_dims(c, axis=0)

        f = base_model.predict(c)
        f = scaler.transform(f)

        p = svm.predict_proba(f)[0]
        probs_list.append(p)

    final_prob = np.mean(probs_list, axis=0)
    pred = np.argmax(final_prob)
    confidence = np.max(final_prob)

    return pred, confidence

# ============================
# SINGLE IMAGE PREDICTION
# ============================

def predict_single(img_path):
    # Load and display image
    original = cv2.imread(img_path)

    if original is None:
        raise ValueError("Invalid image path")

    # Convert BGR → RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    plt.imshow(original)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    # Now preprocess
    img = preprocess_image(img_path)

    # Feature extraction
    features = base_model.predict(img)
    features = scaler.transform(features)

    # Prediction
    pred = svm.predict(features)[0]
    prob = svm.predict_proba(features)[0]

    confidence = np.max(prob)

    label = class_names[pred]

    if confidence < 0.70:
        label = "No clear disease / Possibly healthy"

    print("\n===== SINGLE IMAGE RESULT =====")
    print("Predicted:", class_names[pred])
    print(f"Confidence: {confidence * 100:.2f}%")

# ============================
# DATASET EVALUATION
# ============================

def evaluate_dataset():
    print("\nEvaluating on test dataset...")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_data = datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    # Extract features
    features = base_model.predict(test_data)
    labels = test_data.classes

    # Scale
    features = scaler.transform(features)

    # Predict
    preds = svm.predict(features)

    # Metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    print("\n===== DATASET METRICS =====")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

# ============================
# RUN
# ============================

if __name__ == "__main__":
    choice = input("1: Predict Image | 2: Evaluate Model\nEnter choice: ")

    if choice == "1":
        path = input("Enter image path: ")
        predict_single(path)

    elif choice == "2":
        evaluate_dataset()

    else:
        print("Invalid option")


import joblib
import os

print("\nSaving models...")

# create folder in ROOT (not inside train)
os.makedirs("../models", exist_ok=True)

joblib.dump(svm, "../models/svm_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Models saved in /models folder!")    