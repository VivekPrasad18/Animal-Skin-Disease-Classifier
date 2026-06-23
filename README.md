# Animal Skin Disease Classifier

## 📌 Overview
A hybrid deep learning computer vision pipeline designed to classify 6 distinct classes of canine skin diseases from image data. The system utilizes a dual-model approach, leveraging a pretrained Convolutional Neural Network for feature extraction and a hyperparameter-tuned machine learning classifier for final prediction.

## 📊 Dataset & Classes
The model was trained on a meticulously curated dataset of 4,315 images (Train: 3022 | Valid: 860 | Test: 433). 
It successfully identifies the following 6 classes:
* Dermatitis
* Fungal Infections
* Healthy
* Hypersensitivity
* Demodicosis
* Ringworm

## 🧠 System Architecture
The pipeline is structured to maximize accuracy while mitigating overfitting:
1. **Feature Extraction:** Images are resized to 224x224 and passed through a pretrained **ResNet50** (include_top=False, pooling='avg') to extract a 2048-dimensional feature vector.
2. **Normalization:** Features are normalized using a fitted `StandardScaler`.
3. **Classification:** Final predictions are made using a **Support Vector Machine (SVM)** with an RBF kernel (C=100, gamma=0.001), tuned with balanced class weights.

*(Note: Ensure your horizontal architecture diagram file—e.g., `architecture.png`—is in your main folder, then remove this text and uncomment the line below to display it)*
<!-- ![System Architecture](architecture.png) -->

## 📈 Performance Metrics
The model achieved highly reliable diagnostic metrics on the final test set:
* **Accuracy:** 94.46%
* **F1-Score:** 94.47%
* **Precision:** 94.49%
* **Recall:** 94.46%
* **ROC-AUC:** 0.9934

## 🛡️ Anti-Overfitting Techniques
To ensure the model generalizes well to new, unseen data, several strict anti-overfitting measures were implemented:
* Aggressive image data augmentation (rotation, zoom, flip, brightness, channel shifts)
* Dropout (0.5 + 0.3) and L2 Regularization
* Label smoothing (0.1)
* Early Stopping (patience=5) and ReduceLROnPlateau

## ⚙️ Tech Stack
* **Languages:** Python 3.12
* **Frameworks:** TensorFlow 2.21, Keras, Scikit-learn 1.8
* **Tools:** OpenCV, NumPy, Matplotlib, joblib, VSCode

## 🚀 How to Run Locally
1. Clone this repository to your local machine.
2. Open the project folder in **VSCode**.
3. Open the integrated terminal and install dependencies:
   `pip install -r requirements.txt`
4. Run the full prediction pipeline:
   `python predict.py`
