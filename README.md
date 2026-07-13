---
title: Dog Skin Disease Classifier
emoji: 🐾
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# 🐾 Dog Skin Disease Classifier

A hybrid deep learning system to detect **6 dog skin diseases** from images.

## Model Performance
- **Accuracy:** 94.46%
- **ROC-AUC:** 0.9934

## Diseases Detected
| Disease | F1 Score |
|---|---|
| Demodicosis | 99% |
| Ringworm | 96% |
| Dermatitis | 95% |
| Healthy | 93% |
| Hypersensitivity | 92% |
| Fungal Infections | 87% |

## Architecture
**ResNet50** (feature extraction) → **StandardScaler** → **SVM (RBF kernel)**

## Tech Stack
TensorFlow • Keras • scikit-learn • FastAPI • OpenCV
