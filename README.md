Animal Skin Disease Diagnostic Engine

🌐 Executive Summary

The Animal Skin Disease Diagnostic Engine is a high-performance, production-ready computer vision solution designed to automate the classification of canine dermatological conditions. By integrating a deep learning feature extraction pipeline with a robust RESTful microservice architecture, the system provides accurate, scalable diagnostics with a decoupled, modern frontend interface.

🚀 Architectural Overview

The engine employs a hybrid classification strategy, optimizing for both feature-rich representations and computational efficiency:

Model Pipeline: Utilizes ResNet50 (pretrained on ImageNet) for feature extraction, followed by a hyperparameter-tuned Support Vector Machine (SVM) classifier for high-precision inference.

Infrastructure: A decoupled architecture featuring a FastAPI backend (microservice) and a Tailwind-powered web frontend, allowing for independent scaling and modular maintenance.

📊 Performance Metrics

The diagnostic pipeline demonstrates industry-standard reliability with rigorous validation:

Metric

Performance

Accuracy

94.46%

Precision

94.49%

Recall

94.46%

F1-Score

94.47%

ROC-AUC

0.9934

🛠 Tech Stack

Deep Learning: TensorFlow 2.21, Keras, ResNet50, OpenCV.

Backend Engineering: Python 3.12, FastAPI (REST API), Uvicorn, Scikit-learn 1.8.

Frontend Engineering: HTML5, Tailwind CSS, JavaScript (Fetch API).

Deployment Tools: Git, VSCode, RESTful API Design.

📂 Key Features

Production-Ready Microservice: Refactored inference logic into an independent API endpoint (/api/v1/diagnose), ready for integration with mobile apps or clinical dashboards.

Robust Generalization: Implemented advanced anti-overfitting protocols, including multi-modal image augmentation, L2 regularization, and Early Stopping.

Decoupled Frontend: Built a responsive, asynchronous user interface that handles image upload and real-time result visualization without page refreshes.

⚙️ Deployment Instructions

1. Environment Setup

# Clone the repository
git clone [https://github.com/VivekPrasad18/Animal-Skin-Disease-Classifier](https://github.com/VivekPrasad18/Animal-Skin-Disease-Classifier)
cd Animal-Skin-Disease-Classifier

# Install dependencies
pip install -r requirements.txt


2. Launching the System

 1. Initialize the Backend:

  python api.py


 2. Launch the User Interface:Open index.html in any standard web browser. The interface will automatically sync with the local FastAPI microservice.

Developed as a scalable AI/ML solution for clinical dermatological support.
