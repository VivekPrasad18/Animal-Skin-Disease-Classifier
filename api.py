import io
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import uvicorn

app = FastAPI(
    title="Dog Skin Disease Diagnostic Engine API",
    description="Hybrid ResNet50 + SVM classifier for 6 dog skin diseases."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at root
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("index.html")

# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading models...")
resnet = ResNet50(weights='imagenet', include_top=False,
                  pooling='avg', input_shape=(224, 224, 3))
svm    = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("Models loaded!")

# Exact class order (alphabetical as Keras loads)
CLASS_NAMES = [
    'Dermatitis',
    'Fungal Infections',
    'Healthy',
    'Hypersensitivity',
    'Demodicosis',
    'Ringworm'
]

CLASS_INFO = {
    'Dermatitis':        'Skin inflammation causing redness and itching.',
    'Fungal Infections': 'Fungal infection with scaly or crusty patches.',
    'Healthy':           'No disease detected. Skin appears normal.',
    'Hypersensitivity':  'Allergic reaction causing intense itching.',
    'Demodicosis':       'Demodex mite infection causing hair loss.',
    'Ringworm':          'Circular bald patches with red border.'
}

# ── Predict Endpoint ──────────────────────────────────────────────────────────
@app.post("/api/v1/diagnose")
async def diagnose_image(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np  = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Preprocess → ResNet50 features → scale → SVM predict
        img_resized = cv2.resize(img_bgr, (224, 224))
        inp = np.expand_dims(preprocess_input(img_resized.astype(np.float32)), 0)
        feats = resnet.predict(inp, verbose=0)
        feats = scaler.transform(feats)
        proba = svm.predict_proba(feats)[0]

        idx        = int(np.argmax(proba))
        disease    = CLASS_NAMES[idx]
        confidence = float(np.max(proba))

        return {
            "status": "success",
            "filename": file.filename,
            "diagnostics": {
                "prediction_class": disease,
                "confidence_score": round(confidence, 4),
                "description": CLASS_INFO[disease],
                "model_version": "ResNet50+SVM-v2.0",
                "architecture": "Hybrid CNN-SVM Pipeline",
                "all_probabilities": {
                    CLASS_NAMES[i]: round(float(proba[i]), 4)
                    for i in range(len(CLASS_NAMES))
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("\n" + "="*50)
    print(f"Server running at: http://0.0.0.0:{port}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
