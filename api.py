import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn

app = FastAPI(
    title="Deep Learning Diagnostic Engine API",
    description="REST API microservice for automated animal skin disease classification."
)

# Allow cross-origin requests (essential for microservices connecting to frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Teleport users automatically to the docs page so you don't have to type it!
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

print("Loading ResNet50 Diagnostic Model into memory...")
try:
    # We use the exact path found in your workspace
    model = tf.keras.models.load_model("train/resnet_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model. Ensure 'train/resnet_model.h5' exists. Error: {e}")
    model = None

# TODO: Update these to match your exact 6 skin disease classes
CLASS_NAMES = [
    "demodicosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "ringworm"
]

def preprocess_image(image_bytes: bytes):
    """
    Transforms raw uploaded image bytes into the exact mathematical tensor 
    format expected by the ResNet50 architecture.
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Ensure it has 3 channels (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to the standard ResNet50 input size
    image = image.resize((300, 300))
    img_array = np.array(image)
    
    # Expand dimensions to create a "batch" of 1 image
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values between 0 and 1
    img_array = img_array / 255.0  
    
    return img_array

@app.post("/api/v1/diagnose")
async def diagnose_image(file: UploadFile = File(...)):
    """
    Accepts an image file payload, runs it through the preprocessing pipeline, 
    and returns a structured diagnostic prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="The AI model is currently offline or failed to load.")
    
    try:
        # 1. Read the raw file stream
        image_bytes = await file.read()
        
        # 2. Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # 3. Run the Deep Learning prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = float(predictions[0][predicted_class_idx])
        
        # 4. Return a structured JSON response (Microservice standard)
        return {
            "status": "success",
            "filename": file.filename,
            "diagnostics": {
                "prediction_class": CLASS_NAMES[predicted_class_idx],
                "confidence_score": round(confidence_score, 4),
                "model_version": "ResNet50-v1.2",
                "architecture": "Microservice API"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 API is running! Ctrl+Click the link below to test it:")
    print("👉 http://127.0.0.1:8000 👈")
    print("="*50 + "\n")
    # Changed host to 127.0.0.1 so it creates a valid link in your browser
    uvicorn.run(app, host="127.0.0.1", port=8000)