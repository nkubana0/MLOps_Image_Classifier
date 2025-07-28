from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import os
import sys
from datetime import datetime
import numpy as np
import asyncio
import uuid

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from prediction import predict_image, load_inference_model, CLASS_NAMES
from preprocessing import load_and_preprocess_data
from model import build_model, train_model

app = FastAPI(
    title="MLOps CIFAR-10 Image Classifier API",
    description="API for predicting CIFAR-10 images and triggering model retraining.",
    version="1.0.0"
)

# --- API Models for Request/Response ---

class PredictionResponse(BaseModel):
    predicted_class_index: int
    predicted_class_name: str
    probabilities: dict[str, float]

class RetrainTriggerResponse(BaseModel):
    message: str
    status: str
    retraining_initiated_at: str
    model_update_behavior: str = "new_model_loaded_in_memory" # Indicates how the new model is made active

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    model_loaded: bool

# --- Global state for model and retraining ---
is_retraining_in_progress = False
# Directory to simulate new incoming data storage
INCOMING_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'incoming_for_retraining')
os.makedirs(INCOMING_DATA_DIR, exist_ok=True)


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """
    Load the model when the FastAPI application starts.
    """
    try:
        load_inference_model()
        print("Model pre-loaded during API startup.")
    except Exception as e:
        print(f"Failed to pre-load model during startup: {e}. API may not function correctly.")
        # In a real app, you might want to raise an exception here to prevent startup
        # if the model is absolutely critical, or implement a retry mechanism.

# --- Core retraining function to run in background ---
async def perform_retraining_task():
    """
    Performs the full model retraining process. This function is designed to be run
    as a background task, allowing the API to return immediately.
    """
    global is_retraining_in_progress
    print("\n--- Background retraining task started ---")
    try:
        # Step 1: Data Acquisition and Preprocessing
        # In a real scenario, this would involve loading data from a persistent store,
        # potentially including the newly uploaded data. For CIFAR-10, we re-load
        # the entire dataset, simulating fresh data incorporation.
        print("Retraining: Loading and preprocessing data (simulating new data integration)...")
        x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()
        print("Retraining: Data loaded and preprocessed.")

        # Step 2: Build a new model instance
        print("Retraining: Building new model instance...")
        input_shape = x_train.shape[1:]
        num_classes = y_train.shape[1]
        new_model_instance = build_model(input_shape=input_shape, num_classes=num_classes)
        print("Retraining: New model instance built.")

        # Step 3: Train the new model
        print("Retraining: Starting model training...")
        # Crucial for TensorFlow to clear previous graph and prevent memory leaks/conflicts
        tf.keras.backend.clear_session()
        _, _ = train_model(new_model_instance, x_train, y_train, x_val, y_val, epochs=50, batch_size=64)
        print("Retraining: Model training completed.")

        # Step 4: Reload the newly trained model for inference
        # This "hot-swaps" the model in the running API process.
        load_inference_model()
        print("Retraining: Newly trained model loaded into API for live inference.")
        print("--- Background retraining task completed successfully ---")

    except Exception as e:
        print(f"--- Background retraining task failed: {e} ---")
    finally:
        is_retraining_in_progress = False
        print("Retraining status flag reset to False.")


# --- API Endpoints ---

@app.get("/health", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    """
    Checks the health status of the API and if the ML model is loaded.
    """
    model_status = True if load_inference_model() is not None else False
    return {
        "status": "healthy",
        "message": "API is operational.",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_status
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Image Class")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and returns the predicted class
    and class probabilities.
    - **file**: Upload your image file (e.g., JPEG, PNG).
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        image_stream = io.BytesIO(image_bytes)
        prediction_result = predict_image(image_stream)
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/retrain/trigger", response_model=RetrainTriggerResponse, summary="Trigger Model Retraining")
async def trigger_retraining_endpoint(background_tasks: BackgroundTasks, new_data: UploadFile = File(...)):
    """
    Accepts new data and triggers the model retraining process in the background.
    The API will return immediately, allowing retraining to happen asynchronously.

    In a real-world scenario, this `new_data` would be saved to a persistent data
    store (e.g., Google Cloud Storage, S3). For this CIFAR-10 based solution,
    we simulate saving the file, and its upload acts as the signal to
    re-run the full preprocessing and training pipeline with the original source.

    - **new_data**: Upload a file containing new data (e.g., a zip of images). Its
      content will be simulated as saved to disk, and its presence signals intent to retrain.
    """
    global is_retraining_in_progress

    if is_retraining_in_progress:
        raise HTTPException(status_code=409, detail="Retraining already in progress. Please wait.")

    # --- Real-world simulation for handling new_data ---
    # In a production system:
    # 1. New data is typically saved to a versioned data lake/storage (e.g., GCS, S3).
    # 2. This API endpoint would then trigger a separate, possibly long-running,
    #    training job (e.g., via a message queue like Pub/Sub, or directly launching
    #    a Vertex AI Training job).
    #
    # For this assignment, we'll simulate saving the uploaded file and then
    # adding our `perform_retraining_task` to FastAPI's background tasks.
    # The `new_data` content is just a placeholder trigger for CIFAR-10.
    new_data_filename = new_data.filename
    try:
        # Generate a unique filename to avoid overwrites and simulate persistent storage
        unique_filename = f"{uuid.uuid4()}_{new_data_filename}"
        file_path = os.path.join(INCOMING_DATA_DIR, unique_filename)

        # Read the uploaded file content and save it
        file_content = await new_data.read()
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        print(f"Simulated new data '{new_data_filename}' saved to '{file_path}'.")
        # In a real system, you might now add this file path/metadata to a database
        # or a manifest for the training job to pick up.

    except Exception as e:
        print(f"Failed to process and save new data file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process and save new data file: {e}")


    is_retraining_in_progress = True
    # Add the retraining function to background tasks.
    # This makes the API return immediately, while perform_retraining_task runs in the background.
    background_tasks.add_task(perform_retraining_task)

    return {
        "message": f"Model retraining successfully triggered in background with new data '{new_data_filename}'. Check API logs for progress.",
        "status": "triggered",
        "retraining_initiated_at": datetime.now().isoformat(),
        "model_update_behavior": "new_model_loaded_in_memory"
    }