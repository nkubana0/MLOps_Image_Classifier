import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image # For image processing in prediction
import io # New import for handling bytes in memory

# Define paths relative to the project root
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_FILENAME = 'cifar10_cnn_model.h5'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# Global variable to store the loaded model
_model = None
# Global variable for class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_inference_model():
    """
    Loads the trained Keras model from the specified path.
    Uses a global variable to ensure the model is loaded only once.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first by running src/model.py.")
        try:
            _model = load_model(MODEL_PATH)
            print(f"Model '{MODEL_FILENAME}' loaded successfully for inference.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            _model = None # Ensure model is None if loading fails
    return _model

def preprocess_image_for_prediction(image_bytes_io): # Changed parameter name for clarity
    """
    Preprocesses a raw image (bytes in BytesIO) for model prediction.
    Converts to PIL Image, resizes, normalizes, and adds batch dimension.
    """
    try:
        # Image.open can take a file-like object (BytesIO) directly
        image = Image.open(image_bytes_io).convert('RGB') # Ensure 3 channels
        image = image.resize((32, 32)) # Resize to CIFAR-10 expected input
        image_array = np.array(image).astype('float32') / 255.0 # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension (1, 32, 32, 3)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_image(image_bytes_io): # Changed parameter name for clarity
    """
    Makes a prediction on a single image provided as BytesIO object.
    Loads the model if not already loaded, preprocesses the image,
    and returns the predicted class name and probabilities.
    """
    model = load_inference_model()
    if model is None:
        raise RuntimeError("Machine Learning model is not loaded. Cannot make prediction.")

    processed_image = preprocess_image_for_prediction(image_bytes_io)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    probabilities = predictions[0].tolist() # Convert numpy array to list for JSON response

    return {
        "predicted_class_index": predicted_class_index,
        "predicted_class_name": predicted_class_name,
        "probabilities": dict(zip(CLASS_NAMES, probabilities)) # Map class names to probabilities
    }

if __name__ == "__main__":
    print("Running prediction.py directly for testing...")

    # For testing, let's create a very simple in-memory image (e.g., solid red)
    # This completely avoids file path issues for the internal test.
    # In the actual FastAPI app, you'll receive real image bytes from an upload.
    test_image_pil = Image.new('RGB', (32, 32), color='red')
    byte_io = io.BytesIO()
    test_image_pil.save(byte_io, format='PNG') # Save to BytesIO object
    byte_io.seek(0) # Rewind the buffer to the beginning

    try:
        prediction_result = predict_image(byte_io) # Pass BytesIO directly
        print("\nPrediction Result:")
        print(prediction_result)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your model 'cifar10_cnn_model.h5' exists in the 'models/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        print("This might be due to an issue with image processing or model inference.")
        # Re-raise the exception to see full traceback if still an issue
        # raise
    print("\nLocal prediction test complete.")