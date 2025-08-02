import streamlit as st
import requests
import os
import json
import base64
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Update this URL to your deployed Hugging Face Space URL
# Replace 'YOUR_USERNAME-YOUR_SPACE_NAME.hf.space' with your actual Space URL
API_BASE_URL = "https://nkubana-cifar10-mlops-demo.hf.space" # <--- IMPORTANT: UPDATE THIS LINE
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
RETRAIN_TRIGGER_ENDPOINT = f"{API_BASE_URL}/retrain/trigger"

# Directory where your visualization images are saved from the Jupyter notebook
VIZ_DIR = os.path.join(os.getcwd(), 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True) # Ensure the visualization directory exists

# Class names for CIFAR-10 (must match those used in src/prediction.py)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# --- Helper Functions to Interact with FastAPI ---

def get_health_status():
    """Fetches health status from the FastAPI backend."""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI backend at {API_BASE_URL}. Is it running?")
        return {"status": "error", "message": "Backend unavailable", "model_loaded": False}
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error checking health: {e}")
        return {"status": "error", "message": f"HTTP Error: {e}", "model_loaded": False}
    except Exception as e:
        st.error(f"An unexpected error occurred while checking health: {e}")
        return {"status": "error", "message": f"Unexpected Error: {e}", "model_loaded": False}

def predict_image_via_api(image_file):
    """Sends an image file to the FastAPI /predict endpoint."""
    try:
        # Streamlit's file_uploader provides a SpooledTemporaryFile-like object
        # requests.post needs to read its content.
        # Ensure the file pointer is at the beginning before reading.
        image_file.seek(0)
        files = {'file': (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(PREDICT_ENDPOINT, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI backend at {API_BASE_URL}. Is it running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed with HTTP error {response.status_code}: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None

def trigger_retraining_via_api(dummy_file_for_trigger):
    """Triggers retraining via the FastAPI /retrain/trigger endpoint."""
    st.info("Sending retraining trigger to backend...")
    try:
        # Ensure the dummy file pointer is at the beginning
        dummy_file_for_trigger.seek(0)
        files = {'new_data': (dummy_file_for_trigger.name, dummy_file_for_trigger.getvalue(), dummy_file_for_trigger.type)}
        response = requests.post(RETRAIN_TRIGGER_ENDPOINT, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI backend at {API_BASE_URL}. Is it running?")
        return {"status": "error", "message": "Backend unavailable"}
    except requests.exceptions.HTTPError as e:
        st.error(f"Retraining trigger failed with HTTP error {response.status_code}: {response.text}")
        return {"status": "error", "message": f"HTTP Error: {response.text}"}
    except Exception as e:
        st.error(f"An unexpected error occurred while triggering retraining: {e}")
        return {"status": "error", "message": f"Unexpected Error: {e}"}

# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="MLOps CIFAR-10 Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 MLOps CIFAR-10 Image Classifier")
st.markdown("A demonstration of an end-to-end MLOps pipeline for image classification.")

# --- Sidebar for Navigation and Controls ---
st.sidebar.title("Navigation & Controls")
page_selection = st.sidebar.radio("Go to", ["Dashboard", "Predict Image", "Retrain Model", "Data Insights"])

st.sidebar.markdown("---")
st.sidebar.header("API Status")

# Fetch health status using st.rerun for live updates if needed,
# but for simple sidebar, just get it once per load.
# This will be run every time Streamlit re-executes the script.
health_data = get_health_status()
if health_data:
    st.sidebar.write(f"**Backend Status:** :green[{health_data.get('status', 'N/A')}]")
    st.sidebar.write(f"**Model Loaded:** :blue[{'Yes' if health_data.get('model_loaded') else 'No'}]")
    st.sidebar.write(f"Last checked: {pd.to_datetime('now').strftime('%H:%M:%S')}") # Use pandas for datetime format
    if not health_data.get('model_loaded'):
        st.sidebar.warning("Model not loaded in backend. Predictions might fail.")
else:
    st.sidebar.error("Backend not reachable!")

# --- Main Content Area ---

if page_selection == "Dashboard":
    st.header("Dashboard Overview")
    st.markdown("""
    Welcome to the MLOps CIFAR-10 Image Classifier dashboard!
    This application demonstrates a complete machine learning pipeline including:
    - **Data Preprocessing**
    - **Model Training & Evaluation**
    - **API Deployment**
    - **User Interface for Prediction & Retraining**
    - **Monitoring (via Health Check)**

    Use the sidebar to navigate through the different functionalities.
    """)

    st.subheader("Current System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="API Health", value=health_data.get('status', 'Unknown'))
    with col2:
        st.metric(label="ML Model Loaded", value="Yes" if health_data.get('model_loaded') else "No")

    st.info("Before using Predict or Retrain, ensure your FastAPI backend is running. "
            "Start it by navigating to your project root in the terminal and running: "
            "`uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`")

elif page_selection == "Predict Image":
    st.header("Predict Image Class")
    st.write("Upload an image and the model will predict its class (e.g., airplane, cat, dog).")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Predicting...")

        # Call API for prediction
        prediction_result = predict_image_via_api(uploaded_file)

        if prediction_result:
            st.success("Prediction Complete!")
            st.metric(label="Predicted Class", value=prediction_result['predicted_class_name'])

            st.subheader("Probabilities:")
            # Convert probabilities dict to DataFrame for nice display
            probabilities_df = pd.DataFrame(prediction_result['probabilities'].items(),
                                            columns=['Class', 'Probability']).sort_values(by='Probability', ascending=False)
            st.dataframe(probabilities_df, use_container_width=True, hide_index=True)


elif page_selection == "Retrain Model":
    st.header("Trigger Model Retraining")
    st.write("Initiate a new model training process using a potentially updated dataset.")
    st.warning("Retraining can take several minutes. The API will remain responsive, but the model will only update after retraining is complete.")

    if st.button("Start Retraining"):
        # Create a dummy file to send as `new_data` for the retraining trigger.
        # Its content is not processed for CIFAR-10, but its presence signals the intent to retrain.
        dummy_file_content = "retrain_trigger_signal"
        dummy_file_bytes = io.BytesIO(dummy_file_content.encode())
        dummy_file_bytes.name = "retrain_trigger.txt" # Assign a name for the UploadFile mock
        dummy_file_bytes.type = "text/plain"

        with st.spinner("Retraining in progress in the backend... This may take a while."):
            retrain_response = trigger_retraining_via_api(dummy_file_bytes)

        if retrain_response:
            if retrain_response.get('status') == 'triggered':
                st.success(f"Retraining triggered successfully! {retrain_response.get('message')}")
                st.info("Check your FastAPI backend terminal for live retraining progress.")
                st.experimental_rerun() # Rerun to update API status in sidebar
            else:
                st.error(f"Failed to trigger retraining: {retrain_response.get('message')}")
        else:
            st.error("No response received from retraining trigger.")

    st.markdown("---")
    st.subheader("Retraining Notes (Real-World Context):")
    st.write("""
    In a fully productionized MLOps pipeline, new data uploaded here would be:
    1.  **Saved persistently:** Stored in a cloud object storage (e.g., Google Cloud Storage, S3) with versioning.
    2.  **Orchestrated:** The API's role is to *trigger* a separate, long-running training job. This could be done by sending a message to a queue (e.g., Google Cloud Pub/Sub, Kafka), which a dedicated training service (e.g., Google Cloud Run job, Vertex AI Training, Kubeflow Pipeline) listens to.
    3.  **Non-blocking:** The API itself returns immediately, allowing it to continue serving predictions while the retraining happens independently.
    4.  **Model Registry:** Once retraining is complete, the new model is typically registered in a Model Registry (e.g., MLflow Model Registry, Vertex AI Model Registry), where it can be versioned and approved.
    5.  **Deployment:** A continuous deployment (CD) pipeline would then pick up the new model from the registry and deploy it to the serving endpoint, potentially with A/B testing or canary deployments.

    For this assignment, we implement the non-blocking trigger and hot-swapping of the model in memory. The `new_data` file is consumed and conceptually signals the retraining, which re-loads the built-in CIFAR-10 dataset to simulate the effect of 'new' data.
    """)

elif page_selection == "Data Insights":
    st.header("Data Insights & Visualizations")
    st.write("Explore key characteristics of the CIFAR-10 dataset used for training the model, and evaluate model performance.")
    st.markdown("---")

    st.subheader("Class Distribution")
    class_dist_path = os.path.join(VIZ_DIR, 'class_distribution.png')
    if os.path.exists(class_dist_path):
        st.image(class_dist_path, caption='Class Distribution in Training Set', use_column_width=True)
        st.markdown("""
        **Insight:** The CIFAR-10 dataset is perfectly balanced across its 10 classes.
        This is ideal for classification tasks as it prevents the model from being biased
        towards any particular class due to unequal representation, leading to more generalized learning.
        """)
    else:
        st.warning(f"Class distribution plot not found at {class_dist_path}. "
                   "Please ensure the Jupyter notebook (`notebook/project_name.ipynb`) has been run "
                   "and the plots are saved to the 'visualizations' directory.")

    st.subheader("Sample Images")
    sample_images_path = os.path.join(VIZ_DIR, 'sample_images.png')
    if os.path.exists(sample_images_path):
        st.image(sample_images_path, caption='Sample Training Images', use_column_width=True)
        st.markdown("""
        **Insight:** These sample images illustrate the 32x32 pixel resolution and color complexity
        of the CIFAR-10 dataset. Despite their small size, human perception can easily identify
        the objects. The model must learn to extract features from these compact representations.
        """)
    else:
        st.warning(f"Sample images plot not found at {sample_images_path}. "
                   "Please ensure the Jupyter notebook (`notebook/project_name.ipynb`) has been run "
                   "and the plots are saved to the 'visualizations' directory.")

    st.subheader("Pixel Value Distribution (Normalized)")
    pixel_dist_path = os.path.join(VIZ_DIR, 'pixel_distribution.png')
    if os.path.exists(pixel_dist_path):
        st.image(pixel_dist_path, caption='Distribution of Pixel Intensities (Normalized)', use_column_width=True)
        st.markdown("""
        **Insight:** The histograms clearly show that pixel intensities for all three color channels
        are uniformly distributed within the 0 to 1 range, validating the normalization step of our preprocessing pipeline.
        This critical scaling ensures numerical stability during neural network training and prevents gradient explosion/vanishing issues.
        """)
    else:
        st.warning(f"Pixel distribution plot not found at {pixel_dist_path}. "
                   "Please ensure the Jupyter notebook (`notebook/project_name.ipynb`) has been run "
                   "and the plots are saved to the 'visualizations' directory.")

    st.subheader("Model Evaluation: Confusion Matrix")
    confusion_matrix_path = os.path.join(VIZ_DIR, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption='Confusion Matrix for CIFAR-10 Classification', use_column_width=True)
        st.markdown("""
        **Insight from Confusion Matrix:** The diagonal elements of the confusion matrix represent the number of correct predictions for each class.
        High values on the diagonal indicate strong performance. Off-diagonal elements highlight misclassifications; for example,
        a high value in row 'cat' and column 'dog' means the model frequently misidentifies cats as dogs.
        Analyzing these patterns helps in understanding specific areas where the model might be struggling (e.g., visually similar classes).
        """)
    else:
        st.warning(f"Confusion Matrix plot not found at {confusion_matrix_path}. "
                   "Please ensure the Jupyter notebook (`notebook/project_name.ipynb`) has been run "
                   "and the plots are saved to the 'visualizations' directory.")

    st.subheader("Training History (Accuracy & Loss)")
    accuracy_plot_path = os.path.join(VIZ_DIR, 'training_accuracy.png')
    loss_plot_path = os.path.join(VIZ_DIR, 'training_loss.png')

    col_acc_loss_1, col_acc_loss_2 = st.columns(2)
    with col_acc_loss_1:
        if os.path.exists(accuracy_plot_path):
            st.image(accuracy_plot_path, caption='Model Training & Validation Accuracy', use_column_width=True)
        else:
            st.warning(f"Training Accuracy plot not found at {accuracy_plot_path}.")
    with col_acc_loss_2:
        if os.path.exists(loss_plot_path):
            st.image(loss_plot_path, caption='Model Training & Validation Loss', use_column_width=True)
        else:
            st.warning(f"Training Loss plot not found at {loss_plot_path}.")

    if os.path.exists(accuracy_plot_path) or os.path.exists(loss_plot_path):
        st.markdown("""
        **Insight from Training History:** These plots illustrate the model's learning progression over epochs.
        The `accuracy` plot shows how well the model performed on training and validation
        data, while the `loss` plot indicates the error. Early stopping prevented
        overfitting by stopping training when validation loss ceased to improve,
        restoring the best weights. This demonstrates effective regularization.
        """)
    else:
        st.warning("Training history plots not found. Please ensure `training_accuracy.png` and `training_loss.png` "
                   "are saved in the 'visualizations' directory from your Jupyter notebook.")