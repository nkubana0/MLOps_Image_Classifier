# MLOps Image Classifier for CIFAR-10

This project demonstrates a complete MLOps pipeline for an image classification task using the CIFAR-10 dataset. The pipeline covers the entire lifecycle, from model development and containerization to cloud deployment, a web-based user interface, and model retraining capabilities. The deployed service provides real-time predictions and is monitored for performance.

-----

## Project Structure

The repository is organized to follow best practices for MLOps projects, separating code for model, preprocessing, and API logic.

```
MLOps_Image_Classifier/
├── README.md                  # Project overview and instructions
├── notebook/
│   └── cifar10_classification.ipynb  # Jupyter Notebook for initial model development and evaluation
├── src/
│   ├── preprocessing.py       # Script for data loading, normalization, and splitting
│   ├── model.py               # Script for defining the CNN model architecture and retraining logic
│   ├── prediction.py          # Script for model loading and prediction function
│   ├── api.py                 # FastAPI application with prediction and retraining endpoints
│   └── app.py                 # Streamlit-based web user interface
├── Dockerfile                 # Dockerfile for containerizing the FastAPI application
├── requirements.txt           # Python dependencies for the project
└── .gitignore                 # Files to ignore in Git (e.g., model files, virtual environments)
```

## Solution Overview

  * **Model:** A Convolutional Neural Network (CNN) is used for image classification, trained on the CIFAR-10 dataset. The model incorporates regularization techniques like Dropout and L2 regularization to prevent overfitting.

  * **API:** A REST API is built using **FastAPI** to serve model predictions. It exposes three key endpoints:

      * `/predict`: Accepts an image and returns a class prediction.

      * `/upload_data`: Accepts a ZIP file of new images for model retraining.

      * `/trigger_retrain`: Triggers an asynchronous retraining process.

  * **UI:** An interactive web interface is developed with **Streamlit**. It provides a dashboard for data visualization, an interface to make predictions on new images, and a mechanism to upload new data and trigger model retraining.

  * **Deployment:** The entire application is containerized using **Docker**. The Docker image is built in the cloud and deployed as a live web service on **Render.com**.

  * **Retraining:** The pipeline supports model retraining. New data can be uploaded via the UI, and a dedicated API endpoint triggers a background process to fine-tune the existing model with the new data.

  * **Performance Monitoring:** The API's performance and resilience are validated through a flood request simulation using **Locust**, an open-source load-testing tool.

## Setup Instructions

### Prerequisites

  * A GitHub account

  * A Render account (signed up with GitHub)

  * A code editor (e.g., VS Code)

  * Python 3.9+ and `pip`

  * Docker Desktop (for local testing, if desired)

### Deployment on Render.com

This project is designed for seamless deployment on Render.com directly from your GitHub repository.

1.  **Fork or Clone this Repository:**

      * Ensure your `Dockerfile`, `requirements.txt`, and the `src/` directory are in the root of your repository.

2.  **Create a New Render Web Service:**

      * Log in to your Render dashboard.

      * Click **"New" \> "Web Service"**.

      * Select your GitHub repository (`MLOps_Image_Classifier`).

      * **Configuration:**

          * **Name:** `cifar10-mlops-api` (or a name of your choice).

          * **Root Directory:** Leave this empty.

          * **Region:** Choose a region close to you.

          * **Branch:** `main` (or `master`).

          * **Runtime:** `Docker`.

          * **Start Command:** Leave empty.

          * **Plan:** Select the **`Free`** plan.

3.  **Deploy:**

      * Click **"Create Web Service"**.

      * Render will automatically build the Docker image and deploy your application. You can monitor the build progress in the logs.

## Demo Instructions

### Live Application

The live API and UI for this project can be accessed at:

  * **Render API URL:** `https://your-service-name.onrender.com`

  * **Swagger UI:** `https://your-service-name.onrender.com/docs`

The Streamlit UI can be run locally and configured to point to the deployed API.

### Video Demonstration

The video demo showcases the complete end-to-end functionality of the MLOps pipeline.

  * **Video Demo Link:** `[Insert your YouTube video link here]`

The video covers the following key aspects:

  * **UI Walkthrough:** A tour of the Streamlit dashboard, including data visualizations for class distribution and sample images.

  * **Live Prediction:** Demonstrates uploading an image and receiving a real-time prediction from the deployed model.

  * **Retraining Process:** Shows the UI elements for uploading new data and triggering a model retraining event.

  * **Performance Monitoring:** Presents the results of a flood request simulation using Locust, highlighting the model's latency and throughput under different loads.

## Flood Request Simulation Results

To evaluate the API's performance, a load test was performed using Locust. The test simulated multiple concurrent users sending prediction requests to the `/predict` endpoint on the Render Free Plan (1 instance).

| **Test Scenario** | **Concurrent Users** | **Requests/Sec (RPS)** | **Average Response Time (ms)** | **Notes** |
|---|---|---|---|---|
| **Test 1: Low Load** | **10** | **4.8** | **470.87** | The API successfully handles a low load, but begins to show a high failure rate. |
| **Test 2: High Load** | **40** | **19.9** | **484.02** | The API's throughput (RPS) increases significantly, but failures remain high, indicating a performance bottleneck. |

**Observations:**

  * The test results show a direct correlation between increased user load and a significant increase in the API's throughput (RPS), from **4.8 to 19.9**. This demonstrates the system's ability to handle more requests per second.

  * However, the high failure rate (over 50% in both tests) and a slight increase in latency highlight a key limitation of the single-instance free-tier deployment under heavy load. The single container is struggling with a high volume of concurrent, computationally expensive prediction requests.

  * In a production MLOps pipeline, these metrics would signal a need for **horizontal scaling**. The solution would be to deploy multiple Docker containers to distribute the load, which would drastically reduce the failure rate and improve overall latency and throughput.
