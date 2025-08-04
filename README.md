# MLOps CIFAR-10 Image Classifier

Hi, and welcome to my MLOps project! This project demonstrates a full-stack MLOps pipeline for an image classification model. The application provides a user interface for making real-time predictions and a backend system that can be triggered for model retraining.

### Video Demonstration

I've created a video to walk you through the full functionality of this application. It shows a live demo of the prediction service and the background retraining process.

Watch the Demo Video: [https://drive.google.com/file/d/1qqM-1UAKC-pw98FWUBw4xt6HuP1m_YJ_/view](https://drive.google.com/file/d/1qqM-1UAKC-pw98FWUBw4xt6HuP1m_YJ_/view)

### Core Features

* Real-time Prediction Service: My FastAPI backend exposes a /predict endpoint that receives an image and returns the predicted class (from the CIFAR-10 dataset) with its confidence scores.
* Asynchronous Model Retraining: The /retrain/trigger endpoint kicks off a model retraining job in the background. This process is non-blocking, meaning the API remains fully available to serve predictions while the new model is being trained.
* Interactive UI: I built a Streamlit front-end that allows for easy image uploads, visualizes prediction results, and provides a dashboard for a high-level overview of the system's health.

### Architecture and Technology Stack

This project is built using a modern, scalable architecture:

* Front-end: Streamlit for the interactive web application.
* Back-end: FastAPI for building a high-performance and robust API.
* Machine Learning: TensorFlow with Keras for the image classification model.
* Data Processing: Python scripts to handle the downloading, preprocessing, and splitting of the CIFAR-10 dataset.
* Containerization: I have a Dockerfile ready for containerizing the application.
* Deployment Manifest: I've prepared a render.yaml file to define the services for seamless deployment.

### Getting Started (Local Setup)

To run this project on your local machine, follow these steps.

#### 1. Clone the Repository

git clone [https://github.com/nkubana0/MLOps_Image_Classifier.git](https://github.com/nkubana0/MLOps_Image_Classifier.git)
cd MLOps_Image_Classifier

#### 2. Set up the Virtual Environment

I highly recommend using a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

#### 3. Install Dependencies

Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

#### 4. Run the Preprocessing and Model Training Scripts

Before you can run the application, you need to prepare the data and train the initial model.

# This will download the CIFAR-10 dataset and save it to the 'data' directory
python src/preprocessing.py

# This will train the model and save the model file to the 'models' directory
python src/model.py

#### 5. Run the FastAPI Backend

Open a new terminal window (or a new tab) and start the backend API.

# Make sure your virtual environment is active in this new terminal
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000

#### 6. Run the Streamlit Front-end

In your original terminal window, start the Streamlit application.

streamlit run app.py

Your web browser should automatically open to the Streamlit UI. If not, navigate to http://localhost:8501.

