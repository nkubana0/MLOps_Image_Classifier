# MLOps_Image_Classifier/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for Pillow etc. (optional but good practice)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure the models and data directories exist inside the container
RUN mkdir -p /app/models \
    && mkdir -p /app/data/train \
    && mkdir -p /app/data/test \
    && mkdir -p /app/data/validation \
    && mkdir -p /app/data/incoming_for_retraining \
    && mkdir -p /app/visualizations

# --- Pre-run data preprocessing and initial model training within the Docker image build ---
RUN python /app/src/preprocessing.py \
    && python /app/src/model.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]