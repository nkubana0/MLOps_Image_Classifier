# Use a recent and secure Python base image.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, apply security patches, and clean up.
# We add curl and wget for downloading files.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    wget \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install the Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and models
RUN mkdir -p /app/data/test \
    && mkdir -p /app/models

# --- DOWNLOAD LARGE FILES DURING BUILD ---
# This is the new section to handle the Git LFS issue.
# You MUST replace the URLs below with the public, direct download links
# for your x_test.npy and cifar10_cnn_model.h5 files.
RUN curl -o /app/data/test/x_test.npy "YOUR_PUBLIC_URL_FOR_X_TEST.NPY"
RUN curl -o /app/models/cifar10_cnn_model.h5 "YOUR_PUBLIC_URL_FOR_MODEL.H5"

# Copy the rest of the application code into the container.
# This copies everything EXCEPT the large files we just downloaded.
COPY . .

# Expose the port that your FastAPI application will listen on.
EXPOSE 8000

# Define the command to run your application using Uvicorn.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
