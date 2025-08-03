#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Change to the src directory to run the preprocessing script
# This script downloads and prepares the dataset, which is necessary for the model to be built.
echo "Running data preprocessing script to ensure data is available..."
python src/preprocessing.py

# Start the FastAPI application with Uvicorn, binding to the port provided by Render
# The --reload flag is removed as it's not needed in production.
echo "Starting the FastAPI server with Uvicorn..."
uvicorn src.api:app --host 0.0.0.0 --port $PORT
