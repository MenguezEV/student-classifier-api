#!/usr/bin/env bash

# Make sure the script will fail if any command fails
set -o errexit

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# --- CRITICAL STEP: Copy/Verify Model Files ---
# This step ensures the binary files are copied to the correct working directory 
# where 'app.py' expects them to be loaded from.

echo "Verifying model files exist in the build environment..."

# The files should be in the current directory if pushed to GitHub
MODEL_FILE="random_forest_model.joblib"
SCALER_FILE="standard_scaler.joblib"

# Check if the files exist. If they don't, the deployment must fail.
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in the repository."
    exit 1
fi
if [ ! -f "$SCALER_FILE" ]; then
    echo "Error: $SCALER_FILE not found in the repository."
    exit 1
fi

echo "Model files verified. Proceeding to deployment."


