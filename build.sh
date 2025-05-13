#!/bin/bash

# Install only production dependencies with pip
pip install --no-cache-dir -r requirements.txt

# Create required directories
mkdir -p /tmp/uploads /tmp/results

# Ensure the face detection model is in place
if [ ! -f "haarcascade_frontalface_alt.xml" ]; then
    echo "Downloading face detection model"
    curl -s -o haarcascade_frontalface_alt.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
fi

# Clean up unnecessary files to reduce size
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete 