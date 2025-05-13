#!/bin/bash

# Install only minimal dependencies with pip
pip install --no-cache-dir -r requirements.txt

# Create required directories
mkdir -p /tmp/uploads /tmp/results

# Clean up unnecessary files to reduce size
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.jpg" -delete
find . -name "*.jpeg" -delete
find . -name "*.png" -delete

# Remove heavy files we don't need in the minimal version
rm -f haarcascade_frontalface_alt.xml
rm -f project.py
rm -f api/index.py
rm -f api/index-lite.py

# Print deployment size for debugging
echo "Deployment size:"
du -sh . 