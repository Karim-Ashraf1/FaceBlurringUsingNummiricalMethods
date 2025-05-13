#!/bin/bash
# Script to help with Vercel deployment

# Ensure pip is updated to a compatible version
pip install --upgrade pip==21.3.1

# Install dependencies
pip install -r requirements-vercel.txt 