#!/bin/bash
set -e

echo "Installing system dependencies for MediaPipe + OpenCV..."
apt-get update -qq
apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn boto3

echo "Build complete."
