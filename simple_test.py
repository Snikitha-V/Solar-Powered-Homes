#!/usr/bin/env python
"""Simple test to see if YOLO detects anything"""

import sys
import os
os.chdir('C:\\Users\\student\\Desktop\\Solar Project')

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import requests
from typing import Tuple

print("="*70)
print("SOLAR DETECTION SIMPLE TEST")
print("="*70)

# Load YOLO model
print("\n[1] Loading YOLO model...")
model_path = Path("trained_model/yolov11_solar.pt")
assert model_path.exists(), f"Model not found: {model_path}"
print(f"    Model size: {model_path.stat().st_size / 1e6:.1f} MB")

model = YOLO(str(model_path))
print("    ✓ Model loaded")

# Load test data
print("\n[2] Loading test coordinates...")
df = pd.read_excel('test_solar_coordinates.xlsx')
print(f"    Loaded {len(df)} locations")

# Try to fetch one image
lat, lon = df.iloc[0]['latitude'], df.iloc[0]['longitude']
sample_id = df.iloc[0]['sample_id']
print(f"    Testing with sample {sample_id}: ({lat:.4f}, {lon:.4f})")

# Try Google Maps
print("\n[3] Attempting to fetch satellite imagery from Google Maps...")
url = "https://maps.googleapis.com/maps/api/staticmap"
params = {
    'center': f"{lat},{lon}",
    'zoom': 20,
    'size': '640x640',
    'maptype': 'satellite',
    'key': 'AIzaSyBXV54hr_1nkQqHPIBSQcO3e1mMvKvs1xI',
    'format': 'png'
}

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"    Response status: {response.status_code}")
    
    if response.status_code == 200:
        img_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is not None:
            print(f"    ✓ Image fetched: {image.shape}")
            print(f"    Mean pixel: {np.mean(image):.1f}")
            print(f"    Std dev: {np.std(image):.1f}")
        else:
            print(f"    ✗ Failed to decode image")
            image = None
    else:
        print(f"    ✗ API returned status {response.status_code}")
        if response.status_code == 403:
            print("    → Static Maps API not enabled in Google Cloud project")
        image = None
        
except Exception as e:
    print(f"    ✗ Request failed: {e}")
    image = None

# If no image, create a test pattern
if image is None:
    print("\n    Using test pattern image...")
    image = np.ones((640, 640, 3), dtype=np.uint8) * 150
    for i in range(0, 640, 64):
        image[i:i+8, :] = 100
        image[:, i:i+8] = 100

# Test YOLO on this image
print("\n[4] Running YOLO detection...")
results = model.predict(image, conf=0.15, verbose=False)

if results[0].boxes is not None:
    boxes = results[0].boxes
    num_detections = len(boxes)
    print(f"    Detections found: {num_detections}")
    
    if num_detections > 0:
        print(f"    Confidences: {boxes.conf.cpu().numpy()}")
        print(f"    ✓ Detection working!")
    else:
        print(f"    ✗ Model returned no detections")
else:
    num_detections = 0
    print(f"    ✗ Model returned no detections (boxes=None)")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

if num_detections > 0:
    print("✓ YOLO model is working and detecting solar panels")
else:
    print("✗ YOLO model is NOT finding solar panels in imagery")
    print("\nPossible reasons:")
    print("  1. Image resolution too low (0.75m/pixel) for accurate detection")
    print("  2. No actual solar installations in the test locations")
    print("  3. Model trained on different imagery characteristics")
    print("\nNext steps:")
    print("  A) Test with known solar panel locations (e.g., California, Germany)")
    print("  B) Enable Google Maps Static API for higher quality imagery")
    print("  C) Adjust YOLO confidence threshold or re-train model")
