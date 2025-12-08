#!/usr/bin/env python
"""Debug script to diagnose why confidence scores are zero"""

import sys
import os
os.chdir('C:\\Users\\student\\Desktop\\Solar Project\\pipeline_code')
sys.path.insert(0, 'C:\\Users\\student\\Desktop\\Solar Project\\pipeline_code')

import numpy as np
import pandas as pd
from pathlib import Path
from config import config
from ultralytics import YOLO
import cv2
import requests
from typing import Tuple, Dict

# Don't load SAM for this debug script
import cv2

print("="*70)
print("SOLAR DETECTION DEBUGGING")
print("="*70)

# Test 1: Verify model loads
print("\n[TEST 1] Loading YOLO model...")
model_path = Path("trained_model/yolov11_solar.pt")
print(f"  Model exists: {model_path.exists()}")
if model_path.exists():
    print(f"  Model size: {model_path.stat().st_size / 1e6:.1f} MB")
    model = YOLO(str(model_path))
    print("  ✓ Model loaded successfully")
else:
    print("  ✗ Model NOT found!")
    sys.exit(1)

# Test 2: Load first test location
print("\n[TEST 2] Loading test data...")
df = pd.read_excel('test_solar_coordinates.xlsx')
first_row = df.iloc[0]
lat, lon = first_row['latitude'], first_row['longitude']
sample_id = first_row['sample_id']
print(f"  Sample {sample_id}: ({lat}, {lon})")

# Test 3: Fetch imagery
print("\n[TEST 3] Fetching satellite imagery...")
try:
    image, metadata = fetch_google_static_map(lat, lon, config.GOOGLE_MAPS_API_KEY)
    print(f"  ✓ Image fetched")
    print(f"    Shape: {image.shape}")
    print(f"    Mean pixel: {np.mean(image):.1f}")
    print(f"    Std dev: {np.std(image):.1f}")
    print(f"    Min/Max: {np.min(image)}/{np.max(image)}")
    print(f"    Source: {metadata['source']}")
    print(f"    Resolution: {metadata['resolution_m']:.2f} m/pixel")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  Using placeholder image instead...")
    image = np.ones((640, 640, 3), dtype=np.uint8) * 128

# Test 4: Image quality check
print("\n[TEST 4] Image quality assessment...")
quality_ok, issues = assess_image_quality(image, metadata)
print(f"  Quality OK: {quality_ok}")
if issues:
    print(f"  Issues: {issues}")

# Test 5: Direct YOLO prediction
print("\n[TEST 5] Running YOLO detection...")
results = model.predict(image, conf=0.15, verbose=False)
print(f"  Results returned: {len(results)} predictions")
if results[0].boxes is not None:
    boxes = results[0].boxes
    print(f"  Detections: {len(boxes)}")
    print(f"  Confidences: {boxes.conf.cpu().numpy()}")
    print(f"  Boxes: {boxes.xyxy.cpu().numpy()}")
else:
    print(f"  ✗ NO DETECTIONS FOUND")

# Test 6: Full detector
print("\n[TEST 6] Using SolarDetectorIntegrated...")
detector = SolarDetectorIntegrated()
buffer_masks = create_buffer_masks(image.shape[:2], (image.shape[0]//2, image.shape[1]//2))
detection_result = detector.detect_panels(image, buffer_masks)
print(f"  Has solar: {detection_result['has_solar']}")
print(f"  Confidence: {detection_result['confidence']}")
print(f"  Detections: {len(detection_result['detections'])}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

# Summary
print("\nSUMMARY:")
if detection_result['confidence'] == 0.0:
    print("✗ The model is NOT detecting solar panels in your imagery.")
    print("\nPossible causes:")
    print("  1. Image resolution too low (~0.75m/pixel) - model trained on higher res")
    print("  2. No actual solar panels in the test locations")
    print("  3. Image source using fallback (placeholder/OSM) instead of satellite")
    print("\nSolutions:")
    print("  A. Enable Google Maps Static API for native satellite imagery")
    print("  B. Use different test locations with known solar installations")
    print("  C. Re-train model on lower-resolution imagery")
else:
    print(f"✓ Detection working! Found {len(detection_result['detections'])} panels")
