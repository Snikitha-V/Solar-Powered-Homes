# ============================================
# pipeline_code/generate_artifacts.py
# ============================================

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from config import config

def create_audit_overlay(image: np.ndarray, detections: List[Dict], 
                        buffer_masks: Dict, metadata: Dict,
                        output_path: Path) -> None:
    """
    Create audit-friendly visualization with:
    - Original image
    - Buffer zone circles
    - Detection bounding boxes/masks
    - Confidence scores
    - Metadata overlay
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Draw buffer zones
    meters_per_pixel = 300 / w
    primary_radius_px = int(config.PRIMARY_BUFFER_RADIUS_M / meters_per_pixel)
    secondary_radius_px = int(config.SECONDARY_BUFFER_RADIUS_M / meters_per_pixel)
    
    cv2.circle(overlay, center, primary_radius_px, (0, 255, 255), 2)  # Yellow
    cv2.circle(overlay, center, secondary_radius_px, (255, 165, 0), 2)  # Orange
    
    # Draw detections
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
        conf = detection['confidence']
        
        # Color based on buffer
        if detection['in_primary_buffer']:
            color = (0, 255, 0)  # Green
            buffer_label = "Primary"
        elif detection['in_secondary_buffer']:
            color = (0, 255, 255)  # Yellow
            buffer_label = "Secondary"
        else:
            color = (0, 0, 255)  # Red
            buffer_label = "Outside"
        
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Draw mask if available
        if detection.get('mask') is not None:
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[detection['mask'] > 0] = color
            overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        # Add label
        label = f"{buffer_label}: {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add metadata panel
    info_panel = np.zeros((150, w, 3), dtype=np.uint8)
    cv2.putText(info_panel, f"Source: {metadata.get('source', 'Unknown')}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(info_panel, f"Resolution: {metadata.get('resolution_m', 0):.2f}m/px", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(info_panel, f"Detections: {len(detections)}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(info_panel, f"Buffer: 1200/2400 sq.ft circles", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    # Combine
    final = np.vstack([overlay, info_panel])
    
    # Save
    cv2.imwrite(str(output_path), final)