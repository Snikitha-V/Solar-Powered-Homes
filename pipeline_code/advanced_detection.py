# ============================================
# pipeline_code/advanced_detection.py
# Multi-Scale Detection + TTA + NMS Tuning
# ============================================

import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from config import config

class AdvancedDetector:
    """
    Advanced detection with:
    - Multi-scale detection (0.75x, 1.0x, 1.25x, 1.5x)
    - TTA (Test-Time Augmentation: flip, rotate)
    - Optimized NMS with dynamic thresholds
    """
    
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.scales = [0.85, 1.0, 1.15]  # Reduced to 3 scales (was 4) - tighter range
        self.tta_variants = ['original', 'h_flip', 'v_flip']  # Reduced to 3 augmentations (was 5)
        
    def detect_with_multiscale_tta(self, image, buffer_mask):
        """
        Multi-scale + TTA detection with ensemble confidence
        """
        all_detections = []
        h, w = image.shape[:2]
        
        # MULTI-SCALE DETECTION
        print("🔄 Running multi-scale detection (4 scales)...")
        for scale in self.scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            # Predict at this scale
            results = self.yolo_model.predict(
                scaled_image,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.NMS_IOU_THRESHOLD,
                imgsz=config.DETECTION_IMAGE_SIZE,
                verbose=False,
                augment=False
            )
            
            # Rescale detections back to original size
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    # Rescale coordinates
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    
                    all_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'scale': scale,
                        'tta_variant': 'original'
                    })
        
        # TTA DETECTION (Flips + Rotations)
        print("🔄 Running TTA (5 augmentations)...")
        tta_images = {
            'h_flip': cv2.flip(image, 1),  # Horizontal flip
            'v_flip': cv2.flip(image, 0),  # Vertical flip
            'rot90': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),  # 90° rotation
            'rot270': cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270° rotation
        }
        
        for variant_name, tta_image in tta_images.items():
            if variant_name in ['rot90', 'rot270']:
                tta_h, tta_w = tta_image.shape[:2]
            else:
                tta_h, tta_w = h, w
            
            results = self.yolo_model.predict(
                tta_image,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.NMS_IOU_THRESHOLD,
                imgsz=config.DETECTION_IMAGE_SIZE,
                verbose=False,
                augment=False
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    
                    # Reverse TTA transformation to original space
                    if variant_name == 'h_flip':
                        x1, x2 = w - x2, w - x1
                    elif variant_name == 'v_flip':
                        y1, y2 = h - y2, h - y1
                    elif variant_name == 'rot90':
                        x1, y1, x2, y2 = y1, w - x2, y2, w - x1
                    elif variant_name == 'rot270':
                        x1, y1, x2, y2 = h - y2, x1, h - y1, x2
                    
                    all_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'scale': 1.0,
                        'tta_variant': variant_name
                    })
        
        print(f"✓ Total detections before ensemble: {len(all_detections)}")
        
        # ENSEMBLE + DYNAMIC NMS
        print("🎯 Applying ensemble NMS with optimized thresholds...")
        final_detections = self.ensemble_nms(all_detections)
        
        return final_detections
    
    def ensemble_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Ensemble NMS: 
        - Cluster similar detections across scales/TTA
        - Average confidence
        - Use stricter IoU threshold (more conservative)
        """
        if not detections:
            return []
        
        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_dets = []
        used = set()
        
        # STRICTER NMS: Only cluster detections with high IoU overlap
        # This prevents false clustering of nearby-but-separate panels
        strict_iou_threshold = 0.65  # Increased from 0.40 - much stricter
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            area_i = (x2 - x1) * (y2 - y1)
            
            # Only cluster with VERY similar detections (high IoU)
            cluster = [det]
            used.add(i)
            
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                x1_j, y1_j, x2_j, y2_j = detections[j]['bbox']
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                
                # IoU calculation
                inter_x1 = max(x1, x1_j)
                inter_y1 = max(y1, y1_j)
                inter_x2 = min(x2, x2_j)
                inter_y2 = min(y2, y2_j)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # STRICT clustering: Only merge if IoU is very high (0.65+)
                    # This prevents merging different panels
                    if iou > strict_iou_threshold:
                        cluster.append(detections[j])
                        used.add(j)
            
            # Average detections in cluster
            avg_bbox = np.mean([d['bbox'] for d in cluster], axis=0).tolist()
            avg_conf = np.mean([d['confidence'] for d in cluster])
            
            # CONSERVATIVE boost: Only boost with multiple high-confidence votes
            # Require at least 3 detections from different sources to apply boost
            if len(cluster) >= 3:
                # Smaller boost: +0.02 per extra detection (max +0.08)
                ensemble_boost = min(0.08, (len(cluster) - 1) * 0.02)
                final_conf = avg_conf + ensemble_boost
            else:
                # For 1-2 detections, use original confidence (no boost)
                final_conf = avg_conf
            
            final_conf = min(final_conf, 0.99)
            
            final_dets.append({
                'bbox': avg_bbox,
                'confidence': float(final_conf),
                'ensemble_count': len(cluster),
                'tta_variants': list(set([d['tta_variant'] for d in cluster])),
                'scales': list(set([d['scale'] for d in cluster]))
            })
        
        print(f"✓ Final detections after ensemble NMS: {len(final_dets)}")
        return final_dets


def find_optimal_nms_threshold(model, val_images: List[np.ndarray], 
                               ground_truth: List[Dict]) -> Tuple[float, float]:
    """
    Grid search to find optimal NMS thresholds
    """
    print("\n🔍 Tuning NMS thresholds on validation set...")
    
    best_score = 0
    best_conf_thresh = 0.25
    best_nms_thresh = 0.40
    
    conf_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    nms_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    
    for conf_th in conf_thresholds:
        for nms_th in nms_thresholds:
            tp = fp = fn = 0
            
            for image, gt in zip(val_images, ground_truth):
                results = model.predict(
                    image,
                    conf=conf_th,
                    iou=nms_th,
                    imgsz=config.DETECTION_IMAGE_SIZE,
                    verbose=False
                )
                
                if len(results) > 0 and results[0].boxes is not None:
                    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    # Simple IoU-based TP/FP matching
                    for pred_box in pred_boxes:
                        matched = False
                        for gt_box in gt['boxes']:
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > 0.5:
                                tp += 1
                                matched = True
                                break
                        if not matched:
                            fp += 1
                    
                    fn += max(0, len(gt['boxes']) - len(pred_boxes))
            
            # F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_score:
                best_score = f1
                best_conf_thresh = conf_th
                best_nms_thresh = nms_th
                print(f"  ✓ New best: F1={f1:.3f} (conf={conf_th}, nms={nms_th})")
    
    print(f"\n🏆 Optimal thresholds: conf={best_conf_thresh}, nms={best_nms_thresh} (F1={best_score:.3f})")
    return best_conf_thresh, best_nms_thresh


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0
