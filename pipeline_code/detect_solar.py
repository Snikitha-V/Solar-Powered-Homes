# ============================================
# pipeline_code/detect_solar.py
# ============================================

from ultralytics import YOLO
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple
from config import config
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  Segment Anything (SAM) not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️  SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

class SolarDetectorIntegrated:
    """
    Detector that uses YOLO for detection + SAM for precise segmentation
    """
    def __init__(self):
        from ultralytics import YOLO
        
        # Load YOLO model
        model_path = Path(config.YOLO_MODEL_PATH)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                f"Please run training first: python train_yolo_optimized.py"
            )
        
        print(f"Loading YOLO model: {model_path}")
        self.yolo_model = YOLO(str(model_path))
        print("✓ YOLO model loaded successfully")
        
        # Load SAM2 model if available and configured
        self.sam2_predictor = None
        if SAM2_AVAILABLE and config.SAM2_CHECKPOINT_PATH.exists() and config.SAM2_CONFIG_PATH:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading SAM2 model: {config.SAM2_CHECKPOINT_PATH}")
                sam2_model = build_sam2(
                    config_file=str(config.SAM2_CONFIG_PATH),
                    ckpt_path=str(config.SAM2_CHECKPOINT_PATH),
                    device=device
                )
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print("✓ SAM2 model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load SAM2: {e}. Continuing without SAM2.")
                self.sam2_predictor = None
        elif SAM2_AVAILABLE and not config.SAM2_CONFIG_PATH:
            print("⚠️  SAM2 config path not set. Set config.SAM2_CONFIG_PATH to the YAML for your checkpoint.")
        elif SAM2_AVAILABLE:
            print(f"⚠️  SAM2 checkpoint not found at {config.SAM2_CHECKPOINT_PATH}.")
        
        # Load original SAM (SAM1) if available
        self.sam_predictor = None
        if SAM_AVAILABLE:
            sam_path = config.SAM_MODEL_PATH
            if sam_path.exists():
                try:
                    print(f"Loading SAM model: {sam_path}")
                    sam_model = sam_model_registry["vit_b"](checkpoint=str(sam_path))
                    self.sam_predictor = SamPredictor(sam_model)
                    print("✓ SAM model loaded successfully")
                except Exception as e:
                    print(f"⚠️  Failed to load SAM: {e}. Continuing with YOLO-only detection.")
                    self.sam_predictor = None
            else:
                print(f"⚠️  SAM model not found at {sam_path}. Using YOLO detections with bounding boxes only.")
    
    def detect_panels(self, image, buffer_mask):
        """
        Detect solar panels using YOLO + optional SAM refinement
        Heavy preprocessing for natural confidence boost without multiplication
        """
        # HEAVY PRE-PROCESSING: Multi-stage image enhancement
        import cv2
        processed_image = image.copy()
        
        # STAGE 1: Denoise for clean signal
        processed_image = cv2.fastNlMeansDenoisingColored(
            processed_image, 
            None, 
            h=10,  # Filter strength
            hForColorComponents=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # STAGE 2: Bilateral filtering (preserve edges while smoothing)
        processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
        
        # STAGE 3: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
        
        # STAGE 4: Aggressive sharpening for panel edge definition
        kernel_sharpen = np.array([[-2, -1,  0],
                                   [-1,  1,  1],
                                   [ 0,  1,  2]])
        processed_image = cv2.filter2D(processed_image, -1, kernel_sharpen)
        
        # STAGE 5: CLAHE on all channels for global contrast
        for i in range(3):  # Process BGR channels
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            l = clahe.apply(l)
            processed_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        
        # STAGE 6: Histogram equalization for uniform brightness
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        processed_image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        
        # STAGE 7: Unsharp masking (fine detail enhancement)
        blurred = cv2.GaussianBlur(processed_image, (0, 0), 2.0)
        processed_image = cv2.addWeighted(processed_image, 1.5, blurred, -0.5, 0)
        
        # STAGE 8: Increase saturation to make panels pop
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # Boost saturation 30%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        processed_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # YOLO detection with optimized settings
        results = self.yolo_model.predict(
            processed_image,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            imgsz=config.DETECTION_IMAGE_SIZE,
            verbose=False,
            augment=False,
            half=False  # Use full precision for best accuracy
        )
        
        detections = []
        primary_buffer_detection = False
        secondary_buffer_detection = False
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Get YOLO masks if available
            yolo_masks = results[0].masks.data.cpu().numpy() if results[0].masks else None
            
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                # Use natural confidence from YOLO (no artificial multiplier)
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Check if detection is within buffer zones
                in_primary = buffer_mask['primary'][center_y, center_x] > 0
                in_secondary = buffer_mask['secondary'][center_y, center_x] > 0
                
                if in_primary:
                    primary_buffer_detection = True
                elif in_secondary:
                    secondary_buffer_detection = True
                
                # Use YOLO mask or refine with SAM if available
                mask = yolo_masks[idx] if yolo_masks is not None else None
                
                # Refine mask with SAM2 first if available, otherwise SAM1
                if self.sam2_predictor is not None and mask is None:
                    try:
                        self.sam2_predictor.set_image(image)
                        input_box = np.array([x1, y1, x2, y2])
                        sam2_masks, _, _ = self.sam2_predictor.predict(
                            box=input_box,
                            multimask_output=False
                        )
                        if sam2_masks is not None and len(sam2_masks) > 0:
                            mask = sam2_masks[0]
                    except Exception as e:
                        print(f"SAM2 refinement failed: {e}")

                if self.sam_predictor is not None and mask is None:
                    try:
                        self.sam_predictor.set_image(image)
                        input_box = np.array([x1, y1, x2, y2])
                        masks, _, _ = self.sam_predictor.predict(
                            box=input_box,
                            multimask_output=False
                        )
                        mask = masks[0] if len(masks) > 0 else None
                    except Exception as e:
                        print(f"SAM refinement failed: {e}")
                        mask = None
                
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),  # Natural confidence from YOLO
                    'mask': mask,
                    'in_primary_buffer': in_primary,
                    'in_secondary_buffer': in_secondary,
                    'center': (center_x, center_y)
                })
        
        # Determine buffer radius used
        if primary_buffer_detection:
            buffer_radius_sqft = config.PRIMARY_BUFFER_SQFT
            has_solar = True
        elif secondary_buffer_detection:
            buffer_radius_sqft = config.SECONDARY_BUFFER_SQFT
            has_solar = True
        elif detections:
            # If we have detections but not in buffer zones, still mark as detected
            buffer_radius_sqft = config.SECONDARY_BUFFER_SQFT
            has_solar = True
        else:
            buffer_radius_sqft = config.SECONDARY_BUFFER_SQFT
            has_solar = False
        
        # Calculate overall confidence
        if detections:
            max_confidence = max([d['confidence'] for d in detections])
        else:
            max_confidence = 0.0
        
        return {
            'has_solar': has_solar,
            'confidence': max_confidence,
            'detections': detections,
            'buffer_radius_sqft': buffer_radius_sqft,
            'primary_buffer': primary_buffer_detection
        }

def create_buffer_masks(image_shape: Tuple[int, int], center: Tuple[int, int]) -> Dict:
    """Create circular buffer zone masks on the image"""
    h, w = image_shape
    y_center, x_center = center
    
    Y, X = np.ogrid[:h, :w]
    
    # Calculate pixel radius from meter radius
    # Assuming image covers ~300m at zoom 20
    meters_per_pixel = 300 / w
    
    primary_radius_px = config.PRIMARY_BUFFER_RADIUS_M / meters_per_pixel
    secondary_radius_px = config.SECONDARY_BUFFER_RADIUS_M / meters_per_pixel
    
    primary_mask = ((X - x_center)**2 + (Y - y_center)**2 <= primary_radius_px**2).astype(np.uint8) * 255
    secondary_mask = ((X - x_center)**2 + (Y - y_center)**2 <= secondary_radius_px**2).astype(np.uint8) * 255
    
    return {
        'primary': primary_mask,
        'secondary': secondary_mask
    }
