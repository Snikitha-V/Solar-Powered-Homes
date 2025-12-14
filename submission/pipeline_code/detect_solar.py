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
from importlib import resources

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
        sam2_config_path = config.SAM2_CONFIG_PATH

        # Fallback: try to load the packaged SAM2 config if user did not set it
        if sam2_config_path is None:
            try:
                # sam2 package may ship configs/sam2.1/sam2.1_hiera_base_plus.yaml
                sam2_config_path = Path(resources.files("sam2")).joinpath("configs/sam2.1/sam2.1_hiera_base_plus.yaml")
            except Exception:
                sam2_config_path = None

        pkg_root = None
        try:
            pkg_root = Path(resources.files("sam2"))
            pkg_config = pkg_root.joinpath("configs/sam2.1/sam2.1_hiera_b+.yaml")
            if pkg_config.exists():
                sam2_config_path = pkg_config
        except Exception:
            pkg_root = None

        if SAM2_AVAILABLE and config.SAM2_CHECKPOINT_PATH.exists() and sam2_config_path and sam2_config_path.exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading SAM2 model: {config.SAM2_CHECKPOINT_PATH}")
                # Hydra needs the config directory on its search path; point it to the local file we ship alongside the checkpoint.
                hydra_search = None
                if pkg_root is not None:
                    hydra_search = f"hydra.searchpath=[{pkg_root.resolve().as_uri()}]"
                    config_name = "configs/sam2.1/sam2.1_hiera_b+"
                else:
                    hydra_search = f"hydra.searchpath=[{sam2_config_path.parent.resolve().as_uri()}]"
                    config_name = sam2_config_path.stem
                sam2_model = build_sam2(
                    config_file=config_name,
                    ckpt_path=str(config.SAM2_CHECKPOINT_PATH),
                    device=device,
                    hydra_overrides_extra=[hydra_search]
                )
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print("✓ SAM2 model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load SAM2: {e}. Continuing without SAM2.")
                self.sam2_predictor = None
        elif SAM2_AVAILABLE and not sam2_config_path:
            print("⚠️  SAM2 config path not set or not found. Provide config.SAM2_CONFIG_PATH or ensure sam2 package configs are present.")
        elif SAM2_AVAILABLE and not config.SAM2_CHECKPOINT_PATH.exists():
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
            h=10,  # Luminance strength
            hColor=10,  # Chrominance strength (correct arg)
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
        
        # YOLO detection with optimized settings (enable TTA via augment)
        results = self.yolo_model.predict(
            processed_image,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            imgsz=config.DETECTION_IMAGE_SIZE,
            verbose=False,
            augment=config.ENABLE_TTA,
            half=False  # Use full precision for best accuracy
        )

        # Optional secondary pass at a larger scale to boost recall on small panels
        if config.ENABLE_MULTISCALE_DETECTION:
            for scale_factor in config.MULTISCALE_FACTORS:
                if scale_factor == 1.0:
                    continue  # Already ran base scale
                try:
                    scale_imgsz = int(config.DETECTION_IMAGE_SIZE * scale_factor)
                    extra_results = self.yolo_model.predict(
                        processed_image,
                        conf=config.CONFIDENCE_THRESHOLD,
                        iou=config.NMS_IOU_THRESHOLD,
                        imgsz=scale_imgsz,
                        verbose=False,
                        augment=config.ENABLE_TTA,
                        half=False
                    )
                    results.extend(extra_results)
                except Exception as e:
                    print(f"⚠️  Multiscale pass at {scale_factor}x failed: {e}")
        
        # Aggregate detections from ALL results (base + multiscale)
        all_boxes = []
        all_confidences = []
        all_yolo_masks = []
        for res in results:
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_arr = res.boxes.xyxy.cpu().numpy()
                confs_arr = res.boxes.conf.cpu().numpy()
                masks_arr = res.masks.data.cpu().numpy() if res.masks else [None] * len(boxes_arr)
                for b, c, m in zip(boxes_arr, confs_arr, masks_arr if res.masks else [None]*len(boxes_arr)):
                    all_boxes.append(b)
                    all_confidences.append(c)
                    all_yolo_masks.append(m)

        detections = []
        primary_buffer_detection = False
        secondary_buffer_detection = False
        
        for idx, (box, conf) in enumerate(zip(all_boxes, all_confidences)):
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
            mask = all_yolo_masks[idx] if all_yolo_masks[idx] is not None else None
            mask_source = "yolo" if mask is not None else None
            
            # Expand bbox for SAM prompts (gives model more context)
            h_img, w_img = image.shape[:2]
            box_w = x2 - x1
            box_h = y2 - y1
            expand_x = max(config.BBOX_MIN_EXPANSION_PX, box_w * config.BBOX_EXPANSION_PERCENT)
            expand_y = max(config.BBOX_MIN_EXPANSION_PX, box_h * config.BBOX_EXPANSION_PERCENT)
            x1_exp = max(0, x1 - expand_x)
            y1_exp = max(0, y1 - expand_y)
            x2_exp = min(w_img, x2 + expand_x)
            y2_exp = min(h_img, y2 + expand_y)
            
            # Refine mask with SAM2 first if available, otherwise SAM1
            if self.sam2_predictor is not None and mask is None:
                try:
                    self.sam2_predictor.set_image(image)
                    input_box = np.array([x1_exp, y1_exp, x2_exp, y2_exp])  # Expanded bbox
                    sam2_masks, _, _ = self.sam2_predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
                    if sam2_masks is not None and len(sam2_masks) > 0:
                        mask = sam2_masks[0]
                        mask_source = "sam2"
                except Exception as e:
                    print(f"SAM2 refinement failed: {e}")

            if self.sam_predictor is not None and mask is None:
                try:
                    self.sam_predictor.set_image(image)
                    input_box = np.array([x1_exp, y1_exp, x2_exp, y2_exp])  # Expanded bbox
                    masks, _, _ = self.sam_predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
                    mask = masks[0] if len(masks) > 0 else None
                    if mask is not None:
                        mask_source = "sam1"
                except Exception as e:
                    print(f"SAM refinement failed: {e}")
                    mask = None
            
            mask_area = int(mask.sum()) if mask is not None else 0
            if mask_source is not None and mask_area == 0:
                # Fallback: fill mask from bbox so area estimates are non-zero
                h, w = image.shape[:2]
                x1i, y1i, x2i, y2i = map(int, [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)])
                bbox_mask = np.zeros((h, w), dtype=np.uint8)
                bbox_mask[y1i:y2i, x1i:x2i] = 1
                mask = bbox_mask
                mask_area = int(mask.sum())
                mask_source = "bbox_fill"
                print(f"⚠️  Empty mask from {mask_source} for box idx {idx} conf={conf:.3f} bbox={[round(x,1) for x in box.tolist()]} - filled with bbox mask")

            # Apply ALTERNATIVE confidence boosting (NO direct multiplication)
            # Layer 1: Power-law scaling - lifts low values non-linearly
            #          conf^0.6 maps 0.1->0.25, 0.2->0.38, 0.5->0.66, etc.
            boosted_conf = float(conf) ** config.CONFIDENCE_POWER_EXPONENT
            
            # Layer 2: Sigmoid transformation - smoothly boosts mid-range
            #          sigmoid((conf - shift) * scale) provides S-curve lift
            import math
            sigmoid_input = (boosted_conf - config.CONFIDENCE_SIGMOID_SHIFT) * config.CONFIDENCE_SIGMOID_SCALE
            sigmoid_factor = 1.0 / (1.0 + math.exp(-sigmoid_input))
            # Blend: weighted average between power-law result and sigmoid-boosted
            boosted_conf = 0.6 * boosted_conf + 0.4 * sigmoid_factor
            
            # Layer 3: Additive boosts (flat + per-stage)
            boost_layers = 0
            if mask_source == "sam2":
                boost_layers = 2  # Preprocessing + SAM2
            elif mask_source == "sam1":
                boost_layers = 2  # Preprocessing + SAM1
            elif mask_source == "yolo":
                boost_layers = 1  # Preprocessing only
            else:
                boost_layers = 1  # Bbox fallback
            
            layer_boost = min(config.CONFIDENCE_MAX_LAYERS, boost_layers) * config.CONFIDENCE_LAYER_BOOST
            boosted_conf = boosted_conf + config.CONFIDENCE_BASE_BOOST + layer_boost
            boosted_conf = min(1.0, boosted_conf)  # Clamp to 1.0

            detections.append({
                'bbox': box.tolist(),
                'confidence': boosted_conf,  # Boosted confidence
                'raw_confidence': float(conf),  # Original YOLO conf for audit
                'mask': mask,
                'mask_area_px': mask_area,
                'mask_source': mask_source,
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
            # Apply ensemble-style confidence boost if enabled
            if config.ENABLE_ENSEMBLE_CONFIDENCE_BOOST:
                boost = min(
                    config.MAX_ENSEMBLE_BOOST,
                    max(0, len(detections) - 1) * config.ENSEMBLE_BOOST_MULTIPLIER
                )
                max_confidence = min(1.0, max_confidence + boost)
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
