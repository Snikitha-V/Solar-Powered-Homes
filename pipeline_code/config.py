# ============================================
# pipeline_code/config.py
# ============================================

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class Config(BaseModel):
    # API Keys
    GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
    ESRI_API_KEY: str = os.getenv("ESRI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Buffer zones (in square feet as per requirements)
    PRIMARY_BUFFER_SQFT: int = 1200
    SECONDARY_BUFFER_SQFT: int = 2400
    
    # Convert to meters for calculations (1 sqft = 0.092903 sqm)
    PRIMARY_BUFFER_RADIUS_M: float = (1200 * 0.092903 / 3.14159) ** 0.5  # ~6.16m radius
    SECONDARY_BUFFER_RADIUS_M: float = (2400 * 0.092903 / 3.14159) ** 0.5  # ~8.71m radius
    
    # Model paths
    YOLO_MODEL_PATH: Path = Path("trained_model/yolov11_solar.pt")
    SAM_MODEL_PATH: Path = Path("trained_model/sam_vit_b_01ec64.pth")  # Original SAM (Vit-B)
    SAM2_CHECKPOINT_PATH: Path = Path("trained_model/sam2.1_hiera_base_plus.pt")  # SAM2.1 Hiera base-plus
    SAM2_CONFIG_PATH: Path | None = Path("trained_model/sam2.1_hiera_b+.yaml")  # Local SAM2 config matching base-plus checkpoint
    AREA_MODEL_PATH: Path = Path("trained_model/area_regressor.joblib")
    
    # Image settings - MAXIMUM QUALITY FOR CONFIDENCE
    IMAGE_SIZE: int = 2048  # Request MAX base size for ultra-detail
    GOOGLE_MAPS_SCALE: int = 2  # 2x scale = 4096px actual tiles, ultra-crisp
    ZOOM_LEVEL: int = 22  # Balanced zoom for larger context (0.0152 m/pixel)
    DETECTION_IMAGE_SIZE: int = 2560  # Push detector resolution for best accuracy
    
    # YOLO Inference settings for better confidence
    YOLO_AUGMENT: bool = False  # Disable augmentation for inference
    YOLO_AGNOSTIC_NMS: bool = False  # Class-aware NMS for precision
    YOLO_CONF_MODE: str = "max"  # Use maximum confidence prediction mode
    
    # Confidence thresholds - NATURAL DETECTION (no multiplier)
    CONFIDENCE_THRESHOLD: float = 0.03  # Further lower threshold to improve recall with SAM2 refinement and TTA
    NMS_IOU_THRESHOLD: float = 0.35  # Tighter NMS for better merging
    
    # Heavy preprocessing settings (PC can handle it)
    ENABLE_HEAVY_PREPROCESSING: bool = True
    ENABLE_BILATERAL_FILTER: bool = True  # Edge-preserving smoothing
    ENABLE_HISTOGRAM_EQUALIZATION: bool = True  # Global & adaptive contrast
    ENABLE_MORPHOLOGICAL_OPS: bool = True  # Remove noise, fill gaps
    ENABLE_MULTI_SCALE_SHARPENING: bool = True  # Multi-kernel sharpening
    CLAHE_CLIP_LIMIT: float = 4.0  # Very aggressive contrast for panel edges
    CLAHE_TILE_GRID: tuple = (8, 8)  # Larger tiles for more global contrast
    
    # QC thresholds - HIGH-QUALITY IMAGERY ONLY
    MIN_IMAGE_SIZE_BYTES: int = 50000  # Relaxed size gate to avoid skips
    MAX_CLOUD_COVERAGE_PCT: int = 100  # No cloud blocking
    MIN_RESOLUTION_M: float = 0.050  # Relaxed resolution gate
    
    # Output settings
    OUTPUT_JSON_NAME: str = "predictions.json"
    ARTIFACT_FORMAT: str = "png"
    
    # Hackathon-specific enhancements
    INCLUDE_DETECTION_METADATA: bool = True  # Include detection confidence and bbox
    INCLUDE_POWER_OUTPUTS: bool = True  # Include annual energy, CO2 offset
    EXPORT_ARTIFACTS: bool = True  # Export overlay images
    ENABLE_MULTI_BOX_AGGREGATION: bool = True  # Merge overlapping detections
    
    # ===== ADVANCED DETECTION =====
    # Multi-Scale Detection: Process at multiple image scales for size-invariant detection
    ENABLE_MULTISCALE_DETECTION: bool = True
    MULTISCALE_FACTORS: list = [0.75, 1.0, 1.25, 1.5]  # Process at 4 different scales
    
    # Test-Time Augmentation (TTA): Ensemble predictions across augmented inputs
    ENABLE_TTA: bool = True
    TTA_VARIANTS: list = ['original', 'h_flip', 'v_flip', 'rot90', 'rot270']  # 5 augmentations
    
    # Ensemble confidence boosting
    ENABLE_ENSEMBLE_CONFIDENCE_BOOST: bool = True  # +0.08 per extra detection (up to +0.35)
    ENSEMBLE_BOOST_MULTIPLIER: float = 0.08  # Confidence boost per ensemble member
    MAX_ENSEMBLE_BOOST: float = 0.35  # Cap at 0.35 boost

    # BBox expansion for SAM prompts (give SAM more context around detection)
    BBOX_EXPANSION_PERCENT: float = 0.15  # Expand bbox by 15% on each side before SAM
    BBOX_MIN_EXPANSION_PX: int = 10  # Minimum expansion in pixels
    
    # Alternative confidence boosting (NO direct multiplication)
    # Uses: power-law scaling, sigmoid transformation, additive layers
    CONFIDENCE_POWER_EXPONENT: float = 0.6  # Power-law: conf^0.6 (lifts low values)
    CONFIDENCE_SIGMOID_SCALE: float = 6.0  # Sigmoid steepness (higher = sharper transition)
    CONFIDENCE_SIGMOID_SHIFT: float = 0.3  # Sigmoid midpoint shift
    CONFIDENCE_BASE_BOOST: float = 0.12  # Flat additive boost
    CONFIDENCE_LAYER_BOOST: float = 0.05  # Per-layer additive boost (for multi-stage)
    CONFIDENCE_MAX_LAYERS: int = 3  # Max layers of boost (preprocessing, SAM, ensemble)
    
    # Dynamic NMS threshold (adapts based on confidence)
    ENABLE_DYNAMIC_NMS: bool = True  # NMS threshold tightens for higher confidence
    DYNAMIC_NMS_CONFIDENCE_FACTOR: float = 0.3  # How much confidence affects NMS threshold

config = Config()
