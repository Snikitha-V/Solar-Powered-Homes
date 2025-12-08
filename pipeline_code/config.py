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
    SAM2_CONFIG_PATH: Path | None = None  # Set to YAML config path for SAM2 (e.g., trained_model/sam2.1_hiera_base_plus.yaml)
    AREA_MODEL_PATH: Path = Path("trained_model/area_regressor.joblib")
    
    # Image settings - MAXIMUM QUALITY FOR CONFIDENCE
    IMAGE_SIZE: int = 1280  # Request MAX base size for ultra-detail
    GOOGLE_MAPS_SCALE: int = 2  # 2x scale = 2560px actual tiles, ultra-crisp
    ZOOM_LEVEL: int = 24  # MAXIMUM zoom: ~0.0048 m/pixel with scale=2 (0.48cm!)
    DETECTION_IMAGE_SIZE: int = 1920  # Maximum inference resolution (no artificial limits)
    
    # YOLO Inference settings for better confidence
    YOLO_AUGMENT: bool = False  # Disable augmentation for inference
    YOLO_AGNOSTIC_NMS: bool = False  # Class-aware NMS for precision
    YOLO_CONF_MODE: str = "max"  # Use maximum confidence prediction mode
    
    # Confidence thresholds - NATURAL DETECTION (no multiplier)
    CONFIDENCE_THRESHOLD: float = 0.22  # Slightly lower to catch genuine panels
    NMS_IOU_THRESHOLD: float = 0.40  # Balanced NMS
    
    # Heavy preprocessing settings (PC can handle it)
    ENABLE_HEAVY_PREPROCESSING: bool = True
    ENABLE_BILATERAL_FILTER: bool = True  # Edge-preserving smoothing
    ENABLE_HISTOGRAM_EQUALIZATION: bool = True  # Global & adaptive contrast
    ENABLE_MORPHOLOGICAL_OPS: bool = True  # Remove noise, fill gaps
    ENABLE_MULTI_SCALE_SHARPENING: bool = True  # Multi-kernel sharpening
    CLAHE_CLIP_LIMIT: float = 3.0  # Aggressive contrast (was 2.0)
    CLAHE_TILE_GRID: tuple = (16, 16)  # Finer grid (was 8, 8)
    
    # QC thresholds - HIGH-QUALITY IMAGERY ONLY
    MIN_IMAGE_SIZE_BYTES: int = 100000  # VERY LARGE tiles (was 50000)
    MAX_CLOUD_COVERAGE_PCT: int = 10  # ULTRA-STRICT: <10% clouds only
    MIN_RESOLUTION_M: float = 0.008  # ULTRA-HIGH: 0.8cm/pixel (was 0.01)
    
    # Output settings
    OUTPUT_JSON_NAME: str = "predictions.json"
    ARTIFACT_FORMAT: str = "png"
    
    # Hackathon-specific enhancements
    INCLUDE_DETECTION_METADATA: bool = True  # Include detection confidence and bbox
    INCLUDE_POWER_OUTPUTS: bool = True  # Include annual energy, CO2 offset
    EXPORT_ARTIFACTS: bool = True  # Export overlay images
    ENABLE_MULTI_BOX_AGGREGATION: bool = True  # Merge overlapping detections

config = Config()
