# PM Surya Ghar: Rooftop Solar Verification System

## Overview
AI-powered pipeline for verifying rooftop solar panel installations for the PM Surya Ghar Muft Bijli Yojana subsidy program. Uses YOLOv11 for detection and SAM2 for precise segmentation.

## Performance Metrics
- **F1 Score:** 0.971
- **Precision:** 1.0 (100%)
- **Recall:** 0.944 (94.4%)
- **Average Confidence:** 0.729
- **Runtime:** ~8 minutes for 18 samples

## Folder Structure
```
submission/
├── pipeline_code/          # System code for inference (.py files)
│   ├── main.py             # Main inference pipeline
│   ├── config.py           # Configuration settings
│   ├── detect_solar.py     # YOLO + SAM2 detection
│   ├── fetch_imagery.py    # Google Maps / ESRI API fetching
│   ├── quantify_area.py    # Area calculation from masks
│   ├── generate_artifact.py # Audit overlay generation
│   └── rag_explainer.py    # HyDE RAG explanation system
│
├── environment/            # Environment details
│   ├── requirements.txt    # pip dependencies
│   ├── environment.yml     # conda environment
│   └── python_version.txt  # Python version (3.11.9)
│
├── trained_model/          # Trained model files
│   ├── yolov11_solar.pt    # YOLOv11 trained weights
│   ├── sam2.1_hiera_base_plus.pt  # SAM2.1 checkpoint
│   ├── sam2.1_hiera_b+.yaml       # SAM2 config
│   └── training_config.yaml       # Training configuration
│
├── model_card/             # Model documentation
│   └── MODEL_CARD.md       # Model card (see below for PDF)
│
├── predictions/            # Prediction files
│   ├── predictions.json    # Core predictions (hackathon format)
│   └── predictions_with_explanations.json  # Extended with RAG
│
├── artefacts/              # Visual artifacts
│   └── sample_*_overlay.png  # Overlay images for each sample
│
├── training_logs/          # Training metrics
│   ├── training_metrics.csv  # Epoch-wise metrics
│   └── metrics.json          # MLflow export
│
└── README.md               # This file
```

## Quick Start

### 1. Setup Environment

**Option A: Conda (Recommended)**
```bash
cd submission/environment
conda env create -f environment.yml
conda activate solar-verification
```

**Option B: pip**
```bash
pip install -r submission/environment/requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
ESRI_API_KEY=your_esri_api_key  # Optional fallback
```

### 3. Run Inference

```bash
cd submission/pipeline_code
python main.py <input_file.csv> <output_directory>
```

**Example:**
```bash
python main.py test_known_solar.csv results/
```

### 4. Input Format

CSV file with columns:
```csv
sample_id,latitude,longitude
1,28.6139,77.2090
2,19.0760,72.8777
```

### 5. Output Format

**predictions.json** (exact hackathon format):
```json
{
    "sample_id": 1234,
    "lat": 12.9716,
    "lon": 77.5946,
    "has_solar": true,
    "confidence": 0.92,
    "pv_area_sqm_est": 23.5,
    "buffer_radius_sqft": 1200,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": "<encoded polygon>",
    "image_metadata": {
        "source": "Google Maps Static API",
        "capture_date": "YYYY-MM-DD"
    }
}
```

## Pipeline Stages

1. **FETCH** - Retrieves high-resolution satellite imagery via Google Maps API
2. **CLASSIFY** - YOLOv11 detection within 1200/2400 sqft buffer zones
3. **QUANTIFY** - SAM2 segmentation for precise area estimation (m²)
4. **EXPLAIN** - HyDE RAG generates audit-friendly explanations
5. **STORE** - JSON predictions + PNG artifact overlays

## QC Status Values

| Status | Meaning |
|--------|---------|
| `VERIFIABLE` | Clear evidence either way (present/not present) |
| `NOT_VERIFIABLE` | Insufficient evidence (low resolution, shadow, cloud, occlusion) |

## Key Technologies

- **YOLOv11** - Object detection for solar panels
- **SAM2.1** (Segment Anything Model 2) - Precise mask segmentation
- **HyDE RAG** - Hypothetical Document Embeddings for explainability
- **Google Maps Static API** - Satellite imagery fetching
- **OpenCV** - Heavy preprocessing for enhanced detection

## Requirements

- Python 3.11.9
- CUDA-capable GPU (recommended)
- Google Maps API key with Static Maps enabled
- 8GB+ VRAM for SAM2

## Authors

PM Surya Ghar Hackathon Team

## License

For hackathon evaluation purposes only.
