# PM Surya Ghar: Rooftop Solar Verification System

## Overview
AI-powered pipeline for verifying rooftop solar panel installations for the PM Surya Ghar Muft Bijli Yojana subsidy program. Uses YOLOv11 for detection and SAM2 for precise segmentation.

## Performance Metrics
- **F1 Score:** 0.971
- **Precision:** 0.964 (96%)
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

Ecoforage Team

## License

This project is licensed under the **MIT License** - see below:

```
MIT License

Copyright (c) 2025 PM Surya Ghar Hackathon Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---


## Data Sources & Licensing

### Imagery Sources
| Source | License | Usage |
|--------|---------|-------|
| **Google Maps Static API** | [Google Maps Platform ToS](https://cloud.google.com/maps-platform/terms) | Primary satellite imagery (requires API key) |
| **ESRI World Imagery** | [Esri Master License Agreement](https://www.esri.com/en-us/legal/terms/full-master-agreement) | Fallback imagery source |

### Training Data
| Dataset | License | Citation |
|---------|---------|----------|
| **Roboflow Solar Panels Dataset** | CC BY 4.0 | Solar panel detection training images from Roboflow Universe |
| **Custom Annotations** | MIT | Team-annotated samples for Indian rooftop conditions |

### Pre-trained Models
| Model | License | Source |
|-------|---------|--------|
| **YOLOv11** | AGPL-3.0 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **SAM2.1** | Apache 2.0 | [Meta AI - Segment Anything](https://github.com/facebookresearch/sam2) |

> ⚠️ **Note:** When using this system in production, ensure compliance with Google Maps Platform Terms of Service, particularly regarding caching and display requirements.

---

## Known Biases & Limitations

### Identified Biases

| Bias Type | Description | Impact |
|-----------|-------------|--------|
| **Urban vs Rural Gap** | Model performs better on urban installations with regular panel layouts | Rural areas with non-standard installations may have 10-15% lower recall |
| **Panel Type Bias** | Trained primarily on blue/black crystalline panels | Thin-film or building-integrated PV (BIPV) may be missed |
| **Roof Material Bias** | Better performance on concrete/RCC roofs | Tiled, thatched, or metal sheet roofs may cause false positives/negatives |
| **Geographic Bias** | Training data concentrated on North/South Indian metros | Northeast and rural installations underrepresented |
| **Seasonal Bias** | Training imagery mostly from dry season | Monsoon cloud cover and wet surfaces may reduce accuracy |
| **Scale Bias** | Optimized for residential (5-25 m²) installations | Very large commercial or very small installations may have lower accuracy |

### Mitigation Steps Implemented

1. **Multi-scale Detection** - Processes images at multiple resolutions to handle varying panel sizes
2. **Heavy Preprocessing** - 8-stage image enhancement (CLAHE, sharpening, denoising) improves detection in poor conditions
3. **Bounding Box Expansion (25%)** - Ensures full panel capture even with partial initial detections
4. **SAM2 Refinement** - Precise segmentation corrects YOLO's rough bounding boxes
5. **QC Status System** - `NOT_VERIFIABLE` flag for low-confidence or ambiguous cases
6. **RAG Explanations** - Human-readable justifications for audit review

### Recommended Mitigations for Production

1. **Continuous Learning** - Retrain quarterly with newly verified installations
2. **Regional Fine-tuning** - Create regional model variants for underrepresented areas
3. **Human-in-the-loop** - Mandatory manual review for `NOT_VERIFIABLE` cases
4. **Multi-temporal Analysis** - Use multiple imagery dates to confirm installations
5. **Ground Truth Validation** - Random sampling with on-site verification
6. **Bias Monitoring Dashboard** - Track performance metrics by region, roof type, and panel size

### Performance by Context

| Context | Estimated Accuracy | Notes |
|---------|-------------------|-------|
| Urban Metro (Tier 1) | 95%+ | Best performance |
| Urban (Tier 2/3) | 90-95% | Good performance |
| Semi-urban | 85-90% | Moderate performance |
| Rural | 75-85% | Lower due to varied conditions |
| Commercial (large scale) | 80-90% | May need parameter tuning |

---

## Ethical Considerations

- This system is designed to **assist** human reviewers, not replace them
- Final subsidy decisions should always involve human oversight
- False negatives may deny legitimate subsidies; false positives may enable fraud
- Regular audits recommended to ensure fairness across demographics and geographies
