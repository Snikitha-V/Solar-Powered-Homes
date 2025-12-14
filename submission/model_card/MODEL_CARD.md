# Model Card: Solar Panel Detection System

## Model Overview

| Attribute | Details |
|-----------|---------|
| **Model Name** | PM Surya Ghar Solar Verification Pipeline |
| **Version** | 1.0 |
| **Type** | Multi-stage Detection + Segmentation |
| **Primary Task** | Rooftop solar panel detection and area quantification |
| **Framework** | PyTorch (YOLOv11 + SAM2.1) |

---

## 1. Model Architecture

### Detection Stage: YOLOv11
- **Base Model:** Ultralytics YOLOv11n
- **Fine-tuned on:** Custom solar panel dataset
- **Input Size:** 1280×1280 pixels
- **Output:** Bounding boxes + confidence scores

### Segmentation Stage: SAM2.1
- **Model:** Segment Anything Model 2.1 (Hiera Base Plus)
- **Checkpoint:** sam2.1_hiera_base_plus.pt (323 MB)
- **Purpose:** Pixel-precise mask generation from YOLO bboxes

### Explainability: HyDE RAG
- **Technique:** Hypothetical Document Embeddings
- **Retrieval:** Keyword-based from solar knowledge base
- **Output:** Structured explanations for each prediction

---

## 2. Training Data

### Dataset Composition
| Source | Count | Description |
|--------|-------|-------------|
| Roboflow Solar Panels | ~2,000 | Aerial views of residential rooftops |
| Google Maps Imagery | ~500 | Indian urban/suburban locations |
| Negative Samples | ~800 | Rooftops without solar panels |

### Annotation Format
- Bounding boxes in YOLO format (class, x_center, y_center, width, height)
- Single class: "solar_panel"

### Geographic Distribution
- Primary: India (various states)
- Secondary: Australia, USA (for generalization)

---

## 3. Performance Metrics

### Test Set Results (18 known solar locations)

| Metric | Value |
|--------|-------|
| **F1 Score** | 0.971 |
| **Precision** | 1.000 |
| **Recall** | 0.944 |
| **True Positives** | 17/18 |
| **False Negatives** | 1/18 |
| **Average Confidence** | 0.729 |

### Confidence Score Calibration
- Power-law scaling (conf^0.6) for low-value lift
- Sigmoid transformation for mid-range boosting
- Additive layer boosts (+0.12 base, +0.05 per stage)

---

## 4. Assumptions

1. **Satellite Imagery Quality**
   - Resolution ≤ 0.05 m/pixel (zoom level 21+)
   - Clear weather conditions (no heavy clouds)
   - Recent imagery (within 2 years of installation)

2. **Panel Characteristics**
   - Standard rectangular photovoltaic modules
   - Blue/black coloring with grid patterns
   - Minimum detectable size: ~0.5 m²

3. **Buffer Zone Definitions**
   - Primary buffer: 1200 sq ft radius from coordinates
   - Secondary buffer: 2400 sq ft radius from coordinates

4. **Geographic Context**
   - Model optimized for Indian residential rooftops
   - Urban and suburban settings

---

## 5. Known Limitations & Bias

### Detection Limitations
| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Unusual panel colors** | May miss non-standard panels (red, brown) | Lower confidence threshold |
| **Partial occlusion** | Trees/shadows reduce detection | Multi-scale processing |
| **Small installations** | <0.5 m² may be missed | Increased image resolution |
| **Ground-mounted panels** | May detect non-rooftop installations | Buffer zone filtering |

### Known Biases
1. **Geographic Bias:** Trained primarily on Australian/Indian imagery; may underperform in other regions
2. **Panel Type Bias:** Crystalline silicon panels detected better than thin-film
3. **Rooftop Type Bias:** Flat concrete roofs detected better than complex tiled roofs
4. **Temporal Bias:** Model uses static imagery; new installations may not appear

### Failure Modes
| Failure Mode | Cause | Recommended Action |
|--------------|-------|-------------------|
| **False Negative** | Heavy shadows, tree cover | Request site photos |
| **False Positive** | Skylights, blue tarps, water tanks | Manual review |
| **Low Confidence** | Unusual panel appearance | Enhanced verification |
| **NOT_VERIFIABLE** | Poor image quality | Request updated imagery |

---

## 6. Ethical Considerations

### Fairness
- Model treats all geographic regions equally
- No demographic data used in predictions
- Transparent confidence scores for all predictions

### Privacy
- Only processes satellite imagery (publicly available)
- No personal data stored or processed
- Coordinates used only for image fetching

### Transparency
- Full explainability via HyDE RAG system
- QC status clearly indicates verification confidence
- Audit artifacts provided for human review

---

## 7. Retraining Guidance

### When to Retrain
1. Detection rate drops below 80% on new data
2. New panel types become common (e.g., bifacial, BIPV)
3. Imagery source changes (different satellite provider)
4. Significant false positive increase

### Retraining Steps
```bash
# 1. Prepare new dataset in YOLO format
# 2. Update training config
python training/train_yolo.py --data new_dataset.yaml --epochs 100

# 3. Evaluate on validation set
python training/evaluate.py --model new_model.pt

# 4. Update checkpoint paths in config.py
```

### Data Requirements for Retraining
- Minimum 500 new annotated images
- 30% negative samples (no solar)
- Geographic diversity matching deployment region
- Multiple zoom levels (19-22)

---

## 8. Deployment Recommendations

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB |
| CPU | 4 cores | 8+ cores |
| Storage | 2GB | 5GB |

### API Rate Limits
- Google Maps Static API: 25,000 requests/day (free tier)
- Recommended: 1-2 second delay between requests

### Monitoring
- Log detection confidence distributions
- Track QC status ratios (VERIFIABLE vs NOT_VERIFIABLE)
- Monitor API response times and failures

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial release for PM Surya Ghar hackathon |

---

## 10. Contact

For questions about this model, contact the PM Surya Ghar Hackathon Team.

---

*This model card follows the format recommended by Mitchell et al. (2019) "Model Cards for Model Reporting"*
