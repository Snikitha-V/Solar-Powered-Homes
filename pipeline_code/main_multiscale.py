# ============================================
# pipeline_code/main_multiscale.py - SIMPLE MULTI-SCALE ONLY (Fast + Effective)
# ============================================

# Torch 2.1 compatibility fix
try:
    from torch.utils import _pytree as torch_pytree
    if not hasattr(torch_pytree, 'register_pytree_node') and hasattr(torch_pytree, '_register_pytree_node'):
        def _compat_register(node_type, flatten_fn, unflatten_fn, *, serialized_type_name=None, serialized_context_fn=None):
            return torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)
        torch_pytree.register_pytree_node = _compat_register
except Exception:
    pass

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
from typing import Dict
import cv2

from config import config
from fetch_imagery import fetch_google_static_map, fetch_esri_imagery, assess_image_quality
from detect_solar import SolarDetectorIntegrated, create_buffer_masks
from quantify_area import calculate_panel_area, encode_polygon_mask, calculate_power_output
from generate_artifact import create_audit_overlay

def detect_multiscale(image, detector, confidence_threshold=0.22, nms_threshold=0.40):
    """
    Multi-scale detection: Process at 2 scales and ensemble results
    Fast and effective approach without TTA
    """
    h, w = image.shape[:2]
    all_detections = []
    
    # Scale 1: Original size
    results = detector.yolo_model.predict(
        image,
        conf=confidence_threshold,
        iou=nms_threshold,
        imgsz=config.DETECTION_IMAGE_SIZE,
        verbose=False,
        augment=False
    )
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            all_detections.append({
                'bbox': box.tolist(),
                'confidence': float(conf),
                'scale': 1.0
            })
    
    # Scale 2: Upscaled (1.2x) - catches small panels
    scale2_factor = 1.2
    scaled_h = int(h * scale2_factor)
    scaled_w = int(w * scale2_factor)
    scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    
    results = detector.yolo_model.predict(
        scaled_image,
        conf=confidence_threshold,
        iou=nms_threshold,
        imgsz=config.DETECTION_IMAGE_SIZE,
        verbose=False,
        augment=False
    )
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            # Rescale back to original size
            x1, y1, x2, y2 = x1/scale2_factor, y1/scale2_factor, x2/scale2_factor, y2/scale2_factor
            
            all_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'scale': scale2_factor
            })
    
    # Simple NMS: Remove near-duplicates
    if all_detections:
        all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        final_dets = []
        used = set()
        
        for i, det in enumerate(all_detections):
            if i in used:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            area_i = (x2 - x1) * (y2 - y1)
            cluster = [det]
            used.add(i)
            
            for j in range(i + 1, len(all_detections)):
                if j in used:
                    continue
                
                x1_j, y1_j, x2_j, y2_j = all_detections[j]['bbox']
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                
                inter_x1 = max(x1, x1_j)
                inter_y1 = max(y1, y1_j)
                inter_x2 = min(x2, x2_j)
                inter_y2 = min(y2, y2_j)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # Stricter NMS threshold
                    if iou > 0.55:
                        cluster.append(all_detections[j])
                        used.add(j)
            
            # Average cluster
            avg_bbox = np.mean([d['bbox'] for d in cluster], axis=0).tolist()
            avg_conf = np.mean([d['confidence'] for d in cluster])
            
            # Small boost if multiple votes from different scales
            if len(cluster) > 1:
                avg_conf = min(avg_conf + 0.03, 0.99)
            
            final_dets.append({
                'bbox': avg_bbox,
                'confidence': float(avg_conf),
                'num_scales': len(cluster)
            })
        
        return final_dets
    
    return []

def process_single_site_multiscale(sample_id: int, lat: float, lon: float, 
                                   output_dir: Path, detector) -> Dict:
    """Process a single site with multi-scale detection"""
    
    result = {
        'sample_id': sample_id,
        'lat': lat,
        'lon': lon,
        'has_solar': False,
        'confidence': 0.0,
        'pv_area_sqm_est': 0.0,
        'buffer_radius_sqft': config.SECONDARY_BUFFER_SQFT,
        'qc_status': 'NOT_VERIFIABLE',
        'bbox_or_mask': '',
        'detections': [],
        'image_metadata': {},
        'installed_capacity_kw': 0.0,
        'daily_energy_kwh': 0.0,
        'annual_energy_kwh': 0.0,
        'co2_offset_kg_per_year': 0.0,
        'detection_method': 'multiscale_simple'
    }
    
    try:
        # STAGE 1: Fetch imagery
        try:
            image, metadata = fetch_google_static_map(lat, lon, config.GOOGLE_MAPS_API_KEY)
        except:
            image, metadata = fetch_esri_imagery(lat, lon, config.ESRI_API_KEY)
        
        result['image_metadata'] = metadata
        
        # Check image quality
        quality_ok, issues = assess_image_quality(image, metadata)
        if not quality_ok:
            result['qc_status'] = 'NOT_VERIFIABLE'
            result['qc_notes'] = str(issues) if issues else ''
            return result
        
        # STAGE 2: Create buffer masks
        buffer_masks = create_buffer_masks(image.shape[:2], (image.shape[0]//2, image.shape[1]//2))
        
        # STAGE 3: Multi-scale detection
        multiscale_detections = detect_multiscale(image, detector)
        
        # STAGE 4: Process detections
        has_solar = len(multiscale_detections) > 0
        max_confidence = 0.0
        primary_buffer_hit = False
        
        if multiscale_detections:
            for det in multiscale_detections:
                max_confidence = max(max_confidence, det['confidence'])
                
                # Check buffer zones
                x1, y1, x2, y2 = det['bbox']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if center_y < buffer_masks['primary'].shape[0] and center_x < buffer_masks['primary'].shape[1]:
                    if buffer_masks['primary'][int(center_y), int(center_x)] > 0:
                        primary_buffer_hit = True
            
            result['detections'] = [
                {
                    'bbox': det.get('bbox', []),
                    'confidence': float(det.get('confidence', 0.0)),
                    'num_scales': int(det.get('num_scales', 1))
                }
                for det in multiscale_detections
            ]
            
            result['has_solar'] = True
            result['confidence'] = float(max_confidence)
            result['buffer_radius_sqft'] = config.PRIMARY_BUFFER_SQFT if primary_buffer_hit else config.SECONDARY_BUFFER_SQFT
        
        # STAGE 5: Quantify area
        if has_solar:
            meters_per_pixel = metadata['resolution_m']
            buffer_type = 'primary' if primary_buffer_hit else 'secondary'
            
            area_sqm = calculate_panel_area(
                multiscale_detections,
                buffer_masks,
                meters_per_pixel,
                buffer_type
            )
            
            result['pv_area_sqm_est'] = round(area_sqm, 2)
            
            power_output = calculate_power_output(area_sqm)
            result['installed_capacity_kw'] = power_output['installed_capacity_kw']
            result['daily_energy_kwh'] = power_output['daily_energy_kwh']
            result['annual_energy_kwh'] = power_output['annual_energy_kwh']
            result['co2_offset_kg_per_year'] = power_output['co2_offset_kg_per_year']
            
            result['qc_status'] = 'VERIFIABLE'
        else:
            result['qc_status'] = 'NOT_VERIFIABLE'
        
        # STAGE 6: Generate artifacts
        artifact_path = output_dir / 'artifacts' / f"sample_{sample_id}_multiscale.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_audit_overlay(
            image,
            multiscale_detections,
            buffer_masks,
            metadata,
            artifact_path
        )
        
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing sample {sample_id}: {str(e)}")
        traceback.print_exc()
        return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_multiscale.py <input_csv_or_xlsx> [output_directory]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pipeline_results_multiscale")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"📂 Loading {input_file}...")
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file, skipinitialspace=True)
    else:
        df = pd.read_excel(input_file)
    
    print(f"✓ Loaded {len(df)} samples")
    
    # Initialize detector
    print("\n🔧 Initializing detector...")
    detector = SolarDetectorIntegrated()
    print("✓ Detector ready")
    
    # Process samples
    print(f"\n🎯 Processing {len(df)} samples with MULTI-SCALE detection...\n")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = row.get('Sample_ID', row.get('sample_id', idx))
        lat = row.get('Latitude', row.get('latitude'))
        lon = row.get('Longitude', row.get('longitude'))
        
        result = process_single_site_multiscale(sample_id, lat, lon, output_dir, detector)
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_json_path = output_dir / config.OUTPUT_JSON_NAME
    results_csv_path = output_dir / "results.csv"
    
    results_dict = {
        'summary': {
            'total_samples': len(results),
            'samples_with_solar': sum(1 for r in results if r['has_solar']),
            'average_confidence': np.mean([r['confidence'] for r in results if r['confidence'] > 0]) if any(r['confidence'] for r in results) else 0,
            'average_area_sqm': np.mean([r['pv_area_sqm_est'] for r in results if r['pv_area_sqm_est'] > 0]) if any(r['pv_area_sqm_est'] for r in results) else 0,
            'total_capacity_kw': sum(r['installed_capacity_kw'] for r in results),
            'total_annual_kwh': sum(r['annual_energy_kwh'] for r in results),
            'total_co2_offset_kg': sum(r['co2_offset_kg_per_year'] for r in results),
            'detection_method': 'multiscale_simple'
        },
        'results': results
    }
    
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    results_df.to_csv(results_csv_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("🏁 MULTI-SCALE DETECTION COMPLETE")
    print("="*70)
    print(f"Total samples: {results_dict['summary']['total_samples']}")
    print(f"Solar detected: {results_dict['summary']['samples_with_solar']}")
    print(f"Detection rate: {100*results_dict['summary']['samples_with_solar']/results_dict['summary']['total_samples']:.1f}%")
    print(f"Avg confidence: {results_dict['summary']['average_confidence']:.4f}")
    print(f"Avg area: {results_dict['summary']['average_area_sqm']:.2f} m²")
    print(f"Total capacity: {results_dict['summary']['total_capacity_kw']:.2f} kW")
    print(f"Total annual energy: {results_dict['summary']['total_annual_kwh']:.0f} kWh")
    print(f"Total CO₂ offset: {results_dict['summary']['total_co2_offset_kg']:.0f} kg/year")
    print(f"\n✓ Results saved to {results_json_path}")
    print("="*70)

if __name__ == "__main__":
    main()
