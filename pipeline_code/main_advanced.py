# ============================================
# pipeline_code/main_advanced.py - ADVANCED INFERENCE with Multi-Scale + TTA
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

# Import local modules
from config import config
from fetch_imagery import fetch_google_static_map, fetch_esri_imagery, assess_image_quality
from detect_solar import SolarDetectorIntegrated, create_buffer_masks
from advanced_detection import AdvancedDetector
from quantify_area import calculate_panel_area, encode_polygon_mask, calculate_power_output
from generate_artifact import create_audit_overlay

def process_single_site_advanced(sample_id: int, lat: float, lon: float, 
                                 output_dir: Path, detector, advanced_detector) -> Dict:
    """
    Process a single site through the ADVANCED pipeline:
    - Multi-scale detection (4 scales)
    - TTA (5 augmentations)
    - Ensemble NMS with confidence boosting
    """
    
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
        'detection_method': 'advanced_multiscale_tta',  # New field
        'ensemble_count': 0  # Track ensemble voting
    }
    
    try:
        # STAGE 1: Fetch imagery
        try:
            image, metadata = fetch_google_static_map(lat, lon, config.GOOGLE_MAPS_API_KEY)
        except:
            print(f"  Google Maps failed for {sample_id}, trying ESRI...")
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
        
        # STAGE 3: ADVANCED DETECTION (Multi-Scale + TTA)
        print(f"  🚀 Running ADVANCED detection (multi-scale + TTA)...")
        
        if config.ENABLE_MULTISCALE_DETECTION and config.ENABLE_TTA:
            # Full advanced pipeline
            advanced_detections = advanced_detector.detect_with_multiscale_tta(image, buffer_masks)
        else:
            # Fallback to standard detection
            detection_result = detector.detect_panels(image, buffer_masks)
            advanced_detections = detection_result['detections']
        
        # STAGE 4: Process detections
        has_solar = len(advanced_detections) > 0
        max_confidence = 0.0
        primary_buffer_hit = False
        
        if advanced_detections:
            for det in advanced_detections:
                max_confidence = max(max_confidence, det['confidence'])
                
                # Check buffer zones
                x1, y1, x2, y2 = det['bbox']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if center_y < buffer_masks['primary'].shape[0] and center_x < buffer_masks['primary'].shape[1]:
                    if buffer_masks['primary'][int(center_y), int(center_x)] > 0:
                        primary_buffer_hit = True
            
            # Store detections with ensemble info
            result['detections'] = [
                {
                    'bbox': det.get('bbox', []),
                    'confidence': float(det.get('confidence', 0.0)),
                    'ensemble_count': int(det.get('ensemble_count', 1)),
                    'ensemble_variants': det.get('tta_variants', []),
                    'scales_detected': det.get('scales', [])
                }
                for det in advanced_detections
            ]
            
            result['has_solar'] = True
            result['confidence'] = float(max_confidence)
            result['buffer_radius_sqft'] = config.PRIMARY_BUFFER_SQFT if primary_buffer_hit else config.SECONDARY_BUFFER_SQFT
            result['ensemble_count'] = sum([d.get('ensemble_count', 1) for d in advanced_detections])
        
        # STAGE 5: Quantify area (if solar detected)
        if has_solar:
            meters_per_pixel = metadata['resolution_m']
            buffer_type = 'primary' if primary_buffer_hit else 'secondary'
            
            area_sqm = calculate_panel_area(
                advanced_detections,
                buffer_masks,
                meters_per_pixel,
                buffer_type
            )
            
            result['pv_area_sqm_est'] = round(area_sqm, 2)
            
            # Calculate power output
            power_output = calculate_power_output(area_sqm)
            result['installed_capacity_kw'] = power_output['installed_capacity_kw']
            result['daily_energy_kwh'] = power_output['daily_energy_kwh']
            result['annual_energy_kwh'] = power_output['annual_energy_kwh']
            result['co2_offset_kg_per_year'] = power_output['co2_offset_kg_per_year']
            
            result['qc_status'] = 'VERIFIABLE'
        else:
            result['qc_status'] = 'NOT_VERIFIABLE'
        
        # STAGE 6: Generate artifacts
        artifact_path = output_dir / 'artifacts' / f"sample_{sample_id}_advanced.{config.ARTIFACT_FORMAT}"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_audit_overlay(
            image,
            advanced_detections,
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
    """Main entry point for advanced pipeline"""
    
    if len(sys.argv) < 2:
        print("Usage: python main_advanced.py <input_csv_or_xlsx> [output_directory]")
        print("Example: python main_advanced.py test_known_solar.csv pipeline_results_advanced")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pipeline_results_advanced")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"📂 Loading input data from {input_file}...")
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file, skipinitialspace=True)
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
    else:
        df = pd.read_excel(input_file)
    
    print(f"✓ Loaded {len(df)} samples")
    
    # Initialize detectors
    print("\n🔧 Initializing detectors...")
    detector = SolarDetectorIntegrated()
    advanced_detector = AdvancedDetector(detector.yolo_model)
    print("✓ Detectors ready")
    
    # Process each sample
    print(f"\n🎯 Processing {len(df)} samples with ADVANCED detection...\n")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Handle both uppercase and lowercase column names
        sample_id = row.get('Sample_ID', row.get('sample_id', idx))
        lat = row.get('Latitude', row.get('latitude', None))
        lon = row.get('Longitude', row.get('longitude', None))
        
        if lat is None or lon is None:
            print(f"⚠️  Skipping sample {idx}: missing coordinates")
            continue
        
        result = process_single_site_advanced(
            sample_id, lat, lon, output_dir, detector, advanced_detector
        )
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_json_path = output_dir / config.OUTPUT_JSON_NAME
    results_csv_path = output_dir / "results.csv"
    
    # Convert to JSON-serializable format
    results_dict = {
        'summary': {
            'total_samples': len(results),
            'samples_with_solar': sum(1 for r in results if r['has_solar']),
            'average_confidence': np.mean([r['confidence'] for r in results if r['confidence'] > 0]),
            'average_area_sqm': np.mean([r['pv_area_sqm_est'] for r in results if r['pv_area_sqm_est'] > 0]),
            'total_capacity_kw': sum(r['installed_capacity_kw'] for r in results),
            'total_annual_kwh': sum(r['annual_energy_kwh'] for r in results),
            'total_co2_offset_kg': sum(r['co2_offset_kg_per_year'] for r in results),
            'detection_method': 'advanced_multiscale_tta',
            'avg_ensemble_votes': np.mean([r['ensemble_count'] for r in results if r['ensemble_count'] > 0])
        },
        'results': results
    }
    
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    results_df.to_csv(results_csv_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("🏁 ADVANCED DETECTION PIPELINE COMPLETE")
    print("="*70)
    print(f"Total samples: {results_dict['summary']['total_samples']}")
    if results_dict['summary']['total_samples'] > 0:
        print(f"Solar detected: {results_dict['summary']['samples_with_solar']} ({100*results_dict['summary']['samples_with_solar']/results_dict['summary']['total_samples']:.1f}%)")
        print(f"Avg confidence: {results_dict['summary']['average_confidence']:.3f}")
        print(f"Avg ensemble votes per detection: {results_dict['summary']['avg_ensemble_votes']:.1f}")
        print(f"Avg area: {results_dict['summary']['average_area_sqm']:.2f} m²")
    else:
        print("⚠️  No samples processed!")
    print(f"Total capacity: {results_dict['summary']['total_capacity_kw']:.2f} kW")
    print(f"Total annual energy: {results_dict['summary']['total_annual_kwh']:.0f} kWh")
    print(f"Total CO₂ offset: {results_dict['summary']['total_co2_offset_kg']:.0f} kg/year")
    print(f"\n✓ Results saved to {results_json_path}")
    print(f"✓ CSV saved to {results_csv_path}")
    print(f"✓ Artifacts in {output_dir / 'artifacts'}")
    print("="*70)

if __name__ == "__main__":
    main()
