# ============================================
# pipeline_code/main.py - MAIN INFERENCE PIPELINE
# ============================================

# Torch 2.1 compatibility fix for transformers library
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

# Import local modules
from config import config
from fetch_imagery import fetch_google_static_map, fetch_esri_imagery, assess_image_quality
from detect_solar import SolarDetectorIntegrated, create_buffer_masks
from quantify_area import calculate_panel_area, encode_polygon_mask, calculate_power_output
from generate_artifact import create_audit_overlay
from rag_explainer import HyDERAGExplainer

def process_single_site(sample_id: int, lat: float, lon: float, 
                       output_dir: Path, detector) -> Dict:
    """Process a single site through the complete pipeline"""
    
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
        'detections': [],  # lightweight detections summary
        'image_metadata': {},
        # Power output fields
        'installed_capacity_kw': 0.0,
        'daily_energy_kwh': 0.0,
        'annual_energy_kwh': 0.0,
        'co2_offset_kg_per_year': 0.0
    }
    
    try:
        # STAGE 1: Fetch imagery
        try:
            image, metadata = fetch_google_static_map(lat, lon, config.GOOGLE_MAPS_API_KEY)
        except:
            print(f"Google Maps failed for {sample_id}, trying ESRI...")
            image, metadata = fetch_esri_imagery(lat, lon, config.ESRI_API_KEY)
        
        result['image_metadata'] = metadata
        
        # Check image quality and determine verifiability
        quality_ok, issues = assess_image_quality(image, metadata)
        
        # Store QC issues for explainability
        result['qc_issues'] = issues if issues else []
        
        if not quality_ok:
            result['qc_status'] = 'NOT_VERIFIABLE'
            result['qc_reason'] = f"Insufficient evidence: {'; '.join(issues)}"
            return result
        
        # STAGE 2: Create buffer masks
        h, w = image.shape[:2]
        center = (h // 2, w // 2)
        buffer_masks = create_buffer_masks(image.shape[:2], center)
        
        # STAGE 3: Detect solar panels (reusing detector)
        detection_result = detector.detect_panels(image, buffer_masks)
        
        result['has_solar'] = detection_result['has_solar']
        result['confidence'] = detection_result['confidence']
        result['buffer_radius_sqft'] = detection_result['buffer_radius_sqft']
        
        # Store all detections for full traceability
        if detection_result['detections']:
            result['detections'] = [
                {
                    'bbox': [float(x) for x in det['bbox']],
                    'confidence': float(det['confidence']),
                    'in_primary_buffer': bool(det['in_primary_buffer']),
                    'in_secondary_buffer': bool(det['in_secondary_buffer'])
                }
                for det in detection_result['detections']
            ]
        
        # STAGE 4: Quantify area (if solar detected)
        if detection_result['has_solar']:
            meters_per_pixel = metadata['resolution_m']
            buffer_type = 'primary' if detection_result['primary_buffer'] else 'secondary'
            
            area_sqm = calculate_panel_area(
                detection_result['detections'],
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
            
            # Encode polygon/bbox from first detection
            if detection_result['detections']:
                first_mask = detection_result['detections'][0].get('mask')
                if first_mask is not None:
                    result['bbox_or_mask'] = encode_polygon_mask(first_mask)
                else:
                    result['bbox_or_mask'] = str(detection_result['detections'][0]['bbox'])
            
            # QC Status: VERIFIABLE = clear evidence of solar presence
            result['qc_status'] = 'VERIFIABLE'
            result['qc_reason'] = 'Clear evidence of solar panel presence detected with high-quality imagery'
        else:
            # No solar detected - still VERIFIABLE if we have clear evidence of absence
            if detection_result['detections']:
                first_det = detection_result['detections'][0]
                first_mask = first_det.get('mask')
                if first_mask is not None:
                    result['bbox_or_mask'] = encode_polygon_mask(first_mask)
                else:
                    result['bbox_or_mask'] = str(first_det['bbox'])
            
            # VERIFIABLE = clear evidence of no solar (high-quality image, no detections)
            # NOT_VERIFIABLE = cannot determine (image quality issues)
            result['qc_status'] = 'VERIFIABLE'
            result['qc_reason'] = 'Clear evidence: high-quality imagery analyzed, no solar panels detected in buffer zones'
        
        # STAGE 5: Generate artifacts
        artifact_path = output_dir / 'artifacts' / f"sample_{sample_id}_overlay.{config.ARTIFACT_FORMAT}"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_audit_overlay(
            image,
            detection_result['detections'],
            buffer_masks,
            metadata,
            artifact_path
        )
        
        return result
        
    except Exception as e:
        print(f"Error processing sample {sample_id}: {str(e)}")
        result['qc_status'] = 'NOT_VERIFIABLE'
        result['qc_reason'] = f"Insufficient evidence: Processing error - {str(e)}"
        result['qc_issues'] = [str(e)]
        return result

def main(input_xlsx: Path, output_dir: Path):
    """
    Main pipeline: Process all samples from Excel file
    
    Args:
        input_xlsx: Path to .xlsx file with columns: sample_id, latitude, longitude
        output_dir: Path to output directory
    """
    print("="*60)
    print("PM SURYA GHAR: ROOFTOP SOLAR VERIFICATION SYSTEM")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"\nLoading data from: {input_xlsx}")
    input_path = Path(input_xlsx)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, skipinitialspace=True)
    else:
        df = pd.read_excel(input_path)

    # Normalize column names to avoid whitespace issues
    df.columns = [c.strip() for c in df.columns]
    
    required_cols = ['sample_id', 'latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input must contain columns: {required_cols}")
    
    print(f"Found {len(df)} samples to process")
    
    # Create detector once (expensive operation - saves ~seconds per sample)
    print("Initializing solar panel detector...")
    detector = SolarDetectorIntegrated()
    print("✓ Detector ready")
    
    # Initialize HyDE RAG explainer
    print("Initializing HyDE RAG explainer...")
    explainer = HyDERAGExplainer()
    print("✓ RAG Explainer ready\n")
    
    # Process each sample
    all_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        result = process_single_site(
            sample_id=int(row['sample_id']),
            lat=float(row['latitude']),
            lon=float(row['longitude']),
            output_dir=output_dir,
            detector=detector
        )
        
        # Generate HyDE RAG explanation for this prediction
        explanation_result = explainer.explain(result)
        result['explanation'] = explainer.to_dict(explanation_result)
        
        all_results.append(result)
    
    # Save all predictions to JSON
    predictions_path = output_dir / config.OUTPUT_JSON_NAME
    with open(predictions_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Predictions saved to: {predictions_path}")
    print(f"✓ Artifacts saved to: {output_dir / 'artifacts'}")
    
    # Print summary statistics
    has_solar_count = sum(1 for r in all_results if r['has_solar'])
    verifiable_count = sum(1 for r in all_results if r['qc_status'] == 'VERIFIABLE')
    
    print(f"\nSummary:")
    print(f"  Total samples: {len(all_results)}")
    print(f"  Solar detected: {has_solar_count} ({has_solar_count/len(all_results)*100:.1f}%)")
    print(f"  Verifiable: {verifiable_count} ({verifiable_count/len(all_results)*100:.1f}%)")
    print(f"  Avg confidence: {np.mean([r['confidence'] for r in all_results]):.2f}")
    print(f"  Total PV area: {sum(r['pv_area_sqm_est'] for r in all_results):.1f} m²")
    
    # HACKATHON IMPACT METRICS
    total_capacity_kw = sum(r['installed_capacity_kw'] for r in all_results)
    total_annual_energy = sum(r['annual_energy_kwh'] for r in all_results)
    total_co2_offset = sum(r['co2_offset_kg_per_year'] for r in all_results)

    print(f"  Total Installed Capacity: {total_capacity_kw:.2f} kW")
    print(f"  Total Annual Energy: {total_annual_energy:,.0f} kWh/year")
    print(f"  Total CO₂ Offset: {total_co2_offset:,.0f} kg/year (~{total_co2_offset/1000:.1f} tonnes)")
    if has_solar_count > 0:
        print(f"  Avg Confidence (Solar): {np.mean([r['confidence'] for r in all_results if r['has_solar']]):.3f}")
        print(f"  Avg Area (Solar): {np.mean([r['pv_area_sqm_est'] for r in all_results if r['has_solar']]):.2f} m²")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_xlsx_path> <output_directory>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    main(input_file, output_dir)
