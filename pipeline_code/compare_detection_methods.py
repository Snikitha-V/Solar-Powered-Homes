# ============================================
# pipeline_code/compare_detection_methods.py
# Compare baseline vs advanced (multi-scale+TTA) detection
# ============================================

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def load_results(results_path: str) -> Dict:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)

def compare_results(baseline_json: str, advanced_json: str) -> Dict:
    """
    Compare two detection result sets
    baseline_json: Path to baseline results (standard detection)
    advanced_json: Path to advanced results (multi-scale + TTA)
    """
    
    baseline = load_results(baseline_json)
    advanced = load_results(advanced_json)
    
    baseline_results = baseline['results']
    advanced_results = advanced['results']
    
    # Match results by sample_id
    comparison = {}
    improvements = {
        'confidence_boost': [],
        'new_detections': [],
        'false_positives': [],
        'area_adjustments': []
    }
    
    for adv_res in advanced_results:
        sample_id = adv_res['sample_id']
        
        # Find matching baseline result
        base_res = next((r for r in baseline_results if r['sample_id'] == sample_id), None)
        if base_res is None:
            continue
        
        comparison[sample_id] = {
            'sample_id': sample_id,
            'lat': adv_res['lat'],
            'lon': adv_res['lon'],
            'baseline': {
                'has_solar': base_res['has_solar'],
                'confidence': base_res['confidence'],
                'area_sqm': base_res['pv_area_sqm_est'],
                'detected': 1 if base_res['has_solar'] else 0
            },
            'advanced': {
                'has_solar': adv_res['has_solar'],
                'confidence': adv_res['confidence'],
                'area_sqm': adv_res['pv_area_sqm_est'],
                'detected': 1 if adv_res['has_solar'] else 0,
                'ensemble_votes': adv_res.get('ensemble_count', 1)
            }
        }
        
        # Track improvements
        conf_delta = adv_res['confidence'] - base_res['confidence']
        
        if conf_delta > 0.05:  # Significant boost
            improvements['confidence_boost'].append({
                'sample_id': sample_id,
                'baseline_conf': base_res['confidence'],
                'advanced_conf': adv_res['confidence'],
                'boost': conf_delta,
                'pct_improvement': (conf_delta / base_res['confidence'] * 100) if base_res['confidence'] > 0 else np.inf
            })
        
        if adv_res['has_solar'] and not base_res['has_solar']:
            improvements['new_detections'].append({
                'sample_id': sample_id,
                'advanced_conf': adv_res['confidence'],
                'area_sqm': adv_res['pv_area_sqm_est'],
                'ensemble_votes': adv_res.get('ensemble_count', 1)
            })
        
        if base_res['has_solar'] and not adv_res['has_solar']:
            improvements['false_positives'].append({
                'sample_id': sample_id,
                'baseline_conf': base_res['confidence']
            })
        
        if adv_res['has_solar'] and base_res['has_solar']:
            area_delta = adv_res['pv_area_sqm_est'] - base_res['pv_area_sqm_est']
            if abs(area_delta) > 0.1:
                improvements['area_adjustments'].append({
                    'sample_id': sample_id,
                    'baseline_area': base_res['pv_area_sqm_est'],
                    'advanced_area': adv_res['pv_area_sqm_est'],
                    'delta_sqm': area_delta
                })
    
    # Calculate aggregate statistics
    baseline_confs = [r['confidence'] for r in baseline_results if r['confidence'] > 0]
    advanced_confs = [r['confidence'] for r in advanced_results if r['confidence'] > 0]
    
    stats = {
        'baseline': {
            'total_detected': sum(1 for r in baseline_results if r['has_solar']),
            'detection_rate': sum(1 for r in baseline_results if r['has_solar']) / len(baseline_results),
            'avg_confidence': np.mean(baseline_confs) if baseline_confs else 0,
            'median_confidence': np.median(baseline_confs) if baseline_confs else 0,
            'max_confidence': np.max(baseline_confs) if baseline_confs else 0,
            'avg_area': np.mean([r['pv_area_sqm_est'] for r in baseline_results if r['pv_area_sqm_est'] > 0]) if any(r['pv_area_sqm_est'] for r in baseline_results) else 0
        },
        'advanced': {
            'total_detected': sum(1 for r in advanced_results if r['has_solar']),
            'detection_rate': sum(1 for r in advanced_results if r['has_solar']) / len(advanced_results),
            'avg_confidence': np.mean(advanced_confs) if advanced_confs else 0,
            'median_confidence': np.median(advanced_confs) if advanced_confs else 0,
            'max_confidence': np.max(advanced_confs) if advanced_confs else 0,
            'avg_area': np.mean([r['pv_area_sqm_est'] for r in advanced_results if r['pv_area_sqm_est'] > 0]) if any(r['pv_area_sqm_est'] for r in advanced_results) else 0
        }
    }
    
    # Calculate deltas
    stats['improvements'] = {
        'confidence_boost_avg': stats['advanced']['avg_confidence'] - stats['baseline']['avg_confidence'],
        'detection_rate_delta': stats['advanced']['detection_rate'] - stats['baseline']['detection_rate'],
        'new_detections_count': len(improvements['new_detections']),
        'false_positives_count': len(improvements['false_positives']),
        'significant_boosts_count': len(improvements['confidence_boost'])
    }
    
    return {
        'comparison': comparison,
        'improvements': improvements,
        'stats': stats
    }

def print_comparison_report(analysis: Dict):
    """Print formatted comparison report"""
    
    stats = analysis['stats']
    improvements = analysis['improvements']
    
    print("\n" + "="*80)
    print("🔬 DETECTION METHOD COMPARISON: Baseline vs Advanced (Multi-Scale + TTA)")
    print("="*80)
    
    print("\n📊 BASELINE DETECTION (Standard YOLO)")
    print("-" * 80)
    print(f"  Total detected:        {stats['baseline']['total_detected']} / {len(analysis['comparison'])}")
    print(f"  Detection rate:        {stats['baseline']['detection_rate']*100:.1f}%")
    print(f"  Avg confidence:        {stats['baseline']['avg_confidence']:.4f}")
    print(f"  Median confidence:     {stats['baseline']['median_confidence']:.4f}")
    print(f"  Max confidence:        {stats['baseline']['max_confidence']:.4f}")
    print(f"  Avg area (m²):         {stats['baseline']['avg_area']:.2f}")
    
    print("\n🚀 ADVANCED DETECTION (Multi-Scale + TTA)")
    print("-" * 80)
    print(f"  Total detected:        {stats['advanced']['total_detected']} / {len(analysis['comparison'])}")
    print(f"  Detection rate:        {stats['advanced']['detection_rate']*100:.1f}%")
    print(f"  Avg confidence:        {stats['advanced']['avg_confidence']:.4f}")
    print(f"  Median confidence:     {stats['advanced']['median_confidence']:.4f}")
    print(f"  Max confidence:        {stats['advanced']['max_confidence']:.4f}")
    print(f"  Avg area (m²):         {stats['advanced']['avg_area']:.2f}")
    
    print("\n✨ IMPROVEMENTS SUMMARY")
    print("-" * 80)
    print(f"  Confidence boost:      {improvements['improvements']['confidence_boost_avg']:+.4f} ({improvements['improvements']['confidence_boost_avg']/stats['baseline']['avg_confidence']*100:+.1f}%)")
    print(f"  Detection rate delta:  {improvements['improvements']['detection_rate_delta']:+.1f}%")
    print(f"  New detections:        {improvements['improvements']['new_detections_count']}")
    print(f"  False positives:       {improvements['improvements']['false_positives_count']}")
    print(f"  Significant boosts:    {improvements['improvements']['significant_boosts_count']}")
    
    if improvements['improvements']['new_detections_count'] > 0:
        print("\n🎯 NEW DETECTIONS (Advanced found, Baseline missed):")
        for det in improvements['improvements']['new_detections'][:5]:
            print(f"    Sample {det['sample_id']}: conf={det['advanced_conf']:.3f}, area={det['area_sqm']:.2f}m², votes={det['ensemble_votes']}")
        if len(improvements['improvements']['new_detections']) > 5:
            print(f"    ... and {len(improvements['improvements']['new_detections']) - 5} more")
    
    if improvements['improvements']['significant_boosts_count'] > 0:
        print("\n⬆️  SIGNIFICANT CONFIDENCE BOOSTS (>0.05):")
        for boost in sorted(improvements['improvements']['confidence_boost'], key=lambda x: x['boost'], reverse=True)[:5]:
            print(f"    Sample {boost['sample_id']}: {boost['baseline_conf']:.3f} → {boost['advanced_conf']:.3f} (+{boost['boost']:.3f}, {boost['pct_improvement']:.1f}%)")
        if len(improvements['improvements']['confidence_boost']) > 5:
            print(f"    ... and {len(improvements['improvements']['confidence_boost']) - 5} more")
    
    if improvements['improvements']['false_positives_count'] > 0:
        print(f"\n⚠️  FALSE POSITIVES CORRECTED: {improvements['improvements']['false_positives_count']}")
        for fp in improvements['improvements']['false_positives'][:3]:
            print(f"    Sample {fp['sample_id']}: baseline={fp['baseline_conf']:.3f}")
    
    print("\n" + "="*80)

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_detection_methods.py <baseline_json> <advanced_json>")
        print("Example: python compare_detection_methods.py pipeline_results/predictions.json pipeline_results_advanced/predictions.json")
        sys.exit(1)
    
    baseline_json = sys.argv[1]
    advanced_json = sys.argv[2]
    
    if not Path(baseline_json).exists():
        print(f"❌ Baseline results not found: {baseline_json}")
        sys.exit(1)
    
    if not Path(advanced_json).exists():
        print(f"❌ Advanced results not found: {advanced_json}")
        sys.exit(1)
    
    print(f"📂 Loading results...")
    print(f"  Baseline: {baseline_json}")
    print(f"  Advanced: {advanced_json}")
    
    analysis = compare_results(baseline_json, advanced_json)
    print_comparison_report(analysis)
    
    # Save detailed report
    report_path = Path(advanced_json).parent / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n✓ Detailed report saved to {report_path}")

if __name__ == "__main__":
    main()
