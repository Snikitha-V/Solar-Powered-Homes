# ============================================
# pipeline_code/quantify_area.py
# ============================================

import numpy as np
from typing import List, Dict
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import cv2

def calculate_panel_area(detections: List[Dict], buffer_mask: np.ndarray, 
                        meters_per_pixel: float, buffer_type: str) -> float:
    """
    Calculate total PV area (m²) with largest overlap in selected buffer zone
    
    Args:
        detections: List of detection dictionaries with masks
        buffer_mask: Buffer zone mask (primary or secondary)
        meters_per_pixel: Ground sampling distance
        buffer_type: 'primary' or 'secondary'
    
    Returns:
        Total area in square meters
    """
    total_area_sqm = 0.0
    
    # Use all detections (not just those in buffer zones)
    # Since we already filtered by has_solar flag in detection stage
    relevant_detections = detections if detections else []
    
    if not relevant_detections:
        return 0.0
    
    # Collect all masks
    all_masks = []
    for detection in relevant_detections:
        if detection.get('mask') is not None:
            mask = detection['mask']
            
            # Apply buffer mask if available
            if buffer_mask and buffer_type in buffer_mask:
                masked_area = cv2.bitwise_and(mask, mask, mask=buffer_mask[buffer_type])
            else:
                masked_area = mask
            
            # Calculate area in pixels
            area_pixels = np.sum(masked_area > 0)
            
            # Convert to square meters
            area_sqm = area_pixels * (meters_per_pixel ** 2)
            all_masks.append(masked_area)
            total_area_sqm += area_sqm
        else:
            # Fallback: use bounding box
            x1, y1, x2, y2 = detection['bbox']
            width_m = (x2 - x1) * meters_per_pixel
            height_m = (y2 - y1) * meters_per_pixel
            total_area_sqm += width_m * height_m * 0.85  # 85% fill factor
    
    return total_area_sqm

def encode_polygon_mask(mask: np.ndarray) -> str:
    """Encode binary mask as polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return ""
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of coordinates
    coords = approx_polygon.reshape(-1, 2).tolist()
    
    # Return as string
    return str(coords)

def calculate_power_output(area_sqm: float, efficiency: float = 0.18, 
                          insolation_kwh_per_sqm_per_day: float = 5.0) -> Dict:
    """
    Calculate solar panel power output and energy generation
    
    Args:
        area_sqm: Total panel area in square meters
        efficiency: Panel efficiency (default 18% - typical modern panels)
        insolation_kwh_per_sqm_per_day: Daily solar insolation (default 5 kWh/m²/day - US average)
    
    Returns:
        Dictionary with power metrics
    """
    if area_sqm <= 0:
        return {
            'installed_capacity_kw': 0.0,
            'daily_energy_kwh': 0.0,
            'monthly_energy_kwh': 0.0,
            'annual_energy_kwh': 0.0,
            'co2_offset_kg_per_year': 0.0
        }
    
    # Installed capacity (kW) = Area (m²) × Efficiency × 1 kW/m² (standard test conditions)
    installed_capacity_kw = area_sqm * efficiency / 1000
    
    # Daily energy generation (kWh) = Capacity (kW) × Daily Insolation (kWh/m²/day)
    daily_energy_kwh = area_sqm * efficiency * insolation_kwh_per_sqm_per_day / 1000
    
    # Monthly and annual
    monthly_energy_kwh = daily_energy_kwh * 30
    annual_energy_kwh = daily_energy_kwh * 365
    
    # CO2 offset (kg/year) - average US grid: 0.92 kg CO2 per kWh
    co2_offset_kg_per_year = annual_energy_kwh * 0.92
    
    return {
        'installed_capacity_kw': round(installed_capacity_kw, 3),
        'daily_energy_kwh': round(daily_energy_kwh, 2),
        'monthly_energy_kwh': round(monthly_energy_kwh, 2),
        'annual_energy_kwh': round(annual_energy_kwh, 2),
        'co2_offset_kg_per_year': round(co2_offset_kg_per_year, 1)
    }
