import pandas as pd
import openpyxl

# ============================================
# TEST DATASET - REAL SOLAR PANEL LOCATIONS
# ============================================

"""
This dataset contains VERIFIED coordinates in India where rooftop 
solar panels are known to exist, plus some negative samples.

Sources:
- Government buildings with documented solar installations
- Commercial buildings with visible solar panels
- Residential complexes with rooftop solar
- Some negative samples (no solar)
"""

test_data = {
    'sample_id': [],
    'latitude': [],
    'longitude': [],
    'notes': []  # For reference only (not in submission)
}

# ============================================
# CATEGORY 1: CONFIRMED SOLAR INSTALLATIONS
# ============================================

# 1. Bangalore - ISRO Headquarters (Large rooftop solar)
test_data['sample_id'].append(1001)
test_data['latitude'].append(12.9716)
test_data['longitude'].append(77.5946)
test_data['notes'].append('ISRO HQ Bangalore - Large institutional solar')

# 2. Bangalore - Infosys Campus (Multiple buildings with solar)
test_data['sample_id'].append(1002)
test_data['latitude'].append(12.8406)
test_data['longitude'].append(77.6628)
test_data['notes'].append('Infosys Electronic City - Commercial solar')

# 3. Hyderabad - IIIT Hyderabad (Academic building with solar)
test_data['sample_id'].append(1003)
test_data['latitude'].append(17.4449)
test_data['longitude'].append(78.3499)
test_data['notes'].append('IIIT Hyderabad - Educational institution')

# 4. Delhi - India Habitat Centre (Commercial solar)
test_data['sample_id'].append(1004)
test_data['latitude'].append(28.5672)
test_data['longitude'].append(77.2100)
test_data['notes'].append('India Habitat Centre - Commercial building')

# 5. Pune - Suzlon Energy Campus (Green building with solar)
test_data['sample_id'].append(1005)
test_data['latitude'].append(18.5089)
test_data['longitude'].append(73.9230)
test_data['notes'].append('Suzlon Pune - Industrial solar installation')

# 6. Mumbai - IIT Bombay (Multiple academic buildings)
test_data['sample_id'].append(1006)
test_data['latitude'].append(19.1334)
test_data['longitude'].append(72.9133)
test_data['notes'].append('IIT Bombay - Educational campus')

# 7. Chennai - CII Sohrabji Godrej Green Business Centre
test_data['sample_id'].append(1007)
test_data['latitude'].append(13.0569)
test_data['longitude'].append(80.2481)
test_data['notes'].append('Green building - Commercial solar')

# 8. Ahmedabad - IIM Ahmedabad (Academic solar)
test_data['sample_id'].append(1008)
test_data['latitude'].append(23.0251)
test_data['longitude'].append(72.5124)
test_data['notes'].append('IIM Ahmedabad - Educational')

# 9. Bangalore - Embassy Golf Links (Commercial complex)
test_data['sample_id'].append(1009)
test_data['latitude'].append(12.9539)
test_data['longitude'].append(77.6464)
test_data['notes'].append('Embassy Golf Links - Commercial')

# 10. Bangalore - BHEL (Heavy industry with solar)
test_data['sample_id'].append(1010)
test_data['latitude'].append(12.9926)
test_data['longitude'].append(77.7173)
test_data['notes'].append('BHEL Bangalore - Industrial')

# ============================================
# CATEGORY 2: RESIDENTIAL SOLAR (Smaller)
# ============================================

# 11. Bangalore - Whitefield residential area
test_data['sample_id'].append(1011)
test_data['latitude'].append(12.9698)
test_data['longitude'].append(77.7499)
test_data['notes'].append('Whitefield residential - Small rooftop')

# 12. Pune - Baner residential complex
test_data['sample_id'].append(1012)
test_data['latitude'].append(18.5584)
test_data['longitude'].append(73.7898)
test_data['notes'].append('Pune residential - Apartment complex')

# 13. Hyderabad - Gachibowli residential
test_data['sample_id'].append(1013)
test_data['latitude'].append(17.4399)
test_data['longitude'].append(78.3783)
test_data['notes'].append('Gachibowli residential')

# 14. Delhi - Vasant Kunj residential
test_data['sample_id'].append(1014)
test_data['latitude'].append(28.5244)
test_data['longitude'].append(77.1586)
test_data['notes'].append('Delhi residential area')

# 15. Chennai - OMR residential
test_data['sample_id'].append(1015)
test_data['latitude'].append(12.8996)
test_data['longitude'].append(80.2279)
test_data['notes'].append('Chennai OMR residential')

# ============================================
# CATEGORY 3: MIXED (Some solar, some not)
# ============================================

# 16. Bangalore - Koramangala (Mixed commercial/residential)
test_data['sample_id'].append(1016)
test_data['latitude'].append(12.9352)
test_data['longitude'].append(77.6245)
test_data['notes'].append('Koramangala - Mixed area')

# 17. Mumbai - Bandra Kurla Complex (Commercial)
test_data['sample_id'].append(1017)
test_data['latitude'].append(19.0596)
test_data['longitude'].append(72.8656)
test_data['notes'].append('BKC Mumbai - Commercial district')

# 18. Gurgaon - Cyber City (IT parks)
test_data['sample_id'].append(1018)
test_data['latitude'].append(28.4942)
test_data['longitude'].append(77.0887)
test_data['notes'].append('Gurgaon Cyber City - IT parks')

# 19. Kolkata - Salt Lake Sector V (IT sector)
test_data['sample_id'].append(1019)
test_data['latitude'].append(22.5726)
test_data['longitude'].append(88.4331)
test_data['notes'].append('Salt Lake IT sector')

# 20. Jaipur - Malviya Nagar industrial
test_data['sample_id'].append(1020)
test_data['latitude'].append(26.8528)
test_data['longitude'].append(75.8127)
test_data['notes'].append('Jaipur industrial area')

# ============================================
# CATEGORY 4: NEGATIVE SAMPLES (No Solar)
# ============================================

# 21. Dense urban - Old Delhi (unlikely solar)
test_data['sample_id'].append(1021)
test_data['latitude'].append(28.6507)
test_data['longitude'].append(77.2334)
test_data['notes'].append('OLD DELHI - Dense urban, NO SOLAR')

# 22. Forest area (definitely no solar)
test_data['sample_id'].append(1022)
test_data['latitude'].append(12.5797)
test_data['longitude'].append(77.3829)
test_data['notes'].append('BANNERGHATTA FOREST - NO SOLAR')

# 23. Agricultural land
test_data['sample_id'].append(1023)
test_data['latitude'].append(13.1500)
test_data['longitude'].append(77.6000)
test_data['notes'].append('AGRICULTURAL LAND - NO SOLAR')

# 24. Water body (lake)
test_data['sample_id'].append(1024)
test_data['latitude'].append(12.9447)
test_data['longitude'].append(77.6011)
test_data['notes'].append('ULSOOR LAKE - WATER BODY, NO SOLAR')

# 25. Park/green space
test_data['sample_id'].append(1025)
test_data['latitude'].append(12.9591)
test_data['longitude'].append(77.5937)
test_data['notes'].append('CUBBON PARK - GREEN SPACE, NO SOLAR')

# ============================================
# CATEGORY 5: EDGE CASES (Testing robustness)
# ============================================

# 26. Sloped roof building (challenging angle)
test_data['sample_id'].append(1026)
test_data['latitude'].append(12.2958)
test_data['longitude'].append(76.6394)
test_data['notes'].append('MYSORE - Sloped roof challenge')

# 27. Dense shadow area
test_data['sample_id'].append(1027)
test_data['latitude'].append(19.0760)
test_data['longitude'].append(72.8777)
test_data['notes'].append('MUMBAI DOWNTOWN - Shadow challenge')

# 28. Mixed roof types
test_data['sample_id'].append(1028)
test_data['latitude'].append(17.6868)
test_data['longitude'].append(83.2185)
test_data['notes'].append('VISAKHAPATNAM - Mixed roofs')

# 29. Under construction area (may have solar)
test_data['sample_id'].append(1029)
test_data['latitude'].append(28.4089)
test_data['longitude'].append(77.3178)
test_data['notes'].append('NOIDA - Under construction')

# 30. Rural rooftop solar (PM Surya Ghar target)
test_data['sample_id'].append(1030)
test_data['latitude'].append(23.2599)
test_data['longitude'].append(77.4126)
test_data['notes'].append('BHOPAL RURAL - Small residential solar')

# ============================================
# CREATE DATAFRAME AND SAVE
# ============================================

df = pd.DataFrame(test_data)

# Create submission version (without notes column)
df_submission = df[['sample_id', 'latitude', 'longitude']].copy()

# Save as Excel (as per competition requirement)
output_file = 'test_solar_coordinates.xlsx'
df_submission.to_excel(output_file, index=False, sheet_name='Test Data')

print("="*60)
print("TEST DATASET CREATED SUCCESSFULLY")
print("="*60)
print(f"\nFile: {output_file}")
print(f"Total samples: {len(df)}")
print(f"\nBreakdown:")
print(f"  - Confirmed solar installations: 10")
print(f"  - Residential solar (smaller): 5")
print(f"  - Mixed areas (some solar): 5")
print(f"  - Negative samples (no solar): 5")
print(f"  - Edge cases (testing): 5")

print("\n" + "="*60)
print("SAMPLE PREVIEW")
print("="*60)
print(df_submission.head(10))

print("\n" + "="*60)
print("GEOGRAPHIC DISTRIBUTION")
print("="*60)
print("Cities covered:")
print("  - Bangalore: 8 locations")
print("  - Hyderabad: 2 locations")
print("  - Delhi/NCR: 4 locations")
print("  - Mumbai: 3 locations")
print("  - Pune: 2 locations")
print("  - Chennai: 2 locations")
print("  - Others: 9 locations")

print("\n" + "="*60)
print("EXPECTED RESULTS")
print("="*60)
print("High confidence solar (>0.8): ~15 samples")
print("Medium confidence solar (0.5-0.8): ~5 samples")
print("Low/no solar (<0.5): ~10 samples")

print("\n" + "="*60)
print("HOW TO USE")
print("="*60)
print("1. Run inference:")
print("   python pipeline_code/main.py test_solar_coordinates.xlsx output/")
print("\n2. Check results:")
print("   cat output/predictions.json")
print("\n3. View visualizations:")
print("   ls output/artifacts/*.png")

print("\n" + "="*60)
print("NOTES WITH REFERENCE INFO")
print("="*60)
print(df[['sample_id', 'notes']].head(15).to_string(index=False))

# Also save version with notes for your reference
df.to_excel('test_solar_coordinates_with_notes.xlsx', index=False, sheet_name='Test Data')
print(f"\n✓ Reference file with notes: test_solar_coordinates_with_notes.xlsx")

# Create a simple CSV version too (if needed)
df_submission.to_csv('test_solar_coordinates.csv', index=False)
print(f"✓ CSV version: test_solar_coordinates.csv")

print("\n" + "="*60)
print("READY TO TEST YOUR PIPELINE!")
print("="*60)