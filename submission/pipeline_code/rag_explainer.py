# ============================================
# pipeline_code/rag_explainer.py - HyDE RAG Explainer
# ============================================
"""
HyDE (Hypothetical Document Embeddings) RAG module for explaining solar panel detections.

HyDE Process:
1. Generate a hypothetical explanation based on detection data
2. Use that to retrieve relevant knowledge from the solar knowledge base
3. Synthesize a final, grounded explanation combining detection data + retrieved knowledge
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re

# ============================================
# SOLAR KNOWLEDGE BASE
# ============================================
# Curated knowledge about solar panels, detection, and PM Surya Ghar program

SOLAR_KNOWLEDGE_BASE = [
    # Detection confidence knowledge
    {
        "topic": "high_confidence_detection",
        "keywords": ["high confidence", "confident", "clear", "obvious", "strong detection"],
        "content": """High confidence detections (>0.8) indicate the model found clear visual signatures 
of solar panels: distinct rectangular shapes, characteristic blue/black coloring, regular grid patterns, 
and reflective surfaces typical of photovoltaic modules. These detections are highly reliable for 
verification under the PM Surya Ghar scheme."""
    },
    {
        "topic": "medium_confidence_detection", 
        "keywords": ["medium confidence", "moderate", "partial", "possible"],
        "content": """Medium confidence detections (0.5-0.8) suggest probable solar panel presence 
but with some uncertainty. This may occur due to partial occlusion, shadows, image quality issues, 
or panels that differ from typical appearances. Manual verification is recommended for subsidy approval."""
    },
    {
        "topic": "low_confidence_detection",
        "keywords": ["low confidence", "uncertain", "weak", "marginal"],
        "content": """Low confidence detections (<0.5) indicate possible but uncertain solar panel presence. 
The model detected features that could be solar panels but lack definitive characteristics. This may be 
due to unusual panel types, poor image quality, extreme angles, or false positives from similar-looking 
structures like skylights or water tanks."""
    },
    # Panel area knowledge
    {
        "topic": "small_installation",
        "keywords": ["small area", "small installation", "1kw", "2kw", "residential small"],
        "content": """Small installations (1-5 m²) typically represent 1-2 kW residential systems, 
suitable for basic household needs. Under PM Surya Ghar, these qualify for the highest per-kW subsidy 
tier and can offset 30-40% of average household electricity consumption."""
    },
    {
        "topic": "medium_installation",
        "keywords": ["medium area", "medium installation", "3kw", "4kw", "5kw"],
        "content": """Medium installations (5-15 m²) represent 3-5 kW systems ideal for typical Indian 
households. These systems can generate 4,000-7,000 kWh annually, significantly reducing electricity bills 
and carbon footprint. PM Surya Ghar provides substantial subsidies for this category."""
    },
    {
        "topic": "large_installation",
        "keywords": ["large area", "large installation", "commercial", "10kw", "multiple panels"],
        "content": """Large installations (>15 m²) indicate commercial-scale or premium residential systems 
of 10+ kW. These multi-panel arrays can achieve near-complete energy independence and may qualify for 
net metering benefits, selling excess power back to the grid."""
    },
    # Buffer zone knowledge
    {
        "topic": "primary_buffer",
        "keywords": ["primary buffer", "1200 sqft", "within buffer", "core zone"],
        "content": """Detections within the primary buffer zone (1200 sq ft radius from coordinates) 
are highly likely to be on the target property. This zone represents the core rooftop area where 
solar panels would be installed for the claimed residence."""
    },
    {
        "topic": "secondary_buffer",
        "keywords": ["secondary buffer", "2400 sqft", "extended zone", "neighboring"],
        "content": """Detections in the secondary buffer zone (2400 sq ft radius) may include 
neighboring properties. While solar presence is confirmed, additional verification may be needed 
to ensure panels belong to the claimed address rather than adjacent buildings."""
    },
    # Technical quality
    {
        "topic": "sam2_refinement",
        "keywords": ["sam2", "segmentation", "precise mask", "refined boundary"],
        "content": """SAM2 (Segment Anything Model 2) refinement provides pixel-perfect panel boundaries, 
enabling accurate area estimation. This advanced segmentation ensures measurement precision 
within ±5% for subsidy calculations and capacity verification."""
    },
    {
        "topic": "multi_detection",
        "keywords": ["multiple detections", "ensemble", "multiple panels", "aggregated"],
        "content": """Multiple detection points indicate either a multi-panel array or high-confidence 
ensemble agreement. When multiple model passes detect the same region, confidence is boosted through 
ensemble voting, indicating robust and reliable solar presence verification."""
    },
    # PM Surya Ghar specific
    {
        "topic": "pm_surya_ghar_program",
        "keywords": ["pm surya ghar", "subsidy", "government scheme", "verification"],
        "content": """PM Surya Ghar Muft Bijli Yojana aims to provide free electricity to 1 crore 
households through rooftop solar. The scheme offers subsidies up to ₹78,000 for installations 
up to 3 kW. This verification system uses AI to validate installations for subsidy approval, 
ensuring transparency and reducing fraud in the disbursement process."""
    },
    {
        "topic": "no_solar_detected",
        "keywords": ["no solar", "not detected", "negative", "no panels found"],
        "content": """When no solar panels are detected, it may indicate: (1) No installation present, 
(2) Panels obscured by trees or shadows, (3) Unusual panel types not recognized, (4) Image quality 
issues, or (5) Very new installations not yet in satellite imagery. Site visit may be recommended 
for claimed installations with negative detection results."""
    },
    # Environmental impact
    {
        "topic": "environmental_benefits",
        "keywords": ["co2", "carbon", "environmental", "green energy", "offset"],
        "content": """Each kW of rooftop solar prevents approximately 1,500 kg of CO₂ emissions annually. 
A typical 3 kW residential system saves about 4.5 tonnes of CO₂ per year, equivalent to planting 
75 trees or removing one car from the road. This aligns with India's climate commitments."""
    },
    {
        "topic": "energy_generation",
        "keywords": ["energy", "kwh", "electricity", "generation", "annual output"],
        "content": """In India's solar-rich climate, rooftop panels generate 4-5 kWh per kW installed 
daily, or approximately 1,500 kWh/kW annually. Peak generation occurs during summer months. 
Net metering allows excess generation to be exported to the grid for credits."""
    }
]


@dataclass
class ExplanationResult:
    """Structured explanation output"""
    summary: str  # One-line summary
    detailed_explanation: str  # Full paragraph explanation
    key_factors: List[str]  # Bullet points of key factors
    confidence_reasoning: str  # Why this confidence level
    area_interpretation: str  # What the area means
    recommendation: str  # Verification recommendation
    qc_explanation: str  # Why VERIFIABLE or NOT_VERIFIABLE
    retrieved_knowledge: List[str]  # Knowledge sources used


class HyDERAGExplainer:
    """
    HyDE (Hypothetical Document Embeddings) RAG Explainer for solar detections.
    
    Uses a lightweight in-memory approach suitable for hackathon:
    1. Generate hypothetical explanation from detection data
    2. Keyword-based retrieval from knowledge base (fast, no embeddings needed)
    3. Template-based synthesis with retrieved knowledge
    """
    
    def __init__(self, knowledge_base: List[Dict] = None):
        self.knowledge_base = knowledge_base or SOLAR_KNOWLEDGE_BASE
        
    def _generate_hypothesis(self, detection_data: Dict) -> str:
        """
        HyDE Step 1: Generate a hypothetical explanation based on detection data.
        This "imagined" answer guides retrieval.
        """
        has_solar = detection_data.get('has_solar', False)
        confidence = detection_data.get('confidence', 0.0)
        area = detection_data.get('pv_area_sqm_est', 0.0)
        num_detections = len(detection_data.get('detections', []))
        
        if not has_solar:
            return "No solar panels detected in the satellite image. The property may not have rooftop solar installed, or panels may be obscured or unrecognizable."
        
        # Build hypothesis based on data
        conf_level = "high confidence" if confidence > 0.8 else "medium confidence" if confidence > 0.5 else "low confidence"
        size_desc = "large installation" if area > 15 else "medium installation" if area > 5 else "small installation"
        
        hypothesis = f"This is a {conf_level} detection of a {size_desc} with {area:.1f} square meters of solar panels. "
        
        if num_detections > 1:
            hypothesis += f"Multiple detections ({num_detections}) indicate ensemble agreement and robust verification. "
        
        if confidence > 0.9:
            hypothesis += "Clear visual signatures confirm solar panel presence for PM Surya Ghar subsidy verification."
        elif confidence > 0.6:
            hypothesis += "Detection shows probable solar installation suitable for verification with standard review."
        else:
            hypothesis += "Marginal detection may require additional verification or site visit."
            
        return hypothesis
    
    def _retrieve_relevant_knowledge(self, hypothesis: str, detection_data: Dict, top_k: int = 4) -> List[Dict]:
        """
        HyDE Step 2: Retrieve relevant knowledge using hypothesis + detection data as query.
        Uses keyword matching for speed (embeddings would be better but slower).
        """
        # Extract query terms from hypothesis and detection data
        query_text = hypothesis.lower()
        
        # Add detection-specific keywords
        has_solar = detection_data.get('has_solar', False)
        confidence = detection_data.get('confidence', 0.0)
        area = detection_data.get('pv_area_sqm_est', 0.0)
        
        if not has_solar:
            query_text += " no solar not detected negative"
        elif confidence > 0.8:
            query_text += " high confidence confident clear strong"
        elif confidence > 0.5:
            query_text += " medium confidence moderate partial"
        else:
            query_text += " low confidence uncertain weak marginal"
            
        if area > 15:
            query_text += " large installation commercial"
        elif area > 5:
            query_text += " medium installation 3kw 4kw 5kw"
        elif area > 0:
            query_text += " small installation 1kw 2kw residential"
            
        # Check buffer info
        detections = detection_data.get('detections', [])
        if any(d.get('in_primary_buffer') for d in detections):
            query_text += " primary buffer core zone"
        elif any(d.get('in_secondary_buffer') for d in detections):
            query_text += " secondary buffer extended"
            
        # Score each knowledge item
        scored_items = []
        for item in self.knowledge_base:
            score = 0
            keywords = item.get('keywords', [])
            for kw in keywords:
                if kw.lower() in query_text:
                    score += 2
            # Also check topic match
            if item['topic'].replace('_', ' ') in query_text:
                score += 3
            if score > 0:
                scored_items.append((score, item))
        
        # Sort by score and take top_k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:top_k]]
    
    def _synthesize_explanation(self, detection_data: Dict, hypothesis: str, 
                                 retrieved_knowledge: List[Dict]) -> ExplanationResult:
        """
        HyDE Step 3: Synthesize final explanation from detection data + retrieved knowledge.
        """
        has_solar = detection_data.get('has_solar', False)
        confidence = detection_data.get('confidence', 0.0)
        area = detection_data.get('pv_area_sqm_est', 0.0)
        sample_id = detection_data.get('sample_id', 'N/A')
        num_detections = len(detection_data.get('detections', []))
        annual_energy = detection_data.get('annual_energy_kwh', 0.0)
        co2_offset = detection_data.get('co2_offset_kg_per_year', 0.0)
        
        # Extract knowledge content
        knowledge_texts = [k['content'] for k in retrieved_knowledge]
        knowledge_topics = [k['topic'] for k in retrieved_knowledge]
        
        # Build explanation components
        if not has_solar:
            summary = f"Sample {sample_id}: No solar panels detected"
            detailed = self._build_no_detection_explanation(detection_data, knowledge_texts)
            key_factors = [
                "No recognizable solar panel patterns found",
                "Confidence below detection threshold",
                "Property may not have solar installation or panels may be obscured"
            ]
            conf_reasoning = "The model did not detect any features consistent with solar panels in the imagery."
            area_interp = "No area measurement possible without detection."
            recommendation = "If solar installation is claimed, recommend site visit or updated imagery for verification."
        else:
            # Confidence-based summary
            conf_emoji = "✅" if confidence > 0.8 else "⚠️" if confidence > 0.5 else "❓"
            conf_word = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            summary = f"Sample {sample_id}: {conf_emoji} {conf_word} confidence solar detection ({confidence:.1%}), {area:.1f} m² estimated"
            
            detailed = self._build_detection_explanation(detection_data, knowledge_texts)
            
            key_factors = self._extract_key_factors(detection_data, retrieved_knowledge)
            conf_reasoning = self._build_confidence_reasoning(detection_data, retrieved_knowledge)
            area_interp = self._build_area_interpretation(area, annual_energy, co2_offset, knowledge_texts)
            recommendation = self._build_recommendation(detection_data)
        
        # Build QC explanation
        qc_explanation = self._build_qc_explanation(detection_data)
        
        return ExplanationResult(
            summary=summary,
            detailed_explanation=detailed,
            key_factors=key_factors,
            confidence_reasoning=conf_reasoning,
            area_interpretation=area_interp,
            recommendation=recommendation,
            qc_explanation=qc_explanation,
            retrieved_knowledge=knowledge_topics
        )
    
    def _build_no_detection_explanation(self, data: Dict, knowledge: List[str]) -> str:
        """Build explanation for no-detection case"""
        sample_id = data.get('sample_id', 'N/A')
        lat, lon = data.get('lat', 0), data.get('lon', 0)
        
        explanation = f"""Analysis of satellite imagery at coordinates ({lat:.5f}, {lon:.5f}) 
for Sample {sample_id} did not identify any solar panel installations.

The AI detection system scanned the rooftop area within the verification buffer zone 
but found no visual patterns consistent with photovoltaic panels. This could indicate:

1. No solar installation present at this location
2. Panels obscured by trees, shadows, or other structures
3. Recent installation not yet captured in satellite imagery
4. Unconventional panel types not recognized by the model

"""
        # Add relevant knowledge
        for k in knowledge[:2]:
            if "no solar" in k.lower() or "not detected" in k.lower():
                explanation += f"\n{k}\n"
                break
                
        return explanation.strip()
    
    def _build_detection_explanation(self, data: Dict, knowledge: List[str]) -> str:
        """Build detailed explanation for positive detection"""
        sample_id = data.get('sample_id', 'N/A')
        lat, lon = data.get('lat', 0), data.get('lon', 0)
        confidence = data.get('confidence', 0.0)
        area = data.get('pv_area_sqm_est', 0.0)
        num_dets = len(data.get('detections', []))
        annual_energy = data.get('annual_energy_kwh', 0.0)
        co2_offset = data.get('co2_offset_kg_per_year', 0.0)
        
        # Determine installation size category
        if area > 15:
            size_cat = "large-scale"
            capacity_est = "10+ kW commercial-grade"
        elif area > 5:
            size_cat = "medium-sized"
            capacity_est = "3-5 kW residential"
        else:
            size_cat = "small"
            capacity_est = "1-2 kW starter"
            
        explanation = f"""**Solar Panel Detection Report - Sample {sample_id}**

📍 Location: ({lat:.5f}, {lon:.5f})
🔍 Detection Confidence: {confidence:.1%}
📐 Estimated Panel Area: {area:.2f} m²
🔢 Detection Points: {num_dets}

**Analysis Summary:**
The AI verification system identified a {size_cat} solar installation with 
{confidence:.1%} confidence. The estimated {area:.2f} square meters of panel 
coverage suggests a {capacity_est} system.

"""
        
        if num_dets > 1:
            explanation += f"""**Multi-Detection Validation:**
{num_dets} separate detection points were identified and aggregated, providing 
robust ensemble confidence. Multiple detections indicate clear, distinguishable 
solar panel features in the imagery.

"""

        if annual_energy > 0:
            explanation += f"""**Estimated Impact:**
• Annual Energy Generation: ~{annual_energy:,.0f} kWh/year
• CO₂ Offset: ~{co2_offset:,.0f} kg/year ({co2_offset/1000:.2f} tonnes)
• Equivalent to {int(co2_offset/60)} trees planted or {co2_offset/4500:.1f} cars off road

"""

        # Add most relevant knowledge
        for k in knowledge[:2]:
            if len(k) > 50:  # Skip very short entries
                explanation += f"{k}\n\n"
                
        return explanation.strip()
    
    def _extract_key_factors(self, data: Dict, knowledge: List[Dict]) -> List[str]:
        """Extract key factors as bullet points"""
        factors = []
        confidence = data.get('confidence', 0.0)
        area = data.get('pv_area_sqm_est', 0.0)
        num_dets = len(data.get('detections', []))
        
        # Confidence factor
        if confidence > 0.9:
            factors.append("Very high model confidence (>90%) - clear solar signatures detected")
        elif confidence > 0.7:
            factors.append("Strong model confidence (>70%) - reliable detection")
        elif confidence > 0.5:
            factors.append("Moderate confidence (50-70%) - probable solar presence")
        else:
            factors.append("Lower confidence (<50%) - possible but uncertain detection")
            
        # Area factor
        if area > 10:
            factors.append(f"Significant panel coverage ({area:.1f} m²) - substantial installation")
        elif area > 3:
            factors.append(f"Moderate panel coverage ({area:.1f} m²) - typical residential system")
        else:
            factors.append(f"Small panel coverage ({area:.1f} m²) - starter or partial installation")
            
        # Detection count
        if num_dets > 3:
            factors.append(f"Multiple detection points ({num_dets}) - high ensemble agreement")
        elif num_dets > 1:
            factors.append(f"Multiple detections ({num_dets}) - corroborated finding")
        else:
            factors.append("Single detection point - focused panel region")
            
        # Buffer zone
        detections = data.get('detections', [])
        if any(d.get('in_primary_buffer') for d in detections):
            factors.append("Detection within primary buffer zone (1200 sqft) - on-property")
        elif any(d.get('in_secondary_buffer') for d in detections):
            factors.append("Detection in secondary buffer zone - verify property boundary")
            
        return factors
    
    def _build_confidence_reasoning(self, data: Dict, knowledge: List[Dict]) -> str:
        """Explain why this confidence level"""
        confidence = data.get('confidence', 0.0)
        num_dets = len(data.get('detections', []))
        
        reasoning = f"The {confidence:.1%} confidence score reflects "
        
        if confidence > 0.9:
            reasoning += "exceptional visual clarity - the model detected definitive solar panel patterns including rectangular shapes, characteristic coloring, and regular grid structures. "
        elif confidence > 0.7:
            reasoning += "strong visual indicators - clear panel boundaries and typical photovoltaic appearance were identified. "
        elif confidence > 0.5:
            reasoning += "probable solar features - the detection shows characteristics consistent with solar panels but with some uncertainty due to image quality or partial visibility. "
        else:
            reasoning += "marginal visual cues - some features suggest possible solar presence but lack definitive confirmation. "
            
        if num_dets > 1:
            reasoning += f"Confidence was boosted by ensemble agreement across {num_dets} detection points."
        
        return reasoning
    
    def _build_area_interpretation(self, area: float, annual_energy: float, 
                                    co2_offset: float, knowledge: List[str]) -> str:
        """Interpret what the detected area means"""
        if area > 15:
            capacity = area * 0.15  # ~150W per m²
            return f"""The detected {area:.1f} m² represents a large installation, likely a commercial 
or premium residential system of approximately {capacity:.1f} kW capacity. This size typically 
generates {annual_energy:,.0f} kWh annually and offsets {co2_offset:,.0f} kg of CO₂, equivalent 
to significant environmental impact. Under PM Surya Ghar, installations of this size provide 
maximum energy independence benefits."""
        elif area > 5:
            capacity = area * 0.15
            return f"""The detected {area:.1f} m² suggests a medium-sized residential installation 
of approximately {capacity:.1f} kW. This is typical for Indian households seeking substantial 
electricity savings. Annual generation of ~{annual_energy:,.0f} kWh can offset 40-60% of 
typical household consumption, qualifying for PM Surya Ghar subsidies."""
        else:
            capacity = area * 0.15
            return f"""The detected {area:.1f} m² indicates a small or starter installation 
of approximately {capacity:.1f} kW. While modest, this system still contributes to renewable 
energy goals and qualifies for the highest per-kW subsidy tier under PM Surya Ghar. 
Expected annual generation: ~{annual_energy:,.0f} kWh."""
    
    def _build_recommendation(self, data: Dict) -> str:
        """Generate verification recommendation"""
        confidence = data.get('confidence', 0.0)
        qc_status = data.get('qc_status', 'NOT_VERIFIABLE')
        
        if confidence > 0.85 and qc_status == 'VERIFIABLE':
            return "✅ RECOMMENDED FOR APPROVAL: High-confidence detection with good image quality. Subsidy verification can proceed based on this analysis."
        elif confidence > 0.6:
            return "⚠️ STANDARD REVIEW: Moderate confidence detection. Recommend standard documentation review before subsidy approval."
        elif confidence > 0.3:
            return "📋 ENHANCED VERIFICATION: Lower confidence detection. Recommend additional documentation or site photo submission."
        else:
            return "🔍 SITE VISIT RECOMMENDED: Marginal detection confidence. Physical verification advised before subsidy disbursement."
    
    def _build_qc_explanation(self, data: Dict) -> str:
        """
        Explain why the sample is VERIFIABLE or NOT_VERIFIABLE.
        
        VERIFIABLE: Clear evidence either way (present/not present)
        NOT_VERIFIABLE: Insufficient evidence (low resolution, shadow, cloud, occlusion, stale imagery)
        """
        qc_status = data.get('qc_status', 'NOT_VERIFIABLE')
        qc_reason = data.get('qc_reason', '')
        qc_issues = data.get('qc_issues', [])
        has_solar = data.get('has_solar', False)
        confidence = data.get('confidence', 0.0)
        
        # Get image metadata for explanation
        metadata = data.get('image_metadata', {})
        resolution = metadata.get('resolution_m', 0)
        source = metadata.get('source', 'Unknown')
        
        if qc_status == 'VERIFIABLE':
            if has_solar:
                explanation = f"""✅ QC STATUS: VERIFIABLE

**Reason:** Clear evidence of solar panel presence detected.

**Evidence Quality:**
• Image Source: {source}
• Resolution: {resolution:.3f} m/pixel (sufficient for panel detection)
• Detection Confidence: {confidence:.1%}
• No quality issues detected (no clouds, shadows, or occlusions)

**Conclusion:** The satellite imagery provides clear, unambiguous evidence of rooftop 
solar panel installation. The detection confidence and image quality meet verification 
standards for PM Surya Ghar subsidy processing."""
            else:
                explanation = f"""✅ QC STATUS: VERIFIABLE

**Reason:** Clear evidence of NO solar panel presence.

**Evidence Quality:**
• Image Source: {source}
• Resolution: {resolution:.3f} m/pixel (sufficient for panel detection)
• Rooftop clearly visible with no solar panel signatures
• No quality issues detected (no clouds, shadows, or occlusions)

**Conclusion:** The satellite imagery provides clear visibility of the rooftop area. 
After comprehensive AI analysis, no solar panel patterns were detected within the 
buffer zones. This constitutes clear evidence that rooftop PV is NOT present at 
this location (or was not present when imagery was captured)."""
        else:
            # NOT_VERIFIABLE - explain why
            issues_text = ""
            if qc_issues:
                issues_text = "\n• ".join(qc_issues)
            
            explanation = f"""❌ QC STATUS: NOT_VERIFIABLE

**Reason:** Insufficient evidence to determine solar presence.

**Quality Issues Detected:**
• {issues_text if issues_text else 'Unknown quality issue'}

**Possible Causes:**
• Low resolution imagery (>{resolution:.3f} m/pixel may miss panel details)
• Heavy shadows obscuring rooftop
• Cloud coverage blocking view
• Tree canopy or other occlusion
• Stale imagery (installation may be newer than satellite capture)
• Processing error during analysis

**Recommendation:** This sample requires additional verification. Consider:
1. Requesting updated satellite imagery
2. On-site photo submission by applicant
3. Physical site visit by verification officer

**Note:** NOT_VERIFIABLE does not mean solar is absent - it means the evidence 
is insufficient to make a determination either way."""
        
        return explanation

    def explain(self, detection_data: Dict) -> ExplanationResult:
        """
        Main HyDE RAG explanation pipeline.
        
        Args:
            detection_data: Dictionary with detection results (from predictions.json)
            
        Returns:
            ExplanationResult with structured explanation
        """
        # Step 1: Generate hypothesis
        hypothesis = self._generate_hypothesis(detection_data)
        
        # Step 2: Retrieve relevant knowledge
        retrieved = self._retrieve_relevant_knowledge(hypothesis, detection_data, top_k=4)
        
        # Step 3: Synthesize explanation
        explanation = self._synthesize_explanation(detection_data, hypothesis, retrieved)
        
        return explanation
    
    def explain_batch(self, predictions: List[Dict]) -> List[ExplanationResult]:
        """Explain all predictions in a batch"""
        return [self.explain(pred) for pred in predictions]
    
    def to_dict(self, result: ExplanationResult) -> Dict:
        """Convert ExplanationResult to dictionary for JSON serialization"""
        return {
            'summary': result.summary,
            'detailed_explanation': result.detailed_explanation,
            'key_factors': result.key_factors,
            'confidence_reasoning': result.confidence_reasoning,
            'area_interpretation': result.area_interpretation,
            'recommendation': result.recommendation,
            'qc_explanation': result.qc_explanation,
            'knowledge_sources': result.retrieved_knowledge
        }


def explain_predictions_file(predictions_path: str, output_path: str = None) -> List[Dict]:
    """
    Convenience function to explain all predictions in a JSON file.
    
    Args:
        predictions_path: Path to predictions.json
        output_path: Optional path to save explained predictions
        
    Returns:
        List of predictions with explanations added
    """
    import json
    from pathlib import Path
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Initialize explainer
    explainer = HyDERAGExplainer()
    
    # Explain each prediction
    explained = []
    for pred in predictions:
        explanation = explainer.explain(pred)
        pred_with_explanation = pred.copy()
        pred_with_explanation['explanation'] = explainer.to_dict(explanation)
        explained.append(pred_with_explanation)
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(explained, f, indent=2)
        print(f"✓ Explained predictions saved to: {output_path}")
    
    return explained


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rag_explainer.py <predictions.json> [output.json]")
        print("\nExample:")
        print("  python rag_explainer.py pipeline_results/predictions.json pipeline_results/predictions_explained.json")
        sys.exit(1)
    
    predictions_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else predictions_path.replace('.json', '_explained.json')
    
    print("="*60)
    print("HyDE RAG EXPLAINER - Solar Detection Explanations")
    print("="*60)
    
    explained = explain_predictions_file(predictions_path, output_path)
    
    # Print sample explanations
    print(f"\n✓ Explained {len(explained)} predictions")
    print("\n" + "="*60)
    print("SAMPLE EXPLANATIONS:")
    print("="*60)
    
    for pred in explained[:3]:  # Show first 3
        exp = pred['explanation']
        print(f"\n{exp['summary']}")
        print("-" * 50)
        print(f"Key Factors:")
        for factor in exp['key_factors']:
            print(f"  • {factor}")
        print(f"\nRecommendation: {exp['recommendation']}")
        print()
