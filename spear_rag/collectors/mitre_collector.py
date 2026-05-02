import requests
import json
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

from config import config
from schemas import VulnerabilityDocument

class MITRECollector:
    """
    Collector for MITRE ATT&CK for ICS techniques
    """
    
    def __init__(self):
        self.stix_url = config.MITRE_STIX_URL
        logger.info("Initialized MITRECollector")
    
    def collect_ics_techniques(self) -> List[Dict]:
        """
        Download and parse MITRE ATT&CK for ICS STIX data
        
        Returns:
            List of raw MITRE technique objects
        """
        logger.info(f"Downloading MITRE ATT&CK for ICS from {self.stix_url}")
        
        try:
            response = requests.get(self.stix_url, timeout=60)
            response.raise_for_status()
            stix_data = response.json()
            
            techniques = []
            for obj in stix_data.get('objects', []):
                if obj.get('type') == 'attack-pattern':
                    techniques.append(obj)
            
            logger.info(f"Collected {len(techniques)} MITRE ICS techniques")
            return techniques
            
        except Exception as e:
            logger.error(f"Failed to download MITRE data: {e}")
            return []
    
    def process_technique_to_document(self, technique_obj: Dict) -> VulnerabilityDocument:
        """
        Convert MITRE technique to VulnerabilityDocument schema
        
        Args:
            technique_obj: Raw MITRE STIX technique object
            
        Returns:
            VulnerabilityDocument instance
        """
        technique_id = self._extract_technique_id(technique_obj)
        technique_name = technique_obj.get('name', '')
        description = technique_obj.get('description', '')
        
        tactics = self._extract_tactics(technique_obj)
        stride_categories = self._map_tactics_to_stride(tactics)
        
        mitigations = self._extract_mitigations(technique_obj)
        detections = self._extract_detections(technique_obj)
        
        affected_systems = self._infer_affected_systems(description)
        severity = self._infer_severity(tactics, description)
        
        references = self._extract_references(technique_obj)
        keywords = self._extract_keywords(description)
        
        embedding_text = f"{technique_id} {technique_name} {description}"
        
        document = VulnerabilityDocument(
            doc_id=technique_id,
            type="mitre_technique",
            title=technique_name,
            description=description,
            source=f"https://attack.mitre.org/techniques/{technique_id}",
            date_published=technique_obj.get('created', ''),
            last_updated=technique_obj.get('modified', ''),
            stride_categories=stride_categories,
            mitre_tactics=tactics,
            mitre_techniques=[technique_id],
            attack_vector="Network",
            severity=severity,
            affected_systems=affected_systems,
            affected_components=[],
            industry_sectors=["Energy", "Manufacturing", "Water", "Critical Infrastructure"],
            mitigation_strategies=mitigations,
            detection_methods=detections,
            defensive_actions=mitigations,
            countermeasures=mitigations,
            prerequisites="",
            impact=description,
            cvss_score=0.0,
            exploitability="Medium",
            references=references,
            embedding_text=embedding_text,
            keywords=keywords,
            relevance_tags=["mitre", "attack", "ics", "technique"]
        )
        
        return document
    
    def _extract_technique_id(self, technique_obj: Dict) -> str:
        """Extract MITRE technique ID"""
        external_refs = technique_obj.get('external_references', [])
        for ref in external_refs:
            if ref.get('source_name') == 'mitre-ics-attack':
                tech_id = ref.get('external_id', '')
                if tech_id:
                    return tech_id
        
        # Fallback: use STIX ID if no external ID found
        stix_id = technique_obj.get('id', '')
        if stix_id:
            # Convert STIX ID to a usable format (e.g., attack-pattern--xxx -> MITRE-xxx)
            return f"MITRE-{stix_id.split('--')[-1][:8]}"
        
        # Last resort: use name as ID
        name = technique_obj.get('name', 'Unknown')
        return f"MITRE-{name.replace(' ', '-')[:30]}"
    
    def _extract_tactics(self, technique_obj: Dict) -> List[str]:
        """Extract tactics from kill chain phases"""
        tactics = []
        if 'kill_chain_phases' in technique_obj:
            for phase in technique_obj['kill_chain_phases']:
                tactic = phase.get('phase_name', '').replace('-', ' ').title()
                tactics.append(tactic)
        return tactics
    
    def _map_tactics_to_stride(self, tactics: List[str]) -> List[str]:
        """Map MITRE tactics to STRIDE categories"""
        stride_mapping = {
            'Initial Access': ['Spoofing', 'Elevation of Privilege'],
            'Execution': ['Tampering'],
            'Persistence': ['Tampering'],
            'Privilege Escalation': ['Elevation of Privilege'],
            'Defense Evasion': ['Repudiation'],
            'Discovery': ['Information Disclosure'],
            'Lateral Movement': ['Tampering', 'Elevation of Privilege'],
            'Collection': ['Information Disclosure'],
            'Command And Control': ['Tampering'],
            'Inhibit Response Function': ['Denial of Service'],
            'Impair Process Control': ['Tampering', 'Denial of Service'],
            'Impact': ['Denial of Service', 'Tampering']
        }
        
        stride_categories = []
        for tactic in tactics:
            stride_categories.extend(stride_mapping.get(tactic, []))
        
        return list(set(stride_categories)) if stride_categories else ["Unknown"]
    
    def _extract_mitigations(self, technique_obj: Dict) -> List[str]:
        """Extract mitigation information"""
        mitigations = []
        
        if 'x_mitre_mitigation' in technique_obj:
            mitigations.append(technique_obj['x_mitre_mitigation'])
        
        return mitigations
    
    def _extract_detections(self, technique_obj: Dict) -> List[str]:
        """Extract detection information"""
        detections = []
        
        detection_text = technique_obj.get('x_mitre_detection', '')
        if detection_text:
            detections.append(detection_text)
        
        return detections
    
    def _extract_references(self, technique_obj: Dict) -> List[str]:
        """Extract reference URLs"""
        references = []
        external_refs = technique_obj.get('external_references', [])
        
        for ref in external_refs:
            if 'url' in ref:
                references.append(ref['url'])
        
        return references[:5]
    
    def _infer_affected_systems(self, description: str) -> List[str]:
        """Infer affected systems from description"""
        systems = []
        desc_lower = description.lower()
        
        system_keywords = {
            "EVSE": ['charging', 'electric vehicle'],
            "SCADA": ['scada', 'supervisory'],
            "ICS": ['industrial control', 'ics', 'control system'],
            "PLC": ['plc', 'programmable logic'],
            "HMI": ['hmi', 'human machine'],
            "DCS": ['dcs', 'distributed control'],
            "Grid": ['grid', 'power', 'electrical'],
            "DER": ['distributed energy', 'microgrid']
        }
        
        for system, keywords in system_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                systems.append(system)
        
        return systems if systems else ["ICS", "SCADA"]
    
    def _infer_severity(self, tactics: List[str], description: str) -> str:
        """Infer severity based on tactics and description"""
        high_impact_tactics = ['Impact', 'Inhibit Response Function', 'Impair Process Control']
        
        if any(tactic in tactics for tactic in high_impact_tactics):
            return "High"
        
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['critical', 'severe', 'catastrophic']):
            return "High"
        elif any(word in desc_lower for word in ['moderate', 'significant']):
            return "Medium"
        
        return "Medium"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        common_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'this', 'that', 'with', 'from', 'by', 'as', 'is', 'was', 'are', 'be',
            'can', 'may', 'will', 'would', 'could', 'should'
        }
        
        words = text.lower().split()
        keywords = [w.strip('.,;:!?()[]{}') for w in words if len(w) > 4 and w not in common_words]
        
        return list(set(keywords))[:15]
    
    def save_raw_data(self, techniques: List[Dict], filename: str = "mitre_techniques_raw.json"):
        """Save raw MITRE data to file"""
        filepath = config.RAW_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(techniques, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(techniques)} raw MITRE techniques to {filepath}")
    
    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "mitre_documents.json"):
        """Save processed documents to file"""
        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} processed documents to {filepath}")
