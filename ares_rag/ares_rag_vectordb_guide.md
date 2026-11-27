# ARES RAG Vector Database Creation Guide

## Overview

This guide provides a step-by-step process for creating a Retrieval-Augmented Generation (RAG) vector database containing 200-250 recent cybersecurity documents focused on:
- Electric Vehicle Charging Systems (EVCS) vulnerabilities
- STRIDE threat modeling patterns
- MITRE ATT&CK for ICS techniques
- Power system cybersecurity incidents
- Critical infrastructure protection

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Collection Strategy](#data-collection-strategy)
3. [Document Schema Design](#document-schema-design)
4. [Data Collection & Preparation](#data-collection--preparation)
5. [Vector Database Setup](#vector-database-setup)
6. [Document Processing Pipeline](#document-processing-pipeline)
7. [Embedding & Indexing](#embedding--indexing)
8. [Validation & Quality Assurance](#validation--quality-assurance)
9. [Maintenance & Updates](#maintenance--updates)

---

## Prerequisites

### Required Libraries

```bash
# Core dependencies
pip install chromadb
pip install sentence-transformers
pip install openai  # if using OpenAI embeddings
pip install pandas
pip install requests
pip install beautifulsoup4
pip install python-dotenv

# Optional but recommended
pip install langchain
pip install pypdf
pip install python-docx
pip install xmltodict
```

### API Keys (Optional)

```bash
# Create .env file
OPENAI_API_KEY=your_key_here  # For alternative embeddings
NVD_API_KEY=your_nvd_key      # For NVD API access (higher rate limits)
```

---

## Data Collection Strategy

### Target Document Distribution (Total: 200-250 documents)

| Category | Target Count | Rationale |
|----------|--------------|-----------|
| **CVE/Vulnerability Reports** | 60-70 | Recent vulnerabilities in ICS, SCADA, EV charging infrastructure |
| **ICS-CERT Advisories** | 40-50 | Industry-specific security alerts and mitigations |
| **MITRE ATT&CK Techniques** | 50-60 | ICS-specific techniques and sub-techniques |
| **STRIDE Patterns** | 30-40 | Threat modeling scenarios for CPS/power systems |
| **Incident Reports** | 20-30 | Real-world cyber incidents in energy sector |
| **Academic Papers** | 10-15 | Recent research on EVCS/grid security (2023-2025) |

### Recency Criteria

- **CVEs**: Published within last 3 years (2022-2025)
- **Advisories**: Last 2 years (2023-2025)
- **Incidents**: Last 5 years (2020-2025) for major incidents
- **MITRE Techniques**: Current matrix version (v15+)
- **Papers**: Last 2 years (2023-2025)

---

## Document Schema Design

### Unified Document Schema

Each document in the vector database will follow this structure:

```python
document_schema = {
    # Core Identification
    "doc_id": str,              # Unique identifier (e.g., "CVE-2024-1234", "MITRE-T0830")
    "type": str,                # Document type (vulnerability, incident, mitre, stride, paper)
    "title": str,               # Document title
    "description": str,         # Main content/description
    "source": str,              # Source URL or reference
    "date_published": str,      # ISO format date
    "last_updated": str,        # ISO format date
    
    # Security Classifications
    "stride_categories": list,  # ["Spoofing", "Tampering", ...]
    "mitre_tactics": list,      # ["Initial Access", "Execution", ...]
    "mitre_techniques": list,   # ["T0802", "T0830", ...]
    "attack_vector": str,       # Network, Physical, Local
    "severity": str,            # Critical, High, Medium, Low
    
    # Domain-Specific
    "affected_systems": list,   # ["EVCS", "ADMS", "SCADA", "DER"]
    "affected_components": list,# ["charging_controller", "billing_system", ...]
    "industry_sectors": list,   # ["Energy", "Transportation", ...]
    
    # Actionable Information
    "mitigation_strategies": list,  # List of mitigation approaches
    "detection_methods": list,      # How to detect this threat
    "defensive_actions": list,      # Immediate response actions
    "countermeasures": list,        # Long-term countermeasures
    
    # Additional Context
    "prerequisites": str,       # Attack prerequisites
    "impact": str,              # Potential impact description
    "cvss_score": float,        # If applicable
    "exploitability": str,      # High, Medium, Low
    "references": list,         # External references/URLs
    
    # Metadata for RAG
    "embedding_text": str,      # Pre-processed text for embedding
    "keywords": list,           # Key terms for filtering
    "relevance_tags": list      # Custom tags for domain relevance
}
```

---

## Data Collection & Preparation

### 1. CVE/Vulnerability Collection

#### Using NVD API

```python
import requests
import json
from datetime import datetime, timedelta
import time

class NVDCollector:
    def __init__(self, api_key=None):
        self.base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.api_key = "your_nvd_api_key_here"
        self.headers = {}
        if api_key:
            self.headers['apiKey'] = api_key
    
    def collect_evcs_cves(self, start_date="2022-01-01", max_results=70):
        """
        Collect CVEs related to EV charging, SCADA, ICS, power systems
        """
        # Keywords for filtering
        keywords = [
            "SCADA", "ICS", "industrial control",
            "electric vehicle", "EV charging", "EVSE",
            "power grid", "smart grid", "substation",
            "DER", "distributed energy", "microgrid",
            "charging station", "EMS", "ADMS"
        ]
        
        all_cves = []
        
        for keyword in keywords:
            params = {
                'keywordSearch': keyword,
                'pubStartDate': f"{start_date}T00:00:00.000",
                'pubEndDate': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000"),
                'resultsPerPage': 20
            }
            
            try:
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'vulnerabilities' in data:
                        all_cves.extend(data['vulnerabilities'])
                
                # Rate limiting
                time.sleep(6 if not self.api_key else 0.6)
                
            except Exception as e:
                print(f"Error fetching CVEs for {keyword}: {e}")
                continue
        
        # Remove duplicates
        unique_cves = {cve['cve']['id']: cve for cve in all_cves}
        return list(unique_cves.values())[:max_results]
    
    def process_cve_to_schema(self, cve_data):
        """Convert NVD CVE format to our document schema"""
        cve = cve_data['cve']
        cve_id = cve['id']
        
        # Extract description
        description = ""
        if 'descriptions' in cve:
            for desc in cve['descriptions']:
                if desc['lang'] == 'en':
                    description = desc['value']
                    break
        
        # Extract CVSS score
        cvss_score = 0.0
        severity = "Unknown"
        if 'metrics' in cve:
            if 'cvssMetricV31' in cve['metrics']:
                cvss_score = cve['metrics']['cvssMetricV31'][0]['cvssData']['baseScore']
                severity = cve['metrics']['cvssMetricV31'][0]['cvssData']['baseSeverity']
            elif 'cvssMetricV30' in cve['metrics']:
                cvss_score = cve['metrics']['cvssMetricV30'][0]['cvssData']['baseScore']
                severity = cve['metrics']['cvssMetricV30'][0]['cvssData']['baseSeverity']
        
        # Map to STRIDE (basic heuristic)
        stride_categories = self._infer_stride(description)
        
        # Extract references
        references = []
        if 'references' in cve:
            references = [ref['url'] for ref in cve['references'][:5]]
        
        document = {
            "doc_id": cve_id,
            "type": "vulnerability",
            "title": cve_id,
            "description": description,
            "source": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            "date_published": cve.get('published', ''),
            "last_updated": cve.get('lastModified', ''),
            "stride_categories": stride_categories,
            "mitre_tactics": [],  # To be enriched later
            "mitre_techniques": [],
            "attack_vector": "Network",  # Default
            "severity": severity,
            "affected_systems": self._infer_affected_systems(description),
            "affected_components": [],
            "industry_sectors": ["Energy", "Transportation"],
            "mitigation_strategies": [],  # To be added manually or via LLM
            "detection_methods": [],
            "defensive_actions": [],
            "countermeasures": [],
            "prerequisites": "",
            "impact": description,
            "cvss_score": cvss_score,
            "exploitability": self._map_cvss_to_exploitability(cvss_score),
            "references": references,
            "embedding_text": f"{cve_id} {description}",
            "keywords": self._extract_keywords(description),
            "relevance_tags": ["cve", "vulnerability"]
        }
        
        return document
    
    def _infer_stride(self, description):
        """Basic STRIDE classification based on keywords"""
        stride = []
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['spoof', 'imperson', 'fake identity']):
            stride.append("Spoofing")
        if any(word in desc_lower for word in ['tamper', 'modify', 'alter', 'inject']):
            stride.append("Tampering")
        if any(word in desc_lower for word in ['deny', 'log', 'audit', 'trace']):
            stride.append("Repudiation")
        if any(word in desc_lower for word in ['disclosure', 'leak', 'expose', 'information']):
            stride.append("Information Disclosure")
        if any(word in desc_lower for word in ['denial', 'dos', 'crash', 'unavailable']):
            stride.append("Denial of Service")
        if any(word in desc_lower for word in ['privilege', 'escalat', 'unauthorized', 'bypass']):
            stride.append("Elevation of Privilege")
        
        return stride if stride else ["Unknown"]
    
    def _infer_affected_systems(self, description):
        """Infer which systems are affected"""
        systems = []
        desc_lower = description.lower()
        
        system_keywords = {
            "EVCS": ['ev charging', 'evse', 'charging station', 'charger'],
            "SCADA": ['scada', 'supervisory control'],
            "ICS": ['industrial control', 'ics'],
            "ADMS": ['adms', 'distribution management'],
            "DER": ['der', 'distributed energy', 'microgrid'],
            "Grid": ['power grid', 'smart grid', 'electrical grid']
        }
        
        for system, keywords in system_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                systems.append(system)
        
        return systems if systems else ["General"]
    
    def _map_cvss_to_exploitability(self, cvss_score):
        """Map CVSS score to exploitability level"""
        if cvss_score >= 9.0:
            return "Critical"
        elif cvss_score >= 7.0:
            return "High"
        elif cvss_score >= 4.0:
            return "Medium"
        else:
            return "Low"
    
    def _extract_keywords(self, text):
        """Extract relevant keywords"""
        # Simple keyword extraction (can be enhanced with NLP)
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 4 and w not in common_words]
        return list(set(keywords))[:10]


# Usage
collector = NVDCollector(api_key="your_api_key_optional")
cve_documents = []

raw_cves = collector.collect_evcs_cves(start_date="2022-01-01", max_results=70)
for cve_data in raw_cves:
    doc = collector.process_cve_to_schema(cve_data)
    cve_documents.append(doc)

print(f"Collected {len(cve_documents)} CVE documents")
```

### 2. ICS-CERT Advisory Collection

```python
import requests
from bs4 import BeautifulSoup
import re

class ICSCERTCollector:
    def __init__(self):
        self.base_url = "https://www.cisa.gov/news-events/ics-advisories"
        
    def collect_advisories(self, max_results=50):
        """
        Collect ICS-CERT advisories from CISA
        Note: This is a simplified scraper - actual implementation may need updates
        """
        advisories = []
        
        # CISA provides an API/RSS feed - check their documentation
        # For now, this is a placeholder structure
        
        # Example manual curation approach for high-quality data
        # You would extract from: https://www.cisa.gov/uscert/ics/advisories
        
        return advisories
    
    def process_advisory_to_schema(self, advisory_data):
        """Convert advisory to document schema"""
        document = {
            "doc_id": advisory_data['id'],
            "type": "ics_advisory",
            "title": advisory_data['title'],
            "description": advisory_data['description'],
            "source": advisory_data['url'],
            "date_published": advisory_data['date'],
            "last_updated": advisory_data['date'],
            "stride_categories": advisory_data.get('stride', []),
            "mitre_tactics": advisory_data.get('mitre_tactics', []),
            "mitre_techniques": advisory_data.get('mitre_techniques', []),
            "attack_vector": advisory_data.get('attack_vector', 'Network'),
            "severity": advisory_data.get('severity', 'Medium'),
            "affected_systems": advisory_data.get('affected_systems', []),
            "affected_components": advisory_data.get('affected_components', []),
            "industry_sectors": ["Energy", "Critical Infrastructure"],
            "mitigation_strategies": advisory_data.get('mitigations', []),
            "detection_methods": [],
            "defensive_actions": [],
            "countermeasures": advisory_data.get('mitigations', []),
            "prerequisites": "",
            "impact": advisory_data.get('impact', ''),
            "cvss_score": advisory_data.get('cvss_score', 0.0),
            "exploitability": "Medium",
            "references": advisory_data.get('references', []),
            "embedding_text": f"{advisory_data['title']} {advisory_data['description']}",
            "keywords": [],
            "relevance_tags": ["ics", "advisory", "cisa"]
        }
        
        return document


# Manual curation template for ICS-CERT advisories
# Create a JSON file with curated advisories
ics_advisories_template = [
    {
        "id": "ICSA-24-001-01",
        "title": "Siemens SIMATIC Energy Management",
        "description": "Vulnerability in energy management system...",
        "url": "https://www.cisa.gov/...",
        "date": "2024-01-15",
        "stride": ["Elevation of Privilege"],
        "mitre_tactics": ["Initial Access"],
        "mitre_techniques": ["T0866"],
        "severity": "High",
        "affected_systems": ["ADMS", "EMS"],
        "mitigations": ["Update to version X.Y", "Network segmentation"]
    }
    # Add 40-50 more entries...
]
```

### 3. MITRE ATT&CK for ICS Collection

```python
import requests
import json

class MITRECollector:
    def __init__(self):
        # MITRE provides STIX data
        self.stix_url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"
        
    def collect_ics_techniques(self):
        """Download and parse MITRE ATT&CK for ICS"""
        response = requests.get(self.stix_url)
        stix_data = response.json()
        
        techniques = []
        
        for obj in stix_data['objects']:
            if obj['type'] == 'attack-pattern':
                technique = self.process_technique_to_schema(obj)
                techniques.append(technique)
        
        return techniques
    
    def process_technique_to_schema(self, technique_obj):
        """Convert MITRE technique to document schema"""
        
        # Extract technique ID
        external_refs = technique_obj.get('external_references', [])
        technique_id = ""
        for ref in external_refs:
            if ref.get('source_name') == 'mitre-ics-attack':
                technique_id = ref.get('external_id', '')
                break
        
        # Extract tactics
        tactics = []
        if 'kill_chain_phases' in technique_obj:
            tactics = [phase['phase_name'].replace('-', ' ').title() 
                      for phase in technique_obj['kill_chain_phases']]
        
        # Map to STRIDE (general mapping)
        stride_mapping = {
            'Initial Access': ['Spoofing', 'Elevation of Privilege'],
            'Execution': ['Tampering'],
            'Persistence': ['Tampering'],
            'Privilege Escalation': ['Elevation of Privilege'],
            'Defense Evasion': ['Repudiation'],
            'Discovery': ['Information Disclosure'],
            'Lateral Movement': ['Tampering'],
            'Collection': ['Information Disclosure'],
            'Command and Control': ['Tampering'],
            'Inhibit Response Function': ['Denial of Service'],
            'Impair Process Control': ['Tampering', 'Denial of Service'],
            'Impact': ['Denial of Service']
        }
        
        stride_categories = []
        for tactic in tactics:
            stride_categories.extend(stride_mapping.get(tactic, []))
        stride_categories = list(set(stride_categories))
        
        description = technique_obj.get('description', '')
        
        document = {
            "doc_id": technique_id,
            "type": "mitre_technique",
            "title": technique_obj.get('name', ''),
            "description": description,
            "source": f"https://attack.mitre.org/techniques/{technique_id}",
            "date_published": technique_obj.get('created', ''),
            "last_updated": technique_obj.get('modified', ''),
            "stride_categories": stride_categories,
            "mitre_tactics": tactics,
            "mitre_techniques": [technique_id],
            "attack_vector": "Network",  # Most common
            "severity": "Medium",  # Default
            "affected_systems": ["ICS", "SCADA"],
            "affected_components": [],
            "industry_sectors": ["Energy", "Manufacturing", "Water"],
            "mitigation_strategies": self._extract_mitigations(technique_obj),
            "detection_methods": self._extract_detections(technique_obj),
            "defensive_actions": [],
            "countermeasures": [],
            "prerequisites": "",
            "impact": description,
            "cvss_score": 0.0,
            "exploitability": "Medium",
            "references": [ref.get('url', '') for ref in external_refs if 'url' in ref],
            "embedding_text": f"{technique_id} {technique_obj.get('name', '')} {description}",
            "keywords": self._extract_keywords(description),
            "relevance_tags": ["mitre", "attack", "ics", "technique"]
        }
        
        return document
    
    def _extract_mitigations(self, technique_obj):
        """Extract mitigation information if available"""
        # This would require cross-referencing with mitigation objects
        # Simplified version returns empty list
        return []
    
    def _extract_detections(self, technique_obj):
        """Extract detection information if available"""
        # Detection info is in 'x_mitre_detection' field
        detection = technique_obj.get('x_mitre_detection', '')
        if detection:
            return [detection]
        return []
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'this', 'that'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 4 and w not in common_words]
        return list(set(keywords))[:15]


# Usage
mitre_collector = MITRECollector()
mitre_documents = mitre_collector.collect_ics_techniques()
print(f"Collected {len(mitre_documents)} MITRE techniques")
```

### 4. STRIDE Pattern Collection

```python
class STRIDECollector:
    def __init__(self):
        self.stride_categories = [
            "Spoofing",
            "Tampering",
            "Repudiation",
            "Information Disclosure",
            "Denial of Service",
            "Elevation of Privilege"
        ]
    
    def create_stride_patterns(self):
        """
        Create comprehensive STRIDE patterns for EVCS/Power Systems
        Based on threat modeling best practices
        """
        stride_patterns = []
        
        # Template for each STRIDE category with EVCS-specific scenarios
        patterns = {
            "Spoofing": [
                {
                    "scenario": "EV Authentication Spoofing",
                    "description": "Attacker spoofs vehicle identity to gain unauthorized charging access or manipulate billing",
                    "attack_vector": "Network",
                    "affected_systems": ["EVCS", "Billing System"],
                    "affected_components": ["authentication_module", "billing_interface"],
                    "mitigations": [
                        "Implement PKI-based vehicle authentication",
                        "Use ISO 15118 Plug & Charge with certificate validation",
                        "Multi-factor authentication for user accounts"
                    ],
                    "detection": [
                        "Monitor authentication attempts for anomalies",
                        "Certificate validation failures",
                        "Unusual charging patterns from single vehicle ID"
                    ],
                    "mitre_techniques": ["T0866", "T0862"]
                },
                # Add 5-7 more spoofing scenarios
            ],
            "Tampering": [
                {
                    "scenario": "Charging Data Manipulation",
                    "description": "Attacker modifies charging session data to alter billing or hide malicious activity",
                    "attack_vector": "Network",
                    "affected_systems": ["EVCS", "ADMS"],
                    "affected_components": ["data_logger", "billing_system", "energy_meter"],
                    "mitigations": [
                        "Implement cryptographic signing of charging records",
                        "Use tamper-evident logging",
                        "Real-time integrity checking"
                    ],
                    "detection": [
                        "Hash validation failures",
                        "Discrepancies between metering and billing",
                        "Unexpected data modifications"
                    ],
                    "mitre_techniques": ["T0836", "T0832"]
                },
                # Add 5-7 more tampering scenarios
            ],
            "Repudiation": [
                {
                    "scenario": "Charging Session Repudiation",
                    "description": "User or attacker denies conducting charging session to avoid payment",
                    "attack_vector": "Local",
                    "affected_systems": ["EVCS", "Billing System"],
                    "affected_components": ["transaction_log", "audit_system"],
                    "mitigations": [
                        "Immutable audit logs with timestamps",
                        "Digital signatures on transactions",
                        "Blockchain-based transaction recording"
                    ],
                    "detection": [
                        "Missing log entries",
                        "Log tampering attempts",
                        "Billing disputes without valid records"
                    ],
                    "mitre_techniques": ["T0831"]
                },
                # Add 3-5 more repudiation scenarios
            ],
            "Information Disclosure": [
                {
                    "scenario": "PII Leakage from Charging Data",
                    "description": "Attacker extracts personally identifiable information from charging session logs",
                    "attack_vector": "Network",
                    "affected_systems": ["EVCS", "Cloud Backend"],
                    "affected_components": ["user_database", "session_logs", "api_endpoints"],
                    "mitigations": [
                        "Encrypt sensitive data at rest and in transit",
                        "Implement data minimization",
                        "Access control and least privilege"
                    ],
                    "detection": [
                        "Unusual database queries",
                        "Large data exports",
                        "Unauthorized API access"
                    ],
                    "mitre_techniques": ["T0868", "T0877"]
                },
                # Add 5-7 more information disclosure scenarios
            ],
            "Denial of Service": [
                {
                    "scenario": "Charging Station Flooding Attack",
                    "description": "Attacker floods charging station with requests causing service unavailability",
                    "attack_vector": "Network",
                    "affected_systems": ["EVCS", "Network Infrastructure"],
                    "affected_components": ["communication_module", "control_system"],
                    "mitigations": [
                        "Rate limiting on API endpoints",
                        "DDoS protection mechanisms",
                        "Network segmentation"
                    ],
                    "detection": [
                        "Abnormal request rates",
                        "Resource exhaustion",
                        "Service degradation alerts"
                    ],
                    "mitre_techniques": ["T0814", "T0816"]
                },
                # Add 5-7 more DoS scenarios
            ],
            "Elevation of Privilege": [
                {
                    "scenario": "Privilege Escalation via API Exploitation",
                    "description": "Attacker exploits API vulnerability to gain administrative access to charging network",
                    "attack_vector": "Network",
                    "affected_systems": ["EVCS", "Management Platform"],
                    "affected_components": ["api_gateway", "authentication_service", "admin_panel"],
                    "mitigations": [
                        "Regular security audits and penetration testing",
                        "Principle of least privilege",
                        "Input validation and sanitization"
                    ],
                    "detection": [
                        "Unexpected privilege changes",
                        "Unauthorized admin actions",
                        "API abuse patterns"
                    ],
                    "mitre_techniques": ["T0890", "T0866"]
                },
                # Add 5-7 more privilege escalation scenarios
            ]
        }
        
        # Process into document schema
        for category, scenarios in patterns.items():
            for idx, scenario in enumerate(scenarios):
                doc = {
                    "doc_id": f"STRIDE-{category.upper().replace(' ', '_')}-{idx+1:02d}",
                    "type": "stride_pattern",
                    "title": scenario['scenario'],
                    "description": scenario['description'],
                    "source": "ARES Threat Modeling",
                    "date_published": "2024-01-01",
                    "last_updated": "2024-01-01",
                    "stride_categories": [category],
                    "mitre_tactics": [],
                    "mitre_techniques": scenario.get('mitre_techniques', []),
                    "attack_vector": scenario['attack_vector'],
                    "severity": "High",
                    "affected_systems": scenario['affected_systems'],
                    "affected_components": scenario['affected_components'],
                    "industry_sectors": ["Energy", "Transportation"],
                    "mitigation_strategies": scenario['mitigations'],
                    "detection_methods": scenario['detection'],
                    "defensive_actions": scenario['mitigations'],
                    "countermeasures": scenario['mitigations'],
                    "prerequisites": "",
                    "impact": scenario['description'],
                    "cvss_score": 7.5,  # Default high
                    "exploitability": "Medium",
                    "references": [],
                    "embedding_text": f"{scenario['scenario']} {scenario['description']} {' '.join(scenario['mitigations'])}",
                    "keywords": [category.lower(), "evcs", "charging"],
                    "relevance_tags": ["stride", "threat_model", category.lower()]
                }
                stride_patterns.append(doc)
        
        return stride_patterns


# Usage
stride_collector = STRIDECollector()
stride_documents = stride_collector.create_stride_patterns()
print(f"Created {len(stride_documents)} STRIDE pattern documents")
```

### 5. Incident Report Collection

```python
class IncidentCollector:
    def __init__(self):
        pass
    
    def create_incident_reports(self):
        """
        Curate real-world cyber incidents in energy/EV sector
        Sources: News, security reports, academic papers
        """
        incidents = [
            {
                "id": "INC-2023-001",
                "title": "Colonial Pipeline Ransomware Attack",
                "date": "2021-05-07",
                "description": "Ransomware attack on Colonial Pipeline leading to fuel supply disruption across US East Coast",
                "attack_vector": "Network - VPN compromise",
                "affected_systems": ["SCADA", "IT Systems"],
                "stride": ["Denial of Service", "Tampering"],
                "mitre_techniques": ["T1133", "T1486"],
                "impact": "6-day operational shutdown, ~100 million in damages",
                "lessons_learned": [
                    "Importance of network segmentation between IT and OT",
                    "Multi-factor authentication for VPN access",
                    "Incident response planning for ransomware"
                ],
                "defensive_actions": [
                    "Implement zero-trust architecture",
                    "Regular backup and disaster recovery testing",
                    "Enhanced monitoring of OT networks"
                ],
                "references": ["https://www.cisa.gov/colonial-pipeline"]
            },
            {
                "id": "INC-2022-002",
                "title": "EV Charging Network API Vulnerability",
                "date": "2022-03-15",
                "description": "Security researchers discovered API vulnerabilities allowing unauthorized control of EV chargers",
                "attack_vector": "Network - API exploitation",
                "affected_systems": ["EVCS", "Cloud Backend"],
                "stride": ["Elevation of Privilege", "Tampering"],
                "mitre_techniques": ["T0866", "T0836"],
                "impact": "Potential for unauthorized charging control, billing manipulation",
                "lessons_learned": [
                    "Secure API design with proper authentication",
                    "Rate limiting and input validation",
                    "Regular security assessments"
                ],
                "defensive_actions": [
                    "API security testing",
                    "Implement OAuth 2.0 with scopes",
                    "Monitoring for API abuse"
                ],
                "references": []
            },
            # Add 18-28 more incidents covering:
            # - Ukraine power grid attacks (2015, 2016)
            # - Triton/TRISIS malware
            # - EV charger vulnerabilities
            # - Smart meter attacks
            # - DER manipulation incidents
            # - Recent EVCS security disclosures
        ]
        
        documents = []
        for incident in incidents:
            doc = {
                "doc_id": incident['id'],
                "type": "incident_report",
                "title": incident['title'],
                "description": incident['description'],
                "source": "Incident Database",
                "date_published": incident['date'],
                "last_updated": incident['date'],
                "stride_categories": incident['stride'],
                "mitre_tactics": [],
                "mitre_techniques": incident['mitre_techniques'],
                "attack_vector": incident['attack_vector'],
                "severity": "Critical",
                "affected_systems": incident['affected_systems'],
                "affected_components": [],
                "industry_sectors": ["Energy"],
                "mitigation_strategies": incident.get('defensive_actions', []),
                "detection_methods": [],
                "defensive_actions": incident.get('defensive_actions', []),
                "countermeasures": incident.get('defensive_actions', []),
                "prerequisites": "",
                "impact": incident['impact'],
                "cvss_score": 9.0,
                "exploitability": "High",
                "references": incident.get('references', []),
                "embedding_text": f"{incident['title']} {incident['description']} {incident['impact']}",
                "keywords": ["incident", "attack", "energy"],
                "relevance_tags": ["incident", "real_world", "case_study"]
            }
            documents.append(doc)
        
        return documents


# Usage
incident_collector = IncidentCollector()
incident_documents = incident_collector.create_incident_reports()
print(f"Created {len(incident_documents)} incident report documents")
```

---

## Vector Database Setup

### Using ChromaDB (Recommended for Local Development)

```python
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

class VectorDatabaseManager:
    def __init__(self, persist_directory="./ares_vectordb"):
        # Initialize ChromaDB with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Choose embedding function
        # Option 1: Sentence Transformers (local, free)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Fast, good quality
            # Alternative: "all-mpnet-base-v2" for better quality but slower
        )
        
        # Option 2: OpenAI embeddings (requires API key, higher quality)
        # self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key="your-openai-key",
        #     model_name="text-embedding-3-small"
        # )
        
        # Create or get collection
        self.collection = None
    
    def create_collection(self, collection_name="ares_cybersecurity_kb"):
        """Create a new collection for documents"""
        try:
            # Delete if exists (for fresh start)
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "ARES Cybersecurity Knowledge Base",
                    "version": "1.0",
                    "created_date": "2024-01-01"
                }
            )
            print(f"Created collection: {collection_name}")
            return self.collection
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self, collection_name="ares_cybersecurity_kb"):
        """Get existing collection"""
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            return self.collection
        except:
            return self.create_collection(collection_name)


# Initialize
db_manager = VectorDatabaseManager(persist_directory="./ares_vectordb")
collection = db_manager.create_collection()
```

---

## Document Processing Pipeline

### Combining All Data Sources

```python
import json
from datetime import datetime

class DocumentProcessor:
    def __init__(self):
        self.all_documents = []
    
    def collect_all_documents(self):
        """Collect documents from all sources"""
        
        print("Collecting CVE documents...")
        nvd_collector = NVDCollector()
        cve_docs = []
        raw_cves = nvd_collector.collect_evcs_cves(max_results=70)
        for cve in raw_cves:
            cve_docs.append(nvd_collector.process_cve_to_schema(cve))
        
        print(f"✓ Collected {len(cve_docs)} CVE documents")
        
        print("Collecting MITRE ATT&CK techniques...")
        mitre_collector = MITRECollector()
        mitre_docs = mitre_collector.collect_ics_techniques()
        # Filter to most relevant 50-60 techniques
        mitre_docs = mitre_docs[:60]
        print(f"✓ Collected {len(mitre_docs)} MITRE documents")
        
        print("Creating STRIDE patterns...")
        stride_collector = STRIDECollector()
        stride_docs = stride_collector.create_stride_patterns()
        print(f"✓ Created {len(stride_docs)} STRIDE documents")
        
        print("Creating incident reports...")
        incident_collector = IncidentCollector()
        incident_docs = incident_collector.create_incident_reports()
        print(f"✓ Created {len(incident_docs)} incident documents")
        
        # Combine all
        self.all_documents = cve_docs + mitre_docs + stride_docs + incident_docs
        
        print(f"\n{'='*50}")
        print(f"Total documents collected: {len(self.all_documents)}")
        print(f"{'='*50}")
        
        return self.all_documents
    
    def validate_documents(self):
        """Validate document schema compliance"""
        required_fields = ['doc_id', 'type', 'description', 'embedding_text']
        
        valid_docs = []
        invalid_docs = []
        
        for doc in self.all_documents:
            is_valid = all(field in doc for field in required_fields)
            if is_valid:
                valid_docs.append(doc)
            else:
                invalid_docs.append(doc)
        
        print(f"\nValidation Results:")
        print(f"Valid documents: {len(valid_docs)}")
        print(f"Invalid documents: {len(invalid_docs)}")
        
        return valid_docs
    
    def save_to_json(self, filename="ares_documents.json"):
        """Save all documents to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_documents, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(self.all_documents)} documents to {filename}")
    
    def load_from_json(self, filename="ares_documents.json"):
        """Load documents from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.all_documents = json.load(f)
        print(f"Loaded {len(self.all_documents)} documents from {filename}")
        return self.all_documents


# Usage
processor = DocumentProcessor()
all_docs = processor.collect_all_documents()
valid_docs = processor.validate_documents()
processor.save_to_json("ares_knowledge_base.json")
```

---

## Embedding & Indexing

### Adding Documents to Vector Database

```python
class VectorDatabaseIndexer:
    def __init__(self, collection):
        self.collection = collection
    
    def prepare_for_embedding(self, documents):
        """Prepare documents for vector database insertion"""
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            # Generate unique ID
            doc_id = doc['doc_id']
            ids.append(doc_id)
            
            # Create rich embedding text
            embedding_text = self._create_embedding_text(doc)
            texts.append(embedding_text)
            
            # Prepare metadata (exclude long text fields)
            metadata = {k: v for k, v in doc.items() 
                       if k != 'embedding_text' and not isinstance(v, list) or k in ['stride_categories', 'mitre_techniques']}
            
            # Convert lists to strings for metadata
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ', '.join(map(str, value))
            
            metadatas.append(metadata)
        
        return ids, texts, metadatas
    
    def _create_embedding_text(self, doc):
        """Create comprehensive text for embedding"""
        parts = [
            f"Title: {doc['title']}",
            f"Type: {doc['type']}",
            f"Description: {doc['description']}",
            f"STRIDE Categories: {', '.join(doc.get('stride_categories', []))}",
            f"MITRE Techniques: {', '.join(doc.get('mitre_techniques', []))}",
            f"Affected Systems: {', '.join(doc.get('affected_systems', []))}",
            f"Severity: {doc.get('severity', 'Unknown')}",
            f"Mitigations: {' '.join(doc.get('mitigation_strategies', []))}",
            f"Detection: {' '.join(doc.get('detection_methods', []))}"
        ]
        
        return " | ".join(parts)
    
    def index_documents(self, documents, batch_size=100):
        """Index documents in batches"""
        print(f"\nIndexing {len(documents)} documents...")
        
        ids, texts, metadatas = self.prepare_for_embedding(documents)
        
        # Add in batches to avoid memory issues
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            print(f"Indexed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
        
        print(f"✓ Successfully indexed {len(documents)} documents")
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        print(f"\nCollection Statistics:")
        print(f"Total documents: {count}")
        return count


# Usage - Complete Pipeline
print("="*60)
print("ARES Vector Database Creation Pipeline")
print("="*60)

# Step 1: Initialize database
db_manager = VectorDatabaseManager(persist_directory="./ares_vectordb")
collection = db_manager.create_collection("ares_cybersecurity_kb")

# Step 2: Collect and process documents
processor = DocumentProcessor()
all_documents = processor.collect_all_documents()
valid_documents = processor.validate_documents()

# Step 3: Save backup
processor.save_to_json("ares_knowledge_base_backup.json")

# Step 4: Index documents
indexer = VectorDatabaseIndexer(collection)
indexer.index_documents(valid_documents)

# Step 5: Verify
stats = indexer.get_collection_stats()

print("\n" + "="*60)
print("Vector Database Creation Complete!")
print("="*60)
```

---

## Validation & Quality Assurance

### Testing Vector Database

```python
class VectorDatabaseTester:
    def __init__(self, collection):
        self.collection = collection
    
    def test_retrieval(self):
        """Test retrieval with various queries"""
        test_queries = [
            "false data injection attack on power grid",
            "EV charging station authentication vulnerability",
            "denial of service SCADA systems",
            "privilege escalation industrial control",
            "tampering with charging session data",
            "STRIDE spoofing attacks",
            "MITRE techniques for initial access"
        ]
        
        print("\n" + "="*60)
        print("Testing Vector Database Retrieval")
        print("="*60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            print("Top 3 Results:")
            for i, (doc_id, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            ), 1):
                print(f"  {i}. {doc_id} (similarity: {1-distance:.3f})")
                print(f"     Type: {metadata.get('type', 'N/A')}")
                print(f"     Title: {metadata.get('title', 'N/A')[:60]}...")
    
    def analyze_coverage(self):
        """Analyze document type distribution"""
        all_docs = self.collection.get()
        
        type_counts = {}
        for metadata in all_docs['metadatas']:
            doc_type = metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print("\n" + "="*60)
        print("Document Type Distribution")
        print("="*60)
        for doc_type, count in sorted(type_counts.items()):
            print(f"{doc_type:25s}: {count:3d} documents")


# Run tests
tester = VectorDatabaseTester(collection)
tester.test_retrieval()
tester.analyze_coverage()
```

---

## Maintenance & Updates

### Adding New Documents

```python
def add_new_documents(collection, new_documents):
    """Add new documents to existing collection"""
    indexer = VectorDatabaseIndexer(collection)
    indexer.index_documents(new_documents)
    print(f"Added {len(new_documents)} new documents to collection")


def update_document(collection, doc_id, updated_document):
    """Update an existing document"""
    # ChromaDB update
    collection.update(
        ids=[doc_id],
        documents=[updated_document['embedding_text']],
        metadatas=[{k: v for k, v in updated_document.items() 
                   if k != 'embedding_text'}]
    )
    print(f"Updated document: {doc_id}")


def delete_document(collection, doc_id):
    """Delete a document from collection"""
    collection.delete(ids=[doc_id])
    print(f"Deleted document: {doc_id}")
```

---

## Complete Example Usage

```python
def main():
    """Complete pipeline execution"""
    
    # Configuration
    PERSIST_DIR = "./ares_vectordb"
    COLLECTION_NAME = "ares_cybersecurity_kb"
    BACKUP_FILE = "ares_knowledge_base.json"
    
    print("Starting ARES Vector Database Creation...")
    print("="*60)
    
    # Step 1: Initialize
    db_manager = VectorDatabaseManager(persist_directory=PERSIST_DIR)
    collection = db_manager.create_collection(COLLECTION_NAME)
    
    # Step 2: Collect documents
    processor = DocumentProcessor()
    all_documents = processor.collect_all_documents()
    
    # Step 3: Validate
    valid_documents = processor.validate_documents()
    
    # Step 4: Save backup
    processor.save_to_json(BACKUP_FILE)
    
    # Step 5: Index
    indexer = VectorDatabaseIndexer(collection)
    indexer.index_documents(valid_documents)
    
    # Step 6: Test
    tester = VectorDatabaseTester(collection)
    tester.test_retrieval()
    tester.analyze_coverage()
    
    print("\n" + "="*60)
    print("✓ Vector Database Creation Complete!")
    print(f"✓ Persisted to: {PERSIST_DIR}")
    print(f"✓ Backup saved: {BACKUP_FILE}")
    print("="*60)
    
    return collection


if __name__ == "__main__":
    collection = main()
```

---

## Next Steps

After creating the vector database, you can:

1. **Integrate with Gemini Agents**: Use the retrieval functions in your RL agent decision-making
2. **Expand Knowledge Base**: Continuously add new CVEs, incidents, and research findings
3. **Fine-tune Embeddings**: Consider domain-specific embedding models for better retrieval
4. **Implement Hybrid Search**: Combine vector similarity with keyword filtering for better results
5. **Add Feedback Loop**: Track which retrievals are useful and refine the knowledge base

## References

- NVD API: https://nvd.nist.gov/developers/vulnerabilities
- MITRE ATT&CK: https://attack.mitre.org/
- ICS-CERT: https://www.cisa.gov/uscert/ics
- ChromaDB Documentation: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Maintained By**: ARES Research Team
