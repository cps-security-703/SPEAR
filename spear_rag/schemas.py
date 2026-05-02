from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class VulnerabilityDocument(BaseModel):
    doc_id: str = Field(..., description="Unique identifier")
    type: str = Field(..., description="Document type: vulnerability, mitre_technique, stride_pattern, incident, dataset")
    title: str = Field(..., description="Document title")
    description: str = Field(..., description="Main content/description")
    source: str = Field(..., description="Source URL or reference")
    date_published: str = Field(default="", description="ISO format date")
    last_updated: str = Field(default="", description="ISO format date")
    
    stride_categories: List[str] = Field(default_factory=list, description="STRIDE categories")
    mitre_tactics: List[str] = Field(default_factory=list, description="MITRE tactics")
    mitre_techniques: List[str] = Field(default_factory=list, description="MITRE technique IDs")
    attack_vector: str = Field(default="Network", description="Attack vector type")
    severity: str = Field(default="Medium", description="Severity level")
    
    affected_systems: List[str] = Field(default_factory=list, description="Affected systems")
    affected_components: List[str] = Field(default_factory=list, description="Affected components")
    industry_sectors: List[str] = Field(default_factory=list, description="Industry sectors")
    
    mitigation_strategies: List[str] = Field(default_factory=list, description="Mitigation approaches")
    detection_methods: List[str] = Field(default_factory=list, description="Detection methods")
    defensive_actions: List[str] = Field(default_factory=list, description="Defensive actions")
    countermeasures: List[str] = Field(default_factory=list, description="Countermeasures")
    
    prerequisites: str = Field(default="", description="Attack prerequisites")
    impact: str = Field(default="", description="Potential impact")
    cvss_score: float = Field(default=0.0, description="CVSS score")
    exploitability: str = Field(default="Medium", description="Exploitability level")
    references: List[str] = Field(default_factory=list, description="External references")
    
    embedding_text: str = Field(default="", description="Text for embedding")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    relevance_tags: List[str] = Field(default_factory=list, description="Relevance tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "CVE-2024-1234",
                "type": "vulnerability",
                "title": "Authentication Bypass in EVSE Controller",
                "description": "A vulnerability in the authentication mechanism...",
                "source": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234",
                "stride_categories": ["Spoofing", "Elevation of Privilege"],
                "affected_systems": ["EVSE", "Charging Management System"],
                "severity": "High",
                "cvss_score": 8.5
            }
        }

class CICEVSE2024Record(BaseModel):
    record_id: str
    timestamp: str
    attack_type: str
    source_ip: Optional[str] = ""
    destination_ip: Optional[str] = ""
    protocol: Optional[str] = ""
    payload: Optional[str] = ""
    label: str
    features: dict = Field(default_factory=dict)
    
class MITRESTRIDEMapping(BaseModel):
    mitre_technique_id: str
    mitre_technique_name: str
    mitre_tactic: str
    stride_categories: List[str]
    description: str
    applicability_to_evse: str
    attack_scenarios: List[str]
    detection_strategies: List[str]
    mitigation_approaches: List[str]
