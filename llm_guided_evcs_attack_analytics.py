#!/usr/bin/env python3
"""
LLM-Guided RL Attack Analytics for Federated PINN EVCS CMS
Integrates STRIDE-MITRE threat modeling with Ollama deepseek-r1:8b for intelligent attack analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Ollama Integration
try:
    from openai import OpenAI
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI client not available. Install with: pip install openai")
    OLLAMA_AVAILABLE = False

# RL and ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@dataclass
class EVCSAttackVector:
    """Attack vector targeting EVCS systems"""
    attack_id: str
    attack_type: str
    target_component: str
    magnitude: float
    duration: float
    stealth_level: float
    impact_score: float
    mitre_technique: str
    stride_category: str
    prerequisites: List[str]
    detection_difficulty: float

@dataclass
class EVCSVulnerability:
    """Vulnerability in EVCS system"""
    vuln_id: str
    component: str
    vulnerability_type: str
    severity: float
    exploitability: float
    impact: float
    cvss_score: float
    mitigation: str
    detection_methods: List[str]

class OllamaLLMThreatAnalyzer:
    """LLM-powered threat analyzer using Ollama deepseek-r1:8b"""
    
    def __init__(self, base_url="http://localhost:11434/v1", model="deepseek-r1:8b"):
        self.client = None
        self.model = model
        self.base_url = base_url
        self.is_available = False
        
        if OLLAMA_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url=base_url,
                    api_key="ollama"  # Dummy key for Ollama
                )
                # Test connection
                self._test_connection()
            except Exception as e:
                print(f"Failed to initialize Ollama client: {e}")
                self.is_available = False
        else:
            print("Ollama client not available")
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert."},
                    {"role": "user", "content": "Test connection"}
                ],
                max_tokens=10
            )
            self.is_available = True
            print(f"✅ Ollama connection successful with model: {self.model}")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            self.is_available = False
    
    def analyze_evcs_vulnerabilities(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Analyze EVCS vulnerabilities using LLM"""
        if not self.is_available:
            return self._fallback_analysis(evcs_state, system_config)
        
        # Create comprehensive prompt for EVCS vulnerability analysis
        prompt = self._create_vulnerability_prompt(evcs_state, system_config)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_vulnerability_response(llm_response, evcs_state)
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_analysis(evcs_state, system_config)
    
    def generate_attack_strategy(self, vulnerabilities: List[EVCSVulnerability], 
                               evcs_state: Dict, constraints: Dict) -> Dict:
        """Generate attack strategy using LLM insights"""
        if not self.is_available:
            return self._fallback_attack_strategy(vulnerabilities, evcs_state, constraints)
        
        prompt = self._create_attack_strategy_prompt(vulnerabilities, evcs_state, constraints)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_attack_strategy_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.8
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_attack_strategy_response(llm_response, vulnerabilities)
            
        except Exception as e:
            print(f"Attack strategy generation failed: {e}")
            return self._fallback_attack_strategy(vulnerabilities, evcs_state, constraints)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for vulnerability analysis"""
        return """You are an expert cybersecurity analyst specializing in Electric Vehicle Charging Station (EVCS) systems and smart grid infrastructure. 

Your expertise includes:
- EVCS hardware vulnerabilities (power electronics, communication protocols)
- Charging Management System (CMS) security flaws
- Federated PINN model attack vectors
- STRIDE threat modeling for power systems
- MITRE ATT&CK techniques for industrial control systems
- Smart grid communication protocols (IEC 61850, Modbus, DNP3)

Analyze the provided EVCS system state and identify critical vulnerabilities, attack vectors, and potential impact scenarios. Focus on:
1. Power electronics manipulation attacks
2. Communication protocol exploitation
3. Federated learning model poisoning
4. Charging session hijacking
5. Grid stability impact vectors

Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization."""

    def _get_attack_strategy_system_prompt(self) -> str:
        """Get system prompt for attack strategy generation"""
        return """You are a red team cybersecurity expert developing sophisticated attack strategies against EVCS systems.

Your role is to:
1. Design multi-stage attack campaigns targeting EVCS infrastructure
2. Develop evasion techniques to avoid detection
3. Create coordinated attacks across multiple charging stations
4. Plan attacks that exploit federated PINN model vulnerabilities
5. Design attacks that maximize impact while minimizing detection

Focus on:
- Stealth and persistence techniques
- Coordinated multi-vector attacks
- Exploitation of federated learning vulnerabilities
- Grid stability manipulation
- Economic impact maximization

Provide detailed attack sequences with timing, coordination, and evasion strategies."""

    def _create_vulnerability_prompt(self, evcs_state: Dict, system_config: Dict) -> str:
        """Create vulnerability analysis prompt"""
        return f"""
EVCS System Analysis Request:

Current System State:
- Number of charging stations: {evcs_state.get('num_stations', 'Unknown')}
- Active charging sessions: {evcs_state.get('active_sessions', 'Unknown')}
- Grid voltage levels: {evcs_state.get('voltage_levels', 'Unknown')}
- System frequency: {evcs_state.get('frequency', 'Unknown')}
- PINN model status: {evcs_state.get('pinn_status', 'Unknown')}
- Communication status: {evcs_state.get('comm_status', 'Unknown')}

System Configuration:
- Federated learning enabled: {system_config.get('federated_enabled', False)}
- Security measures: {system_config.get('security_measures', 'Unknown')}
- Communication protocols: {system_config.get('protocols', 'Unknown')}
- Power electronics topology: {system_config.get('topology', 'Unknown')}

Please analyze this EVCS system and identify:
1. Top 5 critical vulnerabilities
2. Potential attack vectors for each vulnerability
3. MITRE ATT&CK technique mappings
4. STRIDE threat categorization
5. Impact assessment and risk scores
6. Recommended detection methods

Format your response as structured JSON with specific technical details.
"""

    def _create_attack_strategy_prompt(self, vulnerabilities: List[EVCSVulnerability], 
                                     evcs_state: Dict, constraints: Dict) -> str:
        """Create attack strategy generation prompt"""
        vuln_summary = "\n".join([
            f"- {v.vuln_id}: {v.vulnerability_type} (Severity: {v.severity:.2f}, Impact: {v.impact:.2f})"
            for v in vulnerabilities[:5]
        ])
        
        return f"""
Attack Strategy Development Request:

Identified Vulnerabilities:
{vuln_summary}

Target System State:
- Stations: {evcs_state.get('num_stations', 'Unknown')}
- Active sessions: {evcs_state.get('active_sessions', 'Unknown')}
- Grid conditions: {evcs_state.get('grid_conditions', 'Unknown')}

Attack Constraints:
- Stealth requirement: {constraints.get('stealth_level', 'Medium')}
- Time window: {constraints.get('time_window', 'Unknown')}
- Resource limits: {constraints.get('resource_limits', 'Unknown')}
- Detection avoidance: {constraints.get('avoid_detection', True)}

Develop a comprehensive attack strategy that:
1. Exploits the identified vulnerabilities
2. Maximizes impact on EVCS operations
3. Maintains stealth and avoids detection
4. Coordinates multiple attack vectors
5. Targets federated PINN model weaknesses
6. Includes timing and sequencing details

Provide specific attack steps, techniques, and coordination methods.
"""

    def _parse_vulnerability_response(self, llm_response: str, evcs_state: Dict) -> Dict:
        """Parse LLM vulnerability analysis response"""
        try:
            # Try to extract JSON from response
            if "```json" in llm_response:
                json_start = llm_response.find("```json") + 7
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
            else:
                json_str = llm_response
            
            # Parse JSON response
            parsed = json.loads(json_str)
            return parsed
            
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing
            return self._extract_vulnerabilities_from_text(llm_response, evcs_state)
    
    def _parse_attack_strategy_response(self, llm_response: str, vulnerabilities: List[EVCSVulnerability]) -> Dict:
        """Parse LLM attack strategy response"""
        try:
            if "```json" in llm_response:
                json_start = llm_response.find("```json") + 7
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
            else:
                json_str = llm_response
            
            parsed = json.loads(json_str)
            return parsed
            
        except (json.JSONDecodeError, ValueError):
            return self._extract_attack_strategy_from_text(llm_response, vulnerabilities)
    
    def _extract_vulnerabilities_from_text(self, text: str, evcs_state: Dict) -> Dict:
        """Extract vulnerability information from text response"""
        vulnerabilities = []
        
        # Simple text parsing for vulnerabilities
        lines = text.split('\n')
        current_vuln = {}
        
        for line in lines:
            line = line.strip()
            if 'vulnerability' in line.lower() or 'vuln' in line.lower():
                if current_vuln:
                    vulnerabilities.append(current_vuln)
                current_vuln = {'description': line}
            elif 'severity' in line.lower():
                current_vuln['severity'] = 0.7  # Default
            elif 'mitre' in line.lower():
                current_vuln['mitre_technique'] = 'T0000'  # Default
        
        if current_vuln:
            vulnerabilities.append(current_vuln)
        
        return {
            'vulnerabilities': vulnerabilities[:5],
            'llm_raw_response': text,
            'analysis_confidence': 0.6
        }
    
    def _extract_attack_strategy_from_text(self, text: str, vulnerabilities: List[EVCSVulnerability]) -> Dict:
        """Extract attack strategy from text response"""
        return {
            'strategy_name': 'LLM Generated Strategy',
            'attack_sequence': [
                {
                    'step': 1,
                    'action': 'Initial reconnaissance',
                    'target': 'EVCS communication',
                    'technique': 'T1590',
                    'description': 'Gather information about EVCS infrastructure'
                },
                {
                    'step': 2,
                    'action': 'Exploit vulnerability',
                    'target': 'Charging management system',
                    'technique': 'T1190',
                    'description': 'Exploit identified vulnerabilities'
                }
            ],
            'coordination_method': 'Sequential',
            'evasion_techniques': ['Stealth timing', 'Noise injection'],
            'llm_raw_response': text
        }
    
    def _fallback_analysis(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Fallback analysis when LLM is not available"""
        return {
            'vulnerabilities': [
                {
                    'vuln_id': 'VULN_001',
                    'component': 'Communication Protocol',
                    'vulnerability_type': 'Unencrypted Communication',
                    'severity': 0.8,
                    'mitre_technique': 'T1040',
                    'stride_category': 'Information_Disclosure'
                },
                {
                    'vuln_id': 'VULN_002',
                    'component': 'PINN Model',
                    'vulnerability_type': 'Model Poisoning',
                    'severity': 0.9,
                    'mitre_technique': 'T1573',
                    'stride_category': 'Tampering'
                }
            ],
            'analysis_confidence': 0.5,
            'fallback_mode': True
        }
    
    def _fallback_attack_strategy(self, vulnerabilities: List[EVCSVulnerability], 
                                evcs_state: Dict, constraints: Dict) -> Dict:
        """Fallback attack strategy when LLM is not available"""
        return {
            'strategy_name': 'Rule-Based Strategy',
            'attack_sequence': [
                {
                    'step': 1,
                    'action': 'Reconnaissance',
                    'target': 'EVCS Infrastructure',
                    'technique': 'T1590',
                    'description': 'Gather system information'
                }
            ],
            'coordination_method': 'Sequential',
            'evasion_techniques': ['Basic stealth'],
            'fallback_mode': True
        }

class STRIDEMITREThreatMapper:
    """Maps STRIDE threats to MITRE ATT&CK techniques for EVCS systems"""
    
    def __init__(self):
        self.stride_categories = {
            'Spoofing': 'Impersonating system components or users',
            'Tampering': 'Modifying system data or components', 
            'Repudiation': 'Denying actions or events',
            'Information_Disclosure': 'Exposing sensitive system information',
            'Denial_of_Service': 'Disrupting system availability',
            'Elevation_of_Privilege': 'Gaining unauthorized access levels'
        }
        
        # EVCS-specific STRIDE to MITRE mappings
        self.stride_mitre_mappings = {
            'Spoofing': {
                'sensor_spoofing': ['T0856', 'T0832'],  # Spoof Reporting Message, Manipulation of View
                'communication_spoofing': ['T0855', 'T1036'],  # Unauthorized Command Message, Masquerading
                'device_spoofing': ['T0808', 'T0849'],  # Control Device Identification, Masquerading
                'certificate_spoofing': ['T1552', 'T1588']  # Unsecured Credentials, Obtain Capabilities
            },
            'Tampering': {
                'data_tampering': ['T0832', 'T0835'],  # Manipulation of View, Modify Parameter
                'control_tampering': ['T0831', 'T0834'],  # Manipulation of Control, Modify Control Logic
                'model_tampering': ['T1573', 'T1574'],  # Data Manipulation, Hijack Execution Flow
                'firmware_tampering': ['T0857', 'T0839']  # System Firmware, Module Firmware
            },
            'Information_Disclosure': {
                'data_exfiltration': ['T1213', 'T1005'],  # Data from Information Repositories, Data from Local System
                'credential_theft': ['T1552', 'T1555'],  # Unsecured Credentials, Credentials from Password Stores
                'network_sniffing': ['T0842', 'T1040'],  # Network Sniffing, Network Sniffing
                'api_exploitation': ['T1213', 'T1552']  # Data from Information Repositories, Unsecured Credentials
            },
            'Denial_of_Service': {
                'communication_flooding': ['T0815', 'T1499'],  # Denial of Service, Endpoint Denial of Service
                'resource_exhaustion': ['T0815', 'T1499'],  # Denial of Service, Endpoint Denial of Service
                'control_blocking': ['T0803', 'T0814'],  # Block Command Message, Denial of View
                'system_shutdown': ['T0816', 'T1529']  # Device Restart/Shutdown, System Shutdown/Reboot
            },
            'Elevation_of_Privilege': {
                'privilege_escalation': ['T1055', 'T1548'],  # Process Injection, Abuse Elevation Control Mechanism
                'access_manipulation': ['T1078', 'T1556'],  # Valid Accounts, Modify Authentication Process
                'control_takeover': ['T0831', 'T1543'],  # Manipulation of Control, Create or Modify System Process
                'admin_access': ['T1078', 'T1548']  # Valid Accounts, Abuse Elevation Control Mechanism
            },
            'Repudiation': {
                'log_tampering': ['T1070', 'T1562'],  # Indicator Removal, Impair Defenses
                'audit_disabling': ['T1562', 'T1070'],  # Impair Defenses, Indicator Removal
                'evidence_destruction': ['T1485', 'T1070'],  # Data Destruction, Indicator Removal
                'activity_hiding': ['T1070', 'T1562']  # Indicator Removal, Impair Defenses
            }
        }
        
        # EVCS-specific attack techniques
        self.evcs_specific_techniques = {
            'charging_session_hijacking': {
                'stride': 'Tampering',
                'mitre': ['T1190', 'T1078'],  # Exploit Public-Facing Application, Valid Accounts
                'description': 'Hijack active charging sessions to manipulate power flow'
            },
            'pinn_model_poisoning': {
                'stride': 'Tampering', 
                'mitre': ['T1573', 'T1574'],  # Data Manipulation, Hijack Execution Flow
                'description': 'Poison federated PINN model with malicious data'
            },
            'grid_frequency_manipulation': {
                'stride': 'Tampering',
                'mitre': ['T0831', 'T0835'],  # Manipulation of Control, Modify Parameter
                'description': 'Manipulate grid frequency through coordinated EVCS attacks'
            },
            'charging_demand_manipulation': {
                'stride': 'Tampering',
                'mitre': ['T0832', 'T0835'],  # Manipulation of View, Modify Parameter
                'description': 'Manipulate charging demand signals to destabilize grid'
            }
        }
    
    def map_stride_to_mitre(self, stride_category: str, specific_threat: str = None) -> List[str]:
        """Map STRIDE category to MITRE ATT&CK techniques"""
        if stride_category in self.stride_mitre_mappings:
            if specific_threat and specific_threat in self.stride_mitre_mappings[stride_category]:
                return self.stride_mitre_mappings[stride_category][specific_threat]
            else:
                # Return all techniques for the category
                all_techniques = []
                for techniques in self.stride_mitre_mappings[stride_category].values():
                    all_techniques.extend(techniques)
                return list(set(all_techniques))  # Remove duplicates
        
        return ['T0000']  # Unknown technique
    
    def get_evcs_attack_techniques(self) -> Dict[str, Dict]:
        """Get EVCS-specific attack techniques"""
        return self.evcs_specific_techniques
    
    def analyze_threat_landscape(self, evcs_components: List[str]) -> Dict:
        """Analyze threat landscape for given EVCS components"""
        threat_analysis = {}
        
        for component in evcs_components:
            component_threats = {}
            
            for stride_category, techniques in self.stride_mitre_mappings.items():
                component_threats[stride_category] = {
                    'techniques': techniques,
                    'risk_level': self._calculate_risk_level(component, stride_category),
                    'mitigation_priority': self._get_mitigation_priority(stride_category)
                }
            
            threat_analysis[component] = component_threats
        
        return threat_analysis
    
    def _calculate_risk_level(self, component: str, stride_category: str) -> str:
        """Calculate risk level for component and STRIDE category"""
        # High-risk combinations
        high_risk = [
            ('charging_controller', 'Tampering'),
            ('communication_protocol', 'Spoofing'),
            ('pinn_model', 'Tampering'),
            ('power_electronics', 'Tampering'),
            ('grid_interface', 'Tampering')
        ]
        
        if (component, stride_category) in high_risk:
            return 'High'
        elif stride_category in ['Tampering', 'Denial_of_Service']:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_mitigation_priority(self, stride_category: str) -> int:
        """Get mitigation priority (1=highest, 5=lowest)"""
        priority_map = {
            'Tampering': 1,
            'Spoofing': 2,
            'Denial_of_Service': 2,
            'Elevation_of_Privilege': 3,
            'Information_Disclosure': 4,
            'Repudiation': 5
        }
        return priority_map.get(stride_category, 3)

if __name__ == "__main__":
    # Test the LLM threat analyzer
    print("Testing LLM-Guided EVCS Attack Analytics System")
    print("=" * 60)
    
    # Initialize components
    llm_analyzer = OllamaLLMThreatAnalyzer()
    threat_mapper = STRIDEMITREThreatMapper()
    
    # Test EVCS state
    test_evcs_state = {
        'num_stations': 6,
        'active_sessions': 12,
        'voltage_levels': {'bus1': 0.98, 'bus2': 1.02, 'bus3': 0.99},
        'frequency': 59.8,
        'pinn_status': 'active',
        'comm_status': 'encrypted'
    }
    
    test_system_config = {
        'federated_enabled': True,
        'security_measures': ['encryption', 'authentication', 'anomaly_detection'],
        'protocols': ['IEC 61850', 'Modbus'],
        'topology': 'hierarchical'
    }
    
    # Test vulnerability analysis
    print("\n1. Testing LLM Vulnerability Analysis...")
    vuln_analysis = llm_analyzer.analyze_evcs_vulnerabilities(test_evcs_state, test_system_config)
    print(f"Vulnerabilities found: {len(vuln_analysis.get('vulnerabilities', []))}")
    
    # Test STRIDE-MITRE mapping
    print("\n2. Testing STRIDE-MITRE Mapping...")
    mitre_techniques = threat_mapper.map_stride_to_mitre('Tampering', 'data_tampering')
    print(f"MITRE techniques for data tampering: {mitre_techniques}")
    
    # Test threat landscape analysis
    print("\n3. Testing Threat Landscape Analysis...")
    components = ['charging_controller', 'communication_protocol', 'pinn_model']
    threat_landscape = threat_mapper.analyze_threat_landscape(components)
    print(f"Threat landscape analyzed for {len(components)} components")
    
    print("\n✅ LLM-Guided EVCS Attack Analytics System initialized successfully!")
