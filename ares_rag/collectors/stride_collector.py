import json
from typing import List, Dict
from loguru import logger

from config import config
from schemas import VulnerabilityDocument

class STRIDECollector:
    """
    Collector for STRIDE threat patterns specific to EVSE and Power Systems
    """
    
    def __init__(self):
        self.stride_categories = [
            "Spoofing",
            "Tampering",
            "Repudiation",
            "Information Disclosure",
            "Denial of Service",
            "Elevation of Privilege"
        ]
        logger.info("Initialized STRIDECollector")
    
    def create_evse_stride_patterns(self) -> List[VulnerabilityDocument]:
        """
        Create comprehensive STRIDE patterns for EVSE/Power Systems
        
        Returns:
            List of VulnerabilityDocument instances
        """
        logger.info("Creating STRIDE patterns for EVSE and Power Systems")
        
        patterns = self._define_stride_patterns()
        documents = []
        
        for category, scenarios in patterns.items():
            for idx, scenario in enumerate(scenarios):
                doc = self._create_document_from_scenario(category, idx, scenario)
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} STRIDE pattern documents")
        return documents
    
    def _define_stride_patterns(self) -> Dict[str, List[Dict]]:
        """Define STRIDE patterns for EVSE infrastructure"""
        return {
            "Spoofing": [
                {
                    "scenario": "EV Authentication Spoofing",
                    "description": "Attacker spoofs vehicle identity to gain unauthorized charging access or manipulate billing records",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "Billing System", "CMS"],
                    "affected_components": ["authentication_module", "billing_interface", "iso15118_handler"],
                    "mitigations": [
                        "Implement PKI-based vehicle authentication (ISO 15118)",
                        "Use certificate pinning and validation",
                        "Multi-factor authentication for user accounts",
                        "Implement mutual TLS authentication"
                    ],
                    "detection": [
                        "Monitor authentication attempts for anomalies",
                        "Certificate validation failures",
                        "Unusual charging patterns from single vehicle ID",
                        "Behavioral analysis of charging sessions"
                    ],
                    "mitre_techniques": ["T0866", "T0862"]
                },
                {
                    "scenario": "OCPP Message Spoofing",
                    "description": "Attacker spoofs OCPP messages between charging station and central system to manipulate operations",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "CCMS"],
                    "affected_components": ["ocpp_handler", "message_queue", "websocket_connection"],
                    "mitigations": [
                        "Use OCPP 2.0.1 with security profile 3",
                        "Implement message signing and encryption",
                        "Certificate-based authentication for OCPP connections",
                        "Network segmentation and VPN tunnels"
                    ],
                    "detection": [
                        "Message integrity validation failures",
                        "Unexpected message sources",
                        "Anomalous command sequences",
                        "TLS handshake failures"
                    ],
                    "mitre_techniques": ["T0855", "T0866"]
                },
                {
                    "scenario": "Grid Operator Identity Spoofing",
                    "description": "Attacker impersonates grid operator to send fraudulent demand response or load control commands",
                    "attack_vector": "Network",
                    "affected_systems": ["DMS", "AGC", "CCMS"],
                    "affected_components": ["demand_response_module", "load_control", "scada_interface"],
                    "mitigations": [
                        "Strong authentication for grid operator interfaces",
                        "Digital signatures on control commands",
                        "Role-based access control (RBAC)",
                        "Command verification workflows"
                    ],
                    "detection": [
                        "Unauthorized access attempts",
                        "Commands from unverified sources",
                        "Unusual command patterns",
                        "Geographic anomalies in access"
                    ],
                    "mitre_techniques": ["T0866", "T0871"]
                }
            ],
            "Tampering": [
                {
                    "scenario": "Energy Metering Data Manipulation",
                    "description": "Attacker modifies charging session energy measurements to alter billing or hide malicious activity",
                    "attack_vector": "Physical/Network",
                    "affected_systems": ["EVSE", "Billing System"],
                    "affected_components": ["energy_meter", "data_logger", "billing_calculator"],
                    "mitigations": [
                        "Cryptographic signing of metering data",
                        "Tamper-evident hardware seals",
                        "Real-time integrity checking",
                        "Blockchain-based transaction logging"
                    ],
                    "detection": [
                        "Hash validation failures",
                        "Discrepancies between meter and billing",
                        "Unexpected data modifications",
                        "Physical tamper alerts"
                    ],
                    "mitre_techniques": ["T0836", "T0832"]
                },
                {
                    "scenario": "Firmware Tampering",
                    "description": "Attacker injects malicious firmware into charging station or DMS components",
                    "attack_vector": "Physical/Network",
                    "affected_systems": ["EVSE", "DMS", "SCADA"],
                    "affected_components": ["firmware_update_module", "bootloader", "control_processor"],
                    "mitigations": [
                        "Secure boot with verified signatures",
                        "Firmware signing and validation",
                        "Over-the-air update authentication",
                        "Hardware security modules (HSM)"
                    ],
                    "detection": [
                        "Firmware signature validation failures",
                        "Unexpected firmware versions",
                        "Boot integrity check failures",
                        "Anomalous system behavior post-update"
                    ],
                    "mitre_techniques": ["T0873", "T0839"]
                },
                {
                    "scenario": "Load Balancing Algorithm Manipulation",
                    "description": "Attacker modifies AGC or DMS load balancing algorithms to destabilize grid or favor specific charging stations",
                    "attack_vector": "Network",
                    "affected_systems": ["AGC", "DMS", "CCMS"],
                    "affected_components": ["load_balancer", "optimization_engine", "control_algorithm"],
                    "mitigations": [
                        "Code integrity monitoring",
                        "Configuration management and version control",
                        "Anomaly detection on control outputs",
                        "Redundant control systems"
                    ],
                    "detection": [
                        "Unexpected load distribution patterns",
                        "Algorithm output validation failures",
                        "Configuration change alerts",
                        "Performance degradation"
                    ],
                    "mitre_techniques": ["T0836", "T0856"]
                }
            ],
            "Repudiation": [
                {
                    "scenario": "Charging Transaction Repudiation",
                    "description": "User or attacker denies conducting charging session to avoid payment or accountability",
                    "attack_vector": "Local",
                    "affected_systems": ["EVSE", "Billing System"],
                    "affected_components": ["transaction_log", "audit_system", "payment_gateway"],
                    "mitigations": [
                        "Immutable audit logs with timestamps",
                        "Digital signatures on transactions",
                        "Blockchain-based transaction recording",
                        "Multi-party transaction verification"
                    ],
                    "detection": [
                        "Missing log entries",
                        "Log tampering attempts",
                        "Billing disputes without valid records",
                        "Audit trail inconsistencies"
                    ],
                    "mitre_techniques": ["T0831"]
                },
                {
                    "scenario": "Control Command Repudiation",
                    "description": "Operator denies issuing critical control commands during incident investigation",
                    "attack_vector": "Local",
                    "affected_systems": ["DMS", "AGC", "SCADA"],
                    "affected_components": ["command_logger", "audit_trail", "operator_interface"],
                    "mitigations": [
                        "Comprehensive audit logging",
                        "Non-repudiation mechanisms (digital signatures)",
                        "Video surveillance of control rooms",
                        "Multi-factor command authorization"
                    ],
                    "detection": [
                        "Audit log gaps",
                        "Unsigned commands",
                        "Conflicting operator statements",
                        "Missing authentication records"
                    ],
                    "mitre_techniques": ["T0831", "T0858"]
                }
            ],
            "Information Disclosure": [
                {
                    "scenario": "PII Leakage from Charging Data",
                    "description": "Attacker extracts personally identifiable information from charging session logs and user profiles",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "CMS", "Cloud Backend"],
                    "affected_components": ["user_database", "session_logs", "api_endpoints"],
                    "mitigations": [
                        "Data encryption at rest and in transit",
                        "Data minimization and anonymization",
                        "Access control and least privilege",
                        "GDPR/privacy compliance measures"
                    ],
                    "detection": [
                        "Unusual database queries",
                        "Large data exports",
                        "Unauthorized API access",
                        "Data exfiltration patterns"
                    ],
                    "mitre_techniques": ["T0868", "T0877"]
                },
                {
                    "scenario": "Grid Topology Information Disclosure",
                    "description": "Attacker obtains sensitive grid topology and configuration data from DMS/SCADA systems",
                    "attack_vector": "Network",
                    "affected_systems": ["DMS", "SCADA", "AGC"],
                    "affected_components": ["topology_database", "configuration_files", "network_diagrams"],
                    "mitigations": [
                        "Network segmentation (IT/OT separation)",
                        "Encryption of sensitive data",
                        "Access controls and authentication",
                        "Data classification and handling policies"
                    ],
                    "detection": [
                        "Unauthorized file access",
                        "Abnormal network traffic patterns",
                        "Failed authentication attempts",
                        "Data loss prevention alerts"
                    ],
                    "mitre_techniques": ["T0868", "T0842"]
                },
                {
                    "scenario": "Charging Pattern Analysis",
                    "description": "Attacker analyzes charging patterns to infer user behavior, location, and schedules",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "CMS", "Analytics Platform"],
                    "affected_components": ["analytics_engine", "data_warehouse", "reporting_module"],
                    "mitigations": [
                        "Differential privacy techniques",
                        "Data aggregation and anonymization",
                        "Access controls on analytics systems",
                        "Privacy-preserving analytics"
                    ],
                    "detection": [
                        "Unusual analytics queries",
                        "Unauthorized data mining activities",
                        "Privacy policy violations",
                        "Abnormal data access patterns"
                    ],
                    "mitre_techniques": ["T0877", "T0802"]
                }
            ],
            "Denial of Service": [
                {
                    "scenario": "Charging Station Flooding Attack",
                    "description": "Attacker floods charging stations with requests causing service unavailability",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "Network Infrastructure"],
                    "affected_components": ["communication_module", "api_gateway", "websocket_handler"],
                    "mitigations": [
                        "Rate limiting on API endpoints",
                        "DDoS protection mechanisms",
                        "Network traffic filtering",
                        "Resource allocation limits"
                    ],
                    "detection": [
                        "Abnormal request rates",
                        "Resource exhaustion alerts",
                        "Service degradation",
                        "Network congestion"
                    ],
                    "mitre_techniques": ["T0814", "T0816"]
                },
                {
                    "scenario": "AGC Disruption Attack",
                    "description": "Attacker disrupts automatic generation control causing frequency instability",
                    "attack_vector": "Network",
                    "affected_systems": ["AGC", "DMS", "Generation Units"],
                    "affected_components": ["frequency_controller", "generation_dispatch", "communication_links"],
                    "mitigations": [
                        "Redundant control systems",
                        "Failover mechanisms",
                        "Network resilience measures",
                        "Manual override capabilities"
                    ],
                    "detection": [
                        "Frequency deviation alerts",
                        "Control signal anomalies",
                        "Communication failures",
                        "System instability indicators"
                    ],
                    "mitre_techniques": ["T0816", "T0826"]
                },
                {
                    "scenario": "Coordinated Charging Disruption",
                    "description": "Attacker coordinates simultaneous charging start/stop to create grid instability",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "CCMS", "DMS"],
                    "affected_components": ["charging_scheduler", "load_management", "grid_interface"],
                    "mitigations": [
                        "Load ramping controls",
                        "Randomization of charging schedules",
                        "Grid stability monitoring",
                        "Emergency load shedding"
                    ],
                    "detection": [
                        "Synchronized charging patterns",
                        "Unusual load spikes",
                        "Grid stability warnings",
                        "Coordinated command sequences"
                    ],
                    "mitre_techniques": ["T0816", "T0831"]
                }
            ],
            "Elevation of Privilege": [
                {
                    "scenario": "API Privilege Escalation",
                    "description": "Attacker exploits API vulnerability to gain administrative access to charging network",
                    "attack_vector": "Network",
                    "affected_systems": ["EVSE", "CMS", "CCMS"],
                    "affected_components": ["api_gateway", "authentication_service", "admin_panel"],
                    "mitigations": [
                        "Regular security audits and penetration testing",
                        "Principle of least privilege",
                        "Input validation and sanitization",
                        "API security best practices (OAuth 2.0, JWT)"
                    ],
                    "detection": [
                        "Unexpected privilege changes",
                        "Unauthorized admin actions",
                        "API abuse patterns",
                        "Privilege escalation attempts"
                    ],
                    "mitre_techniques": ["T0890", "T0866"]
                },
                {
                    "scenario": "SCADA HMI Privilege Escalation",
                    "description": "Attacker escalates privileges on SCADA HMI to gain control over critical infrastructure",
                    "attack_vector": "Network/Local",
                    "affected_systems": ["SCADA", "DMS", "AGC"],
                    "affected_components": ["hmi_interface", "user_management", "access_control"],
                    "mitigations": [
                        "Strong authentication mechanisms",
                        "Role-based access control",
                        "Privilege separation",
                        "Regular access reviews"
                    ],
                    "detection": [
                        "Unauthorized privilege modifications",
                        "Suspicious user account activities",
                        "Access control violations",
                        "Privilege escalation exploits"
                    ],
                    "mitre_techniques": ["T0890", "T0874"]
                },
                {
                    "scenario": "Database Privilege Escalation",
                    "description": "Attacker gains elevated database privileges to modify critical charging or grid data",
                    "attack_vector": "Network",
                    "affected_systems": ["CMS", "DMS", "Billing System"],
                    "affected_components": ["database_server", "user_accounts", "stored_procedures"],
                    "mitigations": [
                        "Database security hardening",
                        "Least privilege database accounts",
                        "SQL injection prevention",
                        "Database activity monitoring"
                    ],
                    "detection": [
                        "Unauthorized database modifications",
                        "Privilege grant operations",
                        "SQL injection attempts",
                        "Abnormal database queries"
                    ],
                    "mitre_techniques": ["T0890", "T0866"]
                }
            ]
        }
    
    def _create_document_from_scenario(self, category: str, idx: int, scenario: Dict) -> VulnerabilityDocument:
        """Create VulnerabilityDocument from STRIDE scenario"""
        doc_id = f"STRIDE-{category.upper().replace(' ', '_')}-{idx+1:03d}"
        
        embedding_text = (
            f"{scenario['scenario']} {scenario['description']} "
            f"{' '.join(scenario['mitigations'])} {' '.join(scenario['detection'])}"
        )
        
        keywords = [category.lower(), "evse", "charging", "power", "grid"]
        keywords.extend([word.lower() for word in scenario['scenario'].split()[:3]])
        
        document = VulnerabilityDocument(
            doc_id=doc_id,
            type="stride_pattern",
            title=scenario['scenario'],
            description=scenario['description'],
            source="ARES STRIDE Threat Modeling Framework",
            date_published="2024-01-01",
            last_updated="2024-01-01",
            stride_categories=[category],
            mitre_tactics=[],
            mitre_techniques=scenario.get('mitre_techniques', []),
            attack_vector=scenario['attack_vector'],
            severity="High",
            affected_systems=scenario['affected_systems'],
            affected_components=scenario['affected_components'],
            industry_sectors=["Energy", "Transportation", "Critical Infrastructure"],
            mitigation_strategies=scenario['mitigations'],
            detection_methods=scenario['detection'],
            defensive_actions=scenario['mitigations'],
            countermeasures=scenario['mitigations'],
            prerequisites="",
            impact=scenario['description'],
            cvss_score=7.5,
            exploitability="Medium",
            references=[],
            embedding_text=embedding_text,
            keywords=keywords,
            relevance_tags=["stride", "threat_model", category.lower(), "evse", "power_systems"]
        )
        
        return document
    
    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "stride_documents.json"):
        """Save processed documents to file"""
        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} STRIDE documents to {filepath}")
