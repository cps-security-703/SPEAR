import json
from typing import List, Dict
from loguru import logger

from config import config
from schemas import VulnerabilityDocument, MITRESTRIDEMapping

class MITRESTRIDEMapper:
    """
    Creates comprehensive mappings between MITRE ATT&CK for ICS and STRIDE
    with specific focus on EVSE and Power Systems
    """
    
    def __init__(self):
        logger.info("Initialized MITRESTRIDEMapper")
    
    def create_comprehensive_mappings(self) -> List[VulnerabilityDocument]:
        """
        Create comprehensive MITRE-STRIDE mappings for EVSE context
        
        Returns:
            List of VulnerabilityDocument instances
        """
        logger.info("Creating comprehensive MITRE-STRIDE mappings")
        
        mappings = self._define_mitre_stride_mappings()
        documents = []
        
        for mapping in mappings:
            doc = self._create_document_from_mapping(mapping)
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} MITRE-STRIDE mapping documents")
        return documents
    
    def _define_mitre_stride_mappings(self) -> List[Dict]:
        """Define detailed MITRE-STRIDE mappings for EVSE/Power Systems"""
        return [
            {
                'mitre_id': 'T0866',
                'mitre_name': 'Exploitation of Remote Services',
                'mitre_tactic': 'Initial Access',
                'stride': ['Spoofing', 'Elevation of Privilege'],
                'description': 'Adversaries may exploit remote services to gain unauthorized access to EVSE or DMS systems',
                'evse_applicability': 'Attackers exploit vulnerabilities in OCPP, OSCP, or proprietary charging protocols to gain access',
                'attack_scenarios': [
                    'Exploiting unpatched OCPP WebSocket vulnerabilities',
                    'Attacking weak authentication in charging station APIs',
                    'Leveraging default credentials in EVSE management interfaces'
                ],
                'detection': [
                    'Monitor for unusual remote access attempts',
                    'Log authentication failures and successes',
                    'Detect exploitation attempts through IDS/IPS',
                    'Analyze network traffic for protocol anomalies'
                ],
                'mitigation': [
                    'Regular patching and updates',
                    'Strong authentication mechanisms',
                    'Network segmentation',
                    'Disable unnecessary remote services'
                ]
            },
            {
                'mitre_id': 'T0836',
                'mitre_name': 'Modify Parameter',
                'mitre_tactic': 'Impair Process Control',
                'stride': ['Tampering'],
                'description': 'Adversaries may modify parameters in control systems to disrupt operations',
                'evse_applicability': 'Attackers modify charging parameters, load balancing settings, or AGC control parameters',
                'attack_scenarios': [
                    'Modifying charging current limits to damage batteries',
                    'Altering load balancing parameters to destabilize grid',
                    'Changing AGC setpoints to cause frequency deviations',
                    'Tampering with billing rate parameters'
                ],
                'detection': [
                    'Parameter change logging and alerting',
                    'Baseline deviation detection',
                    'Integrity checking of configuration files',
                    'Real-time monitoring of critical parameters'
                ],
                'mitigation': [
                    'Access control on parameter modification',
                    'Change management processes',
                    'Parameter validation and bounds checking',
                    'Cryptographic signing of configurations'
                ]
            },
            {
                'mitre_id': 'T0814',
                'mitre_name': 'Denial of Service',
                'mitre_tactic': 'Inhibit Response Function',
                'stride': ['Denial of Service'],
                'description': 'Adversaries may perform DoS attacks to disrupt charging services or grid operations',
                'evse_applicability': 'DoS attacks against charging stations, CCMS, or grid management systems',
                'attack_scenarios': [
                    'Network flooding of charging station interfaces',
                    'Resource exhaustion attacks on CCMS',
                    'Protocol-specific DoS (OCPP message flooding)',
                    'Physical DoS through emergency stop abuse'
                ],
                'detection': [
                    'Network traffic anomaly detection',
                    'Resource utilization monitoring',
                    'Service availability monitoring',
                    'Rate-based detection algorithms'
                ],
                'mitigation': [
                    'DDoS protection mechanisms',
                    'Rate limiting and throttling',
                    'Resource allocation controls',
                    'Redundant infrastructure'
                ]
            },
            {
                'mitre_id': 'T0868',
                'mitre_name': 'Detect Operating Mode',
                'mitre_tactic': 'Collection',
                'stride': ['Information Disclosure'],
                'description': 'Adversaries gather information about system operating modes for reconnaissance',
                'evse_applicability': 'Attackers detect charging modes, grid state, or system configurations',
                'attack_scenarios': [
                    'Monitoring charging station states for attack timing',
                    'Detecting grid operating modes (normal, emergency)',
                    'Identifying peak load periods',
                    'Reconnaissance of DMS topology'
                ],
                'detection': [
                    'Monitor for unusual information queries',
                    'Detect unauthorized access to status interfaces',
                    'Log all state information requests',
                    'Behavioral analysis of access patterns'
                ],
                'mitigation': [
                    'Access control on status information',
                    'Encryption of operational data',
                    'Network segmentation',
                    'Information classification and protection'
                ]
            },
            {
                'mitre_id': 'T0831',
                'mitre_name': 'Manipulation of Control',
                'mitre_tactic': 'Impair Process Control',
                'stride': ['Tampering', 'Repudiation'],
                'description': 'Adversaries manipulate control systems to alter physical processes',
                'evse_applicability': 'Manipulation of charging control, load management, or AGC systems',
                'attack_scenarios': [
                    'Unauthorized control of charging sessions',
                    'Manipulation of load shedding commands',
                    'Altering AGC control signals',
                    'Tampering with demand response controls'
                ],
                'detection': [
                    'Control command logging and verification',
                    'Anomaly detection on control sequences',
                    'Physical process monitoring',
                    'Command source authentication'
                ],
                'mitigation': [
                    'Multi-factor authorization for critical commands',
                    'Command validation and bounds checking',
                    'Audit trails with non-repudiation',
                    'Physical process limits and safeguards'
                ]
            },
            {
                'mitre_id': 'T0890',
                'mitre_name': 'Exploitation for Privilege Escalation',
                'mitre_tactic': 'Privilege Escalation',
                'stride': ['Elevation of Privilege'],
                'description': 'Adversaries exploit vulnerabilities to gain higher privileges',
                'evse_applicability': 'Escalating from user to admin in charging systems or DMS',
                'attack_scenarios': [
                    'Exploiting EVSE firmware vulnerabilities',
                    'Privilege escalation in charging management software',
                    'Gaining admin access to SCADA/DMS systems',
                    'Exploiting API vulnerabilities for elevated access'
                ],
                'detection': [
                    'Monitor privilege changes',
                    'Detect exploitation attempts',
                    'Log all privilege escalation events',
                    'Behavioral analysis of user activities'
                ],
                'mitigation': [
                    'Regular vulnerability patching',
                    'Principle of least privilege',
                    'Application whitelisting',
                    'Exploit mitigation technologies'
                ]
            },
            {
                'mitre_id': 'T0816',
                'mitre_name': 'Device Restart/Shutdown',
                'mitre_tactic': 'Inhibit Response Function',
                'stride': ['Denial of Service'],
                'description': 'Adversaries restart or shutdown devices to disrupt operations',
                'evse_applicability': 'Forcing restart/shutdown of charging stations or grid control systems',
                'attack_scenarios': [
                    'Remote shutdown of charging stations',
                    'Forcing AGC system restarts during critical periods',
                    'Coordinated shutdown of multiple EVSE units',
                    'Disrupting DMS through system restarts'
                ],
                'detection': [
                    'Monitor unexpected restarts/shutdowns',
                    'Log all shutdown commands',
                    'Detect coordinated shutdown patterns',
                    'Alert on critical system unavailability'
                ],
                'mitigation': [
                    'Access control on shutdown commands',
                    'Physical security of devices',
                    'Redundant systems and failover',
                    'Restart authorization requirements'
                ]
            },
            {
                'mitre_id': 'T0855',
                'mitre_name': 'Unauthorized Command Message',
                'mitre_tactic': 'Impair Process Control',
                'stride': ['Spoofing', 'Tampering'],
                'description': 'Adversaries send unauthorized command messages to control systems',
                'evse_applicability': 'Sending unauthorized OCPP commands or grid control messages',
                'attack_scenarios': [
                    'Injecting malicious OCPP commands',
                    'Sending fraudulent demand response signals',
                    'Unauthorized AGC control commands',
                    'Spoofed emergency stop commands'
                ],
                'detection': [
                    'Command message authentication verification',
                    'Sequence number validation',
                    'Source verification',
                    'Protocol anomaly detection'
                ],
                'mitigation': [
                    'Message authentication and signing',
                    'Encryption of command channels',
                    'Command whitelisting',
                    'Source authentication mechanisms'
                ]
            },
            {
                'mitre_id': 'T0873',
                'mitre_name': 'Project File Infection',
                'mitre_tactic': 'Persistence',
                'stride': ['Tampering'],
                'description': 'Adversaries infect project files to maintain persistence',
                'evse_applicability': 'Infecting EVSE configuration files or DMS project files',
                'attack_scenarios': [
                    'Malware in charging station configuration files',
                    'Infected DMS project files',
                    'Backdoors in firmware update packages',
                    'Compromised SCADA project databases'
                ],
                'detection': [
                    'File integrity monitoring',
                    'Antivirus and anti-malware scanning',
                    'Digital signature verification',
                    'Behavioral analysis of file modifications'
                ],
                'mitigation': [
                    'Code signing and verification',
                    'File integrity checking',
                    'Access control on project files',
                    'Regular security scanning'
                ]
            },
            {
                'mitre_id': 'T0842',
                'mitre_name': 'Network Sniffing',
                'mitre_tactic': 'Discovery',
                'stride': ['Information Disclosure'],
                'description': 'Adversaries sniff network traffic to gather sensitive information',
                'evse_applicability': 'Capturing charging session data, credentials, or grid operational data',
                'attack_scenarios': [
                    'Sniffing unencrypted OCPP traffic',
                    'Capturing authentication credentials',
                    'Intercepting billing information',
                    'Monitoring grid control communications'
                ],
                'detection': [
                    'Detect promiscuous mode on network interfaces',
                    'Monitor for unusual network traffic patterns',
                    'Detect unauthorized network taps',
                    'Encrypted traffic analysis'
                ],
                'mitigation': [
                    'Encrypt all network communications (TLS/SSL)',
                    'Network segmentation',
                    'Physical security of network infrastructure',
                    'Use of VPNs for sensitive communications'
                ]
            }
        ]
    
    def _create_document_from_mapping(self, mapping: Dict) -> VulnerabilityDocument:
        """Create VulnerabilityDocument from MITRE-STRIDE mapping"""
        doc_id = f"MITRE-STRIDE-{mapping['mitre_id']}"
        
        embedding_text = (
            f"{mapping['mitre_id']} {mapping['mitre_name']} {mapping['description']} "
            f"{mapping['evse_applicability']} {' '.join(mapping['attack_scenarios'])} "
            f"{' '.join(mapping['mitigation'])}"
        )
        
        keywords = [
            mapping['mitre_id'].lower(),
            mapping['mitre_name'].lower().replace(' ', '_'),
            'evse', 'power_systems', 'mitre', 'stride'
        ]
        
        document = VulnerabilityDocument(
            doc_id=doc_id,
            type="mitre_stride_mapping",
            title=f"{mapping['mitre_id']}: {mapping['mitre_name']} - STRIDE Mapping",
            description=f"{mapping['description']}. EVSE Context: {mapping['evse_applicability']}",
            source=f"https://attack.mitre.org/techniques/{mapping['mitre_id']}",
            date_published="2024-01-01",
            last_updated="2024-01-01",
            stride_categories=mapping['stride'],
            mitre_tactics=[mapping['mitre_tactic']],
            mitre_techniques=[mapping['mitre_id']],
            attack_vector="Network",
            severity="High",
            affected_systems=['EVSE', 'DMS', 'AGC', 'SCADA', 'CMS', 'CCMS'],
            affected_components=[],
            industry_sectors=["Energy", "Transportation", "Critical Infrastructure"],
            mitigation_strategies=mapping['mitigation'],
            detection_methods=mapping['detection'],
            defensive_actions=mapping['mitigation'],
            countermeasures=mapping['mitigation'],
            prerequisites="",
            impact=mapping['evse_applicability'],
            cvss_score=7.5,
            exploitability="Medium",
            references=[f"https://attack.mitre.org/techniques/{mapping['mitre_id']}"],
            embedding_text=embedding_text,
            keywords=keywords,
            relevance_tags=["mitre", "stride", "mapping", "evse", "ics"] + mapping['stride']
        )
        
        return document
    
    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "mitre_stride_mappings.json"):
        """Save processed documents to file"""
        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} MITRE-STRIDE mapping documents to {filepath}")
