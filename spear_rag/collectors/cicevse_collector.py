import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from config import config
from schemas import VulnerabilityDocument, CICEVSE2024Record

class CICEVSECollector:


    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        logger.info("Initialized CICEVSECollector")

    def load_dataset(self, filepath: str) -> pd.DataFrame:

        logger.info(f"Loading CICEVSE2024 dataset from {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from CICEVSE2024 dataset")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return pd.DataFrame()

    def analyze_attack_patterns(self, df: pd.DataFrame) -> List[Dict]:

        logger.info("Analyzing attack patterns from CICEVSE2024")

        attack_patterns = []

        if 'Label' in df.columns or 'label' in df.columns:
            label_col = 'Label' if 'Label' in df.columns else 'label'

            attack_types = df[label_col].value_counts()

            for attack_type, count in attack_types.items():
                if attack_type.lower() != 'benign' and attack_type.lower() != 'normal':
                    pattern = self._create_attack_pattern(df, attack_type, label_col)
                    attack_patterns.append(pattern)

        logger.info(f"Identified {len(attack_patterns)} attack patterns")
        return attack_patterns

    def _create_attack_pattern(self, df: pd.DataFrame, attack_type: str, label_col: str) -> Dict:

        attack_df = df[df[label_col] == attack_type]

        pattern = {
            'attack_type': attack_type,
            'sample_count': len(attack_df),
            'description': self._generate_attack_description(attack_type),
            'stride_categories': self._map_attack_to_stride(attack_type),
            'mitre_techniques': self._map_attack_to_mitre(attack_type),
            'affected_systems': ['EVSE', 'Network Infrastructure', 'CMS'],
            'severity': self._determine_severity(attack_type),
            'detection_features': self._extract_key_features(attack_df)
        }

        return pattern

    def _generate_attack_description(self, attack_type: str) -> str:

        descriptions = {
            'DDoS': 'Distributed Denial of Service attack targeting EV charging infrastructure to disrupt service availability',
            'DoS': 'Denial of Service attack causing charging station unavailability',
            'MITM': 'Man-in-the-Middle attack intercepting communication between charging station and management system',
            'Spoofing': 'Attack where attacker impersonates legitimate charging station or vehicle',
            'UDP Flood': 'UDP flooding attack overwhelming network resources of charging infrastructure',
            'ICMP Flood': 'ICMP flooding attack targeting charging station network interfaces',
            'SYN Flood': 'TCP SYN flooding attack exhausting charging station connection resources',
            'Port Scan': 'Network reconnaissance scanning for vulnerable ports on charging infrastructure',
            'Brute Force': 'Credential brute force attack against charging station authentication',
            'SQL Injection': 'SQL injection attack targeting charging management system database',
            'XSS': 'Cross-site scripting attack against charging station web interfaces',
            'Malware': 'Malicious software infection of charging station or management system',
            'Ransomware': 'Ransomware attack encrypting charging infrastructure data'
        }

        return descriptions.get(attack_type, f'Network attack of type {attack_type} against EV charging infrastructure')

    def _map_attack_to_stride(self, attack_type: str) -> List[str]:

        mapping = {
            'DDoS': ['Denial of Service'],
            'DoS': ['Denial of Service'],
            'MITM': ['Spoofing', 'Information Disclosure', 'Tampering'],
            'Spoofing': ['Spoofing'],
            'UDP Flood': ['Denial of Service'],
            'ICMP Flood': ['Denial of Service'],
            'SYN Flood': ['Denial of Service'],
            'Port Scan': ['Information Disclosure'],
            'Brute Force': ['Elevation of Privilege'],
            'SQL Injection': ['Elevation of Privilege', 'Information Disclosure'],
            'XSS': ['Elevation of Privilege', 'Tampering'],
            'Malware': ['Tampering', 'Elevation of Privilege'],
            'Ransomware': ['Denial of Service', 'Tampering']
        }

        return mapping.get(attack_type, ['Unknown'])

    def _map_attack_to_mitre(self, attack_type: str) -> List[str]:

        mapping = {
            'DDoS': ['T0814', 'T0816'],
            'DoS': ['T0814', 'T0816'],
            'MITM': ['T0830', 'T0868'],
            'Spoofing': ['T0866', 'T0862'],
            'Port Scan': ['T0840', 'T0846'],
            'Brute Force': ['T0859', 'T0890'],
            'SQL Injection': ['T0866', 'T0868'],
            'Malware': ['T0873', 'T0874'],
            'Ransomware': ['T0881', 'T0882']
        }

        return mapping.get(attack_type, [])

    def _determine_severity(self, attack_type: str) -> str:

        high_severity = ['DDoS', 'Ransomware', 'Malware', 'SQL Injection']
        medium_severity = ['DoS', 'MITM', 'Brute Force', 'XSS']

        if attack_type in high_severity:
            return 'High'
        elif attack_type in medium_severity:
            return 'Medium'
        else:
            return 'Low'

    def _extract_key_features(self, attack_df: pd.DataFrame) -> List[str]:

        features = []

        numeric_cols = attack_df.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_cols[:10]:
            if attack_df[col].std() > 0:
                mean_val = attack_df[col].mean()
                features.append(f"{col}: mean={mean_val:.2f}")

        return features

    def create_documents_from_patterns(self, patterns: List[Dict]) -> List[VulnerabilityDocument]:

        logger.info("Creating documents from CICEVSE2024 attack patterns")

        documents = []

        for idx, pattern in enumerate(patterns):
            doc = VulnerabilityDocument(
                doc_id=f"CICEVSE-{pattern['attack_type'].upper().replace(' ', '_')}-{idx+1:03d}",
                type="dataset",
                title=f"CICEVSE2024 Attack Pattern: {pattern['attack_type']}",
                description=pattern['description'],
                source="CICEVSE2024 Dataset",
                date_published="2024-01-01",
                last_updated="2024-01-01",
                stride_categories=pattern['stride_categories'],
                mitre_tactics=[],
                mitre_techniques=pattern['mitre_techniques'],
                attack_vector="Network",
                severity=pattern['severity'],
                affected_systems=pattern['affected_systems'],
                affected_components=['network_interface', 'communication_module', 'protocol_handler'],
                industry_sectors=["Energy", "Transportation"],
                mitigation_strategies=self._generate_mitigations(pattern['attack_type']),
                detection_methods=pattern['detection_features'],
                defensive_actions=self._generate_defensive_actions(pattern['attack_type']),
                countermeasures=self._generate_mitigations(pattern['attack_type']),
                prerequisites="Network access to charging infrastructure",
                impact=pattern['description'],
                cvss_score=self._estimate_cvss(pattern['severity']),
                exploitability="Medium",
                references=["https://www.unb.ca/cic/datasets/"],
                embedding_text=f"{pattern['attack_type']} {pattern['description']} CICEVSE2024 dataset",
                keywords=[pattern['attack_type'].lower(), 'cicevse', 'network', 'attack'],
                relevance_tags=["cicevse2024", "dataset", "network_attack", pattern['attack_type'].lower()]
            )

            documents.append(doc)

        logger.info(f"Created {len(documents)} documents from CICEVSE2024 patterns")
        return documents

    def _generate_mitigations(self, attack_type: str) -> List[str]:

        mitigations = {
            'DDoS': [
                'Implement DDoS protection and traffic filtering',
                'Use rate limiting on network interfaces',
                'Deploy intrusion detection/prevention systems',
                'Network segmentation and isolation'
            ],
            'DoS': [
                'Resource allocation limits',
                'Connection rate limiting',
                'Anomaly detection systems',
                'Redundant infrastructure'
            ],
            'MITM': [
                'Use TLS/SSL encryption for all communications',
                'Implement certificate pinning',
                'Mutual authentication',
                'Network monitoring for ARP spoofing'
            ],
            'Port Scan': [
                'Firewall rules to block scanning',
                'Network segmentation',
                'Intrusion detection systems',
                'Disable unnecessary services'
            ],
            'Brute Force': [
                'Account lockout policies',
                'Multi-factor authentication',
                'Strong password requirements',
                'Rate limiting on authentication attempts'
            ],
            'SQL Injection': [
                'Input validation and sanitization',
                'Parameterized queries',
                'Web application firewall',
                'Regular security audits'
            ]
        }

        return mitigations.get(attack_type, ['Implement security best practices', 'Regular monitoring and updates'])

    def _generate_defensive_actions(self, attack_type: str) -> List[str]:

        actions = {
            'DDoS': ['Activate DDoS mitigation service', 'Increase bandwidth capacity', 'Block malicious IPs'],
            'DoS': ['Isolate affected systems', 'Implement emergency protocols', 'Failover to backup systems'],
            'MITM': ['Terminate suspicious connections', 'Verify certificate validity', 'Alert security team'],
            'Brute Force': ['Lock affected accounts', 'Block source IPs', 'Increase authentication requirements']
        }

        return actions.get(attack_type, ['Monitor and alert', 'Investigate suspicious activity'])

    def _estimate_cvss(self, severity: str) -> float:

        mapping = {
            'Critical': 9.5,
            'High': 8.0,
            'Medium': 6.0,
            'Low': 3.5
        }
        return mapping.get(severity, 5.0)

    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "cicevse_documents.json"):

        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} CICEVSE documents to {filepath}")
