import json
from typing import List, Dict
from loguru import logger

from config import config
from schemas import VulnerabilityDocument

class ProtocolVulnerabilityCollector:


    def __init__(self):
        logger.info("Initialized ProtocolVulnerabilityCollector")

    def create_protocol_vulnerabilities(self) -> List[VulnerabilityDocument]:

        logger.info("Creating protocol vulnerability documents")

        vulnerabilities = self._define_protocol_vulnerabilities()
        documents = []

        for protocol, vulns in vulnerabilities.items():
            for idx, vuln in enumerate(vulns):
                doc = self._create_document_from_vulnerability(protocol, idx, vuln)
                documents.append(doc)

        logger.info(f"Created {len(documents)} protocol vulnerability documents")
        return documents

    def _define_protocol_vulnerabilities(self) -> Dict[str, List[Dict]]:

        return {
            "OCPP": [
                {
                    "version": "1.6",
                    "vulnerability": "Weak Authentication in OCPP 1.6",
                    "cve": "N/A",
                    "description": "OCPP 1.6 does not mandate mutual authentication between charging stations and central systems. Basic authentication (username/password) over WebSocket is optional and often not implemented, allowing unauthorized devices to connect.",
                    "severity": "High",
                    "cvss_score": 8.1,
                    "affected_systems": ["EVSE", "CCMS"],
                    "mitre_techniques": ["T0866", "T0862"],
                    "mitigations": [
                        "Upgrade to OCPP 2.0.1 with Security Profile 3",
                        "Implement certificate-based mutual authentication",
                        "Use VPN tunnels for OCPP communication",
                        "Enable TLS 1.2+ with strong cipher suites"
                    ],
                    "detection": [
                        "Monitor for unauthorized charging station connections",
                        "Log all authentication attempts",
                        "Detect connections without TLS",
                        "Alert on missing client certificates"
                    ],
                    "references": [
                        "OWASP EVSE Security Top 10",
                        "OCPP 1.6 Security Whitepaper"
                    ]
                },
                {
                    "version": "1.6",
                    "vulnerability": "Lack of Message Integrity in OCPP 1.6",
                    "cve": "N/A",
                    "description": "OCPP 1.6 does not provide built-in message signing or integrity verification. Attackers performing man-in-the-middle attacks can modify OCPP messages (e.g., RemoteStartTransaction, ChangeConfiguration) without detection.",
                    "severity": "High",
                    "cvss_score": 7.5,
                    "affected_systems": ["EVSE", "CCMS"],
                    "mitre_techniques": ["T0855", "T0831"],
                    "mitigations": [
                        "Implement application-level message signing",
                        "Use OCPP 2.0.1 with message authentication",
                        "Deploy network-level integrity checks",
                        "Monitor for message tampering indicators"
                    ],
                    "detection": [
                        "Validate message sequence numbers",
                        "Check for unexpected message modifications",
                        "Monitor for replay attacks",
                        "Log all critical command messages"
                    ],
                    "references": [
                        "OCPP Security Best Practices Guide"
                    ]
                },
                {
                    "version": "2.0.1",
                    "vulnerability": "OCPP 2.0.1 Security Profile Downgrade",
                    "cve": "N/A",
                    "description": "OCPP 2.0.1 defines three security profiles (1: Unsecured, 2: Basic Auth, 3: TLS with client certificates). If not properly configured, systems may downgrade to lower security profiles, negating security improvements.",
                    "severity": "Medium",
                    "cvss_score": 6.5,
                    "affected_systems": ["EVSE", "CCMS"],
                    "mitre_techniques": ["T0866"],
                    "mitigations": [
                        "Enforce Security Profile 3 in all deployments",
                        "Disable Security Profiles 1 and 2",
                        "Implement configuration management",
                        "Audit security profile settings regularly"
                    ],
                    "detection": [
                        "Monitor for security profile changes",
                        "Alert on downgrade attempts",
                        "Verify TLS configuration",
                        "Check certificate usage"
                    ],
                    "references": [
                        "OCPP 2.0.1 Security Specification"
                    ]
                }
            ],
            "ISO_15118": [
                {
                    "version": "ISO 15118-2",
                    "vulnerability": "Plug & Charge Certificate Validation Issues",
                    "cve": "N/A",
                    "description": "ISO 15118 Plug & Charge relies on X.509 certificate chains for vehicle authentication. Improper certificate validation (e.g., not checking revocation, accepting expired certificates, weak signature algorithms) can allow unauthorized vehicles to charge.",
                    "severity": "High",
                    "cvss_score": 8.2,
                    "affected_systems": ["EVSE"],
                    "mitre_techniques": ["T0862", "T0866"],
                    "mitigations": [
                        "Implement strict certificate validation (RFC 5280)",
                        "Check certificate revocation lists (CRL) or OCSP",
                        "Reject expired or self-signed certificates",
                        "Use strong signature algorithms (SHA-256+)",
                        "Implement certificate pinning for root CAs"
                    ],
                    "detection": [
                        "Log all certificate validation failures",
                        "Monitor for expired certificate usage",
                        "Alert on revoked certificate attempts",
                        "Track certificate chain anomalies"
                    ],
                    "references": [
                        "ISO 15118-2 Security Annex",
                        "CharIN Security Guidelines"
                    ]
                },
                {
                    "version": "ISO 15118-20",
                    "vulnerability": "TLS Configuration Weaknesses",
                    "cve": "N/A",
                    "description": "ISO 15118-20 mandates TLS 1.2+ for secure communication between EV and EVSE. Weak TLS configurations (e.g., supporting TLS 1.0/1.1, weak cipher suites like RC4, CBC mode) expose communications to attacks.",
                    "severity": "High",
                    "cvss_score": 7.4,
                    "affected_systems": ["EVSE"],
                    "mitre_techniques": ["T0868"],
                    "mitigations": [
                        "Enforce TLS 1.2 minimum (TLS 1.3 preferred)",
                        "Use only strong cipher suites (AEAD modes)",
                        "Disable weak ciphers (RC4, 3DES, CBC)",
                        "Implement perfect forward secrecy (PFS)",
                        "Regular TLS configuration audits"
                    ],
                    "detection": [
                        "Monitor TLS handshake negotiations",
                        "Alert on weak cipher usage",
                        "Detect TLS downgrade attempts",
                        "Log TLS version mismatches"
                    ],
                    "references": [
                        "ISO 15118-20 Specification",
                        "NIST TLS Guidelines"
                    ]
                }
            ],
            "Modbus_TCP": [
                {
                    "version": "All",
                    "vulnerability": "Lack of Authentication in Modbus TCP",
                    "cve": "N/A",
                    "description": "Modbus TCP protocol has no built-in authentication mechanism. Any device with network access can send Modbus commands to PLCs, RTUs, or charging controllers, allowing unauthorized control of critical systems.",
                    "severity": "Critical",
                    "cvss_score": 9.8,
                    "affected_systems": ["SCADA", "DMS", "EVSE", "Grid"],
                    "mitre_techniques": ["T0831", "T0836"],
                    "mitigations": [
                        "Implement Modbus/TCP Security (RFC 8551)",
                        "Use VPN or encrypted tunnels for Modbus traffic",
                        "Deploy firewall rules restricting Modbus access",
                        "Implement application-level authentication",
                        "Network segmentation (isolate Modbus devices)"
                    ],
                    "detection": [
                        "Monitor all Modbus traffic sources",
                        "Alert on unauthorized Modbus connections",
                        "Detect unexpected write commands",
                        "Log all Modbus function codes"
                    ],
                    "references": [
                        "Modbus Security Best Practices",
                        "ICS-CERT Modbus Advisories"
                    ]
                },
                {
                    "version": "All",
                    "vulnerability": "Modbus Command Injection and Manipulation",
                    "cve": "CVE-2022-2003",
                    "description": "Modbus TCP lacks message integrity checks. Attackers can inject or modify Modbus commands (e.g., Write Single Coil, Write Multiple Registers) to manipulate charging parameters, grid controls, or cause equipment damage.",
                    "severity": "Critical",
                    "cvss_score": 9.1,
                    "affected_systems": ["SCADA", "DMS", "EVSE", "AGC"],
                    "mitre_techniques": ["T0831", "T0836", "T0814"],
                    "mitigations": [
                        "Implement command validation and bounds checking",
                        "Use digital signatures for critical commands",
                        "Deploy IDS/IPS with Modbus protocol awareness",
                        "Enable read-only mode where possible",
                        "Implement rate limiting on write operations"
                    ],
                    "detection": [
                        "Monitor for unexpected register writes",
                        "Alert on out-of-bounds values",
                        "Detect rapid command sequences",
                        "Log all write operations with timestamps"
                    ],
                    "references": [
                        "CVE-2022-2003",
                        "Modbus Protocol Specification"
                    ]
                }
            ],
            "DNP3": [
                {
                    "version": "All",
                    "vulnerability": "DNP3 Authentication Bypass (SAv5 Weaknesses)",
                    "cve": "N/A",
                    "description": "DNP3 Secure Authentication v5 (SAv5) has known weaknesses including replay attacks, weak key management, and optional implementation. Many deployments don't enable SAv5, leaving communications unauthenticated.",
                    "severity": "High",
                    "cvss_score": 8.6,
                    "affected_systems": ["SCADA", "DMS", "Grid"],
                    "mitre_techniques": ["T0855", "T0831"],
                    "mitigations": [
                        "Enable DNP3 SAv5 authentication",
                        "Use strong cryptographic keys (256-bit minimum)",
                        "Implement key rotation policies",
                        "Deploy DNP3 over TLS (DNP3-TLS)",
                        "Monitor for authentication failures"
                    ],
                    "detection": [
                        "Alert on unauthenticated DNP3 sessions",
                        "Monitor for replay attack patterns",
                        "Detect SAv5 challenge-response failures",
                        "Log all DNP3 authentication events"
                    ],
                    "references": [
                        "IEEE 1815-2012 (DNP3 Specification)",
                        "DNP3 Security Best Practices"
                    ]
                },
                {
                    "version": "All",
                    "vulnerability": "DNP3 Unsolicited Response Exploitation",
                    "cve": "N/A",
                    "description": "DNP3 supports unsolicited responses where outstations send data without master requests. Attackers can exploit this to inject false data, trigger alarms, or cause operators to make incorrect decisions.",
                    "severity": "High",
                    "cvss_score": 7.8,
                    "affected_systems": ["SCADA", "DMS", "AGC"],
                    "mitre_techniques": ["T0832", "T0855"],
                    "mitigations": [
                        "Disable unsolicited responses if not needed",
                        "Validate source of unsolicited messages",
                        "Implement data integrity checks",
                        "Use DNP3 SAv5 for message authentication",
                        "Deploy anomaly detection for data patterns"
                    ],
                    "detection": [
                        "Monitor for unexpected unsolicited responses",
                        "Validate data against expected ranges",
                        "Alert on data quality flags",
                        "Correlate with other sensor data"
                    ],
                    "references": [
                        "DNP3 Security Considerations"
                    ]
                }
            ],
            "IEC_61850": [
                {
                    "version": "All",
                    "vulnerability": "IEC 61850 MMS Protocol Vulnerabilities",
                    "cve": "CVE-2022-4156",
                    "description": "IEC 61850 uses Manufacturing Message Specification (MMS) protocol which has buffer overflow, authentication bypass, and denial of service vulnerabilities affecting substation automation and grid-connected EV charging.",
                    "severity": "Critical",
                    "cvss_score": 9.4,
                    "affected_systems": ["SCADA", "DMS", "Grid", "EVSE"],
                    "mitre_techniques": ["T0866", "T0814", "T0836"],
                    "mitigations": [
                        "Apply vendor patches for IEC 61850 stacks",
                        "Implement IEC 62351 security extensions",
                        "Use TLS for MMS communication",
                        "Deploy network segmentation",
                        "Enable authentication and access control"
                    ],
                    "detection": [
                        "Monitor for malformed MMS messages",
                        "Detect buffer overflow attempts",
                        "Alert on authentication failures",
                        "Log all MMS transactions"
                    ],
                    "references": [
                        "CVE-2022-4156",
                        "IEC 61850 Security Whitepaper"
                    ]
                },
                {
                    "version": "All",
                    "vulnerability": "GOOSE Message Spoofing",
                    "cve": "N/A",
                    "description": "IEC 61850 GOOSE (Generic Object Oriented Substation Event) messages are multicast and lack authentication by default. Attackers can inject false GOOSE messages to trip breakers, manipulate protection relays, or disrupt grid operations.",
                    "severity": "Critical",
                    "cvss_score": 9.6,
                    "affected_systems": ["SCADA", "Grid", "DMS"],
                    "mitre_techniques": ["T0855", "T0816"],
                    "mitigations": [
                        "Implement IEC 62351-6 for GOOSE security",
                        "Use VLAN isolation for GOOSE traffic",
                        "Deploy GOOSE message authentication",
                        "Enable sequence number validation",
                        "Physical security for substation networks"
                    ],
                    "detection": [
                        "Monitor for duplicate GOOSE messages",
                        "Validate GOOSE sequence numbers",
                        "Alert on unexpected GOOSE sources",
                        "Detect GOOSE message flooding"
                    ],
                    "references": [
                        "IEC 62351-6 Security Standard",
                        "GOOSE Security Best Practices"
                    ]
                }
            ],
            "MQTT": [
                {
                    "version": "3.1.1",
                    "vulnerability": "MQTT Weak Authentication and Authorization",
                    "cve": "N/A",
                    "description": "MQTT brokers used in IoT-based charging infrastructure often have weak or default credentials, lack proper topic-level authorization, and allow anonymous connections, enabling unauthorized access to charging data and control.",
                    "severity": "High",
                    "cvss_score": 8.3,
                    "affected_systems": ["EVSE", "CMS"],
                    "mitre_techniques": ["T0866", "T0868"],
                    "mitigations": [
                        "Disable anonymous MQTT connections",
                        "Implement strong authentication (TLS client certs)",
                        "Use topic-level ACLs (Access Control Lists)",
                        "Deploy MQTT over TLS (MQTTS)",
                        "Change default broker credentials"
                    ],
                    "detection": [
                        "Monitor for anonymous connections",
                        "Alert on unauthorized topic subscriptions",
                        "Detect brute-force authentication attempts",
                        "Log all MQTT publish/subscribe events"
                    ],
                    "references": [
                        "MQTT Security Best Practices",
                        "OWASP IoT Top 10"
                    ]
                }
            ],
            "CoAP": [
                {
                    "version": "All",
                    "vulnerability": "CoAP Amplification DDoS",
                    "cve": "CVE-2018-19417",
                    "description": "CoAP protocol used in IoT charging devices can be exploited for amplification DDoS attacks. Attackers send small requests to CoAP servers with spoofed source IPs, causing large responses to flood victims.",
                    "severity": "Medium",
                    "cvss_score": 6.5,
                    "affected_systems": ["EVSE", "Network Infrastructure"],
                    "mitre_techniques": ["T0814"],
                    "mitigations": [
                        "Implement rate limiting on CoAP servers",
                        "Use DTLS for CoAP (CoAPS)",
                        "Deploy ingress/egress filtering",
                        "Disable CoAP on public interfaces",
                        "Monitor for amplification patterns"
                    ],
                    "detection": [
                        "Detect high-volume CoAP responses",
                        "Monitor for spoofed source addresses",
                        "Alert on unusual CoAP request rates",
                        "Track CoAP amplification ratios"
                    ],
                    "references": [
                        "CVE-2018-19417",
                        "CoAP Security Considerations RFC 7252"
                    ]
                }
            ]
        }

    def _create_document_from_vulnerability(self, protocol: str, idx: int, vuln: Dict) -> VulnerabilityDocument:

        doc_id = f"PROTOCOL-{protocol.upper().replace('_', '-')}-{idx+1:03d}"

        embedding_text = (
            f"{protocol} {vuln['vulnerability']} {vuln['description']} "
            f"{' '.join(vuln['mitigations'])}"
        )

        keywords = [
            protocol.lower(),
            "protocol",
            "vulnerability",
            vuln['severity'].lower()
        ]
        if vuln.get('cve') and vuln['cve'] != "N/A":
            keywords.append(vuln['cve'].lower())

        cve_ids = [vuln['cve']] if vuln.get('cve') and vuln['cve'] != "N/A" else []

        document = VulnerabilityDocument(
            doc_id=doc_id,
            type="protocol_vulnerability",
            title=f"{protocol} - {vuln['vulnerability']}",
            description=vuln['description'],
            source="ARES Protocol Vulnerability Database",
            date_published="2024-01-01",
            last_updated="2024-01-01",
            cve_ids=cve_ids,
            stride_categories=[],
            mitre_tactics=[],
            mitre_techniques=vuln.get('mitre_techniques', []),
            attack_vector="Network",
            severity=vuln['severity'],
            affected_systems=vuln['affected_systems'],
            affected_components=[protocol],
            industry_sectors=["Energy", "Transportation", "Critical Infrastructure"],
            mitigation_strategies=vuln['mitigations'],
            detection_methods=vuln['detection'],
            defensive_actions=vuln['mitigations'],
            countermeasures=vuln['mitigations'],
            prerequisites="",
            impact=vuln['description'],
            cvss_score=vuln['cvss_score'],
            exploitability="Medium",
            references=vuln.get('references', []),
            embedding_text=embedding_text,
            keywords=keywords,
            relevance_tags=[
                "protocol",
                protocol.lower(),
                "vulnerability",
                "evse" if "EVSE" in vuln['affected_systems'] else "power_systems"
            ]
        )

        return document

    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "protocol_documents.json"):

        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} protocol vulnerability documents to {filepath}")
