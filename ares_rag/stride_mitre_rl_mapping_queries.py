"""
Test queries for STRIDE-MITRE-RL Action Mapping Evaluation
Focus on EVCS and Distribution Grid (Communication Links 1-6)
"""

# Communication links from the architecture diagram:
# 1. Charging Info (SoC, power demand), Optimal Charging (V, I, P) (OCPP) - EV <-> EVCS
# 2. Customer Authentication, Queue Management - EVCS <-> CMS
# 3. Load Measurement from CMS (DNP3) - CMS <-> Distribution Grid
# 4. Load Measurement from CMS node (DNP3) - Distribution Grid <-> DSM
# 5. Load Forecasting Info (DNP3) - DSM <-> EMS
# 6. Load Measurement from EMS (TCP/IP) - EMS <-> AGC
# [7-9 skipped: Transmission grid focus]

STRIDE_MITRE_RL_MAPPING_QUERIES = [
    {
        "query_id": "Q1_SPOOFING_EV_EVCS",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "stride_category": "Spoofing",
        "query": "What are the spoofing vulnerabilities in OCPP communication between electric vehicles and charging stations? Map them to MITRE ATT&CK techniques and suggest relevant RL attack actions.",
        "ground_truth": {
            "stride": ["Spoofing"],
            "mitre_techniques": ["T0866", "T0855", "T0862"],  # Exploitation of Remote Services, Unauthorized Command Message, Improper Input Validation
            "projected_rl_action": "Communication spoofing",
            "protocols": ["OCPP"],
            "affected_systems": ["EV", "EVCS"],
            "expected_cves": ["CVE-2022-3203", "CVE-2022-3204"]
        }
    },
    {
        "query_id": "Q2_TAMPERING_EVCS_CMS",
        "communication_link": "Link 2: EVCS <-> CMS (Authentication/Queue)",
        "stride_category": "Tampering",
        "query": "What tampering attacks can occur in the authentication and queue management between charging stations and central management systems? Identify MITRE techniques and corresponding RL actions.",
        "ground_truth": {
            "stride": ["Tampering"],
            "mitre_techniques": ["T0871", "T0836", "T0831"],  # Execution through API, Modify Parameter, Manipulation of Control
            "projected_rl_action": "Data injection",
            "protocols": ["OCPP", "HTTPS"],
            "affected_systems": ["EVCS", "CMS", "CCMS"],
            "expected_cves": ["CVE-2023-5412", "CVE-2023-5413"]
        }
    },
    {
        "query_id": "Q3_INFO_DISCLOSURE_CMS_GRID",
        "communication_link": "Link 3: CMS <-> Distribution Grid (DNP3)",
        "stride_category": "Information Disclosure",
        "query": "What information disclosure vulnerabilities exist in DNP3 load measurement communication from charging management systems to the distribution grid? Map to MITRE ATT&CK and suggest RL attack strategies.",
        "ground_truth": {
            "stride": ["Information Disclosure"],
            "mitre_techniques": ["T0840", "T0802", "T0888", "T0842"],  # Network Connection Enumeration, Automated Collection, Remote System Discovery, Network Sniffing
            "projected_rl_action": "Voltage manipulation",
            "protocols": ["DNP3"],
            "affected_systems": ["CMS", "Distribution Grid", "DMS"],
            "expected_cves": []  # DNP3 protocol vulnerabilities may not have specific CVEs
        }
    },
    {
        "query_id": "Q4_DOS_GRID_DSM",
        "communication_link": "Link 4: Distribution Grid <-> DSM (DNP3)",
        "stride_category": "Denial of Service",
        "query": "What denial of service attacks can target DNP3 load measurement communication between distribution grid and distribution system management? Identify MITRE techniques and RL attack actions.",
        "ground_truth": {
            "stride": ["Denial of Service"],
            "mitre_techniques": ["T0816", "T0814", "T0800", "T0831"],  # Device Restart/Shutdown, Denial of Service, Activate Firmware Update, Manipulation of Control
            "projected_rl_action": "Power disruption",
            "protocols": ["DNP3"],
            "affected_systems": ["Distribution Grid", "DSM", "DMS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q5_REPUDIATION_DSM_EMS",
        "communication_link": "Link 5: DSM <-> EMS (DNP3)",
        "stride_category": "Repudiation",
        "query": "What repudiation vulnerabilities exist in load forecasting communication between distribution system management and energy management systems using DNP3? Map to MITRE ATT&CK and suggest RL actions.",
        "ground_truth": {
            "stride": ["Repudiation"],
            "mitre_techniques": ["T0872", "T0858", "T0820"],  # Indicator Removal, Change Operating Mode, Exploitation for Evasion
            "projected_rl_action": "Protocol manipulation",
            "protocols": ["DNP3"],
            "affected_systems": ["DSM", "EMS", "DMS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q6_ELEVATION_EMS_AGC",
        "communication_link": "Link 6: EMS <-> AGC (TCP/IP)",
        "stride_category": "Elevation of Privilege",
        "query": "What privilege escalation vulnerabilities exist in TCP/IP load measurement communication from energy management systems to automatic generation control? Identify MITRE techniques and RL attack actions.",
        "ground_truth": {
            "stride": ["Elevation of Privilege"],
            "mitre_techniques": ["T0890", "T0891", "T0839"],  # Exploit for Privilege Escalation, Hardcoded Credentials, Module Firmware
            "projected_rl_action": "Current injection",
            "protocols": ["TCP/IP", "Modbus TCP"],
            "affected_systems": ["EMS", "AGC", "SCADA"],
            "expected_cves": ["CVE-2022-2003", "CVE-2022-2004", "CVE-2022-2005"]
        }
    },
    {
        "query_id": "Q7_TAMPERING_EV_CHARGING",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "stride_category": "Tampering",
        "query": "What are the data tampering vulnerabilities in OCPP charging parameter communication (voltage, current, power)? Map to MITRE ATT&CK and suggest RL attack actions.",
        "ground_truth": {
            "stride": ["Tampering"],
            "mitre_techniques": ["T0836", "T0831", "T0855"],  # Modify Parameter, Manipulation of Control, Unauthorized Command Message
            "projected_rl_action": "Data injection",
            "protocols": ["OCPP"],
            "affected_systems": ["EV", "EVCS"],
            "expected_cves": ["CVE-2022-3203", "CVE-2022-3204"]
        }
    },
    {
        "query_id": "Q8_DOS_EVCS_CMS",
        "communication_link": "Link 2: EVCS <-> CMS",
        "stride_category": "Denial of Service",
        "query": "What denial of service attacks can disrupt queue management and customer authentication between charging stations and central management? Identify MITRE techniques and RL actions.",
        "ground_truth": {
            "stride": ["Denial of Service"],
            "mitre_techniques": ["T0816", "T0814", "T0831"],  # Device Restart/Shutdown, Denial of Service, Manipulation of Control
            "projected_rl_action": "Power disruption",
            "protocols": ["OCPP", "HTTPS"],
            "affected_systems": ["EVCS", "CMS", "CCMS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q9_SPOOFING_DNP3",
        "communication_link": "Link 3-5: DNP3 Communication",
        "stride_category": "Spoofing",
        "query": "What spoofing vulnerabilities exist in DNP3 protocol used for load measurement and forecasting across CMS, distribution grid, DSM, and EMS? Map to MITRE ATT&CK and suggest RL attack strategies.",
        "ground_truth": {
            "stride": ["Spoofing"],
            "mitre_techniques": ["T0855", "T0866", "T0868"],  # Unauthorized Command Message, Exploitation of Remote Services, Detect Operating Mode
            "projected_rl_action": "Communication spoofing",
            "protocols": ["DNP3"],
            "affected_systems": ["CMS", "Distribution Grid", "DSM", "EMS", "DMS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q10_INFO_DISCLOSURE_OCPP",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "stride_category": "Information Disclosure",
        "query": "What information disclosure vulnerabilities exist in OCPP that could expose charging information like state of charge and power demand? Map to MITRE ATT&CK and suggest RL actions.",
        "ground_truth": {
            "stride": ["Information Disclosure"],
            "mitre_techniques": ["T0842", "T0840", "T0868"],  # Network Sniffing, Network Connection Enumeration, Detect Operating Mode
            "projected_rl_action": "Voltage manipulation",
            "protocols": ["OCPP"],
            "affected_systems": ["EV", "EVCS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q11_ELEVATION_EVCS",
        "communication_link": "Link 1-2: EV <-> EVCS <-> CMS",
        "stride_category": "Elevation of Privilege",
        "query": "What privilege escalation vulnerabilities could allow unauthorized control over charging current and voltage limits in EVCS systems? Identify MITRE techniques and RL attack actions.",
        "ground_truth": {
            "stride": ["Elevation of Privilege"],
            "mitre_techniques": ["T0890", "T0891", "T0871"],  # Exploit for Privilege Escalation, Hardcoded Credentials, Execution through API
            "projected_rl_action": "Current injection",
            "protocols": ["OCPP", "HTTPS"],
            "affected_systems": ["EVCS", "CMS", "EV"],
            "expected_cves": ["CVE-2023-2891"]  # Tesla authentication bypass
        }
    },
    {
        "query_id": "Q12_TAMPERING_LOAD_MEASUREMENT",
        "communication_link": "Link 3-4: Load Measurement (DNP3)",
        "stride_category": "Tampering",
        "query": "What tampering attacks can manipulate DNP3 load measurement data from charging management systems to distribution grid and DSM? Map to MITRE ATT&CK and suggest RL actions.",
        "ground_truth": {
            "stride": ["Tampering"],
            "mitre_techniques": ["T0836", "T0831", "T0871"],  # Modify Parameter, Manipulation of Control, Execution through API
            "projected_rl_action": "Data injection",
            "protocols": ["DNP3"],
            "affected_systems": ["CMS", "Distribution Grid", "DSM", "DMS"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q13_REPUDIATION_OCPP",
        "communication_link": "Link 1-2: OCPP Communication",
        "stride_category": "Repudiation",
        "query": "What repudiation vulnerabilities in OCPP could allow attackers to deny charging transactions or manipulate billing records? Identify MITRE techniques and RL attack strategies.",
        "ground_truth": {
            "stride": ["Repudiation"],
            "mitre_techniques": ["T0872", "T0858"],  # Indicator Removal, Change Operating Mode
            "projected_rl_action": "Protocol manipulation",
            "protocols": ["OCPP"],
            "affected_systems": ["EVCS", "CMS", "Billing System"],
            "expected_cves": []
        }
    },
    {
        "query_id": "Q14_DOS_AGC",
        "communication_link": "Link 6: EMS <-> AGC",
        "stride_category": "Denial of Service",
        "query": "What denial of service attacks can disrupt TCP/IP load measurement communication to automatic generation control systems? Map to MITRE ATT&CK and suggest RL actions.",
        "ground_truth": {
            "stride": ["Denial of Service"],
            "mitre_techniques": ["T0816", "T0814", "T0800"],  # Device Restart/Shutdown, Denial of Service, Activate Firmware Update
            "projected_rl_action": "Power disruption",
            "protocols": ["TCP/IP", "Modbus TCP"],
            "affected_systems": ["EMS", "AGC", "SCADA"],
            "expected_cves": ["CVE-2022-2003", "CVE-2022-2004"]
        }
    },
    {
        "query_id": "Q15_MULTI_STRIDE_COORDINATED",
        "communication_link": "Multiple Links: EVCS ecosystem",
        "stride_category": "Multiple",
        "query": "What are the coordinated attack scenarios combining spoofing, tampering, and denial of service across OCPP and DNP3 protocols in the EVCS-to-grid infrastructure? Map to MITRE ATT&CK and suggest comprehensive RL attack strategies.",
        "ground_truth": {
            "stride": ["Spoofing", "Tampering", "Denial of Service"],
            "mitre_techniques": ["T0866", "T0855", "T0836", "T0831", "T0816", "T0814"],
            "projected_rl_action": "Communication spoofing, Data injection, Power disruption",
            "protocols": ["OCPP", "DNP3", "TCP/IP"],
            "affected_systems": ["EV", "EVCS", "CMS", "Distribution Grid", "DSM", "EMS", "AGC"],
            "expected_cves": ["CVE-2022-3203", "CVE-2022-3204", "CVE-2022-2003"]
        }
    }
]

def get_queries_by_stride(stride_category: str):
    """Get all queries for a specific STRIDE category"""
    return [q for q in STRIDE_MITRE_RL_MAPPING_QUERIES 
            if stride_category in q['ground_truth']['stride']]

def get_queries_by_link(link_number: int):
    """Get all queries for a specific communication link"""
    return [q for q in STRIDE_MITRE_RL_MAPPING_QUERIES 
            if f"Link {link_number}" in q['communication_link']]

def get_queries_by_protocol(protocol: str):
    """Get all queries for a specific protocol"""
    return [q for q in STRIDE_MITRE_RL_MAPPING_QUERIES 
            if protocol.upper() in [p.upper() for p in q['ground_truth']['protocols']]]

if __name__ == "__main__":
    print(f"Total queries: {len(STRIDE_MITRE_RL_MAPPING_QUERIES)}")
    print("\nQueries by STRIDE category:")
    for category in ["Spoofing", "Tampering", "Repudiation", "Information Disclosure", "Denial of Service", "Elevation of Privilege"]:
        count = len(get_queries_by_stride(category))
        print(f"  {category}: {count}")
    
    print("\nQueries by Protocol:")
    for protocol in ["OCPP", "DNP3", "TCP/IP"]:
        count = len(get_queries_by_protocol(protocol))
        print(f"  {protocol}: {count}")
