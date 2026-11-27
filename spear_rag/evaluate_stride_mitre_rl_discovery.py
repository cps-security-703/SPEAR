

import json
import re
from typing import Dict, List, Set, Tuple
from loguru import logger
from datetime import datetime
from evaluation.rag_evaluator import RAGEvaluator


DISCOVERY_QUERIES = [
    {
        "query_id": "Q1_SPOOFING_EVCS_CMS",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "focus_area": "Spoofing vulnerabilities in OCPP",
        "query": "What are the spoofing vulnerabilities in OCPP communication between electric vehicles and charging stations? For each vulnerability, provide: (1) STRIDE category, (2) relevant MITRE ATT&CK for ICS technique IDs, (3) suggested RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q2_TAMPERING_EVCS_CMS",
        "communication_link": "Link 2: EVCS <-> CMS",
        "focus_area": "Tampering in authentication/queue management",
        "query": "What tampering attacks can occur in the authentication and queue management between charging stations and central management systems? For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) corresponding RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q3_INFO_DISCLOSURE_DNP3",
        "communication_link": "Link 3: CMS <-> Distribution Grid (DNP3)",
        "focus_area": "Information disclosure in DNP3",
        "query": "What information disclosure vulnerabilities exist in DNP3 load measurement communication from charging management systems to the distribution grid? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack strategy, (4) CVE IDs if available."
    },
    {
        "query_id": "Q4_DOS_DNP3",
        "communication_link": "Link 4: Distribution Grid <-> DSM (DNP3)",
        "focus_area": "Denial of service in DNP3",
        "query": "What denial of service attacks can target DNP3 load measurement communication between distribution grid and distribution system management? For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q5_REPUDIATION_DNP3",
        "communication_link": "Link 5: DSM <-> EMS (DNP3)",
        "focus_area": "Repudiation in load forecasting",
        "query": "What repudiation vulnerabilities exist in load forecasting communication between distribution system management and energy management systems using DNP3? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q6_ELEVATION_TCP",
        "communication_link": "Link 6: EMS <-> AGC (TCP/IP)",
        "focus_area": "Privilege escalation in TCP/IP",
        "query": "What privilege escalation vulnerabilities exist in TCP/IP load measurement communication from energy management systems to automatic generation control? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q7_TAMPERING_CHARGING_PARAMS",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "focus_area": "Tampering with charging parameters",
        "query": "What are the data tampering vulnerabilities in OCPP charging parameter communication (voltage, current, power)? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q8_DOS_QUEUE_MGMT",
        "communication_link": "Link 2: EVCS <-> CMS",
        "focus_area": "DoS in queue management",
        "query": "What denial of service attacks can disrupt queue management and customer authentication between charging stations and central management? For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q9_SPOOFING_DNP3",
        "communication_link": "Links 3-5: DNP3 Communication Chain",
        "focus_area": "Spoofing in DNP3 protocol",
        "query": "What spoofing vulnerabilities exist in DNP3 protocol used for load measurement and forecasting across CMS, distribution grid, DSM, and EMS? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack strategy, (4) CVE IDs if available."
    },
    {
        "query_id": "Q10_INFO_DISCLOSURE_OCPP",
        "communication_link": "Link 1: EV <-> EVCS (OCPP)",
        "focus_area": "Information disclosure in OCPP",
        "query": "What information disclosure vulnerabilities exist in OCPP that could expose charging information like state of charge and power demand? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q11_ELEVATION_CMS",
        "communication_link": "Link 2b: CMS <-> CCMS (TCP/IP)",
        "focus_area": "Privilege escalation from CMS to CCMS over TCP/IP",
        "query": "What privilege escalation vulnerabilities exist in the TCP/IP communication between a local Charging Management System (CMS) and a Central Charging Management System (CCMS)? Consider exploits such as buffer overflows, weak API authentication, remote service flaws, and hardcoded credentials that could allow an attacker to escalate from CMS-level access to CCMS administrator control, enabling override of fleet-wide charging authorization and demand setpoints. For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK for ICS technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q12_TAMPERING_LOAD_DATA",
        "communication_link": "Links 3-4: Load Measurement (DNP3)",
        "focus_area": "Tampering with load measurement",
        "query": "What tampering attacks can manipulate DNP3 load measurement data from charging management systems to distribution grid and DSM? For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q13_REPUDIATION_OCPP",
        "communication_link": "Links 1-2: OCPP Communication",
        "focus_area": "Repudiation in billing/transactions",
        "query": "What repudiation vulnerabilities in OCPP could allow attackers to deny charging transactions or manipulate billing records? For each vulnerability, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack strategy, (4) CVE IDs if available."
    },
    {
        "query_id": "Q14_DOS_AGC",
        "communication_link": "Link 6: EMS <-> AGC (TCP/IP)",
        "focus_area": "DoS targeting AGC",
        "query": "What denial of service attacks can disrupt TCP/IP load measurement communication to automatic generation control systems? For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK technique IDs, (3) RL attack action, (4) CVE IDs if available."
    },
    {
        "query_id": "Q15_TAMPERING_SPOOFING_CMS_CCMS",
        "communication_link": "Link 2b: CMS <-> CCMS (TCP/IP)",
        "focus_area": "Tampering and spoofing in CMS to CCMS aggregated data communication",
        "query": "What tampering and spoofing attacks can occur in the aggregated data communication between local charging management systems (CMS) and the central charging management system (CCMS)? Consider attacks on aggregated load data, authentication token forwarding, queue synchronization, and billing record aggregation flowing over TCP/IP. For each attack, provide: (1) STRIDE category, (2) MITRE ATT&CK for ICS technique IDs, (3) RL attack action, (4) CVE IDs if available."
    }
]

class DiscoveryModeEvaluator:


    def __init__(self):
        self.rag_evaluator = RAGEvaluator()

    def extract_mappings(self, response: str) -> Dict:


        mitre_techniques = set(re.findall(r'T\d{4}', response.upper()))


        stride_categories = []
        stride_keywords = {
            "Spoofing": ["spoof", "impersonat", "fake identity"],
            "Tampering": ["tamper", "modif", "alter", "manipulat"],
            "Repudiation": ["repudiat", "deny", "log", "audit"],
            "Information Disclosure": ["disclosure", "leak", "expos", "sniff"],
            "Denial of Service": ["denial", "dos", "disrupt", "unavailable"],
            "Elevation of Privilege": ["privilege", "escalat", "unauthorized access"]
        }

        response_lower = response.lower()
        for category, keywords in stride_keywords.items():
            if any(kw in response_lower for kw in keywords):
                stride_categories.append(category)


        rl_action_patterns = [
            r'rl action[s]?:?\s*([^\n\.]+)',
            r'attack action[s]?:?\s*([^\n\.]+)',
            r'suggested action[s]?:?\s*([^\n\.]+)',
            r'projected action[s]?:?\s*([^\n\.]+)'
        ]

        rl_actions = []
        for pattern in rl_action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            rl_actions.extend(matches)


        cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', response.upper()))


        protocols = []
        protocol_keywords = ["OCPP", "DNP3", "TCP/IP", "Modbus", "HTTPS", "ISO 15118"]
        for protocol in protocol_keywords:
            if protocol.upper() in response.upper():
                protocols.append(protocol)

        return {
            "mitre_techniques": sorted(list(mitre_techniques)),
            "stride_categories": stride_categories,
            "rl_actions": [action.strip() for action in rl_actions if action.strip()],
            "cves": sorted(list(cves)),
            "protocols": protocols,
            "response_length": len(response),
            "has_structured_format": self._check_structured_format(response)
        }

    def _check_structured_format(self, response: str) -> bool:

        required_sections = ["stride", "mitre", "attack"]
        response_lower = response.lower()
        return sum(1 for section in required_sections if section in response_lower) >= 2

    def compare_mappings(self, rag_mapping: Dict, non_rag_mapping: Dict) -> Dict:


        comparison = {
            "mitre_techniques": {
                "rag_count": len(rag_mapping["mitre_techniques"]),
                "non_rag_count": len(non_rag_mapping["mitre_techniques"]),
                "rag_only": sorted(list(set(rag_mapping["mitre_techniques"]) - set(non_rag_mapping["mitre_techniques"]))),
                "non_rag_only": sorted(list(set(non_rag_mapping["mitre_techniques"]) - set(rag_mapping["mitre_techniques"]))),
                "common": sorted(list(set(rag_mapping["mitre_techniques"]) & set(non_rag_mapping["mitre_techniques"])))
            },
            "stride_categories": {
                "rag_count": len(rag_mapping["stride_categories"]),
                "non_rag_count": len(non_rag_mapping["stride_categories"]),
                "rag_only": list(set(rag_mapping["stride_categories"]) - set(non_rag_mapping["stride_categories"])),
                "non_rag_only": list(set(non_rag_mapping["stride_categories"]) - set(rag_mapping["stride_categories"])),
                "common": list(set(rag_mapping["stride_categories"]) & set(non_rag_mapping["stride_categories"]))
            },
            "cves": {
                "rag_count": len(rag_mapping["cves"]),
                "non_rag_count": len(non_rag_mapping["cves"]),
                "rag_cves": rag_mapping["cves"],
                "non_rag_cves": non_rag_mapping["cves"]
            },
            "quality_indicators": {
                "rag_structured": rag_mapping["has_structured_format"],
                "non_rag_structured": non_rag_mapping["has_structured_format"],
                "rag_length": rag_mapping["response_length"],
                "non_rag_length": non_rag_mapping["response_length"],
                "rag_more_comprehensive": (
                    len(rag_mapping["mitre_techniques"]) > len(non_rag_mapping["mitre_techniques"]) and
                    len(rag_mapping["cves"]) >= len(non_rag_mapping["cves"])
                )
            }
        }

        return comparison

    def evaluate_query(self, query_data: Dict) -> Dict:

        query = query_data['query']

        logger.info(f"Evaluating: {query_data['query_id']}")


        logger.info("Getting RAG response...")
        rag_response = self.rag_evaluator.get_rag_response(query)


        logger.info("Getting non-RAG response...")
        non_rag_response = self.rag_evaluator.get_non_rag_response(query)


        rag_mapping = self.extract_mappings(rag_response)
        non_rag_mapping = self.extract_mappings(non_rag_response)


        comparison = self.compare_mappings(rag_mapping, non_rag_mapping)

        return {
            "query_id": query_data["query_id"],
            "query": query,
            "communication_link": query_data["communication_link"],
            "focus_area": query_data["focus_area"],
            "rag_response": rag_response,
            "non_rag_response": non_rag_response,
            "rag_mapping": rag_mapping,
            "non_rag_mapping": non_rag_mapping,
            "comparison": comparison
        }

    def evaluate_all(self) -> Dict:

        results = []

        for query_data in DISCOVERY_QUERIES:
            try:
                result = self.evaluate_query(query_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query_data['query_id']}: {e}")
                continue


        summary = self.calculate_summary(results)

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "discovery",
            "total_queries": len(results),
            "summary": summary,
            "individual_results": results
        }

    def calculate_summary(self, results: List[Dict]) -> Dict:


        total_rag_mitre = sum(r["comparison"]["mitre_techniques"]["rag_count"] for r in results)
        total_non_rag_mitre = sum(r["comparison"]["mitre_techniques"]["non_rag_count"] for r in results)

        total_rag_cves = sum(r["comparison"]["cves"]["rag_count"] for r in results)
        total_non_rag_cves = sum(r["comparison"]["cves"]["non_rag_count"] for r in results)

        rag_more_comprehensive = sum(1 for r in results if r["comparison"]["quality_indicators"]["rag_more_comprehensive"])

        return {
            "avg_mitre_per_query": {
                "rag": round(total_rag_mitre / len(results), 2) if results else 0,
                "non_rag": round(total_non_rag_mitre / len(results), 2) if results else 0
            },
            "avg_cves_per_query": {
                "rag": round(total_rag_cves / len(results), 2) if results else 0,
                "non_rag": round(total_non_rag_cves / len(results), 2) if results else 0
            },
            "rag_more_comprehensive_count": rag_more_comprehensive,
            "rag_more_comprehensive_percentage": round(rag_more_comprehensive / len(results) * 100, 2) if results else 0
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Discovery Mode: Compare RAG vs Non-RAG STRIDE-MITRE-RL mappings')
    parser.add_argument('--query-id', type=str, help='Evaluate specific query')
    parser.add_argument('--output', type=str, default='stride_mitre_rl_discovery_results.json')

    args = parser.parse_args()

    evaluator = DiscoveryModeEvaluator()

    if args.query_id:
        queries = [q for q in DISCOVERY_QUERIES if q['query_id'] == args.query_id]
        if not queries:
            logger.error(f"Query ID {args.query_id} not found")
            return
        result = evaluator.evaluate_query(queries[0])
        results = {"individual_results": [result]}
    else:
        results = evaluator.evaluate_all()


    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output}")


    if "summary" in results:
        print("\n" + "="*80)
        print("DISCOVERY MODE EVALUATION SUMMARY")
        print("="*80)
        print(f"\nTotal Queries: {results['total_queries']}")
        print(f"\nAverage MITRE Techniques per Query:")
        print(f"  RAG: {results['summary']['avg_mitre_per_query']['rag']}")
        print(f"  Non-RAG: {results['summary']['avg_mitre_per_query']['non_rag']}")
        print(f"\nAverage CVEs per Query:")
        print(f"  RAG: {results['summary']['avg_cves_per_query']['rag']}")
        print(f"  Non-RAG: {results['summary']['avg_cves_per_query']['non_rag']}")
        print(f"\nRAG More Comprehensive: {results['summary']['rag_more_comprehensive_count']}/{results['total_queries']} ({results['summary']['rag_more_comprehensive_percentage']}%)")
        print("="*80)

if __name__ == "__main__":
    main()
