

import re
from typing import List, Dict, Set
from loguru import logger

class EvaluationMetrics:


    @staticmethod
    def extract_mitre_techniques(text: str) -> Set[str]:

        pattern = r'T\d{4}'
        matches = re.findall(pattern, text)
        return set(matches)

    @staticmethod
    def extract_stride_categories(text: str) -> Set[str]:

        stride_categories = [
            "Spoofing", "Tampering", "Repudiation",
            "Information Disclosure", "Denial of Service",
            "Elevation of Privilege"
        ]

        found = set()
        text_lower = text.lower()

        for category in stride_categories:
            if category.lower() in text_lower:
                found.add(category)

        return found

    @staticmethod
    def extract_cve_ids(text: str) -> Set[str]:

        pattern = r'CVE-\d{4}-\d{4,7}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return set([m.upper() for m in matches])

    @staticmethod
    def count_specific_terms(text: str, terms: List[str]) -> Dict[str, int]:

        text_lower = text.lower()
        counts = {}

        for term in terms:
            counts[term] = text_lower.count(term.lower())

        return counts

    @staticmethod
    def calculate_specificity_score(text: str) -> Dict[str, any]:


        technical_terms = [
            'vulnerability', 'exploit', 'attack vector', 'mitigation',
            'detection', 'cvss', 'severity', 'authentication',
            'encryption', 'protocol', 'configuration', 'patch',
            'firewall', 'monitoring', 'access control', 'network segmentation'
        ]

        evse_specific_terms = [
            'evse', 'ocpp', 'charging station', 'ev charging',
            'iso 15118', 'charging management', 'agc', 'dms',
            'scada', 'ics', 'power grid', 'distribution system'
        ]

        technical_counts = EvaluationMetrics.count_specific_terms(text, technical_terms)
        evse_counts = EvaluationMetrics.count_specific_terms(text, evse_specific_terms)

        total_technical = sum(technical_counts.values())
        total_evse = sum(evse_counts.values())

        return {
            'technical_term_count': total_technical,
            'evse_specific_count': total_evse,
            'total_specificity_score': total_technical + total_evse,
            'technical_terms': technical_counts,
            'evse_terms': evse_counts
        }

    @staticmethod
    def calculate_completeness_score(text: str, expected_sections: List[str]) -> Dict[str, any]:

        text_lower = text.lower()
        found_sections = {}

        for section in expected_sections:
            found_sections[section] = section.lower() in text_lower

        completeness_percentage = (sum(found_sections.values()) / len(expected_sections)) * 100

        return {
            'found_sections': found_sections,
            'completeness_percentage': completeness_percentage,
            'missing_sections': [s for s, found in found_sections.items() if not found]
        }

    @staticmethod
    def calculate_actionability_score(text: str) -> Dict[str, any]:

        actionable_keywords = [
            'implement', 'configure', 'enable', 'disable', 'update',
            'patch', 'monitor', 'restrict', 'enforce', 'validate',
            'should', 'must', 'recommend', 'ensure', 'verify'
        ]

        mitigation_indicators = [
            'mitigation', 'countermeasure', 'defense', 'protection',
            'prevention', 'remediation', 'fix', 'solution'
        ]

        actionable_counts = EvaluationMetrics.count_specific_terms(text, actionable_keywords)
        mitigation_counts = EvaluationMetrics.count_specific_terms(text, mitigation_indicators)

        total_actionable = sum(actionable_counts.values())
        total_mitigation = sum(mitigation_counts.values())

        return {
            'actionable_term_count': total_actionable,
            'mitigation_term_count': total_mitigation,
            'actionability_score': total_actionable + total_mitigation,
            'has_recommendations': total_actionable > 0 or total_mitigation > 0
        }

    @staticmethod
    def compare_responses(rag_response: str, non_rag_response: str,
                         ground_truth: Dict = None) -> Dict[str, any]:


        rag_mitre = EvaluationMetrics.extract_mitre_techniques(rag_response)
        non_rag_mitre = EvaluationMetrics.extract_mitre_techniques(non_rag_response)

        rag_stride = EvaluationMetrics.extract_stride_categories(rag_response)
        non_rag_stride = EvaluationMetrics.extract_stride_categories(non_rag_response)

        rag_cves = EvaluationMetrics.extract_cve_ids(rag_response)
        non_rag_cves = EvaluationMetrics.extract_cve_ids(non_rag_response)


        rag_specificity = EvaluationMetrics.calculate_specificity_score(rag_response)
        non_rag_specificity = EvaluationMetrics.calculate_specificity_score(non_rag_response)


        rag_actionability = EvaluationMetrics.calculate_actionability_score(rag_response)
        non_rag_actionability = EvaluationMetrics.calculate_actionability_score(non_rag_response)


        rag_length = len(rag_response)
        non_rag_length = len(non_rag_response)

        comparison = {
            'mitre_techniques': {
                'rag': list(rag_mitre),
                'non_rag': list(non_rag_mitre),
                'rag_count': len(rag_mitre),
                'non_rag_count': len(non_rag_mitre),
                'rag_advantage': len(rag_mitre) - len(non_rag_mitre)
            },
            'stride_categories': {
                'rag': list(rag_stride),
                'non_rag': list(non_rag_stride),
                'rag_count': len(rag_stride),
                'non_rag_count': len(non_rag_stride),
                'rag_advantage': len(rag_stride) - len(non_rag_stride)
            },
            'cve_references': {
                'rag': list(rag_cves),
                'non_rag': list(non_rag_cves),
                'rag_count': len(rag_cves),
                'non_rag_count': len(non_rag_cves),
                'rag_advantage': len(rag_cves) - len(non_rag_cves)
            },
            'specificity': {
                'rag': rag_specificity,
                'non_rag': non_rag_specificity,
                'rag_advantage': rag_specificity['total_specificity_score'] -
                                non_rag_specificity['total_specificity_score']
            },
            'actionability': {
                'rag': rag_actionability,
                'non_rag': non_rag_actionability,
                'rag_advantage': rag_actionability['actionability_score'] -
                                non_rag_actionability['actionability_score']
            },
            'response_length': {
                'rag': rag_length,
                'non_rag': non_rag_length,
                'rag_advantage': rag_length - non_rag_length
            }
        }


        if ground_truth:
            comparison['ground_truth_validation'] = EvaluationMetrics.validate_against_ground_truth(
                rag_response, non_rag_response, ground_truth
            )

        return comparison

    @staticmethod
    def validate_against_ground_truth(rag_response: str, non_rag_response: str,
                                     ground_truth: Dict) -> Dict[str, any]:

        validation = {}


        if 'expected_mitre' in ground_truth:
            expected_mitre = set(ground_truth['expected_mitre'])
            rag_mitre = EvaluationMetrics.extract_mitre_techniques(rag_response)
            non_rag_mitre = EvaluationMetrics.extract_mitre_techniques(non_rag_response)

            validation['mitre_accuracy'] = {
                'rag_recall': len(rag_mitre & expected_mitre) / len(expected_mitre) if expected_mitre else 0,
                'non_rag_recall': len(non_rag_mitre & expected_mitre) / len(expected_mitre) if expected_mitre else 0,
                'rag_found': list(rag_mitre & expected_mitre),
                'non_rag_found': list(non_rag_mitre & expected_mitre),
                'rag_missed': list(expected_mitre - rag_mitre),
                'non_rag_missed': list(expected_mitre - non_rag_mitre)
            }


        if 'expected_stride' in ground_truth:
            expected_stride = set(ground_truth['expected_stride'])
            rag_stride = EvaluationMetrics.extract_stride_categories(rag_response)
            non_rag_stride = EvaluationMetrics.extract_stride_categories(non_rag_response)

            validation['stride_accuracy'] = {
                'rag_recall': len(rag_stride & expected_stride) / len(expected_stride) if expected_stride else 0,
                'non_rag_recall': len(non_rag_stride & expected_stride) / len(expected_stride) if expected_stride else 0,
                'rag_found': list(rag_stride & expected_stride),
                'non_rag_found': list(non_rag_stride & expected_stride)
            }


        if 'expected_cves' in ground_truth:
            expected_cves = set(ground_truth['expected_cves'])
            rag_cves = EvaluationMetrics.extract_cve_ids(rag_response)
            non_rag_cves = EvaluationMetrics.extract_cve_ids(non_rag_response)

            validation['cve_accuracy'] = {
                'rag_recall': len(rag_cves & expected_cves) / len(expected_cves) if expected_cves else 0,
                'non_rag_recall': len(non_rag_cves & expected_cves) / len(expected_cves) if expected_cves else 0,
                'rag_found': list(rag_cves & expected_cves),
                'non_rag_found': list(non_rag_cves & expected_cves)
            }

        return validation

    @staticmethod
    def generate_evaluation_report(comparison: Dict) -> str:

        report = []
        report.append("=" * 80)
        report.append("RAG EVALUATION REPORT")
        report.append("=" * 80)


        report.append("\n1. MITRE ATT&CK Techniques:")
        report.append(f"   RAG: {comparison['mitre_techniques']['rag_count']} techniques")
        report.append(f"   Non-RAG: {comparison['mitre_techniques']['non_rag_count']} techniques")
        report.append(f"   Advantage: +{comparison['mitre_techniques']['rag_advantage']} techniques")
        if comparison['mitre_techniques']['rag']:
            report.append(f"   RAG found: {', '.join(comparison['mitre_techniques']['rag'])}")


        report.append("\n2. STRIDE Categories:")
        report.append(f"   RAG: {comparison['stride_categories']['rag_count']} categories")
        report.append(f"   Non-RAG: {comparison['stride_categories']['non_rag_count']} categories")
        report.append(f"   Advantage: +{comparison['stride_categories']['rag_advantage']} categories")
        if comparison['stride_categories']['rag']:
            report.append(f"   RAG found: {', '.join(comparison['stride_categories']['rag'])}")


        report.append("\n3. CVE References:")
        report.append(f"   RAG: {comparison['cve_references']['rag_count']} CVEs")
        report.append(f"   Non-RAG: {comparison['cve_references']['non_rag_count']} CVEs")
        report.append(f"   Advantage: +{comparison['cve_references']['rag_advantage']} CVEs")


        report.append("\n4. Specificity Score:")
        report.append(f"   RAG: {comparison['specificity']['rag']['total_specificity_score']}")
        report.append(f"   Non-RAG: {comparison['specificity']['non_rag']['total_specificity_score']}")
        report.append(f"   Advantage: +{comparison['specificity']['rag_advantage']}")


        report.append("\n5. Actionability Score:")
        report.append(f"   RAG: {comparison['actionability']['rag']['actionability_score']}")
        report.append(f"   Non-RAG: {comparison['actionability']['non_rag']['actionability_score']}")
        report.append(f"   Advantage: +{comparison['actionability']['rag_advantage']}")


        if 'ground_truth_validation' in comparison:
            report.append("\n6. Ground Truth Validation:")
            gt = comparison['ground_truth_validation']

            if 'mitre_accuracy' in gt:
                report.append(f"   MITRE Recall - RAG: {gt['mitre_accuracy']['rag_recall']:.2%}, "
                            f"Non-RAG: {gt['mitre_accuracy']['non_rag_recall']:.2%}")

            if 'stride_accuracy' in gt:
                report.append(f"   STRIDE Recall - RAG: {gt['stride_accuracy']['rag_recall']:.2%}, "
                            f"Non-RAG: {gt['stride_accuracy']['non_rag_recall']:.2%}")

        report.append("\n" + "=" * 80)


        advantages = 0
        if comparison['mitre_techniques']['rag_advantage'] > 0:
            advantages += 1
        if comparison['stride_categories']['rag_advantage'] > 0:
            advantages += 1
        if comparison['specificity']['rag_advantage'] > 0:
            advantages += 1
        if comparison['actionability']['rag_advantage'] > 0:
            advantages += 1

        report.append(f"\nRAG Advantages: {advantages}/4 metrics")

        if advantages >= 3:
            report.append(" RAG significantly improves response quality")
        elif advantages >= 2:
            report.append("~ RAG moderately improves response quality")
        else:
            report.append(" RAG shows limited improvement")

        report.append("=" * 80)

        return "\n".join(report)
