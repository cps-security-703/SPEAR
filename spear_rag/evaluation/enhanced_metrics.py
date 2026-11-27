

import re
from typing import List, Dict, Set, Tuple
from loguru import logger

class EnhancedMetrics:


    @staticmethod
    def calculate_context_usage_score(response: str, context_docs: List[Dict]) -> Dict[str, any]:

        if not context_docs:
            return {
                'context_usage_score': 0.0,
                'documents_referenced': 0,
                'unique_facts_used': 0,
                'context_relevance': 0.0
            }

        response_lower = response.lower()


        docs_referenced = 0
        unique_facts = set()

        for doc in context_docs:
            doc_id = doc.get('id', '')
            metadata = doc.get('metadata', {})


            if doc_id.lower() in response_lower:
                docs_referenced += 1


            mitre_techniques = eval(metadata.get('mitre_techniques', '[]'))
            for technique in mitre_techniques:
                if technique.lower() in response_lower:
                    unique_facts.add(f"mitre_{technique}")
                    docs_referenced += 1
                    break


            if metadata.get('type') == 'vulnerability':
                cve_id = metadata.get('title', '')
                if cve_id and cve_id.lower() in response_lower:
                    unique_facts.add(f"cve_{cve_id}")
                    docs_referenced += 1


            stride_cats = eval(metadata.get('stride_categories', '[]'))
            for cat in stride_cats:
                if cat.lower() in response_lower:
                    unique_facts.add(f"stride_{cat}")


        docs_referenced = min(docs_referenced, len(context_docs))
        context_usage_score = docs_referenced / len(context_docs) if context_docs else 0.0

        return {
            'context_usage_score': context_usage_score,
            'documents_referenced': docs_referenced,
            'total_documents': len(context_docs),
            'unique_facts_used': len(unique_facts),
            'context_relevance': context_usage_score * 100
        }

    @staticmethod
    def calculate_technical_depth_score(response: str) -> Dict[str, any]:


        mitre_techniques = re.findall(r'T\d{4}', response)
        cve_ids = re.findall(r'CVE-\d{4}-\d{4,7}', response, re.IGNORECASE)
        cvss_scores = re.findall(r'CVSS[:\s]+(\d+\.?\d*)', response, re.IGNORECASE)


        technical_sentences = 0
        sentences = response.split('.')

        technical_indicators = [
            'exploit', 'vulnerability', 'attack', 'mitigation', 'detection',
            'authentication', 'authorization', 'encryption', 'protocol',
            'configuration', 'implementation', 'payload', 'injection'
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in technical_indicators):

                if len(sentence) > 50:
                    technical_sentences += 1


        has_structure = bool(re.search(r'\n\d+\.|\n-|\n\*', response))


        has_steps = bool(re.search(r'step \d+|first|second|third|then|next|finally', response, re.IGNORECASE))


        depth_score = 0
        depth_score += len(set(mitre_techniques)) * 10
        depth_score += len(set(cve_ids)) * 10
        depth_score += len(cvss_scores) * 5
        depth_score += technical_sentences * 2
        depth_score += 10 if has_structure else 0
        depth_score += 10 if has_steps else 0

        return {
            'technical_depth_score': depth_score,
            'unique_mitre_techniques': len(set(mitre_techniques)),
            'unique_cves': len(set(cve_ids)),
            'cvss_references': len(cvss_scores),
            'technical_sentences': technical_sentences,
            'has_structure': has_structure,
            'has_step_by_step': has_steps
        }

    @staticmethod
    def calculate_actionability_quality(response: str) -> Dict[str, any]:


        specific_actions = []


        action_patterns = [
            r'implement\s+([A-Z][^.!?]{10,80})',
            r'configure\s+([A-Z][^.!?]{10,80})',
            r'enable\s+([A-Z][^.!?]{10,80})',
            r'disable\s+([A-Z][^.!?]{10,80})',
            r'deploy\s+([A-Z][^.!?]{10,80})',
            r'install\s+([A-Z][^.!?]{10,80})',
            r'update\s+([A-Z][^.!?]{10,80})',
            r'patch\s+([A-Z][^.!?]{10,80})',
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, response)
            specific_actions.extend(matches)


        has_prioritization = bool(re.search(r'critical|high priority|medium priority|low priority', response, re.IGNORECASE))


        has_implementation = bool(re.search(r'implementation|deployment|configuration|setup', response, re.IGNORECASE))


        standards = re.findall(r'(IEC \d+|NIST|ISO \d+|IEEE \d+)', response, re.IGNORECASE)


        has_mitigation_section = bool(re.search(r'mitigation|countermeasure|defense|protection', response, re.IGNORECASE))


        quality_score = 0
        quality_score += len(specific_actions) * 5
        quality_score += 15 if has_prioritization else 0
        quality_score += 10 if has_implementation else 0
        quality_score += len(set(standards)) * 5
        quality_score += 10 if has_mitigation_section else 0

        return {
            'actionability_quality_score': quality_score,
            'specific_recommendations': len(specific_actions),
            'has_prioritization': has_prioritization,
            'has_implementation_guidance': has_implementation,
            'standards_referenced': len(set(standards)),
            'has_mitigation_section': has_mitigation_section
        }

    @staticmethod
    def calculate_framework_integration_score(response: str) -> Dict[str, any]:


        mitre_techniques = set(re.findall(r'T\d{4}', response))
        mitre_tactics = []

        tactic_keywords = {
            'Initial Access': ['initial access', 'entry point'],
            'Execution': ['execution', 'code execution'],
            'Persistence': ['persistence', 'maintain access'],
            'Privilege Escalation': ['privilege escalation', 'elevated privilege'],
            'Defense Evasion': ['defense evasion', 'evade detection'],
            'Discovery': ['discovery', 'reconnaissance'],
            'Lateral Movement': ['lateral movement', 'move laterally'],
            'Collection': ['collection', 'data collection'],
            'Command and Control': ['command and control', 'c2', 'c&c'],
            'Inhibit Response': ['inhibit response', 'disable'],
            'Impair Process': ['impair process', 'disrupt'],
            'Impact': ['impact', 'damage', 'disruption']
        }

        response_lower = response.lower()
        for tactic, keywords in tactic_keywords.items():
            if any(kw in response_lower for kw in keywords):
                mitre_tactics.append(tactic)


        stride_categories = []
        stride_keywords = {
            'Spoofing': ['spoof', 'impersonat', 'fake identity'],
            'Tampering': ['tamper', 'modif', 'alter', 'manipulat'],
            'Repudiation': ['repudiat', 'deny', 'non-repudiation'],
            'Information Disclosure': ['information disclosure', 'data leak', 'exposure'],
            'Denial of Service': ['denial of service', 'dos', 'ddos', 'availability'],
            'Elevation of Privilege': ['elevation of privilege', 'privilege escalation', 'unauthorized access']
        }

        for category, keywords in stride_keywords.items():
            if any(kw in response_lower for kw in keywords):
                stride_categories.append(category)


        mentions_mitre = bool(re.search(r'MITRE|ATT&CK', response, re.IGNORECASE))
        mentions_stride = bool(re.search(r'STRIDE', response, re.IGNORECASE))


        integration_score = 0
        integration_score += len(mitre_techniques) * 5
        integration_score += len(mitre_tactics) * 3
        integration_score += len(stride_categories) * 4
        integration_score += 10 if mentions_mitre else 0
        integration_score += 10 if mentions_stride else 0


        if mitre_techniques and stride_categories:
            integration_score += 15

        return {
            'framework_integration_score': integration_score,
            'mitre_techniques_count': len(mitre_techniques),
            'mitre_tactics_count': len(mitre_tactics),
            'stride_categories_count': len(stride_categories),
            'mentions_mitre_framework': mentions_mitre,
            'mentions_stride_framework': mentions_stride,
            'uses_both_frameworks': bool(mitre_techniques and stride_categories)
        }

    @staticmethod
    def calculate_overall_quality_score(
        response: str,
        context_docs: List[Dict] = None,
        ground_truth: Dict = None
    ) -> Dict[str, any]:


        context_usage = EnhancedMetrics.calculate_context_usage_score(
            response, context_docs or []
        )
        technical_depth = EnhancedMetrics.calculate_technical_depth_score(response)
        actionability = EnhancedMetrics.calculate_actionability_quality(response)
        framework_integration = EnhancedMetrics.calculate_framework_integration_score(response)


        weights = {
            'context_usage': 0.25,
            'technical_depth': 0.30,
            'actionability': 0.25,
            'framework_integration': 0.20
        }

        overall_score = (
            context_usage['context_usage_score'] * 100 * weights['context_usage'] +
            technical_depth['technical_depth_score'] * weights['technical_depth'] +
            actionability['actionability_quality_score'] * weights['actionability'] +
            framework_integration['framework_integration_score'] * weights['framework_integration']
        )


        accuracy_score = 0.0
        if ground_truth:
            accuracy_metrics = EnhancedMetrics.calculate_ground_truth_accuracy(
                response, ground_truth
            )
            accuracy_score = accuracy_metrics['overall_accuracy']

            overall_score = overall_score * 0.8 + accuracy_score * 0.2

        return {
            'overall_quality_score': overall_score,
            'context_usage': context_usage,
            'technical_depth': technical_depth,
            'actionability': actionability,
            'framework_integration': framework_integration,
            'has_ground_truth': ground_truth is not None,
            'accuracy_score': accuracy_score if ground_truth else None
        }

    @staticmethod
    def calculate_ground_truth_accuracy(response: str, ground_truth: Dict) -> Dict[str, any]:

        mitre_found = set(re.findall(r'T\d{4}', response))
        stride_found = set()

        stride_categories = ['Spoofing', 'Tampering', 'Repudiation',
                            'Information Disclosure', 'Denial of Service',
                            'Elevation of Privilege']

        response_lower = response.lower()
        for cat in stride_categories:
            if cat.lower() in response_lower:
                stride_found.add(cat)


        mitre_precision = 0.0
        mitre_recall = 0.0
        stride_precision = 0.0
        stride_recall = 0.0

        if 'expected_mitre' in ground_truth:
            expected_mitre = set(ground_truth['expected_mitre'])
            if mitre_found:
                mitre_precision = len(mitre_found & expected_mitre) / len(mitre_found)
            if expected_mitre:
                mitre_recall = len(mitre_found & expected_mitre) / len(expected_mitre)

        if 'expected_stride' in ground_truth:
            expected_stride = set(ground_truth['expected_stride'])
            if stride_found:
                stride_precision = len(stride_found & expected_stride) / len(stride_found)
            if expected_stride:
                stride_recall = len(stride_found & expected_stride) / len(expected_stride)


        mitre_f1 = (2 * mitre_precision * mitre_recall / (mitre_precision + mitre_recall)) if (mitre_precision + mitre_recall) > 0 else 0
        stride_f1 = (2 * stride_precision * stride_recall / (stride_precision + stride_recall)) if (stride_precision + stride_recall) > 0 else 0


        overall_accuracy = (mitre_f1 + stride_f1) / 2 * 100

        return {
            'overall_accuracy': overall_accuracy,
            'mitre_precision': mitre_precision,
            'mitre_recall': mitre_recall,
            'mitre_f1': mitre_f1,
            'stride_precision': stride_precision,
            'stride_recall': stride_recall,
            'stride_f1': stride_f1
        }

    @staticmethod
    def compare_rag_vs_non_rag(
        rag_response: str,
        non_rag_response: str,
        rag_context_docs: List[Dict] = None,
        ground_truth: Dict = None
    ) -> Dict[str, any]:

        rag_quality = EnhancedMetrics.calculate_overall_quality_score(
            rag_response, rag_context_docs, ground_truth
        )

        non_rag_quality = EnhancedMetrics.calculate_overall_quality_score(
            non_rag_response, None, ground_truth
        )


        quality_improvement = rag_quality['overall_quality_score'] - non_rag_quality['overall_quality_score']

        context_advantage = rag_quality['context_usage']['context_usage_score'] * 100

        technical_improvement = (
            rag_quality['technical_depth']['technical_depth_score'] -
            non_rag_quality['technical_depth']['technical_depth_score']
        )

        actionability_improvement = (
            rag_quality['actionability']['actionability_quality_score'] -
            non_rag_quality['actionability']['actionability_quality_score']
        )

        framework_improvement = (
            rag_quality['framework_integration']['framework_integration_score'] -
            non_rag_quality['framework_integration']['framework_integration_score']
        )

        return {
            'rag_quality': rag_quality,
            'non_rag_quality': non_rag_quality,
            'quality_improvement': quality_improvement,
            'quality_improvement_percentage': (quality_improvement / non_rag_quality['overall_quality_score'] * 100) if non_rag_quality['overall_quality_score'] > 0 else 0,
            'context_advantage': context_advantage,
            'technical_improvement': technical_improvement,
            'actionability_improvement': actionability_improvement,
            'framework_improvement': framework_improvement,
            'rag_wins': quality_improvement > 0
        }

    @staticmethod
    def generate_enhanced_report(comparison: Dict) -> str:

        report = []
        report.append("=" * 80)
        report.append("ENHANCED RAG EVALUATION REPORT")
        report.append("=" * 80)

        rag_q = comparison['rag_quality']
        non_rag_q = comparison['non_rag_quality']


        report.append(f"\n# OVERALL QUALITY SCORE:")
        report.append(f"   RAG: {rag_q['overall_quality_score']:.1f}/100")
        report.append(f"   Non-RAG: {non_rag_q['overall_quality_score']:.1f}/100")
        report.append(f"   Improvement: {comparison['quality_improvement']:+.1f} ({comparison['quality_improvement_percentage']:+.1f}%)")


        report.append(f"\n# CONTEXT USAGE:")
        cu = rag_q['context_usage']
        report.append(f"   Documents Referenced: {cu['documents_referenced']}/{cu['total_documents']}")
        report.append(f"   Unique Facts Used: {cu['unique_facts_used']}")
        report.append(f"   Context Relevance: {cu['context_relevance']:.1f}%")


        report.append(f"\n TECHNICAL DEPTH:")
        rag_td = rag_q['technical_depth']
        non_rag_td = non_rag_q['technical_depth']
        report.append(f"   RAG Score: {rag_td['technical_depth_score']}")
        report.append(f"   Non-RAG Score: {non_rag_td['technical_depth_score']}")
        report.append(f"   MITRE Techniques: RAG={rag_td['unique_mitre_techniques']}, Non-RAG={non_rag_td['unique_mitre_techniques']}")
        report.append(f"   CVE References: RAG={rag_td['unique_cves']}, Non-RAG={non_rag_td['unique_cves']}")
        report.append(f"   Technical Sentences: RAG={rag_td['technical_sentences']}, Non-RAG={non_rag_td['technical_sentences']}")


        report.append(f"\n ACTIONABILITY QUALITY:")
        rag_aq = rag_q['actionability']
        non_rag_aq = non_rag_q['actionability']
        report.append(f"   RAG Score: {rag_aq['actionability_quality_score']}")
        report.append(f"   Non-RAG Score: {non_rag_aq['actionability_quality_score']}")
        report.append(f"   Specific Recommendations: RAG={rag_aq['specific_recommendations']}, Non-RAG={non_rag_aq['specific_recommendations']}")
        report.append(f"   Has Prioritization: RAG={rag_aq['has_prioritization']}, Non-RAG={non_rag_aq['has_prioritization']}")


        report.append(f"\n FRAMEWORK INTEGRATION:")
        rag_fi = rag_q['framework_integration']
        non_rag_fi = non_rag_q['framework_integration']
        report.append(f"   RAG Score: {rag_fi['framework_integration_score']}")
        report.append(f"   Non-RAG Score: {non_rag_fi['framework_integration_score']}")
        report.append(f"   MITRE Techniques: RAG={rag_fi['mitre_techniques_count']}, Non-RAG={non_rag_fi['mitre_techniques_count']}")
        report.append(f"   STRIDE Categories: RAG={rag_fi['stride_categories_count']}, Non-RAG={non_rag_fi['stride_categories_count']}")
        report.append(f"   Uses Both Frameworks: RAG={rag_fi['uses_both_frameworks']}, Non-RAG={non_rag_fi['uses_both_frameworks']}")


        if rag_q['has_ground_truth']:
            report.append(f"\n# GROUND TRUTH ACCURACY:")
            report.append(f"   RAG Accuracy: {rag_q['accuracy_score']:.1f}%")
            report.append(f"   Non-RAG Accuracy: {non_rag_q['accuracy_score']:.1f}%")

        report.append("\n" + "=" * 80)


        if comparison['quality_improvement'] > 20:
            verdict = " RAG SIGNIFICANTLY IMPROVES response quality"
        elif comparison['quality_improvement'] > 5:
            verdict = " RAG IMPROVES response quality"
        elif comparison['quality_improvement'] > -5:
            verdict = "~ RAG shows MARGINAL improvement"
        else:
            verdict = " RAG shows NO improvement (check configuration)"

        report.append(f"\n{verdict}")
        report.append(f"Overall Quality Improvement: {comparison['quality_improvement']:+.1f} points")
        report.append("=" * 80)

        return "\n".join(report)
