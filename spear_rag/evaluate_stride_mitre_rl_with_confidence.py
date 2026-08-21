

import json
import re
from typing import Dict, List, Set, Tuple
from loguru import logger
from datetime import datetime
from evaluation.rag_evaluator import RAGEvaluator


from evaluate_stride_mitre_rl_discovery import DISCOVERY_QUERIES

class ConfidenceBasedRLActionSelector:


    def __init__(self):
        self.rag_evaluator = RAGEvaluator()

    def calculate_confidence_score(self, response: str, mapping: Dict, context_docs: List[Dict] = None) -> Dict:


        score_components = {}


        verified_cves, unverified_cves = self._validate_cves(mapping['cves'], context_docs)
        score_components['cve_score'] = min(len(verified_cves) * 12.5, 25)


        mitre_count = len(mapping['mitre_techniques'])
        score_components['mitre_score'] = min(mitre_count * 7, 20)


        rl_action_specificity = self._score_rl_action_specificity(mapping['rl_actions'])
        score_components['rl_action_score'] = min(rl_action_specificity, 15)


        protocol_count = len(mapping['protocols'])
        score_components['protocol_score'] = min(protocol_count * 5, 10)


        if context_docs:
            context_score = self._score_context_usage(response, context_docs)
            score_components['context_score'] = context_score
        else:
            score_components['context_score'] = 0


        score_components['structure_score'] = 5 if mapping['has_structured_format'] else 0


        hallucination_penalty = min(len(unverified_cves) * 5, 15)
        score_components['hallucination_penalty'] = -hallucination_penalty


        total_score = max(sum(score_components.values()), 0)

        return {
            'total_confidence': round(total_score, 2),
            'components': score_components,
            'confidence_level': self._get_confidence_level(total_score),
            'verified_cves': list(verified_cves),
            'unverified_cves': list(unverified_cves)
        }

    def _validate_cves(self, cves: List[str], context_docs: List[Dict] = None) -> Tuple[Set[str], Set[str]]:

        if not cves:
            return set(), set()


        context_cves = set()
        if context_docs:
            for doc in context_docs:
                doc_id = doc.get('id', '').upper()

                if doc_id.startswith('CVE-'):
                    context_cves.add(doc_id)

                metadata = doc.get('metadata', {})
                content = doc.get('content', '')

                content_cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', (content + ' ' + str(metadata)).upper()))
                context_cves.update(content_cves)

        cves_upper = {c.upper() for c in cves}
        verified = cves_upper & context_cves
        unverified = cves_upper - context_cves

        return verified, unverified

    def _score_rl_action_specificity(self, rl_actions: List[str]) -> float:


        clean_actions = self._clean_rl_actions(rl_actions)

        if not clean_actions:
            return 0

        specificity_keywords = {
            'high': ['inject', 'manipulate', 'modify', 'intercept', 'forge', 'replay', 'parameter', 'message', 'packet'],
            'medium': ['spoofing', 'tampering', 'disruption', 'manipulation', 'injection'],
            'low': ['attack', 'exploit', 'compromise']
        }

        max_score = 0
        for action in clean_actions:
            action_lower = action.lower()


            high_count = sum(1 for kw in specificity_keywords['high'] if kw in action_lower)
            if high_count >= 2:
                max_score = max(max_score, 15)
            elif high_count == 1:
                max_score = max(max_score, 12)


            elif any(kw in action_lower for kw in specificity_keywords['medium']):
                max_score = max(max_score, 8)


            elif any(kw in action_lower for kw in specificity_keywords['low']):
                max_score = max(max_score, 4)

        return max_score

    def _clean_rl_actions(self, rl_actions: List[str]) -> List[str]:

        if not rl_actions:
            return []

        clean = []
        for action in rl_actions:
            stripped = action.strip()

            if len(stripped) < 10:
                continue

            if stripped in ('---', '**', '***', '----', '* *', '**:**'):
                continue

            if re.match(r'^[\s\*\-\#\|\:]+$', stripped):
                continue

            if stripped.lower().startswith(('and relevant', 'and notes', 'the following')):
                continue
            clean.append(stripped)

        return clean

    def _score_context_usage(self, response: str, context_docs: List[Dict]) -> float:

        if not context_docs:
            return 0

        docs_referenced = 0
        response_lower = response.lower()

        for doc in context_docs:
            doc_id = doc.get('id', '').lower()
            metadata = doc.get('metadata', {})


            if doc_id in response_lower:
                docs_referenced += 1
                continue


            title = metadata.get('title', '').lower()
            if title and len(title) > 5 and title in response_lower:
                docs_referenced += 1
                continue


            try:
                mitre_techniques = eval(metadata.get('mitre_techniques', '[]'))
                if any(tech.lower() in response_lower for tech in mitre_techniques):
                    docs_referenced += 1
                    continue
            except:
                pass


            content = doc.get('content', '')
            doc_cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()))
            if any(cve.lower() in response_lower for cve in doc_cves):
                docs_referenced += 1
                continue


            try:
                affected = eval(metadata.get('affected_systems', '[]'))
                if any(sys.lower() in response_lower for sys in affected if len(sys) > 2):
                    docs_referenced += 1
                    continue
            except:
                pass


        usage_rate = docs_referenced / len(context_docs)
        return round(usage_rate * 20, 2)

    def _get_confidence_level(self, score: float) -> str:

        if score >= 80:
            return "Very High"
        elif score >= 65:
            return "High"
        elif score >= 50:
            return "Medium"
        elif score >= 35:
            return "Low"
        else:
            return "Very Low"

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
            r'rl (?:attack )?action[s]?:?\s*(.{15,}?)(?:\n|$)',
            r'attack action[s]?:?\s*(.{15,}?)(?:\n|$)',
            r'suggested action[s]?:?\s*(.{15,}?)(?:\n|$)',
            r'projected action[s]?:?\s*(.{15,}?)(?:\n|$)',
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

    def evaluate_query_with_confidence(self, query_data: Dict) -> Dict:

        query = query_data['query']

        logger.info(f"Evaluating: {query_data['query_id']}")


        logger.info("Getting RAG response...")
        rag_result = self.rag_evaluator.get_rag_response(query, n_context_docs=10)
        rag_response = rag_result['response']
        context_docs = rag_result.get('context_docs', [])


        logger.info("Getting non-RAG response...")
        non_rag_result = self.rag_evaluator.get_non_rag_response(query)
        non_rag_response = non_rag_result['response']


        rag_mapping = self.extract_mappings(rag_response)
        non_rag_mapping = self.extract_mappings(non_rag_response)


        rag_confidence = self.calculate_confidence_score(rag_response, rag_mapping, context_docs)
        non_rag_confidence = self.calculate_confidence_score(non_rag_response, non_rag_mapping, None)


        winner = "RAG" if rag_confidence['total_confidence'] > non_rag_confidence['total_confidence'] else "Non-RAG"
        confidence_advantage = abs(rag_confidence['total_confidence'] - non_rag_confidence['total_confidence'])

        return {
            "query_id": query_data["query_id"],
            "query": query,
            "communication_link": query_data["communication_link"],
            "focus_area": query_data["focus_area"],
            "rag": {
                "response": rag_response,
                "mapping": rag_mapping,
                "confidence": rag_confidence,
                "context_docs_used": len(context_docs),
                "context_docs_summary": [
                    {
                        "id": doc.get('id', ''),
                        "type": doc.get('metadata', {}).get('type', ''),
                        "title": doc.get('metadata', {}).get('title', ''),
                    }
                    for doc in context_docs[:5]
                ]
            },
            "non_rag": {
                "response": non_rag_response,
                "mapping": non_rag_mapping,
                "confidence": non_rag_confidence
            },
            "winner": winner,
            "confidence_advantage": round(confidence_advantage, 2)
        }

    def select_top_rl_actions(self, results: List[Dict], n_actions: int = 6) -> List[Dict]:


        action_candidates = []

        for result in results:

            rag_mapping = result['rag']['mapping']
            rag_confidence = result['rag']['confidence']

            clean_actions = self._clean_rl_actions(rag_mapping['rl_actions'])
            for rl_action in clean_actions:
                action_candidates.append({
                    'action': rl_action,
                    'query_id': result['query_id'],
                    'communication_link': result['communication_link'],
                    'stride_categories': rag_mapping['stride_categories'],
                    'mitre_techniques': rag_mapping['mitre_techniques'],
                    'cves': rag_mapping['cves'],
                    'protocols': rag_mapping['protocols'],
                    'confidence_score': rag_confidence['total_confidence'],
                    'confidence_level': rag_confidence['confidence_level'],
                    'confidence_components': rag_confidence['components']
                })


        action_candidates.sort(key=lambda x: x['confidence_score'], reverse=True)


        selected_actions = []
        seen_actions = set()

        for candidate in action_candidates:
            action_normalized = candidate['action'].lower().strip()


            if action_normalized not in seen_actions:
                selected_actions.append(candidate)
                seen_actions.add(action_normalized)

                if len(selected_actions) >= n_actions:
                    break

        return selected_actions

    def evaluate_all_and_select_actions(self, n_actions: int = 6) -> Dict:

        results = []

        logger.info(f"Evaluating {len(DISCOVERY_QUERIES)} queries...")

        for query_data in DISCOVERY_QUERIES:
            try:
                result = self.evaluate_query_with_confidence(query_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query_data['query_id']}: {e}")
                continue


        logger.info(f"Selecting top {n_actions} RL actions...")
        top_actions = self.select_top_rl_actions(results, n_actions)


        summary = self._calculate_summary(results, top_actions)

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "confidence_based_selection",
            "total_queries": len(results),
            "actions_requested": n_actions,
            "actions_selected": len(top_actions),
            "summary": summary,
            "top_rl_actions": top_actions,
            "all_results": results
        }

    def _calculate_summary(self, results: List[Dict], top_actions: List[Dict]) -> Dict:


        rag_wins = sum(1 for r in results if r['winner'] == 'RAG')
        avg_rag_confidence = sum(r['rag']['confidence']['total_confidence'] for r in results) / len(results) if results else 0
        avg_non_rag_confidence = sum(r['non_rag']['confidence']['total_confidence'] for r in results) / len(results) if results else 0

        return {
            "rag_wins": rag_wins,
            "non_rag_wins": len(results) - rag_wins,
            "avg_confidence": {
                "rag": round(avg_rag_confidence, 2),
                "non_rag": round(avg_non_rag_confidence, 2),
                "advantage": round(avg_rag_confidence - avg_non_rag_confidence, 2)
            },
            "top_actions_confidence_range": {
                "highest": round(top_actions[0]['confidence_score'], 2) if top_actions else 0,
                "lowest": round(top_actions[-1]['confidence_score'], 2) if top_actions else 0
            },
            "stride_coverage": list(set(cat for action in top_actions for cat in action['stride_categories'])),
            "protocol_coverage": list(set(proto for action in top_actions for proto in action['protocols']))
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Select top RL actions based on confidence scores')
    parser.add_argument('--n-actions', type=int, default=6, help='Number of RL actions to select')
    parser.add_argument('--output', type=str, default='top_rl_actions_for_simulation.json')

    args = parser.parse_args()

    selector = ConfidenceBasedRLActionSelector()

    logger.info(f"Running confidence-based evaluation to select top {args.n_actions} RL actions...")

    results = selector.evaluate_all_and_select_actions(n_actions=args.n_actions)


    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output}")


    print("\n" + "="*80)
    print("TOP RL ACTIONS FOR SIMULATION (CONFIDENCE-BASED SELECTION)")
    print("="*80)

    print(f"\nTotal Queries Evaluated: {results['total_queries']}")
    print(f"RAG Wins: {results['summary']['rag_wins']}/{results['total_queries']}")
    print(f"\nAverage Confidence Scores:")
    print(f"  RAG: {results['summary']['avg_confidence']['rag']}")
    print(f"  Non-RAG: {results['summary']['avg_confidence']['non_rag']}")
    print(f"  RAG Advantage: +{results['summary']['avg_confidence']['advantage']}")

    print(f"\n{'='*80}")
    print(f"TOP {args.n_actions} RL ACTIONS SELECTED FOR SIMULATION")
    print("="*80)

    for i, action in enumerate(results['top_rl_actions'], 1):
        print(f"\n{i}. {action['action']}")
        print(f"   Confidence: {action['confidence_score']:.2f}/100 ({action['confidence_level']})")
        print(f"   Communication Link: {action['communication_link']}")
        print(f"   STRIDE: {', '.join(action['stride_categories'])}")
        print(f"   MITRE: {', '.join(action['mitre_techniques'][:3])}{'...' if len(action['mitre_techniques']) > 3 else ''}")
        if action['cves']:
            print(f"   CVEs: {', '.join(action['cves'])}")
        print(f"   Protocols: {', '.join(action['protocols'])}")

    print("\n" + "="*80)
    print(f"\nSTRIDE Coverage: {', '.join(results['summary']['stride_coverage'])}")
    print(f"Protocol Coverage: {', '.join(results['summary']['protocol_coverage'])}")
    print("="*80)

if __name__ == "__main__":
    main()
