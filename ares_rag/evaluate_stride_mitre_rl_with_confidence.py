"""
Discovery Mode with Confidence Scoring for RL Action Selection
Generates mappings freely, scores them by confidence, selects top 6 for simulation
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple
from loguru import logger
from datetime import datetime
from evaluation.rag_evaluator import RAGEvaluator

# Same queries as discovery mode
from evaluate_stride_mitre_rl_discovery import DISCOVERY_QUERIES

# ---------------------------------------------------------------------------
# Weight-sensitivity analysis
#
# calculate_confidence_score() below scores six dimensions under a single
# fixed point allocation (see _COMPONENT_CEILINGS): CVE citation accuracy
# (0-25), MITRE technique coverage (0-20), RL action specificity (0-15),
# protocol specificity (0-10), context document utilization (0-20), and
# structured format adherence (0-5), plus an unweighted hallucination
# penalty. To verify the RAG-vs-non-RAG ordering is not an artifact of this
# particular allocation, we re-express each stored component as a 0-1
# quality fraction (score / its ceiling) and re-apply three published point
# schemes that each sum to 100. Reweighting only needs the already-computed
# per-query component breakdown (from a prior evaluate_all_and_select_actions
# run, cached in top_rl_actions_for_simulation.json) -- it does not re-query
# the LLM, so all three schemes are evaluated on identical underlying data.
WEIGHT_SCHEMES: Dict[str, Dict[str, float]] = {
    "Primary (operational)": {
        "cve": 25, "mitre": 25, "rl_action": 20,
        "protocol": 15, "context": 10, "structure": 5,
    },
    "Uniform (w_i = 1/6)": {
        "cve": 100 / 6, "mitre": 100 / 6, "rl_action": 100 / 6,
        "protocol": 100 / 6, "context": 100 / 6, "structure": 100 / 6,
    },
    "RL-centric (w_RL = w_proto = 0.25)": {
        "cve": 12.5, "mitre": 12.5, "rl_action": 25,
        "protocol": 25, "context": 12.5, "structure": 12.5,
    },
}

# Point ceilings implicitly used by calculate_confidence_score()'s component
# scorers -- needed to normalize a stored component score back to a 0-1
# quality fraction before re-weighting.
_COMPONENT_CEILINGS = {
    "cve": 25, "mitre": 20, "rl_action": 15,
    "protocol": 10, "context": 20, "structure": 5,
}
_COMPONENT_KEY_MAP = {
    "cve": "cve_score", "mitre": "mitre_score", "rl_action": "rl_action_score",
    "protocol": "protocol_score", "context": "context_score", "structure": "structure_score",
}


def reweight_confidence(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Recompute a total confidence score from an already-extracted component
    breakdown under a different weighting scheme, without re-querying the
    LLM or re-scoring the response. hallucination_penalty is carried
    through unweighted (it is a penalty, not one of the six weighted
    dimensions, and none of the three published schemes reweight it).
    """
    total = 0.0
    for dim, weight in weights.items():
        ceiling = _COMPONENT_CEILINGS[dim]
        raw = components.get(_COMPONENT_KEY_MAP[dim], 0) or 0
        fraction = max(min(raw / ceiling, 1.0), 0.0) if ceiling else 0.0
        total += fraction * weight
    total += components.get("hallucination_penalty", 0) or 0
    return max(round(total, 2), 0.0)


def weight_sensitivity_table(results: List[Dict]) -> List[Dict]:
    """
    For each weighting scheme, recompute the average RAG and non-RAG
    confidence score across all queries in `results` (each result must have
    the standard rag/non_rag/confidence/components shape produced by
    evaluate_query_with_confidence), and return the RAG vs non-RAG margin.
    """
    rows = []
    for scheme_name, weights in WEIGHT_SCHEMES.items():
        rag_scores = [reweight_confidence(r['rag']['confidence']['components'], weights) for r in results]
        non_rag_scores = [reweight_confidence(r['non_rag']['confidence']['components'], weights) for r in results]
        rag_avg = sum(rag_scores) / len(rag_scores) if rag_scores else 0.0
        non_rag_avg = sum(non_rag_scores) / len(non_rag_scores) if non_rag_scores else 0.0
        rows.append({
            "scheme": scheme_name,
            "rag": round(rag_avg, 1),
            "non_rag": round(non_rag_avg, 1),
            "delta": round(rag_avg - non_rag_avg, 1),
        })
    return rows


def print_weight_sensitivity_table(rows: List[Dict]) -> None:
    """Pretty-print the weight-sensitivity comparison table to stdout."""
    print("\n" + "=" * 80)
    print("WEIGHT-SENSITIVITY ANALYSIS OF THE CONFIDENCE SCORE")
    print("(RAG > Non-RAG ordering should hold under all three weighting schemes)")
    print("=" * 80)
    header = f"{'Weighting Scheme':<38}{'RAG':>8}{'Non-RAG':>10}{'Delta':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        delta_str = f"+{row['delta']:.1f}" if row['delta'] >= 0 else f"{row['delta']:.1f}"
        print(f"{row['scheme']:<38}{row['rag']:>8.1f}{row['non_rag']:>10.1f}{delta_str:>10}")
    print("=" * 80)
    min_delta = min(row['delta'] for row in rows)
    if min_delta > 0:
        print(f"\nMinimum RAG advantage across all schemes: +{min_delta:.1f} points")
        print("RAG > Non-RAG ordering holds under all three weighting schemes.")
    else:
        print(f"\nWARNING: minimum margin is {min_delta:.1f} -- ordering does NOT hold under all schemes.")


class ConfidenceBasedRLActionSelector:
    """
    Evaluate STRIDE-MITRE-RL mappings and rank by confidence
    Select top 6 RL actions for simulation implementation
    """
    
    def __init__(self):
        self.rag_evaluator = RAGEvaluator()
    
    def calculate_confidence_score(self, response: str, mapping: Dict, context_docs: List[Dict] = None) -> Dict:
        """
        Calculate confidence score for the mapping based on multiple indicators
        
        Confidence Indicators:
        1. CVE citations - VERIFIED against context docs (0-25)
        2. MITRE technique count (0-20)
        3. Specificity of RL action (0-15)
        4. Protocol-specific details (0-10)
        5. Context document usage / groundedness (0-20)
        6. Structured format (0-5)
        7. Hallucination penalty (0 to -15)
        """
        
        score_components = {}
        
        # 1. CVE Citation Score (0-25 points) - NOW VALIDATES against context
        verified_cves, unverified_cves = self._validate_cves(mapping['cves'], context_docs)
        score_components['cve_score'] = min(len(verified_cves) * 12.5, 25)  # Max 25 for 2+ verified CVEs
        
        # 2. MITRE Technique Coverage (0-20 points)
        mitre_count = len(mapping['mitre_techniques'])
        score_components['mitre_score'] = min(mitre_count * 7, 20)  # Max 20 for 3+ techniques
        
        # 3. RL Action Specificity (0-15 points)
        rl_action_specificity = self._score_rl_action_specificity(mapping['rl_actions'])
        score_components['rl_action_score'] = min(rl_action_specificity, 15)
        
        # 4. Protocol Specificity (0-10 points)
        protocol_count = len(mapping['protocols'])
        score_components['protocol_score'] = min(protocol_count * 5, 10)  # Max 10 for 2+ protocols
        
        # 5. Context Usage / Groundedness (0-20 points) - RAG's key advantage
        if context_docs:
            context_score = self._score_context_usage(response, context_docs)
            score_components['context_score'] = context_score
        else:
            score_components['context_score'] = 0
        
        # 6. Structured Format (0-5 points)
        score_components['structure_score'] = 5 if mapping['has_structured_format'] else 0
        
        # 7. Hallucination Penalty (-15 to 0) - penalize unverified CVEs
        hallucination_penalty = min(len(unverified_cves) * 5, 15)
        score_components['hallucination_penalty'] = -hallucination_penalty
        
        # Total confidence score
        total_score = max(sum(score_components.values()), 0)  # Floor at 0
        
        return {
            'total_confidence': round(total_score, 2),
            'components': score_components,
            'confidence_level': self._get_confidence_level(total_score),
            'verified_cves': list(verified_cves),
            'unverified_cves': list(unverified_cves)
        }
    
    def _validate_cves(self, cves: List[str], context_docs: List[Dict] = None) -> Tuple[Set[str], Set[str]]:
        """
        Validate CVEs against context documents from the knowledge base.
        CVEs not found in context are considered unverified (potentially hallucinated).
        
        Returns:
            Tuple of (verified_cves, unverified_cves)
        """
        if not cves:
            return set(), set()
        
        # Build set of CVEs present in context documents
        context_cves = set()
        if context_docs:
            for doc in context_docs:
                doc_id = doc.get('id', '').upper()
                # Check if doc ID itself is a CVE
                if doc_id.startswith('CVE-'):
                    context_cves.add(doc_id)
                # Check metadata and content for CVE references
                metadata = doc.get('metadata', {})
                content = doc.get('content', '')
                # Extract CVEs from content and metadata
                content_cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', (content + ' ' + str(metadata)).upper()))
                context_cves.update(content_cves)
        
        cves_upper = {c.upper() for c in cves}
        verified = cves_upper & context_cves
        unverified = cves_upper - context_cves
        
        return verified, unverified
    
    def _score_rl_action_specificity(self, rl_actions: List[str]) -> float:
        """
        Score RL action specificity based on detail level
        
        High specificity: "Communication spoofing - inject false OCPP StartTransaction messages"
        Medium specificity: "Communication spoofing"
        Low specificity: "Spoofing attack"
        """
        # Filter out garbage/markdown artifacts before scoring
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
            
            # Check for high specificity
            high_count = sum(1 for kw in specificity_keywords['high'] if kw in action_lower)
            if high_count >= 2:
                max_score = max(max_score, 15)
            elif high_count == 1:
                max_score = max(max_score, 12)
            
            # Check for medium specificity
            elif any(kw in action_lower for kw in specificity_keywords['medium']):
                max_score = max(max_score, 8)
            
            # Low specificity
            elif any(kw in action_lower for kw in specificity_keywords['low']):
                max_score = max(max_score, 4)
        
        return max_score
    
    def _clean_rl_actions(self, rl_actions: List[str]) -> List[str]:
        """
        Filter out markdown artifacts and garbage from extracted RL actions.
        Returns only meaningful action strings.
        """
        if not rl_actions:
            return []
        
        clean = []
        for action in rl_actions:
            stripped = action.strip()
            # Skip empty, too short, or markdown-only strings
            if len(stripped) < 10:
                continue
            # Skip strings that are just markdown formatting
            if stripped in ('---', '**', '***', '----', '* *', '**:**'):
                continue
            # Skip strings that are just punctuation/formatting
            if re.match(r'^[\s\*\-\#\|\:]+$', stripped):
                continue
            # Skip strings that look like metadata rather than actions
            if stripped.lower().startswith(('and relevant', 'and notes', 'the following')):
                continue
            clean.append(stripped)
        
        return clean
    
    def _score_context_usage(self, response: str, context_docs: List[Dict]) -> float:
        """
        Score how well the response uses context documents (0-20 points).
        This is RAG's key differentiator - grounded responses score higher.
        """
        if not context_docs:
            return 0
        
        docs_referenced = 0
        response_lower = response.lower()
        
        for doc in context_docs:
            doc_id = doc.get('id', '').lower()
            metadata = doc.get('metadata', {})
            
            # Check if document ID or key metadata is mentioned
            if doc_id in response_lower:
                docs_referenced += 1
                continue
            
            # Check document title reference
            title = metadata.get('title', '').lower()
            if title and len(title) > 5 and title in response_lower:
                docs_referenced += 1
                continue
            
            # Check if MITRE techniques from this doc are used
            try:
                mitre_techniques = eval(metadata.get('mitre_techniques', '[]'))
                if any(tech.lower() in response_lower for tech in mitre_techniques):
                    docs_referenced += 1
                    continue
            except:
                pass
            
            # Check if CVE IDs from this doc are referenced
            content = doc.get('content', '')
            doc_cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()))
            if any(cve.lower() in response_lower for cve in doc_cves):
                docs_referenced += 1
                continue
            
            # Check if affected systems from this doc are mentioned
            try:
                affected = eval(metadata.get('affected_systems', '[]'))
                if any(sys.lower() in response_lower for sys in affected if len(sys) > 2):
                    docs_referenced += 1
                    continue
            except:
                pass
        
        # Score based on percentage of docs used - max 20 points
        usage_rate = docs_referenced / len(context_docs)
        return round(usage_rate * 20, 2)  # Max 20 points
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level"""
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
        """Extract STRIDE-MITRE-RL mappings from response"""
        
        # Extract MITRE techniques
        mitre_techniques = set(re.findall(r'T\d{4}', response.upper()))
        
        # Extract STRIDE categories
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
        
        # Extract RL actions with better pattern matching
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
        
        # Extract CVEs
        cves = set(re.findall(r'CVE-\d{4}-\d{4,7}', response.upper()))
        
        # Extract protocols
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
        """Check if response follows structured format"""
        required_sections = ["stride", "mitre", "attack"]
        response_lower = response.lower()
        return sum(1 for section in required_sections if section in response_lower) >= 2
    
    def evaluate_query_with_confidence(self, query_data: Dict) -> Dict:
        """Evaluate query and calculate confidence scores"""
        query = query_data['query']
        
        logger.info(f"Evaluating: {query_data['query_id']}")
        
        # Get RAG response with context
        logger.info("Getting RAG response...")
        rag_result = self.rag_evaluator.get_rag_response(query, n_context_docs=10)
        rag_response = rag_result['response']
        context_docs = rag_result.get('context_docs', [])
        
        # Get non-RAG response
        logger.info("Getting non-RAG response...")
        non_rag_result = self.rag_evaluator.get_non_rag_response(query)
        non_rag_response = non_rag_result['response']
        
        # Extract mappings
        rag_mapping = self.extract_mappings(rag_response)
        non_rag_mapping = self.extract_mappings(non_rag_response)
        
        # Calculate confidence scores
        rag_confidence = self.calculate_confidence_score(rag_response, rag_mapping, context_docs)
        non_rag_confidence = self.calculate_confidence_score(non_rag_response, non_rag_mapping, None)
        
        # Determine winner
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
                    for doc in context_docs[:5]  # Store top 5 for reference
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
        """
        Select top N RL actions based on confidence scores
        Returns list of actions ready for simulation implementation
        """
        
        # Extract all RL actions with their confidence scores
        action_candidates = []
        
        for result in results:
            # Use RAG results (typically higher confidence)
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
        
        # Sort by confidence score (descending)
        action_candidates.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Select top N unique actions (avoid duplicates)
        selected_actions = []
        seen_actions = set()
        
        for candidate in action_candidates:
            action_normalized = candidate['action'].lower().strip()
            
            # Skip if we've seen a very similar action
            if action_normalized not in seen_actions:
                selected_actions.append(candidate)
                seen_actions.add(action_normalized)
                
                if len(selected_actions) >= n_actions:
                    break
        
        return selected_actions
    
    def evaluate_all_and_select_actions(self, n_actions: int = 6) -> Dict:
        """
        Evaluate all queries and select top N RL actions for simulation
        """
        results = []
        
        logger.info(f"Evaluating {len(DISCOVERY_QUERIES)} queries...")
        
        for query_data in DISCOVERY_QUERIES:
            try:
                result = self.evaluate_query_with_confidence(query_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query_data['query_id']}: {e}")
                continue
        
        # Select top RL actions
        logger.info(f"Selecting top {n_actions} RL actions...")
        top_actions = self.select_top_rl_actions(results, n_actions)
        
        # Calculate summary statistics
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
        """Calculate summary statistics"""
        
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
    parser.add_argument('--weight-sensitivity', action='store_true',
                         help='Run the weight-sensitivity analysis (Primary/Uniform/RL-centric schemes) '
                              'and print a comparison table, instead of the normal single-scheme run')
    parser.add_argument('--results-cache', type=str, default='top_rl_actions_for_simulation.json',
                         help='Cached results JSON (from a prior run) to reweight without new LLM calls')
    parser.add_argument('--regenerate', action='store_true',
                         help='Force a fresh RAG/non-RAG evaluation instead of reusing --results-cache '
                              '(used with --weight-sensitivity)')
    parser.add_argument('--weight-sensitivity-output', type=str, default='weight_sensitivity_results.json')

    args = parser.parse_args()

    selector = ConfidenceBasedRLActionSelector()

    if args.weight_sensitivity:
        if args.regenerate or not os.path.exists(args.results_cache):
            logger.info("No usable cache found (or --regenerate set); running one full evaluation pass "
                        f"over {len(DISCOVERY_QUERIES)} queries...")
            full = selector.evaluate_all_and_select_actions(n_actions=args.n_actions)
            results = full['all_results']
        else:
            logger.info(f"Reusing cached results from {args.results_cache} (no new LLM calls)...")
            with open(args.results_cache, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            results = cached['all_results']

        table = weight_sensitivity_table(results)
        print_weight_sensitivity_table(table)

        with open(args.weight_sensitivity_output, 'w', encoding='utf-8') as f:
            json.dump({"generated_from": args.results_cache if not args.regenerate else "fresh_run",
                       "n_queries": len(results), "schemes": table}, f, indent=2)
        logger.info(f"Weight-sensitivity table saved to {args.weight_sensitivity_output}")
        return

    logger.info(f"Running confidence-based evaluation to select top {args.n_actions} RL actions...")
    
    results = selector.evaluate_all_and_select_actions(n_actions=args.n_actions)
    
    # Save full results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
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
