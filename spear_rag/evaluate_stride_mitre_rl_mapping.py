"""
Evaluation script for STRIDE-MITRE-RL Action Mapping
Compares RAG vs Non-RAG responses for mapping accuracy
"""

import json
import re
from typing import Dict, List, Set
from loguru import logger
from datetime import datetime
from evaluation.rag_evaluator import RAGEvaluator
from stride_mitre_rl_mapping_queries import STRIDE_MITRE_RL_MAPPING_QUERIES

class STRIDEMITREMappingEvaluator:
    """Evaluator for STRIDE-MITRE-RL action mapping accuracy"""
    
    def __init__(self):
        self.rag_evaluator = RAGEvaluator()
        
    def extract_mitre_techniques(self, response: str) -> Set[str]:
        """Extract MITRE technique IDs from response"""
        # Match T followed by 4 digits
        techniques = re.findall(r'T\d{4}', response.upper())
        return set(techniques)
    
    def extract_stride_categories(self, response: str) -> Set[str]:
        """Extract STRIDE categories from response"""
        stride_categories = [
            "Spoofing", "Tampering", "Repudiation", 
            "Information Disclosure", "Denial of Service", 
            "Elevation of Privilege"
        ]
        found = set()
        response_lower = response.lower()
        
        for category in stride_categories:
            if category.lower() in response_lower:
                found.add(category)
        
        return found
    
    def extract_rl_actions(self, response: str) -> Set[str]:
        """Extract RL action keywords from response"""
        rl_actions = [
            "communication spoofing", "data injection", 
            "protocol manipulation", "voltage manipulation",
            "power disruption", "current injection"
        ]
        found = set()
        response_lower = response.lower()
        
        for action in rl_actions:
            if action.lower() in response_lower:
                found.add(action)
        
        return found
    
    def extract_protocols(self, response: str) -> Set[str]:
        """Extract protocol names from response"""
        protocols = ["OCPP", "DNP3", "TCP/IP", "Modbus", "HTTPS", "ISO 15118"]
        found = set()
        response_upper = response.upper()
        
        for protocol in protocols:
            if protocol.upper() in response_upper:
                found.add(protocol)
        
        return found
    
    def extract_cves(self, response: str) -> Set[str]:
        """Extract CVE IDs from response"""
        cves = re.findall(r'CVE-\d{4}-\d{4,7}', response.upper())
        return set(cves)
    
    def calculate_mapping_accuracy(
        self, 
        response: str, 
        ground_truth: Dict
    ) -> Dict:
        """
        Calculate accuracy of STRIDE-MITRE-RL mapping
        
        Returns:
            Dictionary with precision, recall, F1 scores for each component
        """
        # Extract from response
        extracted_mitre = self.extract_mitre_techniques(response)
        extracted_stride = self.extract_stride_categories(response)
        extracted_rl = self.extract_rl_actions(response)
        extracted_protocols = self.extract_protocols(response)
        extracted_cves = self.extract_cves(response)
        
        # Ground truth
        gt_mitre = set(ground_truth.get('mitre_techniques', []))
        gt_stride = set(ground_truth.get('stride', []))
        gt_rl = set([ground_truth.get('projected_rl_action', '')])
        gt_protocols = set(ground_truth.get('protocols', []))
        gt_cves = set(ground_truth.get('expected_cves', []))
        
        def calc_metrics(extracted: Set, ground_truth: Set) -> Dict:
            """Calculate precision, recall, F1"""
            if not ground_truth:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'count': 0}
            
            true_positives = len(extracted & ground_truth)
            false_positives = len(extracted - ground_truth)
            false_negatives = len(ground_truth - extracted)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1': round(f1 * 100, 2),
                'count': true_positives,
                'extracted': list(extracted),
                'ground_truth': list(ground_truth),
                'correct': list(extracted & ground_truth),
                'missed': list(ground_truth - extracted),
                'extra': list(extracted - ground_truth)
            }
        
        return {
            'mitre_techniques': calc_metrics(extracted_mitre, gt_mitre),
            'stride_categories': calc_metrics(extracted_stride, gt_stride),
            'rl_actions': calc_metrics(extracted_rl, gt_rl),
            'protocols': calc_metrics(extracted_protocols, gt_protocols),
            'cves': calc_metrics(extracted_cves, gt_cves),
            'overall_f1': 0.0  # Will be calculated below
        }
    
    def evaluate_query(self, query_data: Dict) -> Dict:
        """Evaluate a single query with both RAG and non-RAG"""
        query = query_data['query']
        ground_truth = query_data['ground_truth']
        
        logger.info(f"Evaluating: {query_data['query_id']}")
        
        # Get RAG response
        logger.info("Getting RAG response...")
        rag_response = self.rag_evaluator.get_rag_response(query)
        
        # Get non-RAG response
        logger.info("Getting non-RAG response...")
        non_rag_response = self.rag_evaluator.get_non_rag_response(query)
        
        # Calculate accuracy for both
        rag_accuracy = self.calculate_mapping_accuracy(rag_response, ground_truth)
        non_rag_accuracy = self.calculate_mapping_accuracy(non_rag_response, ground_truth)
        
        # Calculate overall F1 scores
        rag_f1_scores = [
            rag_accuracy['mitre_techniques']['f1'],
            rag_accuracy['stride_categories']['f1'],
            rag_accuracy['rl_actions']['f1'],
            rag_accuracy['protocols']['f1']
        ]
        rag_accuracy['overall_f1'] = round(sum(rag_f1_scores) / len(rag_f1_scores), 2)
        
        non_rag_f1_scores = [
            non_rag_accuracy['mitre_techniques']['f1'],
            non_rag_accuracy['stride_categories']['f1'],
            non_rag_accuracy['rl_actions']['f1'],
            non_rag_accuracy['protocols']['f1']
        ]
        non_rag_accuracy['overall_f1'] = round(sum(non_rag_f1_scores) / len(non_rag_f1_scores), 2)
        
        return {
            'query_id': query_data['query_id'],
            'query': query,
            'communication_link': query_data['communication_link'],
            'stride_category': query_data['stride_category'],
            'ground_truth': ground_truth,
            'rag_response': rag_response,
            'non_rag_response': non_rag_response,
            'rag_accuracy': rag_accuracy,
            'non_rag_accuracy': non_rag_accuracy,
            'improvement': {
                'overall_f1': round(rag_accuracy['overall_f1'] - non_rag_accuracy['overall_f1'], 2),
                'mitre_f1': round(rag_accuracy['mitre_techniques']['f1'] - non_rag_accuracy['mitre_techniques']['f1'], 2),
                'stride_f1': round(rag_accuracy['stride_categories']['f1'] - non_rag_accuracy['stride_categories']['f1'], 2),
                'rl_f1': round(rag_accuracy['rl_actions']['f1'] - non_rag_accuracy['rl_actions']['f1'], 2),
                'cve_f1': round(rag_accuracy['cves']['f1'] - non_rag_accuracy['cves']['f1'], 2)
            }
        }
    
    def evaluate_all_queries(self, queries: List[Dict] = None) -> Dict:
        """Evaluate all queries and generate comprehensive report"""
        if queries is None:
            queries = STRIDE_MITRE_RL_MAPPING_QUERIES
        
        results = []
        
        for query_data in queries:
            try:
                result = self.evaluate_query(query_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query_data['query_id']}: {e}")
                continue
        
        # Calculate aggregated metrics
        aggregated = self.calculate_aggregated_metrics(results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'aggregated_metrics': aggregated,
            'individual_results': results
        }
    
    def calculate_aggregated_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregated metrics across all queries"""
        if not results:
            return {}
        
        # Aggregate F1 scores
        rag_overall_f1 = [r['rag_accuracy']['overall_f1'] for r in results]
        non_rag_overall_f1 = [r['non_rag_accuracy']['overall_f1'] for r in results]
        
        rag_mitre_f1 = [r['rag_accuracy']['mitre_techniques']['f1'] for r in results]
        non_rag_mitre_f1 = [r['non_rag_accuracy']['mitre_techniques']['f1'] for r in results]
        
        rag_stride_f1 = [r['rag_accuracy']['stride_categories']['f1'] for r in results]
        non_rag_stride_f1 = [r['non_rag_accuracy']['stride_categories']['f1'] for r in results]
        
        rag_rl_f1 = [r['rag_accuracy']['rl_actions']['f1'] for r in results]
        non_rag_rl_f1 = [r['non_rag_accuracy']['rl_actions']['f1'] for r in results]
        
        rag_cve_f1 = [r['rag_accuracy']['cves']['f1'] for r in results]
        non_rag_cve_f1 = [r['non_rag_accuracy']['cves']['f1'] for r in results]
        
        def avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0.0
        
        return {
            'rag_metrics': {
                'avg_overall_f1': avg(rag_overall_f1),
                'avg_mitre_f1': avg(rag_mitre_f1),
                'avg_stride_f1': avg(rag_stride_f1),
                'avg_rl_f1': avg(rag_rl_f1),
                'avg_cve_f1': avg(rag_cve_f1)
            },
            'non_rag_metrics': {
                'avg_overall_f1': avg(non_rag_overall_f1),
                'avg_mitre_f1': avg(non_rag_mitre_f1),
                'avg_stride_f1': avg(non_rag_stride_f1),
                'avg_rl_f1': avg(non_rag_rl_f1),
                'avg_cve_f1': avg(non_rag_cve_f1)
            },
            'improvement': {
                'overall_f1': round(avg(rag_overall_f1) - avg(non_rag_overall_f1), 2),
                'mitre_f1': round(avg(rag_mitre_f1) - avg(non_rag_mitre_f1), 2),
                'stride_f1': round(avg(rag_stride_f1) - avg(non_rag_stride_f1), 2),
                'rl_f1': round(avg(rag_rl_f1) - avg(non_rag_rl_f1), 2),
                'cve_f1': round(avg(rag_cve_f1) - avg(non_rag_cve_f1), 2)
            },
            'rag_wins': sum(1 for r in results if r['rag_accuracy']['overall_f1'] > r['non_rag_accuracy']['overall_f1']),
            'non_rag_wins': sum(1 for r in results if r['non_rag_accuracy']['overall_f1'] > r['rag_accuracy']['overall_f1']),
            'ties': sum(1 for r in results if r['rag_accuracy']['overall_f1'] == r['non_rag_accuracy']['overall_f1'])
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate STRIDE-MITRE-RL Mapping')
    parser.add_argument('--query-id', type=str, help='Evaluate specific query by ID')
    parser.add_argument('--stride-category', type=str, help='Evaluate queries for specific STRIDE category')
    parser.add_argument('--output', type=str, default='stride_mitre_rl_evaluation_results.json', 
                       help='Output file for results')
    
    args = parser.parse_args()
    
    evaluator = STRIDEMITREMappingEvaluator()
    
    # Select queries to evaluate
    queries = STRIDE_MITRE_RL_MAPPING_QUERIES
    
    if args.query_id:
        queries = [q for q in queries if q['query_id'] == args.query_id]
        if not queries:
            logger.error(f"Query ID {args.query_id} not found")
            return
    
    if args.stride_category:
        queries = [q for q in queries if args.stride_category in q['ground_truth']['stride']]
        if not queries:
            logger.error(f"No queries found for STRIDE category {args.stride_category}")
            return
    
    logger.info(f"Evaluating {len(queries)} queries...")
    
    # Run evaluation
    results = evaluator.evaluate_all_queries(queries)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("STRIDE-MITRE-RL MAPPING EVALUATION SUMMARY")
    print("="*80)
    
    agg = results['aggregated_metrics']
    
    print(f"\nTotal Queries Evaluated: {results['total_queries']}")
    print(f"RAG Wins: {agg['rag_wins']}")
    print(f"Non-RAG Wins: {agg['non_rag_wins']}")
    print(f"Ties: {agg['ties']}")
    
    print("\n" + "-"*80)
    print("Average F1 Scores:")
    print("-"*80)
    print(f"{'Metric':<25} {'RAG':<15} {'Non-RAG':<15} {'Improvement':<15}")
    print("-"*80)
    print(f"{'Overall F1':<25} {agg['rag_metrics']['avg_overall_f1']:<15} {agg['non_rag_metrics']['avg_overall_f1']:<15} {agg['improvement']['overall_f1']:<15}")
    print(f"{'MITRE Techniques F1':<25} {agg['rag_metrics']['avg_mitre_f1']:<15} {agg['non_rag_metrics']['avg_mitre_f1']:<15} {agg['improvement']['mitre_f1']:<15}")
    print(f"{'STRIDE Categories F1':<25} {agg['rag_metrics']['avg_stride_f1']:<15} {agg['non_rag_metrics']['avg_stride_f1']:<15} {agg['improvement']['stride_f1']:<15}")
    print(f"{'RL Actions F1':<25} {agg['rag_metrics']['avg_rl_f1']:<15} {agg['non_rag_metrics']['avg_rl_f1']:<15} {agg['improvement']['rl_f1']:<15}")
    print(f"{'CVE Citations F1':<25} {agg['rag_metrics']['avg_cve_f1']:<15} {agg['non_rag_metrics']['avg_cve_f1']:<15} {agg['improvement']['cve_f1']:<15}")
    print("="*80)

if __name__ == "__main__":
    main()
