"""
Script to evaluate RAG system performance
"""

import argparse
from loguru import logger
import sys
import json

from evaluation import RAGEvaluator

def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

def create_default_test_set():
    """Create a default test set for evaluation"""
    return [
        {
            'query': 'What are the main vulnerabilities in OCPP protocol for EV charging stations?',
            'n_context_docs': 10,
            'ground_truth': {
                'expected_mitre': ['T0866', 'T0855', 'T0868'],
                'expected_stride': ['Spoofing', 'Tampering', 'Information Disclosure']
            }
        },
        {
            'query': 'How can an attacker manipulate AGC systems in power grids?',
            'n_context_docs': 10,
            'ground_truth': {
                'expected_mitre': ['T0831', 'T0836', 'T0816'],
                'expected_stride': ['Tampering', 'Denial of Service']
            }
        },
        {
            'query': 'What are the authentication vulnerabilities in EV charging management systems?',
            'n_context_docs': 10,
            'ground_truth': {
                'expected_mitre': ['T0866', 'T0890'],
                'expected_stride': ['Spoofing', 'Elevation of Privilege']
            }
        },
        {
            'query': 'Describe DoS attacks on charging station networks',
            'n_context_docs': 10,
            'ground_truth': {
                'expected_mitre': ['T0814', 'T0816'],
                'expected_stride': ['Denial of Service']
            }
        },
        {
            'query': 'What are the risks of data leakage in EV charging systems?',
            'n_context_docs': 10,
            'ground_truth': {
                'expected_mitre': ['T0868', 'T0877', 'T0842'],
                'expected_stride': ['Information Disclosure']
            }
        }
    ]

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system performance by comparing responses with and without RAG"
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to evaluate'
    )
    
    parser.add_argument(
        '--test-set',
        type=str,
        help='JSON file containing test queries'
    )
    
    parser.add_argument(
        '--use-default-test-set',
        action='store_true',
        help='Use default test set of 5 queries'
    )
    
    parser.add_argument(
        '--n-context',
        type=int,
        default=10,
        help='Number of context documents for RAG'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        evaluator = RAGEvaluator()
        
        if args.query:
            # Single query evaluation
            logger.info(f"Evaluating single query: {args.query}")
            
            result = evaluator.evaluate_query(args.query, args.n_context)
            
            print("\n" + result['report'])
            
            # Save result
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {args.output}")
        
        elif args.test_set:
            # Load test set from file
            logger.info(f"Loading test set from {args.test_set}")
            
            with open(args.test_set, 'r', encoding='utf-8') as f:
                test_queries = json.load(f)
            
            aggregated = evaluator.evaluate_test_set(test_queries, args.output)
            
            # Print aggregated results
            print("\n" + "=" * 80)
            print("AGGREGATED RESULTS")
            print("=" * 80)
            print(json.dumps(aggregated, indent=2))
        
        elif args.use_default_test_set:
            # Use default test set
            logger.info("Using default test set")
            
            test_queries = create_default_test_set()
            
            aggregated = evaluator.evaluate_test_set(test_queries, args.output)
            
            # Print aggregated results
            print("\n" + "=" * 80)
            print("AGGREGATED RESULTS")
            print("=" * 80)
            print(json.dumps(aggregated, indent=2))
        
        else:
            parser.print_help()
            print("\nPlease provide --query, --test-set, or --use-default-test-set")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
