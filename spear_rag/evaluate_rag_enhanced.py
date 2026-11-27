

import argparse
from loguru import logger
import sys
import json

from evaluation import RAGEvaluator
from evaluation.enhanced_metrics import EnhancedMetrics
from evaluation.context_analyzer import ContextAnalyzer

def setup_logging():

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

def create_default_test_set():

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

def evaluate_single_query_enhanced(evaluator, query, n_context, ground_truth=None):

    logger.info(f"Evaluating query: {query}")


    rag_result = evaluator.get_rag_response(query, n_context)


    non_rag_result = evaluator.get_non_rag_response(query)


    enhanced_comparison = EnhancedMetrics.compare_rag_vs_non_rag(
        rag_result['response'],
        non_rag_result['response'],
        rag_result.get('context_docs', []),
        ground_truth
    )


    context_analysis = ContextAnalyzer.analyze_context_utilization(
        rag_result['response'],
        rag_result.get('context_docs', []),
        query
    )


    enhanced_report = EnhancedMetrics.generate_enhanced_report(enhanced_comparison)
    context_report = ContextAnalyzer.generate_context_usage_report(context_analysis)

    return {
        'query': query,
        'rag_result': rag_result,
        'non_rag_result': non_rag_result,
        'enhanced_comparison': enhanced_comparison,
        'context_analysis': context_analysis,
        'enhanced_report': enhanced_report,
        'context_report': context_report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RAG evaluation with improved metrics and context analysis"
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
        default='enhanced_evaluation_results.json',
        help='Output file for results'
    )

    parser.add_argument(
        '--show-context-analysis',
        action='store_true',
        help='Show detailed context usage analysis'
    )

    args = parser.parse_args()

    setup_logging()

    try:
        evaluator = RAGEvaluator()

        if args.query:

            logger.info(f"Evaluating single query: {args.query}")

            result = evaluate_single_query_enhanced(
                evaluator,
                args.query,
                args.n_context
            )


            print("\n" + result['enhanced_report'])

            if args.show_context_analysis:
                print("\n" + result['context_report'])


            with open(args.output, 'w', encoding='utf-8') as f:

                save_result = {k: v for k, v in result.items()
                              if k not in ['enhanced_report', 'context_report']}
                json.dump(save_result, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {args.output}")

        elif args.test_set:

            logger.info(f"Loading test set from {args.test_set}")

            with open(args.test_set, 'r', encoding='utf-8') as f:
                test_queries = json.load(f)

            results = []
            for i, test_case in enumerate(test_queries, 1):
                logger.info(f"Evaluating query {i}/{len(test_queries)}")

                result = evaluate_single_query_enhanced(
                    evaluator,
                    test_case['query'],
                    test_case.get('n_context_docs', args.n_context),
                    test_case.get('ground_truth')
                )

                results.append(result)


                print(f"\n{'='*80}")
                print(f"Query {i}/{len(test_queries)}")
                print(f"{'='*80}")
                print(result['enhanced_report'])

                if args.show_context_analysis:
                    print("\n" + result['context_report'])


                import time
                time.sleep(2)


            aggregated = aggregate_enhanced_results(results)


            print("\n" + "="*80)
            print("AGGREGATED ENHANCED RESULTS")
            print("="*80)
            print(json.dumps(aggregated, indent=2))


            save_data = {
                'aggregated': aggregated,
                'individual_results': [
                    {k: v for k, v in r.items() if k not in ['enhanced_report', 'context_report']}
                    for r in results
                ]
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {args.output}")

        elif args.use_default_test_set:

            logger.info("Using default test set")

            test_queries = create_default_test_set()
            results = []

            for i, test_case in enumerate(test_queries, 1):
                logger.info(f"Evaluating query {i}/{len(test_queries)}")

                result = evaluate_single_query_enhanced(
                    evaluator,
                    test_case['query'],
                    test_case.get('n_context_docs', args.n_context),
                    test_case.get('ground_truth')
                )

                results.append(result)


                print(f"\n{'='*80}")
                print(f"Query {i}/{len(test_queries)}")
                print(f"{'='*80}")
                print(result['enhanced_report'])

                if args.show_context_analysis:
                    print("\n" + result['context_report'])


                import time
                time.sleep(2)


            aggregated = aggregate_enhanced_results(results)


            print("\n" + "="*80)
            print("AGGREGATED ENHANCED RESULTS")
            print("="*80)
            print(json.dumps(aggregated, indent=2))


            save_data = {
                'aggregated': aggregated,
                'individual_results': [
                    {k: v for k, v in r.items() if k not in ['enhanced_report', 'context_report']}
                    for r in results
                ]
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {args.output}")

        else:
            parser.print_help()
            print("\nPlease provide --query, --test-set, or --use-default-test-set")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception(e)
        sys.exit(1)

def aggregate_enhanced_results(results: list) -> dict:

    total = len(results)


    avg_rag_quality = sum(r['enhanced_comparison']['rag_quality']['overall_quality_score'] for r in results) / total
    avg_non_rag_quality = sum(r['enhanced_comparison']['non_rag_quality']['overall_quality_score'] for r in results) / total
    avg_improvement = avg_rag_quality - avg_non_rag_quality


    avg_context_usage = sum(r['context_analysis']['utilization_score'] for r in results) / total
    avg_docs_used = sum(r['context_analysis']['documents_used'] for r in results) / total


    quality_wins = sum(1 for r in results if r['enhanced_comparison']['quality_improvement'] > 0)

    return {
        'total_queries': total,
        'average_quality_scores': {
            'rag': avg_rag_quality,
            'non_rag': avg_non_rag_quality,
            'improvement': avg_improvement,
            'improvement_percentage': (avg_improvement / avg_non_rag_quality * 100) if avg_non_rag_quality > 0 else 0
        },
        'context_usage': {
            'average_utilization_score': avg_context_usage,
            'average_documents_used': avg_docs_used
        },
        'rag_wins': {
            'quality_wins': f"{quality_wins}/{total} ({quality_wins/total*100:.1f}%)"
        },
        'verdict': 'RAG significantly improves quality' if avg_improvement > 20 else
                  'RAG improves quality' if avg_improvement > 5 else
                  'RAG shows marginal improvement' if avg_improvement > -5 else
                  'RAG needs improvement'
    }

if __name__ == "__main__":
    main()
