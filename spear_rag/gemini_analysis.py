"""
Script to perform vulnerability analysis using Gemini RAG
"""

import argparse
from loguru import logger
import sys

from gemini_rag import GeminiRAG

def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Perform vulnerability analysis using Gemini RAG"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Vulnerability analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze system vulnerabilities")
    analyze_parser.add_argument(
        "system_description",
        type=str,
        help="Description of the system to analyze"
    )
    analyze_parser.add_argument(
        "--attack-scenario",
        type=str,
        help="Specific attack scenario to analyze"
    )
    analyze_parser.add_argument(
        "--n-context",
        type=int,
        default=10,
        help="Number of context documents to retrieve"
    )
    
    # RL attack strategy command
    rl_parser = subparsers.add_parser("rl-attack", help="Suggest RL attack strategies")
    rl_parser.add_argument(
        "system_description",
        type=str,
        help="Description of the target system"
    )
    rl_parser.add_argument(
        "objective",
        type=str,
        help="Attack objective"
    )
    rl_parser.add_argument(
        "--n-context",
        type=int,
        default=10,
        help="Number of context documents to retrieve"
    )
    
    # Interactive query command
    query_parser = subparsers.add_parser("query", help="Interactive query")
    query_parser.add_argument(
        "question",
        type=str,
        help="Question to ask"
    )
    query_parser.add_argument(
        "--n-context",
        type=int,
        default=5,
        help="Number of context documents to retrieve"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging()
    
    try:
        rag = GeminiRAG()
        
        if args.command == "analyze":
            logger.info("Performing vulnerability analysis...")
            result = rag.analyze_vulnerability(
                system_description=args.system_description,
                attack_scenario=args.attack_scenario,
                n_context_docs=args.n_context
            )
            
            print("\n" + "=" * 80)
            print("VULNERABILITY ANALYSIS RESULTS")
            print("=" * 80)
            print(result)
            print("=" * 80 + "\n")
        
        elif args.command == "rl-attack":
            logger.info("Generating RL attack strategies...")
            result = rag.suggest_rl_attack_strategies(
                system_description=args.system_description,
                objective=args.objective,
                n_context_docs=args.n_context
            )
            
            print("\n" + "=" * 80)
            print("RL ATTACK STRATEGY SUGGESTIONS")
            print("=" * 80)
            print(result)
            print("=" * 80 + "\n")
        
        elif args.command == "query":
            logger.info("Processing query...")
            result = rag.interactive_query(
                query=args.question,
                n_context_docs=args.n_context
            )
            
            print("\n" + "=" * 80)
            print("QUERY RESPONSE")
            print("=" * 80)
            print(result)
            print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
