

import argparse
from loguru import logger
import sys
import json

from vector_db import ChromaDBManager, DocumentEmbedder
from config import config

def setup_logging():

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

def print_results(results, show_full: bool = False):

    if not results['ids'] or not results['ids'][0]:
        logger.info("No results found.")
        return

    logger.info(f"\nFound {len(results['ids'][0])} results:\n")

    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]

        print("=" * 80)
        print(f"Result {i+1}")
        print("=" * 80)
        print(f"ID: {doc_id}")
        print(f"Similarity Score: {1 - distance:.4f}")
        print(f"Type: {metadata.get('type', 'N/A')}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Severity: {metadata.get('severity', 'N/A')} | CVSS: {metadata.get('cvss_score', 'N/A')}")
        print(f"STRIDE: {metadata.get('stride_categories', '[]')}")
        print(f"MITRE: {metadata.get('mitre_techniques', '[]')}")
        print(f"Affected Systems: {metadata.get('affected_systems', '[]')}")
        print(f"\nDescription:")
        if show_full:
            print(document)
        else:
            print(document[:300] + "..." if len(document) > 300 else document)
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Query the EVSE Vulnerability Vector Database"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Query string"
    )

    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of results to return"
    )

    parser.add_argument(
        "--severity",
        type=str,
        choices=["Critical", "High", "Medium", "Low"],
        help="Filter by severity"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["vulnerability", "mitre_technique", "stride_pattern", "mitre_stride_mapping", "dataset"],
        help="Filter by document type"
    )

    parser.add_argument(
        "--min-cvss",
        type=float,
        help="Minimum CVSS score"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full document content"
    )

    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )

    args = parser.parse_args()

    setup_logging()

    logger.info(f"Querying database: {args.query}")

    try:
        embedder = DocumentEmbedder()
        db_manager = ChromaDBManager()

        query_embedding = embedder.embed_text(args.query)

        where_filter = {}
        if args.severity:
            where_filter["severity"] = args.severity
        if args.type:
            where_filter["type"] = args.type
        if args.min_cvss is not None:
            where_filter["cvss_score"] = {"$gte": args.min_cvss}

        results = db_manager.query(
            query_embedding=query_embedding,
            n_results=args.n_results,
            where=where_filter if where_filter else None
        )

        print_results(results, show_full=args.full)

        if args.export:
            export_data = {
                "query": args.query,
                "filters": where_filter,
                "n_results": len(results['ids'][0]) if results['ids'] else 0,
                "results": []
            }

            if results['ids']:
                for i in range(len(results['ids'][0])):
                    result = {
                        "id": results['ids'][0][i],
                        "distance": results['distances'][0][i],
                        "similarity": 1 - results['distances'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "document": results['documents'][0][i]
                    }
                    export_data['results'].append(result)

            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Results exported to {args.export}")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
