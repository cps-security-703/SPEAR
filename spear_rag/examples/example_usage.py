

import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_rag import GeminiRAG
from vector_db import ChromaDBManager, DocumentEmbedder
from loguru import logger

def example_1_vulnerability_analysis():

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Vulnerability Analysis")
    print("=" * 80)

    rag = GeminiRAG()

    system_description = """
    Electric Vehicle Charging Station Network:
    - 50 Level 2 AC chargers (7.2 kW each)
    - 10 DC fast chargers (150 kW each)
    - OCPP 1.6 protocol over WebSocket
    - Central Management System (CMS) in cloud
    - Cellular (4G LTE) connectivity for each charger
    - Mobile app for user authentication and payment
    - Integration with utility Distribution Management System (DMS)
    - Basic username/password authentication
    - No encryption on OCPP messages
    """

    print("\nSystem Description:")
    print(system_description)

    print("\nPerforming vulnerability analysis...")
    analysis = rag.analyze_vulnerability(system_description)

    print("\nAnalysis Results:")
    print(analysis)

def example_2_rl_attack_strategies():

    print("\n" + "=" * 80)
    print("EXAMPLE 2: RL Attack Strategy Generation")
    print("=" * 80)

    rag = GeminiRAG()

    system_description = """
    Transmission Distribution System with EV Charging Integration:
    - 200 EV charging stations distributed across grid
    - Automatic Generation Control (AGC) system
    - Distribution Management System (DMS)
    - Real-time load balancing
    - Demand response capabilities
    """

    objective = """
    Learn optimal attack sequences to:
    1. Maximize grid frequency deviation
    2. Cause load imbalance
    3. Disrupt AGC control loops
    4. Evade detection systems
    """

    print("\nTarget System:")
    print(system_description)
    print("\nAttack Objective:")
    print(objective)

    print("\nGenerating RL attack strategies...")
    strategies = rag.suggest_rl_attack_strategies(system_description, objective)

    print("\nRL Attack Strategies:")
    print(strategies)

def example_3_direct_database_query():

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Direct Database Query")
    print("=" * 80)

    db = ChromaDBManager()
    embedder = DocumentEmbedder()


    print("\nQuery 1: STRIDE threats for charging management systems")
    query1 = "STRIDE threats for charging management systems"
    query1_embedding = embedder.embed_text(query1)

    results1 = db.query(
        query_embedding=query1_embedding,
        n_results=3
    )

    print(f"\nFound {len(results1['ids'][0])} results:")
    for i in range(len(results1['ids'][0])):
        metadata = results1['metadatas'][0][i]
        print(f"\n{i+1}. {metadata['title']}")
        print(f"   Type: {metadata['type']}")
        print(f"   Severity: {metadata['severity']}")
        print(f"   STRIDE: {metadata['stride_categories']}")


    print("\n\nQuery 2: High-severity vulnerabilities in SCADA/ICS")
    query2 = "SCADA ICS vulnerabilities"
    query2_embedding = embedder.embed_text(query2)

    results2 = db.query(
        query_embedding=query2_embedding,
        n_results=3,
        where={"severity": "High"}
    )

    print(f"\nFound {len(results2['ids'][0])} high-severity results:")
    for i in range(len(results2['ids'][0])):
        metadata = results2['metadatas'][0][i]
        print(f"\n{i+1}. {metadata['title']}")
        print(f"   CVSS: {metadata.get('cvss_score', 'N/A')}")
        print(f"   MITRE: {metadata['mitre_techniques']}")

def example_4_interactive_qa():

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Interactive Q&A")
    print("=" * 80)

    rag = GeminiRAG()

    questions = [
        "What are the most critical vulnerabilities in OCPP protocol?",
        "How can an attacker manipulate AGC systems?",
        "What STRIDE threats apply to EV charging billing systems?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\nQuestion {i}: {question}")
        print("-" * 80)

        answer = rag.interactive_query(question, n_context_docs=5)
        print(answer)

def example_5_filtered_search():

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Advanced Filtered Search")
    print("=" * 80)

    db = ChromaDBManager()
    embedder = DocumentEmbedder()

    query = "denial of service attacks"
    query_embedding = embedder.embed_text(query)


    print("\nSearching for DoS attacks with CVSS >= 7.0")

    results = db.query_by_filters(
        query_embedding=query_embedding,
        n_results=5,
        min_cvss=7.0,
        stride_category="Denial of Service"
    )

    print(f"\nFound {len(results['ids'][0])} results:")
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance

        print(f"\n{i+1}. {metadata['title']}")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   CVSS: {metadata.get('cvss_score', 'N/A')}")
        print(f"   Affected Systems: {metadata['affected_systems']}")

def example_6_database_stats():

    print("\n" + "=" * 80)
    print("EXAMPLE 6: Database Statistics")
    print("=" * 80)

    db = ChromaDBManager()
    stats = db.get_collection_stats()

    print("\nDatabase Statistics:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"\nDocument Types:")
    for doc_type, count in stats.get('document_types', {}).items():
        print(f"  - {doc_type}: {count}")

    print(f"\nSeverity Distribution:")
    for severity, count in stats.get('severity_distribution', {}).items():
        print(f"  - {severity}: {count}")

def main():

    print("\n" + "=" * 80)
    print("ARES RAG SYSTEM - USAGE EXAMPLES")
    print("=" * 80)

    try:

        example_6_database_stats()
        example_3_direct_database_query()
        example_5_filtered_search()


        print("\n\nNote: Skipping Gemini-based examples (1, 2, 4).")
        print("To run these examples, set GOOGLE_API_KEY in your .env file.")
        print("\nYou can run them individually:")
        print("  - example_1_vulnerability_analysis()")
        print("  - example_2_rl_attack_strategies()")
        print("  - example_4_interactive_qa()")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
