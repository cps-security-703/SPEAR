"""
Quick test script to verify CVE citations are working after prompt improvements
"""
import os
from gemini_rag import GeminiRAG
from loguru import logger

def test_cve_citations():
    """Test if CVE IDs are being extracted and cited in responses"""
    
    # Initialize RAG
    logger.info("Initializing GeminiRAG...")
    rag = GeminiRAG()
    
    # Test query that should return ICS-CERT advisories with CVEs
    test_query = "What are the OCPP protocol vulnerabilities with CVE IDs?"
    
    logger.info(f"Testing query: {test_query}")
    logger.info("=" * 80)
    
    # Get RAG response
    response = rag.interactive_query(test_query, n_context_docs=5)
    
    print("\n" + "=" * 80)
    print("RAG RESPONSE:")
    print("=" * 80)
    print(response)
    print("\n" + "=" * 80)
    
    # Check for CVE citations
    cve_count = response.count("CVE-")
    
    print(f"\n✓ CVE citations found: {cve_count}")
    
    if cve_count > 0:
        print("✅ SUCCESS: CVE IDs are being cited in the response!")
        
        # Extract CVE IDs
        import re
        cves = re.findall(r'CVE-\d{4}-\d{4,7}', response)
        print(f"\nCVE IDs found: {', '.join(set(cves))}")
    else:
        print("❌ ISSUE: No CVE IDs found in response")
        print("The prompt improvements may need adjustment or context docs don't contain CVEs")
    
    return cve_count > 0

if __name__ == "__main__":
    try:
        success = test_cve_citations()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        exit(1)
