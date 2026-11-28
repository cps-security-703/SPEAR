# RAG Evaluation Guide

## Overview

This guide explains how to evaluate and verify the effectiveness of the RAG system by comparing Gemini responses **with RAG** (using vector database context) vs **without RAG** (no context).

## Evaluation Metrics

The system measures:

### 1. **Technical Accuracy**
- **MITRE Techniques**: Number of MITRE ATT&CK technique IDs mentioned (e.g., T0866, T0814)
- **STRIDE Categories**: Number of STRIDE categories identified
- **CVE References**: Number of specific CVE IDs mentioned

### 2. **Specificity Score**
- Count of technical security terms (vulnerability, exploit, mitigation, etc.)
- Count of EVSE-specific terms (OCPP, ISO 15118, charging station, AGC, etc.)
- Higher score = more specific and detailed response

### 3. **Actionability Score**
- Count of actionable recommendations (implement, configure, enable, etc.)
- Count of mitigation-related terms (mitigation, countermeasure, defense, etc.)
- Higher score = more actionable guidance

### 4. **Completeness**
- Presence of expected sections (vulnerabilities, mitigations, detection, etc.)
- Percentage of expected content covered

### 5. **Ground Truth Validation** (Optional)
- Recall: Percentage of expected MITRE techniques found
- Recall: Percentage of expected STRIDE categories found
- Recall: Percentage of expected CVEs found

## Usage

### Option 1: Evaluate Single Query

```bash
python evaluate_rag.py --query "What are the main vulnerabilities in OCPP protocol?"
```

**Output:**
- Side-by-side comparison of RAG vs non-RAG responses
- Detailed metrics for both responses
- Evaluation report showing advantages

### Option 2: Use Default Test Set (5 Queries)

```bash
python evaluate_rag.py --use-default-test-set
```

**Default Test Queries:**
1. OCPP protocol vulnerabilities
2. AGC system manipulation
3. Authentication vulnerabilities in charging systems
4. DoS attacks on charging networks
5. Data leakage risks in EV systems

**Output:**
- Individual evaluation for each query
- Aggregated statistics across all queries
- Results saved to `evaluation_results.json`

### Option 3: Custom Test Set

Create a JSON file with your test queries:

```json
[
  {
    "query": "Your question here",
    "n_context_docs": 10,
    "ground_truth": {
      "expected_mitre": ["T0866", "T0814"],
      "expected_stride": ["Spoofing", "Denial of Service"],
      "expected_cves": ["CVE-2024-1234"]
    }
  },
  {
    "query": "Another question",
    "n_context_docs": 10
  }
]
```

Run evaluation:

```bash
python evaluate_rag.py --test-set my_test_queries.json --output my_results.json
```

## Example Evaluation Report

```
================================================================================
RAG EVALUATION REPORT
================================================================================

1. MITRE ATT&CK Techniques:
   RAG: 5 techniques
   Non-RAG: 1 techniques
   Advantage: +4 techniques
   RAG found: T0866, T0855, T0868, T0814, T0836

2. STRIDE Categories:
   RAG: 4 categories
   Non-RAG: 2 categories
   Advantage: +2 categories
   RAG found: Spoofing, Tampering, Information Disclosure, Denial of Service

3. CVE References:
   RAG: 3 CVEs
   Non-RAG: 0 CVEs
   Advantage: +3 CVEs

4. Specificity Score:
   RAG: 45
   Non-RAG: 18
   Advantage: +27

5. Actionability Score:
   RAG: 32
   Non-RAG: 12
   Advantage: +20

6. Ground Truth Validation:
   MITRE Recall - RAG: 80%, Non-RAG: 20%
   STRIDE Recall - RAG: 100%, Non-RAG: 50%

================================================================================
RAG Advantages: 4/4 metrics
✓ RAG significantly improves response quality
================================================================================
```

## Programmatic Usage

### Python Script Example

```python
from evaluation import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate single query
result = evaluator.evaluate_query(
    query="What are authentication vulnerabilities in EVSE?",
    n_context_docs=10,
    ground_truth={
        'expected_mitre': ['T0866', 'T0890'],
        'expected_stride': ['Spoofing', 'Elevation of Privilege']
    }
)

# Print report
print(result['report'])

# Access detailed metrics
print(f"RAG MITRE count: {result['comparison']['mitre_techniques']['rag_count']}")
print(f"Non-RAG MITRE count: {result['comparison']['mitre_techniques']['non_rag_count']}")

# Get RAG response
rag_response = result['rag_response']['response']
print(f"\nRAG Response:\n{rag_response}")

# Get non-RAG response
non_rag_response = result['non_rag_response']['response']
print(f"\nNon-RAG Response:\n{non_rag_response}")
```

### Batch Evaluation

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator()

test_queries = [
    {
        'query': 'OCPP vulnerabilities',
        'ground_truth': {
            'expected_mitre': ['T0866', 'T0855'],
            'expected_stride': ['Spoofing', 'Tampering']
        }
    },
    {
        'query': 'AGC attacks',
        'ground_truth': {
            'expected_mitre': ['T0831', 'T0836'],
            'expected_stride': ['Tampering']
        }
    }
]

# Evaluate all queries
aggregated = evaluator.evaluate_test_set(
    test_queries,
    output_file='batch_results.json'
)

# Print aggregated statistics
print(f"Total queries: {aggregated['total_queries']}")
print(f"RAG advantages: {aggregated['rag_advantages']}")
print(f"Average improvements: {aggregated['average_scores']}")
```

## Understanding Results

### What Good Results Look Like

**RAG should show advantages in:**
- ✅ More MITRE techniques mentioned (specific attack methods)
- ✅ More STRIDE categories covered (comprehensive threat analysis)
- ✅ Specific CVE references (real-world vulnerabilities)
- ✅ Higher specificity score (technical depth)
- ✅ Higher actionability score (practical recommendations)
- ✅ Better ground truth recall (accuracy)

### Example Comparison

**Query:** "What are OCPP protocol vulnerabilities?"

**Non-RAG Response (Generic):**
```
OCPP protocol may have authentication issues and encryption weaknesses.
Consider implementing strong security measures and regular updates.
```
- MITRE: 0
- STRIDE: 1 (vague)
- CVEs: 0
- Specificity: Low
- Actionability: Low

**RAG Response (Specific):**
```
OCPP protocol vulnerabilities include:

1. Authentication Bypass (T0866, STRIDE: Spoofing)
   - CVE-2023-XXXX affects OCPP 1.6 implementations
   - Mitigation: Implement OCPP 2.0.1 with security profile 3

2. Message Tampering (T0855, STRIDE: Tampering)
   - Unencrypted WebSocket communications
   - Mitigation: Use TLS 1.3, certificate pinning

3. Information Disclosure (T0868, STRIDE: Information Disclosure)
   - Charging session data exposure
   - Mitigation: Encrypt data at rest and in transit

Detection: Monitor for authentication failures, unexpected message sources
```
- MITRE: 3 techniques
- STRIDE: 3 categories
- CVEs: 1+
- Specificity: High
- Actionability: High

## Validation Strategies

### 1. Manual Review
- Read both responses
- Verify technical accuracy
- Check if RAG response uses database context

### 2. Quantitative Metrics
- Compare MITRE/STRIDE counts
- Measure specificity scores
- Calculate ground truth recall

### 3. Domain Expert Review
- Have security experts rate response quality
- Assess practical applicability
- Verify correctness of recommendations

### 4. A/B Testing
- Use responses in real security assessments
- Measure time to identify vulnerabilities
- Track false positive/negative rates

## Tips for Effective Evaluation

1. **Create Diverse Test Queries**
   - Cover different attack types
   - Include various system components
   - Mix broad and specific questions

2. **Define Ground Truth**
   - Research expected MITRE techniques
   - Identify relevant STRIDE categories
   - Find applicable CVEs from your database

3. **Run Multiple Evaluations**
   - Test with different context sizes (5, 10, 15 docs)
   - Evaluate at different times
   - Compare consistency

4. **Analyze Context Quality**
   - Check which documents RAG retrieved
   - Verify relevance of context
   - Identify gaps in database

5. **Iterate and Improve**
   - Add missing data sources
   - Refine STRIDE patterns
   - Update MITRE-STRIDE mappings

## Common Issues

### Issue: RAG shows no improvement

**Possible Causes:**
- Database not populated yet (run `python main.py` first)
- Query doesn't match database content
- Context documents not relevant

**Solutions:**
- Verify database has documents: `python query_db.py "test"`
- Try more specific queries related to EVSE/power systems
- Increase `n_context_docs` parameter

### Issue: Both responses are similar

**Possible Causes:**
- Query is too general
- Gemini has strong base knowledge
- Context not being used effectively

**Solutions:**
- Ask more specific technical questions
- Include system details in query
- Check retrieved context documents

### Issue: Ground truth validation fails

**Possible Causes:**
- Expected techniques not in database
- Response uses different terminology
- Ground truth too strict

**Solutions:**
- Verify expected techniques exist in database
- Adjust ground truth expectations
- Use broader matching criteria

## Output Files

### evaluation_results.json

Contains:
```json
{
  "aggregated_results": {
    "total_queries": 5,
    "rag_advantages": {
      "mitre_techniques": "4/5 (80%)",
      "stride_categories": "5/5 (100%)",
      "specificity": "5/5 (100%)"
    },
    "average_scores": {
      "mitre_techniques": {
        "rag": 4.2,
        "non_rag": 1.4,
        "improvement": 2.8
      }
    }
  },
  "individual_results": [...]
}
```

## Next Steps

1. **Run Initial Evaluation**
   ```bash
   python evaluate_rag.py --use-default-test-set
   ```

2. **Review Results**
   - Check evaluation_results.json
   - Analyze where RAG helps most
   - Identify improvement areas

3. **Create Custom Tests**
   - Design queries for your specific use case
   - Add ground truth from your research
   - Run comprehensive evaluation

4. **Iterate**
   - Add more data sources if needed
   - Refine database content
   - Re-evaluate to measure improvement

## Example Workflow

```bash
# 1. Create database (if not done)
python main.py --nvd-max-results 50 --skip-cicevse

# 2. Run quick evaluation
python evaluate_rag.py --query "OCPP authentication vulnerabilities"

# 3. Run full test set
python evaluate_rag.py --use-default-test-set --output full_eval.json

# 4. Analyze results
# Review full_eval.json for detailed metrics

# 5. Query database to verify context
python query_db.py "OCPP authentication" --n-results 5
```

## Conclusion

The evaluation system provides quantitative and qualitative metrics to verify that RAG improves Gemini's responses by:
- Adding specific technical details from the database
- Referencing real vulnerabilities (CVEs)
- Providing MITRE ATT&CK and STRIDE framework context
- Offering actionable, evidence-based recommendations

Use this system to validate your RAG implementation and demonstrate its value for vulnerability analysis.
