# STRIDE-MITRE-RL Action Mapping Evaluation Guide

## Overview

This evaluation framework validates the RAG system's ability to generate accurate **STRIDE-MITRE-RL action mappings** for EVCS and distribution grid vulnerabilities. This is critical for designing reinforcement learning (RL) attack agents for security testing.

## Research Objective

Compare RAG vs Non-RAG responses to determine which approach produces more accurate mappings between:
- **STRIDE threat categories** (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
- **MITRE ATT&CK for ICS techniques** (T0866, T0855, T0836, etc.)
- **Projected RL attack actions** (Communication spoofing, Data injection, Protocol manipulation, etc.)

## Architecture Coverage

### Communication Links (Focus on EVCS & Distribution Grid)

Based on the cyber-physical architecture:

**✅ Included (Links 1-6)**:
1. **EV ↔ EVCS** (OCPP): Charging info, optimal charging parameters
2. **EVCS ↔ CMS**: Customer authentication, queue management
3. **CMS ↔ Distribution Grid** (DNP3): Load measurement
4. **Distribution Grid ↔ DSM** (DNP3): Load measurement
5. **DSM ↔ EMS** (DNP3): Load forecasting
6. **EMS ↔ AGC** (TCP/IP): Load measurement

**❌ Excluded (Links 7-9)**:
- Transmission grid focus (out of scope)

### Systems Covered
- **Cyber Layer**: CCMS, CMS, DSM, EMS, AGC
- **Physical Layer**: EV, EVCS, Distribution Grid, Generators

## Test Queries

### 15 Queries Across 6 STRIDE Categories

| Query ID | STRIDE Category | Communication Link | Protocol |
|----------|----------------|-------------------|----------|
| Q1 | Spoofing | EV ↔ EVCS | OCPP |
| Q2 | Tampering | EVCS ↔ CMS | OCPP/HTTPS |
| Q3 | Information Disclosure | CMS ↔ Grid | DNP3 |
| Q4 | Denial of Service | Grid ↔ DSM | DNP3 |
| Q5 | Repudiation | DSM ↔ EMS | DNP3 |
| Q6 | Elevation of Privilege | EMS ↔ AGC | TCP/IP |
| Q7-Q14 | Various | Multiple | OCPP/DNP3/TCP |
| Q15 | Multi-STRIDE | All | OCPP/DNP3/TCP |

**Coverage**:
- OCPP vulnerabilities: 6 queries
- DNP3 vulnerabilities: 6 queries
- TCP/IP vulnerabilities: 3 queries
- Multi-protocol: 1 query

## Evaluation Metrics

### Component-Level Metrics

For each query, we measure **Precision, Recall, and F1 scores** for:

1. **MITRE Techniques Accuracy**
   - Correct identification of MITRE ATT&CK technique IDs (e.g., T0866, T0855)
   - Precision: % of extracted techniques that are correct
   - Recall: % of ground truth techniques that were found

2. **STRIDE Categories Accuracy**
   - Correct identification of STRIDE categories
   - Multi-label classification (some attacks span multiple categories)

3. **RL Actions Accuracy**
   - Correct mapping to projected RL actions
   - Actions: Communication spoofing, Data injection, Protocol manipulation, Voltage manipulation, Power disruption, Current injection

4. **Protocol Identification**
   - Correct identification of vulnerable protocols
   - Protocols: OCPP, DNP3, TCP/IP, Modbus, HTTPS, ISO 15118

5. **CVE Citations**
   - Presence of relevant CVE IDs
   - Expected CVEs from ICS-CERT advisories

### Aggregated Metrics

- **Average F1 scores** across all queries
- **RAG vs Non-RAG improvement** (percentage)
- **Win/Loss/Tie counts**

### Overall F1 Score

Calculated as the average of:
```
Overall F1 = (MITRE F1 + STRIDE F1 + RL Actions F1 + Protocols F1) / 4
```

## Usage

### 1. Run Full Evaluation (All 15 Queries)

```bash
python evaluate_stride_mitre_rl_mapping.py --output stride_mitre_rl_results.json
```

**Expected Runtime**: ~30-45 minutes (15 queries × 2 responses each)

### 2. Evaluate Single Query

```bash
python evaluate_stride_mitre_rl_mapping.py --query-id Q1_SPOOFING_EV_EVCS
```

### 3. Evaluate by STRIDE Category

```bash
# Test all Spoofing queries
python evaluate_stride_mitre_rl_mapping.py --stride-category Spoofing

# Test all Tampering queries
python evaluate_stride_mitre_rl_mapping.py --stride-category Tampering
```

### 4. View Query Details

```python
from stride_mitre_rl_mapping_queries import STRIDE_MITRE_RL_MAPPING_QUERIES, get_queries_by_stride

# Get all spoofing queries
spoofing_queries = get_queries_by_stride("Spoofing")

# Get queries by protocol
from stride_mitre_rl_mapping_queries import get_queries_by_protocol
ocpp_queries = get_queries_by_protocol("OCPP")
```

## Output Format

### JSON Results Structure

```json
{
  "timestamp": "2026-02-05T...",
  "total_queries": 15,
  "aggregated_metrics": {
    "rag_metrics": {
      "avg_overall_f1": 75.5,
      "avg_mitre_f1": 80.2,
      "avg_stride_f1": 85.0,
      "avg_rl_f1": 70.5,
      "avg_cve_f1": 66.3
    },
    "non_rag_metrics": {
      "avg_overall_f1": 45.2,
      "avg_mitre_f1": 35.0,
      "avg_stride_f1": 60.0,
      "avg_rl_f1": 40.0,
      "avg_cve_f1": 0.0
    },
    "improvement": {
      "overall_f1": 30.3,
      "mitre_f1": 45.2,
      "stride_f1": 25.0,
      "rl_f1": 30.5,
      "cve_f1": 66.3
    },
    "rag_wins": 14,
    "non_rag_wins": 0,
    "ties": 1
  },
  "individual_results": [...]
}
```

### Individual Query Result

```json
{
  "query_id": "Q1_SPOOFING_EV_EVCS",
  "query": "What are the spoofing vulnerabilities...",
  "communication_link": "Link 1: EV <-> EVCS (OCPP)",
  "stride_category": "Spoofing",
  "ground_truth": {
    "stride": ["Spoofing"],
    "mitre_techniques": ["T0866", "T0855", "T0862"],
    "projected_rl_action": "Communication spoofing",
    "protocols": ["OCPP"],
    "expected_cves": ["CVE-2022-3203", "CVE-2022-3204"]
  },
  "rag_accuracy": {
    "mitre_techniques": {
      "precision": 100.0,
      "recall": 100.0,
      "f1": 100.0,
      "correct": ["T0866", "T0855", "T0862"],
      "missed": [],
      "extra": []
    },
    "stride_categories": {...},
    "rl_actions": {...},
    "protocols": {...},
    "cves": {...},
    "overall_f1": 85.5
  },
  "non_rag_accuracy": {...},
  "improvement": {
    "overall_f1": 40.2,
    "mitre_f1": 65.0,
    "stride_f1": 20.0,
    "rl_f1": 50.0,
    "cve_f1": 100.0
  }
}
```

## Expected Results

### Hypothesis

**RAG should significantly outperform Non-RAG** because:

1. **ICS-CERT Advisories**: RAG has access to 8 real-world advisories with CVEs
2. **Protocol Vulnerabilities**: 13 protocol-specific vulnerability documents
3. **MITRE-STRIDE Mappings**: Pre-mapped MITRE techniques to STRIDE categories
4. **STRIDE Patterns**: Domain-specific threat patterns for EVCS

### Expected Improvements

| Metric | Non-RAG F1 | RAG F1 | Improvement |
|--------|-----------|--------|-------------|
| MITRE Techniques | 30-40% | 75-85% | +40-50% |
| STRIDE Categories | 50-60% | 80-90% | +25-35% |
| RL Actions | 35-45% | 70-80% | +30-40% |
| CVE Citations | 0-5% | 60-80% | +60-75% |
| **Overall** | **40-50%** | **75-85%** | **+30-40%** |

### Success Criteria

✅ **Excellent Performance**: RAG F1 > 75%, Improvement > 30%
✅ **Good Performance**: RAG F1 > 65%, Improvement > 20%
⚠️ **Needs Improvement**: RAG F1 < 65%, Improvement < 20%

## Analysis & Visualization

### Generate Comparison Table

After evaluation, you can generate a LaTeX table similar to your original:

```python
import json

with open('stride_mitre_rl_results.json', 'r') as f:
    results = json.load(f)

# Extract best mappings from RAG responses
for result in results['individual_results']:
    print(f"STRIDE: {result['ground_truth']['stride']}")
    print(f"MITRE: {result['rag_accuracy']['mitre_techniques']['correct']}")
    print(f"RL Action: {result['rag_accuracy']['rl_actions']['correct']}")
    print(f"F1 Score: {result['rag_accuracy']['overall_f1']}")
    print("-" * 80)
```

### Compare with Original Table

Your original table (without RAG):

| STRIDE | MITRE | RL Action |
|--------|-------|-----------|
| Spoofing | T0817, T0819, T0867 | Communication spoofing |
| Tampering | T0806, T0871, T0807 | Data injection |
| ... | ... | ... |

RAG-generated table will likely have:
- ✅ More specific MITRE techniques (T0866, T0855 for OCPP)
- ✅ CVE references (CVE-2022-3203)
- ✅ Protocol-specific mappings
- ✅ Better alignment with actual vulnerabilities

## Integration with RL Agent Design

### Using Results for RL Action Space

Once you have validated mappings, use them to design RL action space:

```python
# Extract validated RL actions from high-F1 queries
validated_actions = []
for result in results['individual_results']:
    if result['rag_accuracy']['overall_f1'] > 75:
        validated_actions.extend(result['rag_accuracy']['rl_actions']['correct'])

# Use for RL agent action space definition
rl_action_space = {
    'communication_spoofing': {...},
    'data_injection': {...},
    'protocol_manipulation': {...},
    # etc.
}
```

## Troubleshooting

### Issue: Low MITRE F1 Scores

**Cause**: RAG not extracting MITRE technique IDs from context
**Solution**: Check if prompts emphasize MITRE ID extraction (already fixed in Phase 1)

### Issue: Low CVE F1 Scores

**Cause**: CVE IDs not being cited despite being in context
**Solution**: Verify Phase 1 CVE citation fixes are working (test with `test_cve_citations.py`)

### Issue: API Rate Limit

**Cause**: Gemini free tier limit (20 requests/day)
**Solution**: 
- Run evaluation in batches (5 queries at a time)
- Wait 24 hours between batches
- Or upgrade to paid tier

## Next Steps

### After Evaluation

1. **Analyze Results**: Identify which STRIDE categories have best/worst mappings
2. **Refine Mappings**: Use high-F1 RAG responses as ground truth
3. **Generate LaTeX Table**: Create publication-ready table from RAG results
4. **Design RL Actions**: Use validated mappings to define RL action space
5. **Implement RL Agent**: Build attack agent based on validated STRIDE-MITRE-RL mappings

### Publication

Use results to demonstrate:
- ✅ RAG improves mapping accuracy by 30-40%
- ✅ Knowledge base provides domain-specific expertise
- ✅ Automated mapping generation for RL agent design
- ✅ Validation of STRIDE-MITRE-RL framework for EVCS security

## Files Created

1. **`stride_mitre_rl_mapping_queries.py`**: 15 test queries with ground truth
2. **`evaluate_stride_mitre_rl_mapping.py`**: Evaluation script with metrics
3. **`STRIDE_MITRE_RL_MAPPING_GUIDE.md`**: This documentation

## Summary

This evaluation framework provides:
- ✅ 15 comprehensive test queries covering all STRIDE categories
- ✅ Focus on EVCS and distribution grid (Links 1-6)
- ✅ Precision/Recall/F1 metrics for each component
- ✅ RAG vs Non-RAG comparison
- ✅ Ground truth for validation
- ✅ Integration path to RL agent design

**Ready to run after Gemini API rate limit resets!** 🚀
