# Quick Start Guide

## Initial Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file:

```bash
# Google Gemini API Key (get from https://makersuite.google.com/app/apikey)
GOOGLE_API_KEY=your_google_api_key_here

# NVD API Key (already configured, but you can use your own)
NVD_API_KEY=your_nvd_api_key_here
```

## Quick Start: Create Database (30-60 minutes)

### Option 1: Full Database (Recommended)

Collect all data sources except CICEVSE2024:

```bash
python main.py --nvd-max-results 100
```

This will:
- Collect 100 CVEs from NVD related to EVSE, AGC, Power Systems
- Download all MITRE ATT&CK for ICS techniques (~100 techniques)
- Generate 18 STRIDE threat patterns
- Create 10 MITRE-STRIDE mappings
- Generate embeddings and populate vector database

**Time**: ~30-45 minutes (depends on NVD API rate limits)

### Option 2: Full Database with CICEVSE2024 Dataset

Include the CICEVSE2024 dataset for network attack patterns:

```bash
python main.py --nvd-max-results 100 --cicevse-dataset data/CICEVSE2024_NT.csv
```

This adds real network traffic attack patterns from EV charging infrastructure to your database.

### Option 3: Quick Test (5-10 minutes)

Create a smaller database for testing:

```bash
python main.py --nvd-max-results 20 --skip-cicevse
```

### Option 4: Skip NVD (Faster)

If you want to skip NVD collection (which is slow due to rate limits):

```bash
python main.py --skip-nvd
```

This will only include MITRE, STRIDE, and MITRE-STRIDE mappings (~128 documents).

**Time**: ~5 minutes

### Option 5: CICEVSE2024 Only (Testing)

Test CICEVSE2024 processing without other sources:

```bash
python main.py --skip-nvd --skip-mitre --cicevse-dataset data/CICEVSE2024_NT.csv
```

This creates a database with STRIDE patterns, MITRE-STRIDE mappings, and CICEVSE2024 attack patterns.

**Time**: ~5-10 minutes

## Quick Start: Query Database

Once the database is created, try these queries:

### 1. Basic Query

```bash
python query_db.py "authentication vulnerabilities in EV charging"
```

### 2. Filtered Query

```bash
python query_db.py "DoS attacks on power systems" --severity High --n-results 10
```

### 3. Export Results

```bash
python query_db.py "SCADA vulnerabilities" --export results.json --full
```

## Quick Start: Gemini Analysis

### 1. Analyze System Vulnerabilities

```bash
python gemini_analysis.py analyze "EV charging station using OCPP 1.6 with basic authentication"
```

### 2. Generate RL Attack Strategies

```bash
python gemini_analysis.py rl-attack "EV charging network with 50 stations" "Disrupt grid frequency through coordinated charging"
```

### 3. Ask Questions

```bash
python gemini_analysis.py query "What are common vulnerabilities in AGC systems?"
```

## Example Workflow

### Scenario: Analyze Your EVSE System

1. **Create the database** (first time only):
   ```bash
   python main.py --nvd-max-results 50
   ```

2. **Query for relevant vulnerabilities**:
   ```bash
   python query_db.py "OCPP protocol vulnerabilities" --n-results 10
   ```

3. **Perform detailed analysis**:
   ```bash
   python gemini_analysis.py analyze "EVSE network with OCPP 2.0.1, cloud backend, and mobile app interface"
   ```

4. **Design RL attack scenarios**:
   ```bash
   python gemini_analysis.py rl-attack "EVSE network integrated with DMS" "Learn attack sequences to maximize grid impact"
   ```

## Programmatic Usage

### Python Script Example

```python
from gemini_rag import GeminiRAG
from vector_db import ChromaDBManager, DocumentEmbedder

# Initialize RAG system
rag = GeminiRAG()

# Analyze a system
system_desc = """
EV charging station network:
- 100 Level 2 chargers
- 20 DC fast chargers
- OCPP 1.6 protocol
- Cellular connectivity
- Cloud management platform
"""

analysis = rag.analyze_vulnerability(system_desc)
print(analysis)

# Query database directly
db = ChromaDBManager()
embedder = DocumentEmbedder()

query_embedding = embedder.embed_text("AGC control vulnerabilities")
results = db.query(query_embedding=query_embedding, n_results=5)

for i, doc_id in enumerate(results['ids'][0]):
    print(f"{i+1}. {results['metadatas'][0][i]['title']}")
```

## Common Issues

### Issue: NVD API Rate Limit

**Solution**: The system automatically handles rate limiting. If you want faster results, skip NVD:
```bash
python main.py --skip-nvd
```

### Issue: Gemini API Key Not Set

**Solution**: Add your API key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

### Issue: ChromaDB Error

**Solution**: Delete and recreate:
```bash
rmdir /s chroma_db  # Windows
python main.py
```

## Next Steps

1. **Customize STRIDE patterns**: Edit `collectors/stride_collector.py` to add your own threat scenarios
2. **Add CICEVSE2024 data**: If you have the dataset, use `--cicevse-dataset path/to/data.csv`
3. **Integrate with your tools**: Use the Python API to integrate with your existing security tools
4. **Export for analysis**: Export results to JSON for further processing

## Tips

- Start with `--skip-nvd` for quick testing
- Use `--export-db` to backup your database
- Query with filters (`--severity`, `--type`) for more relevant results
- Increase `--n-context` in Gemini analysis for more comprehensive results
- Use `--full` flag in queries to see complete document content

## Support

Check `README.md` for detailed documentation and advanced features.
