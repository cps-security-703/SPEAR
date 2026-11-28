# ARES RAG: Vulnerability Vector Database for EVSE and Power Systems

A comprehensive Retrieval-Augmented Generation (RAG) system for vulnerability analysis of Electric Vehicle Supply Equipment (EVSE), Automatic Generation Control (AGC), and power distribution systems. This system integrates MITRE ATT&CK for ICS, STRIDE threat modeling, NVD CVE data, and the CICEVSE2024 dataset to provide context-aware vulnerability analysis using Google Gemini.

## Overview

This project creates a vector database containing vulnerability information from multiple sources:

1. **NVD CVE Database** - Recent vulnerabilities related to EVSE, AGC, power systems, and ICS
2. **MITRE ATT&CK for ICS** - Industrial control system attack techniques
3. **STRIDE Threat Patterns** - Custom threat patterns for EVSE and power systems
4. **MITRE-STRIDE Mappings** - Comprehensive mappings between MITRE techniques and STRIDE categories
5. **CICEVSE2024 Dataset** - Network attack patterns from EV charging infrastructure

The system uses this knowledge base to:
- Perform vulnerability analysis using Gemini
- Suggest attack strategies for RL agents
- Provide context-aware security recommendations

## Project Structure

```
ares_rag/
├── collectors/              # Data collection modules
│   ├── nvd_collector.py    # NVD CVE collector
│   ├── mitre_collector.py  # MITRE ATT&CK collector
│   ├── stride_collector.py # STRIDE pattern creator
│   ├── cicevse_collector.py # CICEVSE2024 processor
│   └── mitre_stride_mapper.py # MITRE-STRIDE mapper
├── vector_db/              # Vector database components
│   ├── embedder.py         # Document embedding
│   └── chroma_manager.py   # ChromaDB manager
├── data/                   # Data storage
│   ├── raw/               # Raw collected data
│   └── processed/         # Processed documents
├── chroma_db/             # ChromaDB persistent storage
├── config.py              # Configuration
├── schemas.py             # Data schemas
├── pipeline.py            # Main pipeline
├── gemini_rag.py          # Gemini RAG integration
├── main.py                # Main execution script
├── query_db.py            # Database query script
├── gemini_analysis.py     # Gemini analysis script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
# NVD API Key (optional - for higher rate limits)
NVD_API_KEY=your_nvd_api_key_here

# Google Gemini API Key (required for RAG analysis)
GOOGLE_API_KEY=your_google_api_key_here
```

**Note**: The NVD API key is already included in the configuration. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

### Step 1: Create the Vector Database

Run the main pipeline to collect data and create the vector database:

```bash
python main.py
```

**Options:**

```bash
# Collect specific number of CVEs
python main.py --nvd-max-results 150

# Specify start date for CVE collection
python main.py --nvd-start-date 2023-01-01

# Include CICEVSE2024 dataset
python main.py --cicevse-dataset path/to/cicevse2024.csv

# Skip specific data sources
python main.py --skip-nvd --skip-cicevse

# Export database to JSON
python main.py --export-db database_export.json
```

**Pipeline Phases:**

1. **Phase 1**: Collect NVD CVE data (EVSE, AGC, Power Systems)
2. **Phase 2**: Collect MITRE ATT&CK for ICS techniques
3. **Phase 3**: Create STRIDE threat patterns
4. **Phase 4**: Create MITRE-STRIDE mappings
5. **Phase 5**: Process CICEVSE2024 dataset (if provided)
6. **Phase 6**: Generate embeddings and populate vector database

### Step 2: Query the Database

Query the vector database directly:

```bash
# Basic query
python query_db.py "authentication vulnerabilities in EV charging stations"

# Query with filters
python query_db.py "DoS attacks" --severity High --n-results 10

# Filter by document type
python query_db.py "SCADA vulnerabilities" --type vulnerability --min-cvss 7.0

# Export results
python query_db.py "AGC control attacks" --export results.json --full
```

### Step 3: Perform Vulnerability Analysis with Gemini

#### Analyze System Vulnerabilities

```bash
python gemini_analysis.py analyze "EVSE system with OCPP 1.6 protocol, connected to central management system via cellular network, using basic authentication"
```

With specific attack scenario:

```bash
python gemini_analysis.py analyze "Distribution management system with AGC" --attack-scenario "Man-in-the-middle attack on control signals"
```

#### Generate RL Attack Strategies

```bash
python gemini_analysis.py rl-attack "EV charging network with 100 stations" "Maximize grid instability through coordinated charging manipulation"
```

#### Interactive Query

```bash
python gemini_analysis.py query "What are the most critical vulnerabilities in ISO 15118 implementation?"
```

## Use Cases

### 1. Vulnerability Assessment

Analyze a specific system configuration for vulnerabilities:

```python
from gemini_rag import GeminiRAG

rag = GeminiRAG()

system = """
Electric vehicle charging station network with:
- 50 DC fast chargers
- OCPP 2.0.1 protocol
- Cloud-based management system
- Integration with utility DMS
- Payment processing system
"""

analysis = rag.analyze_vulnerability(system)
print(analysis)
```

### 2. RL Agent Attack Strategy Design

Design attack strategies for reinforcement learning agents:

```python
from gemini_rag import GeminiRAG

rag = GeminiRAG()

system = "Transmission distribution system with EV charging integration"
objective = "Learn optimal attack sequences to disrupt grid frequency control"

strategies = rag.suggest_rl_attack_strategies(system, objective)
print(strategies)
```

### 3. Direct Database Queries

Query the vector database programmatically:

```python
from vector_db import ChromaDBManager, DocumentEmbedder

db = ChromaDBManager()
embedder = DocumentEmbedder()

query = "STRIDE threats for charging management systems"
query_embedding = embedder.embed_text(query)

results = db.query(
    query_embedding=query_embedding,
    n_results=5,
    where={"severity": "High"}
)

for i, doc_id in enumerate(results['ids'][0]):
    print(f"Result {i+1}: {results['metadatas'][0][i]['title']}")
```

## Data Sources

### NVD CVE Database

Keywords searched:
- SCADA, ICS, industrial control
- Electric vehicle, EV charging, EVSE
- Power grid, smart grid, substation
- DER, distributed energy, microgrid
- AGC, automatic generation control
- Energy management, DMS, ADMS

### MITRE ATT&CK for ICS

All techniques from the ICS matrix, mapped to STRIDE categories with EVSE-specific context.

### STRIDE Patterns

Custom threat patterns covering:
- Spoofing (authentication, OCPP messages, grid operators)
- Tampering (metering data, firmware, algorithms)
- Repudiation (transactions, control commands)
- Information Disclosure (PII, grid topology, charging patterns)
- Denial of Service (flooding, AGC disruption, coordinated attacks)
- Elevation of Privilege (API, SCADA, database)

### MITRE-STRIDE Mappings

Comprehensive mappings between MITRE ATT&CK techniques and STRIDE categories, with:
- EVSE-specific applicability
- Attack scenarios
- Detection strategies
- Mitigation approaches

## Configuration

Edit `config.py` or use environment variables:

```python
# Vector Database
COLLECTION_NAME = "evse_vulnerability_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Data Collection
MAX_CVE_RESULTS = 100
MAX_MITRE_TECHNIQUES = 60
MAX_STRIDE_PATTERNS = 40

# Target Systems
TARGET_SYSTEMS = [
    "EVSE", "AGC", "DMS", "CMS", "CCMS",
    "SCADA", "ICS", "Power Grid", "DER"
]
```

## Database Statistics

View database statistics:

```python
from vector_db import ChromaDBManager

db = ChromaDBManager()
stats = db.get_collection_stats()

print(f"Total documents: {stats['total_documents']}")
print(f"Document types: {stats['document_types']}")
print(f"Severity distribution: {stats['severity_distribution']}")
```

## Advanced Features

### Custom Filters

Query with multiple filters:

```python
results = db.query_by_filters(
    query_embedding=embedding,
    severity="High",
    doc_type="vulnerability",
    min_cvss=7.0,
    affected_system="EVSE"
)
```

### Export and Backup

```bash
# Export entire database
python main.py --export-db backup.json

# Export query results
python query_db.py "your query" --export results.json
```

### Reset Database

```python
from vector_db import ChromaDBManager

db = ChromaDBManager()
db.reset_collection()  # Warning: Deletes all data
```

## Integration with Gemini

The system is designed to work seamlessly with Google Gemini for:

1. **Context Retrieval**: Automatically retrieves relevant documents from the vector database
2. **Prompt Engineering**: Formats context and queries for optimal Gemini responses
3. **MITRE & STRIDE Analysis**: Provides structured analysis using both frameworks
4. **RL Agent Guidance**: Generates actionable attack strategies for RL agents

## Future Enhancements

Potential additions to the system:

1. Additional data sources (ICS-CERT advisories, academic papers)
2. Real-time vulnerability feeds
3. Custom embedding models fine-tuned for security domain
4. Integration with other LLMs (Claude, GPT-4)
5. Web interface for easier interaction
6. Automated vulnerability scanning integration

## Troubleshooting

### ChromaDB Issues

If you encounter ChromaDB errors, try:

```bash
# Delete and recreate the database
rm -rf chroma_db/
python main.py
```

### API Rate Limits

NVD API has rate limits:
- Without API key: 5 requests per 30 seconds
- With API key: 50 requests per 30 seconds

The collector automatically handles rate limiting.

### Gemini API Errors

Ensure your API key is set correctly:

```bash
export GOOGLE_API_KEY=your_key_here
```

Or add it to `.env` file.

## License

This project is for research and educational purposes.

## Citation

If you use this system in your research, please cite:

```
ARES RAG: Vulnerability Vector Database for EVSE and Power Systems
[Your Institution/Research Group]
2024
```

## Contact

For questions or issues, please contact [your contact information].
