# ARES RAG Project Summary

## Project Overview

This project implements a comprehensive **Retrieval-Augmented Generation (RAG) system** for vulnerability analysis of transmission distribution systems enriched with Electric Vehicle Supply Equipment (EVSE). The system integrates multiple data sources and uses Google Gemini for AI-powered vulnerability analysis and attack strategy generation for reinforcement learning agents.

## Objectives

1. **Create a comprehensive vector database** containing vulnerability information from multiple authoritative sources
2. **Integrate MITRE ATT&CK for ICS and STRIDE** threat modeling frameworks
3. **Provide context-aware vulnerability analysis** using Google Gemini
4. **Support RL agent development** by suggesting attack strategies based on real vulnerabilities
5. **Enable semantic search** across vulnerability data for research and security analysis

## System Components

### Target Systems Covered

- **EVSE** (Electric Vehicle Supply Equipment)
- **Charging Management System (CMS)**
- **Central Charging Management System (CCMS)**
- **Distribution Management System (DMS)**
- **Automatic Generation Control (AGC)**
- **SCADA/ICS** systems
- **Power Grid** infrastructure
- **Distributed Energy Resources (DER)**

## Data Sources

### 1. NVD CVE Database
- **Source**: National Vulnerability Database
- **Coverage**: CVEs from 2022-present
- **Focus Keywords**: EVSE, AGC, Power Systems, SCADA, ICS, Smart Grid
- **API**: NVD REST API 2.0
- **Documents**: ~100 CVEs (configurable)

### 2. MITRE ATT&CK for ICS
- **Source**: MITRE ATT&CK STIX data
- **Coverage**: All ICS attack techniques
- **Mapping**: Techniques mapped to STRIDE categories
- **Documents**: ~100 techniques

### 3. STRIDE Threat Patterns
- **Source**: Custom-created patterns
- **Coverage**: 6 STRIDE categories × 3 scenarios each = 18 patterns
- **Focus**: EVSE and power system specific threats
- **Categories**:
  - Spoofing (EV auth, OCPP messages, grid operators)
  - Tampering (metering, firmware, algorithms)
  - Repudiation (transactions, commands)
  - Information Disclosure (PII, topology, patterns)
  - Denial of Service (flooding, AGC, coordinated)
  - Elevation of Privilege (API, SCADA, database)

### 4. MITRE-STRIDE Mappings
- **Source**: Custom mappings
- **Coverage**: 10 key MITRE techniques mapped to STRIDE
- **Context**: EVSE-specific applicability and scenarios
- **Documents**: 10 comprehensive mappings

### 5. CICEVSE2024 Dataset (Optional)
- **Source**: CIC EVSE 2024 network traffic dataset
- **Coverage**: Network attack patterns on EV charging infrastructure
- **Processing**: Attack pattern extraction and analysis
- **Documents**: Variable (based on dataset)

## Technical Architecture

### Vector Database
- **Technology**: ChromaDB
- **Embedding Model**: sentence-transformers (all-MiniLM-L6-v2)
- **Embedding Dimension**: 384
- **Storage**: Persistent local storage
- **Features**: Semantic search, metadata filtering, similarity scoring

### Document Schema
Each document contains:
- **Core**: ID, type, title, description, source, dates
- **Security**: STRIDE categories, MITRE tactics/techniques, severity, CVSS
- **Context**: Affected systems, components, industry sectors
- **Actionable**: Mitigations, detections, defensive actions
- **Metadata**: Keywords, relevance tags, embedding text

### RAG Integration
- **LLM**: Google Gemini (gemini-pro)
- **Retrieval**: Top-k semantic search with optional filters
- **Context**: Formatted document metadata and content
- **Prompts**: Specialized for vulnerability analysis and RL attack strategies

## Key Features

### 1. Multi-Source Data Collection
- Automated collection from NVD, MITRE, and custom sources
- Rate-limited API calls with error handling
- Raw and processed data storage

### 2. Semantic Search
- Vector-based similarity search
- Metadata filtering (severity, type, CVSS, etc.)
- Relevance scoring

### 3. Gemini-Powered Analysis
- **Vulnerability Analysis**: MITRE + STRIDE framework analysis
- **RL Attack Strategies**: Action space, reward function, attack sequences
- **Interactive Q&A**: Context-aware responses

### 4. Flexible Querying
- Command-line interface
- Programmatic Python API
- Export capabilities (JSON)

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
├── examples/               # Usage examples
├── data/                   # Data storage
│   ├── raw/               # Raw collected data
│   └── processed/         # Processed documents
├── chroma_db/             # ChromaDB persistent storage
├── config.py              # Configuration
├── schemas.py             # Pydantic data schemas
├── pipeline.py            # Main data pipeline
├── gemini_rag.py          # Gemini RAG integration
├── main.py                # Main execution script
├── query_db.py            # Database query script
├── gemini_analysis.py     # Gemini analysis script
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick start guide
└── PROJECT_SUMMARY.md     # This file
```

## Usage Workflow

### Phase 1: Database Creation
```bash
python main.py --nvd-max-results 100
```

**Pipeline Steps:**
1. Collect NVD CVEs (30-45 min with rate limiting)
2. Collect MITRE techniques (5 min)
3. Generate STRIDE patterns (instant)
4. Create MITRE-STRIDE mappings (instant)
5. Process CICEVSE2024 if provided (variable)
6. Generate embeddings (5-10 min)
7. Populate vector database (instant)

**Total Time**: ~45-60 minutes for full database

### Phase 2: Querying
```bash
# Direct database query
python query_db.py "authentication vulnerabilities in EVSE" --n-results 10

# Filtered query
python query_db.py "DoS attacks" --severity High --type vulnerability
```

### Phase 3: Analysis with Gemini
```bash
# Vulnerability analysis
python gemini_analysis.py analyze "Your system description"

# RL attack strategies
python gemini_analysis.py rl-attack "System" "Objective"

# Interactive Q&A
python gemini_analysis.py query "Your question"
```

## Use Cases

### 1. Security Research
- Identify vulnerabilities in EVSE systems
- Understand attack patterns and techniques
- Map threats to MITRE ATT&CK and STRIDE

### 2. Vulnerability Assessment
- Analyze specific system configurations
- Get MITRE and STRIDE-based threat analysis
- Receive mitigation recommendations

### 3. RL Agent Development
- Design attack scenarios for security testing
- Define action spaces and reward functions
- Create realistic attack sequences

### 4. Threat Intelligence
- Stay updated on latest CVEs
- Understand emerging attack techniques
- Cross-reference multiple frameworks

## Key Innovations

1. **Multi-Framework Integration**: Combines NVD, MITRE ATT&CK, and STRIDE in one system
2. **EVSE-Specific Context**: Tailored threat patterns for EV charging infrastructure
3. **MITRE-STRIDE Mappings**: Novel mappings with EVSE applicability
4. **RAG-Enhanced Analysis**: Context-aware AI analysis using Gemini
5. **RL Agent Support**: Specialized prompts for attack strategy generation

## Performance Metrics

### Database Size (Typical)
- **Total Documents**: ~228 documents
  - NVD CVEs: ~100
  - MITRE Techniques: ~100
  - STRIDE Patterns: 18
  - MITRE-STRIDE Mappings: 10
  - CICEVSE2024: Variable

### Query Performance
- **Embedding Generation**: ~50ms per query
- **Vector Search**: <100ms for top-10 results
- **Gemini Analysis**: 5-15 seconds (depends on context size)

### Storage
- **Vector Database**: ~50-100 MB
- **Raw Data**: ~10-20 MB
- **Processed Data**: ~5-10 MB

## Configuration

### API Keys Required
1. **NVD API Key** (optional): Higher rate limits (50 req/30s vs 5 req/30s)
2. **Google Gemini API Key** (required for RAG): Free tier available

### Customization Options
- Embedding model selection
- Collection size limits
- Target system keywords
- STRIDE pattern scenarios
- Query filters and parameters

## Future Enhancements

### Planned Features
1. **Additional Data Sources**:
   - ICS-CERT advisories
   - Academic papers
   - Security blogs and reports

2. **Advanced Analytics**:
   - Trend analysis over time
   - Vulnerability correlation
   - Attack chain reconstruction

3. **Integration Options**:
   - REST API server
   - Web interface
   - Integration with security tools (SIEM, scanners)

4. **Enhanced RAG**:
   - Fine-tuned embeddings for security domain
   - Multi-LLM support (Claude, GPT-4)
   - Automated report generation

5. **Real-Time Updates**:
   - Automated CVE monitoring
   - Incremental database updates
   - Alert system for new vulnerabilities

## Dependencies

### Core Libraries
- **chromadb**: Vector database
- **sentence-transformers**: Embeddings
- **google-generativeai**: Gemini integration
- **pydantic**: Data validation
- **loguru**: Logging
- **requests**: API calls
- **pandas**: Data processing

### Python Version
- Python 3.8+

## Limitations

1. **NVD Rate Limits**: Slow collection without API key
2. **Gemini API**: Requires internet connection and API key
3. **CICEVSE2024**: Dataset not included, must be obtained separately
4. **Embedding Model**: Fixed dimension (384), changing requires re-embedding
5. **Local Storage**: Database grows with more documents

## Security Considerations

1. **API Keys**: Store securely in `.env` file, never commit to version control
2. **Data Privacy**: Be cautious with sensitive system descriptions in Gemini queries
3. **Rate Limiting**: Respect API rate limits to avoid blocking
4. **Data Validation**: All inputs validated through Pydantic schemas

## Conclusion

This ARES RAG system provides a comprehensive, AI-enhanced platform for vulnerability analysis of EVSE and power systems. By integrating multiple authoritative data sources with advanced RAG capabilities, it enables researchers and security professionals to:

- Quickly identify relevant vulnerabilities
- Understand attack techniques through multiple frameworks
- Generate actionable security insights
- Design realistic attack scenarios for testing
- Stay current with evolving threats

The system is designed to be extensible, allowing for additional data sources, custom threat patterns, and integration with existing security workflows.

## Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Configure**: Set `GOOGLE_API_KEY` in `.env`
3. **Create DB**: `python main.py`
4. **Query**: `python query_db.py "your query"`
5. **Analyze**: `python gemini_analysis.py analyze "your system"`

For detailed instructions, see `QUICKSTART.md` and `README.md`.
