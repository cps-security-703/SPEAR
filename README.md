#SPEAR: Semantic Planning and Execution of Adversarial  Agentic Reinforcement Learning for Cyber Physical System
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for adversarial attack research on Electric Vehicle Charging Systems (EVCS) integrated with power distribution networks. This project combines **Physics-Informed Neural Networks (PINN)**, **Reinforcement Learning (RL)**, **Large Language Models (LLM/Gemini)**, and **Retrieval-Augmented Generation (RAG)** for intelligent cyber-physical attack simulation and intrusion detection.

## 📥 Download

**Full Dataset & Pre-trained Models:** [Google Drive Link](https://drive.google.com/drive/folders/1THSModNtE3cSgRIewFD0Yr07_a379vc5?usp=sharing)

> The Google Drive contains pre-trained PINN models, LSTM IDS checkpoints, trained RL agents, and the Chrom)aDB vector database.


### Key Components

| Component | Description |
|-----------|-------------|
| **SPEAR RAG** | Retrieval-Augmented Generation system with CVE, MITRE ATT&CK, and STRIDE threat intelligence |
| **LSTM IDS** | 3-layer Intrusion Detection System (Physical + Pattern + LSTM ML) |
| **Federated PINN** | Physics-Informed Neural Networks for 6 distribution systems with federated learning |
| **DQN/SAC Agents** | Deep Q-Network and Soft Actor-Critic agents for 6 attack types |
| **Gemini LLM** | Google Gemini integration for strategic attack coordination |
| **Hierarchical Co-Sim** | IEEE 14-bus transmission + IEEE 34-bus distribution + EVCS dynamics |

---

## 📁 Project Structure

```
SPEAR/
├── spear_rag/                      # RAG Knowledge Base System
│   ├── collectors/                 # Data collectors (NVD, MITRE, STRIDE, etc.)
│   ├── vector_db/                  # ChromaDB vector database
│   ├── chroma_db/                  # Persistent vector storage
│   ├── main.py                     # RAG pipeline entry point
│   ├── evaluate_stride_mitre_rl_with_confidence.py
│   └── visualize_confidence_results.py
│
├── train_lstm_ids.py               # LSTM IDS training script
├── run_baseline_attacks_actual_system.py  # Baseline attack evaluation
├── compare_rl_vs_baseline_actual.py       # RL vs Baseline comparison
├── enhanced_integrated_evcs_system.py     # Main simulation system
│
├── hierarchical_cosimulation.py    # IEEE 14+34 bus co-simulation
├── federated_pinn_manager.py       # Federated PINN coordination
├── pinn_optimizer.py               # LSTM-PINN charging optimizer
├── lstm_anomaly_detector.py        # LSTM-based anomaly detection
├── lstm_ids_evcs.py                # LSTM IDS model architecture
│
├── attack_specific_rl_agents.py    # Attack-specific DQN/SAC agents
├── central_rl_coordinator.py       # Two-level RL training coordinator
├── dqn_sac_security_evasion.py     # Security evasion environments
│
├── gemini_llm_threat_analyzer.py   # Gemini LLM integration
├── gemini_attack_deployment.py     # Gemini attack deployment prompts
├── enhanced_llm_rl_coordinator.py  # LangGraph workflow coordinator
│
├── trained_rl_agents/              # Pre-trained DQN/SAC models
├── federated_models/               # Federated PINN checkpoints
├── models/                         # LSTM IDS checkpoints
├── detection_results/              # Evaluation outputs
│
├── ieee34Mod1.dss                  # OpenDSS distribution model
├── ieee34Mod2.dss                  # Alternative distribution model
├── requirements_gemini.txt         # Main dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Google Gemini API key (for LLM features)
- NVD API key (optional, for RAG database creation)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SPEAR.git
cd SPEAR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install main dependencies
pip install -r requirements_gemini.txt

# Install RAG dependencies
pip install -r spear_rag/requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Google Gemini API Key (required for LLM features)
GOOGLE_API_KEY=your_gemini_api_key_here

# NVD API Key (optional, for higher rate limits)
NVD_API_KEY=your_nvd_api_key_here
```

---

## 📖 Usage Guide

The framework follows a **three-stage workflow**:

### Stage 1: Create RAG Knowledge Base

Build the vulnerability vector database with CVEs, MITRE ATT&CK techniques, and STRIDE patterns.

```bash
cd spear_rag

# Create the full database
python main.py --nvd-max-results 150 --nvd-start-date 2022-01-01

# Options:
#   --skip-nvd          Skip NVD CVE collection
#   --skip-mitre        Skip MITRE ATT&CK collection
#   --skip-stride       Skip STRIDE pattern creation
#   --export-db FILE    Export database to JSON
```

**Evaluate STRIDE-MITRE-RL Mappings with Confidence Scoring:**

```bash
# Run confidence-based evaluation
python evaluate_stride_mitre_rl_with_confidence.py

# Generate visualization plots
python visualize_confidence_results.py
```

**Output:**
- `chroma_db/` — Persistent ChromaDB vector database
- `top_rl_actions_for_simulation.json` — Top 6 RL actions ranked by confidence
- `plots/` — RAG vs Non-RAG comparison visualizations

---

### Stage 2: Train LSTM IDS & Run Baseline

Train the 3-layer Intrusion Detection System and establish baseline attack performance.

#### 2.1 Train LSTM IDS

```bash
# Train LSTM IDS with default parameters
python train_lstm_ids.py

# Custom training parameters
python train_lstm_ids.py \
    --epochs 500 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_sequences 12000 \
    --val_sequences 3000 \
    --attack_ratio 0.2 \
    --output_dir models
```

**Output:**
- `models/lstm_ids_pretrained.pth` — Trained LSTM model
- `lstm_ids_best_balanced.pth` — Best checkpoint (by F1-score)
- `models/lstm_ids_training_history.png` — Training curves
- `models/lstm_ids_training_report.txt` — Performance metrics

#### 2.2 Run Baseline Attacks

Execute random (non-RL) attacks through the actual system to establish baseline:

```bash
python run_baseline_attacks_actual_system.py
```

**Output:**
- `detection_results/baseline_actual_system_TIMESTAMP.json`

#### 2.3 Compare RL vs Baseline

After running the main simulation (Stage 3), compare RL-coordinated attacks against baseline:

```bash
python compare_rl_vs_baseline_actual.py
```

**Output:**
- `detection_results/actual_system_comparison_TIMESTAMP.json`
- `detection_results/actual_system_comparison_TIMESTAMP.png`

---

### Stage 3: Validate Federated PINN-Based EVCS-CMS Dynamics 

```bash
python evaluate_federated_pinn.py 
```

After running the above file, the corresponding results will be saved in federated_pinn_results.


### Stage 4: Run Full Simulation

Execute the complete hierarchical co-simulation with RL-coordinated attacks.

```bash
python enhanced_integrated_evcs_system.py
```

**Simulation Phases:**

| Phase | Description | Duration |
|-------|-------------|----------|
| **Phase 1** | Load pre-trained PINN models for 6 distribution systems | ~1 min |
| **Phase 2** | DQN/SAC security evasion training (coordinated) | ~10-15 min |
| **Phase 3** | Two-level RL training (18 outer × 100 inner episodes) | ~300-400 min |
| **Phase 4** | Gemini-guided attack deployment | ~5 min |
| **Phase 5** | Hierarchical co-simulation (3600s simulated) | ~10-20 min |

**Key Outputs:**
- `rl_training_rewards_6_systems.png` — Training reward curves
- `reward_history_gemini_guided.json` — Detailed training history
- `rag_enhanced_deployment_plan_TIMESTAMP.json` — Final attack deployment
- `detection_results/ids_detection_report_TIMESTAMP.json` — IDS evaluation
- Hierarchical co-simulation plots (voltage, frequency, power, etc.)

---

## 🎯 Attack Types

The framework supports 6 RL-trained attack types, each mapped to STRIDE categories:

| Attack Type | STRIDE Category | Target Protocol | Description |
|-------------|-----------------|-----------------|-------------|
| `voltage_manipulation` | Information Disclosure | DNP3 | Manipulate voltage setpoints |
| `current_injection` | Elevation of Privilege | OCPP/TCP | Inject false current readings |
| `power_disruption` | Denial of Service | TCP/IEC | Disrupt power flow control |
| `communication_spoofing` | Spoofing | OCPP | Spoof EVCS-CMS communication |
| `data_injection` | Tampering | DNP3 | Inject malicious data packets |
| `protocol_manipulation` | Repudiation | DNP3 | Manipulate protocol messages |

---

## 🔬 Technical Details

### 3-Layer Intrusion Detection System

```
Layer 1: Physical Constraints
├── SOC bounds (0.05-0.95)
├── Grid voltage (0.85-1.15 pu)
├── Grid frequency (59.5-60.5 Hz)
└── Demand/Load factors

Layer 2: Pattern Detection
├── Rate-of-change analysis
├── Oscillation detection
└── Correlation checks

Layer 3: LSTM ML Detection
├── 10-step sequence buffer
├── Classifier softmax P(attack)
└── Corroboration with L1/L2
```

### Federated PINN Architecture

- **6 Local Models**: One per distribution system
- **Global Aggregation**: FedAvg with weighted averaging
- **Input Features**: 14 EVCS operational parameters
- **Output**: Voltage, Current, Power references
- **Physics Loss**: Converter dynamics + efficiency constraints


## 📊 Results & Evaluation

### Sample Results

| Metric | Baseline (Random) | RL-Coordinated | Improvement |
|--------|-------------------|----------------|-------------|
| Detection Rate | 96.7% | 15-25% | ↓75% |
| Attack Success | 65% | 85% | ↑20% |
| Evasion Rate | 3.3% | 75-85% | ↑72% |
| Avg Anomaly Score | 0.85 | 0.25 | ↓70% |

### Visualization Outputs

- **Training Rewards**: Per-system episode rewards with outer-circle boundaries
- **IDS Comparison**: Baseline vs RL detection rates, precision, recall, F1
- **RAG Confidence**: CVE/MITRE/Protocol scoring breakdown
- **Co-Simulation**: Voltage, frequency, power, charging time plots

---

## 🛠️ Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `spear_rag/config.py` | RAG database settings |
| `attack_magnitude_config.py` | Attack parameter ranges |
| `knowledgebase_mapping.md` | STRIDE-MITRE-RL mappings |

### Simulation Parameters

Edit in `enhanced_integrated_evcs_system.py`:


## 📚 References

### Frameworks & Libraries

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
- [PyTorch](https://pytorch.org/) — Deep learning
- [OpenDSS](https://www.epri.com/pages/sa/opendss) — Distribution system simulation
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Google Gemini](https://ai.google.dev/) — LLM integration
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Workflow orchestration

### Academic References

- MITRE ATT&CK for ICS: https://attack.mitre.org/matrices/ics/
- STRIDE Threat Modeling: Microsoft SDL
- IEEE 14-Bus / 34-Bus Test Systems
- CICEVSE2024 Dataset

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the maintainers.

---

## ⚠️ Disclaimer

This research framework is intended for **academic and defensive security research only**. The attack simulations are conducted in isolated environments and should not be used against real infrastructure without proper authorization.
