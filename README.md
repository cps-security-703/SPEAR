# SPEAR: Semantic Planning and Execution of Adversarial Agentic Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenDSS](https://img.shields.io/badge/OpenDSS-9.6+-green.svg)](https://www.epri.com/pages/sa/opendss)

A research framework for adversarial security analysis of **Electric Vehicle Charging Systems (EVCS)** integrated with power distribution networks. SPEAR combines Physics-Informed Neural Networks (PINN), Reinforcement Learning (RL), Large Language Models (LLM), and Retrieval-Augmented Generation (RAG) to simulate and study coordinated cyber-physical attacks against smart charging infrastructure.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SPEAR Framework                                  │
│                                                                          │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  SPEAR-RAG   │    │          Hierarchical Co-Simulation           │   │
│  │  (ChromaDB)  │───▶│  IEEE 14-bus TX ↔ IEEE 34-bus Dist × 6      │   │
│  │  CVE/MITRE/  │    │  EVCS Stations (pinn_optimizer / evcs_dyn)   │   │
│  │  STRIDE KB   │    └──────────────────┬───────────────────────────┘   │
│  └──────────────┘                       │                                │
│                                         ▼                                │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  Gemini /    │    │       Federated PINN Manager (6 nodes)        │   │
│  │  OpenRouter  │───▶│  Local LSTM-PINN × 6 → FedAvg Aggregation   │   │
│  │  LLM         │    │  Physics Loss: converter dynamics + KVL/KCL  │   │
│  └──────────────┘    └──────────────────┬───────────────────────────┘   │
│         │                               │                                │
│         ▼                               ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │            LLM-guided RL Attack Coordinator                       │   │
│  │   LangGraph Workflow → Attack-Specific DQN/SAC Agents × 6        │   │
│  │   (voltage_manip | current_inj | power_disrupt |                  │   │
│  │    comm_spoof | data_inj | protocol_manip)                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                Robust Intrusion Detection System                  │   │
│  │               │   │
│  │     Best-IDS ( Transformer/LSTM/..., 14-D features)               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Module | File | Description |
|--------|------|-------------|
| **Main Simulation** | `enhanced_integrated_evcs_system.py` | End-to-end simulation entry point |
| **Co-Simulation** | `hierarchical_cosimulation.py` | IEEE 14+34-bus co-simulation; hosts the 3-layer IDS (L1 physical → L2 pattern → L3 `BestIDSDetector`) |
| **Federated PINN** | `federated_pinn_manager.py` | FedAvg over 6 local LSTM-PINN models |
| **PINN Optimizer** | `pinn_optimizer.py` | LSTM-PINN charging optimizer with AC-DC physics |
| **EVCS Dynamics** | `evcs_dynamics.py` | AC-DC-DC converter + SOC evolution |
| **ACN Interface** | `acn_sim_interface.py` | ACN-Data loader + PINN training sequences |
| **Attack Agents** | `attack_specific_rl_agents.py` | Per-attack-type DQN/SAC agents |
| **RL Coordinator** | `central_rl_coordinator.py` | Two-level RL (outer 18 + inner 100 episodes) |
| **Security Evasion** | `dqn_sac_security_evasion.py` | Evasion-aware RL training environments |
| **LangGraph** | `langgraph_attack_coordinator.py` | LangGraph workflow for attack orchestration |
| **LLM Coordinator** | `enhanced_llm_rl_coordinator.py` | LLM-guided RL strategy adaptation |
| **LLM Analyzer** | `gemini_llm_threat_analyzer.py` | Multi-provider LLM threat analysis (Gemini / OpenRouter) |
| **Attack Deployment** | `gemini_attack_deployment.py` | LLM prompt construction for attack planning |
| **IDS Benchmark** | `compare_ids_models_update.py` | **Trains 9 IDS classifiers on ACN-Data → selects best by composite score → saves `models/best_ids_model.pkl`** |
| **Runtime IDS** | `best_ids_model.py` | `BestIDSDetector`: loads `best_ids_model.pkl` at simulation startup; called per-timestep per station by `hierarchical_cosimulation.py` |
| **LSTM IDS Model** | `lstm_anomaly_detector.py` | `LSTMIDSModel` architecture + `LSTMIDSDetector` wrapper; used by `train_lstm_ids_acn.py` and `federated_pinn_manager.py` |
| **IDS Architectures** | `ids_neural_models.py` | Bi-LSTM+Attention, Transformer, Autoencoder IDS classes; imported by `compare_ids_models_update.py` |
| **LLM Metrics** | `llm_metrics_logger.py` | Token usage and latency logging for LLM calls |
| **Federated Integration** | `federated_evcs_integration.py` | Federated EVCS data integration layer |
| **Benign Generator** | `benign_data_generator.py` | Synthetic benign traffic generation |
| **DSS Helper** | `dss_function_qsts.py` | OpenDSS load/bus-distance utilities (imported by `pinn_optimizer.py`) |
| **CMS Benchmark** | `compare_pinn_cms_vs_acn_controllers.py` | PINN-CMS vs 5 ACN baseline controllers |
| **IDS Training** | `train_lstm_ids_acn.py` | Train `LSTMIDSModel` on real ACN-Data sessions |
| **RL Evaluation** | `evaluate_rl_evasion_comparison.py` | RL evasion vs. random-baseline comparison |
| **Publication Eval** | `evaluate_publication_ready.py` | Publication-ready evaluation pipeline |

---

## Project Structure

```
SPEAR/
│
├── enhanced_integrated_evcs_system.py  ← Main entry point
│
├── ── Core Simulation ──────────────────────────────────────
├── hierarchical_cosimulation.py        # IEEE 14+34-bus co-simulation
├── evcs_dynamics.py                    # EVCS converter physics
├── dss_function_qsts.py                # OpenDSS utility functions
├── acn_sim_interface.py                # ACN-Data interface
│
├── ── PINN & Federated Learning ────────────────────────────
├── pinn_optimizer.py                   # LSTM-PINN charging optimizer
├── federated_pinn_manager.py           # Federated PINN (6 nodes)
├── federated_evcs_integration.py       # Federated data integration
│
├── ── RL Attack System ─────────────────────────────────────
├── attack_specific_rl_agents.py        # Per-type DQN/SAC agents (6 types)
├── central_rl_coordinator.py           # Two-level RL coordinator
├── dqn_sac_security_evasion.py         # Security evasion environments
│
├── ── LLM Integration ──────────────────────────────────────
├── gemini_llm_threat_analyzer.py       # LLM threat analysis (Gemini/OpenRouter)
├── gemini_attack_deployment.py         # LLM attack deployment prompts
├── langgraph_attack_coordinator.py     # LangGraph attack orchestration
├── enhanced_llm_rl_coordinator.py      # LLM-RL joint coordination
├── llm_metrics_logger.py               # LLM call metrics & token logging
│
├── ── Intrusion Detection ──────────────────────────────────
├── compare_ids_models_update.py        # ← Run first: trains 9 IDS classifiers,
│                                       #   saves best to models/best_ids_model.pkl
├── best_ids_model.py                   # BestIDSDetector: runtime wrapper loaded
│                                       #   per-timestep by hierarchical_cosimulation.py
├── ids_neural_models.py                # Bi-LSTM+Attn, Transformer, Autoencoder classes
│                                       #   (used during benchmark training)
├── lstm_anomaly_detector.py            # LSTMIDSModel + LSTMIDSDetector
│                                       #   (used by train_lstm_ids_acn.py &
│                                       #    federated_pinn_manager.py)
│
├── ── Evaluation & Benchmarks ──────────────────────────────
├── compare_pinn_cms_vs_acn_controllers.py  # PINN-CMS vs ACN controllers
├── train_lstm_ids_acn.py               # Train LSTMIDSModel on ACN-Data
├── evaluate_rl_evasion_comparison.py   # RL evasion comparison
├── evaluate_publication_ready.py       # Publication-ready evaluation
├── benign_data_generator.py            # Synthetic benign traffic
│
├── ── IEEE Distribution System ─────────────────────────────
├── ieee34Mod1.dss                      # IEEE 34-bus OpenDSS model (variant 1)
├── ieee34Mod2.dss                      # IEEE 34-bus OpenDSS model (variant 2)
├── IEEELineCodes.DSS                   # OpenDSS line code definitions
├── IEEE34_BusXY.csv                    # Bus coordinate file
├── Run_IEEE34Mod1.dss                  # OpenDSS run script
├── Run_IEEE34Mod2.dss                  # OpenDSS run script
│
├── ── RAG Knowledge Base ───────────────────────────────────
├── spear_rag/
│   ├── chroma_db/                      # ChromaDB vector store (CVE/MITRE/STRIDE)
│   ├── collectors/                     # NVD, MITRE, STRIDE, CICEVSE data collectors
│   ├── main.py                         # RAG pipeline entry point
│   └── *.md / *.tex                    # Documentation & IEEE paper sections
│
├── ── Datasets (Sample) ────────────────────────────────────
├── evcs_data/
│   ├── ACN-Data-Static-main/
│   │   ├── time series data/caltech/   # Real EV charging CSVs (sample subset)
│   │   └── session data/               # Session metadata JSON
│   └── CICEVSE2024_NT.csv              # EVCS network traffic (10k-row sample)
│
├── ── Model Checkpoints ────────────────────────────────────
├── models/                             # LSTM-IDS checkpoints (.pth, .pkl)
├── trained_rl_agents/                  # Pre-trained DQN/SAC agent weights
├── federated_models/                   # Federated PINN round checkpoints
│
├── ── Outputs ──────────────────────────────────────────────
├── sub_figures/                        # Representative simulation output PDFs
├── plots/                              # IDS & CMS benchmark plots
├── detection_results/                  # IDS detection reports
├── attack_scenarios_logs/              # Attack scenario JSON logs
├── llm_metrics/                        # LLM token/latency metrics
├── evaluation_results/                 # Evaluation output files
│
├── requirements.txt
├── .env                                # API keys (see Environment Setup)
└── README.md
```

---

## Datasets

| Dataset | Size | Source | Used By |
|---------|------|--------|---------|
| ACN-Data-Static (Caltech) | Sample | [GitHub](https://github.com/tongxin-li/ACN-Data-Static) | `acn_sim_interface.py`, IDS benchmark, PINN training |
| CICEVSE2024-NT | Sample (10k rows) | Public | `spear_rag/collectors/cicevse_collector.py` |

> **Full dataset:** Download ACN-Data-Static from `https://github.com/tongxin-li/ACN-Data-Static` and place under `evcs_data/ACN-Data-Static-main/`.

---

## Installation

```bash
git clone https://github.com/zakariahaider14/SPEAR.git
cd SPEAR

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### System dependencies

- **OpenDSS**: Required for distribution-system simulation. Install via `pip install opendssdirect.py`.
- **CUDA**: Optional but recommended for GPU-accelerated PINN and LSTM training.

---

## Environment Setup

Create a `.env` file in the project root:

```bash
# LLM provider (choose one or both)
GEMINI_API_KEY=your_google_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# NVD API key (optional — raises rate limits for RAG database construction)
NVD_API_KEY=your_nvd_api_key
```

Switch between LLM providers in `gemini_llm_threat_analyzer.py`:

```python
USE_GEMINI: bool = False           # True → Gemini  |  False → OpenRouter
GEMINI_MODEL_NAME    = "models/gemini-2.5-flash"
OPENROUTER_MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
```

---

## Usage

The framework follows a four-stage workflow.

---

### Stage 1 — Build the RAG Knowledge Base

Populate the ChromaDB vector store with CVE, MITRE ATT&CK for ICS, and STRIDE data. A pre-built snapshot is included in `spear_rag/chroma_db/`.

```bash
cd spear_rag

# Rebuild from scratch (requires NVD_API_KEY for full CVE collection)
python main.py --nvd-max-results 150 --nvd-start-date 2022-01-01

# Skip individual collectors if needed
python main.py --skip-nvd --skip-mitre      # Only STRIDE + CICEVSE
```

**Outputs:**
- `spear_rag/chroma_db/` — Persistent vector database
- `spear_rag/top_rl_actions_for_simulation.json` — Top-6 RL actions ranked by RAG confidence

---

### Stage 2 — Train and Benchmark the IDS

#### 2a. (Optional) Pre-train the LSTM IDS model on ACN-Data

```bash
python train_lstm_ids_acn.py
```

Trains `LSTMIDSModel` (bidirectional LSTM) on real ACN-Data sessions with FDI injection. Saves checkpoint to `models/lstm_ids_pretrained.pth`. This checkpoint is one of the nine candidates evaluated in the next step.

#### 2b. Run the IDS benchmark — generates `best_ids_model.pkl`

```bash
python compare_ids_models_update.py
```

This is the **critical step for the full simulation**. It builds the ACN-Data dataset, trains all 9 IDS classifiers (LSTM-IDS, Transformer-IDS, Autoencoder-IDS, Random Forest, Gradient Boosting, Extra Trees, SVM-RBF, MLP, Isolation Forest) from scratch, ranks them by composite score, and saves the winner to `models/best_ids_model.pkl`.

`hierarchical_cosimulation.py` loads this file via `best_ids_model.BestIDSDetector` to power Layer 3 of the IDS at runtime. If the file is absent, the simulation runs with L1/L2 only.

**Selection criterion:**
```
Composite = 0.35 × Recall + 0.25 × AUC + 0.25 × (1 - FPR) + 0.15 × F1
```

**Outputs** (in `plots/ids_comparison/`):
- `IDS-01_roc_curves.png` — ROC curves for all 9 models
- `IDS-02_pr_curves.png` — Precision-Recall curves
- `IDS-03_confusion_matrices.png` — Confusion matrix grid
- `IDS-04_prf1_bar.png` — Precision / Recall / F1 / FPR bar chart
- `IDS-05_per_attack_detection.png` — Per-attack-type detection heatmap
- `IDS-06_latency_vs_f1.png` — Inference latency vs. F1 scatter
- `IDS-07_summary_table.png` — Ranked summary table

---

### Stage 3 — Compare PINN-CMS vs. ACN Baseline Controllers

```bash
python compare_pinn_cms_vs_acn_controllers.py
```

Simulates a congested 80-EV busy-day scenario at 40% grid capacity on real ACN-Data sessions. Compares:

| Controller | Strategy |
|-----------|----------|
| Uncontrolled | Full rate on plug-in |
| EDF | Earliest Deadline First |
| MaxRate | Equal allocation up to max rate |
| SortedStayFirst | Largest energy demand first |
| SortedDeptFirst | Departure time + laxity sort |
| **PINN-CMS** | Hybrid urgency × LSTM-PINN priority |

**Outputs** (in `plots/cms_comparison/`):
- `CMS-01_aggregate_power.png` — Power time-series (24h)
- `CMS-02_soc_progression.png` — Fleet state-of-charge progression
- `CMS-03_peak_utilisation.png` — Peak demand and EVSE utilisation
- `CMS-04_throughput.png` — % EVs satisfied and energy delivered
- `CMS-05_queue_depth.png` — Charging queue depth
- `CMS-06_pilot_violin.png` — Pilot signal distributions
- `CMS-07_summary_table.png` — Summary metrics table

---

### Stage 4 — Run the Full SPEAR Simulation

```bash
python enhanced_integrated_evcs_system.py
```

Runs all five phases end-to-end:

| Phase | Description | Est. Duration |
|-------|-------------|--------------|
| **1** | Load / train Federated PINN for 6 distribution systems | ~2 min |
| **2** | DQN/SAC security-evasion training (coordinated, 6 attack types) | ~10–15 min |
| **3** | Two-level RL: 18 outer × 100 inner episodes per attack type | ~4–6 h (GPU) |
| **4** | LLM-guided attack deployment (Gemini / OpenRouter) | ~5 min |
| **5** | Hierarchical co-simulation: 3 600 s IEEE 14+34-bus + EVCS dynamics | ~15–30 min |

**Key outputs:**
- `sub_figures/` — Per-run PDF figures (attack impact, frequency response, load, etc.)
- `attack_scenarios_logs/` — JSON scenario logs per simulation run
- `detection_results/ids_detection_report_<timestamp>.json` — IDS evaluation report
- `llm_metrics/` — LLM call records (token counts, latency, cost estimates)
- Console log redirected to `enhanched_evcs_system_log_<timestamp>.txt`

---

### Stage 5 — Evaluate Results

```bash
# RL evasion comparison (RL-coordinated vs. random baseline)
python evaluate_rl_evasion_comparison.py

# Publication-ready evaluation with full metric tables
python evaluate_publication_ready.py
```

---

## Attack Types

Six FDI attack types are implemented, each with a dedicated DQN/SAC agent:

| ID | Attack Type | STRIDE | Target Feature(s) | MITRE ICS |
|----|------------|--------|------------------|-----------|
| 0 | `voltage_manipulation` | Information Disclosure | Grid voltage, power surge | T0855 |
| 1 | `current_injection` | Elevation of Privilege | Current, demand factor, load | T0831 |
| 2 | `power_disruption` | Denial of Service | Power, demand, utilisation | T0814 |
| 3 | `communication_spoofing` | Spoofing | SOC, urgency, queue length | T0830 |
| 4 | `data_injection` | Tampering | Frequency, demand, grid voltage | T0832 |
| 5 | `protocol_manipulation` | Repudiation | Multi-feature oscillation | T0856 |

---

## Technical Architecture

### Robust Intrusion Detection System

The roust IDS is implemented inside `hierarchical_cosimulation.py` and fires once per timestep per EVCS station.

```

— ML Detection  (best_ids_model.BestIDSDetector)
    Loads models/best_ids_model.pkl produced by compare_ids_models_update.py
    Sliding window of 10 timesteps × 14 features → sklearn predict_proba
    Alert confirmed only when ≥1 of L1/L2 also fires (corroboration gate)
```

**IDS model selection pipeline (`compare_ids_models_update.py`):**
```
ACN-Data sessions  ──▶  build_dataset()  ──▶  14-D feature sequences
                                                        │
                        ┌───────────────────────────────┤
                        ▼                               ▼
               Train 9 classifiers:          Composite score:
               LSTM-IDS, Transformer,        0.35·Recall + 0.25·AUC
               Autoencoder, RF, GradBoost,   + 0.25·(1-FPR) + 0.15·F1
               ExtraTrees, SVM, MLP, IsoF           │
                                                     ▼
                                        Best model → models/best_ids_model.pkl
                                        (loaded at runtime by BestIDSDetector)
```

**14-D feature vector (shared by all IDS components):**
```
[SOC, voltage_norm, current_norm, power_norm, temperature_norm,
 demand_factor, load_factor, grid_voltage_pu, freq_norm, queue_length_norm,
 utilization, urgency_factor, time_of_day_norm, system_id_norm]
```

### Federated PINN Architecture

- **6 local LSTM-PINN models** — one per IEEE 34-bus distribution system
- **Physics loss** — Kirchhoff's laws + AC-DC converter efficiency + SOC dynamics
- **FedAvg aggregation** — weighted average of local gradients after each round
- **ACN-Data fine-tuning** — local models fine-tuned on real charging session CSVs

### LLM Integration (Gemini / OpenRouter)

- **Provider switching**: `USE_GEMINI` flag in `gemini_llm_threat_analyzer.py`
- **Supported models**: Gemini 2.5 Flash,Gemini 3.1 Flash via google AI studio or  GPT-4o, Claude Sonnet 4.5, DeepSeek V3, and others via OpenRouter
- **Conversation memory**: up to 20-turn rolling context + learning context
- **STRIDE/MITRE mapping**: structured prompts return JSON with T-codes, CVSS scores, and countermeasures

### RAG Knowledge Base

```
ChromaDB Vector Store (spear_rag/chroma_db/)
    ├── NVD CVE collection (EVCS-relevant, CVSS ≥ 7.0, 2022–present)
    ├── MITRE ATT&CK for ICS technique embeddings
    ├── STRIDE threat pattern corpus
    └── CICEVSE2024-NT network traffic descriptors

Retrieval → Confidence scoring → Top-K ranked RL action recommendations
```

---

## Results Summary

> See `sub_figures/` for per-run PDF figures and `detection_results/` for JSON reports `plots/` for comparative performance of IDS module or PINN based CMS 

---

## Configuration

### LLM Provider

Edit the top of `gemini_llm_threat_analyzer.py`:

```python
USE_GEMINI: bool = False                        # True → Gemini, False → OpenRouter
GEMINI_MODEL_NAME    = "models/gemini-2.5-flash"
OPENROUTER_MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
```

### Simulation Scale

Key parameters in `enhanced_integrated_evcs_system.py`:

```python
NUM_DISTRIBUTION_SYSTEMS = 6     # Number of IEEE 34-bus feeder instances
SIMULATION_DURATION      = 3600  # Seconds of simulated grid time
OUTER_EPISODES           = 18    # RL outer training episodes
INNER_EPISODES           = 100   # RL inner steps per episode
```

### IDS Benchmark Weights

Edit in `compare_ids_models_update.py`:

```python
IDS_ALPHA = 0.35   # Recall / Detection Rate weight
IDS_BETA  = 0.25   # ROC-AUC weight
IDS_GAMMA = 0.25   # Specificity (1 – FPR) weight
IDS_DELTA = 0.15   # F1-Score weight
```

---

## References

### Datasets
- **ACN-Data-Static**: T. Li *et al.*, "Adaptive Charging Network," ACM e-Energy 2021. [GitHub](https://github.com/tongxin-li/ACN-Data-Static)
- **CICEVSE2024-NT**: Canadian Institute for Cybersecurity, EV Charging Station Network Traffic, 2024.

### Frameworks
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — SAC / DQN implementations
- [PyTorch](https://pytorch.org/) — Deep learning
- [OpenDSS / opendssdirect.py](https://www.epri.com/pages/sa/opendss) — Distribution system simulation
- [ChromaDB](https://www.trychroma.com/) — Vector database for RAG
- [LangGraph](https://langchain-ai.github.io/langgraph/) — LLM workflow orchestration
- [Google Gemini API](https://ai.google.dev/) / [OpenRouter](https://openrouter.ai/) — LLM providers

### Standards & Threat Models
- MITRE ATT&CK for ICS: [attack.mitre.org/matrices/ics](https://attack.mitre.org/matrices/ics/)
- STRIDE Threat Modeling: Microsoft SDL
- IEEE 14-Bus / 34-Bus Test Systems: IEEE PES
- IEC 61850 / OCPP / DNP3 protocol attack surfaces

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

This framework is intended exclusively for **academic and defensive security research**. All attack simulations are conducted in isolated digital environments. Do not use any component of this framework against real infrastructure without explicit written authorization from the system owner.
