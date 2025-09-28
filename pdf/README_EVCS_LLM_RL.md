# EVCS LLM-RL Attack Analytics System

A comprehensive system that integrates Large Language Models (LLMs) with Reinforcement Learning (RL) for advanced attack analytics on Electric Vehicle Charging Station (EVCS) systems with federated PINN models.

## 🚀 Features

- **LLM-Powered Threat Analysis**: Uses Ollama with deepseek-r1:8b for intelligent vulnerability assessment
- **STRIDE-MITRE Threat Mapping**: Maps STRIDE threat categories to MITRE ATT&CK techniques
- **Advanced RL Agents**: Deep Q-Network (DQN) and PPO agents for adaptive attack strategies
- **Federated PINN Integration**: Targets federated learning vulnerabilities in PINN models
- **Comprehensive Visualization**: Detailed analysis and performance visualizations
- **Real-time Attack Coordination**: Multi-agent coordination for sophisticated attacks

## 📋 Prerequisites

### Required Dependencies
```bash
pip install torch numpy matplotlib seaborn scikit-learn pandas
pip install openai  # For Ollama integration
```

### Ollama Setup
1. Install Ollama: https://ollama.ai/
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the deepseek-r1:8b model:
   ```bash
   ollama pull deepseek-r1:8b
   ```

### Optional Dependencies
```bash
pip install pandapower  # For realistic power system modeling
pip install opendssdirect  # For OpenDSS integration
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EVCS LLM-RL System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    LLM      │  │   STRIDE    │  │     RL      │        │
│  │  Analyzer   │  │   MITRE     │  │   Agents    │        │
│  │ (Ollama)    │  │  Mapper     │  │  (DQN/PPO)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Attack Coordinator                         │
│  └─────────────────────────────────────────────────────────┤
│                          │                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │            Federated PINN EVCS System                   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │   Charging  │  │   PINN      │  │   Security  │    │
│  │  │ Management  │  │  Models     │  │  Detection  │    │
│  │  │   System    │  │             │  │   System    │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from evcs_llm_rl_integration import EVCSLLMRLSystem

# Initialize the system
config = {
    'ollama': {
        'base_url': 'http://localhost:11434/v1',
        'model': 'deepseek-r1:8b'
    },
    'rl': {
        'state_dim': 50,
        'action_dim': 8,
        'hidden_dim': 256
    }
}

system = EVCSLLMRLSystem(config)

# Run attack simulation
results = system.run_attack_simulation("SCENARIO_001", episodes=50)
print(f"Average Reward: {results['performance_metrics']['average_reward']:.2f}")
```

### 2. Federated PINN Integration

```python
from federated_pinn_llm_rl_integration import FederatedPINNLLMRLSystem

# Initialize federated system
federated_system = FederatedPINNLLMRLSystem(config)

# Run federated attack simulation
results = federated_system.run_federated_attack_simulation("FED_PINN_001", episodes=50)
print(f"PINN Corruption: {results['performance_metrics']['average_pinn_corruption']:.1%}")
```

### 3. Complete Demo

```python
# Run the complete demonstration
python demo_evcs_llm_rl_system.py
```

## 📊 Attack Scenarios

### Basic EVCS Scenarios

1. **Stealth Grid Manipulation** (`SCENARIO_001`)
   - Target: Charging controllers, grid interface
   - Goal: Manipulate grid frequency through coordinated attacks
   - Stealth Requirement: 80%

2. **PINN Model Poisoning** (`SCENARIO_002`)
   - Target: Federated PINN, CMS communication
   - Goal: Poison federated learning model
   - Stealth Requirement: 90%

3. **Charging Session Hijacking** (`SCENARIO_003`)
   - Target: Charging sessions, payment system
   - Goal: Economic disruption through session hijacking
   - Stealth Requirement: 60%

4. **Communication Protocol Exploitation** (`SCENARIO_004`)
   - Target: Communication protocols, authentication
   - Goal: System takeover through protocol exploitation
   - Stealth Requirement: 70%

### Federated PINN Scenarios

1. **Federated Model Poisoning** (`FED_PINN_001`)
   - Target: Federated PINN, local models, aggregation server
   - Goal: Poison federated learning through malicious updates
   - Stealth Requirement: 90%

2. **Gradient Manipulation Attack** (`FED_PINN_002`)
   - Target: Gradient updates, optimization process
   - Goal: Corrupt global model through gradient manipulation
   - Stealth Requirement: 85%

3. **Coordinated Multi-Client Attack** (`FED_PINN_003`)
   - Target: Multiple clients, communication protocol
   - Goal: Coordinate attacks across federated clients
   - Stealth Requirement: 70%

4. **Backdoor Injection Attack** (`FED_PINN_004`)
   - Target: Model parameters, inference engine
   - Goal: Inject backdoors into federated model
   - Stealth Requirement: 95%

## 🔧 Configuration

### System Configuration

```python
config = {
    'ollama': {
        'base_url': 'http://localhost:11434/v1',
        'model': 'deepseek-r1:8b'
    },
    'federated_pinn': {
        'num_distribution_systems': 6,
        'local_epochs': 100,
        'global_rounds': 10,
        'aggregation_method': 'fedavg',
        'model_path': 'federated_models'
    },
    'rl': {
        'state_dim': 60,
        'action_dim': 10,
        'hidden_dim': 512,
        'num_coordinator_agents': 4
    },
    'evcs': {
        'num_stations': 6,
        'max_power_per_station': 1000,
        'voltage_limits': {'min': 0.95, 'max': 1.05},
        'frequency_limits': {'min': 59.5, 'max': 60.5}
    },
    'simulation': {
        'max_episodes': 100,
        'max_steps_per_episode': 300,
        'time_step': 1.0,
        'federated_attack_probability': 0.4
    }
}
```

## 📈 Output and Visualizations

The system generates comprehensive visualizations including:

- **Episode Performance**: Rewards, success rates, detection rates over time
- **Attack Strategy Analysis**: Attack type distribution, impact analysis
- **Power System State**: Voltage profiles, frequency distribution, generator outputs
- **Threat Analysis Insights**: Vulnerability assessment, evasion techniques
- **Federated Metrics**: Model corruption, learning progress, aggregation quality
- **Comparison Analysis**: LLM vs rule-based approaches

## 🛡️ Security Features

### STRIDE Threat Categories
- **Spoofing**: Impersonating system components
- **Tampering**: Modifying system data or components
- **Repudiation**: Denying actions or events
- **Information Disclosure**: Exposing sensitive information
- **Denial of Service**: Disrupting system availability
- **Elevation of Privilege**: Gaining unauthorized access

### MITRE ATT&CK Techniques
- **T0832**: Manipulation of View
- **T0835**: Modify Parameter
- **T0815**: Denial of Service
- **T1573**: Data Manipulation
- **T0855**: Unauthorized Command Message
- **T0856**: Spoof Reporting Message

## 🔬 Research Applications

This system is designed for cybersecurity research and includes:

- **Vulnerability Assessment**: Identify critical weaknesses in EVCS systems
- **Attack Simulation**: Test defensive mechanisms against sophisticated attacks
- **Federated Learning Security**: Analyze vulnerabilities in distributed learning
- **Threat Intelligence**: Generate insights for improving security measures
- **Defensive Strategy Development**: Inform better protection mechanisms

## 📚 File Structure

```
├── llm_guided_evcs_attack_analytics.py    # LLM threat analyzer
├── evcs_rl_attack_agent.py                # RL attack agents
├── evcs_llm_rl_integration.py             # Basic integration system
├── federated_pinn_llm_rl_integration.py   # Federated PINN integration
├── demo_evcs_llm_rl_system.py             # Complete demo
└── README_EVCS_LLM_RL.md                  # This file
```

## 🚨 Important Notes

1. **Research Purpose**: This system is designed for cybersecurity research and defensive purposes only.

2. **Ollama Requirement**: The LLM functionality requires Ollama to be running with the deepseek-r1:8b model.

3. **Federated PINN**: Integration with your existing federated PINN system requires the appropriate model files.

4. **Performance**: Simulation performance depends on system resources and model complexity.

5. **Security**: Always use this system in controlled, isolated environments for research purposes.

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:

- Additional LLM models and providers
- More sophisticated RL algorithms
- Enhanced federated learning attack vectors
- Real-time attack detection and response
- Integration with more power system simulators

## 📄 License

This project is intended for research purposes. Please ensure compliance with all applicable laws and regulations when using this system.

## 🔗 Integration with Existing Systems

The system is designed to integrate with your existing:
- `focused_demand_analysis.py`
- `hierarchical_cosimulation.py`
- Federated PINN models
- EVCS charging management systems

Simply import the appropriate modules and configure the system according to your specific requirements.
