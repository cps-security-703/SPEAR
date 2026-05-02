
## 1. IDS (Intrusion Detection System) Design

The IDS is designed with both a primary multi-layer architecture and fallback anomaly scoring mechanisms depending on the active environment. 

### Multi-Layer IDS
The primary detector (`anomaly_detector.multi_layer_detection`) assesses 15 distinct physical and operational features in real-time, including:
- **Physical Features**: Voltage, current, power, SOC, temperature.
- **Operational Features**: Demand factor, load factor, queue length, utilization, urgency factor.
- **Grid Features**: Grid voltage, grid frequency, time of day.

It uses a 3-layer approach (Physical Limits -> Statistical Patterns -> LSTM sequence analysis) to produce an `ids_lstm_score`. If the score exceeds an `anomaly_threshold` (default is often 0.3 or 0.8 depending on the environment scale), the attack is flagged as detected.

### Fallback / Simple Anomaly Scoring
When the multi-layer IDS is unavailable or in simplified simulation steps, the system relies on fallback scoring methods:
1. **Parameter-based Score**:
   ```python
   base_score = (impact + magnitude) / 2.0
   stealth_reduction = stealth_factor * 0.3
   anomaly_score = max(0.0, base_score - stealth_reduction)
   ```
2. **Deviation-based Score**:
   Measures absolute deviations in `demand_factor`, `urgency_factor`, and `grid_voltage` from their baseline, capping the normalized sum at 1.0.

## 2. How the >97% Evasion Rate is Achieved

The extraordinary evasion rate is not a coincidence; the system explicitly trains Reinforcement Learning (RL) agents (a combination of DQN and SAC) to exploit the IDS constraints. The agents achieve >97% evasion through the following design mechanisms:

### A. Dedicated Security Evasion Environment
The RL agents are trained inside a specialized `SecurityEvasionEnvironment`. Their primary objective function is to learn the exact threshold boundaries of the IDS. 

### B. Exploitative Reward Function (`SecurityEvasionRewardCalculator`)
The reward calculator aggressively incentivizes stealth over raw destruction:
- **Heavy Detection Penalty**: `-100.0` points for being caught, terminating streaks.
- **Evasion Bonus**: `+50.0` points for successfully completing a step undetected.
- **Stealth Bonus**: `+25.0 * stealth_factor`. The agent receives extra points the further below the detection threshold the anomaly score remains.
- **Consecutive Evasion Bonus**: After 3 successful evasions, the agent gets a compounding bonus (`consecutive_evasions * 5.0`). Mission success is defined as 10 consecutive evasions with a minimum impact, rewarding `+500.0` points.

### C. Continuous Parameter Tuning (SAC Agent)
While the DQN agent selects the *type* and *timing* of the attack, the **SAC (Soft Actor-Critic)** agent has continuous control over the attack's:
1. `magnitude` (0.1 to 2.1)
2. `duration`
3. `stealth_factor` (0.0 to 1.0)

**The Exploit:** Because the fallback anomaly score formula mathematically subtracts `stealth_factor * 0.3` from the base detection score, the SAC agent learns to perfectly maximize the `stealth_factor` parameter while keeping the `magnitude` precisely below the trigger threshold. This algorithmic "tuning" ensures the resulting `anomaly_score` consistently stays below the IDS threshold.

### D. Multi-Agent Coordination
In the `MultiAgentRLEnvironment`, multiple agents coordinate attacks across different charging stations. They receive a **Coordination Bonus** (`len(successful_attacks) * 10.0`) for simultaneously attacking. This allows them to distribute the attack magnitude across multiple nodes (keeping individual station anomalies low) while achieving a high collective system impact.

### Summary
The >97% evasion rate is highly effective because the SAC/DQN agents learn to perfectly map the "safe zone" of the IDS thresholds. By continuously tuning the attack `magnitude` just below detection limits and artificially inflating the `stealth_factor` to suppress the anomaly score, the RL framework effectively blindfolds the IDS while still causing cumulative damage.
