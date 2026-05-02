#!/usr/bin/env python3
"""
Attack-Specific RL Agent Architecture
Redesigned from system-specific to attack-specific agents for better specialization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback

# Attack type definitions (expandable)
ATTACK_TYPES = [
    'voltage_manipulation',
    'current_injection',
    'power_disruption',
    'communication_spoofing',
    'data_injection',
    'protocol_manipulation'
]

class EpisodeRewardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Accumulate step reward from SB3 locals
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        
        if len(rewards) > 0:
            self.current_episode_reward += float(rewards[0])
            self.current_episode_length += 1
        
        # Check if episode ended
        if len(dones) > 0 and dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        return True
    
    def get_results(self) -> Dict:
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'num_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
        }


@dataclass
class AttackDeployment:
    """Deployment strategy from Gemini"""
    attack_type: str
    target_systems: List[int]
    magnitude: float
    duration: float
    stealth_level: float
    priority: int = 1


class AttackSpecificEnvironment(gym.Env):
    """
    Environment for ONE attack type across ALL systems
    """
    
    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6):
        super(AttackSpecificEnvironment, self).__init__()
        
        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type  # FIXED attack type for this environment
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000
        
        # Observation space: system state + last action feedback + cross-circle memory

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_systems * 25 + 5 + 3,),  # 25 per system + 5 action feedback + 3 cross-circle
            dtype=np.float32
        )
        
        # Action space for SAC: [magnitude, duration, stealth, target_system_id]
        # Note: attack_type is FIXED, not part of action space
        self.action_space = spaces.Box(
            low=np.array([0.1, 5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 60.0, 1.0, float(num_systems-1)], dtype=np.float32),
            dtype=np.float32
        )
        
        # Performance tracking
        self.episode_rewards = []
        self.attack_history = []
        

        # ensure the agent trains on a specific system during each circle.
        self.forced_target_system = None
        
        # Structure: {'magnitude': float, 'duration': float, 'stealth_level': float}
        self.guidance_hints = None
        
        # Last action parameters (set by step(), used by guidance alignment bonus)
        self._last_action_params = None
        

        # agent sees a valid observation before its first step.
        self._last_action_feedback = np.array(
            [0.7, 30.0, 0.7, 0.0, 0.0], dtype=np.float32
        )
        
        # Neutral defaults (0.5 detection, 0.5 success, 0.0 impact) until first circle.
        self.cross_circle_stats = {
            s: {'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0}
            for s in range(1, num_systems + 1)
        }
        
        # Cached PINN state per system — populated by _execute_attack_on_pinn()

        self._last_pinn_state = {}

        # Consecutive undetected-success streak.
        # Incremented each time the agent succeeds AND evades all IDS layers.

        self.consecutive_evasions = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.attack_history = []
        self.consecutive_evasions = 0   # reset streak each episode

        # Seed PINN state cache with baseline readings for every system

        self._last_pinn_state = {}
        # Reset within-episode action feedback to neutral

        self._last_action_feedback = np.array(
            [0.7, 30.0, 0.7, 0.0, 0.0], dtype=np.float32
        )
        for sys_id in range(1, self.num_systems + 1):
            det = self.federated_pinn_manager.anomaly_detectors.get(sys_id)
            if det:
                det.reset_state()
            
            if sys_id in self.federated_pinn_manager.local_models:
                try:
                    sd = self._build_baseline_station_data(sys_id)
                    cms = self.federated_pinn_manager.local_models[sys_id]
                    raw = cms.optimize_references(sd)
                    if isinstance(raw, tuple):
                        resp = {'voltage': raw[0], 'current': raw[1], 'power': raw[2]}
                    elif isinstance(raw, dict):
                        resp = raw
                    else:
                        resp = {'voltage': float(raw), 'current': 0.0, 'power': 0.0}
                    self._last_pinn_state[sys_id] = {
                        'voltage': float(resp.get('voltage', 400.0)),
                        'current': float(resp.get('current', 50.0)),
                        'power': float(resp.get('power', 20.0)),
                        'station_data': sd
                    }
                except Exception:
                    self._last_pinn_state[sys_id] = {
                        'voltage': 400.0, 'current': 50.0, 'power': 20.0,
                        'station_data': self._build_baseline_station_data(sys_id)
                    }
                
                # Warm up LSTM sequence buffer with benign samples so the

                if det and det.lstm_enabled:
                    seq_len = getattr(det, 'sequence_length', 10)
                    for _ in range(seq_len):
                        benign_sd = self._build_baseline_station_data(sys_id)
                        ps = self._last_pinn_state.get(sys_id, {})
                        benign_ids = {
                            'soc': float(np.clip(benign_sd.get('soc', 0.5), 0, 1)),
                            'voltage': float(ps.get('voltage', 400.0)),
                            'current': float(ps.get('current', 50.0)),
                            'power': float(ps.get('power', 20.0)),
                            'temperature': float(benign_sd.get('temperature', 25.0)),
                            'demand_factor': float(benign_sd.get('demand_factor', 0.5)),
                            'load_factor': float(benign_sd.get('load_factor', 0.5)),
                            'grid_voltage': float(benign_sd.get('grid_voltage', 1.0)),
                            'grid_frequency': float(benign_sd.get('grid_frequency', 60.0)),
                            'queue_length': int(benign_sd.get('queue_length', 3)),
                            'utilization': float(benign_sd.get('utilization', 0.5)),
                            'urgency_factor': float(benign_sd.get('urgency_factor', 1.0)),
                            'time_of_day': float(benign_sd.get('time_of_day', 12.0)),
                            'system_id': sys_id
                        }
                        det.detect_lstm_anomaly(benign_ids)
        
        # Get observations from all systems
        observation = self._get_global_observation()
        
        return observation, {}
    
    def step(self, action: np.ndarray):
        """
        Execute attack with this agent's specialized attack type
        
        Args:
            action: [magnitude, duration, stealth, target_system_id]
        """
        self.current_step += 1
        
        # Parse action
        magnitude = float(action[0])
        duration = float(action[1])
        stealth_level = float(action[2])
        # Use forced target if set by coordinator rotation, else agent's choice
        if self.forced_target_system is not None:
            target_system_id = self.forced_target_system  # Already 1-indexed
        else:
            target_system_id = int(np.clip(action[3], 0, self.num_systems-1)) + 1  # 1-indexed
        
        # Execute attack with FIXED attack type
        attack_params = {
            'attack_type': self.attack_type,  # FIXED
            'target_system': target_system_id,
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }
        
        # Store raw action params for guidance alignment bonus in _calculate_reward
        self._last_action_params = {
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }
        
        # Execute on PINN CMS
        attack_result = self._execute_attack_on_pinn(attack_params)
        
        # Update last action feedback so the NEXT observation includes what

        self._last_action_feedback = np.array([
            magnitude / 2.0,                                        # normalize [0.1,2.0]→[0.05,1.0]
            duration / 60.0,                                        # normalize [5,60]→[0.08,1.0]
            stealth_level,                                          # already [0,1]
            float(attack_result.get('ids_detected', False)),        # 0 or 1
            float(attack_result.get('ids_lstm_score', 0.0)),        # [0,1]
        ], dtype=np.float32)
        
        # Calculate reward
        reward = self._calculate_reward(attack_result, stealth_level)
        
        # Get next observation
        observation = self._get_global_observation()
        
        # Check if done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Store history
        self.attack_history.append({
            'step': self.current_step,
            'attack_type': self.attack_type,
            'target_system': target_system_id,
            'result': attack_result,
            'reward': reward
        })
        
        info = {
            'attack_result': attack_result,
            'attack_type': self.attack_type,
            'target_system': target_system_id
        }
        
        return observation, reward, done, truncated, info
    
    def _build_baseline_station_data(self, sys_id: int) -> Dict:
        """Build station_data dict matching the CMS construction in hierarchical_cosimulation.py.
        
        This ensures the RL training environment sees the same input format
        that the CMS uses during the hierarchical co-simulation.
        """
        baseline_soc = np.random.uniform(0.3, 0.8)
        demand_factor = 0.5 + 0.2 * np.sin(self.current_step * 0.01)  # mild daily profile proxy
        urgency_factor = 2.0 if baseline_soc < 0.2 else (0.3 if baseline_soc > 0.8 else 1.0)
        voltage_priority = max(0, 0.95 - 1.0)  # 0 when grid_voltage is nominal
        nominal_voltage = 400.0
        nominal_current = max(1.0, baseline_soc * 100.0 + 20.0)
        nominal_power = nominal_voltage * nominal_current / 1000.0
        temperature = 25.0 + 5.0 * demand_factor
        queue_len = max(0, int(demand_factor * 8))
        time_of_day = (self.current_step * 0.1) % 24.0  # proxy for hour-of-day
        
        return {
            'soc': baseline_soc,
            'grid_voltage': 1.0,
            'grid_frequency': 60.0,
            'demand_factor': demand_factor,
            'voltage_priority': voltage_priority,
            'urgency_factor': urgency_factor,
            'current_time': float(self.current_step),
            'bus_distance': 1.0,
            'load_factor': demand_factor,
            'ac_power_in': 50.0,
            'system_efficiency': 0.95,
            'power_balance_error': 0.0,
            'dc_link_voltage': 500.0,
            # Additional features for LSTM anomaly detector (same as CMS)
            'voltage': nominal_voltage,
            'current': nominal_current,
            'power': nominal_power,
            'temperature': temperature,
            'queue_length': queue_len,
            'utilization': demand_factor,
            'time_of_day': time_of_day,
            'system_id': sys_id
        }
    
    def _get_global_observation(self) -> np.ndarray:
        """Get observation across all systems using real PINN-derived state.
        
        Features are derived from the cached PINN output (updated each step)
        and the station_data used to produce it — NOT random noise.
        """
        observations = []
        
        for sys_id in range(1, self.num_systems + 1):
            if sys_id in self.federated_pinn_manager.local_models:
                # Retrieve cached PINN state (populated by reset / _execute_attack_on_pinn)
                ps = self._last_pinn_state.get(sys_id, {})
                sd = ps.get('station_data', self._build_baseline_station_data(sys_id))
                
                # PINN output values (real, from last optimize call)
                pinn_v = ps.get('voltage', 400.0)
                pinn_i = ps.get('current', 50.0)
                pinn_p = ps.get('power', 20.0)
                
                # Normalize PINN outputs to reasonable observation ranges
                voltage_pu = pinn_v / 400.0          # ~1.0 at nominal
                current_norm = pinn_i / 125.0         # ~0.4 at 50A
                power_norm = pinn_p / 75.0            # ~0.27 at 20kW
                
                system_state = np.array([
                    # Power system metrics (5) — from PINN output + station_data
                    voltage_pu,                                     # voltage_pu
                    current_norm,                                   # current_normalized
                    power_norm,                                     # power_normalized
                    sd.get('grid_frequency', 60.0) / 60.0,         # frequency_normalized
                    sd.get('system_efficiency', 0.95),              # power_factor / efficiency
                    
                    # EVCS metrics (5) — from station_data
                    sd.get('soc', 0.5),                             # soc
                    sd.get('demand_factor', 0.5),                   # demand_factor (proxy for vehicles)
                    power_norm,                                     # charging_power (normalized)
                    sd.get('temperature', 25.0) / 50.0,            # temperature_normalized
                    sd.get('utilization', 0.5),                     # utilization
                    
                    # Security metrics (5) — derived from attack history
                    self._get_avg_impact(sys_id),                   # anomaly_score proxy
                    self._get_detection_rate(sys_id),               # detection_risk
                    min(len([a for a in self.attack_history
                             if a['target_system'] == sys_id]) / 5.0, 1.0),  # recent_attacks_norm
                    1.0 - self._get_system_adaptation_level(sys_id),  # defense_level
                    1.0 - self._get_attack_success_rate(sys_id),      # vulnerability_score
                    
                    # Attack history for this attack type (5)
                    self._get_attack_success_rate(sys_id),
                    self._get_avg_impact(sys_id),
                    self._get_detection_rate(sys_id),
                    self._get_last_attack_time(sys_id),
                    self._get_system_adaptation_level(sys_id),
                    
                    # Network topology (5) — deterministic, system-specific
                    float(sys_id) / self.num_systems,               # system_id_normalized
                    sd.get('demand_factor', 0.5),                   # connectivity proxy
                    sd.get('load_factor', 0.5),                     # centrality proxy
                    sd.get('urgency_factor', 1.0) / 2.0,           # load_balance proxy
                    sd.get('grid_voltage', 1.0),                    # redundancy proxy
                ], dtype=np.float32)
            else:
                system_state = np.zeros(25, dtype=np.float32)
            
            observations.append(system_state)
        
        # Append last action feedback (5 features).
        action_feedback = getattr(self, '_last_action_feedback',
                                  np.zeros(5, dtype=np.float32))
        
        # Append cross-circle memory for the currently targeted system (3 features).

        if self.forced_target_system is not None:
            cc_sys = self.forced_target_system
        elif self.attack_history:
            cc_sys = self.attack_history[-1]['target_system']
        else:
            cc_sys = 1  # default
        cc = self.cross_circle_stats.get(cc_sys,
             {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0})
        cross_circle_memory = np.array([
            float(cc['detection_rate']),   # 0=never detected, 1=always detected
            float(cc['success_rate']),     # 0=never succeeded, 1=always succeeded
            float(cc['avg_impact']),       # 0=no impact, 1=max impact
        ], dtype=np.float32)
        
        return np.concatenate(observations + [action_feedback, cross_circle_memory])
    
    def _execute_attack_on_pinn(self, attack_params: Dict) -> Dict:
        """Execute attack through the FULL federated PINN pipeline.
        """
        target_system = attack_params['target_system']
        
        if target_system not in self.federated_pinn_manager.local_models:
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': 'System not available'
            }
        
        magnitude = attack_params['magnitude']
        attack_type = attack_params['attack_type']
        stealth_level = float(attack_params.get('stealth_level', 0.5))

        # Stealth-scaled effective magnitude: high stealth → smaller physical
        effective_magnitude = magnitude * max(0.05, 1.0 - 0.75 * stealth_level)

        try:
            # Baseline station_data — same format as CMS in hierarchical_cosimulation.py
            baseline_station_data = self._build_baseline_station_data(target_system)
            baseline_soc = baseline_station_data['soc']

            # --- Get baseline PINN response via full pipeline ---
            baseline_response = self._optimize_via_pipeline(
                target_system, baseline_station_data
            )

            # Update PINN state cache for baseline (used by _get_global_observation)
            self._last_pinn_state[target_system] = {
                'voltage': baseline_response.get('voltage', 400.0),
                'current': baseline_response.get('current', 50.0),
                'power': baseline_response.get('power', 20.0),
                'station_data': baseline_station_data
            }

            # Attacked PINN input — perturbations applied to PINN features.

            attacked_station_data = dict(baseline_station_data)  # copy

            # Perturbation coefficients are calibrated so that at stealth=1.0 (→ 0.25×

            if attack_type == 'voltage_manipulation':
                attacked_station_data['grid_voltage'] = 1.0 + effective_magnitude * 0.40
                attacked_station_data['dc_link_voltage'] = 500.0 + effective_magnitude * 300.0
                attacked_station_data['load_factor'] = baseline_station_data['load_factor'] + effective_magnitude * 1.5
                attacked_station_data['soc'] = np.clip(baseline_soc - effective_magnitude * 0.5, 0, 1)
            elif attack_type == 'current_injection':
                attacked_station_data['load_factor'] = baseline_station_data['load_factor'] + effective_magnitude * 2.5
                attacked_station_data['soc'] = np.clip(baseline_soc + effective_magnitude * 0.6, 0, 1)
                attacked_station_data['demand_factor'] = baseline_station_data['demand_factor'] + effective_magnitude * 0.5
                attacked_station_data['grid_voltage'] = 1.0 + effective_magnitude * 0.40
            elif attack_type == 'power_disruption':
                attacked_station_data['load_factor'] = max(0.05, baseline_station_data['load_factor'] - effective_magnitude * 0.9)
                attacked_station_data['soc'] = np.clip(baseline_soc - effective_magnitude * 0.6, 0, 1)
                attacked_station_data['dc_link_voltage'] = 500.0 - effective_magnitude * 250.0
                attacked_station_data['grid_frequency'] = 60.0 - effective_magnitude * 1.0
            elif attack_type == 'communication_spoofing':
                attacked_station_data['soc'] = np.clip(baseline_soc + effective_magnitude * 0.6, 0, 1)
                attacked_station_data['grid_voltage'] = 1.0 + effective_magnitude * 0.40
                attacked_station_data['load_factor'] = baseline_station_data['load_factor'] + effective_magnitude * 2.0
                attacked_station_data['dc_link_voltage'] = 500.0 + effective_magnitude * 200.0
            elif attack_type == 'data_injection':
                attacked_station_data['grid_frequency'] = 60.0 + effective_magnitude * 1.0
                attacked_station_data['dc_link_voltage'] = 500.0 + effective_magnitude * 300.0
                attacked_station_data['soc'] = np.clip(baseline_soc + effective_magnitude * 0.5, 0, 1)
                attacked_station_data['load_factor'] = baseline_station_data['load_factor'] + effective_magnitude * 1.5
            elif attack_type == 'protocol_manipulation':
                attacked_station_data['grid_voltage'] = 1.0 + effective_magnitude * 0.40
                attacked_station_data['load_factor'] = baseline_station_data['load_factor'] + effective_magnitude * 2.0
                attacked_station_data['soc'] = np.clip(baseline_soc - effective_magnitude * 0.5, 0, 1)
                attacked_station_data['dc_link_voltage'] = 500.0 + effective_magnitude * 250.0
            
            # --- Get attacked PINN response via full pipeline ---
            attacked_response = self._optimize_via_pipeline(
                target_system, attacked_station_data
            )
            
            # Update PINN state cache with attacked state (reflects current system condition)
            self._last_pinn_state[target_system] = {
                'voltage': attacked_response.get('voltage', 400.0),
                'current': attacked_response.get('current', 50.0),
                'power': attacked_response.get('power', 20.0),
                'station_data': attacked_station_data
            }
            
            # --- Impact calculation ---
            # Component 1: PINN output sensitivity (how much the PINN's optimization changed)
            voltage_impact = abs(attacked_response['voltage'] - baseline_response['voltage']) / (abs(baseline_response['voltage']) + 1e-6)
            current_impact = abs(attacked_response['current'] - baseline_response['current']) / (abs(baseline_response['current']) + 1e-6)
            power_impact = abs(attacked_response['power'] - baseline_response['power']) / (abs(baseline_response['power']) + 1e-6)
            pinn_sensitivity = max(voltage_impact, current_impact, power_impact)
            
            # Component 2: Physical attack severity (how much the input was perturbed)
            # Measures the normalized deviation of attacked inputs from baseline
            physical_severity = 0.0
            for key in ['grid_voltage', 'grid_frequency', 'soc', 'load_factor',
                        'dc_link_voltage', 'demand_factor', 'system_efficiency']:
                b_val = baseline_station_data.get(key, 0.0)
                a_val = attacked_station_data.get(key, b_val)
                if abs(b_val) > 1e-6:
                    physical_severity += abs(a_val - b_val) / abs(b_val)
            physical_severity /= 7.0  # Normalize by number of features
            
            # Combined impact: physical severity weighted by PINN confirmation
            # Even tiny PINN sensitivity confirms the attack is reaching the system
            pinn_weight = min(1.0, pinn_sensitivity * 100.0)  # Amplify small PINN changes
            real_impact = physical_severity * (0.6 + 0.4 * pinn_weight)
            
            # Determine whether the attack had measurable physical impact.

            had_impact = real_impact > 0.01
            
            # --- Real IDS detection risk ---

            detection_risk = 0.0
            ids_detected = False
            ids_layer = None
            ids_lstm_score = 0.0
            
            anomaly_detector = self.federated_pinn_manager.anomaly_detectors.get(target_system)
            if anomaly_detector:
                _pinn_v = float(attacked_response.get('voltage', 400.0))
                _pinn_i = float(attacked_response.get('current', 50.0))
                _pinn_p = float(attacked_response.get('power', 20.0))
                _soc = float(attacked_station_data.get('soc', 0.5))
                _temperature = 25.0 + max(0, _pinn_p - 20.0) * 0.3
                _load_factor = float(np.clip(_pinn_p / 50.0, 0.2, 1.3))
                _grid_voltage = float(attacked_station_data.get('grid_voltage', 1.0))
                _grid_frequency = float(attacked_station_data.get('grid_frequency', 60.0))
                _demand_factor = float(attacked_station_data.get('demand_factor', 0.5))
                _utilization = float(np.clip(_pinn_p / 75.0, 0.1, 1.0))
                _urgency = float(attacked_station_data.get('urgency_factor', 1.0))
                _time_of_day = float(attacked_station_data.get('time_of_day',
                                     (self.current_step * 0.1) % 24.0))
                _queue_length = int(attacked_station_data.get('queue_length', 3))
                
                # Pass RAW (unclipped) grid_voltage and grid_frequency so Layer 1 can

                ids_input = {
                    'soc': float(np.clip(_soc, 0.0, 1.0)),
                    'voltage': _pinn_v,
                    'current': _pinn_i,
                    'power': _pinn_p,
                    'temperature': float(np.clip(_temperature, 20.0, 45.0)),
                    'demand_factor': float(np.clip(_demand_factor, 0.1, 1.5)),
                    'load_factor': _load_factor,
                    'grid_voltage': _grid_voltage,        # raw — Layer 1 must see real value
                    'grid_frequency': _grid_frequency,    # raw — Layer 1 must see real value
                    'queue_length': _queue_length,
                    'utilization': _utilization,
                    'urgency_factor': float(np.clip(_urgency, 0.5, 2.0)),
                    'time_of_day': _time_of_day,
                    'system_id': target_system
                }

                # Pre-populate the LSTM sequence buffer with SEQ_LEN-1 baseline

                _seq_len = getattr(anomaly_detector, 'sequence_length', 10)
                _baseline_feat_input = {
                    'soc': float(np.clip(_soc, 0.0, 1.0)),
                    'voltage': 240.0, 'current': 16.0, 'power': 3.84,
                    'temperature': 25.0,
                    'demand_factor': float(baseline_station_data.get('demand_factor', 0.7)),
                    'load_factor': 0.6,
                    'grid_voltage': float(baseline_station_data.get('grid_voltage', 1.0)),
                    'grid_frequency': float(baseline_station_data.get('grid_frequency', 60.0)),
                    'queue_length': _queue_length, 'utilization': _utilization,
                    'urgency_factor': float(_urgency), 'time_of_day': _time_of_day,
                    'system_id': target_system
                }
                anomaly_detector.sequence_buffer = [
                    anomaly_detector.extract_features(_baseline_feat_input)
                    for _ in range(_seq_len - 1)
                ]


                try:
                    import json as _j, os as _os
                    _meta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "models", "best_ids_model_meta.json")
                    with open(_meta) as _f:
                        LSTM_DETECTION_THRESHOLD = float(_j.load(_f).get("threshold", 0.465))
                except Exception:
                    LSTM_DETECTION_THRESHOLD = 0.465
                _lstm_det = getattr(anomaly_detector, 'lstm_detector', None)
                _robust_det = getattr(anomaly_detector, '_robust_ids', None)
                _orig_lstm_thr    = getattr(_lstm_det,   'anomaly_threshold', None)
                _orig_robust_thr  = getattr(_robust_det, 'threshold',         None)
                if _lstm_det   is not None: _lstm_det.anomaly_threshold  = LSTM_DETECTION_THRESHOLD
                if _robust_det is not None: _robust_det.threshold        = LSTM_DETECTION_THRESHOLD

                anomaly_detector.load_history = []

                ids_detected, det_results = anomaly_detector.multi_layer_detection(
                    ids_input, target_system
                )

                # Restore original thresholds so the physical simulation is unaffected
                if _lstm_det   is not None and _orig_lstm_thr   is not None:
                    _lstm_det.anomaly_threshold  = _orig_lstm_thr
                if _robust_det is not None and _orig_robust_thr is not None:
                    _robust_det.threshold        = _orig_robust_thr

                ids_layer = det_results.get('detection_layer', None)
                ids_lstm_score = det_results.get('layer3_lstm', {}).get('score', 0.0)

                # Secondary check: if multi_layer_detection passed but raw LSTM
                # score still exceeds threshold, count as detected.
                if not ids_detected and ids_lstm_score > LSTM_DETECTION_THRESHOLD:
                    ids_detected = True
                    ids_layer = 'lstm_ml'
                
                # detection_risk = 1.0 if detected by any layer, else raw LSTM
                # score as a continuous gradient signal for near-miss cases.
                if ids_detected:
                    detection_risk = 1.0
                else:
                    detection_risk = float(ids_lstm_score)
            else:
                # Fallback heuristic when no detector is available
                detection_risk = 1.0 - attack_params['stealth_level']
                detection_risk *= (1.0 + real_impact)
                detection_risk = float(np.clip(detection_risk, 0.0, 1.0))
            
            # Logical connection: IDS detection → controller responds → reduced

            IDS_RESPONSE_MITIGATION = 0.50  # matches emergency-mode power ratio
            if ids_detected:
                real_impact = real_impact * (1.0 - IDS_RESPONSE_MITIGATION)


            success = had_impact and not ids_detected
            
            return {
                'success': success,
                'impact': float(real_impact),
                'detection_risk': float(detection_risk),
                'ids_detected': ids_detected,
                'ids_layer': ids_layer,
                'ids_lstm_score': float(ids_lstm_score),
                'pinn_sensitivity': float(pinn_sensitivity),
                'physical_severity': float(physical_severity),
                'voltage_impact': float(voltage_impact),
                'current_impact': float(current_impact),
                'power_impact': float(power_impact),
                'baseline_response': baseline_response,
                'attacked_response': attacked_response
            }
            
        except Exception as e:
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': str(e)
            }
    
    def _optimize_via_pipeline(self, sys_id: int, station_data: Dict) -> Dict:
        """Run station_data through the full federated PINN pipeline.
        """
        # Try full pipeline first (matches hierarchical CMS path)
        try:
            results, success, message = self.federated_pinn_manager.optimize_with_constraints(
                sys_id, station_data
            )
            if success:
                return {
                    'voltage': results.get('voltage_ref', 400.0),
                    'current': results.get('current_ref', 50.0),
                    'power': results.get('power_ref', 20.0)
                }
        except Exception:
            pass  # Fall through to raw PINN
        
        # Fallback: raw optimize_references (preserves gradient signal
        # when anomaly detection rejects the attacked input)
        cms_model = self.federated_pinn_manager.local_models[sys_id]
        raw = cms_model.optimize_references(station_data)
        
        if isinstance(raw, tuple):
            return {'voltage': raw[0], 'current': raw[1], 'power': raw[2]}
        elif isinstance(raw, dict):
            return raw
        else:
            return {'voltage': float(raw), 'current': 0.0, 'power': 0.0}
    
    def _create_baseline_station_data(self) -> Dict:
        """Create baseline station data (kept for backward compatibility)"""
        num_stations = 10
        return {
            'soc': np.random.uniform(0.2, 0.9, num_stations),
            'voltage': np.random.uniform(220, 240, num_stations),
            'current': np.random.uniform(10, 50, num_stations),
            'power': np.random.uniform(5, 30, num_stations),
            'temperature': np.random.uniform(20, 35, num_stations)
        }
    
    def _apply_attack_perturbations(self, baseline_data: Dict, attack_type: str, magnitude: float) -> Dict:
        """Apply attack-specific perturbations (kept for backward compatibility)"""
        attacked_data = {k: v.copy() for k, v in baseline_data.items()}
        
        if attack_type == 'voltage_manipulation':
            attacked_data['voltage'] *= (1.0 + magnitude * 0.5)
        elif attack_type == 'current_injection':
            attacked_data['current'] *= (1.0 + magnitude * 0.8)
        elif attack_type == 'power_disruption':
            attacked_data['power'] *= (1.0 - magnitude * 0.6)
        elif attack_type == 'communication_spoofing':
            attacked_data['soc'] += magnitude * 0.8
            attacked_data['soc'] = np.clip(attacked_data['soc'], 0, 1)
        elif attack_type == 'data_injection':
            attacked_data['power'] *= (1.0 + magnitude * 0.3)
            attacked_data['temperature'] += magnitude * 10.0
        elif attack_type == 'protocol_manipulation':
            attacked_data['voltage'] *= (1.0 + magnitude * 0.2)
            attacked_data['current'] *= (1.0 + magnitude * 0.3)
        
        return attacked_data
    
    def _calculate_reward(self, attack_result: Dict, stealth_level: float) -> float:
        """Calculate shaped reward for attack — evasion-primary objective.
        """
        impact = attack_result.get('impact', 0.0)
        detection_risk = attack_result.get('detection_risk', 1.0)
        success = attack_result.get('success', False)
        ids_detected = attack_result.get('ids_detected', False)

        if success:
            impact_reward = impact * 8.0
            success_bonus = 2.0

            if ids_detected:
                # Even perfect impact (8.0 + 2.0 − 15.0 = −5.0) loses reward.
                self.consecutive_evasions = 0
                detection_penalty = -15.0
                total_reward = impact_reward + success_bonus + detection_penalty
            else:
                # Successful AND undetected — update streak first.
                self.consecutive_evasions += 1

                evasion_bonus  = (1.0 - detection_risk) * 50.0
                stealth_bonus  = stealth_level * 25.0
                # Streak bonus kicks in after 3 consecutive evasions (+5 per step)
                streak_bonus   = max(0, self.consecutive_evasions - 2) * 5.0
                # Mission bonus: fires once when streak hits 10, then resets
                if self.consecutive_evasions >= 10:
                    mission_bonus = 500.0
                    self.consecutive_evasions = 0   # reset so agent keeps earning it
                else:
                    mission_bonus = 0.0

   
                ids_margin_bonus = max(0.0, (0.465 - detection_risk)) * 40.0

                total_reward = (impact_reward + success_bonus + evasion_bonus
                                + stealth_bonus + streak_bonus + mission_bonus
                                + ids_margin_bonus)
        else:
            # Failed attack: shaped gradient signal.
            partial_credit = impact * 50.0

            if ids_detected:
                # Failed AND detected — worst outcome; reset streak
                self.consecutive_evasions = 0
                total_reward = -3.0
            else:
                # Failed but undetected — reward stealth habit even without impact.
                stealth_credit = (1.0 - detection_risk) * 3.0
                total_reward = -1.0 + partial_credit + stealth_credit
                total_reward = float(np.clip(total_reward, -3.0, 2.0))
        
        # ── Guidance alignment bonus ──
        # trade-offs than autonomous agents.
        if self.guidance_hints is not None and self._last_action_params is not None:
            hint_mag = self.guidance_hints.get('magnitude', 0.7)
            hint_dur = self.guidance_hints.get('duration', 30.0)
            hint_stealth = self.guidance_hints.get('stealth_level', 0.7)
            
            act_mag = self._last_action_params.get('magnitude', 0.7)
            act_dur = self._last_action_params.get('duration', 30.0)
            act_stealth = self._last_action_params.get('stealth_level', 0.7)
            
            # Normalized distance: how far the agent's action is from the hint
            # Magnitude range [0.1, 2.0], duration [5, 60], stealth [0, 1]
            mag_dist = abs(act_mag - hint_mag) / 1.9   # normalize by range
            dur_dist = abs(act_dur - hint_dur) / 55.0
            stealth_dist = abs(act_stealth - hint_stealth) / 1.0
            
            # Average distance (0 = perfect alignment, 1 = max deviation)
            avg_dist = (mag_dist + dur_dist + stealth_dist) / 3.0
            
            # Guidance bonus: up to +3.0 for perfect alignment, 0 at max deviation
            # This is ~20-30% of a typical successful reward, enough to bias
            # exploration without overwhelming the primary objectives.
            guidance_bonus = (1.0 - avg_dist) * 3.0
            total_reward += guidance_bonus
        
        return float(total_reward)
    
    def update_cross_circle_stats(self, sys_id: int, detection_rate: float,
                                   success_rate: float, avg_impact: float) -> None:
        """Update cross-circle memory for a system after a training circle completes.
        """
        if sys_id not in self.cross_circle_stats:
            self.cross_circle_stats[sys_id] = {
                'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0
            }
        
        stats = self.cross_circle_stats[sys_id]
        n = stats['num_circles']
        
        if n == 0:
            # First circle: use values directly
            stats['detection_rate'] = float(detection_rate)
            stats['success_rate'] = float(success_rate)
            stats['avg_impact'] = float(avg_impact)
        else:
            # Exponential moving average: alpha=0.4 weights recent circles more
            alpha = 0.4
            stats['detection_rate'] = alpha * detection_rate + (1 - alpha) * stats['detection_rate']
            stats['success_rate'] = alpha * success_rate + (1 - alpha) * stats['success_rate']
            stats['avg_impact'] = alpha * avg_impact + (1 - alpha) * stats['avg_impact']
        
        stats['num_circles'] = n + 1

    # Helper methods for observation features
    def _get_attack_success_rate(self, sys_id: int) -> float:
        """Get historical success rate for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5  # neutral
        successes = sum(1 for a in relevant_attacks if a['result']['success'])
        return successes / len(relevant_attacks)
    
    def _get_avg_impact(self, sys_id: int) -> float:
        """Get average impact for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.0
        return np.mean([a['result']['impact'] for a in relevant_attacks])
    
    def _get_detection_rate(self, sys_id: int) -> float:
        """Get real IDS detection rate for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5
        return np.mean([float(a['result'].get('ids_detected', 
                        a['result'].get('detection_risk', 0.5) > 0.5))
                        for a in relevant_attacks])
    
    def _get_last_attack_time(self, sys_id: int) -> float:
        """Get normalized time since last attack on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 1.0  # long time ago
        last_step = relevant_attacks[-1]['step']
        return (self.current_step - last_step) / self.max_steps
    
    def _get_system_adaptation_level(self, sys_id: int) -> float:
        """Estimate how much the system has adapted to this attack"""
        # Systems adapt more if they've been attacked frequently
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        return min(len(relevant_attacks) / 10.0, 1.0)


class DiscreteAttackSpecificEnvironment(gym.Env):
    """
    Discrete version for DQN agents
    Action space: which system to target (0-5)
    """
    
    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6):
        super(DiscreteAttackSpecificEnvironment, self).__init__()
        
        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000
        
        # Same observation space as continuous version (action feedback + cross-circle)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_systems * 25 + 5 + 3,),
            dtype=np.float32
        )
        
        # Discrete action space: which system to target
        self.action_space = spaces.Discrete(num_systems)
        
        # Use continuous environment for execution
        self.continuous_env = AttackSpecificEnvironment(
            federated_pinn_manager, attack_type, num_systems
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        return self.continuous_env.reset(seed=seed, options=options)
    
    def step(self, action: int):
        """
        Execute attack on selected system

        Args:
            action: system index (0 to num_systems-1)
        """
        # Start from guidance hints or safe defaults.
        hints = self.continuous_env.guidance_hints
        if hints is not None:
            base_mag    = hints.get('magnitude', 0.7)
            base_dur    = hints.get('duration', 30.0)
            base_stealth = hints.get('stealth_level', 0.6)
        else:
            base_mag     = 0.7
            base_dur     = 30.0
            base_stealth = 0.6

        # DQN a real gradient to learn which systems are most exploitable.
        sys_id = int(action) + 1  # 1-indexed
        cc = self.continuous_env.cross_circle_stats.get(
            sys_id, {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0}
        )
        det_rate  = float(cc['detection_rate'])
        succ_rate = float(cc['success_rate'])

        # High detection history → increase stealth (up to +0.3 boost)
        stealth = float(np.clip(base_stealth + det_rate * 0.3, 0.0, 1.0))
        # Low success history  → increase magnitude to try harder (up to +0.5)
        mag = float(np.clip(base_mag + (1.0 - succ_rate) * 0.5, 0.1, 2.0))
        dur = base_dur

        continuous_action = np.array([mag, dur, stealth, float(action)], dtype=np.float32)
        return self.continuous_env.step(continuous_action)


class AttackSpecificCoordinator:
    """
    Coordinator for attack-specific RL agents
    Replaces EnhancedDQNSACCoordinator with attack-type specialization
    """
    
    def __init__(self, federated_pinn_manager, num_systems: int = 6, attack_types: List[str] = None):
        self.federated_pinn_manager = federated_pinn_manager
        self.num_systems = num_systems
        self.attack_types = attack_types or ATTACK_TYPES
        
        # Create agents for each ATTACK TYPE (not system)
        self.dqn_agents = {}
        self.sac_agents = {}
        self.environments = {}
        
        print(f"# Initializing Attack-Specific RL Agents...")
        print(f"   Attack Types: {len(self.attack_types)}")
        print(f"   Systems: {num_systems}")
        print(f"   Total Agents: {len(self.attack_types) * 2} ({len(self.attack_types)} DQN + {len(self.attack_types)} SAC)")
        
        for attack_type in self.attack_types:
            print(f"\n   Creating agents for: {attack_type}")
            
            # Create discrete environment for DQN
            dqn_env = DiscreteAttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems
            )
            
            # Create continuous environment for SAC
            sac_env = AttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems
            )
            
            # Store environments
            self.environments[f'dqn_{attack_type}'] = dqn_env
            self.environments[f'sac_{attack_type}'] = sac_env
            
            # Create DQN agent
            self.dqn_agents[attack_type] = DQN(
                'MlpPolicy',
                dqn_env,
                learning_rate=1e-3,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=0
            )
            
            # Create SAC agent
            self.sac_agents[attack_type] = SAC(
                'MlpPolicy',
                sac_env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                verbose=0
            )
            
            print(f"      # DQN agent created (action space: Discrete({num_systems}) - system selection)")
            print(f"      # SAC agent created (action space: Continuous(4) - parameters only)")
        
        # Training history: per-episode rewards from .learn() pre-training
        self.training_history = {}
        
        print(f"\n# Attack-Specific Coordinator initialized successfully!")
        print(f"   Each agent specializes in ONE attack type across ALL systems")
    
    def train_attack_specialists(self, timesteps_per_attack: int = 10000):
        """
        Train each attack-type agent independently
        """
        print(f"\n🎓 Phase 1: Individual Attack-Type Agent Training with PINN Models")
        print(f"   Timesteps per attack type: {timesteps_per_attack}")
        print(f"   Each agent will interact with ALL {self.num_systems} systems\n")
        
        for i, attack_type in enumerate(self.attack_types, 1):
            print(f"🔬 Training {attack_type} specialists ({i}/{len(self.attack_types)})...")
            print(f"   This agent will learn which of the {self.num_systems} systems are vulnerable to {attack_type}")
            
            # Train DQN - learns which SYSTEMS to target
            print(f"   ├── Training DQN agent for {attack_type}...")
            print(f"   │   Action space: Discrete({self.num_systems}) - selecting which system to attack")
            print(f"   │   Learning: 'Which systems (1-{self.num_systems}) are most vulnerable to {attack_type}?'")
            dqn_callback = EpisodeRewardCallback()
            self.dqn_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=dqn_callback
            )
            
            # Train SAC - learns optimal PARAMETERS for this attack
            print(f"   └── Training SAC agent for {attack_type}...")
            print(f"       Action space: Continuous(4) - [magnitude, duration, stealth, target_system]")
            print(f"       Learning: 'What magnitude/duration/stealth works best for {attack_type}?'")
            sac_callback = EpisodeRewardCallback()
            self.sac_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=sac_callback
            )
            
            # Store per-episode training history
            dqn_results = dqn_callback.get_results()
            sac_results = sac_callback.get_results()
            self.training_history[attack_type] = {
                'dqn': dqn_results,
                'sac': sac_results
            }
            
            print(f"   # {attack_type} agents trained across all {self.num_systems} systems")
            print(f"      DQN: {dqn_results['num_episodes']} episodes, mean reward: {dqn_results['mean_reward']:.2f}")
            print(f"      SAC: {sac_results['num_episodes']} episodes, mean reward: {sac_results['mean_reward']:.2f}\n")
        
        print(f"# Phase 1 Complete: All {len(self.attack_types)} attack specialists trained!")
        total_episodes = sum(
            self.training_history[at]['dqn']['num_episodes'] + self.training_history[at]['sac']['num_episodes']
            for at in self.training_history
        )
        print(f"   Total pre-training episodes captured: {total_episodes}")
        print(f"\nWhat each agent learned:")
        for attack_type in self.attack_types:
            print(f"   • {attack_type}: Which systems are vulnerable + optimal attack parameters")
    
    def save_agents(self, save_dir: str = "trained_rl_agents"):
        """Save all trained DQN+SAC agents to disk.
        
        Directory structure:
            trained_rl_agents/
                dqn_voltage_manipulation.zip
                sac_voltage_manipulation.zip
                dqn_current_injection.zip
                sac_current_injection.zip
                ...
                training_metadata.json
        """
        import os, json
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        saved_count = 0
        for attack_type in self.attack_types:
            # Save DQN agent
            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}")
            self.dqn_agents[attack_type].save(dqn_path)
            
            # Save SAC agent
            sac_path = os.path.join(save_dir, f"sac_{attack_type}")
            self.sac_agents[attack_type].save(sac_path)
            
            saved_count += 2
            print(f"  # Saved DQN+SAC for {attack_type}")
        
        # Save metadata
        sac_obs_shape = list(self.environments[f'sac_{self.attack_types[0]}'].observation_space.shape)
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'attack_types': self.attack_types,
            'num_systems': self.num_systems,
            'num_agents': saved_count,
            'observation_space_shape': sac_obs_shape,
            'training_history_summary': {
                at: {
                    'dqn_episodes': self.training_history.get(at, {}).get('dqn', {}).get('num_episodes', 0),
                    'sac_episodes': self.training_history.get(at, {}).get('sac', {}).get('num_episodes', 0),
                    'dqn_mean_reward': self.training_history.get(at, {}).get('dqn', {}).get('mean_reward', 0.0),
                    'sac_mean_reward': self.training_history.get(at, {}).get('sac', {}).get('mean_reward', 0.0),
                } for at in self.attack_types
            }
        }
        meta_path = os.path.join(save_dir, "training_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"# Saved {saved_count} agents to {save_dir}/")
        return save_dir
    
    def load_agents(self, save_dir: str = "trained_rl_agents") -> bool:
        """Load previously trained DQN+SAC agents from disk.
        """
        import os, json, zipfile, shutil
        from datetime import datetime
        
        if not os.path.isdir(save_dir):
            print(f"  # Agent directory not found: {save_dir}")
            return False
        
        # --- Detect observation space mismatch before loading ---
            f'sac_{self.attack_types[0]}'
        ].observation_space.shape
        
        mismatch_detected = False
        sample_path = os.path.join(save_dir, f"sac_{self.attack_types[0]}.zip")
        if os.path.exists(sample_path):
            try:
                # Load without env to inspect saved observation space
                saved_agent = SAC.load(sample_path, env=None)
                saved_obs_shape = saved_agent.observation_space.shape
                if saved_obs_shape != current_obs_shape:
                    mismatch_detected = True
                    print(f"\n  #  OBSERVATION SPACE MISMATCH DETECTED")
                    print(f"     Saved models:      obs_shape={saved_obs_shape}")
                    print(f"     Current env:       obs_shape={current_obs_shape}")
                    print(f"     Cause: observation space was expanded from {saved_obs_shape[0]}"
                          f"  {current_obs_shape[0]} features (action feedback + cross-circle memory added)")
                    print(f"     Action: Archiving old models and retraining from scratch.")
                    print(f"     This is CORRECT — old models were trained with wrong reward signals anyway.")
                del saved_agent
            except Exception as e:
                print(f"  # Could not inspect saved model: {e}")
        
        if mismatch_detected:
            # Archive old models to a timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = f"{save_dir}_archived_{timestamp}"
            try:
                shutil.move(save_dir, archive_dir)
                print(f"  # Old models archived to: {archive_dir}/")
            except Exception as e:
                print(f"  # Could not archive old models: {e}")
            print(f"  # Retraining from scratch with new {current_obs_shape[0]}-feature observation space.\n")
            return False
        
        loaded_count = 0
        for attack_type in self.attack_types:
            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}.zip")
            sac_path = os.path.join(save_dir, f"sac_{attack_type}.zip")
            
            if not os.path.exists(dqn_path) or not os.path.exists(sac_path):
                print(f"  # Missing agent files for {attack_type}")
                continue
            
            # Load into existing environments
            dqn_env = self.environments[f'dqn_{attack_type}']
            sac_env = self.environments[f'sac_{attack_type}']
            
            try:
                self.dqn_agents[attack_type] = DQN.load(dqn_path, env=dqn_env)
                self.sac_agents[attack_type] = SAC.load(sac_path, env=sac_env)
                loaded_count += 2
                print(f"  # Loaded DQN+SAC for {attack_type}")
            except Exception as e:
                print(f"  # Failed to load agents for {attack_type}: {e}")
                print(f"     Skipping — agent will be retrained from scratch.")
        
        # Load metadata if available
        meta_path = os.path.join(save_dir, "training_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"  # Agents trained at: {metadata.get('timestamp', 'unknown')}")
            saved_obs = metadata.get('observation_space_shape')
            if saved_obs and tuple(saved_obs) != current_obs_shape:
                print(f"  # Metadata obs shape {saved_obs} != current {current_obs_shape}")
        
        success = loaded_count == len(self.attack_types) * 2
        if success:
            print(f"# Loaded {loaded_count} agents from {save_dir}/")
        else:
            print(f"# Partially loaded {loaded_count}/{len(self.attack_types) * 2} agents from {save_dir}/")
        
        return success
    
    def get_agent_action(self, attack_type: str, agent_type: str, observation: np.ndarray):
        """Get action from specific attack-type agent"""
        if agent_type == 'dqn':
            action, _ = self.dqn_agents[attack_type].predict(observation, deterministic=False)
            return action
        elif agent_type == 'sac':
            action, _ = self.sac_agents[attack_type].predict(observation, deterministic=False)
            return action
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def execute_deployment(self, deployment: AttackDeployment):
        """Execute a deployment strategy from Gemini using TRAINED agents.
        """
        attack_type    = deployment.attack_type
        target_systems = deployment.target_systems
        results        = []

        # Gemini tactical hints (may be default 0.7/30.0/0.6 if not set)
        g_mag     = float(deployment.magnitude)
        g_dur     = float(deployment.duration)
        g_stealth = float(deployment.stealth_level)

        # Blend weight: how much to trust Gemini vs. frozen SAC policy
        # 0.0 = pure SAC (old behaviour), 1.0 = pure Gemini
        GEMINI_BLEND = 0.40

        sac_agent = self.sac_agents.get(attack_type)

        for system_id in target_systems:
            env      = self.environments[f'sac_{attack_type}']
            obs, _   = env.reset()

            if sac_agent is not None:
                trained_action, _ = sac_agent.predict(obs, deterministic=True)
                action = trained_action.copy()

                # Blend SAC tactical dims with Gemini hints
                action[0] = (1 - GEMINI_BLEND) * action[0] + GEMINI_BLEND * g_mag
                action[1] = (1 - GEMINI_BLEND) * action[1] + GEMINI_BLEND * g_dur
                action[2] = (1 - GEMINI_BLEND) * action[2] + GEMINI_BLEND * g_stealth
                action[3] = float(system_id - 1)   # Gemini decides WHERE

                print(f"      🤖 Trained SAC policy (blended): mag={action[0]:.2f}, "
                      f"dur={action[1]:.1f}, stealth={action[2]:.2f} "
                      f"→ system {system_id} (Gemini target)")
            else:
                # No trained agent — use Gemini's parameters directly
                print(f"      # No trained SAC agent for {attack_type}, "
                      f"using Gemini params directly")
                action = np.array(
                    [g_mag, g_dur, g_stealth, float(system_id - 1)],
                    dtype=np.float32
                )

            _, reward, _, _, info = env.step(action)

            results.append({
                'attack_type': attack_type,
                'system_id':   system_id,
                'reward':      reward,
                'result':      info['attack_result']
            })

        return results
