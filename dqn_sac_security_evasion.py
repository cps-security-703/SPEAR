#!/usr/bin/env python3

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import EvalCallback
from typing import Dict, List, Tuple, Optional
import time
from collections import deque

class SecurityEvasionRewardCalculator:
    """Calculate rewards for RL agents learning to bypass anomaly detection"""
    
    def __init__(self):
        self.detection_penalty = -100.0  # Heavy penalty for being detected
        self.evasion_bonus = 50.0        # Bonus for successful evasion
        self.impact_multiplier = 10.0    # Multiplier for actual system impact
        self.stealth_bonus = 25.0        # Bonus for staying under detection thresholds
        
    def calculate_evasion_reward(self, attack_result: Dict, security_result: Dict, 
                               evcs_output_change: Dict) -> float:
        """Calculate reward based on security evasion and system impact"""
        
        total_reward = 0.0
        
        # 1. Security Evasion Component
        if security_result.get('attack_detected', False):
            total_reward += self.detection_penalty
            detection_time = security_result.get('detection_time', 1)
            total_reward += -50.0 / max(detection_time, 1)
        else:
            total_reward += self.evasion_bonus
            
        # 2. System Impact Component (only if not detected)
        if not security_result.get('attack_detected', False):
            voltage_change = abs(evcs_output_change.get('voltage_deviation', 0.0))
            power_change = abs(evcs_output_change.get('power_deviation', 0.0))
            current_change = abs(evcs_output_change.get('current_deviation', 0.0))
            
            impact_reward = (voltage_change + power_change + current_change) * self.impact_multiplier
            total_reward += impact_reward
            
        # 3. Stealth Bonus
        anomaly_score = security_result.get('anomaly_score', 0.0)
        detection_threshold = security_result.get('detection_threshold', 0.3)
        
        if anomaly_score < detection_threshold * 0.8:
            stealth_factor = (detection_threshold - anomaly_score) / detection_threshold
            total_reward += self.stealth_bonus * stealth_factor
            
        # 4. Consecutive Evasion Bonus
        consecutive_evasions = security_result.get('consecutive_evasions', 0)
        if consecutive_evasions > 3:
            total_reward += consecutive_evasions * 5.0
            
        return total_reward

class SecurityEvasionEnvironment(gym.Env):
    """Gym environment for training DQN/SAC agents to bypass security"""
    
    def __init__(self, cms_system, num_stations: int = 6):
        super(SecurityEvasionEnvironment, self).__init__()
        
        self.cms = cms_system
        self.num_stations = num_stations
        self.reward_calculator = SecurityEvasionRewardCalculator()
        
        # Track security state for RL learning
        self.security_history = deque(maxlen=20)
        self.consecutive_evasions = 0
        self.consecutive_detections = 0
        self.current_step = 0
        self.max_steps = 1000
        
        # Define observation space: [system_state(15) + security_state(10)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        
        # Define action space: [attack_type(6), magnitude(1), duration(1), timing(1), stealth(1), target(1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.1, 5.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 2.1, 35.0, 1.0, 1.0, float(num_stations-1)], dtype=np.float32),
            dtype=np.float32
        )
        
        self.current_station = 0
        self.baseline_outputs = {}
        self.security_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_station = np.random.randint(0, self.num_stations)
        self.consecutive_evasions = 0
        self.consecutive_detections = 0
        
        # Initialize baseline outputs
        for station_id in range(self.num_stations):
            self.baseline_outputs[station_id] = {
                'voltage': 400.0, 'current': 25.0, 'power': 10.0,
                'voltage_reference': 400.0, 'current_reference': 25.0, 'power_reference': 10.0,
                'soc': 0.5, 'grid_voltage': 1.0, 'grid_frequency': 60.0,
                'demand_factor': 1.0, 'voltage_priority': 0.0, 'urgency_factor': 1.0
            }
        
        observation = self._get_observation()
        info = {'station_id': self.current_station}
        
        return observation, info
    
    def step(self, action):
        """Execute one step with combined DQN+SAC actions"""
        self.current_step += 1
        
        # Get decisions from both agents
        if hasattr(self, 'trainer') and self.trainer:
            # Get DQN strategic decision
            dqn_obs = self._get_observation()  # Use full 25-feature observation
            dqn_action = self.trainer.dqn_agent.predict(dqn_obs, deterministic=True)[0]
            self.dqn_decision = self._decode_dqn_action(dqn_action)
            
            # Get SAC continuous control
            sac_obs = self._get_observation()  # Use full 25-feature observation
            sac_action = self.trainer.sac_agent.predict(sac_obs, deterministic=True)[0]
            self.sac_control = self._decode_sac_action(sac_action)
        else:
            # Fallback: treat action as combined
            self.dqn_decision = None
            self.sac_control = None
        
        # Convert combined actions to attack parameters
        attack_params = self._action_to_attack_params(action)
        
        # Ensure baseline outputs exist for current station
        if self.current_station not in self.baseline_outputs:
            self.baseline_outputs[self.current_station] = {
                'voltage': 400.0, 'current': 25.0, 'power': 10.0,
                'voltage_reference': 400.0, 'current_reference': 25.0, 'power_reference': 10.0,
                'soc': 0.5, 'grid_voltage': 1.0, 'grid_frequency': 60.0,
                'demand_factor': 1.0, 'voltage_priority': 0.0, 'urgency_factor': 1.0
            }
        
        # Execute attack and get security feedback
        security_result, evcs_output_change, final_output = self._execute_adaptive_attack(
            self.current_station, attack_params, self.baseline_outputs[self.current_station]
        )
        
        # Calculate reward based on security evasion and system impact
        if hasattr(self, 'reward_calculator') and self.reward_calculator:
            reward = self.reward_calculator.calculate_evasion_reward(
                attack_params, security_result, evcs_output_change
            )
        else:
            # Fallback reward calculation
            reward = -1.0 if security_result.get('attack_detected', True) else 1.0
        
        # Update counters
        if security_result.get('attack_detected', False):
            self.consecutive_detections += 1
            self.consecutive_evasions = 0
        else:
            self.consecutive_evasions += 1
            self.consecutive_detections = 0
        
        # Store in history
        self.security_history.append({
            'detected': security_result.get('attack_detected', False),
            'timestamp': time.time(),
            'attack_params': attack_params,
            'output_change': evcs_output_change
        })
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        
        info = {
            'station_id': self.current_station,
            'attack_detected': security_result.get('attack_detected', False),
            'evcs_impact': evcs_output_change.get('total_impact', 0.0),
            'consecutive_evasions': self.consecutive_evasions,
            'security_result': security_result
        }
        
        # Switch to random station for next step
        self.current_station = np.random.randint(0, self.num_stations)
        
        return observation, reward, terminated, truncated, info
    
    def _get_dqn_observation(self):
        """Get observation for DQN agent (discrete strategic decisions)"""
        if self.current_station not in self.baseline_outputs:
            # Initialize if missing
            self.baseline_outputs[self.current_station] = {
                'voltage_reference': 400.0, 'current_reference': 25.0, 'power_reference': 10.0,
                'soc': 0.5, 'grid_voltage': 1.0, 'grid_frequency': 60.0
            }
        
        station_data = self.baseline_outputs[self.current_station]
        security_history_len = len(getattr(self, 'security_history', []))
        
        # DQN focuses on strategic context
        return np.array([
            station_data.get('power_reference', 10.0) / 100.0,  # Normalized power
            station_data.get('voltage_reference', 400.0) / 500.0,  # Normalized voltage
            station_data.get('current_reference', 25.0) / 200.0,  # Normalized current
            self.current_step / self.max_steps,  # Time progress
            security_history_len / 10.0,  # Security event density
            self.current_station / self.num_stations  # Station context
        ], dtype=np.float32)
    
    def _get_sac_observation(self):
        """Get observation for SAC agent (continuous control)"""
        if self.current_station not in self.baseline_outputs:
            # Initialize if missing
            self.baseline_outputs[self.current_station] = {
                'voltage_reference': 400.0, 'current_reference': 25.0, 'power_reference': 10.0,
                'soc': 0.5, 'grid_voltage': 1.0, 'grid_frequency': 60.0,
                'demand_factor': 1.0, 'urgency_factor': 1.0
            }
        
        station_data = self.baseline_outputs[self.current_station]
        
        # SAC focuses on fine-grained control parameters
        return np.array([
            station_data.get('power_reference', 10.0) / 100.0,
            station_data.get('voltage_reference', 400.0) / 500.0,
            station_data.get('current_reference', 25.0) / 200.0,
            station_data.get('soc', 0.5),
            station_data.get('grid_voltage', 1.0),
            station_data.get('grid_frequency', 60.0) / 60.0,
            station_data.get('demand_factor', 1.0),
            station_data.get('urgency_factor', 1.0)
        ], dtype=np.float32)
    
    def _decode_dqn_action(self, action):
        """Decode DQN discrete action to strategic decisions"""
        attack_types = ['demand_increase', 'demand_decrease', 'oscillating_demand', 
                       'voltage_spoofing', 'frequency_spoofing', 'soc_manipulation']
        
        return {
            'attack_type': int(action) % len(attack_types),
            'target_strategy': (int(action) // len(attack_types)) % 3,  # 0: single, 1: multiple, 2: random
            'timing_window': (int(action) // (len(attack_types) * 3)) % 4,  # 0: immediate, 1: delayed, 2: peak, 3: off-peak
            'evasion_strategy': (int(action) // (len(attack_types) * 12)) % 3,  # 0: stealth, 1: burst, 2: adaptive
            'primary_target': self.current_station
        }
    
    def _decode_sac_action(self, action):
        """Decode SAC continuous action to control parameters"""
        return {
            'magnitude': np.clip(action[0], -1.0, 1.0),  # Attack magnitude
            'stealth_factor': np.clip(action[1], 0.0, 1.0),  # Stealth level
            'duration': np.clip(action[2], 0.1, 2.0),  # Duration multiplier
            'fine_tuning': np.clip(action[3], -0.5, 0.5)  # Fine adjustment
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation combining system and security state"""
        
        # System state features (15 dimensions)
        system_features = self._get_system_state(self.current_station)
        
        # Security state features (10 dimensions)
        security_features = self._get_security_state(self.current_station)
        
        # Combine into single observation
        observation = np.concatenate([system_features, security_features])
        
        return observation.astype(np.float32)
    
    def _get_system_state(self, station_id: int) -> np.ndarray:
        """Get system state features"""
        features = []
        
        # Station-specific features
        if self.cms and hasattr(self.cms, 'stations') and station_id < len(self.cms.stations):
            station = self.cms.stations[station_id]
            # Use available attributes from EVChargingStation
            soc_estimate = 0.5  # Default SOC estimate
            max_power_norm = getattr(station, 'max_power', 50000.0) / 100000.0  # Normalize to 100kW
            efficiency_estimate = 0.95  # Default efficiency
            
            # Try to get actual values if available
            if hasattr(station, 'evcs_controller') and station.evcs_controller:
                # Get current state from EVCS controller if available
                try:
                    current_state = station.evcs_controller.get_current_state()
                    if 'soc' in current_state:
                        soc_estimate = current_state['soc']
                except:
                    pass
            
            features.extend([soc_estimate, max_power_norm, efficiency_estimate])
        else:
            # Fallback when CMS is None or station not available
            features.extend([0.5, 0.5, 0.95])  # Default values
        
        # Grid conditions and other features
        features.extend([
            1.0,    # Grid voltage (pu)
            60.0,   # Grid frequency  
            0.8,    # Load factor
            time.time() % 86400 / 86400.0,  # Time of day normalized
            0.5,    # Historical attack success
            0.1,    # Voltage deviation
            0.05,   # Frequency deviation
            0.2,    # System congestion
            float(self.consecutive_evasions) / 10.0,  # Normalized evasion streak
            float(self.consecutive_detections) / 10.0,  # Normalized detection streak
            0.5,    # System stability factor
            0.3,    # Power quality index
            0.7     # Grid reliability score
        ])
        
        # Pad to 15 dimensions
        while len(features) < 15:
            features.append(0.0)
            
        return np.array(features[:15], dtype=np.float32)
    
    def _get_security_state(self, station_id: int) -> np.ndarray:
        """Get security system state"""
        security_features = []
        
        # Anomaly detection thresholds and current values
        if self.cms and hasattr(self.cms, 'anomaly_counters') and station_id in self.cms.anomaly_counters:
            security_features.append(self.cms.anomaly_counters[station_id] / 3.0)
        else:
            security_features.append(0.0)
            
        # Security parameters
        if self.cms:
            security_features.extend([
                getattr(self.cms, 'rate_change_limit', 0.5),
                getattr(self.cms, 'max_power_reference', 100.0) / 100.0,
                getattr(self.cms, 'max_voltage_reference', 500.0) / 500.0,
                getattr(self.cms, 'max_current_reference', 200.0) / 200.0
            ])
        else:
            # Fallback when CMS is None
            security_features.extend([0.5, 1.0, 1.0, 1.0])
        
        # Historical detection patterns
        recent_detections = sum(1 for h in list(self.security_history)[-5:] if h.get('detected', False))
        security_features.append(recent_detections / 5.0)
        
        # Time since last detection
        time_since_detection = 0
        for i, h in enumerate(reversed(self.security_history)):
            if h.get('detected', False):
                time_since_detection = i
                break
        security_features.append(min(time_since_detection / 10.0, 1.0))
        
        # Current thresholds
        if self.cms:
            security_features.extend([
                getattr(self.cms, 'anomaly_threshold', 0.3),
                0.4,  # Statistical detection sensitivity
                float(len(self.security_history)) / 20.0,  # History fullness
                getattr(self.cms, 'detection_sensitivity', 0.5),
                getattr(self.cms, 'response_threshold', 0.6)
            ])
        else:
            # Fallback when CMS is None
            security_features.extend([0.3, 0.4, 0.5, 0.5, 0.6])
        
        # Pad to 10 dimensions
        while len(security_features) < 10:
            security_features.append(0.0)
            
        return np.array(security_features[:10], dtype=np.float32)
    
    def _action_to_attack_params(self, action: np.ndarray) -> Dict:
        """Convert combined DQN+SAC actions to attack parameters"""
        # This method now expects combined actions from both agents
        if hasattr(self, 'dqn_decision') and hasattr(self, 'sac_control'):
            # Use stored decisions from both agents
            dqn_decision = self.dqn_decision
            sac_control = self.sac_control
        else:
            # Fallback to single agent mode
            attack_types = ['demand_increase', 'demand_decrease', 'oscillating_demand', 
                           'voltage_spoofing', 'frequency_spoofing', 'soc_manipulation']
            
            return {
                'type': attack_types[int(action[0]) % len(attack_types)],
                'magnitude': float(action[1]),
                'duration': int(action[2]),
                'timing_strategy': float(action[3]),
                'stealth_factor': float(action[4]),
                'targets': [int(action[5]) % self.num_stations]
            }
        
        # Combined DQN+SAC attack parameters
        attack_types = ['demand_increase', 'demand_decrease', 'oscillating_demand', 
                       'voltage_spoofing', 'frequency_spoofing', 'soc_manipulation']
        
        # Handle None cases with fallbacks
        if dqn_decision is None:
            dqn_decision = {
                'attack_type': int(action[0]) % len(attack_types),
                'target_strategy': 0,
                'timing_window': 0,
                'evasion_strategy': 0,
                'primary_target': self.current_station
            }
        
        if sac_control is None:
            sac_control = {
                'magnitude': action[1] if len(action) > 1 else 1.0,
                'stealth_factor': action[4] if len(action) > 4 else 0.5,
                'duration': action[2] if len(action) > 2 else 10.0,
                'fine_tuning': action[3] if len(action) > 3 else 0.5
            }
        
        return {
            'type': attack_types[dqn_decision['attack_type']],
            'target_selection': dqn_decision['target_strategy'],
            'timing_window': dqn_decision['timing_window'],
            'evasion_strategy': dqn_decision['evasion_strategy'],
            'magnitude': sac_control['magnitude'],
            'stealth_factor': sac_control['stealth_factor'],
            'duration': sac_control['duration'],
            'fine_tuning': sac_control['fine_tuning'],
            'targets': [dqn_decision['primary_target']]
        }
    
    def _execute_adaptive_attack(self, station_id: int, attack_params: Dict, 
                               baseline_output: Dict) -> Tuple[Dict, Dict, Dict]:
        """Execute attack and return security feedback"""
        try:
            # Apply attack to CMS inputs
            station_data = {
                'soc': baseline_output.get('soc', 0.5),
                'grid_voltage': baseline_output.get('grid_voltage', 1.0),
                'grid_frequency': baseline_output.get('grid_frequency', 60.0),
                'demand_factor': baseline_output.get('demand_factor', 1.0),
                'voltage_priority': baseline_output.get('voltage_priority', 0.0),
                'urgency_factor': baseline_output.get('urgency_factor', 1.0),
                'current_time': time.time()
            }
            
            # Simulate attack by modifying inputs based on attack type
            attacked_data = station_data.copy()
            attack_type = attack_params.get('type', 'demand_increase')
            magnitude = attack_params.get('magnitude', 1.0)
            
            if attack_type == 'demand_increase':
                attacked_data['demand_factor'] *= (1.0 + magnitude * 5.0)  # Increased from 0.5 to 5.0 for higher impact
            elif attack_type == 'demand_decrease':
                attacked_data['demand_factor'] *= (1.0 - magnitude * 2.0)  # Increased from 0.3 to 2.0 for higher impact
            elif attack_type == 'voltage_spoofing':
                attacked_data['grid_voltage'] *= (1.0 + magnitude * 1.0)  # Increased from 0.1 to 1.0 for higher impact
            elif attack_type == 'frequency_spoofing':
                attacked_data['grid_frequency'] += magnitude * 10.0  # Increased from 2.0 to 10.0 for higher impact
            elif attack_type == 'soc_manipulation':
                attacked_data['soc'] = min(1.0, attacked_data['soc'] + magnitude * 1.0)  # Increased from 0.2 to 1.0 for higher impact
            elif attack_type == 'oscillating_demand':
                attacked_data['demand_factor'] *= (1.0 + magnitude * 2.0 * np.sin(time.time()))  # Increased from 0.3 to 2.0 for higher impact
            
            # Get CMS response
            if self.cms and hasattr(self.cms, 'federated_manager') and self.cms.federated_manager:
                voltage_ref, current_ref, power_ref = self.cms.federated_manager.optimize_references(
                    station_id, attacked_data
                )
            else:
                voltage_ref, current_ref, power_ref = 400.0, 25.0, 10.0
            
            # Apply security validation if available
            if self.cms and hasattr(self.cms, '_security_validation'):
                final_v, final_i, final_p, not_detected = self.cms._security_validation(
                    station_id, voltage_ref, current_ref, power_ref, attacked_data, time.time()
                )
                attack_detected = not not_detected
            else:
                # Simulate security detection based on deviation magnitude
                final_v, final_i, final_p = voltage_ref, current_ref, power_ref
                attack_detected = magnitude > 1.5  # Simple threshold-based detection
            
            # Calculate output changes
            evcs_output_change = {
                'voltage_deviation': abs(final_v - baseline_output.get('voltage', 400.0)) / 400.0,
                'current_deviation': abs(final_i - baseline_output.get('current', 25.0)) / 25.0,
                'power_deviation': abs(final_p - baseline_output.get('power', 10.0)) / 10.0
            }
            evcs_output_change['total_impact'] = sum(evcs_output_change.values()) / 3.0
            
            # Security system feedback
            security_result = {
                'attack_detected': attack_detected,
                'detection_time': 1,
                'anomaly_score': self._calculate_anomaly_score(attacked_data, station_data),
                'detection_threshold': getattr(self.cms, 'anomaly_threshold', 0.3) if self.cms else 0.3,
                'consecutive_evasions': self.consecutive_evasions if not attack_detected else 0,
                'consecutive_detections': self.consecutive_detections if attack_detected else 0
            }
            
            return security_result, evcs_output_change, {
                'voltage': final_v, 'current': final_i, 'power': final_p
            }
            
        except Exception as e:
            # Return default values instead of error to prevent training interruption
            return {
                'attack_detected': True, 
                'detection_time': 1,
                'anomaly_score': 1.0,
                'detection_threshold': 0.3,
                'consecutive_evasions': 0,
                'consecutive_detections': 1
            }, {
                'voltage_deviation': 0.0,
                'current_deviation': 0.0, 
                'power_deviation': 0.0,
                'total_impact': 0.0
            }, {
                'voltage': 400.0, 'current': 25.0, 'power': 10.0
            }
    
    def _calculate_anomaly_score(self, attacked_data: Dict, original_data: Dict) -> float:
        """Calculate anomaly score"""
        score = 0.0
        for key in ['demand_factor', 'urgency_factor', 'grid_voltage']:
            if key in attacked_data and key in original_data:
                deviation = abs(attacked_data[key] - original_data[key]) / max(original_data[key], 0.1)
                score += deviation
        return min(score / 3.0, 1.0)

class DiscreteSecurityEvasionEnv(SecurityEvasionEnvironment):
    """Discrete action space version for DQN"""
    
    def __init__(self, cms_system, num_stations):
        super().__init__(cms_system, num_stations)
        # Discrete actions: attack_type(6) * magnitude_levels(5) * duration_levels(6) = 180 actions
        self.action_space = spaces.Discrete(180)
        
    def step(self, action):
        # Convert discrete action to continuous parameters
        attack_type = action % 6
        magnitude_level = (action // 6) % 5
        duration_level = (action // 30) % 6
        
        continuous_action = np.array([
            float(attack_type),
            0.1 + magnitude_level * 0.4,  # 0.1 to 2.1
            5.0 + duration_level * 5.0,   # 5 to 35
            0.5,  # Default timing
            0.5,  # Default stealth
            float(self.current_station)  # Current target
        ], dtype=np.float32)
        
        return super().step(continuous_action)

class DQNSACSecurityEvasionTrainer:
    """Trainer using DQN and SAC agents to learn security bypass"""
    
    def __init__(self, cms_system, num_stations: int = 6, use_both: bool = True):
        self.cms = cms_system
        self.num_stations = num_stations
        
        # Create environments
        self.sac_env = SecurityEvasionEnvironment(cms_system, num_stations)
        self.dqn_env = DiscreteSecurityEvasionEnv(cms_system, num_stations)
        
        # Initialize agents
        self.sac_agent = None
        self.dqn_agent = None
        
        if use_both:
            print("🎯 Initializing DQN and SAC agents for security evasion...")
            
            # SAC Agent (continuous actions)
            self.sac_agent = SAC(
                'MlpPolicy',
                self.sac_env,
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
                verbose=1
            )
            
            # DQN Agent (discrete actions)
            self.dqn_agent = DQN(
                'MlpPolicy',
                self.dqn_env,
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
                verbose=1
            )
        
        self.training_history = []
        
        # Try to load pre-trained agents if they exist
        self.load_agents()
    
    def train_agents(self, sac_timesteps: int = 50000, dqn_timesteps: int = 50000):
        """Train both DQN and SAC agents"""
        
        print("🚀 Starting DQN/SAC Security Evasion Training")
        
        if self.sac_agent:
            print(f"\n🤖 Training SAC Agent ({sac_timesteps} timesteps)...")
            self.sac_agent.learn(
                total_timesteps=sac_timesteps,
                log_interval=1000,
                progress_bar=True
            )
            print("✅ SAC training completed")
        
        if self.dqn_agent:
            print(f"\n🤖 Training DQN Agent ({dqn_timesteps} timesteps)...")
            self.dqn_agent.learn(
                total_timesteps=dqn_timesteps,
                log_interval=1000,
                progress_bar=True
            )
            print(" DQN training completed")
    
    def evaluate_agents(self, num_episodes: int = 100):
        """Evaluate both agents"""
        results = {}
        
        if self.sac_agent:
            print("\n Evaluating SAC Agent...")
            results['sac'] = self._evaluate_agent(self.sac_agent, self.sac_env, num_episodes)
        
        if self.dqn_agent:
            print("\n Evaluating DQN Agent...")
            results['dqn'] = self._evaluate_agent(self.dqn_agent, self.dqn_env, num_episodes)
        
        return results
    
    def get_coordinated_attack(self, station_id: int, baseline_outputs: Dict):
        """Get coordinated attack from both DQN and SAC agents"""
        if not (self.dqn_agent and self.sac_agent):
            return None
        
        # Validate and normalize baseline_outputs
        if not baseline_outputs:
            baseline_outputs = {
                'voltage': 400.0, 'current': 25.0, 'power': 10.0,
                'voltage_reference': 400.0, 'current_reference': 25.0, 'power_reference': 10.0,
                'soc': 0.5, 'grid_voltage': 1.0, 'grid_frequency': 60.0,
                'demand_factor': 1.0, 'voltage_priority': 0.0, 'urgency_factor': 1.0
            }
        
        # Ensure all required keys exist
        required_keys = ['voltage_reference', 'current_reference', 'power_reference', 'soc', 'grid_voltage', 'grid_frequency']
        for key in required_keys:
            if key not in baseline_outputs:
                if key.endswith('_reference'):
                    base_key = key.replace('_reference', '')
                    baseline_outputs[key] = baseline_outputs.get(base_key, 400.0 if 'voltage' in key else 25.0 if 'current' in key else 10.0)
                else:
                    baseline_outputs[key] = 0.5 if key == 'soc' else 1.0 if 'grid' in key else 60.0 if 'frequency' in key else 1.0
        
        # Set current station for observations
        self.sac_env.current_station = station_id
        self.dqn_env.current_station = station_id
        self.sac_env.baseline_outputs[station_id] = baseline_outputs
        self.dqn_env.baseline_outputs[station_id] = baseline_outputs
        
        try:
            # Get DQN strategic decision
            dqn_obs = self.dqn_env._get_observation()
            dqn_action = self.dqn_agent.predict(dqn_obs, deterministic=True)[0]
            dqn_decision = self.dqn_env._decode_dqn_action(dqn_action)
            
            # Get SAC continuous control
            sac_obs = self.sac_env._get_observation()
            sac_action = self.sac_agent.predict(sac_obs, deterministic=True)[0]
            sac_control = self.sac_env._decode_sac_action(sac_action)
        except Exception as e:
            print(f" Error in coordinated attack generation: {e}")
            return None
        
        # Store decisions in environment for attack parameter generation
        self.sac_env.dqn_decision = dqn_decision
        self.sac_env.sac_control = sac_control
        
        # Generate combined attack parameters
        combined_action = np.zeros(6)  # Placeholder for compatibility
        attack_params = self.sac_env._action_to_attack_params(combined_action)
        
        return {
            'dqn_decision': dqn_decision,
            'sac_control': sac_control,
            'attack_params': attack_params
        }
    
    def train_coordinated_agents(self, total_timesteps: int = 100000):
        """Train agents with coordinated learning"""
        print("Starting Coordinated DQN/SAC Training")
        
        # Set trainer reference in environments for coordination
        self.sac_env.trainer = self
        self.dqn_env.trainer = self
        
        # Alternate training between agents
        sac_steps = total_timesteps // 2
        dqn_steps = total_timesteps // 2
        
        print(f"\n Phase 1: SAC Agent Training ({sac_steps} timesteps)...")
        if self.sac_agent:
            self.sac_agent.learn(
                total_timesteps=sac_steps,
                log_interval=1000,
                progress_bar=True
            )
        
        print(f"\n Phase 2: DQN Agent Training ({dqn_steps} timesteps)...")
        if self.dqn_agent:
            self.dqn_agent.learn(
                total_timesteps=dqn_steps,
                log_interval=1000,
                progress_bar=True
            )
        
        print(" Coordinated training completed")
    
    def _evaluate_agent(self, agent, env, num_episodes: int):
        """Evaluate a single agent"""
        episode_rewards = []
        detection_rates = []
        evasion_rates = []
        impact_scores = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            detections = []
            impacts = []
            
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                detections.append(info.get('attack_detected', False))
                impacts.append(info.get('evcs_impact', 0.0))
                
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            detection_rates.append(np.mean(detections))
            evasion_rates.append(1.0 - np.mean(detections))
            impact_scores.append(np.mean(impacts))
        
        results = {
            'avg_reward': np.mean(episode_rewards),
            'avg_detection_rate': np.mean(detection_rates),
            'avg_evasion_rate': np.mean(evasion_rates),
            'avg_impact_score': np.mean(impact_scores),
            'reward_std': np.std(episode_rewards)
        }
        
        print(f"   Average Reward: {results['avg_reward']:.2f} ± {results['reward_std']:.2f}")
        print(f"   Evasion Rate: {results['avg_evasion_rate']:.2%}")
        print(f"   Average Impact: {results['avg_impact_score']:.3f}")
        
        return results
    
    def save_agents(self, save_dir: str = "./models/"):
        """Save trained agents"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if self.dqn_agent:
            self.dqn_agent.save(f"{save_dir}/dqn_security_evasion")
            print(f" DQN agent saved to {save_dir}/dqn_security_evasion")
        
        if self.sac_agent:
            self.sac_agent.save(f"{save_dir}/sac_security_evasion")
            print(f" SAC agent saved to {save_dir}/sac_security_evasion")
    
    def load_agents(self, save_dir: str = "./models/"):
        """Load trained agents if they exist, otherwise use fresh agents"""
        import os
        
        try:
            if self.dqn_agent:
                dqn_path = f"{save_dir}/dqn_security_evasion.zip"
                if os.path.exists(dqn_path):
                    self.dqn_agent = DQN.load(dqn_path)
                    print(f"✅ DQN agent loaded from {dqn_path}")
                else:
                    print(f"⚠️ DQN model not found at {dqn_path}, using fresh agent")
            
            if self.sac_agent:
                sac_path = f"{save_dir}/sac_security_evasion.zip"
                if os.path.exists(sac_path):
                    self.sac_agent = SAC.load(sac_path)
                    print(f"✅ SAC agent loaded from {sac_path}")
                else:
                    print(f"⚠️ SAC model not found at {sac_path}, using fresh agent")
        except Exception as e:
            print(f"⚠️ Error loading agents: {e}")
            print("   Using fresh agents instead")

def create_dqn_sac_evasion_system(cms_system):
    """Factory function to create DQN/SAC security evasion system"""
    
    trainer = DQNSACSecurityEvasionTrainer(cms_system)
    
    print(" Created DQN/SAC Security Evasion System")
    print("   - DQN agent for discrete attack decisions")
    print("   - SAC agent for continuous attack parameters") 
    print("   - Rewards based on actual EVCS output changes")
    print("   - Learning to bypass anomaly detection")
    
    return trainer

if __name__ == "__main__":
    print(" DQN/SAC Security Evasion System")
    print("This system uses DQN and SAC agents to learn bypassing security measures")
    print("Run from main simulation to integrate with CMS")
