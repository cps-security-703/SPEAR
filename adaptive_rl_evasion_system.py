#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import os
import sys

# Import the new DQN/SAC security evasion system
try:
    from dqn_sac_security_evasion import (
        DQNSACSecurityEvasionTrainer,
        SecurityEvasionEnvironment,
        DiscreteSecurityEvasionEnv,
        SecurityEvasionRewardCalculator
    )
except ImportError:
    print("Warning: Could not import DQN/SAC security evasion system")
    DQNSACSecurityEvasionTrainer = None

# Legacy reward calculator - now using the one from dqn_sac_security_evasion.py
class LegacySecurityEvasionRewardCalculator:
    """Legacy reward calculator - replaced by DQN/SAC version"""
    
    def __init__(self):
        print("⚠️  Using legacy reward calculator. Consider upgrading to DQN/SAC version.")
        self.detection_penalty = -100.0
        self.evasion_bonus = 50.0
        self.impact_multiplier = 10.0
        self.stealth_bonus = 25.0
        
    def calculate_evasion_reward(self, attack_result: Dict, security_result: Dict, 
                               evcs_output_change: Dict) -> float:
        """Calculate reward based on security evasion and system impact"""
        total_reward = 0.0
        
        if security_result.get('attack_detected', False):
            total_reward += self.detection_penalty
            detection_time = security_result.get('detection_time', 1)
            total_reward += -50.0 / max(detection_time, 1)
        else:
            total_reward += self.evasion_bonus
            
        if not security_result.get('attack_detected', False):
            voltage_change = abs(evcs_output_change.get('voltage_deviation', 0.0))
            power_change = abs(evcs_output_change.get('power_deviation', 0.0))
            current_change = abs(evcs_output_change.get('current_deviation', 0.0))
            impact_reward = (voltage_change + power_change + current_change) * self.impact_multiplier
            total_reward += impact_reward
            
        anomaly_score = security_result.get('anomaly_score', 0.0)
        detection_threshold = security_result.get('detection_threshold', 0.3)
        
        if anomaly_score < detection_threshold * 0.8:
            stealth_factor = (detection_threshold - anomaly_score) / detection_threshold
            total_reward += self.stealth_bonus * stealth_factor
            
        consecutive_evasions = security_result.get('consecutive_evasions', 0)
        if consecutive_evasions > 3:
            total_reward += consecutive_evasions * 5.0
            
        return total_reward

# Legacy custom neural network agent - now replaced by DQN/SAC
class LegacyAdaptiveAttackAgent:
    """Legacy custom RL agent - replaced by DQN/SAC agents"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        print("⚠️  Using legacy custom neural network agent. Consider upgrading to DQN/SAC.")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, state, security_state):
        """Legacy forward pass - placeholder"""
        return torch.zeros(self.action_dim), torch.tensor(0.5)
        
    def learn_from_detection(self, state, action, security_state, detected):
        """Legacy learning - placeholder"""
        pass

class SecurityAwareAttackEnvironment:
    """Environment that provides feedback on security evasion"""
    
    def __init__(self, cms_system, security_system):
        self.cms = cms_system
        self.security = security_system
        self.reward_calculator = SecurityEvasionRewardCalculator()
        
        # Track security state for RL learning
        self.security_history = deque(maxlen=20)
        self.consecutive_evasions = 0
        self.consecutive_detections = 0
        
    def get_security_state(self, station_id: int) -> torch.Tensor:
        """Get current security system state for RL agent"""
        
        security_features = []
        
        # Anomaly detection thresholds and current values
        if hasattr(self.cms, 'anomaly_counters') and station_id in self.cms.anomaly_counters:
            security_features.append(self.cms.anomaly_counters[station_id] / 3.0)  # Normalized
        else:
            security_features.append(0.0)
            
        # Rate change limits
        security_features.append(self.cms.rate_change_limit if hasattr(self.cms, 'rate_change_limit') else 0.5)
        
        # Upper bounds (normalized)
        security_features.extend([
            self.cms.max_power_reference / 100.0 if hasattr(self.cms, 'max_power_reference') else 1.0,
            self.cms.max_voltage_reference / 500.0 if hasattr(self.cms, 'max_voltage_reference') else 1.0,
            self.cms.max_current_reference / 200.0 if hasattr(self.cms, 'max_current_reference') else 1.0
        ])
        
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
        
        # Current anomaly threshold
        security_features.append(self.cms.anomaly_threshold if hasattr(self.cms, 'anomaly_threshold') else 0.3)
        
        # Statistical detection sensitivity
        security_features.append(0.4)  # Z-score threshold / 6.0 (2.5/6.0)
        
        # Pad to fixed size
        while len(security_features) < 10:
            security_features.append(0.0)
            
        return torch.tensor(security_features[:10], dtype=torch.float32)
    
    def execute_adaptive_attack(self, station_id: int, attack_params: Dict, 
                              baseline_output: Dict) -> Tuple[Dict, Dict, Dict]:
        """Execute attack and return security feedback"""
        
        # Store baseline
        pre_attack_output = baseline_output.copy()
        
        # Execute attack through CMS
        attack_detected = False
        security_result = {}
        
        try:
            # Apply attack to CMS inputs (using our fixed attack surface)
            station_data = {
                'soc': baseline_output.get('soc', 0.5),
                'grid_voltage': baseline_output.get('grid_voltage', 1.0),
                'grid_frequency': baseline_output.get('grid_frequency', 60.0),
                'demand_factor': baseline_output.get('demand_factor', 1.0),
                'voltage_priority': baseline_output.get('voltage_priority', 0.0),
                'urgency_factor': baseline_output.get('urgency_factor', 1.0),
                'current_time': time.time()
            }
            
            # Apply attack to inputs
            attacked_data = self.cms._apply_input_attacks(station_data, station_id, time.time())
            
            # Get CMS response
            voltage_ref, current_ref, power_ref = self.cms.federated_manager.optimize_references(
                station_id, attacked_data
            )
            
            # Apply security validation
            final_v, final_i, final_p, not_detected = self.cms._security_validation(
                station_id, voltage_ref, current_ref, power_ref, attacked_data, time.time()
            )
            
            attack_detected = not not_detected
            
            # Calculate actual output changes
            evcs_output_change = {
                'voltage_deviation': abs(final_v - pre_attack_output.get('voltage', 400.0)) / 400.0,
                'current_deviation': abs(final_i - pre_attack_output.get('current', 25.0)) / 25.0,
                'power_deviation': abs(final_p - pre_attack_output.get('power', 10.0)) / 10.0
            }
            
            # Security system feedback
            security_result = {
                'attack_detected': attack_detected,
                'detection_time': 1,
                'anomaly_score': self._calculate_anomaly_score(attacked_data, station_data),
                'detection_threshold': self.cms.anomaly_threshold,
                'consecutive_evasions': self.consecutive_evasions if not attack_detected else 0,
                'consecutive_detections': self.consecutive_detections if attack_detected else 0
            }
            
            # Update counters
            if attack_detected:
                self.consecutive_detections += 1
                self.consecutive_evasions = 0
            else:
                self.consecutive_evasions += 1
                self.consecutive_detections = 0
            
            # Store in history
            self.security_history.append({
                'detected': attack_detected,
                'timestamp': time.time(),
                'attack_params': attack_params,
                'output_change': evcs_output_change
            })
            
            return security_result, evcs_output_change, {
                'voltage': final_v,
                'current': final_i, 
                'power': final_p
            }
            
        except Exception as e:
            print(f"Error in adaptive attack execution: {e}")
            return {'attack_detected': True, 'error': str(e)}, {}, {}
    
    def _calculate_anomaly_score(self, attacked_data: Dict, original_data: Dict) -> float:
        """Calculate how anomalous the attack appears to security system"""
        
        score = 0.0
        
        # Check input deviations
        for key in ['demand_factor', 'urgency_factor', 'grid_voltage']:
            if key in attacked_data and key in original_data:
                deviation = abs(attacked_data[key] - original_data[key]) / max(original_data[key], 0.1)
                score += deviation
        
        return min(score / 3.0, 1.0)  # Normalize to [0,1]

class AdaptiveRLAttackTrainer:
    """Enhanced trainer using DQN/SAC agents for security evasion"""
    
    def __init__(self, cms_system, num_stations: int = 6, use_dqn_sac: bool = True):
        self.cms = cms_system
        self.num_stations = num_stations
        self.use_dqn_sac = use_dqn_sac
        
        if use_dqn_sac and DQNSACSecurityEvasionTrainer:
            print("🚀 Initializing DQN/SAC Security Evasion Trainer")
            self.dqn_sac_trainer = DQNSACSecurityEvasionTrainer(cms_system, num_stations)
            self.environment = None  # Will use DQN/SAC environments
        else:
            print("⚠️  Falling back to legacy custom neural network trainer")
            self.environment = SecurityAwareAttackEnvironment(cms_system, cms_system)
            self.agents = {}
            for station_id in range(num_stations):
                self.agents[station_id] = LegacyAdaptiveAttackAgent(
                    state_dim=15, action_dim=6, hidden_dim=256
                )
            self.dqn_sac_trainer = None
        
        self.training_history = []
        
    def train_adaptive_agents(self, num_episodes: int = 1000, episode_length: int = 100, 
                            sac_timesteps: int = 50000, dqn_timesteps: int = 50000):
        """Train RL agents to learn security evasion"""
        
        if self.use_dqn_sac and self.dqn_sac_trainer:
            print(f"🎯 Starting DQN/SAC adaptive RL training")
            
            # Train using DQN and SAC agents
            self.dqn_sac_trainer.train_agents(sac_timesteps, dqn_timesteps)
            
            # Evaluate the trained agents
            results = self.dqn_sac_trainer.evaluate_agents(num_episodes=100)
            
            # Store results in training history
            self.training_history.append({
                'method': 'DQN_SAC',
                'results': results,
                'timesteps': {'sac': sac_timesteps, 'dqn': dqn_timesteps}
            })
            
            print("✅ DQN/SAC training completed successfully")
            return results
            
        else:
            print(f"🎯 Starting legacy adaptive RL training: {num_episodes} episodes")
            return self._train_legacy_agents(num_episodes, episode_length)
    
    def _train_legacy_agents(self, num_episodes: int, episode_length: int):
        """Legacy training method using custom neural networks"""
        
        for episode in range(num_episodes):
            episode_rewards = []
            episode_detections = []
            
            for step in range(episode_length):
                station_id = np.random.randint(0, self.num_stations)
                agent = self.agents[station_id]
                
                system_state = self._get_system_state(station_id)
                security_state = self.environment.get_security_state(station_id)
                
                action, detection_prob = agent.forward(system_state, security_state)
                attack_params = self._action_to_attack_params(action)
                
                baseline_output = self._get_baseline_output(station_id)
                
                security_result, evcs_output_change, final_output = self.environment.execute_adaptive_attack(
                    station_id, attack_params, baseline_output
                )
                
                reward = self.environment.reward_calculator.calculate_evasion_reward(
                    attack_params, security_result, evcs_output_change
                )
                
                agent.learn_from_detection(
                    system_state, action, security_state, 
                    security_result.get('attack_detected', False)
                )
                
                episode_rewards.append(reward)
                episode_detections.append(security_result.get('attack_detected', False))
            
            avg_reward = np.mean(episode_rewards)
            detection_rate = np.mean(episode_detections)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Detection Rate: {detection_rate:.2%}")
            
            self.training_history.append({
                'episode': episode,
                'avg_reward': avg_reward,
                'detection_rate': detection_rate,
                'evasion_rate': 1.0 - detection_rate
            })
        
        return {'legacy_training': 'completed'}
    
    def _get_system_state(self, station_id: int) -> torch.Tensor:
        """Get current system state for RL agent"""
        
        # System features that RL agent can observe
        features = []
        
        # Station-specific features
        if hasattr(self.cms, 'stations') and station_id < len(self.cms.stations):
            station = self.cms.stations[station_id]
            features.extend([
                station.soc,
                station.max_power / 1000.0,  # Normalized
                station.efficiency
            ])
        else:
            features.extend([0.5, 1.0, 0.95])  # Default values
        
        # Grid conditions
        features.extend([
            1.0,    # Grid voltage (pu)
            60.0,   # Grid frequency
            0.8,    # Load factor
            time.time() % 86400 / 86400.0  # Time of day normalized
        ])
        
        # Historical attack success
        recent_success = 0.5  # Would track actual success rate
        features.append(recent_success)
        
        # System stress indicators
        features.extend([
            0.1,    # Voltage deviation
            0.05,   # Frequency deviation
            0.3,    # Load factor
            0.2     # System congestion
        ])
        
        # Pad to fixed size
        while len(features) < 15:
            features.append(0.0)
            
        return torch.tensor(features[:15], dtype=torch.float32)
    
    def _action_to_attack_params(self, action: torch.Tensor) -> Dict:
        """Convert RL action to attack parameters"""
        
        # Map action values [-1,1] to attack parameters
        attack_types = ['demand_increase', 'demand_decrease', 'oscillating_demand', 
                       'voltage_spoofing', 'frequency_spoofing', 'soc_manipulation']
        
        return {
            'type': attack_types[int((action[0] + 1) * 3) % len(attack_types)],
            'magnitude': abs(action[1]) * 2.0 + 0.1,  # 0.1 to 2.1
            'duration': int(abs(action[2]) * 30 + 5),  # 5 to 35 seconds
            'timing_strategy': action[3],
            'stealth_factor': abs(action[4]),  # How subtle the attack should be
            'targets': [int(abs(action[5]) * self.num_stations) % self.num_stations]
        }
    
    def _get_baseline_output(self, station_id: int) -> Dict:
        """Get baseline system output without attacks"""
        
        return {
            'voltage': 400.0,
            'current': 25.0,
            'power': 10.0,
            'soc': 0.5,
            'grid_voltage': 1.0,
            'grid_frequency': 60.0,
            'demand_factor': 1.0,
            'voltage_priority': 0.0,
            'urgency_factor': 1.0
        }
    
    def evaluate_agents(self, num_episodes: int = 100):
        """Evaluate trained agents"""
        if self.use_dqn_sac and self.dqn_sac_trainer:
            return self.dqn_sac_trainer.evaluate_agents(num_episodes)
        else:
            print("⚠️  Legacy evaluation not fully implemented")
            return {'legacy_evaluation': 'placeholder'}
    
    def save_agents(self, save_dir: str = "./models/"):
        """Save trained agents"""
        if self.use_dqn_sac and self.dqn_sac_trainer:
            self.dqn_sac_trainer.save_agents(save_dir)
        else:
            print("⚠️  Legacy agent saving not implemented")
    
    def load_agents(self, save_dir: str = "./models/"):
        """Load trained agents"""
        if self.use_dqn_sac and self.dqn_sac_trainer:
            self.dqn_sac_trainer.load_agents(save_dir)
        else:
            print("⚠️  Legacy agent loading not implemented")

def create_adaptive_rl_system(cms_system, use_dqn_sac: bool = True):
    """Factory function to create adaptive RL attack system"""
    
    trainer = AdaptiveRLAttackTrainer(cms_system, use_dqn_sac=use_dqn_sac)
    
    if use_dqn_sac and DQNSACSecurityEvasionTrainer:
        print("🚀 Created Enhanced DQN/SAC Adaptive RL Attack System")
        print("   - DQN agent for discrete attack decisions")
        print("   - SAC agent for continuous attack parameters")
        print("   - Professional RL algorithms from stable_baselines3")
        print("   - Gym environments for proper RL training")
    else:
        print("🚀 Created Legacy Adaptive RL Attack System")
        print("   - Custom neural network agents")
        print("   - Basic security evasion learning")
    
    print("   - Agents learn to bypass anomaly detection")
    print("   - Rewards based on actual EVCS output changes") 
    print("   - Security evasion feedback loop")
    
    return trainer

def create_dqn_sac_system(cms_system):
    """Direct factory for DQN/SAC system"""
    if DQNSACSecurityEvasionTrainer:
        return DQNSACSecurityEvasionTrainer(cms_system)
    else:
        print(" DQN/SAC system not available. Check imports.")
        return None

if __name__ == "__main__":
    print("🎯 Adaptive RL Evasion System")
    print("This system trains RL agents to learn bypassing security measures")
    print("Run from main simulation to integrate with CMS")
