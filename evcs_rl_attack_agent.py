#!/usr/bin/env python3
"""
RL Attack Agent for EVCS Systems
Implements Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) for adaptive attack strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Experience buffer for RL
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class AttackAction:
    """Attack action for EVCS systems"""
    action_id: str
    action_type: str
    target_component: str
    magnitude: float
    duration: float
    stealth_level: float
    prerequisites: List[str]
    expected_impact: float

class DQNAttackAgent:
    """Deep Q-Network agent for EVCS attack strategies"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # RL parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.tau = 0.005  # Soft update parameter
        
        # Attack statistics
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'detected_attacks': 0,
            'cumulative_reward': 0.0,
            'attack_history': [],
            'reward_history': [],
            'detection_history': []
        }
        
        # EVCS-specific action space
        self.attack_actions = self._initialize_attack_actions()
    
    def _build_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """Build DQN network architecture"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def _initialize_attack_actions(self) -> List[AttackAction]:
        """Initialize EVCS-specific attack actions"""
        actions = [
            # Communication attacks
            AttackAction(
                action_id="COMM_SPOOF_001",
                action_type="communication_spoofing",
                target_component="charging_controller",
                magnitude=0.1,
                duration=5.0,
                stealth_level=0.7,
                prerequisites=["network_access"],
                expected_impact=0.6
            ),
            AttackAction(
                action_id="COMM_INJECT_001", 
                action_type="data_injection",
                target_component="cms_communication",
                magnitude=0.2,
                duration=3.0,
                stealth_level=0.5,
                prerequisites=["protocol_knowledge"],
                expected_impact=0.8
            ),
            
            # Power electronics attacks
            AttackAction(
                action_id="POWER_MANIP_001",
                action_type="power_manipulation",
                target_component="charging_power",
                magnitude=0.3,
                duration=10.0,
                stealth_level=0.4,
                prerequisites=["power_electronics_access"],
                expected_impact=0.9
            ),
            AttackAction(
                action_id="VOLTAGE_INJECT_001",
                action_type="voltage_injection",
                target_component="grid_interface",
                magnitude=0.15,
                duration=8.0,
                stealth_level=0.6,
                prerequisites=["grid_access"],
                expected_impact=0.7
            ),
            
            # PINN model attacks
            AttackAction(
                action_id="PINN_POISON_001",
                action_type="model_poisoning",
                target_component="federated_pinn",
                magnitude=0.25,
                duration=15.0,
                stealth_level=0.8,
                prerequisites=["model_access"],
                expected_impact=0.85
            ),
            AttackAction(
                action_id="PINN_EVASION_001",
                action_type="evasion_attack",
                target_component="anomaly_detection",
                magnitude=0.1,
                duration=12.0,
                stealth_level=0.9,
                prerequisites=["detection_knowledge"],
                expected_impact=0.4
            ),
            
            # Coordinated attacks
            AttackAction(
                action_id="COORD_MULTI_001",
                action_type="coordinated_attack",
                target_component="multiple_stations",
                magnitude=0.2,
                duration=20.0,
                stealth_level=0.6,
                prerequisites=["multi_station_access"],
                expected_impact=0.95
            ),
            
            # No attack
            AttackAction(
                action_id="NO_ATTACK_001",
                action_type="no_attack",
                target_component="none",
                magnitude=0.0,
                duration=0.0,
                stealth_level=1.0,
                prerequisites=[],
                expected_impact=0.0
            )
        ]
        
        return actions
    
    def select_action(self, state: np.ndarray, threat_recommendations: List[Dict] = None) -> int:
        """Select action using epsilon-greedy policy with threat intelligence"""
        if np.random.random() < self.epsilon:
            # Exploration: use threat recommendations or random
            if threat_recommendations and np.random.random() < 0.7:
                return self._select_recommended_action(threat_recommendations)
            else:
                return random.randint(0, len(self.attack_actions) - 1)
        else:
            # Exploitation: use Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def _select_recommended_action(self, threat_recommendations: List[Dict]) -> int:
        """Select action based on threat recommendations"""
        if not threat_recommendations:
            return random.randint(0, len(self.attack_actions) - 1)
        
        # Find best matching action based on recommendations
        best_action_idx = 0
        best_score = 0.0
        
        for i, action in enumerate(self.attack_actions):
            score = 0.0
            for rec in threat_recommendations:
                # Match by attack type
                if rec.get('attack_type', '').lower() in action.action_type.lower():
                    score += rec.get('impact_score', 0.5) * 0.5
                
                # Match by target component
                if rec.get('target_component', '').lower() in action.target_component.lower():
                    score += rec.get('stealth_score', 0.5) * 0.3
                
                # Match by stealth level
                if abs(rec.get('stealth_score', 0.5) - action.stealth_level) < 0.2:
                    score += 0.2
            
            if score > best_score:
                best_score = score
                best_action_idx = i
        
        return best_action_idx
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def train_step(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _soft_update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def calculate_reward(self, state: Dict, action: AttackAction, next_state: Dict, 
                        detected: bool, impact: float) -> float:
        """Calculate reward for attack action"""
        reward = 0.0
        
        # Base impact reward
        reward += impact * 100.0
        
        # Stealth bonus/penalty
        if detected:
            reward -= 200.0  # Heavy penalty for detection
            self.attack_stats['detected_attacks'] += 1
            self.attack_stats['detection_history'].append(1)
        else:
            reward += 50.0  # Bonus for staying hidden
            if action.action_type != 'no_attack':
                self.attack_stats['successful_attacks'] += 1
            self.attack_stats['detection_history'].append(0)
        
        # EVCS-specific rewards
        if 'grid_instability' in next_state and next_state['grid_instability'] > 0.1:
            reward += 150.0  # High reward for grid impact
        
        if 'charging_disruption' in next_state and next_state['charging_disruption'] > 0.2:
            reward += 100.0  # Reward for disrupting charging
        
        if 'pinn_model_corruption' in next_state and next_state['pinn_model_corruption'] > 0.1:
            reward += 120.0  # Reward for model poisoning
        
        # Stealth level bonus
        reward += action.stealth_level * 30.0
        
        # Duration penalty (longer attacks are riskier)
        reward -= action.duration * 2.0
        
        # Update statistics
        self.attack_stats['total_attacks'] += 1
        self.attack_stats['cumulative_reward'] += reward
        self.attack_stats['attack_history'].append(action.action_id)
        self.attack_stats['reward_history'].append(reward)
        
        return reward
    
    def get_action_by_id(self, action_id: str) -> Optional[AttackAction]:
        """Get attack action by ID"""
        for action in self.attack_actions:
            if action.action_id == action_id:
                return action
        return None
    
    def get_attack_statistics(self) -> Dict:
        """Get attack statistics"""
        stats = self.attack_stats.copy()
        
        if stats['total_attacks'] > 0:
            stats['success_rate'] = stats['successful_attacks'] / stats['total_attacks']
            stats['detection_rate'] = stats['detected_attacks'] / stats['total_attacks']
            stats['average_reward'] = stats['cumulative_reward'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
            stats['detection_rate'] = 0.0
            stats['average_reward'] = 0.0
        
        return stats

class PPOAttackAgent:
    """Proximal Policy Optimization agent for continuous attack strategies"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor and Critic networks
        self.actor = self._build_actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = self._build_critic(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.lam = 0.95
        
        # Experience buffer
        self.memory = []
        self.batch_size = 64
        self.update_epochs = 10
        
        # Attack statistics
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'detected_attacks': 0,
            'cumulative_reward': 0.0,
            'policy_entropy': 0.0
        }
    
    def _build_actor(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Continuous actions in [-1, 1]
        )
    
    def _build_critic(self, state_dim: int, hidden_dim: int) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean = self.actor(state_tensor)
            # Add noise for exploration
            action_std = 0.1
            action = action_mean + torch.randn_like(action_mean) * action_std
            action = torch.clamp(action, -1.0, 1.0)
            
            # Calculate log probability
            log_prob = -0.5 * ((action - action_mean) / action_std).pow(2).sum(1)
        
        return action.cpu().numpy()[0], log_prob.item()
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float):
        """Store experience for PPO training"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def train_step(self):
        """Train PPO agent"""
        if len(self.memory) < self.batch_size:
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor([e['state'] for e in self.memory]).to(self.device)
        actions = torch.FloatTensor([e['action'] for e in self.memory]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in self.memory]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([e['log_prob'] for e in self.memory]).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, dones)
        advantages = self._calculate_advantages(states, returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training
        for _ in range(self.update_epochs):
            # Get current policy
            action_means = self.actor(states)
            values = self.critic(states).squeeze()
            
            # Calculate new log probabilities
            action_std = 0.1
            new_log_probs = -0.5 * ((actions - action_means) / action_std).pow(2).sum(1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(values, returns)
            
            # Calculate entropy loss
            entropy_loss = -new_log_probs.mean()
            
            # Total loss
            total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # Clear memory
        self.memory.clear()
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _calculate_advantages(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using GAE"""
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = returns - values
        
        return advantages

class EVCSAttackCoordinator:
    """Coordinates multiple attack agents for sophisticated EVCS attacks"""
    
    def __init__(self, num_agents: int = 3, state_dim: int = 50):
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Initialize multiple DQN agents
        self.agents = [
            DQNAttackAgent(state_dim, 8, hidden_dim=256) 
            for _ in range(num_agents)
        ]
        
        # Coordination parameters
        self.coordination_threshold = 0.7
        self.attack_synchronization = True
        self.shared_rewards = True
        
        # Global statistics
        self.global_stats = {
            'coordinated_attacks': 0,
            'total_attacks': 0,
            'success_rate': 0.0,
            'detection_rate': 0.0
        }
    
    def coordinate_attack(self, evcs_state: Dict, threat_recommendations: List[Dict]) -> List[AttackAction]:
        """Coordinate attacks across multiple agents"""
        # Get individual agent actions
        agent_actions = []
        for i, agent in enumerate(self.agents):
            action_idx = agent.select_action(
                self._state_to_array(evcs_state), 
                threat_recommendations
            )
            action = agent.get_action_by_id(agent.attack_actions[action_idx].action_id)
            agent_actions.append(action)
        
        # Check for coordination opportunities
        if self._should_coordinate(agent_actions):
            coordinated_actions = self._create_coordinated_attack(agent_actions)
            self.global_stats['coordinated_attacks'] += 1
            return coordinated_actions
        else:
            return agent_actions
    
    def _should_coordinate(self, actions: List[AttackAction]) -> bool:
        """Determine if actions should be coordinated"""
        if not self.attack_synchronization:
            return False
        
        # Check if multiple agents are targeting similar components
        target_components = [action.target_component for action in actions if action.action_type != 'no_attack']
        if len(set(target_components)) < len(target_components):
            return True
        
        # Check if actions have complementary effects
        high_impact_actions = [action for action in actions if action.expected_impact > 0.7]
        if len(high_impact_actions) >= 2:
            return True
        
        return False
    
    def _create_coordinated_attack(self, actions: List[AttackAction]) -> List[AttackAction]:
        """Create coordinated attack sequence"""
        # Sort actions by expected impact
        sorted_actions = sorted(actions, key=lambda x: x.expected_impact, reverse=True)
        
        # Create coordinated sequence
        coordinated = []
        for i, action in enumerate(sorted_actions):
            if action.action_type != 'no_attack':
                # Modify timing for coordination
                coordinated_action = AttackAction(
                    action_id=f"COORD_{action.action_id}",
                    action_type=action.action_type,
                    target_component=action.target_component,
                    magnitude=action.magnitude * 1.2,  # Increase magnitude
                    duration=action.duration * 0.8,    # Reduce duration
                    stealth_level=action.stealth_level * 0.9,  # Slightly reduce stealth
                    prerequisites=action.prerequisites,
                    expected_impact=action.expected_impact * 1.3  # Increase expected impact
                )
                coordinated.append(coordinated_action)
        
        return coordinated
    
    def _state_to_array(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to numpy array"""
        # Extract relevant features from EVCS state
        features = []
        
        # Basic system features
        features.extend([
            state.get('num_stations', 0) / 10.0,  # Normalize
            state.get('active_sessions', 0) / 20.0,
            state.get('frequency', 60.0) / 60.0,
            state.get('grid_stability', 1.0),
            state.get('pinn_model_health', 1.0)
        ])
        
        # Voltage levels
        voltage_levels = state.get('voltage_levels', {})
        for i in range(5):  # Max 5 voltage measurements
            bus_key = f'bus{i+1}'
            features.append(voltage_levels.get(bus_key, 1.0))
        
        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def update_agents(self, rewards: List[float], states: List[Dict], 
                     next_states: List[Dict], dones: List[bool]):
        """Update all agents with experience"""
        for i, agent in enumerate(self.agents):
            if i < len(rewards):
                agent.store_experience(
                    self._state_to_array(states[i]),
                    i,  # Action index
                    rewards[i],
                    self._state_to_array(next_states[i]),
                    dones[i]
                )
                agent.train_step()
        
        # Update global statistics
        self.global_stats['total_attacks'] += len(rewards)
        if rewards:
            self.global_stats['success_rate'] = np.mean([r > 0 for r in rewards])
            self.global_stats['detection_rate'] = np.mean([r < -100 for r in rewards])

if __name__ == "__main__":
    # Test the RL attack agents
    print("Testing EVCS RL Attack Agents")
    print("=" * 50)
    
    # Test DQN agent
    print("\n1. Testing DQN Attack Agent...")
    dqn_agent = DQNAttackAgent(state_dim=20, action_dim=8)
    
    # Test action selection
    test_state = np.random.random(20)
    action_idx = dqn_agent.select_action(test_state)
    action = dqn_agent.attack_actions[action_idx]
    print(f"Selected action: {action.action_id} - {action.action_type}")
    
    # Test reward calculation
    test_state_dict = {'grid_instability': 0.2, 'charging_disruption': 0.3}
    next_state_dict = {'grid_instability': 0.4, 'charging_disruption': 0.5}
    reward = dqn_agent.calculate_reward(test_state_dict, action, next_state_dict, False, 0.8)
    print(f"Calculated reward: {reward:.2f}")
    
    # Test PPO agent
    print("\n2. Testing PPO Attack Agent...")
    ppo_agent = PPOAttackAgent(state_dim=20, action_dim=4)
    
    action_continuous, log_prob = ppo_agent.select_action(test_state)
    print(f"PPO action: {action_continuous}, log_prob: {log_prob:.3f}")
    
    # Test coordinator
    print("\n3. Testing Attack Coordinator...")
    coordinator = EVCSAttackCoordinator(num_agents=3, state_dim=20)
    
    test_evcs_state = {
        'num_stations': 6,
        'active_sessions': 12,
        'frequency': 59.8,
        'grid_stability': 0.9,
        'pinn_model_health': 0.95,
        'voltage_levels': {'bus1': 0.98, 'bus2': 1.02, 'bus3': 0.99}
    }
    
    coordinated_actions = coordinator.coordinate_attack(test_evcs_state, [])
    print(f"Coordinated {len(coordinated_actions)} attack actions")
    
    print("\n✅ EVCS RL Attack Agents initialized successfully!")
