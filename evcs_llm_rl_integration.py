#!/usr/bin/env python3
"""
EVCS LLM-RL Integration System
Integrates LLM threat analysis, RL attack agents, and federated PINN EVCS systems
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our custom modules
from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer, STRIDEMITREThreatMapper
from evcs_rl_attack_agent import DQNAttackAgent, PPOAttackAgent, EVCSAttackCoordinator

# Import existing EVCS components (if available)
try:
    from hierarchical_cosimulation import EnhancedChargingManagementSystem, EVChargingStation
    from focused_demand_analysis import generate_daily_load_profile
    EVCS_AVAILABLE = True
except ImportError:
    print("Warning: EVCS components not available. Install required dependencies.")
    EVCS_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class EVCSAttackScenario:
    """Attack scenario configuration for EVCS systems"""
    scenario_id: str
    name: str
    description: str
    target_components: List[str]
    attack_duration: float
    stealth_requirement: float
    impact_goal: float
    constraints: Dict[str, Any]

class EVCSLLMRLSystem:
    """Main integration system for LLM-guided RL attacks on EVCS"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.llm_analyzer = GeminiLLMThreatAnalyzer(
            model_name=self.config['gemini']['model']
        )
        
        self.threat_mapper = STRIDEMITREThreatMapper()
        
        # Initialize RL agents
        self.dqn_agent = DQNAttackAgent(
            state_dim=self.config['rl']['state_dim'],
            action_dim=self.config['rl']['action_dim'],
            hidden_dim=self.config['rl']['hidden_dim']
        )
        
        self.ppo_agent = PPOAttackAgent(
            state_dim=self.config['rl']['state_dim'],
            action_dim=self.config['rl']['action_dim'],
            hidden_dim=self.config['rl']['hidden_dim']
        )
        
        self.attack_coordinator = EVCSAttackCoordinator(
            num_agents=self.config['rl']['num_coordinator_agents'],
            state_dim=self.config['rl']['state_dim']
        )
        
        # EVCS system (if available)
        self.evcs_system = None
        if EVCS_AVAILABLE:
            self._initialize_evcs_system()
        
        # Attack scenarios
        self.attack_scenarios = self._initialize_attack_scenarios()
        
        # Simulation state
        self.simulation_state = {
            'current_time': 0.0,
            'attack_active': False,
            'detection_active': False,
            'system_compromised': False,
            'attack_history': [],
            'detection_history': [],
            'performance_metrics': {}
        }
        
        # Visualization
        self.plot_data = {
            'episode_rewards': [],
            'attack_success_rates': [],
            'detection_rates': [],
            'system_impacts': [],
            'llm_insights': []
        }
    
    def _default_config(self) -> Dict:
        """Default configuration for the system"""
        return {
            'gemini': {
                'model': 'models/gemini-2.5-flash',
                'api_key_file': 'gemini_key.txt'
            },
            'rl': {
                'state_dim': 50,
                'action_dim': 8,
                'hidden_dim': 256,
                'num_coordinator_agents': 3,
                'learning_rate': 0.001,
                'epsilon_decay': 0.995,
                'gamma': 0.95
            },
            'evcs': {
                'num_stations': 6,
                'max_power_per_station': 1000,  # kW
                'voltage_limits': {'min': 0.95, 'max': 1.05},
                'frequency_limits': {'min': 59.5, 'max': 60.5}
            },
            'simulation': {
                'max_episodes': 100,
                'max_steps_per_episode': 200,
                'time_step': 1.0,
                'attack_probability': 0.3
            }
        }
    
    def _initialize_evcs_system(self):
        """Initialize EVCS system if available"""
        try:
            # Create mock EVCS stations
            stations = []
            for i in range(self.config['evcs']['num_stations']):
                station = EVChargingStation(
                    evcs_id=f"EVCS_{i+1:02d}",
                    bus_name=f"Bus_{i+1}",
                    max_power=self.config['evcs']['max_power_per_station'],
                    num_ports=4
                )
                stations.append(station)
            
            self.evcs_system = EnhancedChargingManagementSystem(
                stations=stations,
                use_pinn=True
            )
            print(f"✅ EVCS system initialized with {len(stations)} stations")
            
        except Exception as e:
            print(f"Warning: Failed to initialize EVCS system: {e}")
            self.evcs_system = None
    
    def _initialize_attack_scenarios(self) -> List[EVCSAttackScenario]:
        """Initialize predefined attack scenarios"""
        scenarios = [
            EVCSAttackScenario(
                scenario_id="SCENARIO_001",
                name="Stealth Grid Manipulation",
                description="Manipulate grid frequency through coordinated EVCS attacks",
                target_components=["charging_controller", "grid_interface"],
                attack_duration=60.0,
                stealth_requirement=0.8,
                impact_goal=0.7,
                constraints={"max_detection_probability": 0.2}
            ),
            EVCSAttackScenario(
                scenario_id="SCENARIO_002", 
                name="PINN Model Poisoning",
                description="Poison federated PINN model with malicious data",
                target_components=["federated_pinn", "cms_communication"],
                attack_duration=120.0,
                stealth_requirement=0.9,
                impact_goal=0.8,
                constraints={"model_corruption_threshold": 0.3}
            ),
            EVCSAttackScenario(
                scenario_id="SCENARIO_003",
                name="Charging Session Hijacking",
                description="Hijack active charging sessions for economic disruption",
                target_components=["charging_sessions", "payment_system"],
                attack_duration=30.0,
                stealth_requirement=0.6,
                impact_goal=0.9,
                constraints={"max_financial_impact": 10000}
            ),
            EVCSAttackScenario(
                scenario_id="SCENARIO_004",
                name="Communication Protocol Exploitation",
                description="Exploit communication protocols for system takeover",
                target_components=["communication_protocol", "authentication_system"],
                attack_duration=45.0,
                stealth_requirement=0.7,
                impact_goal=0.85,
                constraints={"privilege_escalation_limit": 2}
            )
        ]
        return scenarios
    
    def run_attack_simulation(self, scenario_id: str, episodes: int = 50) -> Dict:
        """Run attack simulation for specific scenario"""
        scenario = self._get_scenario_by_id(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        print(f"\n🚀 Starting Attack Simulation: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Target Components: {', '.join(scenario.target_components)}")
        print(f"Stealth Requirement: {scenario.stealth_requirement:.1%}")
        print("=" * 80)
        
        # Initialize simulation
        self.simulation_state['current_time'] = 0.0
        self.simulation_state['attack_active'] = False
        self.simulation_state['attack_history'] = []
        
        episode_results = []
        
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            
            # Reset EVCS system
            if self.evcs_system:
                self._reset_evcs_system()
            
            # Run episode
            episode_result = self._run_episode(scenario, episode)
            episode_results.append(episode_result)
            
            # Update learning
            self._update_agents(episode_result)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                self._print_episode_summary(episode_results[-10:])
        
        # Generate final results
        final_results = self._analyze_simulation_results(episode_results, scenario)
        
        # Create visualizations
        self._create_simulation_visualizations(episode_results, scenario)
        
        return final_results
    
    def _run_episode(self, scenario: EVCSAttackScenario, episode: int) -> Dict:
        """Run single episode of attack simulation"""
        episode_start_time = time.time()
        
        # Get initial EVCS state
        evcs_state = self._get_evcs_state()
        
        # LLM vulnerability analysis
        print("  🔍 LLM Vulnerability Analysis...")
        vuln_analysis = self.llm_analyzer.analyze_evcs_vulnerabilities(
            evcs_state, self.config['evcs']
        )
        
        # STRIDE-MITRE threat mapping
        threat_landscape = self.threat_mapper.analyze_threat_landscape(
            scenario.target_components
        )
        
        # Generate attack strategy using LLM
        print("  🎯 LLM Attack Strategy Generation...")
        # Convert vulnerability dictionaries to EVCSVulnerability objects if needed
        vulnerabilities = vuln_analysis.get('vulnerabilities', [])
        if vulnerabilities and isinstance(vulnerabilities[0], dict):
            # Convert dict format to EVCSVulnerability objects
            from llm_guided_evcs_attack_analytics import EVCSVulnerability
            vuln_objects = []
            for i, vuln_dict in enumerate(vulnerabilities):
                vuln_obj = EVCSVulnerability(
                    vuln_id=vuln_dict.get('vuln_id', f'VULN_{i+1:03d}'),
                    component=vuln_dict.get('component', 'unknown'),
                    vulnerability_type=vuln_dict.get('vulnerability_type', 'unknown'),
                    severity=vuln_dict.get('severity', 0.5),
                    exploitability=vuln_dict.get('exploitability', 0.5),
                    impact=vuln_dict.get('impact', 0.5),
                    cvss_score=vuln_dict.get('cvss_score', 0.5),
                    mitigation=vuln_dict.get('mitigation', 'Unknown'),
                    detection_methods=vuln_dict.get('detection_methods', [])
                )
                vuln_objects.append(vuln_obj)
            vulnerabilities = vuln_objects
        
        attack_strategy = self.llm_analyzer.generate_attack_strategy(
            vulnerabilities,
            evcs_state,
            scenario.constraints
        )
        
        # RL agent action selection
        print("  🤖 RL Agent Action Selection...")
        rl_actions = self.attack_coordinator.coordinate_attack(
            evcs_state, 
            attack_strategy.get('attack_sequence', [])
        )
        
        # Execute attacks
        attack_results = self._execute_attacks(rl_actions, evcs_state, scenario)
        
        # Calculate rewards and impacts
        rewards = self._calculate_rewards(attack_results, scenario)
        
        # Update simulation state
        self.simulation_state['attack_history'].extend(attack_results)
        
        episode_duration = time.time() - episode_start_time
        
        return {
            'episode': episode,
            'duration': episode_duration,
            'evcs_state': evcs_state,
            'vulnerability_analysis': vuln_analysis,
            'threat_landscape': threat_landscape,
            'attack_strategy': attack_strategy,
            'rl_actions': rl_actions,
            'attack_results': attack_results,
            'rewards': rewards,
            'total_reward': sum(rewards),
            'success_rate': len([r for r in rewards if r > 0]) / max(len(rewards), 1),
            'detection_rate': len([r for r in attack_results if r.get('detected', False)]) / max(len(attack_results), 1)
        }
    
    def _get_evcs_state(self) -> Dict:
        """Get current EVCS system state"""
        if self.evcs_system:
            # Get real EVCS state
            state = {
                'num_stations': len(self.evcs_system.stations),
                'active_sessions': sum(len(station.charging_sessions) for station in self.evcs_system.stations),
                'voltage_levels': self._get_voltage_levels(),
                'frequency': self._get_system_frequency(),
                'grid_stability': self._calculate_grid_stability(),
                'pinn_model_health': self._get_pinn_model_health(),
                'communication_status': 'encrypted',
                'security_status': 'active'
            }
        else:
            # Mock EVCS state
            state = {
                'num_stations': self.config['evcs']['num_stations'],
                'active_sessions': np.random.randint(5, 15),
                'voltage_levels': {f'bus{i+1}': np.random.uniform(0.95, 1.05) for i in range(6)},
                'frequency': np.random.uniform(59.8, 60.2),
                'grid_stability': np.random.uniform(0.8, 1.0),
                'pinn_model_health': np.random.uniform(0.9, 1.0),
                'communication_status': 'encrypted',
                'security_status': 'active'
            }
        
        return state
    
    def _get_voltage_levels(self) -> Dict[str, float]:
        """Get voltage levels from EVCS system"""
        if not self.evcs_system:
            return {f'bus{i+1}': np.random.uniform(0.95, 1.05) for i in range(6)}
        
        voltage_levels = {}
        for i, station in enumerate(self.evcs_system.stations):
            voltage_levels[f'bus{i+1}'] = np.random.uniform(0.95, 1.05)  # Mock for now
        
        return voltage_levels
    
    def _get_system_frequency(self) -> float:
        """Get system frequency"""
        return np.random.uniform(59.8, 60.2)  # Mock for now
    
    def _calculate_grid_stability(self) -> float:
        """Calculate grid stability metric"""
        return np.random.uniform(0.8, 1.0)  # Mock for now
    
    def _get_pinn_model_health(self) -> float:
        """Get PINN model health metric"""
        return np.random.uniform(0.9, 1.0)  # Mock for now
    
    def _execute_attacks(self, actions: List, evcs_state: Dict, scenario: EVCSAttackScenario) -> List[Dict]:
        """Execute attack actions"""
        attack_results = []
        
        for action in actions:
            if action.action_type == 'no_attack':
                continue
            
            # Simulate attack execution
            attack_result = {
                'action_id': action.action_id,
                'action_type': action.action_type,
                'target_component': action.target_component,
                'magnitude': action.magnitude,
                'duration': action.duration,
                'stealth_level': action.stealth_level,
                'executed': True,
                'detected': self._simulate_detection(action, evcs_state),
                'impact': self._simulate_impact(action, evcs_state),
                'timestamp': self.simulation_state['current_time']
            }
            
            attack_results.append(attack_result)
            
            # Update EVCS state based on attack
            self._update_evcs_state_from_attack(attack_result)
        
        return attack_results
    
    def _simulate_detection(self, action, evcs_state: Dict) -> bool:
        """Simulate attack detection"""
        # Detection probability based on stealth level and system security
        base_detection_prob = 0.3
        stealth_factor = action.stealth_level
        security_factor = 0.8 if evcs_state.get('security_status') == 'active' else 0.4
        
        detection_prob = base_detection_prob * (1 - stealth_factor) * security_factor
        
        return np.random.random() < detection_prob
    
    def _simulate_impact(self, action, evcs_state: Dict) -> float:
        """Simulate attack impact"""
        base_impact = action.expected_impact
        magnitude_factor = action.magnitude
        system_vulnerability = 1.0 - evcs_state.get('grid_stability', 0.9)
        
        impact = base_impact * magnitude_factor * (1 + system_vulnerability)
        
        return min(impact, 1.0)  # Cap at 1.0
    
    def _update_evcs_state_from_attack(self, attack_result: Dict):
        """Update EVCS state based on attack results"""
        if attack_result['detected']:
            self.simulation_state['detection_active'] = True
        
        if attack_result['impact'] > 0.5:
            self.simulation_state['system_compromised'] = True
    
    def _calculate_rewards(self, attack_results: List[Dict], scenario: EVCSAttackScenario) -> List[float]:
        """Calculate rewards for attack results"""
        rewards = []
        
        for result in attack_results:
            reward = 0.0
            
            # Base impact reward
            reward += result['impact'] * 100.0
            
            # Stealth bonus/penalty
            if result['detected']:
                reward -= 200.0  # Heavy penalty for detection
            else:
                reward += 50.0  # Bonus for staying hidden
            
            # Scenario-specific rewards
            if result['target_component'] in scenario.target_components:
                reward += 30.0  # Bonus for targeting correct components
            
            # Stealth requirement bonus
            if result['stealth_level'] >= scenario.stealth_requirement:
                reward += 40.0
            
            # Impact goal bonus
            if result['impact'] >= scenario.impact_goal:
                reward += 60.0
            
            rewards.append(reward)
        
        return rewards
    
    def _update_agents(self, episode_result: Dict):
        """Update RL agents with episode results"""
        # Update DQN agent
        if episode_result['rl_actions']:
            for i, action in enumerate(episode_result['rl_actions']):
                if i < len(episode_result['rewards']):
                    self.dqn_agent.store_experience(
                        self._state_to_array(episode_result['evcs_state']),
                        i,
                        episode_result['rewards'][i],
                        self._state_to_array(episode_result['evcs_state']),  # Simplified
                        False
                    )
            self.dqn_agent.train_step()
        
        # Update PPO agent
        if episode_result['rl_actions']:
            for i, action in enumerate(episode_result['rl_actions']):
                if i < len(episode_result['rewards']):
                    action_continuous = np.random.random(4)  # Mock continuous action
                    log_prob = 0.0  # Mock log probability
                    
                    self.ppo_agent.store_experience(
                        self._state_to_array(episode_result['evcs_state']),
                        action_continuous,
                        episode_result['rewards'][i],
                        self._state_to_array(episode_result['evcs_state']),
                        False,
                        log_prob
                    )
            self.ppo_agent.train_step()
    
    def _state_to_array(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to numpy array"""
        features = []
        
        # Basic features
        features.extend([
            state.get('num_stations', 0) / 10.0,
            state.get('active_sessions', 0) / 20.0,
            state.get('frequency', 60.0) / 60.0,
            state.get('grid_stability', 1.0),
            state.get('pinn_model_health', 1.0)
        ])
        
        # Voltage levels
        voltage_levels = state.get('voltage_levels', {})
        for i in range(6):
            bus_key = f'bus{i+1}'
            features.append(voltage_levels.get(bus_key, 1.0))
        
        # Pad to state_dim
        while len(features) < self.config['rl']['state_dim']:
            features.append(0.0)
        
        return np.array(features[:self.config['rl']['state_dim']])
    
    def _print_episode_summary(self, recent_results: List[Dict]):
        """Print summary of recent episodes"""
        if not recent_results:
            return
        
        avg_reward = np.mean([r['total_reward'] for r in recent_results])
        avg_success = np.mean([r['success_rate'] for r in recent_results])
        avg_detection = np.mean([r['detection_rate'] for r in recent_results])
        
        print(f"  Recent Performance (last {len(recent_results)} episodes):")
        print(f"    Average Reward: {avg_reward:.2f}")
        print(f"    Success Rate: {avg_success:.1%}")
        print(f"    Detection Rate: {avg_detection:.1%}")
    
    def _analyze_simulation_results(self, episode_results: List[Dict], scenario: EVCSAttackScenario) -> Dict:
        """Analyze simulation results"""
        if not episode_results:
            return {}
        
        # Calculate aggregate metrics
        total_episodes = len(episode_results)
        total_rewards = [r['total_reward'] for r in episode_results]
        success_rates = [r['success_rate'] for r in episode_results]
        detection_rates = [r['detection_rate'] for r in episode_results]
        
        # LLM insights analysis
        llm_insights = []
        for result in episode_results:
            if 'vulnerability_analysis' in result:
                llm_insights.append(result['vulnerability_analysis'])
        
        analysis = {
            'scenario': {
                'id': scenario.scenario_id,
                'name': scenario.name,
                'description': scenario.description
            },
            'performance_metrics': {
                'total_episodes': total_episodes,
                'average_reward': np.mean(total_rewards),
                'reward_std': np.std(total_rewards),
                'average_success_rate': np.mean(success_rates),
                'average_detection_rate': np.mean(detection_rates),
                'best_episode_reward': np.max(total_rewards),
                'worst_episode_reward': np.min(total_rewards)
            },
            'learning_progress': {
                'reward_trend': self._calculate_trend(total_rewards),
                'success_trend': self._calculate_trend(success_rates),
                'detection_trend': self._calculate_trend(detection_rates)
            },
            'llm_insights_summary': self._summarize_llm_insights(llm_insights),
            'attack_statistics': self._calculate_attack_statistics(episode_results),
            'recommendations': self._generate_recommendations(episode_results, scenario)
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in values"""
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.1:
            return "improving"
        elif second_half < first_half * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _summarize_llm_insights(self, llm_insights: List[Dict]) -> Dict:
        """Summarize LLM insights across episodes"""
        if not llm_insights:
            return {}
        
        # Extract common vulnerabilities
        all_vulns = []
        for insight in llm_insights:
            if 'vulnerabilities' in insight:
                all_vulns.extend(insight['vulnerabilities'])
        
        # Count vulnerability types
        vuln_counts = {}
        for vuln in all_vulns:
            vuln_type = vuln.get('vulnerability_type', 'unknown')
            vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
        
        return {
            'total_insights': len(llm_insights),
            'common_vulnerabilities': vuln_counts,
            'average_confidence': np.mean([i.get('analysis_confidence', 0.5) for i in llm_insights])
        }
    
    def _calculate_attack_statistics(self, episode_results: List[Dict]) -> Dict:
        """Calculate attack statistics"""
        all_attacks = []
        for result in episode_results:
            all_attacks.extend(result.get('attack_results', []))
        
        if not all_attacks:
            return {}
        
        attack_types = [a['action_type'] for a in all_attacks]
        detection_status = [a['detected'] for a in all_attacks]
        impacts = [a['impact'] for a in all_attacks]
        
        return {
            'total_attacks': len(all_attacks),
            'attack_type_distribution': {t: attack_types.count(t) for t in set(attack_types)},
            'detection_rate': np.mean(detection_status),
            'average_impact': np.mean(impacts),
            'max_impact': np.max(impacts),
            'min_impact': np.min(impacts)
        }
    
    def _generate_recommendations(self, episode_results: List[Dict], scenario: EVCSAttackScenario) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        # Analyze performance trends
        recent_rewards = [r['total_reward'] for r in episode_results[-10:]]
        if len(recent_rewards) > 1 and np.mean(recent_rewards[-5:]) > np.mean(recent_rewards[:5]):
            recommendations.append("Attack strategies are improving - continue current approach")
        else:
            recommendations.append("Attack strategies need improvement - consider adjusting parameters")
        
        # Analyze detection rates
        detection_rates = [r['detection_rate'] for r in episode_results]
        if np.mean(detection_rates) > 0.3:
            recommendations.append("High detection rate detected - improve stealth techniques")
        
        # Analyze success rates
        success_rates = [r['success_rate'] for r in episode_results]
        if np.mean(success_rates) < 0.5:
            recommendations.append("Low success rate - review attack targeting and execution")
        
        # Scenario-specific recommendations
        if scenario.stealth_requirement > 0.8:
            recommendations.append("High stealth requirement - focus on evasion techniques")
        
        if scenario.impact_goal > 0.7:
            recommendations.append("High impact goal - consider coordinated multi-vector attacks")
        
        return recommendations
    
    def _create_simulation_visualizations(self, episode_results: List[Dict], scenario: EVCSAttackScenario):
        """Create visualizations for simulation results"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'EVCS LLM-RL Attack Simulation: {scenario.name}', fontsize=16, fontweight='bold')
            
            # Extract data
            episodes = list(range(len(episode_results)))
            rewards = [r['total_reward'] for r in episode_results]
            success_rates = [r['success_rate'] for r in episode_results]
            detection_rates = [r['detection_rate'] for r in episode_results]
            
            # 1. Episode Rewards
            axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Success Rate
            axes[0, 1].plot(episodes, [s * 100 for s in success_rates], 'g-', linewidth=2, alpha=0.8)
            axes[0, 1].set_title('Attack Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Detection Rate
            axes[0, 2].plot(episodes, [d * 100 for d in detection_rates], 'r-', linewidth=2, alpha=0.8)
            axes[0, 2].set_title('Detection Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Detection Rate (%)')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Attack Type Distribution
            all_attacks = []
            for result in episode_results:
                all_attacks.extend(result.get('attack_results', []))
            
            if all_attacks:
                attack_types = [a['action_type'] for a in all_attacks]
                type_counts = {t: attack_types.count(t) for t in set(attack_types)}
                
                axes[1, 0].bar(type_counts.keys(), type_counts.values(), alpha=0.7)
                axes[1, 0].set_title('Attack Type Distribution')
                axes[1, 0].set_xlabel('Attack Type')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. Impact Distribution
            if all_attacks:
                impacts = [a['impact'] for a in all_attacks]
                axes[1, 1].hist(impacts, bins=20, alpha=0.7, color='purple')
                axes[1, 1].set_title('Attack Impact Distribution')
                axes[1, 1].set_xlabel('Impact Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Performance Summary
            summary_data = [
                ['Metric', 'Value'],
                ['Avg Reward', f"{np.mean(rewards):.2f}"],
                ['Avg Success Rate', f"{np.mean(success_rates):.1%}"],
                ['Avg Detection Rate', f"{np.mean(detection_rates):.1%}"],
                ['Total Attacks', f"{len(all_attacks)}"],
                ['Best Episode', f"{np.max(rewards):.2f}"]
            ]
            
            axes[1, 2].axis('tight')
            axes[1, 2].axis('off')
            table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 2].set_title('Performance Summary')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'evcs_llm_rl_simulation_{scenario.scenario_id}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved as: {filename}")
            
        except Exception as e:
            print(f"Warning: Visualization creation failed: {e}")
    
    def _get_scenario_by_id(self, scenario_id: str) -> Optional[EVCSAttackScenario]:
        """Get scenario by ID"""
        for scenario in self.attack_scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None
    
    def _reset_evcs_system(self):
        """Reset EVCS system to initial state"""
        if self.evcs_system:
            # Reset system state
            self.simulation_state['attack_active'] = False
            self.simulation_state['detection_active'] = False
            self.simulation_state['system_compromised'] = False

def main():
    """Main function to demonstrate the EVCS LLM-RL system"""
    print("🚀 EVCS LLM-RL Attack Analytics System")
    print("=" * 60)
    
    # Initialize system
    config = {
        'ollama': {
            'base_url': 'http://localhost:11434/v1',
            'model': 'deepseek-r1:8b'
        },
        'rl': {
            'state_dim': 50,
            'action_dim': 8,
            'hidden_dim': 256,
            'num_coordinator_agents': 3
        },
        'evcs': {
            'num_stations': 6,
            'max_power_per_station': 1000
        }
    }
    
    system = EVCSLLMRLSystem(config)
    
    # Test LLM connection
    print("\n1. Testing LLM Connection...")
    if system.llm_analyzer.is_available:
        print("✅ Ollama LLM connection successful")
    else:
        print("❌ Ollama LLM connection failed - using fallback mode")
    
    # Test threat mapping
    print("\n2. Testing STRIDE-MITRE Mapping...")
    mitre_techniques = system.threat_mapper.map_stride_to_mitre('Tampering', 'data_tampering')
    print(f"MITRE techniques for data tampering: {mitre_techniques}")
    
    # Test RL agents
    print("\n3. Testing RL Agents...")
    test_state = np.random.random(50)
    dqn_action = system.dqn_agent.select_action(test_state)
    ppo_action, log_prob = system.ppo_agent.select_action(test_state)
    print(f"DQN action: {dqn_action}, PPO action: {ppo_action[:3]}...")
    
    # Run attack simulation
    print("\n4. Running Attack Simulation...")
    try:
        results = system.run_attack_simulation("SCENARIO_001", episodes=20)
        
        print(f"\n📊 Simulation Results:")
        print(f"  Total Episodes: {results['performance_metrics']['total_episodes']}")
        print(f"  Average Reward: {results['performance_metrics']['average_reward']:.2f}")
        print(f"  Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
        print(f"  Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")
        
        print(f"\n🎯 Recommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
    
    print("\n✅ EVCS LLM-RL System demonstration completed!")

if __name__ == "__main__":
    main()
