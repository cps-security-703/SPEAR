#!/usr/bin/env python3
"""
Federated PINN LLM-RL Integration for EVCS CMS
Integrates the LLM-guided RL attack analytics with your existing federated PINN system
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

# Import our LLM-RL components
from evcs_llm_rl_integration import EVCSLLMRLSystem, EVCSAttackScenario
from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer, STRIDEMITREThreatMapper
from evcs_rl_attack_agent import DQNAttackAgent, EVCSAttackCoordinator

# Import existing EVCS components
try:
    from hierarchical_cosimulation import EnhancedChargingManagementSystem, EVChargingStation
    from focused_demand_analysis import generate_daily_load_profile
    EVCS_AVAILABLE = True
except ImportError:
    print("Warning: EVCS components not available. Install required dependencies.")
    EVCS_AVAILABLE = False

# Import federated PINN components
try:
    from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
    FEDERATED_PINN_AVAILABLE = True
except ImportError:
    print("Warning: Federated PINN components not available.")
    FEDERATED_PINN_AVAILABLE = False

warnings.filterwarnings('ignore')

class FederatedPINNLLMRLSystem:
    """Enhanced system integrating federated PINN with LLM-guided RL attacks"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize core components
        self.llm_analyzer = GeminiLLMThreatAnalyzer(
            model_name=self.config['gemini']['model']
        )
        
        self.threat_mapper = STRIDEMITREThreatMapper()
        
        # Initialize federated PINN system
        self.federated_pinn = None
        if FEDERATED_PINN_AVAILABLE:
            self._initialize_federated_pinn()
        
        # Initialize EVCS system
        self.evcs_system = None
        if EVCS_AVAILABLE:
            self._initialize_evcs_system()
        
        # Initialize RL attack agents
        self._initialize_rl_agents()
        
        # Attack scenarios specific to federated PINN
        self.attack_scenarios = self._initialize_federated_attack_scenarios()
        
        # Simulation state
        self.simulation_state = {
            'current_time': 0.0,
            'federated_round': 0,
            'attack_active': False,
            'pinn_compromised': False,
            'attack_history': [],
            'federated_metrics': {},
            'llm_insights_history': []
        }
    
    def _default_config(self) -> Dict:
        """Default configuration for the federated PINN LLM-RL system"""
        return {
            'gemini': {
                'model': 'models/gemini-2.5-flash',
                'api_key_file': 'gemini_key.txt'
            },
            'federated_pinn': {
                'num_distribution_systems': 6,
                'local_epochs': 100,
                'global_rounds': 10,
                'aggregation_method': 'fedavg',
                'model_path': 'federated_models'
            },
            'rl': {
                'state_dim': 60,  # Increased for federated context
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
    
    def _initialize_federated_pinn(self):
        """Initialize federated PINN system"""
        try:
            federated_config = FederatedPINNConfig(
                num_distribution_systems=self.config['federated_pinn']['num_distribution_systems'],
                local_epochs=self.config['federated_pinn']['local_epochs'],
                global_rounds=self.config['federated_pinn']['global_rounds'],
                aggregation_method=self.config['federated_pinn']['aggregation_method']
            )
            
            self.federated_pinn = FederatedPINNManager(federated_config)
            
            # Try to load existing models
            success = self.federated_pinn.load_federated_models(
                self.config['federated_pinn']['model_path']
            )
            
            if success:
                print("✅ Federated PINN models loaded successfully")
            else:
                print("⚠️  No existing federated models found - will train new ones")
                
        except Exception as e:
            print(f"❌ Failed to initialize federated PINN: {e}")
            self.federated_pinn = None
    
    def _initialize_evcs_system(self):
        """Initialize EVCS system with federated PINN integration"""
        try:
            # Create EVCS stations
            stations = []
            for i in range(self.config['evcs']['num_stations']):
                station = EVChargingStation(
                    evcs_id=f"EVCS_{i+1:02d}",
                    bus_name=f"Bus_{i+1}",
                    max_power=self.config['evcs']['max_power_per_station'],
                    num_ports=4
                )
                stations.append(station)
            
            # Create enhanced CMS with federated PINN
            self.evcs_system = EnhancedChargingManagementSystem(
                stations=stations,
                use_pinn=True
            )
            
            # Integrate federated PINN if available
            if self.federated_pinn:
                self.evcs_system.federated_manager = self.federated_pinn
                print("✅ EVCS system integrated with federated PINN")
            else:
                print("⚠️  EVCS system initialized without federated PINN")
            
            print(f"✅ EVCS system initialized with {len(stations)} stations")
            
        except Exception as e:
            print(f"❌ Failed to initialize EVCS system: {e}")
            self.evcs_system = None
    
    def _initialize_rl_agents(self):
        """Initialize RL attack agents"""
        self.dqn_agent = DQNAttackAgent(
            state_dim=self.config['rl']['state_dim'],
            action_dim=self.config['rl']['action_dim'],
            hidden_dim=self.config['rl']['hidden_dim']
        )
        
        self.attack_coordinator = EVCSAttackCoordinator(
            num_agents=self.config['rl']['num_coordinator_agents'],
            state_dim=self.config['rl']['state_dim']
        )
        
        print("✅ RL attack agents initialized")
    
    def _initialize_federated_attack_scenarios(self) -> List[EVCSAttackScenario]:
        """Initialize attack scenarios specific to federated PINN systems"""
        scenarios = [
            EVCSAttackScenario(
                scenario_id="FED_PINN_001",
                name="Federated Model Poisoning",
                description="Poison federated PINN model through malicious local updates",
                target_components=["federated_pinn", "local_models", "aggregation_server"],
                attack_duration=180.0,
                stealth_requirement=0.9,
                impact_goal=0.8,
                constraints={
                    "max_model_corruption": 0.3,
                    "min_participating_clients": 2,
                    "evasion_threshold": 0.8
                }
            ),
            EVCSAttackScenario(
                scenario_id="FED_PINN_002",
                name="Gradient Manipulation Attack",
                description="Manipulate gradients during federated learning to corrupt global model",
                target_components=["gradient_updates", "federated_pinn", "optimization_process"],
                attack_duration=120.0,
                stealth_requirement=0.85,
                impact_goal=0.75,
                constraints={
                    "gradient_manipulation_limit": 0.5,
                    "detection_avoidance": True
                }
            ),
            EVCSAttackScenario(
                scenario_id="FED_PINN_003",
                name="Coordinated Multi-Client Attack",
                description="Coordinate attacks across multiple federated clients",
                target_components=["multiple_clients", "federated_pinn", "communication_protocol"],
                attack_duration=240.0,
                stealth_requirement=0.7,
                impact_goal=0.9,
                constraints={
                    "min_coordinated_clients": 3,
                    "synchronization_required": True
                }
            ),
            EVCSAttackScenario(
                scenario_id="FED_PINN_004",
                name="Backdoor Injection Attack",
                description="Inject backdoors into federated PINN model",
                target_components=["federated_pinn", "model_parameters", "inference_engine"],
                attack_duration=150.0,
                stealth_requirement=0.95,
                impact_goal=0.85,
                constraints={
                    "backdoor_trigger_activation": 0.1,
                    "model_performance_degradation": 0.2
                }
            )
        ]
        return scenarios
    
    def run_federated_attack_simulation(self, scenario_id: str, episodes: int = 50) -> Dict:
        """Run attack simulation targeting federated PINN system"""
        scenario = self._get_scenario_by_id(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        print(f"\n🚀 Starting Federated PINN Attack Simulation: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Target Components: {', '.join(scenario.target_components)}")
        print(f"Federated Rounds: {self.config['federated_pinn']['global_rounds']}")
        print("=" * 80)
        
        # Initialize simulation
        self.simulation_state['current_time'] = 0.0
        self.simulation_state['federated_round'] = 0
        self.simulation_state['attack_active'] = False
        self.simulation_state['pinn_compromised'] = False
        self.simulation_state['attack_history'] = []
        self.simulation_state['federated_metrics'] = {}
        
        episode_results = []
        
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            
            # Run federated learning round with potential attacks
            episode_result = self._run_federated_episode(scenario, episode)
            episode_results.append(episode_result)
            
            # Update RL agents
            self._update_rl_agents(episode_result)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                self._print_federated_episode_summary(episode_results[-10:])
        
        # Analyze results
        final_results = self._analyze_federated_simulation_results(episode_results, scenario)
        
        # Create visualizations
        self._create_federated_visualizations(episode_results, scenario)
        
        return final_results
    
    def _run_federated_episode(self, scenario: EVCSAttackScenario, episode: int) -> Dict:
        """Run single episode of federated learning with potential attacks"""
        episode_start_time = time.time()
        
        # Get federated system state
        federated_state = self._get_federated_system_state()
        
        # LLM analysis of federated vulnerabilities
        print("  🔍 LLM Federated Vulnerability Analysis...")
        vuln_analysis = self.llm_analyzer.analyze_evcs_vulnerabilities(
            federated_state, self.config['evcs']
        )
        
        # STRIDE-MITRE mapping for federated systems
        federated_threats = self.threat_mapper.analyze_threat_landscape(
            scenario.target_components + ['federated_learning', 'model_aggregation']
        )
        
        # Generate federated attack strategy
        print("  🎯 LLM Federated Attack Strategy...")
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
            federated_state,
            scenario.constraints
        )
        
        # RL agent coordination for federated attacks
        print("  🤖 RL Federated Attack Coordination...")
        rl_actions = self.attack_coordinator.coordinate_attack(
            federated_state,
            attack_strategy.get('attack_sequence', [])
        )
        
        # Execute federated attacks
        attack_results = self._execute_federated_attacks(rl_actions, federated_state, scenario)
        
        # Simulate federated learning round
        federated_metrics = self._simulate_federated_learning_round(attack_results)
        
        # Calculate rewards
        rewards = self._calculate_federated_rewards(attack_results, federated_metrics, scenario)
        
        # Update simulation state
        self.simulation_state['attack_history'].extend(attack_results)
        self.simulation_state['federated_round'] += 1
        self.simulation_state['federated_metrics'] = federated_metrics
        
        episode_duration = time.time() - episode_start_time
        
        return {
            'episode': episode,
            'duration': episode_duration,
            'federated_round': self.simulation_state['federated_round'],
            'federated_state': federated_state,
            'vulnerability_analysis': vuln_analysis,
            'federated_threats': federated_threats,
            'attack_strategy': attack_strategy,
            'rl_actions': rl_actions,
            'attack_results': attack_results,
            'federated_metrics': federated_metrics,
            'rewards': rewards,
            'total_reward': sum(rewards),
            'success_rate': len([r for r in rewards if r > 0]) / max(len(rewards), 1),
            'detection_rate': len([r for r in attack_results if r.get('detected', False)]) / max(len(attack_results), 1),
            'pinn_corruption': federated_metrics.get('model_corruption', 0.0)
        }
    
    def _get_federated_system_state(self) -> Dict:
        """Get federated system state including PINN model health"""
        base_state = {
            'num_stations': self.config['evcs']['num_stations'],
            'active_sessions': np.random.randint(5, 15),
            'voltage_levels': {f'bus{i+1}': np.random.uniform(0.95, 1.05) for i in range(6)},
            'frequency': np.random.uniform(59.8, 60.2),
            'grid_stability': np.random.uniform(0.8, 1.0),
            'communication_status': 'encrypted',
            'security_status': 'active'
        }
        
        # Add federated-specific state
        if self.federated_pinn:
            base_state.update({
                'federated_round': self.simulation_state['federated_round'],
                'pinn_model_health': self._calculate_pinn_model_health(),
                'federated_clients': self.config['federated_pinn']['num_distribution_systems'],
                'aggregation_status': 'active',
                'model_convergence': self._calculate_model_convergence(),
                'gradient_norm': np.random.uniform(0.1, 2.0),
                'participation_rate': np.random.uniform(0.7, 1.0)
            })
        else:
            base_state.update({
                'federated_round': 0,
                'pinn_model_health': 1.0,
                'federated_clients': 0,
                'aggregation_status': 'inactive',
                'model_convergence': 1.0,
                'gradient_norm': 0.0,
                'participation_rate': 0.0
            })
        
        return base_state
    
    def _calculate_pinn_model_health(self) -> float:
        """Calculate PINN model health considering potential corruption"""
        base_health = 1.0
        
        # Reduce health if attacks are active
        if self.simulation_state['attack_active']:
            base_health *= 0.8
        
        # Reduce health if model is compromised
        if self.simulation_state['pinn_compromised']:
            base_health *= 0.6
        
        # Add some randomness
        base_health *= np.random.uniform(0.9, 1.0)
        
        return max(base_health, 0.1)  # Minimum health of 0.1
    
    def _calculate_model_convergence(self) -> float:
        """Calculate model convergence metric"""
        # Simulate convergence based on federated round
        round_num = self.simulation_state['federated_round']
        max_rounds = self.config['federated_pinn']['global_rounds']
        
        # Sigmoid-like convergence curve
        convergence = 1.0 / (1.0 + np.exp(-0.5 * (round_num - max_rounds / 2)))
        
        return convergence
    
    def _execute_federated_attacks(self, actions: List, federated_state: Dict, 
                                  scenario: EVCSAttackScenario) -> List[Dict]:
        """Execute attacks targeting federated PINN system"""
        attack_results = []
        
        for action in actions:
            if action.action_type == 'no_attack':
                continue
            
            # Simulate federated-specific attack execution
            attack_result = {
                'action_id': action.action_id,
                'action_type': action.action_type,
                'target_component': action.target_component,
                'magnitude': action.magnitude,
                'duration': action.duration,
                'stealth_level': action.stealth_level,
                'executed': True,
                'detected': self._simulate_federated_detection(action, federated_state),
                'impact': self._simulate_federated_impact(action, federated_state),
                'federated_impact': self._simulate_federated_specific_impact(action, federated_state),
                'timestamp': self.simulation_state['current_time'],
                'federated_round': self.simulation_state['federated_round']
            }
            
            attack_results.append(attack_result)
            
            # Update federated system state
            self._update_federated_state_from_attack(attack_result)
        
        return attack_results
    
    def _simulate_federated_detection(self, action, federated_state: Dict) -> bool:
        """Simulate detection of federated attacks"""
        base_detection_prob = 0.2  # Lower for federated attacks (more complex)
        stealth_factor = action.stealth_level
        federated_security = 0.9  # High security for federated systems
        
        # Federated-specific detection factors
        if 'federated_pinn' in action.target_component:
            base_detection_prob *= 0.7  # Harder to detect model attacks
        
        if 'gradient' in action.action_type:
            base_detection_prob *= 0.8  # Gradient attacks are subtle
        
        detection_prob = base_detection_prob * (1 - stealth_factor) * federated_security
        
        return np.random.random() < detection_prob
    
    def _simulate_federated_impact(self, action, federated_state: Dict) -> float:
        """Simulate general impact of federated attacks"""
        base_impact = action.expected_impact
        magnitude_factor = action.magnitude
        system_vulnerability = 1.0 - federated_state.get('pinn_model_health', 0.9)
        
        impact = base_impact * magnitude_factor * (1 + system_vulnerability)
        
        return min(impact, 1.0)
    
    def _simulate_federated_specific_impact(self, action, federated_state: Dict) -> Dict:
        """Simulate federated-specific impact metrics"""
        impact_metrics = {
            'model_corruption': 0.0,
            'gradient_manipulation': 0.0,
            'aggregation_disruption': 0.0,
            'client_compromise': 0.0,
            'backdoor_injection': 0.0
        }
        
        # Model poisoning attacks
        if 'model' in action.target_component or 'pinn' in action.target_component:
            impact_metrics['model_corruption'] = action.magnitude * 0.8
        
        # Gradient manipulation
        if 'gradient' in action.action_type:
            impact_metrics['gradient_manipulation'] = action.magnitude * 0.9
        
        # Aggregation disruption
        if 'aggregation' in action.target_component:
            impact_metrics['aggregation_disruption'] = action.magnitude * 0.7
        
        # Client compromise
        if 'client' in action.target_component:
            impact_metrics['client_compromise'] = action.magnitude * 0.6
        
        # Backdoor injection
        if 'backdoor' in action.action_type:
            impact_metrics['backdoor_injection'] = action.magnitude * 0.85
        
        return impact_metrics
    
    def _update_federated_state_from_attack(self, attack_result: Dict):
        """Update federated system state based on attack results"""
        if attack_result['detected']:
            self.simulation_state['attack_active'] = False  # Attack stopped
        
        # Update PINN compromise status
        federated_impact = attack_result.get('federated_impact', {})
        if federated_impact.get('model_corruption', 0) > 0.3:
            self.simulation_state['pinn_compromised'] = True
        
        if attack_result['impact'] > 0.5:
            self.simulation_state['attack_active'] = True
    
    def _simulate_federated_learning_round(self, attack_results: List[Dict]) -> Dict:
        """Simulate a federated learning round with potential attacks"""
        if not self.federated_pinn:
            return {'model_corruption': 0.0, 'learning_progress': 0.0}
        
        # Calculate attack impact on federated learning
        total_corruption = 0.0
        gradient_manipulation = 0.0
        
        for result in attack_results:
            federated_impact = result.get('federated_impact', {})
            total_corruption += federated_impact.get('model_corruption', 0.0)
            gradient_manipulation += federated_impact.get('gradient_manipulation', 0.0)
        
        # Simulate learning progress
        base_progress = 0.1  # Base learning progress per round
        corruption_penalty = total_corruption * 0.5
        gradient_penalty = gradient_manipulation * 0.3
        
        learning_progress = max(0.0, base_progress - corruption_penalty - gradient_penalty)
        
        return {
            'model_corruption': min(total_corruption, 1.0),
            'gradient_manipulation': min(gradient_manipulation, 1.0),
            'learning_progress': learning_progress,
            'aggregation_quality': 1.0 - total_corruption * 0.7,
            'client_participation': max(0.5, 1.0 - total_corruption * 0.4)
        }
    
    def _calculate_federated_rewards(self, attack_results: List[Dict], 
                                   federated_metrics: Dict, scenario: EVCSAttackScenario) -> List[float]:
        """Calculate rewards for federated attacks"""
        rewards = []
        
        for result in attack_results:
            reward = 0.0
            
            # Base impact reward
            reward += result['impact'] * 100.0
            
            # Federated-specific rewards
            federated_impact = result.get('federated_impact', {})
            
            # Model corruption reward
            reward += federated_impact.get('model_corruption', 0) * 150.0
            
            # Gradient manipulation reward
            reward += federated_impact.get('gradient_manipulation', 0) * 120.0
            
            # Aggregation disruption reward
            reward += federated_impact.get('aggregation_disruption', 0) * 100.0
            
            # Backdoor injection reward
            reward += federated_impact.get('backdoor_injection', 0) * 200.0
            
            # Stealth bonus/penalty
            if result['detected']:
                reward -= 300.0  # Higher penalty for federated attacks
            else:
                reward += 80.0  # Higher bonus for stealth
            
            # Scenario-specific rewards
            if result['target_component'] in scenario.target_components:
                reward += 50.0
            
            # Stealth requirement bonus
            if result['stealth_level'] >= scenario.stealth_requirement:
                reward += 60.0
            
            # Impact goal bonus
            if result['impact'] >= scenario.impact_goal:
                reward += 80.0
            
            rewards.append(reward)
        
        return rewards
    
    def _update_rl_agents(self, episode_result: Dict):
        """Update RL agents with federated episode results"""
        if episode_result['rl_actions']:
            for i, action in enumerate(episode_result['rl_actions']):
                if i < len(episode_result['rewards']):
                    self.dqn_agent.store_experience(
                        self._federated_state_to_array(episode_result['federated_state']),
                        i,
                        episode_result['rewards'][i],
                        self._federated_state_to_array(episode_result['federated_state']),
                        False
                    )
            self.dqn_agent.train_step()
    
    def _federated_state_to_array(self, state: Dict) -> np.ndarray:
        """Convert federated state to numpy array"""
        features = []
        
        # Basic EVCS features
        features.extend([
            state.get('num_stations', 0) / 10.0,
            state.get('active_sessions', 0) / 20.0,
            state.get('frequency', 60.0) / 60.0,
            state.get('grid_stability', 1.0),
            state.get('pinn_model_health', 1.0)
        ])
        
        # Federated-specific features
        features.extend([
            state.get('federated_round', 0) / 20.0,
            state.get('pinn_model_health', 1.0),
            state.get('model_convergence', 1.0),
            state.get('gradient_norm', 0.0) / 5.0,
            state.get('participation_rate', 1.0)
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
    
    def _print_federated_episode_summary(self, recent_results: List[Dict]):
        """Print summary of recent federated episodes"""
        if not recent_results:
            return
        
        avg_reward = np.mean([r['total_reward'] for r in recent_results])
        avg_success = np.mean([r['success_rate'] for r in recent_results])
        avg_detection = np.mean([r['detection_rate'] for r in recent_results])
        avg_corruption = np.mean([r['pinn_corruption'] for r in recent_results])
        
        print(f"  Recent Federated Performance (last {len(recent_results)} episodes):")
        print(f"    Average Reward: {avg_reward:.2f}")
        print(f"    Success Rate: {avg_success:.1%}")
        print(f"    Detection Rate: {avg_detection:.1%}")
        print(f"    PINN Corruption: {avg_corruption:.1%}")
    
    def _analyze_federated_simulation_results(self, episode_results: List[Dict], 
                                            scenario: EVCSAttackScenario) -> Dict:
        """Analyze federated simulation results"""
        if not episode_results:
            return {}
        
        # Calculate aggregate metrics
        total_episodes = len(episode_results)
        total_rewards = [r['total_reward'] for r in episode_results]
        success_rates = [r['success_rate'] for r in episode_results]
        detection_rates = [r['detection_rate'] for r in episode_results]
        pinn_corruptions = [r['pinn_corruption'] for r in episode_results]
        
        # Federated-specific analysis
        federated_metrics = [r['federated_metrics'] for r in episode_results]
        avg_model_corruption = np.mean([m.get('model_corruption', 0) for m in federated_metrics])
        avg_learning_progress = np.mean([m.get('learning_progress', 0) for m in federated_metrics])
        
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
                'average_pinn_corruption': np.mean(pinn_corruptions),
                'best_episode_reward': np.max(total_rewards),
                'worst_episode_reward': np.min(total_rewards)
            },
            'federated_metrics': {
                'average_model_corruption': avg_model_corruption,
                'average_learning_progress': avg_learning_progress,
                'federated_rounds_completed': self.simulation_state['federated_round'],
                'pinn_compromise_rate': len([r for r in episode_results if r['pinn_corruption'] > 0.3]) / total_episodes
            },
            'learning_progress': {
                'reward_trend': self._calculate_trend(total_rewards),
                'success_trend': self._calculate_trend(success_rates),
                'detection_trend': self._calculate_trend(detection_rates),
                'corruption_trend': self._calculate_trend(pinn_corruptions)
            },
            'recommendations': self._generate_federated_recommendations(episode_results, scenario)
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
    
    def _generate_federated_recommendations(self, episode_results: List[Dict], 
                                          scenario: EVCSAttackScenario) -> List[str]:
        """Generate recommendations for federated attacks"""
        recommendations = []
        
        # Analyze PINN corruption
        avg_corruption = np.mean([r['pinn_corruption'] for r in episode_results])
        if avg_corruption > 0.3:
            recommendations.append("High PINN model corruption achieved - consider reducing attack intensity")
        elif avg_corruption < 0.1:
            recommendations.append("Low PINN model corruption - increase attack magnitude or frequency")
        
        # Analyze detection rates
        detection_rates = [r['detection_rate'] for r in episode_results]
        if np.mean(detection_rates) > 0.2:
            recommendations.append("High detection rate in federated attacks - improve stealth techniques")
        
        # Analyze success rates
        success_rates = [r['success_rate'] for r in episode_results]
        if np.mean(success_rates) < 0.4:
            recommendations.append("Low success rate - focus on federated-specific attack vectors")
        
        # Scenario-specific recommendations
        if scenario.scenario_id == "FED_PINN_001":
            recommendations.append("Model poisoning scenario - ensure gradient manipulation is subtle")
        elif scenario.scenario_id == "FED_PINN_002":
            recommendations.append("Gradient manipulation - coordinate across multiple clients")
        elif scenario.scenario_id == "FED_PINN_003":
            recommendations.append("Multi-client coordination - synchronize attack timing")
        elif scenario.scenario_id == "FED_PINN_004":
            recommendations.append("Backdoor injection - ensure triggers are well-hidden")
        
        return recommendations
    
    def _create_federated_visualizations(self, episode_results: List[Dict], scenario: EVCSAttackScenario):
        """Create visualizations for federated simulation results"""
        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            fig.suptitle(f'Federated PINN LLM-RL Attack Simulation: {scenario.name}', 
                        fontsize=16, fontweight='bold')
            
            # Extract data
            episodes = list(range(len(episode_results)))
            rewards = [r['total_reward'] for r in episode_results]
            success_rates = [r['success_rate'] for r in episode_results]
            detection_rates = [r['detection_rate'] for r in episode_results]
            pinn_corruptions = [r['pinn_corruption'] for r in episode_results]
            
            # 1. Episode Rewards
            axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. PINN Model Corruption
            axes[0, 1].plot(episodes, [c * 100 for c in pinn_corruptions], 'r-', linewidth=2, alpha=0.8)
            axes[0, 1].set_title('PINN Model Corruption')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Corruption (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Success vs Detection Rate
            axes[0, 2].plot(episodes, [s * 100 for s in success_rates], 'g-', label='Success Rate', linewidth=2, alpha=0.8)
            axes[0, 2].plot(episodes, [d * 100 for d in detection_rates], 'r-', label='Detection Rate', linewidth=2, alpha=0.8)
            axes[0, 2].set_title('Success vs Detection Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Rate (%)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Federated Learning Progress
            federated_metrics = [r['federated_metrics'] for r in episode_results]
            learning_progress = [m.get('learning_progress', 0) for m in federated_metrics]
            model_corruption = [m.get('model_corruption', 0) for m in federated_metrics]
            
            axes[1, 0].plot(episodes, learning_progress, 'purple', linewidth=2, alpha=0.8, label='Learning Progress')
            axes[1, 0].plot(episodes, model_corruption, 'orange', linewidth=2, alpha=0.8, label='Model Corruption')
            axes[1, 0].set_title('Federated Learning Metrics')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Attack Impact Distribution
            all_attacks = []
            for result in episode_results:
                all_attacks.extend(result.get('attack_results', []))
            
            if all_attacks:
                impacts = [a['impact'] for a in all_attacks]
                axes[1, 1].hist(impacts, bins=20, alpha=0.7, color='green')
                axes[1, 1].set_title('Attack Impact Distribution')
                axes[1, 1].set_xlabel('Impact Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Federated Performance Summary
            summary_data = [
                ['Metric', 'Value'],
                ['Avg Reward', f"{np.mean(rewards):.2f}"],
                ['Avg Success Rate', f"{np.mean(success_rates):.1%}"],
                ['Avg Detection Rate', f"{np.mean(detection_rates):.1%}"],
                ['Avg PINN Corruption', f"{np.mean(pinn_corruptions):.1%}"],
                ['Federated Rounds', f"{self.simulation_state['federated_round']}"],
                ['Best Episode', f"{np.max(rewards):.2f}"]
            ]
            
            axes[1, 2].axis('tight')
            axes[1, 2].axis('off')
            table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 2].set_title('Federated Performance Summary')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'federated_pinn_llm_rl_simulation_{scenario.scenario_id}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Federated visualization saved as: {filename}")
            
        except Exception as e:
            print(f"Warning: Federated visualization creation failed: {e}")
    
    def _get_scenario_by_id(self, scenario_id: str) -> Optional[EVCSAttackScenario]:
        """Get scenario by ID"""
        for scenario in self.attack_scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

def main():
    """Main function to demonstrate the Federated PINN LLM-RL system"""
    print("🚀 Federated PINN LLM-RL Attack Analytics System")
    print("=" * 70)
    
    # Initialize system
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
            'max_power_per_station': 1000
        }
    }
    
    system = FederatedPINNLLMRLSystem(config)
    
    # Test components
    print("\n1. Testing System Components...")
    print(f"   LLM Available: {'✅' if system.llm_analyzer.is_available else '❌'}")
    print(f"   Federated PINN: {'✅' if system.federated_pinn else '❌'}")
    print(f"   EVCS System: {'✅' if system.evcs_system else '❌'}")
    print(f"   RL Agents: ✅")
    
    # Test threat mapping
    print("\n2. Testing STRIDE-MITRE Mapping...")
    mitre_techniques = system.threat_mapper.map_stride_to_mitre('Tampering', 'model_tampering')
    print(f"   MITRE techniques for model tampering: {mitre_techniques}")
    
    # Run federated attack simulation
    print("\n3. Running Federated Attack Simulation...")
    try:
        results = system.run_federated_attack_simulation("FED_PINN_001", episodes=30)
        
        print(f"\n📊 Federated Simulation Results:")
        print(f"   Total Episodes: {results['performance_metrics']['total_episodes']}")
        print(f"   Average Reward: {results['performance_metrics']['average_reward']:.2f}")
        print(f"   Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
        print(f"   Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")
        print(f"   PINN Corruption: {results['performance_metrics']['average_pinn_corruption']:.1%}")
        
        print(f"\n🎯 Federated Recommendations:")
        for rec in results['recommendations']:
            print(f"   - {rec}")
        
    except Exception as e:
        print(f"Federated simulation failed: {e}")
    
    print("\n✅ Federated PINN LLM-RL System demonstration completed!")

if __name__ == "__main__":
    main()
