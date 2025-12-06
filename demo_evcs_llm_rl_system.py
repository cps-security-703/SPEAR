#!/usr/bin/env python3
"""
Demo Script for EVCS LLM-RL Attack Analytics System
Demonstrates the complete workflow integrating LLM threat analysis, RL attack agents, and federated PINN EVCS systems
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Import our system components
from evcs_llm_rl_integration import EVCSLLMRLSystem, EVCSAttackScenario
from federated_pinn_llm_rl_integration import FederatedPINNLLMRLSystem
from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer, STRIDEMITREThreatMapper
from evcs_rl_attack_agent import DQNAttackAgent, EVCSAttackCoordinator

warnings.filterwarnings('ignore')

class EVCSLLMRLDemo:
    """Comprehensive demo of the EVCS LLM-RL system"""
    
    def __init__(self):
        self.demo_config = self._create_demo_config()
        self.results = {}
        
    def _create_demo_config(self) -> Dict:
        """Create configuration for the demo"""
        return {
            'gemini': {
                'model': 'models/gemini-2.5-flash',
                'api_key_file': 'gemini_key.txt'
            },
            'federated_pinn': {
                'num_distribution_systems': 6,
                'local_epochs': 50,  # Reduced for demo
                'global_rounds': 5,   # Reduced for demo
                'aggregation_method': 'fedavg',
                'model_path': 'federated_models'
            },
            'rl': {
                'state_dim': 50,
                'action_dim': 8,
                'hidden_dim': 256,
                'num_coordinator_agents': 3
            },
            'evcs': {
                'num_stations': 6,
                'max_power_per_station': 1000,
                'voltage_limits': {'min': 0.95, 'max': 1.05},
                'frequency_limits': {'min': 59.5, 'max': 60.5}
            },
            'demo': {
                'episodes_per_scenario': 20,  # Reduced for demo
                'enable_visualizations': True,
                'save_results': True,
                'detailed_logging': True
            }
        }
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 EVCS LLM-RL Attack Analytics System - Complete Demo")
        print("=" * 80)
        print("This demo showcases the integration of:")
        print("  • Ollama deepseek-r1:8b for LLM threat analysis")
        print("  • STRIDE-MITRE threat mapping for EVCS vulnerabilities")
        print("  • Deep Q-Network (DQN) and PPO RL agents")
        print("  • Federated PINN model attack vectors")
        print("  • Comprehensive visualization and analysis")
        print("=" * 80)
        
        # Step 1: Test system components
        self._test_system_components()
        
        # Step 2: Demonstrate LLM threat analysis
        self._demonstrate_llm_threat_analysis()
        
        # Step 3: Demonstrate STRIDE-MITRE mapping
        self._demonstrate_stride_mitre_mapping()
        
        # Step 4: Demonstrate RL attack agents
        self._demonstrate_rl_agents()
        
        # Step 5: Run basic EVCS attack simulation
        self._run_basic_evcs_simulation()
        
        # Step 6: Run federated PINN attack simulation
        self._run_federated_pinn_simulation()
        
        # Step 7: Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        print("\n✅ Complete demo finished successfully!")
        print("Check the generated files for detailed results and visualizations.")
    
    def _test_system_components(self):
        """Test all system components"""
        print("\n1. 🔧 Testing System Components")
        print("-" * 50)
        
        # Test LLM analyzer
        print("Testing Gemini LLM connection...")
        llm_analyzer = GeminiLLMThreatAnalyzer(
            model_name=self.demo_config['gemini']['model']
        )
        
        if llm_analyzer.is_available:
            print("  ✅ Gemini LLM connection successful")
        else:
            print("  ❌ Gemini LLM connection failed - using fallback mode")
        
        # Test threat mapper
        print("Testing STRIDE-MITRE threat mapper...")
        threat_mapper = STRIDEMITREThreatMapper()
        mitre_techniques = threat_mapper.map_stride_to_mitre('Tampering', 'data_tampering')
        print(f"  ✅ STRIDE-MITRE mapping working: {len(mitre_techniques)} techniques found")
        
        # Test RL agents
        print("Testing RL attack agents...")
        dqn_agent = DQNAttackAgent(
            state_dim=self.demo_config['rl']['state_dim'],
            action_dim=self.demo_config['rl']['action_dim']
        )
        print(f"  ✅ DQN agent initialized with {len(dqn_agent.attack_actions)} attack actions")
        
        coordinator = EVCSAttackCoordinator(
            num_agents=self.demo_config['rl']['num_coordinator_agents'],
            state_dim=self.demo_config['rl']['state_dim']
        )
        print(f"  ✅ Attack coordinator initialized with {coordinator.num_agents} agents")
        
        self.results['component_tests'] = {
            'llm_available': llm_analyzer.is_available,
            'threat_mapper_working': True,
            'rl_agents_working': True,
            'mitre_techniques_count': len(mitre_techniques)
        }
    
    def _demonstrate_llm_threat_analysis(self):
        """Demonstrate LLM threat analysis capabilities"""
        print("\n2. 🧠 Demonstrating LLM Threat Analysis")
        print("-" * 50)
        
        llm_analyzer = GeminiLLMThreatAnalyzer(
            model_name=self.demo_config['gemini']['model']
        )
        
        # Create test EVCS state
        test_evcs_state = {
            'num_stations': 6,
            'active_sessions': 12,
            'voltage_levels': {'bus1': 0.98, 'bus2': 1.02, 'bus3': 0.99, 'bus4': 1.01, 'bus5': 0.97, 'bus6': 1.03},
            'frequency': 59.8,
            'grid_stability': 0.9,
            'pinn_model_health': 0.95,
            'communication_status': 'encrypted',
            'security_status': 'active'
        }
        
        test_system_config = {
            'federated_enabled': True,
            'security_measures': ['encryption', 'authentication', 'anomaly_detection', 'intrusion_detection'],
            'protocols': ['IEC 61850', 'Modbus', 'DNP3'],
            'topology': 'hierarchical_federated'
        }
        
        print("Analyzing EVCS vulnerabilities using LLM...")
        vuln_analysis = llm_analyzer.analyze_evcs_vulnerabilities(test_evcs_state, test_system_config)
        
        print(f"  Vulnerabilities found: {len(vuln_analysis.get('vulnerabilities', []))}")
        print(f"  Analysis confidence: {vuln_analysis.get('analysis_confidence', 0.0):.1%}")
        
        if vuln_analysis.get('vulnerabilities'):
            print("  Top vulnerabilities:")
            for i, vuln in enumerate(vuln_analysis['vulnerabilities'][:3], 1):
                print(f"    {i}. {vuln.get('vulnerability_type', 'Unknown')} (Severity: {vuln.get('severity', 0.0):.2f})")
        
        # Generate attack strategy
        print("\nGenerating attack strategy using LLM...")
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
        
        attack_strategy = llm_analyzer.generate_attack_strategy(
            vulnerabilities,
            test_evcs_state,
            {'stealth_level': 'high', 'time_window': 'extended'}
        )
        
        print(f"  Strategy name: {attack_strategy.get('strategy_name', 'Unknown')}")
        print(f"  Attack sequence steps: {len(attack_strategy.get('attack_sequence', []))}")
        
        self.results['llm_analysis'] = {
            'vulnerabilities_found': len(vuln_analysis.get('vulnerabilities', [])),
            'analysis_confidence': vuln_analysis.get('analysis_confidence', 0.0),
            'strategy_generated': bool(attack_strategy.get('attack_sequence')),
            'llm_available': llm_analyzer.is_available
        }
    
    def _demonstrate_stride_mitre_mapping(self):
        """Demonstrate STRIDE-MITRE threat mapping"""
        print("\n3. 🗺️ Demonstrating STRIDE-MITRE Threat Mapping")
        print("-" * 50)
        
        threat_mapper = STRIDEMITREThreatMapper()
        
        # Test different STRIDE categories
        stride_categories = ['Spoofing', 'Tampering', 'Information_Disclosure', 'Denial_of_Service']
        
        print("Mapping STRIDE categories to MITRE ATT&CK techniques:")
        for category in stride_categories:
            techniques = threat_mapper.map_stride_to_mitre(category)
            print(f"  {category}: {len(techniques)} techniques ({', '.join(techniques[:3])}{'...' if len(techniques) > 3 else ''})")
        
        # Analyze threat landscape for EVCS components
        evcs_components = ['charging_controller', 'communication_protocol', 'pinn_model', 'power_electronics']
        print(f"\nAnalyzing threat landscape for EVCS components: {', '.join(evcs_components)}")
        
        threat_landscape = threat_mapper.analyze_threat_landscape(evcs_components)
        
        for component, threats in threat_landscape.items():
            high_risk_count = sum(1 for t in threats.values() if t['risk_level'] == 'High')
            print(f"  {component}: {high_risk_count} high-risk threat categories")
        
        # Get EVCS-specific attack techniques
        evcs_techniques = threat_mapper.get_evcs_attack_techniques()
        print(f"\nEVCS-specific attack techniques: {len(evcs_techniques)}")
        for technique_id, technique_info in evcs_techniques.items():
            print(f"  {technique_id}: {technique_info['description']}")
        
        self.results['stride_mitre_mapping'] = {
            'categories_tested': len(stride_categories),
            'total_techniques': sum(len(threat_mapper.map_stride_to_mitre(cat)) for cat in stride_categories),
            'evcs_components_analyzed': len(evcs_components),
            'evcs_specific_techniques': len(evcs_techniques)
        }
    
    def _demonstrate_rl_agents(self):
        """Demonstrate RL attack agents"""
        print("\n4. 🤖 Demonstrating RL Attack Agents")
        print("-" * 50)
        
        # Initialize DQN agent
        dqn_agent = DQNAttackAgent(
            state_dim=self.demo_config['rl']['state_dim'],
            action_dim=self.demo_config['rl']['action_dim']
        )
        
        print(f"DQN Agent initialized:")
        print(f"  State dimension: {dqn_agent.state_dim}")
        print(f"  Action dimension: {dqn_agent.action_dim}")
        print(f"  Available attack actions: {len(dqn_agent.attack_actions)}")
        
        # Test action selection
        test_state = np.random.random(self.demo_config['rl']['state_dim'])
        action_idx = dqn_agent.select_action(test_state)
        selected_action = dqn_agent.attack_actions[action_idx]
        
        print(f"\nAction selection test:")
        print(f"  Selected action: {selected_action.action_id}")
        print(f"  Action type: {selected_action.action_type}")
        print(f"  Target component: {selected_action.target_component}")
        print(f"  Expected impact: {selected_action.expected_impact:.2f}")
        print(f"  Stealth level: {selected_action.stealth_level:.2f}")
        
        # Test reward calculation
        test_state_dict = {
            'grid_instability': 0.2,
            'charging_disruption': 0.3,
            'pinn_model_corruption': 0.1
        }
        next_state_dict = {
            'grid_instability': 0.4,
            'charging_disruption': 0.5,
            'pinn_model_corruption': 0.2
        }
        
        reward = dqn_agent.calculate_reward(
            test_state_dict, selected_action, next_state_dict, False, 0.8
        )
        print(f"  Calculated reward: {reward:.2f}")
        
        # Test attack coordinator
        coordinator = EVCSAttackCoordinator(
            num_agents=self.demo_config['rl']['num_coordinator_agents'],
            state_dim=self.demo_config['rl']['state_dim']
        )
        
        print(f"\nAttack Coordinator test:")
        print(f"  Number of agents: {coordinator.num_agents}")
        
        test_evcs_state = {
            'num_stations': 6,
            'active_sessions': 12,
            'frequency': 59.8,
            'grid_stability': 0.9,
            'pinn_model_health': 0.95
        }
        
        coordinated_actions = coordinator.coordinate_attack(test_evcs_state, [])
        print(f"  Coordinated actions generated: {len(coordinated_actions)}")
        
        self.results['rl_agents'] = {
            'dqn_actions_available': len(dqn_agent.attack_actions),
            'coordinator_agents': coordinator.num_agents,
            'action_selection_working': True,
            'reward_calculation_working': True,
            'coordination_working': len(coordinated_actions) > 0
        }
    
    def _run_basic_evcs_simulation(self):
        """Run basic EVCS attack simulation"""
        print("\n5. ⚡ Running Basic EVCS Attack Simulation")
        print("-" * 50)
        
        # Initialize basic EVCS system
        evcs_system = EVCSLLMRLSystem(self.demo_config)
        
        print("Running attack simulation scenario: Stealth Grid Manipulation")
        
        try:
            results = evcs_system.run_attack_simulation(
                scenario_id="SCENARIO_001",
                episodes=self.demo_config['demo']['episodes_per_scenario']
            )
            
            print(f"\nBasic EVCS Simulation Results:")
            print(f"  Total Episodes: {results['performance_metrics']['total_episodes']}")
            print(f"  Average Reward: {results['performance_metrics']['average_reward']:.2f}")
            print(f"  Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
            print(f"  Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")
            print(f"  Best Episode Reward: {results['performance_metrics']['best_episode_reward']:.2f}")
            
            print(f"\nRecommendations:")
            for rec in results['recommendations'][:3]:
                print(f"  - {rec}")
            
            self.results['basic_evcs_simulation'] = results['performance_metrics']
            
        except Exception as e:
            print(f"  ❌ Basic EVCS simulation failed: {e}")
            self.results['basic_evcs_simulation'] = {'error': str(e)}
    
    def _run_federated_pinn_simulation(self):
        """Run federated PINN attack simulation"""
        print("\n6. 🧠 Running Federated PINN Attack Simulation")
        print("-" * 50)
        
        # Initialize federated PINN system
        federated_system = FederatedPINNLLMRLSystem(self.demo_config)
        
        print("Running federated attack simulation scenario: Federated Model Poisoning")
        
        try:
            results = federated_system.run_federated_attack_simulation(
                scenario_id="FED_PINN_001",
                episodes=self.demo_config['demo']['episodes_per_scenario']
            )
            
            print(f"\nFederated PINN Simulation Results:")
            print(f"  Total Episodes: {results['performance_metrics']['total_episodes']}")
            print(f"  Average Reward: {results['performance_metrics']['average_reward']:.2f}")
            print(f"  Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
            print(f"  Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")
            print(f"  PINN Corruption: {results['performance_metrics']['average_pinn_corruption']:.1%}")
            
            print(f"\nFederated Metrics:")
            fed_metrics = results['federated_metrics']
            print(f"  Model Corruption: {fed_metrics['average_model_corruption']:.1%}")
            print(f"  Learning Progress: {fed_metrics['average_learning_progress']:.1%}")
            print(f"  Federated Rounds: {fed_metrics['federated_rounds_completed']}")
            print(f"  PINN Compromise Rate: {fed_metrics['pinn_compromise_rate']:.1%}")
            
            print(f"\nFederated Recommendations:")
            for rec in results['recommendations'][:3]:
                print(f"  - {rec}")
            
            self.results['federated_pinn_simulation'] = {
                'performance_metrics': results['performance_metrics'],
                'federated_metrics': results['federated_metrics']
            }
            
        except Exception as e:
            print(f"  ❌ Federated PINN simulation failed: {e}")
            self.results['federated_pinn_simulation'] = {'error': str(e)}
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        print("\n7. 📊 Generating Comprehensive Analysis")
        print("-" * 50)
        
        # Create comprehensive summary visualization
        self._create_demo_summary_visualization()
        
        # Save results to JSON
        if self.demo_config['demo']['save_results']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f'evcs_llm_rl_demo_results_{timestamp}.json'
            
            with open(results_filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"  📄 Results saved to: {results_filename}")
        
        # Print summary statistics
        self._print_demo_summary()
    
    def _create_demo_summary_visualization(self):
        """Create comprehensive demo summary visualization"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('EVCS LLM-RL Attack Analytics System - Demo Summary', 
                        fontsize=16, fontweight='bold')
            
            # 1. Component Test Results
            component_tests = self.results.get('component_tests', {})
            test_labels = ['LLM Available', 'Threat Mapper', 'RL Agents']
            test_values = [
                component_tests.get('llm_available', False),
                component_tests.get('threat_mapper_working', False),
                component_tests.get('rl_agents_working', False)
            ]
            test_colors = ['green' if v else 'red' for v in test_values]
            
            axes[0, 0].bar(test_labels, [1 if v else 0 for v in test_values], color=test_colors, alpha=0.7)
            axes[0, 0].set_title('System Component Tests')
            axes[0, 0].set_ylabel('Status (1=Pass, 0=Fail)')
            axes[0, 0].set_ylim(0, 1.2)
            
            # 2. LLM Analysis Results
            llm_analysis = self.results.get('llm_analysis', {})
            llm_metrics = ['Vulnerabilities Found', 'Analysis Confidence', 'Strategy Generated']
            llm_values = [
                llm_analysis.get('vulnerabilities_found', 0),
                llm_analysis.get('analysis_confidence', 0) * 100,
                1 if llm_analysis.get('strategy_generated', False) else 0
            ]
            
            bars = axes[0, 1].bar(llm_metrics, llm_values, color=['blue', 'orange', 'green'], alpha=0.7)
            axes[0, 1].set_title('LLM Threat Analysis Results')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, llm_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               f'{value:.1f}', ha='center', va='bottom')
            
            # 3. STRIDE-MITRE Mapping Results
            stride_mapping = self.results.get('stride_mitre_mapping', {})
            mapping_data = [
                stride_mapping.get('categories_tested', 0),
                stride_mapping.get('total_techniques', 0),
                stride_mapping.get('evcs_components_analyzed', 0),
                stride_mapping.get('evcs_specific_techniques', 0)
            ]
            mapping_labels = ['Categories', 'Techniques', 'Components', 'EVCS Specific']
            
            axes[0, 2].bar(mapping_labels, mapping_data, color='purple', alpha=0.7)
            axes[0, 2].set_title('STRIDE-MITRE Mapping Results')
            axes[0, 2].set_ylabel('Count')
            
            # 4. RL Agents Performance
            rl_agents = self.results.get('rl_agents', {})
            rl_metrics = ['Actions Available', 'Coordinator Agents', 'Action Selection', 'Reward Calc', 'Coordination']
            rl_values = [
                rl_agents.get('dqn_actions_available', 0),
                rl_agents.get('coordinator_agents', 0),
                1 if rl_agents.get('action_selection_working', False) else 0,
                1 if rl_agents.get('reward_calculation_working', False) else 0,
                1 if rl_agents.get('coordination_working', False) else 0
            ]
            
            axes[1, 0].bar(rl_metrics, rl_values, color='red', alpha=0.7)
            axes[1, 0].set_title('RL Agents Performance')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. Basic EVCS Simulation Results
            basic_sim = self.results.get('basic_evcs_simulation', {})
            if 'error' not in basic_sim:
                sim_metrics = ['Avg Reward', 'Success Rate', 'Detection Rate', 'Best Reward']
                sim_values = [
                    basic_sim.get('average_reward', 0),
                    basic_sim.get('average_success_rate', 0) * 100,
                    basic_sim.get('average_detection_rate', 0) * 100,
                    basic_sim.get('best_episode_reward', 0)
                ]
                
                axes[1, 1].bar(sim_metrics, sim_values, color='green', alpha=0.7)
                axes[1, 1].set_title('Basic EVCS Simulation Results')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, f"Simulation Failed:\n{basic_sim['error']}", 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Basic EVCS Simulation Results')
            
            # 6. Federated PINN Simulation Results
            fed_sim = self.results.get('federated_pinn_simulation', {})
            if 'error' not in fed_sim:
                fed_perf = fed_sim.get('performance_metrics', {})
                fed_metrics = ['Avg Reward', 'Success Rate', 'Detection Rate', 'PINN Corruption']
                fed_values = [
                    fed_perf.get('average_reward', 0),
                    fed_perf.get('average_success_rate', 0) * 100,
                    fed_perf.get('average_detection_rate', 0) * 100,
                    fed_perf.get('average_pinn_corruption', 0) * 100
                ]
                
                axes[1, 2].bar(fed_metrics, fed_values, color='orange', alpha=0.7)
                axes[1, 2].set_title('Federated PINN Simulation Results')
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].tick_params(axis='x', rotation=45)
            else:
                axes[1, 2].text(0.5, 0.5, f"Simulation Failed:\n{fed_sim['error']}", 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Federated PINN Simulation Results')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'evcs_llm_rl_demo_summary_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Demo summary visualization saved as: {filename}")
            
        except Exception as e:
            print(f"  ⚠️  Visualization creation failed: {e}")
    
    def _print_demo_summary(self):
        """Print comprehensive demo summary"""
        print(f"\n📋 Demo Summary")
        print("=" * 50)
        
        # Component tests
        component_tests = self.results.get('component_tests', {})
        print(f"System Components:")
        print(f"  LLM Available: {'✅' if component_tests.get('llm_available', False) else '❌'}")
        print(f"  Threat Mapper: {'✅' if component_tests.get('threat_mapper_working', False) else '❌'}")
        print(f"  RL Agents: {'✅' if component_tests.get('rl_agents_working', False) else '❌'}")
        
        # LLM analysis
        llm_analysis = self.results.get('llm_analysis', {})
        print(f"\nLLM Threat Analysis:")
        print(f"  Vulnerabilities Found: {llm_analysis.get('vulnerabilities_found', 0)}")
        print(f"  Analysis Confidence: {llm_analysis.get('analysis_confidence', 0.0):.1%}")
        print(f"  Strategy Generated: {'✅' if llm_analysis.get('strategy_generated', False) else '❌'}")
        
        # STRIDE-MITRE mapping
        stride_mapping = self.results.get('stride_mitre_mapping', {})
        print(f"\nSTRIDE-MITRE Mapping:")
        print(f"  Categories Tested: {stride_mapping.get('categories_tested', 0)}")
        print(f"  Total Techniques: {stride_mapping.get('total_techniques', 0)}")
        print(f"  EVCS Components: {stride_mapping.get('evcs_components_analyzed', 0)}")
        print(f"  EVCS-Specific Techniques: {stride_mapping.get('evcs_specific_techniques', 0)}")
        
        # RL agents
        rl_agents = self.results.get('rl_agents', {})
        print(f"\nRL Attack Agents:")
        print(f"  DQN Actions Available: {rl_agents.get('dqn_actions_available', 0)}")
        print(f"  Coordinator Agents: {rl_agents.get('coordinator_agents', 0)}")
        print(f"  Action Selection: {'✅' if rl_agents.get('action_selection_working', False) else '❌'}")
        print(f"  Reward Calculation: {'✅' if rl_agents.get('reward_calculation_working', False) else '❌'}")
        print(f"  Coordination: {'✅' if rl_agents.get('coordination_working', False) else '❌'}")
        
        # Basic simulation
        basic_sim = self.results.get('basic_evcs_simulation', {})
        if 'error' not in basic_sim:
            print(f"\nBasic EVCS Simulation:")
            print(f"  Episodes: {basic_sim.get('total_episodes', 0)}")
            print(f"  Avg Reward: {basic_sim.get('average_reward', 0.0):.2f}")
            print(f"  Success Rate: {basic_sim.get('average_success_rate', 0.0):.1%}")
            print(f"  Detection Rate: {basic_sim.get('average_detection_rate', 0.0):.1%}")
        else:
            print(f"\nBasic EVCS Simulation: ❌ Failed - {basic_sim['error']}")
        
        # Federated simulation
        fed_sim = self.results.get('federated_pinn_simulation', {})
        if 'error' not in fed_sim:
            fed_perf = fed_sim.get('performance_metrics', {})
            fed_metrics = fed_sim.get('federated_metrics', {})
            print(f"\nFederated PINN Simulation:")
            print(f"  Episodes: {fed_perf.get('total_episodes', 0)}")
            print(f"  Avg Reward: {fed_perf.get('average_reward', 0.0):.2f}")
            print(f"  Success Rate: {fed_perf.get('average_success_rate', 0.0):.1%}")
            print(f"  Detection Rate: {fed_perf.get('average_detection_rate', 0.0):.1%}")
            print(f"  PINN Corruption: {fed_perf.get('average_pinn_corruption', 0.0):.1%}")
            print(f"  Model Corruption: {fed_metrics.get('average_model_corruption', 0.0):.1%}")
            print(f"  Learning Progress: {fed_metrics.get('average_learning_progress', 0.0):.1%}")
        else:
            print(f"\nFederated PINN Simulation: ❌ Failed - {fed_sim['error']}")

def main():
    """Main function to run the complete demo"""
    print("🚀 Starting EVCS LLM-RL Attack Analytics System Demo")
    print("=" * 80)
    
    # Check if Gemini is accessible
    try:
        import google.generativeai as genai
        
        # Load API key
        with open('gemini_key.txt', 'r') as f:
            api_key = f.read().strip()
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Test connection
        response = model.generate_content("test")
        if response.text:
            print("✅ Gemini Pro is accessible and working")
        else:
            raise Exception("Empty response from Gemini")
            
    except FileNotFoundError:
        print("⚠️  Gemini API key file (gemini_key.txt) not found. The demo will use fallback mode.")
        print("   Please create gemini_key.txt with your Google API key")
    except Exception as e:
        print("⚠️  Gemini Pro is not accessible. The demo will use fallback mode.")
        print(f"   Error: {e}")
        print("   Please check your API key and internet connection")
    
    # Run the demo
    demo = EVCSLLMRLDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
