#!/usr/bin/env python3
"""
Test script to verify visualization fixes for detailed attack analysis
"""

import sys
import numpy as np
from integrated_evcs_llm_rl_system import IntegratedEVCSLLMRLSystem, IntegratedAttackScenario

def create_mock_episode_results():
    """Create mock episode results with proper attack data structure"""
    return [
        {
            'episode': 1,
            'total_reward': 45.2,
            'success_rate': 0.6,
            'detection_rate': 0.3,
            'stealth_score': 0.7,
            'attack_results': [
                {
                    'action_type': 'power_manipulation',
                    'success': True,
                    'impact': 35.0,
                    'detected': False,
                    'stealth_level': 0.8
                },
                {
                    'action_type': 'pinn_manipulation',
                    'success': True,
                    'impact': 28.5,
                    'detected': True,
                    'stealth_level': 0.4
                }
            ],
            'rl_actions': [
                {'action_type': 'coordinated_attack', 'magnitude': 0.6},
                {'action_type': 'stealth_attack', 'magnitude': 0.4}
            ]
        },
        {
            'episode': 2,
            'total_reward': 32.1,
            'success_rate': 0.4,
            'detection_rate': 0.5,
            'stealth_score': 0.5,
            'attack_results': [
                {
                    'action_type': 'federated_poisoning',
                    'success': False,
                    'impact': 12.0,
                    'detected': True,
                    'stealth_level': 0.3
                }
            ]
        }
    ]

def main():
    """Test visualization fixes"""
    print("="*60)
    print(" TESTING VISUALIZATION FIXES")
    print("="*60)
    
    try:
        # Create system instance
        config = {
            'hierarchical': {'num_distribution_systems': 2},
            'attack': {'max_episodes': 2}
        }
        
        system = IntegratedEVCSLLMRLSystem(config)
        
        # Set mock episode results
        system.simulation_results = {
            'episode_results': create_mock_episode_results(),
            'scenario': IntegratedAttackScenario(
                scenario_id="TEST_001",
                name="Visualization Test",
                description="Test attack visualization",
                target_systems=[1, 2],
                attack_duration=30.0,
                stealth_requirement=0.7,
                impact_goal=0.6,
                constraints={}
            )
        }
        
        print(" Testing attack execution data extraction...")
        attack_data = system._extract_attack_execution_details(
            system.simulation_results['episode_results']
        )
        
        print(f"  Attack Types: {attack_data['attack_types']}")
        print(f"  CMS Disruptions: {len(attack_data['cms_disruptions'])} data points")
        print(f"  PINN Impacts: {len(attack_data['pinn_impacts'])} data points")
        
        print(" Testing detection mechanism data extraction...")
        detection_data = system._extract_detection_mechanism_details(
            system.simulation_results['episode_results']
        )
        
        print(f"  Detection Types: {detection_data['detection_types']}")
        print(f"  Stealth Levels: {len(detection_data['stealth_levels'])} data points")
        print(f"  Detection Episodes: {detection_data['detection_episodes']}")
        
        print(" Testing visualization generation...")
        system._create_integrated_visualizations()
        
        print("✓ Visualization fixes working correctly!")
        print("  Check for new detailed_attack_analysis_*.png file")
        
        return 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
