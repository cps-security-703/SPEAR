#!/usr/bin/env python3
"""
Demo script showing LangGraph integration with EVCS system
Enhanced attack coordination with better separation of concerns
"""

import sys
import time
from typing import Dict, Any

def main():
    """Demonstrate LangGraph enhanced attack coordination"""
    print("="*80)
    print(" LANGGRAPH ENHANCED EVCS ATTACK COORDINATION DEMO")
    print("="*80)
    
    # Check if LangGraph is available
    try:
        from langgraph_attack_coordinator import LangGraphAttackCoordinator, AttackState
        print("✓ LangGraph coordinator available")
        langgraph_available = True
    except ImportError as e:
        print(f"✗ LangGraph coordinator not available: {e}")
        print("  Install with: pip install -r requirements_langgraph.txt")
        langgraph_available = False
    
    # Initialize integrated system with LangGraph
    try:
        from integrated_evcs_llm_rl_system import IntegratedEVCSLLMRLSystem
        
        # Configuration optimized for LangGraph demo
        config = {
            'hierarchical': {
                'use_enhanced_pinn': True,
                'use_dqn_sac_security': True,
                'total_duration': 30.0,  # Short demo
                'num_distribution_systems': 3  # Reduced for demo
            },
            'llm': {
                'base_url': 'http://localhost:11434/v1',
                'model': 'deepseek-r1:8b'
            },
            'rl': {
                'state_dim': 20,  # Reduced for demo
                'action_dim': 4,
                'num_coordinator_agents': 2
            },
            'attack': {
                'max_episodes': 2,  # Short demo
                'max_steps_per_episode': 3
            }
        }
        
        print("\n Initializing Integrated EVCS System with LangGraph...")
        system = IntegratedEVCSLLMRLSystem(config)
        
        # Show workflow visualization if LangGraph is available
        if system.langgraph_coordinator:
            print("\n" + "="*60)
            print(" LANGGRAPH WORKFLOW VISUALIZATION")
            print("="*60)
            print(system.langgraph_coordinator.get_workflow_visualization())
        
        # Run demo simulation
        print("\n" + "="*60)
        print(" RUNNING DEMO SIMULATION")
        print("="*60)
        
        results = system.run_integrated_simulation(
            scenario_id="INTEGRATED_001",
            episodes=2,
            max_wall_time_sec=30
        )
        
        # Display results
        print("\n" + "="*60)
        print(" DEMO RESULTS")
        print("="*60)
        
        performance = results.get('performance_metrics', {})
        print(f"Average Reward: {performance.get('average_reward', 0.0):.2f}")
        print(f"Success Rate: {performance.get('average_success_rate', 0.0):.1%}")
        print(f"Detection Rate: {performance.get('average_detection_rate', 0.0):.1%}")
        
        # Show LangGraph specific metrics
        episode_results = results.get('episode_results', [])
        langgraph_episodes = [ep for ep in episode_results if ep.get('langgraph_enhanced', False)]
        
        if langgraph_episodes:
            print(f"\n LangGraph Enhanced Episodes: {len(langgraph_episodes)}/{len(episode_results)}")
            
            for i, episode in enumerate(langgraph_episodes):
                print(f"\n Episode {episode.get('episode', i+1)} (LangGraph Enhanced):")
                print(f"  Stealth Score: {episode.get('stealth_score', 0.0):.3f}")
                print(f"  Impact Score: {episode.get('impact_score', 0.0):.2f}")
                print(f"  Workflow Phases: {len(episode.get('workflow_phases', []))}")
                
                # Show debug info
                debug_info = episode.get('debug_info', [])
                if debug_info:
                    print(f"  Debug Info:")
                    for info in debug_info[-3:]:  # Show last 3 debug messages
                        print(f"    - {info}")
        
        # Show workflow benefits
        print("\n" + "="*60)
        print(" LANGGRAPH WORKFLOW BENEFITS DEMONSTRATED")
        print("="*60)
        
        benefits = [
            "✓ Better separation of concerns (LLM strategic vs RL tactical)",
            "✓ Improved state management across attack phases",
            "✓ Enhanced debugging and monitoring capabilities",
            "✓ Flexible routing between attack phases",
            "✓ Conditional adaptation based on performance",
            "✓ Persistent workflow state and history"
        ]
        
        for benefit in benefits:
            print(f"  {benefit}")
        
        # Installation instructions
        if not langgraph_available:
            print("\n" + "="*60)
            print(" TO ENABLE FULL LANGGRAPH FEATURES")
            print("="*60)
            print("  1. Install LangGraph dependencies:")
            print("     pip install -r requirements_langgraph.txt")
            print("  2. Re-run this demo script")
            print("  3. Observe enhanced workflow coordination")
        
        print("\n" + "="*80)
        print(" DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n Demo failed: {e}")
        print("  Please check that all dependencies are installed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
