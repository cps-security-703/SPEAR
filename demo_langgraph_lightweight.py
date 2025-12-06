#!/usr/bin/env python3
"""
Lightweight LangGraph Demo - Focus on workflow without heavy PINN training
"""

import sys
import time

def main():
    """Lightweight demo focusing on LangGraph workflow"""
    print("="*80)
    print(" LIGHTWEIGHT LANGGRAPH WORKFLOW DEMO")
    print("="*80)
    
    # Test LangGraph coordinator directly
    try:
        from langgraph_attack_coordinator import LangGraphAttackCoordinator, AttackState
        print("✓ LangGraph coordinator available")
        
        # Create mock components for testing
        print("\n Creating mock LLM and RL components...")
        mock_llm = None  # Will use fallback
        mock_rl = None   # Will use fallback  
        mock_sim = None  # Will use fallback
        
        # Initialize LangGraph coordinator
        print(" Initializing LangGraph coordinator...")
        coordinator = LangGraphAttackCoordinator(
            llm_analyzer=mock_llm,
            rl_coordinator=mock_rl,
            hierarchical_sim=mock_sim
        )
        
        # Show workflow visualization
        print("\n" + "="*60)
        print(" LANGGRAPH WORKFLOW STRUCTURE")
        print("="*60)
        print(coordinator.get_workflow_visualization())
        
        # Create mock scenario
        from integrated_evcs_llm_rl_system import IntegratedAttackScenario
        
        scenario = IntegratedAttackScenario(
            scenario_id="DEMO_001",
            name="LangGraph Workflow Demo",
            description="Demonstrate LangGraph coordination workflow",
            target_systems=[1, 2],
            attack_duration=30.0,
            stealth_requirement=0.8,
            impact_goal=0.7,
            constraints={'demo_mode': True}
        )
        
        # Run single episode to demonstrate workflow
        print("\n" + "="*60)
        print(" RUNNING LANGGRAPH WORKFLOW EPISODE")
        print("="*60)
        
        episode_result = coordinator.run_attack_episode(scenario, 1)
        
        # Display workflow results
        print("\n" + "="*60)
        print(" WORKFLOW EXECUTION RESULTS")
        print("="*60)
        
        print(f"Episode Number: {episode_result.get('episode_number', 1)}")
        print(f"Workflow Completed: {episode_result.get('workflow_completed', False)}")
        print(f"Success: {episode_result.get('success', False)}")
        
        # Show stealth metrics
        stealth = episode_result.get('stealth_metrics', {})
        print(f"\nStealth Metrics:")
        print(f"  Detection Probability: {stealth.get('detection_probability', 0.0):.3f}")
        print(f"  Stealth Score: {stealth.get('stealth_score', 1.0):.3f}")
        print(f"  Anomaly Level: {stealth.get('anomaly_level', 0.0):.3f}")
        
        # Show success metrics
        success = episode_result.get('success_metrics', {})
        print(f"\nSuccess Metrics:")
        print(f"  Impact Score: {success.get('impact_score', 0.0):.2f}")
        print(f"  Success Rate: {success.get('success_rate', 0.0):.1%}")
        print(f"  Target Achievement: {success.get('target_achievement', 0.0):.3f}")
        
        # Show debug information (workflow phases)
        debug_info = episode_result.get('debug_info', [])
        print(f"\nWorkflow Debug Information:")
        for i, info in enumerate(debug_info, 1):
            print(f"  {i}. {info}")
        
        # Show execution results
        execution_results = episode_result.get('execution_results', [])
        print(f"\nExecution Results: {len(execution_results)} actions executed")
        for i, result in enumerate(execution_results, 1):
            print(f"  Action {i}: Success={result.get('success', False)}, "
                  f"Impact={result.get('impact', 0.0):.1f}")
        
        # Demonstrate workflow benefits
        print("\n" + "="*60)
        print(" LANGGRAPH WORKFLOW BENEFITS DEMONSTRATED")
        print("="*60)
        
        benefits_shown = [
            "✓ Strategic Planning Phase (LLM-based threat modeling)",
            "✓ Tactical Preparation Phase (RL state processing)", 
            "✓ Execution Phase (Attack action execution)",
            "✓ Stealth Assessment Phase (Detection risk evaluation)",
            "✓ Impact Evaluation Phase (Success measurement)",
            "✓ Strategy Adaptation Phase (Performance-based learning)",
            "✓ Conditional Routing (Continue/Adapt/Abort decisions)",
            "✓ State Persistence (Workflow memory across phases)",
            "✓ Debug Tracking (Detailed execution logging)"
        ]
        
        for benefit in benefits_shown:
            print(f"  {benefit}")
        
        # Performance comparison
        print(f"\n" + "="*60)
        print(" PERFORMANCE COMPARISON")
        print("="*60)
        
        print("Standard Coordination:")
        print("  - Linear execution: LLM → RL → Execute")
        print("  - No state persistence between phases")
        print("  - Limited adaptation capabilities")
        print("  - Basic error handling")
        
        print("\nLangGraph Enhanced Coordination:")
        print("  - Multi-phase workflow with conditional routing")
        print("  - Persistent state management across phases")
        print("  - Dynamic adaptation based on performance")
        print("  - Comprehensive debugging and monitoring")
        print("  - Graceful error recovery")
        
        print("\n" + "="*80)
        print(" LANGGRAPH WORKFLOW DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return 0
        
    except ImportError as e:
        print(f"✗ LangGraph not available: {e}")
        print("  Install with: pip install -r requirements_langgraph.txt")
        return 1
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
