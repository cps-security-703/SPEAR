#!/usr/bin/env python3
"""
Test script to verify RL feedback loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_llm_rl_coordinator import EnhancedLLMRLCoordinator

def test_rl_feedback_loading():
    """Test the RL feedback loading functionality"""
    print("🧪 Testing RL Feedback Loading Functionality")
    print("=" * 60)
    
    try:
        # Initialize the coordinator with mock dependencies
        coordinator = EnhancedLLMRLCoordinator(
            llm_analyzer=None,
            rl_coordinator=None, 
            hierarchical_sim=None,
            federated_manager=None
        )
        
        # Test the RL feedback loading method
        print("\n📖 Testing _load_latest_rl_feedback_data method...")
        rl_attacks = coordinator._load_latest_rl_feedback_data()
        
        if rl_attacks:
            print(f"✅ Successfully loaded {len(rl_attacks)} RL attack suggestions")
            print("\n📋 Sample RL attacks loaded:")
            for i, attack in enumerate(rl_attacks[:3]):  # Show first 3
                print(f"  Attack {i+1}:")
                print(f"    Type: {attack.get('attack_type', 'N/A')}")
                print(f"    Target System: {attack.get('target_system', 'N/A')}")
                print(f"    Magnitude: {attack.get('magnitude', 'N/A')}")
                print(f"    Stealth: {attack.get('stealth', 'N/A')}")
                print(f"    Success: {attack.get('success', 'N/A')}")
                print(f"    Impact: {attack.get('impact', 'N/A')}")
                print()
            
            # Test if we can use these for Gemini optimization
            print("🧠 Testing Gemini strategic combination with actual RL data...")
            try:
                llm_response = coordinator._gemini_strategic_attack_combination(rl_attacks[:5], 3600.0, 6)
                print(f"✅ Gemini strategic combination successful!")
                print(f"   Response type: {type(llm_response)}")
                if isinstance(llm_response, list):
                    print(f"   Generated {len(llm_response)} optimized scenarios")
                    if len(llm_response) > 0:
                        sample_scenario = llm_response[0]
                        print(f"   Sample optimized scenario keys: {list(sample_scenario.keys()) if isinstance(sample_scenario, dict) else 'Not a dict'}")
                else:
                    print(f"   Response preview: {str(llm_response)[:200]}...")
                    
            except Exception as e:
                print(f"⚠️ Gemini strategic combination failed: {e}")
                
        else:
            print("❌ No RL attack data loaded - this indicates the issue is not fixed")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rl_feedback_loading()
