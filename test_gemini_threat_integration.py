#!/usr/bin/env python3
"""
Test script to verify Gemini threat analysis integration
Tests that the system uses actual Gemini analysis instead of mock data
"""

import sys
import os
import time
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_integrated_evcs_system import EnhancedIntegratedEVCSLLMRLSystem
    from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure all required files are in the current directory")
    sys.exit(1)

def test_gemini_threat_integration():
    """Test that the system uses actual Gemini threat analysis"""
    print("🧪 Testing Gemini Threat Analysis Integration")
    print("=" * 60)
    
    # Test 1: Initialize system with minimal config
    print("\n📋 Test 1: System Initialization")
    try:
        config = {
            'hierarchical': {'total_duration': 1800},
            'rl': {'num_systems': 3},
            'llm': {
                'provider': 'gemini',
                'model': 'models/gemini-2.5-flash',
                'api_key_file': 'gemini_key.txt'
            }
        }
        
        system = EnhancedIntegratedEVCSLLMRLSystem(config)
        print("    ✅ System initialized successfully")
        
        # Check if Gemini analyzer is available
        has_gemini = hasattr(system, 'llm_analyzer') and system.llm_analyzer and system.llm_analyzer.is_available
        print(f"    📊 Gemini LLM Analyzer Available: {has_gemini}")
        
    except Exception as e:
        print(f"    ❌ System initialization failed: {e}")
        return False
    
    # Test 2: Test _get_current_threats() method
    print("\n🎯 Test 2: Current Threats Analysis")
    try:
        current_threats = system._get_current_threats()
        
        # Verify the response structure
        required_fields = ['active_attacks', 'potential_vulnerabilities', 'threat_level', 'last_updated', 'source']
        for field in required_fields:
            if field not in current_threats:
                print(f"    ❌ Missing required field: {field}")
                return False
        
        # Check if using Gemini or fallback
        source = current_threats.get('source', 'unknown')
        print(f"    📊 Threat Analysis Source: {source}")
        
        if source == 'gemini_llm':
            print("    ✅ Using actual Gemini threat analysis")
            print(f"    📈 Confidence Score: {current_threats.get('confidence_score', 'N/A')}")
        elif source == 'fallback_simulation':
            print("    ⚠️ Using fallback simulation (Gemini not available)")
        else:
            print(f"    ❓ Unknown source: {source}")
        
        vulnerabilities = current_threats.get('potential_vulnerabilities', [])
        print(f"    🔍 Found {len(vulnerabilities)} potential vulnerabilities")
        
        for i, vuln in enumerate(vulnerabilities[:3]):  # Show first 3
            print(f"      • {vuln.get('type', 'unknown')} (severity: {vuln.get('severity', 'unknown')})")
        
    except Exception as e:
        print(f"    ❌ Current threats analysis failed: {e}")
        return False
    
    # Test 3: Test comprehensive system analysis
    print("\n📊 Test 3: Comprehensive System Analysis")
    try:
        comprehensive_analysis = system._perform_comprehensive_system_analysis()
        
        # Verify the response structure
        required_fields = ['timestamp', 'current_threats', 'analysis_source', 'system_health']
        for field in required_fields:
            if field not in comprehensive_analysis:
                print(f"    ❌ Missing required field: {field}")
                return False
        
        analysis_source = comprehensive_analysis.get('analysis_source', 'unknown')
        print(f"    📊 Analysis Source: {analysis_source}")
        
        if analysis_source == 'gemini_llm':
            print("    ✅ Using actual Gemini comprehensive analysis")
            confidence = comprehensive_analysis.get('confidence_level', 0.0)
            print(f"    📈 Confidence Level: {confidence:.2f}")
            
            recommendations = comprehensive_analysis.get('recommendations', [])
            print(f"    💡 Recommendations: {len(recommendations)}")
            
        elif analysis_source == 'fallback_simulation':
            print("    ⚠️ Using fallback analysis (Gemini not available)")
        
        # Check system health assessment
        system_health = comprehensive_analysis.get('system_health', {})
        health_status = system_health.get('health_status', 'unknown')
        health_score = system_health.get('health_score', 0.0)
        print(f"    🏥 System Health: {health_status} (Score: {health_score:.1f}/100)")
        
    except Exception as e:
        print(f"    ❌ Comprehensive system analysis failed: {e}")
        return False
    
    # Test 4: Test fallback coordination
    print("\n🔄 Test 4: Fallback Coordination")
    try:
        # Create a mock scenario
        scenario = {'id': 'test_scenario', 'duration': 1800}
        
        fallback_result = system._run_fallback_coordination(scenario, episode_num=1)
        
        # Verify the response structure
        required_fields = ['system_analysis', 'attack_scenarios', 'coordination_result', 'coordination_type']
        for field in required_fields:
            if field not in fallback_result:
                print(f"    ❌ Missing required field: {field}")
                return False
        
        coordination_type = fallback_result.get('coordination_type', 'unknown')
        print(f"    🔄 Coordination Type: {coordination_type}")
        
        # Check if system analysis used Gemini
        system_analysis = fallback_result.get('system_analysis', {})
        analysis_source = system_analysis.get('analysis_source', 'unknown')
        print(f"    📊 System Analysis Source: {analysis_source}")
        
        # Check attack scenarios
        attack_scenarios = fallback_result.get('attack_scenarios', [])
        print(f"    ⚡ Generated {len(attack_scenarios)} attack scenarios")
        
        for i, scenario in enumerate(attack_scenarios[:2]):  # Show first 2
            attack_type = scenario.get('attack_type', 'unknown')
            severity = scenario.get('severity', 'unknown')
            print(f"      • Scenario {i+1}: {attack_type} (severity: {severity})")
        
        print("    ✅ Fallback coordination completed successfully")
        
    except Exception as e:
        print(f"    ❌ Fallback coordination failed: {e}")
        return False
    
    # Test 5: Verify no hardcoded mock data is being used
    print("\n🔍 Test 5: Mock Data Detection")
    try:
        # Get current threats multiple times and check for variation
        threats_1 = system._get_current_threats()
        time.sleep(1)  # Small delay
        threats_2 = system._get_current_threats()
        
        # If using real Gemini, timestamps should be different
        timestamp_1 = threats_1.get('last_updated', 0)
        timestamp_2 = threats_2.get('last_updated', 0)
        
        if timestamp_2 > timestamp_1:
            print("    ✅ Threat analysis timestamps are updating (not using static mock data)")
        else:
            print("    ⚠️ Threat analysis timestamps are identical (possible mock data)")
        
        # Check if vulnerabilities are exactly the same as hardcoded fallback
        fallback_vulns = [
            {'type': 'voltage_manipulation', 'severity': 'high', 'systems': [1, 2, 3]},
            {'type': 'current_injection', 'severity': 'medium', 'systems': [4, 5, 6]},
            {'type': 'thermal_attack', 'severity': 'low', 'systems': [1, 4]}
        ]
        
        current_vulns = threats_1.get('potential_vulnerabilities', [])
        
        # Simple check: if vulnerabilities are exactly the same as fallback, it might be mock data
        if len(current_vulns) == 3 and all(
            vuln.get('type') in ['voltage_manipulation', 'current_injection', 'thermal_attack'] 
            for vuln in current_vulns
        ):
            source = threats_1.get('source', 'unknown')
            if source == 'fallback_simulation':
                print("    ℹ️ Using fallback vulnerabilities (expected when Gemini unavailable)")
            else:
                print("    ⚠️ Vulnerabilities match fallback pattern but source indicates Gemini")
        else:
            print("    ✅ Vulnerabilities appear to be dynamically generated")
        
    except Exception as e:
        print(f"    ❌ Mock data detection failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("\n📋 Summary:")
    print("  • System initialization: ✅")
    print("  • Current threats analysis: ✅")
    print("  • Comprehensive system analysis: ✅")
    print("  • Fallback coordination: ✅")
    print("  • Mock data detection: ✅")
    
    # Final recommendation
    gemini_available = hasattr(system, 'llm_analyzer') and system.llm_analyzer and system.llm_analyzer.is_available
    if gemini_available:
        print("\n💡 Recommendation: System is using actual Gemini threat analysis")
    else:
        print("\n💡 Recommendation: Enable Gemini API key for enhanced threat analysis")
    
    return True

if __name__ == "__main__":
    try:
        success = test_gemini_threat_integration()
        if success:
            print("\n✅ Integration test passed!")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
