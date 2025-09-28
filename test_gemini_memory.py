#!/usr/bin/env python3
"""
Test script to demonstrate Gemini LLM conversation memory and context awareness
"""

from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer
import json
import time

def test_conversation_memory():
    """Test conversation memory functionality"""
    print("🧠 Testing Gemini LLM Conversation Memory")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GeminiLLMThreatAnalyzer(model_name="models/gemini-2.5-flash", max_history=5)
    
    if not analyzer.is_available:
        print("❌ Gemini not available for testing")
        return
    
    print("✅ Gemini analyzer initialized with conversation memory")
    
    # Test 1: Initial vulnerability analysis
    print("\n📊 Test 1: Initial vulnerability analysis")
    test_evcs_state_1 = {
        'charging_stations': 6,
        'active_sessions': 12,
        'grid_frequency': 59.8,
        'system_load': 850.5,
        'voltage_issues': True
    }
    
    test_config_1 = {
        'max_power': 1000,
        'voltage_range': [0.95, 1.05]
    }
    
    result1 = analyzer.analyze_evcs_vulnerabilities(test_evcs_state_1, test_config_1)
    print(f"Found {len(result1.get('vulnerabilities', []))} vulnerabilities")
    
    # Test 2: Follow-up analysis (should reference previous findings)
    print("\n🔄 Test 2: Follow-up analysis (should reference previous)")
    time.sleep(1)  # Small delay to differentiate timestamps
    
    test_evcs_state_2 = {
        'charging_stations': 6,
        'active_sessions': 8,  # Reduced sessions
        'grid_frequency': 60.1,  # Improved frequency
        'system_load': 750.0,   # Reduced load
        'voltage_issues': False  # Fixed voltage issues
    }
    
    result2 = analyzer.analyze_evcs_vulnerabilities(test_evcs_state_2, test_config_1)
    print(f"Found {len(result2.get('vulnerabilities', []))} vulnerabilities in improved state")
    
    # Test 3: Attack strategy generation (should build on vulnerability knowledge)
    print("\n⚔️ Test 3: Attack strategy generation (should build on vuln knowledge)")
    
    # Create mock vulnerabilities based on previous analysis
    from gemini_llm_threat_analyzer import EVCSVulnerability
    mock_vulnerabilities = []
    
    for i, vuln in enumerate(result1.get('vulnerabilities', [])[:3]):  # Limit to 3 for testing
        mock_vuln = EVCSVulnerability(
            vuln_id=f"VULN_{i+1}",
            component=vuln.get('component', 'charging_controller'),
            vulnerability_type=vuln.get('type', 'authentication'),
            severity=vuln.get('severity', 0.7),
            exploitability=0.8,
            impact=0.9,
            cvss_score=vuln.get('cvss_score', 7.5),
            mitigation="Enhanced authentication protocols",
            detection_methods=["Anomaly detection", "Log analysis"]
        )
        mock_vulnerabilities.append(mock_vuln)
    
    strategy_result = analyzer.generate_attack_strategy(
        vulnerabilities=mock_vulnerabilities,
        evcs_state=test_evcs_state_2,
        constraints={'stealth_requirement': 'high', 'time_limit': '24h'}
    )
    
    print(f"Generated strategy: {strategy_result.get('strategy_name', 'Unknown')}")
    print(f"Success probability: {strategy_result.get('success_probability', 0.0):.2f}")
    
    # Test 4: Show conversation summary
    print("\n📈 Test 4: Conversation summary and learning")
    summary = analyzer.get_conversation_summary()
    
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Vulnerabilities analyzed: {summary['learning_summary']['total_vulnerabilities_analyzed']}")
    print(f"Strategies generated: {summary['learning_summary']['total_strategies_generated']}")
    
    if summary['learning_summary']['vulnerability_patterns']:
        print(f"Common vulnerability patterns: {summary['learning_summary']['vulnerability_patterns']}")
    
    # Test 5: Another analysis to show context evolution
    print("\n🔍 Test 5: Third analysis (should show pattern recognition)")
    time.sleep(1)
    
    test_evcs_state_3 = {
        'charging_stations': 8,  # More stations
        'active_sessions': 20,   # High load
        'grid_frequency': 59.5,  # Low frequency
        'system_load': 950.0,    # High load
        'voltage_issues': True,  # Voltage problems again
        'new_attack_detected': True  # New factor
    }
    
    result3 = analyzer.analyze_evcs_vulnerabilities(test_evcs_state_3, test_config_1)
    print(f"Found {len(result3.get('vulnerabilities', []))} vulnerabilities with high load")
    
    # Final summary
    print("\n📊 Final conversation summary:")
    final_summary = analyzer.get_conversation_summary()
    print(json.dumps(final_summary, indent=2, default=str))
    
    print("\n✅ Conversation memory test completed!")
    print("🧠 The analyzer now has context from all interactions and can:")
    print("  - Reference previous vulnerability findings")
    print("  - Build attack strategies based on historical analysis")
    print("  - Recognize patterns across multiple sessions")
    print("  - Provide contextual continuity in analysis")

if __name__ == "__main__":
    test_conversation_memory()
