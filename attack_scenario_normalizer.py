#!/usr/bin/env python3
"""
Attack Scenario Normalizer
Fixes inconsistencies in Gemini-generated attack scenarios by aligning TYPE and ATTACK_TYPE 
fields with the actual attack types described in COMBINED_ATTACK_TYPES and SCENARIO_NAME.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class AttackScenarioNormalizer:
    def __init__(self):
        # Mapping of combined attack types to primary attack type
        self.attack_type_mapping = {
            'voltage_manipulation': 'voltage_manipulation',
            'model_poisoning': 'cyber_attack',
            'thermal_attack': 'thermal_attack',
            'power_disruption': 'power_manipulation',
            'charging_hijacking': 'cyber_attack',
            'load_manipulation': 'load_manipulation',
            'power_manipulation': 'power_manipulation',
            'frequency_manipulation': 'frequency_manipulation',
            'communication_jamming': 'cyber_attack',
            'data_injection': 'cyber_attack'
        }
        
        # Priority order for determining primary attack type when multiple types exist
        self.attack_priority = [
            'cyber_attack',           # Highest priority - most sophisticated
            'thermal_attack',         # Physical attacks
            'voltage_manipulation',   # Grid stability attacks
            'power_manipulation',     # Power system attacks
            'load_manipulation',      # Load-based attacks
            'frequency_manipulation'  # Frequency attacks
        ]

    def parse_attack_scenario_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse the attack scenario file and extract individual scenarios."""
        scenarios = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by attack scenario headers
        scenario_blocks = re.split(r'ATTACK SCENARIO #\d+', content)[1:]  # Skip header
        
        for i, block in enumerate(scenario_blocks, 1):
            scenario = self._parse_single_scenario(block, i)
            if scenario:
                scenarios.append(scenario)
        
        return scenarios

    def _parse_single_scenario(self, block: str, scenario_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single attack scenario block."""
        scenario = {'scenario_number': scenario_num}
        
        # Extract all key-value pairs
        lines = block.strip().split('\n')
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-'):
                continue
                
            if ':' in line and not line.startswith(' '):
                # Save previous key-value pair
                if current_key:
                    scenario[current_key] = self._process_value(current_value)
                
                # Start new key-value pair
                key, value = line.split(':', 1)
                current_key = key.strip()
                current_value = [value.strip()]
            else:
                # Continuation of previous value
                if current_key:
                    current_value.append(line)
        
        # Save last key-value pair
        if current_key:
            scenario[current_key] = self._process_value(current_value)
        
        return scenario if len(scenario) > 1 else None

    def _process_value(self, value_lines: List[str]) -> Any:
        """Process value lines into appropriate data type."""
        value_str = ' '.join(value_lines).strip()
        
        if not value_str:
            return None
        
        # Handle lists
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                # Clean up the list format
                clean_str = value_str.replace('\n', '').replace('  ', ' ')
                return eval(clean_str)  # Safe for our controlled input
            except:
                return value_str
        
        # Handle numbers
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Handle booleans
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        return value_str

    def normalize_attack_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single attack scenario to fix inconsistencies."""
        normalized = scenario.copy()
        
        # Get combined attack types
        combined_types = scenario.get('COMBINED_ATTACK_TYPES', [])
        if isinstance(combined_types, str):
            combined_types = [combined_types]
        
        # Determine the primary attack type based on combined types
        primary_type = self._determine_primary_attack_type(combined_types)
        
        # Update TYPE and ATTACK_TYPE to match the primary type
        if primary_type:
            normalized['TYPE'] = primary_type
            normalized['ATTACK_TYPE'] = primary_type
            
            # Add explanation for the change
            normalized['NORMALIZATION_NOTES'] = f"Updated TYPE from '{scenario.get('TYPE', 'unknown')}' to '{primary_type}' based on COMBINED_ATTACK_TYPES: {combined_types}"
        
        # Ensure consistency in attack parameters based on type
        normalized = self._adjust_attack_parameters(normalized, primary_type, combined_types)
        
        return normalized

    def _determine_primary_attack_type(self, combined_types: List[str]) -> str:
        """Determine the primary attack type from combined attack types."""
        if not combined_types:
            return 'power_manipulation'  # Default fallback
        
        # Map combined types to our standard types
        mapped_types = []
        for attack_type in combined_types:
            mapped_type = self.attack_type_mapping.get(attack_type, attack_type)
            if mapped_type not in mapped_types:
                mapped_types.append(mapped_type)
        
        # If only one type, use it
        if len(mapped_types) == 1:
            return mapped_types[0]
        
        # If multiple types, use priority order
        for priority_type in self.attack_priority:
            if priority_type in mapped_types:
                return priority_type
        
        # Fallback to first mapped type
        return mapped_types[0] if mapped_types else 'power_manipulation'

    def _adjust_attack_parameters(self, scenario: Dict[str, Any], primary_type: str, combined_types: List[str]) -> Dict[str, Any]:
        """Adjust attack parameters based on the primary attack type."""
        adjusted = scenario.copy()
        
        # Adjust impact factors based on attack complexity
        complexity_multiplier = 1.0 + (len(combined_types) - 1) * 0.1  # 10% increase per additional attack type
        
        if 'IMPACT_FACTOR' in adjusted:
            adjusted['IMPACT_FACTOR'] = min(1.0, adjusted['IMPACT_FACTOR'] * complexity_multiplier)
        
        # Adjust success rate based on attack type complexity
        if primary_type == 'cyber_attack':
            # Cyber attacks are more complex, slightly lower success rate
            if 'SUCCESS_RATE' in adjusted:
                adjusted['SUCCESS_RATE'] = max(0.6, adjusted['SUCCESS_RATE'] * 0.95)
        elif primary_type == 'thermal_attack':
            # Physical attacks are more reliable but detectable
            if 'STEALTH_LEVEL' in adjusted:
                adjusted['STEALTH_LEVEL'] = max(0.7, adjusted['STEALTH_LEVEL'] * 0.9)
        
        # Add attack type specific parameters
        adjusted['PRIMARY_ATTACK_TYPE'] = primary_type
        adjusted['ATTACK_COMPLEXITY'] = len(combined_types)
        adjusted['MULTI_VECTOR_ATTACK'] = len(combined_types) > 1
        
        return adjusted

    def generate_normalized_file(self, scenarios: List[Dict[str, Any]], output_path: str):
        """Generate a normalized attack scenarios file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_path, 'w') as f:
            f.write("NORMALIZED Attack Scenarios Generated by GEMINI\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Original scenarios: {len(scenarios)}\n")
            f.write("Normalization: Fixed TYPE/ATTACK_TYPE inconsistencies\n")
            f.write("=" * 80 + "\n\n")
            
            for i, scenario in enumerate(scenarios, 1):
                f.write(f"ATTACK SCENARIO #{i}\n")
                f.write("-" * 40 + "\n")
                
                # Write key fields in logical order
                key_order = [
                    'TYPE', 'ATTACK_TYPE', 'PRIMARY_ATTACK_TYPE',
                    'TARGET_SYSTEM', 'TARGET_SYSTEMS',
                    'SCENARIO_NAME', 'COORDINATION_TYPE', 'COMBINED_ATTACK_TYPES',
                    'STRATEGIC_GOAL',
                    'ATTACK_MAGNITUDE', 'STEALTH_LEVEL', 'STEALTH_FACTOR',
                    'IMPACT_FACTOR', 'SUCCESS_RATE', 'ATTACK_COMPLEXITY', 'MULTI_VECTOR_ATTACK',
                    'VOLTAGE_DEVIATION', 'FREQUENCY_DEVIATION', 'POWER_LOSS', 'LOAD_DISRUPTION',
                    'VOLTAGE_DROP_FACTOR', 'POWER_REDUCTION_FACTOR', 'FREQUENCY_IMPACT',
                    'START_TIME', 'DURATION',
                    'MAGNITUDE', 'ACTIVE', 'GEMINI_OPTIMIZED',
                    'NORMALIZATION_NOTES'
                ]
                
                # Write ordered fields
                for key in key_order:
                    if key in scenario:
                        value = scenario[key]
                        if isinstance(value, list):
                            f.write(f"{key}: [\n")
                            for item in value:
                                f.write(f"  {repr(item)}\n")
                            f.write("]\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                # Write any remaining fields
                for key, value in scenario.items():
                    if key not in key_order and key != 'scenario_number':
                        if isinstance(value, list):
                            f.write(f"{key}: [\n")
                            for item in value:
                                f.write(f"  {repr(item)}\n")
                            f.write("]\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
            
            f.write(f"\nTotal normalized scenarios: {len(scenarios)}\n")
            f.write(f"File generated at: {timestamp}\n")

    def generate_summary_report(self, original_scenarios: List[Dict[str, Any]], 
                              normalized_scenarios: List[Dict[str, Any]]) -> str:
        """Generate a summary report of the normalization changes."""
        report = []
        report.append("ATTACK SCENARIO NORMALIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Total scenarios processed: {len(original_scenarios)}")
        report.append("")
        
        changes_made = 0
        for i, (orig, norm) in enumerate(zip(original_scenarios, normalized_scenarios), 1):
            orig_type = orig.get('TYPE', 'unknown')
            norm_type = norm.get('TYPE', 'unknown')
            
            if orig_type != norm_type:
                changes_made += 1
                combined_types = norm.get('COMBINED_ATTACK_TYPES', [])
                report.append(f"Scenario #{i}: {orig.get('SCENARIO_NAME', 'Unknown')}")
                report.append(f"  Changed TYPE: '{orig_type}' → '{norm_type}'")
                report.append(f"  Based on COMBINED_ATTACK_TYPES: {combined_types}")
                report.append(f"  Complexity: {norm.get('ATTACK_COMPLEXITY', 1)} attack vectors")
                report.append("")
        
        report.append(f"Total changes made: {changes_made}")
        report.append(f"Scenarios unchanged: {len(original_scenarios) - changes_made}")
        
        return "\n".join(report)


def main():
    """Main function to normalize attack scenarios."""
    input_file = "/Users/mohammadzakariahaider/Dropbox/Mohammad Zakaria Haider/PHD 2023_28/1. Fall 25/EEL_6905_Individual_Study/Week-1 (DoE_Presentations)/Agentic AI_working_model_shellhacks_main/attack_scenarios_logs/gemini_attack_scenarios_20251012_021357.txt"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/Users/mohammadzakariahaider/Dropbox/Mohammad Zakaria Haider/PHD 2023_28/1. Fall 25/EEL_6905_Individual_Study/Week-1 (DoE_Presentations)/Agentic AI_working_model_shellhacks_main/attack_scenarios_logs/normalized_gemini_attack_scenarios_{timestamp}.txt"
    report_file = f"/Users/mohammadzakariahaider/Dropbox/Mohammad Zakaria Haider/PHD 2023_28/1. Fall 25/EEL_6905_Individual_Study/Week-1 (DoE_Presentations)/Agentic AI_working_model_shellhacks_main/attack_scenarios_logs/normalization_report_{timestamp}.txt"
    
    normalizer = AttackScenarioNormalizer()
    
    print("🔄 Parsing original attack scenarios...")
    original_scenarios = normalizer.parse_attack_scenario_file(input_file)
    print(f"✅ Found {len(original_scenarios)} scenarios")
    
    print("🔧 Normalizing attack scenarios...")
    normalized_scenarios = []
    for scenario in original_scenarios:
        normalized = normalizer.normalize_attack_scenario(scenario)
        normalized_scenarios.append(normalized)
    
    print("📝 Generating normalized file...")
    normalizer.generate_normalized_file(normalized_scenarios, output_file)
    
    print("📊 Generating summary report...")
    report = normalizer.generate_summary_report(original_scenarios, normalized_scenarios)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Normalization complete!")
    print(f"📁 Normalized scenarios: {output_file}")
    print(f"📋 Summary report: {report_file}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    main()
