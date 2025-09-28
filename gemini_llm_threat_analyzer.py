#!/usr/bin/env python3
"""
Gemini-based LLM Threat Analyzer for EVCS Attack Analytics
Replaces Ollama deepseek with Google's Gemini Pro
"""

import google.generativeai as genai
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

# Try to import existing classes for compatibility
try:
    from llm_guided_evcs_attack_analytics import EVCSVulnerability, STRIDEMITREThreatMapper
except ImportError:
    print("Warning: Could not import EVCSVulnerability. Creating minimal version.")
    
    @dataclass
    class EVCSVulnerability:
        """EVCS vulnerability data class"""
        vuln_id: str
        component: str
        vulnerability_type: str
        severity: float
        exploitability: float
        impact: float
        cvss_score: float
        mitigation: str
        detection_methods: List[str]

class GeminiLLMThreatAnalyzer:
    """Gemini Pro-powered threat analyzer for EVCS systems with conversation memory"""
    
    def __init__(self, api_key: str = None, model_name: str = "models/gemini-2.5-flash", max_history: int = 10):
        """
        Initialize Gemini LLM Threat Analyzer
        
        Args:
            api_key: Google API key for Gemini. If None, reads from gemini_key.txt
            model_name: Gemini model to use (default: models/gemini-2.5-flash)
            max_history: Maximum number of conversation turns to remember
        """
        self.model_name = model_name
        self.is_available = False
        self.model = None
        self.max_history = max_history
        
        # Conversation memory
        self.conversation_history = []
        self.analysis_context = {
            'previous_vulnerabilities': [],
            'previous_strategies': [],
            'system_learning': {},
            'threat_evolution': []
        }
        
        # Load API key
        if api_key is None:
            api_key = self._load_api_key()
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            self.is_available = False
    
    def _load_api_key(self) -> str:
        """Load API key from gemini_key.txt"""
        try:
            with open('gemini_key.txt', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("gemini_key.txt not found. Please create the file with your Gemini API key.")
        except Exception as e:
            raise Exception(f"Error reading API key: {e}")
    
    def _test_connection(self):
        """Test connection to Gemini"""
        try:
            response = self.model.generate_content("Test connection")
            if response.text:
                self.is_available = True
                print(f"✅ Gemini connection successful with model: {self.model_name}")
            else:
                raise Exception("Empty response from Gemini")
        except Exception as e:
            print(f" Gemini connection failed: {e}")
            self.is_available = False
    
    def _add_to_conversation_history(self, user_input: str, assistant_response: str, analysis_type: str = None):
        """Add interaction to conversation history with context"""
        conversation_entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'assistant_response': assistant_response,
            'analysis_type': analysis_type
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Maintain max history limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def _update_analysis_context(self, analysis_type: str, result: Dict):
        """Update the analysis context with new findings"""
        timestamp = time.time()
        
        if analysis_type == 'vulnerability_analysis':
            self.analysis_context['previous_vulnerabilities'].append({
                'timestamp': timestamp,
                'vulnerabilities': result.get('vulnerabilities', []),
                'attack_vectors': result.get('attack_vectors', [])
            })
            
        elif analysis_type == 'attack_strategy':
            self.analysis_context['previous_strategies'].append({
                'timestamp': timestamp,
                'strategy': result.get('strategy_name', 'Unknown'),
                'techniques': result.get('mitre_techniques', []),
                'success_probability': result.get('success_probability', 0.0)
            })
        
        # Learn from patterns
        self._update_system_learning(analysis_type, result)
    
    def _update_system_learning(self, analysis_type: str, result: Dict):
        """Extract patterns and learning from analysis results"""
        if analysis_type not in self.analysis_context['system_learning']:
            self.analysis_context['system_learning'][analysis_type] = {
                'common_patterns': [],
                'effectiveness_metrics': {},
                'trend_analysis': []
            }
        
        learning = self.analysis_context['system_learning'][analysis_type]
        
        # Track common vulnerability types
        if analysis_type == 'vulnerability_analysis':
            vulns = result.get('vulnerabilities', [])
            for vuln in vulns:
                vuln_type = vuln.get('type', 'unknown')
                if vuln_type not in learning['common_patterns']:
                    learning['common_patterns'].append(vuln_type)
        
        # Track strategy effectiveness
        elif analysis_type == 'attack_strategy':
            success_prob = result.get('success_probability', 0.0)
            strategy_name = result.get('strategy_name', 'unknown')
            learning['effectiveness_metrics'][strategy_name] = success_prob
    
    def _get_conversation_context(self, max_recent: int = 3) -> str:
        """Get recent conversation context for prompts"""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_recent:]
        context_lines = ["PREVIOUS INTERACTION CONTEXT:"]
        
        for i, entry in enumerate(recent_history, 1):
            context_lines.append(f"\n--- Interaction {i} ({entry.get('analysis_type', 'general')}) ---")
            context_lines.append(f"Request: {entry['user_input'][:200]}...")
            context_lines.append(f"Response: {entry['assistant_response'][:200]}...")
        
        context_lines.append("\nBased on the above context, provide continuity in your analysis.")
        return "\n".join(context_lines)
    
    def _get_learning_context(self, analysis_type: str) -> str:
        """Get system learning context for improved analysis"""
        if analysis_type not in self.analysis_context['system_learning']:
            return ""
        
        learning = self.analysis_context['system_learning'][analysis_type]
        context_lines = [f"\nSYSTEM LEARNING CONTEXT FOR {analysis_type.upper()}:"]
        
        if learning['common_patterns']:
            context_lines.append(f"Common patterns observed: {', '.join(learning['common_patterns'])}")
        
        if learning['effectiveness_metrics']:
            most_effective = max(learning['effectiveness_metrics'].items(), key=lambda x: x[1])
            context_lines.append(f"Most effective strategy: {most_effective[0]} (success: {most_effective[1]:.2f})")
        
        # Add vulnerability evolution tracking
        if analysis_type == 'vulnerability_analysis' and self.analysis_context['previous_vulnerabilities']:
            recent_vulns = self.analysis_context['previous_vulnerabilities'][-3:]
            vuln_trend = []
            for vuln_set in recent_vulns:
                vuln_trend.extend([v.get('type', 'unknown') for v in vuln_set.get('vulnerabilities', [])])
            
            if vuln_trend:
                from collections import Counter
                common_recent = Counter(vuln_trend).most_common(3)
                context_lines.append(f"Recent vulnerability trends: {[f'{v}({c})' for v, c in common_recent]}")
        
        return "\n".join(context_lines)
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation history and learning"""
        return {
            'total_interactions': len(self.conversation_history),
            'recent_interactions': self.conversation_history[-3:] if self.conversation_history else [],
            'learning_summary': {
                'vulnerability_patterns': self.analysis_context['system_learning'].get('vulnerability_analysis', {}).get('common_patterns', []),
                'strategy_effectiveness': self.analysis_context['system_learning'].get('attack_strategy', {}).get('effectiveness_metrics', {}),
                'total_vulnerabilities_analyzed': len(self.analysis_context['previous_vulnerabilities']),
                'total_strategies_generated': len(self.analysis_context['previous_strategies'])
            }
        }
    
    def clear_conversation_history(self):
        """Clear conversation history (useful for starting fresh)"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def reset_learning_context(self):
        """Reset the learning context while keeping conversation history"""
        self.analysis_context = {
            'previous_vulnerabilities': [],
            'previous_strategies': [],
            'system_learning': {},
            'threat_evolution': []
        }
        print("🧠 Learning context reset")
    
    def analyze_evcs_vulnerabilities(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Analyze EVCS vulnerabilities using Gemini Pro with conversation memory"""
        if not self.is_available:
            return self._fallback_analysis(evcs_state, system_config)
        
        # Create comprehensive prompt with context
        base_prompt = self._create_vulnerability_prompt(evcs_state, system_config)
        conversation_context = self._get_conversation_context()
        learning_context = self._get_learning_context('vulnerability_analysis')
        
        # Combine prompt with memory context
        full_prompt = f"{base_prompt}\n{conversation_context}\n{learning_context}"
        
        try:
            response = self.model.generate_content(full_prompt)
            llm_response = response.text
            result = self._parse_vulnerability_response(llm_response, evcs_state)
            
            # Update conversation history and learning context
            self._add_to_conversation_history(
                user_input=f"Vulnerability analysis for EVCS state: {str(evcs_state)[:100]}...",
                assistant_response=llm_response,
                analysis_type='vulnerability_analysis'
            )
            self._update_analysis_context('vulnerability_analysis', result)
            
            return result
            
        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return self._fallback_analysis(evcs_state, system_config)
    
    def generate_attack_strategy(self, vulnerabilities: List[EVCSVulnerability], 
                               evcs_state: Dict, constraints: Dict) -> Dict:
        """Generate attack strategy using Gemini Pro with conversation memory"""
        if not self.is_available:
            return self._fallback_strategy(vulnerabilities, constraints)
        
        # Create strategy generation prompt with context
        base_prompt = self._create_strategy_prompt(vulnerabilities, evcs_state, constraints)
        conversation_context = self._get_conversation_context()
        learning_context = self._get_learning_context('attack_strategy')
        
        # Combine prompt with memory context
        full_prompt = f"{base_prompt}\n{conversation_context}\n{learning_context}"
        
        try:
            response = self.model.generate_content(full_prompt)
            llm_response = response.text
            result = self._parse_strategy_response(llm_response, vulnerabilities)
            
            # Update conversation history and learning context
            vuln_summary = f"{len(vulnerabilities)} vulnerabilities"
            self._add_to_conversation_history(
                user_input=f"Attack strategy generation for {vuln_summary}",
                assistant_response=llm_response,
                analysis_type='attack_strategy'
            )
            self._update_analysis_context('attack_strategy', result)
            
            return result
            
        except Exception as e:
            print(f"Gemini strategy generation failed: {e}")
            return self._fallback_strategy(vulnerabilities, constraints)
    
    def analyze_system_with_context(self, data: Dict, analysis_type: str, system_prompt: str = None) -> Dict:
        """General analysis method with system context and conversation memory"""
        if not self.is_available:
            return self._fallback_analysis_general(data, analysis_type)
        
        try:
            if analysis_type == 'vulnerability_analysis':
                base_prompt = self._create_vulnerability_analysis_prompt(data, system_prompt)
            elif analysis_type == 'attack_strategy':
                base_prompt = self._create_attack_strategy_prompt(data, system_prompt)
            else:
                base_prompt = f"Analyze the following data: {data}"
            
            # Add conversation and learning context
            conversation_context = self._get_conversation_context()
            learning_context = self._get_learning_context(analysis_type)
            
            # Build full prompt with all context
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt)
            prompt_parts.append(base_prompt)
            if conversation_context:
                prompt_parts.append(conversation_context)
            if learning_context:
                prompt_parts.append(learning_context)
            
            full_prompt = "\n\n".join(prompt_parts)
            
            response = self.model.generate_content(full_prompt)
            llm_response = response.text
            result = self._parse_llm_response(llm_response, analysis_type)
            
            # Update conversation history and learning context
            self._add_to_conversation_history(
                user_input=f"{analysis_type}: {str(data)[:100]}...",
                assistant_response=llm_response,
                analysis_type=analysis_type
            )
            self._update_analysis_context(analysis_type, result)
            
            return result
            
        except Exception as e:
            print(f"Gemini analysis with context failed: {e}")
            return self._fallback_analysis_general(data, analysis_type)
    
    def _create_vulnerability_prompt(self, evcs_state: Dict, system_config: Dict) -> str:
        """Create vulnerability analysis prompt"""
        return f"""
        As a cybersecurity expert specializing in Electric Vehicle Charging Station (EVCS) systems, analyze the following system for vulnerabilities:

        EVCS State:
        {json.dumps(evcs_state, indent=2)}

        System Configuration:
        {json.dumps(system_config, indent=2)}

        Please provide a comprehensive vulnerability assessment including:
        1. Top 5 critical vulnerabilities with CVSS scores
        2. STRIDE threat categorization
        3. Potential attack vectors
        4. Recommended mitigations
        5. Detection methods

        Format your response as structured text that can be parsed.
        """
    
    def _create_strategy_prompt(self, vulnerabilities: List[EVCSVulnerability], 
                              evcs_state: Dict, constraints: Dict) -> str:
        """Create attack strategy generation prompt"""
        vuln_summary = "\n".join([f"- {v.vulnerability_type} (Severity: {v.severity})" for v in vulnerabilities])
        
        return f"""
        As a red team cybersecurity expert, develop a sophisticated attack strategy for an EVCS system based on the following vulnerabilities:

        Identified Vulnerabilities:
        {vuln_summary}

        Current System State:
        {json.dumps(evcs_state, indent=2)}

        Attack Constraints:
        {json.dumps(constraints, indent=2)}

        Please provide:
        1. Multi-stage attack sequence
        2. MITRE ATT&CK technique mappings
        3. Stealth and evasion techniques
        4. Success probability assessment
        5. Risk mitigation recommendations

        Focus on sophisticated, coordinated attacks that could realistically target EVCS infrastructure.
        """
    
    def _create_vulnerability_analysis_prompt(self, data: Dict, system_prompt: str = None) -> str:
        """Create vulnerability analysis prompt with system context"""
        cms_info = data.get('cms_specific_info', {})
        focus_areas = data.get('vulnerability_focus_areas', [])
        
        return f"""
        COMPREHENSIVE EVCS SYSTEM VULNERABILITY ANALYSIS REQUEST:

        Current System State:
        - Distribution Systems: {cms_info.get('num_distribution_systems', 'Unknown')}
        - EVCS Stations: {cms_info.get('total_evcs_stations', 'Unknown')}
        - Active Charging Sessions: {cms_info.get('active_charging_sessions', 'Unknown')}
        - Grid Frequency: {cms_info.get('grid_frequency', 'Unknown')} Hz
        - System Load: {cms_info.get('system_load', 'Unknown')} MW
        - Federated PINN Active: {cms_info.get('pinn_models_active', False)}
        - Hierarchical Simulation Active: {cms_info.get('hierarchical_sim_active', False)}

        Focus Areas for Analysis:
        {chr(10).join([f"- {area}" for area in focus_areas])}

        Based on the comprehensive system architecture, analyze this EVCS system and identify:

        1. TOP 5 CRITICAL VULNERABILITIES with specific focus on:
           - Federated PINN model poisoning attack vectors
           - Hierarchical co-simulation layer vulnerabilities
           - EVCS power electronics manipulation points
           - Grid stability disruption vulnerabilities
           - Real-time constraint exploitation opportunities

        2. ATTACK VECTOR MAPPING for each vulnerability:
           - Entry points and attack paths
           - Required privileges and access levels
           - Technical exploitation methods
           - Potential for lateral movement

        3. MITRE ATT&CK technique mappings specific to:
           - Industrial Control Systems (ICS)
           - Machine Learning model attacks
           - Power system operations

        4. STRIDE threat categorization with impact assessment

        5. DETECTION AND MITIGATION recommendations

        Format your response as structured analysis with specific technical details and actionable intelligence.
        """
    
    def _create_attack_strategy_prompt(self, data: Dict, system_prompt: str = None) -> str:
        """Create attack strategy prompt with system context"""
        scenario = data.get('scenario', {})
        system_state = data.get('current_system_state', {})
        
        return f"""
        COMPREHENSIVE ATTACK STRATEGY DEVELOPMENT REQUEST:

        Target Scenario:
        - ID: {scenario.get('id', 'Unknown')}
        - Name: {scenario.get('name', 'Unknown')}
        - Description: {scenario.get('description', 'Unknown')}
        - Target Systems: {scenario.get('target_systems', [])}
        - Stealth Requirement: {scenario.get('stealth_requirement', 'Unknown')}
        - Impact Goal: {scenario.get('impact_goal', 'Unknown')}

        Current System State:
        - Hierarchical Simulation Active: {system_state.get('hierarchical_sim_active', False)}
        - Federated PINN Active: {system_state.get('federated_pinn_active', False)}
        - Distribution Systems: {system_state.get('num_distribution_systems', 'Unknown')}
        - System Load: {system_state.get('system_load', 'Unknown')}
        - Grid Frequency: {system_state.get('grid_frequency', 'Unknown')}
        - EVCS Utilization: {system_state.get('evcs_utilization', 'Unknown')}

        Based on the comprehensive system architecture, develop a sophisticated multi-stage attack strategy that:

        1. INITIAL ACCESS STRATEGY:
           - Identify optimal entry points in the hierarchical co-simulation layer
           - Exploit federated PINN communication channels
           - Target EVCS management interfaces

        2. PERSISTENCE AND LATERAL MOVEMENT:
           - Establish persistent access across distribution systems
           - Move laterally through federated PINN network
           - Compromise multiple EVCS controllers

        3. FEDERATED LEARNING ATTACK SEQUENCE:
           - Model poisoning injection points
           - Gradual bias accumulation strategy
           - Stealth techniques to avoid detection

        4. GRID DESTABILIZATION COORDINATION:
           - Synchronized EVCS manipulation timing
           - Power system stability attack vectors
           - Cascading failure trigger mechanisms

        5. STEALTH AND EVASION:
           - Detection avoidance techniques
           - Legitimate traffic mimicking
           - Gradual escalation strategies

        Provide detailed technical attack sequences with specific timing, coordination methods, and success probability assessments.
        """
    
    def _parse_vulnerability_response(self, response: str, evcs_state: Dict) -> Dict:
        """Parse vulnerability analysis response from Gemini"""
        try:
            # Extract vulnerabilities, CVSS scores, etc.
            vulnerabilities = self._extract_vulnerabilities_from_text(response)
            attack_vectors = self._extract_attack_vectors_from_text(response)
            mitre_techniques = self._extract_mitre_techniques_from_text(response)
            
            return {
                'vulnerabilities': vulnerabilities,
                'attack_vectors': attack_vectors,
                'mitre_techniques': mitre_techniques,
                'stride_mapping': self._extract_stride_mapping(response),
                'raw_analysis': response,
                'confidence': 0.9  # High confidence for Gemini Pro
            }
        except Exception as e:
            print(f"Failed to parse vulnerability response: {e}")
            return {'raw_analysis': response, 'parse_error': str(e)}
    
    def _parse_strategy_response(self, response: str, vulnerabilities: List[EVCSVulnerability]) -> Dict:
        """Parse attack strategy response from Gemini"""
        try:
            return {
                'strategy_name': self._extract_strategy_name(response),
                'attack_sequence': self._extract_attack_sequence_from_text(response),
                'mitre_techniques': self._extract_mitre_techniques_from_text(response),
                'stealth_measures': self._extract_stealth_measures_from_text(response),
                'success_probability': self._extract_success_probability(response),
                'risk_assessment': self._extract_risk_assessment(response),
                'raw_strategy': response
            }
        except Exception as e:
            print(f"Failed to parse strategy response: {e}")
            return {'raw_strategy': response, 'parse_error': str(e)}
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict:
        """Parse general LLM response into structured format"""
        try:
            # Try to extract JSON if present
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON, create structured response based on analysis type
            if analysis_type == 'vulnerability_analysis':
                return {
                    'vulnerabilities': self._extract_vulnerabilities_from_text(response),
                    'attack_vectors': self._extract_attack_vectors_from_text(response),
                    'mitre_techniques': self._extract_mitre_techniques_from_text(response),
                    'raw_analysis': response
                }
            elif analysis_type == 'attack_strategy':
                return {
                    'attack_sequence': self._extract_attack_sequence_from_text(response),
                    'stealth_measures': self._extract_stealth_measures_from_text(response),
                    'success_probability': self._extract_success_probability(response),
                    'raw_strategy': response
                }
            else:
                return {'analysis': response}
                
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return {'raw_response': response, 'parse_error': str(e)}
    
    def _extract_vulnerabilities_from_text(self, text: str) -> List[Dict]:
        """Extract vulnerability information from text response"""
        vulnerabilities = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['vulnerability', 'weakness', 'flaw', 'exploit']):
                # Extract CVSS score if present
                cvss_match = re.search(r'cvss[:\s]*(\d+\.?\d*)', line.lower())
                cvss_score = float(cvss_match.group(1)) if cvss_match else 5.0
                
                # Extract severity
                severity_match = re.search(r'severity[:\s]*(\d+\.?\d*)', line.lower())
                severity = float(severity_match.group(1)) if severity_match else cvss_score / 10.0
                
                vuln = {
                    'description': line.strip(),
                    'severity': min(severity, 1.0),
                    'cvss_score': cvss_score,
                    'component': self._extract_component(line),
                    'type': self._extract_vulnerability_type(line)
                }
                vulnerabilities.append(vuln)
        
        return vulnerabilities[:5]  # Limit to top 5
    
    def _extract_attack_vectors_from_text(self, text: str) -> List[str]:
        """Extract attack vectors from text response"""
        vectors = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['attack', 'exploit', 'compromise', 'manipulate']):
                vectors.append(line.strip())
        
        return vectors[:10]  # Limit to top 10
    
    def _extract_mitre_techniques_from_text(self, text: str) -> List[str]:
        """Extract MITRE ATT&CK techniques from text response"""
        techniques = re.findall(r'T\d{4}', text)
        return list(set(techniques))  # Remove duplicates
    
    def _extract_attack_sequence_from_text(self, text: str) -> List[str]:
        """Extract attack sequence steps from text response"""
        sequence = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'stage', 'phase', 'first', 'then', 'next', 'finally']):
                sequence.append(line.strip())
        
        return sequence
    
    def _extract_stealth_measures_from_text(self, text: str) -> List[str]:
        """Extract stealth measures from text response"""
        measures = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['stealth', 'evasion', 'avoid', 'hide', 'conceal', 'gradual']):
                measures.append(line.strip())
        
        return measures
    
    def _extract_stride_mapping(self, text: str) -> Dict[str, List[str]]:
        """Extract STRIDE threat categorization"""
        stride_categories = {
            'spoofing': [],
            'tampering': [],
            'repudiation': [],
            'information_disclosure': [],
            'denial_of_service': [],
            'elevation_of_privilege': []
        }
        
        lines = text.split('\n')
        current_category = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect STRIDE category headers
            for category in stride_categories.keys():
                if category.replace('_', ' ') in line_lower or category in line_lower:
                    current_category = category
                    break
            
            # Add items to current category
            if current_category and line.strip() and not any(cat in line_lower for cat in stride_categories.keys()):
                stride_categories[current_category].append(line.strip())
        
        return stride_categories
    
    def _extract_component(self, line: str) -> str:
        """Extract component name from vulnerability description"""
        components = ['charging_controller', 'grid_interface', 'cms', 'pinn', 'communication', 'sensor']
        for comp in components:
            if comp in line.lower():
                return comp
        return 'unknown'
    
    def _extract_vulnerability_type(self, line: str) -> str:
        """Extract vulnerability type from description"""
        vuln_types = ['authentication', 'authorization', 'injection', 'overflow', 'disclosure', 'dos']
        for vtype in vuln_types:
            if vtype in line.lower():
                return vtype
        return 'unknown'
    
    def _extract_strategy_name(self, text: str) -> str:
        """Extract strategy name from response"""
        lines = text.split('\n')
        for line in lines:
            if 'strategy' in line.lower() and len(line.strip()) < 100:
                return line.strip()
        return "Gemini Generated Attack Strategy"
    
    def _extract_success_probability(self, text: str) -> float:
        """Extract success probability from text"""
        prob_match = re.search(r'success.*?(\d+\.?\d*)%', text.lower())
        if prob_match:
            return float(prob_match.group(1)) / 100.0
        
        # Look for probability keywords
        if 'high' in text.lower():
            return 0.8
        elif 'medium' in text.lower():
            return 0.6
        elif 'low' in text.lower():
            return 0.3
        
        return 0.7  # Default
    
    def _extract_risk_assessment(self, text: str) -> Dict:
        """Extract risk assessment from text"""
        return {
            'overall_risk': 'medium',
            'technical_complexity': 'high',
            'resource_requirements': 'medium',
            'detection_likelihood': 'low'
        }
    
    def _fallback_analysis(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Fallback analysis when Gemini is not available"""
        return {
            'vulnerabilities': [
                {
                    'description': 'Charging controller authentication bypass',
                    'severity': 0.8,
                    'cvss_score': 8.1,
                    'component': 'charging_controller',
                    'type': 'authentication'
                }
            ],
            'attack_vectors': ['Authentication bypass', 'Command injection'],
            'mitre_techniques': ['T1078', 'T1059'],
            'fallback': True
        }
    
    def _fallback_strategy(self, vulnerabilities: List[EVCSVulnerability], constraints: Dict) -> Dict:
        """Fallback strategy when Gemini is not available"""
        return {
            'strategy_name': 'Fallback Attack Strategy',
            'attack_sequence': ['reconnaissance', 'initial_access', 'persistence', 'impact'],
            'stealth_measures': ['gradual_escalation', 'legitimate_traffic_mimicking'],
            'success_probability': 0.6,
            'fallback': True
        }
    
    def _fallback_analysis_general(self, data: Dict, analysis_type: str) -> Dict:
        """General fallback analysis"""
        return {
            'analysis': 'Fallback analysis - Gemini not available',
            'fallback': True,
            'analysis_type': analysis_type
        }

# For backward compatibility
class OllamaLLMThreatAnalyzer(GeminiLLMThreatAnalyzer):
    """Backward compatibility wrapper that uses Gemini instead of Ollama"""
    
    def __init__(self, base_url=None, model=None):
        """Initialize with Gemini instead of Ollama"""
        print("🔄 Redirecting from Ollama to Gemini Pro...")
        super().__init__(model_name="gemini-pro")
        
        # Override the model attribute for compatibility
        self.base_url = "https://generativelanguage.googleapis.com"
        self.model = "gemini-pro"
        
        # For compatibility with existing code that checks these attributes
        self.client = self  # Point to self for client calls
    
    def chat(self):
        """Compatibility method for existing code"""
        return self
    
    def completions(self):
        """Compatibility method for existing code"""
        return self
    
    def create(self, model=None, messages=None, max_tokens=None, temperature=None, **kwargs):
        """Compatibility method for existing chat.completions.create calls"""
        if not messages:
            return type('Response', (), {'choices': [type('Choice', (), {'message': type('Message', (), {'content': 'Error: No messages provided'})()})()]})()
        
        try:
            # Extract the user message
            user_message = ""
            system_message = ""
            
            for msg in messages:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                elif msg.get('role') == 'system':
                    system_message = msg.get('content', '')
            
            # Combine system and user messages
            full_prompt = f"{system_message}\n\n{user_message}" if system_message else user_message
            
            # Generate response using Gemini
            response = super().model.generate_content(full_prompt)
            
            # Create compatible response object
            choice = type('Choice', (), {
                'message': type('Message', (), {'content': response.text})()
            })()
            
            return type('Response', (), {'choices': [choice]})()
            
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            # Return fallback response
            choice = type('Choice', (), {
                'message': type('Message', (), {'content': 'Fallback response due to API error'})()
            })()
            
            return type('Response', (), {'choices': [choice]})()

if __name__ == "__main__":
    # Test the Gemini LLM Threat Analyzer
    print("Testing Gemini LLM Threat Analyzer...")
    
    analyzer = GeminiLLMThreatAnalyzer(model_name="models/gemini-2.5-flash")
    
    if analyzer.is_available:
        # Test vulnerability analysis
        test_evcs_state = {
            'charging_stations': 6,
            'active_sessions': 12,
            'grid_frequency': 60.0,
            'system_load': 850.5
        }
        
        test_config = {
            'max_power': 1000,
            'voltage_range': [0.95, 1.05]
        }
        
        print("\n🔍 Testing vulnerability analysis...")
        results = analyzer.analyze_evcs_vulnerabilities(test_evcs_state, test_config)
        print(f"Found {len(results.get('vulnerabilities', []))} vulnerabilities")
        
        print("\n Gemini LLM Threat Analyzer test completed!")
    else:
        print(" Gemini LLM Threat Analyzer not available")
