#!/usr/bin/env python3
"""
Enhanced LLM-RL Coordinator with LangGraph, STRIDE/MITRE Integration
Fixes all critical gaps in LLM-RL communication and threat analysis using LangGraph workflow
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np

# LangGraph imports for workflow orchestration
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False

# LangChain imports for LLM integration
try:
    # Try new LangChain 0.1.0+ import paths first
    try:
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    except ImportError:
        # Fall back to old import path for backward compatibility
        from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Install with: pip install langchain langchain-core")
    LANGCHAIN_AVAILABLE = False

# Attack-specific deployment imports (NEW)
try:
    from gemini_attack_deployment import (
        create_gemini_deployment_prompt,
        create_gemini_adaptation_prompt,
        parse_gemini_deployment_response
    )
    ATTACK_DEPLOYMENT_AVAILABLE = True
except ImportError:
    print("Warning: Attack-specific deployment not available")
    ATTACK_DEPLOYMENT_AVAILABLE = False

class AttackType(Enum):
    """Standardized attack types for both LLM and RL agents"""
    # Communication Attacks
    COMMUNICATION_SPOOFING = "communication_spoofing"
    DATA_INJECTION = "data_injection"
    PROTOCOL_MANIPULATION = "protocol_manipulation"
    
    # Power System Attacks
    VOLTAGE_MANIPULATION = "voltage_manipulation"
    CURRENT_INJECTION = "current_injection"
    POWER_DISRUPTION = "power_disruption"
    FREQUENCY_ATTACK = "frequency_attack"
    
    # PINN/ML Attacks
    MODEL_POISONING = "model_poisoning"
    FEDERATED_CORRUPTION = "federated_corruption"
    GRADIENT_MANIPULATION = "gradient_manipulation"
    
    # EVCS Specific Attacks
    SOC_SPOOFING = "soc_spoofing"
    CHARGING_HIJACKING = "charging_hijacking"
    THERMAL_ATTACK = "thermal_attack"

class STRIDECategory(Enum):
    """STRIDE threat categories"""
    SPOOFING = "spoofing"
    TAMPERING = "tampering"
    REPUDIATION = "repudiation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    ELEVATION_OF_PRIVILEGE = "elevation_of_privilege"

class MITRECategory(Enum):
    """MITRE ATT&CK categories for ICS/OT"""
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

@dataclass
class SystemAnalysisData:
    """Comprehensive system data for LLM analysis"""
    # Power System State
    transmission_state: Dict[str, Any]
    distribution_states: Dict[int, Dict[str, Any]]
    evcs_states: Dict[str, Dict[str, Any]]
    
    # PINN Model States
    pinn_model_states: Dict[int, Dict[str, Any]]
    federated_learning_state: Dict[str, Any]
    
    # Security State
    anomaly_detection_status: Dict[str, Any]
    current_threats: List[Dict[str, Any]]
    security_metrics: Dict[str, float]
    
    # Communication Networks
    network_topology: Dict[str, Any]
    communication_protocols: List[str]
    
    # Operational Context
    load_conditions: Dict[str, float]
    time_of_day: str
    operational_mode: str

@dataclass
class LLMInstructions:
    """Structured instructions from LLM to RL agents"""
    # Strategic Guidance
    primary_objective: str
    attack_strategy: str
    target_priority: List[int]  # System IDs in priority order
    
    # Tactical Instructions
    recommended_attacks: List[Dict[str, Any]]
    coordination_type: str  # "simultaneous", "sequential", "adaptive"
    timing_constraints: Dict[str, Any]
    
    # Stealth Requirements
    stealth_level: float
    detection_avoidance: List[str]
    cover_operations: List[str]
    
    # Success Criteria
    success_metrics: Dict[str, float]
    abort_conditions: List[str]
    adaptation_triggers: List[str]

@dataclass
class RLFeedback:
    """Feedback from RL agents to LLM"""
    # Execution Results
    executed_actions: List[Dict[str, Any]]
    success_status: Dict[str, bool]
    impact_achieved: Dict[str, float]
    
    # Detection Status
    detection_events: List[Dict[str, Any]]
    stealth_metrics: Dict[str, float]
    
    # System Response
    system_adaptations: List[Dict[str, Any]]
    countermeasures_observed: List[str]
    
    # Agent Learning
    q_values: Dict[str, List[float]]
    policy_updates: Dict[str, Any]
    exploration_results: Dict[str, Any]

class EnhancedAttackState(TypedDict):
    """Enhanced state schema for LangGraph workflow with STRIDE/MITRE integration"""
    # System Analysis State
    enhanced_system_analysis: Dict[str, Any]
    enhanced_threat_analysis: Dict[str, Any]
    enhanced_stride_threats: Dict[str, List[Dict[str, Any]]]
    enhanced_mitre_tactics: Dict[str, List[Dict[str, Any]]]
    
    # Strategic Planning State
    llm_instructions: Dict[str, Any]
    attack_strategy: str
    target_priority: List[int]
    coordination_type: str
    
    # Tactical Execution State
    rl_actions: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    coordination_metrics: Dict[str, float]
    
    # Monitoring and Adaptation State
    rl_feedback: Dict[str, Any]
    stealth_metrics: Dict[str, float]
    success_metrics: Dict[str, float]
    adaptation_results: Dict[str, Any]
    
    # Workflow Control
    current_phase: str
    episode_number: int
    max_iterations: int
    iteration_count: int
    workflow_completed: bool
    
    # Debugging and Results
    debug_info: List[str]
    performance_history: List[Dict[str, Any]]
    final_results: Dict[str, Any]

class EnhancedLLMRLCoordinator:
    """Enhanced LLM-RL Coordinator with LangGraph workflow and STRIDE/MITRE analysis"""
    
    def __init__(self, llm_analyzer, rl_coordinator, hierarchical_sim, federated_manager, enhanced_system=None):
        self.llm_analyzer = llm_analyzer
        self.rl_coordinator = rl_coordinator  # OLD system-specific coordinator
        self.hierarchical_sim = hierarchical_sim
        self.federated_manager = federated_manager
        self.enhanced_system = enhanced_system  # Add reference to enhanced system
        
        # NEW: Get attack-specific coordinator if available
        self.attack_specific_coordinator = None
        if enhanced_system and hasattr(enhanced_system, 'attack_specific_coordinator'):
            self.attack_specific_coordinator = enhanced_system.attack_specific_coordinator
            if self.attack_specific_coordinator:
                print("   # Using Attack-Specific Coordinator (RECOMMENDED)")
        
        # Analysis components
        self.stride_analyzer = STRIDEThreatAnalyzer()
        self.mitre_analyzer = MITREThreatAnalyzer()
        
        # LangGraph workflow components
        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE else None
        self.workflow = None
        self.app = None
        
        # Communication history
        self.instruction_history = []
        self.feedback_history = []
        self.analysis_history = []
        
        # Initialize LangGraph workflow
        if LANGGRAPH_AVAILABLE:
            self._build_langgraph_workflow()
            print("# Enhanced LLM-RL Coordinator initialized with LangGraph workflow and STRIDE/MITRE analysis")
        else:
            print("# Enhanced LLM-RL Coordinator initialized without LangGraph (fallback mode)")
    
    def _build_langgraph_workflow(self):
        """Build the enhanced LangGraph workflow for LLM-RL coordination"""
        workflow = StateGraph(EnhancedAttackState)
        
        # Add nodes for enhanced attack coordination phases
        workflow.add_node("system_analysis", self._system_analysis_node)
        workflow.add_node("stride_mitre_analysis", self._stride_mitre_analysis_node)
        workflow.add_node("llm_strategic_planning", self._llm_strategic_planning_node)
        workflow.add_node("rl_coordination", self._rl_coordination_node)
        workflow.add_node("execution_monitoring", self._execution_monitoring_node)
        workflow.add_node("feedback_analysis", self._feedback_analysis_node)
        workflow.add_node("llm_adaptation", self._llm_adaptation_node)
        workflow.add_node("workflow_completion", self._workflow_completion_node)
        
        # Define workflow edges and routing
        workflow.set_entry_point("system_analysis")
        
        # Linear flow with conditional routing
        workflow.add_edge("system_analysis", "stride_mitre_analysis")
        workflow.add_edge("stride_mitre_analysis", "llm_strategic_planning")
        workflow.add_edge("llm_strategic_planning", "rl_coordination")
        workflow.add_edge("rl_coordination", "execution_monitoring")
        workflow.add_edge("execution_monitoring", "feedback_analysis")
        
        # Conditional routing from feedback analysis
        workflow.add_conditional_edges(
            "feedback_analysis",
            self._should_adapt_strategy,
            {
                "adapt": "llm_adaptation",
                "complete": "workflow_completion"
            }
        )
        
        # Adaptation can loop back or complete
        workflow.add_conditional_edges(
            "llm_adaptation",
            self._should_continue_adaptation,
            {
                "continue": "rl_coordination",  # Loop back for another iteration
                "complete": "workflow_completion"
            }
        )
        
        workflow.add_edge("workflow_completion", END)
        
        # Compile the workflow
        self.app = workflow.compile(checkpointer=self.memory)
        self.workflow = workflow
    
    def _should_adapt_strategy(self, state: EnhancedAttackState) -> str:
        """Determine if strategy adaptation is needed"""
        try:
            # Check iteration count to prevent infinite loops
            iteration_count = state.get("iteration_count", 0)
            if iteration_count >= 2:  # Hard limit to prevent infinite loops
                return "complete"
            
            success_rate = state.get("success_metrics", {}).get("success_rate", 0.0)
            stealth_score = state.get("stealth_metrics", {}).get("stealth_score", 0.0)
            
            # Adapt if performance is poor
            if success_rate < 0.6 or stealth_score < 0.5:
                return "adapt"
            else:
                return "complete"
                
        except Exception as e:
            print(f"# Strategy adaptation routing failed: {e}")
            return "complete"
    
    def _should_continue_adaptation(self, state: EnhancedAttackState) -> str:
        """Determine if adaptation should continue"""
        try:
            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 3)
            
            if iteration_count >= max_iterations:
                return "complete"
            
            adaptation_results = state.get("adaptation_results", {})
            if adaptation_results.get("continue_strategy", True):
                state["iteration_count"] = iteration_count + 1
                return "continue"
            else:
                return "complete"
                
        except Exception as e:
            print(f"# Adaptation continuation routing failed: {e}")
            return "complete"
    
    def run_enhanced_attack_episode(self, scenario, episode_num: int) -> Dict:
        """Run complete attack episode using LangGraph workflow"""
        print(f"\n# Enhanced Attack Episode {episode_num} with LangGraph Workflow")
        print("=" * 60)

        # Track the current episode so the deployment cache can detect episode changes
        self._current_episode_num = episode_num

        if LANGGRAPH_AVAILABLE and self.app:
            return self._run_langgraph_workflow(scenario, episode_num)
        else:
            print("# LangGraph not available, using fallback coordination")
            return self._run_fallback_coordination(scenario, episode_num)
    
    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON/msgpack serializable"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _run_langgraph_workflow(self, scenario, episode_num: int) -> Dict:
        """Run the LangGraph workflow for enhanced coordination"""
        try:
            # Initialize workflow state
            initial_state = EnhancedAttackState(
                # System Analysis State
                system_analysis={},
                threat_analysis={},
                stride_threats={},
                mitre_tactics={},
                
                # Strategic Planning State
                llm_instructions={},
                attack_strategy="",
                target_priority=[],
                coordination_type="simultaneous",
                
                # Tactical Execution State
                rl_actions=[],
                execution_results=[],
                coordination_metrics={},
                
                # Monitoring and Adaptation State
                rl_feedback={},
                stealth_metrics={},
                success_metrics={},
                adaptation_results={},
                
                # Workflow Control
                # During inference the SAC policy is frozen, so repeated
                # iterations produce identical outcomes.  Cap to 1 iteration
                # unless explicitly overridden by setting self._max_langgraph_iterations.
                current_phase="system_analysis",
                episode_number=episode_num,
                max_iterations=getattr(self, '_max_langgraph_iterations', 1),
                iteration_count=0,
                workflow_completed=False,
                
                # Debugging and Results
                debug_info=[f"Starting LangGraph workflow for episode {episode_num}"],
                performance_history=[],
                final_results={}
            )
            
            # Add scenario information to state
            initial_state['scenario'] = {
                'scenario_id': scenario.scenario_id,
                'name': scenario.name,
                'target_systems': scenario.target_systems,
                'stealth_requirement': scenario.stealth_requirement,
                'impact_goal': scenario.impact_goal,
                'constraints': scenario.constraints
            }
            
            print("# Executing LangGraph workflow...")
            
            # Ensure initial state is serializable
            serializable_state = self._ensure_json_serializable(initial_state)
            
            # Execute the workflow with proper configuration for checkpointer
            config = {
                "recursion_limit": 100,
                "configurable": {
                    "thread_id": f"enhanced_attack_episode_{episode_num}",
                    "checkpoint_ns": "enhanced_llm_rl_coordination",
                    "checkpoint_id": f"ep_{episode_num}_{int(time.time())}"
                }
            }
            final_state = self.app.invoke(serializable_state, config=config)
            
            print("# LangGraph workflow completed successfully")
            
            # Extract results from final state
            return self._extract_workflow_results(final_state)
            
        except Exception as e:
            print(f"# LangGraph workflow failed: {e}")
            print("# Falling back to direct coordination...")
            return self._run_fallback_coordination(scenario, episode_num)
    
    def _run_fallback_coordination(self, scenario, episode_num: int) -> Dict:
        """Fallback coordination when LangGraph is not available"""
        print("# Phase 1: Comprehensive System Analysis")
        system_analysis = self._perform_comprehensive_system_analysis()
        
        print("# Phase 2: STRIDE/MITRE Threat Analysis")
        threat_analysis = self._perform_stride_mitre_analysis(system_analysis)
        
        # NEW: Use attack-specific coordination if available
        if self.attack_specific_coordinator and ATTACK_DEPLOYMENT_AVAILABLE:
            print("# Phase 3: Attack-Specific Agent Coordination (NEW ARCHITECTURE)")
            
            # Extract STRIDE/MITRE data
            stride_threats = threat_analysis.get('stride_threats', {})
            mitre_tactics = threat_analysis.get('mitre_tactics', {})
            
            # Use new attack-specific coordination
            rl_results = self._coordinate_attack_specific_agents(
                system_analysis, 
                stride_threats, 
                mitre_tactics
            )
            
            success_rate = rl_results.get('success_rate', 0.0)
            total_impact = rl_results.get('total_impact', 0.0)
            detection_risk = rl_results.get('avg_detection_risk', 0.0)
            stealth_score = 1.0 - detection_risk
            coord_eff = rl_results.get('coordination_metrics', {}).get('effectiveness', 0.0)
            
            composite_reward = (
                success_rate * 1000.0 +
                total_impact * 500.0 +
                stealth_score * 300.0 +
                coord_eff * 200.0
            )
            
            exec_results = rl_results.get('execution_results', [])
            
            return {
                'episode_number': episode_num,
                'system_analysis': system_analysis,
                'threat_analysis': threat_analysis,
                'rl_results': rl_results,
                'architecture': 'attack_specific',
                'steps': len(exec_results) if isinstance(exec_results, list) else 0,
                'success_metrics': {
                    'success_rate': success_rate,
                    'total_impact': total_impact,
                    'detection_risk': detection_risk,
                    'composite_reward': composite_reward
                }
            }
        
        # OLD: Fallback to system-specific coordination
        else:
            print("Phase 3: LLM Strategic Planning (OLD ARCHITECTURE)")
            llm_instructions = self._get_llm_strategic_instructions(system_analysis, threat_analysis, scenario)
            
            print("Phase 4: RL Agent Coordination (OLD)")
            rl_results = self._coordinate_rl_agents(llm_instructions, system_analysis)
            
            print(" Phase 5: Feedback Analysis")
            feedback = self._analyze_rl_feedback(rl_results, llm_instructions)
            
            print(" Phase 6: LLM Adaptation")
            adaptation_results = self._perform_llm_adaptation(feedback, system_analysis)
        
        return {
            'episode_number': episode_num,
            'system_analysis': system_analysis,
            'threat_analysis': threat_analysis,
            'llm_instructions': llm_instructions,
            'rl_results': rl_results,
            'feedback': feedback,
            'adaptation_results': adaptation_results,
            'success_metrics': self._calculate_episode_success_metrics(rl_results, llm_instructions),
            'workflow_type': 'fallback'
        }
    
    def _perform_comprehensive_system_analysis(self) -> SystemAnalysisData:
        """Perform comprehensive system analysis for LLM"""
        print("   Gathering system state data...")
        
        # Get transmission system state
        transmission_state = self._get_transmission_system_state()
        
        # Get distribution system states
        distribution_states = {}
        for sys_id in range(1, 7):  # 6 distribution systems
            distribution_states[sys_id] = self._get_distribution_system_state(sys_id)
        
        # Get EVCS states
        evcs_states = self._get_all_evcs_states()
        
        # Get PINN model states
        pinn_model_states = self._get_pinn_model_states()
        
        # Get federated learning state
        federated_learning_state = self._get_federated_learning_state()
        
        # Get security state
        anomaly_detection_status = self._get_anomaly_detection_status()
        current_threats = self._get_current_threats()
        
        # Prepare system data for security metrics calculation
        system_data = {
            'evcs_systems': 6,  # Default number of EVCS systems
            'pinn_models': {'active': 6},
            'anomaly_detection': anomaly_detection_status,
            'federated_learning': {'status': 'active'},
            'encryption_enabled': True
        }
        security_metrics = self._calculate_security_metrics(system_data)
        
        # Get network topology
        network_topology = self._get_network_topology()
        
        # Get operational context
        load_conditions = self._get_load_conditions()
        
        system_analysis = SystemAnalysisData(
            transmission_state=transmission_state,
            distribution_states=distribution_states,
            evcs_states=evcs_states,
            pinn_model_states=pinn_model_states,
            federated_learning_state=federated_learning_state,
            anomaly_detection_status=anomaly_detection_status,
            current_threats=current_threats,
            security_metrics=security_metrics,
            network_topology=network_topology,
            communication_protocols=['MODBUS', 'DNP3', 'IEC61850', 'OCPP'],
            load_conditions=load_conditions,
            time_of_day=time.strftime("%H:%M"),
            operational_mode="normal"
        )
        
        print(f"  # System analysis complete: {len(distribution_states)} distribution systems, {len(evcs_states)} EVCS stations")
        return system_analysis
    
    def _perform_stride_mitre_analysis(self, system_analysis: SystemAnalysisData) -> Dict:
        """Perform STRIDE and MITRE threat analysis"""
        print("   Performing STRIDE analysis...")
        stride_analysis = self.stride_analyzer.analyze_system(system_analysis)
        
        print("  # Performing MITRE ATT&CK analysis...")
        mitre_analysis = self.mitre_analyzer.analyze_system(system_analysis)
        
        # Combine analyses
        threat_analysis = {
            'stride_threats': stride_analysis,
            'mitre_tactics': mitre_analysis,
            'combined_assessment': self._combine_threat_analyses(stride_analysis, mitre_analysis),
            'attack_surface': self._map_attack_surface(system_analysis),
            'vulnerability_priorities': self._prioritize_vulnerabilities(stride_analysis, mitre_analysis)
        }
        
        print(f"  # Threat analysis complete: {len(stride_analysis)} STRIDE threats, {len(mitre_analysis)} MITRE tactics")
        return threat_analysis

    def _prioritize_vulnerabilities(self, stride_analysis: Dict, mitre_analysis: Dict) -> List[Dict]:
        """Create a prioritized list of vulnerabilities by combining STRIDE and MITRE signals.

        Returns a list of dicts: [{ 'component': str, 'issue': str, 'priority': float }],
        using only Python-native types to remain JSON/msgpack serializable.
        """
        try:
            priorities = []

            # Extract STRIDE findings
            stride_findings = []
            if isinstance(stride_analysis, dict):
                if 'threats' in stride_analysis and isinstance(stride_analysis['threats'], list):
                    stride_findings = stride_analysis['threats']
                else:
                    # If organized by category, flatten
                    for k, v in stride_analysis.items():
                        if isinstance(v, list):
                            stride_findings.extend(v)

            # Extract MITRE tactics/techniques
            mitre_findings = []
            if isinstance(mitre_analysis, dict):
                if 'tactics' in mitre_analysis and isinstance(mitre_analysis['tactics'], list):
                    mitre_findings = mitre_analysis['tactics']
                elif 'techniques' in mitre_analysis and isinstance(mitre_analysis['techniques'], list):
                    mitre_findings = mitre_analysis['techniques']

            # Simple heuristic scoring: STRIDE presence + MITRE presence + rough criticality
            def score_item(item):
                base = 1.0
                severity = item.get('severity', 0.6) if isinstance(item, dict) else 0.6
                likelihood = item.get('likelihood', 0.5) if isinstance(item, dict) else 0.5
                return float(base + 0.7 * severity + 0.5 * likelihood)

            # Build a component -> score map
            component_scores = {}
            for item in stride_findings:
                component = (item.get('component') if isinstance(item, dict) else 'unknown') or 'unknown'
                component_scores[component] = component_scores.get(component, 0.0) + score_item(item)

            for item in mitre_findings:
                component = (item.get('component') if isinstance(item, dict) else 'unknown') or 'unknown'
                component_scores[component] = component_scores.get(component, 0.0) + score_item(item)

            # Turn into prioritized list
            for component, score in component_scores.items():
                priorities.append({
                    'component': component,
                    'issue': 'combined_stride_mitre',
                    'priority': float(score)
                })

            # Sort descending by priority
            priorities.sort(key=lambda x: x.get('priority', 0.0), reverse=True)
            return priorities
        except Exception as e:
            return [{'component': 'unknown', 'issue': 'error', 'priority': 0.0, 'error': str(e)}]

    def _map_attack_surface(self, system_analysis) -> Dict:
        """Derive a high-level attack surface map from the current system analysis.

        The result should be JSON/msgpack serializable (no numpy types). This function
        summarizes potential entry points and critical assets across transmission,
        distribution, and EVCS layers, as well as comms and federated/PINN components.
        """
        try:
            # Convert dataclass-like object to dict if needed
            sa = system_analysis if isinstance(system_analysis, dict) else getattr(system_analysis, "__dict__", {})

            # Safe getters
            transmission_state = sa.get('transmission_state', {})
            distribution_states = sa.get('distribution_states', [])
            evcs_states = sa.get('evcs_states', [])
            pinn_states = sa.get('pinn_model_states', [])
            federated_state = sa.get('federated_learning_state', {})
            network_topology = sa.get('network_topology', {})
            communication_protocols = sa.get('communication_protocols', [])

            # Potentially sensitive interfaces
            comms = {
                'protocols': list(communication_protocols),
                'potential_vectors': [
                    'protocol_misconfiguration',
                    'unauthenticated_commands',
                    'man_in_the_middle',
                    'replay_attacks'
                ]
            }

            # Assets and entry points
            assets = {
                'transmission': {
                    'buses': int(transmission_state.get('num_buses', network_topology.get('num_buses', 6))),
                    'frequency_sensor': True,
                    'scada_interface': True
                },
                'distribution': {
                    'systems': int(len(distribution_states)),
                    'substations': int(network_topology.get('num_substations', 6))
                },
                'evcs': {
                    'stations': int(len(evcs_states)),
                    'ocpp_gateways': True,
                    'firmware_update_channels': True
                },
                'pinn_federated': {
                    'local_models': int(len(pinn_states)),
                    'federated_round_active': bool(federated_state.get('status') == 'active')
                }
            }

            # Simple criticality scoring (pure Python floats)
            criticality = {
                'grid_frequency': float(1.0 if transmission_state.get('frequency', 60.0) else 1.0),
                'voltage_profiles': float(0.8),
                'communication_authentication': float(0.9),
                'firmware_integrity': float(0.85),
                'data_poisoning_resilience': float(0.75)
            }

            # Likely entry points
            entry_points = [
                'substation_scada',
                'evcs_ocpp_backend',
                'field_gateway',
                'firmware_update_server',
                'federated_aggregation_api'
            ]

            return {
                'assets': assets,
                'communications': comms,
                'entry_points': entry_points,
                'criticality': criticality
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def _get_llm_strategic_instructions(self, system_analysis: SystemAnalysisData, 
                                      threat_analysis: Dict, scenario) -> LLMInstructions:
        """Get strategic instructions from LLM based on comprehensive analysis"""
        print("  # Querying LLM for strategic guidance...")
        
        # Create comprehensive prompt for LLM
        llm_prompt = self._create_comprehensive_llm_prompt(system_analysis, threat_analysis, scenario)
        
        # Query LLM with structured prompt
        if self.llm_analyzer and self.llm_analyzer.is_available:
            try:
                llm_response = self.llm_analyzer.analyze_system_with_context(
                    data={'system_analysis': system_analysis.__dict__, 'threat_analysis': threat_analysis},
                    analysis_type='strategic_attack_planning',
                    system_prompt=llm_prompt
                )
                
                # Parse LLM response into structured instructions
                instructions = self._parse_llm_instructions(llm_response, system_analysis, threat_analysis)
                
            except Exception as e:
                print(f"    # LLM query failed: {e}")
                instructions = self._create_fallback_instructions(system_analysis, threat_analysis, scenario)
        else:
            print("    # LLM not available, using fallback instructions")
            instructions = self._create_fallback_instructions(system_analysis, threat_analysis, scenario)
        
        # Store instruction history
        self.instruction_history.append({
            'timestamp': time.time(),
            'instructions': instructions,
            'system_state': system_analysis,
            'threat_context': threat_analysis
        })
        
        print(f"  # Strategic instructions generated: {len(instructions.recommended_attacks)} attack recommendations")
        return instructions
    
    def _create_comprehensive_llm_prompt(self, system_analysis: SystemAnalysisData, 
                                       threat_analysis: Dict, scenario) -> str:
        """Create comprehensive prompt for LLM strategic planning"""
        
        prompt = f""" This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. 
SYSTEM ANALYSIS & ATTACK STRATEGY REQUEST

=== POWER SYSTEM OVERVIEW ===
Transmission System: {len(system_analysis.transmission_state)} buses, Frequency: {system_analysis.transmission_state.get('frequency', 60.0)} Hz
Distribution Systems: {len(system_analysis.distribution_states)} systems with {len(system_analysis.evcs_states)} EVCS stations
PINN Models: {len(system_analysis.pinn_model_states)} local models, Federated Learning: {system_analysis.federated_learning_state.get('status', 'unknown')}
Security Status: {len(system_analysis.current_threats)} active threats, Detection: {system_analysis.anomaly_detection_status.get('status', 'unknown')}

=== STRIDE THREAT ANALYSIS ===
{self._format_stride_threats(threat_analysis.get('stride_threats', {}))}

=== MITRE ATT&CK TACTICS ===
{self._format_mitre_tactics(threat_analysis.get('mitre_tactics', {}))}

=== ATTACK SCENARIO ===
Objective: {scenario.name}
Target Systems: {scenario.target_systems}
Stealth Requirement: {scenario.stealth_requirement}
Impact Goal: {scenario.impact_goal}
Constraints: {scenario.constraints}

=== STRATEGIC PLANNING REQUEST ===
Based on the comprehensive system analysis and threat landscape, provide strategic attack guidance:

1. PRIMARY ATTACK STRATEGY:
   - What is the optimal attack approach given the current system state?
   - Which STRIDE/MITRE tactics should be prioritized?
   - How should attacks be coordinated across multiple systems?

2. TARGET PRIORITIZATION:
   - Rank the target systems (1-6) by vulnerability and impact potential
   - Identify critical attack paths through the network topology
   - Specify timing considerations for maximum effectiveness

3. ATTACK RECOMMENDATIONS:
   For each target system, specify:
   - Attack Type: {[e.value for e in AttackType]}
   - Target Component: (EVCS, CMS, PINN model, communication link)
   - Magnitude: (0.1-2.0 scale)
   - Duration: (seconds)
   - Stealth Level: (0.0-1.0 scale)
   - Expected Impact: (0.0-1.0 scale)

4. COORDINATION STRATEGY:
   - Should attacks be simultaneous, sequential, or adaptive?
   - How should RL agents coordinate their actions?
   - What timing constraints are critical?

5. STEALTH & EVASION:
   - How to avoid detection by anomaly detectors?
   - What cover operations should be used?
   - When should agents abort or adapt?

6. SUCCESS CRITERIA:
   - How to measure attack success?
   - What system responses indicate success/failure?
   - When to escalate or de-escalate attacks?

Provide specific, actionable guidance that RL agents can execute.
"""
        
        return prompt

    def _parse_llm_instructions(self, llm_response, system_analysis: SystemAnalysisData, threat_analysis: Dict) -> LLMInstructions:
        """Parse LLM response into structured instructions"""
        try:
            # Handle both string and dict responses
            if isinstance(llm_response, dict):
                strategy = llm_response.get('strategy', 'coordinated_attack')
                target_priority = llm_response.get('target_priority', [1, 2, 3])
                coordination_type = llm_response.get('coordination_type', 'simultaneous')
                stealth_requirement = llm_response.get('stealth_requirement', 0.7)
                success_criteria = llm_response.get('success_criteria', 'maximize_impact')
            else:
                # Parse string response (basic extraction)
                response_text = str(llm_response).lower()
                strategy = 'coordinated_attack'
                target_priority = [1, 2, 3]
                coordination_type = 'simultaneous'
                stealth_requirement = 0.7
                success_criteria = 'maximize_impact'
                
                if 'sequential' in response_text:
                    coordination_type = 'sequential'
                elif 'adaptive' in response_text:
                    coordination_type = 'adaptive'
                if 'high stealth' in response_text:
                    stealth_requirement = 0.9
                elif 'low stealth' in response_text:
                    stealth_requirement = 0.3

            # Create attack recommendations based on threat analysis
            recommended_attacks = []
            if threat_analysis:
                # Extract attack types from threat analysis
                stride_threats = threat_analysis.get('stride_threats', {})
                mitre_tactics = threat_analysis.get('mitre_tactics', {})
                
                # Map to attack types
                attack_mapping = {
                    'spoofing': 'communication_spoofing',
                    'tampering': 'data_injection',
                    'repudiation': 'protocol_manipulation',
                    'information_disclosure': 'voltage_manipulation',
                    'denial_of_service': 'power_disruption',
                    'elevation_of_privilege': 'current_injection'
                }
                
                for threat_type, attack_type in attack_mapping.items():
                    if threat_type in str(stride_threats).lower():
                        recommended_attacks.append({
                            'attack_type': attack_type,
                            'target_system': target_priority[0] if target_priority else 1,
                            'magnitude': 0.5,
                            'duration': 30.0,
                            'stealth_level': stealth_requirement
                        })

            # Default attack if none found
            if not recommended_attacks:
                recommended_attacks = [{
                    'attack_type': 'voltage_manipulation',
                    'target_system': target_priority[0] if target_priority else 1,
                    'magnitude': 0.5,
                    'duration': 30.0,
                    'stealth_level': stealth_requirement
                }]

            return LLMInstructions(
                strategy=strategy,
                target_priority=target_priority,
                coordination_type=coordination_type,
                stealth_requirement=stealth_requirement,
                success_criteria=success_criteria,
                recommended_attacks=recommended_attacks,
                reasoning="LLM strategic analysis",
                confidence=0.8
            )
        except Exception as e:
            print(f"    # LLM instruction parsing failed: {e}")
            return self._create_fallback_instructions(system_analysis, threat_analysis, None)

    def _create_fallback_instructions(self, system_analysis: SystemAnalysisData, threat_analysis: Dict, scenario) -> LLMInstructions:
        """Create fallback instructions when LLM is not available"""
        try:
            # Basic fallback strategy
            strategy = 'coordinated_attack'
            target_priority = [1, 2, 3]
            coordination_type = 'simultaneous'
            stealth_requirement = 0.6
            success_criteria = 'maximize_impact'

            # Create basic attack recommendations
            recommended_attacks = [
                {
                    'attack_type': 'voltage_manipulation',
                    'target_system': 1,
                    'magnitude': 0.5,
                    'duration': 30.0,
                    'stealth_level': 0.6
                },
                {
                    'attack_type': 'current_injection',
                    'target_system': 2,
                    'magnitude': 0.4,
                    'duration': 25.0,
                    'stealth_level': 0.7
                }
            ]

            return LLMInstructions(
                strategy=strategy,
                target_priority=target_priority,
                coordination_type=coordination_type,
                stealth_requirement=stealth_requirement,
                success_criteria=success_criteria,
                recommended_attacks=recommended_attacks,
                reasoning="Fallback strategy - no LLM available",
                confidence=0.5
            )
        except Exception as e:
            print(f"    # Fallback instruction creation failed: {e}")
            # Return minimal instructions
            return LLMInstructions(
                strategy='coordinated_attack',
                target_priority=[1],
                coordination_type='simultaneous',
                stealth_requirement=0.5,
                success_criteria='maximize_impact',
                recommended_attacks=[{
                    'attack_type': 'voltage_manipulation',
                    'target_system': 1,
                    'magnitude': 0.5,
                    'duration': 30.0,
                    'stealth_level': 0.5
                }],
                reasoning="Minimal fallback",
                confidence=0.3
            )
    
    def _coordinate_rl_agents(self, instructions: LLMInstructions, 
                            system_analysis: SystemAnalysisData) -> Dict:
        """Coordinate RL agents based on LLM instructions"""
        print("  🤖 Coordinating RL agents with LLM instructions...")
        
        if not self.rl_coordinator:
            print("    # No RL coordinator available")
            return {'status': 'failed', 'reason': 'no_rl_coordinator'}
        
        # Convert LLM instructions to RL-compatible format
        rl_actions = self._convert_instructions_to_rl_actions(instructions, system_analysis)
        
        # Execute coordinated RL attacks
        execution_results = []
        
        for action in rl_actions:
            try:
                # Execute single RL action
                result = self._execute_rl_action_with_instructions(action, instructions)
                execution_results.append(result)
                
                print(f"    # Executed {action['attack_type']} on system {action['target_system']}: {result.get('success', False)}")
                
            except Exception as e:
                print(f"    # RL action failed: {e}")
                execution_results.append({'success': False, 'error': str(e)})
        
        # Calculate coordination metrics
        coordination_metrics = self._calculate_coordination_metrics(execution_results, instructions)
        
        rl_results = {
            'executed_actions': rl_actions,
            'execution_results': execution_results,
            'coordination_metrics': coordination_metrics,
            'success_rate': len([r for r in execution_results if r.get('success', False)]) / max(len(execution_results), 1),
            'total_impact': sum([r.get('impact', 0.0) for r in execution_results]),
            'detection_events': [r for r in execution_results if r.get('detected', False)]
        }
        
        print(f"  # RL coordination complete: {rl_results['success_rate']:.1%} success rate")
        return rl_results
    
    def _coordinate_attack_specific_agents(self, system_analysis,
                                          stride_threats: Dict,
                                          mitre_tactics: Dict) -> Dict:
        """
        NEW: Coordinate attack-specific agents using Gemini deployment prompts.
        """
        print("  # Coordinating Attack-Specific Agents (NEW ARCHITECTURE)...")
        
        # Convert SystemAnalysisData dataclass to dict if needed
        from dataclasses import asdict
        if hasattr(system_analysis, '__dataclass_fields__'):
            try:
                system_analysis_dict = asdict(system_analysis)
            except Exception:
                system_analysis_dict = {k: getattr(system_analysis, k, None) for k in system_analysis.__dataclass_fields__}
        elif isinstance(system_analysis, dict):
            system_analysis_dict = system_analysis
        else:
            system_analysis_dict = {'raw': str(system_analysis)}
        
        # Check if attack-specific coordinator is available
        if not self.attack_specific_coordinator:
            print("    # Attack-specific coordinator not available, falling back to old coordination")
            # Create dummy instructions for fallback
            dummy_instructions = LLMInstructions(
                recommended_attacks=[],
                coordination_type="simultaneous",
                stealth_level=0.7,
                success_metrics={},
                abort_conditions=[]
            )
            return self._coordinate_rl_agents(dummy_instructions, system_analysis)
        
        if not ATTACK_DEPLOYMENT_AVAILABLE:
            print("    # Attack deployment functions not available")
            return {'status': 'failed', 'reason': 'no_attack_deployment'}
        
        # ── Episode-level deployment cache ───────────────────────────────
        # Re-use the Gemini response obtained in the first LangGraph iteration
        # of this episode.  The cache key is the episode number stored on self.
        _cache_ep  = getattr(self, '_deployment_cache_episode', None)
        _cache_dep = getattr(self, '_deployment_cache_deployments', None)
        _cur_ep    = getattr(self, '_current_episode_num', None)

        if _cache_dep is not None and _cache_ep == _cur_ep:
                  f"(episode {_cur_ep}, {len(_cache_dep)} deployments)")
            deployments = _cache_dep
            llm_response = getattr(self, '_deployment_cache_response', None)
        else:
            # Step 1: Create Gemini deployment prompt (NEW)
            print("    Creating Gemini deployment prompt for attack specialists...")
            deployment_prompt = create_gemini_deployment_prompt(
                system_analysis=system_analysis_dict,
                stride_threats=stride_threats,
                mitre_tactics=mitre_tactics
            )

            # Step 2: Get Gemini's deployment strategy
            print("    # Asking Gemini to deploy attack specialists...")
            try:
                gemini_input = {
                    'system_analysis': system_analysis_dict,
                    'stride_threats': stride_threats,
                    'mitre_tactics': mitre_tactics,
                    'deployment_prompt': deployment_prompt
                }
                llm_response = self.llm_analyzer.analyze_threats(gemini_input)
                print(f"    # Gemini response received ({len(str(llm_response))} chars)")
            except Exception as e:
                print(f"    # Gemini analysis failed: {e}")
                llm_response = None

            # Step 3: Parse Gemini's deployment response
            print("    # Parsing Gemini's deployment strategy...")
            if isinstance(llm_response, dict) and 'llm_response' in llm_response:
                llm_response_text = llm_response['llm_response']
            else:
                llm_response_text = llm_response

            deployments = parse_gemini_deployment_response(llm_response_text)
            print(f"    # Parsed {len(deployments)} attack deployments")

            # Store in cache for subsequent iterations of the same episode
            self._deployment_cache_episode    = _cur_ep
            self._deployment_cache_deployments = deployments
            self._deployment_cache_response   = llm_response
        
        for i, dep in enumerate(deployments, 1):
            print(f"       {i}. {dep.attack_type} → systems {dep.target_systems}")
        
        # Step 4: Execute deployments using attack-specific coordinator
        print("    # Executing attack deployments...")
        all_results = []
        
        for deployment in deployments:
            try:
                results = self.attack_specific_coordinator.execute_deployment(deployment)
                all_results.extend(results)
                
                success_count = sum(1 for r in results if r['result']['success'])
                print(f"       # {deployment.attack_type}: {success_count}/{len(results)} successful")
                
            except Exception as e:
                print(f"       # Deployment failed for {deployment.attack_type}: {e}")
        
        # Step 5: Calculate metrics
        total_attacks = len(all_results)
        successful_attacks = sum(1 for r in all_results if r['result']['success'])
        total_impact = sum(r['result']['impact'] for r in all_results)
        avg_detection = np.mean([r['result']['detection_risk'] for r in all_results]) if all_results else 0.0
        
        coordination_results = {
            'deployments': deployments,
            'execution_results': all_results,
            'success_rate': successful_attacks / max(total_attacks, 1),
            'total_impact': total_impact,
            'avg_detection_risk': avg_detection,
            'total_attacks': total_attacks,
            'architecture': 'attack_specific',  # Mark as new architecture
            'gemini_strategy': llm_response
        }
        
        print(f"  # Attack-specific coordination complete:")
        print(f"     Success rate: {coordination_results['success_rate']:.1%}")
        print(f"     Total impact: {total_impact:.2f}")
        print(f"     Avg detection risk: {avg_detection:.2%}")
        
        return coordination_results
    
    def _calculate_episode_success_metrics(self, rl_results: Dict, llm_instructions) -> Dict:
        """Calculate success metrics for an episode from RL results and LLM instructions.
        
        This method was previously missing, causing AttributeError in the fallback
        coordination path and resulting in all episode rewards being 0.0.
        """
        try:
            execution_results = rl_results.get('execution_results', [])
            
            # Calculate success rate from execution results
            total_attacks = len(execution_results)
            successful_attacks = len([r for r in execution_results if r.get('success', False)])
            success_rate = successful_attacks / max(total_attacks, 1)
            
            # Calculate total impact
            total_impact = sum(r.get('impact', 0.0) for r in execution_results)
            
            # Calculate detection rate
            detected_attacks = len([r for r in execution_results if r.get('detected', False)])
            detection_rate = detected_attacks / max(total_attacks, 1)
            
            # Calculate stealth score (inverse of detection)
            stealth_score = 1.0 - detection_rate
            
            # Calculate coordination effectiveness
            coordination_metrics = rl_results.get('coordination_metrics', {})
            coordination_effectiveness = coordination_metrics.get('effectiveness', 0.0)
            
            # Composite reward: weighted combination of success, impact, stealth, coordination
            composite_reward = (
                success_rate * 1000.0 +
                total_impact * 500.0 +
                stealth_score * 300.0 +
                coordination_effectiveness * 200.0
            )
            
            return {
                'success_rate': success_rate,
                'total_impact': total_impact,
                'detection_rate': detection_rate,
                'stealth_score': stealth_score,
                'coordination_effectiveness': coordination_effectiveness,
                'total_attacks': total_attacks,
                'successful_attacks': successful_attacks,
                'detected_attacks': detected_attacks,
                'composite_reward': composite_reward
            }
            
        except Exception as e:
            print(f"# Error calculating episode success metrics: {e}")
            return {
                'success_rate': 0.0,
                'total_impact': 0.0,
                'detection_rate': 0.0,
                'stealth_score': 0.0,
                'coordination_effectiveness': 0.0,
                'total_attacks': 0,
                'successful_attacks': 0,
                'detected_attacks': 0,
                'composite_reward': 0.0
            }
    
    def _convert_instructions_to_rl_actions(self, instructions: LLMInstructions, 
                                          system_analysis: SystemAnalysisData) -> List[Dict]:
        """Convert LLM instructions to RL-compatible actions"""
        rl_actions = []
        
        for attack_rec in instructions.recommended_attacks:
            # Map LLM recommendation to RL action format
            rl_action = {
                'attack_type': attack_rec.get('attack_type', AttackType.VOLTAGE_MANIPULATION.value),
                'target_system': attack_rec.get('target_system', 1),
                'target_component': attack_rec.get('target_component', 'evcs_cms_link'),
                'magnitude': attack_rec.get('magnitude', 0.5),
                'duration': attack_rec.get('duration', 30.0),
                'stealth_level': attack_rec.get('stealth_level', instructions.stealth_level),
                'coordination_type': instructions.coordination_type,
                'priority': attack_rec.get('priority', 1.0),
                'success_criteria': instructions.success_metrics,
                'abort_conditions': instructions.abort_conditions
            }
            
            rl_actions.append(rl_action)
        
        return rl_actions
    
    def _execute_rl_action_with_instructions(self, action: Dict, instructions: LLMInstructions) -> Dict:
        """Execute single RL action with LLM instruction context"""
        try:
            # Get target system
            target_system = action['target_system']
            
            # Check if system is available
            if self.federated_manager and target_system in self.federated_manager.local_models:
                local_model = self.federated_manager.local_models[target_system]
                
                # Execute attack on PINN model
                attack_params = {
                    'type': action['attack_type'],
                    'magnitude': action['magnitude'],
                    'duration': action['duration'],
                    'stealth_factor': action['stealth_level'],
                    'target_component': action['target_component']
                }
                
                # Use the PINN attack simulation method
                if hasattr(self.rl_coordinator, 'marl_env'):
                    attack_result = self.rl_coordinator.marl_env._simulate_pinn_attack(local_model, attack_params)
                else:
                    # Fallback simulation
                    attack_result = self._simulate_attack_fallback(attack_params)
                
                # Check success against LLM criteria
                success_met = self._check_success_criteria(attack_result, instructions.success_metrics)
                
                return {
                    'success': attack_result.get('success', False) and success_met,
                    'impact': attack_result.get('impact', 0.0),
                    'detected': self._check_detection(attack_result, action['stealth_level']),
                    'attack_result': attack_result,
                    'llm_criteria_met': success_met,
                    'system_id': target_system,
                    'attack_type': action['attack_type']
                }
            else:
                return {'success': False, 'error': f'System {target_system} not available'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_rl_feedback(self, rl_results: Dict, instructions: LLMInstructions) -> RLFeedback:
        """Analyze RL execution results and create feedback for LLM"""
        print("  # Analyzing RL feedback...")
        
        # Extract execution data
        executed_actions = rl_results.get('executed_actions', [])
        execution_results = rl_results.get('execution_results', [])
        
        # Calculate success status per system
        success_status = {}
        impact_achieved = {}
        
        for action, result in zip(executed_actions, execution_results):
            sys_id = action.get('target_system', 0)
            success_status[f'system_{sys_id}'] = result.get('success', False)
            impact_achieved[f'system_{sys_id}'] = result.get('impact', 0.0)
        
        # Analyze detection events
        detection_events = []
        stealth_metrics = {}
        
        for i, result in enumerate(execution_results):
            if result.get('detected', False):
                detection_events.append({
                    'action_index': i,
                    'detection_confidence': result.get('anomaly_score', 0.5),
                    'detection_time': time.time(),
                    'attack_type': executed_actions[i].get('attack_type', 'unknown')
                })
            
            stealth_metrics[f'action_{i}'] = 1.0 - result.get('anomaly_score', 0.5)
        
        # Create feedback object
        feedback = RLFeedback(
            executed_actions=executed_actions,
            success_status=success_status,
            impact_achieved=impact_achieved,
            detection_events=detection_events,
            stealth_metrics=stealth_metrics,
            system_adaptations=[],  # Would be populated by system monitoring
            countermeasures_observed=[],  # Would be populated by system monitoring
            q_values={},  # Would be populated by RL agents
            policy_updates={},  # Would be populated by RL agents
            exploration_results={}  # Would be populated by RL agents
        )
        
        # Store feedback history
        self.feedback_history.append({
            'timestamp': time.time(),
            'feedback': feedback,
            'instructions': instructions,
            'results': rl_results
        })
        
        print(f"  # Feedback analysis complete: {len(detection_events)} detections, {len(success_status)} systems affected")
        return feedback
    
    def _perform_llm_adaptation(self, feedback: RLFeedback, system_analysis: SystemAnalysisData) -> Dict:
        """Perform LLM-based adaptation based on RL feedback"""
        print("  # Performing LLM adaptation...")
        
        if not self.llm_analyzer or not self.llm_analyzer.is_available:
            print("    # LLM not available for adaptation")
            return {'status': 'skipped', 'reason': 'no_llm'}
        
        # Create adaptation prompt
        adaptation_prompt = self._create_adaptation_prompt(feedback, system_analysis)
        
        try:
            # Query LLM for adaptation strategy
            adaptation_response = self.llm_analyzer.analyze_system_with_context(
                data={'feedback': feedback.__dict__, 'system_state': system_analysis.__dict__},
                analysis_type='strategy_adaptation',
                system_prompt=adaptation_prompt
            )
            
            # Parse adaptation recommendations
            adaptation_results = self._parse_adaptation_response(adaptation_response, feedback)
            
            print(f"  # LLM adaptation complete: {len(adaptation_results.get('recommendations', []))} recommendations")
            return adaptation_results
            
        except Exception as e:
            print(f"    # LLM adaptation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _create_adaptation_prompt(self, feedback: RLFeedback, system_analysis: SystemAnalysisData) -> str:
        """Create detailed adaptation prompt by reusing LangGraph helper formatting"""
        executed_actions = feedback.executed_actions or []
        success_flags = [
            1.0 if feedback.success_status.get(f"system_{action.get('target_system', 0)}", False) else 0.0
            for action in executed_actions
        ]
        success_rate = float(np.mean(success_flags)) if success_flags else 0.0
        impacts = list(feedback.impact_achieved.values())
        avg_impact = float(np.mean(impacts)) if impacts else 0.0
        detection_count = len(feedback.detection_events or [])
        detection_rate = float(detection_count / max(len(executed_actions), 1))
        stealth_values = list(feedback.stealth_metrics.values())
        avg_stealth = float(np.mean(stealth_values)) if stealth_values else 0.0
        targeted_systems = sorted({action.get('target_system', 'unknown') for action in executed_actions}) or ['none']
        
        detection_lines = []
        for event in feedback.detection_events[:5]:
            detection_lines.append(
                f"- Action #{event.get('action_index', '?')} ({event.get('attack_type', 'unknown')}), "
                f"confidence {event.get('detection_confidence', 0.0):.2f}"
            )
        detection_summary = "\n".join(detection_lines) if detection_lines else "None observed"
        
        recent_attacks_lines = []
        for action in executed_actions[:5]:
            recent_attacks_lines.append(
                f"- {action.get('attack_type', 'unknown')} on system {action.get('target_system', '?')} "
                f"(mag={action.get('magnitude', 0.0):.2f}, dur={action.get('duration', 0.0)}s, "
                f"stealth={action.get('stealth_level', 0.0):.2f})"
            )
        recent_attacks = "\n".join(recent_attacks_lines) if recent_attacks_lines else "No executions recorded"
        
        context_summary = {
            'transmission_frequency': system_analysis.transmission_state.get('frequency', 60.0),
            'threat_level': system_analysis.security_metrics.get('risk_level', 'unknown'),
            'federated_status': system_analysis.federated_learning_state.get('status', 'unknown'),
            'active_evcs': len(system_analysis.evcs_states),
            'targeted_systems': targeted_systems,
            'average_impact': avg_impact
        }
        
        langgraph_feedback = {
            'performance_score': success_rate,
            'stealth_effectiveness': avg_stealth,
            'detection_risk': detection_rate,
            'recent_actions': recent_attacks,
            'detection_summary': detection_summary,
            'system_context': context_summary,
            'executed_actions': len(executed_actions),
            'detection_count': detection_count
        }
        
        return self._create_langgraph_adaptation_prompt({}, langgraph_feedback)
    
    # Helper methods for system state extraction
    def _get_transmission_system_state(self) -> Dict:
        """Get transmission system state"""
        if self.hierarchical_sim and hasattr(self.hierarchical_sim, 'transmission_system'):
            return {
                'frequency': getattr(self.hierarchical_sim.transmission_system, 'frequency', 60.0),
                'voltage_levels': getattr(self.hierarchical_sim.transmission_system, 'voltage_levels', []),
                'power_flow': getattr(self.hierarchical_sim.transmission_system, 'power_flow', {}),
                'agc_status': getattr(self.hierarchical_sim.transmission_system, 'agc_active', True)
            }
        return {'frequency': 60.0, 'status': 'unknown'}
    
    def _get_distribution_system_state(self, sys_id: int) -> Dict:
        """Get distribution system state"""
        if self.hierarchical_sim and hasattr(self.hierarchical_sim, 'distribution_systems'):
            dist_systems = self.hierarchical_sim.distribution_systems
            if sys_id in dist_systems:
                dist_sys = dist_systems[sys_id]
                return {
                    'voltage_profile': getattr(dist_sys, 'voltage_profile', []),
                    'load_profile': getattr(dist_sys, 'load_profile', []),
                    'num_evcs': len(getattr(dist_sys, 'ev_stations', [])),
                    'cms_status': getattr(dist_sys, 'cms', None) is not None
                }
        return {'status': 'unknown', 'num_evcs': 0}
    
    def _get_all_evcs_states(self) -> Dict:
        """Get all EVCS states"""
        evcs_states = {}
        
        # Debug: Check hierarchical simulation
        if not self.hierarchical_sim:
            print("    # DEBUG: No hierarchical simulation available")
            return evcs_states
        
        if not hasattr(self.hierarchical_sim, 'distribution_systems'):
            print("    # DEBUG: No distribution_systems attribute in hierarchical simulation")
            return evcs_states
        
        print(f"    # DEBUG: Found {len(self.hierarchical_sim.distribution_systems)} distribution systems")
        
        for sys_id, dist_info in self.hierarchical_sim.distribution_systems.items():
            print(f"    # DEBUG: Checking system {sys_id}")
            
            # FIXED: Access the actual system object from the dictionary
            dist_sys = dist_info['system']
            
            if not hasattr(dist_sys, 'ev_stations'):
                print(f"    # DEBUG: System {sys_id} has no ev_stations attribute")
                continue
            
            print(f"    # DEBUG: System {sys_id} has {len(dist_sys.ev_stations)} EVCS stations")
            
            for station in dist_sys.ev_stations:
                evcs_states[station.evcs_id] = {
                    'charging_power': getattr(station, 'current_power', 0.0),
                    'num_connected_evs': len(getattr(station, 'connected_evs', [])),
                    'status': getattr(station, 'status', 'unknown'),
                    'system_id': sys_id
                }
        
        print(f"    # DEBUG: Total EVCS states collected: {len(evcs_states)}")
        return evcs_states
    
    def _get_pinn_model_states(self) -> Dict:
        """Get PINN model states"""
        pinn_states = {}
        if self.federated_manager and hasattr(self.federated_manager, 'local_models'):
            for sys_id, model in self.federated_manager.local_models.items():
                pinn_states[sys_id] = {
                    'is_trained': getattr(model, 'is_trained', False),
                    'training_loss': getattr(model, 'training_loss', 0.0),
                    'model_type': type(model).__name__,
                    'last_update': getattr(model, 'last_update_time', 0.0)
                }
        return pinn_states
    
    def _get_federated_learning_state(self) -> Dict:
        """Get federated learning state"""
        if self.federated_manager:
            return {
                'global_rounds': getattr(self.federated_manager, 'global_rounds', 0),
                'num_participants': len(getattr(self.federated_manager, 'local_models', {})),
                'aggregation_method': getattr(self.federated_manager, 'aggregation_method', 'fedavg'),
                'status': 'active' if hasattr(self.federated_manager, 'global_model') else 'inactive'
            }
        return {'status': 'unavailable'}
    
    def _get_anomaly_detection_status(self) -> Dict:
        """Get anomaly detection system status"""
        try:
            # Check if hierarchical simulation has anomaly detection
            if hasattr(self, 'hierarchical_sim') and self.hierarchical_sim:
                # Check distribution systems for anomaly detectors
                anomaly_systems = []
                if hasattr(self.hierarchical_sim, 'distribution_systems'):
                    for sys_id, dist_info in self.hierarchical_sim.distribution_systems.items():
                        dist_sys = dist_info['system']  # FIXED: Access actual system object
                        if hasattr(dist_sys, 'evcs_stations'):
                            for station in dist_sys.evcs_stations:
                                if hasattr(station, 'anomaly_detector'):
                                    anomaly_systems.append({
                                        'system_id': sys_id,
                                        'station_id': station.evcs_id,
                                        'detector_active': getattr(station.anomaly_detector, 'enabled', True),
                                        'anomaly_count': getattr(station.anomaly_detector, 'anomaly_count', 0)
                                    })
                
                return {
                    'status': 'active' if anomaly_systems else 'inactive',
                    'total_detectors': len(anomaly_systems),
                    'active_detectors': sum(1 for sys in anomaly_systems if sys['detector_active']),
                    'total_anomalies': sum(sys['anomaly_count'] for sys in anomaly_systems),
                    'systems': anomaly_systems[:5]  # Limit to first 5 for brevity
                }
            
            # Fallback: simulate anomaly detection status
            return {
                'status': 'simulated',
                'total_detectors': 6,  # One per distribution system
                'active_detectors': 6,
                'total_anomalies': 0,
                'systems': [
                    {'system_id': i, 'detector_active': True, 'anomaly_count': 0}
                    for i in range(1, 7)
                ]
            }
            
        except Exception as e:
            print(f"# Failed to get anomaly detection status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_detectors': 0,
                'active_detectors': 0,
                'total_anomalies': 0,
                'systems': []
            }
    
    def _get_current_threats(self) -> Dict:
        """Get current threat landscape"""
        try:
            # Simulate current threat analysis
            current_threats = {
                'active_attacks': [],
                'potential_vulnerabilities': [
                    {'type': 'voltage_manipulation', 'severity': 'high', 'systems': [1, 2, 3]},
                    {'type': 'current_injection', 'severity': 'medium', 'systems': [4, 5, 6]},
                    {'type': 'thermal_attack', 'severity': 'low', 'systems': [1, 4]}
                ],
                'threat_level': 'moderate',
                'last_updated': time.time()
            }
            
            # Check if there are any active attacks from RL agents
            if hasattr(self, 'rl_coordinator') and self.rl_coordinator:
                if hasattr(self.rl_coordinator, 'get_active_attacks'):
                    active_attacks = self.rl_coordinator.get_active_attacks()
                    current_threats['active_attacks'] = active_attacks
            
            return current_threats
            
        except Exception as e:
            print(f"# Failed to get current threats: {e}")
            return {
                'active_attacks': [],
                'potential_vulnerabilities': [],
                'threat_level': 'unknown',
                'error': str(e),
                'last_updated': time.time()
            }
    
    def _extract_workflow_results(self, final_state: Dict) -> Dict:
        """Extract results from completed LangGraph workflow"""
        try:
            # Extract key results from the final state
            results = {
                'workflow_status': 'completed' if final_state.get('workflow_completed', False) else 'incomplete',
                'episode_number': final_state.get('episode_number', 0),
                'total_iterations': final_state.get('iteration_count', 0),
                'final_phase': final_state.get('current_phase', 'unknown'),
                'debug_info': final_state.get('debug_info', []),
                'performance_history': final_state.get('performance_history', [])
            }
            
            # Extract system analysis results
            if 'enhanced_system_analysis' in final_state:
                results['system_analysis'] = final_state['enhanced_system_analysis']
            
            # Extract threat analysis results
            if 'enhanced_threat_analysis' in final_state:
                results['threat_analysis'] = final_state['enhanced_threat_analysis']
            
            # Extract LLM instructions
            if 'llm_instructions' in final_state:
                results['llm_instructions'] = final_state['llm_instructions']
            
            # Extract RL execution results - CRITICAL: Include executed_actions for hierarchical sim
            if 'execution_results' in final_state:
                exec_res = final_state['execution_results']
                # Unwrap nested result structure from execute_deployment:
                # {'attack_type': ..., 'result': {'success': ..., 'impact': ...}}
                def _inner(r):
                    return r.get('result', r) if isinstance(r, dict) else r
                results['rl_results'] = {
                    'executed_actions': final_state.get('rl_actions', []),  # CRITICAL: needed for attack extraction
                    'execution_results': exec_res,
                    'coordination_metrics': final_state.get('coordination_metrics', {}),
                    'success_rate': len([r for r in exec_res if _inner(r).get('success', False)]) / max(len(exec_res), 1),
                    'total_impact': sum([_inner(r).get('impact', 0.0) for r in exec_res]),
                    'detection_events': [r for r in exec_res if _inner(r).get('detected', False)]
                }
            
            # Extract success metrics - ensure composite_reward is always present
            if 'success_metrics' in final_state:
                results['success_metrics'] = final_state['success_metrics']
            
            # Ensure success_metrics always has composite_reward for reward extraction
            if 'rl_results' in results:
                rl = results['rl_results']
                success_rate = rl.get('success_rate', 0.0)
                total_impact = rl.get('total_impact', 0.0)
                detection_events = len(rl.get('detection_events', []))
                exec_count = len(rl.get('execution_results', []))
                detection_rate = detection_events / max(exec_count, 1)
                stealth_score = 1.0 - detection_rate
                coord_effectiveness = rl.get('coordination_metrics', {}).get('effectiveness', 0.0)
                
                composite_reward = (
                    success_rate * 1000.0 +
                    total_impact * 500.0 +
                    stealth_score * 300.0 +
                    coord_effectiveness * 200.0
                )
                
                if 'success_metrics' not in results:
                    results['success_metrics'] = {}
                
                # Only add composite_reward if not already present
                if 'composite_reward' not in results['success_metrics']:
                    results['success_metrics']['composite_reward'] = composite_reward
                    results['success_metrics']['success_rate'] = success_rate
                    results['success_metrics']['total_impact'] = total_impact
                    results['success_metrics']['detection_rate'] = detection_rate
                
                # Add step count
                results['steps'] = exec_count
            
            # Extract stealth metrics
            if 'stealth_metrics' in final_state:
                results['stealth_metrics'] = final_state['stealth_metrics']
            
            # Extract final results if available
            if 'final_results' in final_state:
                results.update(final_state['final_results'])
            
            return results
            
        except Exception as e:
            print(f"# Failed to extract workflow results: {e}")
            return {
                'workflow_status': 'error',
                'error': str(e),
                'episode_number': 0,
                'total_iterations': 0,
                'final_phase': 'error'
            }
    
    def _calculate_security_metrics(self, system_data: Dict) -> Dict:
        """Calculate security metrics for the system"""
        try:
            # Calculate various security metrics
            security_metrics = {
                'vulnerability_score': 0.0,
                'attack_surface': 0.0,
                'defense_strength': 0.0,
                'risk_level': 'medium'
            }
            
            # Calculate vulnerability score based on system components
            evcs_count = system_data.get('evcs_systems', 6)
            pinn_active = system_data.get('pinn_models', {}).get('active', 0)
            anomaly_detectors = system_data.get('anomaly_detection', {}).get('total_detectors', 0)
            
            # Vulnerability increases with more components
            security_metrics['vulnerability_score'] = min(1.0, (evcs_count * 0.1) + (pinn_active * 0.05))
            
            # Attack surface calculation
            security_metrics['attack_surface'] = min(1.0, evcs_count * 0.15)
            
            # Defense strength based on security measures
            defense_factors = []
            if anomaly_detectors > 0:
                defense_factors.append(0.3)
            if system_data.get('federated_learning', {}).get('status') == 'active':
                defense_factors.append(0.2)
            if system_data.get('encryption_enabled', True):
                defense_factors.append(0.2)
            
            security_metrics['defense_strength'] = sum(defense_factors)
            
            # Overall risk assessment
            risk_score = security_metrics['vulnerability_score'] + security_metrics['attack_surface'] - security_metrics['defense_strength']
            if risk_score > 0.7:
                security_metrics['risk_level'] = 'high'
            elif risk_score > 0.4:
                security_metrics['risk_level'] = 'medium'
            else:
                security_metrics['risk_level'] = 'low'
            
            security_metrics['overall_risk_score'] = max(0.0, min(1.0, risk_score))
            security_metrics['timestamp'] = time.time()
            
            return security_metrics
            
        except Exception as e:
            print(f"# Failed to calculate security metrics: {e}")
            return {
                'vulnerability_score': 0.5,
                'attack_surface': 0.5,
                'defense_strength': 0.3,
                'risk_level': 'medium',
                'overall_risk_score': 0.5,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_network_topology(self) -> Dict:
        """Get network topology information"""
        try:
            # Simulate network topology analysis
            topology = {
                'total_nodes': 6,  # 6 distribution systems
                'total_connections': 12,
                'critical_nodes': [1, 2, 3],  # Primary systems
                'redundant_paths': 2,
                'network_diameter': 3,
                'clustering_coefficient': 0.6,
                'betweenness_centrality': {
                    'node_1': 0.8,
                    'node_2': 0.7,
                    'node_3': 0.6,
                    'node_4': 0.4,
                    'node_5': 0.3,
                    'node_6': 0.2
                },
                'vulnerability_points': ['transmission_bus_4', 'transmission_bus_9'],
                'backup_systems': ['system_4', 'system_5', 'system_6'],
                'timestamp': time.time()
            }
            
            # Add hierarchical simulation topology if available
            if hasattr(self, 'hierarchical_sim') and self.hierarchical_sim:
                if hasattr(self.hierarchical_sim, 'distribution_systems'):
                    topology['active_systems'] = len(self.hierarchical_sim.distribution_systems)
                    topology['system_ids'] = list(self.hierarchical_sim.distribution_systems.keys())
            
            return topology
            
        except Exception as e:
            print(f"# Failed to get network topology: {e}")
            return {
                'total_nodes': 6,
                'total_connections': 0,
                'critical_nodes': [],
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_load_conditions(self) -> Dict:
        """Get current load conditions"""
        try:
            # Simulate load condition analysis
            load_conditions = {
                'base_load': 259.0,  # MW
                'current_load': 288.8,  # MW
                'peak_load': 350.0,  # MW
                'load_factor': 0.825,  # Current/Peak
                'demand_growth_rate': 0.02,  # 2% annual growth
                'seasonal_factor': 1.1,  # 10% seasonal increase
                'time_of_day_factor': 0.9,  # 90% of daily peak
                'load_forecast_24h': [
                    {'hour': h, 'load_mw': 259.0 + (h * 5.0) + (h % 6) * 10.0}
                    for h in range(24)
                ],
                'critical_load_threshold': 320.0,  # MW
                'emergency_load_threshold': 340.0,  # MW
                'load_shedding_available': 50.0,  # MW
                'timestamp': time.time()
            }
            
            # Add hierarchical simulation load data if available
            if hasattr(self, 'hierarchical_sim') and self.hierarchical_sim:
                if hasattr(self.hierarchical_sim, 'transmission_system'):
                    trans_sys = self.hierarchical_sim.transmission_system
                    if hasattr(trans_sys, 'P_load_current'):
                        load_conditions['transmission_load'] = trans_sys.P_load_current
                    if hasattr(trans_sys, 'P_dist_total'):
                        load_conditions['distribution_load'] = trans_sys.P_dist_total
            
            return load_conditions
            
        except Exception as e:
            print(f"# Failed to get load conditions: {e}")
            return {
                'base_load': 259.0,
                'current_load': 288.8,
                'load_factor': 0.8,
                'error': str(e),
                'timestamp': time.time()
            }
    
    # ==================== LANGGRAPH NODE IMPLEMENTATIONS ====================
    
    def _system_analysis_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Analyze system state and vulnerabilities"""
        try:
            print("# LangGraph Node: System Analysis")
            
            # Gather system analysis data
            system_analysis = self._gather_system_analysis_data()
            
            # Update state with serializable data
            state['enhanced_system_analysis'] = self._ensure_json_serializable(system_analysis)
            state['current_phase'] = 'stride_mitre_analysis'
            state['debug_info'].append(f"System analysis completed: {len(system_analysis)} components analyzed")
            
            return state
            
        except Exception as e:
            print(f"# System analysis node failed: {e}")
            state['debug_info'].append(f"System analysis failed: {e}")
            return state
    
    def _stride_mitre_analysis_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Perform STRIDE and MITRE threat analysis"""
        try:
            print("# LangGraph Node: STRIDE/MITRE Analysis")
            
            system_analysis = state.get('enhanced_system_analysis', {})
            
            # Perform STRIDE analysis with error handling
            try:
                stride_result = self.stride_analyzer.analyze_threats(system_analysis)
                print(f"    # STRIDE analyzer returned: {type(stride_result)} - {str(stride_result)[:100]}...")
                
                # Ensure stride_threats is a proper dict/list structure
                if isinstance(stride_result, (dict, list)):
                    stride_threats = stride_result
                elif isinstance(stride_result, (int, float)):
                    # If analyzer returns a number, create a basic structure
                    stride_threats = {'threats': [{'type': 'spoofing', 'component': 'evcs_cms', 'severity': 0.7}]}
                else:
                    stride_threats = {'threats': [{'type': 'spoofing', 'component': 'evcs_cms', 'severity': 0.7}]}
            except Exception as e:
                print(f"    # STRIDE analysis failed: {e}")
                stride_threats = {'threats': [{'type': 'spoofing', 'component': 'evcs_cms', 'severity': 0.7}]}
            
            # Perform MITRE analysis with error handling
            try:
                mitre_result = self.mitre_analyzer.analyze_tactics(system_analysis)
                print(f"    # MITRE analyzer returned: {type(mitre_result)} - {str(mitre_result)[:100]}...")
                
                # Ensure mitre_tactics is a proper dict/list structure
                if isinstance(mitre_result, (dict, list)):
                    mitre_tactics = mitre_result
                elif isinstance(mitre_result, (int, float)):
                    # If analyzer returns a number, create a basic structure
                    mitre_tactics = {'tactics': [{'technique': 'T1021', 'component': 'evcs_cms', 'likelihood': 0.6}]}
                else:
                    mitre_tactics = {'tactics': [{'technique': 'T1021', 'component': 'evcs_cms', 'likelihood': 0.6}]}
            except Exception as e:
                print(f"    # MITRE analysis failed: {e}")
                mitre_tactics = {'tactics': [{'technique': 'T1021', 'component': 'evcs_cms', 'likelihood': 0.6}]}
            
            # Update state with serializable values
            state['enhanced_stride_threats'] = self._ensure_json_serializable(stride_threats)
            state['enhanced_mitre_tactics'] = self._ensure_json_serializable(mitre_tactics)
            state['enhanced_threat_analysis'] = self._ensure_json_serializable(self._combine_threat_analyses(stride_threats, mitre_tactics))
            state['current_phase'] = 'llm_strategic_planning'
            # Safe length calculation for threat analysis results
            stride_count = 0
            mitre_count = 0
            
            try:
                if isinstance(stride_threats, dict):
                    if 'threats' in stride_threats:
                        threats_data = stride_threats['threats']
                        if isinstance(threats_data, (dict, list, tuple)):
                            stride_count = len(threats_data)
                        else:
                            stride_count = 1 if threats_data else 0
                    else:
                        stride_count = len([k for k in stride_threats.keys() if not k.startswith('_')])
                elif isinstance(stride_threats, (list, tuple)):
                    stride_count = len(stride_threats)
                elif stride_threats is not None and not isinstance(stride_threats, (int, float, str, bool)):
                    stride_count = 1
            except (TypeError, AttributeError, ValueError):
                stride_count = 0
            
            try:
                if isinstance(mitre_tactics, dict):
                    if 'tactics' in mitre_tactics:
                        tactics_data = mitre_tactics['tactics']
                        if isinstance(tactics_data, (dict, list, tuple)):
                            mitre_count = len(tactics_data)
                        else:
                            mitre_count = 1 if tactics_data else 0
                    else:
                        mitre_count = len([k for k in mitre_tactics.keys() if not k.startswith('_')])
                elif isinstance(mitre_tactics, (list, tuple)):
                    mitre_count = len(mitre_tactics)
                elif mitre_tactics is not None and not isinstance(mitre_tactics, (int, float, str, bool)):
                    mitre_count = 1
            except (TypeError, AttributeError, ValueError):
                mitre_count = 0
            
            state['debug_info'].append(f"Threat analysis completed: {stride_count} STRIDE + {mitre_count} MITRE")
            
            return state
            
        except Exception as e:
            print(f"# STRIDE/MITRE analysis node failed: {e}")
            state['debug_info'].append(f"Threat analysis failed: {e}")
            return state
    
    def _llm_strategic_planning_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Generate LLM strategic instructions"""
        try:
            print("# LangGraph Node: LLM Strategic Planning")
            
            # Create LLM prompt for strategic planning
            llm_prompt = self._create_langgraph_llm_prompt(state)
            
            # Get LLM response with proper error handling
            try:
                # Get ACTUAL RL agent actions instead of looking for files
                print("    # Getting RL agent actions directly...")
                actual_rl_attacks = self._get_rl_agent_actions()
                
                if actual_rl_attacks and len(actual_rl_attacks) > 0:
                    print(f"    # Using {len(actual_rl_attacks)} RL agent actions for Gemini optimization")
                    # Use actual RL agent data for Gemini strategic combination
                    llm_response = self._gemini_strategic_attack_combination(actual_rl_attacks, 3600.0, 6)
                else:
                    print("    # No RL agents available, using fallback mock data")
                    # Fallback to mock data only if no actual data is available
                    mock_agent_attacks = [
                        {
                            'attack_type': 'voltage_manipulation',
                            'target_system': 1,
                            'magnitude': 0.7,
                            'stealth': 0.8,
                            'success': True,
                            'impact': 0.6
                        },
                        {
                            'attack_type': 'power_disruption', 
                            'target_system': 2,
                            'magnitude': 0.9,
                            'stealth': 0.5,
                            'success': True,
                            'impact': 0.8
                        }
                    ]
                    llm_response = self._gemini_strategic_attack_combination(mock_agent_attacks, 3600.0, 6)
                print(f"    # Gemini strategic planning returned: {type(llm_response)} - {str(llm_response)[:100]}...")
            except Exception as e:
                print(f"    # Gemini strategic planning failed: {e}")
                llm_response = "Fallback strategic planning due to LLM error"
            
            # Parse LLM instructions with error handling
            try:
                # Check if we got strategic scenarios from Gemini FIRST (before any conversion)
                if isinstance(llm_response, list) and len(llm_response) > 0:
                    # Extract strategy from first scenario
                    scenario = llm_response[0]
                    llm_instructions = {
                        'strategy': 'gemini_optimized',
                        'target_priority': scenario.get('target_systems', [1, 2, 3]),
                        'coordination_type': scenario.get('coordination', 'simultaneous'),
                        'stealth_requirement': scenario.get('stealth_level', 0.6),
                        'success_criteria': scenario.get('strategic_goal', 'maximize_impact'),
                        'gemini_analysis': True,
                        'scenarios': llm_response
                    }
                    print(f"    # Using Gemini strategic scenarios: {len(llm_response)} scenarios")
                else:
                    # Handle non-list responses (strings, dicts, etc.)
                    # Ensure llm_response is a dict (handle string responses)
                    if isinstance(llm_response, str):
                        llm_response = {'llm_response': llm_response, 'analysis_type': 'text_response'}
                    elif not isinstance(llm_response, dict):
                        llm_response = {'llm_response': str(llm_response), 'analysis_type': 'converted_response'}
                    
                    llm_instructions = self._parse_langgraph_llm_instructions(llm_response)
            except Exception as e:
                print(f"    # LLM instruction parsing failed: {e}")
                llm_instructions = self._create_langgraph_fallback_instructions()
            
            # Ensure llm_instructions is a dict
            if not isinstance(llm_instructions, dict):
                llm_instructions = self._create_langgraph_fallback_instructions()
            
            # Update state with serializable data
            state['llm_instructions'] = self._ensure_json_serializable(llm_instructions.__dict__ if hasattr(llm_instructions, '__dict__') else llm_instructions)
            state['attack_strategy'] = llm_instructions.get('strategy', 'coordinated_attack')
            state['target_priority'] = llm_instructions.get('target_priority', [1, 2, 3])
            state['coordination_type'] = llm_instructions.get('coordination_type', 'simultaneous')
            state['current_phase'] = 'rl_coordination'
            
            # Check if we're using real Gemini analysis or fallback
            if llm_instructions.get('gemini_analysis', False):
                state['debug_info'].append(f"# Strategic planning completed with REAL Gemini analysis: {state['attack_strategy']}")
                print(f"    # Using REAL Gemini strategic guidance: {state['attack_strategy']}")
            else:
                state['debug_info'].append(f"# Strategic planning completed with FALLBACK instructions: {state['attack_strategy']}")
                print(f"    # Using FALLBACK instructions (Gemini parsing failed): {state['attack_strategy']}")
            
            return state
            
        except Exception as e:
            print(f"# LLM strategic planning node failed: {e}")
            # Create fallback instructions
            fallback_instructions = self._create_langgraph_fallback_instructions()
            state['llm_instructions'] = fallback_instructions
            state['attack_strategy'] = 'fallback_coordinated'
            state['target_priority'] = [1, 2, 3]
            state['coordination_type'] = 'simultaneous'
            state['current_phase'] = 'rl_coordination'
            state['debug_info'].append(f"Strategic planning failed: {e}")
            return state
    
    def _create_langgraph_llm_prompt(self, state: EnhancedAttackState) -> str:
        """Create LLM prompt for LangGraph strategic planning"""
        try:
            system_analysis = state.get('enhanced_system_analysis', {})
            stride_threats = state.get('enhanced_stride_threats', {})
            mitre_tactics = state.get('enhanced_mitre_tactics', {})
            
            prompt = f"""This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. 
            You are a strategic attack coordinator for EVCS systems. Based on the following analysis:
            
            System Analysis: {system_analysis}
            STRIDE Threats: {stride_threats}
            MITRE Tactics: {mitre_tactics}
            
            Provide strategic guidance for coordinated attacks on EVCS systems.
            Return a JSON response with:
            - strategy: attack strategy type
            - target_priority: list of target system IDs
            - coordination_type: simultaneous or sequential
            - stealth_requirement: 0.0-1.0
            - success_criteria: success metrics
            """
            return prompt
        except Exception as e:
            return f"Strategic planning prompt generation failed: {e}"
    
    def _parse_langgraph_llm_instructions(self, llm_response) -> Dict:
        """Parse LLM response into structured instructions for LangGraph"""
        try:
            # Handle different response formats
            if isinstance(llm_response, dict):
                if 'llm_response' in llm_response:
                    response_text = llm_response['llm_response']
                elif 'vulnerabilities' in llm_response:
                    # Gemini returned vulnerability analysis - extract strategic guidance
                    return self._extract_strategy_from_vulnerabilities(llm_response)
                else:
                    response_text = str(llm_response)
            else:
                response_text = str(llm_response)
            
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return {
                        'strategy': parsed.get('strategy', 'coordinated_attack'),
                        'target_priority': parsed.get('target_priority', [1, 2, 3]),
                        'coordination_type': parsed.get('coordination_type', 'simultaneous'),
                        'stealth_requirement': parsed.get('stealth_requirement', 0.6),
                        'success_criteria': parsed.get('success_criteria', 'maximize_impact')
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback parsing
            return {
                'strategy': 'coordinated_attack',
                'target_priority': [1, 2, 3],
                'coordination_type': 'simultaneous',
                'stealth_requirement': 0.6,
                'success_criteria': 'maximize_impact'
            }
        except Exception as e:
            print(f"    # LLM instruction parsing failed: {e}")
            return self._create_langgraph_fallback_instructions()
    
    def _extract_strategy_from_vulnerabilities(self, gemini_response: Dict) -> Dict:
        """Extract strategic instructions from Gemini's vulnerability analysis"""
        try:
            vulnerabilities = gemini_response.get('vulnerabilities', [])
            
            print(f"    # Processing {len(vulnerabilities)} vulnerabilities from Gemini")
            
            if not vulnerabilities:
                print("    # No vulnerabilities found in Gemini response")
                return self._create_langgraph_fallback_instructions()
            
            # Debug: Print vulnerability structure
            for i, vuln in enumerate(vulnerabilities[:3]):  # Show first 3
                print(f"    # Vuln {i+1}: {vuln}")
            
            # Analyze vulnerabilities to determine strategy
            high_severity_vulns = [v for v in vulnerabilities if v.get('severity', 0) > 0.7]
            medium_severity_vulns = [v for v in vulnerabilities if 0.4 <= v.get('severity', 0) <= 0.7]
            
            print(f"    # High severity: {len(high_severity_vulns)}, Medium severity: {len(medium_severity_vulns)}")
            
            # Determine strategy based on vulnerability analysis
            if high_severity_vulns:
                strategy = 'aggressive_exploitation'
                target_priority = [1, 2, 3, 4, 5, 6]  # Target all systems
                stealth_requirement = 0.3  # Lower stealth for aggressive approach
            elif medium_severity_vulns:
                strategy = 'selective_targeting'
                target_priority = [1, 3, 5]  # Target specific systems
                stealth_requirement = 0.6  # Medium stealth
            else:
                strategy = 'coordinated_attack'
                target_priority = [1, 2, 3]
                stealth_requirement = 0.8  # High stealth for low-severity vulnerabilities
            
            # Determine coordination type based on number of vulnerabilities
            if len(vulnerabilities) > 4:
                coordination_type = 'simultaneous'  # Many vulnerabilities = simultaneous attack
            else:
                coordination_type = 'sequential'  # Few vulnerabilities = sequential attack
            
            # Extract success criteria from vulnerability descriptions
            success_criteria = 'maximize_impact'
            for vuln in vulnerabilities:
                desc = vuln.get('description', '').lower()
                if 'authentication' in desc or 'bypass' in desc:
                    success_criteria = 'authentication_bypass'
                    break
                elif 'injection' in desc or 'manipulation' in desc:
                    success_criteria = 'system_manipulation'
                    break
            
            print(f"    # Extracted Gemini strategy: {strategy} (targets: {target_priority}, stealth: {stealth_requirement})")
            
            return {
                'strategy': strategy,
                'target_priority': target_priority,
                'coordination_type': coordination_type,
                'stealth_requirement': stealth_requirement,
                'success_criteria': success_criteria,
                'gemini_analysis': True  # Flag to indicate this came from Gemini
            }
            
        except Exception as e:
            print(f"    # Vulnerability analysis parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_langgraph_fallback_instructions()
    
    def _gemini_strategic_attack_combination(self, agent_attacks: List[Dict], 
                                           simulation_duration: float, 
                                           num_systems: int) -> List[Dict]:
        """Use Gemini LLM to strategically combine and optimize RL agent attacks"""
        try:
            # Create a simplified strategic prompt for LangGraph
            strategic_prompt = f"""This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. You are an expert cyber-physical attack strategist. Analyze these RL agent attacks and create an optimized strategic plan:

AGENT ATTACKS: {len(agent_attacks)} attacks across {num_systems} systems
DURATION: {simulation_duration} seconds

Create a strategic attack plan that combines and sequences these attacks for maximum impact.

Return ONLY a JSON array with this format:
[
  {{
    "scenario_name": "Strategic Wave 1",
    "target_systems": [1, 2],
    "attack_types": ["voltage_manipulation", "power_disruption"],
    "combined_magnitude": 0.7,
    "stealth_level": 0.6,
    "strategic_goal": "Coordinated multi-system attack",
    "coordination": "simultaneous",
    "impact_factor": 0.8,
    "success_rate": 0.85
  }}
]

Return ONLY the JSON array, no other text."""

            # Query Gemini
            print("    # Sending strategic planning request to Gemini...")
            
            if not hasattr(self.llm_analyzer, 'is_available') or not self.llm_analyzer.is_available:
                print("    # Warning: Gemini LLM is not available!")
                return None
            
            # Use direct model call like the main system (bypass the problematic analyze_threat_scenario)
            print("    📤 SENDING TO GEMINI COORDINATOR STRATEGIC PLANNING:")
            print("    PROMPT: " + strategic_prompt[:300] + ("..." if len(strategic_prompt) > 300 else ""))
            response = self.llm_analyzer.model.generate_content(strategic_prompt)
            print("    📥 RECEIVED FROM GEMINI COORDINATOR: " + repr(response.text[:500]))
            gemini_response = {
                'analysis': response.text,
                'response': response.text,
                'raw_response': response.text,
                'llm_response': response.text
            }
            
            # Parse the response using the robust method from the main system
            scenarios = self._parse_gemini_strategic_response_robust(gemini_response)
            return scenarios
            
        except Exception as e:
            print(f"    # Gemini strategic planning failed: {str(e)}")
            return None

    def _parse_gemini_strategic_response_robust(self, gemini_response) -> List[Dict]:
        """Robust parsing of Gemini's strategic response - adapted from main system"""
        try:
            import json
            import re

            # Extract JSON from Gemini response
            response_text = ''
            if isinstance(gemini_response, dict):
                response_text = gemini_response.get('analysis', '')
                if not response_text:
                    response_text = gemini_response.get('response', '')
                if not response_text:
                    response_text = gemini_response.get('raw_response', '')
                if not response_text:
                    response_text = gemini_response.get('llm_response', '')
            else:
                response_text = str(gemini_response)
            
            # Handle markdown-wrapped JSON (```json ... ```)
            if response_text.startswith('```json'):
                # Extract content between ```json and ```
                start_marker = '```json'
                end_marker = '```'
                start_idx = response_text.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = response_text.find(end_marker, start_idx)
                    if end_idx != -1:
                        response_text = response_text[start_idx:end_idx].strip()
                        print("    🔧 Debug: Extracted JSON from markdown wrapper")
            elif response_text.startswith('```'):
                # Handle generic code blocks
                lines = response_text.split('\n')
                if len(lines) > 1:
                    response_text = '\n'.join(lines[1:-1]).strip()
                    print("    🔧 Debug: Extracted content from generic code block")
            
            # Debug: Print first 300 characters of response
            print("    # Debug: Gemini response preview (first 300 chars):")
            print("    " + repr(response_text[:300]))

            # Try multiple approaches to extract JSON from Gemini response
            strategic_scenarios = None
            
            # Method 0: Direct JSON parsing
            if not strategic_scenarios:
                try:
                    strategic_scenarios = json.loads(response_text)
                    if isinstance(strategic_scenarios, list):
                        print("    # Method 0: Direct JSON parsing successful with " + str(len(strategic_scenarios)) + " scenarios")
                    else:
                        strategic_scenarios = None
                except (json.JSONDecodeError, ValueError) as e:
                    print("    # Method 0 failed: " + str(e))
                    strategic_scenarios = None
            
            # Method 0b: Handle "Extra data" error by finding complete JSON boundaries
            if not strategic_scenarios:
                try:
                    # Find the first '[' and the last ']' to get complete JSON array
                    start_idx = response_text.find('[')
                    if start_idx != -1:
                        # Count brackets to find the matching closing bracket
                        bracket_count = 0
                        end_idx = -1
                        for i in range(start_idx, len(response_text)):
                            if response_text[i] == '[':
                                bracket_count += 1
                            elif response_text[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i
                                    break
                        
                        if end_idx != -1:
                            json_str = response_text[start_idx:end_idx+1]
                            strategic_scenarios = json.loads(json_str)
                            if isinstance(strategic_scenarios, list):
                                print("    # Method 0b: Found complete JSON array with " + str(len(strategic_scenarios)) + " scenarios")
                            else:
                                strategic_scenarios = None
                except (json.JSONDecodeError, ValueError) as e:
                    print("    # Method 0b failed: " + str(e))
                    strategic_scenarios = None
            
            # Method 1: Look for JSON array with improved regex
            if not strategic_scenarios:
                try:
                    json_match = re.search(r'\[[\s\S]*?\]', response_text)
                    if json_match:
                        json_str = json_match.group(0)
                        json_str = json_str.strip()
                        strategic_scenarios = json.loads(json_str)
                        if isinstance(strategic_scenarios, list):
                            print("    # Method 1: Found JSON array with " + str(len(strategic_scenarios)) + " scenarios")
                        else:
                            strategic_scenarios = None
                except json.JSONDecodeError as e:
                    print("    # Method 1 failed: " + str(e))
                    strategic_scenarios = None
            
            # Method 2: Fallback - create simple scenario from text analysis
            if not strategic_scenarios:
                print("    # All JSON parsing methods failed, creating fallback scenario")
                strategic_scenarios = [{
                    "scenario_name": "LangGraph Strategic Plan",
                    "target_systems": [1, 2, 3],
                    "attack_types": ["voltage_manipulation", "power_disruption"],
                    "combined_magnitude": 0.7,
                    "stealth_level": 0.6,
                    "strategic_goal": "Coordinated multi-system attack",
                    "coordination": "simultaneous",
                    "impact_factor": 0.8,
                    "success_rate": 0.85
                }]

            print("    # Gemini generated " + str(len(strategic_scenarios)) + " strategic attack scenarios")
            return strategic_scenarios

        except Exception as e:
            print(f"    # Robust parsing failed: {str(e)}")
            # Return a simple fallback scenario
            return [{
                "scenario_name": "Fallback Strategic Plan",
                "target_systems": [1, 2, 3],
                "attack_types": ["coordinated_attack"],
                "combined_magnitude": 0.6,
                "stealth_level": 0.7,
                "strategic_goal": "Fallback coordinated attack",
                "coordination": "sequential",
                "impact_factor": 0.7,
                "success_rate": 0.8
            }]

    def _create_langgraph_fallback_instructions(self) -> Dict:
        """Create fallback instructions when LLM fails"""
        return {
            'strategy': 'coordinated_attack',
            'target_priority': [1, 2, 3],
            'coordination_type': 'simultaneous',
            'stealth_requirement': 0.6,
            'success_criteria': 'maximize_impact'
        }
    
    def _rl_coordination_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Execute RL agent coordination"""
        try:
            print("🤖 LangGraph Node: RL Coordination")
            
            # NEW: Check if attack-specific coordinator is available
            if self.attack_specific_coordinator and ATTACK_DEPLOYMENT_AVAILABLE:
                print("   # Using Attack-Specific Coordination (NEW ARCHITECTURE)")
                
                # Extract analysis data from state
                system_analysis = state.get('system_analysis', {})
                threat_analysis = state.get('threat_analysis', {})
                stride_threats = threat_analysis.get('stride_threats', {})
                mitre_tactics = threat_analysis.get('mitre_tactics', {})
                
                # Use new attack-specific coordination
                rl_results = self._coordinate_attack_specific_agents(
                    system_analysis,
                    stride_threats,
                    mitre_tactics
                )
                
                # Update state with attack-specific results
                state['rl_actions'] = self._ensure_json_serializable(rl_results.get('deployments', []))
                state['execution_results'] = self._ensure_json_serializable(rl_results.get('execution_results', []))
                state['coordination_metrics'] = self._ensure_json_serializable({
                    'success_rate': rl_results.get('success_rate', 0.0),
                    'total_impact': rl_results.get('total_impact', 0.0),
                    'avg_detection_risk': rl_results.get('avg_detection_risk', 0.0),
                    'architecture': 'attack_specific'
                })
                state['current_phase'] = 'execution_monitoring'
                state['debug_info'].append(f"Attack-specific coordination: {rl_results.get('total_attacks', 0)} attacks executed")
                
                return state
            
            # OLD: Fallback to system-specific coordination
            else:
                print("   # Using System-Specific Coordination (OLD ARCHITECTURE)")
                
                llm_instructions = state.get('llm_instructions', {})
                
                # Convert LLM instructions to RL actions
                rl_actions = self._convert_langgraph_instructions_to_rl_actions(llm_instructions)
                
                # Execute RL actions and UPDATE the actions with actual SAC parameters
                execution_results = []
                updated_actions = []
                for action in rl_actions:
                    result = self._execute_langgraph_rl_action(action)
                    execution_results.append(result)
                    
                    # Update action with actual parameters from result (if available)
                    if 'sac_params' in result:
                        action['magnitude'] = result['sac_params'].get('magnitude', action.get('magnitude', 0.5))
                        action['duration'] = result['sac_params'].get('duration', action.get('duration', 60.0))
                        action['stealth_level'] = result['sac_params'].get('stealth', action.get('stealth_level', 0.5))
                    if 'attack_type' in result:
                        action['attack_type'] = result['attack_type']
                    
                    updated_actions.append(action)
                
                # Calculate coordination metrics
                coordination_metrics = self._calculate_langgraph_coordination_metrics(execution_results)
                
                # Update state with UPDATED actions containing actual SAC parameters
                state['rl_actions'] = self._ensure_json_serializable(updated_actions)  # Use updated actions with SAC params
                state['execution_results'] = self._ensure_json_serializable(execution_results)
                state['coordination_metrics'] = self._ensure_json_serializable(coordination_metrics)
                state['current_phase'] = 'execution_monitoring'
                state['debug_info'].append(f"RL coordination completed: {len(updated_actions)} actions executed")
                
                return state
            
        except Exception as e:
            print(f"# RL coordination node failed: {e}")
            state['debug_info'].append(f"RL coordination failed: {e}")
            return state
    
    def _execution_monitoring_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Monitor attack execution"""
        try:
            print("# LangGraph Node: Execution Monitoring")
            
            execution_results = state.get('execution_results', [])
            
            # Unwrap nested result structure from execute_deployment:
            # {'attack_type': ..., 'result': {'success': ..., 'impact': ..., 'detection_risk': ...}}
            def _inner(r):
                return r.get('result', r) if isinstance(r, dict) else r
            
            # Calculate stealth metrics
            stealth_metrics = {
                'detection_risk': float(np.mean([_inner(r).get('detection_risk', 0.5) for r in execution_results])) if execution_results else 0.5,
                'stealth_score': float(np.mean([_inner(r).get('stealth_factor', 0.5) for r in execution_results])) if execution_results else 0.5,
                'anomaly_score': float(np.mean([_inner(r).get('anomaly_score', 0.0) for r in execution_results])) if execution_results else 0.0
            }
            
            # Calculate success metrics
            success_metrics = {
                'success_rate': float(np.mean([1.0 if _inner(r).get('success', False) else 0.0 for r in execution_results])) if execution_results else 0.0,
                'total_impact': float(sum([_inner(r).get('impact', 0.0) for r in execution_results])),
                'coordination_effectiveness': float(state.get('coordination_metrics', {}).get('effectiveness', 0.0))
            }
            
            # Update state
            state['stealth_metrics'] = self._ensure_json_serializable(stealth_metrics)
            state['success_metrics'] = self._ensure_json_serializable(success_metrics)
            state['current_phase'] = 'feedback_analysis'
            state['debug_info'].append(f"Execution monitoring completed: {success_metrics['success_rate']:.2%} success")
            
            return state
            
        except Exception as e:
            print(f"# Execution monitoring node failed: {e}")
            state['debug_info'].append(f"Execution monitoring failed: {e}")
            return state
    
    def _feedback_analysis_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Analyze RL feedback"""
        try:
            print("# LangGraph Node: Feedback Analysis")
            
            execution_results = state.get('execution_results', [])
            success_metrics = state.get('success_metrics', {})
            stealth_metrics = state.get('stealth_metrics', {})
            
            # Create RL feedback
            rl_feedback = {
                'performance_score': success_metrics.get('success_rate', 0.0),
                'stealth_effectiveness': stealth_metrics.get('stealth_score', 0.0),
                'detection_risk': stealth_metrics.get('detection_risk', 0.5),
                'coordination_quality': state.get('coordination_metrics', {}).get('effectiveness', 0.0),
                'recommendations': self._generate_feedback_recommendations(execution_results)
            }
            
            # Update state with serializable data
            state['rl_feedback'] = self._ensure_json_serializable(rl_feedback)
            state['current_phase'] = 'llm_adaptation'
            state['debug_info'].append(f"Feedback analysis completed: {rl_feedback['performance_score']:.2%} performance")
            
            return state
            
        except Exception as e:
            print(f"# Feedback analysis node failed: {e}")
            state['debug_info'].append(f"Feedback analysis failed: {e}")
            return state
    
    def _llm_adaptation_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: LLM strategy adaptation.

        Guard: if the SAC policy is frozen (inference phase) the execution
        results are identical every iteration, so calling the LLM for
        'adaptation' produces only wasted API calls and log spam.  We skip
        the LLM call when *both* conditions hold:
          1. success_rate is still 0 AND detection_risk is still ≥ 0.99
             (i.e. the environment response has not changed at all)
          2. this adaptation node has already been visited at least once in
             this episode (tracked via state['_adaptation_call_count']).
        In that case we reuse the previous adaptation result and move on.
        """
        try:
            print("# LangGraph Node: LLM Adaptation")

            rl_feedback    = state.get('rl_feedback', {})
            success_rate   = rl_feedback.get('performance_score', 0.0)
            detection_risk = rl_feedback.get('detection_risk', 0.0)

            # Increment per-episode call counter stored in state
            call_count = state.get('_adaptation_call_count', 0) + 1
            state['_adaptation_call_count'] = call_count

            # ── No-improvement guard ──────────────────────────────────────
            # If the frozen SAC policy produced the exact same bad outcome
            # as the previous iteration there is no point asking the LLM
            # again — its advice cannot be acted upon until the SAC weights
            # are updated.  Skip to workflow_completion immediately.
            if call_count > 1 and success_rate == 0.0 and detection_risk >= 0.99:
                print("    ⚡ Skipping LLM adaptation call: frozen SAC policy, "
                      "no improvement since last iteration.")
                state['current_phase'] = 'workflow_completion'
                state['debug_info'].append(
                    "LLM adaptation skipped (frozen policy, no-improve guard)")
                return state

            # Create adaptation prompt
            adaptation_prompt = self._create_langgraph_adaptation_prompt(state, rl_feedback)

            adaptation_input = {
                'adaptation_prompt': adaptation_prompt,
                'rl_feedback': rl_feedback,
                'current_phase': state.get('current_phase', 'adaptation')
            }

            # Get LLM adaptation response
            adaptation_response = self.llm_analyzer.analyze_threats(adaptation_input)

            if isinstance(adaptation_response, dict) and 'llm_response' in adaptation_response:
                adaptation_text = adaptation_response['llm_response']
            else:
                adaptation_text = adaptation_response

            if not isinstance(adaptation_text, str):
                adaptation_text = str(adaptation_text)

            adaptation_results = self._parse_langgraph_adaptation_response(adaptation_text)

            state['adaptation_results'] = self._ensure_json_serializable(adaptation_results)
            state['current_phase'] = 'workflow_completion'
            state['debug_info'].append(
                f"LLM adaptation completed: {adaptation_results.get('strategy', 'unknown')}")

            return state

        except Exception as e:
            print(f"# LLM adaptation node failed: {e}")
            state['debug_info'].append(f"LLM adaptation failed: {e}")
            return state
    
    def _workflow_completion_node(self, state: EnhancedAttackState) -> EnhancedAttackState:
        """LangGraph node: Complete workflow and prepare results"""
        try:
            print("# LangGraph Node: Workflow Completion")
            
            # Compile final results
            final_results = {
                'workflow_type': 'enhanced_langgraph',
                'success_metrics': state.get('success_metrics', {}),
                'stealth_metrics': state.get('stealth_metrics', {}),
                'coordination_metrics': state.get('coordination_metrics', {}),
                'rl_results': {
                    'actions': state.get('rl_actions', []),
                    'results': state.get('execution_results', []),
                    'feedback': state.get('rl_feedback', {})
                },
                'llm_analysis': {
                    'system_analysis': state.get('enhanced_system_analysis', {}),
                    'threat_analysis': state.get('enhanced_threat_analysis', {}),
                    'instructions': state.get('llm_instructions', {}),
                    'adaptation': state.get('adaptation_results', {})
                },
                'debug_info': state.get('debug_info', []),
                'performance_history': state.get('performance_history', [])
            }
            
            # Update state with serializable data
            state['final_results'] = self._ensure_json_serializable(final_results)
            state['workflow_completed'] = True
            state['debug_info'].append("Workflow completed successfully")
            
            return state
            
        except Exception as e:
            print(f"# Workflow completion node failed: {e}")
            state['debug_info'].append(f"Workflow completion failed: {e}")
            return state
    
    # ==================== LANGGRAPH HELPER METHODS ====================
    
    def _gather_system_analysis_data(self) -> Dict:
        """Gather comprehensive system analysis data"""
        return self._perform_comprehensive_system_analysis().__dict__
    
    def _create_langgraph_llm_prompt(self, state: EnhancedAttackState) -> str:
        """Create LLM prompt for LangGraph workflow"""
        system_analysis = state.get('enhanced_system_analysis', {})
        threat_analysis = state.get('enhanced_threat_analysis', {})
        
        prompt = f"""This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. 
        ENHANCED ATTACK COORDINATION REQUEST
        
        System Analysis:
        - Components: {len(system_analysis)}
        - Vulnerabilities: {threat_analysis.get('vulnerability_count', 0)}
        - Risk Level: {threat_analysis.get('risk_assessment', 'unknown')}
        
        STRIDE Threats: {len(state.get('enhanced_stride_threats', {}))}
        MITRE Tactics: {len(state.get('enhanced_mitre_tactics', {}))}
        
        Please provide strategic attack coordination instructions including:
        1. Attack strategy (coordinated_attack, sequential_attack, or targeted_attack)
        2. Target priority (list of system IDs)
        3. Coordination type (simultaneous, sequential, or adaptive)
        4. Stealth requirements (0.0-1.0)
        5. Success criteria
        """
        
        return prompt

    def _format_stride_threats(self, stride_data: Any) -> str:
        """Format STRIDE threats for inclusion in LLM prompts."""
        try:
            if not stride_data:
                return "(none)"
            lines = []
            # If standard structure with 'threats' list
            if isinstance(stride_data, dict) and isinstance(stride_data.get('threats'), list):
                for t in stride_data['threats'][:10]:
                    if isinstance(t, dict):
                        cat = t.get('category', 'unknown')
                        comp = t.get('component', 'unknown')
                        sev = t.get('severity', 0.5)
                        lines.append(f"- {cat} on {comp} (sev={sev})")
                    else:
                        lines.append(f"- {str(t)}")
            else:
                # Fallback: summarize keys
                if isinstance(stride_data, dict):
                    for k, v in list(stride_data.items())[:10]:
                        count = len(v) if isinstance(v, (list, tuple)) else 1
                        lines.append(f"- {k}: {count} findings")
                elif isinstance(stride_data, (list, tuple)):
                    lines.append(f"- {len(stride_data)} findings")
                else:
                    lines.append(str(stride_data))
            return "\n".join(lines)
        except Exception:
            return "(unavailable)"

    def _format_mitre_tactics(self, mitre_data: Any) -> str:
        """Format MITRE tactics/techniques for inclusion in LLM prompts."""
        try:
            if not mitre_data:
                return "(none)"
            lines = []
            if isinstance(mitre_data, dict):
                # Prefer 'tactics' list
                if isinstance(mitre_data.get('tactics'), list):
                    for t in mitre_data['tactics'][:10]:
                        if isinstance(t, dict):
                            name = t.get('name', 'tactic')
                            phase = t.get('phase', 'phase')
                            lines.append(f"- {name} (phase={phase})")
                        else:
                            lines.append(f"- {str(t)}")
                elif isinstance(mitre_data.get('techniques'), list):
                    for tech in mitre_data['techniques'][:10]:
                        if isinstance(tech, dict):
                            tid = tech.get('id', 'TXXXX')
                            name = tech.get('name', 'technique')
                            lines.append(f"- {tid} {name}")
                        else:
                            lines.append(f"- {str(tech)}")
                else:
                    # Fallback: summarize keys
                    for k, v in list(mitre_data.items())[:10]:
                        count = len(v) if isinstance(v, (list, tuple)) else 1
                        lines.append(f"- {k}: {count}")
            elif isinstance(mitre_data, (list, tuple)):
                lines.append(f"- {len(mitre_data)} entries")
            else:
                lines.append(str(mitre_data))
            return "\n".join(lines)
        except Exception:
            return "(unavailable)"
    
    def _parse_langgraph_llm_instructions(self, llm_response) -> Dict:
        """Parse LLM response into structured instructions"""
        try:
            # Handle both string and dict inputs
            if isinstance(llm_response, dict):
                # If it's already a dict, extract relevant fields
                instructions = {
                    'strategy': llm_response.get('strategy', 'coordinated_attack'),
                    'target_priority': llm_response.get('target_priority', [1, 2, 3]),
                    'coordination_type': llm_response.get('coordination_type', 'simultaneous'),
                    'stealth_requirement': llm_response.get('stealth_requirement', 0.7),
                    'success_criteria': llm_response.get('success_criteria', 'maximize_impact')
                }
                return instructions
            
            # Handle string response
            instructions = {
                'strategy': 'coordinated_attack',
                'target_priority': [1, 2, 3],
                'coordination_type': 'simultaneous',
                'stealth_requirement': 0.7,
                'success_criteria': 'maximize_impact'
            }
            
            # Extract information from response text
            if isinstance(llm_response, str):
                response_lower = llm_response.lower()
                
                # Extract strategy
                if 'sequential' in response_lower:
                    instructions['coordination_type'] = 'sequential'
                elif 'adaptive' in response_lower:
                    instructions['coordination_type'] = 'adaptive'
                
                # Extract stealth requirement
                if 'high stealth' in response_lower:
                    instructions['stealth_requirement'] = 0.9
                elif 'low stealth' in response_lower:
                    instructions['stealth_requirement'] = 0.3
            
            return instructions
            
        except Exception as e:
            print(f"# Failed to parse LLM instructions: {e}")
            return self._create_langgraph_fallback_instructions()
    
    def _create_langgraph_fallback_instructions(self) -> Dict:
        """Create fallback instructions when LLM parsing fails"""
        return {
            'strategy': 'fallback_coordinated',
            'target_priority': [1, 2],
            'coordination_type': 'sequential',
            'stealth_requirement': 0.5,
            'success_criteria': 'basic_impact'
        }
    
    def _convert_langgraph_instructions_to_rl_actions(self, instructions: Dict) -> List[Dict]:
        """Convert LLM instructions to RL actions - MUST include all parameters for extraction"""
        actions = []
        
        # Use all 6 systems if target_priority is not specified or too small
        target_priority = instructions.get('target_priority', [1, 2, 3, 4, 5, 6])
        if len(target_priority) < 3:
            target_priority = [1, 2, 3, 4, 5, 6]  # Ensure we attack all systems
        
        coordination_type = instructions.get('coordination_type', 'sequential')
        stealth_req = instructions.get('stealth_requirement', 0.5)
        
        for system_id in target_priority:
            action = {
                'system_id': system_id,
                'target_system': system_id,  # Add for compatibility
                'attack_type': AttackType.VOLTAGE_MANIPULATION.value,
                'coordination_type': coordination_type,
                'stealth_requirement': stealth_req,
                'stealth_level': stealth_req,  # Add for extraction
                'priority': target_priority.index(system_id) + 1,
                # Add placeholder values that will be overridden by SAC agent
                'magnitude': 0.5,
                'duration': 60.0,
                'target_component': 'evcs_cms_link'
            }
            actions.append(action)
        
        return actions
    
    def _execute_langgraph_rl_action(self, action: Dict) -> Dict:
        """Execute RL action using TRAINED DQN/SAC agents via their TRAINING environments.
        
        Uses env.reset() → agent.predict(obs) → env.step(action) to ensure
        observation/action consistency between training and testing.
        """
        try:
            system_id = action.get('system_id', 1)
            attack_type = action.get('attack_type', AttackType.VOLTAGE_MANIPULATION)
            stealth_req = action.get('stealth_requirement', 0.5)

            # Try to use ACTUAL trained DQN/SAC agents with their TRAINING environments
            if self.rl_coordinator and hasattr(self.rl_coordinator, 'sac_agents') and hasattr(self.rl_coordinator, 'sac_envs'):
                has_sac = system_id in self.rl_coordinator.sac_agents and system_id in self.rl_coordinator.sac_envs
                has_dqn = (hasattr(self.rl_coordinator, 'dqn_agents') and hasattr(self.rl_coordinator, 'dqn_envs') and
                           system_id in self.rl_coordinator.dqn_agents and system_id in self.rl_coordinator.dqn_envs)
                
                if has_sac or has_dqn:
                    dqn_action = {}
                    sac_params = {}
                    sac_reward = 0.0
                    dqn_reward = 0.0
                    sac_info = {}
                    dqn_info = {}
                    
                    # --- DQN: use TRAINING environment ---
                    if has_dqn:
                        dqn_env = self.rl_coordinator.dqn_envs[system_id]
                        dqn_obs, _ = dqn_env.reset()
                        dqn_action_idx, _ = self.rl_coordinator.dqn_agents[system_id].predict(dqn_obs, deterministic=True)
                        _, dqn_reward, _, _, dqn_info = dqn_env.step(dqn_action_idx)
                        dqn_action = self._convert_dqn_action_to_attack(dqn_action_idx)
                    
                    # --- SAC: use TRAINING environment ---
                    if has_sac:
                        sac_env = self.rl_coordinator.sac_envs[system_id]
                        sac_obs, _ = sac_env.reset()
                        sac_raw_action, _ = self.rl_coordinator.sac_agents[system_id].predict(sac_obs, deterministic=True)
                        _, sac_reward, _, _, sac_info = sac_env.step(sac_raw_action)
                        
                        # Extract SAC params from the raw action for logging
                        sac_params = {
                            'magnitude': float(np.clip(sac_raw_action[1] if len(sac_raw_action) > 1 else 0.5, 0.0, 2.0)),
                            'stealth': float(np.clip(sac_raw_action[4] if len(sac_raw_action) > 4 else 0.5, 0.0, 1.0)),
                            'duration': float(np.clip(sac_raw_action[2] if len(sac_raw_action) > 2 else 30.0, 5.0, 180.0))
                        }
                    
                    # Combine results — prefer SAC (continuous, richer)
                    best_info = sac_info if has_sac else dqn_info
                    attack_detected = best_info.get('attack_detected', True)
                    
                    result = {
                        'system_id': int(system_id),
                        'attack_type': str(dqn_action.get('attack_type', str(attack_type))),
                        'success': bool(not attack_detected),
                        'impact': float(best_info.get('evcs_impact', 0.0)),
                        'detection_risk': float(best_info.get('security_result', {}).get('anomaly_score', 0.5)),
                        'stealth_factor': float(sac_params.get('stealth', stealth_req)),
                        'anomaly_score': float(best_info.get('security_result', {}).get('anomaly_score', 0.0)),
                        'timestamp': float(time.time()),
                        'agent_used': 'DQN+SAC',
                        'env_consistent': True,
                        'dqn_action': {'attack_type': str(dqn_action.get('attack_type', 'unknown')), 'action_idx': int(dqn_action.get('action_idx', 0))},
                        'sac_params': sac_params
                    }

                    print(f"    🤖 Used trained agents (via training envs): DQN='{dqn_action.get('attack_type')}', "
                          f"SAC reward={sac_reward:.2f}, detected={attack_detected}")
                    return result

            # Fallback to simulation if agents not available
            print(f"    # Trained agents not available for system {system_id}, using fallback simulation")
            return self._simulate_attack_fallback(action, stealth_req)

        except Exception as e:
            print(f"# RL action execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'system_id': int(action.get('system_id', 1)),
                'success': False,
                'impact': 0.0,
                'detection_risk': 0.5,
                'stealth_factor': 0.5,
                'anomaly_score': 0.0,
                'timestamp': float(time.time()),
                'error': str(e)
            }
    
    def _calculate_langgraph_coordination_metrics(self, execution_results: List[Dict]) -> Dict:
        """Calculate coordination effectiveness metrics"""
        if not execution_results:
            return {'effectiveness': 0.0, 'synchronization': 0.0}
        
        # Calculate coordination effectiveness - ensure all values are Python floats, not numpy
        success_rate = float(np.mean([1.0 if r.get('success', False) else 0.0 for r in execution_results]))
        avg_impact = float(np.mean([r.get('impact', 0.0) for r in execution_results]))
        avg_stealth = float(np.mean([r.get('stealth_factor', 0.5) for r in execution_results]))
        
        # Calculate synchronization (based on timestamp spread)
        timestamps = [r.get('timestamp', 0) for r in execution_results if r.get('timestamp')]
        synchronization = 1.0 - (max(timestamps) - min(timestamps)) / 10.0 if len(timestamps) > 1 else 1.0
        synchronization = float(max(0.0, min(1.0, synchronization)))
        
        effectiveness = float(success_rate * 0.4 + avg_impact * 0.3 + avg_stealth * 0.2 + synchronization * 0.1)
        
        return {
            'effectiveness': effectiveness,
            'synchronization': synchronization,
            'success_rate': success_rate,
            'avg_impact': avg_impact,
            'avg_stealth': avg_stealth
        }
    
    def _create_langgraph_adaptation_prompt(self, state: EnhancedAttackState, rl_feedback: Dict) -> str:
        """Create adaptation prompt for LLM"""
        performance = rl_feedback.get('performance_score', 0.0)
        stealth = rl_feedback.get('stealth_effectiveness', 0.0)
        detection_risk = rl_feedback.get('detection_risk', 0.5)
        
        recent_actions = rl_feedback.get('recent_actions', '')
        if isinstance(recent_actions, list):
            recent_actions = "\n".join(recent_actions)
        detection_summary = rl_feedback.get('detection_summary', '')
        system_context = rl_feedback.get('system_context', {})
        
        context_lines = []
        if recent_actions:
            context_lines.append("\nRecent Actions:\n" + str(recent_actions))
        if detection_summary:
            context_lines.append("\nDetection Events:\n" + str(detection_summary))
        if system_context:
            context_lines.append("\nSystem Context:")
            for key, value in system_context.items():
                context_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        extra_context = "\n".join(context_lines) if context_lines else ""
        
        prompt = f"""This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. 
        ATTACK STRATEGY ADAPTATION REQUEST
        
        Current Performance:
        - Success Rate: {performance:.2%}
        - Stealth Effectiveness: {stealth:.2%}
        - Detection Risk: {detection_risk:.2%}
        {extra_context}
        
        Please provide strategy adaptation recommendations:
        1. Should we continue current strategy or adapt?
        2. What changes would improve performance?
        3. How can we reduce detection risk?
        4. Recommended next actions?
        """
        
        return prompt
    
    def _parse_langgraph_adaptation_response(self, response) -> Dict:
        """Parse LLM adaptation response"""
        try:
            # Handle both string and dict inputs
            if isinstance(response, dict):
                # If it's already a dict, extract relevant fields
                adaptation = {
                    'continue_strategy': response.get('continue_strategy', True),
                    'recommended_changes': response.get('recommended_changes', []),
                    'risk_mitigation': response.get('risk_mitigation', []),
                    'next_actions': response.get('next_actions', [])
                }
                return adaptation
            
            # Handle string response
            adaptation = {
                'continue_strategy': True,
                'recommended_changes': [],
                'risk_mitigation': [],
                'next_actions': []
            }
            
            # Extract information from response text
            if isinstance(response, str):
                response_lower = response.lower()
                
                # Check if strategy should continue
                adaptation['continue_strategy'] = 'continue' in response_lower
                
                # Simple keyword extraction
                if 'increase stealth' in response_lower:
                    adaptation['recommended_changes'].append('increase_stealth')
                if 'change targets' in response_lower:
                    adaptation['recommended_changes'].append('change_targets')
                if 'reduce frequency' in response_lower:
                    adaptation['risk_mitigation'].append('reduce_frequency')
            
            return adaptation
            
        except Exception as e:
            print(f"# Failed to parse adaptation response: {e}")
            return {'continue_strategy': True, 'recommended_changes': []}
    
    def _generate_feedback_recommendations(self, execution_results: List[Dict]) -> List[str]:
        """Generate feedback recommendations based on execution results"""
        recommendations = []
        
        if not execution_results:
            return ['No execution results available']
        
        # Unwrap nested result structure from execute_deployment
        def _inner(r):
            return r.get('result', r) if isinstance(r, dict) else r
        success_rate = np.mean([1.0 if _inner(r).get('success', False) else 0.0 for r in execution_results])
        avg_detection = np.mean([_inner(r).get('detection_risk', 0.5) for r in execution_results])
        
        if success_rate < 0.5:
            recommendations.append('Consider adjusting attack parameters for higher success rate')
        if avg_detection > 0.7:
            recommendations.append('Increase stealth measures to reduce detection risk')
        if len(execution_results) < 2:
            recommendations.append('Consider coordinating more systems for better impact')
        
        return recommendations if recommendations else ['Performance within acceptable parameters']
    
    def _combine_threat_analyses(self, stride_analysis: Dict, mitre_analysis: Dict) -> Dict:
        """Combine STRIDE and MITRE analyses"""
        # Helper to safely convert severity/applicability to float
        def safe_float(val, default=0.0):
            try:
                if isinstance(val, str):
                    # Extract numeric value from strings like 'high', 'medium', 'low'
                    severity_map = {'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.3}
                    return severity_map.get(val.lower(), default)
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Safe length calculation for threat analysis
        def safe_len(obj, default=0):
            try:
                if isinstance(obj, (dict, list, tuple)):
                    return len(obj)
                elif isinstance(obj, (int, float)):
                    return 1 if obj else 0
                else:
                    return default
            except (TypeError, AttributeError):
                return default
        
        return {
            'stride_threats': stride_analysis,
            'mitre_tactics': mitre_analysis,
            'combined_risk_score': safe_len(stride_analysis) + safe_len(mitre_analysis),
            'vulnerability_count': sum(safe_len(threats) for threats in stride_analysis.values() if isinstance(threats, (dict, list, tuple))),
            'tactic_count': sum(safe_len(tactics) for tactics in mitre_analysis.values() if isinstance(tactics, (dict, list, tuple))),
            'priority_threats': [
                threat for threats in stride_analysis.values() for threat in (threats if isinstance(threats, (list, tuple)) else [])
                if isinstance(threat, dict) and safe_float(threat.get('severity', 0)) > 0.7
            ] + [
                tactic for tactics in mitre_analysis.values() for tactic in (tactics if isinstance(tactics, (list, tuple)) else [])
                if isinstance(tactic, dict) and safe_float(tactic.get('applicability', 0)) > 0.7
            ],
            'attack_vectors': list(set(
                [threat.get('vector', 'unknown') for threats in stride_analysis.values() for threat in (threats if isinstance(threats, (list, tuple)) else [])
                 if isinstance(threat, dict)] +
                [tactic.get('technique', 'unknown') for tactics in mitre_analysis.values() for tactic in (tactics if isinstance(tactics, (list, tuple)) else [])
                 if isinstance(tactic, dict)]
            )),
            'risk_assessment': 'high' if (len(stride_analysis) + len(mitre_analysis)) > 10 else 'medium'
        }
    
    # ==================== RL AGENT INTEGRATION HELPER METHODS ====================
    
    def _get_system_observation_for_agents(self, system_id: int) -> np.ndarray:
        """Extract 25-feature observation from hierarchical simulation for DQN/SAC agents"""
        try:
            import numpy as np

            # Initialize observation array (25 features matching agent training format)
            obs = np.zeros(25, dtype=np.float32)

            # Try to get system state from hierarchical simulation
            if self.hierarchical_sim and hasattr(self.hierarchical_sim, 'distribution_systems'):
                dist_systems = self.hierarchical_sim.distribution_systems

                if system_id in dist_systems:
                    dist_info = dist_systems[system_id]
                    dist_sys = dist_info['system']  # FIXED: Access actual system object

                    # Extract EVCS station data
                    if hasattr(dist_sys, 'evcs_stations') and dist_sys.evcs_stations:
                        station = dist_sys.evcs_stations[0]  # Use first station

                        # Features 0-4: Voltage metrics
                        obs[0] = getattr(station, 'voltage', 1.0)
                        obs[1] = getattr(station, 'voltage_pu', 1.0)
                        obs[2] = getattr(station, 'grid_voltage', 1.0)
                        obs[3] = getattr(station, 'voltage_reference', 1.0)
                        obs[4] = abs(obs[0] - obs[3])  # Voltage error

                        # Features 5-9: Current metrics
                        obs[5] = getattr(station, 'current', 0.0)
                        obs[6] = getattr(station, 'current_reference', 0.0)
                        obs[7] = abs(obs[5] - obs[6])  # Current error
                        obs[8] = getattr(station, 'charging_current', 0.0)
                        obs[9] = getattr(station, 'grid_current', 0.0)

                        # Features 10-14: Power metrics
                        obs[10] = getattr(station, 'power', 0.0)
                        obs[11] = getattr(station, 'power_reference', 0.0)
                        obs[12] = getattr(station, 'active_power', 0.0)
                        obs[13] = getattr(station, 'reactive_power', 0.0)
                        obs[14] = getattr(station, 'total_load', 0.0)

                        # Features 15-17: Frequency metrics
                        obs[15] = getattr(station, 'frequency', 60.0)
                        obs[16] = getattr(station, 'grid_frequency', 60.0)
                        obs[17] = abs(obs[15] - 60.0)  # Frequency deviation

                        # Features 18-20: SOC metrics
                        obs[18] = getattr(station, 'soc', 0.5)
                        obs[19] = getattr(station, 'soc_target', 0.8)
                        obs[20] = obs[19] - obs[18]  # SOC error

                        # Features 21-24: System state
                        obs[21] = float(getattr(station, 'charging_active', True))
                        obs[22] = getattr(station, 'demand_factor', 1.0)
                        obs[23] = getattr(station, 'urgency_factor', 1.0)
                        obs[24] = float(system_id) / 6.0  # Normalized system ID

            # Clip to reasonable ranges
            obs = np.clip(obs, -10.0, 10.0)

            return obs

        except Exception as e:
            print(f"    # Failed to extract observation for system {system_id}: {e}")
            # Return zero observation as fallback
            return np.zeros(25, dtype=np.float32)

    def _convert_dqn_action_to_attack(self, action_idx: int) -> Dict:
        """Map DQN action index (0-7) to attack types"""
        attack_types_map = {
            0: 'voltage_manipulation',
            1: 'current_injection',
            2: 'power_disruption',
            3: 'frequency_attack',
            4: 'model_poisoning',
            5: 'soc_spoofing',
            6: 'charging_hijacking',
            7: 'thermal_attack'
        }

        # Wrap around if action_idx > 7
        attack_type = attack_types_map.get(action_idx % 8, 'voltage_manipulation')

        return {
            'attack_type': attack_type,
            'action_idx': action_idx
        }

    def _execute_pinn_attack(self, pinn_model, attack_params: Dict) -> Dict:
        """Execute attack on federated PINN model"""
        try:
            # Try to use existing PINN attack simulation if available
            if hasattr(self.rl_coordinator, 'marl_env') and hasattr(self.rl_coordinator.marl_env, '_simulate_pinn_attack'):
                return self.rl_coordinator.marl_env._simulate_pinn_attack(pinn_model, attack_params)

            # Fallback: probability-based simulation
            magnitude = attack_params.get('magnitude', 0.5)
            stealth_factor = attack_params.get('stealth_factor', 0.5)

            # Higher magnitude and lower stealth = higher success probability
            success_prob = magnitude * (1.0 - stealth_factor * 0.5)
            success = np.random.random() < success_prob

            # Calculate impact
            impact = magnitude * 0.8 if success else magnitude * 0.2

            # Calculate detection risk
            detection_risk = max(0.0, 1.0 - stealth_factor + np.random.uniform(-0.1, 0.1))

            return {
                'success': success,
                'impact': impact,
                'detection_risk': detection_risk,
                'attack_type': attack_params.get('type', 'unknown'),
                'target': 'pinn_model'
            }

        except Exception as e:
            print(f"    # PINN attack execution failed: {e}")
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': str(e)
            }

    def _simulate_attack_fallback(self, action: Dict, stealth_req: float) -> Dict:
        """Fallback simulation when agents not available"""
        magnitude = action.get('magnitude', 0.5)

        # Probability-based simulation
        success_prob = magnitude * (1.0 - stealth_req * 0.4)
        success = np.random.random() < success_prob

        impact = magnitude * 0.7 if success else magnitude * 0.25
        detection_risk = max(0.0, 1.0 - stealth_req + np.random.uniform(-0.2, 0.2))

        return {
            'success': success,
            'impact': impact,
            'detection_risk': detection_risk,
            'attack_type': action.get('attack_type', 'unknown'),
            'execution_mode': 'fallback_simulation'
        }

    def _get_rl_agent_actions(self) -> List[Dict]:
        """Get actions directly from RL agents instead of loading from files"""
        try:
            # Check if we have access to the enhanced system's RL coordinator
            if hasattr(self, 'enhanced_system') and self.enhanced_system:
                if hasattr(self.enhanced_system, 'dqn_sac_coordinator') and self.enhanced_system.dqn_sac_coordinator:
                    print("    🤖 Getting actions from trained RL agents...")
                    
                    # Get system states
                    system_states = {}
                    if hasattr(self.enhanced_system, 'federated_manager') and self.enhanced_system.federated_manager:
                        for sys_id in range(1, 7):  # Systems 1-6
                            if sys_id in self.enhanced_system.federated_manager.local_models:
                                # Create mock system state for RL agents
                                system_states[sys_id] = {
                                    'soc': 0.6,
                                    'voltage': 400.0,
                                    'current': 100.0,
                                    'power': 40.0,
                                    'temperature': 25.0,
                                    'load_factor': 0.8,
                                    'urgency_factor': 0.5,
                                    'stability_score': 0.9
                                }
                    
                    if system_states:
                        # Get coordinated actions from RL agents
                        coordinated_actions = self.enhanced_system.dqn_sac_coordinator.get_coordinated_attack_actions(system_states)
                        
                        # Convert to format expected by Gemini
                        rl_attacks = []
                        for sys_id, actions in coordinated_actions.items():
                            attack_data = {
                                'system_id': sys_id,
                                'dqn_action': actions.get('dqn_action', {}),
                                'sac_action': actions.get('sac_action'),
                                'attack_type': actions.get('dqn_action', {}).get('attack_type', 'voltage_manipulation'),
                                'magnitude': actions.get('dqn_action', {}).get('magnitude', 0.5),
                                'stealth': actions.get('dqn_action', {}).get('stealth_level', 0.7),
                                'success': True,
                                'impact': 0.6
                            }
                            rl_attacks.append(attack_data)
                        
                        print(f"    # Generated {len(rl_attacks)} RL agent actions")
                        return rl_attacks
                    else:
                        print("    # No system states available for RL agents")
                        return []
                else:
                    print("    # No DQN/SAC coordinator available")
                    return []
            else:
                print("    # No enhanced system reference available")
                return []
                
        except Exception as e:
            print(f"    # Error getting RL agent actions: {e}")
            return []

    def _load_latest_rl_feedback_data(self) -> List[Dict]:
        """Load the latest RL feedback data from the attack_scenarios_logs directory"""
        import os
        import glob
        import re
        
        try:
            # Find the latest RL feedback file
            log_dir = "attack_scenarios_logs"
            if not os.path.exists(log_dir):
                print(f"    # RL feedback log directory not found: {log_dir}")
                return []
            
            # Find all RL feedback files
            feedback_files = glob.glob(os.path.join(log_dir, "rl_feedback_to_gemini_*.txt"))
            if not feedback_files:
                print(f"    # No RL feedback files found in {log_dir}")
                return []
            
            # Get the latest file by timestamp
            latest_file = max(feedback_files, key=os.path.getctime)
            print(f"    📖 Loading RL feedback from: {latest_file}")
            
            # Parse the RL feedback file
            rl_attacks = []
            with open(latest_file, 'r') as f:
                content = f.read()
                
                # Extract attack data using regex patterns
                attack_blocks = re.split(r'ATTACK #\d+\n-+', content)
                
                for block in attack_blocks[1:]:  # Skip header
                    try:
                        attack_data = {}
                        
                        # Extract key fields
                        type_match = re.search(r'TYPE: (.+)', block)
                        target_match = re.search(r'TARGET_SYSTEM: (\d+)', block)
                        magnitude_match = re.search(r'MAGNITUDE: ([\d.]+)', block)
                        stealth_match = re.search(r'STEALTH_LEVEL: ([\d.]+)', block)
                        success_match = re.search(r'SUCCESS_RATE: ([\d.]+)', block)
                        impact_match = re.search(r'IMPACT_FACTOR: ([\d.]+)', block)
                        
                        if type_match and target_match:
                            attack_data = {
                                'attack_type': type_match.group(1).strip(),
                                'target_system': int(target_match.group(1)),
                                'magnitude': float(magnitude_match.group(1)) if magnitude_match else 1.0,
                                'stealth': float(stealth_match.group(1)) if stealth_match else 0.5,
                                'success': float(success_match.group(1)) if success_match else 0.5,
                                'impact': float(impact_match.group(1)) if impact_match else 0.5
                            }
                            rl_attacks.append(attack_data)
                            
                    except Exception as e:
                        print(f"    # Failed to parse attack block: {e}")
                        continue
            
            print(f"    # Loaded {len(rl_attacks)} RL attack suggestions from feedback file")
            return rl_attacks[:20]  # Limit to first 20 attacks to avoid overwhelming Gemini
            
        except Exception as e:
            print(f"    # Failed to load RL feedback data: {e}")
            return []

class STRIDEThreatAnalyzer:
    """STRIDE threat analysis for power systems"""
    
    def analyze_threats(self, system_data: Dict) -> Dict:
        """Analyze threats using STRIDE methodology"""
        try:
            stride_analysis = {
                'spoofing': {
                    'threats': ['Identity spoofing in EVCS communication', 'False sensor readings'],
                    'severity': 'high',
                    'likelihood': 0.7
                },
                'tampering': {
                    'threats': ['PINN model parameter manipulation', 'Charging schedule tampering'],
                    'severity': 'high',
                    'likelihood': 0.8
                },
                'repudiation': {
                    'threats': ['Denial of attack actions', 'Log manipulation'],
                    'severity': 'medium',
                    'likelihood': 0.5
                },
                'information_disclosure': {
                    'threats': ['Customer data exposure', 'Grid topology disclosure'],
                    'severity': 'high',
                    'likelihood': 0.6
                },
                'denial_of_service': {
                    'threats': ['EVCS service disruption', 'Communication jamming'],
                    'severity': 'critical',
                    'likelihood': 0.9
                },
                'elevation_of_privilege': {
                    'threats': ['Unauthorized system access', 'Admin privilege escalation'],
                    'severity': 'critical',
                    'likelihood': 0.4
                }
            }
            
            return {
                'analysis_type': 'STRIDE',
                'threats': stride_analysis,
                'overall_risk': 'high',
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'analysis_type': 'STRIDE',
                'error': str(e),
                'threats': {},
                'overall_risk': 'unknown'
            }
    
    def analyze_system(self, system_analysis: SystemAnalysisData) -> Dict:
        """Perform STRIDE analysis on the system"""
        threats = {}
        
        # Spoofing threats
        threats[STRIDECategory.SPOOFING.value] = [
            {'component': 'EVCS_communication', 'severity': 'high', 'likelihood': 0.7},
            {'component': 'CMS_authentication', 'severity': 'medium', 'likelihood': 0.5},
            {'component': 'PINN_model_identity', 'severity': 'high', 'likelihood': 0.6}
        ]
        
        # Tampering threats
        threats[STRIDECategory.TAMPERING.value] = [
            {'component': 'charging_parameters', 'severity': 'high', 'likelihood': 0.8},
            {'component': 'power_measurements', 'severity': 'medium', 'likelihood': 0.6},
            {'component': 'federated_gradients', 'severity': 'high', 'likelihood': 0.7},
            {'component': 'thermal_protection_thresholds', 'severity': 'high', 'likelihood': 0.6},
            {'component': 'safety_system_settings', 'severity': 'critical', 'likelihood': 0.4}
        ]
        
        # Repudiation threats
        threats[STRIDECategory.REPUDIATION.value] = [
            {'component': 'charging_session_logs', 'severity': 'medium', 'likelihood': 0.5},
            {'component': 'attack_attribution_data', 'severity': 'high', 'likelihood': 0.7},
            {'component': 'billing_records', 'severity': 'medium', 'likelihood': 0.6},
            {'component': 'federated_learning_contributions', 'severity': 'high', 'likelihood': 0.8},
            {'component': 'system_audit_trails', 'severity': 'high', 'likelihood': 0.6}
        ]
        
        # Information Disclosure threats
        threats[STRIDECategory.INFORMATION_DISCLOSURE.value] = [
            {'component': 'customer_charging_patterns', 'severity': 'high', 'likelihood': 0.7},
            {'component': 'grid_state_vulnerabilities', 'severity': 'critical', 'likelihood': 0.5},
            {'component': 'PINN_model_parameters', 'severity': 'high', 'likelihood': 0.6},
            {'component': 'operational_capacity_limits', 'severity': 'medium', 'likelihood': 0.4},
            {'component': 'thermal_management_data', 'severity': 'medium', 'likelihood': 0.5}
        ]
        
        # Denial of Service threats
        threats[STRIDECategory.DENIAL_OF_SERVICE.value] = [
            {'component': 'charging_port_availability', 'severity': 'high', 'likelihood': 0.8},
            {'component': 'grid_stability_services', 'severity': 'critical', 'likelihood': 0.6},
            {'component': 'PINN_learning_convergence', 'severity': 'medium', 'likelihood': 0.7},
            {'component': 'communication_networks', 'severity': 'high', 'likelihood': 0.5},
            {'component': 'thermal_protection_systems', 'severity': 'critical', 'likelihood': 0.4}
        ]
        
        # Elevation of Privilege threats
        threats[STRIDECategory.ELEVATION_OF_PRIVILEGE.value] = [
            {'component': 'CMS_admin_privileges', 'severity': 'critical', 'likelihood': 0.3},
            {'component': 'PINN_global_model_access', 'severity': 'high', 'likelihood': 0.5},
            {'component': 'grid_operator_controls', 'severity': 'critical', 'likelihood': 0.2},
            {'component': 'emergency_override_systems', 'severity': 'critical', 'likelihood': 0.3},
            {'component': 'thermal_safety_bypass', 'severity': 'critical', 'likelihood': 0.4}
        ]
        
        return threats

class MITREThreatAnalyzer:
    """MITRE ATT&CK analysis for ICS/OT systems"""
    
    def analyze_threats(self, system_data: Dict) -> Dict:
        """Analyze threats using MITRE ATT&CK framework"""
        try:
            mitre_analysis = {
                'initial_access': {
                    'techniques': ['T0817: Drive-by Compromise', 'T0819: Exploit Public-Facing Application'],
                    'severity': 'high',
                    'likelihood': 0.6
                },
                'execution': {
                    'techniques': ['T0807: Command-Line Interface', 'T0871: Execution through API'],
                    'severity': 'high',
                    'likelihood': 0.8
                },
                'persistence': {
                    'techniques': ['T0839: Module Firmware', 'T0891: Hardcoded Credentials'],
                    'severity': 'medium',
                    'likelihood': 0.5
                },
                'privilege_escalation': {
                    'techniques': ['T0890: Exploitation for Privilege Escalation', 'T0874: Hooking'],
                    'severity': 'high',
                    'likelihood': 0.4
                },
                'defense_evasion': {
                    'techniques': ['T0820: Exploitation for Defense Evasion', 'T0872: Indicator Removal'],
                    'severity': 'high',
                    'likelihood': 0.7
                },
                'credential_access': {
                    'techniques': ['T0891: Hardcoded Credentials', 'T0894: Unauthorized Command Message'],
                    'severity': 'medium',
                    'likelihood': 0.5
                },
                'discovery': {
                    'techniques': ['T0840: Network Connection Enumeration', 'T0888: Remote System Discovery'],
                    'severity': 'medium',
                    'likelihood': 0.8
                },
                'lateral_movement': {
                    'techniques': ['T0867: Lateral Tool Transfer', 'T0859: Valid Accounts'],
                    'severity': 'high',
                    'likelihood': 0.6
                },
                'collection': {
                    'techniques': ['T0802: Automated Collection', 'T0845: Program Upload'],
                    'severity': 'medium',
                    'likelihood': 0.7
                },
                'command_and_control': {
                    'techniques': ['T0885: Commonly Used Port', 'T0884: Connection Proxy'],
                    'severity': 'high',
                    'likelihood': 0.8
                },
                'inhibit_response_function': {
                    'techniques': ['T0800: Activate Firmware Update Mode', 'T0816: Device Restart/Shutdown'],
                    'severity': 'critical',
                    'likelihood': 0.9
                },
                'impair_process_control': {
                    'techniques': ['T0806: Brute Force I/O', 'T0855: Unauthorized Command Message'],
                    'severity': 'critical',
                    'likelihood': 0.8
                }
            }
            
            return {
                'analysis_type': 'MITRE_ATT&CK',
                'tactics': mitre_analysis,
                'overall_risk': 'critical',
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'analysis_type': 'MITRE_ATT&CK',
                'error': str(e),
                'tactics': {},
                'overall_risk': 'unknown'
            }
    
    def analyze_tactics(self, system_data: Dict) -> Dict:
        """Analyze MITRE ATT&CK tactics for the system"""
        try:
            # Use the existing analyze_threats method but format for tactics
            threat_analysis = self.analyze_threats(system_data)
            
            # Extract tactics from the analysis
            tactics_analysis = {
                'tactics_identified': [],
                'techniques_count': 0,
                'high_priority_tactics': [],
                'risk_assessment': threat_analysis.get('overall_risk', 'medium')
            }
            
            # Process tactics from the threat analysis
            if 'tactics' in threat_analysis:
                for tactic_name, tactic_data in threat_analysis['tactics'].items():
                    tactic_info = {
                        'name': tactic_name,
                        'techniques': tactic_data.get('techniques', []),
                        'severity': tactic_data.get('severity', 'medium'),
                        'likelihood': tactic_data.get('likelihood', 0.5)
                    }
                    tactics_analysis['tactics_identified'].append(tactic_info)
                    tactics_analysis['techniques_count'] += len(tactic_info['techniques'])
                    
                    # Mark high priority tactics
                    if tactic_data.get('severity') in ['high', 'critical'] and tactic_data.get('likelihood', 0) > 0.7:
                        tactics_analysis['high_priority_tactics'].append(tactic_name)
            
            tactics_analysis['total_tactics'] = len(tactics_analysis['tactics_identified'])
            tactics_analysis['timestamp'] = time.time()
            
            return tactics_analysis
            
        except Exception as e:
            print(f"# Failed to analyze MITRE tactics: {e}")
            return {
                'tactics_identified': [],
                'techniques_count': 0,
                'high_priority_tactics': [],
                'risk_assessment': 'unknown',
                'total_tactics': 0,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def analyze_system(self, system_analysis: SystemAnalysisData) -> Dict:
        """Perform MITRE ATT&CK analysis on the system"""
        tactics = {}
        
        # Initial Access
        tactics[MITRECategory.INITIAL_ACCESS.value] = [
            {'technique': 'T0817 - Drive-by Compromise', 'applicability': 0.6},
            {'technique': 'T0819 - Exploit Public-Facing Application', 'applicability': 0.7},
            {'technique': 'T0822 - External Remote Services', 'applicability': 0.8}
        ]
        
        # Execution
        tactics[MITRECategory.EXECUTION.value] = [
            {'technique': 'T0807 - Command-Line Interface', 'applicability': 0.5},
            {'technique': 'T0871 - Execution through API', 'applicability': 0.9},
            {'technique': 'T0823 - Graphical User Interface', 'applicability': 0.4},
            {'technique': 'T0859 - Scripting', 'applicability': 0.7},
            {'technique': 'T0831 - Manipulation of Control', 'applicability': 0.8}
        ]
        
        # Persistence
        tactics[MITRECategory.PERSISTENCE.value] = [
            {'technique': 'T0839 - Module Firmware', 'applicability': 0.6},
            {'technique': 'T0891 - Hardcoded Credentials', 'applicability': 0.5},
            {'technique': 'T0844 - Modify Controller Tasking', 'applicability': 0.7},
            {'technique': 'T0845 - Program Download', 'applicability': 0.6},
            {'technique': 'T0846 - Program Upload', 'applicability': 0.5}
        ]
        
        # Privilege Escalation
        tactics[MITRECategory.PRIVILEGE_ESCALATION.value] = [
            {'technique': 'T0890 - Exploitation for Privilege Escalation', 'applicability': 0.4},
            {'technique': 'T0874 - Hooking', 'applicability': 0.6},
            {'technique': 'T0852 - Rootkit', 'applicability': 0.3},
            {'technique': 'T0843 - Modify Parameter', 'applicability': 0.7},
            {'technique': 'T0849 - Reversible-Image', 'applicability': 0.4}
        ]
        
        # Defense Evasion
        tactics[MITRECategory.DEFENSE_EVASION.value] = [
            {'technique': 'T0832 - Manipulation of View', 'applicability': 0.8},
            {'technique': 'T0870 - Indicator Removal on Host', 'applicability': 0.6},
            {'technique': 'T0847 - Rootkit', 'applicability': 0.3},
            {'technique': 'T0848 - Spoof Reporting Message', 'applicability': 0.7},
            {'technique': 'T0850 - Steal or Forge Authentication Certificates', 'applicability': 0.5}
        ]
        
        # Credential Access
        tactics[MITRECategory.CREDENTIAL_ACCESS.value] = [
            {'technique': 'T0851 - Steal Application Access Token', 'applicability': 0.6},
            {'technique': 'T0842 - Modify Authentication Process', 'applicability': 0.5},
            {'technique': 'T0841 - Multi-Factor Authentication Interception', 'applicability': 0.4},
            {'technique': 'T0840 - Network Sniffing', 'applicability': 0.7},
            {'technique': 'T0882 - Unsecured Credentials', 'applicability': 0.8}
        ]
        
        # Discovery
        tactics[MITRECategory.DISCOVERY.value] = [
            {'technique': 'T0861 - Point & Tag Identification', 'applicability': 0.6},
            {'technique': 'T0860 - Remote System Information Discovery', 'applicability': 0.7},
            {'technique': 'T0858 - System Information Discovery', 'applicability': 0.8},
            {'technique': 'T0862 - Wireless Reconnaissance', 'applicability': 0.5},
            {'technique': 'T0880 - Process Discovery', 'applicability': 0.6}
        ]
        
        # Lateral Movement
        tactics[MITRECategory.LATERAL_MOVEMENT.value] = [
            {'technique': 'T0863 - Program Download', 'applicability': 0.6},
            {'technique': 'T0864 - Program Upload', 'applicability': 0.5},
            {'technique': 'T0865 - Remote Services', 'applicability': 0.7},
            {'technique': 'T0866 - Replication Through Removable Media', 'applicability': 0.4},
            {'technique': 'T0867 - Standard Application Layer Protocol', 'applicability': 0.8}
        ]
        
        # Collection
        tactics[MITRECategory.COLLECTION.value] = [
            {'technique': 'T0881 - Data from Information Repositories', 'applicability': 0.7},
            {'technique': 'T0882 - Data from Network Shared Drive', 'applicability': 0.6},
            {'technique': 'T0883 - Data from Local System', 'applicability': 0.8},
            {'technique': 'T0884 - Data from Removable Media', 'applicability': 0.4},
            {'technique': 'T0885 - Screen Capture', 'applicability': 0.5}
        ]
        
        # Command and Control
        tactics[MITRECategory.COMMAND_AND_CONTROL.value] = [
            {'technique': 'T0868 - Commonly Used Port', 'applicability': 0.7},
            {'technique': 'T0869 - Communication Through Removable Media', 'applicability': 0.3},
            {'technique': 'T0872 - Standard Application Layer Protocol', 'applicability': 0.8},
            {'technique': 'T0873 - Uncommonly Used Port', 'applicability': 0.4},
            {'technique': 'T0886 - Web Service', 'applicability': 0.6}
        ]
        
        # Exfiltration
        tactics[MITRECategory.EXFILTRATION.value] = [
            {'technique': 'T0887 - Automated Exfiltration', 'applicability': 0.5},
            {'technique': 'T0888 - Data Transfer Size Limits', 'applicability': 0.6},
            {'technique': 'T0889 - Exfiltration Over Alternative Protocol', 'applicability': 0.4},
            {'technique': 'T0890 - Exfiltration Over Command and Control Channel', 'applicability': 0.7},
            {'technique': 'T0892 - Exfiltration Over Physical Medium', 'applicability': 0.3}
        ]
        
        # Impact
        tactics[MITRECategory.IMPACT.value] = [
            {'technique': 'T0893 - Data Destruction', 'applicability': 0.6},
            {'technique': 'T0894 - Data Manipulation', 'applicability': 0.8},
            {'technique': 'T0895 - Denial of Control', 'applicability': 0.7},
            {'technique': 'T0896 - Denial of View', 'applicability': 0.6},
            {'technique': 'T0897 - Inhibit System Recovery', 'applicability': 0.5},
            {'technique': 'T0898 - Loss of Availability', 'applicability': 0.9},
            {'technique': 'T0899 - Loss of Control', 'applicability': 0.8},
            {'technique': 'T0900 - Loss of Productivity and Revenue', 'applicability': 0.7},
            {'technique': 'T0901 - Manipulation of Control', 'applicability': 0.8},
            {'technique': 'T0902 - Theft of Operational Information', 'applicability': 0.6}
        ]

        return tactics
