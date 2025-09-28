#!/usr/bin/env python3
"""
LangGraph-based Attack Coordination Workflow
Enhanced attack coordination with better separation of concerns
"""

import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Literal
from dataclasses import dataclass
import numpy as np

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False

# LangChain imports for LLM integration
try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Install with: pip install langchain")
    LANGCHAIN_AVAILABLE = False

class AttackState(TypedDict):
    """State schema for attack workflow"""
    # Strategic planning state
    threat_model: Dict[str, Any]
    attack_strategy: Dict[str, Any]
    target_systems: List[int]
    
    # Tactical execution state
    rl_actions: List[Dict[str, Any]]
    system_state: Dict[str, Any]
    execution_results: List[Dict[str, Any]]
    
    # Monitoring and adaptation state
    stealth_metrics: Dict[str, float]
    success_metrics: Dict[str, float]
    adaptation_needed: bool
    
    # Workflow control
    current_phase: str
    episode_number: int
    max_iterations: int
    iteration_count: int
    
    # Debugging and logging
    debug_info: List[str]
    performance_history: List[Dict[str, Any]]

@dataclass
class AttackAction:
    """Enhanced attack action with LangGraph integration"""
    action_type: str
    target_system: int
    magnitude: float
    stealth_level: float
    execution_time: float
    metadata: Dict[str, Any]

class LangGraphAttackCoordinator:
    """LangGraph-based attack coordination workflow"""
    
    def __init__(self, llm_analyzer=None, rl_coordinator=None, hierarchical_sim=None):
        self.llm_analyzer = llm_analyzer
        self.rl_coordinator = rl_coordinator
        self.hierarchical_sim = hierarchical_sim
        
        # Initialize workflow components
        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE else None
        self.workflow = None
        self.app = None
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
        
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
        else:
            print("LangGraph not available, using fallback coordination")
    
    def _build_workflow(self):
        """Build the LangGraph workflow for attack coordination"""
        workflow = StateGraph(AttackState)
        
        # Add nodes for different attack phases
        workflow.add_node("strategic_planning", self._strategic_planning_node)
        workflow.add_node("tactical_preparation", self._tactical_preparation_node)
        workflow.add_node("rl_execution", self._rl_execution_node)
        workflow.add_node("stealth_assessment", self._stealth_assessment_node)
        workflow.add_node("impact_evaluation", self._impact_evaluation_node)
        workflow.add_node("strategy_adaptation", self._strategy_adaptation_node)
        workflow.add_node("episode_completion", self._episode_completion_node)
        
        # Define workflow edges and routing
        workflow.set_entry_point("strategic_planning")
        
        # Strategic planning -> Tactical preparation
        workflow.add_edge("strategic_planning", "tactical_preparation")
        
        # Tactical preparation -> RL execution
        workflow.add_edge("tactical_preparation", "rl_execution")
        
        # RL execution -> Stealth assessment
        workflow.add_edge("rl_execution", "stealth_assessment")
        
        # Conditional routing from stealth assessment
        workflow.add_conditional_edges(
            "stealth_assessment",
            self._should_continue_attack,
            {
                "continue": "impact_evaluation",
                "adapt": "strategy_adaptation",
                "abort": "episode_completion"
            }
        )
        
        # Impact evaluation -> Check for adaptation
        workflow.add_conditional_edges(
            "impact_evaluation",
            self._should_adapt_strategy,
            {
                "adapt": "strategy_adaptation",
                "continue": "tactical_preparation",
                "complete": "episode_completion"
            }
        )
        
        # Strategy adaptation -> Tactical preparation
        workflow.add_edge("strategy_adaptation", "tactical_preparation")
        
        # Episode completion -> END
        workflow.add_edge("episode_completion", END)
        
        # Compile the workflow
        self.app = workflow.compile(checkpointer=self.memory)
        
    def _strategic_planning_node(self, state: AttackState) -> AttackState:
        """LLM-based strategic planning node"""
        state["debug_info"].append(f"Phase: Strategic Planning (Episode {state['episode_number']})")
        state["current_phase"] = "strategic_planning"
        
        if self.llm_analyzer:
            try:
                # Get LLM threat model (strategic level only)
                threat_model = self._get_llm_threat_model(state)
                attack_strategy = self._generate_attack_strategy(threat_model, state)
                
                state["threat_model"] = threat_model
                state["attack_strategy"] = attack_strategy
                
                state["debug_info"].append(f"Generated strategy: {attack_strategy.get('strategy_type', 'unknown')}")
                
            except Exception as e:
                state["debug_info"].append(f"Strategic planning error: {str(e)}")
                # Fallback strategy
                state["attack_strategy"] = self._get_fallback_strategy()
        else:
            state["attack_strategy"] = self._get_fallback_strategy()
        
        return state
    
    def _tactical_preparation_node(self, state: AttackState) -> AttackState:
        """Prepare tactical execution based on strategic guidance"""
        state["debug_info"].append("Phase: Tactical Preparation")
        state["current_phase"] = "tactical_preparation"
        
        # Convert strategic guidance to RL-compatible actions
        if self.rl_coordinator and state.get("attack_strategy"):
            try:
                # Get current system state for RL agents
                system_state = self._get_system_state()
                state["system_state"] = system_state
                
                # Prepare RL actions based on LLM strategy
                rl_actions = self._prepare_rl_actions(state["attack_strategy"], system_state)
                state["rl_actions"] = rl_actions
                
                state["debug_info"].append(f"Prepared {len(rl_actions)} RL actions")
                
            except Exception as e:
                state["debug_info"].append(f"Tactical preparation error: {str(e)}")
                state["rl_actions"] = []
        else:
            state["rl_actions"] = []
        
        return state
    
    def _rl_execution_node(self, state: AttackState) -> AttackState:
        """Execute attacks using RL agents"""
        state["debug_info"].append("Phase: RL Execution")
        state["current_phase"] = "rl_execution"
        
        execution_results = []
        
        if self.rl_coordinator and state.get("rl_actions"):
            try:
                # Execute RL actions on the system
                for action_data in state["rl_actions"]:
                    action = AttackAction(**action_data)
                    result = self._execute_rl_action(action, state)
                    execution_results.append(result)
                
                state["execution_results"] = execution_results
                state["debug_info"].append(f"Executed {len(execution_results)} actions")
                
            except Exception as e:
                state["debug_info"].append(f"RL execution error: {str(e)}")
                state["execution_results"] = []
        else:
            state["execution_results"] = []
        
        return state
    
    def _stealth_assessment_node(self, state: AttackState) -> AttackState:
        """Assess attack stealth and detection risk"""
        state["debug_info"].append("Phase: Stealth Assessment")
        state["current_phase"] = "stealth_assessment"
        
        stealth_metrics = {
            "detection_probability": 0.0,
            "stealth_score": 1.0,
            "anomaly_level": 0.0
        }
        
        if state.get("execution_results"):
            try:
                # Calculate stealth metrics from execution results
                detection_scores = []
                stealth_scores = []
                
                for result in state["execution_results"]:
                    detection_scores.append(result.get("detection_probability", 0.0))
                    stealth_scores.append(result.get("stealth_level", 1.0))
                
                stealth_metrics["detection_probability"] = np.mean(detection_scores)
                stealth_metrics["stealth_score"] = np.mean(stealth_scores)
                stealth_metrics["anomaly_level"] = max(detection_scores)
                
                state["debug_info"].append(f"Stealth score: {stealth_metrics['stealth_score']:.3f}")
                
            except Exception as e:
                state["debug_info"].append(f"Stealth assessment error: {str(e)}")
        
        state["stealth_metrics"] = stealth_metrics
        return state
    
    def _impact_evaluation_node(self, state: AttackState) -> AttackState:
        """Evaluate attack impact and success"""
        state["debug_info"].append("Phase: Impact Evaluation")
        state["current_phase"] = "impact_evaluation"
        
        success_metrics = {
            "impact_score": 0.0,
            "success_rate": 0.0,
            "target_achievement": 0.0
        }
        
        if state.get("execution_results"):
            try:
                # Calculate success metrics
                impact_scores = []
                success_flags = []
                
                for result in state["execution_results"]:
                    impact_scores.append(result.get("impact", 0.0))
                    success_flags.append(result.get("success", False))
                
                success_metrics["impact_score"] = np.mean(impact_scores)
                success_metrics["success_rate"] = np.mean(success_flags)
                success_metrics["target_achievement"] = min(success_metrics["impact_score"] / 100.0, 1.0)
                
                state["debug_info"].append(f"Impact score: {success_metrics['impact_score']:.2f}")
                
            except Exception as e:
                state["debug_info"].append(f"Impact evaluation error: {str(e)}")
        
        state["success_metrics"] = success_metrics
        return state
    
    def _strategy_adaptation_node(self, state: AttackState) -> AttackState:
        """Adapt strategy based on performance feedback"""
        state["debug_info"].append("Phase: Strategy Adaptation")
        state["current_phase"] = "strategy_adaptation"
        
        if self.llm_analyzer and state.get("stealth_metrics") and state.get("success_metrics"):
            try:
                # Provide feedback to LLM for strategy adaptation
                feedback = {
                    "stealth_performance": state["stealth_metrics"],
                    "success_performance": state["success_metrics"],
                    "execution_results": state["execution_results"]
                }
                
                # Get adapted strategy from LLM
                adapted_strategy = self._adapt_strategy_with_llm(state["attack_strategy"], feedback)
                state["attack_strategy"] = adapted_strategy
                
                state["debug_info"].append("Strategy adapted based on performance feedback")
                
            except Exception as e:
                state["debug_info"].append(f"Strategy adaptation error: {str(e)}")
        
        state["adaptation_needed"] = False
        return state
    
    def _episode_completion_node(self, state: AttackState) -> AttackState:
        """Complete episode and prepare for next iteration"""
        state["debug_info"].append("Phase: Episode Completion")
        state["current_phase"] = "episode_completion"
        
        # Record performance history
        episode_performance = {
            "episode": state["episode_number"],
            "stealth_metrics": state.get("stealth_metrics", {}),
            "success_metrics": state.get("success_metrics", {}),
            "execution_count": len(state.get("execution_results", [])),
            "adaptation_count": state.get("iteration_count", 0)
        }
        
        state["performance_history"].append(episode_performance)
        state["debug_info"].append(f"Episode {state['episode_number']} completed")
        
        return state
    
    def _should_continue_attack(self, state: AttackState) -> Literal["continue", "adapt", "abort"]:
        """Decide whether to continue, adapt, or abort attack"""
        stealth_metrics = state.get("stealth_metrics", {})
        detection_prob = stealth_metrics.get("detection_probability", 0.0)
        stealth_score = stealth_metrics.get("stealth_score", 1.0)
        
        # Abort if detection risk is too high
        if detection_prob > 0.8:
            return "abort"
        
        # Adapt if stealth is compromised but not critically
        if stealth_score < 0.5 or detection_prob > 0.6:
            return "adapt"
        
        # Continue if stealth is maintained
        return "continue"
    
    def _should_adapt_strategy(self, state: AttackState) -> Literal["adapt", "continue", "complete"]:
        """Decide whether to adapt strategy, continue, or complete episode"""
        success_metrics = state.get("success_metrics", {})
        success_rate = success_metrics.get("success_rate", 0.0)
        impact_score = success_metrics.get("impact_score", 0.0)
        
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        
        # Complete if max iterations reached
        if iteration_count >= max_iterations:
            return "complete"
        
        # Adapt if performance is poor
        if success_rate < 0.3 or impact_score < 30.0:
            state["iteration_count"] = iteration_count + 1
            return "adapt"
        
        # Continue if performance is acceptable but not optimal
        if success_rate < 0.7 or impact_score < 70.0:
            state["iteration_count"] = iteration_count + 1
            return "continue"
        
        # Complete if performance is good
        return "complete"
    
    def run_attack_episode(self, scenario, episode_number: int) -> Dict[str, Any]:
        """Run a complete attack episode using LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE or not self.app:
            return self._run_fallback_episode(scenario, episode_number)
        
        # Initialize episode state
        initial_state = AttackState(
            threat_model={},
            attack_strategy={},
            target_systems=scenario.target_systems,
            rl_actions=[],
            system_state={},
            execution_results=[],
            stealth_metrics={},
            success_metrics={},
            adaptation_needed=False,
            current_phase="initialization",
            episode_number=episode_number,
            max_iterations=5,
            iteration_count=0,
            debug_info=[f"Starting episode {episode_number}"],
            performance_history=[]
        )
        
        try:
            # Run the workflow with recursion limit
            thread_config = {
                "configurable": {"thread_id": f"episode_{episode_number}"},
                "recursion_limit": 10  # Limit iterations to prevent infinite loops
            }
            final_state = self.app.invoke(initial_state, config=thread_config)
            
            # Extract results
            results = {
                "episode_number": episode_number,
                "success": len(final_state.get("execution_results", [])) > 0,
                "stealth_metrics": final_state.get("stealth_metrics", {}),
                "success_metrics": final_state.get("success_metrics", {}),
                "execution_results": final_state.get("execution_results", []),
                "debug_info": final_state.get("debug_info", []),
                "performance_history": final_state.get("performance_history", []),
                "workflow_completed": True
            }
            
            # Store in execution history
            self.execution_history.append(results)
            
            return results
            
        except Exception as e:
            print(f"LangGraph workflow error: {e}")
            return self._run_fallback_episode(scenario, episode_number)
    
    # Helper methods for workflow nodes
    def _get_llm_threat_model(self, state: AttackState) -> Dict[str, Any]:
        """Get threat model from LLM (strategic level only)"""
        if not self.llm_analyzer:
            return {"strategy_type": "generic", "priority": "medium"}
        
        # This would call your existing LLM analyzer
        # Following the memory pattern of strategic-only guidance
        threat_context = {
            "target_systems": state["target_systems"],
            "episode_number": state["episode_number"],
            "performance_history": state.get("performance_history", [])
        }
        
        return self.llm_analyzer.generate_threat_model(threat_context)
    
    def _generate_attack_strategy(self, threat_model: Dict, state: AttackState) -> Dict[str, Any]:
        """Generate attack strategy from threat model"""
        return {
            "strategy_type": threat_model.get("strategy_type", "coordinated_disruption"),
            "priority_targets": state["target_systems"][:2],  # Focus on first 2 systems
            "stealth_priority": 0.8,
            "impact_goal": 0.7,
            "execution_phases": ["preparation", "execution", "monitoring"]
        }
    
    def _get_fallback_strategy(self) -> Dict[str, Any]:
        """Fallback strategy when LLM is not available"""
        return {
            "strategy_type": "basic_disruption",
            "priority_targets": [1, 2],
            "stealth_priority": 0.6,
            "impact_goal": 0.5,
            "execution_phases": ["execution"]
        }
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for RL agents"""
        if self.hierarchical_sim:
            return self.hierarchical_sim.get_current_state()
        else:
            # Mock system state
            return {
                "voltage": np.random.normal(1.0, 0.05, 6).tolist(),
                "power": np.random.normal(50.0, 10.0, 6).tolist(),
                "frequency": 60.0 + np.random.normal(0, 0.1)
            }
    
    def _prepare_rl_actions(self, strategy: Dict, system_state: Dict) -> List[Dict[str, Any]]:
        """Prepare RL actions based on strategy and system state"""
        actions = []
        
        for target_system in strategy.get("priority_targets", [1, 2]):
            action = {
                "action_type": "power_manipulation",
                "target_system": target_system,
                "magnitude": 0.3 * strategy.get("impact_goal", 0.5),
                "stealth_level": strategy.get("stealth_priority", 0.6),
                "execution_time": time.time(),
                "metadata": {"strategy_type": strategy.get("strategy_type", "unknown")}
            }
            actions.append(action)
        
        return actions
    
    def _execute_rl_action(self, action: AttackAction, state: AttackState) -> Dict[str, Any]:
        """Execute single RL action and return results"""
        if self.rl_coordinator:
            # Use existing RL coordinator
            return self.rl_coordinator.execute_coordinated_attack([action])
        else:
            # Mock execution
            return {
                "success": np.random.random() > 0.3,
                "impact": action.magnitude * 50 + np.random.normal(0, 10),
                "detection_probability": max(0, 1 - action.stealth_level + np.random.normal(0, 0.1)),
                "stealth_level": action.stealth_level,
                "execution_time": action.execution_time
            }
    
    def _adapt_strategy_with_llm(self, current_strategy: Dict, feedback: Dict) -> Dict[str, Any]:
        """Adapt strategy using LLM feedback"""
        if not self.llm_analyzer:
            return current_strategy
        
        # This would call your LLM analyzer with performance feedback
        # Following the memory pattern of strategic adaptation
        adapted_strategy = current_strategy.copy()
        
        # Simple adaptation logic (you can enhance this)
        if feedback["stealth_performance"]["detection_probability"] > 0.6:
            adapted_strategy["stealth_priority"] = min(0.9, adapted_strategy.get("stealth_priority", 0.6) + 0.1)
        
        if feedback["success_performance"]["success_rate"] < 0.3:
            adapted_strategy["impact_goal"] = min(0.9, adapted_strategy.get("impact_goal", 0.5) + 0.1)
        
        return adapted_strategy
    
    def _run_fallback_episode(self, scenario, episode_number: int) -> Dict[str, Any]:
        """Fallback episode execution when LangGraph is not available"""
        print(f"Running fallback episode {episode_number}")
        
        # Simple fallback logic
        results = {
            "episode_number": episode_number,
            "success": True,
            "stealth_metrics": {"detection_probability": 0.3, "stealth_score": 0.7},
            "success_metrics": {"impact_score": 45.0, "success_rate": 0.6},
            "execution_results": [{"success": True, "impact": 45.0}],
            "debug_info": [f"Fallback episode {episode_number} completed"],
            "performance_history": [],
            "workflow_completed": False
        }
        
        return results
    
    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow"""
        if not LANGGRAPH_AVAILABLE:
            return "LangGraph not available - workflow visualization unavailable"
        
        return """
        LangGraph Attack Coordination Workflow:
        
        [START] → Strategic Planning (LLM)
                      ↓
                 Tactical Preparation (RL State Processing)
                      ↓
                 RL Execution (Attack Actions)
                      ↓
                 Stealth Assessment
                      ↓
                 ┌─ Continue? ─┐
                 │             │
            [Abort]       [Continue/Adapt]
                 │             │
               [END]      Impact Evaluation
                              ↓
                         ┌─ Adapt? ─┐
                         │          │
                    [Complete]  Strategy Adaptation
                         │          │
                       [END]        ↓
                              Tactical Preparation
                                   (Loop)
        """
