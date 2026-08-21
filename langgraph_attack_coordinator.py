#!/usr/bin/env python3


import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Literal
from dataclasses import dataclass
import numpy as np


try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False


try:

    try:
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    except ImportError:

        from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Install with: pip install langchain langchain-core")
    LANGCHAIN_AVAILABLE = False

class AttackState(TypedDict):


    threat_model: Dict[str, Any]
    attack_strategy: Dict[str, Any]
    target_systems: List[int]


    rl_actions: List[Dict[str, Any]]
    system_state: Dict[str, Any]
    execution_results: List[Dict[str, Any]]


    stealth_metrics: Dict[str, float]
    success_metrics: Dict[str, float]
    adaptation_needed: bool


    current_phase: str
    episode_number: int
    max_iterations: int
    iteration_count: int


    debug_info: List[str]
    performance_history: List[Dict[str, Any]]

@dataclass
class AttackAction:

    action_type: str
    target_system: int
    magnitude: float
    stealth_level: float
    execution_time: float
    metadata: Dict[str, Any]

class LangGraphAttackCoordinator:


    def __init__(self, llm_analyzer=None, rl_coordinator=None, hierarchical_sim=None):
        self.llm_analyzer = llm_analyzer
        self.rl_coordinator = rl_coordinator
        self.hierarchical_sim = hierarchical_sim


        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE else None
        self.workflow = None
        self.app = None


        self.execution_history = []
        self.performance_metrics = {}

        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
        else:
            print("LangGraph not available, using fallback coordination")

    def _build_workflow(self):

        workflow = StateGraph(AttackState)


        workflow.add_node("strategic_planning", self._strategic_planning_node)
        workflow.add_node("tactical_preparation", self._tactical_preparation_node)
        workflow.add_node("rl_execution", self._rl_execution_node)
        workflow.add_node("stealth_assessment", self._stealth_assessment_node)
        workflow.add_node("impact_evaluation", self._impact_evaluation_node)
        workflow.add_node("strategy_adaptation", self._strategy_adaptation_node)
        workflow.add_node("episode_completion", self._episode_completion_node)


        workflow.set_entry_point("strategic_planning")


        workflow.add_edge("strategic_planning", "tactical_preparation")


        workflow.add_edge("tactical_preparation", "rl_execution")


        workflow.add_edge("rl_execution", "stealth_assessment")


        workflow.add_conditional_edges(
            "stealth_assessment",
            self._should_continue_attack,
            {
                "continue": "impact_evaluation",
                "adapt": "strategy_adaptation",
                "abort": "episode_completion"
            }
        )


        workflow.add_conditional_edges(
            "impact_evaluation",
            self._should_adapt_strategy,
            {
                "adapt": "strategy_adaptation",
                "continue": "tactical_preparation",
                "complete": "episode_completion"
            }
        )


        workflow.add_edge("strategy_adaptation", "tactical_preparation")


        workflow.add_edge("episode_completion", END)


        self.app = workflow.compile(checkpointer=self.memory)

    def _strategic_planning_node(self, state: AttackState) -> AttackState:

        state["debug_info"].append(f"Phase: Strategic Planning (Episode {state['episode_number']})")
        state["current_phase"] = "strategic_planning"

        if self.llm_analyzer:
            try:

                threat_model = self._get_llm_threat_model(state)
                attack_strategy = self._generate_attack_strategy(threat_model, state)

                state["threat_model"] = threat_model
                state["attack_strategy"] = attack_strategy

                state["debug_info"].append(f"Generated strategy: {attack_strategy.get('strategy_type', 'unknown')}")

            except Exception as e:
                state["debug_info"].append(f"Strategic planning error: {str(e)}")

                state["attack_strategy"] = self._get_fallback_strategy()
        else:
            state["attack_strategy"] = self._get_fallback_strategy()

        return state

    def _tactical_preparation_node(self, state: AttackState) -> AttackState:

        state["debug_info"].append("Phase: Tactical Preparation")
        state["current_phase"] = "tactical_preparation"


        if self.rl_coordinator and state.get("attack_strategy"):
            try:

                system_state = self._get_system_state()
                state["system_state"] = system_state


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

        state["debug_info"].append("Phase: RL Execution")
        state["current_phase"] = "rl_execution"

        execution_results = []

        if self.rl_coordinator and state.get("rl_actions"):
            try:

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

        state["debug_info"].append("Phase: Stealth Assessment")
        state["current_phase"] = "stealth_assessment"

        stealth_metrics = {
            "detection_probability": 0.0,
            "stealth_score": 1.0,
            "anomaly_level": 0.0
        }

        if state.get("execution_results"):
            try:

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

        state["debug_info"].append("Phase: Impact Evaluation")
        state["current_phase"] = "impact_evaluation"

        success_metrics = {
            "impact_score": 0.0,
            "success_rate": 0.0,
            "target_achievement": 0.0
        }

        if state.get("execution_results"):
            try:

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

        state["debug_info"].append("Phase: Strategy Adaptation")
        state["current_phase"] = "strategy_adaptation"

        if self.llm_analyzer and state.get("stealth_metrics") and state.get("success_metrics"):
            try:

                feedback = {
                    "stealth_performance": state["stealth_metrics"],
                    "success_performance": state["success_metrics"],
                    "execution_results": state["execution_results"]
                }


                adapted_strategy = self._adapt_strategy_with_llm(state["attack_strategy"], feedback)
                state["attack_strategy"] = adapted_strategy

                state["debug_info"].append("Strategy adapted based on performance feedback")

            except Exception as e:
                state["debug_info"].append(f"Strategy adaptation error: {str(e)}")

        state["adaptation_needed"] = False
        return state

    def _episode_completion_node(self, state: AttackState) -> AttackState:

        state["debug_info"].append("Phase: Episode Completion")
        state["current_phase"] = "episode_completion"


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

        stealth_metrics = state.get("stealth_metrics", {})
        detection_prob  = stealth_metrics.get("detection_probability", 0.0)
        stealth_score   = stealth_metrics.get("stealth_score", 1.0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations  = state.get("max_iterations", 5)


        if detection_prob >= 1.0 and iteration_count >= max_iterations:
            return "abort"


        if stealth_score < 0.5 or detection_prob > 0.6:
            return "adapt"

        return "continue"

    def _should_adapt_strategy(self, state: AttackState) -> Literal["adapt", "continue", "complete"]:

        success_metrics = state.get("success_metrics", {})
        success_rate    = success_metrics.get("success_rate", 0.0)
        impact_score    = success_metrics.get("impact_score", 0.0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations  = state.get("max_iterations", 5)


        state["iteration_count"] = iteration_count + 1


        if iteration_count >= max_iterations:
            return "complete"


        if success_rate >= 0.7 and impact_score >= 70.0:
            return "complete"


        if success_rate < 0.3 or impact_score < 30.0:
            return "adapt"


        return "continue"

    def run_attack_episode(self, scenario, episode_number: int) -> Dict[str, Any]:

        if not LANGGRAPH_AVAILABLE or not self.app:
            return self._run_fallback_episode(scenario, episode_number)


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

            thread_config = {
                "configurable": {"thread_id": f"episode_{episode_number}"},
                "recursion_limit": 100
            }
            final_state = self.app.invoke(initial_state, config=thread_config)


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


            self.execution_history.append(results)

            return results

        except Exception as e:
            print(f"LangGraph workflow error: {e}")
            return self._run_fallback_episode(scenario, episode_number)


    def _get_llm_threat_model(self, state: AttackState) -> Dict[str, Any]:

        if not self.llm_analyzer:
            return {"strategy_type": "generic", "priority": "medium"}


        threat_context = {
            "target_systems": state["target_systems"],
            "episode_number": state["episode_number"],
            "performance_history": state.get("performance_history", [])
        }

        return self.llm_analyzer.generate_threat_model(threat_context)

    def _generate_attack_strategy(self, threat_model: Dict, state: AttackState) -> Dict[str, Any]:

        return {
            "strategy_type": threat_model.get("strategy_type", "coordinated_disruption"),
            "priority_targets": state["target_systems"][:2],
            "stealth_priority": 0.8,
            "impact_goal": 0.7,
            "execution_phases": ["preparation", "execution", "monitoring"]
        }

    def _get_fallback_strategy(self) -> Dict[str, Any]:

        return {
            "strategy_type": "basic_disruption",
            "priority_targets": [1, 2],
            "stealth_priority": 0.6,
            "impact_goal": 0.5,
            "execution_phases": ["execution"]
        }

    def _get_system_state(self) -> Dict[str, Any]:

        if self.hierarchical_sim:
            return self.hierarchical_sim.get_current_state()
        else:

            return {
                "voltage": np.random.normal(1.0, 0.05, 6).tolist(),
                "power": np.random.normal(50.0, 10.0, 6).tolist(),
                "frequency": 60.0 + np.random.normal(0, 0.1)
            }

    def _prepare_rl_actions(self, strategy: Dict, system_state: Dict) -> List[Dict[str, Any]]:

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

        if self.rl_coordinator:

            return self.rl_coordinator.execute_coordinated_attack([action])
        else:

            return {
                "success": np.random.random() > 0.3,
                "impact": action.magnitude * 50 + np.random.normal(0, 10),
                "detection_probability": max(0, 1 - action.stealth_level + np.random.normal(0, 0.1)),
                "stealth_level": action.stealth_level,
                "execution_time": action.execution_time
            }

    def _adapt_strategy_with_llm(self, current_strategy: Dict, feedback: Dict) -> Dict[str, Any]:

        if not self.llm_analyzer:
            return current_strategy


        adapted_strategy = current_strategy.copy()


        if feedback["stealth_performance"]["detection_probability"] > 0.6:
            adapted_strategy["stealth_priority"] = min(0.9, adapted_strategy.get("stealth_priority", 0.6) + 0.1)

        if feedback["success_performance"]["success_rate"] < 0.3:
            adapted_strategy["impact_goal"] = min(0.9, adapted_strategy.get("impact_goal", 0.5) + 0.1)

        return adapted_strategy

    def _run_fallback_episode(self, scenario, episode_number: int) -> Dict[str, Any]:

        print(f"Running fallback episode {episode_number}")


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

        if not LANGGRAPH_AVAILABLE:
            return "LangGraph not available - workflow visualization unavailable"

        return """
        LangGraph Attack Coordination Workflow:

        [START]  Strategic Planning (LLM)

                 Tactical Preparation (RL State Processing)

                 RL Execution (Attack Actions)

                 Stealth Assessment

                  Continue?

            [Abort]       [Continue/Adapt]

               [END]      Impact Evaluation

                          Adapt?

                    [Complete]  Strategy Adaptation

                       [END]
                              Tactical Preparation
                                   (Loop)
        """
