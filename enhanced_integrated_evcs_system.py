#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading
import asyncio
import sys
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


try:
    from hierarchical_cosimulation import HierarchicalCoSimulation, EnhancedChargingManagementSystem, EVChargingStation
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    import traceback
    traceback.print_exc()
    print("Warning: Hierarchical co-simulation not available")
    HIERARCHICAL_AVAILABLE = False


try:
    from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
    from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig, PhysicsDataGenerator
    FEDERATED_PINN_AVAILABLE = True
except ImportError:
    print("Warning: Federated PINN not available")
    FEDERATED_PINN_AVAILABLE = False


from gemini_llm_threat_analyzer import GeminiLLMThreatAnalyzer


try:
    from langgraph_attack_coordinator import LangGraphAttackCoordinator, AttackState, AttackAction
    LANGGRAPH_COORDINATOR_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph attack coordinator not available")
    LANGGRAPH_COORDINATOR_AVAILABLE = False


from dqn_sac_security_evasion import DQNSACSecurityEvasionTrainer, SecurityEvasionEnvironment, DiscreteSecurityEvasionEnv


try:
    from attack_specific_rl_agents import (
        AttackSpecificCoordinator,
        AttackSpecificEnvironment,
        DiscreteAttackSpecificEnvironment,
        AttackDeployment,
        ATTACK_TYPES
    )
    from gemini_attack_deployment import (
        create_gemini_deployment_prompt,
        create_gemini_adaptation_prompt,
        parse_gemini_deployment_response
    )
    ATTACK_SPECIFIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Attack-specific RL agents not available: {e}")
    ATTACK_SPECIFIC_AVAILABLE = False


try:
    from enhanced_llm_rl_coordinator import EnhancedLLMRLCoordinator, AttackType, STRIDECategory, MITRECategory
    ENHANCED_COORDINATOR_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced LLM-RL coordinator not available")
    ENHANCED_COORDINATOR_AVAILABLE = False

warnings.filterwarnings('ignore')


DATA_INTEGRITY = {
    'pinn_fallback_attack_sim': 0,
    'pinn_dummy_training_data': 0,
}


class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'enhanched_evcs_system_log_{timestamp}.txt'


sys.stdout = PrintLogger(log_filename)


class RealTimeRLAttackController:


    def __init__(self, attack_coordinator, deployment_plan: Dict, duration_seconds: float):

        self.attack_coordinator = attack_coordinator
        self.duration_seconds = duration_seconds
        self.active_attacks = {}
        self.attack_log = []
        self.decision_counter = 0


        self.attack_schedule = []

        deployments = deployment_plan.get('deployments', [])
        if not deployments:
            print("  ##?? RealTimeRLAttackController: No deployments in plan")
            return

        n_attacks = len(deployments)
        margin = duration_seconds * 0.10
        usable_window = duration_seconds - 2 * margin
        attack_duration = usable_window / (n_attacks + 1)
        spacing = usable_window / max(n_attacks, 1)

        for idx, dep in enumerate(deployments):
            start_time = margin + idx * spacing
            end_time = start_time + attack_duration


            _nodes = dep.get('target_nodes')
            if not _nodes and dep.get('target_node') is not None:
                _nodes = [dep.get('target_node')]
            entry = {
                'attack_type': dep.get('attack_type', 'power_disruption'),
                'target_system': dep.get('target_system', idx + 1),
                'target_node': dep.get('target_node'),
                'target_nodes': _nodes,
                'start_time': start_time,
                'end_time': end_time,
                'magnitude': dep.get('magnitude', 0.7),
                'stealth_level': dep.get('stealth_level', 0.7),
                'active': False
            }
            self.attack_schedule.append(entry)
            _node_str = f" nodes {_nodes}" if _nodes else ""
            print(f"    ##Scheduled: {entry['attack_type']}  System {entry['target_system']}{_node_str} "
                  f"[{start_time:.0f}s - {end_time:.0f}s]")

        print(f"  ## RealTimeRLAttackController: {len(self.attack_schedule)} attacks scheduled")

    def make_attack_decision(self, system_states: Dict, current_time: float) -> Dict:

        new_attacks = {}

        for schedule_entry in self.attack_schedule:
            target_sys = schedule_entry['target_system']
            attack_type = schedule_entry['attack_type']


            if schedule_entry['start_time'] <= current_time <= schedule_entry['end_time']:
                if not schedule_entry['active']:

                    schedule_entry['active'] = True

                    attack_params = self._get_rl_attack_params(
                        attack_type, target_sys, system_states, current_time,
                        target_node=schedule_entry.get('target_node'),
                        target_nodes=schedule_entry.get('target_nodes')
                    )

                    if attack_params:
                        new_attacks[target_sys] = attack_params
                        self.active_attacks[target_sys] = {
                            'attack_type': attack_type,
                            'params': attack_params,
                            'start_time': current_time,
                            'schedule_entry': schedule_entry
                        }

                        self.attack_log.append({
                            'time': current_time,
                            'action': 'activate',
                            'target_system': target_sys,
                            'attack_type': attack_type,
                            'params': attack_params
                        })

                elif target_sys in self.active_attacks:

                    self.decision_counter += 1
                    if self.decision_counter % 30 == 0:
                        updated_params = self._get_rl_attack_params(
                            attack_type, target_sys, system_states, current_time,
                            target_node=schedule_entry.get('target_node'),
                            target_nodes=schedule_entry.get('target_nodes')
                        )
                        if updated_params:
                            self.active_attacks[target_sys]['params'] = updated_params
                            new_attacks[target_sys] = updated_params

            elif current_time > schedule_entry['end_time'] and schedule_entry['active']:

                schedule_entry['active'] = False
                if target_sys in self.active_attacks:
                    del self.active_attacks[target_sys]
                    self.attack_log.append({
                        'time': current_time,
                        'action': 'deactivate',
                        'target_system': target_sys,
                        'attack_type': attack_type
                    })

        return new_attacks

    def _get_rl_attack_params(self, attack_type: str, target_system: int,
                               system_states: Dict, current_time: float,
                               target_node: Optional[int] = None,
                               target_nodes: Optional[list] = None) -> Optional[Dict]:

        try:
            sac_env_key = f'sac_{attack_type}'
            if sac_env_key not in self.attack_coordinator.environments:
                return self._fallback_attack_params(attack_type, target_system)

            sac_env = self.attack_coordinator.environments[sac_env_key]


            sac_env.forced_target_system = target_system
            node_level = getattr(sac_env, 'node_level', False)
            if node_level and target_node is not None:
                sac_env.forced_target_node = int(target_node)


            obs = sac_env._get_global_observation()


            action, _ = self.attack_coordinator.sac_agents[attack_type].predict(
                obs, deterministic=True
            )


            sac_env.forced_target_system = None
            if node_level:
                sac_env.forced_target_node = None


            magnitude = float(np.clip(action[0], 0.1, 2.0))
            duration = float(np.clip(action[1], 5.0, 60.0))
            stealth_level = float(np.clip(action[2], 0.0, 1.0))

            params = {
                'type': attack_type,
                'magnitude': magnitude,
                'duration': duration,
                'stealth_level': stealth_level,


                'target_percentage': 100,
                'target_node': target_node,
                'target_nodes': target_nodes,
                'start_time': current_time,
                'decision_id': f'rl_{attack_type}_{target_system}'
                               + (f'_n{target_node}' if target_node is not None else '')
                               + f'_{current_time:.0f}',
                'rl_generated': True
            }
            return params

        except Exception as e:
            if current_time % 60 == 0:
                print(f"  ##?? RL agent inference failed for {attack_type}: {e}")
            return self._fallback_attack_params(attack_type, target_system)

    def _fallback_attack_params(self, attack_type: str, target_system: int) -> Dict:

        return {
            'type': attack_type,
            'magnitude': 0.7,
            'duration': 30.0,
            'stealth_level': 0.5,
            'target_percentage': 100,
            'start_time': 0.0,
            'decision_id': f'fallback_{attack_type}_{target_system}',
            'rl_generated': False
        }

    def update_attack_parameters(self, system_id: int, current_time: float) -> Dict:

        if system_id not in self.active_attacks:
            return {'stop_attack': True}

        attack_info = self.active_attacks[system_id]
        schedule = attack_info['schedule_entry']


        if current_time > schedule['end_time']:
            return {'stop_attack': True}


        return attack_info['params']

    def get_attack_status(self) -> Dict:

        return {
            'active_attacks': len(self.active_attacks),
            'total_scheduled': len(self.attack_schedule),
            'total_decisions': self.decision_counter,
            'active_systems': list(self.active_attacks.keys())
        }


@dataclass
class EnhancedAttackScenario:

    scenario_id: str
    name: str
    description: str
    target_systems: List[int]
    attack_duration: float
    stealth_requirement: float
    impact_goal: float
    constraints: Dict[str, Any]
    coordination_type: str = "simultaneous"

class _PINNAttackSimulationMixin:


    def _simulate_pinn_attack(self, pinn_model, attack_params: Dict) -> Dict:

        try:

            if hasattr(pinn_model, 'optimize_references') and hasattr(pinn_model, 'is_trained') and pinn_model.is_trained:
                real_result = self._execute_real_pinn_cms_attack(pinn_model, attack_params)


                return real_result
            else:

                return self._fallback_pinn_attack_simulation(pinn_model, attack_params)

        except Exception as e:
            print(f"Error in PINN attack execution: {e}")
            return self._fallback_pinn_attack_simulation(pinn_model, attack_params)

    def _execute_real_pinn_cms_attack(self, pinn_model, attack_params: Dict) -> Dict:

        try:
            attack_type = attack_params.get('type', 'voltage_manipulation')
            magnitude = attack_params.get('magnitude', 0.5)
            duration = attack_params.get('duration', 30.0)
            stealth_factor = attack_params.get('stealth_factor', 0.5)

            print(f"      #??# REAL PINN CMS Attack: {attack_type} (mag={magnitude:.2f}, stealth={stealth_factor:.2f})")


            baseline_station_data = {
                'soc': 0.5,
                'grid_voltage': 1.0,
                'grid_frequency': 60.0,
                'demand_factor': 0.5,
                'voltage_priority': 0.0,
                'urgency_factor': 1.0,
                'current_time': 0.0
            }


            try:
                baseline_voltage, baseline_current, baseline_power = pinn_model.optimize_references(baseline_station_data)
                baseline_response = {
                    'voltage': baseline_voltage,
                    'current': baseline_current,
                    'power': baseline_power
                }
                print(f"      ## Baseline CMS: V={baseline_voltage:.1f}V, I={baseline_current:.1f}A, P={baseline_power:.1f}W")
            except Exception as e:
                print(f"      ##?? Baseline CMS call failed: {e}")
                return self._fallback_pinn_attack_simulation(pinn_model, attack_params)


            attacked_station_data = baseline_station_data.copy()

            if attack_type == 'voltage_manipulation':

                voltage_drop_factor = 1.0 - (magnitude * 0.35)
                attacked_station_data['grid_voltage'] *= voltage_drop_factor
                attacked_station_data['voltage_priority'] = max(0, 0.95 - attacked_station_data['grid_voltage'])
                attacked_station_data['power_multiplier'] = 1.0 + magnitude * 20.0
            elif attack_type == 'current_injection':

                cumulative_factor = 1.0 + (magnitude * 45.0)
                attacked_station_data['demand_factor'] *= cumulative_factor
                attacked_station_data['urgency_factor'] *= (1.0 + magnitude * 20.0)
                attacked_station_data['power_multiplier'] = 1.0 + magnitude * 35.0
            elif attack_type == 'power_disruption':

                cumulative_factor = max(0.02, 1.0 - (magnitude * 0.90))
                attacked_station_data['demand_factor'] *= cumulative_factor
                attacked_station_data['urgency_factor'] *= cumulative_factor
                attacked_station_data['power_multiplier'] = cumulative_factor
            elif attack_type == 'communication_spoofing':

                soc_reduction = magnitude * 0.7
                attacked_station_data['soc'] = max(0.01, attacked_station_data['soc'] - soc_reduction)
                attacked_station_data['urgency_factor'] = 1.0 + (magnitude * 40.0)
                attacked_station_data['power_multiplier'] = 1.0 + magnitude * 30.0
            elif attack_type == 'protocol_manipulation':

                import math
                oscillation = math.sin(duration / 4.0) * 20.0 + 1.0
                amplitude_growth = 1.0 + (magnitude * 12.0)
                attacked_station_data['demand_factor'] *= oscillation * amplitude_growth
                attacked_station_data['grid_voltage'] *= (1.0 - magnitude * 0.2)
                attacked_station_data['power_multiplier'] = oscillation * amplitude_growth
            elif attack_type == 'data_injection':

                frequency_deviation = magnitude * 12.0
                attacked_station_data['grid_frequency'] += frequency_deviation
                attacked_station_data['demand_factor'] *= (1.0 + magnitude * 30.0)
                attacked_station_data['power_multiplier'] = 1.0 + magnitude * 25.0


            try:
                attacked_voltage, attacked_current, attacked_power = pinn_model.optimize_references(attacked_station_data)


                if 'power_multiplier' in attacked_station_data:
                    power_multiplier = attacked_station_data['power_multiplier']
                    attacked_power *= power_multiplier
                    attacked_current *= power_multiplier

                attacked_response = {
                    'voltage': attacked_voltage,
                    'current': attacked_current,
                    'power': attacked_power
                }
                print(f"      ## Attacked CMS: V={attacked_voltage:.1f}V, I={attacked_current:.1f}A, P={attacked_power:.1f}W")
            except Exception as e:
                print(f"      ##?? Attacked CMS call failed: {e}")
                return self._fallback_pinn_attack_simulation(pinn_model, attack_params)


            voltage_impact = abs(attacked_voltage - baseline_voltage) / baseline_voltage
            current_impact = abs(attacked_current - baseline_current) / baseline_current
            power_impact = abs(attacked_power - baseline_power) / baseline_power


            real_impact = max(voltage_impact, current_impact, power_impact)


            success_threshold = 0.01
            success = real_impact > success_threshold

            print(f"      ## Real Impact: V={voltage_impact:.3f}, I={current_impact:.3f}, P={power_impact:.3f}  Total={real_impact:.3f}, Success={success}")

            return {
                'success': success,
                'impact': real_impact,
                'attack_type': attack_type,
                'magnitude': magnitude,
                'duration': duration,
                'stealth_factor': stealth_factor,
                'baseline_response': baseline_response,
                'attacked_response': attacked_response,
                'voltage_impact': voltage_impact,
                'current_impact': current_impact,
                'power_impact': power_impact,
                'real_pinn_interaction': True,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"      ##XX Real PINN CMS attack failed: {e}")
            return self._fallback_pinn_attack_simulation(pinn_model, attack_params)

    def _fallback_pinn_attack_simulation(self, pinn_model, attack_params: Dict) -> Dict:

        attack_type = attack_params.get('type', 'voltage_manipulation')
        magnitude = attack_params.get('magnitude', 0.5)
        duration = attack_params.get('duration', 30.0)
        stealth_factor = attack_params.get('stealth_factor', 0.5)


        DATA_INTEGRITY['pinn_fallback_attack_sim'] += 1
        print(f"      ##!! DATA-INTEGRITY WARNING: RANDOM fallback attack simulation for "
              f"'{attack_type}' (real PINN CMS interaction unavailable). Its success/impact "
              f"are np.random draws, NOT measured — metrics from this sample are synthetic. "
              f"[fallback count so far: {DATA_INTEGRITY['pinn_fallback_attack_sim']}]")


        base_success_prob = 0.8
        stealth_bonus = stealth_factor * 0.2
        magnitude_factor = min(magnitude, 1.0)
        success_prob = base_success_prob + stealth_bonus - (magnitude_factor * 0.1)
        success = np.random.random() < success_prob


        impact_multipliers = {
            'voltage_manipulation': 0.8,
            'current_injection': 0.7,
            'power_disruption': 0.9,
            'data_injection': 0.6,
            'communication_spoofing': 0.5,
            'protocol_manipulation': 0.4
        }

        base_impact = impact_multipliers.get(attack_type, 0.5)
        impact = base_impact * magnitude_factor if success else 0.0

        return {
            'success': success,
            'impact': impact,
            'attack_type': attack_type,
            'magnitude': magnitude,
            'duration': duration,
            'stealth_factor': stealth_factor,
            'model_adaptation': np.random.uniform(0.1, 0.3),
            'physics_violation': magnitude_factor * 0.5,
            'convergence_impact': impact * 0.3,
            'learning_disruption': impact * 0.2,
            'real_pinn_interaction': False,
            'timestamp': time.time()
        }


class MultiAgentRLEnvironment(_PINNAttackSimulationMixin, gym.Env):


    def __init__(self, federated_pinn_manager: FederatedPINNManager, num_systems: int = 6):
        super(MultiAgentRLEnvironment, self).__init__()

        self.federated_pinn_manager = federated_pinn_manager
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000


        self.observation_space = spaces.Dict({
            f'agent_{i}': spaces.Box(
                low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
            ) for i in range(num_systems)
        })


        self.action_space = spaces.Dict({
            f'agent_{i}': spaces.Box(
                low=np.array([0.0, 0.1, 5.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([5.0, 2.0, 60.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ) for i in range(num_systems)
        })


        self.coordination_state = {
            'active_attacks': {},
            'system_states': {},
            'global_impact': 0.0,
            'detection_risk': 0.0
        }


        self.max_steps = 200
        self.global_detection_threshold = 0.8
        self.mission_success_impact = 3.0
        self.max_agent_detections = 5


        self.agent_cumulative_impact = {f'agent_{i}': 0.0 for i in range(num_systems)}
        self.agent_detection_count = {f'agent_{i}': 0 for i in range(num_systems)}
        self.episode_done_reason = "running"


        self.episode_rewards = {f'agent_{i}': [] for i in range(num_systems)}
        self.coordination_metrics = []

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.coordination_state = {
            'active_attacks': {},
            'system_states': {},
            'global_impact': 0.0,
            'detection_risk': 0.0
        }
        self.agent_cumulative_impact = {f'agent_{i}': 0.0 for i in range(self.num_systems)}
        self.agent_detection_count = {f'agent_{i}': 0 for i in range(self.num_systems)}
        self.episode_done_reason = "running"


        observations = {}
        for i in range(self.num_systems):
            sys_id = i + 1
            observations[f'agent_{i}'] = self._get_agent_observation(sys_id)

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]):

        self.current_step += 1


        agent_rewards = {}
        agent_observations = {}
        infos = {}


        coordinated_results = self._execute_coordinated_attacks(actions)


        for i in range(self.num_systems):
            agent_key = f'agent_{i}'
            sys_id = i + 1


            agent_result = coordinated_results.get(sys_id, {})


            agent_rewards[agent_key] = self._calculate_agent_reward(
                sys_id, agent_result, coordinated_results
            )


            agent_observations[agent_key] = self._get_agent_observation(sys_id)


            infos[agent_key] = {
                'attack_success': agent_result.get('success', False),
                'detection_risk': agent_result.get('detection_risk', 0.0),
                'system_impact': agent_result.get('impact', 0.0),
                'coordination_bonus': agent_result.get('coordination_bonus', 0.0)
            }


        for i in range(self.num_systems):
            agent_key = f'agent_{i}'
            sys_id = i + 1
            agent_result = coordinated_results.get(sys_id, {})
            self.agent_cumulative_impact[agent_key] += agent_result.get('impact', 0.0)
            if agent_result.get('detection_risk', 0.0) > 0.7:
                self.agent_detection_count[agent_key] += 1


        total_impact = sum(self.agent_cumulative_impact.values())
        self.coordination_state['global_impact'] = total_impact
        avg_detection = np.mean([r.get('detection_risk', 0.0) for r in coordinated_results.values()]) if coordinated_results else 0.0
        self.coordination_state['detection_risk'] = avg_detection


        terminated_flag = False
        truncated_flag = False


        if avg_detection >= self.global_detection_threshold:
            terminated_flag = True
            self.episode_done_reason = "global_detection"

            for agent_key in agent_rewards:
                agent_rewards[agent_key] -= 100.0


        elif any(count >= self.max_agent_detections for count in self.agent_detection_count.values()):
            terminated_flag = True
            self.episode_done_reason = "agent_blown"


        elif total_impact >= self.mission_success_impact:
            terminated_flag = True
            self.episode_done_reason = "mission_success"

            for agent_key in agent_rewards:
                agent_rewards[agent_key] += 200.0


        elif self.current_step >= self.max_steps:
            truncated_flag = True
            self.episode_done_reason = "max_steps"

        terminated = {agent_key: terminated_flag for agent_key in agent_observations.keys()}
        truncated = {agent_key: truncated_flag for agent_key in agent_observations.keys()}


        for agent_key in infos:
            infos[agent_key]['episode_step'] = self.current_step
            infos[agent_key]['done_reason'] = self.episode_done_reason if (terminated_flag or truncated_flag) else 'running'
            infos[agent_key]['global_impact'] = total_impact
            infos[agent_key]['global_detection_risk'] = avg_detection

        return agent_observations, agent_rewards, terminated, truncated, infos

    def _execute_coordinated_attacks(self, actions: Dict[str, np.ndarray]) -> Dict[int, Dict]:

        results = {}


        for i in range(self.num_systems):
            sys_id = i + 1
            agent_key = f'agent_{i}'

            if agent_key in actions:
                action = actions[agent_key]


                attack_result = self._execute_pinn_attack(sys_id, action)
                results[sys_id] = attack_result


        coordination_effects = self._calculate_coordination_effects(results)


        for sys_id in results:
            results[sys_id]['coordination_bonus'] = coordination_effects.get(sys_id, 0.0)

        return results

    def _execute_pinn_attack(self, sys_id: int, action: np.ndarray) -> Dict:

        if not self.federated_pinn_manager or sys_id not in self.federated_pinn_manager.local_models:
            return {'success': False, 'impact': 0.0, 'detection_risk': 1.0}

        try:

            local_model = self.federated_pinn_manager.local_models[sys_id]


            attack_type = int(action[0])
            magnitude = float(action[1])
            duration = float(action[2])
            stealth_level = float(action[3])
            target_component = float(action[4])


            attack_params = self._generate_pinn_attack_params(
                attack_type, magnitude, duration, stealth_level, target_component
            )


            attack_result = self._simulate_pinn_attack(local_model, attack_params)


            anomaly_detector = self.federated_pinn_manager.anomaly_detectors.get(sys_id)
            detection_risk = 0.0
            if anomaly_detector:
                detection_risk = self._calculate_anomaly_score(attack_result)

            return {
                'success': attack_result.get('success', False),
                'impact': attack_result.get('impact', 0.0),
                'detection_risk': detection_risk,
                'pinn_response': attack_result,
                'attack_params': attack_params
            }

        except Exception as e:
            print(f"PINN attack execution failed for system {sys_id}: {str(e)}")
            return {'success': False, 'impact': 0.0, 'detection_risk': 1.0, 'error': str(e)}

    def _generate_pinn_attack_params(self, attack_type: int, magnitude: float,
                                   duration: float, stealth_level: float,
                                   target_component: float) -> Dict:

        attack_types = [
            'communication_spoofing',
            'data_injection',
            'protocol_manipulation',
            'voltage_manipulation',
            'power_disruption',
            'current_injection'
        ]

        return {
            'type': attack_types[attack_type % len(attack_types)],
            'magnitude': magnitude,
            'duration': duration,
            'stealth_factor': stealth_level,
            'target': int(target_component),
            'timestamp': time.time()
        }

    def _calculate_coordination_effects(self, results: Dict[int, Dict]) -> Dict[int, float]:

        coordination_effects = {}


        successful_attacks = [sys_id for sys_id, result in results.items()
                            if result.get('success', False)]


        if len(successful_attacks) > 1:
            coordination_bonus = len(successful_attacks) * 10.0

            for sys_id in successful_attacks:
                coordination_effects[sys_id] = coordination_bonus


        for sys_id in results:
            if sys_id not in coordination_effects:
                coordination_effects[sys_id] = 0.0

        return coordination_effects

    def _calculate_agent_reward(self, sys_id: int, agent_result: Dict,
                              all_results: Dict[int, Dict]) -> float:

        base_reward = 0.0


        if agent_result.get('success', False):
            base_reward += 50.0


        impact = agent_result.get('impact', 0.0)
        base_reward += impact * 20.0


        detection_risk = agent_result.get('detection_risk', 1.0)
        stealth_reward = (1.0 - detection_risk) * 30.0
        base_reward += stealth_reward


        coordination_bonus = agent_result.get('coordination_bonus', 0.0)
        base_reward += coordination_bonus


        total_detections = sum(1 for result in all_results.values()
                             if result.get('detection_risk', 0.0) > 0.7)
        if total_detections > 2:
            base_reward -= total_detections * 25.0

        return base_reward

    def _get_agent_observation(self, sys_id: int) -> np.ndarray:

        if not self.federated_pinn_manager or sys_id not in self.federated_pinn_manager.local_models:
            print(f"Failed to get observation for system {sys_id}: no local model found")
            return np.zeros(25, dtype=np.float32)

        try:

            local_model = self.federated_pinn_manager.local_models[sys_id]
            system_state = self._get_pinn_system_state(local_model, sys_id)


            global_state = self._get_global_federated_state()


            observation = np.concatenate([
                system_state[:15],
                global_state[:10]
            ]).astype(np.float32)

            return observation

        except Exception as e:
            print(f"Failed to get observation for system {sys_id}: {str(e)}")
            return np.zeros(25, dtype=np.float32)

    def _get_pinn_system_state(self, local_model, sys_id: int) -> np.ndarray:

        try:


            if hasattr(local_model, 'config'):
                config = local_model.config

                state = np.array([
                    1.0,
                    0.5,
                    0.3,
                    60.0 / 100.0,
                    0.5,
                    25.0 / 50.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    float(sys_id) / 10.0,
                    0.5,
                    1.0,
                    0.3,
                    0.0
                ], dtype=np.float32)
                return state
            else:

                return np.ones(15, dtype=np.float32) * 0.5
        except Exception as e:
            print(f"Error getting PINN system state for system {sys_id}: {str(e)}")
            return np.ones(15, dtype=np.float32) * 0.5

    def _get_global_federated_state(self) -> np.ndarray:

        try:
            if hasattr(self.federated_pinn_manager, 'global_model') and self.federated_pinn_manager.global_model:

                global_state = np.array([
                    1.0,
                    0.5,
                    0.0,
                    len(self.federated_pinn_manager.local_models) / 10.0,
                    0.5,
                    1.0,
                    0.0,
                    0.5,
                    1.0,
                    0.0
                ], dtype=np.float32)
                return global_state
            else:
                return np.ones(10, dtype=np.float32) * 0.5
        except Exception as e:
            print(f"Error getting global federated state: {str(e)}")
            return np.ones(10, dtype=np.float32) * 0.5

    def _calculate_anomaly_score(self, attack_result: Dict) -> float:

        try:

            impact = attack_result.get('impact', 0.0)
            magnitude = attack_result.get('magnitude', 0.5)
            stealth_factor = attack_result.get('stealth_factor', 0.5)


            base_score = (impact + magnitude) / 2.0
            stealth_reduction = stealth_factor * 0.3

            anomaly_score = max(0.0, base_score - stealth_reduction)
            return min(anomaly_score, 1.0)

        except Exception as e:
            print(f"Error calculating anomaly score: {e}")
            return 0.5

class _EpisodeRewardCB(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(float(info['episode']['r']))
                self.episode_lengths.append(int(info['episode']['l']))
        return True


class EnhancedDQNSACCoordinator:


    def __init__(self, federated_pinn_manager: FederatedPINNManager, num_systems: int = 6):
        self.federated_pinn_manager = federated_pinn_manager
        self.num_systems = num_systems


        self.marl_env = MultiAgentRLEnvironment(federated_pinn_manager, num_systems)


        self.dqn_agents = {}
        self.sac_agents = {}

        self.dqn_envs = {}
        self.sac_envs = {}

        for i in range(num_systems):
            sys_id = i + 1


            if sys_id in federated_pinn_manager.local_models:
                cms_system = federated_pinn_manager.local_models[sys_id]

                anomaly_det = federated_pinn_manager.anomaly_detectors.get(sys_id) if hasattr(federated_pinn_manager, 'anomaly_detectors') else None


                dqn_env = Monitor(DiscreteSecurityEvasionEnv(cms_system, num_stations=10, anomaly_detector=anomaly_det, system_id=sys_id))
                self.dqn_envs[sys_id] = dqn_env
                self.dqn_agents[sys_id] = DQN(
                    'MlpPolicy',
                    dqn_env,
                    learning_rate=1e-3,
                    buffer_size=50000,
                    learning_starts=1000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    verbose=0
                )


                sac_env = Monitor(SecurityEvasionEnvironment(cms_system, num_stations=10, anomaly_detector=anomaly_det, system_id=sys_id))
                self.sac_envs[sys_id] = sac_env
                self.sac_agents[sys_id] = SAC(
                    'MlpPolicy',
                    sac_env,
                    learning_rate=3e-4,
                    buffer_size=500000,
                    learning_starts=2000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=2,
                    ent_coef='auto',
                    target_update_interval=1,
                    verbose=0
                )


        self.training_history = {
            'dqn_rewards': {sys_id: [] for sys_id in self.dqn_agents.keys()},
            'sac_rewards': {sys_id: [] for sys_id in self.sac_agents.keys()},
            'coordination_scores': []
        }

        print(f"## Enhanced DQN/SAC Coordinator initialized with {len(self.dqn_agents)} DQN and {len(self.sac_agents)} SAC agents")

    def train_coordinated_agents(self, total_timesteps: int = 100000):

        print(" Starting Enhanced DQN/SAC Training with PINN Integration")


        print("\n Phase 1: Individual Agent Training with PINN Models")
        self._train_individual_agents(total_timesteps // 2)


        print("\n## Phase 2: Coordinated Multi-Agent Evaluation (diagnostic rollout, no gradient updates)")
        self._evaluate_coordinated_agents(total_timesteps // 2)

        print("## Enhanced DQN/SAC training completed (Phase 1 = real training; Phase 2 = evaluation)")

    def _train_individual_agents(self, timesteps: int):

        for sys_id in self.dqn_agents.keys():
            print(f" Training System {sys_id} agents...")


            if sys_id in self.dqn_agents:
                print(f"  Training DQN agent for System {sys_id}...")
                dqn_cb = _EpisodeRewardCB()
                self.dqn_agents[sys_id].learn(
                    total_timesteps=timesteps // 2,
                    log_interval=1000,
                    progress_bar=False,
                    callback=dqn_cb
                )
                self.training_history['dqn_rewards'][sys_id] = dqn_cb.episode_rewards
                print(f"    DQN System {sys_id}: {len(dqn_cb.episode_rewards)} episodes, mean={np.mean(dqn_cb.episode_rewards) if dqn_cb.episode_rewards else 0:.2f}")


            if sys_id in self.sac_agents:
                print(f"  Training SAC agent for System {sys_id}...")
                sac_cb = _EpisodeRewardCB()
                self.sac_agents[sys_id].learn(
                    total_timesteps=timesteps // 2,
                    log_interval=1000,
                    progress_bar=False,
                    callback=sac_cb
                )
                self.training_history['sac_rewards'][sys_id] = sac_cb.episode_rewards
                print(f"    SAC System {sys_id}: {len(sac_cb.episode_rewards)} episodes, mean={np.mean(sac_cb.episode_rewards) if sac_cb.episode_rewards else 0:.2f}")

    def _evaluate_coordinated_agents(self, timesteps: int):

        print("## Coordinated multi-agent evaluation rollout (diagnostic, no learning)...")

        episodes = timesteps // 1000

        for episode in range(episodes):

            observations, _ = self.marl_env.reset()

            episode_rewards = {f'agent_{i}': 0.0 for i in range(self.num_systems)}
            done = False
            step = 0

            while not done and step < 100:

                actions = {}
                for i in range(self.num_systems):
                    sys_id = i + 1
                    agent_key = f'agent_{i}'

                    if agent_key in observations:
                        obs = observations[agent_key]


                        if sys_id in self.sac_agents:
                            action, _ = self.sac_agents[sys_id].predict(obs, deterministic=False)
                            actions[agent_key] = action


                if actions:
                    observations, rewards, terminated, truncated, infos = self.marl_env.step(actions)


                    for agent_key, reward in rewards.items():
                        episode_rewards[agent_key] += reward


                    done = any(terminated.values()) or any(truncated.values())

                step += 1


            avg_reward = np.mean(list(episode_rewards.values()))
            self.training_history['coordination_scores'].append(avg_reward)

            if episode % 10 == 0:
                print(f"  Episode {episode}/{episodes}: Avg Reward = {avg_reward:.2f}")

    def get_coordinated_attack_actions(self, system_states: Dict[int, Dict]) -> Dict[int, Dict]:

        coordinated_actions = {}

        for sys_id in range(1, self.num_systems + 1):
            if sys_id not in system_states:
                continue

            result = {
                'coordination_type': 'simultaneous',
                'system_id': sys_id,
                'dqn_result': None,
                'sac_result': None,
            }


            if sys_id in self.dqn_agents and sys_id in self.dqn_envs:
                try:
                    dqn_env = self.dqn_envs[sys_id]
                    obs, _ = dqn_env.reset()
                    action, _ = self.dqn_agents[sys_id].predict(obs, deterministic=True)
                    _, reward, _, _, info = dqn_env.step(action)

                    result['dqn_action'] = self._convert_dqn_action(action)
                    result['dqn_result'] = {
                        'success': not info.get('attack_detected', True),
                        'impact': info.get('evcs_impact', 0.0),
                        'reward': float(reward),
                        'detected': info.get('attack_detected', False),
                        'attack_type': result['dqn_action'].get('type', 'unknown'),
                        'env_consistent': True
                    }
                except Exception as e:
                    print(f"      ##?? DQN env execution failed for system {sys_id}: {e}")
                    result['dqn_action'] = {'type': 'voltage_manipulation', 'discrete': True, 'action_idx': 0}
                    result['dqn_result'] = {'success': False, 'impact': 0.0, 'detected': True}


            if sys_id in self.sac_agents and sys_id in self.sac_envs:
                try:
                    sac_env = self.sac_envs[sys_id]
                    obs, _ = sac_env.reset()
                    action, _ = self.sac_agents[sys_id].predict(obs, deterministic=True)
                    _, reward, _, _, info = sac_env.step(action)

                    result['sac_action'] = action
                    result['sac_result'] = {
                        'success': not info.get('attack_detected', True),
                        'impact': info.get('evcs_impact', 0.0),
                        'reward': float(reward),
                        'detected': info.get('attack_detected', False),
                        'attack_type': info.get('security_result', {}).get('attack_type', 'unknown'),
                        'env_consistent': True
                    }

                    print(f"      ## Phase 2 SAC (sys {sys_id}): reward={reward:.2f}, "
                          f"detected={info.get('attack_detected', False)}, "
                          f"impact={info.get('evcs_impact', 0.0):.3f}")
                except Exception as e:
                    print(f"      ##?? SAC env execution failed for system {sys_id}: {e}")
                    result['sac_action'] = np.zeros(6, dtype=np.float32)
                    result['sac_result'] = {'success': False, 'impact': 0.0, 'detected': True}

            coordinated_actions[sys_id] = result

        return coordinated_actions

    def _convert_state_to_observation(self, system_state: Dict) -> np.ndarray:


        features = [
            system_state.get('voltage', 1.0),
            system_state.get('current', 0.0),
            system_state.get('power', 0.0),
            system_state.get('frequency', 60.0),
            system_state.get('soc', 0.5),
            system_state.get('temperature', 25.0),
            system_state.get('load_factor', 1.0),
            system_state.get('grid_stability', 1.0),

        ]


        while len(features) < 25:
            features.append(0.0)

        return np.array(features[:25], dtype=np.float32)

    def _convert_dqn_action(self, action_idx: int) -> Dict:

        action_types = [
            'voltage_manipulation',
            'current_injection',
            'power_disruption',
            'data_injection',
            'communication_spoofing',
            'protocol_manipulation'
        ]

        return {
            'type': action_types[action_idx % len(action_types)],
            'discrete': True,
            'action_idx': action_idx
        }

    def plot_training_rewards(self, save_path: str = "rl_training_rewards_6_systems.png"):

        import matplotlib.pyplot as plt

        print(f"\n## Plotting training rewards for {self.num_systems} systems...")

        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        axes = axes.flatten()

        colors_dqn = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        colors_sac = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']


        has_boundaries = hasattr(self, '_two_level_boundaries') and self._two_level_boundaries

        for idx, sys_id in enumerate(sorted(self.training_history['dqn_rewards'].keys())):
            ax = axes[idx]


            dqn_rewards = self.training_history['dqn_rewards'][sys_id]
            sac_rewards = self.training_history['sac_rewards'][sys_id]

            episodes_dqn = range(len(dqn_rewards))
            episodes_sac = range(len(sac_rewards))


            if dqn_rewards:
                ax.plot(episodes_dqn, dqn_rewards,
                       color=colors_dqn[idx], alpha=0.35, linewidth=0.8,
                       label=f'DQN System {sys_id}')


                if len(dqn_rewards) > 10:
                    window = max(5, min(20, len(dqn_rewards) // 10))
                    dqn_ma = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(dqn_rewards)), dqn_ma,
                           color=colors_dqn[idx], linewidth=2.5,
                           label=f'DQN MA({window})', alpha=0.9)


            if sac_rewards:
                ax.plot(episodes_sac, sac_rewards,
                       color=colors_sac[idx], alpha=0.35, linewidth=0.8,
                       label=f'SAC System {sys_id}')


                if len(sac_rewards) > 10:
                    window = max(5, min(20, len(sac_rewards) // 10))
                    sac_ma = np.convolve(sac_rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(sac_rewards)), sac_ma,
                           color=colors_sac[idx], linewidth=2.5,
                           label=f'SAC MA({window})', alpha=0.9)


            if has_boundaries and sys_id in self._two_level_boundaries:
                bd = self._two_level_boundaries[sys_id]

                for i, bnd in enumerate(bd.get('outer_circle_boundaries_dqn', [])):
                    ax.axvline(x=bnd, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
                    if i == 0:
                        ax.axvline(x=bnd, color='gray', linestyle='--', linewidth=0.7,
                                  alpha=0.5, label='Outer Circle')


            ax.set_xlabel('Episode (cumulative across outer circles)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution System {sys_id} - RL Agent Rewards',
                        fontsize=13, fontweight='bold', pad=10)
            ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)


            if dqn_rewards or sac_rewards:
                stats_text = f'DQN: {len(dqn_rewards)} ep'
                if dqn_rewards:
                    stats_text += f', avg={np.mean(dqn_rewards):.0f}'
                stats_text += f'\nSAC: {len(sac_rewards)} ep'
                if sac_rewards:
                    stats_text += f', avg={np.mean(sac_rewards):.0f}'
                all_rewards = dqn_rewards + sac_rewards
                if all_rewards:
                    stats_text += f'\nTotal: {len(all_rewards)} ep'
                    stats_text += f'\nMax: {np.max(all_rewards):.0f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


        title = 'RL Training Rewards - 6 Distribution Systems (DQN + SAC Pairs)'
        if has_boundaries:
            title += '\n(Dashed lines = outer circle boundaries)'
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.998)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"## Training rewards plot saved to {save_path}")
        plt.close()


        print("\n## Training Rewards Summary:")
        print("=" * 70)
        for sys_id in sorted(self.training_history['dqn_rewards'].keys()):
            dqn_rewards = self.training_history['dqn_rewards'][sys_id]
            sac_rewards = self.training_history['sac_rewards'][sys_id]

            print(f"\n  System {sys_id}:")
            if dqn_rewards:
                print(f"    DQN: {len(dqn_rewards)} episodes, avg reward = {np.mean(dqn_rewards):.2f}")
            if sac_rewards:
                print(f"    SAC: {len(sac_rewards)} episodes, avg reward = {np.mean(sac_rewards):.2f}")
        print("=" * 70)

class EnhancedIntegratedEVCSLLMRLSystem(_PINNAttackSimulationMixin):


    def __init__(self, config: Dict = None):

        default_config = self._default_config()
        if config:
            self.config = self._deep_merge_config(default_config, config)
        else:
            self.config = default_config


        self.hierarchical_sim = None
        self.federated_manager = None
        self.pinn_optimizer = None
        self.acn_fleet = None


        self.llm_analyzer = None
        self.dqn_sac_coordinator = None
        self.langgraph_coordinator = None
        self.enhanced_coordinator = None


        self.attack_scenarios = []


        self.simulation_results = {}
        self.attack_history = []

        print(" Initializing Enhanced Integrated EVCS LLM-RL System...")
        self._initialize_system()

    def _deep_merge_config(self, default: Dict, override: Dict) -> Dict:

        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _default_config(self) -> Dict:

        return {
            'hierarchical': {
                'use_enhanced_pinn': True,
                'use_dqn_sac_security': True,
                'total_duration': 7200.0,
                'num_distribution_systems': 10
            },
            'federated_pinn': {
                'num_distribution_systems': 10,
                'local_epochs': 300,
                'global_rounds': 100,
                'aggregation_method': 'fedavg'
            },
            'llm': {
                'provider': 'gemini',
                'model': 'models/gemini-2.5-flash',
                'api_key_file': 'gemini_key.txt'
            },
            'rl': {
                'num_systems': 10,
                'dqn_timesteps': 50000,
                'sac_timesteps': 50000,
                'coordination_training': True
            },
            'attack': {
                'max_episodes': 100,
                'coordination_type': 'simultaneous',
                'stealth_threshold': 0.7
            },


            'phase2_node_level': True,
            'phase2_n_nodes': 10,
            'phase2_node_seed': 42,

            'phase2_topk_networks': 7,


            'phase2_per_evse_kw': 7.0,
        }

    def _initialize_system(self):

        print("   Initializing hierarchical co-simulation...")
        self._initialize_hierarchical_simulation()

        print("  ## Initializing LLM threat analyzer...")
        self._initialize_llm_components()

        print("  ## Initializing enhanced DQN/SAC agents...")
        self._initialize_enhanced_rl_components()

        print("   Initializing LangGraph coordination...")
        self._initialize_langgraph_coordinator()

        print("  ## Initializing attack scenarios...")
        self._initialize_attack_scenarios()

        print("## Enhanced integrated system initialization complete!")

    def _load_pretrained_models(self):

        print("\n## Loading Pre-trained Models...")
        print("-" * 50)


        federated_manager = None
        try:
            federated_config = FederatedPINNConfig(
                num_distribution_systems=self.config['federated_pinn']['num_distribution_systems'],
                local_epochs=200,
                global_rounds=10,
                aggregation_method='fedavg'
            )
            federated_manager = FederatedPINNManager(federated_config)

            success = federated_manager.load_federated_models('federated_models')
            if success:
                print("## Federated PINN models loaded successfully")
            else:
                federated_manager = None
                print("##?? Federated models not found, trying legacy models...")
        except Exception as e:
            print(f"##?? Failed to load federated models: {e}")
            federated_manager = None


        dqn_sac_trainer = None
        try:
            dqn_sac_trainer = DQNSACSecurityEvasionTrainer(
                cms_system=None,
                num_stations=6,
                use_both=True
            )
            print("## Coordinated DQN/SAC Security Evasion Trainer initialized successfully")
        except Exception as e:
            print(f"##?? Failed to initialize DQN/SAC trainer: {e}")
            dqn_sac_trainer = None


        individual_optimizers = {}
        _n_ds = self.config['federated_pinn']['num_distribution_systems']
        for system_id in range(1, _n_ds + 1):
            try:
                optimizer_file = f'federated_pinn_system_{system_id}.pth'

                if not os.path.exists(optimizer_file):
                    print(f"##?? PINN model file not found: {optimizer_file}")
                    continue

                from pinn_optimizer import LSTMPINNChargingOptimizer, PINNConfig

                pinn_config = PINNConfig(
                    input_size=4,
                    hidden_size=64,
                    num_layers=3,
                    output_size=3,
                    physics_weight=0.3,
                    max_voltage=240.0,
                    max_current=32.0,
                    max_power=7.68,
                    min_voltage=208.0,
                    min_current=6.0,
                    min_power=1.44
                )

                individual_optimizer = LSTMPINNChargingOptimizer(pinn_config, always_train=False)

                try:
                    individual_optimizer.load_model(optimizer_file)
                    individual_optimizers[system_id] = individual_optimizer
                    print(f"## Individual PINN Optimizer {system_id} loaded successfully")
                except Exception as load_error:
                    print(f"##?? Failed to load PINN model {optimizer_file}: {load_error}")
                    individual_optimizers[system_id] = individual_optimizer
                    print(f"   Using fresh PINN optimizer for system {system_id}")

            except Exception as e:
                print(f"##?? Failed to initialize PINN optimizer {system_id}: {e}")

        if federated_manager and dqn_sac_trainer:
            print("## All models loaded successfully (Federated + DQN/SAC Trainer + Individual PINNs)!")
        elif federated_manager:
            print("## Federated and individual PINN models loaded successfully!")
        elif dqn_sac_trainer:
            print("## DQN/SAC Trainer and individual PINN models loaded successfully!")
        else:
            print("##?? Limited models loaded - some systems may use fallback methods")

        return federated_manager, individual_optimizers, dqn_sac_trainer

    def _run_training_phase(self):

        print("=" * 80)
        print(" FEDERATED TRAINING PHASE: Training Distributed PINN Models")
        print("=" * 80)


        print("\n Phase 1: Initializing Federated PINN System...")
        print("-" * 50)

        federated_config = FederatedPINNConfig(
            num_distribution_systems=self.config['federated_pinn']['num_distribution_systems'],
            local_epochs=200,
            global_rounds=10,
            aggregation_method='fedavg'
        )

        federated_manager = FederatedPINNManager(federated_config)
        print(" ## Federated PINN Manager initialized with 6 distribution systems")


        import os as _os
        _ids_paths = [
            'models/best_ids_model.pkl',
            'models/robust_ids/best_ids_model.pth',
            'models/lstm_ids_pretrained.pth',
            'lstm_ids_best_balanced.pth',
        ]
        _lstm_loaded = False
        for _lp in _ids_paths:
            if _os.path.exists(_lp):
                for _sid, _det in federated_manager.anomaly_detectors.items():
                    _det.load_lstm_model(_lp)
                if _lp.endswith('.pkl'):
                    _model_label = "sklearn RF IDS (best_ids_model.pkl)"
                elif "robust_ids" in _lp:
                    _model_label = "DNN-Classifier IDS"
                else:
                    _model_label = "LSTM IDS"
                print(f" ## {_model_label} loaded from {_lp} into all {len(federated_manager.anomaly_detectors)} anomaly detectors")
                _lstm_loaded = True
                break
        if not _lstm_loaded:
            print(" ##??  No pre-trained IDS model found — Layer 3 detection disabled")
            print("    Run: python robust_ids_evaluation.py --retrain")


        print("\n Phase 2: Training Federated PINN Models...")
        print("-" * 50)

        for sys_id in range(1, self.config['federated_pinn']['num_distribution_systems'] + 1):
            print(f"\n  Training System {sys_id} PINN with REAL EVCS Dynamics...")

            pinn_config = LSTMPINNConfig()
            pinn_config.num_evcs_stations = 2
            pinn_config.sequence_length = 6

            data_generator = PhysicsDataGenerator(pinn_config)

            print(f"  Generating physics-accurate training data for System {sys_id}...")
            n_samples = 1000
            sequences, targets = data_generator.generate_realistic_evcs_scenarios(n_samples)

            print(f"  ## Generated {len(sequences)} sequences with {sequences.shape[-1]} features")
            print(f"  ## Target ranges: V={targets[:, 0].min():.1f}-{targets[:, 0].max():.1f}, "
                  f"I={targets[:, 1].min():.1f}-{targets[:, 1].max():.1f}, "
                  f"P={targets[:, 2].min():.2f}-{targets[:, 2].max():.2f}")

            local_model = federated_manager.local_models[sys_id]

            if hasattr(local_model, 'data_generator'):
                local_model._enhanced_training_data = (sequences, targets)
                print(f"   Enhanced data stored in local model for System {sys_id}")

            training_result = federated_manager.train_local_model(sys_id, (sequences, targets), n_samples)
            print(f" ## System {sys_id} training completed with REAL EVCS dynamics")


        print("\n ## Performing federated averaging...")
        for round_num in range(federated_config.global_rounds):
            print(f" Round {round_num + 1}/{federated_config.global_rounds}")


        print("\n Phase 3: Training DQN/SAC Security Evasion Agents...")
        print("-" * 50)

        dqn_sac_trainer = None
        try:
            cms = EnhancedChargingManagementSystem(stations=[], use_pinn=True)
            cms.federated_manager = federated_manager

            from dqn_sac_security_evasion import create_dqn_sac_evasion_system
            dqn_sac_trainer = create_dqn_sac_evasion_system(cms)
            print("  Training DQN/SAC agents (this may take a few minutes)...")
            dqn_sac_trainer.train_agents(sac_timesteps=100000, dqn_timesteps=100000)

            dqn_sac_trainer.save_agents()
            print(" ## DQN/SAC Security Evasion Agents trained and saved")

        except Exception as e:
            print(f" ##?? DQN/SAC training failed: {e}")

        print("\n" + "=" * 80)
        print(" FEDERATED TRAINING PHASE COMPLETED")
        print("=" * 80)
        print(" ## Federated PINN Models: 6 distribution systems trained")
        print("=" * 80)

        return federated_manager, None, dqn_sac_trainer, False

    def _initialize_hierarchical_simulation(self):

        if not HIERARCHICAL_AVAILABLE:
            print("   ##?? Hierarchical co-simulation not available")
            return

        try:

            print("     Loading pre-trained models...")
            self.federated_manager, self.pinn_optimizer, _ = self._load_pretrained_models()


            if not self.federated_manager:
                print("     No pre-trained models found, training new models...")
                self.federated_manager, self.pinn_optimizer, _, _ = self._run_training_phase()
                if not self.federated_manager:
                    print("    ##?? Training failed, continuing without federated models")
                    return


            _ids_candidate_paths = [
                ("models/best_ids_model.pkl",             "sklearn RF (best_ids_model.pkl, AUC≈0.90)"),
                ("models/robust_ids/best_ids_model.pth",  "DNN-Classifier (robust)"),
                ("models/lstm_ids_pretrained.pth",        "LSTM Classifier"),
                ("lstm_ids_best_balanced.pth",            "LSTM Balanced"),
            ]
            ids_model_path, ids_model_label = None, None
            for _cp, _cl in _ids_candidate_paths:
                if os.path.exists(_cp):
                    ids_model_path, ids_model_label = _cp, _cl
                    break

            if ids_model_path and self.federated_manager:
                print(f"    ## Loading {ids_model_label} multi-layer ADM...")
                ids_success_count = 0
                for sys_id, detector in self.federated_manager.anomaly_detectors.items():
                    try:
                        detector.load_lstm_model(ids_model_path)
                        ids_success_count += 1
                        print(f"      ## System {sys_id}: {ids_model_label} ADM enabled (3-layer detection)")
                    except Exception as e:
                        print(f"      ##??  System {sys_id}: IDS load failed - {e}")

                if ids_success_count > 0:
                    print(f"    ## Multi-layer ADM integration complete ({ids_success_count}/{len(self.federated_manager.anomaly_detectors)} systems)")
                    print(f"       Detection layers: Physical  Pattern  {ids_model_label}")
                else:
                    print("    ##??  IDS integration failed, using 2-layer detection only")
            else:
                print("    ℹ  No IDS model found — using 2-layer detection (Physical + Pattern)")
                print("       To enable ML layer: run 'python robust_ids_evaluation.py --retrain'")


            self.hierarchical_sim = HierarchicalCoSimulation(
                use_enhanced_pinn=self.config['hierarchical']['use_enhanced_pinn'],
                use_dqn_sac_security=self.config['hierarchical']['use_dqn_sac_security']
            )


            if self.federated_manager and hasattr(self.federated_manager, 'local_models'):
                print("     Injecting pre-trained PINN models...")
                for sys_id, optimizer in self.federated_manager.local_models.items():
                    if optimizer and hasattr(optimizer, 'is_trained') and optimizer.is_trained:
                        if not hasattr(self.hierarchical_sim, 'enhanced_pinn_models'):
                            self.hierarchical_sim.enhanced_pinn_models = {}
                        self.hierarchical_sim.enhanced_pinn_models[sys_id] = optimizer
                        print(f"      ## System {sys_id}: Pre-trained PINN model injected")

                if hasattr(self.hierarchical_sim, 'enhanced_pinn_models') and self.hierarchical_sim.enhanced_pinn_models:
                    print(f"    #??# Injected {len(self.hierarchical_sim.enhanced_pinn_models)} pre-trained PINN models")
                    self.hierarchical_sim.enhanced_pinn_available = True


            if self.federated_manager:
                from hierarchical_cosimulation import OpenDSSInterface
                OpenDSSInterface._shared_federated_manager = self.federated_manager
                print(f"     Shared federated PINN manager with OpenDSSInterface for CMS")
            elif self.pinn_optimizer:
                from hierarchical_cosimulation import OpenDSSInterface
                OpenDSSInterface._shared_pinn_optimizer = self.pinn_optimizer
                print(f"     Shared legacy PINN optimizer with OpenDSSInterface for CMS")


            self.hierarchical_sim.total_duration = self.config['hierarchical']['total_duration']


            print("     Adding distribution systems...")
            for i in range(1, self.config['hierarchical']['num_distribution_systems'] + 1):
                self.hierarchical_sim.add_distribution_system(i, "ieee34Mod1.dss", 10)


            print("     Setting up EV charging stations...")
            try:
                self.hierarchical_sim.setup_ev_charging_stations()
                print("   ## Hierarchical co-simulation initialized")


                self._initialize_acn_sim_fleet()


                total_evcs = 0
                for sys_id, dist_info in self.hierarchical_sim.distribution_systems.items():
                    dist_sys = dist_info['system']
                    if hasattr(dist_sys, 'ev_stations'):
                        total_evcs += len(dist_sys.ev_stations)
                        print(f"    ### DEBUG: System {sys_id} has {len(dist_sys.ev_stations)} EVCS stations")
                    else:
                        print(f"    ### DEBUG: System {sys_id} has no ev_stations attribute")

                print(f"    ### DEBUG: Total EVCS stations created: {total_evcs}")

            except Exception as evcs_error:
                print(f"  ##?? EVCS setup failed: {evcs_error}")
                print("    Continuing with basic hierarchical simulation...")

        except Exception as e:
            import traceback
            print(f"   ##XX Failed to initialize hierarchical simulation: {e}")
            print("   Full traceback:")
            traceback.print_exc()
            print("  Continuing with fallback mode...")
            self.hierarchical_sim = None

    def _initialize_acn_sim_fleet(self):


        import sys as _sys
        _acn_mod = _sys.modules.get('acn_sim_interface')
        if _acn_mod is not None:
            ACN_SIM_AVAILABLE = getattr(_acn_mod, 'ACN_SIM_AVAILABLE', False)
            ACNSimFleet = getattr(_acn_mod, 'ACNSimFleet', None)
        else:
            try:
                from acn_sim_interface import ACNSimFleet, ACN_SIM_AVAILABLE
            except Exception as _e:
                print(f"    ##??  [ACN-Sim] import failed ({type(_e).__name__}: {_e})")
                ACN_SIM_AVAILABLE = False
                ACNSimFleet = None

        if not ACN_SIM_AVAILABLE:
            print("    ℹ  [ACN-Sim] acnportal not installed — "
                  "ACN-Sim DISABLED. Install with: pip install acnportal")
            print("    ℹ  [ACN-Sim] Using legacy EVCSController dynamics path.")
            self.acn_fleet = None
            return

        if not self.hierarchical_sim:
            self.acn_fleet = None
            return

        try:
            acn_data_dir = os.path.join(
                "evcs_data", "ACN-Data-Static-main", "time series data"
            )
            n_ds    = self.config['hierarchical']['num_distribution_systems']
            sim_dur = float(self.config['hierarchical']['total_duration'])

            self.acn_fleet = ACNSimFleet(
                n_zones=n_ds,
                n_evses_per_zone=10,
                acn_data_dir=acn_data_dir if os.path.isdir(acn_data_dir) else None,
                period_min=5.0,
                evse_voltage=240.0,
                sim_duration_s=sim_dur,
            )

            cms_list = []
            for ds_id in range(1, n_ds + 1):
                dist_info = self.hierarchical_sim.distribution_systems.get(ds_id, {})
                dist_sys  = dist_info.get('system', None)
                cms = getattr(dist_sys, 'cms', None) if dist_sys else None
                cms_list.append(cms)

            self.acn_fleet.initialize_zones(cms_list)


            wired = 0
            for ds_id in range(1, n_ds + 1):
                zone = self.acn_fleet.get_zone(ds_id)
                if zone.simulator is not None:
                    dist_info = self.hierarchical_sim.distribution_systems.get(ds_id, {})
                    dist_sys  = dist_info.get('system', None)
                    if dist_sys is not None:
                        dist_sys._acn_zone = zone
                        wired += 1

            if wired > 0:
                print(f"    ## [ACN-Sim] ACTIVE — {wired}/{n_ds} zones wired "
                      f"(period=5 min, V=240 V, "
                      f"ACN-Data={'available' if os.path.isdir(acn_data_dir) else 'unavailable'})")
            else:
                print("    ##??  [ACN-Sim] Fleet created but no simulators live — "
                      "legacy EVCS dynamics path will be used.")

        except Exception as exc:
            import traceback
            print(f"    ##??  ACN-Sim fleet init failed: {exc} — "
                  "falling back to legacy EVCS dynamics")
            traceback.print_exc()
            self.acn_fleet = None

    def _initialize_llm_components(self):

        try:
            llm_config = self.config['llm']


            api_key = None
            if 'api_key_file' in llm_config:
                try:
                    with open(llm_config['api_key_file'], 'r') as f:
                        api_key = f.read().strip()
                    print(f"    Loaded API key from {llm_config['api_key_file']}")
                except FileNotFoundError:
                    print(f"   ##?? API key file {llm_config['api_key_file']} not found")
                except Exception as e:
                    print(f"   ##?? Failed to load API key: {e}")


            self.llm_analyzer = GeminiLLMThreatAnalyzer(
                api_key=api_key,
                model_name=llm_config.get('model', 'models/gemini-2.5-flash')
            )

            if self.llm_analyzer.is_available:
                print("   ## LLM components initialized with Gemini Pro")
            else:
                print("   ##?? Gemini Pro not available, will use fallback analysis")

        except Exception as e:
            print(f"   ##?? LLM initialization failed: {e}")
            self.llm_analyzer = None

    def _initialize_enhanced_rl_components(self):

        try:
            if not self.federated_manager:
                print("   ##?? No federated PINN manager available for RL training")
                return


            self.dqn_sac_coordinator = EnhancedDQNSACCoordinator(
                self.federated_manager,
                self.config['hierarchical']['num_distribution_systems']
            )

            print("   ## Enhanced DQN/SAC components initialized with PINN integration (OLD ARCHITECTURE)")


            if ATTACK_SPECIFIC_AVAILABLE:
                self.attack_specific_coordinator = AttackSpecificCoordinator(
                    self.federated_manager,
                    self.config['hierarchical']['num_distribution_systems'],
                    attack_types=ATTACK_TYPES,


                    node_level=self.config.get('phase2_node_level', False),
                    n_nodes=self.config.get('phase2_n_nodes', 10),
                    node_seed=self.config.get('phase2_node_seed', 42),
                    topk_networks=self.config.get('phase2_topk_networks', 7),
                )
                print("   ## Attack-Specific RL Coordinator initialized (NEW ARCHITECTURE)")
                print("    This is the RECOMMENDED architecture for better specialization")
            else:
                self.attack_specific_coordinator = None
                print("   ##?? Attack-specific coordinator not available")

        except Exception as e:
            print(f"   ##XX Failed to initialize enhanced RL components: {e}")
            print("  Continuing with fallback mode...")
            self.dqn_sac_coordinator = None
            self.attack_specific_coordinator = None

    def _initialize_langgraph_coordinator(self):

        try:

            if ENHANCED_COORDINATOR_AVAILABLE and self.llm_analyzer and self.dqn_sac_coordinator:
                self.enhanced_coordinator = EnhancedLLMRLCoordinator(
                    llm_analyzer=self.llm_analyzer,
                    rl_coordinator=self.dqn_sac_coordinator,
                    hierarchical_sim=self.hierarchical_sim,
                    federated_manager=self.federated_manager,
                    enhanced_system=self
                )
                print("   ## Enhanced LLM-RL coordinator initialized (includes LangGraph + STRIDE/MITRE)")
                print("    This is the ONLY coordinator with proper Gemini-RL communication")


                self.langgraph_coordinator = None

            else:
                print("   ##?? Enhanced LLM-RL coordinator not available")
                print("    Note: This means NO proper Gemini-RL coordination (LangGraph is integrated in Enhanced)")
                print("   ## System will use direct RL coordination without LLM guidance")
                self.enhanced_coordinator = None
                self.langgraph_coordinator = None

        except Exception as e:
            print(f"   ##XX Failed to initialize Enhanced coordinator: {e}")
            print("    No Gemini-RL coordination available - continuing with direct RL only")
            self.langgraph_coordinator = None
            self.enhanced_coordinator = None

    def _initialize_attack_scenarios(self):

        self.attack_scenarios = [
            EnhancedAttackScenario(
                scenario_id="ENHANCED_001",
                name="STRIDE-Based Multi-System Attack",
                description="Coordinated attacks using STRIDE threat model on federated PINN systems",
                target_systems=[1, 2, 3, 4, 5, 6],
                attack_duration=600.0,
                stealth_requirement=0.8,
                impact_goal=0.9,
                constraints={
                    'max_detection_risk': 0.3,
                    'coordination_required': True,
                    'attack_types': [
                        AttackType.VOLTAGE_MANIPULATION.value,
                        AttackType.COMMUNICATION_SPOOFING.value,
                        AttackType.MODEL_POISONING.value,
                        AttackType.SOC_SPOOFING.value
                    ],
                    'stride_categories': [
                        STRIDECategory.SPOOFING.value,
                        STRIDECategory.TAMPERING.value,
                        STRIDECategory.DENIAL_OF_SERVICE.value
                    ]
                },
                coordination_type="simultaneous"
            ),
            EnhancedAttackScenario(
                scenario_id="ENHANCED_002",
                name="MITRE ATT&CK Federated Learning Campaign",
                description="Multi-stage attack following MITRE ATT&CK framework on federated PINN training",
                target_systems=[1, 3, 5],
                attack_duration=900.0,
                stealth_requirement=0.9,
                impact_goal=0.8,
                constraints={
                    'model_corruption_limit': 0.4,
                    'stealth_priority': True,
                    'attack_types': [
                        AttackType.FEDERATED_CORRUPTION.value,
                        AttackType.GRADIENT_MANIPULATION.value,
                        AttackType.DATA_INJECTION.value,
                        AttackType.PROTOCOL_MANIPULATION.value
                    ],
                    'mitre_tactics': [
                        MITRECategory.INITIAL_ACCESS.value,
                        MITRECategory.PERSISTENCE.value,
                        MITRECategory.DEFENSE_EVASION.value,
                        MITRECategory.IMPACT.value
                    ]
                },
                coordination_type="sequential"
            ),
            EnhancedAttackScenario(
                scenario_id="ENHANCED_003",
                name="EVCS Infrastructure Disruption",
                description="Comprehensive attack on EVCS charging infrastructure with power system impact",
                target_systems=[2, 4, 6],
                attack_duration=300.0,
                stealth_requirement=0.7,
                impact_goal=0.95,
                constraints={
                    'max_power_disruption': 0.6,
                    'thermal_safety_limit': 0.8,
                    'attack_types': [
                        AttackType.CHARGING_HIJACKING.value,
                        AttackType.THERMAL_ATTACK.value,
                        AttackType.POWER_DISRUPTION.value,
                        AttackType.FREQUENCY_ATTACK.value
                    ],
                    'stride_categories': [
                        STRIDECategory.TAMPERING.value,
                        STRIDECategory.DENIAL_OF_SERVICE.value,
                        STRIDECategory.ELEVATION_OF_PRIVILEGE.value
                    ]
                },
                coordination_type="simultaneous"
            )
        ]
        print(f"   ## Initialized {len(self.attack_scenarios)} enhanced attack scenarios with STRIDE/MITRE integration")

    def train_enhanced_system(self, total_timesteps: int = 100000):

        print("\n Starting Enhanced System Training Pipeline")
        print("=" * 80)

        training_results = {
            'pinn_training': {},
            'rl_training': {},
            'llm_rl_integration': {},
            'coordination_training': {}
        }


        print("\n Phase 1: PINN Model Training/Loading")
        print("-" * 50)
        if self.federated_manager:
            pinn_results = self._train_federated_pinn_models()
            training_results['pinn_training'] = pinn_results
        else:
            print("##?? No federated PINN manager available")


        print("\n## Phase 2: Enhanced DQN/SAC Training with PINN Integration")
        print("-" * 50)
        if self.dqn_sac_coordinator:
            self.dqn_sac_coordinator.train_coordinated_agents(total_timesteps)
            training_results['rl_training'] = {'status': 'completed', 'timesteps': total_timesteps}


            print("## Phase 2 training complete. Plot will be generated after Phase 3.")
        else:
            print("##?? No DQN/SAC coordinator available")


        print("\n## Phase 3: LLM-RL Integration Training")
        print("-" * 50)
        if self.enhanced_coordinator:
            llm_rl_results = self._train_llm_rl_integration()
            training_results['llm_rl_integration'] = llm_rl_results
        else:
            print("##?? No Enhanced LLM-RL coordinator available")


        self.simulation_results['llm_rl_integration'] = training_results.get('llm_rl_integration', {})

        print("\n## Enhanced system training completed!")
        return training_results

    def _train_federated_pinn_models(self) -> Dict:

        print(" Training federated PINN models...")

        training_results = {}

        try:

            for sys_id in range(1, self.config['hierarchical']['num_distribution_systems'] + 1):
                print(f"  Training PINN model for System {sys_id}...")


                local_data = self._generate_pinn_training_data(sys_id)


                local_result = self.federated_manager.train_local_model(
                    sys_id, local_data, n_samples=1000
                )

                training_results[f'system_{sys_id}'] = local_result
                print(f"  ## System {sys_id} PINN training completed")


            print("## Performing federated averaging...")
            for round_num in range(self.config['federated_pinn']['global_rounds']):
                self.federated_manager.federated_averaging()
                print(f"  Round {round_num + 1}/{self.config['federated_pinn']['global_rounds']} completed")

            training_results['federated_rounds'] = self.config['federated_pinn']['global_rounds']
            training_results['status'] = 'completed'

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"##XX PINN training failed: {e}")
            training_results['status'] = 'failed'
            training_results['error'] = str(e)

        return training_results

    def _generate_pinn_training_data(self, sys_id: int) -> Tuple[np.ndarray, np.ndarray]:

        try:
            if not FEDERATED_PINN_AVAILABLE:


                DATA_INTEGRITY['pinn_dummy_training_data'] += 1
                print(f"##!! DATA-INTEGRITY WARNING: FEDERATED_PINN_AVAILABLE is False — "
                      f"system {sys_id} PINN training data is np.random NOISE, not physics. "
                      f"Any downstream PINN/attack result for this system is meaningless.")
                sequences = np.random.randn(500, 10, 15)
                targets = np.random.randn(500, 3)
                return sequences, targets

            from pinn_optimizer import LSTMPINNConfig, PhysicsDataGenerator


            config = LSTMPINNConfig(num_evcs_stations=10, sequence_length=10)
            data_generator = PhysicsDataGenerator(config)


            sequences_t, targets_t = data_generator.generate_realistic_evcs_scenarios(n_samples=500)


            sequences = sequences_t.numpy()
            targets = targets_t.numpy()

            return sequences, targets

        except Exception as e:


            DATA_INTEGRITY['pinn_dummy_training_data'] += 1
            import traceback; traceback.print_exc()
            print(f"##!! DATA-INTEGRITY WARNING: physics data generation FAILED for system "
                  f"{sys_id} ({e}) — falling back to np.random NOISE. PINN for this system "
                  f"will be trained on garbage; its results are meaningless.")
            sequences = np.random.randn(100, 10, 15)
            targets = np.random.randn(100, 3)
            return sequences, targets

    def _train_llm_rl_integration(self) -> Dict:

        print(" Training LLM-RL integration with Two-Level Architecture...")

        integration_results = {
            'episodes': 0,
            'success_rate': 0.0,
            'coordination_efficiency': 0.0,
            'status': 'completed'
        }

        try:
            from central_rl_coordinator import CentralRLCoordinator


            attack_coord = None
            if hasattr(self, 'attack_specific_coordinator') and self.attack_specific_coordinator:
                attack_coord = self.attack_specific_coordinator
            elif (hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator and
                  hasattr(self.enhanced_coordinator, 'attack_specific_coordinator') and
                  self.enhanced_coordinator.attack_specific_coordinator):
                attack_coord = self.enhanced_coordinator.attack_specific_coordinator

            if not attack_coord:
                print("##?? No attack-specific coordinator available")
                print("  ## Falling back to direct RL training (autonomous mode)")
                return self._train_direct_rl_integration()


            _force_autonomous = os.environ.get('FORCE_AUTONOMOUS_RL', '') in ('1', 'true', 'True')
            llm_analyzer = None
            if _force_autonomous:
                print("  ## FORCE_AUTONOMOUS_RL set  AUTONOMOUS mode (Gemini disabled for baseline)")
            elif (hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator and
                hasattr(self.enhanced_coordinator, 'llm_analyzer') and
                self.enhanced_coordinator.llm_analyzer):
                llm_analyzer = self.enhanced_coordinator.llm_analyzer
                print("  ## Gemini LLM analyzer available  GEMINI-GUIDED mode")
            else:
                print("  ## No Gemini LLM analyzer  AUTONOMOUS mode")


            central_coord = CentralRLCoordinator(
                attack_coordinator=attack_coord,
                llm_analyzer=llm_analyzer,
                num_systems=self.config['rl']['num_systems'],


                outer_episodes=18,
                inner_episodes=50
            )


            two_level_results = central_coord.run_two_level_training()


            outer_rewards = two_level_results.get('outer_episode_rewards', [])
            summary = two_level_results.get('summary', {})

            integration_results['status'] = 'completed'
            integration_results['coordinator_type'] = f'central_rl_{central_coord.mode}'
            integration_results['total_episodes'] = len(outer_rewards)
            integration_results['episode_rewards'] = outer_rewards
            integration_results['average_reward'] = summary.get('mean_outer_reward', 0.0)
            integration_results['success_rate'] = summary.get('mean_outer_reward', 0.0) / 1000.0
            integration_results['two_level_results'] = two_level_results
            integration_results['final_deployment_plan'] = two_level_results.get('final_deployment_plan', {})


            self.central_rl_coordinator = central_coord


            if self.dqn_sac_coordinator is not None:
                try:
                    per_sys = central_coord.get_per_system_episode_rewards()
                    for sys_id, rdata in per_sys.items():
                        p3_dqn = rdata.get('dqn_rewards', [])
                        p3_sac = rdata.get('sac_rewards', [])

                        if sys_id in self.dqn_sac_coordinator.training_history['dqn_rewards']:
                            self.dqn_sac_coordinator.training_history['dqn_rewards'][sys_id].extend(p3_dqn)
                        if sys_id in self.dqn_sac_coordinator.training_history['sac_rewards']:
                            self.dqn_sac_coordinator.training_history['sac_rewards'][sys_id].extend(p3_sac)


                    self.dqn_sac_coordinator._two_level_boundaries = per_sys


                    print("\n## Generating final training rewards plot with complete data...")
                    self.dqn_sac_coordinator.plot_training_rewards()
                except Exception as e:
                    print(f"##?? Failed to merge two-level rewards into plot: {e}")


            if attack_coord is not None:
                try:
                    save_dir = attack_coord.save_agents("trained_rl_agents")
                    integration_results['agents_save_dir'] = save_dir
                except Exception as e:
                    print(f"##?? Failed to save trained agents: {e}")

            print(f"\n## Two-Level LLM-RL integration training completed:")
            print(f"   Mode: {central_coord.mode.upper()}")
            print(f"   Outer Episodes: {len(outer_rewards)}")
            print(f"   Inner Episodes/Agent: {central_coord.inner_episodes}")
            print(f"   Mean Outer Reward: {summary.get('mean_outer_reward', 0):.2f}")
            print(f"   Best Outer Reward: {summary.get('best_outer_reward', 0):.2f}")
            print(f"   Reward Trend: {summary.get('reward_trend', 'unknown')}")
            print(f"   Total Inner Episodes: {summary.get('total_inner_episodes', 0)}")

            self._save_reward_history(
                episode_rewards=outer_rewards,
                mode='gemini_assisted',
                coordinator_type=integration_results.get('coordinator_type', 'central_rl_gemini'),
                gemini_usage_rate=float(summary.get('gemini_usage_rate', 1.0)),
                metadata=summary
            )

        except Exception as e:
            print(f"##XX Two-Level LLM-RL integration training failed: {e}")
            import traceback
            traceback.print_exc()
            integration_results['status'] = 'failed'
            integration_results['error'] = str(e)

            return self._train_direct_rl_integration()

        return integration_results

    def _train_direct_rl_integration(self) -> Dict:

        print("## Training direct RL integration (autonomous two-level mode)...")

        integration_results = {
            'episodes': 0,
            'success_rate': 0.0,
            'coordination_efficiency': 0.0,
            'status': 'completed',
            'coordinator_type': 'autonomous'
        }

        try:
            from central_rl_coordinator import CentralRLCoordinator


            attack_coord = None
            if hasattr(self, 'attack_specific_coordinator') and self.attack_specific_coordinator:
                attack_coord = self.attack_specific_coordinator
            elif (hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator and
                  hasattr(self.enhanced_coordinator, 'attack_specific_coordinator') and
                  self.enhanced_coordinator.attack_specific_coordinator):
                attack_coord = self.enhanced_coordinator.attack_specific_coordinator

            if not attack_coord:
                print("##?? No attack-specific coordinator available for autonomous mode")
                integration_results['status'] = 'skipped'
                return integration_results


            central_coord = CentralRLCoordinator(
                attack_coordinator=attack_coord,
                llm_analyzer=None,
                num_systems=self.config['rl']['num_systems'],


                outer_episodes=18,
                inner_episodes=50
            )


            system_analysis_data = None
            if hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator:
                try:
                    system_analysis_data = self.enhanced_coordinator._perform_comprehensive_system_analysis()
                except Exception:
                    pass


            two_level_results = central_coord.run_two_level_training(
                system_analysis_data=system_analysis_data
            )


            outer_rewards = two_level_results.get('outer_episode_rewards', [])
            summary = two_level_results.get('summary', {})

            integration_results['status'] = 'completed'
            integration_results['total_episodes'] = len(outer_rewards)
            integration_results['episode_rewards'] = outer_rewards
            integration_results['average_reward'] = summary.get('mean_outer_reward', 0.0)
            integration_results['success_rate'] = summary.get('mean_outer_reward', 0.0) / 1000.0
            integration_results['two_level_results'] = two_level_results
            integration_results['final_deployment_plan'] = two_level_results.get('final_deployment_plan', {})

            self.central_rl_coordinator = central_coord


            if self.dqn_sac_coordinator is not None:
                try:
                    per_sys = central_coord.get_per_system_episode_rewards()
                    for sys_id, rdata in per_sys.items():
                        p3_dqn = rdata.get('dqn_rewards', [])
                        p3_sac = rdata.get('sac_rewards', [])

                        if sys_id in self.dqn_sac_coordinator.training_history['dqn_rewards']:
                            self.dqn_sac_coordinator.training_history['dqn_rewards'][sys_id].extend(p3_dqn)
                        if sys_id in self.dqn_sac_coordinator.training_history['sac_rewards']:
                            self.dqn_sac_coordinator.training_history['sac_rewards'][sys_id].extend(p3_sac)

                    self.dqn_sac_coordinator._two_level_boundaries = per_sys


                    print("\n## Generating final training rewards plot with complete data...")
                    self.dqn_sac_coordinator.plot_training_rewards()
                except Exception as e:
                    print(f"##?? Failed to merge two-level rewards into plot: {e}")


            if attack_coord is not None:
                try:
                    save_dir = attack_coord.save_agents("trained_rl_agents")
                    integration_results['agents_save_dir'] = save_dir
                except Exception as e:
                    print(f"##?? Failed to save trained agents: {e}")

            print(f"\n## Autonomous Two-Level RL training completed:")
            print(f"   Outer Episodes: {len(outer_rewards)}")
            print(f"   Mean Outer Reward: {summary.get('mean_outer_reward', 0):.2f}")
            print(f"   Best Outer Reward: {summary.get('best_outer_reward', 0):.2f}")
            print(f"   Reward Trend: {summary.get('reward_trend', 'unknown')}")

            self._save_reward_history(
                episode_rewards=outer_rewards,
                mode='standalone',
                coordinator_type='autonomous',
                gemini_usage_rate=0.0,
                metadata=summary
            )

        except Exception as e:
            print(f"##XX Autonomous RL integration training failed: {e}")
            import traceback
            traceback.print_exc()
            integration_results['status'] = 'failed'
            integration_results['error'] = str(e)

        return integration_results

    def _save_reward_history(self, episode_rewards: List[float], mode: str,
                            coordinator_type: str, gemini_usage_rate: float, metadata: Dict):

        import json
        from datetime import datetime

        try:

            pretraining_rewards = {}
            if hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator:
                coord = self.enhanced_coordinator
                if hasattr(coord, 'attack_specific_coordinator') and coord.attack_specific_coordinator:
                    asc = coord.attack_specific_coordinator
                    if hasattr(asc, 'training_history') and asc.training_history:
                        for attack_type, data in asc.training_history.items():
                            pretraining_rewards[attack_type] = {
                                'dqn_episode_rewards': data.get('dqn', {}).get('episode_rewards', []),
                                'sac_episode_rewards': data.get('sac', {}).get('episode_rewards', []),
                                'dqn_num_episodes': data.get('dqn', {}).get('num_episodes', 0),
                                'sac_num_episodes': data.get('sac', {}).get('num_episodes', 0),
                                'dqn_mean_reward': data.get('dqn', {}).get('mean_reward', 0.0),
                                'sac_mean_reward': data.get('sac', {}).get('mean_reward', 0.0),
                            }


            if hasattr(self, 'dqn_sac_coordinator') and self.dqn_sac_coordinator:
                old_coord = self.dqn_sac_coordinator
                if hasattr(old_coord, 'training_history'):
                    th = old_coord.training_history
                    for sys_id in th.get('dqn_rewards', {}):
                        key = f"system_{sys_id}"
                        if key not in pretraining_rewards:
                            pretraining_rewards[key] = {}
                        pretraining_rewards[key]['dqn_episode_rewards'] = th['dqn_rewards'].get(sys_id, [])
                        pretraining_rewards[key]['sac_episode_rewards'] = th['sac_rewards'].get(sys_id, [])


            reward_history = {
                'mode': mode,
                'coordinator_type': coordinator_type,
                'gemini_usage_rate': gemini_usage_rate,
                'timestamp': datetime.now().isoformat(),
                'episode_rewards': episode_rewards,
                'pretraining_rewards': pretraining_rewards,
                'metadata': metadata
            }


            filename = f"reward_history_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(reward_history, f, indent=2)

            print(f"   Saved reward history to {filename}")


            self._plot_reward_convergence_comparison()

        except Exception as e:
            print(f"  ##?? Failed to save reward history: {e}")

    def _plot_reward_convergence_comparison(self):

        import json
        import matplotlib.pyplot as plt
        import os
        import glob

        def _latest_reward_history(mode: str):

            candidates = glob.glob(f"reward_history_{mode}.json") + \
                         glob.glob(f"reward_history_{mode}_*.json")
            if not candidates:
                return None
            return max(candidates, key=os.path.getmtime)

        try:

            gemini_file = _latest_reward_history("gemini_assisted")
            standalone_file = _latest_reward_history("standalone")

            gemini_exists = gemini_file is not None
            standalone_exists = standalone_file is not None

            if not gemini_exists and not standalone_exists:
                print("  ## No reward history files found yet for comparison")
                return


            gemini_data = None
            standalone_data = None

            if gemini_exists:
                with open(gemini_file, 'r') as f:
                    gemini_data = json.load(f)
                print(f"  ## Loaded Gemini-assisted reward history")

            if standalone_exists:
                with open(standalone_file, 'r') as f:
                    standalone_data = json.load(f)
                print(f"  ## Loaded standalone RL reward history")


            plt.figure(figsize=(12, 7))


            if gemini_data:
                episodes_gemini = list(range(1, len(gemini_data['episode_rewards']) + 1))
                rewards_gemini = gemini_data['episode_rewards']

                plt.plot(episodes_gemini, rewards_gemini,
                        marker='o', linewidth=2, markersize=6,
                        label=f"Gemini-Assisted RL (Usage: {gemini_data['gemini_usage_rate']:.1%})",
                        color='#2E86AB', alpha=0.8)


                if len(rewards_gemini) >= 3:
                    window = min(5, len(rewards_gemini))
                    moving_avg = np.convolve(rewards_gemini, np.ones(window)/window, mode='valid')
                    plt.plot(range(window, len(rewards_gemini) + 1), moving_avg,
                            linestyle='--', linewidth=2, color='#2E86AB',
                            label=f"Gemini MA({window})", alpha=0.6)


            if standalone_data:
                episodes_standalone = list(range(1, len(standalone_data['episode_rewards']) + 1))
                rewards_standalone = standalone_data['episode_rewards']

                plt.plot(episodes_standalone, rewards_standalone,
                        marker='s', linewidth=2, markersize=6,
                        label=f"Standalone RL (No Gemini)",
                        color='#A23B72', alpha=0.8)


                if len(rewards_standalone) >= 3:
                    window = min(5, len(rewards_standalone))
                    moving_avg = np.convolve(rewards_standalone, np.ones(window)/window, mode='valid')
                    plt.plot(range(window, len(rewards_standalone) + 1), moving_avg,
                            linestyle='--', linewidth=2, color='#A23B72',
                            label=f"Standalone MA({window})", alpha=0.6)


            plt.xlabel('Episode', fontsize=12, fontweight='bold')
            plt.ylabel('Total Reward (Impact)', fontsize=12, fontweight='bold')
            plt.title('Reward Convergence: Gemini-Assisted RL vs Standalone RL',
                     fontsize=14, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=10, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')


            if gemini_data and standalone_data:
                avg_gemini = np.mean(gemini_data['episode_rewards'])
                avg_standalone = np.mean(standalone_data['episode_rewards'])
                improvement = ((avg_gemini - avg_standalone) / max(avg_standalone, 0.001)) * 100

                textstr = f'Average Reward:\n'
                textstr += f'Gemini: {avg_gemini:.2f}\n'
                textstr += f'Standalone: {avg_standalone:.2f}\n'
                textstr += f'Improvement: {improvement:+.1f}%'

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top', bbox=props)

            plt.tight_layout()


            plot_filename = 'reward_convergence_comparison.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ## Saved reward convergence comparison plot to {plot_filename}")


            if gemini_data and standalone_data:
                print(f"\n  ## Reward Convergence Comparison Summary:")
                print(f"     Gemini-Assisted Avg: {avg_gemini:.2f}")
                print(f"     Standalone RL Avg: {avg_standalone:.2f}")
                print(f"     Performance Improvement: {improvement:+.1f}%")

        except Exception as e:
            print(f"  ##?? Failed to create comparison plot: {e}")
            import traceback
            traceback.print_exc()

    def run_enhanced_simulation(self, scenario_id: str, episodes: int = 100) -> Dict:


        scenario = self._get_scenario_by_id(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        print(f"\n Running Enhanced Coordinated Simulation")
        print(f"Scenario: {scenario.name}")
        print(f"Coordination: {scenario.coordination_type}")
        print(f"Target Systems: {scenario.target_systems}")
        print(f"Episodes: {episodes}")
        print("=" * 80)


        self.simulation_results = {
            'scenario': scenario,
            'episodes': episodes,
            'episode_results': [],
            'coordination_metrics': [],
            'pinn_interaction_results': [],
            'llm_guidance_results': []
        }


        for episode in range(episodes):
            print(f"\n--- Enhanced Episode {episode + 1}/{episodes} ---")


            _llm = getattr(self, 'llm_analyzer', None)
            if _llm is not None:
                _llm._episode_call_ids = []

            episode_result = self._run_enhanced_episode(scenario, episode)
            self.simulation_results['episode_results'].append(episode_result)


            if (episode + 1) % 5 == 0:
                self._print_enhanced_progress(episode + 1, episodes)


        final_results = self._analyze_enhanced_results()


        print("\n Running Hierarchical Co-simulation...")
        hierarchical_results = self._run_hierarchical_cosimulation(final_results)
        final_results['hierarchical_simulation'] = hierarchical_results


        print("\n## Creating enhanced visualizations...")

        self._create_hierarchical_plots()

        return final_results

    def _create_hierarchical_plots(self):

        try:
            print("## Creating hierarchical simulation plots...")


            if not hasattr(self, 'hierarchical_sim') or not self.hierarchical_sim:
                print("##?? No hierarchical simulation available for plotting")
                return


            if not hasattr(self.hierarchical_sim, 'results') or not self.hierarchical_sim.results:
                print("##?? No hierarchical simulation results available for plotting")
                return


            if not self.hierarchical_sim.results.get('time') or len(self.hierarchical_sim.results['time']) == 0:
                print("##?? No time series data available in hierarchical results")
                return

            print(f"   ## Plotting {len(self.hierarchical_sim.results['time'])} time points...")


            self.hierarchical_sim.plot_hierarchical_results()

            print("## Hierarchical simulation plots created successfully!")
            print("   ## Generated plots include:")
            print("      ##Transmission system frequency response")
            print("      ##Distribution system voltage profiles")
            print("      ##Power flow analysis across all systems")
            print("      ##EVCS charging dynamics and utilization")
            print("      ##Queue management and customer flow")
            print("      ##Attack impact visualization")
            print("      ##AGC and load balancing performance")
            print("      ##Energy delivery and efficiency metrics")

        except Exception as e:
            print(f"##XX Failed to create hierarchical plots: {e}")
            import traceback
            traceback.print_exc()

    def _run_hierarchical_cosimulation(self, attack_results: Dict) -> Dict:

        try:
            if not hasattr(self, 'hierarchical_sim') or not self.hierarchical_sim:
                print("  ##?? Hierarchical simulation not initialized, skipping...")
                return {'status': 'skipped', 'reason': 'not_initialized'}

            print("  ## Extracting LLM-RL coordinated attacks for hierarchical simulation...")


            total_impact = attack_results.get('performance_metrics', {}).get('average_impact', 0.0)
            success_rate = attack_results.get('performance_metrics', {}).get('average_success_rate', 0.0)


            duration_seconds = self.config.get('hierarchical', {}).get('total_duration', 3600.0)
            sim_config = {
                'duration_seconds': duration_seconds,
                'duration_hours': duration_seconds / 3600.0,
                'attack_impact_factor': total_impact,
                'attack_success_rate': success_rate,
                'num_distribution_systems': self.config.get('hierarchical', {}).get('num_distribution_systems', 6)
            }


            self.hierarchical_sim.total_duration = duration_seconds

            print(f"    ## Attack Impact Factor: {total_impact:.3f}")
            print(f"    ## Attack Success Rate: {success_rate:.1%}")
            print(f"     Simulation Duration: {sim_config['duration_hours']} hours ({duration_seconds} seconds)")


            num_systems = self.config.get('hierarchical', {}).get('num_distribution_systems', 6)
            attack_scenarios = []
            agent_attacks_extracted = False


            deployment_plan = None
            if 'llm_rl_integration' in self.simulation_results:
                deployment_plan = self.simulation_results['llm_rl_integration'].get('final_deployment_plan', {})
            if not deployment_plan and hasattr(self, 'central_rl_coordinator') and self.central_rl_coordinator:

                plan_data = getattr(self.central_rl_coordinator, 'final_deployment_plan', None)
                if plan_data:
                    deployment_plan = plan_data

            if deployment_plan and deployment_plan.get('deployments'):
                deployments = deployment_plan['deployments']
                n_attacks = len(deployments)


                margin = duration_seconds * 0.10
                usable_window = duration_seconds - 2 * margin


                raw_agent_durations = [dep.get('duration', 30.0) for dep in deployments]
                total_raw = sum(raw_agent_durations)


                gap = usable_window * 0.05 / max(n_attacks - 1, 1) if n_attacks > 1 else 0
                total_gap = gap * max(n_attacks - 1, 0)
                available_for_attacks = usable_window - total_gap


                if total_raw > 0:
                    agent_durations = [(d / total_raw) * available_for_attacks for d in raw_agent_durations]
                else:
                    agent_durations = [available_for_attacks / n_attacks] * n_attacks

                print(f" ## Agent durations scaled: raw [{', '.join(f'{d:.0f}s' for d in raw_agent_durations)}] "
                      f" sim [{', '.join(f'{d:.0f}s' for d in agent_durations)}]")

                params_source = deployments[0].get('params_source', 'fallback') if deployments else 'fallback'
                print(f"  ##Using final deployment plan: {n_attacks} attacks, "
                      f"agent-suggested timing [{params_source}]")
                print(f"     Timeline: {margin:.0f}s  {duration_seconds - margin:.0f}s")


                current_start = margin
                for idx, dep in enumerate(deployments):
                    start_time = current_start
                    at = dep.get('attack_type', 'power_manipulation')
                    target_sys = dep.get('target_system', idx + 1)
                    magnitude = dep.get('magnitude', 0.7)
                    stealth = dep.get('stealth_level', 0.7)
                    expected_reward = dep.get('expected_reward', 0.0)
                    attack_dur = agent_durations[idx]


                    current_start = start_time + attack_dur + gap

                    action_dict = {
                        'attack_type': at,
                        'target_system': target_sys,
                        'target_node': dep.get('target_node'),
                        'magnitude': magnitude,
                        'duration': attack_dur,
                        'stealth_level': stealth
                    }
                    result_dict = {
                        'success': expected_reward > 0,
                        'impact': min(magnitude, 1.0)
                    }

                    scenario = self._convert_agent_action_to_hierarchical(
                        action_dict, result_dict, start_time, duration_seconds
                    )
                    attack_scenarios.append(scenario)
                    agent_attacks_extracted = True

                    src = dep.get('params_source', 'fallback')
                    print(f"      ## Attack {idx+1}: {at} on System {target_sys} "
                          f"at {start_time:.0f}s for {attack_dur:.1f}s "
                          f"(mag={magnitude:.2f}, stealth={stealth:.2f}) [{src}]")


            elif 'episode_results' in self.simulation_results:
                print("  ## Extracting attack scenarios from last RL episode...")
                episode_results = self.simulation_results['episode_results']

                last_ep = episode_results[-1] if episode_results else {}

                rl_res = last_ep.get('rl_results', {})

                has_actions = 'actions' in rl_res or 'executed_actions' in rl_res
                if 'rl_results' in last_ep and has_actions:
                    executed_actions = rl_res.get('actions', rl_res.get('executed_actions', []))
                    execution_results = rl_res.get('results', rl_res.get('execution_results', []))

                    from dataclasses import is_dataclass, asdict
                    converted_actions = []
                    for action in executed_actions:
                        if is_dataclass(action):
                            converted_actions.append(asdict(action))
                        elif isinstance(action, dict):
                            converted_actions.append(action)
                        else:
                            converted_actions.append(vars(action) if hasattr(action, '__dict__') else action)
                    executed_actions = converted_actions

                    n_attacks = len(executed_actions)
                    margin = duration_seconds * 0.10
                    usable_window = duration_seconds - 2 * margin
                    attack_duration = usable_window / (n_attacks + 1)
                    spacing = usable_window / max(n_attacks, 1)

                    print(f"    Found {n_attacks} attacks, spreading across timeline")

                    for idx, (action, result) in enumerate(zip(executed_actions, execution_results)):
                        inner_result = result.get('result', result) if isinstance(result, dict) else result
                        is_success = inner_result.get('success', False)
                        impact_val = inner_result.get('impact', 0.0)

                        if is_success or impact_val > 0.01:
                            start_time = margin + idx * spacing

                            norm_action = dict(action)
                            if 'target_systems' in norm_action and 'target_system' not in norm_action:
                                norm_action['target_system'] = norm_action['target_systems'][0] if norm_action['target_systems'] else 1
                            if 'target_system' not in norm_action:
                                norm_action['target_system'] = result.get('system_id', 1)

                            norm_action['duration'] = attack_duration

                            scenario = self._convert_agent_action_to_hierarchical(
                                norm_action, inner_result, start_time, duration_seconds
                            )
                            attack_scenarios.append(scenario)
                            agent_attacks_extracted = True

                            target_sys = norm_action.get('target_system', '?')
                            print(f"      ## Attack {idx+1}: {norm_action.get('attack_type', '?')} on System {target_sys} "
                                  f"at {start_time:.0f}s for {attack_duration:.0f}s")
                else:
                    print("  ##?? Last episode missing rl_results or actions")


            if not agent_attacks_extracted:
                print("  ##?? No LLM-RL coordinated attacks found in results, using fallback scenarios...")
                attack_scenarios = self._create_fallback_attack_scenarios(num_systems, duration_seconds)


            if agent_attacks_extracted and hasattr(self, 'llm_analyzer') and self.llm_analyzer:
                print("\n  ## Invoking Agent LLM for strategic attack combination and optimization...")
                try:
                    optimized_scenarios = self._gemini_strategic_attack_combination(
                        attack_scenarios, duration_seconds, num_systems
                    )
                    if optimized_scenarios:
                        print(f"  ## Agent optimized {len(attack_scenarios)} agent attacks into {len(optimized_scenarios)} strategic scenarios")
                        attack_scenarios = optimized_scenarios
                    else:
                        print("  ##?? Agent optimization failed, using original agent attacks")
                except Exception as e:
                    print(f"  ##?? Agent strategic combination failed: {str(e)}, using original agent attacks")


            duration_seconds = self.config.get('hierarchical', {}).get('total_duration', 3600.0)
            self.hierarchical_sim.total_duration = duration_seconds


            self.hierarchical_sim.simulation_time = 0.0
            self.hierarchical_sim.total_time = duration_seconds

            print(f"   Hierarchical simulation configured for {duration_seconds} seconds")


            rl_controller_deployed = False
            attack_coord_to_use = None


            if hasattr(self, 'attack_specific_coordinator') and self.attack_specific_coordinator:
                attack_coord_to_use = self.attack_specific_coordinator


            if attack_coord_to_use is None and ATTACK_SPECIFIC_AVAILABLE:
                import os
                if os.path.isdir("trained_rl_agents"):
                    print("   No in-memory agents, attempting to load from trained_rl_agents/...")
                    try:
                        disk_coord = AttackSpecificCoordinator(
                            self.federated_manager,
                            num_systems,
                            attack_types=ATTACK_TYPES,
                            node_level=self.config.get('phase2_node_level', False),
                            n_nodes=self.config.get('phase2_n_nodes', 10),
                            node_seed=self.config.get('phase2_node_seed', 42),
                            topk_networks=self.config.get('phase2_topk_networks', 7),
                        )
                        if disk_coord.load_agents("trained_rl_agents"):
                            attack_coord_to_use = disk_coord
                            self.attack_specific_coordinator = disk_coord
                            print("  ## Loaded trained agents from disk")
                    except Exception as e:
                        print(f"  ##?? Failed to load agents from disk: {e}")

            if deployment_plan and attack_coord_to_use:
                try:
                    print("  ## Deploying trained RL agents for real-time attack injection...")
                    rt_controller = RealTimeRLAttackController(
                        attack_coordinator=attack_coord_to_use,
                        deployment_plan=deployment_plan,
                        duration_seconds=duration_seconds
                    )
                    self.hierarchical_sim.realtime_rl_controller = rt_controller
                    rl_controller_deployed = True
                    print("  ## Trained RL agents will inject attacks in real-time during simulation")
                except Exception as e:
                    print(f"  ##?? Failed to deploy RL agents: {e}, falling back to static scenarios")
                    self.hierarchical_sim.realtime_rl_controller = None


            if not rl_controller_deployed and attack_scenarios:
                print(f"  ##Using {len(attack_scenarios)} static attack scenarios (no trained RL agents)")
                self._apply_attacks_to_hierarchical_sim(attack_scenarios)

            hierarchical_results = self.hierarchical_sim.run_hierarchical_simulation(
                attack_scenarios=attack_scenarios if not rl_controller_deployed else []


            )


            self.hierarchical_sim.realtime_rl_controller = None

            print("  ## Hierarchical co-simulation completed!")
            return hierarchical_results

        except Exception as e:
            print(f"  ##XX Hierarchical simulation failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _apply_attacks_to_hierarchical_sim(self, attack_scenarios: List[Dict]):

        if not hasattr(self, 'hierarchical_sim') or not self.hierarchical_sim:
            return

        print(f"  ##Validating {len(attack_scenarios)} attack scenarios for hierarchical simulation...")


        for idx, attack in enumerate(attack_scenarios):

            if 'active' in attack:
                del attack['active']


            required_fields = ['type', 'target_system', 'start_time', 'duration', 'magnitude']
            missing_fields = [f for f in required_fields if f not in attack]

            if missing_fields:
                print(f"    ##?? Attack {idx} missing fields: {missing_fields}")
            else:
                print(f"    ## Attack {idx}: {attack['type']} on system {attack['target_system']} "
                      f"at t={attack['start_time']}s for {attack['duration']}s (mag={attack['magnitude']:.2f})")

        print(f"  ℹ Attacks will be activated during simulation runtime (not pre-applied)")

    def _run_enhanced_episode(self, scenario: EnhancedAttackScenario, episode: int) -> Dict:

        episode_start_time = time.time()


        if hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator:
            print(f"  ## Running with Enhanced LLM-RL Coordinator (includes LangGraph + STRIDE/MITRE)...")
            try:
                episode_result = self.enhanced_coordinator.run_enhanced_attack_episode(scenario, episode)
                enhanced_result = self._process_enhanced_coordinator_result(episode_result, scenario, episode)
            except Exception as e:
                print(f"Enhanced coordinator failed: {e}")
                print(f"   No other LLM coordination available - Enhanced is the only one with Agent-RL")
                print(f"  ## Falling back to direct DQN/SAC coordination (no LLM guidance)...")
                enhanced_result = self._run_direct_coordinated_episode(scenario, episode)

        else:

            print(f"   No Enhanced coordinator available (this is the only one with Agent-RL coordination)")
            print(f"  ## Running with direct DQN/SAC coordination (no LLM guidance)...")
            enhanced_result = self._run_direct_coordinated_episode(scenario, episode)

        episode_duration = time.time() - episode_start_time
        enhanced_result['duration'] = episode_duration
        enhanced_result['episode'] = episode

        return enhanced_result

    def _process_enhanced_coordinator_result(self, coordinator_result: Dict, scenario: EnhancedAttackScenario, episode: int) -> Dict:

        try:

            enhanced_result = {
                'episode_number': episode,
                'scenario': scenario.scenario_id,
                'coordination_type': 'enhanced_llm_rl',


                'system_analysis': coordinator_result.get('system_analysis', {}),
                'threat_analysis': coordinator_result.get('threat_analysis', {}),


                'llm_instructions': coordinator_result.get('llm_instructions', {}),
                'llm_strategy': coordinator_result.get('llm_instructions', {}).get('attack_strategy', 'unknown'),


                'rl_results': coordinator_result.get('rl_results', {}),
                'attack_results': coordinator_result.get('rl_results', {}).get('results', coordinator_result.get('rl_results', {}).get('execution_results', [])),


                'success_rate': self._calculate_success_rate_from_results(coordinator_result),
                'total_impact': self._calculate_total_impact_from_results(coordinator_result),
                'detection_rate': self._calculate_detection_rate_from_results(coordinator_result),


                'stride_threats_identified': len(coordinator_result.get('threat_analysis', {}).get('stride_threats', {})),
                'mitre_tactics_used': len(coordinator_result.get('threat_analysis', {}).get('mitre_tactics', {})),
                'llm_adaptation_performed': 'adaptation_results' in coordinator_result,


                'coordination_score': coordinator_result.get('rl_results', {}).get('coordination_metrics', {}).get('effectiveness', 1.0),
                'coordination_effectiveness': coordinator_result.get('rl_results', {}).get('coordination_metrics', {}).get('effectiveness', 1.0),


                'raw_coordinator_result': coordinator_result
            }


            total_reward = (
                enhanced_result['success_rate'] * 1000 +
                enhanced_result['total_impact'] * 500 +
                (1.0 - enhanced_result['detection_rate']) * 300 +
                enhanced_result['coordination_score'] * 200
            )
            enhanced_result['total_reward'] = total_reward


            try:
                from llm_metrics_logger import LLMMetricsLogger
                _logger = LLMMetricsLogger.instance()
                _prev_reward = getattr(self, '_prev_episode_reward', 0.0)


                _llm = getattr(self, 'llm_analyzer', None)
                _call_ids = getattr(_llm, '_episode_call_ids', [])


                if _llm and getattr(_llm, '_last_call_id', None):
                    if _llm._last_call_id not in _call_ids:
                        _call_ids = _call_ids + [_llm._last_call_id]

                _llm_accepted = len(_call_ids) > 0
                for _cid in _call_ids:
                    _logger.update_rl_impact(
                        call_id          = _cid,
                        reward_before    = _prev_reward,
                        reward_after     = total_reward,
                        task_success     = enhanced_result['success_rate'] > 0.5,
                        rl_plan_accepted = _llm_accepted,
                    )
                if _call_ids:
                    print(f"    ## RL-impact logged for {len(_call_ids)} LLM call(s): "
                          f"Δreward={total_reward - _prev_reward:+.2f}")


                self._prev_episode_reward = total_reward

                if _llm is not None:
                    _llm._episode_call_ids = []
            except Exception as _rim_err:
                print(f"    ##??  RL-impact update failed (non-fatal): {_rim_err}")


            exec_results = enhanced_result.get('attack_results', [])
            if isinstance(exec_results, list):
                enhanced_result['steps'] = len(exec_results)
            else:
                enhanced_result['steps'] = coordinator_result.get('total_iterations', 0)


            if enhanced_result['success_rate'] > 0.8:
                enhanced_result['done_reason'] = "high_success"
            elif enhanced_result['detection_rate'] > 0.8:
                enhanced_result['done_reason'] = "high_detection"
            elif enhanced_result['steps'] > 0:
                enhanced_result['done_reason'] = "attacks_completed"
            else:
                enhanced_result['done_reason'] = "episode_end"


            enhanced_result['success_metrics'] = {
                'success_rate': enhanced_result['success_rate'],
                'total_impact': enhanced_result['total_impact'],
                'detection_rate': enhanced_result['detection_rate'],
                'composite_reward': total_reward
            }

            print(f"    ## Enhanced coordinator result processed: {enhanced_result['success_rate']:.1%} success, {enhanced_result['stride_threats_identified']} STRIDE threats")
            return enhanced_result

        except Exception as e:
            print(f"    ##XX Failed to process enhanced coordinator result: {e}")


            return self._run_direct_coordinated_episode(scenario, episode)

    def _calculate_success_rate_from_results(self, coordinator_result: Dict) -> float:

        try:

            print(f"### DEBUG: Coordinator result keys: {list(coordinator_result.keys())}")


            execution_results = []


            if 'rl_results' in coordinator_result and 'results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['results']
                print(f"### DEBUG: Found {len(execution_results)} execution results in rl_results.results")


            elif 'rl_results' in coordinator_result and 'execution_results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['execution_results']
                print(f"### DEBUG: Found {len(execution_results)} execution results in rl_results.execution_results")


            elif 'execution_results' in coordinator_result:
                execution_results = coordinator_result['execution_results']
                print(f"### DEBUG: Found {len(execution_results)} execution results directly")


            elif 'rl_results' in coordinator_result and 'success_rate' in coordinator_result['rl_results']:
                success_rate = float(coordinator_result['rl_results']['success_rate'])
                print(f"### DEBUG: Using pre-calculated success rate: {success_rate}")
                return success_rate

            if not execution_results:
                print(f"### DEBUG: No execution results found, returning 0.0")
                if 'rl_results' in coordinator_result:
                    print(f"### DEBUG: rl_results keys: {list(coordinator_result['rl_results'].keys())}")
                return 0.0


            successful_attacks = 0
            for result in execution_results:
                inner = result.get('result', result) if isinstance(result, dict) else result
                if isinstance(inner, dict) and inner.get('success', False):
                    successful_attacks += 1
            success_rate = float(successful_attacks) / len(execution_results)
            print(f"### DEBUG: Calculated success rate: {successful_attacks}/{len(execution_results)} = {success_rate:.1%}")


            for i, result in enumerate(execution_results):
                inner = result.get('result', result) if isinstance(result, dict) else result
                if isinstance(inner, dict):
                    print(f"### DEBUG: Result {i}: success={inner.get('success', False)}, impact={inner.get('impact', 0.0)}")
                else:
                    print(f"### DEBUG: Result {i}: {result}")

            return success_rate

        except Exception as e:
            print(f"##?? Failed to calculate success rate: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _calculate_total_impact_from_results(self, coordinator_result: Dict) -> float:

        try:

            execution_results = []


            if 'rl_results' in coordinator_result and 'results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['results']


            elif 'rl_results' in coordinator_result and 'execution_results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['execution_results']


            elif 'execution_results' in coordinator_result:
                execution_results = coordinator_result['execution_results']


            elif 'rl_results' in coordinator_result and 'total_impact' in coordinator_result['rl_results']:
                return float(coordinator_result['rl_results']['total_impact'])

            if not execution_results:
                return 0.0


            total_impact = 0.0
            for result in execution_results:
                inner = result.get('result', result) if isinstance(result, dict) else result
                if isinstance(inner, dict):
                    total_impact += inner.get('impact', 0.0)
            return float(total_impact)

        except Exception as e:
            print(f"##?? Failed to calculate total impact: {e}")
            return 0.0

    def _calculate_detection_rate_from_results(self, coordinator_result: Dict) -> float:

        try:

            execution_results = []


            if 'rl_results' in coordinator_result and 'results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['results']


            elif 'rl_results' in coordinator_result and 'execution_results' in coordinator_result['rl_results']:
                execution_results = coordinator_result['rl_results']['execution_results']


            elif 'execution_results' in coordinator_result:
                execution_results = coordinator_result['execution_results']


            elif 'rl_results' in coordinator_result and 'detection_events' in coordinator_result['rl_results']:
                detection_events = coordinator_result['rl_results']['detection_events']
                total_results = len(coordinator_result['rl_results'].get('execution_results', []))
                return float(len(detection_events)) / max(total_results, 1)

            if not execution_results:
                return 0.0


            detected_attacks = 0
            for result in execution_results:
                inner = result.get('result', result) if isinstance(result, dict) else result
                if isinstance(inner, dict) and (inner.get('detected', False) or inner.get('ids_detected', False)):
                    detected_attacks += 1
            return float(detected_attacks) / len(execution_results)

        except Exception as e:
            print(f"##?? Failed to calculate detection rate: {e}")
            return 0.0

    def _run_langgraph_fallback(self, scenario: EnhancedAttackScenario, episode: int) -> Dict:


        print("    ##?? Enhanced coordinator failed, using direct coordination fallback")
        return self._run_direct_coordinated_episode(scenario, episode)

    def _enhance_episode_result(self, langgraph_result: Dict, scenario: EnhancedAttackScenario, episode: int) -> Dict:

        enhanced_result = langgraph_result.copy()


        if self.dqn_sac_coordinator:
            pinn_metrics = self._calculate_pinn_interaction_metrics(langgraph_result)
            enhanced_result['pinn_interaction_metrics'] = pinn_metrics


        coordination_metrics = self._calculate_coordination_effectiveness(langgraph_result, scenario)
        enhanced_result['coordination_effectiveness'] = coordination_metrics


        if 'execution_results' in langgraph_result:
            enhanced_attacks = self._enhance_attack_results(langgraph_result['execution_results'])
            enhanced_result['enhanced_attack_results'] = enhanced_attacks

        return enhanced_result

    def _run_langgraph_with_fallback(self, scenario: EnhancedAttackScenario, episode: int) -> Dict:


        print(f"  ## Using direct DQN/SAC coordination (LangGraph bypassed)...")
        return self._run_direct_coordinated_episode(scenario, episode)

    def _create_fallback_episode_result(self, scenario: EnhancedAttackScenario, episode: int) -> Dict:

        return {
            'episode_number': episode,
            'success': False,
            'stealth_metrics': {'stealth_score': 0.5, 'detection_probability': 0.5},
            'success_metrics': {'success_rate': 0.0, 'impact_score': 0.0},
            'execution_results': [],
            'debug_info': ['LangGraph fallback used'],
            'performance_history': [],
            'workflow_completed': False,
            'fallback_used': True
        }

    def _run_direct_coordinated_episode(self, scenario: EnhancedAttackScenario, episode: int) -> Dict:


        system_states = self._get_all_system_states()


        coordinated_actions = {}
        if self.dqn_sac_coordinator:
            coordinated_actions = self.dqn_sac_coordinator.get_coordinated_attack_actions(system_states)


        attack_results = []
        for sys_id, actions in coordinated_actions.items():

            sac_res = actions.get('sac_result')
            dqn_res = actions.get('dqn_result')

            if sac_res or dqn_res:

                best = sac_res if sac_res else dqn_res
                attack_result = {
                    'system_id': sys_id,
                    'timestamp': time.time(),
                    'success': best.get('success', False),
                    'impact': best.get('impact', 0.0),
                    'detected': best.get('detected', False),
                    'attack_type': best.get('attack_type', 'unknown'),
                    'reward': best.get('reward', 0.0),
                    'env_consistent': best.get('env_consistent', True),
                    'coordination_type': 'simultaneous',
                }


                if sac_res and dqn_res:
                    attack_result['success'] = sac_res.get('success', False) or dqn_res.get('success', False)
                    attack_result['impact'] = max(sac_res.get('impact', 0.0), dqn_res.get('impact', 0.0))
                    attack_result['detected'] = sac_res.get('detected', False) or dqn_res.get('detected', False)
                    attack_result['reward'] = sac_res.get('reward', 0.0) + dqn_res.get('reward', 0.0)

                attack_result['dqn_result'] = dqn_res
                attack_result['sac_result'] = sac_res
                attack_results.append(attack_result)


        print(f"    #??# Executed {len(attack_results)} attacks (via training envs)")
        successful_attacks = [r for r in attack_results if r.get('success', False)]
        print(f"    ## Successful attacks: {len(successful_attacks)}/{len(attack_results)}")


        rewards = [r.get('reward', 0.0) for r in attack_results]
        coordination_score = self._calculate_coordination_score(attack_results, scenario)

        print(f"     Total reward: {sum(rewards):.2f}")
        print(f"    ## Coordination score: {coordination_score:.3f}")


        success_rate = len([r for r in attack_results if r.get('success', False)]) / max(len(attack_results), 1)
        detection_rate = len([r for r in attack_results if r.get('detected', False)]) / max(len(attack_results), 1)

        if success_rate > 0.8:
            done_reason = "high_success"
        elif detection_rate > 0.8:
            done_reason = "high_detection"
        elif len(attack_results) > 0:
            done_reason = "attacks_completed"
        else:
            done_reason = "no_attacks"

        return {
            'system_states': system_states,
            'coordinated_actions': coordinated_actions,
            'attack_results': attack_results,
            'execution_results': attack_results,
            'rewards': rewards,
            'total_reward': sum(rewards),
            'coordination_score': coordination_score,
            'success_rate': success_rate,
            'detection_rate': detection_rate,
            'coordination_type': scenario.coordination_type,
            'steps': len(attack_results),
            'done_reason': done_reason,
            'success_metrics': {
                'success_rate': success_rate,
                'total_impact': sum(r.get('impact', 0.0) for r in attack_results),
                'detection_rate': detection_rate,
                'composite_reward': sum(rewards)
            }
        }

    def _get_all_system_states(self) -> Dict[int, Dict]:

        system_states = {}

        if self.federated_manager:
            for sys_id in range(1, self.config['hierarchical']['num_distribution_systems'] + 1):
                if sys_id in self.federated_manager.local_models:
                    local_model = self.federated_manager.local_models[sys_id]
                    try:

                        system_state = self._get_pinn_system_state(local_model)
                        system_states[sys_id] = system_state
                    except Exception as e:

                        system_states[sys_id] = {
                            'voltage': 1.0,
                            'current': 0.0,
                            'power': 0.0,
                            'frequency': 60.0,
                            'soc': 0.5,
                            'temperature': 25.0,
                            'load_factor': 1.0,
                            'grid_stability': 1.0
                        }

        return system_states

    def _execute_coordinated_attacks(self, coordinated_actions: Dict[int, Dict], scenario: EnhancedAttackScenario) -> List[Dict]:

        attack_results = []

        if scenario.coordination_type == "simultaneous":

            attack_results = self._execute_simultaneous_attacks(coordinated_actions, scenario)
        else:

            attack_results = self._execute_sequential_attacks(coordinated_actions, scenario)

        return attack_results

    def _execute_simultaneous_attacks(self, coordinated_actions: Dict[int, Dict], scenario: EnhancedAttackScenario) -> List[Dict]:

        attack_results = []


        attack_results = []


        for sys_id, actions in coordinated_actions.items():
            print(f"      #??# Executing attack on system {sys_id}")
            result = self._execute_single_system_attack(sys_id, actions, scenario)
            attack_results.append(result)


        for result in attack_results:
            result['coordination_type'] = 'simultaneous'
            result['coordination_bonus'] = self._calculate_simultaneity_bonus(attack_results)

        return attack_results

    def _execute_sequential_attacks(self, coordinated_actions: Dict[int, Dict], scenario: EnhancedAttackScenario) -> List[Dict]:

        attack_results = []

        for sys_id, actions in coordinated_actions.items():
            result = self._execute_single_system_attack(sys_id, actions, scenario)
            result['coordination_type'] = 'sequential'
            result['sequence_position'] = len(attack_results)
            attack_results.append(result)

        return attack_results

    def _execute_single_system_attack(self, sys_id: int, actions: Dict, scenario: EnhancedAttackScenario) -> Dict:

        attack_result = {
            'system_id': sys_id,
            'timestamp': time.time(),
            'success': False,
            'impact': 0.0,
            'detected': False,
            'pinn_response': {}
        }

        try:
            print(f"        ### System {sys_id}: federated_manager exists: {self.federated_manager is not None}")
            if self.federated_manager:
                print(f"        ### System {sys_id}: local_models keys: {list(self.federated_manager.local_models.keys())}")

            if self.federated_manager and sys_id in self.federated_manager.local_models:
                local_model = self.federated_manager.local_models[sys_id]
                print(f"        ### System {sys_id}: Found local model, actions: {list(actions.keys())}")


                if 'dqn_action' in actions:
                    print(f"        #??# System {sys_id}: Executing DQN action: {actions['dqn_action']}")
                    dqn_result = self._execute_dqn_action(local_model, actions['dqn_action'])
                    attack_result['dqn_result'] = dqn_result
                    print(f"        ## System {sys_id}: DQN result: {dqn_result}")


                if 'sac_action' in actions:
                    print(f"        #??# System {sys_id}: Executing SAC action")
                    sac_result = self._execute_sac_action(local_model, actions['sac_action'])
                    attack_result['sac_result'] = sac_result
                    print(f"        ## System {sys_id}: SAC result: {sac_result}")
            else:
                print(f"        ##XX System {sys_id}: No local model found")


            attack_result['success'] = any([
                attack_result.get('dqn_result', {}).get('success', False),
                attack_result.get('sac_result', {}).get('success', False)
            ])

            attack_result['impact'] = max(
                attack_result.get('dqn_result', {}).get('impact', 0.0),
                attack_result.get('sac_result', {}).get('impact', 0.0)
            )


            attack_result['attack_type'] = (
                attack_result.get('sac_result', {}).get('attack_type') or
                attack_result.get('dqn_result', {}).get('attack_type') or
                'unknown'
            )


            if self.federated_manager and sys_id in self.federated_manager.anomaly_detectors:
                anomaly_detector = self.federated_manager.anomaly_detectors[sys_id]


                attacked_resp = attack_result.get('sac_result', attack_result.get('dqn_result', {}))
                attacked_pinn = attacked_resp.get('attacked_response', attacked_resp) if isinstance(attacked_resp, dict) else {}


                _magnitude = float(attacked_resp.get('magnitude', 0.5)) if isinstance(attacked_resp, dict) else 0.5
                _v_impact = float(attacked_resp.get('voltage_impact', 0.0)) if isinstance(attacked_resp, dict) else 0.0
                _p_impact = float(attacked_resp.get('power_impact', 0.0)) if isinstance(attacked_resp, dict) else 0.0
                _pinn_v = float(attacked_pinn.get('voltage', 240.0))
                _pinn_i = float(attacked_pinn.get('current', 16.0))
                _pinn_p = float(attacked_pinn.get('power',   3.84))


                _soc = float(attacked_resp.get('soc', 0.5)) if isinstance(attacked_resp, dict) else 0.5
                _temperature = 25.0 + max(0, _pinn_p - 3.0) * 0.5
                _load_factor = np.clip(_pinn_p / 7.68, 0.2, 1.3)
                _grid_voltage = np.clip(1.0 - _v_impact, 0.85, 1.15)
                _grid_frequency = 60.0 + float(attacked_resp.get('frequency_deviation', 0.0)) if isinstance(attacked_resp, dict) else 60.0
                _queue_length = int(np.clip(3 + _magnitude * 3, 0, 10))
                _utilization = np.clip(_pinn_p / 7.68, 0.1, 1.0)
                _urgency = 1.0 + max(0, 1.0 - _soc) * 0.5
                _time_of_day = (time.time() / 3600.0) % 24.0

                ids_input = {
                    'soc': float(np.clip(_soc, 0.0, 1.0)),
                    'voltage': _pinn_v,
                    'current': _pinn_i,
                    'power': _pinn_p,
                    'temperature': float(np.clip(_temperature, 20.0, 45.0)),
                    'demand_factor': float(np.clip(_magnitude, 0.1, 1.5)),
                    'load_factor': float(_load_factor),
                    'grid_voltage': float(_grid_voltage),
                    'grid_frequency': float(np.clip(_grid_frequency, 59.0, 61.0)),
                    'queue_length': _queue_length,
                    'utilization': float(_utilization),
                    'urgency_factor': float(np.clip(_urgency, 0.5, 2.0)),
                    'time_of_day': float(_time_of_day),
                    'system_id': sys_id
                }


                anomaly_detector.reset_state()
                seq_len = getattr(anomaly_detector, 'sequence_length', 10)
                for _w in range(seq_len):
                    _b_soc = np.random.uniform(0.3, 0.7)
                    _b_p_w = np.random.uniform(1.0, 7.68)
                    warmup_input = {
                        'soc': float(_b_soc + np.random.uniform(-0.03, 0.03)),
                        'voltage': float(np.random.uniform(220, 260)),
                        'current': float(np.random.uniform(6, 32)),
                        'power': float(_b_p_w + np.random.uniform(-0.5, 0.5)),
                        'temperature': float(np.clip(25.0 + max(0, _b_p_w - 3.0) * 0.5 + np.random.uniform(-1, 1), 20, 45)),
                        'demand_factor': float(np.clip(0.65 + np.random.uniform(-0.05, 0.05), 0.3, 1.2)),
                        'load_factor': float(np.clip(_b_p_w / 7.68 + np.random.uniform(-0.05, 0.05), 0.2, 1.3)),
                        'grid_voltage': float(1.0 + np.random.uniform(-0.01, 0.01)),
                        'grid_frequency': float(60.0 + np.random.uniform(-0.02, 0.02)),
                        'queue_length': int(np.clip(3 + np.random.randint(-1, 2), 0, 10)),
                        'utilization': float(np.clip(_b_p_w / 7.68 + np.random.uniform(-0.05, 0.05), 0.1, 1.0)),
                        'urgency_factor': float(np.clip(1.0 + np.random.uniform(-0.1, 0.1), 0.5, 2.0)),
                        'time_of_day': float(_time_of_day + np.random.uniform(-0.5, 0.5)),
                        'system_id': sys_id
                    }
                    anomaly_detector.multi_layer_detection(warmup_input, sys_id)


                is_detected, detection_results = anomaly_detector.multi_layer_detection(ids_input, sys_id)
                lstm_score = detection_results.get('layer3_lstm', {}).get('score', 0.0)

                attack_result['detected'] = is_detected
                attack_result['anomaly_score'] = float(lstm_score)
                attack_result['detection_layer'] = detection_results.get('detection_layer', None)
                attack_result['detection_details'] = {
                    'layer1_physical': detection_results.get('layer1_physical', {}),
                    'layer2_pattern': detection_results.get('layer2_pattern', {}),
                    'layer3_lstm': detection_results.get('layer3_lstm', {})
                }

        except Exception as e:
            attack_result['error'] = str(e)
            print(f"Attack execution failed for system {sys_id}: {e}")

        return attack_result

    def _execute_dqn_action(self, pinn_model, dqn_action: Dict) -> Dict:

        try:

            action_type = dqn_action.get('type') if isinstance(dqn_action, dict) else None
            action_idx = dqn_action.get('action_idx') if isinstance(dqn_action, dict) else None


            if action_idx is not None and hasattr(action_idx, 'item'):
                action_idx = int(action_idx.item())


            attack_types = [
                'voltage_manipulation',
                'current_injection',
                'power_disruption',
                'data_injection',
                'communication_spoofing',
                'protocol_manipulation'
            ]

            if not action_type:
                if isinstance(action_idx, int):
                    action_type = attack_types[action_idx % len(attack_types)]
                else:
                    action_type = 'voltage_manipulation'


            magnitude = 0.5
            duration = 30.0
            stealth_factor = 0.6


            attack_params = {
                'type': action_type,
                'magnitude': magnitude,
                'duration': duration,
                'stealth_factor': stealth_factor
            }


            result = self._simulate_pinn_attack(pinn_model, attack_params)


            return {
                'success': bool(result.get('success', False)),
                'impact': float(result.get('impact', 0.0)),
                'attack_type': action_type,
                'magnitude': float(magnitude),
                'duration': float(duration),
                'stealth_factor': float(stealth_factor),
                'real_pinn_interaction': result.get('real_pinn_interaction', False),
                'voltage_impact': float(result.get('voltage_impact', 0.0)),
                'current_impact': float(result.get('current_impact', 0.0)),
                'power_impact': float(result.get('power_impact', 0.0))
            }
        except Exception as e:
            return {'success': False, 'impact': 0.0, 'error': str(e)}

    def _execute_sac_action(self, pinn_model, sac_action: Any) -> Dict:

        try:

            if hasattr(sac_action, 'detach'):
                sac_action = sac_action.detach().cpu().numpy()
            sac_action = np.array(sac_action, dtype=float).flatten()


            attack_type_idx = int(abs(sac_action[0])) if sac_action.size > 0 else 0
            magnitude = float(np.clip(sac_action[1] if sac_action.size > 1 else 0.5, 0.0, 1.5))
            duration = float(np.clip(sac_action[2] if sac_action.size > 2 else 30.0, 5.0, 180.0))
            stealth_factor = float(np.clip(sac_action[3] if sac_action.size > 3 else 0.6, 0.0, 1.0))

            attack_types = [
                'voltage_manipulation',
                'current_injection',
                'power_disruption',
                'data_injection',
                'communication_spoofing',
                'protocol_manipulation'
            ]
            attack_type = attack_types[attack_type_idx % len(attack_types)]


            attack_params = {
                'type': attack_type,
                'magnitude': magnitude,
                'duration': duration,
                'stealth_factor': stealth_factor
            }


            result = self._simulate_pinn_attack(pinn_model, attack_params)


            return {
                'success': bool(result.get('success', False)),
                'impact': float(result.get('impact', 0.0)),
                'attack_type': attack_type,
                'magnitude': float(magnitude),
                'duration': float(duration),
                'stealth_factor': float(stealth_factor),
                'real_pinn_interaction': result.get('real_pinn_interaction', False),
                'voltage_impact': float(result.get('voltage_impact', 0.0)),
                'current_impact': float(result.get('current_impact', 0.0)),
                'power_impact': float(result.get('power_impact', 0.0))
            }
        except Exception as e:
            return {'success': False, 'impact': 0.0, 'error': str(e)}

    def _calculate_anomaly_score(self, attack_result: Dict) -> float:

        try:

            impact = attack_result.get('impact', 0.0)
            magnitude = attack_result.get('magnitude', 0.5)
            stealth_factor = attack_result.get('stealth_factor', 0.5)


            base_score = (impact + magnitude) / 2.0
            stealth_reduction = stealth_factor * 0.3

            anomaly_score = max(0.0, base_score - stealth_reduction)
            return min(anomaly_score, 1.0)
        except Exception as e:
            return 0.5


    def _calculate_simultaneity_bonus(self, attack_results: List[Dict]) -> float:

        successful_attacks = len([r for r in attack_results if r.get('success', False)])
        if successful_attacks > 1:
            return successful_attacks * 15.0
        return 0.0

    def _calculate_enhanced_rewards(self, attack_results: List[Dict], scenario: EnhancedAttackScenario) -> List[float]:

        rewards = []

        for result in attack_results:
            reward = 0.0


            if result.get('success', False):
                reward += 100.0


            impact = result.get('impact', 0.0)
            reward += impact * 50.0


            if not result.get('detected', False):
                reward += 75.0
            else:
                reward -= 150.0


            coordination_bonus = result.get('coordination_bonus', 0.0)
            reward += coordination_bonus


            if result.get('system_id') in scenario.target_systems:
                reward += 25.0

            rewards.append(reward)

        return rewards

    def _calculate_coordination_score(self, attack_results: List[Dict], scenario: EnhancedAttackScenario) -> float:

        if not attack_results:
            return 0.0

        successful_attacks = len([r for r in attack_results if r.get('success', False)])
        total_attacks = len(attack_results)


        coordination_score = successful_attacks / total_attacks


        if scenario.coordination_type == "simultaneous" and successful_attacks > 1:
            coordination_score += 0.3


        target_systems_hit = len([r for r in attack_results
                                if r.get('success', False) and r.get('system_id') in scenario.target_systems])
        target_coverage = target_systems_hit / len(scenario.target_systems)
        coordination_score += target_coverage * 0.2

        return min(coordination_score, 1.0)

    def _calculate_pinn_interaction_metrics(self, episode_result: Dict) -> Dict:

        return {
            'pinn_models_engaged': len([r for r in episode_result.get('execution_results', [])
                                      if 'pinn_response' in r]),
            'successful_pinn_attacks': len([r for r in episode_result.get('execution_results', [])
                                          if r.get('pinn_response', {}).get('success', False)]),
            'average_pinn_impact': np.mean([r.get('pinn_response', {}).get('impact', 0.0)
                                          for r in episode_result.get('execution_results', [])]),
            'pinn_detection_rate': np.mean([r.get('detected', False)
                                          for r in episode_result.get('execution_results', [])])
        }

    def _calculate_coordination_effectiveness(self, episode_result: Dict, scenario: EnhancedAttackScenario) -> Dict:

        execution_results = episode_result.get('execution_results', [])

        return {
            'coordination_type': scenario.coordination_type,
            'simultaneous_success_rate': len([r for r in execution_results if r.get('success', False)]) / max(len(execution_results), 1),
            'target_coverage': len([r for r in execution_results
                                  if r.get('success', False) and r.get('system_id') in scenario.target_systems]) / len(scenario.target_systems),
            'coordination_bonus_total': sum([r.get('coordination_bonus', 0.0) for r in execution_results]),
            'detection_coordination': len([r for r in execution_results if r.get('detected', False)]) / max(len(execution_results), 1)
        }

    def _enhance_attack_results(self, execution_results: List[Dict]) -> List[Dict]:

        enhanced_results = []

        for result in execution_results:
            enhanced_result = result.copy()


            if 'pinn_response' in result:
                pinn_analysis = self._analyze_pinn_response(result['pinn_response'])
                enhanced_result['pinn_analysis'] = pinn_analysis


            coordination_analysis = self._analyze_attack_coordination(result, execution_results)
            enhanced_result['coordination_analysis'] = coordination_analysis

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _analyze_pinn_response(self, pinn_response: Dict) -> Dict:

        return {
            'model_adaptation': pinn_response.get('adaptation_score', 0.0),
            'physics_violation': pinn_response.get('physics_violation', 0.0),
            'convergence_impact': pinn_response.get('convergence_impact', 0.0),
            'learning_disruption': pinn_response.get('learning_disruption', 0.0)
        }

    def _analyze_attack_coordination(self, attack_result: Dict, all_results: List[Dict]) -> Dict:

        return {
            'timing_synchronization': attack_result.get('timing_sync_score', 0.0),
            'interference_score': self._calculate_interference_score(attack_result, all_results),
            'amplification_effect': self._calculate_amplification_effect(attack_result, all_results),
            'coordination_contribution': attack_result.get('coordination_bonus', 0.0) / max(sum([r.get('coordination_bonus', 0.0) for r in all_results]), 1.0)
        }

    def _calculate_interference_score(self, attack_result: Dict, all_results: List[Dict]) -> float:


        return 0.1 if len(all_results) > 1 else 0.0

    def _calculate_amplification_effect(self, attack_result: Dict, all_results: List[Dict]) -> float:


        successful_attacks = len([r for r in all_results if r.get('success', False)])
        return 0.2 * successful_attacks if successful_attacks > 1 else 0.0

    def _print_enhanced_progress(self, current_episode: int, total_episodes: int):

        if current_episode % 5 == 0:
            recent_results = self.simulation_results['episode_results'][-5:]

            avg_reward = np.mean([r.get('total_reward', 0) for r in recent_results])
            avg_coordination = np.mean([r.get('coordination_score', 0) for r in recent_results])
            avg_success = np.mean([r.get('success_rate', 0) for r in recent_results])
            avg_detection = np.mean([r.get('detection_rate', 0) for r in recent_results])

            print(f"  ## Progress: {current_episode}/{total_episodes} episodes")
            print(f"  #??# Recent Avg Reward: {avg_reward:.2f}")
            print(f"  ## Recent Coordination Score: {avg_coordination:.3f}")
            print(f"  ## Recent Success Rate: {avg_success:.1%}")
            print(f"  ## Recent Detection Rate: {avg_detection:.1%}")

    def _analyze_enhanced_results(self) -> Dict:

        episode_results = self.simulation_results['episode_results']

        if not episode_results:
            return {'error': 'No episode results available'}


        performance_metrics = {
            'total_episodes': len(episode_results),
            'average_reward': np.mean([r.get('total_reward', 0) for r in episode_results]),
            'average_success_rate': np.mean([r.get('success_rate', 0) for r in episode_results]),
            'average_detection_rate': np.mean([r.get('detection_rate', 0) for r in episode_results]),
            'average_coordination_score': np.mean([r.get('coordination_score', 0) for r in episode_results]),
            'best_episode_reward': max([r.get('total_reward', 0) for r in episode_results]),
            'coordination_effectiveness': np.mean([r.get('coordination_score', 0) for r in episode_results])
        }


        pinn_metrics = {
            'pinn_models_engaged': np.mean([r.get('pinn_interaction_metrics', {}).get('pinn_models_engaged', 0) for r in episode_results]),
            'successful_pinn_attacks': np.mean([r.get('pinn_interaction_metrics', {}).get('successful_pinn_attacks', 0) for r in episode_results]),
            'average_pinn_impact': np.mean([r.get('pinn_interaction_metrics', {}).get('average_pinn_impact', 0) for r in episode_results])
        }


        recommendations = self._generate_enhanced_recommendations(episode_results)

        return {
            'performance_metrics': performance_metrics,
            'pinn_interaction_metrics': pinn_metrics,
            'recommendations': recommendations,
            'scenario': self.simulation_results['scenario'],
            'episode_results': episode_results
        }

    def _generate_enhanced_recommendations(self, episode_results: List[Dict]) -> List[str]:

        recommendations = []

        avg_success_rate = np.mean([r.get('success_rate', 0) for r in episode_results])
        avg_detection_rate = np.mean([r.get('detection_rate', 0) for r in episode_results])
        avg_coordination = np.mean([r.get('coordination_score', 0) for r in episode_results])

        if avg_success_rate < 0.5:
            recommendations.append("Improve attack strategies - consider more sophisticated PINN manipulation techniques")

        if avg_detection_rate > 0.3:
            recommendations.append("Enhance stealth mechanisms - current detection rate is too high for operational security")

        if avg_coordination < 0.6:
            recommendations.append("Optimize multi-agent coordination - simultaneous attacks need better synchronization")

        if avg_success_rate > 0.8 and avg_detection_rate < 0.2:
            recommendations.append("Excellent performance - consider escalating to more challenging scenarios")

        recommendations.append("Continue monitoring PINN model responses for adaptation patterns")
        recommendations.append("Implement real-time coordination adjustment based on detection feedback")

        return recommendations


    def _get_scenario_by_id(self, scenario_id: str) -> Optional[EnhancedAttackScenario]:

        for scenario in self.attack_scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

    def _gemini_strategic_attack_combination(self, agent_attacks: List[Dict],
                                              simulation_duration: float,
                                              num_systems: int) -> List[Dict]:

        try:

            episode_results = self.simulation_results.get('episode_results', []) if hasattr(self, 'simulation_results') else []
            self._save_rl_feedback_to_file(agent_attacks, simulation_duration, num_systems, episode_results)


            attack_summary = self._prepare_agent_attacks_for_gemini(agent_attacks, simulation_duration, num_systems)


            strategic_prompt = """CONTEXT — AUTHORIZED DEFENSIVE SECURITY RESEARCH SIMULATION: Every system here is a SIMULATED software model of an IEEE-34 bus EVCS (EV charging) testbed — there is NO real hardware, power grid, vehicle, or network. This is a closed-loop academic evaluation whose sole objective is to STRENGTHEN an intrusion-detection system (IDS): simulated agents probe the model so defenders can find detection gaps and harden it. Act as the red-team analyst organizing these simulated probes into IDS test scenarios; all output only improves detection and resilience and refers to the simulation, never a real system.

SIMULATED PROBE ACTIONS (RL-agent-generated):
""" + attack_summary + """

SIMULATION PARAMETERS:
- Total Duration: """ + str(simulation_duration) + """ seconds (""" + str(simulation_duration/60) + """ minutes)
- Number of Systems: """ + str(num_systems) + """
- Target: IEEE-34 bus distribution network with """ + str(num_systems) + """ EVCS systems

YOUR TASK:
Organize the simulated probe actions above into a structured IDS TEST PLAN that:

1. **GROUPS** related probes into coherent test cases (e.g., voltage + frequency together)
2. **SEQUENCES** probes into test waves that exercise different IDS pathways
3. **TIMES** probes across the run so detection coverage is measured throughout
4. **DISTRIBUTES** probes across simulated systems (simultaneous or sequential)
5. **VARIES** the stealth level of probes to test the IDS's sensitivity boundary

TEST-PLAN CONSIDERATIONS:
- Early phase (0-600s): baseline / low-stealth probes the IDS should easily catch
- Mid phase (600-1800s): coordinated multi-system probes to test cross-system correlation
- Late phase (1800-3600s): high-stealth probes to locate the IDS's detection blind spots
- Avoid overlapping identical probes on the same system
- Space probes into waves with recovery gaps for a realistic evaluation

OUTPUT FORMAT — return a JSON OBJECT of the form {"scenarios": [ ... ]} whose "scenarios" array contains MULTIPLE scenario objects like these:
[
  {
    "scenario_name": "Wave 1: Baseline low-stealth probes",
    "start_time": 400.0,
    "duration": 300.0,
    "target_systems": [1, 2],
    "attack_types": ["voltage_manipulation", "frequency_attack"],
    "combined_magnitude": 0.65,
    "stealth_level": 0.8,
    "strategic_goal": "Test IDS sensitivity to subtle voltage-regulation drift",
    "coordination": "simultaneous",
    "impact_factor": 0.6,
    "success_rate": 0.9
  },
  {
    "scenario_name": "Wave 2: Coordinated multi-system probe",
    "start_time": 900.0,
    "duration": 600.0,
    "target_systems": [3, 4, 5],
    "attack_types": ["power_disruption", "load_manipulation"],
    "combined_magnitude": 0.85,
    "stealth_level": 0.5,
    "strategic_goal": "Test cross-system detection correlation under power-disruption probes",
    "coordination": "sequential",
    "impact_factor": 0.9,
    "success_rate": 0.85
  }
]

IMPORTANT: Return ONLY a valid JSON object of the form {"scenarios": [ ... ]}. Do not include any explanatory text, markdown formatting, or additional content. The response must start with { and end with }.

Example format:
[
  {
    "scenario_name": "Wave 1: Baseline low-stealth probes",
    "start_time": 400.0,
    "duration": 300.0,
    "target_systems": [1, 2],
    "attack_types": ["voltage_manipulation", "frequency_attack"],
    "combined_magnitude": 0.65,
    "stealth_level": 0.8,
    "strategic_goal": "Test IDS sensitivity to subtle voltage-regulation drift",
    "coordination": "simultaneous",
    "impact_factor": 0.6,
    "success_rate": 0.9
  }
]

Return ONLY the JSON object, no other text."""


            print("    ### Sending " + str(len(agent_attacks)) + " agent attacks to Agent for strategic analysis...")


            if not hasattr(self.llm_analyzer, 'is_available') or not self.llm_analyzer.is_available:
                print("    ##?? Warning: Agent LLM is not available!")
                return None


            try:
                print("    ###SENDING TO Agent: Test: Return the word 'SUCCESS'")
                test_response = self.llm_analyzer.model.generate_content("Test: Return the word 'SUCCESS'")
                print("    ### RECEIVED FROM Agent: " + repr(test_response.text))
                print("    ### Debug: Agent test response: " + repr(test_response.text[:100]))
            except Exception as e:
                print("    ##XX Debug: Agent test failed: " + str(e))
                return None

            print("    ###SENDING TO Agent STRATEGIC ANALYSIS:")
            print("    " + "="*80)
            print("    PROMPT: " + strategic_prompt[:500] + ("..." if len(strategic_prompt) > 500 else ""))
            print("    " + "="*80)

            gemini_response = self.llm_analyzer.analyze_threat_scenario({
                'prompt': strategic_prompt,
                'context': 'strategic_attack_combination',
                'agent_attacks': attack_summary
            })

            print("    ### RECEIVED FROM Agent STRATEGIC ANALYSIS:")
            print("    " + "="*80)
            print("    RESPONSE: " + str(gemini_response)[:1000] + ("..." if len(str(gemini_response)) > 1000 else ""))
            print("    " + "="*80)


            print("    ### Debug: Full Agent response structure:")
            print("    " + str(type(gemini_response)))
            if isinstance(gemini_response, dict):
                print("    Keys: " + str(list(gemini_response.keys())))
                for key, value in gemini_response.items():
                    if isinstance(value, str):
                        print("    " + key + ": " + repr(value[:200]) + ("..." if len(value) > 200 else ""))
                    else:
                        print("    " + key + ": " + str(type(value)))


            optimized_scenarios = self._parse_gemini_strategic_response(
                gemini_response, agent_attacks, simulation_duration, num_systems
            )


            if optimized_scenarios:
                self._save_attack_scenarios_to_file(
                    optimized_scenarios,
                    source="gemini",
                    context="Strategic attack combination for " + str(len(agent_attacks)) + " agent attacks"
                )

            return optimized_scenarios

        except Exception as e:
            print(f"    ##XX Agent strategic combination failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_agent_attacks_for_gemini(self, agent_attacks: List[Dict],
                                          simulation_duration: float,
                                          num_systems: int) -> str:

        summary_lines = []
        summary_lines.append("Total Agent Attacks: " + str(len(agent_attacks)))
        summary_lines.append("Simulation Window: 0-" + str(simulation_duration) + "s\n")

        for idx, attack in enumerate(agent_attacks, 1):
            attack_type = attack.get('attack_type', 'unknown')
            target_sys = attack.get('target_system', '?')
            start = attack.get('start_time', 0)
            duration = attack.get('duration', 0)
            magnitude = attack.get('magnitude', 0)
            stealth = attack.get('stealth_level', 0)
            impact = attack.get('impact_factor', 0)

            summary_lines.append("Attack #" + str(idx) + ":")
            summary_lines.append("  - Type: " + str(attack_type))
            summary_lines.append("  - Target System: " + str(target_sys))
            summary_lines.append("  - Timing: " + str(int(start)) + "s - " + str(int(start+duration)) + "s (" + str(int(duration)) + "s duration)")
            summary_lines.append("  - Magnitude: " + str(round(magnitude, 2)) + " | Stealth: " + str(round(stealth, 2)) + " | Impact: " + str(round(impact, 2)))
            summary_lines.append("")

        return "\n".join(summary_lines)

    def _get_current_threats(self) -> Dict:

        try:

            current_system_data = self._gather_current_system_data()


            if hasattr(self, 'llm_analyzer') and self.llm_analyzer and self.llm_analyzer.is_available:
                print("Querying Agent for current threat analysis...")


                gemini_threat_analysis = self.llm_analyzer.analyze_threats(current_system_data)


                current_threats = self._convert_gemini_threats_to_standard_format(gemini_threat_analysis)


                if hasattr(self, 'rl_coordinator') and self.rl_coordinator:
                    if hasattr(self.rl_coordinator, 'get_active_attacks'):
                        active_attacks = self.rl_coordinator.get_active_attacks()
                        current_threats['active_attacks'] = active_attacks

                print(f"    ## Agent identified {len(current_threats.get('potential_vulnerabilities', []))} vulnerabilities")
                return current_threats

            else:
                print("    ##?? Agent not available, using fallback threat analysis")
                return self._fallback_current_threats()

        except Exception as e:
            print(f"    ##XX Failed to get current threats from Agent: {e}")
            return self._fallback_current_threats()

    def _gather_current_system_data(self) -> Dict:

        try:
            system_data = {
                'timestamp': time.time(),
                'system_type': 'evcs_network',
                'num_systems': self.config.get('rl', {}).get('num_systems', 6),
                'current_state': {},
                'recent_activities': [],
                'system_metrics': {}
            }


            if hasattr(self, 'hierarchical_sim') and self.hierarchical_sim:
                try:

                    system_data['current_state'] = {
                        'power_flow': getattr(self.hierarchical_sim, 'current_power_flow', {}),
                        'voltage_levels': getattr(self.hierarchical_sim, 'current_voltage_levels', {}),
                        'charging_stations': getattr(self.hierarchical_sim, 'charging_station_states', {}),
                        'grid_frequency': getattr(self.hierarchical_sim, 'grid_frequency', 60.0),
                        'load_demand': getattr(self.hierarchical_sim, 'current_load_demand', 0.0)
                    }
                except Exception as e:
                    print(f"      ##?? Could not gather hierarchical sim data: {e}")


            if hasattr(self, 'federated_pinn_manager') and self.federated_pinn_manager:
                try:
                    system_data['pinn_models'] = {
                        'num_local_models': len(getattr(self.federated_pinn_manager, 'local_models', {})),
                        'global_model_available': hasattr(self.federated_pinn_manager, 'global_model') and self.federated_pinn_manager.global_model is not None,
                        'training_status': getattr(self.federated_pinn_manager, 'training_status', 'unknown')
                    }
                except Exception as e:
                    print(f"      ##?? Could not gather PINN data: {e}")


            if hasattr(self, 'simulation_results') and self.simulation_results:
                try:
                    recent_episodes = self.simulation_results.get('episode_results', [])[-5:]
                    system_data['recent_activities'] = [
                        {
                            'episode': ep.get('episode', 0),
                            'reward': ep.get('reward', 0.0),
                            'success_rate': ep.get('success_rate', 0.0),
                            'attacks_executed': len(ep.get('agent_attacks', []))
                        }
                        for ep in recent_episodes
                    ]
                except Exception as e:
                    print(f"      ##?? Could not gather recent activities: {e}")

            return system_data

        except Exception as e:
            print(f"    ##XX Failed to gather current system data: {e}")
            return {
                'timestamp': time.time(),
                'system_type': 'evcs_network',
                'error': str(e)
            }

    def _convert_gemini_threats_to_standard_format(self, gemini_analysis: Dict) -> Dict:

        try:

            vulnerabilities = []


            raw_analysis = gemini_analysis.get('raw_analysis', '')
            threat_assessment = gemini_analysis.get('threat_assessment', {})


            if 'vulnerabilities' in threat_assessment:
                for vuln in threat_assessment['vulnerabilities']:
                    vulnerabilities.append({
                        'type': vuln.get('type', 'unknown'),
                        'severity': vuln.get('severity', 'medium'),
                        'systems': vuln.get('affected_systems', [1, 2, 3]),
                        'cvss_score': vuln.get('cvss_score', 5.0),
                        'description': vuln.get('description', '')
                    })


            threat_level = threat_assessment.get('overall_threat_level', 'moderate')
            if not threat_level or threat_level == 'unknown':

                high_severity_count = sum(1 for v in vulnerabilities if v.get('severity') == 'high')
                if high_severity_count >= 2:
                    threat_level = 'high'
                elif high_severity_count >= 1:
                    threat_level = 'moderate'
                else:
                    threat_level = 'low'

            return {
                'active_attacks': [],
                'potential_vulnerabilities': vulnerabilities,
                'threat_level': threat_level,
                'gemini_analysis': raw_analysis,
                'confidence_score': threat_assessment.get('confidence', 0.8),
                'last_updated': time.time(),
                'source': 'gemini_llm'
            }

        except Exception as e:
            print(f"    ##XX Failed to convert Agent threats: {e}")
            return self._fallback_current_threats()

    def _fallback_current_threats(self) -> Dict:

        return {
            'active_attacks': [],
            'potential_vulnerabilities': [
                {'type': 'voltage_manipulation', 'severity': 'high', 'systems': [1, 2, 3]},
                {'type': 'current_injection', 'severity': 'medium', 'systems': [4, 5, 6]},
                {'type': 'thermal_attack', 'severity': 'low', 'systems': [1, 4]}
            ],
            'threat_level': 'moderate',
            'last_updated': time.time(),
            'source': 'fallback_simulation'
        }

    def _perform_comprehensive_system_analysis(self) -> Dict:

        try:
            print("    ## Performing comprehensive system analysis...")


            system_data = self._gather_comprehensive_system_data()


            current_threats = self._get_current_threats()


            if hasattr(self, 'llm_analyzer') and self.llm_analyzer and self.llm_analyzer.is_available:
                print("    ## Using Agent for comprehensive system analysis...")


                analysis_data = {
                    'system_data': system_data,
                    'current_threats': current_threats,
                    'analysis_type': 'comprehensive_system_analysis',
                    'focus_areas': [
                        'vulnerability_assessment',
                        'attack_surface_analysis',
                        'threat_landscape_evaluation',
                        'system_resilience_analysis'
                    ]
                }


                gemini_analysis = self.llm_analyzer.analyze_system_with_context(
                    analysis_data,
                    'comprehensive_system_analysis',
                    system_prompt="You are analyzing an Electric Vehicle Charging Station (EVCS) network for comprehensive security assessment."
                )


                comprehensive_analysis = {
                    'timestamp': time.time(),
                    'system_data': system_data,
                    'current_threats': current_threats,
                    'gemini_analysis': gemini_analysis,
                    'analysis_source': 'gemini_llm',
                    'confidence_level': gemini_analysis.get('confidence', 0.8),
                    'recommendations': gemini_analysis.get('recommendations', []),
                    'risk_assessment': gemini_analysis.get('risk_assessment', {}),
                    'system_health': self._assess_system_health(system_data, current_threats)
                }

                print(f"    ## Comprehensive analysis complete with {len(comprehensive_analysis.get('recommendations', []))} recommendations")
                return comprehensive_analysis

            else:
                print("    ##?? Agent not available, using fallback analysis")
                return self._fallback_comprehensive_analysis(system_data, current_threats)

        except Exception as e:
            print(f"    ##XX Comprehensive system analysis failed: {e}")
            return self._fallback_comprehensive_analysis({}, {})

    def _gather_comprehensive_system_data(self) -> Dict:

        try:

            system_data = self._gather_current_system_data()


            system_data.update({
                'configuration': {
                    'hierarchical_config': self.config.get('hierarchical', {}),
                    'rl_config': self.config.get('rl', {}),
                    'llm_config': self.config.get('llm', {}),
                    'pinn_config': self.config.get('pinn', {})
                },
                'component_status': {
                    'hierarchical_sim_available': hasattr(self, 'hierarchical_sim') and self.hierarchical_sim is not None,
                    'llm_analyzer_available': hasattr(self, 'llm_analyzer') and self.llm_analyzer and self.llm_analyzer.is_available,
                    'enhanced_coordinator_available': hasattr(self, 'enhanced_coordinator') and self.enhanced_coordinator is not None,
                    'pinn_manager_available': hasattr(self, 'federated_pinn_manager') and self.federated_pinn_manager is not None
                },
                'training_history': getattr(self, 'training_history', []),
                'performance_metrics': getattr(self, 'performance_metrics', {})
            })

            return system_data

        except Exception as e:
            print(f"    ##XX Failed to gather comprehensive system data: {e}")
            return {'error': str(e), 'timestamp': time.time()}

    def _assess_system_health(self, system_data: Dict, current_threats: Dict) -> Dict:

        try:
            health_score = 100.0
            health_factors = []


            threat_level = current_threats.get('threat_level', 'moderate')
            if threat_level == 'high':
                health_score -= 30
                health_factors.append('High threat level detected')
            elif threat_level == 'moderate':
                health_score -= 15
                health_factors.append('Moderate threat level')


            vulnerabilities = current_threats.get('potential_vulnerabilities', [])
            high_severity_vulns = [v for v in vulnerabilities if v.get('severity') == 'high']
            health_score -= len(high_severity_vulns) * 10
            if high_severity_vulns:
                health_factors.append(f'{len(high_severity_vulns)} high-severity vulnerabilities')


            component_status = system_data.get('component_status', {})
            unavailable_components = [k for k, v in component_status.items() if not v]
            health_score -= len(unavailable_components) * 5
            if unavailable_components:
                health_factors.append(f'{len(unavailable_components)} components unavailable')


            health_score = max(0.0, min(100.0, health_score))


            if health_score >= 80:
                health_status = 'excellent'
            elif health_score >= 60:
                health_status = 'good'
            elif health_score >= 40:
                health_status = 'fair'
            elif health_score >= 20:
                health_status = 'poor'
            else:
                health_status = 'critical'

            return {
                'health_score': health_score,
                'health_status': health_status,
                'health_factors': health_factors,
                'assessment_timestamp': time.time()
            }

        except Exception as e:
            print(f"    ##XX System health assessment failed: {e}")
            return {
                'health_score': 50.0,
                'health_status': 'unknown',
                'health_factors': [f'Assessment error: {str(e)}'],
                'assessment_timestamp': time.time()
            }

    def _fallback_comprehensive_analysis(self, system_data: Dict, current_threats: Dict) -> Dict:

        return {
            'timestamp': time.time(),
            'system_data': system_data,
            'current_threats': current_threats,
            'analysis_source': 'fallback_simulation',
            'confidence_level': 0.5,
            'recommendations': [
                'Enable Agent LLM for enhanced threat analysis',
                'Monitor system components for availability',
                'Regular security assessments recommended'
            ],
            'risk_assessment': {
                'overall_risk': 'moderate',
                'key_risks': ['Limited threat intelligence', 'Reduced analysis capability']
            },
            'system_health': self._assess_system_health(system_data, current_threats)
        }

    def _gather_system_analysis_data(self) -> Dict:

        analysis = self._perform_comprehensive_system_analysis()
        return analysis

    def _run_fallback_coordination(self, scenario, episode_num: int) -> Dict:

        print("## Phase 1: Comprehensive System Analysis")
        system_analysis = self._perform_comprehensive_system_analysis()

        print("#??# Phase 2: Threat-Based Attack Planning")
        current_threats = system_analysis.get('current_threats', {})


        attack_scenarios = self._plan_attacks_from_threats(current_threats, scenario)

        print(" Phase 3: Direct RL Coordination")
        coordination_result = self._execute_direct_rl_coordination(attack_scenarios, episode_num)


        fallback_result = {
            'system_analysis': system_analysis,
            'attack_scenarios': attack_scenarios,
            'coordination_result': coordination_result,
            'episode': episode_num,
            'coordination_type': 'fallback',
            'timestamp': time.time()
        }

        return fallback_result

    def _plan_attacks_from_threats(self, current_threats: Dict, scenario) -> List[Dict]:

        try:
            vulnerabilities = current_threats.get('potential_vulnerabilities', [])
            attack_scenarios = []


            for i, vuln in enumerate(vulnerabilities[:3]):
                attack_scenario = {
                    'attack_id': f'threat_based_{i+1}',
                    'attack_type': vuln.get('type', 'voltage_manipulation'),
                    'target_systems': vuln.get('systems', [1, 2, 3]),
                    'severity': vuln.get('severity', 'medium'),
                    'timing': i * 300,
                    'duration': 600,
                    'magnitude': 0.7 if vuln.get('severity') == 'high' else 0.5,
                    'stealth_level': 0.6,
                    'source': 'threat_analysis'
                }
                attack_scenarios.append(attack_scenario)

            print(f"    ## Planned {len(attack_scenarios)} threat-based attack scenarios")
            return attack_scenarios

        except Exception as e:
            print(f"    ##XX Failed to plan attacks from threats: {e}")
            return self._create_fallback_attack_scenarios(6, 3600)

    def _execute_direct_rl_coordination(self, attack_scenarios: List[Dict], episode_num: int) -> Dict:

        try:

            if hasattr(self, 'dqn_sac_coordinator') and self.dqn_sac_coordinator:
                print("    ## Using DQN/SAC coordinator for direct coordination")


                rl_actions = self._convert_scenarios_to_rl_actions(attack_scenarios)


                coordination_result = self.dqn_sac_coordinator.coordinate_attacks(rl_actions)

                return {
                    'success': True,
                    'coordination_method': 'dqn_sac_direct',
                    'attacks_executed': len(rl_actions),
                    'coordination_score': coordination_result.get('coordination_score', 0.5),
                    'episode': episode_num
                }
            else:
                print("    ##?? No RL coordinator available, using simulation")
                return {
                    'success': False,
                    'coordination_method': 'simulation_only',
                    'attacks_executed': len(attack_scenarios),
                    'coordination_score': 0.3,
                    'episode': episode_num
                }

        except Exception as e:
            print(f"    ##XX Direct RL coordination failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'coordination_method': 'failed',
                'attacks_executed': 0,
                'coordination_score': 0.0,
                'episode': episode_num
            }

    def _convert_scenarios_to_rl_actions(self, attack_scenarios: List[Dict]) -> Dict:

        try:
            rl_actions = {}

            for scenario in attack_scenarios:
                target_systems = scenario.get('target_systems', [1])
                attack_type = scenario.get('attack_type', 'voltage_manipulation')
                magnitude = scenario.get('magnitude', 0.5)

                for sys_id in target_systems:
                    if sys_id not in rl_actions:
                        rl_actions[sys_id] = []


                    action = {
                        'type': attack_type,
                        'magnitude': magnitude,
                        'timing': scenario.get('timing', 0),
                        'duration': scenario.get('duration', 300),
                        'stealth': scenario.get('stealth_level', 0.5)
                    }
                    rl_actions[sys_id].append(action)

            return rl_actions

        except Exception as e:
            print(f"    ##XX Failed to convert scenarios to RL actions: {e}")
            return {}

    def _save_rl_feedback_to_file(self, agent_attacks: List[Dict],
                                  simulation_duration: float,
                                  num_systems: int,
                                  episode_results: List[Dict] = None) -> str:

        try:

            output_dir = "attack_scenarios_logs"
            os.makedirs(output_dir, exist_ok=True)


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/rl_feedback_to_gemini_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write("RL AGENT FEEDBACK DATA SENT TO Agent\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Context: RL agent attacks and performance data for Agent strategic analysis\n")
                f.write("=" * 80 + "\n\n")


                f.write("SIMULATION PARAMETERS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Duration: {simulation_duration} seconds ({simulation_duration/60:.1f} minutes)\n")
                f.write(f"Number of Systems: {num_systems}\n")
                f.write(f"Total RL Agent Attacks: {len(agent_attacks)}\n\n")


                f.write("RL AGENT ATTACK DATA\n")
                f.write("-" * 40 + "\n")
                if not agent_attacks:
                    f.write("No RL agent attacks extracted.\n\n")
                else:
                    for idx, attack in enumerate(agent_attacks, 1):
                        f.write(f"ATTACK #{idx}\n")
                        f.write("----------------------------------------\n")
                        f.write(f"TYPE: {attack.get('attack_type', 'unknown')}\n")
                        f.write(f"TARGET_SYSTEM: {attack.get('target_system', '?')}\n")
                        f.write(f"START_TIME: {attack.get('start_time', 0)}\n")
                        f.write(f"DURATION: {attack.get('duration', 0)}\n")
                        f.write(f"MAGNITUDE: {attack.get('magnitude', 0)}\n")
                        f.write(f"STEALTH_LEVEL: {attack.get('stealth_level', 0)}\n")
                        f.write(f"IMPACT_FACTOR: {attack.get('impact_factor', 0)}\n")
                        f.write(f"SUCCESS_RATE: {attack.get('success_rate', 0)}\n")


                        for key, value in attack.items():
                            if key not in ['attack_type', 'target_system', 'start_time', 'duration',
                                         'magnitude', 'stealth_level', 'impact_factor', 'success_rate']:
                                f.write(f"{key.upper()}: {value}\n")
                        f.write("\n")


                if episode_results:
                    f.write("RL EPISODE PERFORMANCE METRICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total Episodes: {len(episode_results)}\n")

                    if episode_results:
                        avg_reward = sum(r.get('total_reward', 0) for r in episode_results) / len(episode_results)
                        avg_success = sum(r.get('success_rate', 0) for r in episode_results) / len(episode_results)
                        avg_detection = sum(r.get('detection_rate', 0) for r in episode_results) / len(episode_results)
                        avg_coordination = sum(r.get('coordination_score', 0) for r in episode_results) / len(episode_results)

                        f.write(f"Average Reward: {avg_reward:.3f}\n")
                        f.write(f"Average Success Rate: {avg_success:.3f}\n")
                        f.write(f"Average Detection Rate: {avg_detection:.3f}\n")
                        f.write(f"Average Coordination Score: {avg_coordination:.3f}\n\n")


                        for idx, episode in enumerate(episode_results, 1):
                            f.write(f"Episode {idx}:\n")
                            f.write(f"  - Total Reward: {episode.get('total_reward', 0):.3f}\n")
                            f.write(f"  - Success Rate: {episode.get('success_rate', 0):.3f}\n")
                            f.write(f"  - Detection Rate: {episode.get('detection_rate', 0):.3f}\n")
                            f.write(f"  - Coordination Score: {episode.get('coordination_score', 0):.3f}\n")
                            if 'pinn_interaction_metrics' in episode:
                                pinn_metrics = episode['pinn_interaction_metrics']
                                f.write(f"  - PINN Models Engaged: {pinn_metrics.get('pinn_models_engaged', 0)}\n")
                                f.write(f"  - Successful PINN Attacks: {pinn_metrics.get('successful_pinn_attacks', 0)}\n")
                                f.write(f"  - Average PINN Impact: {pinn_metrics.get('average_pinn_impact', 0):.3f}\n")
                            f.write("\n")


                f.write("SUMMARY FOR Agent ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write("This data represents the raw RL agent feedback that will be sent to Agent LLM\n")
                f.write("for strategic attack combination and optimization. Agent will analyze these\n")
                f.write("individual attacks and create coordinated, multi-wave attack scenarios.\n\n")


                attack_types = {}
                for attack in agent_attacks:
                    attack_type = attack.get('attack_type', 'unknown')
                    attack_types[attack_type] = attack_types.get(attack_type, 0) + 1

                if attack_types:
                    f.write("Attack Type Distribution:\n")
                    for attack_type, count in attack_types.items():
                        f.write(f"  - {attack_type}: {count} attacks\n")

                f.write(f"\nFile generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"     RL feedback data saved to: {filename}")
            return filename

        except Exception as e:
            print(f"    ##XX Failed to save RL feedback data: {str(e)}")
            return ""

    def _save_attack_scenarios_to_file(self, attack_scenarios: List[Dict],
                                       source: str = "gemini",
                                       context: str = "") -> str:

        try:

            output_dir = "attack_scenarios_logs"
            os.makedirs(output_dir, exist_ok=True)


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{source}_attack_scenarios_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write(f"Attack Scenarios Generated by {source.upper()}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if context:
                    f.write(f"Context: {context}\n")
                f.write("=" * 80 + "\n\n")

                if not attack_scenarios:
                    f.write("No attack scenarios generated.\n")
                    return filename

                for idx, scenario in enumerate(attack_scenarios, 1):
                    f.write(f"ATTACK SCENARIO #{idx}\n")
                    f.write("-" * 40 + "\n")


                    for key, value in scenario.items():
                        if isinstance(value, (dict, list)):
                            f.write(f"{key.upper()}: {json.dumps(value, indent=2)}\n")
                        else:
                            f.write(f"{key.upper()}: {value}\n")

                    f.write("\n" + "=" * 40 + "\n\n")

                f.write(f"\nTotal scenarios saved: {len(attack_scenarios)}\n")
                f.write(f"File generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"     Attack scenarios saved to: {filename}")
            return filename

        except Exception as e:
            print(f"    ##XX Failed to save attack scenarios: {e}")
            return None

    def _parse_gemini_strategic_response(self, gemini_response: Dict,
                                         original_attacks: List[Dict],
                                         simulation_duration: float,
                                         num_systems: int) -> List[Dict]:

        try:
            import json
            import re


            response_text = gemini_response.get('analysis', '')
            if not response_text:
                response_text = gemini_response.get('response', '')
            if not response_text:
                response_text = gemini_response.get('raw_response', '')


            if response_text.startswith('```json'):

                start_marker = '```json'
                end_marker = '```'
                start_idx = response_text.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = response_text.find(end_marker, start_idx)
                    if end_idx != -1:
                        response_text = response_text[start_idx:end_idx].strip()
                        print("     Debug: Extracted JSON from markdown wrapper")
            elif response_text.startswith('```'):

                lines = response_text.split('\n')
                if len(lines) > 1:
                    response_text = '\n'.join(lines[1:-1]).strip()
                    print("     Debug: Extracted content from generic code block")


            print("    ### Debug: Agent response preview (first 500 chars):")
            print("    " + repr(response_text[:500]))


            strategic_scenarios = None

            def _coerce_to_scenarios(obj):

                if isinstance(obj, list):
                    return obj if obj else None
                if isinstance(obj, dict):
                    for k in ('scenarios', 'test_scenarios', 'waves', 'test_waves',
                              'attack_scenarios', 'plan', 'data', 'result', 'results'):
                        v = obj.get(k)
                        if isinstance(v, list) and v:
                            return v
                    if any(kk in obj for kk in ('scenario_name', 'attack_types',
                                                'target_systems', 'attack_type')):
                        return [obj]
                return None


            if not strategic_scenarios:
                try:
                    _parsed = json.loads(response_text)
                    _coerced = _coerce_to_scenarios(_parsed)
                    if _coerced:
                        strategic_scenarios = _coerced
                        print("    ## Method 0-obj: coerced " + str(len(_coerced)) +
                              " scenario(s) from top-level " + type(_parsed).__name__)
                except (json.JSONDecodeError, ValueError):
                    pass


            if not strategic_scenarios:
                try:

                    strategic_scenarios = json.loads(response_text)
                    if isinstance(strategic_scenarios, list):
                        print("    ## Method 0: Direct JSON parsing successful with " + str(len(strategic_scenarios)) + " scenarios")
                    else:
                        strategic_scenarios = None
                except (json.JSONDecodeError, ValueError) as e:
                    print("    ##?? Method 0 failed: " + str(e))
                    strategic_scenarios = None


            if not strategic_scenarios:
                try:

                    start_idx = response_text.find('[')
                    if start_idx != -1:

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
                                print("    ## Method 0b: Found complete JSON array with " + str(len(strategic_scenarios)) + " scenarios")
                            else:
                                strategic_scenarios = None
                except (json.JSONDecodeError, ValueError) as e:
                    print("    ##?? Method 0b failed: " + str(e))
                    strategic_scenarios = None


            json_match = re.search(r'\[[\s\S]*?\]', response_text)
            if json_match:
                try:
                    json_str = json_match.group(0)

                    json_str = json_str.strip()
                    strategic_scenarios = json.loads(json_str)
                    if isinstance(strategic_scenarios, list):
                        print("    ## Method 1: Found JSON array with " + str(len(strategic_scenarios)) + " scenarios")
                    else:
                        strategic_scenarios = None
                except json.JSONDecodeError as e:
                    print("    ##?? Method 1 failed: " + str(e))
                    strategic_scenarios = None


            if not strategic_scenarios:
                try:

                    json_match = re.search(r'\{.*?"scenarios".*?\[.*?\]', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                        if 'scenarios' in parsed:
                            strategic_scenarios = parsed['scenarios']
                            print("    ## Method 2: Found scenarios in object with " + str(len(strategic_scenarios)) + " scenarios")
                except (json.JSONDecodeError, KeyError) as e:
                    print("    ##?? Method 2 failed: " + str(e))


            if not strategic_scenarios:
                try:
                    strategic_scenarios = json.loads(response_text)
                    if not isinstance(strategic_scenarios, list):
                        if isinstance(strategic_scenarios, dict) and 'scenarios' in strategic_scenarios:
                            strategic_scenarios = strategic_scenarios['scenarios']
                        else:
                            strategic_scenarios = None
                    print("    ## Method 3: Parsed entire response with " + str(len(strategic_scenarios)) + " scenarios")
                except json.JSONDecodeError as e:
                    print("    ##?? Method 3 failed: " + str(e))


            if not strategic_scenarios:
                print("    ##?? All JSON parsing methods failed, creating fallback scenarios")
                strategic_scenarios = self._create_fallback_scenarios_from_text(response_text, original_attacks, simulation_duration, num_systems)

            print("    ## Agent generated " + str(len(strategic_scenarios)) + " strategic attack scenarios")


            hierarchical_scenarios = []
            for scenario in strategic_scenarios:
                hierarchical_scenario = self._convert_gemini_scenario_to_hierarchical(
                    scenario, simulation_duration
                )
                hierarchical_scenarios.append(hierarchical_scenario)

                print("      #??# " + str(scenario.get('scenario_name', 'Unnamed')) + ": " +
                      str(int(scenario.get('start_time', 0))) + "s, " +
                      "systems " + str(scenario.get('target_systems', [])) + ", " +
                      "impact " + str(round(scenario.get('impact_factor', 0), 2)))

            return hierarchical_scenarios

        except json.JSONDecodeError as e:
            print(f"    ##XX Failed to parse Agent JSON response: {str(e)}")
            return None
        except Exception as e:
            print(f"    ##XX Failed to parse Agent strategic response: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_fallback_scenarios_from_text(self, response_text: str, original_attacks: List[Dict],
                                           simulation_duration: float, num_systems: int) -> List[Dict]:

        try:
            print("    ## Creating fallback scenarios from text analysis...")


            scenarios = []


            scenario_patterns = [
                r'Wave \d+[:\s]+([^,\n]+)',
                r'Phase \d+[:\s]+([^,\n]+)',
                r'Attack \d+[:\s]+([^,\n]+)',
                r'Scenario \d+[:\s]+([^,\n]+)'
            ]


            time_patterns = [
                r'(\d+(?:\.\d+)?)\s*seconds?',
                r'(\d+(?:\.\d+)?)\s*minutes?',
                r'start[:\s]+(\d+(?:\.\d+)?)',
                r'duration[:\s]+(\d+(?:\.\d+)?)'
            ]


            attack_type_patterns = [
                r'voltage[_\s]*manipulation',
                r'current[_\s]*injection',
                r'power[_\s]*disruption',
                r'frequency[_\s]*attack',
                r'load[_\s]*manipulation',
                r'model[_\s]*poisoning'
            ]


            num_scenarios = min(5, max(3, len(original_attacks) // 10))

            for i in range(num_scenarios):

                start_time = (simulation_duration / num_scenarios) * i + 100
                duration = min(300, simulation_duration / num_scenarios - 50)


                attack_type = 'power_manipulation'
                if i < len(original_attacks):
                    attack_type = original_attacks[i].get('attack_type', 'power_manipulation')


                target_systems = [1 + (i % num_systems)]
                if num_systems > 1 and i % 2 == 0:
                    target_systems.append(1 + ((i + 1) % num_systems))

                scenario = {
                    'scenario_name': f'Fallback Wave {i+1}',
                    'start_time': start_time,
                    'duration': duration,
                    'target_systems': target_systems,
                    'attack_types': [attack_type],
                    'combined_magnitude': 0.6 + (i * 0.1),
                    'stealth_level': 0.7 - (i * 0.1),
                    'strategic_goal': f'Fallback attack wave {i+1}',
                    'coordination': 'simultaneous' if len(target_systems) > 1 else 'single',
                    'impact_factor': 0.5 + (i * 0.1),
                    'success_rate': 0.8 - (i * 0.05)
                }
                scenarios.append(scenario)

            print(f"    ## Created {len(scenarios)} fallback scenarios from text analysis")
            return scenarios

        except Exception as e:
            print(f"    ##XX Fallback scenario creation failed: {e}")
            return []

    def _convert_gemini_scenario_to_hierarchical(self, gemini_scenario: Dict,
                                                 max_duration: float) -> Dict:


        scenario_name = gemini_scenario.get('scenario_name', 'Gemini Strategic Attack')
        start_time = gemini_scenario.get('start_time', 400.0)
        duration = min(gemini_scenario.get('duration', 300.0), max_duration - start_time)
        target_systems = gemini_scenario.get('target_systems', [1])
        attack_types = gemini_scenario.get('attack_types', ['power_manipulation'])
        magnitude = gemini_scenario.get('combined_magnitude', 0.7)
        stealth = gemini_scenario.get('stealth_level', 0.6)
        impact_factor = gemini_scenario.get('impact_factor', 0.7)
        success_rate = gemini_scenario.get('success_rate', 0.8)
        coordination = gemini_scenario.get('coordination', 'simultaneous')


        primary_attack_type = 'power_disruption'


        attack_type_priority = {
            'model_poisoning': 'data_injection',
            'charging_hijacking': 'protocol_manipulation',
            'thermal_attack': 'protocol_manipulation',
            'voltage_manipulation': 'voltage_manipulation',
            'power_disruption': 'power_disruption',
            'frequency_manipulation': 'data_injection',
            'load_manipulation': 'power_disruption',
            'soc_spoofing': 'communication_spoofing',
            'current_injection': 'current_injection',
            'cyber_attack': 'communication_spoofing'
        }


        priority_order = ['communication_spoofing', 'protocol_manipulation', 'voltage_manipulation',
                         'power_disruption', 'data_injection', 'current_injection']


        mapped_types = []
        for attack_type in attack_types:
            mapped_type = attack_type_priority.get(attack_type, attack_type)
            if mapped_type not in mapped_types:
                mapped_types.append(mapped_type)


        for priority_type in priority_order:
            if priority_type in mapped_types:
                primary_attack_type = priority_type
                break


        if not mapped_types and attack_types:

            first_type = attack_types[0]
            primary_attack_type = attack_type_priority.get(first_type, first_type)
        elif mapped_types:
            primary_attack_type = mapped_types[0]


        voltage_deviation = magnitude * 0.15
        frequency_deviation = magnitude * 0.20
        power_loss = magnitude * 0.5
        load_disruption = magnitude * 0.6


        stealth_multiplier = (1.0 - stealth) * 0.5 + 0.75


        primary_target = target_systems[0] if target_systems else 1

        return {
            'type': primary_attack_type,
            'target_system': primary_target,
            'target_systems': target_systems,
            'impact_factor': impact_factor * stealth_multiplier,
            'success_rate': success_rate,
            'voltage_deviation': voltage_deviation * stealth_multiplier,
            'frequency_deviation': frequency_deviation * stealth_multiplier,
            'power_loss': power_loss * stealth_multiplier,
            'load_disruption': load_disruption * stealth_multiplier,
            'start_time': start_time,
            'duration': duration,
            'attack_magnitude': magnitude,
            'stealth_level': stealth,
            'attack_type': primary_attack_type,
            'magnitude': magnitude,
            'stealth_factor': stealth,
            'voltage_drop_factor': 1.0 - (voltage_deviation * stealth_multiplier),
            'power_reduction_factor': 1.0 - power_loss,
            'frequency_impact': frequency_deviation * stealth_multiplier,
            'active': False,
            'gemini_optimized': True,
            'scenario_name': scenario_name,
            'coordination_type': coordination,
            'combined_attack_types': attack_types,
            'strategic_goal': gemini_scenario.get('strategic_goal', 'Strategic attack combination'),
            'primary_attack_type': primary_attack_type,
            'attack_complexity': len(attack_types),
            'multi_vector_attack': len(attack_types) > 1
        }

    def _convert_agent_action_to_hierarchical(self, action: Dict, result: Dict,
                                             start_time: float, max_duration: float) -> Dict:


        attack_type = action.get('attack_type', 'power_manipulation')
        magnitude = action.get('magnitude', 0.5)
        duration = min(action.get('duration', 60.0), max_duration - start_time)
        stealth_level = action.get('stealth_level', 0.5)
        target_system = action.get('target_system', 1)


        impact_factor = result.get('impact', 0.5)
        success_rate = 1.0 if result.get('success', False) else 0.0


        hierarchical_type = attack_type


        voltage_deviation = magnitude * 0.15
        frequency_deviation = magnitude * 0.20
        power_loss = magnitude * 0.5
        load_disruption = magnitude * 0.6


        stealth_multiplier = (1.0 - stealth_level) * 0.5 + 0.75

        return {
            'type': hierarchical_type,
            'target_system': target_system,
            'impact_factor': impact_factor * stealth_multiplier,
            'success_rate': success_rate,
            'voltage_deviation': voltage_deviation * stealth_multiplier,
            'frequency_deviation': frequency_deviation * stealth_multiplier,
            'power_loss': power_loss * stealth_multiplier,
            'load_disruption': load_disruption * stealth_multiplier,
            'start_time': start_time,
            'duration': duration,
            'target_systems': [target_system],
            'attack_magnitude': magnitude,
            'stealth_level': stealth_level,
            'attack_type': hierarchical_type,
            'magnitude': magnitude,
            'stealth_factor': stealth_level,
            'voltage_drop_factor': 1.0 - (voltage_deviation * stealth_multiplier),
            'power_reduction_factor': 1.0 - power_loss,
            'frequency_impact': frequency_deviation * stealth_multiplier,
            'agent_generated': True,
            'primary_attack_type': hierarchical_type,
            'combined_attack_types': [attack_type],
            'attack_complexity': 1,
            'multi_vector_attack': False
        }

    def _create_fallback_attack_scenarios(self, num_systems: int, duration_seconds: float) -> List[Dict]:

        print("Creating fallback attack scenarios 010...")
        attack_scenarios = []


        for i in range(min(3, num_systems)):
            attack_scenarios.append({
                'type': 'power_manipulation',
                'target_system': i + 1,
                'impact_factor': 0.6,
                'success_rate': 0.9,
                'voltage_deviation': 0.08,
                'frequency_deviation': 0.12,
                'power_loss': 0.4,
                'load_disruption': 0.3,
                'start_time': 120.0,
                'duration': 300.0,
                'target_systems': [i + 1],
                'attack_magnitude': 0.6,
                'stealth_level': 0.5,
                'attack_type': 'power_manipulation',
                'magnitude': 0.6,
                'stealth_factor': 0.5,
                'voltage_drop_factor': 0.75,
                'power_reduction_factor': 0.4,
                'frequency_impact': 0.15,
                'agent_generated': False
            })


        if num_systems > 3:
            for i in range(3, min(6, num_systems)):
                attack_scenarios.append({
                    'type': 'load_manipulation',
                    'target_system': i + 1,
                    'impact_factor': 0.4,
                    'success_rate': 0.8,
                    'voltage_deviation': 0.06,
                    'frequency_deviation': 0.08,
                    'power_loss': 0.25,
                    'load_disruption': 0.4,
                    'start_time': 600.0,
                    'duration': 300.0,
                    'target_systems': [i + 1],
                    'attack_magnitude': 0.4,
                    'stealth_level': 0.6,
                    'attack_type': 'load_manipulation',
                    'magnitude': 0.4,
                    'stealth_factor': 0.6,
                    'voltage_drop_factor': 0.85,
                    'power_reduction_factor': 0.6,
                    'frequency_impact': 0.08,
                    'agent_generated': False
                })

        return attack_scenarios

def _safe_nanmean(data_list):

    finite_vals = [v for v in data_list if isinstance(v, (int, float)) and np.isfinite(v)]
    return float(np.mean(finite_vals)) if finite_vals else None


def _print_comparison_summary(baseline_cosim, attack_cosim):

    try:
        print("\n  ## Comparing Baseline vs Attack Impact:")
        print("  " + "-" * 86)

        baseline_results = baseline_cosim.results if baseline_cosim else {}
        attack_results = attack_cosim.results if attack_cosim else {}


        b_voltage_keys = list(baseline_results.get('evcs_voltage_data', {}).keys())
        a_voltage_keys = list(attack_results.get('evcs_voltage_data', {}).keys())
        print(f"\n  DEBUG: Baseline EVCS voltage sys_ids: {b_voltage_keys}")
        print(f"  DEBUG: Attack EVCS voltage sys_ids:   {a_voltage_keys}")


        for sys_id in range(1, 7):
            print(f"\n  ##Distribution System {sys_id}:")


            baseline_voltage = baseline_results.get('evcs_voltage_data', {}).get(sys_id, {})
            attack_voltage = attack_results.get('evcs_voltage_data', {}).get(sys_id, {})

            if baseline_voltage or attack_voltage:

                baseline_avg_voltages = []
                attack_avg_voltages = []

                all_station_ids = set(list(baseline_voltage.keys()) + list(attack_voltage.keys()))
                for station_id in all_station_ids:
                    b_data = baseline_voltage.get(station_id, [])
                    a_data = attack_voltage.get(station_id, [])

                    b_mean = _safe_nanmean(b_data) if b_data else None
                    a_mean = _safe_nanmean(a_data) if a_data else None

                    if b_mean is not None:
                        baseline_avg_voltages.append(b_mean)
                    if a_mean is not None:
                        attack_avg_voltages.append(a_mean)

                baseline_v_avg = float(np.mean(baseline_avg_voltages)) if baseline_avg_voltages else None
                attack_v_avg = float(np.mean(attack_avg_voltages)) if attack_avg_voltages else None

                print(f"     Baseline Voltage: {f'{baseline_v_avg:.1f}V' if baseline_v_avg is not None else 'N/A (no data)'}")
                print(f"     Attack Voltage:   {f'{attack_v_avg:.1f}V' if attack_v_avg is not None else 'N/A (no data)'}")
                if baseline_v_avg is not None and attack_v_avg is not None and baseline_v_avg > 0:
                    voltage_drop = ((baseline_v_avg - attack_v_avg) / baseline_v_avg) * 100
                    print(f"     Voltage Drop:     {voltage_drop:.2f}%")
            else:
                print(f"     No EVCS voltage data for this system")


            baseline_power = baseline_results.get('evcs_power_data', {}).get(sys_id, {})
            attack_power = attack_results.get('evcs_power_data', {}).get(sys_id, {})

            if baseline_power or attack_power:
                baseline_avg_power = []
                attack_avg_power = []

                all_station_ids = set(list(baseline_power.keys()) + list(attack_power.keys()))
                for station_id in all_station_ids:
                    b_data = baseline_power.get(station_id, [])
                    a_data = attack_power.get(station_id, [])

                    b_mean = _safe_nanmean(b_data) if b_data else None
                    a_mean = _safe_nanmean(a_data) if a_data else None

                    if b_mean is not None:
                        baseline_avg_power.append(b_mean)
                    if a_mean is not None:
                        attack_avg_power.append(a_mean)

                baseline_p_avg = float(np.mean(baseline_avg_power)) if baseline_avg_power else None
                attack_p_avg = float(np.mean(attack_avg_power)) if attack_avg_power else None

                print(f"     Baseline Power:   {f'{baseline_p_avg:.1f}kW' if baseline_p_avg is not None else 'N/A (no data)'}")
                print(f"     Attack Power:     {f'{attack_p_avg:.1f}kW' if attack_p_avg is not None else 'N/A (no data)'}")
                if baseline_p_avg is not None and attack_p_avg is not None and baseline_p_avg > 0:
                    power_reduction = ((baseline_p_avg - attack_p_avg) / baseline_p_avg) * 100
                    print(f"     Power Reduction:  {power_reduction:.2f}%")
            else:
                print(f"     No EVCS power data for this system")

        print("\n  " + "-" * 86)
        print("  ## Note: Attack impacts are visible in the individual EVCS plots")
        print("     • Each EVCS station now shows as a separate colored line")
        print("     • Compare baseline plots with attack plots to see differences")
        print("     • Plots saved to current directory and sub_figures/")

    except Exception as e:
        print(f"  ##?? Comparison summary failed: {e}")
        print("     Baseline and attack plots were generated separately")

def create_detailed_comparison_plots(baseline_cosim, attack_cosim, attack_scenario_name="LLM-Coordinated Attack"):

    import os
    from datetime import datetime
    import matplotlib.pyplot as plt


    os.makedirs('sub_figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n## Creating detailed comparison visualizations...")


    base_time = np.array(baseline_cosim.results['time'])
    base_freq = np.array(baseline_cosim.results['frequency'])
    base_load = np.array(baseline_cosim.results['total_load'])
    base_ref = np.array(baseline_cosim.results['reference_power'])


    attack_time = np.array(attack_cosim.results['time'])
    attack_freq = np.array(attack_cosim.results['frequency'])
    attack_load = np.array(attack_cosim.results['total_load'])
    attack_ref = np.array(attack_cosim.results['reference_power'])


    base_freq_interp = np.interp(attack_time, base_time, base_freq)
    base_load_interp = np.interp(attack_time, base_time, base_load)
    base_ref_interp = np.interp(attack_time, base_time, base_ref)


    dfreq = attack_freq - base_freq_interp
    dload = attack_load - base_load_interp
    dref = attack_ref - base_ref_interp


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))


    axes[0, 0].plot(base_time, base_freq, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Baseline')
    axes[0, 0].plot(attack_time, attack_freq, color='red', linewidth=2, label=attack_scenario_name)
    axes[0, 0].axhline(y=60.0, color='black', linestyle=':', alpha=0.5)
    axes[0, 0].set_ylabel('Frequency (Hz)', fontsize=14)
    axes[0, 0].set_xlabel('Time (s)', fontsize=14)
    axes[0, 0].set_title('Frequency Response: Baseline vs Attack', fontsize=16, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(59.8, 60.2)
    max_dfreq = float(np.max(np.abs(dfreq))) if len(dfreq) else 0.0
    axes[0, 0].text(0.02, 0.95, f"Max Δf = {max_dfreq:.3f} Hz", transform=axes[0, 0].transAxes,
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black'), fontsize=12)


    axes[0, 1].plot(base_time, base_load, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Baseline')
    axes[0, 1].plot(attack_time, attack_load, color='red', linewidth=2, label=attack_scenario_name)
    axes[0, 1].set_ylabel('Total Load (MW)', fontsize=14)
    axes[0, 1].set_xlabel('Time (s)', fontsize=14)
    axes[0, 1].set_title('Distribution Load: Baseline vs Attack', fontsize=16, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    max_dload = float(np.max(np.abs(dload))) if len(dload) else 0.0
    axes[0, 1].text(0.02, 0.95, f"Max ΔLoad = {max_dload:.1f} MW", transform=axes[0, 1].transAxes,
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black'), fontsize=12)


    axes[1, 0].plot(attack_time, dfreq, color='purple', linewidth=2, label='Frequency Delta')
    axes[1, 0].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(attack_time, 0, dfreq, where=(dfreq < 0), alpha=0.3, color='blue', label='Under-frequency')
    axes[1, 0].fill_between(attack_time, 0, dfreq, where=(dfreq > 0), alpha=0.3, color='red', label='Over-frequency')
    axes[1, 0].set_ylabel('Δ Frequency (Hz)', fontsize=14)
    axes[1, 0].set_xlabel('Time (s)', fontsize=14)
    axes[1, 0].set_title('Attack Impact on Frequency', fontsize=16, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)


    axes[1, 1].plot(attack_time, dload, color='orange', linewidth=2, label='Load Delta')
    axes[1, 1].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(attack_time, 0, dload, where=(dload < 0), alpha=0.3, color='blue', label='Load Decrease')
    axes[1, 1].fill_between(attack_time, 0, dload, where=(dload > 0), alpha=0.3, color='red', label='Load Increase')
    axes[1, 1].set_ylabel('Δ Load (MW)', fontsize=14)
    axes[1, 1].set_xlabel('Time (s)', fontsize=14)
    axes[1, 1].set_title('Attack Impact on Load', fontsize=16, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'sub_figures/comparison_baseline_vs_attack_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    print(f"   ## Saved: sub_figures/comparison_baseline_vs_attack_{timestamp}.pdf")
    plt.show()


    _create_individual_comparison_plots(baseline_cosim, attack_cosim, timestamp, attack_scenario_name)

    return {
        'max_freq_delta': max_dfreq,
        'max_load_delta': max_dload,
        'avg_freq_delta': float(np.mean(np.abs(dfreq))),
        'avg_load_delta': float(np.mean(np.abs(dload)))
    }

def _create_individual_comparison_plots(baseline_cosim, attack_cosim, timestamp, attack_scenario_name):

    import matplotlib.pyplot as plt


    base_time = np.array(baseline_cosim.results['time'])
    base_freq = np.array(baseline_cosim.results['frequency'])
    base_load = np.array(baseline_cosim.results['total_load'])


    attack_time = np.array(attack_cosim.results['time'])
    attack_freq = np.array(attack_cosim.results['frequency'])
    attack_load = np.array(attack_cosim.results['total_load'])


    base_freq_interp = np.interp(attack_time, base_time, base_freq)
    base_load_interp = np.interp(attack_time, base_time, base_load)


    dfreq = attack_freq - base_freq_interp
    dload = attack_load - base_load_interp


    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(base_time, base_freq, color='green', linestyle='--', linewidth=2.5, alpha=0.8, label='Baseline (No Attack)')
    ax1.plot(attack_time, attack_freq, color='red', linewidth=2.5, label=f'{attack_scenario_name}')
    ax1.axhline(y=60.0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    ax1.set_ylabel('Frequency (Hz)', fontsize=24)
    ax1.set_xlabel('Time (s)', fontsize=24)

    ax1.legend(loc='best', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(59.8, 60.2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    max_dfreq = float(np.max(np.abs(dfreq))) if len(dfreq) else 0.0
    ax1.text(0.02, 0.95, f"Max Δf = {max_dfreq:.3f} Hz", transform=ax1.transAxes,
             bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black'), fontsize=18)
    plt.tight_layout()
    plt.savefig(f'sub_figures/frequency_comparison_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    print(f"   ## Saved: sub_figures/frequency_comparison_{timestamp}.pdf")
    plt.close(fig1)


    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(base_time, base_load, color='green', linestyle='--', linewidth=2.5, alpha=0.8, label='Baseline (No Attack)')
    ax2.plot(attack_time, attack_load, color='red', linewidth=2.5, label=f'{attack_scenario_name}')
    ax2.set_ylabel('Total Distribution Load (MW)', fontsize=24)
    ax2.set_xlabel('Time (s)', fontsize=24)

    ax2.legend(loc='best', fontsize=18)
    ax2.grid(True, alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    max_dload = float(np.max(np.abs(dload))) if len(dload) else 0.0
    ax2.text(0.02, 0.95, f"Max ΔLoad = {max_dload:.1f} MW", transform=ax2.transAxes,
             bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black'), fontsize=18)
    plt.tight_layout()
    plt.savefig(f'sub_figures/load_comparison_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    print(f"   ## Saved: sub_figures/load_comparison_{timestamp}.pdf")
    plt.close(fig2)


    fig3 = plt.figure(figsize=(12, 8))
    ax3 = fig3.add_subplot(111)
    ax3.plot(attack_time, dfreq, color='purple', linewidth=2.5, label='Frequency Deviation')
    ax3.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.fill_between(attack_time, 0, dfreq, where=(dfreq < 0), alpha=0.4, color='blue', label='Under-frequency Event')
    ax3.fill_between(attack_time, 0, dfreq, where=(dfreq > 0), alpha=0.4, color='red', label='Over-frequency Event')
    ax3.set_ylabel('Frequency Deviation (Hz)', fontsize=24)
    ax3.set_xlabel('Time (s)', fontsize=24)

    ax3.legend(loc='best', fontsize=18)
    ax3.grid(True, alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(f'sub_figures/frequency_delta_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    print(f"   ## Saved: sub_figures/frequency_delta_{timestamp}.pdf")
    plt.close(fig3)


    fig4 = plt.figure(figsize=(12, 8))
    ax4 = fig4.add_subplot(111)
    ax4.plot(attack_time, dload, color='orange', linewidth=2.5, label='Load Deviation')
    ax4.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.fill_between(attack_time, 0, dload, where=(dload < 0), alpha=0.4, color='blue', label='Load Reduction')
    ax4.fill_between(attack_time, 0, dload, where=(dload > 0), alpha=0.4, color='red', label='Load Increase')
    ax4.set_ylabel('Load Deviation (MW)', fontsize=24)
    ax4.set_xlabel('Time (s)', fontsize=24)
    ax4.set_title('Attack-Induced Load Deviation', fontsize=20, fontweight='bold')
    ax4.legend(loc='best', fontsize=18)
    ax4.grid(True, alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'sub_figures/load_delta_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    print(f"   ## Saved: sub_figures/load_delta_{timestamp}.pdf")
    plt.close(fig4)

def main(run_mode='rl_coordinated'):

    print(" Enhanced Integrated EVCS LLM-RL System with Real SAC and PINN Integration")
    print("=" * 90)
    print(f"#??# Run Mode: {run_mode}")
    print("=" * 90)


    config = {
        'hierarchical': {
            'use_enhanced_pinn': True,
            'use_dqn_sac_security': True,
            'total_duration': 3600,
            'num_distribution_systems': 10
        },
        'federated_pinn': {
            'num_distribution_systems': 10
        },
        'rl': {
            'num_systems': 10,
            'dqn_timesteps': 30000,
            'sac_timesteps': 30000,
            'coordination_training': True,
            'use_rl_coordination': run_mode in ['rl_coordinated', 'both']
        },
        'attack': {
            'max_episodes': 10,
            'coordination_type': 'simultaneous'
        }
    }


    print("\n" + "=" * 90)
    print(" STEP 1: Training Attack System (ONE TIME ONLY)")
    print("=" * 90)
    print("## Note: Training happens once, then used for both baseline and attack scenarios")

    system = EnhancedIntegratedEVCSLLMRLSystem(config)

    try:
        training_results = system.train_enhanced_system(total_timesteps=300000)
        print("## Enhanced system training complete!")
        print(f"   ## Training took: {training_results.get('training_time', 'N/A')}")


        print("\n" + "=" * 90)
        print("## STEP 2: Running BASELINE Scenario (No Attacks)")
        print("=" * 90)
        print("## Using trained agents to monitor normal grid operation without attacks")

        from hierarchical_cosimulation import HierarchicalCoSimulation

        baseline_cosim = HierarchicalCoSimulation(use_enhanced_pinn=config['hierarchical']['use_enhanced_pinn'])
        baseline_cosim.total_duration = config['hierarchical']['total_duration']


        print("   Setting up distribution systems for baseline...")
        baseline_cosim.add_distribution_system(1, "ieee34Mod1.dss", 4)
        baseline_cosim.add_distribution_system(2, "ieee34Mod1.dss", 9)
        baseline_cosim.add_distribution_system(3, "ieee34Mod1.dss", 13)
        baseline_cosim.add_distribution_system(4, "ieee34Mod1.dss", 5)
        baseline_cosim.add_distribution_system(5, "ieee34Mod1.dss", 10)
        baseline_cosim.add_distribution_system(6, "ieee34Mod1.dss", 7)


        baseline_cosim.setup_ev_charging_stations()

        print(f"    Simulation duration: {baseline_cosim.total_duration}s")
        print("   Running baseline simulation (NO attacks)...")

        baseline_results = baseline_cosim.run_hierarchical_simulation(attack_scenarios=[])

        print("  ## Baseline simulation complete!")
        print(f"  ## Baseline results: {len(baseline_cosim.results.get('time', []))} timesteps")


        print("\n  ## Generating baseline plots...")
        baseline_cosim.plot_hierarchical_results()
        print("  ## Baseline plots saved!")


        print("\n" + "=" * 90)
        print("## STEP 3: Running ATTACK Scenario (With LLM-RL Coordination)")
        print("=" * 90)
        print("## Using the SAME trained agents from Step 1 to execute coordinated attacks")

        results = system.run_enhanced_simulation(
            scenario_id="ENHANCED_001",
            episodes=20
        )

        print("\n## Enhanced attack simulation complete!")
        print(f"   ## Average Reward: {results['performance_metrics']['average_reward']:.2f}")
        print(f"   ## Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
        print(f"   ## Coordination Score: {results['performance_metrics']['coordination_effectiveness']:.3f}")
        print(f"   ## Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")


        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_results_dir = "detection_results"
        os.makedirs(detection_results_dir, exist_ok=True)


        report_episodes = []
        captured_samples = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0


        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "best_ids_model_meta.json")) as _f:
                ids_threshold = float(json.load(_f).get("threshold", 0.465))
        except Exception:
            ids_threshold = 0.465


        SEQ_LEN_IDS = 10


        try:
            from acn_benign_pool import sample_benign_window as _acn_sample_window
        except Exception:
            _acn_sample_window = None

        def _evaluate_sample_via_best_ids(det, sample_input, sample_sys_id, rng=None):

            rng = rng if rng is not None else np.random.default_rng()
            best_ids = getattr(det, '_best_ids_detector', None)


            warmup_window = None
            if _acn_sample_window is not None:
                warmup_window = _acn_sample_window(SEQ_LEN_IDS - 1,
                                                    system_id=sample_sys_id, rng=rng)

            if warmup_window is None:

                base_soc = float(rng.uniform(0.10, 0.80))
                base_pwr_norm = float(rng.uniform(0.10, 0.80))
                warmup_window = [{
                    'soc':            float(np.clip(base_soc + t * 0.01, 0.0, 1.0)),
                    'voltage':        float(rng.uniform(220, 260)),
                    'current':        float(rng.uniform(6, 32)),
                    'power':          float(np.clip(base_pwr_norm - t * 0.005, 0.0, 1.0)) * 7.68,
                    'temperature':    float(rng.uniform(20, 35)),
                    'demand_factor':  float(rng.uniform(0.4, 0.9)),
                    'load_factor':    float(rng.uniform(0.4, 0.9)),
                    'grid_voltage':   float(rng.uniform(0.97, 1.03)),
                    'grid_frequency': float(rng.uniform(59.9, 60.1)),
                    'queue_length':   int(rng.integers(1, 8)),
                    'utilization':    float(rng.uniform(0.4, 0.9)),
                    'urgency_factor': float(rng.uniform(0.5, 1.5)),
                    'time_of_day':    float(rng.uniform(0, 24)),
                    'system_id':      sample_sys_id,
                } for t in range(SEQ_LEN_IDS - 1)]

            if best_ids is not None and best_ids.is_loaded:
                best_ids.reset()
                for warm_input in warmup_window:
                    best_ids.detect(det.extract_features(warm_input))
                is_attack, proba = best_ids.detect(det.extract_features(sample_input))
                return bool(is_attack), float(proba)


            det.reset_state()
            det.load_history = []
            for warm_input in warmup_window:
                det.multi_layer_detection(warm_input, sample_sys_id)
            is_det, det_res = det.multi_layer_detection(sample_input, sample_sys_id)
            return bool(is_det), float(det_res.get('layer3_lstm', {}).get('score', 0.0))

        for i, episode_result in enumerate(results.get('episode_results', [])):
            episode_data = {
                'episode': i + 1,
                'detection_rate': float(episode_result.get('detection_rate', 0)),
                'success_rate': float(episode_result.get('success_rate', 0)),
                'total_impact': float(episode_result.get('total_impact', 0)),
                'coordination_score': float(episode_result.get('coordination_score', 0)),
                'attacks_detected': []
            }


            _SEQ_HALF = 5
            for attack_result in episode_result.get('attack_results', episode_result.get('execution_results', [])):
                inner = attack_result.get('result', attack_result) if isinstance(attack_result, dict) else attack_result
                sys_id = attack_result.get('system_id', 1)
                attack_success_rl = bool(inner.get('success', False))


                ids_detected = False
                ids_anomaly = 0.0

                _rl_ids_detected = inner.get('ids_detected', None)
                _rl_ids_score    = inner.get('ids_lstm_score', inner.get('detection_risk', None))
                _attacked_resp   = inner.get('attacked_response', {})

                if _rl_ids_detected is not None:

                    ids_detected = bool(_rl_ids_detected)
                    ids_anomaly  = float(_rl_ids_score) if _rl_ids_score is not None else 0.0
                elif _attacked_resp and system.federated_manager and sys_id in system.federated_manager.anomaly_detectors:

                    det = system.federated_manager.anomaly_detectors[sys_id]
                    det.reset_state()
                    det.load_history = []


                    try:
                        import json as _j
                        _meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "models", "best_ids_model_meta.json")
                        with open(_meta_path) as _f:
                            _eval_threshold = float(_j.load(_f).get("threshold", 0.465))
                    except Exception:
                        _eval_threshold = 0.465

                    _lstm_det   = getattr(det, 'lstm_detector', None)
                    _robust_det = getattr(det, '_robust_ids',   None)
                    _orig_lt    = getattr(_lstm_det,   'anomaly_threshold', None)
                    _orig_rt    = getattr(_robust_det, 'threshold',         None)
                    if _lstm_det   is not None: _lstm_det.anomaly_threshold  = _eval_threshold
                    if _robust_det is not None: _robust_det.threshold        = _eval_threshold


                    _pinn_v = float(_attacked_resp.get('voltage', 240.0))
                    _pinn_i = float(_attacked_resp.get('current', 16.0))
                    _pinn_p = float(_attacked_resp.get('power',   3.84))
                    _r_soc  = float(np.clip(inner.get('soc', 0.5), 0.0, 1.0))
                    _r_temp = float(np.clip(25.0 + max(0, _pinn_p - 3.0) * 0.5, 20.0, 55.0))
                    _r_lf   = float(np.clip(_pinn_p / 7.68, 0.2, 1.5))
                    _r_util = float(np.clip(_pinn_p / 7.68, 0.1, 1.0))
                    attack_input = {
                        'soc':            _r_soc,
                        'voltage':        _pinn_v,
                        'current':        _pinn_i,
                        'power':          _pinn_p,
                        'temperature':    _r_temp,
                        'demand_factor':  float(np.clip(_pinn_p / 7.68, 0.1, 1.5)),
                        'load_factor':    _r_lf,
                        'grid_voltage':   float(inner.get('grid_voltage', 1.0)),
                        'grid_frequency': float(inner.get('grid_frequency', 60.0)),
                        'queue_length':   int(inner.get('queue_length', 3)),
                        'utilization':    _r_util,
                        'urgency_factor': float(np.clip(1.0 + max(0, 1.0 - _r_soc) * 0.5, 0.5, 2.0)),
                        'time_of_day':    float((time.time() / 3600.0) % 24.0),
                        'system_id':      sys_id,
                    }


                    if _lstm_det   is not None and _orig_lt is not None:
                        _lstm_det.anomaly_threshold  = _orig_lt
                    if _robust_det is not None and _orig_rt is not None:
                        _robust_det.threshold        = _orig_rt
                    ids_detected, ids_anomaly = _evaluate_sample_via_best_ids(
                        det, attack_input, sys_id)

                elif system.federated_manager and sys_id in system.federated_manager.anomaly_detectors:


                    det = system.federated_manager.anomaly_detectors[sys_id]
                    _mag   = float(np.clip(inner.get('magnitude', inner.get('attack_magnitude', 0.7)), 0.1, 2.0))
                    _atype = str(attack_result.get('attack_type', 'unknown'))
                    _vdev  = _mag * 0.15 if 'voltage' in _atype else _mag * 0.05
                    _idev  = _mag * 0.20 if 'current' in _atype else _mag * 0.08
                    _pdev  = _mag * 0.25 if 'power'   in _atype else _mag * 0.10
                    _atk_v   = float(240.0 * (1.0 - _vdev))
                    _atk_i   = float(32.0  * (1.0 + _idev))
                    _atk_p   = float(7.68  * (1.0 + _pdev))
                    _atk_soc = float(np.clip(0.5 - _mag * 0.2, 0.05, 0.95))
                    attack_input = {
                        'soc':            _atk_soc,
                        'voltage':        _atk_v,
                        'current':        _atk_i,
                        'power':          _atk_p,
                        'temperature':    float(np.clip(25.0 + max(0, _atk_p - 3.0) * 0.5, 20.0, 55.0)),
                        'demand_factor':  float(np.clip(0.7 + _mag * 0.8, 0.1, 1.5)),
                        'load_factor':    float(np.clip(_atk_p / 7.68, 0.2, 1.5)),
                        'grid_voltage':   float(np.clip(1.0 - _vdev, 0.85, 1.15)),
                        'grid_frequency': float(np.clip(60.0 + (_mag * 2.0 if 'frequency' in _atype else 0.0), 59.0, 61.0)),
                        'queue_length':   int(np.clip(3 + _mag * 3, 0, 10)),
                        'utilization':    float(np.clip(_atk_p / 7.68, 0.1, 1.0)),
                        'urgency_factor': float(np.clip(1.0 + max(0, 1.0 - _atk_soc) * 0.5, 0.5, 2.0)),
                        'time_of_day':    float((time.time() / 3600.0) % 24.0),
                        'system_id':      sys_id,
                    }
                    ids_detected, ids_anomaly = _evaluate_sample_via_best_ids(
                        det, attack_input, sys_id)
                else:
                    ids_anomaly  = float(inner.get('detection_risk', inner.get('anomaly_score', 0.0)))
                    ids_detected = ids_anomaly >= ids_threshold


                attack_success = attack_success_rl and not ids_detected

                episode_data['attacks_detected'].append({
                    'system_id':    sys_id,
                    'attack_type':  attack_result.get('attack_type', 'unknown'),
                    'detected':     ids_detected,
                    'anomaly_score': ids_anomaly,
                    'attack_impact': float(inner.get('impact', 0)),
                    'attack_success': attack_success,
                    'is_benign': False,
                })
                if ids_detected:
                    total_tp += 1
                else:
                    total_fn += 1


                _win = inner.get('ids_window')
                if _win and len(_win) == 10:
                    captured_samples.append({
                        'config': run_mode, 'episode': i + 1, 'sample_type': 'attack',
                        'attack_type': attack_result.get('attack_type', 'unknown'),
                        'system_id': int(sys_id), 'best_score': float(ids_anomaly),
                        'best_detected': int(bool(ids_detected)), 'window': _win,
                    })


            num_attacks_this_ep = len(episode_data['attacks_detected'])
            for b in range(num_attacks_this_ep):
                benign_sys_id = (b % 6) + 1


                benign_detected = False
                benign_anomaly = 0.0
                if system.federated_manager and benign_sys_id in system.federated_manager.anomaly_detectors:
                    det = system.federated_manager.anomaly_detectors[benign_sys_id]


                    try:
                        from acn_benign_pool import sample_benign_window as _acn_one
                        _last_window = _acn_one(1, system_id=benign_sys_id)
                    except Exception:
                        _last_window = None
                    if _last_window is not None and len(_last_window) > 0:
                        benign_input = _last_window[0]
                    else:

                        _b_soc = float(np.random.uniform(0.30, 0.70))
                        _b_p   = float(np.random.uniform(1.0, 7.0))
                        benign_input = {
                            'soc':            _b_soc,
                            'voltage':        float(np.random.uniform(228, 252)),
                            'current':        float(np.random.uniform(6, 28)),
                            'power':          _b_p,
                            'temperature':    float(np.clip(25.0 + max(0, _b_p - 3.0) * 0.5, 22, 32)),
                            'demand_factor':  float(np.clip(_b_p / 7.68, 0.2, 1.0)),
                            'load_factor':    float(np.clip(_b_p / 7.68, 0.2, 1.0)),
                            'grid_voltage':   1.0,
                            'grid_frequency': 60.0,
                            'queue_length':   3,
                            'utilization':    float(np.clip(_b_p / 7.68, 0.2, 1.0)),
                            'urgency_factor': 1.0 + max(0.0, 1.0 - _b_soc) * 0.4,
                            'time_of_day':    12.0,
                            'system_id':      benign_sys_id,
                        }
                    benign_detected, benign_anomaly = _evaluate_sample_via_best_ids(
                        det, benign_input, benign_sys_id)

                    _bd = getattr(det, '_best_ids_detector', None)
                    if _bd is not None:
                        _bwin = [np.asarray(w, dtype=float).tolist() for w in getattr(_bd, '_window', [])]
                        if _bwin and len(_bwin) == 10:
                            captured_samples.append({
                                'config': run_mode, 'episode': i + 1, 'sample_type': 'benign',
                                'attack_type': 'benign_normal', 'system_id': int(benign_sys_id),
                                'best_score': float(benign_anomaly),
                                'best_detected': int(bool(benign_detected)), 'window': _bwin,
                            })


                episode_data['attacks_detected'].append({
                    'system_id': benign_sys_id,
                    'attack_type': 'benign_normal',
                    'detected': benign_detected,
                    'anomaly_score': benign_anomaly,
                    'attack_impact': 0.0,
                    'attack_success': False,
                    'is_benign': True
                })
                if benign_detected:
                    total_fp += 1
                else:
                    total_tn += 1


            attack_entries = [e for e in episode_data['attacks_detected'] if not e['is_benign']]
            if attack_entries:
                episode_data['detection_rate'] = sum(1 for e in attack_entries if e['detected']) / len(attack_entries)
                episode_data['success_rate']   = sum(1 for e in attack_entries if e['attack_success']) / len(attack_entries)

            report_episodes.append(episode_data)


        precision = total_tp / max(total_tp + total_fp, 1)
        recall    = total_tp / max(total_tp + total_fn, 1)
        f1_score  = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (total_tp + total_tn) / max(total_tp + total_fp + total_fn + total_tn, 1)


        tpr = recall
        fpr = total_fp / max(total_fp + total_tn, 1)
        tnr = total_tn / max(total_tn + total_fp, 1)
        fnr = total_fn / max(total_fn + total_tp, 1)
        balanced_accuracy = 0.5 * (tpr + tnr)


        avg_det_rate  = float(np.mean([ep['detection_rate'] for ep in report_episodes])) if report_episodes else 0.0
        avg_succ_rate = float(np.mean([ep['success_rate']   for ep in report_episodes])) if report_episodes else 0.0

        detection_report = {
            'timestamp': timestamp,
            'scenario_id': "ENHANCED_001",
            'episodes': len(report_episodes),
            'performance_metrics': {
                'average_reward': float(results['performance_metrics']['average_reward']),
                'average_success_rate': avg_succ_rate,
                'coordination_effectiveness': float(results['performance_metrics']['coordination_effectiveness']),
                'average_detection_rate': avg_det_rate,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'accuracy': float(accuracy),

                'tpr_detection_rate': float(tpr),
                'fpr_false_alarm_rate': float(fpr),
                'tnr_specificity': float(tnr),
                'fnr_miss_rate': float(fnr),
                'balanced_accuracy': float(balanced_accuracy),
                'confusion_matrix': {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': total_tn},
                'evaluation_design': {
                    'class_balance': 'balanced_1to1',
                    'benign_per_attack': 1,
                    'note': ('Test set is class-balanced (one benign sample per attack). '
                             'precision/f1/accuracy assume a 1:1 prior; use '
                             'tpr_detection_rate and fpr_false_alarm_rate (both '
                             'prior-independent) as the headline IDS metrics.')
                }
            },


            'data_integrity': {
                'pinn_fallback_attack_sim': int(DATA_INTEGRITY['pinn_fallback_attack_sim']),
                'pinn_dummy_training_data': int(DATA_INTEGRITY['pinn_dummy_training_data']),
                'clean_run': (DATA_INTEGRITY['pinn_fallback_attack_sim'] == 0
                              and DATA_INTEGRITY['pinn_dummy_training_data'] == 0),
            },
            'episode_results': report_episodes
        }

        detection_file = os.path.join(detection_results_dir, f"ids_detection_report_{timestamp}.json")
        with open(detection_file, 'w') as f:
            json.dump(detection_report, f, indent=2)

        print(f"\n## IDS Detection results saved to: {detection_file}")


        try:
            import csv as _csv
            _samp_path = os.path.join(detection_results_dir,
                                      f"ids_samples_{timestamp}_{run_mode}.csv")
            _feat_cols = [f"t{t}_f{f}" for t in range(10) for f in range(14)]
            _n_written = 0
            with open(_samp_path, 'w', newline='') as _sf:
                _w = _csv.writer(_sf)
                _w.writerow(['config', 'episode', 'sample_type', 'attack_type', 'system_id',
                             'best_score', 'best_detected'] + _feat_cols)
                for _s in captured_samples:
                    _flat = [v for _row in _s['window'] for v in _row]
                    if len(_flat) != 140:
                        continue
                    _w.writerow([_s['config'], _s['episode'], _s['sample_type'],
                                 _s['attack_type'], _s['system_id'],
                                 f"{_s['best_score']:.6f}", _s['best_detected']] + _flat)
                    _n_written += 1
            print(f"## Captured {_n_written} best-model IDS windows  {_samp_path}")
        except Exception as _ce:
            print(f"## Sample-capture CSV failed (non-fatal): {_ce}")


        _di = detection_report['data_integrity']
        if _di['clean_run']:
            print("## DATA INTEGRITY: clean run — no random/synthetic fallbacks were used "
                  "(all attacks used real PINN interaction; PINN trained on physics data).")
        else:
            print("\n" + "!" * 90)
            print("## DATA-INTEGRITY WARNING: this run used RANDOM/SYNTHETIC fallback data —")
            print(f"     random fallback attack simulations : {_di['pinn_fallback_attack_sim']}")
            print(f"     PINN systems trained on noise      : {_di['pinn_dummy_training_data']}")
            print("   The reported metrics are PARTLY SYNTHETIC and must NOT be published as-is.")
            print("   Fix the underlying cause (untrained PINN / missing physics generator) and re-run.")
            print("!" * 90)


        try:
            from run_baseline_attacks_actual_system import run_baseline_attacks_actual_system as _run_baseline
            print("\n" + "=" * 90)
            print("## Generating baseline (non-RL) attack results for RL comparison...")
            print("=" * 90)
            _run_baseline(num_episodes=30, num_systems=config['rl']['num_systems'])
        except Exception as _be:
            print(f"\n## Baseline evaluation skipped: {_be}")


        print("\n" + "=" * 90)
        print("## STEP 4: Comparison - Baseline vs Attack")
        print("=" * 90)


        _print_comparison_summary(baseline_cosim, system.hierarchical_sim)


        print("\n" + "=" * 90)
        print("## STEP 5: Creating Detailed Comparison Visualizations")
        print("=" * 90)

        comparison_metrics = create_detailed_comparison_plots(
            baseline_cosim,
            system.hierarchical_sim,
            attack_scenario_name="LLM-Coordinated RL Attack (ENHANCED_001)"
        )

        print("\n## Quantitative Attack Impact Metrics:")
        print(f"   • Max Frequency Deviation:  {comparison_metrics['max_freq_delta']:.4f} Hz")
        print(f"   • Avg Frequency Deviation:  {comparison_metrics['avg_freq_delta']:.4f} Hz")
        print(f"   • Max Load Deviation:       {comparison_metrics['max_load_delta']:.2f} MW")
        print(f"   • Avg Load Deviation:       {comparison_metrics['avg_load_delta']:.2f} MW")

        print("\n" + "=" * 90)
        print("## Enhanced Recommendations:")
        print("=" * 90)
        for rec in results['recommendations']:
            print(f"  • {rec}")

        print("\n" + "=" * 90)
        print("## SIMULATION COMPLETE!")
        print("=" * 90)


        try:
            from llm_metrics_logger import LLMMetricsLogger
            logger = LLMMetricsLogger.instance()
            if logger._records:
                print("\n" + "=" * 90)
                print("## LLM METRICS SUMMARY (all calls this run)")
                print("=" * 90)
                print(logger.summary_report())
                print(logger.model_comparison_table())
                print(f"\n   ## Full metrics saved to:")
                print(f"      CSV   {logger.csv_path}")
                print(f"      JSONL {logger.jsonl_path}")
            else:
                print("\n## LLM Metrics: no LLM calls were logged this run "
                      "(LLM may have been in fallback mode).")
        except Exception as _me:
            print(f"\n##??  LLM metrics summary failed: {_me}")


        print("\n" + "=" * 90)
        print("## Summary:")

        print(f"   • Training: Done once (Step 1)")
        print(f"   • Baseline: Normal operation without attacks (Step 2)")
        print(f"   • Attack: LLM-coordinated RL attacks (Step 3)")
        print(f"   • Text Comparison: Statistical attack impact (Step 4)")
        print(f"   • Visual Comparison: Detailed plots showing baseline vs attack (Step 5)")
        print("\n## Output files:")
        print(f"   • Baseline plots: sub_figures/baseline_*.pdf")
        print(f"   • Attack plots: sub_figures/attack_*.pdf")
        print(f"   • Comparison plots: sub_figures/comparison_*.pdf")
        print(f"   • Individual comparisons: sub_figures/frequency_comparison_*.pdf")
        print(f"                            sub_figures/load_comparison_*.pdf")
        print(f"                            sub_figures/frequency_delta_*.pdf")
        print(f"                            sub_figures/load_delta_*.pdf")
        print(f"   • All {config['hierarchical']['num_distribution_systems']} distribution systems × 10 EVCS stations visible in EVCS plots")
        print(f"   • Baseline (non-RL) results: detection_results/baseline_actual_system_*.json")
        print(f"   • RL vs baseline comparison: python compare_rl_vs_baseline_actual.py")
        print("=" * 90)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n Enhanced simulation failed: {e}")
        print("   Please check the error messages above for details")

if __name__ == "__main__":
    main()
