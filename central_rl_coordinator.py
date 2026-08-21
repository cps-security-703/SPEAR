#!/usr/bin/env python3


import numpy as np
import time
import json
import re
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from stable_baselines3.common.callbacks import BaseCallback

from attack_specific_rl_agents import (
    AttackSpecificCoordinator,
    AttackDeployment,
    ATTACK_TYPES,
    EpisodeRewardCallback
)


try:
    from gemini_attack_deployment import (
        create_gemini_deployment_prompt,
        create_gemini_adaptation_prompt,
        parse_gemini_deployment_response,
        create_fallback_deployments,
        format_deployment_results
    )
    GEMINI_DEPLOYMENT_AVAILABLE = True
except ImportError:
    GEMINI_DEPLOYMENT_AVAILABLE = False


@dataclass
class AttackAssignment:

    attack_type: str
    target_system: int
    magnitude: float = 0.7
    duration: float = 30.0
    stealth_level: float = 0.7
    priority: int = 1
    rationale: str = ""


@dataclass
class OuterEpisodeResult:

    outer_episode: int
    assignments: List[AttackAssignment]
    inner_results: Dict[str, Dict]
    aggregate_reward: float = 0.0
    aggregate_impact: float = 0.0
    aggregate_success_rate: float = 0.0
    aggregate_detection_risk: float = 0.0
    gemini_evaluation: str = ""
    reassignment_needed: bool = False


class CentralRLCoordinator:


    def __init__(self,
                 attack_coordinator: AttackSpecificCoordinator,
                 llm_analyzer=None,
                 system_analyzer=None,
                 num_systems: int = 6,
                 outer_episodes: int = 15,
                 inner_episodes: int = 50):

        self.attack_coordinator = attack_coordinator
        self.llm_analyzer = llm_analyzer
        self.system_analyzer = system_analyzer
        self.num_systems = num_systems
        self.outer_episodes = outer_episodes
        self.inner_episodes = inner_episodes
        self.attack_types = attack_coordinator.attack_types


        self.gemini_available = llm_analyzer is not None
        self.mode = "gemini_guided" if self.gemini_available else "autonomous"


        self.rotation_schedule = {}
        for c in range(self.outer_episodes):
            self.rotation_schedule[c] = {}
            for i, at in enumerate(self.attack_types):
                sys_id = ((i + c) % self.num_systems) + 1
                self.rotation_schedule[c][at] = sys_id


        self.outer_episode_rewards = []
        self.outer_episode_results = []


        self.inner_episode_rewards = {at: {} for at in self.attack_types}


        self.attack_system_performance = {
            at: {s: [] for s in range(1, self.num_systems + 1)}
            for at in self.attack_types
        }


        self.system_colocation_impact = {s: [] for s in range(1, self.num_systems + 1)}


        self.EXPLORE_COEFF = 0.30
        self.SYNERGY_COEFF = 0.20


        self.final_deployment_plan = None

        print(f"\n{'='*70}")
        print(f"\U0001f39b\ufe0f  CENTRAL RL COORDINATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"   Mode: {self.mode.upper()}")
        print(f"   Outer circles:  {outer_episodes}")
        print(f"   Inner episodes: {inner_episodes} per agent per circle")
        print(f"   Attack types:   {len(self.attack_types)}")
        print(f"   Systems:        {num_systems}")
        print(f"   Total agents:   {len(self.attack_types) * 2} (DQN + SAC)")
        if self.gemini_available:
            print(f"   Assignment: Gemini strategic + adaptive scorer (consolidation allowed)")
        else:
            print(f"   Assignment: Adaptive UCB bandit (coverage  value + consolidation)")
            print(f"   Explore coeff={self.EXPLORE_COEFF}, synergy coeff={self.SYNERGY_COEFF}")
        print(f"{'='*70}\n")


    def run_two_level_training(self,
                                system_analysis_data=None,
                                stride_threats: Dict = None,
                                mitre_tactics: Dict = None) -> Dict:

        print(f"\n{'#'*70}")
        print(f"# TWO-LEVEL TRAINING: {self.mode.upper()} MODE")
        print(f"# Outer: {self.outer_episodes} guidance episodes")
        print(f"# Inner: {self.inner_episodes} RL episodes per agent per guidance round")
        print(f"{'#'*70}\n")

        previous_assignments = None
        previous_results = None
        gemini_success_count = 0

        for outer_ep in range(self.outer_episodes):

            if outer_ep < self.num_systems:
                pass_label = "PASS 1 (full coverage)"
            elif outer_ep < 2 * self.num_systems:
                pass_label = "PASS 2 (refinement)"
            else:
                pass_label = "PASS 3 (final tuning)"

            print(f"\n{'='*60}")
            print(f"  OUTER CIRCLE {outer_ep + 1}/{self.outer_episodes}  [{pass_label}]")
            print(f"{'='*60}")


            if self.llm_analyzer is not None:
                self.llm_analyzer._episode_call_ids = []

            self._last_assignment_source = 'autonomous'
            assignments = self._get_attack_assignments(
                outer_ep=outer_ep,
                previous_assignments=previous_assignments,
                previous_results=previous_results
            )
            if self._last_assignment_source == 'gemini':
                gemini_success_count += 1

            self._print_assignments(outer_ep, assignments)


            inner_results = self._run_inner_training_loop(outer_ep, assignments)


            outer_result = self._aggregate_inner_results(outer_ep, assignments, inner_results)

            self.outer_episode_rewards.append(outer_result.aggregate_reward)
            self.outer_episode_results.append(outer_result)

            print(f"\n  ## Outer Episode {outer_ep + 1} Summary:")
            print(f"     Aggregate Reward:    {outer_result.aggregate_reward:.2f}")
            print(f"     Success Rate:        {outer_result.aggregate_success_rate:.1%}")
            print(f"     Total Impact:        {outer_result.aggregate_impact:.4f}")
            print(f"     Avg Detection Risk:  {outer_result.aggregate_detection_risk:.2%}")


            try:
                from llm_metrics_logger import LLMMetricsLogger
                _logger = LLMMetricsLogger.instance()
                _prev_reward = self.outer_episode_rewards[-2] if len(self.outer_episode_rewards) >= 2 else 0.0
                _call_ids = getattr(self.llm_analyzer, '_episode_call_ids', [])

                _last_cid = getattr(self.llm_analyzer, '_last_call_id', None)
                if _last_cid and _last_cid not in _call_ids:
                    _call_ids = list(_call_ids) + [_last_cid]
                for _cid in _call_ids:
                    _logger.update_rl_impact(
                        call_id          = _cid,
                        reward_before    = _prev_reward,
                        reward_after     = outer_result.aggregate_reward,
                        task_success     = outer_result.aggregate_success_rate > 0.5,
                        rl_plan_accepted = (self._last_assignment_source == 'gemini'),
                    )
                if _call_ids:
                    _delta = outer_result.aggregate_reward - _prev_reward
                    print(f"    ## RL-impact logged for {len(_call_ids)} LLM call(s): Δreward={_delta:+.2f}")
            except Exception as _rim_err:
                print(f"    ## RL-impact update failed (non-fatal): {_rim_err}")

            previous_assignments = assignments
            previous_results = outer_result


        if self.mode == 'gemini_guided' and gemini_success_count == 0:
            print(f"\n  ##Gemini was never successfully used ({gemini_success_count}/{self.outer_episodes} circles)")
            print(f"  ## Downgrading mode: gemini_guided  autonomous")
            self.mode = 'autonomous'
        elif self.mode == 'gemini_guided':
            print(f"\n  ## Gemini was used in {gemini_success_count}/{self.outer_episodes} circles")


        print(f"\n{'='*60}")
        print(f"  FINAL PHASE: Producing Deployment Plan")
        print(f"{'='*60}")
        self.final_deployment_plan = self._produce_final_deployment_plan()


        results = self._build_training_results()


        self._save_two_level_reward_history(results)

        return results


    def _get_attack_assignments(self, outer_ep=0, previous_assignments=None,
                                 previous_results=None) -> List[AttackAssignment]:

        if self.gemini_available and GEMINI_DEPLOYMENT_AVAILABLE:
            return self._get_gemini_assignments(
                outer_ep, previous_assignments, previous_results
            )
        else:
            return self._get_adaptive_assignments(outer_ep)

    def _get_gemini_assignments(self, outer_ep, previous_assignments, previous_results) -> List[AttackAssignment]:


        gemini_available = (
            hasattr(self.llm_analyzer, 'is_available') and self.llm_analyzer.is_available
        )
        if not gemini_available:
            print(f"  ##Gemini unavailable (quota exhausted or not configured)")
            print(f"  ## Using ADAPTIVE assignment (bandit) for circle {outer_ep + 1}")
            self._last_assignment_source = 'adaptive'
            return self._get_adaptive_assignments(outer_ep)

        try:
            if outer_ep == 0:
                print("  ## Asking Gemini for initial attack assignments...")
                prompt = create_gemini_deployment_prompt(
                    system_analysis={},
                    num_systems=self.num_systems
                )
            else:
                print("  ## Asking Gemini to evaluate results and reassign...")
                prev_exec_results = []
                if previous_results:
                    for at, ir in previous_results.inner_results.items():
                        prev_exec_results.append({
                            'attack_type': at,
                            'system_id': ir.get('assigned_system', 0),
                            'result': {
                                'success': ir.get('success_rate', 0) > 0.5,
                                'impact': ir.get('best_impact', 0),
                                'detection_risk': ir.get('avg_detection_risk', 0.5),
                                'mean_reward': ir.get('mean_reward', 0),
                                'success_rate': ir.get('success_rate', 0)
                            }
                        })

                prev_strategy = {
                    'deployments': [
                        {'attack_type': a.attack_type, 'target_systems': [a.target_system]}
                        for a in (previous_assignments or [])
                    ]
                }

                cumulative_coverage = {}
                for at in self.attack_types:
                    systems_trained = []
                    for s in range(1, self.num_systems + 1):
                        if self.attack_system_performance.get(at, {}).get(s, []):
                            systems_trained.append(s)
                    cumulative_coverage[at] = systems_trained


                prompt = create_gemini_adaptation_prompt(
                    prev_exec_results, prev_strategy,
                    circle_num=outer_ep + 1,
                    total_circles=self.outer_episodes,
                    num_systems=self.num_systems,
                    cumulative_coverage=cumulative_coverage
                )


            gemini_input = {
                'deployment_prompt': prompt
            }
            llm_response = self.llm_analyzer.analyze_threats(gemini_input)


            if isinstance(llm_response, dict) and 'llm_response' in llm_response:
                llm_text = llm_response['llm_response']
            else:
                llm_text = llm_response

            deployments = parse_gemini_deployment_response(llm_text)


            is_static_fallback = (len(deployments) == len(self.attack_types))
            if is_static_fallback:
                fallback_mapping = {at: s for at, s in zip(self.attack_types, range(1, self.num_systems + 1))}
                for dep in deployments:
                    expected_sys = fallback_mapping.get(dep.attack_type)
                    if expected_sys is None or dep.target_systems != [expected_sys]:
                        is_static_fallback = False
                        break

            if is_static_fallback and outer_ep > 0:

                print(f"  ##Gemini returned static fallback pattern (parser fell back)")
                print(f"  ## Using ADAPTIVE assignment (bandit) for circle {outer_ep + 1}")
                self._last_assignment_source = 'adaptive'
                return self._get_adaptive_assignments(outer_ep)


            assignments = self._deployments_to_assignments(deployments)
            if len(assignments) < len(self.attack_types):

                existing_types = {a.attack_type for a in assignments}
                rotation = self._get_adaptive_assignments(outer_ep)
                for ra in rotation:
                    if ra.attack_type not in existing_types:
                        assignments.append(ra)
                print(f"   Gemini provided {len(assignments) - len(rotation) + len([r for r in rotation if r.attack_type not in existing_types])} assignments, supplemented {len(assignments) - len([a for a in assignments if a.attack_type in existing_types])} from rotation")

            if is_static_fallback:
                print(f"  ## Using initial deployment ({len(assignments)} assignments)")
            else:
                print(f"  ## Gemini provided {len(assignments)} strategic assignments")
            self._last_assignment_source = 'gemini'
            return assignments

        except Exception as e:
            print(f"   Gemini assignment failed: {e}")
            print(f"  ## Falling back to ADAPTIVE assignment (bandit) for circle {outer_ep + 1}")
            self._last_assignment_source = 'adaptive'
            return self._get_adaptive_assignments(outer_ep)

    def _get_adaptive_assignments(self, outer_ep: int) -> List[AttackAssignment]:


        all_means = [
            float(np.mean(v))
            for at in self.attack_types
            for v in self.attack_system_performance[at].values() if v
        ]
        v_min = min(all_means) if all_means else 0.0
        v_max = max(all_means) if all_means else 1.0
        v_span = (v_max - v_min) if (v_max - v_min) > 1e-9 else 1.0

        def _visits(at, s):
            return len(self.attack_system_performance[at][s])

        def _value_norm(at, s):
            hist = self.attack_system_performance[at][s]
            return (float(np.mean(hist)) - v_min) / v_span if hist else 0.0


        coloc_mean = {
            s: (float(np.mean(v)) if v else 0.0)
            for s, v in self.system_colocation_impact.items()
        }
        c_max = max(coloc_mean.values()) if coloc_mean else 0.0
        c_span = c_max if c_max > 1e-9 else 1.0

        def _coloc_norm(s):
            return coloc_mean[s] / c_span

        assignments = []
        system_count = {s: 0 for s in range(1, self.num_systems + 1)}

        for at in self.attack_types:
            unvisited = [s for s in range(1, self.num_systems + 1) if _visits(at, s) == 0]
            if unvisited:
                s_choice = min(unvisited, key=lambda s: system_count[s])
                source = "explore-coverage"
            else:
                total_v = sum(_visits(at, s) for s in range(1, self.num_systems + 1))
                best_s, best_score = 1, -float('inf')
                for s in range(1, self.num_systems + 1):
                    ucb = _value_norm(at, s) + self.EXPLORE_COEFF * float(
                        np.sqrt(np.log(total_v + 1.0) / (_visits(at, s) + 1.0))
                    )
                    synergy = self.SYNERGY_COEFF * system_count[s] * _coloc_norm(s)
                    score = ucb + synergy
                    if score > best_score:
                        best_score, best_s = score, s
                s_choice = best_s
                source = "adaptive-ucb"

            system_count[s_choice] += 1
            assignments.append(AttackAssignment(
                attack_type=at,
                target_system=s_choice,
                magnitude=0.7,
                stealth_level=0.7,
                rationale=f"Adaptive circle {outer_ep + 1}: {at}  System {s_choice} [{source}]"
            ))

        self._last_assignment_source = 'adaptive'

        print(f"  ## Adaptive assignments for circle {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s}  System {a.target_system}")
        consolidated = {s: c for s, c in system_count.items() if c > 1}
        if consolidated:
            summary = ", ".join(f"System {s}×{c}" for s, c in sorted(consolidated.items()))
            print(f"      Consolidation this circle: {summary}")
        else:
            print(f"       Spread across systems (no consolidation this circle)")

        return assignments

    def _get_autonomous_assignments(self, outer_ep: int) -> List[AttackAssignment]:

        schedule = self.rotation_schedule[outer_ep]

        assignments = []
        for attack_type in self.attack_types:
            system_id = schedule[attack_type]
            assignments.append(AttackAssignment(
                attack_type=attack_type,
                target_system=system_id,
                magnitude=0.7,
                stealth_level=0.7,
                rationale=f"Rotation circle {outer_ep + 1}: {attack_type}  System {system_id}"
            ))

        print(f"  ## Rotation schedule for circle {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s}  System {a.target_system}")

        return assignments

    def _deployments_to_assignments(self, deployments: List[AttackDeployment]) -> List[AttackAssignment]:

        assignments = []
        used_attack_types = set()

        for dep in deployments:

            target = dep.target_systems[0] if dep.target_systems else 1
            if dep.attack_type not in used_attack_types:
                assignments.append(AttackAssignment(
                    attack_type=dep.attack_type,
                    target_system=target,
                    magnitude=dep.magnitude,
                    duration=dep.duration,
                    stealth_level=dep.stealth_level,
                    priority=dep.priority
                ))
                used_attack_types.add(dep.attack_type)


        missing_types = [at for at in self.attack_types if at not in used_attack_types]
        for i, at in enumerate(missing_types):
            best_sys, best_mean = None, -float('inf')
            for s in range(1, self.num_systems + 1):
                hist = self.attack_system_performance.get(at, {}).get(s, [])
                if hist:
                    m = float(np.mean(hist))
                    if m > best_mean:
                        best_mean, best_sys = m, s
            system_id = best_sys if best_sys is not None else (i % self.num_systems) + 1
            assignments.append(AttackAssignment(
                attack_type=at,
                target_system=system_id,
                magnitude=0.7,
                stealth_level=0.7,
                rationale=f"Adaptive gap-fill  System {system_id}"
                          f"{' (best recorded)' if best_sys is not None else ' (no history)'}"
            ))

        return assignments


    def _run_inner_training_loop(self, outer_ep: int,
                                  assignments: List[AttackAssignment]) -> Dict[str, Dict]:


        timesteps_per_agent = self.inner_episodes * 1000

        print(f"\n  ## Inner Training Loop: {timesteps_per_agent} timesteps per agent "
              f"(~{self.inner_episodes} episodes)")

        inner_results = {}

        for assignment in assignments:
            at = assignment.attack_type
            sys_id = assignment.target_system

            print(f"\n    ##Training {at} agents on System {sys_id} "
                  f"({timesteps_per_agent} timesteps)...")


            sac_env_key = f'sac_{at}'
            dqn_env_key = f'dqn_{at}'

            if sac_env_key not in self.attack_coordinator.environments:
                print(f"      ##No environment for {at}, skipping")
                inner_results[at] = {
                    'assigned_system': sys_id,
                    'dqn_rewards': [], 'sac_rewards': [],
                    'best_reward': 0.0, 'best_impact': 0.0,
                    'success_rate': 0.0, 'avg_detection_risk': 0.5,
                    'num_episodes': 0
                }
                continue

            sac_env = self.attack_coordinator.environments[sac_env_key]


            sac_env.forced_target_system = [sys_id]


            sac_env.guidance_hints = {
                'magnitude': assignment.magnitude,
                'duration': assignment.duration,
                'stealth_level': assignment.stealth_level
            }
            print(f"       Guidance hints: mag={assignment.magnitude:.2f}, "
                  f"dur={assignment.duration:.1f}s, stealth={assignment.stealth_level:.2f}")


            node_level = getattr(self.attack_coordinator, 'node_level', False)
            if dqn_env_key in self.attack_coordinator.environments:
                dqn_env = self.attack_coordinator.environments[dqn_env_key]
                if hasattr(dqn_env, 'continuous_env'):
                    dqn_env.continuous_env.forced_target_system = (
                        [sys_id] if node_level
                        else list(range(1, self.num_systems + 1))
                    )
                    dqn_env.continuous_env.guidance_hints = {
                        'magnitude': assignment.magnitude,
                        'duration': assignment.duration,
                        'stealth_level': assignment.stealth_level
                    }


            sac_callback = EpisodeRewardCallback()
            try:
                self.attack_coordinator.sac_agents[at].learn(
                    total_timesteps=timesteps_per_agent,
                    callback=sac_callback,
                    reset_num_timesteps=False,
                    progress_bar=False
                )
                sac_results = sac_callback.get_results()
                sac_rewards = sac_results.get('episode_rewards', [])
                print(f"      SAC: {sac_results['num_episodes']} episodes, "
                      f"mean_reward={sac_results['mean_reward']:.3f}")
            except Exception as e:
                print(f"      ##SAC training error: {e}")
                sac_rewards = []


            dqn_callback = EpisodeRewardCallback()
            try:
                self.attack_coordinator.dqn_agents[at].learn(
                    total_timesteps=timesteps_per_agent,
                    callback=dqn_callback,
                    reset_num_timesteps=False,
                    progress_bar=False
                )
                dqn_results = dqn_callback.get_results()
                dqn_rewards = dqn_results.get('episode_rewards', [])
                print(f"      DQN: {dqn_results['num_episodes']} episodes, "
                      f"mean_reward={dqn_results['mean_reward']:.3f}")
            except Exception as e:


                import traceback
                print(f"      ##DQN training error: {e}")
                traceback.print_exc()
                dqn_rewards = []


            sac_env.forced_target_system = None
            sac_env.guidance_hints = None
            if dqn_env_key in self.attack_coordinator.environments:
                dqn_env = self.attack_coordinator.environments[dqn_env_key]
                if hasattr(dqn_env, 'continuous_env'):
                    dqn_env.continuous_env.forced_target_system = None
                    dqn_env.continuous_env.guidance_hints = None


            best_reward = -float('inf')
            best_impact = 0.0
            total_successes = 0
            total_detection_risk = 0.0
            eval_rewards = []
            num_eval = 10

            sac_env.forced_target_system = sys_id
            for _ in range(num_eval):
                obs, _ = sac_env.reset()
                action, _ = self.attack_coordinator.sac_agents[at].predict(
                    obs, deterministic=True
                )
                _, reward, _, _, info = sac_env.step(action)
                eval_rewards.append(float(reward))

                attack_result = info.get('attack_result', {})
                impact = attack_result.get('impact', 0.0)
                success = attack_result.get('success', False)
                detection = attack_result.get('detection_risk', 0.5)

                if float(reward) > best_reward:
                    best_reward = float(reward)
                    best_impact = impact
                if success:
                    total_successes += 1
                total_detection_risk += detection
            sac_env.forced_target_system = None


            circle_detection_rate = total_detection_risk / num_eval
            circle_success_rate = total_successes / num_eval
            circle_avg_impact = best_impact
            sac_env.update_cross_circle_stats(
                sys_id=sys_id,
                detection_rate=circle_detection_rate,
                success_rate=circle_success_rate,
                avg_impact=circle_avg_impact
            )
            print(f"      ## Cross-circle memory updated for sys {sys_id}: "
                  f"det={circle_detection_rate:.2f}, "
                  f"succ={circle_success_rate:.2f}, "
                  f"impact={circle_avg_impact:.3f} "
                  f"(circle #{sac_env.cross_circle_stats[sys_id]['num_circles']})")


            all_sac_rewards = sac_rewards if sac_rewards else eval_rewards


            inner_results[at] = {
                'assigned_system': sys_id,
                'dqn_rewards': dqn_rewards,
                'sac_rewards': all_sac_rewards,
                'best_reward': best_reward,
                'best_impact': best_impact,
                'mean_reward': float(np.mean(all_sac_rewards)) if all_sac_rewards else 0.0,
                'std_reward': float(np.std(all_sac_rewards)) if all_sac_rewards else 0.0,
                'success_rate': total_successes / num_eval,
                'avg_detection_risk': total_detection_risk / num_eval,
                'num_episodes': len(all_sac_rewards),
                'training_timesteps': timesteps_per_agent
            }


            self.inner_episode_rewards[at][outer_ep] = {
                'dqn_rewards': dqn_rewards,
                'sac_rewards': all_sac_rewards,
                'assigned_system': sys_id,
                'best_impact': float(best_impact),
                'success_rate': float(total_successes / num_eval),
                'avg_detection_risk': float(total_detection_risk / num_eval),
            }

            print(f"    ## {at} on System {sys_id}: "
                  f"mean={inner_results[at]['mean_reward']:.3f}, "
                  f"best={best_reward:.3f}, "
                  f"success={inner_results[at]['success_rate']:.0%}")

        return inner_results


    def _aggregate_inner_results(self, outer_ep: int,
                                  assignments: List[AttackAssignment],
                                  inner_results: Dict[str, Dict]) -> OuterEpisodeResult:


        all_rewards = []
        all_impacts = []
        all_success_rates = []
        all_detection_risks = []

        for at, ir in inner_results.items():
            all_rewards.append(ir.get('mean_reward', 0.0))
            all_impacts.append(ir.get('best_impact', 0.0))
            all_success_rates.append(ir.get('success_rate', 0.0))
            all_detection_risks.append(ir.get('avg_detection_risk', 0.5))


        system_to_impacts = {}
        for assignment in assignments:
            imp = inner_results.get(assignment.attack_type, {}).get('best_impact', 0.0)
            system_to_impacts.setdefault(assignment.target_system, []).append(imp)

        colocation_bonus = 0.0
        for s, imps in system_to_impacts.items():
            if len(imps) >= 2:
                combined = float(sum(imps))
                self.system_colocation_impact[s].append(combined)
                colocation_bonus += combined
        if colocation_bonus > 0.0:
            n_consolidated = sum(1 for v in system_to_impacts.values() if len(v) >= 2)
            print(f"     Consolidation measured: co-located impact={colocation_bonus:.4f} "
                  f"across {n_consolidated} system(s)")


        avg_success = np.mean(all_success_rates) if all_success_rates else 0.0
        total_impact = sum(all_impacts)
        avg_detection = np.mean(all_detection_risks) if all_detection_risks else 0.5
        stealth_score = 1.0 - avg_detection

        composite_reward = (
            avg_success * 1000.0 +
            total_impact * 500.0 +
            stealth_score * 300.0 +
            self.SYNERGY_COEFF * colocation_bonus * 500.0
        )


        for assignment in assignments:
            at = assignment.attack_type
            sys_id = assignment.target_system
            ir = inner_results.get(at, {})
            mean_r = ir.get('mean_reward', 0.0)
            if at in self.attack_system_performance and sys_id in self.attack_system_performance[at]:
                self.attack_system_performance[at][sys_id].append(mean_r)

        return OuterEpisodeResult(
            outer_episode=outer_ep,
            assignments=assignments,
            inner_results=inner_results,
            aggregate_reward=composite_reward,
            aggregate_impact=total_impact,
            aggregate_success_rate=avg_success,
            aggregate_detection_risk=avg_detection
        )


    def _dqn_qvalues(self, dqn_agent, obs):

        try:
            import torch
            obs_t, _ = dqn_agent.policy.obs_to_tensor(obs)
            with torch.no_grad():
                q = dqn_agent.q_net(obs_t)
            return q.cpu().numpy().flatten()
        except Exception as e:
            print(f"     ##Q-value extraction failed ({e}); falling back to argmax")
            return None

    def _produce_final_deployment_plan(self) -> Dict:

        print("  ## Analyzing per-attack per-system performance across all circles...")

        if not self.outer_episode_results:
            return {'deployments': [], 'status': 'no_results'}


        final_deployments = []
        best_system_map = {}

        for at in self.attack_types:
            best_sys = 1
            best_mean = -float('inf')

            for sys_id in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance.get(at, {}).get(sys_id, [])
                if rewards:
                    sys_mean = float(np.mean(rewards))
                    if sys_mean > best_mean:
                        best_mean = sys_mean
                        best_sys = sys_id

            best_system_map[at] = (best_sys, best_mean)


            agent_magnitude = 0.7
            agent_duration = 30.0
            agent_stealth = 0.7
            agent_params_source = 'fallback'

            sac_agent = self.attack_coordinator.sac_agents.get(at)
            sac_env_key = f'sac_{at}'
            if sac_agent and sac_env_key in self.attack_coordinator.environments:
                try:
                    env = self.attack_coordinator.environments[sac_env_key]
                    env.forced_target_system = best_sys
                    obs, _ = env.reset()
                    trained_action, _ = sac_agent.predict(obs, deterministic=True)

                    agent_magnitude = float(np.clip(trained_action[0], 0.1, 2.0))
                    agent_duration = float(np.clip(trained_action[1], 5.0, 60.0))
                    agent_stealth = float(np.clip(trained_action[2], 0.0, 1.0))
                    agent_params_source = 'trained_sac_agent'
                    env.forced_target_system = None
                except Exception as e:
                    print(f"     ##Could not query SAC agent for {at}: {e}")


            node_level = getattr(self.attack_coordinator, 'node_level', False)
            target_node = None
            target_nodes = None
            dqn_choice = None
            dqn_agent = self.attack_coordinator.dqn_agents.get(at)
            dqn_env = self.attack_coordinator.environments.get(f'dqn_{at}')
            if dqn_agent is not None and dqn_env is not None:
                try:
                    if node_level and hasattr(dqn_env, 'continuous_env'):

                        dqn_env.continuous_env.forced_target_system = [best_sys]
                    obs, _ = dqn_env.reset()
                    if node_level:
                        n_nodes = getattr(self.attack_coordinator, 'n_nodes', 10)
                        cap = int(getattr(self.attack_coordinator, 'topk_networks', 7))
                        cap = max(1, min(cap, n_nodes))


                        qvals = self._dqn_qvalues(dqn_agent, obs)
                        if qvals is not None and len(qvals) >= n_nodes:
                            order = list(np.argsort(qvals[:n_nodes])[::-1])
                            target_nodes = [int(x) for x in order[:cap]]
                        else:
                            act, _ = dqn_agent.predict(obs, deterministic=True)
                            target_nodes = [int(act) % n_nodes]
                        target_node = target_nodes[0]
                        dqn_choice = f"nodes{target_nodes}"
                        if hasattr(dqn_env, 'continuous_env'):
                            dqn_env.continuous_env.forced_target_system = None
                    else:
                        act, _ = dqn_agent.predict(obs, deterministic=True)
                        dqn_choice = f"system{int(act) + 1}"
                except Exception as e:
                    print(f"     ##Could not query DQN agent for {at}: {e}")

            _dep = {
                'attack_type': at,
                'target_system': best_sys,
                'magnitude': agent_magnitude,
                'duration': agent_duration,
                'stealth_level': agent_stealth,
                'expected_reward': best_mean if best_mean > -float('inf') else 0.0,
                'params_source': agent_params_source,
                'dqn_choice': dqn_choice,
                'systems_trained_on': [
                    s for s in range(1, self.num_systems + 1)
                    if self.attack_system_performance.get(at, {}).get(s, [])
                ],
                'rationale': f"Best system for {at}: System {best_sys} (mean_reward={best_mean:.3f})"
                             + (f", DQN nodes={target_nodes}" if target_nodes else "")
            }
            if target_node is not None:
                _dep['target_node'] = target_node
            if target_nodes:
                _dep['target_nodes'] = target_nodes
            final_deployments.append(_dep)

            print(f"     {at:30s}  System {best_sys} "
                  f"(reward={best_mean:.3f}, mag={agent_magnitude:.2f}, "
                  f"dur={agent_duration:.1f}s, stealth={agent_stealth:.2f}) "
                  f"[{agent_params_source}]")

        best_outer = max(self.outer_episode_results, key=lambda r: r.aggregate_reward)


        if self.gemini_available and self.llm_analyzer:
            final_deployments = self._gemini_refine_timing(final_deployments)


        rag_analysis = self._rag_enhance_deployment_plan(final_deployments, best_system_map)

        plan = {
            'deployments': final_deployments,
            'best_system_per_attack': {at: {'system': s, 'mean_reward': r} for at, (s, r) in best_system_map.items()},
            'best_outer_episode': best_outer.outer_episode,
            'best_aggregate_reward': best_outer.aggregate_reward,
            'mode': self.mode,
            'total_outer_episodes': self.outer_episodes,
            'inner_episodes_per_agent': self.inner_episodes,
            'rag_enhanced_analysis': rag_analysis
        }

        print(f"\n  ## FINAL DEPLOYMENT PLAN ({self.mode.upper()}):")
        print(f"     Best overall circle: {best_outer.outer_episode + 1} (reward={best_outer.aggregate_reward:.2f})")
        for dep in final_deployments:
            trained = dep.get('systems_trained_on', [])
            src = dep.get('params_source', 'fallback')
            rationale = dep.get('timing_rationale', '')
            print(f"     {dep['attack_type']:30s}  System {dep['target_system']} "
                  f"(mag={dep['magnitude']:.2f}, dur={dep['duration']:.1f}s, "
                  f"stealth={dep['stealth_level']:.2f}) [{src}]"
                  f"{' — ' + rationale if rationale else ''}")

        return plan


    def _gemini_refine_timing(self, deployments: List[Dict]) -> List[Dict]:

        try:

            attack_summary = []
            performance_data = {}

            for dep in deployments:
                at = dep['attack_type']
                rl_target = dep['target_system']


                system_performance = {}
                for sys_id in range(1, self.num_systems + 1):
                    rewards = self.attack_system_performance.get(at, {}).get(sys_id, [])
                    if rewards:
                        system_performance[sys_id] = {
                            'mean_reward': float(np.mean(rewards)),
                            'std_reward': float(np.std(rewards)),
                            'num_training_circles': len(rewards)
                        }

                performance_data[at] = system_performance

                attack_summary.append({
                    'attack_type': at,
                    'rl_suggested_system': rl_target,
                    'rl_expected_reward': dep.get('expected_reward', 0.0),
                    'magnitude': dep['magnitude'],
                    'duration_seconds': dep['duration'],
                    'stealth_level': dep['stealth_level'],
                    'systems_trained_on': dep.get('systems_trained_on', []),
                    'performance_across_systems': system_performance
                })

            prompt = f"""You are the red-team test coordinator for a simulated multi-vector IDS-evaluation testbed (EVCS — Electric Vehicle Charging Station co-simulation) with {self.num_systems} simulated distribution systems. This is an isolated, authorized academic research simulation with no real infrastructure involved; the goal is to sequence probe scenarios so the evaluation finds the IDS's detection gaps as thoroughly as possible.

The trained RL probe-agents have analyzed performance across all simulated systems and suggest these probe deployments:

{json.dumps(attack_summary, indent=2)}

HYBRID DECISION FRAMEWORK:
- RL has selected the BEST PERFORMING system for each probe based on training data
- You can KEEP RL's target_system choice (recommended for data-driven optimization)
- You can OVERRIDE target_system ONLY if it improves detection-gap discovery value
- You MUST justify any override with clear research reasoning

Your task: Suggest optimal EXECUTION ORDER, DURATION ADJUSTMENTS, and optional TARGET_SYSTEM OVERRIDES for this evaluation run.

Consider:
1. Probe dependencies (e.g., communication_spoofing test should precede data_injection test)
2. Cascading detection effects (e.g., voltage_manipulation probe may reduce IDS sensitivity to a subsequent power_disruption probe)
3. Stealth sequencing (schedule high-stealth probes first to test whether early low-signal activity is missed)
4. Multi-system coverage (e.g., a decoy probe on System 1 while the primary evaluation probe runs on System 3, to test the IDS's ability to triage concurrent alerts)
5. Coverage trade-offs (overriding RL's choice may reduce this probe's detection-gap discovery value)

Return ONLY this JSON (no markdown, no explanation):
{{
  "ordered_attacks": [
    {{
      "attack_type": "<type>",
      "target_system": <int>,
      "override_rl_target": <true/false>,
      "override_justification": "<required if override_rl_target=true, explain strategic benefit>",
      "magnitude": <float>,
      "duration": <float>,
      "stealth_level": <float>,
      "timing_rationale": "<brief reason for this position in sequence>"
    }}
  ]
}}"""

            response = self.llm_analyzer.analyze_threats({'deployment_prompt': prompt})

            if isinstance(response, dict) and 'llm_response' in response:
                llm_text = response['llm_response']
            else:
                llm_text = str(response)


            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_text)
            if json_match:
                parsed = json.loads(json_match.group())
                ordered = parsed.get('ordered_attacks', [])

                if len(ordered) >= len(deployments):

                    dep_by_type = {d['attack_type']: d for d in deployments}
                    refined = []
                    overrides_applied = 0

                    for item in ordered:
                        at = item.get('attack_type', '')
                        if at in dep_by_type:
                            dep = dep_by_type[at].copy()
                            rl_target = dep['target_system']
                            gemini_target = item.get('target_system', rl_target)
                            override_flag = item.get('override_rl_target', False)
                            override_justification = item.get('override_justification', '')


                            if override_flag and gemini_target != rl_target:
                                dep['target_system'] = gemini_target
                                dep['rl_suggested_system'] = rl_target
                                dep['override_justification'] = override_justification
                                dep['params_source'] = 'trained_sac_agent+gemini_override'
                                overrides_applied += 1
                                print(f"     ## {at}: Gemini OVERRODE target System {rl_target}  {gemini_target}")
                                print(f"         Justification: {override_justification}")
                            else:
                                dep['params_source'] = 'trained_sac_agent+gemini_timing'


                            if 'duration' in item and item['duration'] > 0:
                                dep['duration'] = float(np.clip(item['duration'], 5.0, 120.0))

                            dep['timing_rationale'] = item.get('timing_rationale', '')
                            refined.append(dep)
                            del dep_by_type[at]


                    for dep in dep_by_type.values():
                        refined.append(dep)

                    print(f"  ## Gemini refined attack ordering ({len(refined)} attacks, {overrides_applied} target overrides):")
                    for i, dep in enumerate(refined):
                        rationale = dep.get('timing_rationale', '')
                        override_marker = '##' if 'rl_suggested_system' in dep else ''
                        print(f"     {i+1}. {override_marker}{dep['attack_type']:30s}  Sys {dep['target_system']} "
                              f"(dur={dep['duration']:.1f}s) — {rationale}")
                    return refined

            print(f"  ##Could not parse Gemini timing response, keeping agent order")
            return deployments

        except Exception as e:
            print(f"  ##Gemini timing refinement failed: {e}")
            return deployments


    ATTACK_STRIDE_MAP = {
        'voltage_manipulation': {
            'stride': 'Information Disclosure',
            'protocol': 'DNP3',
            'link': 'CMS  DG / DG  DSM (Links 3-4)',
            'data_flows': 'DF-3, DF-4 (Load Measurement)',
        },
        'current_injection': {
            'stride': 'Elevation of Privilege',
            'protocol': 'OCPP/TCP',
            'link': 'EV  EVCS  CMS (Links 1-2)',
            'data_flows': 'DF-1 (Charging Info V,I,P), DF-2 (Authentication)',
        },
        'power_disruption': {
            'stride': 'Denial of Service',
            'protocol': 'TCP/IEC 61850',
            'link': 'EMS  AGC / DG  DSM (Links 4,6)',
            'data_flows': 'DF-4 (Load Measurement), DF-9 (Optimal Reference)',
        },
        'communication_spoofing': {
            'stride': 'Spoofing',
            'protocol': 'OCPP',
            'link': 'EV  EVCS (Link 1)',
            'data_flows': 'DF-1 (Charging Info: SoC, power demand)',
        },
        'data_injection': {
            'stride': 'Tampering',
            'protocol': 'DNP3',
            'link': 'CMS  DG / DG  DSM (Links 3-4)',
            'data_flows': 'DF-3, DF-4 (Load Measurement)',
        },
        'protocol_manipulation': {
            'stride': 'Repudiation',
            'protocol': 'DNP3',
            'link': 'DSM  EMS (Link 5)',
            'data_flows': 'DF-5 (Load Forecasting)',
        },
    }

    def _rag_enhance_deployment_plan(self, deployments: List[Dict],
                                     best_system_map: Dict) -> Optional[Dict]:


        ares_rag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ares_rag')
        if ares_rag_dir not in sys.path:
            sys.path.insert(0, ares_rag_dir)

        try:
            from vector_db import ChromaDBManager, DocumentEmbedder
            db_manager = ChromaDBManager()
            embedder = DocumentEmbedder()
            doc_count = db_manager.collection.count()
            print(f"\n   RAG: Connected to ARES vector DB ({doc_count} documents)")
        except Exception as e:
            print(f"\n  ##RAG unavailable ({e}) — using pure RL deployment plan")
            return None


        rag_context_per_attack = {}

        for dep in deployments:
            at = dep['attack_type']
            stride_info = self.ATTACK_STRIDE_MAP.get(at, {})
            stride_cat = stride_info.get('stride', at)
            protocol = stride_info.get('protocol', '')


            query = (
                f"{stride_cat} vulnerability in {protocol} protocol "
                f"for EVCS electric vehicle charging station "
                f"{at.replace('_', ' ')} attack"
            )

            try:
                query_embedding = embedder.embed_text(query)
                results = db_manager.query(
                    query_embedding=query_embedding,
                    n_results=5
                )

                docs = []
                if results.get('ids') and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        docs.append({
                            'id': results['ids'][0][i],
                            'distance': results['distances'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'content': results['documents'][0][i][:300]
                        })

                rag_context_per_attack[at] = {
                    'query': query,
                    'stride_category': stride_cat,
                    'protocol': protocol,
                    'link': stride_info.get('link', ''),
                    'data_flows': stride_info.get('data_flows', ''),
                    'context_docs': docs,
                    'num_docs_retrieved': len(docs)
                }


                cves = set()
                mitre_techs = set()
                for doc in docs:
                    meta = doc.get('metadata', {})
                    try:
                        techs = json.loads(meta.get('mitre_techniques', '[]'))
                        mitre_techs.update(techs)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    content_cves = re.findall(r'CVE-\d{4}-\d{4,7}',
                                              doc.get('content', '') + ' ' + str(meta))
                    cves.update(content_cves)

                rag_context_per_attack[at]['cves'] = sorted(list(cves))
                rag_context_per_attack[at]['mitre_techniques'] = sorted(list(mitre_techs))

                print(f"     {at:30s}  {len(docs)} docs, "
                      f"{len(cves)} CVEs, {len(mitre_techs)} MITRE techniques")

            except Exception as e:
                print(f"     {at:30s}  RAG query failed: {e}")
                rag_context_per_attack[at] = {'error': str(e)}


        gemini_analysis = self._gemini_rag_final_analysis(
            deployments, best_system_map, rag_context_per_attack
        )

        return {
            'rag_context': rag_context_per_attack,
            'gemini_analysis': gemini_analysis,
            'rag_db_documents': doc_count
        }

    def _gemini_rag_final_analysis(self, deployments: List[Dict],
                                    best_system_map: Dict,
                                    rag_context: Dict) -> Optional[str]:

        if not self.llm_analyzer:
            print("  ##No Gemini analyzer — skipping RAG-enhanced analysis")
            return None


        rl_summary_lines = []
        for dep in deployments:
            at = dep['attack_type']
            stride_info = self.ATTACK_STRIDE_MAP.get(at, {})
            trained_on = dep.get('systems_trained_on', [])


            sys_perf = []
            for s in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance.get(at, {}).get(s, [])
                if rewards:
                    sys_perf.append(f"Sys{s}={np.mean(rewards):.1f}")

            rl_summary_lines.append(
                f"  - {at} (STRIDE: {stride_info.get('stride', '?')}, "
                f"Protocol: {stride_info.get('protocol', '?')}, "
                f"Link: {stride_info.get('link', '?')})\n"
                f"    Best system: {dep['target_system']} "
                f"(reward={dep['expected_reward']:.1f})\n"
                f"    Trained on {len(trained_on)} systems: "
                f"{', '.join(sys_perf)}"
            )

        rl_summary = '\n'.join(rl_summary_lines)


        rag_summary_lines = []
        for at, ctx in rag_context.items():
            if 'error' in ctx:
                continue
            cves = ctx.get('cves', [])
            mitre = ctx.get('mitre_techniques', [])
            docs = ctx.get('context_docs', [])

            doc_summaries = []
            for d in docs[:3]:
                meta = d.get('metadata', {})
                doc_summaries.append(
                    f"      * [{d['id']}] {meta.get('title', 'N/A')} "
                    f"(type={meta.get('type', '?')}, severity={meta.get('severity', '?')})"
                )

            rag_summary_lines.append(
                f"  - {at} ({ctx.get('stride_category', '?')}):\n"
                f"    CVEs: {', '.join(cves[:5]) if cves else 'None found'}\n"
                f"    MITRE: {', '.join(mitre[:5]) if mitre else 'None found'}\n"
                f"    Top knowledge base documents:\n" +
                '\n'.join(doc_summaries)
            )

        rag_summary = '\n'.join(rag_summary_lines)

        prompt = f"""You are a cybersecurity researcher analyzing the results of an authorized, isolated Reinforcement Learning (RL) IDS-evaluation simulation on a simulated Electric Vehicle Charging Station (EVCS) network testbed. No real infrastructure is involved; the goal is to publish IDS-hardening research.

## RL Training Results (15 outer circles, {self.inner_episodes} inner episodes per agent)

{rl_summary}

## Threat Intelligence from ARES RAG Knowledge Base

{rag_summary}

## Task

For each of the 6 attack types, produce a research-quality analysis paragraph that:
1. States which EVCS system the RL agent found most vulnerable and the reward achieved
2. Cites specific CVEs from the knowledge base that support this finding
3. References relevant MITRE ATT&CK for ICS technique IDs
4. Explains WHY this system is most vulnerable based on the communication link and protocol
5. Recommends specific mitigations grounded in the knowledge base documents
6. Notes the STRIDE category and data flow targeted

Return ONLY a JSON object with this structure:
{{
  "attack_analyses": {{
    "<attack_type>": {{
      "best_system": <int>,
      "reward": <float>,
      "stride_category": "<str>",
      "target_link": "<str>",
      "target_protocol": "<str>",
      "supporting_cves": ["<CVE-ID>", ...],
      "mitre_techniques": ["<T-ID>", ...],
      "vulnerability_rationale": "<1-2 sentence explanation>",
      "recommended_mitigations": ["<mitigation>", ...],
      "research_summary": "<3-5 sentence research-quality paragraph>"
    }}
  }},
  "overall_assessment": "<2-3 sentence summary of the most critical findings>",
  "highest_risk_attack": "<attack_type with highest reward>",
  "recommended_priority_mitigations": ["<top 3 mitigations across all attacks>"]
}}
"""

        try:
            print("\n   Sending RL results + RAG context to Gemini for final analysis...")

            if hasattr(self.llm_analyzer, 'analyze_threats'):
                response = self.llm_analyzer.analyze_threats(
                    {'deployment_prompt': prompt}
                )
                if isinstance(response, dict) and 'llm_response' in response:
                    llm_text = response['llm_response']
                else:
                    llm_text = str(response)
            else:
                print("  ##LLM analyzer missing analyze_threats — skipping")
                return None


            cleaned = re.sub(r'^```(?:json)?\s*', '', llm_text.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)

            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())


                    attack_analyses = analysis.get('attack_analyses', {})
                    for dep in deployments:
                        at = dep['attack_type']
                        if at in attack_analyses:
                            aa = attack_analyses[at]
                            dep['rag_rationale'] = aa.get('research_summary', '')
                            dep['supporting_cves'] = aa.get('supporting_cves', [])
                            dep['mitre_techniques'] = aa.get('mitre_techniques', [])
                            dep['recommended_mitigations'] = aa.get('recommended_mitigations', [])
                            dep['vulnerability_rationale'] = aa.get('vulnerability_rationale', '')
                            dep['stride_category'] = aa.get('stride_category', '')
                            dep['target_link'] = aa.get('target_link', '')
                            dep['target_protocol'] = aa.get('target_protocol', '')

                    print(f"  ## RAG-enhanced analysis complete:")
                    print(f"     Highest risk: {analysis.get('highest_risk_attack', '?')}")
                    print(f"     Overall: {analysis.get('overall_assessment', '?')[:120]}...")


                    self._save_rag_deployment_plan(analysis, deployments)

                    return analysis

                except json.JSONDecodeError as e:
                    print(f"  ##Could not parse Gemini RAG analysis JSON: {e}")

                    self._save_rag_deployment_plan({'raw_response': llm_text}, deployments)
                    return {'raw_response': llm_text}
            else:
                print("  ##No JSON found in Gemini RAG analysis response")
                self._save_rag_deployment_plan({'raw_response': llm_text}, deployments)
                return {'raw_response': llm_text}

        except Exception as e:
            print(f"  ##Gemini RAG analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_rag_deployment_plan(self, analysis: Dict, deployments: List[Dict]):

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rag_enhanced_deployment_plan_{timestamp}.json"

            output = {
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
                'mode': self.mode,
                'outer_episodes': self.outer_episodes,
                'inner_episodes': self.inner_episodes,
                'gemini_analysis': analysis,
                'deployments': deployments
            }

            with open(filename, 'w') as f:
                json.dump(output, f, indent=2, default=str)

            print(f"  ## Saved RAG-enhanced deployment plan to {filename}")
        except Exception as e:
            print(f"  ##Failed to save RAG deployment plan: {e}")


    def get_per_system_episode_rewards(self) -> Dict[int, Dict[str, list]]:

        per_system = {
            s: {'dqn_rewards': [], 'sac_rewards': [], 'outer_circle_boundaries_dqn': [], 'outer_circle_boundaries_sac': []}
            for s in range(1, self.num_systems + 1)
        }


        for outer_ep in range(self.outer_episodes):
            for at in self.attack_types:
                ep_data = self.inner_episode_rewards.get(at, {}).get(outer_ep)
                if ep_data is None:
                    continue

                sys_id = ep_data.get('assigned_system')
                if sys_id is None or sys_id not in per_system:
                    continue

                dqn_r = ep_data.get('dqn_rewards', [])
                sac_r = ep_data.get('sac_rewards', [])


                if dqn_r:
                    per_system[sys_id]['outer_circle_boundaries_dqn'].append(
                        len(per_system[sys_id]['dqn_rewards']))
                    per_system[sys_id]['dqn_rewards'].extend(dqn_r)

                if sac_r:
                    per_system[sys_id]['outer_circle_boundaries_sac'].append(
                        len(per_system[sys_id]['sac_rewards']))
                    per_system[sys_id]['sac_rewards'].extend(sac_r)


        for s in range(1, self.num_systems + 1):
            nd = len(per_system[s]['dqn_rewards'])
            ns = len(per_system[s]['sac_rewards'])
            if nd or ns:
                print(f"  System {s}: {nd} DQN episodes, {ns} SAC episodes across {self.outer_episodes} outer circles")

        return per_system


    def _build_training_results(self) -> Dict:


        inner_rewards_flat = {}
        for at in self.attack_types:
            inner_rewards_flat[at] = {}
            for outer_ep, data in self.inner_episode_rewards[at].items():
                inner_rewards_flat[at][str(outer_ep)] = {
                    'dqn_rewards': data['dqn_rewards'],
                    'sac_rewards': data['sac_rewards'],
                    'assigned_system': data['assigned_system'],

                    'best_impact': data.get('best_impact'),
                    'success_rate': data.get('success_rate'),
                    'avg_detection_risk': data.get('avg_detection_risk'),
                }

        return {
            'mode': self.mode,
            'outer_episodes': self.outer_episodes,
            'inner_episodes': self.inner_episodes,
            'outer_episode_rewards': self.outer_episode_rewards,
            'inner_episode_rewards': inner_rewards_flat,
            'final_deployment_plan': self.final_deployment_plan,
            'outer_episode_details': [
                {
                    'outer_episode': r.outer_episode,
                    'aggregate_reward': r.aggregate_reward,
                    'aggregate_impact': r.aggregate_impact,
                    'aggregate_success_rate': r.aggregate_success_rate,
                    'aggregate_detection_risk': r.aggregate_detection_risk,
                    'assignments': [
                        {'attack_type': a.attack_type, 'target_system': a.target_system,
                         'rationale': a.rationale}
                        for a in r.assignments
                    ]
                }
                for r in self.outer_episode_results
            ],
            'summary': {
                'mean_outer_reward': float(np.mean(self.outer_episode_rewards)) if self.outer_episode_rewards else 0.0,
                'best_outer_reward': float(max(self.outer_episode_rewards)) if self.outer_episode_rewards else 0.0,
                'reward_trend': self._compute_reward_trend(),
                'total_inner_episodes': self.outer_episodes * self.inner_episodes * len(self.attack_types),
            }
        }

    def _compute_reward_trend(self) -> str:

        if len(self.outer_episode_rewards) < 3:
            return "insufficient_data"

        first_half = np.mean(self.outer_episode_rewards[:len(self.outer_episode_rewards)//2])
        second_half = np.mean(self.outer_episode_rewards[len(self.outer_episode_rewards)//2:])

        diff = second_half - first_half
        if diff > 10:
            return "improving"
        elif diff < -10:
            return "declining"
        else:
            return "stable"

    def _save_two_level_reward_history(self, results: Dict):

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"reward_history_{self.mode}_{timestamp}.json"


            serializable = self._make_serializable(results)
            serializable['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

            with open(filename, 'w') as f:
                json.dump(serializable, f, indent=2)

            print(f"\n  ## Saved two-level reward history to {filename}")
            print(f"     Outer rewards: {len(results['outer_episode_rewards'])} episodes")

            total_inner = sum(
                len(data.get('sac_rewards', []))
                for at_data in results['inner_episode_rewards'].values()
                for data in at_data.values()
            )
            print(f"     Inner rewards: {total_inner} total agent episodes")

        except Exception as e:
            print(f"  ##Failed to save reward history: {e}")
            import traceback
            traceback.print_exc()


    def _to_dict(self, obj):

        if obj is None:
            return {}
        if hasattr(obj, '__dataclass_fields__'):
            try:
                return asdict(obj)
            except Exception:
                return {k: getattr(obj, k, None) for k in obj.__dataclass_fields__}
        if isinstance(obj, dict):
            return obj
        return {'raw': str(obj)}

    def _make_serializable(self, obj):

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        else:
            return obj

    def _print_assignments(self, outer_ep, assignments):

        print(f"\n  ## Attack Assignments for Outer Episode {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s}  System {a.target_system}  "
                  f"(mag={a.magnitude:.1f}, stealth={a.stealth_level:.1f})")
            if a.rationale:
                print(f"       Rationale: {a.rationale}")
