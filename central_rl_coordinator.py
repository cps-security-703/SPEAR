#!/usr/bin/env python3
"""
Central RL Coordinator — Two-Level Training Architecture

Architecture:
  OUTER LOOP (15 guidance circles):
    Two modes of operation:
    
    A) GEMINI-GUIDED (15 circles):
       - Gemini strategically assigns attack_type → system each circle
       - Gemini evaluates results after each circle and reassigns
       - Gemini decides rotation/reassignment — no forced schedule
       - Falls back to autonomous rotation if Gemini fails
    
    B) AUTONOMOUS (15 circles with SYSTEMATIC ROTATION):
       - Each outer circle rotates all 6 attacks to different systems
       - Circles 1-6:   Each attack visits every system once (full coverage)
       - Circles 7-12:  Second pass — refine training on all systems
       - Circles 13-15: Third pass — final tuning on first 3 systems
       - Agent weights & replay buffers persist across circles (cumulative training)
       - After all circles, every attack is trained on every system
    
  INNER LOOP (50 RL episodes per agent pair per circle):
    - Each of 6 attack-specialist pairs (DQN+SAC) trains against assigned system
    - Agents converge on optimal attack parameters for their assignment
    - Per-agent episode rewards tracked for convergence plots
    
  FINAL PHASE:
    - Pick best-performing system for each attack based on all circles
    - Deploy trained agents into hierarchical co-simulation with trained CMS
    - Any attack can be deployed on any system (fully trained)

Two Scenarios:
  1. Gemini-guided: Gemini decides all assignments strategically
  2. Autonomous: Systematic rotation ensures full coverage
"""

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

# Gemini deployment helpers (optional)
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


class AnnealPruneCallback(BaseCallback):
    """Relaxes the LLM-pruned action box toward the full action range as SAC
    training proceeds within a circle.

    Motivation (the honest SPEAR claim = "converges faster, not capped lower"):
    a FIXED prune band around the LLM's magnitude/stealth/duration hints speeds
    early convergence (sample efficiency) but, when the LLM biases toward low
    magnitude for stealth, permanently CAPS the impact ceiling below what the
    full-space autonomous agent reaches. Annealing keeps the tight band early
    (fast guidance) and linearly opens it to the full range by `anneal_frac` of
    the circle's budget (so the asymptotic ceiling is not capped).

    At progress p in [0,1] over the first `anneal_frac` of `anneal_steps`:
        band[dim] = lerp(tight_band[dim], full_range[dim], p)
    p=0 → tight LLM band; p>=1 → full range (no capping). The env reads
    `sac_env.action_prune` every step, so updating it here is sufficient.
    """

    def __init__(self, sac_env, tight_band: Dict[str, tuple],
                 full_range: Dict[str, tuple], anneal_steps: int,
                 anneal_frac: float = 0.6, verbose: int = 0):
        super().__init__(verbose)
        self._env = sac_env
        self._tight = {k: (float(v[0]), float(v[1])) for k, v in tight_band.items()}
        self._full = {k: (float(v[0]), float(v[1])) for k, v in full_range.items()}
        self._anneal_steps = max(1, int(anneal_steps * max(1e-6, anneal_frac)))
        self._start = None

    def _on_training_start(self) -> None:
        # num_timesteps is cumulative across circles (reset_num_timesteps=False),
        # so anchor progress to this circle's starting timestep.
        self._start = int(self.num_timesteps)
        self._env.action_prune = dict(self._tight)

    def _on_step(self) -> bool:
        if self._start is None:
            self._start = int(self.num_timesteps)
        p = (int(self.num_timesteps) - self._start) / float(self._anneal_steps)
        p = min(1.0, max(0.0, p))
        band = {}
        for dim, (tlo, thi) in self._tight.items():
            flo, fhi = self._full.get(dim, (tlo, thi))
            band[dim] = (tlo + p * (flo - tlo), thi + p * (fhi - thi))
        self._env.action_prune = band
        return True


@dataclass
class AttackAssignment:
    """A single attack_type → system mapping from Gemini or autonomous coordinator"""
    attack_type: str
    target_system: int
    magnitude: float = 0.7
    duration: float = 30.0
    stealth_level: float = 0.7
    priority: int = 1
    rationale: str = ""
    # Gemini-chosen ACN network (node 0-indexed) within target_system, ceded from
    # DQN in reduced-action mode.  None → autonomous / DQN selects the node.
    target_node: Optional[int] = None


@dataclass
class OuterEpisodeResult:
    """Result of one Gemini guidance episode (outer loop)"""
    outer_episode: int
    assignments: List[AttackAssignment]
    inner_results: Dict[str, Dict]  # attack_type → {rewards, best_reward, ...}
    aggregate_reward: float = 0.0
    aggregate_impact: float = 0.0
    aggregate_success_rate: float = 0.0
    aggregate_detection_risk: float = 0.0
    gemini_evaluation: str = ""
    reassignment_needed: bool = False


class CentralRLCoordinator:
    """
    Central coordinator that manages the two-level training loop.
    
    Level 1 (Outer): Gemini guidance episodes (15-20)
      - Gemini assigns attack_type → system mappings
      - Evaluates results and reassigns as needed
      
    Level 2 (Inner): RL training episodes per agent pair (50-100)
      - Each DQN+SAC pair trains against their assigned system
      - Convergence tracked per agent
    """
    
    def __init__(self, 
                 attack_coordinator: AttackSpecificCoordinator,
                 llm_analyzer=None,
                 system_analyzer=None,
                 num_systems: int = 6,
                 outer_episodes: int = 15,
                 inner_episodes: int = 50):
        """
        Args:
            attack_coordinator: AttackSpecificCoordinator with 12 agents (6 DQN + 6 SAC)
            llm_analyzer: Gemini LLM analyzer (None = autonomous mode)
            system_analyzer: Function to get system analysis data
            num_systems: Number of EVCS systems
            outer_episodes: Gemini guidance episodes (15-20)
            inner_episodes: RL training episodes per agent per outer episode (50-100)
        """
        self.attack_coordinator = attack_coordinator
        self.llm_analyzer = llm_analyzer
        self.system_analyzer = system_analyzer
        self.num_systems = num_systems
        self.outer_episodes = outer_episodes
        self.inner_episodes = inner_episodes
        self.attack_types = attack_coordinator.attack_types
        
        # Determine mode
        self.gemini_available = llm_analyzer is not None
        self.mode = "gemini_guided" if self.gemini_available else "autonomous"
        
        # ===== ROTATION SCHEDULE =====
        # Pre-compute which system each attack trains on in each outer circle.
        # With 6 attacks and 6 systems, attack_i in circle_c trains on:
        #   system = ((i + c) % num_systems) + 1   (1-indexed)
        # This guarantees every attack visits every system in 6 circles.
        # Circles 7-12 repeat the same rotation (second pass / refinement).
        # Circles 13-15 repeat again (third pass / final tuning).
        self.rotation_schedule = {}  # {outer_ep: {attack_type: system_id}}
        for c in range(self.outer_episodes):
            self.rotation_schedule[c] = {}
            for i, at in enumerate(self.attack_types):
                sys_id = ((i + c) % self.num_systems) + 1
                self.rotation_schedule[c][at] = sys_id
        
        # ===== REWARD TRACKING =====
        # Outer loop: one reward per guidance episode
        self.outer_episode_rewards = []
        self.outer_episode_results = []
        
        # Inner loop: per-agent convergence rewards
        # Structure: {attack_type: {outer_ep: {dqn_rewards: [...], sac_rewards: [...]}}}
        self.inner_episode_rewards = {at: {} for at in self.attack_types}
        
        # Per-attack per-system performance tracking (for best-system selection)
        # Structure: {attack_type: {system_id: [mean_reward_circle_1, ...]}}
        self.attack_system_performance = {
            at: {s: [] for s in range(1, self.num_systems + 1)}
            for at in self.attack_types
        }

        # ===== ADAPTIVE ASSIGNMENT STATE (Phase 1) =====
        # Consolidation signal: total measured impact achieved when >=2 attacks
        # were assigned to the same system in a circle.  Populated in
        # _aggregate_inner_results.  Structure: {system_id: [colocated_impact, ...]}
        # NOTE: this is the SUM of the co-located attacks' solo impacts — a real,
        # measured "how much can this system absorb if we pile attacks on it"
        # signal.  It is NOT physically-compounded super-additivity (that needs
        # a shared-state joint attack in the co-sim, which is Phase 2).  The
        # optional super-additivity assumption is modelled by SYNERGY_COEFF below.
        self.system_colocation_impact = {s: [] for s in range(1, self.num_systems + 1)}

        # Adaptive-assignment hyper-parameters (tunable knobs).
        #   EXPLORE_COEFF : UCB exploration weight (higher = explore longer).
        #   SYNERGY_COEFF : how strongly to bias toward consolidation.  0.0 =>
        #                   pure per-attack value (attacks spread by merit);
        #                   higher => attacks pile onto historically high-impact
        #                   systems.  Consolidation is thus ADAPTIVE, not forced.
        self.EXPLORE_COEFF = 0.30
        self.SYNERGY_COEFF = 0.20

        # ── LLM search-space pruning (KEYSTONE mechanism) ───────────────────────
        # When True, each circle narrows the SAC env's reachable action box to a
        # band around the coordinator/LLM's suggested magnitude/stealth/duration
        # BEFORE training — implementing the paper's "the LLM prunes the RL search
        # space" claim (as opposed to the pre-existing soft guidance_hints bonus,
        # which left SAC searching the full space and produced SPEAR≈autonomous).
        # Enabled only in LLM-guided mode so the autonomous baseline keeps the
        # full search space → clean ablation isolating the pruning benefit.
        # Reversible: set to False to restore exact prior behaviour.
        self.llm_prune_action = bool(getattr(self, 'gemini_available', False))
        self.prune_halfwidth = {'magnitude': 0.30, 'stealth_level': 0.15, 'duration': 10.0}
        # Pruning geometry (SPEAR_PRUNE_MODE):
        #   'floor'  (default) — STATIONARY band: the LLM hint sets a FLOOR
        #       (lo = max(full_lo, hint - halfwidth)) but the ceiling stays at the
        #       FULL range (hi = full_hi). Removes the unproductive low-magnitude /
        #       low-stealth dead-zone to speed convergence, while NEVER capping the
        #       impact ceiling. Fixed for the whole circle → no non-stationarity.
        #   'band'   — original symmetric tight band (hint ± halfwidth). Caps both
        #       sides; this was shown to cap impact when the LLM biases magnitude low.
        #   'anneal' — tight band that opens to full range over the circle (fast
        #       early guidance, uncapped late), but makes the action→param remap
        #       non-stationary, which degraded the training-reward curve.
        self.prune_mode = os.environ.get('SPEAR_PRUNE_MODE', 'floor').strip().lower()
        self.prune_anneal_frac = float(os.environ.get('SPEAR_PRUNE_ANNEAL_FRAC', '0.6'))

        # Final deployment plan (produced after all outer episodes)
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
            print(f"   Assignment: Adaptive UCB bandit (coverage → value + consolidation)")
            print(f"   Explore coeff={self.EXPLORE_COEFF}, synergy coeff={self.SYNERGY_COEFF}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    
    def run_two_level_training(self, 
                                system_analysis_data=None,
                                stride_threats: Dict = None,
                                mitre_tactics: Dict = None) -> Dict:
        """
        Run the complete two-level training loop.
        
        STRIDE/MITRE mappings are pre-established via agentic RAG
        (see knowledgebase_mapping.md). Gemini's role during training
        is to evaluate RL feedback and reassign attack→system mappings.
        stride_threats and mitre_tactics params kept for backward compat but unused.
        
        Returns:
            Dict with outer_rewards, inner_rewards, final_deployment, etc.
        """
        print(f"\n{'#'*70}")
        print(f"# TWO-LEVEL TRAINING: {self.mode.upper()} MODE")
        print(f"# Outer: {self.outer_episodes} guidance episodes")
        print(f"# Inner: {self.inner_episodes} RL episodes per agent per guidance round")
        print(f"{'#'*70}\n")
        
        previous_assignments = None
        previous_results = None
        gemini_success_count = 0  # Track how many circles actually used Gemini
        
        for outer_ep in range(self.outer_episodes):
            # Determine which pass this circle belongs to
            if outer_ep < self.num_systems:
                pass_label = "PASS 1 (full coverage)"
            elif outer_ep < 2 * self.num_systems:
                pass_label = "PASS 2 (refinement)"
            else:
                pass_label = "PASS 3 (final tuning)"
            
            print(f"\n{'='*60}")
            print(f"  OUTER CIRCLE {outer_ep + 1}/{self.outer_episodes}  [{pass_label}]")
            print(f"{'='*60}")
            
            # ----- STEP 1: Get attack assignments (Gemini feedback-based or rotation) -----
            # Reset per-circle call-id list so update_rl_impact() only back-fills
            # calls that belong to THIS outer episode.
            if self.llm_analyzer is not None:
                self.llm_analyzer._episode_call_ids = []

            self._last_assignment_source = 'autonomous'  # Default
            assignments = self._get_attack_assignments(
                outer_ep=outer_ep,
                previous_assignments=previous_assignments,
                previous_results=previous_results
            )
            if self._last_assignment_source == 'gemini':
                gemini_success_count += 1

            self._print_assignments(outer_ep, assignments)

            # ----- STEP 2: Inner loop — train each agent pair -----
            inner_results = self._run_inner_training_loop(outer_ep, assignments)

            # ----- STEP 3: Aggregate results -----
            outer_result = self._aggregate_inner_results(outer_ep, assignments, inner_results)

            self.outer_episode_rewards.append(outer_result.aggregate_reward)
            self.outer_episode_results.append(outer_result)

            print(f"\n  ## Outer Episode {outer_ep + 1} Summary:")
            print(f"     Aggregate Reward:    {outer_result.aggregate_reward:.2f}")
            print(f"     Success Rate:        {outer_result.aggregate_success_rate:.1%}")
            print(f"     Total Impact:        {outer_result.aggregate_impact:.4f}")
            print(f"     Avg Detection Risk:  {outer_result.aggregate_detection_risk:.2%}")

            # ----- STEP 4: Back-fill RL-impact columns in LLM metrics -----
            # Now that the outer-episode reward is known, annotate every LLM
            # call made during STEP 1 with reward_before / reward_after / delta.
            try:
                from llm_metrics_logger import LLMMetricsLogger
                _logger = LLMMetricsLogger.instance()
                _prev_reward = self.outer_episode_rewards[-2] if len(self.outer_episode_rewards) >= 2 else 0.0
                _call_ids = getattr(self.llm_analyzer, '_episode_call_ids', [])
                # Also include _last_call_id as a safety net
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
        
        # Update mode based on actual Gemini usage
        if self.mode == 'gemini_guided' and gemini_success_count == 0:
            print(f"\n  ##Gemini was never successfully used ({gemini_success_count}/{self.outer_episodes} circles)")
            print(f"  ## Downgrading mode: gemini_guided → autonomous")
            self.mode = 'autonomous'
        elif self.mode == 'gemini_guided':
            print(f"\n  ## Gemini was used in {gemini_success_count}/{self.outer_episodes} circles")
        
        # ----- FINAL PHASE: Produce deployment plan -----
        print(f"\n{'='*60}")
        print(f"  FINAL PHASE: Producing Deployment Plan")
        print(f"{'='*60}")
        self.final_deployment_plan = self._produce_final_deployment_plan()
        
        # Build complete results
        results = self._build_training_results()
        
        # Save reward history
        self._save_two_level_reward_history(results)
        
        return results
    
    # =========================================================================
    # STEP 1: GET ATTACK ASSIGNMENTS (Gemini or Autonomous)
    # =========================================================================
    
    def _get_attack_assignments(self, outer_ep=0, previous_assignments=None,
                                 previous_results=None) -> List[AttackAssignment]:
        """Get attack→system assignments.
        
        Gemini mode: Gemini evaluates RL feedback and reassigns.
        Autonomous mode: Deterministic rotation schedule.
        """
        if self.gemini_available and GEMINI_DEPLOYMENT_AVAILABLE:
            return self._get_gemini_assignments(
                outer_ep, previous_assignments, previous_results
            )
        else:
            return self._get_adaptive_assignments(outer_ep)
    
    def _node_layout_desc(self, systems=None) -> str:
        """Compact ACN node listing for the Gemini prompt, so Gemini can pick a
        target_node within each assigned DS (reduced-action node ceding).  Empty
        string when node-level layout is unavailable → prompt stays node-free."""
        lay = getattr(self.attack_coordinator, 'network_layout', None)
        if not lay:
            return ""
        try:
            from acn_network_layout import describe_layout_for_prompt
            return describe_layout_for_prompt(lay, systems=systems)
        except Exception:
            return ""

    def _get_gemini_assignments(self, outer_ep, previous_assignments, previous_results) -> List[AttackAssignment]:
        """Ask Gemini to assign attack→system based on RL feedback.
        
        STRIDE/MITRE mappings are pre-established (knowledgebase_mapping.md).
        Gemini evaluates RL agent performance and rotates assignments.
        Falls back to autonomous rotation if Gemini is unavailable or quota exhausted.
        """
        # ── Guard: check if Gemini is actually available ──
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
                    # No node_layout_desc: Gemini assigns the DS only; the ACN
                    # node is selected by the trained DQN, not Gemini.
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
                # Build CUMULATIVE coverage: which systems each attack has trained on so far
                cumulative_coverage = {}
                for at in self.attack_types:
                    systems_trained = []
                    for s in range(1, self.num_systems + 1):
                        if self.attack_system_performance.get(at, {}).get(s, []):
                            systems_trained.append(s)
                    cumulative_coverage[at] = systems_trained
                
                # Pass circle number, coverage, and system info so Gemini can rotate
                prompt = create_gemini_adaptation_prompt(
                    prev_exec_results, prev_strategy,
                    circle_num=outer_ep + 1,
                    total_circles=self.outer_episodes,
                    num_systems=self.num_systems,
                    cumulative_coverage=cumulative_coverage
                    # No node_layout_desc: DQN selects the ACN node, not Gemini.
                )
            
            # Call Gemini with the deployment prompt directly
            gemini_input = {
                'deployment_prompt': prompt
            }
            llm_response = self.llm_analyzer.analyze_threats(gemini_input)
            
            # Parse response into deployments
            if isinstance(llm_response, dict) and 'llm_response' in llm_response:
                llm_text = llm_response['llm_response']
            else:
                llm_text = llm_response
            
            deployments = parse_gemini_deployment_response(llm_text)
            
            # ── Detect static fallback (Gemini didn't actually respond) ──
            # The fallback always produces EXACTLY 6 deployments with
            # attack_i → system_(i+1) in order.  Only flag as static if
            # ALL 6 match that exact pattern.
            is_static_fallback = (len(deployments) == len(self.attack_types))
            if is_static_fallback:
                fallback_mapping = {at: s for at, s in zip(self.attack_types, range(1, self.num_systems + 1))}
                for dep in deployments:
                    expected_sys = fallback_mapping.get(dep.attack_type)
                    if expected_sys is None or dep.target_systems != [expected_sys]:
                        is_static_fallback = False
                        break
            
            if is_static_fallback and outer_ep > 0:
                # Gemini returned the static fallback — use the adaptive bandit instead
                print(f"  ##Gemini returned static fallback pattern (parser fell back)")
                print(f"  ## Using ADAPTIVE assignment (bandit) for circle {outer_ep + 1}")
                self._last_assignment_source = 'adaptive'
                return self._get_adaptive_assignments(outer_ep)

            # Convert deployments to assignments (one attack_type → one system)
            assignments = self._deployments_to_assignments(deployments)
            if len(assignments) < len(self.attack_types):
                # Partial recovery — supplement missing attacks adaptively
                existing_types = {a.attack_type for a in assignments}
                rotation = self._get_adaptive_assignments(outer_ep)
                for ra in rotation:
                    if ra.attack_type not in existing_types:
                        assignments.append(ra)
                print(f"  🔧 Gemini provided {len(assignments) - len(rotation) + len([r for r in rotation if r.attack_type not in existing_types])} assignments, supplemented {len(assignments) - len([a for a in assignments if a.attack_type in existing_types])} from rotation")
            
            if is_static_fallback:
                print(f"  ## Using initial deployment ({len(assignments)} assignments)")
            else:
                print(f"  ## Gemini provided {len(assignments)} strategic assignments")
            self._last_assignment_source = 'gemini'
            return assignments
            
        except Exception as e:
            print(f"  ❌ Gemini assignment failed: {e}")
            print(f"  ## Falling back to ADAPTIVE assignment (bandit) for circle {outer_ep + 1}")
            self._last_assignment_source = 'adaptive'
            return self._get_adaptive_assignments(outer_ep)
    
    def _get_adaptive_assignments(self, outer_ep: int) -> List[AttackAssignment]:
        """
        ADAPTIVE assignment (Phase 1) — replaces the fixed round-robin.

        Per attack type, pick the target system from learned RL feedback rather
        than a hard-coded schedule:

          (1) COVERAGE FIRST (cold start): if any system is still unvisited for
              this attack, pick among the unvisited ones — the least globally
              used system this circle — so every (attack, system) pair is seen
              once before the bandit starts exploiting.  This is data-driven
              (keyed on visit counts), not a fixed rotation: it self-corrects if
              coverage is uneven.  Needs > num_systems circles to finish.

          (2) EXPLOIT + CONSOLIDATE: once all systems have been tried for this
              attack, score each system with a UCB term
                  value_norm(at,s) + EXPLORE_COEFF * sqrt(ln(N+1)/(n+1))
              plus a consolidation bonus
                  SYNERGY_COEFF * (#attacks already on s this circle) * coloc_norm(s)
              and take the argmax.  Repeats are allowed, so multiple attacks can
              land on the same system when that system's measured co-located
              impact (coloc_norm) makes piling on worthwhile.  If SYNERGY_COEFF
              is 0 or no co-location history exists, attacks spread by merit.
              => consolidation vs spread is chosen adaptively, not forced.

        Used in autonomous mode and as the gap-filler / scorer for Gemini mode.
        """
        # Value normalization range across all recorded (attack, system) means.
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

        # Co-location impact normalization (measured consolidation signal).
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
                rationale=f"Adaptive circle {outer_ep + 1}: {at} → System {s_choice} [{source}]"
            ))

        self._last_assignment_source = 'adaptive'

        print(f"  ## Adaptive assignments for circle {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s} → System {a.target_system}")
        consolidated = {s: c for s, c in system_count.items() if c > 1}
        if consolidated:
            summary = ", ".join(f"System {s}×{c}" for s, c in sorted(consolidated.items()))
            print(f"     🔗 Consolidation this circle: {summary}")
        else:
            print(f"     ↔️  Spread across systems (no consolidation this circle)")

        return assignments

    def _get_autonomous_assignments(self, outer_ep: int) -> List[AttackAssignment]:
        """
        Systematic rotation: attack_i in circle_c trains on system ((i+c) % 6) + 1.
        
        This guarantees:
          - Circles 1-6:   Every attack visits every system once (full coverage)
          - Circles 7-12:  Second pass (refinement with accumulated weights)
          - Circles 13-15: Third pass (final tuning)
          - Agent weights & replay buffers persist → cumulative training
          - After all circles, any attack can be deployed on any system
        """
        schedule = self.rotation_schedule[outer_ep]
        
        assignments = []
        for attack_type in self.attack_types:
            system_id = schedule[attack_type]
            assignments.append(AttackAssignment(
                attack_type=attack_type,
                target_system=system_id,
                magnitude=0.7,
                stealth_level=0.7,
                rationale=f"Rotation circle {outer_ep + 1}: {attack_type} → System {system_id}"
            ))
        
        print(f"  ## Rotation schedule for circle {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s} → System {a.target_system}")
        
        return assignments
    
    def _deployments_to_assignments(self, deployments: List[AttackDeployment]) -> List[AttackAssignment]:
        """Convert Gemini's AttackDeployment list to one-to-one AttackAssignment list."""
        assignments = []
        used_attack_types = set()
        
        for dep in deployments:
            # Take the first target system for each attack type
            target = dep.target_systems[0] if dep.target_systems else 1
            if dep.attack_type not in used_attack_types:
                # Validate Gemini's node choice against the DS's node range.
                tnode = getattr(dep, 'target_node', None)
                n_nodes = int(getattr(self.attack_coordinator, 'n_nodes', 10))
                if tnode is not None:
                    tnode = int(tnode) % max(1, n_nodes)
                assignments.append(AttackAssignment(
                    attack_type=dep.attack_type,
                    target_system=target,
                    magnitude=dep.magnitude,
                    duration=dep.duration,
                    stealth_level=dep.stealth_level,
                    priority=dep.priority,
                    target_node=tnode
                ))
                used_attack_types.add(dep.attack_type)
        
        # Fill in any missing attack types using recorded performance (best
        # system so far), allowing repeats so consolidation is NOT artificially
        # prevented — Gemini or the data may prefer piling attacks on one system.
        # Falls back to spreading only when there is no history yet.
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
                rationale=f"Adaptive gap-fill → System {system_id}"
                          f"{' (best recorded)' if best_sys is not None else ' (no history)'}"
            ))

        return assignments
    
    # =========================================================================
    # STEP 2: INNER TRAINING LOOP (RL episodes per agent pair)
    # =========================================================================
    
    def _build_prune_band(self, sac_env, assignment) -> Dict[str, tuple]:
        """Build the LLM-pruned action box around the assignment's suggested
        magnitude/stealth/duration, clipped to the env's full range.

        mode='floor' (default): the LLM hint sets a FLOOR only
            band = (max(full_lo, hint - halfwidth), full_hi)
        so SAC searches from the hint upward with the impact ceiling intact —
        stationary, and cannot cap impact. For a low hint this approaches the
        full range (→ autonomous parity); for an aggressive hint it trims the
        unproductive low tail (→ faster convergence).

        mode='band' (legacy): symmetric tight band hint ± halfwidth (caps both
        sides). Degenerate bands (width <= 0) fall back to the full range.
        """
        hw = getattr(self, 'prune_halfwidth',
                     {'magnitude': 0.30, 'stealth_level': 0.15, 'duration': 10.0})
        mode = getattr(self, 'prune_mode', 'floor')
        full = sac_env._full_action_range
        centers = {'magnitude': assignment.magnitude,
                   'stealth_level': assignment.stealth_level,
                   'duration': assignment.duration}
        band = {}
        for dim, c in centers.items():
            flo, fhi = full[dim]
            if mode == 'floor':
                lo = max(flo, float(c) - hw[dim]); hi = fhi
            else:  # 'band' (and 'anneal' uses this as its tight starting band)
                lo = max(flo, float(c) - hw[dim]); hi = min(fhi, float(c) + hw[dim])
            band[dim] = (lo, hi) if hi > lo else (flo, fhi)
        return band

    def _run_inner_training_loop(self, outer_ep: int,
                                  assignments: List[AttackAssignment]) -> Dict[str, Dict]:
        """
        For each attack assignment, run actual SB3 .learn() training.
        
        This calls agent.learn(total_timesteps=N) which performs real gradient
        updates via replay buffer, unlike the old manual reset()/step() loop
        that only ran inference without updating weights.
        
        The environment's forced_target_system is set so all training steps
        target the assigned system for this rotation circle.
        
        Returns:
            Dict[attack_type → {dqn_rewards, sac_rewards, best_reward, best_impact, ...}]
        """
        # Convert inner_episodes to timesteps for SB3
        # Each "episode" in the env is max_steps=1000 timesteps.
        # inner_episodes=50 means 50 episodes worth of training.
        # We use inner_episodes * 1000 timesteps = 50,000 per agent per circle.
        # This ensures 50+ episodes complete and are captured by the callback.
        timesteps_per_agent = self.inner_episodes * 1000
        
        print(f"\n  ## Inner Training Loop: {timesteps_per_agent} timesteps per agent "
              f"(~{self.inner_episodes} episodes)")
        
        inner_results = {}
        
        for assignment in assignments:
            at = assignment.attack_type
            sys_id = assignment.target_system
            
            print(f"\n    ##Training {at} agents on System {sys_id} "
                  f"({timesteps_per_agent} timesteps)...")
            
            # Get environments for this attack type
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

            # ===== FOCUS training on the ASSIGNED system (Phase-1-full) =====
            # SAC gradient training is locked to this circle's assigned system so
            # the policy actually learns to attack WHERE the outer loop sent it —
            # this is what makes the adaptive assignment (and consolidation, when
            # multiple attacks share a system) something the agents LEARN, not
            # just something the planner scores.
            #
            # Prior code rotated across all systems for generalisation, because
            # at deploy time any (attack, system) pair could be picked.  That is
            # neutralised here by an invariant enforced downstream: the final
            # deployment plan only deploys (attack, system) pairs that were
            # actually trained (best_sys is chosen from recorded performance in
            # _produce_final_deployment_plan).  Coverage across circles comes from
            # the adaptive assignment's explore-coverage phase.
            # NOTE: DQN's env is intentionally left rotating (see below) — DQN is
            # unchanged in Phase 1 and repurposed to node selection in Phase 2.
            sac_env.forced_target_system = [sys_id]
            # Node stays a per-step random draw during SAC training (DQN learns
            # the node separately); never ceded to Gemini under the final design.
            sac_env.forced_target_node = None
            _reduced = getattr(self.attack_coordinator, 'reduced_action', False)

            # ===== SET GUIDANCE HINTS from coordinator assignment =====
            # Gemini mode: magnitude/duration/stealth come from Gemini's
            #   strategic analysis (e.g., lower magnitude for stealth attacks,
            #   longer duration for persistence).
            # Autonomous mode: uses defaults (0.7/30.0/0.7).
            # The guidance hints add a reward bonus for staying close to
            # these values, creating differentiated training dynamics.
            sac_env.guidance_hints = {
                'magnitude': assignment.magnitude,
                'duration': assignment.duration,
                'stealth_level': assignment.stealth_level
            }
            # Remember the ceded dims per attack so the final deployment plan can
            # recover Gemini's duration/stealth when SAC learned magnitude only
            # (reduced-action mode).
            if not hasattr(self, '_last_assignment_hints'):
                self._last_assignment_hints = {}
            self._last_assignment_hints[at] = {
                'magnitude': assignment.magnitude,
                'duration': assignment.duration,
                'stealth_level': assignment.stealth_level,
                'target_node': getattr(assignment, 'target_node', None)
            }
            print(f"      📐 Guidance hints: mag={assignment.magnitude:.2f}, "
                  f"dur={assignment.duration:.1f}s, stealth={assignment.stealth_level:.2f}")

            # ===== KEYSTONE: LLM search-space pruning =====
            # Narrow SAC's reachable action box to a band around the LLM's
            # suggested params so the agent searches ONLY the LLM-curated region.
            # Autonomous mode (llm_prune_action False) leaves it None → full space.
            _anneal_prune_cb = None
            if getattr(self, 'llm_prune_action', False):
                _pmode = getattr(self, 'prune_mode', 'floor')
                _band = self._build_prune_band(sac_env, assignment)
                sac_env.action_prune = _band
                if _pmode == 'anneal':
                    _anneal_prune_cb = AnnealPruneCallback(
                        sac_env, _band, sac_env._full_action_range,
                        anneal_steps=timesteps_per_agent,
                        anneal_frac=getattr(self, 'prune_anneal_frac', 0.6),
                    )
                    print(f"      ✂️  Pruned action box (anneal→full by "
                          f"{getattr(self, 'prune_anneal_frac', 0.6):.0%}): {_band}")
                else:
                    print(f"      ✂️  Pruned action box ({_pmode}, "
                          f"ceiling={'full' if _pmode=='floor' else 'capped'}): {_band}")
            else:
                sac_env.action_prune = None

            # DQN target focus depends on the phase:
            #   Phase 1 (system-level): DQN learns SYSTEM selection → do NOT force
            #     its target; rotate the continuous env across all systems so its
            #     observations span them and it picks its preferred system.
            #   Phase 2 (node_level):  the outer loop already picked the DS, so
            #     FOCUS DQN's continuous env on the assigned system — DQN now
            #     learns NODE (ACN-network) selection WITHIN that DS.
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
            
            # ===== Train SAC agent (parameter optimization) =====
            sac_callback = EpisodeRewardCallback()
            _sac_cb = sac_callback
            if _anneal_prune_cb is not None:
                from stable_baselines3.common.callbacks import CallbackList
                _sac_cb = CallbackList([sac_callback, _anneal_prune_cb])
            try:
                self.attack_coordinator.sac_agents[at].learn(
                    total_timesteps=timesteps_per_agent,
                    callback=_sac_cb,
                    reset_num_timesteps=False,  # Continue from previous training
                    progress_bar=False
                )
                sac_results = sac_callback.get_results()
                sac_rewards = sac_results.get('episode_rewards', [])
                print(f"      SAC: {sac_results['num_episodes']} episodes, "
                      f"mean_reward={sac_results['mean_reward']:.3f}")
            except Exception as e:
                print(f"      ##SAC training error: {e}")
                sac_rewards = []
            
            # ===== Train DQN agent (node selection within the DS) =====
            # DQN learns the ACN node in BOTH modes — the node is always an RL
            # decision (Gemini never picks it under the final design).
            dqn_rewards = []
            dqn_callback = EpisodeRewardCallback()
            try:
                self.attack_coordinator.dqn_agents[at].learn(
                    total_timesteps=timesteps_per_agent,
                    callback=dqn_callback,
                    reset_num_timesteps=False,  # Continue from previous training
                    progress_bar=False
                )
                dqn_results = dqn_callback.get_results()
                dqn_rewards = dqn_results.get('episode_rewards', [])
                print(f"      DQN: {dqn_results['num_episodes']} episodes, "
                      f"mean_reward={dqn_results['mean_reward']:.3f}")
            except Exception as e:
                # Print FULL traceback so silent DQN failures (like the
                # "unhashable type: 'list'" bug from Fix-A list rotation)
                # don't get masked behind a one-line error message.
                import traceback
                print(f"      ##DQN training error: {e}")
                traceback.print_exc()
                dqn_rewards = []

            # ===== UNLOCK environment =====
            sac_env.forced_target_system = None
            sac_env.guidance_hints = None
            sac_env.action_prune = None
            sac_env.forced_target_node = None
            if dqn_env_key in self.attack_coordinator.environments:
                dqn_env = self.attack_coordinator.environments[dqn_env_key]
                if hasattr(dqn_env, 'continuous_env'):
                    dqn_env.continuous_env.forced_target_system = None
                    dqn_env.continuous_env.guidance_hints = None
            
            # ===== Post-training evaluation (10 episodes) =====
            best_reward = -float('inf')
            best_impact = 0.0
            total_successes = 0
            total_detection_risk = 0.0
            eval_rewards = []
            num_eval = 10
            
            sac_env.forced_target_system = sys_id  # Lock for eval too
            # Evaluate on the FULL action range (action_prune=None) so the trained
            # policy's achievable impact is measured the SAME way as autonomous —
            # a fair, symmetric comparison. Previously eval re-applied the tight
            # LLM band, which artificially capped Gemini's reported impact below
            # autonomous even when the policy (trained under annealed pruning) had
            # learned higher-magnitude actions. With annealing the band is already
            # the full range by end-of-training, so this is consistent with how the
            # policy was trained in its final phase.
            sac_env.action_prune = None
            # Reduced (Gemini) mode: SAC's action carries magnitude only, so the
            # env needs Gemini's duration/stealth via guidance_hints to build the
            # full attack during eval (they were cleared after training above).
            if getattr(sac_env, 'reduced_action', False):
                # Reduced mode: SAC's action is [magnitude, stealth]; supply
                # Gemini's DURATION via guidance_hints so eval builds the full
                # attack.  Node stays random (matches training).
                sac_env.guidance_hints = {
                    'magnitude': assignment.magnitude,
                    'duration': assignment.duration,
                    'stealth_level': assignment.stealth_level
                }
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
            sac_env.forced_target_system = None  # Unlock after eval
            sac_env.action_prune = None
            if getattr(sac_env, 'reduced_action', False):
                sac_env.guidance_hints = None
                sac_env.forced_target_node = None
            
            # ===== UPDATE CROSS-CIRCLE MEMORY =====
            # Persist this circle's detection/success/impact stats into the
            # environment's cross_circle_stats so the NEXT circle's SAC agent
            # starts with knowledge of what happened on this system historically.
            circle_detection_rate = total_detection_risk / num_eval  # avg detection risk as proxy
            circle_success_rate = total_successes / num_eval
            circle_avg_impact = best_impact  # best impact this circle
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
            
            # Combine training + eval rewards
            all_sac_rewards = sac_rewards if sac_rewards else eval_rewards
            
            # Store inner results
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
            
            # Store in tracking dict. Persist the CLEAN per-attack physical metrics
            # (impact / success / detection) alongside the shaped rewards so the
            # gemini-vs-autonomous comparison can judge attack QUALITY per attack
            # type — not just the bimodal, guidance-confounded reward. These come
            # from the co-simulation, so they are guidance-bonus-free.
            # Convergence tracking: cumulative SAC timesteps for this attack, plus
            # a within-circle "convergence episode" (where the smoothed training
            # reward first reaches 90% of its min→max range).  Reduced-action
            # (Gemini) runs should reach comparable performance at FEWER cumulative
            # timesteps / earlier convergence episode than full-action (autonomous)
            # runs — that is the sample-efficiency hypothesis.
            if not hasattr(self, '_cum_timesteps'):
                self._cum_timesteps = {}
            self._cum_timesteps[at] = self._cum_timesteps.get(at, 0) + timesteps_per_agent

            def _conv_ep(rewards):
                """Within-circle convergence episode: index where the smoothed
                reward first reaches 90% of its min→max span AND the curve
                actually rose (final level meaningfully above the start).

                Returns None for degenerate curves (too short, flat, or no net
                improvement) rather than 0 — a flat curve is 'no learning signal',
                NOT 'converged instantly', and reporting 0 there biases the
                sample-efficiency comparison. The authoritative cross-circle
                sample-efficiency metric is computed in spear_sample_efficiency.py
                from the concatenated per-circle trajectories; this per-circle
                value is a cheap in-run diagnostic only."""
                r = np.asarray(rewards, dtype=float)
                if r.size < 5:
                    return None
                w = max(1, r.size // 10)
                sm = np.convolve(r, np.ones(w) / w, mode='valid')
                rng = float(sm.max() - sm.min())
                # Flat within noise → no convergence signal.
                scale = abs(float(np.mean(sm))) + 1e-6
                if rng <= 0.05 * scale:
                    return None
                # Require a net upward trend (end third above first third).
                third = max(1, sm.size // 3)
                if float(np.mean(sm[-third:])) <= float(np.mean(sm[:third])):
                    return None
                thr = sm.min() + 0.9 * rng
                return int(np.argmax(sm >= thr))

            self.inner_episode_rewards[at][outer_ep] = {
                'dqn_rewards': dqn_rewards,
                'sac_rewards': all_sac_rewards,
                'assigned_system': sys_id,
                'best_impact': float(best_impact),
                'success_rate': float(total_successes / num_eval),
                'avg_detection_risk': float(total_detection_risk / num_eval),
                # convergence / sample-efficiency instrumentation
                'reduced_action': bool(getattr(sac_env, 'reduced_action', False)),
                'sac_mean_reward': float(np.mean(sac_rewards)) if sac_rewards else 0.0,
                'sac_num_episodes': len(sac_rewards) if sac_rewards else 0,
                'cum_timesteps': int(self._cum_timesteps[at]),
                'convergence_episode': _conv_ep(sac_rewards),
            }
            
            print(f"    ## {at} on System {sys_id}: "
                  f"mean={inner_results[at]['mean_reward']:.3f}, "
                  f"best={best_reward:.3f}, "
                  f"success={inner_results[at]['success_rate']:.0%}")
        
        return inner_results
    
    # =========================================================================
    # STEP 3: AGGREGATE RESULTS
    # =========================================================================
    
    def _aggregate_inner_results(self, outer_ep: int, 
                                  assignments: List[AttackAssignment],
                                  inner_results: Dict[str, Dict]) -> OuterEpisodeResult:
        """Aggregate inner loop results into an outer episode result."""
        
        all_rewards = []
        all_impacts = []
        all_success_rates = []
        all_detection_risks = []
        
        for at, ir in inner_results.items():
            all_rewards.append(ir.get('mean_reward', 0.0))
            all_impacts.append(ir.get('best_impact', 0.0))
            all_success_rates.append(ir.get('success_rate', 0.0))
            all_detection_risks.append(ir.get('avg_detection_risk', 0.5))
        
        # ===== Measure CONSOLIDATION signal (co-located impact) =====
        # Group this circle's assignments by system; when >=2 attacks share a
        # system, record the SUM of their impacts as that system's co-located
        # impact.  This measured data is what the adaptive assigner reads to
        # decide whether piling attacks onto a system is worthwhile.  (Physical
        # super-additive compounding of simultaneous attacks needs shared-state
        # co-sim execution — that is Phase 2.)
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
            print(f"    🔗 Consolidation measured: co-located impact={colocation_bonus:.4f} "
                  f"across {n_consolidated} system(s)")

        # Composite reward (prior formula + measured consolidation bonus)
        avg_success = np.mean(all_success_rates) if all_success_rates else 0.0
        total_impact = sum(all_impacts)
        avg_detection = np.mean(all_detection_risks) if all_detection_risks else 0.5
        stealth_score = 1.0 - avg_detection

        composite_reward = (
            avg_success * 1000.0 +
            total_impact * 500.0 +
            stealth_score * 300.0 +
            self.SYNERGY_COEFF * colocation_bonus * 500.0  # measured consolidation bonus
        )
        
        # Record per-attack per-system performance for best-system selection
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
    
    # =========================================================================
    # FINAL PHASE: PRODUCE DEPLOYMENT PLAN
    # =========================================================================
    
    def _dqn_qvalues(self, dqn_agent, obs):
        """Return the DQN's Q-values for all discrete actions (one per ACN node),
        or None on failure.  Used to rank networks for top-K subset selection."""
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
        """
        After all outer episodes, produce the final deployment plan.
        
        Two stages:
          1. RL-based: Pick the best system for each attack from performance data.
          2. RAG-enhanced: Query the ARES RAG vector DB for real-world threat
             intelligence (CVEs, MITRE techniques, mitigations) and ask Gemini
             to produce a research-quality rationale grounding RL results in
             real-world evidence.  Falls back to pure-RL plan if RAG unavailable.
        """
        print("  ## Analyzing per-attack per-system performance across all circles...")
        
        if not self.outer_episode_results:
            return {'deployments': [], 'status': 'no_results'}
        
        # ── Stage 1: RL-based best-system selection (unchanged) ──
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
            
            # ── Query trained SAC agent for its learned parameters ──
            # SAC action space: [magnitude, duration, stealth, target_system_id]
            # The agent has learned optimal values through training.
            agent_magnitude = 0.7   # fallback
            agent_duration = 30.0   # fallback
            agent_stealth = 0.7     # fallback
            agent_params_source = 'fallback'
            
            sac_agent = self.attack_coordinator.sac_agents.get(at)
            sac_env_key = f'sac_{at}'
            if sac_agent and sac_env_key in self.attack_coordinator.environments:
                try:
                    env = self.attack_coordinator.environments[sac_env_key]
                    env.forced_target_system = best_sys
                    obs, _ = env.reset()
                    trained_action, _ = sac_agent.predict(obs, deterministic=True)
                    ta = np.atleast_1d(trained_action)
                    # Extract learned parameters from the action vector.
                    agent_magnitude = float(np.clip(ta[0], 0.1, 2.0))
                    if getattr(env, 'reduced_action', False):
                        # Reduced (Gemini) mode: SAC action = [magnitude, stealth];
                        # only DURATION was ceded to Gemini, so take it from the
                        # last Gemini assignment for this attack.
                        _h = getattr(self, '_last_assignment_hints', {}).get(at, {})
                        agent_stealth = float(np.clip(ta[1], 0.0, 1.0))
                        agent_duration = float(np.clip(_h.get('duration', 30.0), 5.0, 60.0))
                        agent_params_source = 'trained_sac_agent(mag,stealth)+gemini(dur)'
                    else:
                        agent_duration = float(np.clip(ta[1], 5.0, 60.0))
                        agent_stealth = float(np.clip(ta[2], 0.0, 1.0))
                        agent_params_source = 'trained_sac_agent'
                    env.forced_target_system = None
                except Exception as e:
                    print(f"     ##Could not query SAC agent for {at}: {e}")

            # ── Query trained DQN agent for its learned TARGET selection ──
            # Previously the plan ignored DQN entirely (target came only from
            # SAC-eval performance).  Now DQN's learned choice is wired in:
            #   Phase 2 (node_level): DQN picks the ACN network (node) WITHIN the
            #     best system → written as `target_node` (drives node-level test).
            #   Phase 1 (system-level): DQN's preferred system is recorded as
            #     metadata (`dqn_choice`) without overriding the performance-best
            #     target, to keep Phase-1 plan behaviour stable.
            node_level = getattr(self.attack_coordinator, 'node_level', False)
            target_node = None
            target_nodes = None
            dqn_choice = None
            dqn_agent = self.attack_coordinator.dqn_agents.get(at)
            dqn_env = self.attack_coordinator.environments.get(f'dqn_{at}')
            if dqn_agent is not None and dqn_env is not None:
                try:
                    if node_level and hasattr(dqn_env, 'continuous_env'):
                        # Focus DQN on the chosen DS so its actions rank the networks.
                        dqn_env.continuous_env.forced_target_system = [best_sys]
                    obs, _ = dqn_env.reset()
                    if node_level:
                        n_nodes = getattr(self.attack_coordinator, 'n_nodes', 10)
                        cap = int(getattr(self.attack_coordinator, 'topk_networks', 7))
                        cap = max(1, min(cap, n_nodes))
                        # TOP-K subset selection: rank the ACN networks by the DQN's
                        # learned Q-values and take the best `cap` of them (instead of
                        # a single argmax).  This realises "DQN picks a subset of
                        # networks to attack (max K of 10)".
                        qvals = self._dqn_qvalues(dqn_agent, obs)
                        if qvals is not None and len(qvals) >= n_nodes:
                            order = list(np.argsort(qvals[:n_nodes])[::-1])
                            target_nodes = [int(x) for x in order[:cap]]
                        else:
                            act, _ = dqn_agent.predict(obs, deterministic=True)
                            target_nodes = [int(act) % n_nodes]
                        target_node = target_nodes[0]  # primary (for backward compat)
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
                'dqn_choice': dqn_choice,       # metadata: what DQN picked
                'systems_trained_on': [
                    s for s in range(1, self.num_systems + 1)
                    if self.attack_system_performance.get(at, {}).get(s, [])
                ],
                'rationale': f"Best system for {at}: System {best_sys} (mean_reward={best_mean:.3f})"
                             + (f", DQN nodes={target_nodes}" if target_nodes else "")
            }
            if target_node is not None:
                _dep['target_node'] = target_node    # primary node (backward compat)
            if target_nodes:
                _dep['target_nodes'] = target_nodes  # Phase 2: DQN-chosen network SUBSET
            final_deployments.append(_dep)
            
            print(f"     {at:30s} → System {best_sys} "
                  f"(reward={best_mean:.3f}, mag={agent_magnitude:.2f}, "
                  f"dur={agent_duration:.1f}s, stealth={agent_stealth:.2f}) "
                  f"[{agent_params_source}]")
        
        best_outer = max(self.outer_episode_results, key=lambda r: r.aggregate_reward)
        
        # ── Stage 2: Gemini timing refinement (if available) ──
        # Agents suggest magnitude/duration/stealth; Gemini suggests optimal
        # attack ORDERING for strategic coordination (e.g., spoofing before
        # data injection, simultaneous vs sequential).
        if self.gemini_available and self.llm_analyzer:
            final_deployments = self._gemini_refine_timing(final_deployments)
        
        # ── Stage 3: RAG-enhanced rationale via Gemini + ChromaDB ──
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
            print(f"     {dep['attack_type']:30s} → System {dep['target_system']} "
                  f"(mag={dep['magnitude']:.2f}, dur={dep['duration']:.1f}s, "
                  f"stealth={dep['stealth_level']:.2f}) [{src}]"
                  f"{' — ' + rationale if rationale else ''}")
        
        return plan
    
    # =========================================================================
    # GEMINI TIMING REFINEMENT
    # =========================================================================
    
    def _gemini_refine_timing(self, deployments: List[Dict]) -> List[Dict]:
        """Ask Gemini to suggest optimal attack ordering and timing coordination.
        
        HYBRID APPROACH: Gemini can override RL's target_system choice with justification.
        - RL provides data-driven defaults (best system per attack based on training)
        - Gemini can override for strategic reasons (e.g., diversionary attacks)
        - Gemini must justify any override (why strategic benefit > performance loss)
        
        Returns reordered (and optionally target/duration-adjusted) deployments list.
        """
        try:
            # Build performance data summary for Gemini
            attack_summary = []
            performance_data = {}
            
            for dep in deployments:
                at = dep['attack_type']
                rl_target = dep['target_system']
                
                # Gather performance data for this attack across all systems
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
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_text)
            if json_match:
                parsed = json.loads(json_match.group())
                ordered = parsed.get('ordered_attacks', [])
                
                if len(ordered) >= len(deployments):
                    # Rebuild deployments in Gemini's suggested order
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
                            
                            # Apply target_system override if Gemini explicitly requests it
                            if override_flag and gemini_target != rl_target:
                                dep['target_system'] = gemini_target
                                dep['rl_suggested_system'] = rl_target
                                dep['override_justification'] = override_justification
                                dep['params_source'] = 'trained_sac_agent+gemini_override'
                                overrides_applied += 1
                                print(f"     ## {at}: Gemini OVERRODE target System {rl_target} → {gemini_target}")
                                print(f"         Justification: {override_justification}")
                            else:
                                dep['params_source'] = 'trained_sac_agent+gemini_timing'
                            
                            # Use Gemini-adjusted duration if provided
                            if 'duration' in item and item['duration'] > 0:
                                dep['duration'] = float(np.clip(item['duration'], 5.0, 120.0))
                            
                            dep['timing_rationale'] = item.get('timing_rationale', '')
                            refined.append(dep)
                            del dep_by_type[at]
                    
                    # Append any attacks Gemini missed
                    for dep in dep_by_type.values():
                        refined.append(dep)
                    
                    print(f"  ## Gemini refined attack ordering ({len(refined)} attacks, {overrides_applied} target overrides):")
                    for i, dep in enumerate(refined):
                        rationale = dep.get('timing_rationale', '')
                        override_marker = '##' if 'rl_suggested_system' in dep else ''
                        print(f"     {i+1}. {override_marker}{dep['attack_type']:30s} → Sys {dep['target_system']} "
                              f"(dur={dep['duration']:.1f}s) — {rationale}")
                    return refined
            
            print(f"  ##Could not parse Gemini timing response, keeping agent order")
            return deployments
            
        except Exception as e:
            print(f"  ##Gemini timing refinement failed: {e}")
            return deployments
    
    # =========================================================================
    # RAG-ENHANCED FINAL ANALYSIS
    # =========================================================================
    
    # Pre-established STRIDE mapping from knowledgebase_mapping.md
    ATTACK_STRIDE_MAP = {
        'voltage_manipulation': {
            'stride': 'Information Disclosure',
            'protocol': 'DNP3',
            'link': 'CMS ↔ DG / DG ↔ DSM (Links 3-4)',
            'data_flows': 'DF-3, DF-4 (Load Measurement)',
        },
        'current_injection': {
            'stride': 'Elevation of Privilege',
            'protocol': 'OCPP/TCP',
            'link': 'EV ↔ EVCS ↔ CMS (Links 1-2)',
            'data_flows': 'DF-1 (Charging Info V,I,P), DF-2 (Authentication)',
        },
        'power_disruption': {
            'stride': 'Denial of Service',
            'protocol': 'TCP/IEC 61850',
            'link': 'EMS ↔ AGC / DG ↔ DSM (Links 4,6)',
            'data_flows': 'DF-4 (Load Measurement), DF-9 (Optimal Reference)',
        },
        'communication_spoofing': {
            'stride': 'Spoofing',
            'protocol': 'OCPP',
            'link': 'EV ↔ EVCS (Link 1)',
            'data_flows': 'DF-1 (Charging Info: SoC, power demand)',
        },
        'data_injection': {
            'stride': 'Tampering',
            'protocol': 'DNP3',
            'link': 'CMS ↔ DG / DG ↔ DSM (Links 3-4)',
            'data_flows': 'DF-3, DF-4 (Load Measurement)',
        },
        'protocol_manipulation': {
            'stride': 'Repudiation',
            'protocol': 'DNP3',
            'link': 'DSM ↔ EMS (Link 5)',
            'data_flows': 'DF-5 (Load Forecasting)',
        },
    }
    
    def _rag_enhance_deployment_plan(self, deployments: List[Dict],
                                     best_system_map: Dict) -> Optional[Dict]:
        """
        Query ARES RAG vector DB for each attack type, then ask Gemini to
        produce a research-quality analysis grounding RL results in real-world
        threat intelligence (CVEs, MITRE ATT&CK, mitigations).
        
        Returns dict with per-attack RAG context + Gemini analysis, or None
        if RAG infrastructure is unavailable.
        """
        # ── Try to import RAG infrastructure ──
        ares_rag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ares_rag')
        if ares_rag_dir not in sys.path:
            sys.path.insert(0, ares_rag_dir)
        
        try:
            from vector_db import ChromaDBManager, DocumentEmbedder
            db_manager = ChromaDBManager()
            embedder = DocumentEmbedder()
            doc_count = db_manager.collection.count()
            print(f"\n  🔍 RAG: Connected to ARES vector DB ({doc_count} documents)")
        except Exception as e:
            print(f"\n  ##RAG unavailable ({e}) — using pure RL deployment plan")
            return None
        
        # ── Query ChromaDB for each attack type ──
        rag_context_per_attack = {}
        
        for dep in deployments:
            at = dep['attack_type']
            stride_info = self.ATTACK_STRIDE_MAP.get(at, {})
            stride_cat = stride_info.get('stride', at)
            protocol = stride_info.get('protocol', '')
            
            # Build a targeted query for this attack type
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
                
                # Extract CVEs and MITRE techniques from retrieved docs
                cves = set()
                mitre_techs = set()
                for doc in docs:
                    meta = doc.get('metadata', {})
                    try:
                        techs = json.loads(meta.get('mitre_techniques', '[]'))
                        mitre_techs.update(techs)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    # Extract CVE IDs from content
                    content_cves = re.findall(r'CVE-\d{4}-\d{4,7}', 
                                              doc.get('content', '') + ' ' + str(meta))
                    cves.update(content_cves)
                
                rag_context_per_attack[at]['cves'] = sorted(list(cves))
                rag_context_per_attack[at]['mitre_techniques'] = sorted(list(mitre_techs))
                
                print(f"     {at:30s} → {len(docs)} docs, "
                      f"{len(cves)} CVEs, {len(mitre_techs)} MITRE techniques")
                
            except Exception as e:
                print(f"     {at:30s} → RAG query failed: {e}")
                rag_context_per_attack[at] = {'error': str(e)}
        
        # ── Build Gemini prompt combining RL results + RAG context ──
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
        """
        Send a single Gemini prompt combining RL training results with RAG
        context to produce a research-quality final deployment analysis.
        """
        if not self.llm_analyzer:
            print("  ##No Gemini analyzer — skipping RAG-enhanced analysis")
            return None
        
        # Build the RL results summary
        rl_summary_lines = []
        for dep in deployments:
            at = dep['attack_type']
            stride_info = self.ATTACK_STRIDE_MAP.get(at, {})
            trained_on = dep.get('systems_trained_on', [])
            
            # Per-system performance breakdown
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
        
        # Build RAG context summary
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
            print("\n  🤖 Sending RL results + RAG context to Gemini for final analysis...")
            
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
            
            # Try to parse JSON from Gemini response
            # Strip markdown code fences
            cleaned = re.sub(r'^```(?:json)?\s*', '', llm_text.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)
            
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    
                    # Enrich deployments with RAG-grounded rationale
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
                    
                    # Save the full RAG-enhanced plan to file
                    self._save_rag_deployment_plan(analysis, deployments)
                    
                    return analysis
                    
                except json.JSONDecodeError as e:
                    print(f"  ##Could not parse Gemini RAG analysis JSON: {e}")
                    # Save raw text anyway
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
        """Save the RAG-enhanced deployment plan to a timestamped file."""
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
    
    # =========================================================================
    # PER-SYSTEM REWARD EXTRACTION (for plotting)
    # =========================================================================
    
    def get_per_system_episode_rewards(self) -> Dict[int, Dict[str, list]]:
        """Extract per-system DQN and SAC episode rewards across ALL outer circles.
        
        The inner_episode_rewards structure is:
            {attack_type: {outer_ep: {dqn_rewards: [...], sac_rewards: [...], assigned_system: int}}}
        
        This method flattens it into:
            {system_id: {'dqn_rewards': [...all episodes...], 'sac_rewards': [...],
                         'outer_circle_boundaries': [idx1, idx2, ...]}}
        
        so it can be fed directly into EnhancedDQNSACCoordinator.training_history
        for the rl_training_rewards_6_systems.png plot.
        """
        per_system = {
            s: {'dqn_rewards': [], 'sac_rewards': [], 'outer_circle_boundaries_dqn': [], 'outer_circle_boundaries_sac': []}
            for s in range(1, self.num_systems + 1)
        }
        
        # Walk outer circles in order so episodes are chronological
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
                
                # Record boundary before appending (for vertical lines on plot)
                if dqn_r:
                    per_system[sys_id]['outer_circle_boundaries_dqn'].append(
                        len(per_system[sys_id]['dqn_rewards']))
                    per_system[sys_id]['dqn_rewards'].extend(dqn_r)
                
                if sac_r:
                    per_system[sys_id]['outer_circle_boundaries_sac'].append(
                        len(per_system[sys_id]['sac_rewards']))
                    per_system[sys_id]['sac_rewards'].extend(sac_r)
        
        # Log summary
        for s in range(1, self.num_systems + 1):
            nd = len(per_system[s]['dqn_rewards'])
            ns = len(per_system[s]['sac_rewards'])
            if nd or ns:
                print(f"  System {s}: {nd} DQN episodes, {ns} SAC episodes across {self.outer_episodes} outer circles")
        
        return per_system
    
    # =========================================================================
    # BUILD RESULTS & SAVE
    # =========================================================================
    
    def _build_training_results(self) -> Dict:
        """Build comprehensive training results dict."""
        
        # Flatten inner rewards for JSON serialization
        inner_rewards_flat = {}
        for at in self.attack_types:
            inner_rewards_flat[at] = {}
            for outer_ep, data in self.inner_episode_rewards[at].items():
                inner_rewards_flat[at][str(outer_ep)] = {
                    'dqn_rewards': data['dqn_rewards'],
                    'sac_rewards': data['sac_rewards'],
                    'assigned_system': data['assigned_system'],
                    # clean per-attack physical outcomes (guidance-free)
                    'best_impact': data.get('best_impact'),
                    'success_rate': data.get('success_rate'),
                    'avg_detection_risk': data.get('avg_detection_risk'),
                    # convergence / sample-efficiency instrumentation
                    'reduced_action': data.get('reduced_action'),
                    'sac_mean_reward': data.get('sac_mean_reward'),
                    'sac_num_episodes': data.get('sac_num_episodes'),
                    'cum_timesteps': data.get('cum_timesteps'),
                    'convergence_episode': data.get('convergence_episode'),
                }
        
        # Record the run seed so multi-seed aggregation can group/label files.
        try:
            from spear_seeding import get_run_seed
            _seed = get_run_seed()
        except Exception:
            _seed = None

        return {
            'mode': self.mode,
            'seed': _seed,
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
        """Compute whether outer rewards are improving, stable, or declining."""
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
        """Save the two-level reward history to JSON."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            try:
                from spear_seeding import get_run_seed
                _seed_tag = f"seed{get_run_seed()}_"
            except Exception:
                _seed_tag = ""
            # Optional run tag (e.g. prune mechanism) so parallel runs that share
            # mode+seed (annealed vs floor gemini) write distinct, race-free files.
            _run_tag = os.environ.get('SPEAR_REWARD_TAG', '').strip()
            _run_tag = f"{_run_tag}_" if _run_tag else ""
            filename = f"reward_history_{self.mode}_{_run_tag}{_seed_tag}{timestamp}.json"

            # Make JSON-serializable
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
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _to_dict(self, obj):
        """Convert dataclass or other object to dict."""
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
        """Recursively make object JSON-serializable."""
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
        """Print attack assignments for an outer episode."""
        print(f"\n  ## Attack Assignments for Outer Episode {outer_ep + 1}:")
        for a in assignments:
            print(f"     {a.attack_type:30s} → System {a.target_system}  "
                  f"(mag={a.magnitude:.1f}, stealth={a.stealth_level:.1f})")
            if a.rationale:
                print(f"       Rationale: {a.rationale}")
