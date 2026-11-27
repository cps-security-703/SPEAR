#!/usr/bin/env python3
"""
Attack-Specific RL Agent Architecture
Redesigned from system-specific to attack-specific agents for better specialization

Architecture:
- 6 attack types × 2 agents (DQN + SAC) = 12 total agents
- Each agent pair specializes in ONE attack type across ALL systems
- Gemini deploys attack-specialist agents to target systems
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback

# Attack type definitions (expandable)
ATTACK_TYPES = [
    'voltage_manipulation',
    'current_injection',
    'power_disruption',
    'communication_spoofing',
    'data_injection',
    'protocol_manipulation'
]

class EpisodeRewardCallback(BaseCallback):
    """
    SB3 callback to capture per-episode rewards during .learn() training.
    Stores episode rewards, lengths, and timestamps for convergence analysis.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Accumulate step reward from SB3 locals
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        
        if len(rewards) > 0:
            self.current_episode_reward += float(rewards[0])
            self.current_episode_length += 1
        
        # Check if episode ended
        if len(dones) > 0 and dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        return True
    
    def get_results(self) -> Dict:
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'num_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
        }


@dataclass
class AttackDeployment:
    """Deployment strategy from Gemini"""
    attack_type: str
    target_systems: List[int]
    magnitude: float
    duration: float
    stealth_level: float
    priority: int = 1
    # Reduced-action (Gemini) mode: Gemini also chooses the ACN network (node)
    # within the target DS, ceded from DQN.  None → autonomous / DQN picks it.
    target_node: Optional[int] = None


class AttackSpecificEnvironment(gym.Env):
    """
    Environment for ONE attack type across ALL systems
    
    Key differences from old architecture:
    - Attack type is FIXED (determined by which agent is selected)
    - Action space is system selection (DQN) or parameters only (SAC)
    - Agent specializes in mastering ONE attack across different systems
    """
    
    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6,
                 node_level: bool = False, network_layout=None,
                 n_nodes: int = 10, node_seed: int = 42, reduced_action: bool = False):
        super(AttackSpecificEnvironment, self).__init__()
        # Reduced-action (Gemini-guided) mode: SAC learns MAGNITUDE ONLY; the
        # ceded tactical dims (duration, stealth) are supplied by Gemini via
        # guidance_hints in step().  This shrinks the continuous search 3-D→1-D
        # so the agent converges faster; because Gemini fills the ceded dims,
        # the reachable impact×stealth is unchanged (equal-performance / less-
        # time hypothesis).  Full mode (autonomous) keeps the 4-D action space.
        self.reduced_action = bool(reduced_action)

        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type  # FIXED attack type for this environment
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000

        # ── Phase 2: node-level (ACN-network) targeting ──────────────────────
        # When node_level is True, each DS contains `n_nodes` ACN networks
        # (Caltech ~54 / JPL ~52 EVSEs, seeded layout from acn_network_layout).
        # The attack targets ONE network's CMS input (all its EVSEs impacted).
        # DEFAULT (False) preserves the exact Phase-1 system-level behaviour —
        # every node_level branch below is additive.
        self.node_level = bool(node_level)
        self.n_nodes = int(n_nodes)
        if self.node_level and network_layout is None:
            try:
                from acn_network_layout import build_layout
                network_layout = build_layout(num_systems=num_systems,
                                              n_nodes=n_nodes, seed=node_seed)
            except Exception as _e:
                print(f"      ##?? node_level requested but layout unavailable: {_e}")
                self.node_level = False
                network_layout = None
        self.network_layout = network_layout
        # Which network within the (forced) DS this attack targets.  Mirrors
        # forced_target_system: None => agent/last-used; int => that node index.
        self.forced_target_node = None
        self._last_target_node = 0
        
        # Observation space: system state + last action feedback + cross-circle memory
        # Layout: [system_1_state(25), ..., system_6_state(25),   <- 150 features
        #          last_action_feedback(5),                        <- 5 features
        #          cross_circle_memory(3)]                         <- 3 features
        # last_action_feedback: [magnitude, duration, stealth, ids_detected, lstm_score]
        # cross_circle_memory:  [cc_detection_rate, cc_success_rate, cc_avg_impact]
        #   for the CURRENTLY TARGETED system, aggregated across all past circles.
        # This gives the agent persistent memory: "across all circles, system X
        # had 80% detection rate" — so it knows to try different parameters.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            # 25 per system + 5 action feedback + 3 cross-circle + 3 guidance hint (Fix 1)
            shape=(num_systems * 25 + 5 + 3 + 3,),
            dtype=np.float32
        )

        # Action space for SAC: [magnitude, duration, stealth, target_system_id]
        # Note: attack_type is FIXED, not part of action space
        if self.reduced_action:
            # Gemini mode: SAC learns [magnitude, stealth]; DURATION is ceded to
            # Gemini (injected from guidance_hints in step()).  The target system
            # is set by the coordinator (bandit/Gemini) and the node by DQN, so
            # neither needs a SAC action dim.
            self.action_space = spaces.Box(
                low=np.array([0.1, 0.0], dtype=np.float32),
                high=np.array([2.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Autonomous mode: SAC learns [magnitude, duration, stealth].
            # target_system is removed (DS comes from the bandit, node from DQN).
            self.action_space = spaces.Box(
                low=np.array([0.1, 5.0, 0.0], dtype=np.float32),
                high=np.array([2.0, 60.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        # Performance tracking
        self.episode_rewards = []
        self.attack_history = []
        
        # Forced target system: when set (1-indexed), overrides the agent's
        # target_system action. Used by CentralRLCoordinator rotation to
        # ensure the agent trains on a specific system during each circle.
        self.forced_target_system = None
        
        # Guidance hints from Gemini or autonomous coordinator.
        # When set, the reward function adds a bonus for staying close to
        # these suggested parameters, biasing exploration without hard-
        # constraining the agent.  This is how Gemini's strategic suggestions
        # (e.g., lower magnitude for stealth, longer duration for persistence)
        # actually influence the SAC agent's learned policy.
        # Structure: {'magnitude': float, 'duration': float, 'stealth_level': float}
        self.guidance_hints = None

        # ── LLM search-space pruning (KEYSTONE; reversible, off by default) ──────
        # Unlike guidance_hints (a soft reward bonus that leaves SAC searching the
        # full action space), action_prune implements the paper's actual claimed
        # mechanism: the LLM's deployment plan NARROWS the reachable action box.
        # When set to {'magnitude':(lo,hi), 'stealth_level':(lo,hi), 'duration':(lo,hi)},
        # each raw action value is affine-remapped from its full range into the
        # LLM-curated [lo,hi] band in step(), so SAC explores ONLY that region.
        # None => full action space (identical to prior behaviour → autonomous
        # baseline is unchanged).  Only named dims are pruned.
        self.action_prune = None
        self._full_action_range = {
            'magnitude':     (0.1, 2.0),
            'stealth_level': (0.0, 1.0),
            'duration':      (5.0, 60.0),
        }

        # Last action parameters (set by step(), used by guidance alignment bonus)
        self._last_action_params = None
        
        # Last action feedback for observation: [magnitude, duration, stealth,
        # ids_detected, lstm_score].  Initialized to neutral values so the
        # agent sees a valid observation before its first step.
        self._last_action_feedback = np.array(
            [0.7, 30.0, 0.7, 0.0, 0.0], dtype=np.float32
        )
        
        # Cross-circle memory: persists across reset() calls.
        # Populated by the coordinator after each circle via update_cross_circle_stats().
        # Structure: {sys_id: {'detection_rate': float, 'success_rate': float,
        #                      'avg_impact': float, 'num_circles': int}}
        # Neutral defaults (0.5 detection, 0.5 success, 0.0 impact) until first circle.
        self.cross_circle_stats = {
            s: {'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0}
            for s in range(1, num_systems + 1)
        }
        
        # Cached PINN state per system — populated by _execute_attack_on_pinn()
        # and used by _get_global_observation() to provide real (not random) obs.
        # Structure: {sys_id: {'voltage': V, 'current': I, 'power': P,
        #                      'station_data': {...}, 'attacked_response': {...}}}
        self._last_pinn_state = {}

        # Consecutive undetected-success streak.
        # Incremented each time the agent succeeds AND evades all IDS layers.
        # Reset to 0 on any detection.  Drives the streak/mission bonuses in
        # _calculate_reward() — the primary mechanism from the >97% evasion design.
        self.consecutive_evasions = 0

        # Fix 4 — per-attack running reward scale (RMS via EMA).  This env instance
        # handles ONE fixed attack type, so this normalizes reward PER ATTACK: the
        # ~1e5-scale evadable attacks (protocol/power/voltage/data) and the ~0-scale
        # stuck attacks (current/comms) both land near unit scale.  That keeps SAC/DQN
        # value & gradient magnitudes comparable across attacks (no single attack's
        # 1e5 reward dominates) and stops the guidance signal being swamped.  Persists
        # across reset()/circles (defined here, never cleared in reset()).
        self._rew_sq_ema = 1.0        # running E[reward^2]
        self._rew_norm_beta = 0.999   # EMA decay

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.attack_history = []
        self.consecutive_evasions = 0   # reset streak each episode

        # Seed PINN state cache with baseline readings for every system
        # so _get_global_observation() returns real values from step 0.
        # Reset anomaly detector state (sequence buffers, load history) so
        # the IDS starts fresh each episode — matches deployment behavior.
        # Then warm up the LSTM sequence buffer with benign baseline samples
        # so the LSTM can produce scores from step 1 (it needs sequence_length
        # samples before it can run inference).
        self._last_pinn_state = {}
        # Reset within-episode action feedback to neutral
        # NOTE: cross_circle_stats is intentionally NOT reset here —
        # it must survive across episodes and circles.
        self._last_action_feedback = np.array(
            [0.7, 30.0, 0.7, 0.0, 0.0], dtype=np.float32
        )
        for sys_id in range(1, self.num_systems + 1):
            det = self.federated_pinn_manager.anomaly_detectors.get(sys_id)
            if det:
                det.reset_state()
            
            if sys_id in self.federated_pinn_manager.local_models:
                try:
                    sd = self._build_baseline_station_data(sys_id)
                    cms = self.federated_pinn_manager.local_models[sys_id]
                    raw = cms.optimize_references(sd)
                    if isinstance(raw, tuple):
                        resp = {'voltage': raw[0], 'current': raw[1], 'power': raw[2]}
                    elif isinstance(raw, dict):
                        resp = raw
                    else:
                        resp = {'voltage': float(raw), 'current': 0.0, 'power': 0.0}
                    self._last_pinn_state[sys_id] = {
                        'voltage': float(resp.get('voltage', 400.0)),
                        'current': float(resp.get('current', 50.0)),
                        'power': float(resp.get('power', 20.0)),
                        'station_data': sd
                    }
                except Exception:
                    self._last_pinn_state[sys_id] = {
                        'voltage': 400.0, 'current': 50.0, 'power': 20.0,
                        'station_data': self._build_baseline_station_data(sys_id)
                    }
                
                # Warm up LSTM sequence buffer with benign samples so the
                # detector can score from step 1.  In deployment the IDS has
                # been monitoring normal traffic before any attack starts.
                if det and det.lstm_enabled:
                    seq_len = getattr(det, 'sequence_length', 10)
                    for _ in range(seq_len):
                        benign_sd = self._build_baseline_station_data(sys_id)
                        ps = self._last_pinn_state.get(sys_id, {})
                        benign_ids = {
                            'soc': float(np.clip(benign_sd.get('soc', 0.5), 0, 1)),
                            'voltage': float(ps.get('voltage', 400.0)),
                            'current': float(ps.get('current', 50.0)),
                            'power': float(ps.get('power', 20.0)),
                            'temperature': float(benign_sd.get('temperature', 25.0)),
                            'demand_factor': float(benign_sd.get('demand_factor', 0.5)),
                            'load_factor': float(benign_sd.get('load_factor', 0.5)),
                            'grid_voltage': float(benign_sd.get('grid_voltage', 1.0)),
                            'grid_frequency': float(benign_sd.get('grid_frequency', 60.0)),
                            'queue_length': int(benign_sd.get('queue_length', 3)),
                            'utilization': float(benign_sd.get('utilization', 0.5)),
                            'urgency_factor': float(benign_sd.get('urgency_factor', 1.0)),
                            'time_of_day': float(benign_sd.get('time_of_day', 12.0)),
                            'system_id': sys_id
                        }
                        det.detect_lstm_anomaly(benign_ids)
        
        # Get observations from all systems
        observation = self._get_global_observation()
        
        return observation, {}
    
    def _remap_pruned(self, dim: str, val: float) -> float:
        """Affine-remap a raw action value from its full action range into the
        LLM's pruned [lo,hi] band for this dim (true search-space reduction:
        the agent keeps full resolution inside the narrowed region).  Returns
        val unchanged when dim is not in action_prune."""
        band = (self.action_prune or {}).get(dim)
        if not band:
            return val
        lo, hi = float(band[0]), float(band[1])
        flo, fhi = self._full_action_range.get(dim, (0.0, 1.0))
        if fhi <= flo:
            return val
        frac = (float(val) - flo) / (fhi - flo)
        frac = min(1.0, max(0.0, frac))          # guard actions outside the box
        return lo + frac * (hi - lo)

    def step(self, action: np.ndarray):
        """
        Execute attack with this agent's specialized attack type
        
        Args:
            action: [magnitude, duration, stealth, target_system_id]
        """
        self.current_step += 1

        # Parse action.  Reduced (Gemini) mode: action = [magnitude, stealth];
        # DURATION is pulled from Gemini's guidance_hints so the executed attack
        # is still fully specified.  Full (autonomous) mode: [magnitude, duration,
        # stealth].  target_system is no longer an action dim in either mode.
        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        if self.reduced_action:
            magnitude = float(action[0])
            stealth_level = float(action[1])
            h = self.guidance_hints or {}
            duration = float(h.get('duration', 30.0))
        else:
            magnitude = float(action[0])
            duration = float(action[1])
            stealth_level = float(action[2])

        # ── LLM search-space pruning (keystone) ─────────────────────────────────
        # Confine the EXECUTED attack — and therefore the region SAC actually
        # explores — to the LLM's physically-reasoned sub-band.  Off (None) leaves
        # the full-space behaviour untouched (autonomous baseline).
        if self.action_prune:
            magnitude     = self._remap_pruned('magnitude', magnitude)
            stealth_level = self._remap_pruned('stealth_level', stealth_level)
            if not self.reduced_action:
                duration  = self._remap_pruned('duration', duration)

        # Use forced target if set by coordinator rotation, else agent's choice.
        # forced_target_system can be:
        #   - None  → agent's action[3] picks the system (default)
        #   - int   → all steps target this system (legacy single-system focus)
        #   - list  → uniformly random pick per step.  This is the generalisation
        #            mode: the policy sees diverse (attack, system) combinations
        #            so it transfers to any deployment-time assignment, not just
        #            the LLM's per-circle hint pair.
        if self.forced_target_system is not None:
            if isinstance(self.forced_target_system, (list, tuple, np.ndarray)):
                target_system_id = int(np.random.choice(self.forced_target_system))
            else:
                target_system_id = int(self.forced_target_system)  # already 1-indexed
        else:
            # target_system is no longer a SAC action dim (DS comes from the
            # bandit/Gemini via forced_target_system).  Without a forced target
            # fall back to a random system.
            target_system_id = int(np.random.randint(1, self.num_systems + 1))

        # Phase 2: resolve the target NODE (ACN network within the DS).
        #   forced_target_node set (by the DQN env) → that node (deterministic).
        #   unforced (SAC training) → random node each step so SAC params
        #   generalise across the DS's Caltech/JPL networks.
        # No-op when node_level is False (target_node stays None).
        target_node = None
        if self.node_level:
            if self.forced_target_node is not None:
                target_node = int(self.forced_target_node) % self.n_nodes
            else:
                target_node = int(np.random.randint(0, self.n_nodes))
            self._last_target_node = target_node

        # Execute attack with FIXED attack type
        attack_params = {
            'attack_type': self.attack_type,  # FIXED
            'target_system': target_system_id,
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }
        if target_node is not None:
            attack_params['target_node'] = target_node
        
        # Store raw action params for guidance alignment bonus in _calculate_reward
        self._last_action_params = {
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }
        
        # Execute on PINN CMS
        attack_result = self._execute_attack_on_pinn(attack_params)
        
        # Update last action feedback so the NEXT observation includes what
        # happened: magnitude/duration/stealth chosen + whether LSTM detected.
        # This is the key memory signal: agent sees "I chose X → detected Y"
        # and can learn to adjust parameters to evade detection.
        self._last_action_feedback = np.array([
            magnitude / 2.0,                                        # normalize [0.1,2.0]→[0.05,1.0]
            duration / 60.0,                                        # normalize [5,60]→[0.08,1.0]
            stealth_level,                                          # already [0,1]
            float(attack_result.get('ids_detected', False)),        # 0 or 1
            float(attack_result.get('ids_lstm_score', 0.0)),        # [0,1]
        ], dtype=np.float32)
        
        # Calculate reward
        reward = self._calculate_reward(attack_result, stealth_level)

        # Fix 4 — per-attack reward normalization (running RMS). Keeps reward ~O(1)
        # across attacks whose raw scale spans ~1e5 (evadable) to ~0 (stuck), so
        # value/gradient magnitudes are comparable and no attack dominates. Sign is
        # preserved (scale-only, like VecNormalize norm_reward).
        reward = self._normalize_reward(reward)

        # Get next observation
        observation = self._get_global_observation()
        
        # Check if done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Store history
        self.attack_history.append({
            'step': self.current_step,
            'attack_type': self.attack_type,
            'target_system': target_system_id,
            'result': attack_result,
            'reward': reward
        })
        
        info = {
            'attack_result': attack_result,
            'attack_type': self.attack_type,
            'target_system': target_system_id
        }
        
        return observation, reward, done, truncated, info
    
    def _build_baseline_station_data(self, sys_id: int) -> Dict:
        """Build station_data dict matching the CMS construction in hierarchical_cosimulation.py.
        
        This ensures the RL training environment sees the same input format
        that the CMS uses during the hierarchical co-simulation.
        """
        baseline_soc = np.random.uniform(0.3, 0.8)
        demand_factor = 0.5 + 0.2 * np.sin(self.current_step * 0.01)  # mild daily profile proxy
        urgency_factor = 2.0 if baseline_soc < 0.2 else (0.3 if baseline_soc > 0.8 else 1.0)
        voltage_priority = max(0, 0.95 - 1.0)  # 0 when grid_voltage is nominal
        nominal_voltage = 400.0
        nominal_current = max(1.0, baseline_soc * 100.0 + 20.0)
        nominal_power = nominal_voltage * nominal_current / 1000.0
        temperature = 25.0 + 5.0 * demand_factor
        queue_len = max(0, int(demand_factor * 8))
        time_of_day = (self.current_step * 0.1) % 24.0  # proxy for hour-of-day

        # ── Phase 2: shape the aggregate CMS input by the TARGET network ──────
        # Caltech (bigger/busier) vs JPL yields a different baseline, so the
        # DQN's node choice produces a real, differentiated attack signal.  No
        # effect when node_level is False.
        _node_util = demand_factor
        node_type = "n/a"
        if self.node_level and self.network_layout is not None:
            try:
                from acn_network_layout import sample_network_profile
                nodes = self.network_layout.get(sys_id, [])
                if nodes:
                    node = nodes[int(self._last_target_node) % len(nodes)]
                    prof = sample_network_profile(node, step=self.current_step)
                    demand_factor = prof['demand_factor']
                    # Scale current/power by active-EVSE load (normalised to the
                    # canonical caltech size) so a larger network is a physically
                    # larger target.
                    load_scale = prof['active_evse'] / 54.0
                    nominal_current = max(1.0, nominal_current * (0.5 + load_scale))
                    nominal_power = nominal_voltage * nominal_current / 1000.0
                    queue_len = max(0, int(prof['active_evse'] * 0.15))
                    temperature = 25.0 + 5.0 * demand_factor
                    _node_util = prof['utilization']
                    node_type = node.network_type
            except Exception:
                pass

        return {
            'soc': baseline_soc,
            'grid_voltage': 1.0,
            'grid_frequency': 60.0,
            'demand_factor': demand_factor,
            'voltage_priority': voltage_priority,
            'urgency_factor': urgency_factor,
            'current_time': float(self.current_step),
            'bus_distance': 1.0,
            'load_factor': demand_factor,
            'ac_power_in': 50.0,
            'system_efficiency': 0.95,
            'power_balance_error': 0.0,
            'dc_link_voltage': 500.0,
            # Additional features for LSTM anomaly detector (same as CMS)
            'voltage': nominal_voltage,
            'current': nominal_current,
            'power': nominal_power,
            'temperature': temperature,
            'queue_length': queue_len,
            'utilization': _node_util,
            'time_of_day': time_of_day,
            'system_id': sys_id,
            # Phase 2 node context (harmless extra keys when node_level=False)
            'target_node': int(self._last_target_node) if self.node_level else -1,
            'network_type': node_type,
        }
    
    def _get_global_observation(self) -> np.ndarray:
        """Get observation across all systems using real PINN-derived state.
        
        Features are derived from the cached PINN output (updated each step)
        and the station_data used to produce it — NOT random noise.
        """
        observations = []
        
        for sys_id in range(1, self.num_systems + 1):
            if sys_id in self.federated_pinn_manager.local_models:
                # Retrieve cached PINN state (populated by reset / _execute_attack_on_pinn)
                ps = self._last_pinn_state.get(sys_id, {})
                sd = ps.get('station_data', self._build_baseline_station_data(sys_id))
                
                # PINN output values (real, from last optimize call)
                pinn_v = ps.get('voltage', 400.0)
                pinn_i = ps.get('current', 50.0)
                pinn_p = ps.get('power', 20.0)
                
                # Normalize PINN outputs to reasonable observation ranges
                voltage_pu = pinn_v / 400.0          # ~1.0 at nominal
                current_norm = pinn_i / 125.0         # ~0.4 at 50A
                power_norm = pinn_p / 75.0            # ~0.27 at 20kW
                
                system_state = np.array([
                    # Power system metrics (5) — from PINN output + station_data
                    voltage_pu,                                     # voltage_pu
                    current_norm,                                   # current_normalized
                    power_norm,                                     # power_normalized
                    sd.get('grid_frequency', 60.0) / 60.0,         # frequency_normalized
                    sd.get('system_efficiency', 0.95),              # power_factor / efficiency
                    
                    # EVCS metrics (5) — from station_data
                    sd.get('soc', 0.5),                             # soc
                    sd.get('demand_factor', 0.5),                   # demand_factor (proxy for vehicles)
                    power_norm,                                     # charging_power (normalized)
                    sd.get('temperature', 25.0) / 50.0,            # temperature_normalized
                    sd.get('utilization', 0.5),                     # utilization
                    
                    # Security metrics (5) — derived from attack history
                    self._get_avg_impact(sys_id),                   # anomaly_score proxy
                    self._get_detection_rate(sys_id),               # detection_risk
                    min(len([a for a in self.attack_history
                             if a['target_system'] == sys_id]) / 5.0, 1.0),  # recent_attacks_norm
                    1.0 - self._get_system_adaptation_level(sys_id),  # defense_level
                    1.0 - self._get_attack_success_rate(sys_id),      # vulnerability_score
                    
                    # Attack history for this attack type (5)
                    self._get_attack_success_rate(sys_id),
                    self._get_avg_impact(sys_id),
                    self._get_detection_rate(sys_id),
                    self._get_last_attack_time(sys_id),
                    self._get_system_adaptation_level(sys_id),
                    
                    # Network topology (5) — deterministic, system-specific
                    float(sys_id) / self.num_systems,               # system_id_normalized
                    sd.get('demand_factor', 0.5),                   # connectivity proxy
                    sd.get('load_factor', 0.5),                     # centrality proxy
                    sd.get('urgency_factor', 1.0) / 2.0,           # load_balance proxy
                    sd.get('grid_voltage', 1.0),                    # redundancy proxy
                ], dtype=np.float32)
            else:
                system_state = np.zeros(25, dtype=np.float32)
            
            observations.append(system_state)
        
        # Append last action feedback (5 features).
        action_feedback = getattr(self, '_last_action_feedback',
                                  np.zeros(5, dtype=np.float32))
        
        # Append cross-circle memory for the currently targeted system (3 features).
        # If forced_target_system is set use that; otherwise use the last targeted system
        # from attack_history; fall back to neutral if no history yet.
        # NOTE: forced_target_system can be a list (rotation mode from Fix A) —
        # in that case fall back to the most-recent attack's system, since a
        # list isn't hashable and the cross-circle memory is per-system anyway.
        # The previous code passed the raw list to cross_circle_stats.get(),
        # which raised "unhashable type: 'list'" inside DQN training and got
        # silently swallowed by the DQN try/except, leaving dqn_rewards = [].
        _fts = self.forced_target_system
        if _fts is not None and not isinstance(_fts, (list, tuple, np.ndarray)):
            cc_sys = int(_fts)
        elif self.attack_history:
            cc_sys = int(self.attack_history[-1]['target_system'])
        else:
            cc_sys = 1  # default
        cc = self.cross_circle_stats.get(cc_sys,
             {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0})
        cross_circle_memory = np.array([
            float(cc['detection_rate']),   # 0=never detected, 1=always detected
            float(cc['success_rate']),     # 0=never succeeded, 1=always succeeded
            float(cc['avg_impact']),       # 0=no impact, 1=max impact
        ], dtype=np.float32)

        # Fix 1 — guidance hint IN THE OBSERVATION so the policy can CONDITION on it
        # (goal-conditioned RL).  Previously the hint reached the agent only as a tiny
        # reward bonus it could not see; a per-circle-varying hint was therefore
        # non-stationary reward noise the agent could not attribute (and could not
        # follow).  As an observable input, the hint becomes something the policy can
        # act on: it can learn "given this suggested magnitude/stealth, here is the
        # best action," and a changing hint is a conditioning variable, not noise.
        # Defaults to the autonomous baseline (0.7/30/0.7) when no hint is set, so the
        # observation is always well-defined and gemini/autonomous share the same shape.
        h = self.guidance_hints or {}
        hint_obs = np.array([
            (float(h.get('magnitude', 0.7)) - 0.1) / 1.9,     # [0.1,2.0] → [0,1]
            (float(h.get('duration', 30.0)) - 5.0) / 55.0,    # [5,60]   → [0,1]
            float(h.get('stealth_level', 0.7)),               # already [0,1]
        ], dtype=np.float32)

        return np.concatenate(
            observations + [action_feedback, cross_circle_memory, hint_obs]
        )
    
    def _execute_attack_on_pinn(self, attack_params: Dict) -> Dict:
        """Execute attack through the FULL federated PINN pipeline.
        
        Uses optimize_with_constraints() — the same pipeline the CMS uses
        during hierarchical co-simulation — so the RL agent trains against
        the same anomaly detection, input sanitization, and output clamping
        it will face during deployment.
        
        Fallback: if optimize_with_constraints rejects the input (e.g.
        physical constraint violation), falls back to raw optimize_references()
        to preserve gradient signal for the RL agent.
        """
        target_system = attack_params['target_system']
        
        if target_system not in self.federated_pinn_manager.local_models:
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': 'System not available'
            }
        
        magnitude = attack_params['magnitude']
        attack_type = attack_params['attack_type']
        stealth_level = float(attack_params.get('stealth_level', 0.5))

        # Stealth-scaled effective magnitude: high stealth → smaller physical
        # footprint → lower IDS detection probability.  This gives the agent a
        # REAL lever: choosing high stealth actually reduces the perturbation
        # amplitude, not just the reward multiplier.
        #   stealth=1.0 → effective = magnitude × 0.25  (very subtle)
        #   stealth=0.5 → effective = magnitude × 0.625
        #   stealth=0.0 → effective = magnitude × 1.0   (full amplitude)
        effective_magnitude = magnitude * max(0.05, 1.0 - 0.75 * stealth_level)

        try:
            # Baseline station_data — same format as CMS in hierarchical_cosimulation.py
            baseline_station_data = self._build_baseline_station_data(target_system)
            baseline_soc = baseline_station_data['soc']

            # --- Get baseline PINN response via full pipeline ---
            baseline_response = self._optimize_via_pipeline(
                target_system, baseline_station_data
            )

            # Update PINN state cache for baseline (used by _get_global_observation)
            self._last_pinn_state[target_system] = {
                'voltage': baseline_response.get('voltage', 400.0),
                'current': baseline_response.get('current', 50.0),
                'power': baseline_response.get('power', 20.0),
                'station_data': baseline_station_data
            }

            # ── Attack injection — VERBATIM port of hierarchical_cosimulation
            # `_apply_input_attacks` (lines 1516-1599).  Training MUST perturb
            # the same dict keys with the same coefficients as deployment so
            # that the trained policy transfers; otherwise the IDS sees
            # different feature distributions in the two phases.
            #
            # Differences from deployment (intentional):
            #   • progress = 1.0 (deployment ramps from 0→1 over duration; here
            #     each RL step is a single full-attack snapshot).
            #   • effective_magnitude (stealth-scaled) replaces raw magnitude
            #     so the agent's stealth knob actually shrinks the perturbation
            #     amplitude.  Deployment will need a matching change later
            #     (Fix #6) to honour the trained policy's stealth choice.
            attacked_station_data = dict(baseline_station_data)  # copy
            gradual_magnitude = effective_magnitude   # progress=1.0 in training

            if attack_type == 'voltage_manipulation':
                voltage_drop_factor = 1.0 - (gradual_magnitude * 0.35)
                attacked_station_data['grid_voltage'] = (
                    baseline_station_data.get('grid_voltage', 1.0) * voltage_drop_factor)
                attacked_station_data['voltage_priority'] = max(
                    0.0, 0.95 - attacked_station_data['grid_voltage'])
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 20.0

            elif attack_type == 'current_injection':
                cumulative_factor = 1.0 + (gradual_magnitude * 45.0)
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7) * cumulative_factor)
                attacked_station_data['urgency_factor'] = (
                    baseline_station_data.get('urgency_factor', 1.0)
                    * (1.0 + gradual_magnitude * 20.0))
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 35.0

            elif attack_type == 'power_disruption':
                cumulative_factor = max(0.02, 1.0 - (gradual_magnitude * 0.90))
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7) * cumulative_factor)
                attacked_station_data['urgency_factor'] = (
                    baseline_station_data.get('urgency_factor', 1.0) * cumulative_factor)
                attacked_station_data['power_multiplier'] = cumulative_factor

            elif attack_type == 'communication_spoofing':
                soc_reduction = gradual_magnitude * 0.7
                attacked_station_data['soc'] = max(0.01, baseline_soc - soc_reduction)
                attacked_station_data['urgency_factor'] = 1.0 + (gradual_magnitude * 40.0)
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 30.0

            elif attack_type == 'data_injection':
                frequency_deviation = gradual_magnitude * 12.0
                attacked_station_data['grid_frequency'] = (
                    baseline_station_data.get('grid_frequency', 60.0) + frequency_deviation)
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7)
                    * (1.0 + gradual_magnitude * 30.0))
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 25.0

            elif attack_type == 'protocol_manipulation':
                # progress=1.0 → time_factor = duration/4.  Use a fixed phase so
                # each training step is deterministic given the same magnitude.
                duration_s = float(attack_params.get('duration', 30.0))
                time_factor = duration_s / 4.0
                oscillation = float(np.sin(time_factor)) * 20.0 + 1.0
                amplitude_growth = 1.0 + (magnitude * 12.0)   # progress=1.0
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7)
                    * oscillation * amplitude_growth)
                attacked_station_data['grid_voltage'] = (
                    baseline_station_data.get('grid_voltage', 1.0)
                    * (1.0 - gradual_magnitude * 0.2))
                attacked_station_data['power_multiplier'] = oscillation * amplitude_growth

            else:
                # Unknown attack type → default current_injection behaviour
                cumulative_factor = 1.0 + (gradual_magnitude * 30.0)
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7) * cumulative_factor)
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 25.0

            # Defense-in-depth: sanitize NaN/Inf from attack formulas (matches
            # _apply_input_attacks lines 1586-1597).
            _defaults = {
                'grid_voltage': 1.0, 'grid_frequency': 60.0, 'demand_factor': 0.7,
                'urgency_factor': 1.0, 'soc': float(baseline_soc),
                'voltage_priority': 0.0, 'power_multiplier': 1.0, 'load_factor': 0.7,
            }
            for _k, _dv in _defaults.items():
                if _k in attacked_station_data and not np.isfinite(attacked_station_data[_k]):
                    attacked_station_data[_k] = _dv
            
            # --- Get attacked PINN response via full pipeline ---
            attacked_response = self._optimize_via_pipeline(
                target_system, attacked_station_data
            )
            
            # Update PINN state cache with attacked state (reflects current system condition)
            self._last_pinn_state[target_system] = {
                'voltage': attacked_response.get('voltage', 400.0),
                'current': attacked_response.get('current', 50.0),
                'power': attacked_response.get('power', 20.0),
                'station_data': attacked_station_data
            }
            
            # --- Impact calculation ---
            # Component 1: PINN output sensitivity (how much the PINN's optimization changed)
            voltage_impact = abs(attacked_response['voltage'] - baseline_response['voltage']) / (abs(baseline_response['voltage']) + 1e-6)
            current_impact = abs(attacked_response['current'] - baseline_response['current']) / (abs(baseline_response['current']) + 1e-6)
            power_impact = abs(attacked_response['power'] - baseline_response['power']) / (abs(baseline_response['power']) + 1e-6)
            pinn_sensitivity = max(voltage_impact, current_impact, power_impact)
            
            # Component 2: Physical attack severity (how much the input was perturbed)
            # Measures the normalized deviation of attacked inputs from baseline.
            # Key list aligned with the new deployment-style attack injection
            # above — every key that any attack type can mutate is included so
            # physical_severity actually reflects the perturbation magnitude.
            physical_severity = 0.0
            _SEVERITY_KEYS = ['grid_voltage', 'grid_frequency', 'demand_factor',
                              'urgency_factor', 'power_multiplier', 'soc',
                              'voltage_priority']
            for key in _SEVERITY_KEYS:
                b_val = baseline_station_data.get(key, 0.0)
                a_val = attacked_station_data.get(key, b_val)
                if abs(b_val) > 1e-6:
                    physical_severity += abs(a_val - b_val) / abs(b_val)
                else:
                    # Baseline absent (e.g., voltage_priority defaults to 0) —
                    # use absolute deviation so we still capture the attack signal.
                    physical_severity += abs(a_val - b_val)
            physical_severity /= float(len(_SEVERITY_KEYS))
            
            # Combined impact: physical severity weighted by PINN confirmation
            # Even tiny PINN sensitivity confirms the attack is reaching the system
            pinn_weight = min(1.0, pinn_sensitivity * 100.0)  # Amplify small PINN changes
            real_impact = physical_severity * (0.6 + 0.4 * pinn_weight)
            
            # Determine whether the attack had measurable physical impact.
            # Threshold lowered from 0.05 → 0.01 so that stealth-scaled perturbations
            # (effective_magnitude ≈ 0.03–0.07 at high stealth) still register as a
            # real impact.  At 0.05 the threshold excluded ALL stealthy attacks, giving
            # the agent no positive reward signal for learning the evadeable region.
            had_impact = real_impact > 0.01
            
            # --- Real IDS detection risk ---
            # Run the attacked traffic through the ACTUAL multi-layer IDS
            # (physical constraints + pattern detection + LSTM) so the RL
            # agent learns to evade the real detector, not a heuristic proxy.
            detection_risk = 0.0
            ids_detected = False
            ids_layer = None
            ids_lstm_score = 0.0
            ids_window = None   # exact 10x14 window scored by the best IDS (for replay/CSV capture)
            
            anomaly_detector = self.federated_pinn_manager.anomaly_detectors.get(target_system)
            if anomaly_detector:
                _pinn_v = float(attacked_response.get('voltage', 400.0))
                _pinn_i = float(attacked_response.get('current', 50.0))
                _pinn_p = float(attacked_response.get('power', 20.0))
                _soc = float(attacked_station_data.get('soc', 0.5))
                _temperature = 25.0 + max(0, _pinn_p - 20.0) * 0.3
                _load_factor = float(np.clip(_pinn_p / 50.0, 0.2, 1.3))
                _grid_voltage = float(attacked_station_data.get('grid_voltage', 1.0))
                _grid_frequency = float(attacked_station_data.get('grid_frequency', 60.0))
                _demand_factor = float(attacked_station_data.get('demand_factor', 0.5))
                _utilization = float(np.clip(_pinn_p / 75.0, 0.1, 1.0))
                _urgency = float(attacked_station_data.get('urgency_factor', 1.0))
                _time_of_day = float(attacked_station_data.get('time_of_day',
                                     (self.current_step * 0.1) % 24.0))
                _queue_length = int(attacked_station_data.get('queue_length', 3))
                
                # Pass RAW (unclipped) grid_voltage and grid_frequency so Layer 1 can
                # actually catch large perturbations.  The previous pre-clipping to
                # [0.85, 1.15] and [59, 61] made Layer 1 permanently blind: values
                # were already sanitised before the constraint check ran.
                ids_input = {
                    'soc': float(np.clip(_soc, 0.0, 1.0)),
                    'voltage': _pinn_v,
                    'current': _pinn_i,
                    'power': _pinn_p,
                    'temperature': float(np.clip(_temperature, 20.0, 45.0)),
                    'demand_factor': float(np.clip(_demand_factor, 0.1, 1.5)),
                    'load_factor': _load_factor,
                    'grid_voltage': _grid_voltage,        # raw — Layer 1 must see real value
                    'grid_frequency': _grid_frequency,    # raw — Layer 1 must see real value
                    'queue_length': _queue_length,
                    'utilization': _utilization,
                    'urgency_factor': float(np.clip(_urgency, 0.5, 2.0)),
                    'time_of_day': _time_of_day,
                    'system_id': target_system
                }

                # ── Use BestIDSDetector ONLY (matches production environment) ──
                # The production path (hierarchical_cosimulation.py line 1688) calls
                # station.anomaly_detector.detect(feat_14d) directly on the BestIDSDetector
                # — no Layer 1/2 pre-checks.  Training must use the same path so the
                # RL agent learns to evade exactly what's deployed.
                #
                # We reset the detector's internal sliding window and pre-fill it
                # with SEQ_LEN-1 benign warm-up samples, then feed the attacked
                # sample as the final timestep.  This matches:
                #   (a) the training-data construction in compare_ids_models_update.py
                #       (build_dataset → benign warm-up + FDI on last steps)
                #   (b) the report-evaluation path's Priority-2 reconstruction
                # so that the IDS sees the same input distribution in training,
                # evaluation, and deployment.
                _best_ids = getattr(anomaly_detector, '_best_ids_detector', None)

                # ── Benign warm-up: REAL ACN-Data window ──────────────────
                # Sample a SEQ_LEN-1-step slice of an actual ACN charging
                # session.  This matches the IDS training distribution
                # exactly (compare_ids_models_update.py samples from the
                # same pool).  When ACN data isn't available we fall back
                # to the synthetic uniform-jitter construction.
                _seq_len_ids = 10
                _warmup_window = None
                try:
                    from acn_benign_pool import sample_benign_window as _acn_sample
                    _warmup_window = _acn_sample(_seq_len_ids - 1,
                                                 system_id=target_system)
                except Exception:
                    pass

                if _warmup_window is None:
                    # Synthetic fallback (same construction as the IDS training
                    # script's synthetic mode, used only if ACN data is missing).
                    _w_soc_base = float(np.random.uniform(0.10, 0.80))
                    _w_pwr_norm = float(np.random.uniform(0.10, 0.80))
                    _warmup_window = [{
                        'soc':            float(np.clip(_w_soc_base + t * 0.01, 0.0, 1.0)),
                        'voltage':        float(np.random.uniform(220, 260)),
                        'current':        float(np.random.uniform(6, 32)),
                        'power':          float(np.clip(_w_pwr_norm - t * 0.005, 0.0, 1.0)) * 7.68,
                        'temperature':    float(np.random.uniform(20, 35)),
                        'demand_factor':  float(np.random.uniform(0.4, 0.9)),
                        'load_factor':    float(np.random.uniform(0.4, 0.9)),
                        'grid_voltage':   float(np.random.uniform(0.97, 1.03)),
                        'grid_frequency': float(np.random.uniform(59.9, 60.1)),
                        'queue_length':   int(np.random.randint(1, 8)),
                        'utilization':    float(np.random.uniform(0.4, 0.9)),
                        'urgency_factor': float(np.random.uniform(0.5, 1.5)),
                        'time_of_day':    float(np.random.uniform(0, 24)),
                        'system_id':      target_system,
                    } for t in range(_seq_len_ids - 1)]

                def _warmup_input(t):
                    return _warmup_window[t]

                if _best_ids is not None and _best_ids.is_loaded:
                    # Production-aligned path: BestIDSDetector window only.
                    _attacked_feat = anomaly_detector.extract_features(ids_input)

                    # Reset window so stale RL-step features don't pollute the prediction
                    _best_ids.reset()
                    for _t in range(_seq_len_ids - 1):
                        _best_ids.detect(anomaly_detector.extract_features(_warmup_input(_t)))
                    ids_detected_b, ids_proba = _best_ids.detect(_attacked_feat)
                    ids_detected   = bool(ids_detected_b)
                    ids_lstm_score = float(ids_proba)
                    ids_layer = 'best_ids' if ids_detected else None
                    # Capture the EXACT 10x14 window the best model just scored so the
                    # same real attack samples can be replayed through every other
                    # trained IDS model later (evaluate_rl_evasion_comparison.py).
                    ids_window = [np.asarray(w, dtype=float).tolist()
                                  for w in getattr(_best_ids, '_window', [])]
                else:
                    # Fallback: legacy multi_layer_detection if BestIDSDetector unavailable
                    _seq_len = getattr(anomaly_detector, 'sequence_length', 10)
                    anomaly_detector.sequence_buffer = [
                        anomaly_detector.extract_features(_warmup_input(_t))
                        for _t in range(_seq_len - 1)
                    ]
                    anomaly_detector.load_history = []
                    ids_detected, det_results = anomaly_detector.multi_layer_detection(
                        ids_input, target_system
                    )
                    ids_layer = det_results.get('detection_layer', None)
                    ids_lstm_score = det_results.get('layer3_lstm', {}).get('score', 0.0)
                    ids_window = [np.asarray(w, dtype=float).tolist()
                                  for w in (anomaly_detector.sequence_buffer
                                            + [anomaly_detector.extract_features(ids_input)])]

                # detection_risk = raw model probability — gives a CONTINUOUS gradient
                # signal even when detected.  Forcing it to 1.0 on detection (old code)
                # killed the stealth gradient: (1.0 - 1.0) * N = 0, so the agent had
                # no learning signal in the detected region.  Using the raw probability
                # means "barely-detected" (e.g. proba=0.50) gives a stronger reward
                # than "obviously-detected" (e.g. proba=0.99), driving the agent
                # toward the decision boundary.
                detection_risk = float(np.clip(ids_lstm_score, 0.0, 1.0))
            else:
                # Fallback heuristic when no detector is available
                detection_risk = 1.0 - attack_params['stealth_level']
                detection_risk *= (1.0 + real_impact)
                detection_risk = float(np.clip(detection_risk, 0.0, 1.0))
            
            # Logical connection: IDS detection → controller responds → reduced
            # physical impact.  When the IDS triggers, the hierarchical
            # controller enters emergency safe mode and halves charger power
            # (3.84 kW vs 7.68 kW full L2 → 50% mitigation).  Mirror that
            # reduction here so the RL reward reflects the real outcome: a
            # detected attack is less damaging AND incurs a detection penalty.
            IDS_RESPONSE_MITIGATION = 0.50  # matches emergency-mode power ratio
            if ids_detected:
                real_impact = real_impact * (1.0 - IDS_RESPONSE_MITIGATION)

            # True success = had measurable impact AND evaded ALL IDS layers.
            # An attack that is detected — even if it caused damage — is a
            # failure from the attacker's perspective: the defender knows and
            # can respond.  This makes success rate a meaningful metric.
            success = had_impact and not ids_detected
            
            return {
                'success': success,
                'impact': float(real_impact),
                'detection_risk': float(detection_risk),
                'ids_detected': ids_detected,
                'ids_layer': ids_layer,
                'ids_lstm_score': float(ids_lstm_score),
                'pinn_sensitivity': float(pinn_sensitivity),
                'physical_severity': float(physical_severity),
                'voltage_impact': float(voltage_impact),
                'current_impact': float(current_impact),
                'power_impact': float(power_impact),
                'baseline_response': baseline_response,
                'attacked_response': attacked_response,
                'ids_window': ids_window   # 10x14 window scored by best IDS (for multi-model replay)
            }
            
        except Exception as e:
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': str(e)
            }
    
    def _optimize_via_pipeline(self, sys_id: int, station_data: Dict) -> Dict:
        """Run station_data through the full federated PINN pipeline.
        
        Priority 1: optimize_with_constraints() — includes anomaly detection,
                     input sanitization, and output clamping (same as CMS).
        Priority 2: raw optimize_references() — fallback when constraints
                     reject the input, to preserve gradient signal.
        
        Returns dict with 'voltage', 'current', 'power' keys.
        """
        # Try full pipeline first (matches hierarchical CMS path)
        try:
            results, success, message = self.federated_pinn_manager.optimize_with_constraints(
                sys_id, station_data
            )
            if success:
                return {
                    'voltage': results.get('voltage_ref', 400.0),
                    'current': results.get('current_ref', 50.0),
                    'power': results.get('power_ref', 20.0)
                }
        except Exception:
            pass  # Fall through to raw PINN
        
        # Fallback: raw optimize_references (preserves gradient signal
        # when anomaly detection rejects the attacked input)
        cms_model = self.federated_pinn_manager.local_models[sys_id]
        raw = cms_model.optimize_references(station_data)
        
        if isinstance(raw, tuple):
            return {'voltage': raw[0], 'current': raw[1], 'power': raw[2]}
        elif isinstance(raw, dict):
            return raw
        else:
            return {'voltage': float(raw), 'current': 0.0, 'power': 0.0}
    
    def _create_baseline_station_data(self) -> Dict:
        """Create baseline station data (kept for backward compatibility)"""
        num_stations = 10
        return {
            'soc': np.random.uniform(0.2, 0.9, num_stations),
            'voltage': np.random.uniform(220, 240, num_stations),
            'current': np.random.uniform(10, 50, num_stations),
            'power': np.random.uniform(5, 30, num_stations),
            'temperature': np.random.uniform(20, 35, num_stations)
        }
    
    def _apply_attack_perturbations(self, baseline_data: Dict, attack_type: str, magnitude: float) -> Dict:
        """Apply attack-specific perturbations (kept for backward compatibility)"""
        attacked_data = {k: v.copy() for k, v in baseline_data.items()}
        
        if attack_type == 'voltage_manipulation':
            attacked_data['voltage'] *= (1.0 + magnitude * 0.5)
        elif attack_type == 'current_injection':
            attacked_data['current'] *= (1.0 + magnitude * 0.8)
        elif attack_type == 'power_disruption':
            attacked_data['power'] *= (1.0 - magnitude * 0.6)
        elif attack_type == 'communication_spoofing':
            attacked_data['soc'] += magnitude * 0.8
            attacked_data['soc'] = np.clip(attacked_data['soc'], 0, 1)
        elif attack_type == 'data_injection':
            attacked_data['power'] *= (1.0 + magnitude * 0.3)
            attacked_data['temperature'] += magnitude * 10.0
        elif attack_type == 'protocol_manipulation':
            attacked_data['voltage'] *= (1.0 + magnitude * 0.2)
            attacked_data['current'] *= (1.0 + magnitude * 0.3)
        
        return attacked_data
    
    def _calculate_reward(self, attack_result: Dict, stealth_level: float) -> float:
        """Calculate shaped reward for attack — evasion-primary objective.

        Component breakdown (mirrors >97%-evasion design from ids_evasion_analysis.md):

          Successful + undetected path:
            impact_reward   = impact × 8.0                    [max  8.0]
            success_bonus   = 2.0
            evasion_bonus   = (1 − detection_risk) × 50.0     [max 50.0]
            stealth_bonus   = stealth_level × 25.0             [max 25.0]
            streak_bonus    = max(0, streak − 2) × 5.0         [+5 per step after 3rd]
            mission_bonus   = 500.0                            [one-time at streak = 10]

          Successful + detected path:
            detection_penalty = −15.0  (net always negative even at max impact)

          Failed + detected path:
            flat penalty = −3.0

          Failed + undetected path:
            stealth_credit = (1 − detection_risk) × 3.0       [habit-building]
            (clipped to [−3, 0])

        The streak counter (self.consecutive_evasions) is incremented on every
        undetected success and reset to 0 on any detection.  The +500 mission
        bonus fires once at streak=10 and then resets the streak so the agent
        keeps earning it rather than coasting.

        detection_risk comes from the ACTUAL multi-layer IDS (physical +
        pattern + LSTM).  ids_detected=True means the IDS flagged the attack.
        """
        impact = attack_result.get('impact', 0.0)
        detection_risk = attack_result.get('detection_risk', 1.0)
        success = attack_result.get('success', False)
        ids_detected = attack_result.get('ids_detected', False)

        if success:
            impact_reward = impact * 8.0
            success_bonus = 2.0

            if ids_detected:
                # Detected: net-negative, but stealth gradient rewards lower anomaly
                # scores even in the detected state.  This breaks the flat -3.0 wall:
                # agents learn to push detection_risk toward 0 even before they cross
                # the threshold, because lower risk → higher reward here.
                # At detection_risk=0.95: gradient= 0.4; at 0.5: 4.0; at 0.1: 7.2
                self.consecutive_evasions = 0
                detection_penalty = -15.0
                stealth_gradient = (1.0 - detection_risk) * 8.0
                total_reward = impact_reward + success_bonus + detection_penalty + stealth_gradient
            else:
                # Successful AND undetected — update streak first.
                self.consecutive_evasions += 1

                evasion_bonus  = (1.0 - detection_risk) * 50.0
                stealth_bonus  = stealth_level * 25.0
                # Streak bonus kicks in after 3 consecutive evasions (+5 per step)
                streak_bonus   = max(0, self.consecutive_evasions - 2) * 5.0
                # Mission bonus: fires once when streak hits 10, then resets
                if self.consecutive_evasions >= 10:
                    mission_bonus = 500.0
                    self.consecutive_evasions = 0   # reset so agent keeps earning it
                else:
                    mission_bonus = 0.0

                # Continuous gradient toward IDS boundary — rewards distance below
                # the decision threshold (up to +18.6 at score=0.0, zero at 0.465).
                # Gives smooth signal so agent learns to minimise anomaly_score,
                # not just flip the binary detected/not-detected flag.
                ids_margin_bonus = max(0.0, (0.465 - detection_risk)) * 40.0

                total_reward = (impact_reward + success_bonus + evasion_bonus
                                + stealth_bonus + streak_bonus + mission_bonus
                                + ids_margin_bonus)
        else:
            # Failed attack: shaped gradient signal.
            partial_credit = impact * 50.0

            if ids_detected:
                # Failed AND detected — worst outcome; reset streak.
                # Stealth gradient: reward lower anomaly scores so the agent has
                # a continuous signal to minimise detection_risk, breaking the
                # flat-gradient wall that prevents learning when det_rate=1.0.
                # Range: -3.0 (obvious, risk≈1) → +2.0 (borderline, risk≈0).
                self.consecutive_evasions = 0
                stealth_gradient = (1.0 - detection_risk) * 6.0
                total_reward = float(np.clip(-3.0 + stealth_gradient, -3.0, 2.0))
            else:
                # Failed but undetected — reward stealth habit even without impact.
                # Allow small positive rewards (up to +2.0) so the agent gets a
                # real gradient signal for being stealthy.  The old upper clip of
                # 0.0 caused every step to return exactly 0.0 when stealth_credit
                # exceeded 1.0, making DQN's replay buffer useless (all zeros →
                # Q-values never differentiate between systems).
                stealth_credit = (1.0 - detection_risk) * 3.0
                total_reward = -1.0 + partial_credit + stealth_credit
                total_reward = float(np.clip(total_reward, -3.0, 2.0))
        
        # ── Guidance alignment bonus ──
        # When the coordinator provides strategic hints (from Gemini or
        # autonomous defaults), reward the agent for aligning its chosen
        # parameters with those hints.  This creates differentiated training
        # dynamics: Gemini-guided agents learn different magnitude/stealth
        # trade-offs than autonomous agents.
        if self.guidance_hints is not None and self._last_action_params is not None:
            hint_mag = self.guidance_hints.get('magnitude', 0.7)
            hint_dur = self.guidance_hints.get('duration', 30.0)
            hint_stealth = self.guidance_hints.get('stealth_level', 0.7)
            
            act_mag = self._last_action_params.get('magnitude', 0.7)
            act_dur = self._last_action_params.get('duration', 30.0)
            act_stealth = self._last_action_params.get('stealth_level', 0.7)

            # Normalized distance: how far the agent's action is from the hint.
            # Only include dims the SAC agent ACTUALLY controls — otherwise we'd
            # reward/penalise alignment on a dim it can't move.  Reduced (Gemini)
            # mode: SAC controls magnitude + stealth (duration is Gemini's, fixed).
            # Full (autonomous) mode: magnitude + duration + stealth.
            # Ranges: magnitude [0.1, 2.0], duration [5, 60], stealth [0, 1].
            mag_dist = abs(act_mag - hint_mag) / 1.9
            stealth_dist = abs(act_stealth - hint_stealth) / 1.0
            if getattr(self, 'reduced_action', False):
                avg_dist = (mag_dist + stealth_dist) / 2.0
            else:
                dur_dist = abs(act_dur - hint_dur) / 55.0
                avg_dist = (mag_dist + dur_dist + stealth_dist) / 3.0
            
            # Guidance bonus: up to +3.0 for perfect alignment, 0 at max deviation
            # This is ~20-30% of a typical successful reward, enough to bias
            # exploration without overwhelming the primary objectives.
            guidance_bonus = (1.0 - avg_dist) * 3.0
            total_reward += guidance_bonus

        return float(total_reward)

    def _normalize_reward(self, r: float) -> float:
        """Fix 4 — scale reward by a running RMS so magnitudes are comparable across
        attack types (this env is one fixed attack type, so the scale is per-attack).

        Divide-by-scale only (sign/direction preserved, like VecNormalize's
        norm_reward), with EMA-tracked E[r^2] and outlier clipping for early-training
        stability. Persists across circles via self._rew_sq_ema (set in __init__)."""
        r = float(r)
        self._rew_sq_ema = (self._rew_norm_beta * self._rew_sq_ema
                            + (1.0 - self._rew_norm_beta) * r * r)
        rms = float(np.sqrt(self._rew_sq_ema)) + 1e-8
        return float(np.clip(r / rms, -10.0, 10.0))

    def update_cross_circle_stats(self, sys_id: int, detection_rate: float,
                                   success_rate: float, avg_impact: float) -> None:
        """Update cross-circle memory for a system after a training circle completes.
        
        Called by CentralRLCoordinator after each circle's inner training loop.
        Uses an exponential moving average (alpha=0.4) so recent circles have
        more weight than old ones — the agent adapts to changing IDS behavior.
        
        Args:
            sys_id:         1-indexed system that was trained this circle
            detection_rate: fraction of eval episodes where IDS detected the attack
            success_rate:   fraction of eval episodes with success (impact+evasion)
            avg_impact:     mean impact score across eval episodes
        """
        if sys_id not in self.cross_circle_stats:
            self.cross_circle_stats[sys_id] = {
                'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0
            }
        
        stats = self.cross_circle_stats[sys_id]
        n = stats['num_circles']
        
        if n == 0:
            # First circle: use values directly
            stats['detection_rate'] = float(detection_rate)
            stats['success_rate'] = float(success_rate)
            stats['avg_impact'] = float(avg_impact)
        else:
            # Exponential moving average: alpha=0.4 weights recent circles more
            alpha = 0.4
            stats['detection_rate'] = alpha * detection_rate + (1 - alpha) * stats['detection_rate']
            stats['success_rate'] = alpha * success_rate + (1 - alpha) * stats['success_rate']
            stats['avg_impact'] = alpha * avg_impact + (1 - alpha) * stats['avg_impact']
        
        stats['num_circles'] = n + 1

    # Helper methods for observation features
    def _get_attack_success_rate(self, sys_id: int) -> float:
        """Get historical success rate for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5  # neutral
        successes = sum(1 for a in relevant_attacks if a['result']['success'])
        return successes / len(relevant_attacks)
    
    def _get_avg_impact(self, sys_id: int) -> float:
        """Get average impact for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.0
        return np.mean([a['result']['impact'] for a in relevant_attacks])
    
    def _get_detection_rate(self, sys_id: int) -> float:
        """Get real IDS detection rate for this attack type on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5
        return np.mean([float(a['result'].get('ids_detected', 
                        a['result'].get('detection_risk', 0.5) > 0.5))
                        for a in relevant_attacks])
    
    def _get_last_attack_time(self, sys_id: int) -> float:
        """Get normalized time since last attack on this system"""
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 1.0  # long time ago
        last_step = relevant_attacks[-1]['step']
        return (self.current_step - last_step) / self.max_steps
    
    def _get_system_adaptation_level(self, sys_id: int) -> float:
        """Estimate how much the system has adapted to this attack"""
        # Systems adapt more if they've been attacked frequently
        relevant_attacks = [a for a in self.attack_history 
                          if a['target_system'] == sys_id]
        return min(len(relevant_attacks) / 10.0, 1.0)


class DiscreteAttackSpecificEnvironment(gym.Env):
    """
    Discrete version for DQN agents
    Action space: which system to target (0-5)
    """
    
    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6,
                 node_level: bool = False, network_layout=None,
                 n_nodes: int = 10, node_seed: int = 42):
        super(DiscreteAttackSpecificEnvironment, self).__init__()

        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000

        # Phase 2 node-level: DQN selects the ACN NETWORK (node) within the
        # DS assigned by the outer loop, instead of selecting the DS itself.
        self.node_level = bool(node_level)
        self.n_nodes = int(n_nodes)

        # Same observation space as continuous version (action feedback + cross-circle
        # + guidance hint). DQN.step() delegates to continuous_env.step(), whose
        # observation now includes the 3 guidance-hint features (Fix 1), so this shape
        # must match. Kept identical in node_level mode: node identity is stable
        # (seeded layout), so node preference is learnable from reward differentiation.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_systems * 25 + 5 + 3 + 3,),
            dtype=np.float32
        )

        # Discrete action space: Phase 1 = which system; Phase 2 = which node.
        self.action_space = spaces.Discrete(self.n_nodes if self.node_level else num_systems)

        # Use continuous environment for execution (node_level propagated so its
        # station-data builder shapes the input by the targeted network).
        self.continuous_env = AttackSpecificEnvironment(
            federated_pinn_manager, attack_type, num_systems,
            node_level=self.node_level, network_layout=network_layout,
            n_nodes=n_nodes, node_seed=node_seed
        )
        # Keep the discrete env's layout view consistent with the continuous env
        # (build_layout may have run inside the continuous env).
        self.network_layout = self.continuous_env.network_layout
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        return self.continuous_env.reset(seed=seed, options=options)
    
    def step(self, action: int):
        """
        Execute attack on selected system

        Args:
            action: system index (0 to num_systems-1)
        """
        # Start from guidance hints or safe defaults.
        hints = self.continuous_env.guidance_hints
        if hints is not None:
            base_mag    = hints.get('magnitude', 0.7)
            base_dur    = hints.get('duration', 30.0)
            base_stealth = hints.get('stealth_level', 0.6)
        else:
            base_mag     = 0.7
            base_dur     = 30.0
            base_stealth = 0.6

        if self.node_level:
            # Phase 2: `action` = NODE index within the DS the outer loop assigned
            # (that DS is fixed via continuous_env.forced_target_system).  Force
            # the node so this step attacks the chosen network; reward differs by
            # node because the station-data builder shapes the input per network,
            # giving DQN a gradient to learn which network is most exploitable.
            node_idx = int(action) % self.n_nodes
            self.continuous_env.forced_target_node = node_idx
            fts = self.continuous_env.forced_target_system
            if isinstance(fts, (list, tuple, np.ndarray)) and len(fts) > 0:
                sys_id = int(fts[0])
            elif fts is not None:
                sys_id = int(fts)
            else:
                sys_id = 1
        else:
            # Phase 1: `action` = system index.
            sys_id = int(action) + 1  # 1-indexed

        # Adapt parameters to what cross-circle memory says about this system.
        # Systems with high past detection rate → raise stealth to compensate.
        # Systems with high past success rate   → lower magnitude (already works).
        # This creates DIFFERENT rewards for different choices, giving DQN a real
        # gradient to learn which target is most exploitable.
        cc = self.continuous_env.cross_circle_stats.get(
            sys_id, {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0}
        )
        det_rate  = float(cc['detection_rate'])
        succ_rate = float(cc['success_rate'])

        # High detection history → increase stealth (up to +0.3 boost)
        stealth = float(np.clip(base_stealth + det_rate * 0.3, 0.0, 1.0))
        # Low success history  → increase magnitude to try harder (up to +0.5)
        mag = float(np.clip(base_mag + (1.0 - succ_rate) * 0.5, 0.1, 2.0))
        dur = base_dur

        # The DQN's internal continuous_env is full-action [mag, dur, stealth]
        # (target_system removed).  In node_level the DS is forced via
        # forced_target_system and the node via forced_target_node, so no system
        # dim is needed here.
        continuous_action = np.array([mag, dur, stealth], dtype=np.float32)
        return self.continuous_env.step(continuous_action)


class AttackSpecificCoordinator:
    """
    Coordinator for attack-specific RL agents
    Replaces EnhancedDQNSACCoordinator with attack-type specialization
    """
    
    def __init__(self, federated_pinn_manager, num_systems: int = 6, attack_types: List[str] = None,
                 node_level: bool = False, n_nodes: int = 10, node_seed: int = 42,
                 topk_networks: int = 7, reduced_action: bool = False):
        self.federated_pinn_manager = federated_pinn_manager
        self.num_systems = num_systems
        self.attack_types = attack_types or ATTACK_TYPES
        # Reduced-action mode (Gemini-guided): SAC learns magnitude only; the
        # duration/stealth dims are supplied by Gemini.  Autonomous mode leaves
        # this False → full action space.  See AttackSpecificEnvironment.
        self.reduced_action = bool(reduced_action)

        # Phase 2: node-level (ACN-network) targeting.  Build ONE shared layout
        # so every attack type's DQN and SAC env agree on which node is which
        # network.  Default (False) => Phase-1 system-level behaviour.
        self.node_level = bool(node_level)
        self.n_nodes = int(n_nodes)
        # Max # of networks the DQN may attack per DS (top-K subset cap).
        self.topk_networks = int(topk_networks)
        self.network_layout = None
        if self.node_level:
            try:
                from acn_network_layout import build_layout
                self.network_layout = build_layout(
                    num_systems=num_systems, n_nodes=n_nodes, seed=node_seed)
                print(f"   🧩 Node-level ON: {num_systems}×{n_nodes} ACN networks "
                      f"(seed={node_seed})")
            except Exception as _e:
                print(f"   ##?? node_level requested but layout failed: {_e} — falling back")
                self.node_level = False

        # Create agents for each ATTACK TYPE (not system)
        self.dqn_agents = {}
        self.sac_agents = {}
        self.environments = {}

        print(f"🎯 Initializing Attack-Specific RL Agents...")
        print(f"   Attack Types: {len(self.attack_types)}")
        print(f"   Systems: {num_systems}")
        print(f"   Total Agents: {len(self.attack_types) * 2} ({len(self.attack_types)} DQN + {len(self.attack_types)} SAC)")

        for attack_type in self.attack_types:
            print(f"\n   Creating agents for: {attack_type}")

            # Create discrete environment for DQN
            dqn_env = DiscreteAttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems,
                node_level=self.node_level, network_layout=self.network_layout,
                n_nodes=n_nodes, node_seed=node_seed
            )

            # Create continuous environment for SAC
            sac_env = AttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems,
                node_level=self.node_level, network_layout=self.network_layout,
                n_nodes=n_nodes, node_seed=node_seed,
                reduced_action=self.reduced_action
            )
            
            # Store environments
            self.environments[f'dqn_{attack_type}'] = dqn_env
            self.environments[f'sac_{attack_type}'] = sac_env
            
            # Create DQN agent
            self.dqn_agents[attack_type] = DQN(
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
            
            # Create SAC agent
            self.sac_agents[attack_type] = SAC(
                'MlpPolicy',
                sac_env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                verbose=0
            )
            
            print(f"      ✅ DQN agent created (action space: Discrete({num_systems}) - system selection)")
            _sac_dim = int(sac_env.action_space.shape[0])
            _sac_desc = ("magnitude only — reduced/Gemini mode; dur+stealth ceded to Gemini"
                         if self.reduced_action else "magnitude/duration/stealth/system")
            print(f"      ✅ SAC agent created (action space: Continuous({_sac_dim}) - {_sac_desc})")
        
        # Training history: per-episode rewards from .learn() pre-training
        self.training_history = {}
        
        print(f"\n✅ Attack-Specific Coordinator initialized successfully!")
        print(f"   Each agent specializes in ONE attack type across ALL systems")
    
    def train_attack_specialists(self, timesteps_per_attack: int = 10000):
        """
        Train each attack-type agent independently
        
        Key difference from old architecture:
        - OLD: System 1 agents train on System 1 only
        - NEW: voltage_manipulation agents train on ALL 6 systems
        
        Each agent learns which systems are most vulnerable to their attack type
        """
        print(f"\n🎓 Phase 1: Individual Attack-Type Agent Training with PINN Models")
        print(f"   Timesteps per attack type: {timesteps_per_attack}")
        print(f"   Each agent will interact with ALL {self.num_systems} systems\n")
        
        for i, attack_type in enumerate(self.attack_types, 1):
            print(f"🔬 Training {attack_type} specialists ({i}/{len(self.attack_types)})...")
            print(f"   This agent will learn which of the {self.num_systems} systems are vulnerable to {attack_type}")
            
            # Train DQN - learns which SYSTEMS to target
            print(f"   ├── Training DQN agent for {attack_type}...")
            print(f"   │   Action space: Discrete({self.num_systems}) - selecting which system to attack")
            print(f"   │   Learning: 'Which systems (1-{self.num_systems}) are most vulnerable to {attack_type}?'")
            dqn_callback = EpisodeRewardCallback()
            self.dqn_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=dqn_callback
            )
            
            # Train SAC - learns optimal PARAMETERS for this attack
            print(f"   └── Training SAC agent for {attack_type}...")
            print(f"       Action space: Continuous(4) - [magnitude, duration, stealth, target_system]")
            print(f"       Learning: 'What magnitude/duration/stealth works best for {attack_type}?'")
            sac_callback = EpisodeRewardCallback()
            self.sac_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=sac_callback
            )
            
            # Store per-episode training history
            dqn_results = dqn_callback.get_results()
            sac_results = sac_callback.get_results()
            self.training_history[attack_type] = {
                'dqn': dqn_results,
                'sac': sac_results
            }
            
            print(f"   ✅ {attack_type} agents trained across all {self.num_systems} systems")
            print(f"      DQN: {dqn_results['num_episodes']} episodes, mean reward: {dqn_results['mean_reward']:.2f}")
            print(f"      SAC: {sac_results['num_episodes']} episodes, mean reward: {sac_results['mean_reward']:.2f}\n")
        
        print(f"✅ Phase 1 Complete: All {len(self.attack_types)} attack specialists trained!")
        total_episodes = sum(
            self.training_history[at]['dqn']['num_episodes'] + self.training_history[at]['sac']['num_episodes']
            for at in self.training_history
        )
        print(f"   Total pre-training episodes captured: {total_episodes}")
        print(f"\nWhat each agent learned:")
        for attack_type in self.attack_types:
            print(f"   • {attack_type}: Which systems are vulnerable + optimal attack parameters")
    
    def save_agents(self, save_dir: str = "trained_rl_agents"):
        """Save all trained DQN+SAC agents to disk.
        
        Directory structure:
            trained_rl_agents/
                dqn_voltage_manipulation.zip
                sac_voltage_manipulation.zip
                dqn_current_injection.zip
                sac_current_injection.zip
                ...
                training_metadata.json
        """
        import os, json
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        saved_count = 0
        for attack_type in self.attack_types:
            # Save DQN agent
            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}")
            self.dqn_agents[attack_type].save(dqn_path)
            
            # Save SAC agent
            sac_path = os.path.join(save_dir, f"sac_{attack_type}")
            self.sac_agents[attack_type].save(sac_path)
            
            saved_count += 2
            print(f"  💾 Saved DQN+SAC for {attack_type}")
        
        # Save metadata
        sac_obs_shape = list(self.environments[f'sac_{self.attack_types[0]}'].observation_space.shape)
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'attack_types': self.attack_types,
            'num_systems': self.num_systems,
            'num_agents': saved_count,
            'observation_space_shape': sac_obs_shape,
            'training_history_summary': {
                at: {
                    'dqn_episodes': self.training_history.get(at, {}).get('dqn', {}).get('num_episodes', 0),
                    'sac_episodes': self.training_history.get(at, {}).get('sac', {}).get('num_episodes', 0),
                    'dqn_mean_reward': self.training_history.get(at, {}).get('dqn', {}).get('mean_reward', 0.0),
                    'sac_mean_reward': self.training_history.get(at, {}).get('sac', {}).get('mean_reward', 0.0),
                } for at in self.attack_types
            }
        }
        meta_path = os.path.join(save_dir, "training_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved {saved_count} agents to {save_dir}/")
        return save_dir
    
    def load_agents(self, save_dir: str = "trained_rl_agents") -> bool:
        """Load previously trained DQN+SAC agents from disk.
        
        Returns True if all agents loaded successfully, False otherwise.
        
        OBSERVATION SPACE COMPATIBILITY CHECK:
        If the saved models were trained with a different observation space
        (e.g., old 150-feature space vs new 158-feature space), loading them
        will cause a shape mismatch crash on the first predict() call.
        This method detects the mismatch and skips loading, letting the system
        retrain from scratch with the correct observation space.
        Old models are archived to a timestamped subfolder so they are not lost.
        """
        import os, json, zipfile, shutil
        from datetime import datetime
        
        if not os.path.isdir(save_dir):
            print(f"  ⚠️ Agent directory not found: {save_dir}")
            return False
        
        # --- Detect observation space mismatch before loading ---
        # SB3 saves observation space shape in the zip as 'data' (a pickle).
        # We check the first available agent to detect mismatch early.
        current_obs_shape = self.environments[
            f'sac_{self.attack_types[0]}'
        ].observation_space.shape
        
        mismatch_detected = False
        sample_path = os.path.join(save_dir, f"sac_{self.attack_types[0]}.zip")
        if os.path.exists(sample_path):
            try:
                # Load without env to inspect saved observation space
                saved_agent = SAC.load(sample_path, env=None)
                saved_obs_shape = saved_agent.observation_space.shape
                if saved_obs_shape != current_obs_shape:
                    mismatch_detected = True
                    print(f"\n  ⚠️  OBSERVATION SPACE MISMATCH DETECTED")
                    print(f"     Saved models:      obs_shape={saved_obs_shape}")
                    print(f"     Current env:       obs_shape={current_obs_shape}")
                    print(f"     Cause: observation space was expanded from {saved_obs_shape[0]}"
                          f" → {current_obs_shape[0]} features (action feedback + cross-circle memory added)")
                    print(f"     Action: Archiving old models and retraining from scratch.")
                    print(f"     This is CORRECT — old models were trained with wrong reward signals anyway.")
                del saved_agent
            except Exception as e:
                print(f"  ⚠️ Could not inspect saved model: {e}")
        
        if mismatch_detected:
            # Archive old models to a timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = f"{save_dir}_archived_{timestamp}"
            try:
                shutil.move(save_dir, archive_dir)
                print(f"  📦 Old models archived to: {archive_dir}/")
            except Exception as e:
                print(f"  ⚠️ Could not archive old models: {e}")
            print(f"  🔄 Retraining from scratch with new {current_obs_shape[0]}-feature observation space.\n")
            return False
        
        loaded_count = 0
        for attack_type in self.attack_types:
            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}.zip")
            sac_path = os.path.join(save_dir, f"sac_{attack_type}.zip")
            
            if not os.path.exists(dqn_path) or not os.path.exists(sac_path):
                print(f"  ⚠️ Missing agent files for {attack_type}")
                continue
            
            # Load into existing environments
            dqn_env = self.environments[f'dqn_{attack_type}']
            sac_env = self.environments[f'sac_{attack_type}']
            
            try:
                self.dqn_agents[attack_type] = DQN.load(dqn_path, env=dqn_env)
                self.sac_agents[attack_type] = SAC.load(sac_path, env=sac_env)
                loaded_count += 2
                print(f"  📂 Loaded DQN+SAC for {attack_type}")
            except Exception as e:
                print(f"  ⚠️ Failed to load agents for {attack_type}: {e}")
                print(f"     Skipping — agent will be retrained from scratch.")
        
        # Load metadata if available
        meta_path = os.path.join(save_dir, "training_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"  📋 Agents trained at: {metadata.get('timestamp', 'unknown')}")
            saved_obs = metadata.get('observation_space_shape')
            if saved_obs and tuple(saved_obs) != current_obs_shape:
                print(f"  ⚠️ Metadata obs shape {saved_obs} != current {current_obs_shape}")
        
        success = loaded_count == len(self.attack_types) * 2
        if success:
            print(f"✅ Loaded {loaded_count} agents from {save_dir}/")
        else:
            print(f"⚠️ Partially loaded {loaded_count}/{len(self.attack_types) * 2} agents from {save_dir}/")
        
        return success
    
    def get_agent_action(self, attack_type: str, agent_type: str, observation: np.ndarray):
        """Get action from specific attack-type agent"""
        if agent_type == 'dqn':
            action, _ = self.dqn_agents[attack_type].predict(observation, deterministic=False)
            return action
        elif agent_type == 'sac':
            action, _ = self.sac_agents[attack_type].predict(observation, deterministic=False)
            return action
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _sac_q_value(self, sac_agent, obs, action):
        """Q(s,a) under SAC's OWN twin critic (min of the two heads).

        The action is scaled to [-1,1] exactly as SB3 stores it in the replay
        buffer, so the critic sees it inside its training distribution.  Returns
        a float, or None if the critic cannot be queried (version/shape issue)
        — the caller then falls back to trusting the agent's own action rather
        than blindly forcing Gemini's.  This is the arbiter that lets Gemini's
        tactical hint win ONLY when the agent's learned value endorses it.
        """
        try:
            import torch
            policy = sac_agent.policy
            device = sac_agent.device
            low  = np.asarray(sac_agent.action_space.low,  dtype=np.float32)
            high = np.asarray(sac_agent.action_space.high, dtype=np.float32)
            a = np.clip(np.asarray(action, dtype=np.float32), low, high)
            a_scaled = 2.0 * (a - low) / (high - low) - 1.0
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=device).reshape(1, -1)
            act_t = torch.as_tensor(a_scaled, device=device).reshape(1, -1)
            with torch.no_grad():
                qs = policy.critic(obs_t, act_t)          # tuple of twin Q-nets
                q = torch.min(torch.cat(list(qs), dim=1), dim=1).values
            return float(q.item())
        except Exception as e:
            print(f"      [value-gate] SAC critic unavailable: {e}")
            return None

    def _dqn_q_values(self, dqn_agent, obs):
        """Q-values over the discrete system actions from DQN's OWN q_net.

        Returns a 1-D numpy array indexed by (system_id - 1), or None on
        failure (caller then keeps DQN's argmax pick without gating).
        """
        try:
            import torch
            q_net = dqn_agent.q_net
            device = dqn_agent.device
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=device).reshape(1, -1)
            with torch.no_grad():
                q = q_net(obs_t).cpu().numpy().reshape(-1)
            return q
        except Exception as e:
            print(f"      [value-gate] DQN q_net unavailable: {e}")
            return None

    def execute_deployment(self, deployment: AttackDeployment):
        """Execute a deployment strategy using BOTH trained DQN + SAC agents.

        Division of responsibility:
          - DQN: trained system selector — picks the target system based on
            its learned Q-values (which system has historically been most
            evadeable for this attack type).
          - SAC: trained parameter optimiser — picks (magnitude, duration,
            stealth) for the chosen system.
          - Gemini: provides STRATEGIC hints — a candidate target list (used
            to constrain DQN's choice) and tactical hints (blended with SAC
            output).  Gemini does NOT hard-pick the system anymore, so the
            same prompt no longer collapses to the same fixed assignment
            every episode — DQN's per-episode observations introduce real
            variation.

        Why this matters:
          The earlier code never queried DQN in deployment.  It hard-set
          system_id = Gemini's first target, which collapsed across all
          episodes (Gemini consistently produced voltage→sys1, current→sys2,
          etc.).  Now DQN's learned policy actually drives system choice,
          and stochastic sampling gives action diversity even when
          observations are similar.
        """
        attack_type     = deployment.attack_type
        target_systems  = list(deployment.target_systems) if deployment.target_systems else []
        results         = []

        # Gemini tactical hints (may be defaults if Gemini didn't specify)
        g_mag     = float(deployment.magnitude)
        g_dur     = float(deployment.duration)
        g_stealth = float(deployment.stealth_level)

        sac_agent = self.sac_agents.get(attack_type)
        dqn_agent = self.dqn_agents.get(attack_type)
        sac_env   = self.environments.get(f'sac_{attack_type}')
        dqn_env   = self.environments.get(f'dqn_{attack_type}')

        # ── Step 1: DS (outer: bandit/Gemini) + node (DQN, both modes) ──
        # DS comes from the outer loop — the UCB bandit in autonomous, Gemini in
        # guided mode (deployment.target_systems).  The trained DQN selects the
        # ACN node within that DS, in BOTH modes.
        node_level = getattr(sac_env, 'node_level', getattr(dqn_env, 'node_level', False))
        n_nodes = int(getattr(sac_env, 'n_nodes', 10)) if sac_env is not None else 10
        final_sys = int(target_systems[0]) if target_systems else int(np.random.randint(1, self.num_systems + 1))
        system_source = "outer(bandit/Gemini)"

        final_node = None
        node_source = "n/a"
        if node_level and dqn_agent is not None and dqn_env is not None:
            try:
                if hasattr(dqn_env, 'continuous_env'):
                    dqn_env.continuous_env.forced_target_system = [final_sys]
                dqn_obs, _ = dqn_env.reset()
                dqn_action, _ = dqn_agent.predict(dqn_obs, deterministic=False)
                final_node = int(dqn_action) % max(1, n_nodes)
                node_source = "DQN"
                if hasattr(dqn_env, 'continuous_env'):
                    dqn_env.continuous_env.forced_target_system = None
            except Exception as e:
                print(f"      ##?? DQN node inference failed for {attack_type}: {e}")

        if sac_env is None:
            print(f"      ##?? No SAC env for {attack_type}")
            return results

        # ── Step 2: SAC parameters (no value-gate) ──────────────────────
        # Autonomous: SAC deploys [magnitude, duration, stealth].
        # Gemini:     SAC deploys [magnitude, stealth]; DURATION is ceded to
        #             Gemini and injected from guidance_hints inside step().
        reduced = getattr(sac_env, 'reduced_action', False)
        sac_env.forced_target_system = [final_sys]
        sac_env.guidance_hints = {
            'magnitude': g_mag, 'duration': g_dur, 'stealth_level': g_stealth
        }
        if node_level and final_node is not None:
            sac_env.forced_target_node = int(final_node)
        obs, _ = sac_env.reset()

        if sac_agent is not None:
            trained_action, _ = sac_agent.predict(obs, deterministic=False)
            action = np.atleast_1d(trained_action.astype(np.float32))
            if reduced:
                param_source = "SAC(mag,stealth)+Gemini(dur)"
                print(f"      🎯 deploy[gemini]: mag={float(action[0]):.2f} "
                      f"dur={g_dur:.1f}(gem) stealth={float(action[1]):.2f} "
                      f"→ sys {final_sys} node {final_node} "
                      f"[sys_src={system_source}, node_src={node_source}]")
            else:
                param_source = "SAC(mag,dur,stealth)"
                print(f"      🎯 deploy[auto]: mag={float(action[0]):.2f} "
                      f"dur={float(action[1]):.1f} stealth={float(action[2]):.2f} "
                      f"→ sys {final_sys} node {final_node} "
                      f"[sys_src={system_source}, node_src={node_source}]")
        else:
            print(f"      ##?? No trained SAC agent for {attack_type}, using hint params")
            action = (np.array([g_mag, g_stealth], dtype=np.float32) if reduced
                      else np.array([g_mag, g_dur, g_stealth], dtype=np.float32))
            param_source = "hints(no-sac)"

        _, reward, _, _, info = sac_env.step(action)

        # Unlock the env so deploy-time conditioning doesn't leak into later use.
        sac_env.forced_target_system = None
        sac_env.guidance_hints = None
        sac_env.forced_target_node = None

        results.append({
            'attack_type':   attack_type,
            'system_id':     final_sys,
            'target_node':   final_node,
            'node_source':   node_source,
            'reward':        reward,
            'gemini_targets': target_systems,
            'system_source': system_source,
            'param_source':  param_source,
            'result':        info['attack_result'],
        })

        return results
