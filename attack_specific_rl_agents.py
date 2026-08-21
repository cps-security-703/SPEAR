#!/usr/bin/env python3


import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback


ATTACK_TYPES = [
    'voltage_manipulation',
    'current_injection',
    'power_disruption',
    'communication_spoofing',
    'data_injection',
    'protocol_manipulation'
]

class EpisodeRewardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_step(self) -> bool:

        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])

        if len(rewards) > 0:
            self.current_episode_reward += float(rewards[0])
            self.current_episode_length += 1


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

    attack_type: str
    target_systems: List[int]
    magnitude: float
    duration: float
    stealth_level: float
    priority: int = 1


class AttackSpecificEnvironment(gym.Env):


    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6,
                 node_level: bool = False, network_layout=None,
                 n_nodes: int = 10, node_seed: int = 42):
        super(AttackSpecificEnvironment, self).__init__()

        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000


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


        self.forced_target_node = None
        self._last_target_node = 0


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,

            shape=(num_systems * 25 + 5 + 3 + 3,),
            dtype=np.float32
        )


        self.action_space = spaces.Box(
            low=np.array([0.1, 5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 60.0, 1.0, float(num_systems-1)], dtype=np.float32),
            dtype=np.float32
        )


        self.episode_rewards = []
        self.attack_history = []


        self.forced_target_system = None


        self.guidance_hints = None


        self._last_action_params = None


        self._last_action_feedback = np.array(
            [0.7, 30.0, 0.7, 0.0, 0.0], dtype=np.float32
        )


        self.cross_circle_stats = {
            s: {'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0}
            for s in range(1, num_systems + 1)
        }


        self._last_pinn_state = {}


        self.consecutive_evasions = 0


        self._rew_sq_ema = 1.0
        self._rew_norm_beta = 0.999

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.attack_history = []
        self.consecutive_evasions = 0


        self._last_pinn_state = {}


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


        observation = self._get_global_observation()

        return observation, {}

    def step(self, action: np.ndarray):

        self.current_step += 1


        magnitude = float(action[0])
        duration = float(action[1])
        stealth_level = float(action[2])


        if self.forced_target_system is not None:
            if isinstance(self.forced_target_system, (list, tuple, np.ndarray)):
                target_system_id = int(np.random.choice(self.forced_target_system))
            else:
                target_system_id = int(self.forced_target_system)
        else:
            target_system_id = int(np.clip(action[3], 0, self.num_systems-1)) + 1


        target_node = None
        if self.node_level:
            if self.forced_target_node is not None:
                target_node = int(self.forced_target_node) % self.n_nodes
            else:
                target_node = int(np.random.randint(0, self.n_nodes))
            self._last_target_node = target_node


        attack_params = {
            'attack_type': self.attack_type,
            'target_system': target_system_id,
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }
        if target_node is not None:
            attack_params['target_node'] = target_node


        self._last_action_params = {
            'magnitude': magnitude,
            'duration': duration,
            'stealth_level': stealth_level
        }


        attack_result = self._execute_attack_on_pinn(attack_params)


        self._last_action_feedback = np.array([
            magnitude / 2.0,
            duration / 60.0,
            stealth_level,
            float(attack_result.get('ids_detected', False)),
            float(attack_result.get('ids_lstm_score', 0.0)),
        ], dtype=np.float32)


        reward = self._calculate_reward(attack_result, stealth_level)


        reward = self._normalize_reward(reward)


        observation = self._get_global_observation()


        done = self.current_step >= self.max_steps
        truncated = False


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

        baseline_soc = np.random.uniform(0.3, 0.8)
        demand_factor = 0.5 + 0.2 * np.sin(self.current_step * 0.01)
        urgency_factor = 2.0 if baseline_soc < 0.2 else (0.3 if baseline_soc > 0.8 else 1.0)
        voltage_priority = max(0, 0.95 - 1.0)
        nominal_voltage = 400.0
        nominal_current = max(1.0, baseline_soc * 100.0 + 20.0)
        nominal_power = nominal_voltage * nominal_current / 1000.0
        temperature = 25.0 + 5.0 * demand_factor
        queue_len = max(0, int(demand_factor * 8))
        time_of_day = (self.current_step * 0.1) % 24.0


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

            'voltage': nominal_voltage,
            'current': nominal_current,
            'power': nominal_power,
            'temperature': temperature,
            'queue_length': queue_len,
            'utilization': _node_util,
            'time_of_day': time_of_day,
            'system_id': sys_id,

            'target_node': int(self._last_target_node) if self.node_level else -1,
            'network_type': node_type,
        }

    def _get_global_observation(self) -> np.ndarray:

        observations = []

        for sys_id in range(1, self.num_systems + 1):
            if sys_id in self.federated_pinn_manager.local_models:

                ps = self._last_pinn_state.get(sys_id, {})
                sd = ps.get('station_data', self._build_baseline_station_data(sys_id))


                pinn_v = ps.get('voltage', 400.0)
                pinn_i = ps.get('current', 50.0)
                pinn_p = ps.get('power', 20.0)


                voltage_pu = pinn_v / 400.0
                current_norm = pinn_i / 125.0
                power_norm = pinn_p / 75.0

                system_state = np.array([

                    voltage_pu,
                    current_norm,
                    power_norm,
                    sd.get('grid_frequency', 60.0) / 60.0,
                    sd.get('system_efficiency', 0.95),


                    sd.get('soc', 0.5),
                    sd.get('demand_factor', 0.5),
                    power_norm,
                    sd.get('temperature', 25.0) / 50.0,
                    sd.get('utilization', 0.5),


                    self._get_avg_impact(sys_id),
                    self._get_detection_rate(sys_id),
                    min(len([a for a in self.attack_history
                             if a['target_system'] == sys_id]) / 5.0, 1.0),
                    1.0 - self._get_system_adaptation_level(sys_id),
                    1.0 - self._get_attack_success_rate(sys_id),


                    self._get_attack_success_rate(sys_id),
                    self._get_avg_impact(sys_id),
                    self._get_detection_rate(sys_id),
                    self._get_last_attack_time(sys_id),
                    self._get_system_adaptation_level(sys_id),


                    float(sys_id) / self.num_systems,
                    sd.get('demand_factor', 0.5),
                    sd.get('load_factor', 0.5),
                    sd.get('urgency_factor', 1.0) / 2.0,
                    sd.get('grid_voltage', 1.0),
                ], dtype=np.float32)
            else:
                system_state = np.zeros(25, dtype=np.float32)

            observations.append(system_state)


        action_feedback = getattr(self, '_last_action_feedback',
                                  np.zeros(5, dtype=np.float32))


        _fts = self.forced_target_system
        if _fts is not None and not isinstance(_fts, (list, tuple, np.ndarray)):
            cc_sys = int(_fts)
        elif self.attack_history:
            cc_sys = int(self.attack_history[-1]['target_system'])
        else:
            cc_sys = 1
        cc = self.cross_circle_stats.get(cc_sys,
             {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0})
        cross_circle_memory = np.array([
            float(cc['detection_rate']),
            float(cc['success_rate']),
            float(cc['avg_impact']),
        ], dtype=np.float32)


        h = self.guidance_hints or {}
        hint_obs = np.array([
            (float(h.get('magnitude', 0.7)) - 0.1) / 1.9,
            (float(h.get('duration', 30.0)) - 5.0) / 55.0,
            float(h.get('stealth_level', 0.7)),
        ], dtype=np.float32)

        return np.concatenate(
            observations + [action_feedback, cross_circle_memory, hint_obs]
        )

    def _execute_attack_on_pinn(self, attack_params: Dict) -> Dict:

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


        effective_magnitude = magnitude * max(0.05, 1.0 - 0.75 * stealth_level)

        try:

            baseline_station_data = self._build_baseline_station_data(target_system)
            baseline_soc = baseline_station_data['soc']


            baseline_response = self._optimize_via_pipeline(
                target_system, baseline_station_data
            )


            self._last_pinn_state[target_system] = {
                'voltage': baseline_response.get('voltage', 400.0),
                'current': baseline_response.get('current', 50.0),
                'power': baseline_response.get('power', 20.0),
                'station_data': baseline_station_data
            }


            attacked_station_data = dict(baseline_station_data)
            gradual_magnitude = effective_magnitude

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


                duration_s = float(attack_params.get('duration', 30.0))
                time_factor = duration_s / 4.0
                oscillation = float(np.sin(time_factor)) * 20.0 + 1.0
                amplitude_growth = 1.0 + (magnitude * 12.0)
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7)
                    * oscillation * amplitude_growth)
                attacked_station_data['grid_voltage'] = (
                    baseline_station_data.get('grid_voltage', 1.0)
                    * (1.0 - gradual_magnitude * 0.2))
                attacked_station_data['power_multiplier'] = oscillation * amplitude_growth

            else:

                cumulative_factor = 1.0 + (gradual_magnitude * 30.0)
                attacked_station_data['demand_factor'] = (
                    baseline_station_data.get('demand_factor', 0.7) * cumulative_factor)
                attacked_station_data['power_multiplier'] = 1.0 + gradual_magnitude * 25.0


            _defaults = {
                'grid_voltage': 1.0, 'grid_frequency': 60.0, 'demand_factor': 0.7,
                'urgency_factor': 1.0, 'soc': float(baseline_soc),
                'voltage_priority': 0.0, 'power_multiplier': 1.0, 'load_factor': 0.7,
            }
            for _k, _dv in _defaults.items():
                if _k in attacked_station_data and not np.isfinite(attacked_station_data[_k]):
                    attacked_station_data[_k] = _dv


            attacked_response = self._optimize_via_pipeline(
                target_system, attacked_station_data
            )


            self._last_pinn_state[target_system] = {
                'voltage': attacked_response.get('voltage', 400.0),
                'current': attacked_response.get('current', 50.0),
                'power': attacked_response.get('power', 20.0),
                'station_data': attacked_station_data
            }


            voltage_impact = abs(attacked_response['voltage'] - baseline_response['voltage']) / (abs(baseline_response['voltage']) + 1e-6)
            current_impact = abs(attacked_response['current'] - baseline_response['current']) / (abs(baseline_response['current']) + 1e-6)
            power_impact = abs(attacked_response['power'] - baseline_response['power']) / (abs(baseline_response['power']) + 1e-6)
            pinn_sensitivity = max(voltage_impact, current_impact, power_impact)


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


                    physical_severity += abs(a_val - b_val)
            physical_severity /= float(len(_SEVERITY_KEYS))


            pinn_weight = min(1.0, pinn_sensitivity * 100.0)
            real_impact = physical_severity * (0.6 + 0.4 * pinn_weight)


            had_impact = real_impact > 0.01


            detection_risk = 0.0
            ids_detected = False
            ids_layer = None
            ids_lstm_score = 0.0
            ids_window = None

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


                ids_input = {
                    'soc': float(np.clip(_soc, 0.0, 1.0)),
                    'voltage': _pinn_v,
                    'current': _pinn_i,
                    'power': _pinn_p,
                    'temperature': float(np.clip(_temperature, 20.0, 45.0)),
                    'demand_factor': float(np.clip(_demand_factor, 0.1, 1.5)),
                    'load_factor': _load_factor,
                    'grid_voltage': _grid_voltage,
                    'grid_frequency': _grid_frequency,
                    'queue_length': _queue_length,
                    'utilization': _utilization,
                    'urgency_factor': float(np.clip(_urgency, 0.5, 2.0)),
                    'time_of_day': _time_of_day,
                    'system_id': target_system
                }


                _best_ids = getattr(anomaly_detector, '_best_ids_detector', None)


                _seq_len_ids = 10
                _warmup_window = None
                try:
                    from acn_benign_pool import sample_benign_window as _acn_sample
                    _warmup_window = _acn_sample(_seq_len_ids - 1,
                                                 system_id=target_system)
                except Exception:
                    pass

                if _warmup_window is None:


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

                    _attacked_feat = anomaly_detector.extract_features(ids_input)


                    _best_ids.reset()
                    for _t in range(_seq_len_ids - 1):
                        _best_ids.detect(anomaly_detector.extract_features(_warmup_input(_t)))
                    ids_detected_b, ids_proba = _best_ids.detect(_attacked_feat)
                    ids_detected   = bool(ids_detected_b)
                    ids_lstm_score = float(ids_proba)
                    ids_layer = 'best_ids' if ids_detected else None


                    ids_window = [np.asarray(w, dtype=float).tolist()
                                  for w in getattr(_best_ids, '_window', [])]
                else:

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


                detection_risk = float(np.clip(ids_lstm_score, 0.0, 1.0))
            else:

                detection_risk = 1.0 - attack_params['stealth_level']
                detection_risk *= (1.0 + real_impact)
                detection_risk = float(np.clip(detection_risk, 0.0, 1.0))


            IDS_RESPONSE_MITIGATION = 0.50
            if ids_detected:
                real_impact = real_impact * (1.0 - IDS_RESPONSE_MITIGATION)


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
                'ids_window': ids_window
            }

        except Exception as e:
            return {
                'success': False,
                'impact': 0.0,
                'detection_risk': 1.0,
                'error': str(e)
            }

    def _optimize_via_pipeline(self, sys_id: int, station_data: Dict) -> Dict:


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
            pass


        cms_model = self.federated_pinn_manager.local_models[sys_id]
        raw = cms_model.optimize_references(station_data)

        if isinstance(raw, tuple):
            return {'voltage': raw[0], 'current': raw[1], 'power': raw[2]}
        elif isinstance(raw, dict):
            return raw
        else:
            return {'voltage': float(raw), 'current': 0.0, 'power': 0.0}

    def _create_baseline_station_data(self) -> Dict:

        num_stations = 10
        return {
            'soc': np.random.uniform(0.2, 0.9, num_stations),
            'voltage': np.random.uniform(220, 240, num_stations),
            'current': np.random.uniform(10, 50, num_stations),
            'power': np.random.uniform(5, 30, num_stations),
            'temperature': np.random.uniform(20, 35, num_stations)
        }

    def _apply_attack_perturbations(self, baseline_data: Dict, attack_type: str, magnitude: float) -> Dict:

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

        impact = attack_result.get('impact', 0.0)
        detection_risk = attack_result.get('detection_risk', 1.0)
        success = attack_result.get('success', False)
        ids_detected = attack_result.get('ids_detected', False)

        if success:
            impact_reward = impact * 8.0
            success_bonus = 2.0

            if ids_detected:


                self.consecutive_evasions = 0
                detection_penalty = -15.0
                stealth_gradient = (1.0 - detection_risk) * 8.0
                total_reward = impact_reward + success_bonus + detection_penalty + stealth_gradient
            else:

                self.consecutive_evasions += 1

                evasion_bonus  = (1.0 - detection_risk) * 50.0
                stealth_bonus  = stealth_level * 25.0

                streak_bonus   = max(0, self.consecutive_evasions - 2) * 5.0

                if self.consecutive_evasions >= 10:
                    mission_bonus = 500.0
                    self.consecutive_evasions = 0
                else:
                    mission_bonus = 0.0


                ids_margin_bonus = max(0.0, (0.465 - detection_risk)) * 40.0

                total_reward = (impact_reward + success_bonus + evasion_bonus
                                + stealth_bonus + streak_bonus + mission_bonus
                                + ids_margin_bonus)
        else:

            partial_credit = impact * 50.0

            if ids_detected:


                self.consecutive_evasions = 0
                stealth_gradient = (1.0 - detection_risk) * 6.0
                total_reward = float(np.clip(-3.0 + stealth_gradient, -3.0, 2.0))
            else:


                stealth_credit = (1.0 - detection_risk) * 3.0
                total_reward = -1.0 + partial_credit + stealth_credit
                total_reward = float(np.clip(total_reward, -3.0, 2.0))


        if self.guidance_hints is not None and self._last_action_params is not None:
            hint_mag = self.guidance_hints.get('magnitude', 0.7)
            hint_dur = self.guidance_hints.get('duration', 30.0)
            hint_stealth = self.guidance_hints.get('stealth_level', 0.7)

            act_mag = self._last_action_params.get('magnitude', 0.7)
            act_dur = self._last_action_params.get('duration', 30.0)
            act_stealth = self._last_action_params.get('stealth_level', 0.7)


            mag_dist = abs(act_mag - hint_mag) / 1.9
            dur_dist = abs(act_dur - hint_dur) / 55.0
            stealth_dist = abs(act_stealth - hint_stealth) / 1.0


            avg_dist = (mag_dist + dur_dist + stealth_dist) / 3.0


            guidance_bonus = (1.0 - avg_dist) * 3.0
            total_reward += guidance_bonus

        return float(total_reward)

    def _normalize_reward(self, r: float) -> float:

        r = float(r)
        self._rew_sq_ema = (self._rew_norm_beta * self._rew_sq_ema
                            + (1.0 - self._rew_norm_beta) * r * r)
        rms = float(np.sqrt(self._rew_sq_ema)) + 1e-8
        return float(np.clip(r / rms, -10.0, 10.0))

    def update_cross_circle_stats(self, sys_id: int, detection_rate: float,
                                   success_rate: float, avg_impact: float) -> None:

        if sys_id not in self.cross_circle_stats:
            self.cross_circle_stats[sys_id] = {
                'detection_rate': 0.5, 'success_rate': 0.5,
                'avg_impact': 0.0, 'num_circles': 0
            }

        stats = self.cross_circle_stats[sys_id]
        n = stats['num_circles']

        if n == 0:

            stats['detection_rate'] = float(detection_rate)
            stats['success_rate'] = float(success_rate)
            stats['avg_impact'] = float(avg_impact)
        else:

            alpha = 0.4
            stats['detection_rate'] = alpha * detection_rate + (1 - alpha) * stats['detection_rate']
            stats['success_rate'] = alpha * success_rate + (1 - alpha) * stats['success_rate']
            stats['avg_impact'] = alpha * avg_impact + (1 - alpha) * stats['avg_impact']

        stats['num_circles'] = n + 1


    def _get_attack_success_rate(self, sys_id: int) -> float:

        relevant_attacks = [a for a in self.attack_history
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5
        successes = sum(1 for a in relevant_attacks if a['result']['success'])
        return successes / len(relevant_attacks)

    def _get_avg_impact(self, sys_id: int) -> float:

        relevant_attacks = [a for a in self.attack_history
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.0
        return np.mean([a['result']['impact'] for a in relevant_attacks])

    def _get_detection_rate(self, sys_id: int) -> float:

        relevant_attacks = [a for a in self.attack_history
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 0.5
        return np.mean([float(a['result'].get('ids_detected',
                        a['result'].get('detection_risk', 0.5) > 0.5))
                        for a in relevant_attacks])

    def _get_last_attack_time(self, sys_id: int) -> float:

        relevant_attacks = [a for a in self.attack_history
                          if a['target_system'] == sys_id]
        if not relevant_attacks:
            return 1.0
        last_step = relevant_attacks[-1]['step']
        return (self.current_step - last_step) / self.max_steps

    def _get_system_adaptation_level(self, sys_id: int) -> float:


        relevant_attacks = [a for a in self.attack_history
                          if a['target_system'] == sys_id]
        return min(len(relevant_attacks) / 10.0, 1.0)


class DiscreteAttackSpecificEnvironment(gym.Env):


    def __init__(self, federated_pinn_manager, attack_type: str, num_systems: int = 6,
                 node_level: bool = False, network_layout=None,
                 n_nodes: int = 10, node_seed: int = 42):
        super(DiscreteAttackSpecificEnvironment, self).__init__()

        self.federated_pinn_manager = federated_pinn_manager
        self.attack_type = attack_type
        self.num_systems = num_systems
        self.current_step = 0
        self.max_steps = 1000


        self.node_level = bool(node_level)
        self.n_nodes = int(n_nodes)


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_systems * 25 + 5 + 3 + 3,),
            dtype=np.float32
        )


        self.action_space = spaces.Discrete(self.n_nodes if self.node_level else num_systems)


        self.continuous_env = AttackSpecificEnvironment(
            federated_pinn_manager, attack_type, num_systems,
            node_level=self.node_level, network_layout=network_layout,
            n_nodes=n_nodes, node_seed=node_seed
        )


        self.network_layout = self.continuous_env.network_layout

    def reset(self, seed=None, options=None):

        return self.continuous_env.reset(seed=seed, options=options)

    def step(self, action: int):


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

            sys_id = int(action) + 1


        cc = self.continuous_env.cross_circle_stats.get(
            sys_id, {'detection_rate': 0.5, 'success_rate': 0.5, 'avg_impact': 0.0}
        )
        det_rate  = float(cc['detection_rate'])
        succ_rate = float(cc['success_rate'])


        stealth = float(np.clip(base_stealth + det_rate * 0.3, 0.0, 1.0))

        mag = float(np.clip(base_mag + (1.0 - succ_rate) * 0.5, 0.1, 2.0))
        dur = base_dur


        sys_dim = 0.0 if self.node_level else float(action)
        continuous_action = np.array([mag, dur, stealth, sys_dim], dtype=np.float32)
        return self.continuous_env.step(continuous_action)


class AttackSpecificCoordinator:


    def __init__(self, federated_pinn_manager, num_systems: int = 6, attack_types: List[str] = None,
                 node_level: bool = False, n_nodes: int = 10, node_seed: int = 42,
                 topk_networks: int = 7):
        self.federated_pinn_manager = federated_pinn_manager
        self.num_systems = num_systems
        self.attack_types = attack_types or ATTACK_TYPES


        self.node_level = bool(node_level)
        self.n_nodes = int(n_nodes)

        self.topk_networks = int(topk_networks)
        self.network_layout = None
        if self.node_level:
            try:
                from acn_network_layout import build_layout
                self.network_layout = build_layout(
                    num_systems=num_systems, n_nodes=n_nodes, seed=node_seed)
                print(f"    Node-level ON: {num_systems}×{n_nodes} ACN networks "
                      f"(seed={node_seed})")
            except Exception as _e:
                print(f"   ##?? node_level requested but layout failed: {_e} — falling back")
                self.node_level = False


        self.dqn_agents = {}
        self.sac_agents = {}
        self.environments = {}

        print(f" Initializing Attack-Specific RL Agents...")
        print(f"   Attack Types: {len(self.attack_types)}")
        print(f"   Systems: {num_systems}")
        print(f"   Total Agents: {len(self.attack_types) * 2} ({len(self.attack_types)} DQN + {len(self.attack_types)} SAC)")

        for attack_type in self.attack_types:
            print(f"\n   Creating agents for: {attack_type}")


            dqn_env = DiscreteAttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems,
                node_level=self.node_level, network_layout=self.network_layout,
                n_nodes=n_nodes, node_seed=node_seed
            )


            sac_env = AttackSpecificEnvironment(
                federated_pinn_manager, attack_type, num_systems,
                node_level=self.node_level, network_layout=self.network_layout,
                n_nodes=n_nodes, node_seed=node_seed
            )


            self.environments[f'dqn_{attack_type}'] = dqn_env
            self.environments[f'sac_{attack_type}'] = sac_env


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

            print(f"       DQN agent created (action space: Discrete({num_systems}) - system selection)")
            print(f"       SAC agent created (action space: Continuous(4) - parameters only)")


        self.training_history = {}

        print(f"\n Attack-Specific Coordinator initialized successfully!")
        print(f"   Each agent specializes in ONE attack type across ALL systems")

    def train_attack_specialists(self, timesteps_per_attack: int = 10000):

        print(f"\n Phase 1: Individual Attack-Type Agent Training with PINN Models")
        print(f"   Timesteps per attack type: {timesteps_per_attack}")
        print(f"   Each agent will interact with ALL {self.num_systems} systems\n")

        for i, attack_type in enumerate(self.attack_types, 1):
            print(f" Training {attack_type} specialists ({i}/{len(self.attack_types)})...")
            print(f"   This agent will learn which of the {self.num_systems} systems are vulnerable to {attack_type}")


            print(f"    Training DQN agent for {attack_type}...")
            print(f"      Action space: Discrete({self.num_systems}) - selecting which system to attack")
            print(f"      Learning: 'Which systems (1-{self.num_systems}) are most vulnerable to {attack_type}?'")
            dqn_callback = EpisodeRewardCallback()
            self.dqn_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=dqn_callback
            )


            print(f"    Training SAC agent for {attack_type}...")
            print(f"       Action space: Continuous(4) - [magnitude, duration, stealth, target_system]")
            print(f"       Learning: 'What magnitude/duration/stealth works best for {attack_type}?'")
            sac_callback = EpisodeRewardCallback()
            self.sac_agents[attack_type].learn(
                total_timesteps=timesteps_per_attack // 2,
                callback=sac_callback
            )


            dqn_results = dqn_callback.get_results()
            sac_results = sac_callback.get_results()
            self.training_history[attack_type] = {
                'dqn': dqn_results,
                'sac': sac_results
            }

            print(f"    {attack_type} agents trained across all {self.num_systems} systems")
            print(f"      DQN: {dqn_results['num_episodes']} episodes, mean reward: {dqn_results['mean_reward']:.2f}")
            print(f"      SAC: {sac_results['num_episodes']} episodes, mean reward: {sac_results['mean_reward']:.2f}\n")

        print(f" Phase 1 Complete: All {len(self.attack_types)} attack specialists trained!")
        total_episodes = sum(
            self.training_history[at]['dqn']['num_episodes'] + self.training_history[at]['sac']['num_episodes']
            for at in self.training_history
        )
        print(f"   Total pre-training episodes captured: {total_episodes}")
        print(f"\nWhat each agent learned:")
        for attack_type in self.attack_types:
            print(f"   • {attack_type}: Which systems are vulnerable + optimal attack parameters")

    def save_agents(self, save_dir: str = "trained_rl_agents"):

        import os, json
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)

        saved_count = 0
        for attack_type in self.attack_types:

            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}")
            self.dqn_agents[attack_type].save(dqn_path)


            sac_path = os.path.join(save_dir, f"sac_{attack_type}")
            self.sac_agents[attack_type].save(sac_path)

            saved_count += 2
            print(f"   Saved DQN+SAC for {attack_type}")


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

        print(f" Saved {saved_count} agents to {save_dir}/")
        return save_dir

    def load_agents(self, save_dir: str = "trained_rl_agents") -> bool:

        import os, json, zipfile, shutil
        from datetime import datetime

        if not os.path.isdir(save_dir):
            print(f"   Agent directory not found: {save_dir}")
            return False


        current_obs_shape = self.environments[
            f'sac_{self.attack_types[0]}'
        ].observation_space.shape

        mismatch_detected = False
        sample_path = os.path.join(save_dir, f"sac_{self.attack_types[0]}.zip")
        if os.path.exists(sample_path):
            try:

                saved_agent = SAC.load(sample_path, env=None)
                saved_obs_shape = saved_agent.observation_space.shape
                if saved_obs_shape != current_obs_shape:
                    mismatch_detected = True
                    print(f"\n    OBSERVATION SPACE MISMATCH DETECTED")
                    print(f"     Saved models:      obs_shape={saved_obs_shape}")
                    print(f"     Current env:       obs_shape={current_obs_shape}")
                    print(f"     Cause: observation space was expanded from {saved_obs_shape[0]}"
                          f"  {current_obs_shape[0]} features (action feedback + cross-circle memory added)")
                    print(f"     Action: Archiving old models and retraining from scratch.")
                    print(f"     This is CORRECT — old models were trained with wrong reward signals anyway.")
                del saved_agent
            except Exception as e:
                print(f"   Could not inspect saved model: {e}")

        if mismatch_detected:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = f"{save_dir}_archived_{timestamp}"
            try:
                shutil.move(save_dir, archive_dir)
                print(f"   Old models archived to: {archive_dir}/")
            except Exception as e:
                print(f"   Could not archive old models: {e}")
            print(f"   Retraining from scratch with new {current_obs_shape[0]}-feature observation space.\n")
            return False

        loaded_count = 0
        for attack_type in self.attack_types:
            dqn_path = os.path.join(save_dir, f"dqn_{attack_type}.zip")
            sac_path = os.path.join(save_dir, f"sac_{attack_type}.zip")

            if not os.path.exists(dqn_path) or not os.path.exists(sac_path):
                print(f"   Missing agent files for {attack_type}")
                continue


            dqn_env = self.environments[f'dqn_{attack_type}']
            sac_env = self.environments[f'sac_{attack_type}']

            try:
                self.dqn_agents[attack_type] = DQN.load(dqn_path, env=dqn_env)
                self.sac_agents[attack_type] = SAC.load(sac_path, env=sac_env)
                loaded_count += 2
                print(f"   Loaded DQN+SAC for {attack_type}")
            except Exception as e:
                print(f"   Failed to load agents for {attack_type}: {e}")
                print(f"     Skipping — agent will be retrained from scratch.")


        meta_path = os.path.join(save_dir, "training_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"   Agents trained at: {metadata.get('timestamp', 'unknown')}")
            saved_obs = metadata.get('observation_space_shape')
            if saved_obs and tuple(saved_obs) != current_obs_shape:
                print(f"   Metadata obs shape {saved_obs} != current {current_obs_shape}")

        success = loaded_count == len(self.attack_types) * 2
        if success:
            print(f" Loaded {loaded_count} agents from {save_dir}/")
        else:
            print(f" Partially loaded {loaded_count}/{len(self.attack_types) * 2} agents from {save_dir}/")

        return success

    def get_agent_action(self, attack_type: str, agent_type: str, observation: np.ndarray):

        if agent_type == 'dqn':
            action, _ = self.dqn_agents[attack_type].predict(observation, deterministic=False)
            return action
        elif agent_type == 'sac':
            action, _ = self.sac_agents[attack_type].predict(observation, deterministic=False)
            return action
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _sac_q_value(self, sac_agent, obs, action):

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
                qs = policy.critic(obs_t, act_t)
                q = torch.min(torch.cat(list(qs), dim=1), dim=1).values
            return float(q.item())
        except Exception as e:
            print(f"      [value-gate] SAC critic unavailable: {e}")
            return None

    def _dqn_q_values(self, dqn_agent, obs):

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

        attack_type     = deployment.attack_type
        target_systems  = list(deployment.target_systems) if deployment.target_systems else []
        results         = []


        g_mag     = float(deployment.magnitude)
        g_dur     = float(deployment.duration)
        g_stealth = float(deployment.stealth_level)

        sac_agent = self.sac_agents.get(attack_type)
        dqn_agent = self.dqn_agents.get(attack_type)
        sac_env   = self.environments.get(f'sac_{attack_type}')
        dqn_env   = self.environments.get(f'dqn_{attack_type}')


        dqn_chosen_sys = None
        dqn_q = None
        if dqn_agent is not None and dqn_env is not None:
            try:
                dqn_obs, _ = dqn_env.reset()
                dqn_action, _ = dqn_agent.predict(dqn_obs, deterministic=False)
                dqn_chosen_sys = int(dqn_action) + 1
                dqn_q = self._dqn_q_values(dqn_agent, dqn_obs)
            except Exception as e:
                print(f"      ##?? DQN inference failed for {attack_type}: {e}")

        candidates = []
        if dqn_chosen_sys is not None:
            candidates.append(dqn_chosen_sys)
        for t in target_systems:
            ti = int(t)
            if 1 <= ti <= self.num_systems and ti not in candidates:
                candidates.append(ti)
        if not candidates:
            candidates = [int(np.random.randint(1, self.num_systems + 1))]


        if (dqn_q is not None and len(dqn_q) >= self.num_systems
                and len(candidates) > 1):
            final_sys = max(candidates, key=lambda s: float(dqn_q[s - 1]))
        else:
            final_sys = candidates[0]

        if final_sys == dqn_chosen_sys:
            system_source = "DQN"
        elif final_sys in target_systems:
            system_source = "Gemini(gated)"
        else:
            system_source = "fallback"
        gemini_sys_win = (final_sys != dqn_chosen_sys and final_sys in target_systems)

        if sac_env is None:
            print(f"      ##?? No SAC env for {attack_type}")
            return results


        sac_env.forced_target_system = [final_sys]
        sac_env.guidance_hints = {
            'magnitude': g_mag, 'duration': g_dur, 'stealth_level': g_stealth
        }
        obs, _ = sac_env.reset()

        param_source = "SAC"
        gemini_param_win = False
        q_sac_own = q_sac_gem = None

        if sac_agent is not None:
            trained_action, _ = sac_agent.predict(obs, deterministic=False)
            a_sac = trained_action.copy().astype(np.float32)
            a_sac[3] = float(final_sys - 1)
            a_gem = np.array([g_mag, g_dur, g_stealth, float(final_sys - 1)],
                             dtype=np.float32)

            q_sac_own = self._sac_q_value(sac_agent, obs, a_sac)
            q_sac_gem = self._sac_q_value(sac_agent, obs, a_gem)

            if q_sac_own is not None and q_sac_gem is not None:
                if q_sac_gem > q_sac_own:
                    action, param_source, gemini_param_win = a_gem, "Gemini(gated)", True
                else:
                    action = a_sac
            else:
                action = a_sac
                param_source = "SAC(no-critic)"

            _q = lambda v: "n/a" if v is None else round(v, 2)
            print(f"       gated deploy: mag={action[0]:.2f} dur={action[1]:.1f} "
                  f"stealth={action[2]:.2f}  sys {final_sys} "
                  f"[sys_src={system_source}, param_src={param_source}, "
                  f"Q_sac={_q(q_sac_own)}, Q_gem={_q(q_sac_gem)}]")
        else:
            print(f"      ##?? No trained SAC agent for {attack_type}, using Gemini hints")
            action = np.array([g_mag, g_dur, g_stealth, float(final_sys - 1)],
                              dtype=np.float32)
            param_source = "Gemini(no-sac)"

        _, reward, _, _, info = sac_env.step(action)


        sac_env.forced_target_system = None
        sac_env.guidance_hints = None


        gs = getattr(self, '_gate_stats', None)
        if gs is None:
            gs = {'sys_gemini_win': 0, 'sys_total': 0,
                  'param_gemini_win': 0, 'param_total': 0}
            self._gate_stats = gs
        gs['sys_total'] += 1
        gs['param_total'] += 1
        gs['sys_gemini_win'] += int(gemini_sys_win)
        gs['param_gemini_win'] += int(gemini_param_win)

        results.append({
            'attack_type': attack_type,
            'system_id':   final_sys,
            'reward':      reward,
            'dqn_chosen_system': dqn_chosen_sys,
            'gemini_targets':    target_systems,
            'system_source':     system_source,
            'param_source':      param_source,
            'gemini_system_win': gemini_sys_win,
            'gemini_param_win':  gemini_param_win,
            'q_sac_own':         q_sac_own,
            'q_sac_gemini':      q_sac_gem,
            'result':            info['attack_result'],
        })

        return results
