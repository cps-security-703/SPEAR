#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import copy
import os
import time
from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig, ACN_EVSE_VOLTAGE_V, ACN_MAX_PILOT_A, ACN_VOLTAGE_MAX_V, ACN_VOLTAGE_MIN_V, ACN_P_MAX_KW
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FederatedPINNConfig:


    num_distribution_systems: int = 10
    local_epochs: int = 200
    global_rounds: int = 20
    aggregation_method: str = 'fedavg'
    communication_rounds: int = 0


    local_batch_size: int = 32
    local_learning_rate: float = 0.001


    differential_privacy: bool = True
    noise_multiplier: float = 0.1
    max_grad_norm: float = 1.0


    share_frequency: int = 5

class AnomalyDetector:


    def __init__(self, config: LSTMPINNConfig, lstm_model_path: Optional[str] = None):
        self.config = config


        self.max_realistic_power = 100.0
        self.max_realistic_load_change = 50.0
        self.max_system_load = 500.0


        self.attack_detection_window = 10
        self.load_change_threshold = 25.0
        self.frequency_deviation_threshold = 0.5


        self.load_history = []
        self.power_history = []
        self.voltage_history = []


        self.lstm_detector = None
        self._robust_ids = None
        self._best_ids_detector = None
        self.lstm_enabled = False
        self.sequence_buffer = []
        self.sequence_length = 10


        if lstm_model_path and os.path.exists(lstm_model_path):
            self.load_lstm_model(lstm_model_path)

    def reset_state(self):

        self.load_history.clear()
        self.power_history.clear()
        self.voltage_history.clear()
        self.sequence_buffer.clear()


        if self._best_ids_detector is not None:
            self._best_ids_detector.reset()

    def validate_physical_constraints(self, inputs: Dict) -> Tuple[bool, Dict]:

        violations = {}
        is_valid = True

        def _check_and_sanitize(key, default, lo, hi, label):
            nonlocal is_valid
            raw = inputs.get(key, default)
            if not np.isfinite(raw):
                violations[key] = f"{label} is NaN/Inf"
                is_valid = False
                inputs[key] = default
                return
            if not (lo <= raw <= hi):
                violations[key] = f"{label} {raw:.4f} outside range [{lo}, {hi}]"
                is_valid = False
            inputs[key] = float(np.clip(raw, lo, hi))


        _check_and_sanitize('soc', 0.5, 0.0, 1.0, 'SOC')
        _check_and_sanitize('grid_voltage', 1.0, 0.80, 1.20, 'Grid voltage')
        _check_and_sanitize('grid_frequency', 60.0, 59.5, 60.5, 'Frequency')
        _check_and_sanitize('demand_factor', 0.7, 0.0, 2.0, 'Demand factor')
        _check_and_sanitize('load_factor', 0.7, 0.05, 1.8, 'Load factor')

        return is_valid, violations

    def detect_attack_patterns(self, current_load: float, system_id: int) -> Tuple[bool, str]:


        self.load_history.append((time.time(), current_load, system_id))


        current_time = time.time()
        self.load_history = [(t, load, sys_id) for t, load, sys_id in self.load_history
                           if current_time - t < 60.0]


        if len(self.load_history) >= 2:
            recent_loads = [load for _, load, sys_id in self.load_history[-5:] if sys_id == system_id]

            if len(recent_loads) >= 2:
                load_change = abs(recent_loads[-1] - recent_loads[-2])


                if current_load > self.max_system_load:
                    return True, f"Unrealistic load injection: {current_load:.1f} MW exceeds system capacity"


                if load_change > self.load_change_threshold:
                    return True, f"Suspicious load change: {load_change:.1f} kW in single step"


                if len(recent_loads) >= 4:
                    changes = [recent_loads[i+1] - recent_loads[i] for i in range(len(recent_loads)-1)]
                    if all(abs(change) > 10.0 for change in changes):
                        sign_changes = sum(1 for i in range(len(changes)-1)
                                         if changes[i] * changes[i+1] < 0)
                        if sign_changes >= 2:
                            return True, "Oscillating load pattern detected (potential attack)"

        return False, "Normal operation"

    def sanitize_inputs(self, inputs: Dict) -> Dict:

        sanitized = inputs.copy()


        sanitized['soc'] = np.clip(inputs.get('soc', 0.5), 0.05, 0.95)
        sanitized['grid_voltage'] = np.clip(inputs.get('grid_voltage', 1.0), 0.90, 1.10)
        sanitized['grid_frequency'] = np.clip(inputs.get('grid_frequency', 60.0), 59.5, 60.5)
        sanitized['demand_factor'] = np.clip(inputs.get('demand_factor', 0.7), 0.1, 1.5)
        sanitized['load_factor'] = np.clip(inputs.get('load_factor', 0.7), 0.2, 1.2)
        sanitized['urgency_factor'] = np.clip(inputs.get('urgency_factor', 1.0), 0.5, 2.0)

        return sanitized

    def load_lstm_model(self, model_path: str):


        if model_path.endswith('.pkl'):
            try:
                from best_ids_model import BestIDSDetector
                self._best_ids_detector = BestIDSDetector(model_path)
                if self._best_ids_detector.is_loaded:
                    self.lstm_enabled = True
                    print(f"##  sklearn IDS ({self._best_ids_detector.model_name}) loaded from: {model_path}")
                    return
                else:
                    print(f"##??  BestIDSDetector could not load {model_path}")
                    self._best_ids_detector = None
            except Exception as e:
                print(f"##??  Failed to load sklearn IDS from {model_path}: {e}")
            return

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)


            if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
                mtype = checkpoint['model_type']
                if mtype == 'classifier' and 'model_state_dict' in checkpoint:

                    try:
                        from robust_ids_evaluation import DNNClassifier, IDSModelWrapper
                        dnn = DNNClassifier(14, 10, 0.3)
                        dnn.load_state_dict(checkpoint['model_state_dict'])
                        self._robust_ids = IDSModelWrapper(
                            checkpoint.get('model_name', 'DNN'),
                            dnn, 'classifier',
                            threshold=checkpoint.get('threshold', 0.5))
                        self.lstm_enabled = True
                        print(f"## Robust IDS ({checkpoint.get('model_name','DNN')}) loaded from: {model_path}")
                        return
                    except Exception:
                        pass


            from lstm_anomaly_detector import LSTMIDSDetector
            self.lstm_detector = LSTMIDSDetector(
                input_size=14, hidden_size=128, num_layers=2,
                sequence_length=10, anomaly_threshold=0.5)
            self.lstm_detector.load_model(model_path)


            import json as _json, os as _os


            _this_dir  = _os.path.dirname(_os.path.abspath(__file__))
            _meta_path = _os.path.join(_this_dir, 'models', 'best_ids_model_meta.json')
            _loaded_thr = 0.5
            if _os.path.exists(_meta_path):
                try:
                    with open(_meta_path) as _mf:
                        _meta = _json.load(_mf)
                    _loaded_thr = float(_meta.get('threshold', 0.5))
                    print(f"##  Using Youden-J threshold from meta: {_loaded_thr:.3f}")
                except Exception:
                    pass
            self.lstm_detector.anomaly_threshold = _loaded_thr
            self.lstm_enabled = True
            self._robust_ids = None
            print(f"##  IDS loaded from: {model_path}  (threshold={_loaded_thr:.3f})")
        except Exception as e:
            print(f"##  Failed to load IDS model: {e}")
            self.lstm_enabled = False

    def extract_features(self, inputs: Dict) -> np.ndarray:

        freq = inputs.get('grid_frequency', 60.0)
        features = np.array([
            inputs.get('soc',           0.5),
            inputs.get('voltage',     240.0) / 240.0,
            inputs.get('current',      16.0) /  32.0,
            inputs.get('power',        3.84) /  7.68,
            inputs.get('temperature',  25.0) / 50.0,
            inputs.get('demand_factor', 0.7),
            inputs.get('load_factor',   0.7),
            inputs.get('grid_voltage',  1.0),
            freq / 60.0,
            inputs.get('queue_length',  3)   / 10.0,
            inputs.get('utilization',   0.6),
            inputs.get('urgency_factor',1.0),
            inputs.get('time_of_day',  12.0) / 24.0,
            inputs.get('system_id',     1)   / 10.0,
        ], dtype=np.float32)

        return features

    def detect_lstm_anomaly(self, inputs: Dict) -> Tuple[bool, float, str]:

        if not self.lstm_enabled:
            return False, 0.0, "ML IDS not enabled"

        features = self.extract_features(inputs)


        if self._best_ids_detector is not None and self._best_ids_detector.is_loaded:
            try:
                is_attack, confidence = self._best_ids_detector.detect(features)
                if is_attack:
                    return True, confidence, (
                        f"sklearn IDS ({self._best_ids_detector.model_name}) "
                        f"detected anomaly (score: {confidence:.3f})"
                    )
                return False, confidence, "Normal operation"
            except Exception as e:
                print(f"##  sklearn IDS detection error: {e}")


        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        if len(self.sequence_buffer) < self.sequence_length:
            return False, 0.0, "Insufficient sequence data"

        try:
            sequence = np.array(self.sequence_buffer)


            if hasattr(self, '_robust_ids') and self._robust_ids is not None:
                det, scores = self._robust_ids.detect_batch(sequence[np.newaxis])
                score = float(scores[0])
                detected = bool(det[0])
                if detected:
                    return True, score, f"ML IDS detected anomaly (score: {score:.3f})"
                return False, score, "Normal operation"


            if self.lstm_detector:
                is_anomaly, anomaly_score = self.lstm_detector.detect_anomaly(sequence)
                if is_anomaly:
                    return True, anomaly_score, f"LSTM detected anomaly (score: {anomaly_score:.3f})"
                return False, anomaly_score, "Normal operation"

            return False, 0.0, "No ML model available"
        except Exception as e:
            print(f"##  ML IDS detection error: {e}")
            return False, 0.0, f"ML IDS error: {str(e)}"

    def multi_layer_detection(self, inputs: Dict, system_id: int) -> Tuple[bool, Dict]:

        results = {
            'layer1_physical': {'detected': False, 'violations': {}},
            'layer2_pattern': {'detected': False, 'message': ''},
            'layer3_lstm': {'detected': False, 'score': 0.0, 'message': ''},
            'overall_detected': False,
            'detection_layer': None
        }

        first_detection_layer = None


        is_valid, violations = self.validate_physical_constraints(inputs)
        layer1_detected = not is_valid
        if layer1_detected:
            results['layer1_physical'] = {'detected': True, 'violations': violations}
            if first_detection_layer is None:
                first_detection_layer = 'physical_constraints'


        current_load = inputs.get('demand_factor', 0.7) * 100.0
        is_attack, attack_msg = self.detect_attack_patterns(current_load, system_id)
        layer2_detected = is_attack
        if layer2_detected:
            results['layer2_pattern'] = {'detected': True, 'message': attack_msg}
            if first_detection_layer is None:
                first_detection_layer = 'pattern_based'


        is_lstm_anomaly, lstm_score, lstm_msg = self.detect_lstm_anomaly(inputs)
        results['layer3_lstm'] = {'detected': is_lstm_anomaly, 'score': lstm_score, 'message': lstm_msg}


        if is_lstm_anomaly:
            if first_detection_layer is None:
                first_detection_layer = 'lstm_ml'


        if first_detection_layer is not None:
            results['overall_detected'] = True
            results['detection_layer'] = first_detection_layer
            return True, results


        return False, results

class GradualAttackController:


    def __init__(self, max_attack_magnitude: float = 50.0):
        self.max_attack_magnitude = max_attack_magnitude
        self.attack_step_size = 2.0
        self.attack_delay = 5.0


        self.current_attack_level = 0.0
        self.target_attack_level = 0.0
        self.last_attack_time = 0.0
        self.attack_active = False

    def start_gradual_attack(self, target_magnitude: float, attack_type: str = 'increase'):


        self.target_attack_level = np.clip(target_magnitude, 0.0, self.max_attack_magnitude)
        self.attack_active = True
        self.last_attack_time = time.time()

        if attack_type == 'decrease':
            self.target_attack_level = -self.target_attack_level

        print(f"## Starting gradual {attack_type} attack: target {self.target_attack_level:.1f} kW")

    def update_attack_level(self) -> float:

        if not self.attack_active:
            return 0.0

        current_time = time.time()


        if current_time - self.last_attack_time >= self.attack_delay:

            if abs(self.current_attack_level - self.target_attack_level) > self.attack_step_size:
                if self.current_attack_level < self.target_attack_level:
                    self.current_attack_level += self.attack_step_size
                else:
                    self.current_attack_level -= self.attack_step_size

                self.last_attack_time = current_time
            else:

                self.current_attack_level = self.target_attack_level
                self.attack_active = False
                print(f"## Gradual attack completed: {self.current_attack_level:.1f} kW")

        return self.current_attack_level

    def stop_attack(self):

        self.attack_active = False
        self.current_attack_level = 0.0
        self.target_attack_level = 0.0

class PINNCMSScheduler:


    MAX_PILOT_A  = 32.0
    EVSE_VOLTAGE = 240.0
    PERIOD_MIN   = 5
    SEQ_LEN      = 8
    ALPHA        = 0.65
    BETA         = 0.35

    def __init__(self, pinn_optimizer):

        self._opt     = pinn_optimizer
        self._device  = torch.device('cpu')
        self._buffers : Dict = {}

    def _extract_features(self, ev: Dict, n_active: int,
                          current_period: int) -> np.ndarray:

        soc          = float(np.clip(ev.get('soc', 0.5), 0.0, 1.0))
        rem_periods  = max(ev.get('departure_periods', 1), 1)
        dt_h         = self.PERIOD_MIN / 60.0
        max_del_kwh  = self.MAX_PILOT_A * self.EVSE_VOLTAGE / 1000.0 * rem_periods * dt_h
        dem          = float(np.clip(
            ev.get('remaining_kwh', 0.0) / max(max_del_kwh, 1e-6), 0.0, 2.0))
        urgency      = float(ev.get('urgency', np.clip(dem, 0.0, 1.0)))
        tod_norm     = float(ev.get('time_of_day', 12.0)) / 24.0
        load_f       = min(1.0, n_active / max(10, 1))
        return np.array([
            soc,
            1.0,
            float(np.clip(current_period * self.PERIOD_MIN / 60.0 / 24.0, 0, 1)),
            dem,
            max(0.0, 1.0 - dem),
            urgency,
            load_f,
            0.95,
            1.0,
            min(1.0, n_active / 10.0),
            load_f,
            urgency,
            tod_norm,
            0.1,
        ], dtype=np.float32)

    def schedule(self, fleet_state: List[Dict],
                 cap_kw: float,
                 n_evses: int = 10,
                 current_period: int = 0) -> Dict:

        if not fleet_state:
            return {}

        dt_h     = self.PERIOD_MIN / 60.0
        budget_A = cap_kw * 1000.0 / self.EVSE_VOLTAGE


        ev_info: Dict = {}
        for ev in fleet_state:
            eid         = ev['ev_id']
            rem_periods = max(ev.get('departure_periods', 1), 1)
            max_pilot   = ev.get('max_pilot_A', self.MAX_PILOT_A)
            rem_kwh     = ev.get('remaining_kwh', 0.0)

            min_pilot_A = min(max_pilot, max(0.0,
                rem_kwh / max(rem_periods * dt_h, 1e-6) * 1000.0 / self.EVSE_VOLTAGE
            ))
            lax_ratio   = max(0.0, 1.0 - min_pilot_A / max(max_pilot, 1e-6))
            urgency     = 1.0 - lax_ratio
            ev_info[eid] = {
                'ev':          ev,
                'urgency':     urgency,
                'min_pilot_A': min_pilot_A,
                'max_pilot_A': max_pilot,
                'pinn_w':      0.5,
            }


        if self._opt is not None and hasattr(self._opt, 'model'):
            n_active = len(fleet_state)
            self._opt.model.eval()
            with torch.no_grad():
                for ev in fleet_state:
                    eid  = ev['ev_id']
                    feat = self._extract_features(ev, n_active, current_period)
                    if eid not in self._buffers:
                        self._buffers[eid] = [feat.copy()] * self.SEQ_LEN
                    else:
                        self._buffers[eid].append(feat)
                        if len(self._buffers[eid]) > self.SEQ_LEN:
                            self._buffers[eid].pop(0)
                    seq = torch.FloatTensor(
                        np.array(self._buffers[eid])).unsqueeze(0)
                    try:
                        out   = self._opt.model(seq)
                        pinn_w = float(np.clip(float(out[0, 1]), 0.0, 1.0))
                    except Exception:
                        pinn_w = 0.5
                    ev_info[eid]['pinn_w'] = pinn_w


        for info in ev_info.values():
            info['priority'] = self.ALPHA * info['urgency'] + self.BETA * info['pinn_w']


        result = {ev['ev_id']: 0.0 for ev in fleet_state}
        rem_A  = budget_A
        evs_sorted = sorted(fleet_state,
                            key=lambda e: -ev_info[e['ev_id']]['priority'])
        for ev in evs_sorted:
            info  = ev_info[ev['ev_id']]
            min_A = info['min_pilot_A']
            if min_A > 0 and rem_A > 0:
                alloc = min(min_A, info['max_pilot_A'], rem_A)
                result[ev['ev_id']] = alloc
                rem_A -= alloc


        if rem_A > 0.5:
            weights = {}
            for ev in evs_sorted:
                info    = ev_info[ev['ev_id']]
                target  = ev.get('target_soc', 0.80)
                soc_gap = max(0.0, target - ev.get('soc', 0.0))
                weights[ev['ev_id']] = info['priority'] * soc_gap
            total_w = max(sum(weights.values()), 1e-6)
            for ev in evs_sorted:
                eid     = ev['ev_id']
                extra_A = min(
                    ev_info[eid]['max_pilot_A'] - result[eid],
                    rem_A * weights[eid] / total_w,
                )
                result[eid] += max(0.0, extra_A)


        total_A = sum(result.values())
        if total_A > budget_A + 0.01:
            scale = budget_A / total_A
            result = {k: v * scale for k, v in result.items()}
        result = {k: float(np.clip(v, 0.0, ev_info[k]['max_pilot_A']))
                  for k, v in result.items()}
        return result


class FederatedPINNManager:


    def __init__(self, config: FederatedPINNConfig):
        self.config = config
        self.pinn_config = LSTMPINNConfig()


        self.local_models: Dict[int, LSTMPINNChargingOptimizer] = {}
        self.global_model: Optional[LSTMPINNChargingOptimizer] = None


        self.anomaly_detectors: Dict[int, AnomalyDetector] = {}


        self.attack_controllers: Dict[int, GradualAttackController] = {}


        self.training_history = {
            'local_losses': {},
            'global_losses': [],
            'communication_rounds': 0
        }

        self._initialize_systems()

    def _initialize_systems(self):

        print(f" Initializing {self.config.num_distribution_systems} federated PINN systems...")

        for sys_id in range(1, self.config.num_distribution_systems + 1):

            local_config = copy.deepcopy(self.pinn_config)
            local_config.epochs = self.config.local_epochs
            local_config.learning_rate = self.config.local_learning_rate

            self.local_models[sys_id] = LSTMPINNChargingOptimizer(local_config, always_train=False)


            self.anomaly_detectors[sys_id] = AnomalyDetector(self.pinn_config)


            self.attack_controllers[sys_id] = GradualAttackController()


            self.training_history['local_losses'][sys_id] = []

            print(f"  ## System {sys_id}: Local PINN + Anomaly Detector + Attack Controller")


        self.global_model = LSTMPINNChargingOptimizer(self.pinn_config, always_train=False)
        print("## Global federated model initialized")

    def train_local_model(self, sys_id: int, local_data: Union[np.ndarray, Tuple],
                         n_samples: int = 1000) -> Dict:

        if sys_id not in self.local_models:
            raise ValueError(f"System {sys_id} not initialized")

        print(f" Training local PINN for Distribution System {sys_id}...")


        local_model = self.local_models[sys_id]


        if isinstance(local_data, tuple) and len(local_data) == 2:

            sequences, targets = local_data
            print(f"   Using ENHANCED training data: {len(sequences)} sequences with {sequences.shape[-1]} features")
            print(f"  ## Target ranges: V={targets[:, 0].min():.1f}-{targets[:, 0].max():.1f}, "
                  f"I={targets[:, 1].min():.1f}-{targets[:, 1].max():.1f}, "
                  f"P={targets[:, 2].min():.2f}-{targets[:, 2].max():.2f}")


            local_model._enhanced_training_data = (sequences, targets)


            if hasattr(local_model, 'trainer'):
                local_model.trainer._enhanced_training_data = (sequences, targets)
                print(f"   ACN sequences wired directly into LSTM-PINN trainer")

            training_metrics = local_model.train_model(n_samples=len(sequences))
        else:

            print(f"   Using simplified training data: {local_data.shape}")
            training_metrics = local_model.train_model(n_samples=n_samples)


        self.training_history['local_losses'][sys_id].append(training_metrics)


        model_path = f'federated_pinn_system_{sys_id}.pth'
        local_model.save_model(model_path)

        print(f"  ## System {sys_id} training completed, model saved to {model_path}")
        return training_metrics

    def federated_averaging(self) -> Dict:

        print("## Performing federated averaging...")

        if not self.local_models:
            raise ValueError("No local models to aggregate")


        local_state_dicts = []
        for sys_id, model in self.local_models.items():
            local_state_dicts.append(model.model.state_dict())


        global_state_dict = {}

        for key in local_state_dicts[0].keys():

            tensors = [state_dict[key] for state_dict in local_state_dicts]


            if all(t.shape == tensors[0].shape for t in tensors):

                if tensors[0].dtype in [torch.long, torch.int, torch.int32, torch.int64]:

                    global_state_dict[key] = tensors[0].clone()
                else:

                    if self.config.aggregation_method == 'fedavg':

                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = torch.stack(float_tensors).mean(dim=0)

                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)

                    elif self.config.aggregation_method == 'weighted_avg':

                        weights = torch.ones(len(local_state_dicts)) / len(local_state_dicts)
                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = sum(
                            w * tensor for w, tensor in zip(weights, float_tensors)
                        )

                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)

                    elif self.config.aggregation_method == 'median':
                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = torch.median(torch.stack(float_tensors), dim=0)[0]

                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)
            else:

                global_state_dict[key] = tensors[0].clone()


        self.global_model.model.load_state_dict(global_state_dict)


        self.config.communication_rounds += 1

        print(f"  ## Federated averaging completed (Round {self.config.communication_rounds})")

        return {
            'round': self.config.communication_rounds,
            'aggregation_method': self.config.aggregation_method,
            'num_participants': len(local_state_dicts)
        }

    def distribute_global_model(self):

        print("##Distributing global model to local systems...")

        global_state_dict = self.global_model.model.state_dict()

        for sys_id, local_model in self.local_models.items():
            local_model.model.load_state_dict(copy.deepcopy(global_state_dict))
            print(f"  ## System {sys_id}: Global model distributed")

    def optimize_with_constraints(self, sys_id: int, inputs: Dict,
                                   fleet_state: Optional[List[Dict]] = None,
                                   cap_kw: float = 76.8) -> Tuple[Dict, bool, str]:

        if sys_id not in self.local_models:
            raise ValueError(f"System {sys_id} not initialized")


        detector = self.anomaly_detectors[sys_id]
        is_detected, detection_results = detector.multi_layer_detection(inputs, sys_id)

        lstm_advisory_flag = False
        if is_detected:
            layer = detection_results['detection_layer']
            if layer == 'physical_constraints':
                violations = detection_results['layer1_physical']['violations']
                msg = f"Physical violations: {'; '.join(violations.values())}"
                return {}, False, f"[{layer}] {msg}"
            elif layer == 'pattern_based':
                msg = detection_results['layer2_pattern']['message']
                return {}, False, f"[{layer}] {msg}"
            elif layer == 'lstm_ml':
                score = detection_results['layer3_lstm']['score']
                lstm_advisory_flag = True


        sanitized_inputs = detector.sanitize_inputs(inputs)


        attack_controller = self.attack_controllers[sys_id]
        attack_level = attack_controller.update_attack_level()
        if attack_level != 0.0:
            sanitized_inputs['demand_factor'] += attack_level / 100.0
            sanitized_inputs['demand_factor'] = np.clip(sanitized_inputs['demand_factor'], 0.1, 1.5)


        local_model = self.local_models[sys_id]
        try:
            v_ref, i_ref, p_ref = local_model.optimize_references(sanitized_inputs)
            v_ref = float(np.clip(v_ref, ACN_VOLTAGE_MIN_V, ACN_VOLTAGE_MAX_V))
            i_ref = float(np.clip(i_ref, 0.0, ACN_MAX_PILOT_A))
            p_ref = float(np.clip(p_ref, 0.0, ACN_P_MAX_KW))

            results = {
                'voltage_ref':  v_ref,
                'current_ref':  i_ref,
                'power_ref':    p_ref,
                'system_id':    sys_id,
                'attack_level': attack_level,
                'sanitized':    sanitized_inputs != inputs,
                'lstm_advisory':lstm_advisory_flag,
            }


            if fleet_state:
                scheduler = PINNCMSScheduler(local_model)
                pilot_signals = scheduler.schedule(fleet_state, cap_kw)
                results['pilot_signals'] = pilot_signals

            msg = "Optimization successful"
            if lstm_advisory_flag:
                msg += " (IDS-ADVISORY: Anomaly detected but non-blocking)"
            return results, True, msg

        except Exception as e:
            return {}, False, f"Optimization failed: {str(e)}"


    def start_coordinated_attack(self, target_systems: List[int], attack_magnitude: float,
                                attack_type: str = 'increase'):

        print(f"## Starting coordinated {attack_type} attack on systems {target_systems}")
        print(f"   Target magnitude: {attack_magnitude:.1f} kW per system")

        for sys_id in target_systems:
            if sys_id in self.attack_controllers:
                self.attack_controllers[sys_id].start_gradual_attack(attack_magnitude, attack_type)

    def stop_all_attacks(self):

        print("## Stopping all active attacks...")
        for sys_id, controller in self.attack_controllers.items():
            controller.stop_attack()
        print("## All attacks stopped")

    def get_federated_status(self) -> Dict:

        status = {
            'num_systems': len(self.local_models),
            'communication_rounds': self.config.communication_rounds,
            'active_attacks': {},
            'anomaly_detections': {},
            'training_status': {}
        }

        for sys_id in self.local_models.keys():

            controller = self.attack_controllers[sys_id]
            status['active_attacks'][sys_id] = {
                'active': controller.attack_active,
                'current_level': controller.current_attack_level,
                'target_level': controller.target_attack_level
            }


            if sys_id in self.training_history['local_losses']:
                losses = self.training_history['local_losses'][sys_id]
                status['training_status'][sys_id] = {
                    'training_rounds': len(losses),
                    'last_loss': losses[-1] if losses else None
                }

        return status

    def save_federated_models(self, base_path: str = 'federated_models'):

        print(f"## Saving federated models to {base_path}/...")


        import os
        os.makedirs(base_path, exist_ok=True)


        if self.global_model:
            self.global_model.save_model(f'{base_path}/global_federated_pinn.pth')


        for sys_id, model in self.local_models.items():
            model.save_model(f'{base_path}/local_pinn_system_{sys_id}.pth')

        print("## All federated models saved")

    def load_federated_models(self, base_path: str = 'federated_models'):

        print(f" Loading federated models from {base_path}/...")

        try:

            if self.global_model:
                self.global_model.load_model(f'{base_path}/global_federated_pinn.pth')
                print("  ## Global model loaded")


            for sys_id, model in self.local_models.items():
                model.load_model(f'{base_path}/local_pinn_system_{sys_id}.pth')
                print(f"  ## System {sys_id} model loaded")

            print("## All federated models loaded successfully")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"## Failed to load federated models: {e}")
            return False
