#!/usr/bin/env python3


import numpy as np
import json
import os
from datetime import datetime
from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig


def run_baseline_attacks_actual_system(num_episodes=30, num_systems=10):

    print(" Baseline Attack Evaluation - ACTUAL SYSTEM")
    print("="*90)
    print(" Running random attacks through the same system as RL")
    print("="*90)


    print("\n Initializing Federated PINN Manager...")
    config = FederatedPINNConfig(num_distribution_systems=num_systems)
    federated_manager = FederatedPINNManager(config)


    import os as _os
    _lstm_paths = ['models/lstm_ids_pretrained.pth', 'lstm_ids_best_balanced.pth']
    _lstm_loaded = False
    for _lp in _lstm_paths:
        if _os.path.exists(_lp):
            for _sid, _det in federated_manager.anomaly_detectors.items():
                _det.load_lstm_model(_lp)
            print(f"  LSTM IDS loaded from {_lp} into all {len(federated_manager.anomaly_detectors)} anomaly detectors")
            _lstm_loaded = True
            break
    if not _lstm_loaded:
        print("   No pre-trained LSTM IDS model found — Layer 3 detection disabled")


    np.random.seed(42)

    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'scenario_id': 'BASELINE_RANDOM_ACTUAL_SYSTEM',
        'episodes': num_episodes,
        'performance_metrics': {},
        'episode_results': []
    }

    detection_rates = []
    success_rates = []
    anomaly_scores = []


    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}")

        episode_data = {
            'episode': episode + 1,
            'detection_rate': 0.0,
            'success_rate': 0.0,
            'attacks_detected': []
        }


        for sys_id in range(1, num_systems + 1):
            if sys_id not in federated_manager.local_models:
                continue

            local_model = federated_manager.local_models[sys_id]


            attack_type_idx = np.random.randint(0, 6)
            attack_types = [
                'voltage_manipulation',
                'current_injection',
                'power_disruption',
                'frequency_attack',
                'soc_spoofing',
                'thermal_attack'
            ]
            attack_type = attack_types[attack_type_idx]


            magnitude = np.random.uniform(0.9, 1.5)
            duration = np.random.uniform(30.0, 120.0)
            stealth_factor = np.random.uniform(0.05, 0.3)


            attack_params = {
                'type': attack_type,
                'magnitude': magnitude,
                'duration': duration,
                'stealth_factor': stealth_factor
            }


            attack_result = execute_attack_on_system(
                local_model,
                attack_params,
                sys_id,
                federated_manager
            )

            episode_data['attacks_detected'].append(attack_result)


            if attack_result['detected']:
                total_tp += 1
            else:
                total_fn += 1


        for sys_id in range(1, num_systems + 1):
            if sys_id not in federated_manager.anomaly_detectors:
                continue
            det = federated_manager.anomaly_detectors[sys_id]
            det.reset_state()


            _benign_soc = np.random.uniform(0.3, 0.7)
            _benign_v = np.random.uniform(370, 430)
            _benign_i = np.random.uniform(60, 100)
            _benign_p = np.random.uniform(20, 40)
            _benign_temp = 25.0 + max(0, _benign_p - 20.0) * 0.3
            _benign_lf = np.clip(_benign_p / 50.0, 0.2, 1.3)
            _benign_util = np.clip(_benign_p / 75.0, 0.1, 1.0)
            _benign_urg = 1.0 + max(0, 1.0 - _benign_soc) * 0.5
            import time as _time
            _benign_tod = (_time.time() / 3600.0) % 24.0
            for _ in range(det.sequence_length):
                warmup_input = {
                    'soc': float(_benign_soc + np.random.uniform(-0.03, 0.03)),
                    'voltage': float(_benign_v + np.random.uniform(-5.0, 5.0)),
                    'current': float(_benign_i + np.random.uniform(-3.0, 3.0)),
                    'power': float(_benign_p + np.random.uniform(-2.0, 2.0)),
                    'temperature': float(np.clip(_benign_temp + np.random.uniform(-1.0, 1.0), 20.0, 45.0)),
                    'demand_factor': float(np.clip(0.65 + np.random.uniform(-0.05, 0.05), 0.3, 1.2)),
                    'load_factor': float(np.clip(_benign_lf + np.random.uniform(-0.05, 0.05), 0.2, 1.3)),
                    'grid_voltage': float(1.0 + np.random.uniform(-0.01, 0.01)),
                    'grid_frequency': float(60.0 + np.random.uniform(-0.02, 0.02)),
                    'queue_length': int(np.clip(3 + np.random.randint(-1, 2), 0, 10)),
                    'utilization': float(np.clip(_benign_util + np.random.uniform(-0.05, 0.05), 0.1, 1.0)),
                    'urgency_factor': float(np.clip(_benign_urg + np.random.uniform(-0.05, 0.05), 0.5, 2.0)),
                    'time_of_day': float(_benign_tod + np.random.uniform(-0.5, 0.5)),
                    'system_id': sys_id
                }
                det.multi_layer_detection(warmup_input, sys_id)


            benign_input = {
                'soc': float(_benign_soc + np.random.uniform(-0.03, 0.03)),
                'voltage': float(_benign_v + np.random.uniform(-5.0, 5.0)),
                'current': float(_benign_i + np.random.uniform(-3.0, 3.0)),
                'power': float(_benign_p + np.random.uniform(-2.0, 2.0)),
                'temperature': float(np.clip(_benign_temp + np.random.uniform(-1.0, 1.0), 20.0, 45.0)),
                'demand_factor': float(np.clip(0.65 + np.random.uniform(-0.05, 0.05), 0.3, 1.2)),
                'load_factor': float(np.clip(_benign_lf + np.random.uniform(-0.05, 0.05), 0.2, 1.3)),
                'grid_voltage': float(1.0 + np.random.uniform(-0.01, 0.01)),
                'grid_frequency': float(60.0 + np.random.uniform(-0.02, 0.02)),
                'queue_length': int(np.clip(3 + np.random.randint(-1, 2), 0, 10)),
                'utilization': float(np.clip(_benign_util + np.random.uniform(-0.05, 0.05), 0.1, 1.0)),
                'urgency_factor': float(np.clip(_benign_urg + np.random.uniform(-0.05, 0.05), 0.5, 2.0)),
                'time_of_day': float(_benign_tod + np.random.uniform(-0.5, 0.5)),
                'system_id': sys_id
            }
            is_det, det_res = det.multi_layer_detection(benign_input, sys_id)
            if is_det:
                total_fp += 1
            else:
                total_tn += 1


        if episode_data['attacks_detected']:
            episode_data['detection_rate'] = np.mean([
                a['detected'] for a in episode_data['attacks_detected']
            ])
            episode_data['success_rate'] = np.mean([
                a['success'] for a in episode_data['attacks_detected']
            ])
            episode_anomaly = np.mean([
                a['anomaly_score'] for a in episode_data['attacks_detected']
            ])
            anomaly_scores.append(episode_anomaly)
        else:
            episode_data['detection_rate'] = 0.0
            episode_data['success_rate'] = 0.0

        detection_rates.append(episode_data['detection_rate'])
        success_rates.append(episode_data['success_rate'])

        results['episode_results'].append(episode_data)

        if (episode + 1) % 5 == 0:
            print(f"    Detection Rate: {episode_data['detection_rate']:.1%}, "
                  f"Success Rate: {episode_data['success_rate']:.1%}")


    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (total_tp + total_tn) / max(total_tp + total_fp + total_fn + total_tn, 1)

    results['performance_metrics'] = {
        'average_detection_rate': float(np.mean(detection_rates)),
        'average_success_rate': float(np.mean(success_rates)),
        'average_anomaly_score': float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
        'average_reward': 0.0,
        'coordination_effectiveness': 0.0,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy),
        'confusion_matrix': {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': total_tn}
    }

    print("\n" + "="*90)
    print(" BASELINE EVALUATION COMPLETE")
    print("="*90)
    print(f"   Detection Rate: {results['performance_metrics']['average_detection_rate']:.1%}")
    print(f"   Success Rate: {results['performance_metrics']['average_success_rate']:.1%}")
    print(f"   Avg Anomaly Score: {results['performance_metrics']['average_anomaly_score']:.3f}")
    print(f"   Precision:  {precision:.1%}")
    print(f"   Recall:     {recall:.1%}")
    print(f"   F1-Score:   {f1_score:.4f}")
    print(f"   Accuracy:   {accuracy:.1%}")
    print(f"   Confusion Matrix: TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
    print("="*90)


    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"baseline_actual_system_{results['timestamp']}.json")


    serializable_results = convert_to_json_serializable(results)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n Baseline results saved to: {output_file}")

    return serializable_results


def convert_to_json_serializable(obj):

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def execute_attack_on_system(pinn_model, attack_params, sys_id, federated_manager):

    attack_result = {
        'system_id': sys_id,
        'attack_type': attack_params['type'],
        'magnitude': float(attack_params['magnitude']),
        'stealth_factor': float(attack_params['stealth_factor']),
        'success': False,
        'impact': 0.0,
        'detected': False,
        'anomaly_score': 0.0
    }

    try:

        impact_multipliers = {
            'voltage_manipulation': 0.8,
            'current_injection': 0.7,
            'power_disruption': 0.9,
            'frequency_attack': 0.6,
            'soc_spoofing': 0.5,
            'thermal_attack': 0.4,
            'none': 0.0
        }

        base_impact = impact_multipliers.get(attack_params['type'], 0.5)
        magnitude_factor = np.clip(attack_params['magnitude'], 0.0, 1.5)
        impact = base_impact * magnitude_factor

        attack_result['impact'] = float(impact)
        attack_result['success'] = impact > 0.3


        magnitude = float(attack_params['magnitude'])
        stealth = float(attack_params['stealth_factor'])
        attack_type = attack_params['type']


        voltage_dev = magnitude * 0.15 if 'voltage' in attack_type else magnitude * 0.05
        current_dev = magnitude * 0.20 if 'current' in attack_type else magnitude * 0.08
        power_dev = magnitude * 0.25 if 'power' in attack_type else magnitude * 0.10

        _attacked_v = float(400.0 * (1.0 - voltage_dev))
        _attacked_i = float(50.0 * (1.0 + current_dev))
        _attacked_p = float(20.0 * (1.0 + power_dev))
        _soc = float(np.clip(0.5 - magnitude * 0.2, 0.05, 0.95))
        _temperature = float(np.clip(25.0 + max(0, _attacked_p - 20.0) * 0.3, 20.0, 45.0))
        _load_factor = float(np.clip(_attacked_p / 50.0, 0.2, 1.3))
        _grid_voltage = float(np.clip(1.0 - voltage_dev, 0.85, 1.15))
        _grid_frequency = float(np.clip(60.0 + (magnitude * 2.0 if 'frequency' in attack_type else 0.0), 59.0, 61.0))
        _queue_length = int(np.clip(3 + magnitude * 3, 0, 10))
        _utilization = float(np.clip(_attacked_p / 75.0, 0.1, 1.0))
        _urgency = float(np.clip(1.0 + max(0, 1.0 - _soc) * 0.5, 0.5, 2.0))
        import time as _time
        _time_of_day = float((_time.time() / 3600.0) % 24.0)

        ids_input = {
            'soc': _soc,
            'voltage': _attacked_v,
            'current': _attacked_i,
            'power': _attacked_p,
            'temperature': _temperature,
            'demand_factor': float(np.clip(0.7 + magnitude * 0.8, 0.1, 1.5)),
            'load_factor': _load_factor,
            'grid_voltage': _grid_voltage,
            'grid_frequency': _grid_frequency,
            'queue_length': _queue_length,
            'utilization': _utilization,
            'urgency_factor': _urgency,
            'time_of_day': _time_of_day,
            'system_id': sys_id
        }


        if sys_id in federated_manager.anomaly_detectors:
            detector = federated_manager.anomaly_detectors[sys_id]
            is_detected, detection_results = detector.multi_layer_detection(ids_input, sys_id)
            lstm_score = detection_results.get('layer3_lstm', {}).get('score', 0.0)

            attack_result['detected'] = is_detected
            attack_result['anomaly_score'] = float(lstm_score)
            attack_result['detection_layer'] = detection_results.get('detection_layer', None)
        else:

            attack_result['detected'] = False
            attack_result['anomaly_score'] = 0.0

    except Exception as e:
        attack_result['error'] = str(e)

    return attack_result


def main():

    print(" Baseline Attack Evaluation on ACTUAL System")
    print("="*90)
    print("This runs random attacks through the same system infrastructure")
    print("as enhanced_integrated_evcs_system.py (without RL coordination)")
    print("="*90)

    results = run_baseline_attacks_actual_system(num_episodes=30, num_systems=10)

    print("\n" + "="*90)
    print(" COMPLETE!")
    print("="*90)
    print(" Output File:")
    print(f"   • detection_results/baseline_actual_system_{results['timestamp']}.json")
    print("\n Next step: Use compare_rl_vs_baseline_actual.py to compare")
    print("   this baseline with your RL results")
    print("="*90)


if __name__ == "__main__":
    main()
