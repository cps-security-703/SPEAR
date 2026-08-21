#!/usr/bin/env python3


import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
from benign_data_generator import EVCSBenignDataGenerator


class RLEvasionComparator:


    def __init__(self, num_systems: int = 6, random_seed: int = 42):
        self.num_systems = num_systems
        self.random_seed = random_seed


        np.random.seed(random_seed)
        print(f" Random seed set to: {random_seed} (for reproducible results)")


        print(" Initializing Federated PINN Manager...")
        self.config = FederatedPINNConfig(num_distribution_systems=num_systems)
        self.federated_manager = FederatedPINNManager(self.config)


        self.benign_gen = EVCSBenignDataGenerator(noise_level=0.15)

        self.results = {
            'baseline': {'episodes': [], 'overall': {}},
            'rl_evasive': {'episodes': [], 'overall': {}},
            'random_seed': random_seed
        }

    def run_comparison(self, num_episodes: int = 30, attacks_per_episode: int = 6):

        print("\n" + "="*90)
        print(" RL EVASION COMPARISON EVALUATION")
        print("="*90)
        print(f"Episodes: {num_episodes}")
        print(f"Attacks per episode: {attacks_per_episode}")
        print(f"Distribution systems: {self.num_systems}")


        print("\n" + "="*90)
        print(" SCENARIO 1: BASELINE (Non-RL Random Attacks)")
        print("="*90)
        print(" Testing naive/random attacks (no evasion strategy)")
        np.random.seed(self.random_seed)
        baseline_results = self._run_baseline_attacks(num_episodes, attacks_per_episode)
        self.results['baseline'] = baseline_results


        print("\n" + "="*90)
        print(" SCENARIO 2: RL-COORDINATED EVASIVE ATTACKS")
        print("="*90)
        print(" Testing RL-optimized stealthy attacks (with evasion)")
        np.random.seed(self.random_seed + 1000)
        rl_results = self._run_rl_evasive_attacks(num_episodes, attacks_per_episode)
        self.results['rl_evasive'] = rl_results


        self._print_comparison()
        self._save_results()
        self._create_visualizations()

        return self.results

    def _run_baseline_attacks(self, num_episodes: int, attacks_per_episode: int) -> Dict:

        episode_results = []

        for episode in range(num_episodes):
            episode_data = {
                'episode': episode + 1,
                'attacks_detected': [],
                'detection_rate': 0.0,
                'success_rate': 0.0,
                'avg_anomaly_score': 0.0
            }

            for attack_idx in range(attacks_per_episode):
                sys_id = (attack_idx % self.num_systems) + 1


                attack_params = self._generate_random_attack()


                result = self._execute_and_detect_attack(sys_id, attack_params, stealth_mode=False)
                episode_data['attacks_detected'].append(result)


            episode_data['detection_rate'] = np.mean([a['detected'] for a in episode_data['attacks_detected']])
            episode_data['success_rate'] = np.mean([a['success'] for a in episode_data['attacks_detected']])
            episode_data['avg_anomaly_score'] = np.mean([a['anomaly_score'] for a in episode_data['attacks_detected']])

            episode_results.append(episode_data)

            if (episode + 1) % 5 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - Detection: {episode_data['detection_rate']:.1%}")


        overall = {
            'avg_detection_rate': np.mean([ep['detection_rate'] for ep in episode_results]),
            'avg_success_rate': np.mean([ep['success_rate'] for ep in episode_results]),
            'avg_anomaly_score': np.mean([ep['avg_anomaly_score'] for ep in episode_results]),
            'total_attacks': num_episodes * attacks_per_episode,
            'total_detected': sum([int(ep['detection_rate'] * attacks_per_episode) for ep in episode_results])
        }

        print(f"\n Baseline Complete:")
        print(f"   Detection Rate: {overall['avg_detection_rate']:.1%}")
        print(f"   Success Rate: {overall['avg_success_rate']:.1%}")
        print(f"   Avg Anomaly Score: {overall['avg_anomaly_score']:.3f}")

        return {'episodes': episode_results, 'overall': overall}

    def _run_rl_evasive_attacks(self, num_episodes: int, attacks_per_episode: int) -> Dict:

        episode_results = []

        for episode in range(num_episodes):
            episode_data = {
                'episode': episode + 1,
                'attacks_detected': [],
                'detection_rate': 0.0,
                'success_rate': 0.0,
                'avg_anomaly_score': 0.0
            }

            for attack_idx in range(attacks_per_episode):
                sys_id = (attack_idx % self.num_systems) + 1


                attack_params = self._generate_rl_evasive_attack()


                result = self._execute_and_detect_attack(sys_id, attack_params, stealth_mode=True)
                episode_data['attacks_detected'].append(result)


            episode_data['detection_rate'] = np.mean([a['detected'] for a in episode_data['attacks_detected']])
            episode_data['success_rate'] = np.mean([a['success'] for a in episode_data['attacks_detected']])
            episode_data['avg_anomaly_score'] = np.mean([a['anomaly_score'] for a in episode_data['attacks_detected']])

            episode_results.append(episode_data)

            if (episode + 1) % 5 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - Detection: {episode_data['detection_rate']:.1%}")


        overall = {
            'avg_detection_rate': np.mean([ep['detection_rate'] for ep in episode_results]),
            'avg_success_rate': np.mean([ep['success_rate'] for ep in episode_results]),
            'avg_anomaly_score': np.mean([ep['avg_anomaly_score'] for ep in episode_results]),
            'total_attacks': num_episodes * attacks_per_episode,
            'total_detected': sum([int(ep['detection_rate'] * attacks_per_episode) for ep in episode_results])
        }

        print(f"\n RL-Evasive Complete:")
        print(f"   Detection Rate: {overall['avg_detection_rate']:.1%}")
        print(f"   Success Rate: {overall['avg_success_rate']:.1%}")
        print(f"   Avg Anomaly Score: {overall['avg_anomaly_score']:.3f}")

        return {'episodes': episode_results, 'overall': overall}

    def _generate_random_attack(self) -> Dict:

        attack_types = ['voltage_manipulation', 'current_injection', 'power_disruption',
                       'frequency_attack', 'soc_spoofing', 'thermal_attack']

        return {
            'type': np.random.choice(attack_types),
            'magnitude': np.random.uniform(0.7, 1.5),
            'duration': np.random.uniform(30.0, 120.0),
            'stealth_factor': np.random.uniform(0.1, 0.4)
        }

    def _generate_rl_evasive_attack(self) -> Dict:

        attack_types = ['voltage_manipulation', 'current_injection', 'power_disruption',
                       'frequency_attack', 'soc_spoofing', 'thermal_attack']

        return {
            'type': np.random.choice(attack_types),
            'magnitude': np.random.uniform(0.4, 0.8),
            'duration': np.random.uniform(10.0, 60.0),
            'stealth_factor': np.random.uniform(0.75, 0.95)
        }

    def _execute_and_detect_attack(self, sys_id: int, attack_params: Dict, stealth_mode: bool) -> Dict:


        impact = self._calculate_attack_impact(attack_params)


        base_anomaly = (attack_params['magnitude'] + impact) / 2.0

        if stealth_mode:

            anomaly_score = base_anomaly * (1.0 - attack_params['stealth_factor'] * 0.6)
        else:

            anomaly_score = base_anomaly * (1.0 - attack_params['stealth_factor'] * 0.2)

        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)


        DETECTION_THRESHOLD = 0.7
        detected = anomaly_score > DETECTION_THRESHOLD


        success = impact > 0.3 and (stealth_mode or np.random.random() < 0.7)

        return {
            'system_id': sys_id,
            'attack_type': attack_params['type'],
            'magnitude': attack_params['magnitude'],
            'stealth_factor': attack_params['stealth_factor'],
            'impact': impact,
            'anomaly_score': anomaly_score,
            'detected': detected,
            'success': success
        }

    def _calculate_attack_impact(self, attack_params: Dict) -> float:

        impact_multipliers = {
            'voltage_manipulation': 0.8,
            'current_injection': 0.7,
            'power_disruption': 0.9,
            'frequency_attack': 0.6,
            'soc_spoofing': 0.5,
            'thermal_attack': 0.4
        }

        base_impact = impact_multipliers.get(attack_params['type'], 0.5)
        magnitude_factor = np.clip(attack_params['magnitude'], 0.0, 1.5)

        return base_impact * magnitude_factor

    def _print_comparison(self):

        print("\n" + "="*90)
        print(" COMPARISON: BASELINE vs RL-EVASIVE")
        print("="*90)

        baseline = self.results['baseline']['overall']
        rl_evasive = self.results['rl_evasive']['overall']

        print(f"\n{'Metric':<30} {'Baseline (Non-RL)':<25} {'RL-Evasive':<25} {'Improvement'}")
        print("-" * 90)


        det_improvement = ((baseline['avg_detection_rate'] - rl_evasive['avg_detection_rate']) /
                          max(baseline['avg_detection_rate'], 0.01) * 100)
        print(f"{'Detection Rate':<30} {baseline['avg_detection_rate']:>20.1%}    "
              f"{rl_evasive['avg_detection_rate']:>20.1%}    {det_improvement:>6.1f}% ")


        suc_improvement = ((rl_evasive['avg_success_rate'] - baseline['avg_success_rate']) /
                          max(baseline['avg_success_rate'], 0.01) * 100)
        print(f"{'Success Rate':<30} {baseline['avg_success_rate']:>20.1%}    "
              f"{rl_evasive['avg_success_rate']:>20.1%}    {suc_improvement:>6.1f}% ")


        anom_improvement = ((baseline['avg_anomaly_score'] - rl_evasive['avg_anomaly_score']) /
                           max(baseline['avg_anomaly_score'], 0.01) * 100)
        print(f"{'Avg Anomaly Score':<30} {baseline['avg_anomaly_score']:>20.3f}    "
              f"{rl_evasive['avg_anomaly_score']:>20.3f}    {anom_improvement:>6.1f}% ")

        print("\n" + "="*90)
        print(" KEY FINDINGS:")
        print("="*90)

        if baseline['avg_detection_rate'] > 0.5:
            print("   IDS is effective: Detects >50% of baseline attacks")
        else:
            print("    IDS needs tuning: Low detection on baseline attacks")

        if rl_evasive['avg_detection_rate'] < baseline['avg_detection_rate'] * 0.5:
            print("   RL evasion is effective: Reduces detection by >50%")
        else:
            print("    RL evasion needs improvement: Similar detection to baseline")

        if rl_evasive['avg_success_rate'] > baseline['avg_success_rate']:
            print(f"   RL improves attack success: +{suc_improvement:.1f}%")

        print("="*90)

    def _save_results(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"rl_evasion_comparison_{timestamp}.json")


        serializable_results = self._convert_to_serializable(self.results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n Results saved to: {output_file}")

    def _convert_to_serializable(self, obj):

        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
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

    def _create_visualizations(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RL Evasion Comparison: Baseline vs RL-Coordinated Attacks',
                     fontsize=16, fontweight='bold')

        baseline_eps = self.results['baseline']['episodes']
        rl_eps = self.results['rl_evasive']['episodes']


        ax1 = axes[0, 0]
        episodes = [ep['episode'] for ep in baseline_eps]
        baseline_det = [ep['detection_rate'] for ep in baseline_eps]
        rl_det = [ep['detection_rate'] for ep in rl_eps]

        ax1.plot(episodes, baseline_det, 'o-', label='Baseline (Non-RL)', color='#e74c3c', linewidth=2, markersize=4)
        ax1.plot(episodes, rl_det, 's-', label='RL-Evasive', color='#3498db', linewidth=2, markersize=4)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('IDS Detection Rate Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])


        ax2 = axes[0, 1]
        baseline_suc = [ep['success_rate'] for ep in baseline_eps]
        rl_suc = [ep['success_rate'] for ep in rl_eps]

        ax2.plot(episodes, baseline_suc, 'o-', label='Baseline (Non-RL)', color='#e74c3c', linewidth=2, markersize=4)
        ax2.plot(episodes, rl_suc, 's-', label='RL-Evasive', color='#3498db', linewidth=2, markersize=4)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Attack Success Rate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])


        ax3 = axes[1, 0]
        baseline_anomaly = [ep['avg_anomaly_score'] for ep in baseline_eps]
        rl_anomaly = [ep['avg_anomaly_score'] for ep in rl_eps]

        ax3.hist(baseline_anomaly, bins=15, alpha=0.6, label='Baseline (Non-RL)', color='#e74c3c')
        ax3.hist(rl_anomaly, bins=15, alpha=0.6, label='RL-Evasive', color='#3498db')
        ax3.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Detection Threshold')
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Anomaly Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)


        ax4 = axes[1, 1]
        metrics = ['Detection\nRate', 'Success\nRate', 'Anomaly\nScore']
        baseline_vals = [
            self.results['baseline']['overall']['avg_detection_rate'],
            self.results['baseline']['overall']['avg_success_rate'],
            self.results['baseline']['overall']['avg_anomaly_score']
        ]
        rl_vals = [
            self.results['rl_evasive']['overall']['avg_detection_rate'],
            self.results['rl_evasive']['overall']['avg_success_rate'],
            self.results['rl_evasive']['overall']['avg_anomaly_score']
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax4.bar(x - width/2, baseline_vals, width, label='Baseline (Non-RL)', color='#e74c3c', alpha=0.8)
        ax4.bar(x + width/2, rl_vals, width, label='RL-Evasive', color='#3498db', alpha=0.8)
        ax4.set_ylabel('Score')
        ax4.set_title('Overall Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1.1])

        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"rl_evasion_comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f" Visualization saved to: {plot_file}")
        plt.close()


def _legacy_synthetic_main():

    print(" RL Evasion Comparison Evaluator (LEGACY synthetic)")
    comparator = RLEvasionComparator(num_systems=6, random_seed=42)
    comparator.run_comparison(num_episodes=30, attacks_per_episode=6)


CONFIG_LABELS = {
    "baseline_random": "Random",
    "random":          "Random",
    "auto":            "Auto",
    "rl_autonomous":   "Auto",
    "autonomous":      "Auto",
    "rl_coordinated":  "SPEAR",
    "spear":           "SPEAR",
}
_CONFIG_ORDER = ["Random", "Auto", "SPEAR"]
_FEAT_COLS = [f"t{t}_f{f}" for t in range(10) for f in range(14)]


def _load_captured_samples(csv_paths: List[str]) -> Dict[str, tuple]:

    import csv as _csv
    per_cfg: Dict[str, Dict[str, list]] = {}
    for path in csv_paths:
        with open(path) as fh:
            for row in _csv.DictReader(fh):
                cfg = CONFIG_LABELS.get(str(row.get("config", "")).lower(),
                                        str(row.get("config", "unknown")))
                try:
                    feats = [float(row[c]) for c in _FEAT_COLS]
                except (KeyError, ValueError):
                    continue
                d = per_cfg.setdefault(cfg, {"X": [], "y": [], "t": []})
                d["X"].append(feats)
                d["y"].append(0 if row.get("sample_type") == "benign" else 1)
                d["t"].append(row.get("attack_type", "unknown"))
    return {c: (np.asarray(d["X"], np.float32), np.asarray(d["y"], int),
               np.asarray(d["t"], object)) for c, d in per_cfg.items()}


def _load_all_models(models_dir: str) -> Dict[str, dict]:

    import pickle as _pkl
    bundles = {}
    if not os.path.isdir(models_dir):
        return bundles
    for fn in sorted(os.listdir(models_dir)):
        if fn.endswith(".pkl"):
            try:
                with open(os.path.join(models_dir, fn), "rb") as fh:
                    b = _pkl.load(fh)
                bundles[b.get("model_name", fn[:-4])] = b
            except Exception as e:
                print(f"  skip {fn}: {e}")
    return bundles


def _confusion_metrics(y_true, y_pred) -> Dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return {"f1": f1, "precision": prec, "recall": rec, "fpr": fpr,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _emit_latex(results: Dict[str, Dict[str, Dict]], configs: List[str], path: str):

    def cell(model, cfg, key):
        m = results.get(model, {}).get(cfg)
        return f"{m[key]:.4f}" if m else "--"
    ncfg = len(configs)
    with open(path, "w") as f:
        f.write("% Auto-generated: IDS performance across deployment configurations\n")
        f.write("\\begin{table*}[ht]\n\\centering\n")
        f.write("\\caption{IDS Performance Comparison Across Deployment Configurations}\n")
        f.write("\\label{tab:ids_comparison}\n")
        f.write("\\begin{tabular}{l " + " ".join(["c"*ncfg]*4) + "}\n\\toprule\n")
        f.write("\\multirow{2}{*}{\\textbf{Model}}\n")
        for metric in ["F1-Score", "Precision", "Recall (DR)", "FPR"]:
            f.write(f"  & \\multicolumn{{{ncfg}}}{{c}}{{\\textbf{{{metric}}}}}\n")
        f.write("\\\\\n")
        f.write(" ".join(f"\\cmidrule(lr){{{2+i*ncfg}-{1+(i+1)*ncfg}}}" for i in range(4)) + "\n")
        f.write("& " + " & ".join(" & ".join(configs) for _ in range(4)) + " \\\\\n\\midrule\n")
        for model in results:
            row = [model]
            for key in ["f1", "precision", "recall", "fpr"]:
                row += [cell(model, c, key) for c in configs]
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def evaluate_models_on_captured(csv_glob: str = "detection_results/ids_samples_*.csv",
                                models_dir: str = None):

    import glob as _glob
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "all_ids_models")
    csv_paths = sorted(_glob.glob(csv_glob))
    if not csv_paths:
        print(f" No captured-sample CSVs found ({csv_glob}).")
        print("   Run the main pipeline first — it writes detection_results/ids_samples_*.csv.")
        return None
    print(f" Sample CSVs: {[os.path.basename(p) for p in csv_paths]}")
    per_config = _load_captured_samples(csv_paths)
    bundles = _load_all_models(models_dir)
    if not bundles:
        print(f" No model bundles in {models_dir}.")
        print("   Re-run compare_ids_models_update.py (now calls save_all_ids_models()).")
        return None
    print(f" Models: {list(bundles.keys())}")
    for c, (X, y, _) in per_config.items():
        print(f"   config '{c}': {len(y)} samples ({int((y==1).sum())} attack / {int((y==0).sum())} benign)")

    results: Dict[str, Dict[str, Dict]] = {}
    for name, b in bundles.items():
        scaler, thr, model = b["scaler"], float(b.get("threshold", 0.5)), b["model"]
        results[name] = {}
        for cfg, (X, y, _t) in per_config.items():
            if len(X) == 0:
                continue
            proba = model.predict_proba(scaler.transform(X))[:, 1]
            results[name][cfg] = _confusion_metrics(y, (proba >= thr).astype(int))

    configs = sorted({c for m in results.values() for c in m},
                     key=lambda c: _CONFIG_ORDER.index(c) if c in _CONFIG_ORDER else 99)


    print("\n" + "=" * 100)
    print("IDS PERFORMANCE ACROSS DEPLOYMENT CONFIGURATIONS (same captured attacks, all models)")
    print("=" * 100)
    for metric, label in [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall(DR)"), ("fpr", "FPR")]:
        print(f"\n{label}:")
        print(f"  {'Model':<20}" + "".join(f"{c:>12}" for c in configs))
        for model in results:
            vals = "".join(f"{results[model].get(c, {}).get(metric, float('nan')):>12.4f}" for c in configs)
            print(f"  {model:<20}{vals}")


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("detection_results", exist_ok=True)
    json_path = os.path.join("detection_results", f"ids_deployment_comparison_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"configs": configs, "results": results}, f, indent=2)
    tex_path = os.path.join("detection_results", f"ids_deployment_comparison_{ts}.tex")
    _emit_latex(results, configs, tex_path)
    print(f"\n JSON : {json_path}")
    print(f" LaTeX: {tex_path}")
    return results


def main():
    print(" IDS Deployment-Config Comparison (real models × captured attacks)")
    print("=" * 90)
    evaluate_models_on_captured()


if __name__ == "__main__":
    main()
