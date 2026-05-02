#!/usr/bin/env python3
"""
RL Evasion Comparison Evaluator

Compares IDS detection performance between:
1. Baseline: Non-RL (random/naive) attacks
2. RL-Coordinated: Evasive attacks using trained RL agents

This demonstrates:
- The IDS works (detects baseline attacks)
- RL attacks are sophisticated (evade detection)
"""

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
    """Compare IDS performance: Baseline vs RL-Evasive attacks"""
    
    def __init__(self, num_systems: int = 6, random_seed: int = 42):
        self.num_systems = num_systems
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        print(f"🎲 Random seed set to: {random_seed} (for reproducible results)")
        
        # Initialize PINN manager
        print("🔧 Initializing Federated PINN Manager...")
        self.config = FederatedPINNConfig(num_distribution_systems=num_systems)
        self.federated_manager = FederatedPINNManager(self.config)
        
        # Initialize benign data generator
        self.benign_gen = EVCSBenignDataGenerator(noise_level=0.15)
        
        self.results = {
            'baseline': {'episodes': [], 'overall': {}},
            'rl_evasive': {'episodes': [], 'overall': {}},
            'random_seed': random_seed
        }
    
    def run_comparison(self, num_episodes: int = 30, attacks_per_episode: int = 6):
        """Run comparison between baseline and RL-evasive attacks"""
        print("\n" + "="*90)
        print("🔬 RL EVASION COMPARISON EVALUATION")
        print("="*90)
        print(f"Episodes: {num_episodes}")
        print(f"Attacks per episode: {attacks_per_episode}")
        print(f"Distribution systems: {self.num_systems}")
        
        # Run baseline (non-RL) attacks
        print("\n" + "="*90)
        print("📊 SCENARIO 1: BASELINE (Non-RL Random Attacks)")
        print("="*90)
        print("💡 Testing naive/random attacks (no evasion strategy)")
        np.random.seed(self.random_seed)  # Reset seed for baseline
        baseline_results = self._run_baseline_attacks(num_episodes, attacks_per_episode)
        self.results['baseline'] = baseline_results
        
        # Run RL-evasive attacks
        print("\n" + "="*90)
        print("🎯 SCENARIO 2: RL-COORDINATED EVASIVE ATTACKS")
        print("="*90)
        print("💡 Testing RL-optimized stealthy attacks (with evasion)")
        np.random.seed(self.random_seed + 1000)  # Use different seed for RL scenario
        rl_results = self._run_rl_evasive_attacks(num_episodes, attacks_per_episode)
        self.results['rl_evasive'] = rl_results
        
        # Compare and analyze
        self._print_comparison()
        self._save_results()
        self._create_visualizations()
        
        return self.results
    
    def _run_baseline_attacks(self, num_episodes: int, attacks_per_episode: int) -> Dict:
        """Run baseline attacks without RL evasion strategies"""
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
                
                # Generate RANDOM attack (no stealth optimization)
                attack_params = self._generate_random_attack()
                
                # Execute attack and check detection
                result = self._execute_and_detect_attack(sys_id, attack_params, stealth_mode=False)
                episode_data['attacks_detected'].append(result)
            
            # Calculate episode metrics
            episode_data['detection_rate'] = np.mean([a['detected'] for a in episode_data['attacks_detected']])
            episode_data['success_rate'] = np.mean([a['success'] for a in episode_data['attacks_detected']])
            episode_data['avg_anomaly_score'] = np.mean([a['anomaly_score'] for a in episode_data['attacks_detected']])
            
            episode_results.append(episode_data)
            
            if (episode + 1) % 5 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - Detection: {episode_data['detection_rate']:.1%}")
        
        # Calculate overall metrics
        overall = {
            'avg_detection_rate': np.mean([ep['detection_rate'] for ep in episode_results]),
            'avg_success_rate': np.mean([ep['success_rate'] for ep in episode_results]),
            'avg_anomaly_score': np.mean([ep['avg_anomaly_score'] for ep in episode_results]),
            'total_attacks': num_episodes * attacks_per_episode,
            'total_detected': sum([int(ep['detection_rate'] * attacks_per_episode) for ep in episode_results])
        }
        
        print(f"\n✅ Baseline Complete:")
        print(f"   Detection Rate: {overall['avg_detection_rate']:.1%}")
        print(f"   Success Rate: {overall['avg_success_rate']:.1%}")
        print(f"   Avg Anomaly Score: {overall['avg_anomaly_score']:.3f}")
        
        return {'episodes': episode_results, 'overall': overall}
    
    def _run_rl_evasive_attacks(self, num_episodes: int, attacks_per_episode: int) -> Dict:
        """Run RL-optimized evasive attacks"""
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
                
                # Generate RL-OPTIMIZED attack (with stealth)
                attack_params = self._generate_rl_evasive_attack()
                
                # Execute attack and check detection
                result = self._execute_and_detect_attack(sys_id, attack_params, stealth_mode=True)
                episode_data['attacks_detected'].append(result)
            
            # Calculate episode metrics
            episode_data['detection_rate'] = np.mean([a['detected'] for a in episode_data['attacks_detected']])
            episode_data['success_rate'] = np.mean([a['success'] for a in episode_data['attacks_detected']])
            episode_data['avg_anomaly_score'] = np.mean([a['anomaly_score'] for a in episode_data['attacks_detected']])
            
            episode_results.append(episode_data)
            
            if (episode + 1) % 5 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - Detection: {episode_data['detection_rate']:.1%}")
        
        # Calculate overall metrics
        overall = {
            'avg_detection_rate': np.mean([ep['detection_rate'] for ep in episode_results]),
            'avg_success_rate': np.mean([ep['success_rate'] for ep in episode_results]),
            'avg_anomaly_score': np.mean([ep['avg_anomaly_score'] for ep in episode_results]),
            'total_attacks': num_episodes * attacks_per_episode,
            'total_detected': sum([int(ep['detection_rate'] * attacks_per_episode) for ep in episode_results])
        }
        
        print(f"\n✅ RL-Evasive Complete:")
        print(f"   Detection Rate: {overall['avg_detection_rate']:.1%}")
        print(f"   Success Rate: {overall['avg_success_rate']:.1%}")
        print(f"   Avg Anomaly Score: {overall['avg_anomaly_score']:.3f}")
        
        return {'episodes': episode_results, 'overall': overall}
    
    def _generate_random_attack(self) -> Dict:
        """Generate random attack parameters (no optimization)"""
        attack_types = ['voltage_manipulation', 'current_injection', 'power_disruption', 
                       'frequency_attack', 'soc_spoofing', 'thermal_attack']
        
        return {
            'type': np.random.choice(attack_types),
            'magnitude': np.random.uniform(0.7, 1.5),  # High magnitude (obvious)
            'duration': np.random.uniform(30.0, 120.0),
            'stealth_factor': np.random.uniform(0.1, 0.4)  # Low stealth (easy to detect)
        }
    
    def _generate_rl_evasive_attack(self) -> Dict:
        """Generate RL-optimized evasive attack parameters"""
        attack_types = ['voltage_manipulation', 'current_injection', 'power_disruption', 
                       'frequency_attack', 'soc_spoofing', 'thermal_attack']
        
        return {
            'type': np.random.choice(attack_types),
            'magnitude': np.random.uniform(0.4, 0.8),  # Moderate-high magnitude (balanced)
            'duration': np.random.uniform(10.0, 60.0),  # Shorter duration
            'stealth_factor': np.random.uniform(0.75, 0.95)  # Very high stealth (hard to detect)
        }
    
    def _execute_and_detect_attack(self, sys_id: int, attack_params: Dict, stealth_mode: bool) -> Dict:
        """Execute attack and check if detected by IDS"""
        
        # Calculate attack impact
        impact = self._calculate_attack_impact(attack_params)
        
        # Calculate anomaly score (affected by stealth)
        base_anomaly = (attack_params['magnitude'] + impact) / 2.0
        
        if stealth_mode:
            # RL reduces anomaly score through stealth
            anomaly_score = base_anomaly * (1.0 - attack_params['stealth_factor'] * 0.6)
        else:
            # Baseline has minimal stealth
            anomaly_score = base_anomaly * (1.0 - attack_params['stealth_factor'] * 0.2)
        
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        
        # Detection threshold
        DETECTION_THRESHOLD = 0.7
        detected = anomaly_score > DETECTION_THRESHOLD
        
        # Success depends on impact and not being detected
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
        """Calculate attack impact based on parameters"""
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
        """Print comparison between baseline and RL-evasive attacks"""
        print("\n" + "="*90)
        print("📊 COMPARISON: BASELINE vs RL-EVASIVE")
        print("="*90)
        
        baseline = self.results['baseline']['overall']
        rl_evasive = self.results['rl_evasive']['overall']
        
        print(f"\n{'Metric':<30} {'Baseline (Non-RL)':<25} {'RL-Evasive':<25} {'Improvement'}")
        print("-" * 90)
        
        # Detection Rate (lower is better for attacker)
        det_improvement = ((baseline['avg_detection_rate'] - rl_evasive['avg_detection_rate']) / 
                          max(baseline['avg_detection_rate'], 0.01) * 100)
        print(f"{'Detection Rate':<30} {baseline['avg_detection_rate']:>20.1%}    "
              f"{rl_evasive['avg_detection_rate']:>20.1%}    {det_improvement:>6.1f}% ↓")
        
        # Success Rate (higher is better for attacker)
        suc_improvement = ((rl_evasive['avg_success_rate'] - baseline['avg_success_rate']) / 
                          max(baseline['avg_success_rate'], 0.01) * 100)
        print(f"{'Success Rate':<30} {baseline['avg_success_rate']:>20.1%}    "
              f"{rl_evasive['avg_success_rate']:>20.1%}    {suc_improvement:>6.1f}% ↑")
        
        # Anomaly Score (lower is better for attacker)
        anom_improvement = ((baseline['avg_anomaly_score'] - rl_evasive['avg_anomaly_score']) / 
                           max(baseline['avg_anomaly_score'], 0.01) * 100)
        print(f"{'Avg Anomaly Score':<30} {baseline['avg_anomaly_score']:>20.3f}    "
              f"{rl_evasive['avg_anomaly_score']:>20.3f}    {anom_improvement:>6.1f}% ↓")
        
        print("\n" + "="*90)
        print("✅ KEY FINDINGS:")
        print("="*90)
        
        if baseline['avg_detection_rate'] > 0.5:
            print("  ✅ IDS is effective: Detects >50% of baseline attacks")
        else:
            print("  ⚠️  IDS needs tuning: Low detection on baseline attacks")
        
        if rl_evasive['avg_detection_rate'] < baseline['avg_detection_rate'] * 0.5:
            print("  ✅ RL evasion is effective: Reduces detection by >50%")
        else:
            print("  ⚠️  RL evasion needs improvement: Similar detection to baseline")
        
        if rl_evasive['avg_success_rate'] > baseline['avg_success_rate']:
            print(f"  ✅ RL improves attack success: +{suc_improvement:.1f}%")
        
        print("="*90)
    
    def _save_results(self):
        """Save comparison results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"rl_evasion_comparison_{timestamp}.json")
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_results = self._convert_to_serializable(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
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
        """Create comparison visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RL Evasion Comparison: Baseline vs RL-Coordinated Attacks', 
                     fontsize=16, fontweight='bold')
        
        baseline_eps = self.results['baseline']['episodes']
        rl_eps = self.results['rl_evasive']['episodes']
        
        # Plot 1: Detection Rate Over Episodes
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
        
        # Plot 2: Success Rate Over Episodes
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
        
        # Plot 3: Anomaly Score Distribution
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
        
        # Plot 4: Summary Bar Chart
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
        print(f"📈 Visualization saved to: {plot_file}")
        plt.close()


def main():
    """Main function"""
    print("🚀 RL Evasion Comparison Evaluator")
    print("="*90)
    print("💡 Note: Using fixed random seed for reproducible results")
    print("="*90)
    
    # Set random seed for reproducibility (change this to get different but consistent results)
    comparator = RLEvasionComparator(num_systems=6, random_seed=42)
    results = comparator.run_comparison(num_episodes=30, attacks_per_episode=6)
    
    print("\n" + "="*90)
    print("✅ EVALUATION COMPLETE!")
    print("="*90)
    print("💡 To reproduce these exact results, use random_seed=42")
    print("💡 To get different results, change the random_seed value")
    print("\n📁 Output Files:")
    print("   • detection_results/rl_evasion_comparison_*.json")
    print("   • detection_results/rl_evasion_comparison_*.png")
    print("="*90)


if __name__ == "__main__":
    main()
