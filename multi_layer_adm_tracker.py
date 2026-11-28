#!/usr/bin/env python3
"""
Multi-Layer ADM Performance Tracker

Tracks detection performance across all three anomaly detection layers:
1. Physical constraint validation (Layer 1)
2. Pattern-based attack detection (Layer 2)
3. LSTM-based ML detection (Layer 3)

Provides comprehensive statistics for each layer and cascade analysis to identify
where attacks are being caught in the multi-layer defense system.

Usage:
    tracker = MultiLayerADMTracker()
    
    # Record detection at each layer
    tracker.record_attack_flow(
        is_attack=True,
        physical_detected=False,
        pattern_detected=True,
        lstm_detected=False
    )
    
    # Print summary
    tracker.print_summary()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import os


@dataclass
class LayerDetectionStats:
    """Statistics for a single detection layer"""
    layer_name: str
    total_samples: int = 0
    true_positives: int = 0  # Correctly detected attacks
    false_positives: int = 0  # Normal flagged as attack
    true_negatives: int = 0  # Correctly identified normal
    false_negatives: int = 0  # Missed attacks
    
    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall (Detection Rate): TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / Total"""
        if self.total_samples == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_samples
    
    @property
    def f1_score(self) -> float:
        """F1-Score: 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def false_positive_rate(self) -> float:
        """FPR: FP / (FP + TN)"""
        if self.false_positives + self.true_negatives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_negatives)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'layer_name': self.layer_name,
            'total_samples': self.total_samples,
            'confusion_matrix': {
                'TP': self.true_positives,
                'FP': self.false_positives,
                'TN': self.true_negatives,
                'FN': self.false_negatives
            },
            'metrics': {
                'precision': self.precision,
                'recall': self.recall,
                'accuracy': self.accuracy,
                'f1_score': self.f1_score,
                'false_positive_rate': self.false_positive_rate
            }
        }


class MultiLayerADMTracker:
    """
    Tracks detection performance across multiple ADM layers
    
    Designed for EVCS multi-layer anomaly detection system with:
    - Layer 1: Physical constraint validation
    - Layer 2: Pattern-based attack detection
    - Layer 3: LSTM-based ML detection
    
    Usage:
        tracker = MultiLayerADMTracker()
        
        # Record detection at each layer
        tracker.record_attack_flow(
            is_attack=True,
            physical_detected=False,
            pattern_detected=True,
            lstm_detected=False
        )
        
        # Get statistics
        stats = tracker.get_layer_stats('pattern')
        print(f"Pattern layer F1-score: {stats.f1_score:.4f}")
        
        # Print comprehensive summary
        tracker.print_summary()
    """
    
    def __init__(self):
        self.layers = {
            'physical': LayerDetectionStats('Physical Constraints (Layer 1)'),
            'pattern': LayerDetectionStats('Pattern Detection (Layer 2)'),
            'lstm': LayerDetectionStats('LSTM ML Detection (Layer 3)')
        }
        
        # Track which layer caught each attack
        self.attack_caught_at_layer: Dict[str, int] = defaultdict(int)
        self.attacks_escaped_all_layers: int = 0
        
        # Episode-wise tracking for RL evasion analysis
        self.episode_stats: List[Dict] = []
        self.current_episode: int = 0
        
        # Attack type tracking
        self.attack_types_caught: Dict[str, Dict[str, int]] = {
            'physical': defaultdict(int),
            'pattern': defaultdict(int),
            'lstm': defaultdict(int)
        }
    
    def record_detection(self, layer: str, is_attack: bool, detected: bool):
        """
        Record a detection event at a specific layer
        
        Args:
            layer: Layer name ('physical', 'pattern', 'lstm')
            is_attack: True if this is actually an attack
            detected: True if the layer detected it as an attack
        """
        if layer not in self.layers:
            raise ValueError(f"Unknown layer: {layer}. Must be one of: {list(self.layers.keys())}")
        
        stats = self.layers[layer]
        stats.total_samples += 1
        
        if is_attack and detected:
            stats.true_positives += 1
        elif is_attack and not detected:
            stats.false_negatives += 1
        elif not is_attack and detected:
            stats.false_positives += 1
        elif not is_attack and not detected:
            stats.true_negatives += 1
    
    def record_attack_flow(self, is_attack: bool, 
                          physical_detected: bool,
                          pattern_detected: bool,
                          lstm_detected: bool,
                          attack_type: Optional[str] = None):
        """
        Record the complete flow of an attack through all layers
        
        Args:
            is_attack: True if this is actually an attack
            physical_detected: Detected by physical constraint layer
            pattern_detected: Detected by pattern detection layer
            lstm_detected: Detected by LSTM layer
            attack_type: Optional attack type identifier
        """
        # Record at each layer
        self.record_detection('physical', is_attack, physical_detected)
        self.record_detection('pattern', is_attack, pattern_detected)
        self.record_detection('lstm', is_attack, lstm_detected)
        
        # Track where attack was caught (if at all)
        if is_attack:
            if physical_detected:
                self.attack_caught_at_layer['physical'] += 1
                if attack_type:
                    self.attack_types_caught['physical'][attack_type] += 1
            elif pattern_detected:
                self.attack_caught_at_layer['pattern'] += 1
                if attack_type:
                    self.attack_types_caught['pattern'][attack_type] += 1
            elif lstm_detected:
                self.attack_caught_at_layer['lstm'] += 1
                if attack_type:
                    self.attack_types_caught['lstm'][attack_type] += 1
            else:
                self.attacks_escaped_all_layers += 1
    
    def start_episode(self, episode_num: int):
        """Start tracking a new episode"""
        self.current_episode = episode_num
    
    def end_episode(self):
        """End current episode and save statistics"""
        episode_data = {
            'episode': self.current_episode,
            'layers': {
                layer_key: {
                    'detection_rate': stats.recall,
                    'precision': stats.precision,
                    'f1_score': stats.f1_score,
                    'fpr': stats.false_positive_rate
                }
                for layer_key, stats in self.layers.items()
            },
            'cascade': self.get_cascade_analysis()
        }
        self.episode_stats.append(episode_data)
    
    def get_layer_stats(self, layer: str) -> LayerDetectionStats:
        """Get statistics for a specific layer"""
        if layer not in self.layers:
            raise ValueError(f"Unknown layer: {layer}")
        return self.layers[layer]
    
    def get_all_stats(self) -> Dict[str, LayerDetectionStats]:
        """Get statistics for all layers"""
        return self.layers
    
    def get_cascade_analysis(self) -> Dict:
        """
        Analyze how attacks flow through the cascade of layers
        
        Returns statistics on where attacks are caught
        """
        total_attacks = sum(self.attack_caught_at_layer.values()) + self.attacks_escaped_all_layers
        
        if total_attacks == 0:
            return {
                'total_attacks': 0,
                'caught_by_layer': {},
                'escaped_all': 0,
                'overall_detection_rate': 0.0
            }
        
        caught_by_layer = {
            layer: {
                'count': count,
                'percentage': count / total_attacks * 100
            }
            for layer, count in self.attack_caught_at_layer.items()
        }
        
        overall_detection_rate = (total_attacks - self.attacks_escaped_all_layers) / total_attacks
        
        return {
            'total_attacks': total_attacks,
            'caught_by_layer': caught_by_layer,
            'escaped_all': {
                'count': self.attacks_escaped_all_layers,
                'percentage': self.attacks_escaped_all_layers / total_attacks * 100
            },
            'overall_detection_rate': overall_detection_rate
        }
    
    def get_attack_type_analysis(self) -> Dict:
        """Analyze which layers catch which attack types"""
        return {
            layer: dict(attack_types)
            for layer, attack_types in self.attack_types_caught.items()
        }
    
    def print_summary(self):
        """Print a comprehensive summary of all layers"""
        print("\n" + "="*80)
        print("MULTI-LAYER ADM PERFORMANCE SUMMARY")
        print("="*80)
        
        # Layer-by-layer statistics
        for layer_key in ['physical', 'pattern', 'lstm']:
            stats = self.layers[layer_key]
            print(f"\n{stats.layer_name}")
            print("-" * 80)
            print(f"  Total Samples: {stats.total_samples}")
            print(f"  Confusion Matrix:")
            print(f"    TP: {stats.true_positives:6d}  |  FP: {stats.false_positives:6d}")
            print(f"    FN: {stats.false_negatives:6d}  |  TN: {stats.true_negatives:6d}")
            print(f"  Metrics:")
            print(f"    Precision:  {stats.precision:6.2%}")
            print(f"    Recall:     {stats.recall:6.2%}")
            print(f"    Accuracy:   {stats.accuracy:6.2%}")
            print(f"    F1-Score:   {stats.f1_score:6.4f}")
            print(f"    FPR:        {stats.false_positive_rate:6.2%}")
        
        # Cascade analysis
        cascade = self.get_cascade_analysis()
        print("\n" + "="*80)
        print("CASCADE ANALYSIS - Where Attacks Are Caught")
        print("="*80)
        print(f"  Total Attacks: {cascade['total_attacks']}")
        
        if cascade['total_attacks'] > 0:
            print(f"\n  Detection by Layer:")
            for layer, data in cascade['caught_by_layer'].items():
                layer_name = self.layers[layer].layer_name
                print(f"    {layer_name:40s}: {data['count']:4d} ({data['percentage']:5.1f}%)")
            
            print(f"\n  Escaped All Layers: {cascade['escaped_all']['count']:4d} ({cascade['escaped_all']['percentage']:5.1f}%)")
            print(f"\n  Overall Detection Rate: {cascade['overall_detection_rate']:.2%}")
        
        # Attack type analysis
        attack_type_analysis = self.get_attack_type_analysis()
        if any(attack_type_analysis.values()):
            print("\n" + "="*80)
            print("ATTACK TYPE ANALYSIS")
            print("="*80)
            for layer, attack_types in attack_type_analysis.items():
                if attack_types:
                    print(f"\n  {self.layers[layer].layer_name}:")
                    for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {attack_type:30s}: {count:4d}")
        
        print("="*80)
    
    def save_to_file(self, filepath: str):
        """Save statistics to JSON file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        data = {
            'layers': {
                layer_key: stats.to_dict()
                for layer_key, stats in self.layers.items()
            },
            'cascade_analysis': self.get_cascade_analysis(),
            'attack_type_analysis': self.get_attack_type_analysis(),
            'episode_stats': self.episode_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Multi-layer ADM statistics saved to: {filepath}")
    
    def reset(self):
        """Reset all statistics"""
        for stats in self.layers.values():
            stats.total_samples = 0
            stats.true_positives = 0
            stats.false_positives = 0
            stats.true_negatives = 0
            stats.false_negatives = 0
        
        self.attack_caught_at_layer.clear()
        self.attacks_escaped_all_layers = 0
        self.episode_stats.clear()
        self.current_episode = 0
        
        for layer in self.attack_types_caught:
            self.attack_types_caught[layer].clear()


# Example usage
if __name__ == "__main__":
    print("🔧 Multi-Layer ADM Tracker Example\n")
    
    # Create tracker
    tracker = MultiLayerADMTracker()
    
    # Simulate some detections
    print("Simulating detection events...\n")
    
    # Example 1: Attack caught by physical constraints
    tracker.record_attack_flow(
        is_attack=True,
        physical_detected=True,
        pattern_detected=False,
        lstm_detected=False,
        attack_type='voltage_manipulation'
    )
    
    # Example 2: Attack missed by physical, caught by pattern
    tracker.record_attack_flow(
        is_attack=True,
        physical_detected=False,
        pattern_detected=True,
        lstm_detected=False,
        attack_type='load_injection'
    )
    
    # Example 3: Attack missed by physical and pattern, caught by LSTM
    tracker.record_attack_flow(
        is_attack=True,
        physical_detected=False,
        pattern_detected=False,
        lstm_detected=True,
        attack_type='stealthy_attack'
    )
    
    # Example 4: Attack escaped all layers
    tracker.record_attack_flow(
        is_attack=True,
        physical_detected=False,
        pattern_detected=False,
        lstm_detected=False,
        attack_type='evasive_attack'
    )
    
    # Example 5: Normal operation (no false positives)
    for _ in range(10):
        tracker.record_attack_flow(
            is_attack=False,
            physical_detected=False,
            pattern_detected=False,
            lstm_detected=False
        )
    
    # Example 6: False positive by LSTM
    tracker.record_attack_flow(
        is_attack=False,
        physical_detected=False,
        pattern_detected=False,
        lstm_detected=True
    )
    
    # Print summary
    tracker.print_summary()
    
    # Save to file
    tracker.save_to_file('evaluation_results/multi_layer_adm_stats.json')
