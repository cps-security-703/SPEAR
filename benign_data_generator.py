#!/usr/bin/env python3
"""
Benign EVCS Data Generator

Generates realistic benign (normal) operational data for Electric Vehicle Charging Stations.
This data is used to train the LSTM anomaly detection model to recognize normal behavior patterns.

Key Features:
- Realistic charging cycle simulation
- Temporal sequence generation (10 timesteps)
- 14-dimensional feature space matching LSTM input
- Configurable number of sequences and variation levels
- PyTorch-compatible dataset export
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import os


class EVCSBenignDataGenerator:
    """
    Generator for benign EVCS operational data
    
    Simulates realistic charging cycles with:
    - SOC progression (0.2 → 0.8)
    - Voltage stability (380-420V)
    - Current variation based on SOC
    - Power consumption patterns
    - Temperature gradual increase
    - Demand factor daily patterns
    - Normal operational noise
    
    Args:
        sequence_length: Length of temporal sequences (default: 10)
        feature_size: Number of features per timestep (default: 14)
        noise_level: Amount of random noise to add (default: 0.05)
    """
    
    def __init__(self, sequence_length: int = 10, feature_size: int = 14, 
                 noise_level: float = 0.05):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.noise_level = noise_level
    
    def generate_charging_cycle(self) -> np.ndarray:
        """
        Generate a single benign charging cycle sequence
        
        Returns:
            sequence: [sequence_length, feature_size] array
        """
        sequence = np.zeros((self.sequence_length, self.feature_size), dtype=np.float32)
        
        # Initial SOC (random start between 0.2 and 0.4)
        initial_soc = np.random.uniform(0.2, 0.4)
        
        # Target SOC (random end between 0.7 and 0.9)
        target_soc = np.random.uniform(0.7, 0.9)
        
        # SOC increment per timestep
        soc_increment = (target_soc - initial_soc) / self.sequence_length
        
        # Base parameters — cover the full PINN CMS output envelope
        # PINN outputs: V ∈ [300,500], I ∈ [50,150], P ∈ [15,75]
        base_voltage = np.random.uniform(350, 450)  # Full EVCS voltage range
        base_temperature = np.random.uniform(20, 30)  # Initial temperature
        demand_factor_base = np.random.uniform(0.3, 1.0)
        system_id = np.random.randint(1, 7)  # 6 distribution systems
        time_of_day = np.random.uniform(0, 24)  # Hour of day
        
        for t in range(self.sequence_length):
            # Feature 1: SOC (State of Charge) - gradually increases
            soc = initial_soc + soc_increment * t
            soc = np.clip(soc + np.random.normal(0, self.noise_level), 0.0, 1.0)
            
            # Feature 2: Voltage (normalized) - rises during CC-CV charging
            # CC phase: ~350-400V; CV phase: ~400-450V
            voltage = base_voltage + (soc - initial_soc) / (target_soc - initial_soc + 1e-6) * 50.0
            voltage += np.random.normal(0, 5)  # ±5V variation
            voltage = np.clip(voltage, 300, 500) / 500.0  # Normalize to full PINN range
            
            # Feature 3: Current (normalized) - high at low SOC, tapers at high SOC
            # CC phase: 80-150A; CV phase: tapers to 50-80A
            current_factor = 1.0 - (soc * 0.5)  # Decreases 50% from start to end
            current = np.random.uniform(80, 150) * current_factor
            current = np.clip(current, 50, 150) / 150.0  # Normalize to full PINN range
            
            # Feature 4: Power (normalized) - V*I product in kW
            power = (voltage * 500.0) * (current * 150.0) / 1000.0  # kW
            power = np.clip(power, 15, 75) / 100.0  # Normalize to full PINN range
            
            # Feature 5: Temperature (normalized) - gradually increases
            temperature = base_temperature + (t / self.sequence_length) * 10  # +10°C over cycle
            temperature = temperature + np.random.normal(0, 1)
            temperature = np.clip(temperature, 20, 45) / 50.0  # Normalize
            
            # Feature 6: Demand factor - varies with time of day
            time_variation = np.sin(2 * np.pi * (time_of_day + t * 0.1) / 24) * 0.2
            demand_factor = demand_factor_base + time_variation
            demand_factor = np.clip(demand_factor + np.random.normal(0, self.noise_level), 0.3, 1.2)
            
            # Feature 7: Load factor - correlated with demand
            load_factor = demand_factor * np.random.uniform(0.8, 1.2)
            load_factor = np.clip(load_factor, 0.3, 1.3)
            
            # Feature 8: Grid voltage (per unit) - very stable
            grid_voltage = 1.0 + np.random.normal(0, 0.02)
            grid_voltage = np.clip(grid_voltage, 0.95, 1.05)
            
            # Feature 9: Grid frequency (normalized) - very stable around 60Hz
            grid_frequency = 60.0 + np.random.normal(0, 0.1)
            grid_frequency = np.clip(grid_frequency, 59.8, 60.2) / 60.0
            
            # Feature 10: Queue length (normalized) - random but realistic
            queue_length = np.random.poisson(3)  # Poisson distribution, mean=3
            queue_length = np.clip(queue_length, 0, 10) / 10.0
            
            # Feature 11: Utilization - based on queue and demand
            utilization = (queue_length * 10 + demand_factor * 5) / 15
            utilization = np.clip(utilization + np.random.normal(0, 0.1), 0.2, 0.9)
            
            # Feature 12: Urgency factor - increases as SOC is low
            urgency_factor = 1.0 + (1.0 - soc) * 0.5  # Higher urgency at low SOC
            urgency_factor = np.clip(urgency_factor, 0.8, 1.5)
            
            # Feature 13: Time of day (normalized)
            current_time = (time_of_day + t * 0.1) % 24
            time_normalized = current_time / 24.0
            
            # Feature 14: System ID (normalized)
            system_id_normalized = system_id / 10.0
            
            # Assemble feature vector
            sequence[t] = [
                soc, voltage, current, power, temperature,
                demand_factor, load_factor, grid_voltage, grid_frequency,
                queue_length, utilization, urgency_factor,
                time_normalized, system_id_normalized
            ]
        
        return sequence
    
    def generate_steady_state_cycle(self) -> np.ndarray:
        """
        Generate a benign steady-state sequence (small perturbations around a
        fixed operating point).  This represents an EVCS that is mid-charge at
        a stable power level — the kind of traffic the IDS sees when it
        evaluates individual snapshots accumulated over time.
        """
        sequence = np.zeros((self.sequence_length, self.feature_size), dtype=np.float32)

        # Pick a random but fixed operating point — cover full PINN range
        soc_base = np.random.uniform(0.15, 0.85)
        voltage_base = np.random.uniform(320, 480) / 500.0
        current_base = np.random.uniform(50, 140) / 150.0
        power_base = np.random.uniform(15, 70) / 100.0
        temp_base = np.random.uniform(22, 40) / 50.0
        demand_base = np.random.uniform(0.5, 0.8)
        load_base = demand_base * np.random.uniform(0.9, 1.1)
        gv_base = 1.0 + np.random.uniform(-0.02, 0.02)
        gf_base = (60.0 + np.random.uniform(-0.05, 0.05)) / 60.0
        ql_base = np.random.randint(1, 5) / 10.0
        util_base = np.random.uniform(0.4, 0.7)
        urg_base = np.random.uniform(0.9, 1.2)
        tod_base = np.random.uniform(0, 24) / 24.0
        sid_base = np.random.randint(1, 7) / 10.0

        for t in range(self.sequence_length):
            n = self.noise_level
            sequence[t] = [
                np.clip(soc_base + np.random.normal(0, n * 0.5), 0.0, 1.0),
                np.clip(voltage_base + np.random.normal(0, n * 0.3), 0.6, 1.0),
                np.clip(current_base + np.random.normal(0, n * 0.3), 0.33, 1.0),
                np.clip(power_base + np.random.normal(0, n * 0.2), 0.15, 0.75),
                np.clip(temp_base + np.random.normal(0, n * 0.2), 0.3, 0.9),
                np.clip(demand_base + np.random.normal(0, n), 0.3, 1.2),
                np.clip(load_base + np.random.normal(0, n), 0.3, 1.3),
                np.clip(gv_base + np.random.normal(0, 0.005), 0.95, 1.05),
                np.clip(gf_base + np.random.normal(0, 0.001), 59.8 / 60.0, 60.2 / 60.0),
                np.clip(ql_base + np.random.normal(0, 0.02), 0.0, 1.0),
                np.clip(util_base + np.random.normal(0, n), 0.2, 0.9),
                np.clip(urg_base + np.random.normal(0, n * 0.3), 0.8, 1.5),
                np.clip(tod_base + t * 0.002, 0.0, 1.0),
                sid_base,
            ]
        return sequence

    def generate_dataset(self, num_sequences: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset of benign sequences
        
        Args:
            num_sequences: Number of sequences to generate
        
        Returns:
            sequences: [num_sequences, sequence_length, feature_size]
            labels: [num_sequences] - all zeros (benign)
        """
        sequences = np.zeros((num_sequences, self.sequence_length, self.feature_size), 
                            dtype=np.float32)
        labels = np.zeros(num_sequences, dtype=np.int64)  # All benign (label=0)
        
        for i in range(num_sequences):
            # 50% charging cycles, 50% steady-state — so the LSTM learns
            # both temporal patterns and stable operating-point traffic.
            if i < num_sequences // 2:
                sequences[i] = self.generate_charging_cycle()
            else:
                sequences[i] = self.generate_steady_state_cycle()
        
        # Shuffle so training sees both types interleaved
        perm = np.random.permutation(num_sequences)
        sequences = sequences[perm]
        labels = labels[perm]
        
        return sequences, labels
    
    def save_dataset(self, filepath: str, num_sequences: int):
        """
        Generate and save dataset to file
        
        Args:
            filepath: Path to save dataset (.npz format)
            num_sequences: Number of sequences to generate
        """
        sequences, labels = self.generate_dataset(num_sequences)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save to npz format
        np.savez_compressed(filepath, sequences=sequences, labels=labels)
        
        print(f"💾 Benign dataset saved to: {filepath}")
        print(f"   Sequences: {num_sequences}")
        print(f"   Shape: {sequences.shape}")
        print(f"   Labels: all benign (0)")


class EVCSAnomalyDataset(Dataset):
    """
    PyTorch Dataset for EVCS anomaly detection
    
    Can load both benign and attack data for training/validation
    
    Args:
        sequences: Numpy array of sequences [N, seq_len, features]
        labels: Numpy array of labels [N] (0=benign, 1=attack)
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]
    
    @classmethod
    def from_file(cls, filepath: str) -> 'EVCSAnomalyDataset':
        """Load dataset from .npz file"""
        data = np.load(filepath)
        return cls(data['sequences'], data['labels'])
    
    @classmethod
    def combine_datasets(cls, benign_file: str, attack_file: Optional[str] = None,
                        attack_ratio: float = 0.1) -> 'EVCSAnomalyDataset':
        """
        Combine benign and attack datasets
        
        Args:
            benign_file: Path to benign data file
            attack_file: Path to attack data file (optional)
            attack_ratio: Ratio of attack samples if generating synthetic attacks
        
        Returns:
            Combined dataset
        """
        # Load benign data
        benign_data = np.load(benign_file)
        benign_sequences = benign_data['sequences']
        benign_labels = benign_data['labels']
        
        if attack_file and os.path.exists(attack_file):
            # Load real attack data
            attack_data = np.load(attack_file)
            attack_sequences = attack_data['sequences']
            attack_labels = attack_data['labels']
        else:
            # Generate synthetic attack data by perturbing benign data
            num_attacks = int(len(benign_sequences) * attack_ratio)
            attack_sequences = benign_sequences[:num_attacks].copy()
            
            # Apply random perturbations to create attacks
            for i in range(num_attacks):
                # Randomly select attack type
                attack_type = np.random.randint(0, 6)
                
                if attack_type == 0:  # Voltage manipulation
                    attack_sequences[i, :, 1] *= np.random.uniform(1.2, 1.5)
                elif attack_type == 1:  # Current injection
                    attack_sequences[i, :, 2] *= np.random.uniform(1.3, 1.8)
                elif attack_type == 2:  # Power disruption
                    attack_sequences[i, :, 3] *= np.random.uniform(0.3, 0.7)
                elif attack_type == 3:  # SOC spoofing
                    attack_sequences[i, :, 0] = np.random.uniform(0, 1, attack_sequences.shape[1])
                elif attack_type == 4:  # Thermal attack
                    attack_sequences[i, :, 4] *= np.random.uniform(1.5, 2.0)
                elif attack_type == 5:  # Frequency attack
                    attack_sequences[i, :, 8] *= np.random.uniform(0.95, 1.05)
            
            attack_labels = np.ones(num_attacks, dtype=np.int64)
        
        # Combine datasets
        all_sequences = np.concatenate([benign_sequences, attack_sequences], axis=0)
        all_labels = np.concatenate([benign_labels, attack_labels], axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[indices]
        all_labels = all_labels[indices]
        
        return cls(all_sequences, all_labels)


# Example usage
if __name__ == "__main__":
    print("🔧 EVCS Benign Data Generator\n")
    
    # Create generator
    generator = EVCSBenignDataGenerator(
        sequence_length=10,
        feature_size=14,
        noise_level=0.05
    )
    
    # Generate training data
    print("📊 Generating training data...")
    generator.save_dataset(
        filepath='data/evcs_benign_train.npz',
        num_sequences=8000
    )
    
    # Generate validation data
    print("\n📊 Generating validation data...")
    generator.save_dataset(
        filepath='data/evcs_benign_val.npz',
        num_sequences=2000
    )
    
    # Create combined dataset with synthetic attacks
    print("\n🔀 Creating combined dataset with synthetic attacks...")
    dataset = EVCSAnomalyDataset.combine_datasets(
        benign_file='data/evcs_benign_train.npz',
        attack_ratio=0.1  # 10% attack samples
    )
    
    print(f"\n✅ Combined dataset created:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Benign: {(dataset.labels == 0).sum().item()}")
    print(f"   Attack: {(dataset.labels == 1).sum().item()}")
    
    # Show sample
    sample_seq, sample_label = dataset[0]
    print(f"\n📋 Sample sequence shape: {sample_seq.shape}")
    print(f"   Label: {'Attack' if sample_label == 1 else 'Benign'}")
    print(f"   First timestep features: {sample_seq[0].numpy()}")
