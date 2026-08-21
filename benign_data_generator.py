#!/usr/bin/env python3


import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import os


class EVCSBenignDataGenerator:


    def __init__(self, sequence_length: int = 10, feature_size: int = 14,
                 noise_level: float = 0.05):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.noise_level = noise_level

    def generate_charging_cycle(self) -> np.ndarray:

        sequence = np.zeros((self.sequence_length, self.feature_size), dtype=np.float32)


        initial_soc = np.random.uniform(0.2, 0.4)


        target_soc = np.random.uniform(0.7, 0.9)


        soc_increment = (target_soc - initial_soc) / self.sequence_length


        base_voltage = np.random.uniform(350, 450)
        base_temperature = np.random.uniform(20, 30)
        demand_factor_base = np.random.uniform(0.3, 1.0)
        system_id = np.random.randint(1, 7)
        time_of_day = np.random.uniform(0, 24)

        for t in range(self.sequence_length):

            soc = initial_soc + soc_increment * t
            soc = np.clip(soc + np.random.normal(0, self.noise_level), 0.0, 1.0)


            voltage = base_voltage + (soc - initial_soc) / (target_soc - initial_soc + 1e-6) * 50.0
            voltage += np.random.normal(0, 5)
            voltage = np.clip(voltage, 300, 500) / 500.0


            current_factor = 1.0 - (soc * 0.5)
            current = np.random.uniform(80, 150) * current_factor
            current = np.clip(current, 50, 150) / 150.0


            power = (voltage * 500.0) * (current * 150.0) / 1000.0
            power = np.clip(power, 15, 75) / 100.0


            temperature = base_temperature + (t / self.sequence_length) * 10
            temperature = temperature + np.random.normal(0, 1)
            temperature = np.clip(temperature, 20, 45) / 50.0


            time_variation = np.sin(2 * np.pi * (time_of_day + t * 0.1) / 24) * 0.2
            demand_factor = demand_factor_base + time_variation
            demand_factor = np.clip(demand_factor + np.random.normal(0, self.noise_level), 0.3, 1.2)


            load_factor = demand_factor * np.random.uniform(0.8, 1.2)
            load_factor = np.clip(load_factor, 0.3, 1.3)


            grid_voltage = 1.0 + np.random.normal(0, 0.02)
            grid_voltage = np.clip(grid_voltage, 0.95, 1.05)


            grid_frequency = 60.0 + np.random.normal(0, 0.1)
            grid_frequency = np.clip(grid_frequency, 59.8, 60.2) / 60.0


            queue_length = np.random.poisson(3)
            queue_length = np.clip(queue_length, 0, 10) / 10.0


            utilization = (queue_length * 10 + demand_factor * 5) / 15
            utilization = np.clip(utilization + np.random.normal(0, 0.1), 0.2, 0.9)


            urgency_factor = 1.0 + (1.0 - soc) * 0.5
            urgency_factor = np.clip(urgency_factor, 0.8, 1.5)


            current_time = (time_of_day + t * 0.1) % 24
            time_normalized = current_time / 24.0


            system_id_normalized = system_id / 10.0


            sequence[t] = [
                soc, voltage, current, power, temperature,
                demand_factor, load_factor, grid_voltage, grid_frequency,
                queue_length, utilization, urgency_factor,
                time_normalized, system_id_normalized
            ]

        return sequence

    def generate_steady_state_cycle(self) -> np.ndarray:

        sequence = np.zeros((self.sequence_length, self.feature_size), dtype=np.float32)


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

        sequences = np.zeros((num_sequences, self.sequence_length, self.feature_size),
                            dtype=np.float32)
        labels = np.zeros(num_sequences, dtype=np.int64)

        for i in range(num_sequences):


            if i < num_sequences // 2:
                sequences[i] = self.generate_charging_cycle()
            else:
                sequences[i] = self.generate_steady_state_cycle()


        perm = np.random.permutation(num_sequences)
        sequences = sequences[perm]
        labels = labels[perm]

        return sequences, labels

    def save_dataset(self, filepath: str, num_sequences: int):

        sequences, labels = self.generate_dataset(num_sequences)


        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)


        np.savez_compressed(filepath, sequences=sequences, labels=labels)

        print(f"# Benign dataset saved to: {filepath}")
        print(f"   Sequences: {num_sequences}")
        print(f"   Shape: {sequences.shape}")
        print(f"   Labels: all benign (0)")


class EVCSAnomalyDataset(Dataset):


    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

    @classmethod
    def from_file(cls, filepath: str) -> 'EVCSAnomalyDataset':

        data = np.load(filepath)
        return cls(data['sequences'], data['labels'])

    @classmethod
    def combine_datasets(cls, benign_file: str, attack_file: Optional[str] = None,
                        attack_ratio: float = 0.1) -> 'EVCSAnomalyDataset':


        benign_data = np.load(benign_file)
        benign_sequences = benign_data['sequences']
        benign_labels = benign_data['labels']

        if attack_file and os.path.exists(attack_file):

            attack_data = np.load(attack_file)
            attack_sequences = attack_data['sequences']
            attack_labels = attack_data['labels']
        else:

            num_attacks = int(len(benign_sequences) * attack_ratio)
            attack_sequences = benign_sequences[:num_attacks].copy()


            for i in range(num_attacks):

                attack_type = np.random.randint(0, 6)

                if attack_type == 0:
                    attack_sequences[i, :, 1] *= np.random.uniform(1.2, 1.5)
                elif attack_type == 1:
                    attack_sequences[i, :, 2] *= np.random.uniform(1.3, 1.8)
                elif attack_type == 2:
                    attack_sequences[i, :, 3] *= np.random.uniform(0.3, 0.7)
                elif attack_type == 3:
                    attack_sequences[i, :, 0] = np.random.uniform(0, 1, attack_sequences.shape[1])
                elif attack_type == 4:
                    attack_sequences[i, :, 4] *= np.random.uniform(1.5, 2.0)
                elif attack_type == 5:
                    attack_sequences[i, :, 8] *= np.random.uniform(0.95, 1.05)

            attack_labels = np.ones(num_attacks, dtype=np.int64)


        all_sequences = np.concatenate([benign_sequences, attack_sequences], axis=0)
        all_labels = np.concatenate([benign_labels, attack_labels], axis=0)


        indices = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[indices]
        all_labels = all_labels[indices]

        return cls(all_sequences, all_labels)


if __name__ == "__main__":
    print(" EVCS Benign Data Generator\n")


    generator = EVCSBenignDataGenerator(
        sequence_length=10,
        feature_size=14,
        noise_level=0.05
    )


    print("# Generating training data...")
    generator.save_dataset(
        filepath='data/evcs_benign_train.npz',
        num_sequences=8000
    )


    print("\n# Generating validation data...")
    generator.save_dataset(
        filepath='data/evcs_benign_val.npz',
        num_sequences=2000
    )


    print("\n# Creating combined dataset with synthetic attacks...")
    dataset = EVCSAnomalyDataset.combine_datasets(
        benign_file='data/evcs_benign_train.npz',
        attack_ratio=0.1
    )

    print(f"\n# Combined dataset created:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Benign: {(dataset.labels == 0).sum().item()}")
    print(f"   Attack: {(dataset.labels == 1).sum().item()}")


    sample_seq, sample_label = dataset[0]
    print(f"\n# Sample sequence shape: {sample_seq.shape}")
    print(f"   Label: {'Attack' if sample_label == 1 else 'Benign'}")
    print(f"   First timestep features: {sample_seq[0].numpy()}")
