#!/usr/bin/env python3


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import os


class LSTMIDSModel(nn.Module):


    def __init__(self, input_size: int = 14, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMIDSModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )


        self.anomaly_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        lstm_out, (h_n, c_n) = self.lstm(x)


        last_hidden = h_n[-1]


        classification_output = self.classifier(last_hidden)


        anomaly_score = self.anomaly_scorer(last_hidden)

        return classification_output, anomaly_score


class LSTMIDSDetector:


    def __init__(self, input_size: int = 14, hidden_size: int = 128,
                 num_layers: int = 2, sequence_length: int = 10,
                 anomaly_threshold: float = 0.7, device: Optional[str] = None):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold


        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)


        self.model = LSTMIDSModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)

        self.model.eval()

        print(f"# LSTM IDS Detector initialized on {self.device}")
        print(f"   Input size: {input_size}, Hidden size: {hidden_size}")
        print(f"   Sequence length: {sequence_length}, Anomaly threshold: {anomaly_threshold}")

    def extract_features(self, evcs_data: Dict) -> np.ndarray:

        freq = evcs_data.get('grid_frequency', 60.0)

        features = np.array([

            evcs_data.get('soc', 0.5),
            evcs_data.get('voltage', 400.0) / 500.0,
            evcs_data.get('current', 50.0) / 150.0,
            evcs_data.get('power', 20.0) / 100.0,
            evcs_data.get('temperature', 25.0) / 50.0,
            evcs_data.get('demand_factor', 0.7),
            evcs_data.get('load_factor', 0.7),
            evcs_data.get('grid_voltage', 1.0),
            freq / 60.0,
            evcs_data.get('queue_length', 3) / 10.0,
            evcs_data.get('utilization', 0.6),
            evcs_data.get('urgency_factor', 1.0),
            evcs_data.get('time_of_day', 12.0) / 24.0,
            evcs_data.get('system_id', 1) / 10.0,

            (freq - 60.0) / 0.5,
            evcs_data.get('agg_active_power_pu',    0.0),
            evcs_data.get('agg_reactive_power_pu',  0.0),
            evcs_data.get('bus_voltage_min_pu',      1.0),
            evcs_data.get('bus_voltage_max_pu',      1.0),
            float(evcs_data.get('attack_active',     0)),
        ], dtype=np.float32)

        return features

    def detect_anomaly(self, sequence: np.ndarray) -> Tuple[bool, float]:

        if sequence.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {sequence.shape[0]}")

        if sequence.shape[1] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {sequence.shape[1]}")


        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)


        with torch.no_grad():
            classification_output, _anomaly_score_tensor = self.model(sequence_tensor)


            probs = torch.softmax(classification_output, dim=1)
            anomaly_score = probs[0, 1].item()


        is_anomaly = anomaly_score > self.anomaly_threshold

        return is_anomaly, anomaly_score

    def save_model(self, filepath: str):


        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)


        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'anomaly_threshold': self.anomaly_threshold
        }

        torch.save(checkpoint, filepath)
        print(f"# Model saved to: {filepath}")

    def load_model(self, filepath: str):

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)


        if checkpoint['input_size'] != self.input_size:
            print(f"#  Warning: Loaded model input_size ({checkpoint['input_size']}) != current ({self.input_size})")


        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


        self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
        self.anomaly_threshold = checkpoint.get('anomaly_threshold', self.anomaly_threshold)

        print(f"# Model loaded from: {filepath}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Anomaly threshold: {self.anomaly_threshold}")


if __name__ == "__main__":

    detector = LSTMIDSDetector(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        sequence_length=10,
        anomaly_threshold=0.7
    )


    benign_sequence = np.random.rand(10, 14).astype(np.float32) * 0.5 + 0.5


    is_anomaly, score = detector.detect_anomaly(benign_sequence)
    print(f"\nBenign sequence - Anomaly: {is_anomaly}, Score: {score:.4f}")


    anomalous_sequence = np.random.rand(10, 14).astype(np.float32) * 2.0
    is_anomaly, score = detector.detect_anomaly(anomalous_sequence)
    print(f"Anomalous sequence - Anomaly: {is_anomaly}, Score: {score:.4f}")


    detector.save_model('models/lstm_ids_example.pth')


    detector.load_model('models/lstm_ids_example.pth')
