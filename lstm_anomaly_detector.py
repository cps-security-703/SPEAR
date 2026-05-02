#!/usr/bin/env python3
"""
LSTM-based Anomaly Detector for EVCS Systems

This module provides an LSTM-based intrusion detection system specifically
designed for Electric Vehicle Charging Station (EVCS) operations. It detects
anomalies in temporal sequences of EVCS operational data.

Key Features:
- Dual-output LSTM architecture (classification + anomaly score)
- Temporal sequence analysis (10 timesteps)
- 14-dimensional EVCS feature space
- Pre-trained model loading/saving
- Real-time inference capability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import os


class LSTMIDSModel(nn.Module):
    """
    LSTM-based Intrusion Detection System Model

    """
    
    def __init__(self, input_size: int = 14, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMIDSModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head (binary: normal=0, attack=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 2 classes
        )
        
        # Anomaly score head (continuous score)
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
        
        Returns:
            classification_output: Logits for binary classification [batch_size, 2]
            anomaly_score: Anomaly score [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for classification
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Classification output
        classification_output = self.classifier(last_hidden)
        
        # Anomaly score output
        anomaly_score = self.anomaly_scorer(last_hidden)
        
        return classification_output, anomaly_score


class LSTMIDSDetector:
    """
    LSTM IDS Detector Wrapper

    """
    
    def __init__(self, input_size: int = 14, hidden_size: int = 128,
                 num_layers: int = 2, sequence_length: int = 10,
                 anomaly_threshold: float = 0.7, device: Optional[str] = None):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create model
        self.model = LSTMIDSModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode by default
        
        print(f"# LSTM IDS Detector initialized on {self.device}")
        print(f"   Input size: {input_size}, Hidden size: {hidden_size}")
        print(f"   Sequence length: {sequence_length}, Anomaly threshold: {anomaly_threshold}")
    
    def extract_features(self, evcs_data: Dict) -> np.ndarray:
        """
        Extract 20-dimensional feature vector from EVCS + grid data.
        """
        freq = evcs_data.get('grid_frequency', 60.0)

        features = np.array([
            # ── EVCS features 0-13 ───────────────────────────────────────────
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
            # ── Grid-level features 14-19 ────────────────────────────────────
            (freq - 60.0) / 0.5,                             # 14 freq_dev
            evcs_data.get('agg_active_power_pu',    0.0),    # 15
            evcs_data.get('agg_reactive_power_pu',  0.0),    # 16
            evcs_data.get('bus_voltage_min_pu',      1.0),   # 17
            evcs_data.get('bus_voltage_max_pu',      1.0),   # 18
            float(evcs_data.get('attack_active',     0)),    # 19
        ], dtype=np.float32)

        return features
    
    def detect_anomaly(self, sequence: np.ndarray) -> Tuple[bool, float]:
        """
        Detect anomaly in a temporal sequence
        
        Args:
            sequence: Temporal sequence [sequence_length, input_size]
        
        Returns:
            is_anomaly: True if anomaly detected
            anomaly_score: Continuous anomaly score [0, 1]
        """
        if sequence.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {sequence.shape[0]}")
        
        if sequence.shape[1] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {sequence.shape[1]}")
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            classification_output, _anomaly_score_tensor = self.model(sequence_tensor)
            
            # Use softmax probability of class-1 (attack) from the classifier
            # head as the anomaly score.  The classifier was trained with focal
            # loss and is well-calibrated; the separate anomaly_scorer head is
            # poorly calibrated and produces near-random scores on benign data.
            probs = torch.softmax(classification_output, dim=1)
            anomaly_score = probs[0, 1].item()  # P(attack)
        
        # Determine if anomaly using score threshold only.
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        return is_anomaly, anomaly_score
    
    def save_model(self, filepath: str):
        """Save model to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model state and configuration
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
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Verify configuration matches
        if checkpoint['input_size'] != self.input_size:
            print(f"#  Warning: Loaded model input_size ({checkpoint['input_size']}) != current ({self.input_size})")
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Update configuration
        self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
        self.anomaly_threshold = checkpoint.get('anomaly_threshold', self.anomaly_threshold)
        
        print(f"# Model loaded from: {filepath}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Anomaly threshold: {self.anomaly_threshold}")


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = LSTMIDSDetector(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        sequence_length=10,
        anomaly_threshold=0.7
    )
    
    # Example: Create a benign sequence
    benign_sequence = np.random.rand(10, 14).astype(np.float32) * 0.5 + 0.5  # Values in [0.5, 1.0]
    
    # Detect anomaly
    is_anomaly, score = detector.detect_anomaly(benign_sequence)
    print(f"\nBenign sequence - Anomaly: {is_anomaly}, Score: {score:.4f}")
    
    # Example: Create an anomalous sequence
    anomalous_sequence = np.random.rand(10, 14).astype(np.float32) * 2.0  # Values in [0, 2.0] (out of normal range)
    is_anomaly, score = detector.detect_anomaly(anomalous_sequence)
    print(f"Anomalous sequence - Anomaly: {is_anomaly}, Score: {score:.4f}")
    
    # Save model
    detector.save_model('models/lstm_ids_example.pth')
    
    # Load model
    detector.load_model('models/lstm_ids_example.pth')
