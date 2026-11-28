#!/usr/bin/env python3
"""
Fixed LSTM IDS Training with Class Imbalance Handling

This module addresses the critical issue where the LSTM IDS achieves 100% accuracy
but 0% detection rate due to severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Focuses training on hard-to-classify examples and down-weights easy examples.
    Particularly effective for imbalanced datasets.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def calculate_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    
    Returns inverse frequency weights to give more importance to minority class
    """
    labels = []
    for _, label in dataset:
        labels.append(label.item() if torch.is_tensor(label) else label)
    
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    
    # Inverse frequency weighting
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Class 0 (Normal): {class_counts[0]} samples ({class_counts[0]/len(labels)*100:.1f}%)")
    print(f"   Class 1 (Attack): {class_counts[1]} samples ({class_counts[1]/len(labels)*100:.1f}%)")
    print(f"   Class Weights: Normal={class_weights[0]:.4f}, Attack={class_weights[1]:.4f}")
    
    return class_weights


def create_balanced_sampler(dataset):
    """
    Create a weighted random sampler for balanced batch sampling
    
    Ensures each batch has roughly equal representation of both classes
    """
    labels = []
    for _, label in dataset:
        labels.append(label.item() if torch.is_tensor(label) else label)
    
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    
    # Weight for each sample (inverse of its class frequency)
    sample_weights = torch.zeros(len(labels))
    for idx, label in enumerate(labels):
        sample_weights[idx] = 1.0 / class_counts[label]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def validate_dataset_balance(train_dataset, val_dataset):
    """
    Validate dataset balance and warn if severe imbalance detected
    """
    print("\n🔍 Validating Dataset Balance...")
    
    for name, dataset in [("Training", train_dataset), ("Validation", val_dataset)]:
        labels = []
        for _, label in dataset:
            labels.append(label.item() if torch.is_tensor(label) else label)
        
        attack_count = sum(labels)
        total_count = len(labels)
        attack_ratio = attack_count / total_count
        
        print(f"\n{name} Set:")
        print(f"   Total samples: {total_count}")
        print(f"   Attack samples: {attack_count} ({attack_ratio:.2%})")
        print(f"   Normal samples: {total_count - attack_count} ({(1-attack_ratio):.2%})")
        
        if attack_ratio < 0.1:
            print(f"   ⚠️  WARNING: Severe class imbalance detected!")
            print(f"   ⚠️  Attack samples < 10% - Model may ignore minority class")
            print(f"   ✅ Recommendation: Use class weighting or oversampling")
        elif attack_ratio < 0.3:
            print(f"   ⚡ CAUTION: Moderate class imbalance")
            print(f"   ✅ Recommendation: Consider using class weighting")
        else:
            print(f"   ✅ Dataset balance is acceptable")


def train_with_class_balancing(ids_detector, train_dataset, val_dataset,
                               epochs=500, batch_size=32, learning_rate=0.001,
                               use_focal_loss=True, use_balanced_sampling=True,
                               early_stopping_patience=20, verbose=True):
    """
    Enhanced training with class imbalance handling
    
    Args:
        ids_detector: LSTM IDS detector instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_focal_loss: Whether to use focal loss (recommended for imbalance)
        use_balanced_sampling: Whether to use balanced sampling
        early_stopping_patience: Patience for early stopping
        verbose: Print training progress
    """
    # Validate dataset balance
    validate_dataset_balance(train_dataset, val_dataset)
    
    # Create data loaders
    if use_balanced_sampling:
        print("\n✅ Using balanced sampling for training")
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss functions
    if use_focal_loss:
        print("✅ Using Focal Loss for class imbalance")
        classification_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        print("✅ Using Weighted Cross-Entropy Loss")
        class_weights = calculate_class_weights(train_dataset)
        class_weights = class_weights.to(ids_detector.device)
        classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    anomaly_criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(ids_detector.model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Early stopping — track F1-score for best recall/precision balance
    best_val_loss = float('inf')
    best_detection_rate = 0.0
    best_f1 = 0.0
    patience_counter = 0
    
    print(f"\n🏋️ Training LSTM IDS Model with Class Balancing:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'detection_rate': [], 'false_positive_rate': [],
        'precision': [], 'recall': [], 'f1_score': []
    }
    
    for epoch in range(epochs):
        # Training phase
        ids_detector.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(ids_detector.device)
            labels = labels.squeeze().to(ids_detector.device)
            
            # Forward pass
            optimizer.zero_grad()
            classification_output, anomaly_score = ids_detector.model(sequences)
            
            # Calculate losses
            classification_loss = classification_criterion(classification_output, labels)
            anomaly_labels = labels.float().unsqueeze(1)
            anomaly_loss = anomaly_criterion(anomaly_score, anomaly_labels)
            
            # Combined loss
            total_loss = classification_loss + 0.5 * anomaly_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ids_detector.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            _, predicted = torch.max(classification_output, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        ids_detector.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True positives
        val_fp = 0  # False positives
        val_tn = 0  # True negatives
        val_fn = 0  # False negatives
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(ids_detector.device)
                labels = labels.squeeze().to(ids_detector.device)
                
                # Forward pass
                classification_output, anomaly_score = ids_detector.model(sequences)
                
                # Calculate losses
                classification_loss = classification_criterion(classification_output, labels)
                anomaly_labels = labels.float().unsqueeze(1)
                anomaly_loss = anomaly_criterion(anomaly_score, anomaly_labels)
                total_loss = classification_loss + 0.5 * anomaly_loss
                
                # Statistics
                val_loss += total_loss.item()
                _, predicted = torch.max(classification_output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Confusion matrix
                for pred, true in zip(predicted, labels):
                    if pred == 1 and true == 1:
                        val_tp += 1
                    elif pred == 1 and true == 0:
                        val_fp += 1
                    elif pred == 0 and true == 0:
                        val_tn += 1
                    elif pred == 0 and true == 1:
                        val_fn += 1
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Calculate metrics
        detection_rate = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        false_positive_rate = val_fp / (val_fp + val_tn) if (val_fp + val_tn) > 0 else 0.0
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        recall = detection_rate
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['detection_rate'].append(detection_rate)
        training_history['false_positive_rate'].append(false_positive_rate)
        training_history['precision'].append(precision)
        training_history['recall'].append(recall)
        training_history['f1_score'].append(f1_score)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"\n   Epoch {epoch+1}/{epochs}:")
            print(f"      Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"      Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"      Detection Rate (Recall): {detection_rate:.2%}")
            print(f"      Precision: {precision:.2%}")
            print(f"      F1-Score: {f1_score:.4f}")
            print(f"      False Positive Rate: {false_positive_rate:.2%}")
            print(f"      Confusion Matrix: TP={val_tp}, FP={val_fp}, TN={val_tn}, FN={val_fn}")
        
        # Early stopping based on F1-score (balances recall and precision)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_detection_rate = detection_rate
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            ids_detector.save_model('lstm_ids_best_balanced.pth')
            if verbose and (epoch + 1) % 10 == 0:
                print(f"      💾 Best model saved (F1: {best_f1:.4f}, DR: {best_detection_rate:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n   ⏹️  Early stopping at epoch {epoch+1}")
                print(f"      Best F1: {best_f1:.4f}, Best detection rate: {best_detection_rate:.2%}")
                break
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best F1-score: {best_f1:.4f}")
    print(f"   Best detection rate: {best_detection_rate:.2%}")
    print(f"   Final F1-Score: {f1_score:.4f}")
    
    return training_history


# Example usage
if __name__ == "__main__":
    from lstm_ids_detector import LSTMIDSDetector, EVCSAnomalyDataset
    
    # Create detector
    detector = LSTMIDSDetector(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        sequence_length=10
    )
    
    # Load or create datasets
    # train_dataset = EVCSAnomalyDataset(...)
    # val_dataset = EVCSAnomalyDataset(...)
    
    # Train with class balancing
    # history = train_with_class_balancing(
    #     detector, train_dataset, val_dataset,
    #     epochs=500, use_focal_loss=True, use_balanced_sampling=True
    # )
