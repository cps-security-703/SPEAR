#!/usr/bin/env python3
# --- repo-root shim: relocated into scripts/<category>/ (keeps flat imports working) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end shim ---
"""
LSTM IDS Pre-Training Script

Trains the LSTM anomaly detection model on benign EVCS data before RL training begins.
This ensures the model learns normal operational patterns and can detect deviations.

Workflow:
1. Generate benign training and validation data
2. Create balanced dataset with synthetic attacks (10%)
3. Train LSTM with focal loss and balanced sampling
4. Save best model for deployment
5. Generate training report with metrics

Usage:
    python train_lstm_ids.py [--epochs 500] [--batch_size 32] [--output_dir models]
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from lstm_anomaly_detector import LSTMIDSDetector
from benign_data_generator import EVCSBenignDataGenerator, EVCSAnomalyDataset
from lstm_ids_evcs import (
    train_with_class_balancing,
    FocalLoss,
    calculate_class_weights,
    create_balanced_sampler
)


def generate_training_data(output_dir: str = 'data', 
                          train_sequences: int = 8000,
                          val_sequences: int = 2000,
                          attack_ratio: float = 0.1):
    """
    Generate benign and attack data for training
    
    Args:
        output_dir: Directory to save data files
        train_sequences: Number of training sequences
        val_sequences: Number of validation sequences
        attack_ratio: Ratio of attack samples to benign
    
    Returns:
        train_dataset, val_dataset
    """
    print("=" * 80)
    print("STEP 1: GENERATING TRAINING DATA")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix seed for reproducible training data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create generator
    generator = EVCSBenignDataGenerator(
        sequence_length=10,
        feature_size=14,
        noise_level=0.05
    )
    
    # Generate benign training data
    print(f"\n📊 Generating {train_sequences} benign training sequences...")
    train_benign_file = os.path.join(output_dir, 'evcs_benign_train.npz')
    generator.save_dataset(train_benign_file, train_sequences)
    
    # Generate benign validation data
    print(f"\n📊 Generating {val_sequences} benign validation sequences...")
    val_benign_file = os.path.join(output_dir, 'evcs_benign_val.npz')
    generator.save_dataset(val_benign_file, val_sequences)
    
    # Create combined datasets with synthetic attacks
    print(f"\n🔀 Creating training dataset with {attack_ratio:.0%} attack samples...")
    train_dataset = EVCSAnomalyDataset.combine_datasets(
        benign_file=train_benign_file,
        attack_ratio=attack_ratio
    )
    
    print(f"\n🔀 Creating validation dataset with {attack_ratio:.0%} attack samples...")
    val_dataset = EVCSAnomalyDataset.combine_datasets(
        benign_file=val_benign_file,
        attack_ratio=attack_ratio
    )
    
    print(f"\n✅ Data generation complete!")
    print(f"   Training set: {len(train_dataset)} samples")
    print(f"   Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, 
                epochs: int = 500,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                output_dir: str = 'models'):
    """
    Train LSTM IDS model
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save models
    
    Returns:
        detector, training_history
    """
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING LSTM IDS MODEL")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detector
    detector = LSTMIDSDetector(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        sequence_length=10,
        anomaly_threshold=0.7
    )
    
    # Train with class balancing
    training_history = train_with_class_balancing(
        ids_detector=detector,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_focal_loss=True,
        use_balanced_sampling=True,
        early_stopping_patience=50,
        verbose=True
    )
    
    # Reload the best checkpoint (saved during training) so the final model
    # is the best-F1 model, not the last-epoch model.
    best_ckpt = 'lstm_ids_best_balanced.pth'
    if os.path.exists(best_ckpt):
        detector.load_model(best_ckpt)
        print(f"   📂 Restored best checkpoint from {best_ckpt}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'lstm_ids_pretrained.pth')
    detector.save_model(final_model_path)
    
    print(f"\n✅ Training complete! Model saved to: {final_model_path}")
    
    return detector, training_history


def plot_training_history(history: dict, output_dir: str = 'models'):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING TRAINING PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LSTM IDS Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.7)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Detection Rate (Recall)
    axes[0, 2].plot(epochs, history['detection_rate'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Detection Rate')
    axes[0, 2].set_title('Attack Detection Rate (Recall)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
    axes[0, 2].legend()
    
    # Precision
    axes[1, 0].plot(epochs, history['precision'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1, 1].plot(epochs, history['f1_score'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
    axes[1, 1].legend()
    
    # False Positive Rate
    axes[1, 2].plot(epochs, history['false_positive_rate'], 'red', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('False Positive Rate')
    axes[1, 2].set_title('False Positive Rate')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0.05, color='g', linestyle='--', label='Target (<5%)')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'lstm_ids_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {plot_path}")
    
    plt.close()


def generate_training_report(history: dict, output_dir: str = 'models'):
    """
    Generate training report
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save report
    """
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING TRAINING REPORT")
    print("=" * 80)
    
    # Get final metrics
    final_metrics = {
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'train_acc': history['train_acc'][-1],
        'val_acc': history['val_acc'][-1],
        'detection_rate': history['detection_rate'][-1],
        'precision': history['precision'][-1],
        'f1_score': history['f1_score'][-1],
        'false_positive_rate': history['false_positive_rate'][-1]
    }
    
    # Get best metrics
    best_metrics = {
        'best_detection_rate': max(history['detection_rate']),
        'best_precision': max(history['precision']),
        'best_f1_score': max(history['f1_score']),
        'lowest_fpr': min(history['false_positive_rate'])
    }
    
    # Create report
    report_path = os.path.join(output_dir, 'lstm_ids_training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LSTM IDS TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(history['train_loss'])}\n\n")
        
        f.write("FINAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Training Loss:        {final_metrics['train_loss']:.4f}\n")
        f.write(f"  Validation Loss:      {final_metrics['val_loss']:.4f}\n")
        f.write(f"  Training Accuracy:    {final_metrics['train_acc']:.2f}%\n")
        f.write(f"  Validation Accuracy:  {final_metrics['val_acc']:.2f}%\n")
        f.write(f"  Detection Rate:       {final_metrics['detection_rate']:.2%}\n")
        f.write(f"  Precision:            {final_metrics['precision']:.2%}\n")
        f.write(f"  F1-Score:             {final_metrics['f1_score']:.4f}\n")
        f.write(f"  False Positive Rate:  {final_metrics['false_positive_rate']:.2%}\n\n")
        
        f.write("BEST METRICS ACHIEVED\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Best Detection Rate:  {best_metrics['best_detection_rate']:.2%}\n")
        f.write(f"  Best Precision:       {best_metrics['best_precision']:.2%}\n")
        f.write(f"  Best F1-Score:        {best_metrics['best_f1_score']:.4f}\n")
        f.write(f"  Lowest FPR:           {best_metrics['lowest_fpr']:.2%}\n\n")
        
        f.write("PERFORMANCE ASSESSMENT\n")
        f.write("-" * 80 + "\n")
        
        # Check if targets are met
        targets_met = []
        targets_missed = []
        
        if final_metrics['detection_rate'] >= 0.80:
            targets_met.append(f"✅ Detection Rate >= 80% ({final_metrics['detection_rate']:.2%})")
        else:
            targets_missed.append(f"❌ Detection Rate < 80% ({final_metrics['detection_rate']:.2%})")
        
        if final_metrics['f1_score'] >= 0.75:
            targets_met.append(f"✅ F1-Score >= 0.75 ({final_metrics['f1_score']:.4f})")
        else:
            targets_missed.append(f"❌ F1-Score < 0.75 ({final_metrics['f1_score']:.4f})")
        
        if final_metrics['false_positive_rate'] <= 0.05:
            targets_met.append(f"✅ FPR <= 5% ({final_metrics['false_positive_rate']:.2%})")
        else:
            targets_missed.append(f"⚠️  FPR > 5% ({final_metrics['false_positive_rate']:.2%})")
        
        for target in targets_met:
            f.write(f"  {target}\n")
        for target in targets_missed:
            f.write(f"  {target}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📄 Training report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Detection Rate:  {final_metrics['detection_rate']:.2%}")
    print(f"  Precision:       {final_metrics['precision']:.2%}")
    print(f"  F1-Score:        {final_metrics['f1_score']:.4f}")
    print(f"  FPR:             {final_metrics['false_positive_rate']:.2%}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train LSTM IDS for EVCS anomaly detection')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_sequences', type=int, default=8000, help='Number of training sequences')
    parser.add_argument('--val_sequences', type=int, default=2000, help='Number of validation sequences')
    parser.add_argument('--attack_ratio', type=float, default=0.1, help='Ratio of attack samples')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("LSTM IDS PRE-TRAINING FOR EVCS ANOMALY DETECTION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Training sequences: {args.train_sequences}")
    print(f"  Validation sequences: {args.val_sequences}")
    print(f"  Attack ratio: {args.attack_ratio:.0%}")
    print("=" * 80 + "\n")
    
    # Step 1: Generate data
    train_dataset, val_dataset = generate_training_data(
        output_dir=args.data_dir,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        attack_ratio=args.attack_ratio
    )
    
    # Step 2: Train model
    detector, history = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Step 3: Plot training history
    plot_training_history(history, output_dir=args.output_dir)
    
    # Step 4: Generate report
    generate_training_report(history, output_dir=args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ LSTM IDS PRE-TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel ready for deployment:")
    print(f"  📁 {os.path.join(args.output_dir, 'lstm_ids_pretrained.pth')}")
    print(f"\nUse this model with FederatedPINNManager for multi-layer anomaly detection.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
