import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import urllib.request
import zipfile
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. DOWNLOAD AND LOAD UCI HUMAN ACTIVITY RECOGNITION DATASET
# ============================================================================
def download_uci_har_data():
    """
    Download UCI Human Activity Recognition dataset directly from UCI repository
    Dataset: 30 subjects, 6 activities, smartphone sensors (accelerometer & gyroscope)
    """
    # The extracted folder has spaces in the name
    data_dir_with_spaces = 'UCI HAR Dataset'
    data_dir_no_spaces = 'UCI_HAR_Dataset'
    zip_file = 'UCI_HAR_Dataset.zip'
    
    # Check if already downloaded (either version)
    if os.path.exists(data_dir_with_spaces):
        print(f"Dataset already exists in '{data_dir_with_spaces}/'")
        return data_dir_with_spaces
    if os.path.exists(data_dir_no_spaces):
        print(f"Dataset already exists in '{data_dir_no_spaces}/'")
        return data_dir_no_spaces
    
    print("Downloading UCI Human Activity Recognition dataset...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    
    try:
        urllib.request.urlretrieve(url, zip_file)
        print("Download complete! Extracting...")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        os.remove(zip_file)
        
        # Check which folder was created
        if os.path.exists(data_dir_with_spaces):
            print(f"Dataset extracted to '{data_dir_with_spaces}/'")
            return data_dir_with_spaces
        else:
            print(f"Dataset extracted to '{data_dir_no_spaces}/'")
            return data_dir_no_spaces
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternatively, you can manually download from:")
        print("https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones")
        raise

def load_uci_har_data(data_dir):
    """Load UCI HAR dataset from extracted files"""
    print(f"\nLoading UCI HAR dataset from '{data_dir}'...")
    
    # Verify directory structure
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found: {train_dir}")
        print(f"Contents of {data_dir}:")
        print(os.listdir(data_dir))
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load training data
    train_x_path = os.path.join(data_dir, 'train', 'X_train.txt')
    train_y_path = os.path.join(data_dir, 'train', 'y_train.txt')
    
    print(f"  Loading: {train_x_path}")
    train_X = np.loadtxt(train_x_path)
    print(f"  Loading: {train_y_path}")
    train_y = np.loadtxt(train_y_path)
    
    # Load test data
    test_x_path = os.path.join(data_dir, 'test', 'X_test.txt')
    test_y_path = os.path.join(data_dir, 'test', 'y_test.txt')
    
    print(f"  Loading: {test_x_path}")
    test_X = np.loadtxt(test_x_path)
    print(f"  Loading: {test_y_path}")
    test_y = np.loadtxt(test_y_path)
    
    # Combine train and test
    X = np.vstack([train_X, test_X])
    y = np.hstack([train_y, test_y])
    
    # Convert to DataFrames
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y - 1, columns=['Activity'])  # Convert to 0-indexed
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Activities: {len(np.unique(y))}")
    
    activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                     'Sitting', 'Standing', 'Laying']
    print(f"  Activity labels: {activity_names}")
    
    return X_df, y_df

class UCIHARDataset(Dataset):
    """PyTorch Dataset for UCI HAR data"""
    def __init__(self, features, labels, seq_length=128, n_features=9):
        """
        Args:
            features: DataFrame with sensor readings (561 features)
            labels: DataFrame with activity labels (0-5)
            seq_length: Length of sequence windows
            n_features: Number of features to use per time step
        """
        self.seq_length = seq_length
        self.n_features = n_features
        
        # Convert to numpy
        self.features = features.values.astype(np.float32)
        self.labels = labels['Activity'].values.astype(np.int64)
        
        # Select subset of features
        # Use the first n_features (these represent different sensor measurements)
        self.features = self.features[:, :n_features]
        
        print(f"\nDataset initialized:")
        print(f"  Samples: {len(self.features)}")
        print(f"  Features per time step: {self.n_features}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Classes: {len(np.unique(self.labels))}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get features and label
        features = self.features[idx]
        label = self.labels[idx]
        
        # Create time series by adding temporal variation
        # Each feature becomes a time series with some variation
        sequence = np.zeros((self.seq_length, self.n_features), dtype=np.float32)
        
        for t in range(self.seq_length):
            # Add sinusoidal variation to simulate temporal dynamics
            time_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / self.seq_length)
            noise = np.random.normal(0, 0.02, self.n_features)
            sequence[t] = features * time_factor + noise
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])[0]

# ============================================================================
# 2. DEFINE LSTM MODEL (TARGET MODEL)
# ============================================================================
class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, num_classes=6):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use last time step output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def predict_proba(self, x):
        """Get probability distributions"""
        logits = self.forward(x)
        return self.softmax(logits)

# ============================================================================
# 3. DEFINE ADVERSARIAL MODEL (INPUT RECONSTRUCTOR)
# ============================================================================
class AdversarialReconstructor(nn.Module):
    """
    Adversarial model that tries to reconstruct inputs from LSTM outputs.
    This demonstrates a Model Inversion Attack - recovering private sensor data
    from model predictions.
    """
    def __init__(self, num_classes=6, seq_length=128, n_features=9, hidden_size=128):
        super(AdversarialReconstructor, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        
        # Encoder: Take output probabilities and encode them
        self.encoder = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Decoder LSTM: Reconstruct the time series
        self.decoder_lstm = nn.LSTM(512, hidden_size, num_layers=3, 
                                     batch_first=True, dropout=0.3)
        
        # Output layer: Generate features at each time step
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_features)
        )
    
    def forward(self, output_probs):
        # Encode output probabilities
        encoded = self.encoder(output_probs)
        
        # Expand for sequence generation
        encoded = encoded.unsqueeze(1)
        
        # Repeat for each time step
        encoded = encoded.repeat(1, self.seq_length, 1)
        
        # LSTM decoder
        decoded, _ = self.decoder_lstm(encoded)
        
        # Generate time series for each time step
        reconstructed = self.output_layer(decoded)
        
        return reconstructed

# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================
def train_lstm(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train the LSTM classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% '
                  f'(Best: {best_val_acc:.2f}%)')
    
    return train_losses, val_accuracies

def train_adversarial(adv_model, target_model, train_loader, epochs=50, lr=0.001):
    """
    Train adversarial model to reconstruct inputs from LSTM outputs
    This simulates a Model Inversion Attack
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(adv_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    losses = []
    
    print("\n" + "="*60)
    print("TRAINING ADVERSARIAL RECONSTRUCTOR")
    print("="*60)
    print("🎯 Goal: Reconstruct sensor inputs from LSTM outputs")
    print("🔒 Privacy Risk: This demonstrates model inversion attack")
    
    target_model.eval()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        adv_model.train()
        total_loss = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            
            # Get target model's output (predictions)
            with torch.no_grad():
                target_outputs = target_model.predict_proba(batch_data)
            
            # Try to reconstruct input from output
            reconstructed = adv_model(target_outputs)
            
            # Loss: how well did we reconstruct the input?
            loss = criterion(reconstructed, batch_data)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adv_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f} (Best: {best_loss:.6f})')
    
    return losses

# ============================================================================
# 5. EVALUATION FUNCTIONS
# ============================================================================
def evaluate_lstm(model, test_loader):
    """Evaluate LSTM model performance"""
    model.eval()
    correct = 0
    total = 0
    
    print("\n" + "="*60)
    print("EVALUATING LSTM MODEL")
    print("="*60)
    
    # Per-class accuracy
    class_correct = [0] * 6
    class_total = [0] * 6
    activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                     'Sitting', 'Standing', 'Laying']
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            for i in range(batch_labels.size(0)):
                label = batch_labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    accuracy = 100 * correct / total
    print(f'\n✓ Overall Accuracy: {accuracy:.2f}%')
    
    print("\nPer-Activity Accuracy:")
    for i in range(6):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {activity_names[i]:20s}: {acc:.2f}%")
    
    return accuracy

def evaluate_adversarial(adv_model, target_model, test_loader):
    """Evaluate adversarial model's reconstruction capability"""
    adv_model.eval()
    target_model.eval()
    
    all_originals = []
    all_reconstructed = []
    all_predictions = []
    
    print("\n" + "="*60)
    print("EVALUATING MODEL INVERSION ATTACK")
    print("="*60)
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.to(device)
            
            target_outputs = target_model.predict_proba(batch_data)
            reconstructed = adv_model(target_outputs)
            
            all_originals.append(batch_data.cpu().numpy())
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_predictions.append(target_outputs.cpu().numpy())
    
    originals = np.concatenate(all_originals, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(originals.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(originals.flatten(), reconstructed.flatten())
    correlation = np.corrcoef(originals.flatten(), reconstructed.flatten())[0, 1]
    
    print("\n📊 Reconstruction Quality Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Overall Correlation: {correlation:.6f}")
    
    print("\n📈 Per-Feature Correlation:")
    for i in range(originals.shape[2]):
        feat_corr = np.corrcoef(originals[:, :, i].flatten(), 
                                reconstructed[:, :, i].flatten())[0, 1]
        print(f"  Feature {i+1}: {feat_corr:.4f}")
    
    return originals, reconstructed, predictions, mse, mae, correlation

def visualize_reconstruction(originals, reconstructed, predictions, n_samples=3):
    """Visualize original vs reconstructed sensor data"""
    activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                     'Sitting', 'Standing', 'Laying']
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    
    for i in range(n_samples):
        pred_class = np.argmax(predictions[i])
        pred_activity = activity_names[pred_class]
        
        for col_idx, feature_idx in enumerate([0, 3, 6]):
            if feature_idx >= originals.shape[2]:
                feature_idx = col_idx % originals.shape[2]
            
            ax = axes[i, col_idx] if n_samples > 1 else axes[col_idx]
            
            ax.plot(originals[i, :, feature_idx], label='Original', 
                   linewidth=2, alpha=0.8, color='#2E86AB')
            ax.plot(reconstructed[i, :, feature_idx], label='Reconstructed', 
                   linestyle='--', linewidth=2, alpha=0.8, color='#A23B72')
            
            ax.set_title(f'Sample {i+1}, Feature {feature_idx+1}\nActivity: {pred_activity}',
                        fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Sensor Value')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('har_reconstruction.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: 'har_reconstruction.png'")
    plt.close()

def plot_confusion_matrix_lstm(model, test_loader, save_path='lstm_confusion_matrix.png'):
    """Plot confusion matrix for LSTM classification"""
    try:
        model.eval()
        all_preds = []
        all_labels = []
        
        print("\n" + "="*60)
        print("GENERATING LSTM CONFUSION MATRIX")
        print("="*60)
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot
        plt.figure(figsize=(10, 8))
        activity_names = ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=activity_names, yticklabels=activity_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('LSTM Model - Confusion Matrix\n(Actual vs Predicted Activities)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Activity', fontsize=12)
        plt.xlabel('Predicted Activity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ LSTM Confusion Matrix saved: '{save_path}'")
        
        # Verify file was created
        if os.path.exists(save_path):
            print(f"  File size: {os.path.getsize(save_path)} bytes")
        else:
            print(f"  ❌ ERROR: File was not created at {save_path}")
        
        plt.close()
        
        # Print classification report
        print("\nLSTM Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=activity_names, digits=3))
        
        return cm
        
    except Exception as e:
        print(f"❌ ERROR in plot_confusion_matrix_lstm: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_adversarial_confusion_matrix(adv_model, target_model, test_loader, 
                                      save_path='adversarial_confusion_matrix.png'):
    """
    Plot confusion matrix showing if LSTM predictions match when using
    original vs reconstructed inputs (tests if reconstruction preserves classification)
    """
    adv_model.eval()
    target_model.eval()
    
    print("\n" + "="*60)
    print("GENERATING ADVERSARIAL ATTACK CONFUSION MATRIX")
    print("="*60)
    print("Testing: Does LSTM predict same activity for reconstructed data?")
    
    original_preds = []
    reconstructed_preds = []
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.to(device)
            
            # Get predictions on original data
            orig_outputs = target_model(batch_data)
            _, orig_pred = torch.max(orig_outputs.data, 1)
            
            # Get predictions on reconstructed data
            target_probs = target_model.predict_proba(batch_data)
            reconstructed = adv_model(target_probs)
            recon_outputs = target_model(reconstructed)
            _, recon_pred = torch.max(recon_outputs.data, 1)
            
            original_preds.extend(orig_pred.cpu().numpy())
            reconstructed_preds.extend(recon_pred.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(original_preds, reconstructed_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    activity_names = ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=activity_names, yticklabels=activity_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Adversarial Attack - Prediction Consistency\n(Original Input vs Reconstructed Input)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('LSTM Prediction (Original Data)', fontsize=12)
    plt.xlabel('LSTM Prediction (Reconstructed Data)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Calculate consistency
    consistency = np.trace(cm) / np.sum(cm) * 100
    plt.text(0.5, -0.15, f'Prediction Consistency: {consistency:.2f}%', 
             transform=plt.gca().transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Adversarial Confusion Matrix saved: '{save_path}'")
    print(f"✓ Prediction Consistency: {consistency:.2f}%")
    print("  (Higher = reconstructed data yields same predictions)")
    plt.close()
    
    return cm, consistency

def plot_reconstruction_quality_per_activity(originals, reconstructed, predictions, 
                                             save_path='reconstruction_per_activity.png'):
    """Plot reconstruction quality metrics per activity class"""
    activity_names = ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying']
    
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics per activity
    correlations = []
    mse_scores = []
    mae_scores = []
    
    for activity_idx in range(6):
        mask = pred_classes == activity_idx
        if np.sum(mask) > 0:
            orig_subset = originals[mask].flatten()
            recon_subset = reconstructed[mask].flatten()
            
            corr = np.corrcoef(orig_subset, recon_subset)[0, 1]
            mse = mean_squared_error(orig_subset, recon_subset)
            mae = mean_absolute_error(orig_subset, recon_subset)
            
            correlations.append(corr)
            mse_scores.append(mse)
            mae_scores.append(mae)
        else:
            correlations.append(0)
            mse_scores.append(0)
            mae_scores.append(0)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Correlation
    axes[0].bar(range(6), correlations, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Activity', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Correlation', fontsize=11, fontweight='bold')
    axes[0].set_title('Reconstruction Correlation by Activity', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(activity_names, rotation=45, ha='right')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: MSE
    axes[1].bar(range(6), mse_scores, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Activity', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
    axes[1].set_title('Reconstruction MSE by Activity', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels(activity_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: MAE
    axes[2].bar(range(6), mae_scores, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Activity', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[2].set_title('Reconstruction MAE by Activity', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(6))
    axes[2].set_xticklabels(activity_names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Per-Activity Reconstruction Quality saved: '{save_path}'")
    plt.close()

def plot_feature_correlation_heatmap(originals, reconstructed, 
                                     save_path='feature_correlation_heatmap.png'):
    """Plot heatmap showing correlation between original and reconstructed for each feature"""
    n_features = originals.shape[2]
    
    # Calculate correlation for each feature
    correlations = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            orig_feat = originals[:, :, i].flatten()
            recon_feat = reconstructed[:, :, j].flatten()
            correlations[i, j] = np.corrcoef(orig_feat, recon_feat)[0, 1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=[f'Recon-{i+1}' for i in range(n_features)],
                yticklabels=[f'Orig-{i+1}' for i in range(n_features)],
                cbar_kws={'label': 'Correlation'})
    
    plt.title('Feature-wise Correlation Matrix\n(Original vs Reconstructed Features)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Reconstructed Features', fontsize=12)
    plt.ylabel('Original Features', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature Correlation Heatmap saved: '{save_path}'")
    plt.close()

def plot_error_distribution(originals, reconstructed, save_path='error_distribution.png'):
    """Plot distribution of reconstruction errors"""
    errors = reconstructed - originals
    errors_flat = errors.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error histogram
    axes[0, 0].hist(errors_flat, bins=100, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Reconstruction Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Distribution of Reconstruction Errors', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Error over time (averaged across samples and features)
    time_errors = np.mean(np.abs(errors), axis=(0, 2))
    axes[0, 1].plot(time_errors, color='#A23B72', linewidth=2)
    axes[0, 1].set_xlabel('Time Step', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Reconstruction Error Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].fill_between(range(len(time_errors)), time_errors, alpha=0.3, color='#A23B72')
    
    # Plot 3: Error by feature
    feature_errors = np.mean(np.abs(errors), axis=(0, 1))
    axes[1, 0].bar(range(len(feature_errors)), feature_errors, 
                   color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Reconstruction Error by Feature', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Scatter plot - original vs reconstructed
    sample_size = min(10000, len(errors_flat))
    sample_indices = np.random.choice(len(errors_flat), sample_size, replace=False)
    orig_sample = originals.flatten()[sample_indices]
    recon_sample = reconstructed.flatten()[sample_indices]
    
    axes[1, 1].scatter(orig_sample, recon_sample, alpha=0.3, s=1, color='#2E86AB')
    axes[1, 1].plot([orig_sample.min(), orig_sample.max()], 
                    [orig_sample.min(), orig_sample.max()], 
                    'r--', linewidth=2, label='Perfect Reconstruction')
    axes[1, 1].set_xlabel('Original Value', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Reconstructed Value', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Original vs Reconstructed Values', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Error Distribution Plot saved: '{save_path}'")
    plt.close()

def plot_training_history(train_losses, val_accs, adv_losses, 
                          save_path='training_history.png'):
    """Plot training history for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: LSTM training
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    epochs_lstm = range(1, len(train_losses) + 1)
    line1 = ax1.plot(epochs_lstm, train_losses, 'b-', linewidth=2, label='Training Loss')
    line2 = ax1_twin.plot(epochs_lstm, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold', color='b')
    ax1_twin.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='r')
    ax1.set_title('LSTM Model Training History', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Plot 2: Adversarial training
    axes[1].plot(range(1, len(adv_losses) + 1), adv_losses, 
                color='#A23B72', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Reconstruction Loss (MSE)', fontsize=11, fontweight='bold')
    axes[1].set_title('Adversarial Model Training History', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].fill_between(range(1, len(adv_losses) + 1), adv_losses, alpha=0.3, color='#A23B72')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training History Plot saved: '{save_path}'")
    plt.close()

def list_generated_files():
    """List all generated plot files with their sizes"""
    plot_files = [
        'lstm_confusion_matrix.png',
        'adversarial_confusion_matrix.png',
        'har_reconstruction.png',
        'reconstruction_per_activity.png',
        'feature_correlation_heatmap.png',
        'error_distribution.png',
        'training_history.png'
    ]
    
    print("\n" + "="*60)
    print("📁 GENERATED FILES")
    print("="*60)
    
    for filename in plot_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename:40s} ({size:,} bytes)")
        else:
            print(f"✗ {filename:40s} (NOT FOUND)")
    
    print("="*60)

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================
def main():
    print("="*60)
    print("MODEL INVERSION ATTACK ON LSTM")
    print("UCI Human Activity Recognition Dataset")
    print("="*60)
    
    # Test matplotlib
    print("\nTesting matplotlib setup...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title('Test Plot')
        plt.savefig('test_plot.png', dpi=100)
        plt.close()
        if os.path.exists('test_plot.png'):
            print("✓ Matplotlib working correctly!")
            os.remove('test_plot.png')
        else:
            print("✗ Warning: Plot file not created!")
    except Exception as e:
        print(f"✗ Matplotlib error: {e}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    LSTM_EPOCHS = 30
    ADV_EPOCHS = 50
    SEQ_LENGTH = 128
    N_FEATURES = 9
    
    # Download and load dataset
    data_dir = download_uci_har_data()
    X, y = load_uci_har_data(data_dir)
    
    # Split data
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create datasets
    train_dataset = UCIHARDataset(X.iloc[train_idx], y.iloc[train_idx], 
                                  seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    val_dataset = UCIHARDataset(X.iloc[val_idx], y.iloc[val_idx], 
                                seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    test_dataset = UCIHARDataset(X.iloc[test_idx], y.iloc[test_idx], 
                                 seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # Initialize LSTM model
    lstm_model = LSTMClassifier(input_size=N_FEATURES, hidden_size=64, 
                                num_layers=2, num_classes=6).to(device)
    
    print(f"\nLSTM Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Train LSTM
    train_losses, val_accs = train_lstm(lstm_model, train_loader, val_loader, 
                                        epochs=LSTM_EPOCHS, lr=0.001)
    
    # Evaluate LSTM
    lstm_accuracy = evaluate_lstm(lstm_model, test_loader)
    
    # Initialize adversarial model
    adv_model = AdversarialReconstructor(num_classes=6, seq_length=SEQ_LENGTH, 
                                        n_features=N_FEATURES, 
                                        hidden_size=128).to(device)
    
    print(f"\nAdversarial Model Parameters: {sum(p.numel() for p in adv_model.parameters()):,}")
    
    # Train adversarial model
    adv_losses = train_adversarial(adv_model, lstm_model, train_loader, 
                                   epochs=ADV_EPOCHS, lr=0.001)
    
    # Evaluate adversarial model
    originals, reconstructed, predictions, mse, mae, corr = evaluate_adversarial(
        adv_model, lstm_model, test_loader
    )
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Plot 1: LSTM confusion matrix
    plot_confusion_matrix_lstm(lstm_model, test_loader, 'lstm_confusion_matrix.png')
    
    # Plot 2: Adversarial confusion matrix  
    cm_adv, consistency = plot_adversarial_confusion_matrix(adv_model, lstm_model, test_loader, 
                                                           'adversarial_confusion_matrix.png')
    
    # Plot 3: Sample reconstructions
    visualize_reconstruction(originals, reconstructed, predictions, n_samples=3)
    
    # Plot 4: Per-activity reconstruction quality
    plot_reconstruction_quality_per_activity(originals, reconstructed, predictions, 
                                           'reconstruction_per_activity.png')
    
    # Plot 5: Feature correlation heatmap
    plot_feature_correlation_heatmap(originals, reconstructed, 'feature_correlation_heatmap.png')
    
    # Plot 6: Error distribution analysis
    plot_error_distribution(originals, reconstructed, 'error_distribution.png')
    
    # Plot 7: Training history
    plot_training_history(train_losses, val_accs, adv_losses, 'training_history.png')
    
    # List all generated files
    list_generated_files()
    
    # Final Summary
    print("\n" + "="*60)
    print("🎯 EXPERIMENT RESULTS")
    print("="*60)
    print(f"Target LSTM Accuracy: {lstm_accuracy:.2f}%")
    print(f"\nAdversarial Reconstruction:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Prediction Consistency: {consistency:.2f}%")
    
    print("\n" + "="*60)
    print("📊 GENERATED VISUALIZATIONS")
    print("="*60)

    
    print("\n" + "="*60)
    print("🔒 SECURITY IMPLICATIONS")
    print("="*60)
    print(f"\n✓ Attack Success: {corr:.1%} correlation means adversary")
    print("  can recover significant information about private inputs.")
    print(f"\n✓ Prediction Preservation: {consistency:.1f}% of predictions")

    print("="*60)

if __name__ == "__main__":
    main()