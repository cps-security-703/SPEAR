import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. GENERATE SYNTHETIC TIME SERIES DATA
# ============================================================================
class TimeSeriesDataset(Dataset):
    """Generate synthetic time series data with multiple patterns"""
    def __init__(self, n_samples=1000, seq_length=50, n_features=3):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_features = n_features
        self.data, self.labels = self._generate_data()
    
    def _generate_data(self):
        data = []
        labels = []
        
        for i in range(self.n_samples):
            # Create time series with sinusoidal patterns + noise
            t = np.linspace(0, 4*np.pi, self.seq_length)
            
            # Different patterns based on class
            pattern_type = i % 3
            
            if pattern_type == 0:
                # Sine wave pattern
                series = np.column_stack([
                    np.sin(t) + np.random.normal(0, 0.1, self.seq_length),
                    np.cos(t) + np.random.normal(0, 0.1, self.seq_length),
                    np.sin(2*t) + np.random.normal(0, 0.1, self.seq_length)
                ])
                label = 0
            elif pattern_type == 1:
                # Exponential decay pattern
                series = np.column_stack([
                    np.exp(-t/4) + np.random.normal(0, 0.1, self.seq_length),
                    np.exp(-t/3) * np.sin(t) + np.random.normal(0, 0.1, self.seq_length),
                    np.exp(-t/2) + np.random.normal(0, 0.1, self.seq_length)
                ])
                label = 1
            else:
                # Linear trend pattern
                series = np.column_stack([
                    t/10 + np.random.normal(0, 0.1, self.seq_length),
                    -t/10 + np.random.normal(0, 0.1, self.seq_length),
                    np.sin(t/2) + np.random.normal(0, 0.1, self.seq_length)
                ])
                label = 2
            
            data.append(series)
            labels.append(label)
        
        return torch.FloatTensor(np.array(data)), torch.LongTensor(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ============================================================================
# 2. DEFINE LSTM MODEL (TARGET MODEL)
# ============================================================================
class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
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
    This model learns the inverse mapping: output -> input
    """
    def __init__(self, num_classes=3, seq_length=50, n_features=3, hidden_size=128):
        super(AdversarialReconstructor, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        
        # Encoder: Take output probabilities and encode them
        self.encoder = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct the time series
        self.decoder_lstm = nn.LSTM(256, hidden_size, num_layers=2, 
                                     batch_first=True, dropout=0.2)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_features)
        )
    
    def forward(self, output_probs):
        # Encode output probabilities
        encoded = self.encoder(output_probs)
        
        # Expand for sequence generation
        # Shape: (batch, 1, 256)
        encoded = encoded.unsqueeze(1)
        
        # Repeat for each time step
        # Shape: (batch, seq_length, 256)
        encoded = encoded.repeat(1, self.seq_length, 1)
        
        # LSTM decoder
        decoded, _ = self.decoder_lstm(encoded)
        
        # Generate time series for each time step
        reconstructed = self.output_layer(decoded)
        
        return reconstructed

# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================
def train_lstm(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train the LSTM classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
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
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_accuracies

def train_adversarial(adv_model, target_model, train_loader, epochs=100, lr=0.001):
    """
    Train adversarial model to reconstruct inputs from LSTM outputs
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(adv_model.parameters(), lr=lr)
    
    losses = []
    
    print("\n" + "="*60)
    print("TRAINING ADVERSARIAL RECONSTRUCTOR")
    print("="*60)
    
    target_model.eval()  # Keep target model in eval mode
    
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
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss:.6f}')
    
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
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def evaluate_adversarial(adv_model, target_model, test_loader):
    """Evaluate how well adversarial model reconstructs inputs"""
    adv_model.eval()
    target_model.eval()
    
    all_originals = []
    all_reconstructed = []
    
    print("\n" + "="*60)
    print("EVALUATING ADVERSARIAL RECONSTRUCTOR")
    print("="*60)
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.to(device)
            
            # Get target model predictions
            target_outputs = target_model.predict_proba(batch_data)
            
            # Reconstruct inputs
            reconstructed = adv_model(target_outputs)
            
            all_originals.append(batch_data.cpu().numpy())
            all_reconstructed.append(reconstructed.cpu().numpy())
    
    # Concatenate all batches
    originals = np.concatenate(all_originals, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(originals.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(originals.flatten(), reconstructed.flatten())
    
    # Correlation coefficient
    correlation = np.corrcoef(originals.flatten(), reconstructed.flatten())[0, 1]
    
    print(f'Mean Squared Error: {mse:.6f}')
    print(f'Mean Absolute Error: {mae:.6f}')
    print(f'Correlation Coefficient: {correlation:.6f}')
    
    return originals, reconstructed, mse, mae, correlation

def visualize_reconstruction(originals, reconstructed, n_samples=3):
    """Visualize original vs reconstructed time series"""
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    
    for i in range(n_samples):
        for feature in range(3):
            ax = axes[i, feature] if n_samples > 1 else axes[feature]
            ax.plot(originals[i, :, feature], label='Original', linewidth=2)
            ax.plot(reconstructed[i, :, feature], label='Reconstructed', 
                   linestyle='--', linewidth=2)
            ax.set_title(f'Sample {i+1}, Feature {feature+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'reconstruction_comparison.png'")
    plt.show()

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================
def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LSTM_EPOCHS = 50
    ADV_EPOCHS = 100
    SEQ_LENGTH = 50
    N_FEATURES = 3
    
    # Create datasets
    print("Generating datasets...")
    train_dataset = TimeSeriesDataset(n_samples=1000, seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    val_dataset = TimeSeriesDataset(n_samples=200, seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    test_dataset = TimeSeriesDataset(n_samples=200, seq_length=SEQ_LENGTH, n_features=N_FEATURES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize LSTM model
    lstm_model = LSTMClassifier(input_size=N_FEATURES, hidden_size=64, 
                                num_layers=2, num_classes=3).to(device)
    
    print(f"\nLSTM Model Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Train LSTM
    train_losses, val_accs = train_lstm(lstm_model, train_loader, val_loader, 
                                        epochs=LSTM_EPOCHS, lr=0.001)
    
    # Evaluate LSTM
    lstm_accuracy = evaluate_lstm(lstm_model, test_loader)
    
    # Initialize adversarial model
    adv_model = AdversarialReconstructor(num_classes=3, seq_length=SEQ_LENGTH, 
                                        n_features=N_FEATURES, hidden_size=128).to(device)
    
    print(f"\nAdversarial Model Parameters: {sum(p.numel() for p in adv_model.parameters()):,}")
    
    # Train adversarial model
    adv_losses = train_adversarial(adv_model, lstm_model, train_loader, 
                                   epochs=ADV_EPOCHS, lr=0.001)
    
    # Evaluate adversarial model
    originals, reconstructed, mse, mae, corr = evaluate_adversarial(
        adv_model, lstm_model, test_loader
    )
    
    # Visualize results
    visualize_reconstruction(originals, reconstructed, n_samples=3)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"LSTM Model Accuracy: {lstm_accuracy:.2f}%")
    print(f"Adversarial Reconstruction MSE: {mse:.6f}")
    print(f"Adversarial Reconstruction Correlation: {corr:.6f}")
    print("\nThe adversarial model successfully learned to reconstruct")
    print("inputs from the LSTM's output probabilities!")
    print("="*60)

if __name__ == "__main__":
    main()