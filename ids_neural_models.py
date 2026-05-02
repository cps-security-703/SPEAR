"""
ids_neural_models.py
====================
Standalone module containing all neural-network IDS class definitions.
"""

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN     = 10
FEATURE_DIM = 14
DEVICE      = torch.device("cpu")


# ── LSTM-IDS with temporal attention ─────────────────────────────────────────

class LSTMIDSBenchmark(nn.Module):
    """Enhanced LSTM IDS with temporal self-attention."""

    def __init__(self, input_size=FEATURE_DIM, hidden_size=256,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_fc   = nn.Linear(hidden_size, 1)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),          nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )
        self.anomaly_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        attn_scores  = self.attn_fc(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context      = (attn_weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        last  = h_n[-1]
        fused = self.attn_norm(context + last)
        class_logits  = self.classifier(fused)
        anomaly_score = self.anomaly_scorer(fused)
        return class_logits, anomaly_score

    def predict_proba(self, x):
        class_logits, _ = self.forward(x)
        return torch.softmax(class_logits, dim=1)[:, 1]


# ── Transformer IDS ───────────────────────────────────────────────────────────

class TransformerIDS(nn.Module):
    """Temporal Transformer for IDS — self-attention over the 10-step feature window."""

    def __init__(self, feature_dim: int = FEATURE_DIM, seq_len: int = SEQ_LEN,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_emb    = nn.Embedding(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos    = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h      = self.input_proj(x) + self.pos_emb(pos)
        h      = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class TransformerIDSWrapper:
    """Sklearn-compatible, pickle-safe wrapper for TransformerIDS."""

    def __init__(self, model: nn.Module,
                 seq_len: int = SEQ_LEN, feature_dim: int = FEATURE_DIM):
        self._model       = model
        self._seq_len     = seq_len
        self._feature_dim = feature_dim

    def predict_proba(self, X_flat: np.ndarray) -> np.ndarray:
        n    = len(X_flat)
        Xseq = torch.FloatTensor(
            X_flat.reshape(n, self._seq_len, self._feature_dim)).to(DEVICE)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(Xseq)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs   # [N, 2]


# ── Autoencoder IDS ───────────────────────────────────────────────────────────

class AutoencoderIDS(nn.Module):
    """Reconstruction-based anomaly detector (unsupervised, trained on benign only)."""

    def __init__(self, input_dim: int = SEQ_LEN * FEATURE_DIM, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 256),        nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.forward(x)) ** 2).mean(dim=1)
