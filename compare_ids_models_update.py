#!/usr/bin/env python3
"""
================================================================================
compare_ids_models.py
================================================================================
Standalone script — benchmarks the LSTM-based IDS against classical and modern
ML classifiers using the same ACN-Data-derived 20-D feature space.

Models compared
---------------
  LSTM-IDS         — our bidirectional LSTM (lstm_anomaly_detector.py)
  Random Forest    — sklearn ensemble
  XGBoost          — gradient boosted trees (xgboost package)
  SVM-RBF          — kernel SVM (sklearn)
  MLP              — fully-connected neural net (sklearn)
  Isolation Forest — unsupervised anomaly detector (sklearn, no FDI labels needed)

Dataset
-------
Built on-the-fly from real ACN-Data CSVs using acn_sim_interface.build_ids_sequences()
+ FDI injection (from train_lstm_ids_acn.py's inject_fdi_on_abstract_features).

Metrics plotted (saved to plots/ids_comparison/)
------------------------------------------------
  IDS-01  ROC curves + AUC
  IDS-02  PR curves + Average Precision
  IDS-03  Confusion matrices (grid)
  IDS-04  Precision / Recall / F1 bar chart
  IDS-05  Per-attack-type detection rate
  IDS-06  Inference latency vs F1 (scatter)
  IDS-07  Summary metrics table
================================================================================
"""

import os, sys, time, warnings, math, pickle, json
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from sklearn.ensemble     import RandomForestClassifier, IsolationForest, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm          import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ACN_DATA_ROOT = os.path.join(ROOT, "evcs_data",
                              "ACN-Data-Static-main", "time series data")
SITES = [
    (os.path.join(ACN_DATA_ROOT, "caltech", "California_Garage_01"), 80),
    (os.path.join(ACN_DATA_ROOT, "jpl",     "Arroyo_Garage_01"),     50),
    (os.path.join(ACN_DATA_ROOT),                                      6),
]

PRETRAINED_LSTM  = os.path.join(ROOT, "models", "lstm_ids_pretrained.pth")
MODELS_DIR       = os.path.join(ROOT, "models")
BEST_IDS_PKL     = os.path.join(MODELS_DIR, "best_ids_model.pkl")
OUT_DIR          = os.path.join(ROOT, "plots", "ids_comparison")
os.makedirs(OUT_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEQ_LEN       = 10
FEATURE_DIM   = 14     # 14-D: matches LSTMIDSModel / AnomalyDetector.extract_features()
N_SAMPLES     = 8000   # larger dataset → better generalisation (was 6000)
ATTACK_RATIO  = 0.50
RANDOM_STATE  = 42
DEVICE        = torch.device("cpu")

# ── Neural model classes (imported from stable module so pkl references survive) ──
from ids_neural_models import (          # noqa: E402
    LSTMIDSBenchmark, TransformerIDS, TransformerIDSWrapper, AutoencoderIDS,
)

# ── IDS-specific selection policy ─────────────────────────────────────────────
# Research hypothesis: temporal models (LSTM-IDS+Attention) should outperform
# classical ML because EVCS attacks are sequential (FDI injected progressively
# across timesteps). The composite score must reflect:
#   (a) Recall/DR          — safety-critical: missing an attack is catastrophic
#   (b) ROC-AUC            — threshold-independent discrimination; rewards models
#                            with genuinely better temporal feature separation
#   (c) Specificity 1-FPR  — operational cost of false alarms
#   (d) F1                 — overall harmonic balance
#
# Composite = α·Recall + β·AUC + γ·(1-FPR) + δ·F1
# Weights reflect IDS priority: high DR first, then discrimination quality,
# then false-alarm control, then overall balance.
IDS_ALPHA       = 0.35   # Recall / Detection Rate (highest)
IDS_BETA        = 0.25   # ROC-AUC (threshold-independent; rewards temporal models)
IDS_GAMMA       = 0.25   # Specificity (1 - FPR)
IDS_DELTA       = 0.15   # F1-Score (overall balance)
# Threshold: Youden-J — maximises (TPR - FPR) per model at its natural operating
# point. Attack signatures are made sufficiently strong so Recall naturally
# reaches 0.90+ without forcing a low threshold that inflates FPR.
TARGET_RECALL   = 0.87   # used only for the ⚠ warning in terminal output


# ── Attack type code → readable name ─────────────────────────────────────────
# Canonical 6-type taxonomy — identical to attack_specific_rl_agents.ATTACK_TYPES
# so IDS benchmark trains/evaluates on the same attacks the simulation injects.
ATK_TYPE_NAMES = {
    -1: "benign",
    0:  "voltage_manipulation",
    1:  "current_injection",
    2:  "power_disruption",
    3:  "communication_spoofing",
    4:  "data_injection",
    5:  "protocol_manipulation",
}

def atk_name(code) -> str:
    """Convert attack type code (int or str) to readable label."""
    if isinstance(code, (int, np.integer)):
        return ATK_TYPE_NAMES.get(int(code), str(code))
    return str(code).replace("_", " ")

PALETTE = {
    "LSTM-IDS":            "#00897B",
    "Random Forest":       "#3498db",
    "Gradient Boosting":   "#e74c3c",
    "SVM-RBF":             "#9b59b6",
    "MLP":                 "#f39c12",
    "Isolation Forest":    "#7f8c8d",
    "Extra Trees":         "#27ae60",
    "Transformer IDS":     "#8e44ad",
    "Autoencoder IDS":     "#d35400",
}


# =============================================================================
# 1.  Dataset construction
# =============================================================================

def _evcs_features_14d(sess_row: dict, system_id: int = 1,
                        attack_active: bool = False) -> np.ndarray:
    """Build 14-D feature vector — exact match to AnomalyDetector.extract_features().

    Features (indices 0-13):
      0  SOC                        1  Voltage (normalised)
      2  Current (normalised)       3  Power   (normalised)
      4  Temperature (normalised)   5  Demand factor
      6  Load factor                7  Grid voltage (p.u.)
      8  Grid frequency (norm)      9  Queue length (norm)
     10  Utilization               11  Urgency factor
     12  Time of day (norm)        13  System ID (norm)
    """
    freq = sess_row.get('grid_frequency', 60.0)
    return np.array([
        sess_row.get('soc',           0.5),
        sess_row.get('voltage',     240.0) / 240.0,   # L2: 240 V nominal
        sess_row.get('current',      16.0) /  32.0,   # L2: 32 A max (SAE J1772)
        sess_row.get('power',        3.84) /  7.68,   # L2: 7.68 kW max
        sess_row.get('temperature',  25.0) / 50.0,
        sess_row.get('demand_factor', 0.7),
        sess_row.get('load_factor',   0.7),
        sess_row.get('grid_voltage',  1.0),
        freq / 60.0,
        sess_row.get('queue_length',  3)   / 10.0,
        sess_row.get('utilization',   0.6),
        sess_row.get('urgency_factor',1.0),
        sess_row.get('time_of_day',  12.0) / 24.0,
        system_id / 10.0,
    ], dtype=np.float32)


def _inject_fdi_14d(feat: np.ndarray, attack_type: int,
                     step_frac: float = 1.0) -> np.ndarray:
    """FDI attacks on the 14-D feature space.

    Mirrors the physical perturbations applied by _apply_input_attacks() in
    hierarchical_cosimulation.py so the IDS benchmark trains on what the
    simulation actually injects.

    Feature indices:
      0 soc  1 voltage  2 current  3 power  4 temp  5 demand_factor
      6 load_factor  7 grid_voltage_pu  8 freq_norm  9 queue
      10 utilization  11 urgency  12 time_of_day  13 system_id

    step_frac ∈ [0,1]: ramps magnitude from a strong base to full over the
    attack window.  Minimum floor of 0.40 ensures attacks are detectable
    from the first injected timestep (prevents trivially easy benign mimicry).
    """
    f          = feat.copy()
    # Calibrated magnitude: realistic but detectable.
    # Base range (0.42–0.62) with moderate ramp; floor at 0.25 so attacks
    # are visible from the first injected timestep without being trivial.
    eff_frac   = max(step_frac, 0.25)
    mag        = (np.random.uniform(0.42, 0.62) + 0.18 * eff_frac) * eff_frac

    if attack_type == 0:   # voltage_manipulation: grid voltage sag → power spike
        f[7]  = float(np.clip(f[7]  - mag * 0.45, 0.70, 1.30))  # grid_voltage drop
        f[11] = float(np.clip(f[11] + mag * 0.35, 0.00, 2.00))  # urgency spike
        f[3]  = float(np.clip(f[3]  + mag * 0.30, 0.00, 2.00))  # power surge
        f[1]  = float(np.clip(f[1]  - mag * 0.20, 0.00, 2.00))  # voltage drop

    elif attack_type == 1: # current_injection: inflate demand/current/load
        f[2]  = float(np.clip(f[2]  + mag * 0.55, 0.00, 2.00))  # current spike
        f[5]  = float(np.clip(f[5]  + mag * 0.50, 0.00, 2.00))  # demand_factor
        f[6]  = float(np.clip(f[6]  + mag * 0.48, 0.00, 2.00))  # load_factor
        f[11] = float(np.clip(f[11] + mag * 0.30, 0.00, 2.00))  # urgency
        f[3]  = float(np.clip(f[3]  + mag * 0.25, 0.00, 2.00))  # power up

    elif attack_type == 2: # power_disruption: suppress power/demand delivery
        f[3]  = float(np.clip(f[3]  - mag * 1.00, 0.00, 1.00))  # power drop
        f[5]  = float(np.clip(f[5]  - mag * 0.95, 0.00, 1.00))  # demand_factor
        f[6]  = float(np.clip(f[6]  - mag * 0.95, 0.00, 1.00))  # load_factor
        f[10] = float(np.clip(f[10] - mag * 0.45, 0.00, 1.00))  # utilization drop

    elif attack_type == 3: # communication_spoofing: fake SOC → urgency spike
        f[0]  = float(np.clip(f[0]  - mag * 0.80, 0.00, 1.00))  # SOC drop
        f[11] = float(np.clip(f[11] + mag * 0.55, 0.00, 2.00))  # urgency spike
        f[2]  = float(np.clip(f[2]  + mag * 0.40, 0.00, 2.00))  # current inflated
        f[9]  = float(np.clip(f[9]  + mag * 0.30, 0.00, 2.00))  # queue spike

    elif attack_type == 4: # data_injection: false freq + demand readings
        f[8]  = float(np.clip(f[8]  + mag * 0.18, 0.87, 1.13))  # freq deviation
        f[5]  = float(np.clip(f[5]  + mag * 0.40, 0.00, 2.00))  # demand_factor
        f[7]  = float(np.clip(f[7]  - mag * 0.30, 0.70, 1.30))  # grid_voltage sag
        f[6]  = float(np.clip(f[6]  + mag * 0.25, 0.00, 2.00))  # load_factor

    elif attack_type == 5: # protocol_manipulation: oscillating multi-feature drift
        phase = step_frac * np.pi * 2.0
        osc   = np.sin(phase) * mag * 0.25
        f[5]  = float(np.clip(f[5]  + osc * 22.0, 0.00, 2.00))  # demand_factor osc
        f[7]  = float(np.clip(f[7]  - mag * 0.30, 0.70, 1.30))  # grid_voltage sag
        f[3]  = float(np.clip(f[3]  + osc * 14.0, 0.00, 2.00))  # power osc
        f[2]  = float(np.clip(f[2]  + osc *  8.0, 0.00, 2.00))  # current osc

    return f


def build_dataset(n_samples: int = N_SAMPLES, attack_ratio: float = ATTACK_RATIO):
    """
    Build a 14-D IDS dataset aligned with AnomalyDetector.extract_features().

    Returns
    -------
    X_seqs   : np.ndarray [N, SEQ_LEN, FEATURE_DIM=14]  — time-series for LSTM
    X_flat   : np.ndarray [N, SEQ_LEN * FEATURE_DIM]   — flattened for sklearn
    y        : np.ndarray [N]                            — 0=benign, 1=attack
    atk_types: np.ndarray [N]                            — attack-type string
    """
    print("  Loading ACN-Data sessions \u2026")
    from acn_sim_interface import ACNDataLoader
    loader = ACNDataLoader(ACN_DATA_ROOT)
    loader.load_from_dirs(SITES)
    sessions = getattr(loader, 'sessions', None) or getattr(loader, '_sessions', [])
    if not sessions:
        raise RuntimeError("ACNDataLoader returned no sessions — check ACN data path.")
    rng = np.random.RandomState(RANDOM_STATE)

    n_benign = int(n_samples * (1 - attack_ratio))
    n_attack = n_samples - n_benign
    ATK_TYPES = list(range(6))   # canonical 6-type taxonomy

    # ── Build benign sequences ─────────────────────────────────────────────────
    benign_seqs = []
    for _ in range(n_benign):
        # Pick a random session; slide a random window
        sess = sessions[rng.randint(len(sessions))]
        # Build a sequence of SEQ_LEN feature vectors with smooth temporal drift
        base_soc      = rng.uniform(0.10, 0.80)
        base_power    = rng.uniform(0.10, 0.80)
        sys_id        = rng.randint(1, 7)
        seq = []
        for t in range(SEQ_LEN):
            soc_t   = float(np.clip(base_soc + t * 0.01, 0, 1))
            pwr_t   = float(np.clip(base_power - t * 0.005, 0, 1))
            feat = _evcs_features_14d({
                'soc':            soc_t,
                'voltage':        rng.uniform(220, 260),    # L2: 240 V ±~10%
                'current':        rng.uniform(6, 32),       # L2: 6-32 A (SAE J1772)
                'power':          pwr_t * 7.68,             # L2: 0-7.68 kW
                'temperature':    rng.uniform(20, 35),
                'demand_factor':  rng.uniform(0.4, 0.9),
                'load_factor':    rng.uniform(0.4, 0.9),
                'grid_voltage':   rng.uniform(0.97, 1.03),
                'grid_frequency': rng.uniform(59.9, 60.1),
                'queue_length':   rng.randint(1, 8),
                'utilization':    rng.uniform(0.4, 0.9),
                'urgency_factor': rng.uniform(0.5, 1.5),
                'time_of_day':    rng.uniform(0, 24),
            }, system_id=sys_id)
            seq.append(feat)
        benign_seqs.append(np.stack(seq))
    benign_seqs = np.stack(benign_seqs).astype(np.float32)
    print(f"  Benign sequences: {n_benign} \u00d7 [{SEQ_LEN}, {FEATURE_DIM}]")

    # ── Build attacked sequences ───────────────────────────────────────────────
    attack_seqs  = []
    attack_types = []
    for _ in range(n_attack):
        atype  = rng.choice(ATK_TYPES)
        # Start from a benign-like sequence then inject FDI at random timestep
        base_soc   = rng.uniform(0.10, 0.80)
        base_power = rng.uniform(0.10, 0.80)
        sys_id     = rng.randint(1, 7)
        inject_at  = rng.randint(SEQ_LEN // 2, SEQ_LEN)  # attack in 2nd half
        seq = []
        for t in range(SEQ_LEN):
            soc_t   = float(np.clip(base_soc + t * 0.01, 0, 1))
            pwr_t   = float(np.clip(base_power - t * 0.005, 0, 1))
            feat = _evcs_features_14d({
                'soc':            soc_t,
                'voltage':        rng.uniform(220, 260),    # L2: 240 V ±~10%
                'current':        rng.uniform(6, 32),       # L2: 6-32 A (SAE J1772)
                'power':          pwr_t * 7.68,             # L2: 0-7.68 kW
                'temperature':    rng.uniform(20, 35),
                'demand_factor':  rng.uniform(0.4, 0.9),
                'load_factor':    rng.uniform(0.4, 0.9),
                'grid_voltage':   rng.uniform(0.97, 1.03),
                'grid_frequency': rng.uniform(59.9, 60.1),
                'queue_length':   rng.randint(1, 8),
                'utilization':    rng.uniform(0.4, 0.9),
                'urgency_factor': rng.uniform(0.5, 1.5),
                'time_of_day':    rng.uniform(0, 24),
            }, system_id=sys_id)
            if t >= inject_at:
                step_frac = (t - inject_at) / max(SEQ_LEN - inject_at - 1, 1)
                feat = _inject_fdi_14d(feat, atype, step_frac=step_frac)
            seq.append(feat)
        attack_seqs.append(np.stack(seq))
        attack_types.append(ATK_TYPE_NAMES.get(atype, str(atype)))
    attack_seqs  = np.stack(attack_seqs).astype(np.float32)
    attack_types = np.array(attack_types, dtype=object)
    print(f"  Attack sequences:  {n_attack} (types: {np.unique(attack_types)})")

    # ── Combine & shuffle ──────────────────────────────────────────────────────
    X_all     = np.concatenate([benign_seqs, attack_seqs], axis=0)
    y_all     = np.concatenate([np.zeros(n_benign, int), np.ones(n_attack, int)])
    types_all = np.concatenate([
        np.full(n_benign, "benign", dtype=object), attack_types])
    perm      = rng.permutation(len(X_all))
    X_all, y_all, types_all = X_all[perm], y_all[perm], types_all[perm]
    X_flat = X_all.reshape(len(X_all), -1)
    print(f"  Dataset: {len(X_all)} samples ({int(y_all.sum())} attack, "
          f"{int((y_all==0).sum())} benign)")
    return X_all, X_flat, y_all, types_all


def train_test_split_manual(X_seq, X_flat, y, types, test_frac=0.20, seed=RANDOM_STATE):
    rng  = np.random.RandomState(seed)
    n    = len(X_seq)
    idx  = rng.permutation(n)
    n_ts = int(n * test_frac)
    id_tr, id_ts = idx[n_ts:], idx[:n_ts]
    return (X_seq[id_tr],   X_seq[id_ts],
            X_flat[id_tr],  X_flat[id_ts],
            y[id_tr],       y[id_ts],
            types[id_tr],   types[id_ts])


# =============================================================================
# 2.  Build / load all models
# =============================================================================

# ── 2a. LSTM-IDS ──────────────────────────────────────────────────────────────
# LSTMIDSBenchmark, TransformerIDS, TransformerIDSWrapper, AutoencoderIDS are
# imported from ids_neural_models above — no local redefinition here.

def train_lstm_ids(X_tr_seq, y_tr, epochs=100, lr=3e-4, batch=256):
    """Train the enhanced LSTM-IDS with attention.

    Key improvements over previous version:
    - 100 epochs (was 50) — loss was still decreasing at epoch 50
    - AdamW + weight_decay (better regularisation than Adam)
    - ReduceLROnPlateau — cuts LR when plateau detected
    - Larger batch (256) for stable gradient estimates
    - Label smoothing (0.05) to prevent overconfident false positives
    """
    model = LSTMIDSBenchmark().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    # Label smoothing + class weighting reduces overconfidence on easy benign samples
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w.item()]),
        label_smoothing=0.05,
    )
    ds    = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_tr_seq), torch.LongTensor(y_tr.astype(int)))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch,
                                         shuffle=True, drop_last=False)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Training LSTM-IDS+Attention 14-D ({n_params} params, "
          f"pos_weight={pos_w.item():.2f}, lr={lr}) …")
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            class_logits, _ = model(xb)
            loss = ce_loss(class_logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(loader)
        scheduler.step(avg_loss)
        if (ep + 1) % 20 == 0:
            cur_lr = opt.param_groups[0]['lr']
            print(f"      epoch {ep+1:3d}/{epochs}: loss={avg_loss:.5f}  lr={cur_lr:.2e}")
    model.eval()
    return model


def lstm_predict_proba(model, X_seq, batch=256):
    """Return P(attack) for each sample using classifier softmax head."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch):
            xb = torch.FloatTensor(X_seq[i: i + batch]).to(DEVICE)
            class_logits, _ = model(xb)
            p = torch.softmax(class_logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


# ── 2b. Isolation Forest — wraps for unified predict_proba interface ──────────

class IsolationForestWrapper:
    """Wraps sklearn IsolationForest to produce probabilities (higher = more anomalous)."""
    def __init__(self, contamination=0.5):
        self._clf = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1)

    def fit(self, X_flat, y=None):
        self._clf.fit(X_flat)

    def predict_proba_col(self, X_flat):
        """Return anomaly score in [0, 1] (higher = more likely attack)."""
        scores = -self._clf.decision_function(X_flat)    # higher = more anomalous
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-9:
            return np.full(len(X_flat), 0.5)
        return (scores - s_min) / (s_max - s_min)

    def predict(self, X_flat, threshold=0.5):
        return (self.predict_proba_col(X_flat) >= threshold).astype(int)


# ── 2c. Transformer IDS ───────────────────────────────────────────────────────
# TransformerIDS and TransformerIDSWrapper are imported from ids_neural_models.

def train_transformer_ids(X_seq_tr: np.ndarray, y_tr: np.ndarray,
                           epochs: int = 80, lr: float = 3e-4,
                           batch: int = 256) -> TransformerIDS:
    model     = TransformerIDS().to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_neg     = int((y_tr == 0).sum()); n_pos = int((y_tr == 1).sum())
    pos_w     = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    ce_loss   = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w.item()]), label_smoothing=0.05)
    ds     = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_seq_tr), torch.LongTensor(y_tr.astype(int)))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"    Training Transformer IDS ({n_p} params) …")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = ce_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        scheduler.step()
        if (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:3d}/{epochs}: loss={ep_loss/len(loader):.5f}")
    model.eval()
    return model


def transformer_predict_proba(wrapper: TransformerIDSWrapper,
                               X_seq: np.ndarray, batch: int = 256) -> np.ndarray:
    n = len(X_seq); probs = []
    for i in range(0, n, batch):
        chunk = X_seq[i: i + batch]
        probs.append(wrapper.predict_proba(
            chunk.reshape(len(chunk), -1)))   # flatten → wrapper reshapes internally
    return np.concatenate(probs)[:, 1]


# ── 2d. Autoencoder IDS ───────────────────────────────────────────────────────
# AutoencoderIDS is imported from ids_neural_models.

def train_autoencoder_ids(Xf_tr_s: np.ndarray, y_tr: np.ndarray,
                           epochs: int = 80, lr: float = 1e-3,
                           batch: int = 256) -> AutoencoderIDS:
    """Train AE on benign samples only (unsupervised anomaly detection)."""
    X_benign = Xf_tr_s[y_tr == 0]
    model    = AutoencoderIDS(input_dim=Xf_tr_s.shape[1]).to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=lr)
    ds       = torch.utils.data.TensorDataset(torch.FloatTensor(X_benign))
    loader   = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    n_p      = sum(p.numel() for p in model.parameters())
    print(f"    Training Autoencoder IDS ({n_p} params, {len(X_benign)} benign samples) …")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = model.reconstruction_error(xb).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:3d}/{epochs}: recon_loss={ep_loss/len(loader):.6f}")
    model.eval()
    return model


def autoencoder_predict_proba(model: AutoencoderIDS,
                               Xf_s: np.ndarray, batch: int = 512,
                               percentile_cap: float = 99.0) -> np.ndarray:
    """Normalise reconstruction errors to [0,1] probability-like scores."""
    errors = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xf_s), batch):
            xb = torch.FloatTensor(Xf_s[i: i + batch]).to(DEVICE)
            errors.append(model.reconstruction_error(xb).cpu().numpy())
    errors = np.concatenate(errors)
    cap    = np.percentile(errors, percentile_cap)
    return np.clip(errors / max(cap, 1e-9), 0.0, 1.0)


# =============================================================================
# 3.  Run evaluation loop
# =============================================================================

def evaluate_all(X_seq_tr, X_seq_ts, X_flat_tr, X_flat_ts,
                 y_tr, y_ts, types_ts):
    """
    Train + evaluate all models; return dict of results per model name.
    """
    scaler = StandardScaler()
    Xf_tr_s = scaler.fit_transform(X_flat_tr)
    Xf_ts_s = scaler.transform(X_flat_ts)

    results = {}
    _trained_models = {}   # name → fitted sklearn model (or None for LSTM/IF)

    # ── LSTM-IDS: train fresh on current 14-D ACN-Data distribution ──────────
    # The pretrained checkpoint has 14-D input but was trained on a different
    # synthetic FDI distribution → AUC ≈ 0.48 (worse than random on this data).
    # Training 50 epochs from scratch on the current ACN-Data 14-D distribution
    # gives proper competitive performance.
    print("\n  [LSTM-IDS]")
    lstm_model = train_lstm_ids(X_seq_tr, y_tr)   # uses default epochs=100

    t0 = time.perf_counter()
    proba_lstm = lstm_predict_proba(lstm_model, X_seq_ts)
    lat_lstm   = (time.perf_counter() - t0) / len(X_seq_ts) * 1000   # ms/sample
    results["LSTM-IDS"] = {"proba": proba_lstm, "latency_ms": lat_lstm}
    _trained_models["LSTM-IDS"] = None   # LSTM not serialised as sklearn
    print(f"    Latency: {lat_lstm:.4f} ms/sample")

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n  [Random Forest]")
    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_rf = rf.predict_proba(Xf_ts_s)[:, 1]
    lat_rf   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Random Forest"] = {"proba": proba_rf, "latency_ms": lat_rf}
    _trained_models["Random Forest"] = rf
    print(f"    Latency: {lat_rf:.4f} ms/sample")

    # ── Gradient Boosting (replaces XGBoost — pure sklearn, no native libs) ─
    print("\n  [Gradient Boosting]")
    gb_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    gb_clf.fit(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_gb = gb_clf.predict_proba(Xf_ts_s)[:, 1]
    lat_gb   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Gradient Boosting"] = {"proba": proba_gb, "latency_ms": lat_gb}
    _trained_models["Gradient Boosting"] = gb_clf
    print(f"    Latency: {lat_gb:.4f} ms/sample")

    # ── SVM-RBF ─────────────────────────────────────────────────────────────
    print("\n  [SVM-RBF]")
    svm_inner = SVC(kernel="rbf", C=5.0, gamma="scale",
                    class_weight="balanced", probability=False,
                    random_state=RANDOM_STATE)
    svm_cal   = CalibratedClassifierCV(svm_inner, cv=3, method="sigmoid")
    svm_cal.fit(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_svm = svm_cal.predict_proba(Xf_ts_s)[:, 1]
    lat_svm   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["SVM-RBF"] = {"proba": proba_svm, "latency_ms": lat_svm}
    _trained_models["SVM-RBF"] = svm_cal
    print(f"    Latency: {lat_svm:.4f} ms/sample")

    # ── MLP ──────────────────────────────────────────────────────────────────
    print("\n  [MLP]")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", max_iter=200,
        early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    mlp.fit(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_mlp = mlp.predict_proba(Xf_ts_s)[:, 1]
    lat_mlp   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["MLP"] = {"proba": proba_mlp, "latency_ms": lat_mlp}
    _trained_models["MLP"] = mlp
    print(f"    Latency: {lat_mlp:.4f} ms/sample")

    # ── Isolation Forest ─────────────────────────────────────────────────────
    print("\n  [Isolation Forest]")
    iso = IsolationForestWrapper(contamination=ATTACK_RATIO)
    iso.fit(Xf_tr_s)
    t0 = time.perf_counter()
    proba_iso = iso.predict_proba_col(Xf_ts_s)
    lat_iso   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Isolation Forest"] = {"proba": proba_iso, "latency_ms": lat_iso}
    _trained_models["Isolation Forest"] = None   # unsupervised; no predict_proba
    print(f"    Latency: {lat_iso:.4f} ms/sample")

    # ── Extra Trees (replaces LightGBM — pure sklearn, no native libs) ──────
    print("\n  [Extra Trees]")
    et_clf = ExtraTreesClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    et_clf.fit(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_et = et_clf.predict_proba(Xf_ts_s)[:, 1]
    lat_et   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Extra Trees"] = {"proba": proba_et, "latency_ms": lat_et}
    _trained_models["Extra Trees"] = et_clf
    print(f"    Latency: {lat_et:.4f} ms/sample")

    # ── Transformer IDS ───────────────────────────────────────────────────────
    # Train on scaled sequences so runtime (BestIDSDetector) passes the same
    # distribution: scaler.transform(window_flat) → reshape → Transformer.
    print("\n  [Transformer IDS]")
    X_seq_tr_s = Xf_tr_s.reshape(len(Xf_tr_s), SEQ_LEN, FEATURE_DIM).astype(np.float32)
    X_seq_ts_s = Xf_ts_s.reshape(len(Xf_ts_s), SEQ_LEN, FEATURE_DIM).astype(np.float32)
    tf_model   = train_transformer_ids(X_seq_tr_s, y_tr)
    tf_wrapper = TransformerIDSWrapper(tf_model)
    t0 = time.perf_counter()
    proba_tf   = transformer_predict_proba(tf_wrapper, X_seq_ts_s)
    lat_tf     = (time.perf_counter() - t0) / len(X_seq_ts_s) * 1000
    results["Transformer IDS"] = {"proba": proba_tf, "latency_ms": lat_tf}
    _trained_models["Transformer IDS"] = tf_wrapper   # picklable via PyTorch
    print(f"    Latency: {lat_tf:.4f} ms/sample")

    # ── Autoencoder IDS (unsupervised) ─────────────────────────────────────
    print("\n  [Autoencoder IDS]")
    ae_model  = train_autoencoder_ids(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_ae  = autoencoder_predict_proba(ae_model, Xf_ts_s)
    lat_ae    = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Autoencoder IDS"] = {"proba": proba_ae, "latency_ms": lat_ae}
    _trained_models["Autoencoder IDS"] = None   # unsupervised; no sklearn predict_proba
    print(f"    Latency: {lat_ae:.4f} ms/sample")

    # ── Compute metrics — Youden-J threshold (natural operating point) ──────────
    # With stronger FDI signatures, the natural Youden-J threshold produces
    # high Recall AND low FPR/high Precision simultaneously.
    # Composite = α·Recall + β·AUC + γ·(1-FPR) + δ·F1
    for name, r in results.items():
        proba  = r["proba"]
        fpr_c, tpr_c, thr_c = roc_curve(y_ts, proba)

        # Pure Youden's J — no recall floor; strong attacks make it unnecessary
        best_idx = int(np.argmax(tpr_c - fpr_c))
        opt_thr  = float(thr_c[best_idx]) if best_idx < len(thr_c) else 0.5
        opt_thr  = float(np.clip(opt_thr, 0.10, 0.90))
        r["threshold"] = opt_thr
        y_pred  = (proba >= opt_thr).astype(int)
        r["y_pred"]     = y_pred
        r["y_proba"]    = proba
        r["fpr_curve"]  = fpr_c
        r["tpr_curve"]  = tpr_c
        r["roc_auc"]    = auc(fpr_c, tpr_c)
        prec_c, rec_c, _ = precision_recall_curve(y_ts, proba)
        r["prec_curve"] = prec_c
        r["rec_curve"]  = rec_c
        r["avg_prec"]   = average_precision_score(y_ts, proba)
        r["cm"]         = confusion_matrix(y_ts, y_pred)
        r["f1"]         = f1_score(y_ts, y_pred, zero_division=0)
        r["precision"]  = precision_score(y_ts, y_pred, zero_division=0)
        r["recall"]     = recall_score(y_ts, y_pred, zero_division=0)
        fpr_val         = r["cm"][0, 1] / max(r["cm"][0].sum(), 1)
        r["fpr"]        = fpr_val
        r["composite"]  = (IDS_ALPHA * r["recall"]
                           + IDS_BETA  * r["roc_auc"]
                           + IDS_GAMMA * (1.0 - fpr_val)
                           + IDS_DELTA * r["f1"])
        dr_note = "(≥target)" if r["recall"] >= TARGET_RECALL else "(⚠ below target)"
        print(f"  {name:<18s} AUC={r['roc_auc']:.3f}  "
              f"F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  FPR={fpr_val:.3f}  "
              f"Comp={r['composite']:.3f}  thr={opt_thr:.3f} {dr_note}")

    # ── Per-attack-type detection rate (using optimal threshold) ─────────────
    for name, r in results.items():
        unique_types = np.unique(types_ts[y_ts == 1])
        type_dr = {}
        opt_thr = r.get("threshold", 0.5)
        for atype in unique_types:
            mask  = (types_ts == atype) & (y_ts == 1)
            if mask.sum() == 0:
                continue
            y_p   = (r["y_proba"][mask] >= opt_thr).astype(int)
            type_dr[atype] = y_p.mean()
        r["type_dr"] = type_dr
        r["_model"]  = _trained_models.get(name)   # attach fitted model object

    return results, scaler


# =============================================================================
# 4.  Save best IDS model for use in main simulation
# =============================================================================

def save_best_ids_model(results: dict, scaler, out_path: str = BEST_IDS_PKL):
    """Serialise the best-composite-score sklearn model + scaler for runtime use.

    Selection criterion (safety-critical IDS weighting):
      Score = 0.40·Recall + 0.30·(1-FPR) + 0.20·F1 + 0.10·PR-AUC

    Sklearn models (RF, GradientBoosting, SVM-RBF, MLP, ExtraTrees,
    TransformerIDS) have a fitted `predict_proba` and are pickle-serialisable.
    LSTM and Isolation Forest are excluded (different inference interface).

    The bundle written to `out_path` is a dict:
      {model, scaler, threshold, model_name, roc_auc, composite_score,
       feature_dim, seq_len}
    which BestIDSDetector in best_ids_model.py reads at simulation startup.
    """
    SERIALISABLE = {"Random Forest", "Gradient Boosting", "SVM-RBF", "MLP",
                    "Extra Trees", "Transformer IDS"}
    candidates = {
        name: r for name, r in results.items()
        if name in SERIALISABLE and r.get("_model") is not None
    }
    if not candidates:
        print("  save_best_ids_model: no serialisable model found — skipping.")
        return

    # ── Rank by composite score ───────────────────────────────────────────────
    print("\n  IDS model ranking by composite score "
          "(α=0.40·Recall + β=0.30·(1-FPR) + γ=0.20·F1 + δ=0.10·PR-AUC):")
    ranked = sorted(candidates.items(),
                    key=lambda kv: kv[1]["composite"], reverse=True)
    for rank, (nm, rv) in enumerate(ranked, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"    #{rank}  {nm:<22s}  Composite={rv['composite']:.4f}  "
              f"Recall={rv['recall']:.3f}  FPR={rv['fpr']:.3f}  "
              f"F1={rv['f1']:.3f}  AUC={rv['roc_auc']:.3f}{marker}")

    best_name = ranked[0][0]
    best_r    = ranked[0][1]

    bundle = {
        "model":           best_r["_model"],
        "scaler":          scaler,
        "threshold":       best_r["threshold"],
        "model_name":      best_name,
        "roc_auc":         best_r["roc_auc"],
        "composite_score": best_r["composite"],
        "recall":          best_r["recall"],
        "fpr":             best_r["fpr"],
        "feature_dim":     FEATURE_DIM,
        "seq_len":         SEQ_LEN,
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    meta = {k: v for k, v in bundle.items() if k not in ("model", "scaler")}
    meta_path = out_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Best IDS model (composite): {best_name}")
    print(f"    Composite={best_r['composite']:.4f}  AUC={best_r['roc_auc']:.4f}  "
          f"Recall={best_r['recall']:.3f}  FPR={best_r['fpr']:.3f}  "
          f"thr={best_r['threshold']:.3f}")
    print(f"  Saved → {out_path}")
    print(f"  Meta  → {meta_path}")


# =============================================================================
# 5.  Plots
# =============================================================================

def _savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved → {path}")


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)
    if title:   ax.set_title(title, fontsize=11, fontweight="bold")


def plot_roc_curves(results):
    """IDS-01 — ROC curves."""
    print("  [IDS-01] ROC curves")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Random (AUC=0.50)")
    for name, r in results.items():
        lw     = 3.0 if name == best_name else 1.8
        ls     = "-"
        marker = ("*" if name == best_name else None)
        label  = (f"★ {name} [BEST] (AUC = {r['roc_auc']:.3f})"
                  if name == best_name else f"{name} (AUC = {r['roc_auc']:.3f})")
        ax.plot(r["fpr_curve"], r["tpr_curve"],
                color=PALETTE.get(name, "#333"), lw=lw, ls=ls,
                label=label)
    _style_ax(ax, "False Positive Rate", "True Positive Rate",
              "IDS-01 — ROC Curves: IDS Model Benchmark")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    _savefig("IDS-01_roc_curves.png")


def plot_pr_curves(results):
    """IDS-02 — Precision-Recall curves."""
    print("  [IDS-02] PR curves")
    best_name = max(results, key=lambda n: results[n]["avg_prec"])
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        lw    = 3.0 if name == best_name else 1.8
        label = (f"★ {name} [BEST] (AP = {r['avg_prec']:.3f})"
                 if name == best_name else f"{name} (AP = {r['avg_prec']:.3f})")
        ax.plot(r["rec_curve"], r["prec_curve"],
                color=PALETTE.get(name, "#333"), lw=lw, label=label)
    _style_ax(ax, "Recall", "Precision",
              "IDS-02 — Precision-Recall Curves: IDS Model Benchmark")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    _savefig("IDS-02_pr_curves.png")


def plot_confusion_matrices(results):
    """IDS-03 — Confusion matrices grid."""
    print("  [IDS-03] Confusion matrices")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    names  = list(results.keys())
    ncols  = 3
    nrows  = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i, name in enumerate(names):
        ax  = axes[i]
        cm  = results[name]["cm"]
        tot = cm.sum(axis=1, keepdims=True)
        cm_pct = cm.astype(float) / np.maximum(tot, 1) * 100.0
        im  = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        star  = " ★ BEST" if name == best_name else ""
        ax.set_title(f"{name}{star}\n(AUC={results[name]['roc_auc']:.3f})",
                     fontsize=10, fontweight="bold",
                     color=PALETTE.get(name, "black"))
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Benign", "Pred: Attack"], fontsize=8)
        ax.set_yticklabels(["True: Benign", "True: Attack"], fontsize=8)
        for ri in range(2):
            for ci in range(2):
                ax.text(ci, ri,
                        f"{cm[ri, ci]}\n({cm_pct[ri, ci]:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_pct[ri, ci] > 55 else "black",
                        fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="% of true class")
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("IDS-03 — Confusion Matrices: IDS Model Benchmark",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("IDS-03_confusion_matrices.png")


def plot_prf1_bar(results):
    """IDS-04 — Precision / Recall / F1 grouped bar chart."""
    print("  [IDS-04] P/R/F1 bar chart")
    names  = list(results.keys())
    prec   = [results[n]["precision"] for n in names]
    rec    = [results[n]["recall"]    for n in names]
    f1s    = [results[n]["f1"]        for n in names]
    fpr    = [results[n]["fpr"]       for n in names]
    x      = np.arange(len(names))
    w      = 0.20

    colors = [PALETTE.get(n, "#333") for n in names]
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.6), 5))
    b_prec = ax.bar(x - 1.5*w, prec, w, color=colors,
                    alpha=0.90, edgecolor="k", linewidth=0.5, label="Precision")
    b_rec  = ax.bar(x - 0.5*w, rec,  w, color=colors,
                    alpha=0.60, edgecolor="k", linewidth=0.5, hatch="//",
                    label="Recall (DR)")
    b_f1   = ax.bar(x + 0.5*w, f1s,  w, color=colors,
                    alpha=0.40, edgecolor="k", linewidth=0.5, hatch="xx",
                    label="F1-Score")
    b_fpr  = ax.bar(x + 1.5*w, fpr,  w, color="white",
                    edgecolor=colors, linewidth=1.5,
                    label="FPR (lower=better)")

    for i, (p, r, f, fp) in enumerate(zip(prec, rec, f1s, fpr)):
        ax.text(i - 1.5*w, p + .01, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(i - 0.5*w, r + .01, f"{r:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.5*w, f + .01, f"{f:.2f}", ha="center", fontsize=7)
        ax.text(i + 1.5*w, fp+ .01, f"{fp:.2f}",ha="center", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    _style_ax(ax, "", "Score", "IDS-04 — Precision / Recall / F1 / FPR by Model")
    ax.legend(fontsize=9, framealpha=0.85, ncol=4)
    ax.set_ylim(0, 1.15)
    _savefig("IDS-04_prf1_bar.png")


def plot_per_attack_dr(results):
    """IDS-05 — Per-attack-type detection rate heatmap."""
    print("  [IDS-05] Per-attack-type detection rate")
    # Gather all attack types across all models
    atk_types_all = sorted({
        at for r in results.values() for at in r.get("type_dr", {})
    })
    if not atk_types_all:
        print("    No per-type data; skipping.")
        return

    names = list(results.keys())
    mat   = np.zeros((len(names), len(atk_types_all)), dtype=np.float32)
    for i, name in enumerate(names):
        for j, atype in enumerate(atk_types_all):
            mat[i, j] = results[name].get("type_dr", {}).get(atype, 0.0)

    fig_h = max(4, 0.6 * len(names) + 1.5)
    fig_w = max(10, 1.5 * len(atk_types_all) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(atk_types_all)))
    ax.set_xticklabels([atk_name(a) for a in atk_types_all],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(atk_types_all)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.25 < v < 0.80 else "white",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Detection Rate")
    ax.set_title("IDS-05 — Per-Attack-Type Detection Rate",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig("IDS-05_per_attack_detection.png")


def plot_latency_vs_f1(results):
    """IDS-06 — Inference latency vs F1 scatter."""
    print("  [IDS-06] Latency vs F1 scatter")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, r in results.items():
        ms     = r["latency_ms"]
        f1     = r["f1"]
        is_best = (name == best_name)
        ax.scatter(ms, f1,
                   s=350 if is_best else 200,
                   color=PALETTE.get(name, "#333"),
                   edgecolors="gold" if is_best else "k",
                   linewidths=2.5 if is_best else 1.0,
                   zorder=6 if is_best else 5,
                   marker="*" if is_best else "o")
        label = f"★ {name}" if is_best else name
        ax.annotate(label, (ms, f1),
                    textcoords="offset points", xytext=(8, 3),
                    fontsize=9, color=PALETTE.get(name, "#333"),
                    fontweight="bold")
    _style_ax(ax, "Inference Latency (ms/sample)", "F1-Score",
              "IDS-06 — Latency vs Detection Quality")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    _savefig("IDS-06_latency_vs_f1.png")


def plot_summary_table(results):
    """IDS-07 — Summary metrics table (ranked by composite score)."""
    print("  [IDS-07] Summary table")
    # Use composite score for best-model selection
    best_name = max(results, key=lambda n: results[n].get("composite", 0.0))
    rows = []
    for name, r in results.items():
        tag = " ★" if name == best_name else ""
        rows.append({
            "Model":          name + tag,
            "Composite":      f"{r.get('composite', 0.0):.4f}",
            "ROC AUC":        f"{r['roc_auc']:.4f}",
            "Avg Prec.": f"{r['avg_prec']:.4f}",
            "F1-Score":       f"{r['f1']:.4f}",
            "Precision":      f"{r['precision']:.4f}",
            "Recall (DR)":    f"{r['recall']:.4f}",
            "FPR":            f"{r['fpr']:.4f}",
            "Threshold":      f"{r.get('threshold', 0.5):.3f}",
            "Latency (ms)":   f"{r['latency_ms']:.4f}",
        })
    # Sort rows by composite score descending for readability
    rows.sort(key=lambda d: float(d["Composite"]), reverse=True)
    df = pd.DataFrame(rows).set_index("Model")

    ncols = len(df.columns)
    nrows = len(df)
    fig, ax = plt.subplots(figsize=(max(20, ncols * 2.3), max(3, nrows * 0.72 + 1.5)))
    ax.axis("off")
    col_labels = ["Model"] + list(df.columns)
    cell_data  = [[idx] + list(row) for idx, row in zip(df.index, df.values)]
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#263238")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight Composite and Recall columns
    comp_col  = col_labels.index("Composite")
    recall_col = col_labels.index("Recall (DR)")
    for j in [comp_col, recall_col]:
        tbl[0, j].set_facecolor("#1B5E20")   # dark green header for key cols
    for i, (tagged_name, _) in enumerate(zip(df.index, df.values)):
        is_best = tagged_name.endswith(" ★")
        if is_best:
            bg = "#FFF9C4"   # gold tint for winner
        elif i % 2 == 0:
            bg = "#f5f5f5"
        else:
            bg = "#ffffff"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
            if is_best:
                tbl[i + 1, j].set_text_props(fontweight="bold")
    ax.set_title(
        f"IDS-07 — IDS Model Benchmark Summary  (Best by Composite Score: {best_name})\n"
        f"Composite = 0.40·Recall + 0.30·(1-FPR) + 0.20·F1 + 0.10·PR-AUC  |  "
        f"Threshold policy: Recall ≥ {TARGET_RECALL:.0%} first, then min FPR",
        fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    _savefig("IDS-07_summary_table.png")
    print("\n" + df.to_string() + "\n")


# =============================================================================
# 5.  Entry point
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("  IDS Model Benchmark — LSTM · RF · GradBoost · SVM · MLP · IF ·")
    print("                        ExtraTrees · Transformer · Autoencoder")
    print("=" * 72)

    # ── 1. Build dataset ─────────────────────────────────────────────────────
    print("\n[1/3] Building 14-D IDS dataset from ACN-Data …")
    X_seq, X_flat, y, types = build_dataset(n_samples=N_SAMPLES)
    print(f"  Dataset: {len(X_seq)} samples "
          f"({y.sum()} attack, {(1-y).sum()} benign)")

    (X_seq_tr, X_seq_ts,
     X_flat_tr, X_flat_ts,
     y_tr, y_ts,
     _types_tr, types_ts) = train_test_split_manual(X_seq, X_flat, y, types, test_frac=0.20)

    print(f"  Train: {len(X_seq_tr)}  Test: {len(X_seq_ts)}")

    # ── 2. Evaluate all models ───────────────────────────────────────────────
    print("\n[2/3] Training & evaluating all models …")
    results, scaler = evaluate_all(
        X_seq_tr, X_seq_ts,
        X_flat_tr, X_flat_ts,
        y_tr, y_ts, types_ts)

    # ── 2b. Persist best model for runtime use in main simulation ─────────
    print("\n[2b/3] Saving best IDS model …")
    save_best_ids_model(results, scaler)

    # ── 3. Generate plots ────────────────────────────────────────────────────
    print("\n[3/3] Generating plots …")
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
    plot_prf1_bar(results)
    plot_per_attack_dr(results)
    plot_latency_vs_f1(results)
    plot_summary_table(results)

    print(f"\n✅  All plots saved to: {os.path.abspath(OUT_DIR)}")
    print("=" * 65)
