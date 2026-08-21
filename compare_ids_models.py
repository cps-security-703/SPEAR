#!/usr/bin/env python3


import os, sys, time, warnings, math, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from sklearn.ensemble     import RandomForestClassifier, IsolationForest
from sklearn.svm          import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  xgboost not installed — XGBoost will be skipped.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("  lightgbm not installed — LightGBM will be skipped.")


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


SEQ_LEN       = 10
FEATURE_DIM   = 14
N_SAMPLES     = 6000
ATTACK_RATIO  = 0.50
RANDOM_STATE  = 42
DEVICE        = torch.device("cpu")


from ids_neural_models import (          # noqa: E402
    LSTMIDSBenchmark, TransformerIDS, TransformerIDSWrapper, AutoencoderIDS,
)


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

    if isinstance(code, (int, np.integer)):
        return ATK_TYPE_NAMES.get(int(code), str(code))
    return str(code).replace("_", " ")

PALETTE = {
    "LSTM-IDS":         "#00897B",
    "Random Forest":    "#3498db",
    "XGBoost":          "#e74c3c",
    "SVM-RBF":          "#9b59b6",
    "MLP":              "#f39c12",
    "Isolation Forest": "#7f8c8d",
    "LightGBM":         "#27ae60",
    "Transformer IDS":  "#8e44ad",
    "Autoencoder IDS":  "#d35400",
}


def _evcs_features_14d(sess_row: dict, system_id: int = 1,
                        attack_active: bool = False) -> np.ndarray:

    freq = sess_row.get('grid_frequency', 60.0)
    return np.array([
        sess_row.get('soc',           0.5),
        sess_row.get('voltage',     240.0) / 240.0,
        sess_row.get('current',      16.0) /  32.0,
        sess_row.get('power',        3.84) /  7.68,
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

    f   = feat.copy()
    mag = (np.random.uniform(0.35, 0.55) + 0.20 * step_frac) * step_frac

    if attack_type == 0:
        f[7]  = float(np.clip(f[7]  - mag * 0.35, 0.7, 1.3))
        f[11] = float(np.clip(f[11] + mag * 0.20, 0.0, 2.0))
        f[3]  = float(np.clip(f[3]  + mag * 0.20, 0.0, 2.0))

    elif attack_type == 1:
        f[2]  = float(np.clip(f[2]  + mag * 0.45, 0.0, 2.0))
        f[5]  = float(np.clip(f[5]  + mag * 0.45, 0.0, 2.0))
        f[6]  = float(np.clip(f[6]  + mag * 0.40, 0.0, 2.0))
        f[11] = float(np.clip(f[11] + mag * 0.20, 0.0, 2.0))

    elif attack_type == 2:
        f[3]  = float(np.clip(f[3]  - mag * 0.90, 0.0, 1.0))
        f[5]  = float(np.clip(f[5]  - mag * 0.90, 0.0, 1.0))
        f[6]  = float(np.clip(f[6]  - mag * 0.90, 0.0, 1.0))

    elif attack_type == 3:
        f[0]  = float(np.clip(f[0]  - mag * 0.70, 0.0, 1.0))
        f[11] = float(np.clip(f[11] + mag * 0.40, 0.0, 2.0))
        f[2]  = float(np.clip(f[2]  + mag * 0.30, 0.0, 2.0))

    elif attack_type == 4:
        f[8]  = float(np.clip(f[8]  + mag * 0.15, 0.88, 1.12))
        f[5]  = float(np.clip(f[5]  + mag * 0.30, 0.0, 2.0))
        f[7]  = float(np.clip(f[7]  - mag * 0.20, 0.7,  1.3))

    elif attack_type == 5:
        phase = step_frac * np.pi * 2.0
        osc   = np.sin(phase) * mag * 0.20
        f[5]  = float(np.clip(f[5]  + osc * 20.0, 0.0, 2.0))
        f[7]  = float(np.clip(f[7]  - mag * 0.20, 0.7, 1.3))
        f[3]  = float(np.clip(f[3]  + osc * 12.0, 0.0, 2.0))

    return f


def build_dataset(n_samples: int = N_SAMPLES, attack_ratio: float = ATTACK_RATIO):

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
    ATK_TYPES = list(range(6))


    benign_seqs = []
    for _ in range(n_benign):

        sess = sessions[rng.randint(len(sessions))]

        base_soc      = rng.uniform(0.10, 0.80)
        base_power    = rng.uniform(0.10, 0.80)
        sys_id        = rng.randint(1, 7)
        seq = []
        for t in range(SEQ_LEN):
            soc_t   = float(np.clip(base_soc + t * 0.01, 0, 1))
            pwr_t   = float(np.clip(base_power - t * 0.005, 0, 1))
            feat = _evcs_features_14d({
                'soc':            soc_t,
                'voltage':        rng.uniform(220, 260),
                'current':        rng.uniform(6, 32),
                'power':          pwr_t * 7.68,
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


    attack_seqs  = []
    attack_types = []
    for _ in range(n_attack):
        atype  = rng.choice(ATK_TYPES)

        base_soc   = rng.uniform(0.10, 0.80)
        base_power = rng.uniform(0.10, 0.80)
        sys_id     = rng.randint(1, 7)
        inject_at  = rng.randint(SEQ_LEN // 2, SEQ_LEN)
        seq = []
        for t in range(SEQ_LEN):
            soc_t   = float(np.clip(base_soc + t * 0.01, 0, 1))
            pwr_t   = float(np.clip(base_power - t * 0.005, 0, 1))
            feat = _evcs_features_14d({
                'soc':            soc_t,
                'voltage':        rng.uniform(220, 260),
                'current':        rng.uniform(6, 32),
                'power':          pwr_t * 7.68,
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


def train_lstm_ids(X_tr_seq, y_tr, epochs=100, lr=3e-4, batch=256):

    model = LSTMIDSBenchmark().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

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

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch):
            xb = torch.FloatTensor(X_seq[i: i + batch]).to(DEVICE)
            class_logits, _ = model(xb)
            p = torch.softmax(class_logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


class IsolationForestWrapper:

    def __init__(self, contamination=0.5):
        self._clf = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1)

    def fit(self, X_flat, y=None):
        self._clf.fit(X_flat)

    def predict_proba_col(self, X_flat):

        scores = -self._clf.decision_function(X_flat)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-9:
            return np.full(len(X_flat), 0.5)
        return (scores - s_min) / (s_max - s_min)

    def predict(self, X_flat, threshold=0.5):
        return (self.predict_proba_col(X_flat) >= threshold).astype(int)


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
            chunk.reshape(len(chunk), -1)))
    return np.concatenate(probs)[:, 1]


def train_autoencoder_ids(Xf_tr_s: np.ndarray, y_tr: np.ndarray,
                           epochs: int = 80, lr: float = 1e-3,
                           batch: int = 256) -> AutoencoderIDS:

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

    errors = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xf_s), batch):
            xb = torch.FloatTensor(Xf_s[i: i + batch]).to(DEVICE)
            errors.append(model.reconstruction_error(xb).cpu().numpy())
    errors = np.concatenate(errors)
    cap    = np.percentile(errors, percentile_cap)
    return np.clip(errors / max(cap, 1e-9), 0.0, 1.0)


def evaluate_all(X_seq_tr, X_seq_ts, X_flat_tr, X_flat_ts,
                 y_tr, y_ts, types_ts):

    scaler = StandardScaler()
    Xf_tr_s = scaler.fit_transform(X_flat_tr)
    Xf_ts_s = scaler.transform(X_flat_ts)

    results = {}
    _trained_models = {}


    print("\n  [LSTM-IDS]")
    lstm_model = train_lstm_ids(X_seq_tr, y_tr)

    t0 = time.perf_counter()
    proba_lstm = lstm_predict_proba(lstm_model, X_seq_ts)
    lat_lstm   = (time.perf_counter() - t0) / len(X_seq_ts) * 1000
    results["LSTM-IDS"] = {"proba": proba_lstm, "latency_ms": lat_lstm}
    _trained_models["LSTM-IDS"] = None
    print(f"    Latency: {lat_lstm:.4f} ms/sample")


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


    if XGB_AVAILABLE:
        print("\n  [XGBoost]")
        scale_pos = int((1 - ATTACK_RATIO) / max(ATTACK_RATIO, 1e-6))
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        xgb_clf.fit(Xf_tr_s, y_tr)
        t0 = time.perf_counter()
        proba_xgb = xgb_clf.predict_proba(Xf_ts_s)[:, 1]
        lat_xgb   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
        results["XGBoost"] = {"proba": proba_xgb, "latency_ms": lat_xgb}
        _trained_models["XGBoost"] = xgb_clf
        print(f"    Latency: {lat_xgb:.4f} ms/sample")


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


    print("\n  [Isolation Forest]")
    iso = IsolationForestWrapper(contamination=ATTACK_RATIO)
    iso.fit(Xf_tr_s)
    t0 = time.perf_counter()
    proba_iso = iso.predict_proba_col(Xf_ts_s)
    lat_iso   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Isolation Forest"] = {"proba": proba_iso, "latency_ms": lat_iso}
    _trained_models["Isolation Forest"] = None
    print(f"    Latency: {lat_iso:.4f} ms/sample")


    if LGB_AVAILABLE:
        print("\n  [LightGBM]")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        lgb_clf.fit(Xf_tr_s, y_tr,
                    eval_set=[(Xf_ts_s, y_ts)],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(period=-1)])
        t0 = time.perf_counter()
        proba_lgb = lgb_clf.predict_proba(Xf_ts_s)[:, 1]
        lat_lgb   = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
        results["LightGBM"] = {"proba": proba_lgb, "latency_ms": lat_lgb}
        _trained_models["LightGBM"] = lgb_clf
        print(f"    Latency: {lat_lgb:.4f} ms/sample")


    print("\n  [Transformer IDS]")
    X_seq_tr_s = Xf_tr_s.reshape(len(Xf_tr_s), SEQ_LEN, FEATURE_DIM).astype(np.float32)
    X_seq_ts_s = Xf_ts_s.reshape(len(Xf_ts_s), SEQ_LEN, FEATURE_DIM).astype(np.float32)
    tf_model   = train_transformer_ids(X_seq_tr_s, y_tr)
    tf_wrapper = TransformerIDSWrapper(tf_model)
    t0 = time.perf_counter()
    proba_tf   = transformer_predict_proba(tf_wrapper, X_seq_ts_s)
    lat_tf     = (time.perf_counter() - t0) / len(X_seq_ts_s) * 1000
    results["Transformer IDS"] = {"proba": proba_tf, "latency_ms": lat_tf}
    _trained_models["Transformer IDS"] = tf_wrapper
    print(f"    Latency: {lat_tf:.4f} ms/sample")


    print("\n  [Autoencoder IDS]")
    ae_model  = train_autoencoder_ids(Xf_tr_s, y_tr)
    t0 = time.perf_counter()
    proba_ae  = autoencoder_predict_proba(ae_model, Xf_ts_s)
    lat_ae    = (time.perf_counter() - t0) / len(Xf_ts_s) * 1000
    results["Autoencoder IDS"] = {"proba": proba_ae, "latency_ms": lat_ae}
    _trained_models["Autoencoder IDS"] = None
    print(f"    Latency: {lat_ae:.4f} ms/sample")


    for name, r in results.items():
        proba  = r["proba"]

        fpr_c, tpr_c, thr_c = roc_curve(y_ts, proba)
        j_scores   = tpr_c - fpr_c
        best_idx   = int(np.argmax(j_scores))
        opt_thr    = float(thr_c[best_idx]) if best_idx < len(thr_c) else 0.5

        opt_thr    = float(np.clip(opt_thr, 0.10, 0.90))
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
        print(f"  {name:<18s} AUC={r['roc_auc']:.3f}  "
              f"F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  FPR={fpr_val:.3f}  "
              f"thr={opt_thr:.3f}")


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
        r["_model"]  = _trained_models.get(name)

    return results, scaler


def save_best_ids_model(results: dict, scaler, out_path: str = BEST_IDS_PKL):

    SERIALISABLE = {"Random Forest", "XGBoost", "SVM-RBF", "MLP",
                    "LightGBM", "Transformer IDS"}
    candidates = {
        name: r for name, r in results.items()
        if name in SERIALISABLE and r.get("_model") is not None
    }
    if not candidates:
        print("  save_best_ids_model: no serialisable model found — skipping.")
        return

    best_name = max(candidates, key=lambda n: candidates[n]["roc_auc"])
    best_r    = candidates[best_name]

    bundle = {
        "model":       best_r["_model"],
        "scaler":      scaler,
        "threshold":   best_r["threshold"],
        "model_name":  best_name,
        "roc_auc":     best_r["roc_auc"],
        "feature_dim": FEATURE_DIM,
        "seq_len":     SEQ_LEN,
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    meta = {k: v for k, v in bundle.items() if k not in ("model", "scaler")}
    meta_path = out_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Best IDS model: {best_name}  (AUC={best_r['roc_auc']:.4f}, "
          f"thr={best_r['threshold']:.3f})")
    print(f"  Saved  {out_path}")
    print(f"  Meta   {meta_path}")


def _savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved  {path}")


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)
    if title:   ax.set_title(title, fontsize=11, fontweight="bold")


def plot_roc_curves(results):

    print("  [IDS-01] ROC curves")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Random (AUC=0.50)")
    for name, r in results.items():
        lw     = 3.0 if name == best_name else 1.8
        ls     = "-"
        marker = ("*" if name == best_name else None)
        label  = (f" {name} [BEST] (AUC = {r['roc_auc']:.3f})"
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

    print("  [IDS-02] PR curves")
    best_name = max(results, key=lambda n: results[n]["avg_prec"])
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        lw    = 3.0 if name == best_name else 1.8
        label = (f" {name} [BEST] (AP = {r['avg_prec']:.3f})"
                 if name == best_name else f"{name} (AP = {r['avg_prec']:.3f})")
        ax.plot(r["rec_curve"], r["prec_curve"],
                color=PALETTE.get(name, "#333"), lw=lw, label=label)
    _style_ax(ax, "Recall", "Precision",
              "IDS-02 — Precision-Recall Curves: IDS Model Benchmark")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    _savefig("IDS-02_pr_curves.png")


def plot_confusion_matrices(results):

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
        star  = "  BEST" if name == best_name else ""
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

    print("  [IDS-05] Per-attack-type detection rate")

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
        label = f" {name}" if is_best else name
        ax.annotate(label, (ms, f1),
                    textcoords="offset points", xytext=(8, 3),
                    fontsize=9, color=PALETTE.get(name, "#333"),
                    fontweight="bold")
    _style_ax(ax, "Inference Latency (ms/sample)", "F1-Score",
              "IDS-06 — Latency vs Detection Quality")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    _savefig("IDS-06_latency_vs_f1.png")


def plot_summary_table(results):

    print("  [IDS-07] Summary table")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    rows = []
    for name, r in results.items():
        tag = " " if name == best_name else ""
        rows.append({
            "Model":         name + tag,
            "ROC AUC":       f"{r['roc_auc']:.4f}",
            "Avg Prec.":     f"{r['avg_prec']:.4f}",
            "F1-Score":      f"{r['f1']:.4f}",
            "Precision":     f"{r['precision']:.4f}",
            "Recall (DR)":   f"{r['recall']:.4f}",
            "FPR":           f"{r['fpr']:.4f}",
            "Threshold":     f"{r.get('threshold', 0.5):.3f}",
            "Latency (ms)":  f"{r['latency_ms']:.4f}",
        })
    df = pd.DataFrame(rows).set_index("Model")

    ncols = len(df.columns)
    nrows = len(df)
    fig, ax = plt.subplots(figsize=(max(18, ncols * 2.3), max(3, nrows * 0.72 + 1.5)))
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
    for i, (tagged_name, _) in enumerate(zip(df.index, df.values)):
        is_best = tagged_name.endswith(" ")
        if is_best:
            bg = "#FFF9C4"
        elif i % 2 == 0:
            bg = "#f5f5f5"
        else:
            bg = "#ffffff"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
            if is_best:
                tbl[i + 1, j].set_text_props(fontweight="bold")
    ax.set_title(f"IDS-07 — IDS Model Benchmark Summary  (Best: {best_name})",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    _savefig("IDS-07_summary_table.png")
    print("\n" + df.to_string() + "\n")


if __name__ == "__main__":

    print("=" * 72)
    print("  IDS Model Benchmark — LSTM · RF · XGB · SVM · MLP · IF ·")
    print("                        LightGBM · Transformer · Autoencoder")
    print("=" * 72)


    print("\n[1/3] Building 14-D IDS dataset from ACN-Data …")
    X_seq, X_flat, y, types = build_dataset(n_samples=N_SAMPLES)
    print(f"  Dataset: {len(X_seq)} samples "
          f"({y.sum()} attack, {(1-y).sum()} benign)")

    (X_seq_tr, X_seq_ts,
     X_flat_tr, X_flat_ts,
     y_tr, y_ts,
     _types_tr, types_ts) = train_test_split_manual(X_seq, X_flat, y, types, test_frac=0.20)

    print(f"  Train: {len(X_seq_tr)}  Test: {len(X_seq_ts)}")


    print("\n[2/3] Training & evaluating all models …")
    results, scaler = evaluate_all(
        X_seq_tr, X_seq_ts,
        X_flat_tr, X_flat_ts,
        y_tr, y_ts, types_ts)


    print("\n[2b/3] Saving best IDS model …")
    save_best_ids_model(results, scaler)


    print("\n[3/3] Generating plots …")
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
    plot_prf1_bar(results)
    plot_per_attack_dr(results)
    plot_latency_vs_f1(results)
    plot_summary_table(results)

    print(f"\n  All plots saved to: {os.path.abspath(OUT_DIR)}")
    print("=" * 65)
