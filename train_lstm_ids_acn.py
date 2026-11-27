#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from lstm_anomaly_detector import LSTMIDSDetector, LSTMIDSModel
from acn_sim_interface import ACNDataLoader


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IDX_SOC           = 0
IDX_VPU           = 1
IDX_FREQ          = 2
IDX_DEMAND        = 3
IDX_V_PRIO        = 4
IDX_URGENCY       = 5
IDX_TOD           = 6
IDX_LOAD          = 8
IDX_PREV_P        = 9
IDX_AC_P_IN       = 10

IDX_FREQ_DEV      = 14
IDX_AGG_P_PU      = 15
IDX_AGG_Q_PU      = 16
IDX_V_MIN_PU      = 17
IDX_V_MAX_PU      = 18
IDX_ATTACK_FLAG   = 19


IDS_FEATURE_DIM = 20


FDI_TYPES = [
    "energy_delivered_inflate",
    "remaining_demand_inflate",
    "remaining_time_extend",
    "dos_pilot",
    "id_spoof",
    "time_shift",
    "grid_frequency_attack",
    "load_injection_attack",
]


def inject_fdi_on_abstract_features(
    clean_sequences: np.ndarray,
    attack_ratio: float = 0.5,
    magnitude_range: Tuple[float, float] = (0.05, 0.30),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    N, T, F = clean_sequences.shape
    attacked = clean_sequences.copy()
    labels = np.zeros(N, dtype=np.int64)
    atk_types = np.full(N, -1, dtype=np.int64)

    n_attacks = int(N * attack_ratio)
    atk_idx = np.random.choice(N, n_attacks, replace=False)

    for k, i in enumerate(atk_idx):
        mag = np.random.uniform(*magnitude_range)

        atype = k % len(FDI_TYPES)
        atk_types[i] = atype
        labels[i] = 1

        seq = attacked[i]

        if FDI_TYPES[atype] == "energy_delivered_inflate":


            seq[:, IDX_SOC]      = np.clip(seq[:, IDX_SOC] + mag,      0.0, 1.0)
            seq[:, IDX_DEMAND]   = np.clip(seq[:, IDX_DEMAND] - mag * 0.5, 0.0, 1.0)
            seq[:, IDX_URGENCY]  = np.clip(seq[:, IDX_URGENCY] - mag * 0.3, 0.0, 1.0)

        elif FDI_TYPES[atype] == "remaining_demand_inflate":


            seq[:, IDX_DEMAND]   = np.clip(seq[:, IDX_DEMAND] + mag,   0.0, 1.5)
            seq[:, IDX_LOAD]     = np.clip(seq[:, IDX_LOAD] + mag,     0.0, 1.5)
            seq[:, IDX_AC_P_IN]  = np.clip(seq[:, IDX_AC_P_IN] + mag,  0.0, 1.5)

        elif FDI_TYPES[atype] == "remaining_time_extend":


            seq[:, IDX_URGENCY]  = np.clip(seq[:, IDX_URGENCY] - mag * 0.8, 0.0, 1.0)
            seq[:, IDX_V_PRIO]   = np.clip(seq[:, IDX_V_PRIO]  - mag * 0.3, 0.0, 1.0)

        elif FDI_TYPES[atype] == "dos_pilot":


            seq[:, IDX_PREV_P]   = 0.0
            seq[:, IDX_AC_P_IN]  = 0.0
            seq[:, IDX_DEMAND]   *= (1.0 - mag)

        elif FDI_TYPES[atype] == "id_spoof":


            spoof_soc = np.random.uniform(0.0, 1.0)
            seq[:, IDX_SOC]      = np.clip(spoof_soc + np.random.normal(0, 0.05, T),
                                           0.0, 1.0)
            seq[:, IDX_DEMAND]   = np.clip(seq[:, IDX_DEMAND] +
                                           np.random.uniform(-mag, mag, T),
                                           0.0, 1.5)

        elif FDI_TYPES[atype] == "time_shift":


            shift_hr = np.random.uniform(-6.0, 6.0) * mag
            seq[:, IDX_TOD]      = np.clip((seq[:, IDX_TOD] + shift_hr / 24.0) % 1.0,
                                           0.0, 1.0)
            seq[:, IDX_URGENCY]  = np.clip(seq[:, IDX_URGENCY] +
                                           np.random.normal(0, mag, T),
                                           0.0, 1.0)

        elif FDI_TYPES[atype] == "grid_frequency_attack":


            freq_shift = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0) * mag
            if seq.shape[1] > IDX_FREQ_DEV:
                seq[:, IDX_FREQ_DEV] = np.clip(
                    seq[:, IDX_FREQ_DEV] + freq_shift, -2.0, 2.0)
            seq[:, IDX_URGENCY] = np.clip(
                seq[:, IDX_URGENCY] + np.random.normal(0, mag * 0.3, T), 0.0, 1.0)

        elif FDI_TYPES[atype] == "load_injection_attack":


            load_delta = np.random.choice([-1, 1]) * mag
            if seq.shape[1] > IDX_AGG_P_PU:
                seq[:, IDX_AGG_P_PU] = np.clip(
                    seq[:, IDX_AGG_P_PU] + load_delta, -0.5, 1.5)
                seq[:, IDX_AGG_Q_PU] = np.clip(
                    seq[:, IDX_AGG_Q_PU] + load_delta * 0.3, -0.3, 0.5)
            if load_delta > 0:
                if seq.shape[1] > IDX_V_MIN_PU:
                    seq[:, IDX_V_MIN_PU] = np.clip(
                        seq[:, IDX_V_MIN_PU] - mag * 0.1, 0.85, 1.05)
            else:
                if seq.shape[1] > IDX_V_MAX_PU:
                    seq[:, IDX_V_MAX_PU] = np.clip(
                        seq[:, IDX_V_MAX_PU] + mag * 0.1, 0.95, 1.10)

        attacked[i] = seq

    return attacked, labels, atk_types


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:

    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)

    alpha_t = torch.where(targets == 1,
                          torch.tensor(alpha,     device=logits.device),
                          torch.tensor(1 - alpha, device=logits.device))
    return (alpha_t * (1 - pt) ** gamma * ce).mean()


def run_epoch(model, loader, optimiser=None):

    is_train = optimiser is not None
    model.train(is_train)

    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_attack_seen = 0
    n_attack_caught = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if is_train:
            optimiser.zero_grad()

        logits, _ = model(xb)
        loss = focal_loss(logits, yb)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)

        mask_atk = yb == 1
        n_attack_seen += mask_atk.sum().item()
        n_attack_caught += ((preds == 1) & mask_atk).sum().item()

    avg_loss = total_loss / max(n_total, 1)
    acc = n_correct / max(n_total, 1)
    recall = n_attack_caught / max(n_attack_seen, 1)
    return avg_loss, acc, recall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8000,
                        help="Total sequences to build from ACN CSVs")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length (must match LSTMIDSDetector)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attack-ratio", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--acn-data-dir", type=str,
        default=os.path.join("evcs_data", "ACN-Data-Static-main", "time series data"),
        help="Root of ACN-Data CSVs")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--sites", type=str, nargs="+",
                        default=["caltech/N_Wilson_Garage_01",
                                 "caltech/S_Wilson_Garage_01",
                                 "office_01/Parking_Lot_01"])
    parser.add_argument("--max-files-per-site", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print(" ACN-aligned LSTM IDS trainer (mirrors PINN feature pipeline)")
    print("=" * 78)
    print(f"  Device          : {DEVICE}")
    print(f"  ACN data dir    : {args.acn_data_dir}")
    print(f"  Sites           : {args.sites}")
    print(f"  n_samples       : {args.n_samples}  seq_len: {args.seq_len}")
    print(f"  epochs          : {args.epochs}     batch_size: {args.batch_size}")
    print(f"  attack_ratio    : {args.attack_ratio}")
    print()


    if not os.path.isdir(args.acn_data_dir):
        print(f"# ACN data dir not found: {args.acn_data_dir}")
        sys.exit(1)

    loader = ACNDataLoader(args.acn_data_dir)
    site_dirs = [(os.path.join(args.acn_data_dir, s), args.max_files_per_site)
                 for s in args.sites]
    n_loaded = loader.load_from_dirs(site_dirs)
    print(f"# Loaded {n_loaded} ACN sessions across {len(args.sites)} sites")

    if n_loaded == 0:
        print("# No ACN sessions loaded — check --acn-data-dir / --sites")
        sys.exit(1)


    clean_tensor = loader.build_ids_sequences(
        seq_len=args.seq_len, n_samples=args.n_samples)
    if clean_tensor is None:
        print("# build_ids_sequences returned None — check ACN data format")
        sys.exit(1)
    clean_seq = clean_tensor.numpy().astype(np.float32)
    print(f"# Built clean 20-D sequences of shape {clean_seq.shape}")


    attacked_seq, labels, atk_types = inject_fdi_on_abstract_features(
        clean_seq, attack_ratio=args.attack_ratio)
    print(f"# Injected FDI attacks: "
          f"{labels.sum()} / {len(labels)} sequences "
          f"(types: {[(t, int((atk_types == i).sum())) for i, t in enumerate(FDI_TYPES)]})")


    n_val = int(len(attacked_seq) * args.val_split)
    perm = np.random.permutation(len(attacked_seq))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_train = torch.from_numpy(attacked_seq[tr_idx]).float()
    y_train = torch.from_numpy(labels[tr_idx]).long()
    X_val = torch.from_numpy(attacked_seq[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)


    model = LSTMIDSModel(input_size=IDS_FEATURE_DIM, hidden_size=128, num_layers=2,
                         dropout=0.2).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    print(f"# Model: LSTMIDSModel({IDS_FEATURE_DIM}-dim input, 128-hidden, 2-layer)")


    hist = {"train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_recall": []}
    best_val_acc = 0.0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, optimiser)
        vl_loss, vl_acc, vl_rec = run_epoch(model, val_loader)
        scheduler.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(vl_acc)
        hist["val_recall"].append(vl_rec)

        print(f"  Epoch {ep:3d}/{args.epochs}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"atk-recall {vl_rec:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc

            best_path = os.path.join(args.output_dir, "lstm_ids_acn_trained.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size":       14,
                "hidden_size":      128,
                "num_layers":       2,
                "sequence_length":  args.seq_len,
                "anomaly_threshold": 0.5,
                "metadata": {
                    "train_source":  "ACNDataLoader (real ACN-Data)",
                    "n_sessions":    n_loaded,
                    "sites":         args.sites,
                    "fdi_types":     FDI_TYPES,
                    "epoch":         ep,
                    "val_acc":       vl_acc,
                    "val_recall":    vl_rec,
                },
            }, best_path)

    elapsed = time.time() - t0
    print(f"\n# Training complete in {elapsed:.1f}s   "
          f"best val_acc={best_val_acc:.3f}")


    drop_in = os.path.join(args.output_dir, "lstm_ids_pretrained.pth")
    import shutil
    shutil.copyfile(best_path, drop_in)
    print(f"# Saved drop-in checkpoint  {drop_in}")


    model.eval()
    print("\n Per-attack-type recall on validation set ")
    with torch.no_grad():
        X_val_d = X_val.to(DEVICE)
        logits, _ = model(X_val_d)
        preds = logits.argmax(dim=1).cpu().numpy()
    val_atk_types = atk_types[val_idx]
    for ti, tname in enumerate(FDI_TYPES):
        m = val_atk_types == ti
        if m.sum() == 0:
            continue
        caught = ((preds[m] == 1).sum())
        total = m.sum()
        print(f"    {tname:27s}: {caught:4d}/{total:4d}  "
              f"recall={caught / total:.3f}")
    benign_m = val_atk_types == -1
    if benign_m.sum() > 0:
        fp = (preds[benign_m] == 1).sum()
        total_b = benign_m.sum()
        print(f"    {'(benign)':27s}: {fp:4d}/{total_b:4d}  "
              f"FPR     ={fp / total_b:.3f}")


    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(hist["train_loss"], label="train")
        axes[0].plot(hist["val_loss"],   label="val")
        axes[0].set_title("Focal Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(hist["train_acc"], label="train")
        axes[1].plot(hist["val_acc"],   label="val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(hist["val_recall"], color="#C62828")
        axes[2].set_title("Val attack-recall"); axes[2].set_xlabel("Epoch")
        axes[2].grid(alpha=0.3)

        fig.suptitle("ACN-aligned LSTM IDS — training history",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig("lstm_ids_acn_training_history.png", dpi=120,
                    bbox_inches="tight")
        plt.close()
        print("# Saved lstm_ids_acn_training_history.png")
    except Exception as e:
        print(f"#  Plot skipped: {e}")


if __name__ == "__main__":
    main()
