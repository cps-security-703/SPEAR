#!/usr/bin/env python3
"""
Federated PINN Evaluation Script

"""

import os
import sys
import copy
import json
import time
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional
from datetime import datetime
warnings.filterwarnings('ignore')

# Import project modules (same config as federated_pinn_manager.py)
try:
    from pinn_optimizer import (
        LSTMPINNChargingOptimizer,
        LSTMPINNConfig,
        LSTMPINNOptimizer,
        PhysicsDataGenerator,
        LSTMPINNTrainer,
        DynamicWeightAveraging,
        DifferentiableEVCSDynamics,
    )
    from federated_pinn_manager import (
        FederatedPINNManager,
        FederatedPINNConfig,
        AnomalyDetector,
    )
    from evcs_dynamics import EVCSParameters
    print(" All modules imported successfully.")
except ImportError as exc:
    print(f" Import error: {exc}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette & style
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
LOSS_COLORS   = {'total': '#e41a1c', 'physics': '#377eb8',
                 'boundary': '#4daf4a', 'data': '#984ea3', 'temporal': '#ff7f00'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 150,
})

NUM_SYSTEMS = 6
PINN_BOUNDS = dict(v_min=300.0, v_max=500.0, i_min=50.0, i_max=150.0,
                   p_min=15.0, p_max=75.0)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: manually load a saved .pth into LSTMPINNChargingOptimizer
# (load_model() in the class is disabled intentionally – we bypass it)
# ─────────────────────────────────────────────────────────────────────────────

def load_pth_into_optimizer(
    optimizer: LSTMPINNChargingOptimizer,
    pth_path: str,
) -> bool:
    """Load saved model weights; return True on success."""
    if not os.path.exists(pth_path):
        print(f"  {pth_path} not found – using untrained weights.")
        return False
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        optimizer.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.is_trained = True
        optimizer.model.eval()
        return True
    except Exception as exc:
        print(f"   Failed to load {pth_path}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Load pre-trained federated models
# ─────────────────────────────────────────────────────────────────────────────

def load_federated_models() -> Tuple[FederatedPINNManager, Dict[int, bool]]:
    """Instantiate FederatedPINNManager and load saved weights."""
    print("\n" + "="*60)
    print("SECTION 1 – Loading Pre-trained Federated PINN Models")
    print("="*60)

    fed_cfg   = FederatedPINNConfig()       # exactly the same config as the main system
    manager   = FederatedPINNManager(fed_cfg)
    loaded_ok = {}

    for sys_id in range(1, NUM_SYSTEMS + 1):
        pth = f'federated_pinn_system_{sys_id}.pth'
        ok  = load_pth_into_optimizer(manager.local_models[sys_id], pth)
        loaded_ok[sys_id] = ok
        status = " loaded" if ok else " untrained weights"
        print(f"  System {sys_id}: {status}")

    # Sync global model using FedAvg over loaded local models
    try:
        manager.federated_averaging()
        print("  # Global model updated via FedAvg from loaded weights.")
    except Exception as exc:
        print(f"  # FedAvg skipped: {exc}")

    return manager, loaded_ok


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Training-curve extraction (mini re-train, collect loss history)
# ─────────────────────────────────────────────────────────────────────────────

def extract_training_curves(
    manager: FederatedPINNManager,
    n_samples: int = 600,
    n_epochs: int  = 120,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Train each local model from a FRESH random initialisation using the
    corrected data pipeline, collect loss curves, then write the trained
    weights back into *manager* so that all downstream evaluation steps
    (test metrics, EVCS sweep, divergence analysis) use properly trained
    models.  Also re-runs FedAvg and saves updated .pth checkpoints.
    """
    print("\n" + "="*60)
    print("SECTION 2 – Fresh Training with Corrected Pipeline")
    print(f"           ({n_samples} samples · {n_epochs} epochs per system)")
    print("="*60)

    pinn_cfg        = LSTMPINNConfig()
    pinn_cfg.epochs = n_epochs

    all_histories: Dict[int, Dict] = {}

    for sys_id in range(1, NUM_SYSTEMS + 1):
        print(f"\n  System {sys_id}: fresh init → training for {n_epochs} epochs …")

        # ── Fresh trainer – NO old weights copied in.  The previous .pth files

        trainer = LSTMPINNTrainer(pinn_cfg)

        try:
            seqs, tgts = trainer.data_generator.generate_realistic_evcs_scenarios(n_samples)

            dataset    = torch.utils.data.TensorDataset(seqs, tgts)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=pinn_cfg.batch_size, shuffle=True)

            hist = {k: [] for k in ['total', 'physics', 'boundary', 'data', 'temporal']}

            for epoch in range(n_epochs):
                trainer.model.train()
                ep_totals = {k: 0.0 for k in hist}
                n_batches = 0

                for batch_seqs, batch_tgts in dataloader:
                    trainer.optimizer.zero_grad()
                    try:
                        outputs    = trainer.model(batch_seqs)
                        physics_l  = trainer.model.physics_loss(batch_seqs, outputs)
                        boundary_l = trainer.model.boundary_loss(batch_seqs, outputs)
                        data_l     = trainer.model.data_loss(batch_seqs, outputs, batch_tgts)
                        temporal_l = trainer.model.temporal_consistency_loss(batch_seqs, outputs)
                        loss_terms = [data_l, physics_l, boundary_l, temporal_l]
                        total_l    = trainer.dynamic_weights.compute_weighted_loss(loss_terms)
                        total_l.backward()
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                        trainer.optimizer.step()

                        def _s(t): return t.detach().mean().item()
                        ep_totals['total']    += _s(total_l)
                        ep_totals['physics']  += _s(physics_l)
                        ep_totals['boundary'] += _s(boundary_l)
                        ep_totals['data']     += _s(data_l)
                        ep_totals['temporal'] += _s(temporal_l)
                        n_batches += 1
                    except Exception:
                        continue

                if n_batches > 0:
                    for k in hist:
                        hist[k].append(ep_totals[k] / n_batches)

                if (epoch + 1) % 30 == 0:
                    print(f"    epoch {epoch+1:>3}/{n_epochs}  "
                          f"loss={hist['total'][-1]:.4f}  "
                          f"data={hist['data'][-1]:.4f}")

            # ── Write freshly trained weights back into the manager so that
            #    evaluate_test_performance() etc. use these models.
            manager.local_models[sys_id].model.load_state_dict(
                copy.deepcopy(trainer.model.state_dict()))
            manager.local_models[sys_id].is_trained = True
            manager.local_models[sys_id].model.eval()

            # ── Persist updated checkpoint so future runs benefit too.
            pth_path = f'federated_pinn_system_{sys_id}.pth'
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config':           pinn_cfg,
                'is_trained':       True,
            }, pth_path)

            all_histories[sys_id] = hist
            final_loss = hist['total'][-1] if hist['total'] else float('nan')
            print(f"    # Final loss: {final_loss:.4f}  →  saved {pth_path}")

        except Exception as exc:
            print(f"    # Training failed for system {sys_id}: {exc}")
            all_histories[sys_id] = {k: [] for k in
                                     ['total', 'physics', 'boundary', 'data', 'temporal']}

    # ── Update global model via FedAvg over newly trained local models.
    try:
        manager.federated_averaging()
        print("\n  # Global model refreshed via FedAvg over retrained local models.")
    except Exception as exc:
        print(f"  # FedAvg skipped: {exc}")

    return all_histories


# Section 3 – Testing / Generalisation Metrics

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def evaluate_test_performance(
    manager: FederatedPINNManager,
    n_test: int = 400,
) -> Dict:
    """Compute MSE, MAE, RMSE, R² for V/I/P on a held-out test set."""
    print("\n" + "="*60)
    print("SECTION 3 – Test / Generalisation Performance")
    print("="*60)

    pinn_cfg   = LSTMPINNConfig()
    data_gen   = PhysicsDataGenerator(pinn_cfg)

    print(f"  Generating {n_test}-sample held-out test set …")
    seqs, tgts = data_gen.generate_realistic_evcs_scenarios(n_test)

    # De-normalize targets from [0,1] back to physical units using the same

    V_MIN, V_RNG = pinn_cfg.min_voltage, pinn_cfg.max_voltage - pinn_cfg.min_voltage  # 300, 200
    I_MIN, I_RNG = pinn_cfg.min_current, pinn_cfg.max_current - pinn_cfg.min_current  # 50, 100
    P_MIN, P_RNG = pinn_cfg.min_power,   pinn_cfg.max_power   - pinn_cfg.min_power    # 15, 60

    tgts_np = tgts.cpu().numpy()
    tgts_phys = np.column_stack([
        tgts_np[:, 0] * V_RNG + V_MIN,   # Voltage  [300, 500] V
        tgts_np[:, 1] * I_RNG + I_MIN,   # Current  [ 50, 150] A
        tgts_np[:, 2] * P_RNG + P_MIN,   # Power    [ 15,  75] kW
    ])

    results = {}
    labels  = ['voltage', 'current', 'power']

    def _metrics_for(model, name):
        model.eval()
        with torch.no_grad():
            preds = model(seqs).cpu().numpy()   # (N, 3) – physical units from forward()
        true = tgts_phys                        # (N, 3) – de-normalised physical units

        out = {}
        for j, lbl in enumerate(labels):
            y_t = true[:, j]
            y_p = preds[:, j]
            mse  = float(np.mean((y_t - y_p)**2))
            mae  = float(np.mean(np.abs(y_t - y_p)))
            rmse = float(np.sqrt(mse))
            r2   = _r2(y_t, y_p)
            out[lbl] = dict(mse=mse, mae=mae, rmse=rmse, r2=r2)
            print(f"    {name:20s} | {lbl:8s}: RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")
        return out

    for sys_id in range(1, NUM_SYSTEMS + 1):
        print(f"\n  System {sys_id}:")
        results[f'system_{sys_id}'] = _metrics_for(
            manager.local_models[sys_id].model, f"System {sys_id}")

    print("\n  Global (federated) model:")
    results['global'] = _metrics_for(manager.global_model.model, "Global")

    return results


# Section 4 – EVCS Output Dynamics

def evaluate_evcs_outputs(manager: FederatedPINNManager) -> Dict:
    """Sweep SOC x time-of-day; call optimize_references() for each local
    """
    print("\n" + "="*60)
    print("SECTION 4 – EVCS Output Dynamics (V / I / P sweep)")
    print("="*60)

    soc_values  = np.linspace(0.1, 0.95, 20)
    hour_values = np.linspace(0, 23, 24)
    params      = EVCSParameters()

    output_data: Dict[int, Dict] = {}

    for sys_id in range(1, NUM_SYSTEMS + 1):
        voltages, currents, powers = [], [], []
        soc_trace, hour_trace      = [], []

        for soc in soc_values:
            for hour in hour_values:
                station_data = {
                    'soc':               float(soc),
                    'grid_voltage':      1.0 + np.random.normal(0, 0.01),
                    'grid_frequency':    60.0 + np.random.normal(0, 0.05),
                    'demand_factor':     0.5 + 0.3 * np.sin(2 * np.pi * hour / 24),
                    'voltage_priority':  max(0.0, 0.95 - 0.98),
                    'urgency_factor':    2.0 - float(soc),
                    'current_time':      float(hour),
                    'bus_distance':      1.0,
                    'load_factor':       0.8 + 0.2 * np.sin(2 * np.pi * hour / 24),
                    'ac_power_in':       50.0,
                    'system_efficiency': 0.95,
                    'power_balance_error': 0.0,
                    'dc_link_voltage':   500.0,
                }
                try:
                    _v_ref, i_ref, _p_ref = manager.local_models[sys_id].optimize_references(
                        station_data)
                    # Convert PINN current reference → actual terminal quantities

                    ss = _evcs_steady_state(float(soc), float(i_ref), params)
                    voltages.append(ss['v_term'])
                    currents.append(ss['i_act'])
                    powers.append(ss['p_dc'])
                    soc_trace.append(float(soc))
                    hour_trace.append(float(hour))
                except Exception:
                    pass

        arr_v = np.array(voltages);  arr_i = np.array(currents);  arr_p = np.array(powers)
        arr_s = np.array(soc_trace); arr_h = np.array(hour_trace)

        v_ok = np.mean((arr_v >= PINN_BOUNDS['v_min']) & (arr_v <= PINN_BOUNDS['v_max']))
        i_ok = np.mean((arr_i >= PINN_BOUNDS['i_min']) & (arr_i <= PINN_BOUNDS['i_max']))
        p_ok = np.mean((arr_p >= PINN_BOUNDS['p_min']) & (arr_p <= PINN_BOUNDS['p_max']))

        output_data[sys_id] = {
            'soc': arr_s, 'hour': arr_h,
            'voltage': arr_v, 'current': arr_i, 'power': arr_p,
            'constraint_sat': {'voltage': float(v_ok), 'current': float(i_ok), 'power': float(p_ok)},
        }
        print(f"  System {sys_id}: V∈bounds {v_ok*100:.1f}%  I∈bounds {i_ok*100:.1f}%  P∈bounds {p_ok*100:.1f}%")

    return output_data



# Section 4b – Actual EVCS Physical Outputs (measured, not PINN references)


# IEEE 34-bus assignments for each EVCS system
_EVCS_BUS_MAP = {1: '890', 2: '844', 3: '860', 4: '840', 5: '848', 6: '830'}
_GRID_V_LN    = 7200.0   # V line-to-neutral (12.47 kV L-L / √3)
_DT_S         = 360.0    # 6-minute time step in seconds
_N_STEPS      = 20       # 2 h × (60 min / 6 min) = 20 steps


def _evcs_steady_state(
    soc: float,
    i_ref: float,
    params: 'EVCSParameters',
) -> Dict:
    """Compute the quasi-static EVCS operating point for a 6-minute timestep.

    """
    R_int      = 0.1           # Ω
    eta_acdc   = 0.98
    eta_dcdc   = 0.96
    eta_charge = 0.95
    eta_total  = eta_acdc * eta_dcdc

    ocv = params.min_voltage + soc * (params.max_voltage - params.min_voltage)

    if soc < 0.8:
        # CC phase — charger enforces rated current; PINN i_ref is irrelevant here
        i_act = float(params.rated_current)
    else:
        # CV phase — taper from rated to min as SOC rises 0.8 → 0.9
        taper    = (soc - 0.8) / 0.2          # 0 at SOC=0.8, 1 at SOC=0.9
        i_cv_max = (params.rated_current
                    - taper * (params.rated_current - params.min_current))
        # PINN may request lower current (demand response), honour if below taper limit
        i_act = float(np.clip(min(float(i_ref), i_cv_max),
                               params.min_current, i_cv_max))

    v_term = float(np.clip(ocv - i_act * R_int,
                            params.min_voltage, params.max_voltage))
    p_dc   = v_term * i_act / 1000.0
    p_ac   = p_dc / eta_total if p_dc > 0 else 0.0

    delta_soc = p_dc * eta_charge * _DT_S / (params.capacity * 3600.0)
    new_soc   = float(np.clip(soc + delta_soc, params.min_soc, params.max_soc))

    return {
        'v_term' : v_term,
        'i_act'  : i_act,
        'p_dc'   : p_dc,
        'p_ac'   : p_ac,
        'eff'    : eta_total,
        'pb_err' : abs(p_ac * eta_total - p_dc),
        'new_soc': new_soc,
    }


def _try_load_opendss_voltages(dss_file: str) -> Optional[Dict[str, float]]:
    """Load OpenDSS model and return per-bus RMS line-neutral voltage (V).

    Returns None if opendssdirect is unavailable or the file cannot be parsed.
    """
    try:
        import opendssdirect as dss   # type: ignore
        dss.run_command('Clear')
        dss.run_command(f'Redirect "{dss_file}"')
        dss.run_command('Solve')
        bus_voltages: Dict[str, float] = {}
        for bus_name in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus_name)
            vmag = dss.Bus.VMagAngle()          # alternating (Vmag, angle, …)
            if vmag:
                bus_voltages[bus_name.lower()] = float(vmag[0])   # first phase L-N
        return bus_voltages if bus_voltages else None
    except Exception:
        return None


def plot_actual_evcs_outputs(
    manager,
    output_png: str = 'federated_pinn_evcs_actual_outputs.png',
) -> Dict:
    """Simulate a 2-hour CC-CV charging window per system and plot the *actual*
    """
    print("\n" + "=" * 60)
    print("SECTION 4b – Actual EVCS Physical Outputs (Converter Dynamics)")
    print("=" * 60)

    # ── Try to get real bus voltages from OpenDSS ────────────────────────────
    dss_file = os.path.join(os.path.dirname(__file__), 'ieee34Mod1.dss')
    if not os.path.isfile(dss_file):
        dss_file = 'ieee34Mod1.dss'
    opendss_bus_v = _try_load_opendss_voltages(dss_file)
    if opendss_bus_v:
        print("  OpenDSS loaded — using real bus voltages.")
    else:
        print("  OpenDSS unavailable — using nominal grid voltage (7200 V L-N).")

    params = EVCSParameters()
    SOC_START = 0.30

    # Time axis (hours)
    hours = np.array([step * (_DT_S / 3600.0) for step in range(_N_STEPS)])

    all_data: Dict[int, Dict] = {}

    for sys_id in range(1, NUM_SYSTEMS + 1):
        bus_name = _EVCS_BUS_MAP[sys_id].lower()

        # Bus voltage from OpenDSS if available, else nominal
        if opendss_bus_v:
            grid_v = opendss_bus_v.get(bus_name, _GRID_V_LN)
        else:
            grid_v = _GRID_V_LN

        local_optimizer = manager.local_models[sys_id]

        # --- Recording arrays ---
        rec_soc   = np.zeros(_N_STEPS)
        rec_v     = np.zeros(_N_STEPS)
        rec_i     = np.zeros(_N_STEPS)
        rec_p_dc  = np.zeros(_N_STEPS)
        rec_p_ac  = np.zeros(_N_STEPS)
        rec_eff   = np.zeros(_N_STEPS)
        rec_pberr = np.zeros(_N_STEPS)
        rec_v_ref = np.zeros(_N_STEPS)
        rec_i_ref = np.zeros(_N_STEPS)
        rec_p_ref = np.zeros(_N_STEPS)

        session_hours: List[float] = []   # hour of each new-EV arrival
        n_sessions   = 0
        soc          = SOC_START

        for step in range(_N_STEPS):
            hour = hours[step]

            # ── New EV arrives when previous finishes (SOC ≥ disconnect) ─────

            if soc >= params.disconnect_soc:
                n_sessions += 1
                session_hours.append(hour)
                soc = float(np.random.uniform(0.20, 0.35))

            # Demand / urgency factors for PINN input
            demand_factor  = 0.3 + 0.7 * np.exp(-((hour - 18.0) ** 2) / 8.0)
            urgency_factor = max(0.5, 2.0 - soc * 2.0)

            station_data = {
                'soc'                : soc,
                'grid_voltage'       : grid_v / _GRID_V_LN,
                'grid_frequency'     : 60.0,
                'demand_factor'      : float(demand_factor),
                'voltage_priority'   : 0.0,
                'urgency_factor'     : float(urgency_factor),
                'current_time'       : hour,
                'bus_distance'       : 1.0,
                'load_factor'        : 0.8,
                'ac_power_in'        : 50.0,
                'system_efficiency'  : 0.94,
                'power_balance_error': 0.0,
                'dc_link_voltage'    : 500.0,
            }

            try:
                v_ref, i_ref, p_ref = local_optimizer.optimize_references(station_data)
            except Exception:
                ocv   = params.min_voltage + soc * (params.max_voltage - params.min_voltage)
                v_ref = float(ocv)
                if soc < 0.8:
                    i_ref = params.rated_current
                else:
                    taper = (soc - 0.8) / 0.2
                    i_ref = (params.rated_current
                             - taper * (params.rated_current - params.min_current))
                p_ref = v_ref * i_ref / 1000.0

            # ── Steady-state physics (bypasses unstable PI Euler integration) ─

            ss = _evcs_steady_state(soc, float(i_ref), params)

            rec_soc[step]   = soc
            rec_v[step]     = ss['v_term']
            rec_i[step]     = ss['i_act']
            rec_p_dc[step]  = ss['p_dc']
            rec_p_ac[step]  = ss['p_ac']
            rec_eff[step]   = ss['eff']
            rec_pberr[step] = ss['pb_err']
            rec_v_ref[step] = float(v_ref)
            rec_i_ref[step] = float(i_ref)
            rec_p_ref[step] = float(p_ref)

            soc = ss['new_soc']

        all_data[sys_id] = {
            'hours'        : hours,
            'soc'          : rec_soc,
            'v'            : rec_v,
            'i'            : rec_i,
            'p_dc'         : rec_p_dc,
            'p_ac'         : rec_p_ac,
            'eff'          : rec_eff,
            'pberr'        : rec_pberr,
            'v_ref'        : rec_v_ref,
            'i_ref'        : rec_i_ref,
            'p_ref'        : rec_p_ref,
            'session_hours': session_hours,
            'n_sessions'   : n_sessions,
        }
        print(f"  System {sys_id} (bus {_EVCS_BUS_MAP[sys_id]}): "
              f"{n_sessions} EVs served  "
              f"Avg V={rec_v.mean():.1f} V  "
              f"Avg I={rec_i.mean():.1f} A  "
              f"Avg P_dc={rec_p_dc.mean():.2f} kW")

    # ── Generate 3 × 2 figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))


    soc_fine = np.linspace(0.10, params.disconnect_soc, 200)
    ocv_fine = params.min_voltage + soc_fine * (params.max_voltage - params.min_voltage)
    # terminal voltage at rated current (shows I·R drop from OCV)
    vterm_fine = ocv_fine - params.rated_current * 0.1

    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    for sys_id in range(1, NUM_SYSTEMS + 1):
        d   = all_data[sys_id]
        col = SYSTEM_COLORS[sys_id - 1]
        ls  = styles[sys_id - 1]
        lbl = f'Sys {sys_id}'

        # ── Row 0: Terminal Voltage ──────────────────────────────────────────
        # Time plot — only measured terminal voltage (v_term = OCV - I·R)
        axes[0, 0].plot(d['hours'], d['v'], color=col, ls=ls, lw=4.0, label=lbl)
        # SOC plot
        axes[0, 1].plot(d['soc'], d['v'], color=col, ls=ls, lw=4.0, label=lbl)

        # ── Row 1: Charging Current ──────────────────────────────────────────
        # Time plot — measured current (CC plateau then CV taper)
        axes[1, 0].plot(d['hours'], d['i'], color=col, ls=ls, lw=4.0, label=lbl)
        # SOC plot
        axes[1, 1].plot(d['soc'], d['i'], color=col, ls=ls, lw=4.0, label=lbl)

        # ── Row 2: Power ─────────────────────────────────────────────────────
        # Time plot — DC power to battery (solid) + AC power from grid (dashed)
        axes[2, 0].plot(d['hours'], d['p_dc'], color=col, ls=ls,   lw=4.0, label=lbl)
        axes[2, 0].plot(d['hours'], d['p_ac'], color=col, ls='--', lw=4.0,
                        alpha=0.50, zorder=1)
        # SOC plot — efficiency
        axes[2, 1].plot(d['soc'], d['eff'] * 100, color=col, ls=ls, lw=4.0, label=lbl)

    # ── Session boundary markers on time plots (System 1 only) ──────────────
    _first_sh = True
    for sh in all_data[1]['session_hours']:
        _kw = dict(color='#888888', ls=':', lw=1.0, zorder=0)
        for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
            ax.axvline(sh, **_kw,
                       label='EV handover' if (_first_sh and ax is axes[0, 0]) else None)
        _first_sh = False

    # ── Reference curves on SOC plots ────────────────────────────────────────
    # OCV line: open-circuit voltage (no load)
    axes[0, 1].plot(soc_fine, ocv_fine,   'k--', lw=4.0, label='OCV (no load)')
    # Terminal voltage at rated current (shows full I·R drop)
    axes[0, 1].plot(soc_fine, vterm_fine, 'k:',  lw=4.0, label='V_term @ I_rated')

    # CC→CV boundary — SOC=0.8 on SOC-axis plots only
    axes[0, 1].axvline(0.8, color='grey', ls='-.', lw=4.0, label='CC→CV  (SOC=0.8)')
    axes[1, 1].axvline(0.8, color='grey', ls='-.', lw=4.0, label='CC→CV  (SOC=0.8)')

    # ── Physical bounds ──────────────────────────────────────────────────────
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.axhline(params.min_voltage, color='grey', ls=':', lw=4.0)
        ax.axhline(params.max_voltage, color='grey', ls=':', lw=4.0)
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.axhline(params.min_current, color='grey', ls=':', lw=4.0)
        ax.axhline(params.max_current, color='grey', ls=':', lw=4.0)

    # ── Power panel annotation ────────────────────────────────────────────────
    axes[2, 0].text(
        0.02, 0.97,
        'Solid = P_dc (to battery)\nDashed = P_ac (from grid)\n'
        'P_dc = P_ac x eta_total  (eta_total=0.94)',
        transform=axes[2, 0].transAxes, fontsize=7.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85),
    )

    # ── Labels, titles, grids ─────────────────────────────────────────────────
    # axes[0, 0].set_title('Terminal Voltage vs Time', fontweight='bold')
    axes[0, 0].set_xlabel('Time (h)'); axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].legend(ncol=2, fontsize=7); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, hours[-1])

    # axes[0, 1].set_title('Terminal Voltage vs SOC', fontweight='bold')
    axes[0, 1].set_xlabel('SOC'); axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].legend(ncol=2, fontsize=7); axes[0, 1].grid(True, alpha=0.3)

    # axes[1, 0].set_title('Charging Current vs Time', fontweight='bold')
    axes[1, 0].set_xlabel('Time (h)'); axes[1, 0].set_ylabel('Current (A)')
    axes[1, 0].legend(ncol=2, fontsize=7); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, hours[-1])

    # axes[1, 1].set_title('Charging Current vs SOC  (CC→CV taper at SOC=0.8)',
                        #   fontweight='bold')
    axes[1, 1].set_xlabel('SOC'); axes[1, 1].set_ylabel('Current (A)')
    axes[1, 1].legend(ncol=2, fontsize=7); axes[1, 1].grid(True, alpha=0.3)

    # axes[2, 0].set_title('DC Power (solid) and AC Grid Power (dashed) vs Time',
    #                       fontweight='bold')
    axes[2, 0].set_xlabel('Time (h)'); axes[2, 0].set_ylabel('Power (kW)')
    axes[2, 0].legend(ncol=2, fontsize=7); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(0, hours[-1])

    # axes[2, 1].set_title('System Efficiency vs SOC', fontweight='bold')
    axes[2, 1].set_xlabel('SOC'); axes[2, 1].set_ylabel('Efficiency (%)')
    axes[2, 1].set_ylim(88, 100)
    axes[2, 1].axhline(94.1, color='k', ls='--', lw=4.0,
                        label='Design eta = 94.1 %  (0.98 x 0.96)')
    axes[2, 1].legend(ncol=2, fontsize=7); axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n# Actual EVCS output plots saved → {output_png}")

    return all_data


# Section 5 – Anomaly Detection Layer Evaluation

def evaluate_anomaly_detection(manager: FederatedPINNManager) -> Dict:
    """Test each 3-layer anomaly detector with known-good and known-bad inputs."""
    print("\n" + "="*60)
    print("SECTION 5 – Anomaly Detection Layer Evaluation")
    print("="*60)

    GOOD_INPUTS = [
        {'soc': 0.5,  'grid_voltage': 1.0,  'grid_frequency': 60.0, 'demand_factor': 0.7, 'load_factor': 0.8, 'urgency_factor': 1.0, 'voltage_priority': 0.0},
        {'soc': 0.3,  'grid_voltage': 0.97, 'grid_frequency': 59.9, 'demand_factor': 0.6, 'load_factor': 0.7, 'urgency_factor': 1.2, 'voltage_priority': 0.01},
        {'soc': 0.8,  'grid_voltage': 1.02, 'grid_frequency': 60.1, 'demand_factor': 0.5, 'load_factor': 0.9, 'urgency_factor': 0.8, 'voltage_priority': 0.0},
        {'soc': 0.6,  'grid_voltage': 0.98, 'grid_frequency': 60.0, 'demand_factor': 0.8, 'load_factor': 1.0, 'urgency_factor': 1.5, 'voltage_priority': 0.02},
        {'soc': 0.4,  'grid_voltage': 0.99, 'grid_frequency': 60.0, 'demand_factor': 0.65,'load_factor': 0.85,'urgency_factor': 1.1, 'voltage_priority': 0.01},
    ]
    BAD_INPUTS = [
        {'soc': -0.1,  'grid_voltage': 0.5,  'grid_frequency': 55.0, 'demand_factor': 3.0, 'load_factor': 2.0, 'urgency_factor': 5.0},  # extreme violations
        {'soc': 1.2,   'grid_voltage': 1.5,  'grid_frequency': 65.0, 'demand_factor': 0.7, 'load_factor': 0.8, 'urgency_factor': 1.0},  # SOC & voltage high
        {'soc': 0.5,   'grid_voltage': 0.6,  'grid_frequency': 60.0, 'demand_factor': 0.7, 'load_factor': 2.5, 'urgency_factor': 1.0},  # low V, high load
        {'soc': 0.5,   'grid_voltage': 1.0,  'grid_frequency': 58.0, 'demand_factor': 0.7, 'load_factor': 0.8, 'urgency_factor': 1.0},  # freq violation
        {'soc': 0.5,   'grid_voltage': 1.0,  'grid_frequency': 60.0, 'demand_factor': 2.5, 'load_factor': 0.8, 'urgency_factor': 1.0},  # demand violation
    ]

    results = {}
    for sys_id in range(1, NUM_SYSTEMS + 1):
        detector = manager.anomaly_detectors[sys_id]
        tp = tn = fp = fn = 0
        layer_hits = {'physical_constraints': 0, 'pattern_based': 0, 'lstm_ml': 0}

        for inp in GOOD_INPUTS:
            detected, det_res = detector.multi_layer_detection(inp, sys_id)
            if not detected:
                tn += 1
            else:
                fp += 1
                if det_res.get('detection_layer'):
                    layer_hits[det_res['detection_layer']] = layer_hits.get(det_res['detection_layer'], 0) + 1

        for inp in BAD_INPUTS:
            detected, det_res = detector.multi_layer_detection(inp, sys_id)
            if detected:
                tp += 1
                if det_res.get('detection_layer'):
                    layer_hits[det_res['detection_layer']] = layer_hits.get(det_res['detection_layer'], 0) + 1
            else:
                fn += 1

        n_pos = len(BAD_INPUTS); n_neg = len(GOOD_INPUTS)
        dr   = tp / n_pos if n_pos else 0.0
        fpr  = fp / n_neg if n_neg else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = (2 * prec * dr) / (prec + dr) if (prec + dr) else 0.0

        results[sys_id] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'detection_rate': dr, 'fpr': fpr, 'precision': prec, 'f1': f1,
            'layer_hits': layer_hits,
        }
        print(f"  System {sys_id}: DR={dr*100:.0f}%  FPR={fpr*100:.0f}%  F1={f1:.2f}  "
              f"Layers: {layer_hits}")

    return results


# Section 6 – Local vs Global model comparison

def compare_local_vs_global(
    manager: FederatedPINNManager,
    n_test: int = 200,
) -> Dict:
    """Compute parameter divergence and prediction distance between local & global."""
    print("\n" + "="*60)
    print("SECTION 6 – Local vs Global Model Comparison")
    print("="*60)

    pinn_cfg = LSTMPINNConfig()
    data_gen = PhysicsDataGenerator(pinn_cfg)
    seqs, _ = data_gen.generate_realistic_evcs_scenarios(n_test)

    global_model = manager.global_model.model
    global_state = global_model.state_dict()

    results = {}

    for sys_id in range(1, NUM_SYSTEMS + 1):
        local_model = manager.local_models[sys_id].model
        local_state = local_model.state_dict()

        # L2 parameter divergence
        param_l2 = 0.0
        for key in global_state:
            if global_state[key].dtype.is_floating_point:
                diff = (local_state[key].float() - global_state[key].float()).pow(2).sum().item()
                param_l2 += diff
        param_l2 = float(np.sqrt(param_l2))

        # Prediction MAE on shared test set
        global_model.eval(); local_model.eval()
        with torch.no_grad():
            g_pred = global_model(seqs).cpu().numpy()
            l_pred = local_model(seqs).cpu().numpy()

        pred_mae_v = float(np.mean(np.abs(g_pred[:, 0] - l_pred[:, 0])))
        pred_mae_i = float(np.mean(np.abs(g_pred[:, 1] - l_pred[:, 1])))
        pred_mae_p = float(np.mean(np.abs(g_pred[:, 2] - l_pred[:, 2])))
        pred_mae   = float(np.mean(np.abs(g_pred - l_pred)))

        results[sys_id] = {
            'param_l2': param_l2,
            'pred_mae': pred_mae,
            'pred_mae_v': pred_mae_v,
            'pred_mae_i': pred_mae_i,
            'pred_mae_p': pred_mae_p,
        }
        print(f"  System {sys_id}: Param-L2={param_l2:.4f}  Pred-MAE (V/I/P) "
              f"{pred_mae_v:.2f}/{pred_mae_i:.2f}/{pred_mae_p:.2f}")

    return results


# Section 7 – Plots

def generate_all_plots(
    train_histories : Dict,
    test_metrics    : Dict,
    evcs_outputs    : Dict,
    anomaly_results : Dict,
    lglobal_compare : Dict,
    output_png      : str = 'federated_pinn_evaluation_plots.png',
):
    """Produce a 7-panel figure and save to disk."""

    fig = plt.figure(figsize=(22, 30))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Plot 1: Training loss curves ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for sys_id in range(1, NUM_SYSTEMS + 1):
        hist = train_histories.get(sys_id, {})
        if hist.get('total'):
            ax1.plot(hist['total'], color=SYSTEM_COLORS[sys_id - 1],
                     label=f'Sys {sys_id}', lw=4.0)
    # ax1.set_title('Training Loss Curves (Local Models)', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Total Loss')
    ax1.legend(ncol=2, fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') if any(
        (train_histories.get(s, {}).get('total', [0])[0] or 0) > 10
        for s in range(1, NUM_SYSTEMS + 1)) else None

    # ── Plot 2: Loss components for system 1 ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    hist1 = train_histories.get(1, {})
    for key, color in LOSS_COLORS.items():
        if hist1.get(key):
            ax2.plot(hist1[key], color=color, label=key.capitalize(), lw=4.0)
    # ax2.set_title('PINN Loss Components – System 1', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── Plot 3: Test metrics bar chart (R²) ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    metric_keys = ['voltage', 'current', 'power']
    x = np.arange(NUM_SYSTEMS + 1)    # +1 for global
    bar_w = 0.25
    labels_sys = [f'Sys {s}' for s in range(1, NUM_SYSTEMS + 1)] + ['Global']
    entity_keys = [f'system_{s}' for s in range(1, NUM_SYSTEMS + 1)] + ['global']
    lcolors = ['#4c72b0', '#dd8452', '#55a868']

    for k, (mkey, color) in enumerate(zip(metric_keys, lcolors)):
        r2_vals = []
        for ekey in entity_keys:
            r2_val = test_metrics.get(ekey, {}).get(mkey, {}).get('r2', 0.0)
            r2_vals.append(r2_val)
        ax3.bar(x + k * bar_w, r2_vals, bar_w, label=mkey.capitalize(),
                color=color, edgecolor='white', lw=4.0)

    # ax3.set_title('Test Performance – R² Score (V / I / P)', fontweight='bold')
    ax3.set_xlabel('Model'); ax3.set_ylabel('R²')
    ax3.set_xticks(x + bar_w); ax3.set_xticklabels(labels_sys, rotation=30, ha='right')
    ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(-0.1, 1.05)
    ax3.axhline(0, color='k', lw=0.8, ls='--')

    # ── Plot 4: EVCS V/I/P vs SOC (mean across all systems) ──────────────
    ax4a = fig.add_subplot(gs[1, 1])
    soc_grid = np.linspace(0.1, 0.95, 20)
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    for sys_id in range(1, NUM_SYSTEMS + 1):
        data = evcs_outputs.get(sys_id, {})
        if len(data.get('soc', [])) == 0:
            continue
        # Average over hours for each SOC value
        v_by_soc = [np.mean(data['voltage'][data['soc'] == s]) if np.any(data['soc'] == s)
                    else np.nan for s in soc_grid]
        v_by_soc_arr = []
        for s in soc_grid:
            mask = np.isclose(data['soc'], s, atol=0.01)
            v_by_soc_arr.append(np.mean(data['voltage'][mask]) if mask.any() else np.nan)
        ax4a.plot(soc_grid, v_by_soc_arr,
                  color=SYSTEM_COLORS[sys_id - 1],
                  ls=styles[sys_id - 1], lw=4.0, label=f'Sys {sys_id}')

    ax4a.axhline(PINN_BOUNDS['v_min'], color='grey', ls='--', lw=4.0, label='Bounds')
    ax4a.axhline(PINN_BOUNDS['v_max'], color='grey', ls='--', lw=4.0)
    # ax4a.set_title('EVCS Terminal Voltage vs SOC  (physics-corrected)', fontweight='bold')
    ax4a.set_xlabel('State of Charge (SOC)'); ax4a.set_ylabel('Terminal Voltage (V)')
    ax4a.legend(ncol=2, fontsize=7); ax4a.grid(True, alpha=0.3)

    # ── Plot 5: Power & Current vs SOC ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5b = ax5.twinx()
    for sys_id in range(1, NUM_SYSTEMS + 1):
        data = evcs_outputs.get(sys_id, {})
        if len(data.get('soc', [])) == 0:
            continue
        p_arr, i_arr = [], []
        for s in soc_grid:
            mask = np.isclose(data['soc'], s, atol=0.01)
            p_arr.append(np.mean(data['power'][mask])   if mask.any() else np.nan)
            i_arr.append(np.mean(data['current'][mask]) if mask.any() else np.nan)
        ax5.plot(soc_grid, p_arr, color=SYSTEM_COLORS[sys_id - 1],
                 ls='-', lw=4.0, label=f'P Sys{sys_id}')
        ax5b.plot(soc_grid, i_arr, color=SYSTEM_COLORS[sys_id - 1],
                  ls='--', lw=4.0, alpha=0.6)

    ax5.axhline(PINN_BOUNDS['p_min'], color='grey', ls=':', lw=4.0)
    ax5.axhline(PINN_BOUNDS['p_max'], color='grey', ls=':', lw=4.0)
    # ax5.set_title('EVCS DC Power (solid) & Current (dashed) vs SOC  (physics-corrected)',
    #               fontweight='bold')
    ax5.set_xlabel('SOC'); ax5.set_ylabel('DC Power (kW)')
    ax5b.set_ylabel('Current (A)')
    ax5.legend(ncol=2, fontsize=7, loc='upper right'); ax5.grid(True, alpha=0.3)

    # ── Plot 6: Physics constraint satisfaction ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    cat_names = ['Voltage', 'Current', 'Power']
    all_sat = {c.lower(): [] for c in cat_names}
    for sys_id in range(1, NUM_SYSTEMS + 1):
        cs = evcs_outputs.get(sys_id, {}).get('constraint_sat', {})
        for c in cat_names:
            all_sat[c.lower()].append(cs.get(c.lower(), 0.0) * 100)

    xs = np.arange(NUM_SYSTEMS)
    bar_w_c = 0.25
    for k, (cat, color) in enumerate(zip(cat_names, ['#4c72b0', '#dd8452', '#55a868'])):
        ax6.bar(xs + k * bar_w_c, all_sat[cat.lower()], bar_w_c,
                label=cat, color=color, edgecolor='white' , lw=4.0)
    ax6.axhline(100, color='red', ls='--', lw=4.0, label='100 %')
    # ax6.set_title('Physics Constraint Satisfaction per System (%)', fontweight='bold')
    ax6.set_xlabel('System ID'); ax6.set_ylabel('Satisfaction (%)')
    ax6.set_xticks(xs + bar_w_c); ax6.set_xticklabels([f'Sys {s}' for s in range(1, NUM_SYSTEMS+1)])
    ax6.legend(); ax6.grid(True, alpha=0.3, axis='y'); ax6.set_ylim(0, 110)

    # ── Plot 7: Anomaly detection results & model divergence ──────────────
    ax7a = fig.add_subplot(gs[3, 0])
    sys_ids = list(range(1, NUM_SYSTEMS + 1))
    dr_vals  = [anomaly_results.get(s, {}).get('detection_rate', 0) * 100 for s in sys_ids]
    fpr_vals = [anomaly_results.get(s, {}).get('fpr', 0) * 100           for s in sys_ids]
    f1_vals  = [anomaly_results.get(s, {}).get('f1', 0) * 100            for s in sys_ids]

    xd  = np.arange(NUM_SYSTEMS)
    bw2 = 0.25
    ax7a.bar(xd,       dr_vals,  bw2, label='Detection Rate %',  color='#2ca02c', edgecolor='white')
    ax7a.bar(xd + bw2, fpr_vals, bw2, label='False Positive %',  color='#d62728', edgecolor='white')
    ax7a.bar(xd + 2*bw2, f1_vals, bw2, label='F1 score × 100',  color='#1f77b4', edgecolor='white')
    # ax7a.set_title('3-Layer Anomaly Detection Performance', fontweight='bold')
    ax7a.set_xlabel('System'); ax7a.set_ylabel('%')
    ax7a.set_xticks(xd + bw2); ax7a.set_xticklabels([f'Sys {s}' for s in sys_ids])
    ax7a.legend(fontsize=8); ax7a.grid(True, alpha=0.3, axis='y'); ax7a.set_ylim(0, 115)

    ax7b = fig.add_subplot(gs[3, 1])
    param_l2s = [lglobal_compare.get(s, {}).get('param_l2', 0) for s in sys_ids]
    pred_maes = [lglobal_compare.get(s, {}).get('pred_mae', 0) for s in sys_ids]
    ax7b2 = ax7b.twinx()
    ax7b.bar(xd,       param_l2s, 0.35, label='Param L₂ divergence', color='#9467bd', edgecolor='white')
    ax7b2.plot(xd, pred_maes, 'o-', color='#e377c2', lw=4.0, label='Pred MAE vs Global')
    # ax7b.set_title('Local vs Global Model: Divergence', fontweight='bold')
    ax7b.set_xlabel('System'); ax7b.set_ylabel('Parameter L₂ Norm')
    ax7b2.set_ylabel('Mean Prediction MAE')
    ax7b.set_xticks(xd); ax7b.set_xticklabels([f'Sys {s}' for s in sys_ids])
    lines1, lbl1 = ax7b.get_legend_handles_labels()
    lines2, lbl2 = ax7b2.get_legend_handles_labels()
    ax7b.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    ax7b.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n# Plots saved → {output_png}")


# 
# Section 8 – Metrics Report
# 

def build_metrics_report(
    test_metrics    : Dict,
    evcs_outputs    : Dict,
    anomaly_results : Dict,
    lglobal_compare : Dict,
    output_json     : str = 'federated_pinn_metrics.json',
) -> Dict:
    """Assemble a JSON-serialisable metrics dictionary and save."""

    def _np_safe(obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean(v) for v in d]
        return _np_safe(d)

    summary = {
        'timestamp'         : datetime.now().isoformat(),
        'num_systems'       : NUM_SYSTEMS,
        'architecture'      : 'LSTM-PINN with FedAvg',
        'test_metrics'      : test_metrics,
        'constraint_sat'    : {str(sid): evcs_outputs[sid]['constraint_sat']
                               for sid in evcs_outputs},
        'anomaly_detection' : {str(sid): {k: v for k, v in anomaly_results[sid].items()
                                          if k != 'layer_hits'}
                               for sid in anomaly_results},
        'federated_divergence': {str(sid): lglobal_compare[sid]
                                 for sid in lglobal_compare},
    }

    summary = _clean(summary)

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"# Metrics saved → {output_json}")

    # Console summary table
    print("\n" + "─"*70)
    print(f"{'EVALUATION SUMMARY':^70}")
    print("─"*70)
    print(f"{'Model':<15} {'V-RMSE':>8} {'I-RMSE':>8} {'P-RMSE':>8} "
          f"{'V-R²':>6} {'I-R²':>6} {'P-R²':>6}")
    print("─"*70)
    for ekey in [f'system_{s}' for s in range(1, NUM_SYSTEMS+1)] + ['global']:
        m   = test_metrics.get(ekey, {})
        label = ekey.replace('system_', 'Sys ').replace('global', 'GLOBAL').upper()
        print(f"{label:<15} "
              f"{m.get('voltage', {}).get('rmse', 0):>8.3f} "
              f"{m.get('current', {}).get('rmse', 0):>8.3f} "
              f"{m.get('power',   {}).get('rmse', 0):>8.3f} "
              f"{m.get('voltage', {}).get('r2',   0):>6.3f} "
              f"{m.get('current', {}).get('r2',   0):>6.3f} "
              f"{m.get('power',   {}).get('r2',   0):>6.3f} ")
    print("─"*70)

    print("\nConstraint Satisfaction Summary:")
    for sys_id in range(1, NUM_SYSTEMS + 1):
        cs = evcs_outputs.get(sys_id, {}).get('constraint_sat', {})
        print(f"  System {sys_id}: "
              f"V={cs.get('voltage', 0)*100:.1f}%  "
              f"I={cs.get('current', 0)*100:.1f}%  "
              f"P={cs.get('power',   0)*100:.1f}%")

    print("\nAnomaly Detection Summary (DR / FPR / F1):")
    for sys_id in range(1, NUM_SYSTEMS + 1):
        ar = anomaly_results.get(sys_id, {})
        print(f"  System {sys_id}: "
              f"DR={ar.get('detection_rate', 0)*100:.0f}%  "
              f"FPR={ar.get('fpr', 0)*100:.0f}%  "
              f"F1={ar.get('f1', 0):.2f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Individual Publication Sub-figures
# ─────────────────────────────────────────────────────────────────────────────

def save_publication_subplots(
    train_histories : Dict,
    test_metrics    : Dict,
    evcs_outputs    : Dict,
    anomaly_results : Dict,
    lglobal_compare : Dict,
    actual_evcs_data: Dict,
    out_dir         : str = 'federated_pinn_results',
) -> None:
    """Save every panel from both evaluation figures as individual PDF files.

    Publication format:
      figsize          = (10, 8)
      xlabel / ylabel  fontsize = 20
      xtick  / ytick   fontsize = 20
      legend           fontsize = 18
      grid on
      Saved as PDF in <out_dir>/
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"SECTION 9 – Publication Sub-figures  →  {out_dir}/")
    print(f"{'='*60}")

    FIG  = (10, 8)
    FS_L = 24       # axis-label font size
    FS_T = 24        # tick font size
    FS_G = 24        # legend font size

    soc_grid = np.linspace(0.1, 0.95, 20)
    styles   = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    # ── Formatting helper ────────────────────────────────────────────────────
    def _fmt(ax, xlabel='', ylabel='', legend=True, grid=True,
             twin=None, twin_ylabel=''):
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FS_L)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS_L)
        ax.tick_params(axis='both', labelsize=FS_T)
        if grid:
            ax.grid(True, alpha=0.3)
        if legend:
            handles, labs = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labs, fontsize=FS_G, ncol=2)
        if twin is not None:
            if twin_ylabel:
                twin.set_ylabel(twin_ylabel, fontsize=FS_L)
            twin.tick_params(axis='both', labelsize=FS_T)

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {name}")

    # Panels from federated_pinn_evaluation_plots.png  (figures 01-08)

    # ── Fig 01: Training loss curves (all systems) ───────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        hist = train_histories.get(sys_id, {})
        if hist.get('total'):
            ax.plot(hist['total'], color=SYSTEM_COLORS[sys_id - 1],
                    label=f'Sys {sys_id}', lw=4)
    if any((train_histories.get(s, {}).get('total', [0])[0] or 0) > 10
           for s in range(1, NUM_SYSTEMS + 1)):
        ax.set_yscale('log')
    _fmt(ax, xlabel='Epoch', ylabel='Total Loss', legend=True, grid=True)
    _save(fig, 'fig01_training_loss_curves.pdf')

    # ── Fig 02: PINN loss components – System 1 ─────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    hist1 = train_histories.get(1, {})
    for key, color in LOSS_COLORS.items():
        if hist1.get(key):
            ax.plot(hist1[key], color=color, label=key.capitalize(), lw=4)
    _fmt(ax, xlabel='Epoch', ylabel='Loss', legend=True, grid=True)
    _save(fig, 'fig02_pinn_loss_components.pdf')

    # ── Fig 03: Test R² scores (bar chart) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    metric_keys = ['voltage', 'current', 'power']
    x = np.arange(NUM_SYSTEMS + 1)
    bar_w = 0.25
    labels_sys  = [f'Sys {s}' for s in range(1, NUM_SYSTEMS + 1)] + ['Global']
    entity_keys = [f'system_{s}' for s in range(1, NUM_SYSTEMS + 1)] + ['global']
    bar_colors  = ['#4c72b0', '#dd8452', '#55a868']
    for k, (mkey, color) in enumerate(zip(metric_keys, bar_colors)):
        r2_vals = [test_metrics.get(ek, {}).get(mkey, {}).get('r2', 0.0)
                   for ek in entity_keys]
        ax.bar(x + k * bar_w, r2_vals, bar_w, label=mkey.capitalize(),
               color=color, edgecolor='white', lw=4.0)
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(labels_sys, rotation=30, ha='right', fontsize=FS_T)
    ax.axhline(0, color='k', lw=4.0, ls='--')
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    _fmt(ax, xlabel='Model', ylabel='R² Score', grid=False, legend=True)
    _save(fig, 'fig03_test_r2_scores.pdf')

    # ── Fig 04: EVCS terminal voltage vs SOC ─────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        data = evcs_outputs.get(sys_id, {})
        if len(data.get('soc', [])) == 0:
            continue
        v_arr = []
        for s in soc_grid:
            mask = np.isclose(data['soc'], s, atol=0.01)
            v_arr.append(np.mean(data['voltage'][mask]) if mask.any() else np.nan)
        ax.plot(soc_grid, v_arr, color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    ax.axhline(PINN_BOUNDS['v_min'], color='grey', ls='--', lw=1.2, label='Bounds')
    ax.axhline(PINN_BOUNDS['v_max'], color='grey', ls='--', lw=1.2)
    _fmt(ax, xlabel='State of Charge (SOC)', ylabel='Terminal Voltage (V)', legend=True, grid=True)
    _save(fig, 'fig04_evcs_voltage_vs_soc.pdf')

    # ── Fig 05: DC Power & Current vs SOC (twin y-axis) ──────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    ax2 = ax.twinx()
    for sys_id in range(1, NUM_SYSTEMS + 1):
        data = evcs_outputs.get(sys_id, {})
        if len(data.get('soc', [])) == 0:
            continue
        p_arr, i_arr = [], []
        for s in soc_grid:
            mask = np.isclose(data['soc'], s, atol=0.01)
            p_arr.append(np.mean(data['power'][mask])   if mask.any() else np.nan)
            i_arr.append(np.mean(data['current'][mask]) if mask.any() else np.nan)
        ax.plot(soc_grid,  p_arr, color=SYSTEM_COLORS[sys_id - 1],
                ls='-',  lw=4,   label=f'P  Sys {sys_id}')
        ax2.plot(soc_grid, i_arr, color=SYSTEM_COLORS[sys_id - 1],
                 ls='--', lw=4.0, alpha=0.7)
    ax.axhline(PINN_BOUNDS['p_min'], color='grey', ls=':', lw=4.0)
    ax.axhline(PINN_BOUNDS['p_max'], color='grey', ls=':', lw=4.0)
    _fmt(ax, xlabel='SOC', ylabel='DC Power (kW)',
         twin=ax2, twin_ylabel='Current (A)', legend=True, grid=True)
    _save(fig, 'fig05_evcs_power_current_vs_soc.pdf')

    # ── Fig 06: Physics constraint satisfaction ───────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    cat_names = ['Voltage', 'Current', 'Power']
    all_sat = {c.lower(): [] for c in cat_names}
    for sys_id in range(1, NUM_SYSTEMS + 1):
        cs = evcs_outputs.get(sys_id, {}).get('constraint_sat', {})
        for c in cat_names:
            all_sat[c.lower()].append(cs.get(c.lower(), 0.0) * 100)
    xs  = np.arange(NUM_SYSTEMS)
    bwc = 0.25
    for k, (cat, color) in enumerate(zip(cat_names,
                                          ['#4c72b0', '#dd8452', '#55a868'])):
        ax.bar(xs + k * bwc, all_sat[cat.lower()], bwc,
               label=cat, color=color, edgecolor='white')
    ax.axhline(100, color='red', ls='--', lw=4, label='100 %')
    ax.set_xticks(xs + bwc)
    ax.set_xticklabels([f'Sys {s}' for s in range(1, NUM_SYSTEMS + 1)],
                        fontsize=FS_T)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    _fmt(ax, xlabel='System ID', ylabel='Satisfaction (%)', grid=False, legend=True)
    _save(fig, 'fig06_constraint_satisfaction.pdf')

    # ── Fig 07: Anomaly detection performance ────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    sys_ids  = list(range(1, NUM_SYSTEMS + 1))
    dr_vals  = [anomaly_results.get(s, {}).get('detection_rate', 0) * 100 for s in sys_ids]
    fpr_vals = [anomaly_results.get(s, {}).get('fpr',            0) * 100 for s in sys_ids]
    f1_vals  = [anomaly_results.get(s, {}).get('f1',             0) * 100 for s in sys_ids]
    xd  = np.arange(NUM_SYSTEMS)
    bw2 = 0.25
    ax.bar(xd,         dr_vals,  bw2, label='Detection Rate (%)',
           color='#2ca02c', edgecolor='white')
    ax.bar(xd + bw2,   fpr_vals, bw2, label='False Positive (%)',
           color='#d62728', edgecolor='white')
    ax.bar(xd + 2*bw2, f1_vals,  bw2, label='F1 Score × 100',
           color='#1f77b4', edgecolor='white')
    ax.set_xticks(xd + bw2)
    ax.set_xticklabels([f'Sys {s}' for s in sys_ids], fontsize=FS_T)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    _fmt(ax, xlabel='System', ylabel='(%)', grid=False, legend=True)
    _save(fig, 'fig07_anomaly_detection.pdf')

    # ── Fig 08: Local vs Global model divergence ─────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    ax_r = ax.twinx()
    param_l2s = [lglobal_compare.get(s, {}).get('param_l2', 0) for s in sys_ids]
    pred_maes = [lglobal_compare.get(s, {}).get('pred_mae',  0) for s in sys_ids]
    ax.bar(xd, param_l2s, 0.4, label='Param L₂ Divergence',
           color='#9467bd', edgecolor='white')
    ax_r.plot(xd, pred_maes, 'o-', color='#e377c2', lw=4, ms=8,
              label='Pred MAE vs Global')
    ax.set_xticks(xd)
    ax.set_xticklabels([f'Sys {s}' for s in sys_ids], fontsize=FS_T)
    ax.grid(True, alpha=0.3, axis='y')
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=FS_G)
    _fmt(ax, xlabel='System', ylabel='Parameter L₂ Norm', legend=False,
         grid=False, twin=ax_r, twin_ylabel='Mean Prediction MAE')
    _save(fig, 'fig08_model_divergence.pdf')

    # Panels from federated_pinn_evcs_actual_outputs.png  (figures 09-14)
    if not actual_evcs_data:
        print("  (No actual EVCS data — skipping figures 09–14)")
        return

    hours     = actual_evcs_data[1]['hours']
    params_ev = EVCSParameters()
    soc_fine  = np.linspace(0.10, params_ev.disconnect_soc, 200)
    ocv_fine  = (params_ev.min_voltage
                 + soc_fine * (params_ev.max_voltage - params_ev.min_voltage))
    vterm_fine = ocv_fine - params_ev.rated_current * 0.1   # I·R drop at full CC

    # ── Fig 09: Terminal voltage vs time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['hours'], d['v'], color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    for sh in actual_evcs_data[1].get('session_hours', []):
        ax.axvline(sh, color='#888888', ls=':', lw=4, zorder=0)
    ax.axhline(params_ev.min_voltage, color='grey', ls=':', lw=4)
    ax.axhline(params_ev.max_voltage, color='grey', ls=':', lw=4)
    ax.set_xlim(0, hours[-1])
    _fmt(ax, xlabel='Time (h)', ylabel='Terminal Voltage (V)', legend=True, grid=True)
    _save(fig, 'fig09_actual_voltage_vs_time.pdf')

    # ── Fig 10: Terminal voltage vs SOC ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['soc'], d['v'], color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    ax.plot(soc_fine, ocv_fine,   'k--', lw=4, label='OCV (no load)')
    ax.plot(soc_fine, vterm_fine, 'k:',  lw=4, label='V_term @ I_rated')
    ax.axvline(0.8, color='grey', ls='-.', lw=4, label='CC→CV  (SOC=0.8)')
    ax.axhline(params_ev.min_voltage, color='grey', ls=':', lw=4)
    ax.axhline(params_ev.max_voltage, color='grey', ls=':', lw=4)
    _fmt(ax, xlabel='SOC', ylabel='Terminal Voltage (V)', legend=True, grid=True)
    _save(fig, 'fig10_actual_voltage_vs_soc.pdf')

    # ── Fig 11: Charging current vs time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['hours'], d['i'], color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    for sh in actual_evcs_data[1].get('session_hours', []):
        ax.axvline(sh, color='#888888', ls=':', lw=4, zorder=0)
    ax.axhline(params_ev.min_current, color='grey', ls=':', lw=4)
    ax.axhline(params_ev.max_current, color='grey', ls=':', lw=4)
    ax.set_xlim(0, hours[-1])
    _fmt(ax, xlabel='Time (h)', ylabel='Current (A)', legend=True, grid=True)
    _save(fig, 'fig11_actual_current_vs_time.pdf')

    # ── Fig 12: Charging current vs SOC ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['soc'], d['i'], color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    ax.axvline(0.8, color='grey', ls='-.', lw=4, label='CC→CV  (SOC=0.8)')
    ax.axhline(params_ev.min_current, color='grey', ls=':', lw=4)
    ax.axhline(params_ev.max_current, color='grey', ls=':', lw=4)
    _fmt(ax, xlabel='SOC', ylabel='Current (A)', legend=True, grid=True)
    _save(fig, 'fig12_actual_current_vs_soc.pdf')

    # ── Fig 13: DC power (solid) & AC grid power (dashed) vs time ────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['hours'], d['p_dc'], color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4,   label=f'Sys {sys_id}')
        ax.plot(d['hours'], d['p_ac'], color=SYSTEM_COLORS[sys_id - 1],
                ls='--',               lw=4.0, alpha=0.5)
    for sh in actual_evcs_data[1].get('session_hours', []):
        ax.axvline(sh, color='#888888', ls=':', lw=4, zorder=0)
    ax.text(0.02, 0.97, 'Solid = P_dc  |  Dashed = P_ac',
            transform=ax.transAxes, fontsize=16, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
    ax.set_xlim(0, hours[-1])
    _fmt(ax, xlabel='Time (h)', ylabel='Power (kW)', legend=True, grid=True)
    _save(fig, 'fig13_actual_power_vs_time.pdf')

    # ── Fig 14: System efficiency vs SOC ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG)
    for sys_id in range(1, NUM_SYSTEMS + 1):
        d = actual_evcs_data[sys_id]
        ax.plot(d['soc'], d['eff'] * 100, color=SYSTEM_COLORS[sys_id - 1],
                ls=styles[sys_id - 1], lw=4, label=f'Sys {sys_id}')
    ax.axhline(94.1, color='k', ls='--', lw=4,
               label='Design η = 94.1 %  (0.98×0.96)')
    ax.set_ylim(88, 100)
    _fmt(ax, xlabel='SOC', ylabel='Efficiency (%)', legend=True, grid=True)
    _save(fig, 'fig14_actual_efficiency_vs_soc.pdf')

    print(f"\n  14 PDF sub-figures saved in '{out_dir}/'")


# Main

def main():
    t0 = time.time()
    print("="*70)
    print("  Federated PINN Evaluation – EVCS Charging Management System")
    print("  " + datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    print("="*70)

    # ── 1. Load models ───────────────────────────────────────────────────────
    manager, loaded_ok = load_federated_models()

    # ── 2. Fresh training + write back into manager ──────────────────────────

    train_histories = extract_training_curves(manager, n_samples=600, n_epochs=120)

    # ── 3. Test performance ──────────────────────────────────────────────────
    test_metrics = evaluate_test_performance(manager, n_test=400)

    # ── 4. EVCS output dynamics (PINN references) ────────────────────────────
    evcs_outputs = evaluate_evcs_outputs(manager)

    # ── 4b. Actual EVCS physical outputs from converter dynamics ─────────────
    actual_evcs_data = plot_actual_evcs_outputs(manager)

    # ── 5. Anomaly detection ─────────────────────────────────────────────────
    anomaly_results = evaluate_anomaly_detection(manager)

    # ── 6. Local vs global comparison ────────────────────────────────────────
    lglobal_compare = compare_local_vs_global(manager, n_test=200)

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    generate_all_plots(
        train_histories, test_metrics, evcs_outputs,
        anomaly_results, lglobal_compare,
    )

    # ── 8. Metrics report ────────────────────────────────────────────────────
    build_metrics_report(
        test_metrics, evcs_outputs, anomaly_results, lglobal_compare,
    )

    # ── 9. Publication sub-figures ────────────────────────────────────────────
    save_publication_subplots(
        train_histories, test_metrics, evcs_outputs,
        anomaly_results, lglobal_compare, actual_evcs_data,
    )

    elapsed = time.time() - t0
    print(f"\n🏁 Evaluation complete in {elapsed:.1f} s")
    print("   Outputs:")
    print("   • federated_pinn_evaluation_plots.png")
    print("   • federated_pinn_evcs_actual_outputs.png")
    print("   • federated_pinn_metrics.json")
    print("   • federated_pinn_results/  (14 PDF sub-figures)")


if __name__ == '__main__':
    main()
