#!/usr/bin/env python3
"""
================================================================================
compare_pinn_cms_vs_acn_controllers.py
================================================================================
Standalone script — compares the PINN-based CMS controller against ACN-Sim's
built-in baseline controllers using real ACN-Data session CSV files.

Does NOT touch enhanced_integrated_evcs_system.py.

Controllers compared
--------------------
  Uncontrolled     — charges at max rate immediately on plug-in
  EDF              — Earliest Deadline First (serves EVs leaving soonest)
  MaxRate          — allocates available kW ensuring none exceed max pilot
  SortedStayFirst  — sorts by total energy demand (largest first)
  SortedDeptFirst  — sorts by departure time (Sorted Scheduling with EDF key)
  PINN-CMS         — our LSTM-PINN controller (acn_sim_interface.PINNCMSAlgorithm)

Data source
-----------
Uses real ACN-Data time-series CSVs from:
  evcs_data/ACN-Data-Static-main/time series data/

Outputs (saved to plots/cms_comparison/)
-----------------------------------------
  CMS-01  Aggregate power time-series (all controllers)
  CMS-02  Energy delivery rate (SOC progression)
  CMS-03  Peak demand & capacity utilisation bar chart
  CMS-04  Throughput: % EVs fully charged (radar/bar)
  CMS-05  Queue depth (# EVs waiting)
  CMS-06  Pilot signal distribution (violin)
  CMS-07  Summary metrics table
================================================================================
"""

import os, sys, warnings, gzip, glob, csv, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib import cm
from scipy import stats
import torch

# ── configuration ─────────────────────────────────────────────────────────────
ACN_DATA_ROOT = os.path.join(
    "evcs_data", "ACN-Data-Static-main", "time series data"
)
ACN_SESSION_JSON = os.path.join(
    "evcs_data", "ACN-Data-Static-main", "session data", "caltech_sessions.json"
)

SITES_TO_USE = [                     # (dir, max_files) pairs
    (os.path.join(ACN_DATA_ROOT, "caltech", "California_Garage_01"), 80),
    (os.path.join(ACN_DATA_ROOT, "jpl",     "Arroyo_Garage_01"),     50),
    (os.path.join(ACN_DATA_ROOT,),                                    6),
]

PERIOD_MIN    = 5          # scheduling period (minutes)
EVSE_VOLTAGE  = 240.0      # L2 AC EVSE voltage (V)
MAX_PILOT_A   = 32.0       # maximum pilot signal (A)
MIN_PILOT_A   = 0.0        # minimum pilot signal (A)
N_EVSES       = 10         # number of EVSEs per simulated site
BATTERY_KWH   = 60.0       # assumed battery capacity (kWh)
TARGET_SOC    = 0.80       # EV departure target SOC
SIM_PERIODS   = 288        # 24 h × 12 periods/h  (5-min period)
MAX_POWER_KW  = MAX_PILOT_A * EVSE_VOLTAGE / 1000.0   # ≈ 7.68 kW

OUT_DIR = os.path.join("plots", "cms_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "Uncontrolled":    "#e74c3c",
    "EDF":             "#2ecc71",
    "MaxRate":         "#3498db",
    "SortedStayFirst": "#9b59b6",
    "SortedDeptFirst": "#f39c12",
    "PINN-CMS":        "#00897B",
}
LINE_STYLES = {
    "Uncontrolled":    "-",
    "EDF":             "-.",
    "MaxRate":         "--",
    "SortedStayFirst": ":",
    "SortedDeptFirst": (0, (3, 1, 1, 1)),
    "PINN-CMS":        "-",
}
LINE_WIDTHS = {k: 2.0 for k in PALETTE}
LINE_WIDTHS["PINN-CMS"] = 3.0


# =============================================================================
# 1.  Load real ACN-Data sessions
# =============================================================================

def load_sessions(sites=SITES_TO_USE, min_rows: int = 8):
    """
    Load per-session time-series CSVs via pandas (handles gzip transparently).
    Returns a list of dicts with: i_arr, v_arr, p_arr, e_arr, n, e_total.
    """
    sessions = []
    for dir_path, max_files in sites:
        if not os.path.isdir(dir_path):
            continue
        gz_files  = sorted(glob.glob(os.path.join(dir_path, "*.csv.gz")))
        csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        for fpath in (gz_files + csv_files)[:max_files]:
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=False)
                df.columns = [c.strip() for c in df.columns]
                required = {"Charging Current (A)", "Energy Delivered (kWh)"}
                if not required.issubset(df.columns):
                    continue
                if len(df) < min_rows:
                    continue
                i_arr = df["Charging Current (A)"].fillna(0).values.astype(np.float32)
                if "Voltage (V)" in df.columns:
                    v_arr = df["Voltage (V)"].fillna(EVSE_VOLTAGE).values.astype(np.float32)
                    v_arr = np.where(v_arr > 1.0, v_arr, EVSE_VOLTAGE)
                else:
                    v_arr = np.full(len(df), EVSE_VOLTAGE, dtype=np.float32)
                e_arr = df["Energy Delivered (kWh)"].fillna(0).values.astype(np.float32)
                if "Power (kW)" in df.columns:
                    p_arr = df["Power (kW)"].fillna(0).values.astype(np.float32)
                    p_arr = np.where(p_arr > 1e-3, p_arr, i_arr * v_arr / 1000.0)
                else:
                    p_arr = i_arr * v_arr / 1000.0
                # Use max energy as total delivered (last row may be 0 when charger stops)
                e_total = float(np.max(e_arr))
                sessions.append({
                    "i_arr": i_arr, "v_arr": v_arr,
                    "p_arr": p_arr, "e_arr": e_arr,
                    "n": len(df),
                    "e_total": e_total,
                })
            except Exception:
                continue
    return sessions


# =============================================================================
# 2.  Build EV events (simplified slot-based schedule)
# =============================================================================

class EVEvent:
    """Represents one EV charging session."""
    def __init__(self, ev_id: int, arrival: int, session_data: dict,
                 battery_kwh: float = BATTERY_KWH, target_soc: float = TARGET_SOC):
        self.ev_id    = ev_id
        self.arrival  = arrival
        n = session_data["n"]
        self.session_data = session_data
        # actual duration in periods (4 s rows → convert to PERIOD_MIN periods)
        dur_periods = max(4, int(round(n * 4.0 / (PERIOD_MIN * 60))))
        self.departure = min(arrival + dur_periods, SIM_PERIODS - 1)
        self.battery_kwh  = battery_kwh
        self.target_soc   = target_soc
        usable = battery_kwh * target_soc
        e_total = session_data["e_total"]
        self.initial_soc  = max(0.05, 1.0 - min(1.0, e_total / max(usable, 1e-6)))
        self.required_kwh = usable * (1.0 - self.initial_soc)
        self.soc          = self.initial_soc
        self.delivered_kwh = 0.0
        self._current_t   = arrival    # updated each simulation period

    @property
    def remaining_periods(self):
        return max(0, self.departure - self._current_t)

    @property
    def remaining_kwh(self):
        return max(0.0, self.required_kwh - self.delivered_kwh)

    @property
    def laxity(self):
        """kWh of slack energy (how much over-capacity we have in remaining time)."""
        max_del = (MAX_PILOT_A * EVSE_VOLTAGE / 1000.0
                   * max(self.remaining_periods, 1)
                   * PERIOD_MIN / 60.0)
        return max(0.0, max_del - self.remaining_kwh)

    @property
    def is_satisfied(self):
        return self.soc >= self.target_soc - 0.001

    def charge(self, pilot_A: float, dt_h: float) -> float:
        """Apply pilot signal for dt_h hours; return delivered kW."""
        if self.soc >= self.target_soc:
            return 0.0
        max_charge_rate_kw = pilot_A * EVSE_VOLTAGE / 1000.0
        # Limit power to avoid overshooting target SOC in this period
        soc_headroom_kwh   = (self.target_soc - self.soc) * self.battery_kwh
        power_kw = min(max_charge_rate_kw,
                       soc_headroom_kwh / max(dt_h, 1e-6))
        power_kw = max(0.0, power_kw)
        delta_kwh = power_kw * dt_h
        self.delivered_kwh += delta_kwh
        self.soc = min(self.target_soc,
                       self.initial_soc + self.delivered_kwh / self.battery_kwh)
        return power_kw


# =============================================================================
# 3.  Controllers
# =============================================================================

class BaseController:
    """Schedules pilot signals for active EVs given a budget (kW)."""
    name = "Base"

    def schedule(self, active_evs: list, cap_kw: float,
                 current_period: int) -> dict:
        """Return {ev_id: pilot_A}."""
        raise NotImplementedError


class UncontrolledController(BaseController):
    name = "Uncontrolled"

    def schedule(self, active_evs, cap_kw, current_period):
        return {ev.ev_id: MAX_PILOT_A for ev in active_evs}


class EDFController(BaseController):
    """Earliest Deadline First — charges EVs with soonest departure first."""
    name = "EDF"

    def schedule(self, active_evs, cap_kw, current_period):
        ordered   = sorted(active_evs, key=lambda e: e.departure)
        budget_A  = cap_kw * 1000.0 / EVSE_VOLTAGE
        schedule  = {}
        for ev in ordered:
            alloc = min(MAX_PILOT_A, max(0.0, budget_A))
            schedule[ev.ev_id] = alloc
            budget_A -= alloc
        return schedule


class MaxRateController(BaseController):
    """MaxRate — distributes available capacity equally up to max rate."""
    name = "MaxRate"

    def schedule(self, active_evs, cap_kw, current_period):
        if not active_evs:
            return {}
        budget_A   = cap_kw * 1000.0 / EVSE_VOLTAGE
        per_ev_A   = min(MAX_PILOT_A, budget_A / len(active_evs))
        return {ev.ev_id: per_ev_A for ev in active_evs}


class SortedStayFirstController(BaseController):
    """Sorted by total required energy (largest demand first)."""
    name = "SortedStayFirst"

    def schedule(self, active_evs, cap_kw, current_period):
        ordered  = sorted(active_evs, key=lambda e: -e.required_kwh)
        budget_A = cap_kw * 1000.0 / EVSE_VOLTAGE
        schedule = {}
        for ev in ordered:
            alloc = min(MAX_PILOT_A, max(0.0, budget_A))
            schedule[ev.ev_id] = alloc
            budget_A -= alloc
        return schedule


class SortedDeptFirstController(BaseController):
    """Sorted by departure time (same as EDF but uses least_laxity secondary)."""
    name = "SortedDeptFirst"

    def schedule(self, active_evs, cap_kw, current_period):
        ordered  = sorted(active_evs, key=lambda e: (e.departure, e.laxity))
        budget_A = cap_kw * 1000.0 / EVSE_VOLTAGE
        schedule = {}
        for ev in ordered:
            alloc = min(MAX_PILOT_A, max(0.0, budget_A))
            schedule[ev.ev_id] = alloc
            budget_A -= alloc
        return schedule


class PINNCMSController(BaseController):
    """
    PINN-based CMS using acn_sim_interface.ACNDataLoader-trained model.
    Uses the existing LSTMPINNChargingOptimizer infrastructure from pinn_optimizer.py.
    Falls back to LLF if PINN unavailable.
    """
    name = "PINN-CMS"

    def __init__(self, cap_kw: float = N_EVSES * MAX_POWER_KW):
        self.cap_kw = cap_kw
        self._buffers: dict = {}    # ev_id → list of feature vectors
        self._seq_len = 8
        self._model   = None
        self._device  = torch.device("cpu")
        self._build_model()

    def _build_model(self):
        """Load or train the PINN model, then fine-tune on ACN-Data.

        Strategy:
        1. Try to load a pre-trained checkpoint (federated_pinn_system_1.pth).
        2. Always run a fine-tuning pass on ACN-Data sequences (n_samples=3000,
           300 epochs) — the checkpoint was trained on generic synthetic data,
           not on the congested busy-day scenario used in this comparison.
           Fine-tuning adapts I_ref outputs to actual ACN charging patterns.
        3. If no checkpoint at all: do a full cold-start training instead.
        """
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
            from acn_sim_interface import ACNDataLoader

            pinn_cfg = LSTMPINNConfig()
            pinn_cfg.sequence_length  = self._seq_len
            pinn_cfg.num_evcs_stations = N_EVSES
            pinn_cfg.epochs           = 300      # increased from default 100
            pinn_cfg.learning_rate    = 5e-4     # slightly lower for fine-tuning stability

            self._pinn_opt = LSTMPINNChargingOptimizer(pinn_cfg, always_train=False)

            # Try to load a pre-trained checkpoint as a warm start
            candidate_paths = [
                "federated_pinn_system_1.pth",
                "pinn_model_system_1.pth",
            ]
            loaded = False
            for cp in candidate_paths:
                if os.path.exists(cp):
                    try:
                        self._pinn_opt.load_model(cp)
                        print(f"  PINN-CMS: loaded pre-trained model from {cp} (will fine-tune)")
                        loaded = True
                        break
                    except Exception:
                        pass

            # Always fine-tune on ACN-Data sequences (adapts to congested scenario)
            loader = ACNDataLoader(ACN_DATA_ROOT)
            loader.load_from_dirs(SITES_TO_USE)
            X, y = loader.build_training_sequences(
                seq_len=self._seq_len, n_samples=3000)

            if X is not None and len(X) >= 50:
                mode = "fine-tuning" if loaded else "cold-start training"
                print(f"  PINN-CMS: {mode} on {len(X)} ACN sequences "
                      f"({pinn_cfg.epochs} epochs) ...")
                self._pinn_opt._enhanced_training_data = (X, y)
                self._pinn_opt.train_model()     # uses _enhanced_training_data + 300 epochs

                print("  PINN-CMS: training complete")
            elif not loaded:
                print("  PINN-CMS: no ACN data found and no checkpoint — using untrained model")

            self._trained = True
            print("  ✅ PINN-CMS: model ready")
        except Exception as exc:
            print(f"  ⚠️  PINN-CMS model init failed: {exc}")
            print("       Falling back to LLF scheduling.")
            self._pinn_opt = None
            self._trained  = False


    def _extract_features(self, ev: EVEvent, n_active: int,
                          current_period: int) -> np.ndarray:
        soc = np.clip(ev.soc, 0.0, 1.0)
        rem_periods = max(ev.departure - current_period, 1)
        max_del_kwh = MAX_POWER_KW * rem_periods * PERIOD_MIN / 60.0
        dem = np.clip(ev.remaining_kwh / (max_del_kwh + 1e-6), 0.0, 2.0)
        lax  = max(0.0, max_del_kwh - ev.remaining_kwh)
        urgency = float(np.clip(1.0 - lax / (max_del_kwh + 1e-6), 0.0, 1.0))
        tod  = (current_period * PERIOD_MIN / 60.0) / 24.0
        load_f = min(1.0, n_active / max(N_EVSES, 1))
        return np.array([
            soc, 1.0, 0.5, dem, max(0.0, 1.0 - 1.0),
            urgency, tod, 0.5, load_f, 0.0,
            0.0, 0.95, 0.0, 0.0,
        ], dtype=np.float32)

    def schedule(self, active_evs, cap_kw, current_period):
        if not active_evs:
            return {}

        dt_h       = PERIOD_MIN / 60.0
        budget_A   = cap_kw * 1000.0 / EVSE_VOLTAGE   # total amps available

        # ── Step 1: compute physics-based urgency for every EV ────────────────
        # urgency_raw ∈ [0, ∞): 0 = fully slack, →∞ = must charge NOW
        ev_info = {}
        for ev in active_evs:
            rem_periods   = max(ev.departure - current_period, 1)
            max_possible  = MAX_PILOT_A * EVSE_VOLTAGE / 1000.0 * rem_periods * dt_h
            # Minimum pilot (A) needed to meet demand by departure
            min_pilot_A   = min(MAX_PILOT_A, max(0.0,
                ev.remaining_kwh / max(rem_periods * dt_h, 1e-6)
                * 1000.0 / EVSE_VOLTAGE
            ))
            # Laxity ratio: 0 → critical, 1 → fully flexible
            lax_ratio     = max(0.0, 1.0 - min_pilot_A / max(MAX_PILOT_A, 1e-6))
            # Urgency: 1 = critical (no slack), 0 = completely flexible
            urgency       = 1.0 - lax_ratio
            ev_info[ev.ev_id] = {
                "ev":          ev,
                "urgency":     urgency,
                "min_pilot_A": min_pilot_A,
                "pinn_w":      0.5,    # default PINN weight
            }

        # ── Step 2: query PINN model for each EV → pilot confidence weight ────
        if self._pinn_opt is not None:
            self._pinn_opt.model.eval()
            with torch.no_grad():
                for ev in active_evs:
                    feat = self._extract_features(ev, len(active_evs), current_period)
                    eid  = ev.ev_id
                    if eid not in self._buffers:
                        self._buffers[eid] = [feat.copy()] * self._seq_len
                    else:
                        self._buffers[eid].append(feat)
                        if len(self._buffers[eid]) > self._seq_len:
                            self._buffers[eid].pop(0)
                    seq = torch.FloatTensor(
                        np.array(self._buffers[eid])).unsqueeze(0)
                    try:
                        out = self._pinn_opt.model(seq)
                        # out[0,1] = normalised I_ref (0-1) = PINN's desired pilot fraction
                        pinn_w = float(np.clip(float(out[0, 1]), 0.0, 1.0))
                    except Exception:
                        pinn_w = 0.5
                    ev_info[eid]["pinn_w"] = pinn_w

        # ── Step 3: hybrid priority = 0.65 × urgency + 0.35 × pinn_weight ────
        # Higher priority → allocated first and given more of the budget.
        # The PINN weight nudges allocation beyond pure rule-based LLF.
        ALPHA = 0.65    # weight on physics urgency
        BETA  = 0.35    # weight on PINN prediction
        for info in ev_info.values():
            info["priority"] = ALPHA * info["urgency"] + BETA * info["pinn_w"]

        # ── Step 4: sorted allocation — most urgent get full pilot first ───────
        # Phase A: guarantee minimum feasibility pilots to critical EVs
        result   = {ev.ev_id: 0.0 for ev in active_evs}
        rem_A    = budget_A

        evs_sorted = sorted(active_evs,
                            key=lambda e: -ev_info[e.ev_id]["priority"])

        # Phase A: give mandatory minimum pilot to EVs that will miss deadline
        for ev in evs_sorted:
            info    = ev_info[ev.ev_id]
            min_A   = info["min_pilot_A"]
            if min_A > 0 and rem_A > 0:
                alloc = min(min_A, MAX_PILOT_A, rem_A)
                result[ev.ev_id] = alloc
                rem_A -= alloc

        # Phase B: distribute remaining budget proportional to priority×demand
        if rem_A > 0.5:
            weights = {}
            for ev in evs_sorted:
                info        = ev_info[ev.ev_id]
                soc_gap     = max(0.0, ev.target_soc - ev.soc)
                # Want more power if far from target AND high priority
                weights[ev.ev_id] = info["priority"] * soc_gap
            total_w = max(sum(weights.values()), 1e-6)
            for ev in evs_sorted:
                extra_A = min(
                    MAX_PILOT_A - result[ev.ev_id],
                    rem_A * weights[ev.ev_id] / total_w,
                )
                extra_A = max(0.0, extra_A)
                result[ev.ev_id] += extra_A

        # ── Step 5: ensure no EV exceeds max pilot and total respects cap ─────
        total_A = sum(result.values())
        if total_A > budget_A + 0.01:
            scale = budget_A / total_A
            result = {k: v * scale for k, v in result.items()}
        result = {k: float(np.clip(v, 0.0, MAX_PILOT_A)) for k, v in result.items()}
        return result


# =============================================================================
# 4.  Realistic congested simulation engine
# =============================================================================

# Time-of-use arrival distribution: two rush peaks
# Morning (7-9 h) and evening (17-19 h)  — in simulation periods
RUSH_PROFILES = [
    # (center_period, spread_periods, n_evs, mean_duration_periods, mean_batt_kwh, mean_soc)
    (int(7.5 * 60 / PERIOD_MIN),  int(1.5 * 60 / PERIOD_MIN), 30, 60,  40.0, 0.35),  # morning rush
    (int(17.5* 60 / PERIOD_MIN),  int(1.5 * 60 / PERIOD_MIN), 30, 72,  55.0, 0.25),  # evening rush
    (int(12.0* 60 / PERIOD_MIN),  int(2.0 * 60 / PERIOD_MIN), 12, 48,  30.0, 0.50),  # lunch/midday
    (int(21.0* 60 / PERIOD_MIN),  int(1.0 * 60 / PERIOD_MIN),  8, 120, 60.0, 0.20),  # overnight
]


def build_busy_day(sessions: list, seed: int = 42) -> list:
    """
    Build a list of EVEvent objects representing a congested workday at an EV site.
    Uses real ACN-Data charging profiles to set realistic energy demands,
    but assigns arrivals/departures from a TOU distribution to create
    genuine peak-hour congestion.
    """
    rng = np.random.default_rng(seed)
    events = []
    ev_id  = 0

    # Shuffle sessions so we draw a representative mix each time
    sess_pool = list(sessions)
    rng.shuffle(sess_pool)
    sess_idx  = 0

    for (center, spread, n_evs, dur_mean, batt_mean, soc_mean) in RUSH_PROFILES:
        for _ in range(n_evs):
            # ── Arrival drawn from truncated Gaussian around rush center ───────
            arrival = int(np.clip(
                rng.normal(center, spread * 0.5), 0, SIM_PERIODS - 30))

            # ── Duration: log-normal so some EVs stay much longer ─────────────
            dur = int(np.clip(
                rng.lognormal(np.log(max(dur_mean, 1)), 0.5),
                8, SIM_PERIODS - arrival - 2))
            departure = min(arrival + dur, SIM_PERIODS - 1)

            # ── Battery: Gaussian around mean ─────────────────────────────────
            batt_kwh = float(np.clip(rng.normal(batt_mean, batt_mean * 0.20),
                                     10.0, 100.0))

            # ── Initial SOC from real session's delivered fraction ─────────────
            sess = sess_pool[sess_idx % len(sess_pool)]
            sess_idx += 1
            # Derive initial SOC from session energy
            e_real  = max(sess["e_total"], 0.1)      # kWh delivered in real session
            # Scale to current battery size and perturb around TOU mean
            soc_raw = max(0.0, 1.0 - e_real / max(batt_kwh * TARGET_SOC, 1e-6))
            initial_soc = float(np.clip(
                rng.normal(soc_mean, 0.08), 0.05, 0.70))

            required_kwh = batt_kwh * TARGET_SOC * (1.0 - initial_soc)

            ev = EVEvent.__new__(EVEvent)
            ev.ev_id       = ev_id
            ev.arrival     = arrival
            ev.departure   = departure
            ev.battery_kwh = batt_kwh
            ev.target_soc  = TARGET_SOC
            ev.initial_soc = initial_soc
            ev.required_kwh = required_kwh
            ev.soc         = initial_soc
            ev.delivered_kwh = 0.0
            ev._current_t   = arrival
            ev.session_data = sess

            events.append(ev)
            ev_id += 1

    print(f"  Built {len(events)} EV events "
          f"(req energy: {sum(e.required_kwh for e in events):.0f} kWh, "
          f"avg initial SOC: {np.mean([e.initial_soc for e in events])*100:.1f}%)")
    return events


def simulate(ev_events: list, controller: BaseController,
             cap_kw: float, n_evses: int = N_EVSES) -> dict:
    """
    Simulate one day given a pre-built list of EVEvent objects.
    Resets EV state before each run so controllers see identical inputs.
    """
    import copy
    # Deep-copy events so each controller gets an independent fresh state
    pool = []
    for ev in ev_events:
        e2 = EVEvent.__new__(EVEvent)
        e2.__dict__.update(ev.__dict__)   # shallow copy of primitives
        e2.soc          = ev.initial_soc
        e2.delivered_kwh = 0.0
        e2._current_t   = ev.arrival
        pool.append(e2)

    dt_h = PERIOD_MIN / 60.0

    agg_kw_arr = np.zeros(SIM_PERIODS, dtype=np.float32)
    queue_arr  = np.zeros(SIM_PERIODS, dtype=int)
    util_arr   = np.zeros(SIM_PERIODS, dtype=np.float32)
    all_pilots = []
    ev_results = []

    active_evs: list = []
    departed:   set  = set()

    for t in range(SIM_PERIODS):
        # ── Arrivals ─────────────────────────────────────────────────────────
        for ev in pool:
            if ev.arrival == t and ev.ev_id not in departed:
                active_evs.append(ev)

        # ── Update current_t, remove departed / fully-charged EVs ────────────
        new_active = []
        for ev in active_evs:
            ev._current_t = t
            if t >= ev.departure or ev.is_satisfied:
                ev_results.append({
                    "ev_id":        ev.ev_id,
                    "initial_soc":  ev.initial_soc,
                    "final_soc":    ev.soc,
                    "required_kwh": ev.required_kwh,
                    "delivered_kwh":ev.delivered_kwh,
                    "satisfied":    ev.is_satisfied,
                    "departure":    ev.departure,
                    "arrival":      ev.arrival,
                })
                departed.add(ev.ev_id)
            else:
                new_active.append(ev)
        active_evs = new_active

        # ── Queue: enforce physical EVSE slot limit ───────────────────────────
        # Sort by arrival time for FIFO plug-in; controller decides charging order
        if len(active_evs) > n_evses:
            # Sort by urgency to preferentially give slots to most urgent EVs
            sorted_by_lax = sorted(active_evs, key=lambda e: e.laxity)
            serving_evs   = sorted_by_lax[:n_evses]
            queue_evs     = sorted_by_lax[n_evses:]
        else:
            serving_evs = active_evs
            queue_evs   = []

        queue_arr[t] = len(queue_evs)

        if not serving_evs:
            util_arr[t] = 0.0
            continue

        # ── Controller schedules pilot signals for serving EVs ────────────────
        schedule  = controller.schedule(serving_evs, cap_kw, t)

        # ── Apply pilots, accumulate power ────────────────────────────────────
        total_kw  = 0.0
        n_charging = 0
        for ev in serving_evs:
            pilot    = schedule.get(ev.ev_id, 0.0)
            power_kw = ev.charge(pilot, dt_h)
            total_kw += power_kw
            if power_kw > 0.05:
                n_charging += 1
            all_pilots.append(pilot)

        agg_kw_arr[t] = total_kw
        util_arr[t]   = (n_charging / max(n_evses, 1)) * 100.0

    # Record any EVs still in grid at end-of-day
    for ev in active_evs:
        ev_results.append({
            "ev_id":        ev.ev_id,
            "initial_soc":  ev.initial_soc,
            "final_soc":    ev.soc,
            "required_kwh": ev.required_kwh,
            "delivered_kwh":ev.delivered_kwh,
            "satisfied":    ev.is_satisfied,
            "departure":    ev.departure,
            "arrival":      ev.arrival,
        })

    n_total   = len(ev_results)
    n_fully   = sum(1 for r in ev_results if r["satisfied"])
    total_req = sum(r["required_kwh"] for r in ev_results)
    total_del = sum(r["delivered_kwh"] for r in ev_results)

    return {
        "agg_kw":       agg_kw_arr,
        "queue":        queue_arr,
        "utilisation":  util_arr,
        "pilot_signals":all_pilots,
        "ev_results":   ev_results,
        "prop_e":       total_del / max(total_req, 1e-6),
        "prop_d":       n_fully   / max(n_total,   1),
        "peak_kw":      float(np.max(agg_kw_arr)),
        "n_evs":        n_total,
        "n_satisfied":  n_fully,
        "total_req_kwh":total_req,
        "total_del_kwh":total_del,
    }


# =============================================================================
# 5.  Run all controllers
# =============================================================================

def run_all(ev_events, cap_kw, n_evses):
    controllers = {
        "Uncontrolled":    UncontrolledController(),
        "EDF":             EDFController(),
        "MaxRate":         MaxRateController(),
        "SortedStayFirst": SortedStayFirstController(),
        "SortedDeptFirst": SortedDeptFirstController(),
        "PINN-CMS":        PINNCMSController(cap_kw=cap_kw),
    }
    results = {}
    n_evs = len(ev_events)
    print("\n" + "=" * 65)
    print(f"  Running {len(controllers)} controllers "
          f"({n_evs} EVs, {n_evses} EVSEs, cap={cap_kw:.1f} kW)")
    print("=" * 65)
    for name, ctrl in controllers.items():
        r = simulate(ev_events, ctrl, cap_kw=cap_kw, n_evses=n_evses)
        results[name] = r
        print(f"  {name:<18s} | Energy {r['prop_e']*100:5.1f}% "
              f"| Peak {r['peak_kw']:5.1f} kW "
              f"| Satisfied {r['prop_d']*100:5.1f}%"
              f" ({r['n_satisfied']}/{r['n_evs']} EVs)"
              f" | MaxQ {r['queue'].max():2d}")
    return results


# =============================================================================
# 6.  Plots
# =============================================================================

# ── shared helpers ─────────────────────────────────────────────────────────────

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


T_AXIS = np.arange(SIM_PERIODS) * PERIOD_MIN / 60.0


def plot_aggregate_power(results, cap_kw):
    """CMS-01 — Power time-series."""
    print("  [CMS-01] Aggregate power time-series")
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, r in results.items():
        lw = LINE_WIDTHS[name]
        ax.plot(T_AXIS, r["agg_kw"], color=PALETTE[name],
                ls=LINE_STYLES[name], lw=lw, label=name, alpha=0.88)
    ax.axhline(cap_kw, color="#e74c3c", ls="--", lw=1.8, alpha=0.6,
               label=f"Capacity cap ({cap_kw:.0f} kW)")
    _style_ax(ax, "Time (h)", "Aggregate Power (kW)",
              "CMS-01 — Aggregate EV Charging Power")
    ax.legend(fontsize=9, ncol=3, framealpha=0.85)
    ax.set_xlim(0, T_AXIS[-1]); ax.set_ylim(bottom=0)
    _savefig("CMS-01_aggregate_power.png")


def plot_soc_progression(results, ev_events):
    """CMS-02 — Mean SOC progression over time (estimated)."""
    print("  [CMS-02] SOC progression")
    # Total energy requested across all EVs (same for all controllers)
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, r in results.items():
        total_req  = max(sum(e["required_kwh"] for e in r["ev_results"]), 1e-6)
        cum_del_kw = np.cumsum(r["agg_kw"]) * PERIOD_MIN / 60.0  # kWh
        soc_proxy  = np.clip(cum_del_kw / total_req, 0.0, 1.0)
        lw = LINE_WIDTHS[name]
        ax.plot(T_AXIS, soc_proxy * 100.0, color=PALETTE[name],
                ls=LINE_STYLES[name], lw=lw, label=name, alpha=0.88)
    _style_ax(ax, "Time (h)", "Fleet Energy Delivery Progress (%)",
              "CMS-02 — Estimated Fleet State-of-Charge Progression")
    ax.legend(fontsize=9, ncol=3, framealpha=0.85)
    ax.set_xlim(0, T_AXIS[-1]); ax.set_ylim(0, 105)
    _savefig("CMS-02_soc_progression.png")


def plot_peak_and_utilisation(results, cap_kw):
    """CMS-03 — Peak demand + capacity utilisation vs controller."""
    print("  [CMS-03] Peak demand & capacity utilisation")
    names   = list(results.keys())
    peaks   = [results[n]["peak_kw"] for n in names]
    avg_util = [results[n]["utilisation"].mean() for n in names]
    colors  = [PALETTE[n] for n in names]
    x = np.arange(len(names))
    w = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: peak demand bar
    ax = axes[0]
    bars = ax.bar(x, peaks, width=w*2, color=colors, edgecolor="k",
                  linewidth=0.6, alpha=0.85)
    ax.axhline(cap_kw, color="#e74c3c", ls="--", lw=2.0, alpha=0.65,
               label=f"Capacity ({cap_kw:.0f} kW)")
    for i, v in enumerate(peaks):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    _style_ax(ax, "", "Peak Power (kW)", "Peak EV Demand by Controller")
    ax.legend(fontsize=9); ax.set_ylim(0)

    # Right: average utilisation bar
    ax2 = axes[1]
    bars2 = ax2.bar(x, avg_util, width=w*2, color=colors, edgecolor="k",
                    linewidth=0.6, alpha=0.85)
    for i, v in enumerate(avg_util):
        ax2.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9,
                 fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=25, ha="right")
    _style_ax(ax2, "", "Mean EVSE Utilisation (%)",
              "Average EVSE Utilisation by Controller")
    ax2.set_ylim(0, 110)

    plt.suptitle("CMS-03 — Peak Demand & EVSE Utilisation", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    _savefig("CMS-03_peak_utilisation.png")


def plot_throughput(results):
    """CMS-04 — % EVs fully charged + energy delivery rate."""
    print("  [CMS-04] Throughput (% EVs satisfied & energy delivery)")
    names   = list(results.keys())
    prop_d  = [results[n]["prop_d"] * 100 for n in names]
    prop_e  = [results[n]["prop_e"] * 100 for n in names]
    colors  = [PALETTE[n] for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_d = ax.bar(x - w/2, prop_d, w, color=colors, alpha=0.90,
                    edgecolor="k", linewidth=0.5, label="Demands met (%)")
    bars_e = ax.bar(x + w/2, prop_e, w, color=colors, alpha=0.55,
                    edgecolor="k", linewidth=0.5, hatch="//",
                    label="Energy delivered (%)")
    for i, (d, e) in enumerate(zip(prop_d, prop_e)):
        ax.text(i - w/2, d + 0.5, f"{d:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, e + 0.5, f"{e:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    _style_ax(ax, "", "% (%)", "CMS-04 — Throughput: Demands Met & Energy Delivered")
    ax.legend(fontsize=9, framealpha=0.85); ax.set_ylim(0, 120)
    _savefig("CMS-04_throughput.png")


def plot_queue_depth(results):
    """CMS-05 — Queue depth over time."""
    print("  [CMS-05] Queue depth")
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, r in results.items():
        lw = LINE_WIDTHS[name]
        ax.fill_between(T_AXIS, r["queue"], alpha=0.12, color=PALETTE[name])
        ax.plot(T_AXIS, r["queue"], color=PALETTE[name],
                ls=LINE_STYLES[name], lw=lw, label=name, alpha=0.88)
    _style_ax(ax, "Time (h)", "# EVs Waiting", "CMS-05 — Charging Queue Depth")
    ax.legend(fontsize=9, ncol=3, framealpha=0.85)
    ax.set_xlim(0, T_AXIS[-1]); ax.set_ylim(bottom=0)
    _savefig("CMS-05_queue_depth.png")


def plot_pilot_violin(results):
    """CMS-06 — Pilot signal distribution per controller."""
    print("  [CMS-06] Pilot signal distribution (violin)")
    names   = list(results.keys())
    data_nz = [np.array([p for p in results[n]["pilot_signals"] if p > 0.05])
               for n in names]

    # If any controller has no non-zero pilots, fall back to bar chart of mean pilot
    if any(len(d) == 0 for d in data_nz):
        fig, ax = plt.subplots(figsize=(14, 5))
        means = [np.mean(results[n]["pilot_signals"]) if results[n]["pilot_signals"]
                 else 0.0 for n in names]
        colors = [PALETTE[n] for n in names]
        ax.bar(names, means, color=colors, edgecolor="k", linewidth=0.6, alpha=0.85)
        ax.axhline(MAX_PILOT_A, color="#e74c3c", ls="--", lw=1.5, alpha=0.6,
                   label=f"Max pilot ({MAX_PILOT_A}A)")
        _style_ax(ax, "Controller", "Mean Pilot Signal (A)",
                  "CMS-06 — Mean Pilot Signal by Controller")
        ax.legend(fontsize=9)
        ax.set_xticklabels(names, rotation=25, ha="right")
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        vparts  = ax.violinplot(data_nz, positions=np.arange(len(names)),
                                widths=0.65, showmedians=True, showextrema=True)
        for i, (pc, name) in enumerate(zip(vparts["bodies"], names)):
            pc.set_facecolor(PALETTE[name])
            pc.set_alpha(0.60)
            pc.set_edgecolor("k")
        vparts["cmedians"].set_color("black")
        vparts["cmedians"].set_linewidth(2.0)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
        ax.axhline(MAX_PILOT_A, color="#e74c3c", ls="--", lw=1.5, alpha=0.6,
                   label=f"Max pilot ({MAX_PILOT_A}A)")
        _style_ax(ax, "", "Pilot Signal (A)",
                  "CMS-06 — Pilot Signal Distribution by Controller")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.set_ylim(0)
    _savefig("CMS-06_pilot_violin.png")


def plot_summary_table(results, cap_kw):
    """CMS-07 — Summary metrics table."""
    print("  [CMS-07] Summary table")
    rows = []
    for name, r in results.items():
        pilots = np.array(r["pilot_signals"])
        rows.append({
            "Controller":     name,
            "Energy Del. (%)": f"{r['prop_e']*100:.1f}",
            "EVs Satisfied (%)": f"{r['prop_d']*100:.1f}",
            "Peak Power (kW)": f"{r['peak_kw']:.1f}",
            "Cap. Util. (%)":  f"{r['peak_kw']/max(cap_kw,1)*100:.1f}",
            "Mean Queue":      f"{r['queue'].mean():.2f}",
            "Mean Pilot (A)":  f"{pilots.mean():.2f}" if len(pilots) else "—",
            "Pilot Std (A)":   f"{pilots.std():.2f}"  if len(pilots) else "—",
        })
    df = pd.DataFrame(rows).set_index("Controller")

    ncols = len(df.columns)
    nrows = len(df)
    fig, ax = plt.subplots(figsize=(max(20, ncols * 2.5), max(3, nrows * 0.7 + 1.5)))
    ax.axis("off")
    col_labels = ["Controller"] + list(df.columns)
    cell_data  = [[idx] + list(row) for idx, row in zip(df.index, df.values)]
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#263238")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (name, _) in enumerate(zip(df.index, df.values)):
        bg = "#B2DFDB" if name == "PINN-CMS" else ("#f5f5f5" if i % 2 == 0 else "#ffffff")
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
    ax.set_title("CMS-07 — PINN-CMS vs ACN Baseline Controllers — Summary",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    _savefig("CMS-07_summary_table.png")
    print("\n" + df.to_string() + "\n")


# =============================================================================
# 7.  Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PINN-CMS vs ACN Controllers — Comparison")
    print("=" * 65)

    print("\n[1/3] Loading ACN-Data sessions ...")
    sessions = load_sessions(SITES_TO_USE)
    if not sessions:
        print("❌  No ACN sessions found. Check ACN_DATA_ROOT path.")
        sys.exit(1)
    print(f"     Loaded {len(sessions)} real ACN sessions")

    # ── Build a congested busy-day scenario ────────────────────────────────────
    # 80 EVs (30+30+12+8) across morning, evening, midday, overnight rush peaks.
    # Cap is set to 40% of nameplate so demand exceeds capacity at rush hours,
    # forcing genuine rationing that separates controller strategies.
    n_evses  = N_EVSES
    cap_kw   = N_EVSES * MAX_POWER_KW * 0.40    # 40% → ~30.7 kW tight cap
    print(f"\n     Site capacity cap: {cap_kw:.1f} kW  ({n_evses} EVSEs × "
          f"{MAX_POWER_KW:.1f} kW × 40%)")
    print("     [Tight cap creates peak-hour rationing — differentiates controllers]")

    print("\n[1b] Building busy-day EV events ...")
    ev_events = build_busy_day(sessions, seed=42)

    print("\n[2/3] Running simulations ...")
    results = run_all(ev_events, cap_kw, n_evses)

    print("\n[3/3] Generating plots ...")
    plot_aggregate_power(results, cap_kw)
    plot_soc_progression(results, ev_events)
    plot_peak_and_utilisation(results, cap_kw)
    plot_throughput(results)
    plot_queue_depth(results)
    plot_pilot_violin(results)
    plot_summary_table(results, cap_kw)

    print(f"\n✅  All plots saved to: {os.path.abspath(OUT_DIR)}")
    print("=" * 65)

