"""acn_sim_interface.py
ACN-Sim integration for the EVCS hierarchical co-simulation.
"""

import os
import glob
import gzip
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# ── acnportal imports (graceful fallback) ─────────────────────────────────────
ACN_SIM_AVAILABLE = False
try:
    from acnportal.acnsim import (
        Simulator, ChargingNetwork,
        EV, Linear2StageBattery, EVSE,
        PluginEvent, EventQueue,
    )
    from acnportal.algorithms import BaseAlgorithm
    ACN_SIM_AVAILABLE = True
except Exception:
    warnings.warn(
        "acnportal not installed or failed to import. ACN-Sim integration disabled. "
        "Install with: pip install acnportal",
        ImportWarning, stacklevel=2
    )

    class BaseAlgorithm:                             # minimal stub
        interface = None
        def schedule(self, active_evs):
            return {}
        def register_interface(self, interface):
            self.interface = interface


# ── Constants matching Caltech ACN site ───────────────────────────────────────
CALTECH_EVSE_VOLTAGE_V  = 240.0    # L2 single-phase EVSE voltage (V)
CALTECH_MAX_PILOT_A     = 32.0     # Maximum pilot signal (A)
CALTECH_MIN_PILOT_A     = 0.0      # Minimum pilot signal (A)
DEFAULT_BATTERY_KWH     = 60.0     # Assumed EV battery capacity (kWh)
DEFAULT_MAX_SOC         = 0.80     # Disconnect / target SOC
ACN_PERIOD_MIN          = 5.0      # ACN-Sim scheduling period (minutes)
ACN_PERIOD_S            = ACN_PERIOD_MIN * 60.0   # = 300 s

# ── IDS feature dimension constants ───────────────────────────────────────────
PINN_FEATURE_DIM = 14   # 14-D feature space used by PINN training
IDS_FEATURE_DIM  = 20   # 20-D feature space used by IDS (14 EVCS + 6 grid)


# ACNDataLoader — loads real ACN-Data CSVs for PINN and IDS training

class ACNDataLoader:
    """
    Loads ACN-Data time-series CSVs (.csv and .csv.gz) and builds training
    sequences for the PINN controller and LSTM IDS.
    """

    # --- PINN target names ---
    TARGET_NAMES = ["voltage_V", "current_A", "power_kW"]

    V_NOMINAL   = 240.0                                  # L2 EVSE nominal V
    I_MAX       = CALTECH_MAX_PILOT_A                    # 32 A
    P_MAX_KW    = CALTECH_EVSE_VOLTAGE_V * 32.0 / 1000.0  # ≈ 7.68 kW
    BATT_KWH    = DEFAULT_BATTERY_KWH
    TARGET_SOC  = DEFAULT_MAX_SOC

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._sessions: List[pd.DataFrame] = []
        self._loaded = False

    # ── Low-level loaders ─────────────────────────────────────────────────────

    def _read_csv_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """Read a single CSV or CSV.GZ session file; return None if invalid."""
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as fh:
                    df = pd.read_csv(fh, index_col=0, parse_dates=False)
            else:
                df = pd.read_csv(filepath, index_col=0, parse_dates=False)

            df.columns = [c.strip() for c in df.columns]

            # Require at minimum current and energy columns
            required = {"Charging Current (A)", "Energy Delivered (kWh)"}
            if not required.issubset(df.columns):
                return None

            # Impute missing Voltage from I × V_nominal
            if "Voltage (V)" not in df.columns or df["Voltage (V)"].isna().all():
                df["Voltage (V)"] = self.V_NOMINAL
            else:
                df["Voltage (V)"] = df["Voltage (V)"].fillna(self.V_NOMINAL)

            # Impute missing Power from I × V
            if "Power (kW)" not in df.columns or df["Power (kW)"].isna().all():
                df["Power (kW)"] = (
                    df["Charging Current (A)"] * df["Voltage (V)"] / 1000.0
                )
            else:
                df["Power (kW)"] = df["Power (kW)"].fillna(
                    df["Charging Current (A)"] * df["Voltage (V)"] / 1000.0
                )

            df = df.dropna(subset=["Charging Current (A)", "Energy Delivered (kWh)"])

            if len(df) >= 8:
                return df
        except Exception:
            pass
        return None

    def load_all_csvs(self) -> int:
        """Scan data_dir recursively for *.csv and *.csv.gz files; load them."""
        plain_pattern = os.path.join(self.data_dir, "*.csv")
        gz_pattern    = os.path.join(self.data_dir, "**", "*.csv.gz")
        csv_pattern   = os.path.join(self.data_dir, "**", "*.csv")

        files = (
            sorted(glob.glob(plain_pattern))
            + sorted(glob.glob(gz_pattern,  recursive=True))
            + sorted(glob.glob(csv_pattern, recursive=True))
        )
        # Deduplicate (plain_pattern may overlap csv_pattern)
        seen = set()
        unique_files = []
        for f in files:
            norm = os.path.normpath(f)
            if norm not in seen:
                seen.add(norm)
                unique_files.append(f)

        self._sessions = []
        for filepath in unique_files:
            df = self._read_csv_file(filepath)
            if df is not None:
                self._sessions.append(df)

        self._loaded = True
        return len(self._sessions)

    def load_from_dirs(
        self,
        site_dirs: List[Tuple[str, int]],
    ) -> int:
        """
        Load up to *max_files* CSV/GZ files per site directory.

        Parameters
        ----------
        site_dirs : list of (dir_path, max_files) tuples
        """
        self._sessions = []
        for dir_path, max_files in site_dirs:
            if not os.path.isdir(dir_path):
                continue
            gz_files  = sorted(glob.glob(os.path.join(dir_path, "*.csv.gz")))
            csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
            candidates = (gz_files + csv_files)[:max_files]
            for filepath in candidates:
                df = self._read_csv_file(filepath)
                if df is not None:
                    self._sessions.append(df)

        self._loaded = True
        return len(self._sessions)

    # ── Feature engineering helpers ───────────────────────────────────────────

    def _derive_soc(self, energy_arr: np.ndarray) -> np.ndarray:
        """Derive SOC from cumulative energy delivered."""
        capacity_kwh = self.BATT_KWH * self.TARGET_SOC   # usable capacity
        return np.clip(energy_arr / max(capacity_kwh, 1e-6), 0.0, 1.0)

    def _session_to_pinn_sequences(
        self, df: pd.DataFrame, seq_len: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert one session DataFrame to PINN (14-D) sequences."""
        n = len(df)
        if n < seq_len + 1:
            return None, None

        i_arr = df["Charging Current (A)"].values.astype(float)
        v_arr = df["Voltage (V)"].values.astype(float)
        p_arr = df["Power (kW)"].values.astype(float)
        e_arr = df["Energy Delivered (kWh)"].values.astype(float)

        soc_arr = self._derive_soc(e_arr)
        v_pu    = np.clip(v_arr / max(self.V_NOMINAL, 1e-6), 0.3, 1.5)
        t_norm  = np.linspace(0.0, 1.0, n)
        i_norm  = np.clip(i_arr / max(self.I_MAX, 1e-6), 0.0, 1.1)
        p_norm  = np.clip(p_arr / max(self.P_MAX_KW, 1e-6), 0.0, 1.1)
        p_prev  = np.roll(p_norm, 1); p_prev[0] = 0.0

        features = np.stack([
            soc_arr,                                              # 0 soc
            v_pu,                                                 # 1 voltage_pu
            np.full(n, 0.5),                                      # 2 grid_freq
            p_norm,                                               # 3 demand_factor
            np.clip(1.0 - v_pu, 0.0, 1.0),                       # 4 voltage_priority
            1.0 - soc_arr,                                        # 5 urgency
            t_norm,                                               # 6 time_of_day_norm
            np.full(n, 0.5),                                      # 7 bus_distance_norm
            i_norm,                                               # 8 load_factor
            p_prev,                                               # 9 prev_power_norm
            p_norm,                                               # 10 ac_power_in_norm
            np.full(n, 0.95),                                     # 11 efficiency
            np.zeros(n),                                          # 12 power_balance_err
            np.zeros(n),                                          # 13 dc_link_dev
        ], axis=1)  # (n, 14)

        targets = np.stack([
            v_arr / self.V_NOMINAL,                               # voltage (normalised)
            i_arr / self.I_MAX,                                   # current (normalised)
            p_norm,                                               # power   (normalised)
        ], axis=1)  # (n, 3)

        X, y = [], []
        for start in range(n - seq_len):
            X.append(features[start: start + seq_len])
            y.append(targets[start + seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _session_to_ids_sequences(
        self, df: pd.DataFrame, seq_len: int
    ) -> Optional[np.ndarray]:
        """Return 20-D IDS sequences (benign grid cols = neutral defaults)."""
        pinn_seqs, _ = self._session_to_pinn_sequences(df, seq_len)
        if pinn_seqs is None:
            return None
        N, T, _ = pinn_seqs.shape
        # Grid features — neutral values for real benign ACN data
        grid_feats = np.zeros((N, T, 6), dtype=np.float32)
        grid_feats[:, :, 3] = 1.0  # bus_voltage_min_pu = 1.0
        grid_feats[:, :, 4] = 1.0  # bus_voltage_max_pu = 1.0
        return np.concatenate([pinn_seqs, grid_feats], axis=2)  # (N, T, 20)

    # ── Public build methods ───────────────────────────────────────────────────

    def build_training_sequences(
        self,
        seq_len: int = 8,
        n_samples: Optional[int] = None,
        max_sessions: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """
        Build PINN training sequences (14-D features, 3-D targets).

        Returns torch.FloatTensor or (None, None) on failure.
        """
        if not self._loaded:
            self.load_all_csvs()
        if not self._sessions:
            return None, None

        try:
            import torch
        except ImportError:
            return None, None

        sessions = self._sessions
        if max_sessions is not None:
            sessions = sessions[:max_sessions]

        all_X, all_y = [], []
        for df in sessions:
            X, y = self._session_to_pinn_sequences(df, seq_len)
            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return None, None

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        if n_samples is not None and len(X_all) > n_samples:
            idx = np.random.choice(len(X_all), n_samples, replace=False)
            X_all = X_all[idx]
            y_all = y_all[idx]

        return torch.FloatTensor(X_all), torch.FloatTensor(y_all)

    def build_ids_sequences(
        self,
        seq_len: int = 10,
        n_samples: Optional[int] = None,
        max_sessions: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Build IDS training sequences (20-D features).

        Returns torch.FloatTensor of shape (N, seq_len, 20) or None.
        """
        if not self._loaded:
            self.load_all_csvs()
        if not self._sessions:
            return None

        try:
            import torch
        except ImportError:
            return None

        sessions = self._sessions
        if max_sessions is not None:
            sessions = sessions[:max_sessions]

        all_X = []
        for df in sessions:
            X = self._session_to_ids_sequences(df, seq_len)
            if X is not None and len(X) > 0:
                all_X.append(X)

        if not all_X:
            return None

        X_all = np.concatenate(all_X, axis=0)

        if n_samples is not None and len(X_all) > n_samples:
            idx = np.random.choice(len(X_all), n_samples, replace=False)
            X_all = X_all[idx]

        return torch.FloatTensor(X_all)


# PINNCMSAlgorithm — ACN-Sim scheduling algorithm wrapping the PINN CMS

class PINNCMSAlgorithm(BaseAlgorithm):
    """
    ACN-Sim scheduling algorithm that delegates to the PINN-based CMS.
    Per §7 of the guideline: only I_ref (clipped) is sent to ACN-Sim.
    V_ref and P_ref are still used inside the PINN for physics loss.
    """

    def __init__(self, cms, station_ids: List[str],
                 evse_voltage: float = CALTECH_EVSE_VOLTAGE_V):
        super().__init__()
        self.cms            = cms
        self.station_ids    = station_ids
        self.evse_voltage   = evse_voltage
        self.current_time_s   = 0.0
        self.system_frequency = 60.0
        self.bus_voltages_pu: Dict[str, float] = {}
        self.last_schedule: Dict[str, float] = {sid: 0.0 for sid in station_ids}
        self.last_soc:      Dict[str, float] = {sid: 0.35 for sid in station_ids}

    def set_context(self, current_time_s: float,
                    bus_voltages_pu: Dict[str, float],
                    system_frequency: float) -> None:
        self.current_time_s   = current_time_s
        self.bus_voltages_pu  = bus_voltages_pu
        self.system_frequency = system_frequency

    def schedule(self, active_evs) -> Dict[str, List[float]]:
        schedules: Dict[str, List[float]] = {}

        for ev in active_evs:
            sid = ev.station_id
            station_idx = self._station_idx(sid)

            # v0.3.3: SOC via ev._battery._soc
            try:
                soc = float(ev._battery._soc)
            except Exception:
                soc = max(0.0, 1.0 - float(getattr(ev, "percent_remaining", 0.5)))

            prev_i = self.last_schedule.get(sid, 0.0)
            prev_p = prev_i * self.evse_voltage / 1000.0
            self.last_soc[sid] = soc

            v_pu = self.bus_voltages_pu.get(sid, 1.0)
            dynamics_result = {
                "soc":                 soc,
                "voltage_measured":    self.evse_voltage * v_pu,
                "current_measured":    prev_i,
                "power_measured":      prev_p,
                "total_power":         prev_p,
                "grid_frequency":      self.system_frequency,
                "ac_voltage_rms":      self.evse_voltage * v_pu,
                "ac_current_rms":      prev_i,
                "system_efficiency":   0.95,
                "power_balance_error": 0.0,
                "dc_link_voltage":     self.evse_voltage,
            }

            try:
                _v, i_ref, _p = self.cms.optimize_charging(
                    station_idx, self.current_time_s,
                    {sid: v_pu}, self.system_frequency, dynamics_result,
                )
            except Exception:
                # Heuristic fallback: proportional to remaining SOC headroom
                i_ref = CALTECH_MAX_PILOT_A * (1.0 - soc) * 0.8

            max_p = CALTECH_MAX_PILOT_A
            min_p = CALTECH_MIN_PILOT_A
            if self.interface is not None:
                try:
                    max_p = self.interface.max_pilot_signal(sid)
                    min_p = self.interface.min_pilot_signal(sid)
                except Exception:
                    pass

            i_pilot = float(np.clip(i_ref, min_p, max_p))
            schedules[sid]          = [i_pilot]
            self.last_schedule[sid] = i_pilot

        return schedules

    def _station_idx(self, station_id: str) -> int:
        try:
            return self.station_ids.index(station_id)
        except ValueError:
            digits = "".join(c for c in station_id if c.isdigit())
            return int(digits[-2:]) if len(digits) >= 2 else 0


# ACNSimZone — one distribution system's ACN-Sim instance

class ACNSimZone:
    """Manages one ACN-Sim Simulator instance (one DS, 10 EVSEs)."""

    def __init__(self, ds_id: int, n_evses: int = 10, period_min: float = 5.0,
                 evse_voltage: float = 240.0, sim_duration_s: float = 3600.0,
                 rng_seed: Optional[int] = None):
        self.ds_id        = ds_id
        self.n_evses      = n_evses
        self.period_min   = period_min
        self.period_s     = period_min * 60.0
        self.evse_voltage = evse_voltage
        self.sim_duration_s = sim_duration_s
        self.total_periods  = max(1, int(sim_duration_s / self.period_s))
        self.rng = np.random.default_rng(
            seed=rng_seed if rng_seed is not None else ds_id * 7919)

        self.station_ids = [f"DS{ds_id}_EV{i:02d}" for i in range(n_evses)]

        self.network:   Any = None
        self.algorithm: Optional[PINNCMSAlgorithm] = None
        self.simulator: Any = None

        self.current_period      = -1
        self._cached_metrics:    Dict = {}
        self._session_durations: List[float] = []

        self._pilot_A:     Dict[str, float] = {s: 0.0 for s in self.station_ids}
        self._actual_A:    Dict[str, float] = {s: 0.0 for s in self.station_ids}
        self._station_soc: Dict[str, float] = {
            s: float(np.clip(self.rng.normal(0.35, 0.18), 0.10, 0.65))
            for s in self.station_ids
        }
        self._station_state: Dict[str, str] = {s: "IDLE" for s in self.station_ids}
        self._build_initial_cache()

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, cms) -> None:
        if not ACN_SIM_AVAILABLE:
            print(f"#  ACNSimZone DS{self.ds_id}: acnportal unavailable — fallback mode")
            return
        self.network   = self._build_network()
        events         = self._build_event_queue()
        self.algorithm = PINNCMSAlgorithm(
            cms=cms, station_ids=self.station_ids, evse_voltage=self.evse_voltage)
        self.simulator = Simulator(
            network=self.network, scheduler=self.algorithm,
            events=events, start=0, period=self.period_min,
            store_schedule_history=False)
        self.current_period = -1
        self._build_initial_cache()
        print(f"# ACNSimZone DS{self.ds_id}: {self.n_evses} EVSEs "
              f"@ {self.evse_voltage}V, period={self.period_min} min, "
              f"total_periods={self.total_periods}")

    def _build_network(self):
        network = ChargingNetwork()
        for sid in self.station_ids:
            evse = EVSE(sid, max_rate=CALTECH_MAX_PILOT_A,
                        min_rate=CALTECH_MIN_PILOT_A)
            network.register_evse(evse, voltage=self.evse_voltage, phase_angle=0.0)
        return network

    def _build_event_queue(self):
        events = EventQueue()
        max_power_kw = self.evse_voltage * CALTECH_MAX_PILOT_A / 1000.0
        for evse_idx, sid in enumerate(self.station_ids):
            arr, session_num = 0, 0
            while arr < self.total_periods:
                a_soc = float(np.clip(self.rng.normal(0.35, 0.18), 0.10, 0.65))
                req_energy   = (DEFAULT_MAX_SOC - a_soc) * DEFAULT_BATTERY_KWH
                charge_h     = req_energy / max(max_power_kw, 0.01)
                charge_periods = max(1, int(charge_h * 60 / self.period_min))
                dep = min(arr + charge_periods, self.total_periods - 1)
                try:
                    battery = Linear2StageBattery(
                        capacity=DEFAULT_BATTERY_KWH,
                        init_charge=a_soc * DEFAULT_BATTERY_KWH,
                        max_power=max_power_kw)
                    ev = EV(arrival=arr, departure=dep,
                            requested_energy=req_energy,
                            station_id=sid,
                            session_id=f"sess_{self.ds_id}_{evse_idx}_{session_num}",
                            battery=battery)
                    events.add_event(PluginEvent(arr, ev))
                except Exception:
                    pass
                arr = dep + 2
                session_num += 1
        return events

    def _build_initial_cache(self) -> None:
        metrics: Dict[str, Any] = {}
        for sid in self.station_ids:
            metrics[sid] = {
                "pilot_A": 0.0, "actual_current_A": 0.0,
                "voltage_V": self.evse_voltage, "power_kW": 0.0,
                "soc": self._station_soc.get(sid, 0.35),
                "state": "IDLE", "connected": False,
            }
        metrics["_aggregate"] = {
            "total_power_kW": 0.0, "avg_charging_time_min": 45.0,
            "queue_length": 0, "active_sessions": 0,
        }
        self._cached_metrics = metrics

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self, current_time_s: float,
             bus_voltages_pu: Optional[Dict[str, float]] = None,
             system_frequency: float = 60.0) -> Dict:
        """Advance simulation to current_time_s; return per-EVSE metrics dict."""
        if self.simulator is None:
            return self._cached_metrics

        new_period = int(current_time_s / self.period_s)
        if new_period <= self.current_period:
            return self._cached_metrics   # hold pattern between ACN periods

        bv = bus_voltages_pu or {}
        if self.algorithm is not None:
            self.algorithm.set_context(current_time_s, bv, system_frequency)

        steps_needed = new_period - max(0, self.current_period)
        for _ in range(steps_needed):
            if self.simulator.event_queue.empty():
                break
            try:
                self._acn_step_one_period()
            except Exception as exc:
                warnings.warn(f"ACNSimZone DS{self.ds_id}: step error: {exc}")
                break

        self.current_period = new_period
        self._collect_results()
        return self._cached_metrics

    def _acn_step_one_period(self) -> None:
        """Execute one ACN-Sim iteration for acnportal v0.3.3."""
        try:
            from acnportal.acnsim.simulator import _increase_width
        except ImportError:
            def _increase_width(arr, width):
                if arr.shape[1] >= width:
                    return arr
                pad = np.zeros((arr.shape[0], width - arr.shape[1]))
                return np.hstack([arr, pad])

        sim = self.simulator

        # 1. Process events
        current_events = sim.event_queue.get_current_events(sim._iteration)
        for e in current_events:
            sim.event_history.append(e)
            sim._process_event(e)

        # 2. Get schedule from PINN-CMS
        new_schedule = sim.scheduler.run()
        sim._update_schedules(new_schedule)
        if sim.schedule_history is not None:
            sim.schedule_history[sim._iteration] = new_schedule
        sim._last_schedule_update = sim._iteration
        sim._resolve = False

        # 3. Expand storage arrays
        last_ts = sim.event_queue.get_last_timestamp()
        width = (max(last_ts + 1, sim._iteration + 1)
                 if last_ts is not None else sim._iteration + 1)
        sim.pilot_signals  = _increase_width(sim.pilot_signals,  width)
        sim.charging_rates = _increase_width(sim.charging_rates, width)

        # 4. Update network pilots and collect actual rates
        sim.network.update_pilots(sim.pilot_signals, sim._iteration, sim.period)
        sim._store_actual_charging_rates()
        sim.network.post_charging_update()

        # 5. Advance clock
        sim._iteration += 1

    def _collect_results(self) -> None:
        metrics: Dict[str, Any] = {}
        total_kw, active_cnt = 0.0, 0

        ev_lookup: Dict[str, Any] = {}
        try:
            for ev in self.network.active_evs:
                ev_lookup[ev.station_id] = ev
        except Exception:
            pass

        # v0.3.3: current_charging_rates is numpy array indexed by station pos
        try:
            rates_arr = self.network.current_charging_rates
            net_sids  = list(self.network.station_ids)
            rates: Dict[str, float] = {
                net_sids[i]: float(rates_arr[i]) for i in range(len(net_sids))
            }
        except Exception:
            rates = {}

        for sid in self.station_ids:
            ev = ev_lookup.get(sid)
            if ev is not None:
                try:
                    soc = float(ev._battery._soc)
                except Exception:
                    soc = max(0.0, 1.0 - float(getattr(ev, "percent_remaining", 1.0)))

                act_i    = float(getattr(ev, "current_charging_rate",
                                         rates.get(sid, 0.0)))
                pilot    = (self.algorithm.last_schedule.get(sid, act_i)
                            if self.algorithm else act_i)
                state    = "ADAPTIVE" if act_i > 0.5 else "READY"
                power_kw = act_i * self.evse_voltage / 1000.0
                connected = True

                self._station_soc[sid]   = soc
                self._station_state[sid] = state
                self._pilot_A[sid]       = pilot
                self._actual_A[sid]      = act_i
                total_kw  += power_kw
                active_cnt += 1

                remaining = getattr(ev, "remaining_demand", None)
                if remaining is not None and float(remaining) < 0.1:
                    arr = getattr(ev, "arrival", 0)
                    dep = getattr(ev, "departure", self.current_period)
                    dur_min = (dep - arr) * self.period_min
                    if 5.0 < dur_min < 300.0:
                        self._session_durations.append(dur_min)
            else:
                soc      = self._station_soc.get(sid, 0.35)
                act_i    = 0.0
                pilot    = 0.0
                state    = "IDLE"
                power_kw = 0.0
                connected = False

            metrics[sid] = {
                "pilot_A":          pilot,
                "actual_current_A": act_i,
                "voltage_V":        self.evse_voltage,
                "power_kW":         power_kw,
                "soc":              soc,
                "state":            state,
                "connected":        connected,
            }

        avg_ct = (float(np.mean(self._session_durations[-20:]))
                  if self._session_durations else 45.0)
        metrics["_aggregate"] = {
            "total_power_kW":        total_kw,
            "avg_charging_time_min": avg_ct,
            "queue_length":          max(0, self.n_evses - active_cnt),
            "active_sessions":       active_cnt,
        }
        self._cached_metrics = metrics

    def get_cached_metrics(self) -> Dict:
        return self._cached_metrics


# ACNSimFleet — manages all 6 zone instances

class ACNSimFleet:
    """Manages 6 ACNSimZone instances (Pattern A — one per DS)."""

    def __init__(self, n_zones: int = 6, n_evses_per_zone: int = 10,
                 acn_data_dir: Optional[str] = None,
                 period_min: float = 5.0, evse_voltage: float = 240.0,
                 sim_duration_s: float = 3600.0):
        self.n_zones        = n_zones
        self.n_evses        = n_evses_per_zone
        self.acn_data_dir   = acn_data_dir
        self.period_min     = period_min
        self.evse_voltage   = evse_voltage
        self.sim_duration_s = sim_duration_s
        self._zones: Dict[int, ACNSimZone] = {}
        self._initialized = False

    def initialize_zones(self, cms_list: List) -> None:
        if not ACN_SIM_AVAILABLE:
            print("#  ACNSimFleet: acnportal not installed — all zones in fallback mode.")
        for i in range(self.n_zones):
            ds_id = i + 1
            cms   = cms_list[i] if i < len(cms_list) else None
            zone  = ACNSimZone(
                ds_id=ds_id, n_evses=self.n_evses,
                period_min=self.period_min, evse_voltage=self.evse_voltage,
                sim_duration_s=self.sim_duration_s, rng_seed=ds_id * 31337)
            if cms is not None:
                zone.initialize(cms)
            else:
                zone.initialize(cms=None)   # fallback heuristic inside algorithm
            self._zones[ds_id] = zone
        self._initialized = True
        print(f"# ACNSimFleet: {self.n_zones} zones × {self.n_evses} EVSEs initialized "
              f"(period={self.period_min} min, V={self.evse_voltage} V)")

    def step_all(self, current_time_s: float,
                 bus_voltages_per_ds: Optional[Dict[int, Dict[str, float]]] = None,
                 frequencies: Optional[Dict[int, float]] = None) -> Dict[int, Dict]:
        results: Dict[int, Dict] = {}
        bv_map   = bus_voltages_per_ds or {}
        freq_map = frequencies or {}
        for ds_id, zone in self._zones.items():
            results[ds_id] = zone.step(
                current_time_s, bv_map.get(ds_id, {}), freq_map.get(ds_id, 60.0))
        return results

    def get_zone(self, ds_id: int) -> ACNSimZone:
        return self._zones[ds_id]

    def get_acn_data_loader(self) -> Optional[ACNDataLoader]:
        if self.acn_data_dir and os.path.isdir(self.acn_data_dir):
            return ACNDataLoader(self.acn_data_dir)
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized
