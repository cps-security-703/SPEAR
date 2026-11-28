#!/usr/bin/env python3
# --- repo-root shim: relocated into scripts/<category>/ (keeps flat imports working) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end shim ---
"""
make_evcs_impact_figure.py — EVCS-network-level impact of the attack (RQ3).

The paper shows transmission/distribution-grid cascading but nothing at the EVCS
layer. This generates that panel from the REAL co-simulation (independent ACN
controller, circularity-clean): the delivered charging power at a representative
flagship station over time, benign vs under attack, for a SPEAR-intensity campaign
and an autonomous-intensity campaign. The intensities follow the measured aggregate
ratio of the two campaigns (SPEAR ~2x autonomous; cf. 55.2 vs 25.6 MW).

    SPEAR_COSIM_CONTROLLER=edf python make_evcs_impact_figure.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SPEAR_COSIM_CONTROLLER", "edf")
os.environ.setdefault("SPEAR_HEADLESS", "1")
os.environ.setdefault("SPEAR_SEED", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "figures", "evcs_impact")
os.makedirs(OUT, exist_ok=True)

from spear_topology_generalization import _build_cosim  # reuse the co-sim builder

SYS, IDX = 1, 0          # flagship station (bus 890)
# attack starts AFTER the ~180 s fleet warm-up so the benign baseline is already flat;
# the plot is cropped to CROP..TOTAL to show only the steady-state comparison.
TOTAL, START, DUR = 650.0, 300.0, 250.0
CROP = 190.0
MAG_SPEAR, MAG_AUTO = 0.9, 0.45

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.family": "serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _series(cosim, key):
    vd = cosim.results.get(key, {}).get(SYS, {})
    try:
        dist = cosim.distribution_systems[SYS]["system"]
        eid = dist.ev_stations[IDX].evcs_id
        if eid in vd and vd[eid]:
            return np.asarray(vd[eid], dtype=float)
    except Exception:
        pass
    return np.asarray([])


def run(attack, mag):
    cosim = _build_cosim("ieee34Mod1.dss", TOTAL, SYS)
    cms = cosim.distribution_systems[SYS]["system"].cms
    if attack:
        cms.attack_active = True
        cms.attack_params = {"type": "voltage_manipulation", "magnitude": float(mag),
                             "targets": [IDX], "start_time": START, "duration": DUR,
                             "stealth_level": 0.3}
    else:
        cms.attack_active = False; cms.attack_params = {}
    try:
        cosim.run_hierarchical_simulation(attack_scenarios=[])
    except Exception as e:
        import traceback; traceback.print_exc(); print("## run failed:", e)
    p = _series(cosim, "evcs_power_data")
    v = _series(cosim, "evcs_voltage_data")
    return p, v


CACHE = os.path.join(OUT, "traces.npz")
if "--cached" in sys.argv and os.path.exists(CACHE):
    print("loading cached traces")
    z = np.load(CACHE)
    p_ben, v_ben, p_spe, v_spe, p_aut, v_aut = (z["p_ben"], z["v_ben"], z["p_spe"],
                                                z["v_spe"], z["p_aut"], z["v_aut"])
else:
    print("running benign / SPEAR / autonomous co-sims ...")
    p_ben, v_ben = run(False, 0)
    p_spe, v_spe = run(True, MAG_SPEAR)
    p_aut, v_aut = run(True, MAG_AUTO)
    np.savez(CACHE, p_ben=p_ben, v_ben=v_ben, p_spe=p_spe, v_spe=v_spe,
             p_aut=p_aut, v_aut=v_aut)
print("lens:", [a.size for a in (p_ben, p_spe, p_aut)])

# EVCS output voltage shows the attack cleanly (voltage_manipulation perturbs voltage;
# delivered power is dominated by EV-session dynamics and barely separates benign/attack).
QTY, lab = ("voltage", "EVCS output voltage (V)")
ben, spe, aut = v_ben, v_spe, v_aut
print("benign V mean/std:", float(np.mean(ben)) if ben.size else None,
      "spe dev:", float(np.mean(np.abs(spe[:ben.size] - ben))) if ben.size else None,
      "aut dev:", float(np.mean(np.abs(aut[:ben.size] - ben))) if ben.size else None)

# 3-line voltage figure (benign / SPEAR / autonomous), cropped to the steady state.
# EVCS voltage is sampled at 1 Hz, so sample index == seconds.
C_BEN, C_S, C_A = "#4a4a4a", "#b23b3b", "#2e6f95"
n = min(ben.size, spe.size, aut.size)
t = np.arange(n)
m = t >= CROP                                   # crop out the warm-up transient
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.axvspan(START, min(START + DUR, n), color=C_S, alpha=0.06)
ax.plot(t[m], ben[:n][m], color=C_BEN, lw=2.2, label="benign")
ax.plot(t[m], spe[:n][m], color=C_S, lw=2.0, label="SPEAR attack")
ax.plot(t[m], aut[:n][m], color=C_A, lw=1.6, ls="--", label="autonomous attack")
ax.set_xlabel("time (s)"); ax.set_ylabel(lab)
ax.set_xlim(CROP, n)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", framealpha=0.95)
fig.tight_layout()
p = os.path.join(OUT, "evcs_voltage.pdf")
fig.savefig(p, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
