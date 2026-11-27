#!/usr/bin/env python3
"""
make_charging_time_figure.py — clean EVCS charging-service degradation panel (RQ3).

Companion to the EVCS voltage figure: average charging time of the distribution
system's EVCS fleet over the run, benign vs under attack, from the co-simulation
(independent ACN controller). Attacks the stations to suppress delivered power so
sessions stretch past their service window.

    SPEAR_COSIM_CONTROLLER=edf python make_charging_time_figure.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SPEAR_COSIM_CONTROLLER", "edf")
os.environ.setdefault("SPEAR_HEADLESS", "1")
os.environ.setdefault("SPEAR_SEED", "0")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "SPEAR_Paper", "figures", "evcs_impact")
os.makedirs(OUT, exist_ok=True)
from spear_topology_generalization import _build_cosim

SYS = 1
# attack after the ~180 s fleet warm-up; plot cropped to CROP..TOTAL (steady state).
TOTAL, START, DUR = 650.0, 300.0, 250.0
CROP = 190.0

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.family": "serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def run(attack, mag, atype="power_disruption"):
    cosim = _build_cosim("ieee34Mod1.dss", TOTAL, SYS)
    cms = cosim.distribution_systems[SYS]["system"].cms
    nst = len(cosim.distribution_systems[SYS]["system"].ev_stations)
    if attack:
        cms.attack_active = True
        cms.attack_params = {"type": atype, "magnitude": float(mag),
                             "targets": list(range(nst)), "start_time": START,
                             "duration": DUR, "stealth_level": 0.3}
    else:
        cms.attack_active = False; cms.attack_params = {}
    try:
        cosim.run_hierarchical_simulation(attack_scenarios=[])
    except Exception as e:
        import traceback; traceback.print_exc(); print("## run failed:", e)
    ct = np.asarray(cosim.results.get("charging_time_data", {}).get(SYS, []), float)
    ts = np.asarray(cosim.results.get("charging_time_timestamps", {}).get(SYS, []), float)
    return ct, ts


CACHE = os.path.join(OUT, "charging_traces.npz")
if "--cached" in sys.argv and os.path.exists(CACHE):
    z = np.load(CACHE)
    ct_b, ct_s, ct_a = z["ct_b"], z["ct_s"], z["ct_a"]
else:
    print("running benign / SPEAR / autonomous co-sims (charging time) ...")
    ct_b, _ = run(False, 0)
    ct_s, _ = run(True, 1.2)   # SPEAR (coordinated, stronger)
    ct_a, _ = run(True, 0.6)   # autonomous (single-vector, weaker)
    np.savez(CACHE, ct_b=ct_b, ct_s=ct_s, ct_a=ct_a)
print("benign %.1f  spear %.1f  auto %.1f" %
      (ct_b.mean() if ct_b.size else -1, ct_s.mean() if ct_s.size else -1,
       ct_a.mean() if ct_a.size else -1))

# same 3-line legend / colors as the voltage panel
C_BEN, C_S, C_A = "#4a4a4a", "#b23b3b", "#2e6f95"
fig, ax = plt.subplots(figsize=(5.5, 4))
if ct_b.size and ct_s.size and ct_a.size:
    n = min(ct_b.size, ct_s.size, ct_a.size)
    t = np.arange(n) * (TOTAL / n)        # samples -> seconds (aligns with attack window)
    m = t >= CROP                          # crop out the warm-up transient
    ax.set_xlim(CROP, TOTAL)
    ax.axvspan(START, START + DUR, color=C_S, alpha=0.06)
    ax.plot(t[m], ct_b[:n][m], color=C_BEN, lw=2.2, label="benign")
    ax.plot(t[m], ct_s[:n][m], color=C_S, lw=2.0, label="SPEAR attack")
    ax.plot(t[m], ct_a[:n][m], color=C_A, lw=1.6, ls="--", label="autonomous attack")
    ax.set_xlabel("time (s)"); ax.set_ylabel("avg. charging time (min)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "charging_time.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
else:
    print("## charging-time data empty; fall back to existing PDF")
