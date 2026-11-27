#!/usr/bin/env python3
"""
make_cascade_figures.py — large-font transmission-grid cascade panels (SPEAR vs auto).

Multi-DS co-simulation (6 distribution systems). The SPEAR-vs-autonomous cascade gap
is a *coordination* effect: SPEAR's semantic planner hits several distribution-system
aggregators simultaneously (coordinated kill chain), whereas the autonomous bandit is
single/few-vector at any instant. power_disruption suppresses each attacked system's
delivered power, so the aggregate transmission-load drop (and the resulting frequency
deviation) scales with the number of simultaneously-coordinated aggregators.

Regenerates, in large-font style (figsize 5.5x4, labels 15, ticks 15, legend 13,
no title, 'SPEAR'/'autonomous' legend):
    load_comparison_spear.pdf   (a) benign vs SPEAR load disturbance
    load_comparison_auto.pdf    (b) benign vs autonomous load disturbance
    frequency_delta_spear.pdf   (a) benign vs SPEAR frequency response
    frequency_delta_auto.pdf    (b) benign vs autonomous frequency response

    python make_cascade_figures.py            # run co-sims, write panels
    python make_cascade_figures.py --cached   # re-plot from cached traces only
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SPEAR_HEADLESS", "1")
os.environ.setdefault("SPEAR_SEED", "0")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "SPEAR_Paper", "figures", "new_cascading_effect")
os.makedirs(OUT, exist_ok=True)
from hierarchical_cosimulation import HierarchicalCoSimulation

# ten distribution systems (connection bus is vestigial — loads aggregate into
# P_dist_total; see CLAUDE.md). Canonical setup: add_distribution_system(i, dss, 10)
# for i=1..10, exactly as enhanced_integrated_evcs_system.py.
N_SYS = 10
DS_BUS = 10
TOTAL, START, DUR = 650.0, 300.0, 250.0
CROP = 190.0
F_NOM = 60.0

# SPEAR coordinates the <=6-of-10 budget simultaneously; autonomous is single-vector
SPEAR_SYS = [1, 2, 3, 4, 5, 6]
AUTO_SYS  = [1]
ATK_MAG   = 1.0

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.family": "serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_BEN, C_S, C_A = "#4a4a4a", "#b23b3b", "#2e6f95"


def build():
    cosim = HierarchicalCoSimulation(use_enhanced_pinn=True)
    cosim.total_duration = float(TOTAL)
    cosim.total_time = float(TOTAL)
    cosim.simulation_time = 0.0
    for i in range(1, N_SYS + 1):
        cosim.add_distribution_system(i, "ieee34Mod1.dss", DS_BUS)
    cosim.setup_ev_charging_stations()
    return cosim


def run(attacked_systems):
    """Run one co-sim; attack every station of each system in `attacked_systems`.
    Return (time, total_load[MW], frequency[Hz])."""
    cosim = build()
    for sid, dinfo in cosim.distribution_systems.items():
        sysobj = dinfo["system"]
        cms = getattr(sysobj, "cms", None)
        if cms is None:
            continue
        if sid in attacked_systems:
            nst = len(sysobj.ev_stations)
            cms.attack_active = True
            cms.attack_params = {"type": "power_disruption", "magnitude": float(ATK_MAG),
                                 "targets": list(range(nst)), "start_time": START,
                                 "duration": DUR, "stealth_level": 0.3}
        else:
            cms.attack_active = False; cms.attack_params = {}
    try:
        cosim.run_hierarchical_simulation(attack_scenarios=[])
    except Exception as e:
        import traceback; traceback.print_exc(); print("## run failed:", e)
    r = cosim.results
    return (np.asarray(r.get("time", []), float),
            np.asarray(r.get("total_load", []), float),
            np.asarray(r.get("frequency", []), float))


CACHE = os.path.join(OUT, "cascade_traces.npz")
if "--cached" in sys.argv and os.path.exists(CACHE):
    z = np.load(CACHE)
    t_b, l_b, f_b = z["t_b"], z["l_b"], z["f_b"]
    t_s, l_s, f_s = z["t_s"], z["l_s"], z["f_s"]
    t_a, l_a, f_a = z["t_a"], z["l_a"], z["f_a"]
else:
    print("running benign / SPEAR(6-DS coordinated) / autonomous(1-DS) co-sims ...")
    t_b, l_b, f_b = run([])
    t_s, l_s, f_s = run(SPEAR_SYS)
    t_a, l_a, f_a = run(AUTO_SYS)
    np.savez(CACHE, t_b=t_b, l_b=l_b, f_b=f_b, t_s=t_s, l_s=l_s, f_s=f_s,
             t_a=t_a, l_a=l_a, f_a=f_a)


def _align(tb, yb, ta, ya):
    n = min(tb.size, yb.size, ta.size, ya.size)
    if n == 0:
        return None
    t = np.arange(n) * (TOTAL / n)
    m = t >= CROP
    return t[m], yb[:n][m], ya[:n][m]


def panel(tb, yb, ta, ya, ylabel, attack_label, color_attack, fname, to_delta=None, diff=False, ylim=None):
    al = _align(tb, yb, ta, ya)
    if al is None:
        print("## empty data for", fname); return None
    t, benign, attack = al
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_xlim(CROP, TOTAL)
    ax.axvspan(START, START + DUR, color=color_attack, alpha=0.06)
    if diff:
        # Attack-induced CHANGE (attack - benign). Base-independent: because the
        # transmission base is constant, d(transmission load) = d(distribution load).
        d = attack - benign
        ax.axhline(0.0, color=C_BEN, lw=1.8, ls=(0, (5, 2)), label="no attack")
        ax.plot(t, d, color=color_attack, lw=2.3, label=attack_label)
        peak = float(np.nanmax(np.abs(d))) if d.size else float("nan")
    else:
        if to_delta is not None:
            benign = benign - to_delta; attack = attack - to_delta
        ax.plot(t, benign, color=C_BEN, lw=2.2, label="benign")
        ax.plot(t, attack, color=color_attack, lw=2.0, label=attack_label)
        peak = float(np.nanmax(np.abs(attack - benign))) if attack.size else float("nan")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    if ylim is not None: ax.set_ylim(*ylim)          # shared across the (a)/(b) pair
    ax.grid(True, alpha=0.3); ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, fname); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname, " peak|attack-benign| = %.3f" % peak)
    return peak


# shared y-limits so (a) SPEAR and (b) autonomous are directly comparable
LOAD_YLIM = (-64.0, 4.0)
FREQ_YLIM = (-0.24, 0.20)
pl_s = panel(t_b, l_b, t_s, l_s, "Change in transmission\nload (MW)", "SPEAR attack", C_S,
             "load_comparison_spear.pdf", diff=True, ylim=LOAD_YLIM)
pl_a = panel(t_b, l_b, t_a, l_a, "Change in transmission\nload (MW)", "autonomous attack", C_A,
             "load_comparison_auto.pdf", diff=True, ylim=LOAD_YLIM)
pf_s = panel(t_b, f_b, t_s, f_s, "Change in\nfrequency (Hz)", "SPEAR attack",
             C_S, "frequency_delta_spear.pdf", diff=True, ylim=FREQ_YLIM)
pf_a = panel(t_b, f_b, t_a, f_a, "Change in\nfrequency (Hz)", "autonomous attack",
             C_A, "frequency_delta_auto.pdf", diff=True, ylim=FREQ_YLIM)

print("\n=== caption scalars ===")
print("load peak disturbance:  SPEAR %.1f MW | auto %.1f MW" %
      (pl_s if pl_s else -1, pl_a if pl_a else -1))
print("frequency peak |df|:    SPEAR %.3f Hz | auto %.3f Hz" %
      (pf_s if pf_s else -1, pf_a if pf_a else -1))
