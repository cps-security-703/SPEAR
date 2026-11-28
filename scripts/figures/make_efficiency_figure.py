#!/usr/bin/env python3
# --- repo-root shim: relocated into scripts/<category>/ (keeps flat imports working) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end shim ---
"""
make_efficiency_figure.py — the RQ2 "parameter search is a tie" figure.

Reuses spear_sample_efficiency.py's exact curve methodology so the figure is
consistent with Table V. Reads the 5 autonomous + 5 SPEAR-floor per-seed
reward_history files from ms_results/ and plots:

  (a) mean learning curve (shared-scale normalized SAC reward vs training progress),
      SPEAR-floor vs autonomous, averaged over the 6 attacks and 5 seeds -> the two
      converge to the same plateau (visual parity).
  (b) per-attack interactions-to-90%-plateau (ITP, lower = more sample-efficient);
      SPEAR-floor trends <= autonomous on 4/6 attacks.

    python make_efficiency_figure.py
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spear_sample_efficiency as se   # attack_curve, smooth, plateau, start_level, interactions_to_plateau

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "figures", "efficiency")
os.makedirs(OUT, exist_ok=True)

ATTACKS = ["voltage_manipulation", "current_injection", "power_disruption",
           "communication_spoofing", "data_injection", "protocol_manipulation"]
SHORT   = ["Vol.", "Cur.", "Pow.", "Spoof", "Data", "Prot."]

FLOOR = (sorted(glob.glob("ms_results/gemini_floor/reward_history_gemini_guided_seed0_*.json"))
         + sorted(glob.glob("ms_results/gemini_floor/reward_history_gemini_guided_floor_seed*.json")))
AUTO  = sorted(glob.glob("ms_results/autonomous/reward_history_autonomous_seed*.json"))

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.family": "serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

import json
def load(p): return json.load(open(p))

N = 120  # resample grid for the progress axis
def resample(y):
    ys = se.smooth(y)
    if ys.size < 3: return None
    x_old = np.linspace(0, 1, ys.size)
    return np.interp(np.linspace(0, 1, N), x_old, ys)

raw = {m: [load(p) for p in files] for m, files in (("floor", FLOOR), ("auto", AUTO))}

# ---- per-mode mean LEARNING curve: each attack curve self-normalized to its own
#      [start -> plateau] (0->1), so the convergence trajectory is visible and the
#      0.9 crossing matches interactions-to-plateau. Avg over attacks, then seeds. ----
def mode_curves(m):
    per_seed = []
    for d in raw[m]:
        attack_norm = []
        for a in ATTACKS:
            y, _ = se.attack_curve(d, a)
            ys = se.smooth(y)
            if ys.size < 5:
                continue
            s, p = se.start_level(ys), se.plateau(ys)
            if not np.isfinite(s) or not np.isfinite(p) or p <= s:
                continue                      # curve never learns; excluded (honest)
            z = np.clip((ys - s) / (p - s), -0.15, 1.15)
            x_old = np.linspace(0, 1, z.size)
            attack_norm.append(np.interp(np.linspace(0, 1, N), x_old, z))
        if attack_norm:
            per_seed.append(np.mean(attack_norm, axis=0))
    return np.array(per_seed)

cf, ca = mode_curves("floor"), mode_curves("auto")

def itp_by_attack(m):
    out = {}
    for a in ATTACKS:
        vals = []
        for d in raw[m]:
            y, _ = se.attack_curve(d, a)
            v = se.interactions_to_plateau(y)
            if v is not None: vals.append(v)
        out[a] = np.mean(vals) if vals else np.nan
    return out
itp_s, itp_a = itp_by_attack("floor"), itp_by_attack("auto")

# ---- single plot: parity learning curves ----
fig, ax = plt.subplots(figsize=(5.5, 4))
xg = np.linspace(0, 100, N)
C_S, C_A = "#2e6f95", "#b26b2e"
ax.axhline(0.9, color="gray", lw=1.0, ls=":")
ax.text(4, 0.92, "90% plateau", fontsize=12, color="gray")
for arr, c, lab in ((cf, C_S, "SPEAR (floor)"), (ca, C_A, "Autonomous")):
    mu, sem = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
    ax.plot(xg, mu, color=c, lw=2.0, label=lab)
    ax.fill_between(xg, mu - sem, mu + sem, color=c, alpha=0.18, lw=0)
ax.set_xlabel("training progress (% of SAC budget)")
ax.set_ylabel("normalized reward")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", framealpha=0.9)
ax.set_ylim(-0.05, 1.12)
ax.set_xlim(0, 100)

fig.tight_layout()
p = os.path.join(OUT, "sample_efficiency_parity.pdf")
fig.savefig(p, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
print("floor seeds:", len(cf), " auto seeds:", len(ca))

# ---- companion panel: per-attack interactions-to-plateau ----
figb, axb = plt.subplots(figsize=(5.5, 4))
x = np.arange(len(ATTACKS)); w = 0.38
axb.bar(x - w/2, [itp_s[a] for a in ATTACKS], w, color=C_S, label="SPEAR (floor)")
axb.bar(x + w/2, [itp_a[a] for a in ATTACKS], w, color=C_A, label="Autonomous")
axb.set_xticks(x); axb.set_xticklabels(SHORT)
axb.set_ylabel("interactions to plateau")
axb.set_xlabel("attack type")
axb.grid(True, axis="y", alpha=0.3)
axb.set_ylim(0, max([v for v in list(itp_s.values()) + list(itp_a.values())
                     if np.isfinite(v)]) * 1.28)
axb.legend(loc="upper center", ncol=1, framealpha=0.9)
figb.tight_layout()
pb = os.path.join(OUT, "parity_itp.pdf")
figb.savefig(pb, bbox_inches="tight"); plt.close(figb)
print("wrote", pb)
print("ITP floor:", {s: round(itp_s[a],1) for s,a in zip(SHORT,ATTACKS)})
print("ITP auto :", {s: round(itp_a[a],1) for s,a in zip(SHORT,ATTACKS)})
