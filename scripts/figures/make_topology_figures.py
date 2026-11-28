#!/usr/bin/env python3
# --- repo-root shim: relocated into scripts/<category>/ (keeps flat imports working) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end shim ---
"""
make_topology_figures.py — regenerate the workstream-C paper figures as clean
single-plot figures from the saved result JSONs.

  topology_retention.pdf  : per-seed physical impact under the topology shift
                            (flagship@T0, flagship@T1=0, retarget@T1).
  faithfulness_llm.pdf    : real-LLM target selection across T0/T1 conditions.

    python make_topology_figures.py
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "figures", "topology")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.family": "serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def latest(pat):
    fs = sorted(glob.glob(os.path.join(ROOT, "detection_results", pat)))
    return fs[-1] if fs else None

C_T0    = "#3b5b92"   # flagship nominal
C_BLIND = "#b23b3b"   # frozen RL after shift (dead)
C_AWARE = "#2e8b6b"   # topology-aware retarget

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — per-seed physical impact (single plot)
# ─────────────────────────────────────────────────────────────────────────────
tj = json.load(open(latest("topology_generalization_*.json")))
rows  = tj["rows"]
seeds = [r["seed"] for r in rows]
f_t0  = np.array([r["impact_flagship_T0"] for r in rows])
f_t1  = np.array([r["impact_flagship_T1"] for r in rows])
rt_t1 = np.array([r["impact_retarget_T1"] for r in rows])

fig, ax = plt.subplots(figsize=(5.5, 4))
x = np.arange(len(seeds)); w = 0.26
ax.bar(x - w, f_t0,  w, color=C_T0,    label=r"flagship 890 @ $T_0$")
ax.bar(x,      f_t1,  w, color=C_BLIND, label=r"flagship 890 @ $T_1$ (frozen RL)")
ax.bar(x + w, rt_t1, w, color=C_AWARE, label=r"retarget 848 @ $T_1$ (LLM)")
ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel(r"attack impact  $\overline{|\Delta V|}$")
ax.set_xlabel("random seed")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, max(f_t0) * 1.32)          # headroom so legend clears the bars
ax.legend(loc="upper center", ncol=1, framealpha=0.9)
p1 = os.path.join(OUT, "topology_retention.pdf")
fig.tight_layout(); fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
print("wrote", p1)

# companion panel: impact retained (blind vs aware), mean +/- SD
blind = np.array([r["blind_retention"] for r in rows])
aware = np.array([r["aware_retention"] for r in rows])
figr, axr = plt.subplots(figsize=(5.5, 4))
means = [blind.mean(), aware.mean()]; sds = [blind.std(ddof=1), aware.std(ddof=1)]
axr.bar([0, 1], means, 0.55, yerr=sds, capsize=6, color=[C_BLIND, C_AWARE],
        error_kw=dict(lw=1.2))
axr.set_xticks([0, 1]); axr.set_xticklabels(["frozen RL\n(blind)", "LLM\n(aware)"])
axr.set_ylabel("impact retained after shift")
axr.grid(True, axis="y", alpha=0.3)
axr.set_ylim(0, 0.52)
axr.text(0, 0.03, f"{means[0]:.3f}", ha="center", va="bottom", fontsize=13)
axr.text(1, means[1] + sds[1] + 0.015, f"{means[1]:.3f}", ha="center", va="bottom", fontsize=13)
axr.annotate(r"Cohen's $d=+12.9$", (0.5, 0.47), ha="center", fontsize=13)
axr.annotate("", xy=(1, 0.43), xytext=(0, 0.43), arrowprops=dict(arrowstyle="<->", lw=1.0))
pr = os.path.join(OUT, "retention_bar.pdf")
figr.tight_layout(); figr.savefig(pr, bbox_inches="tight"); plt.close(figr)
print("wrote", pr)

# companion panel: T0 per-station impact sweep (faithfulness)
T0_SWEEP = [("890", 2.2724), ("848", 1.5633), ("844", 1.1679), ("848$'$", 0.9242),
            ("860", 0.9067), ("840", 0.8661), ("830", 0.7770), ("830$'$", 0.7050),
            ("824", 0.5842), ("826", 0.0000)]
labs = [b for b, _ in T0_SWEEP][::-1]; vls = [v for _, v in T0_SWEEP][::-1]
cols = [C_BLIND if b == "890" else "#9bb0c9" for b in labs]
figt, axt = plt.subplots(figsize=(5.5, 4))
axt.barh(np.arange(len(labs)), vls, color=cols, height=0.74)
axt.set_yticks(np.arange(len(labs))); axt.set_yticklabels([f"bus {b}" for b in labs])
axt.set_xlabel(r"$T_0$ attack impact  $\overline{|\Delta V|}$")
axt.grid(True, axis="x", alpha=0.3); axt.set_xlim(0, 2.7)
axt.text(1.95, len(labs) / 2.0 - 0.5, "bus 890 = argmax\n($+45\\%$ over next)",
         ha="center", va="center", fontsize=12, color=C_BLIND,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BLIND, lw=1.3))
pt = os.path.join(OUT, "t0_sweep.pdf")
figt.tight_layout(); figt.savefig(pt, bbox_inches="tight"); plt.close(figt)
print("wrote", pt)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — real-LLM target selection (single plot)
# ─────────────────────────────────────────────────────────────────────────────
lj = json.load(open(latest("llm_topology_reasoning_*.json")))
res = lj["results"]
conds = [("T0", r"$T_0$ nominal"),
         ("T1a", r"$T_1$ telemetry"),
         ("T1b", r"$T_1$ event-only")]
order = ["890", "848", "830"]
bus_color = {"890": C_BLIND, "848": C_AWARE, "830": "#6fae8f"}
bus_label = {"890": "bus 890 (dead at $T_1$)", "848": "bus 848 (best live)",
             "830": "bus 830 (live)"}

fig2, ax2 = plt.subplots(figsize=(5.5, 4))
x = np.arange(len(conds)); w = 0.26
for j, b in enumerate(order):
    counts = [res[c[0]]["dist"].get(b, 0) for c in conds]
    ns     = [res[c[0]]["n"] for c in conds]
    frac   = [cc / nn if nn else 0 for cc, nn in zip(counts, ns)]
    ax2.bar(x + (j - 1) * w, frac, w, color=bus_color[b], label=bus_label[b])
ax2.set_xticks(x); ax2.set_xticklabels([c[1] for c in conds])
ax2.set_ylabel("fraction of LLM trials")
ax2.grid(True, axis="y", alpha=0.3)
ax2.set_ylim(0, 1.28)                      # headroom so legend clears the full bars
ax2.legend(loc="upper center", ncol=1, framealpha=0.9)
ax2.text(0.0, 0.52, "frozen RL: always 890", fontsize=12, color=C_BLIND,
         rotation=90, va="center", ha="center")
p2 = os.path.join(OUT, "faithfulness_llm.pdf")
fig2.tight_layout(); fig2.savefig(p2, bbox_inches="tight"); plt.close(fig2)
print("wrote", p2)
