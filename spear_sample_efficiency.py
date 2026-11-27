#!/usr/bin/env python3
"""
spear_sample_efficiency.py — the SPEAR headline evidence.

The paper's defensible claim (per the rebuttal) is NOT uniform mean-reward
dominance.  It is **sample efficiency**: with the LLM pruning the RL search space
and supplying a domain prior, SPEAR (gemini_guided) reaches a given performance
level in FEWER environment interactions than the autonomous UCB baseline, which
must pay a cold-start coverage cost (visit every attack×system pair) before it
can exploit.

This script produces the committed figure ("reward-vs-interactions curves") and
the scalar sample-efficiency metrics, from the per-circle trajectories already
saved in reward_history_*.json by central_rl_coordinator.py:

  inner_episode_rewards[attack][circle] = {
      'sac_rewards': [...per-episode shaped reward...],
      'best_impact', 'success_rate', 'avg_detection_risk',
      'cum_timesteps', 'reduced_action', ...
  }

Metrics computed per attack type, per mode:
  * interactions_to_plateau : SAC episodes to first reach 90% of the run's
                              plateau (plateau = mean of the last 20% of the
                              learning curve). Lower = more sample efficient.
  * auc_norm                : area under the min-max-normalized learning curve
                              (shared scale across both modes per attack). Higher
                              = rises to high reward sooner.
  * plateau                 : asymptotic mean reward (final 20%).
  * impact_x_stealth        : clean physical objective (best_impact * (1-det))
                              averaged over circles — guidance-free, from co-sim.

Two run modes
-------------
1) Single pair:   python spear_sample_efficiency.py gem.json auto.json
2) Auto / multi-seed (default): discovers ALL reward_history_gemini_guided_*.json
   and reward_history_autonomous_*.json, groups by seed, aggregates per mode, and
   (with >=2 seeds per mode) reports a pooled-SD Cohen's d on each scalar metric.

Outputs: console tables + sub_figures/spear_sample_efficiency_<ts>.{png,pdf}.

NOTE on reading files: result JSONs under Dropbox CloudStorage may be
"online-only" placeholders; if a read times out, make them available offline
(or regenerate the run locally) before running this.
"""
from __future__ import annotations
import json
import os
import sys
import glob
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

BASE = os.path.dirname(os.path.abspath(__file__))

ATTACKS = ["voltage_manipulation", "current_injection", "power_disruption",
           "communication_spoofing", "data_injection", "protocol_manipulation"]

# Attacks where domain semantics (protocol/voltage/cascading-load) matter most —
# the rebuttal scopes SPEAR's per-vector edge to these.
SEMANTIC_ATTACKS = {"voltage_manipulation", "power_disruption", "protocol_manipulation"}


# ----------------------------------------------------------------------------- I/O
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _discover(mode: str) -> List[str]:
    cands = (glob.glob(os.path.join(BASE, f"reward_history_{mode}.json")) +
             glob.glob(os.path.join(BASE, f"reward_history_{mode}_*.json")))
    return sorted(c for c in cands if "assisted" not in os.path.basename(c))


def _seed_of(data: dict, path: str) -> str:
    s = data.get("seed")
    if s is not None:
        return str(s)
    b = os.path.basename(path)
    if "seed" in b:
        try:
            return b.split("seed", 1)[1].split("_", 1)[0]
        except Exception:
            pass
    return b  # fall back to filename as the group key


# --------------------------------------------------------------------- curve build
def attack_curve(data: dict, attack: str) -> Tuple[np.ndarray, List[int]]:
    """Concatenate an attack's SAC episode rewards across circles, chronologically.
    Returns (rewards_array, circle_boundary_indices)."""
    inner = data.get("inner_episode_rewards", {}).get(attack, {})
    # circle keys are strings of ints; sort numerically
    circ_keys = sorted(inner.keys(), key=lambda k: int(k))
    rewards: List[float] = []
    boundaries: List[int] = []
    for k in circ_keys:
        r = inner[k].get("sac_rewards", []) or []
        if r:
            boundaries.append(len(rewards))
            rewards.extend(float(x) for x in r)
    return np.asarray(rewards, dtype=float), boundaries


def attack_clean_series(data: dict, attack: str, field: str) -> np.ndarray:
    inner = data.get("inner_episode_rewards", {}).get(attack, {})
    circ_keys = sorted(inner.keys(), key=lambda k: int(k))
    vals = [inner[k].get(field) for k in circ_keys if inner[k].get(field) is not None]
    return np.asarray(vals, dtype=float) if vals else np.asarray([])


def smooth(y: np.ndarray, frac: float = 0.1) -> np.ndarray:
    if y.size < 3:
        return y
    w = max(1, int(y.size * frac))
    return np.convolve(y, np.ones(w) / w, mode="valid")


def plateau(y: np.ndarray, tail: float = 0.2) -> float:
    if y.size == 0:
        return float("nan")
    k = max(1, int(y.size * tail))
    return float(np.mean(y[-k:]))


def start_level(y: np.ndarray, head: float = 0.1) -> float:
    if y.size == 0:
        return float("nan")
    k = max(1, int(y.size * head))
    return float(np.mean(y[:k]))


def interactions_to_plateau(y: np.ndarray, frac: float = 0.9) -> Optional[int]:
    """Episodes until smoothed reward first reaches `frac` of the way from the
    start level to the plateau. None if the curve never learns (plateau<=start)
    or is too short."""
    if y.size < 5:
        return None
    sm = smooth(y)
    s = start_level(sm)
    p = plateau(sm)
    if not np.isfinite(s) or not np.isfinite(p) or p <= s:
        return None
    target = s + frac * (p - s)
    hits = np.where(sm >= target)[0]
    if hits.size == 0:
        return None
    # offset by the smoothing window so the index maps back to episode count
    return int(hits[0] + (y.size - sm.size))


def auc_norm(y: np.ndarray, lo: float, hi: float) -> float:
    """Mean of the curve normalized to a shared [lo,hi] scale → in [0,1].
    Higher = spends more of training near the top of the shared range (faster)."""
    if y.size == 0 or hi <= lo:
        return float("nan")
    z = np.clip((smooth(y) - lo) / (hi - lo), 0.0, 1.0)
    return float(np.mean(z))


# --------------------------------------------------------------------- aggregation
def cohen_d_pooled(a: List[float], b: List[float]) -> Optional[float]:
    """Cohen's d with the standard pooled SD convention:
        sp = sqrt(((na-1)sa^2 + (nb-1)sb^2) / (na+nb-2)),  d = (mean_a-mean_b)/sp
    Requires >=2 finite values in each group."""
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def welch_p(a: List[float], b: List[float]) -> Optional[float]:
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if not HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return None
    return float(_scipy_stats.ttest_ind(a, b, equal_var=False)[1])


def _fmt(x: float, nd: int = 2) -> str:
    return "   n/a" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


# --------------------------------------------------------------------- main compare
def compute_mode_metrics(files: List[str]) -> Dict:
    """For a list of same-mode run files (one per seed), compute per-attack scalar
    metric lists across seeds, plus the mean learning curve per attack for plotting."""
    per_attack = {a: {"itp": [], "auc": [], "plateau": [], "ixs": [],
                      "curves": []} for a in ATTACKS}
    seeds = []
    datas = []
    for p in files:
        try:
            d = load(p)
        except Exception as e:
            print(f"  ##?? could not read {os.path.basename(p)}: {e}")
            continue
        seeds.append(_seed_of(d, p))
        datas.append(d)

    # shared per-attack [lo,hi] scale across ALL files+modes is applied later;
    # here just collect raw curves and defer AUC. We collect curves now.
    for d in datas:
        for a in ATTACKS:
            y, _ = attack_curve(d, a)
            imp = attack_clean_series(d, a, "best_impact")
            det = attack_clean_series(d, a, "avg_detection_risk")
            if imp.size and det.size:
                n = min(imp.size, det.size)
                ixs = float(np.mean(imp[:n] * (1.0 - det[:n])))
            else:
                ixs = float("nan")
            per_attack[a]["curves"].append(y)
            per_attack[a]["itp"].append(interactions_to_plateau(y))
            per_attack[a]["plateau"].append(plateau(y))
            per_attack[a]["ixs"].append(ixs)
    return {"per_attack": per_attack, "seeds": seeds, "datas": datas}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) >= 2:
        gem_files, aut_files = [args[0]], [args[1]]
    else:
        gem_files = _discover("gemini_guided")
        aut_files = _discover("autonomous")

    if not gem_files:
        print("ERROR: no reward_history_gemini_guided_*.json found.")
        return
    if not aut_files:
        print("ERROR: no reward_history_autonomous_*.json found.")
        print("  Generate a matched autonomous baseline:")
        print("    FORCE_AUTONOMOUS_RL=1 SPEAR_SEED=0 python enhanced_integrated_evcs_system.py")
        return

    print("=" * 100)
    print("SPEAR SAMPLE-EFFICIENCY — reward vs interactions (gemini_guided vs autonomous)")
    print("=" * 100)
    print(f"  gemini files ({len(gem_files)} seed(s)): {[os.path.basename(x) for x in gem_files]}")
    print(f"  auto   files ({len(aut_files)} seed(s)): {[os.path.basename(x) for x in aut_files]}")
    if not HAVE_SCIPY:
        print("  [scipy missing → p-values omitted; Cohen's d still reported]")

    gem = compute_mode_metrics(gem_files)
    aut = compute_mode_metrics(aut_files)

    # shared per-attack [lo,hi] scale across both modes → comparable AUC
    for a in ATTACKS:
        allvals = []
        for m in (gem, aut):
            for y in m["per_attack"][a]["curves"]:
                if y.size:
                    allvals.append(smooth(y))
        if allvals:
            cat = np.concatenate(allvals)
            lo, hi = float(np.min(cat)), float(np.max(cat))
        else:
            lo, hi = 0.0, 1.0
        for m in (gem, aut):
            m["per_attack"][a]["auc"] = [auc_norm(y, lo, hi)
                                         for y in m["per_attack"][a]["curves"]]
        gem["per_attack"][a]["_scale"] = (lo, hi)

    # ---------------- per-attack table ----------------
    print("\n" + "-" * 100)
    print("PER-ATTACK sample efficiency  (ITP=interactions-to-90%-plateau, lower=better; "
          "AUC higher=better)")
    print("-" * 100)
    hdr = (f"{'Attack':<22} {'ITP_gem':>8} {'ITP_aut':>8} {'AUC_gem':>8} {'AUC_aut':>8} "
           f"{'Plt_gem':>8} {'Plt_aut':>8} {'IxS_gem':>8} {'IxS_aut':>8}  winner(sample-eff)")
    print(hdr)

    def _mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    for a in ATTACKS:
        g, u = gem["per_attack"][a], aut["per_attack"][a]
        itp_g, itp_u = _mean(g["itp"]), _mean(u["itp"])
        auc_g, auc_u = _mean(g["auc"]), _mean(u["auc"])
        plt_g, plt_u = _mean(g["plateau"]), _mean(u["plateau"])
        ixs_g, ixs_u = _mean(g["ixs"]), _mean(u["ixs"])
        # winner by sample efficiency: fewer interactions OR (if ITP n/a) higher AUC
        if np.isfinite(itp_g) and np.isfinite(itp_u):
            win = "GEMINI" if itp_g < itp_u else "auto"
        elif np.isfinite(auc_g) and np.isfinite(auc_u):
            win = "GEMINI" if auc_g > auc_u else "auto"
        else:
            win = "  n/a"
        tag = " *" if a in SEMANTIC_ATTACKS else "  "
        print(f"{a+tag:<22} {_fmt(itp_g,0):>8} {_fmt(itp_u,0):>8} {_fmt(auc_g,3):>8} "
              f"{_fmt(auc_u,3):>8} {_fmt(plt_g,1):>8} {_fmt(plt_u,1):>8} "
              f"{_fmt(ixs_g,3):>8} {_fmt(ixs_u,3):>8}  {win}")
    print("  (* = semantic attack where SPEAR's domain-prior edge is expected)")

    # ---------------- cross-seed Cohen's d on scalar metrics ----------------
    print("\n" + "-" * 100)
    print("AGGREGATE (pooled across attacks & seeds) with pooled-SD Cohen's d")
    print("  d sign: for ITP, NEGATIVE d (gemini<auto) = gemini more sample-efficient;")
    print("          for AUC / plateau / IxS, POSITIVE d (gemini>auto) favours gemini.")
    print("-" * 100)

    def _pool(mode_obj, key):
        out = []
        for a in ATTACKS:
            out.extend(x for x in mode_obj["per_attack"][a][key]
                       if x is not None and np.isfinite(x))
        return out

    for key, label, gem_better_if_positive in [
        ("itp", "Interactions-to-plateau (lower=better)", False),
        ("auc", "Normalized AUC (higher=better)", True),
        ("plateau", "Reward plateau (higher=better)", True),
        ("ixs", "Impact x Stealth (higher=better)", True),
    ]:
        gv, uv = _pool(gem, key), _pool(aut, key)
        mg = float(np.mean(gv)) if gv else float("nan")
        mu = float(np.mean(uv)) if uv else float("nan")
        d = cohen_d_pooled(gv, uv)
        p = welch_p(gv, uv)
        if d is None:
            verdict = "n/a"
        else:
            favours_gem = (d > 0) if gem_better_if_positive else (d < 0)
            verdict = "GEMINI" if favours_gem else "auto"
        print(f"  {label:<42} gem={_fmt(mg,3):>8}  aut={_fmt(mu,3):>8}  "
              f"d={_fmt(d,2):>6}  p={_fmt(p,4) if p is not None else '  n/a':>6}  → {verdict}")

    # ---------------- figure ----------------
    if HAVE_MPL:
        _plot(gem, aut)

    print("\n" + "=" * 100)
    print("Interpretation:")
    print("  - The honest SPEAR claim is SAMPLE EFFICIENCY: lower ITP / higher early AUC,")
    print("    concentrated on the semantic attacks (*). It need NOT win the reward plateau.")
    print("  - Report Cohen's d across SEEDS (run >=3 seeds per mode via spear_multiseed_runner.py)")
    print("    so the samples are independent, per reviewer #109A.")
    print("=" * 100)


def _plot(gem: Dict, aut: Dict):
    outdir = os.path.join(BASE, "sub_figures")
    os.makedirs(outdir, exist_ok=True)
    n = len(ATTACKS)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)

    def mean_curve(mode_obj, a):
        curves = [smooth(y) for y in mode_obj["per_attack"][a]["curves"] if y.size]
        if not curves:
            return None
        L = min(len(c) for c in curves)
        if L < 2:
            return None
        arr = np.vstack([c[:L] for c in curves])
        return np.mean(arr, axis=0), (np.std(arr, axis=0) if arr.shape[0] > 1 else None)

    for i, a in enumerate(ATTACKS):
        ax = axes[i // ncol][i % ncol]
        for mode_obj, color, lab in [(gem, "#2563eb", "SPEAR (gemini)"),
                                     (aut, "#dc2626", "autonomous")]:
            mc = mean_curve(mode_obj, a)
            if mc is None:
                continue
            mean, sd = mc
            x = np.arange(mean.size)
            ax.plot(x, mean, color=color, lw=1.8, label=lab)
            if sd is not None:
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
            itp = interactions_to_plateau(
                np.concatenate(mode_obj["per_attack"][a]["curves"])
                if mode_obj["per_attack"][a]["curves"] else np.asarray([]))
        title = a + (" *" if a in SEMANTIC_ATTACKS else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("SAC episodes (interactions)", fontsize=8)
        ax.set_ylabel("smoothed reward", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
    # hide unused axes
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("SPEAR vs Autonomous — reward vs interactions (mean ± SD across seeds)\n"
                 "* = semantic attacks (voltage / power / protocol)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ts = time.strftime("%Y%m%d_%H%M%S")
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"spear_sample_efficiency_{ts}.{ext}")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"\n  📈 saved figure: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
