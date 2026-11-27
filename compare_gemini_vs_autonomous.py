"""
compare_gemini_vs_autonomous.py — CORRECT gemini-guided vs autonomous comparison.

Why this replaces compute_stats.py / compare_significance*.py / generate_*.py:
  Those scripts loaded TWO `reward_history_gemini_guided_*.json` files and labelled
  one "autonomous"/"SPEAR" — i.e. they compared gemini-vs-gemini (run-to-run noise),
  never against a real autonomous run.  They also compared on the raw shaped training
  reward, which is (a) bimodal by attack type (~10^5 for evadable attacks, ~0 for the
  two that can't evade), (b) dominated by per-episode mission/streak accumulation with
  enormous variance, and (c) polluted by a guidance-alignment bonus that differs
  between the two modes.  None of that measures attack QUALITY.

What this script does instead:
  1. Auto-selects the newest real `reward_history_gemini_guided_*.json` and the newest
     `reward_history_autonomous_*.json` (override on the command line).
  2. Refuses to silently compare mismatched eras — warns loudly if outer-episode counts
     differ (a 30-circle 10-system run is not comparable to an 18-circle 6-system run).
  3. Compares on CLEAN, guidance-free PHYSICAL outcomes stored per outer circle in
     `outer_episode_details`:
        - impact          (attack effectiveness; higher = stronger)
        - success_rate    (fraction of attacks that succeeded)
        - stealth         = 1 - detection_risk  (higher = harder for IDS to catch)
        - impact x stealth (the paper's real objective: strong AND stealthy)
     These come from the co-simulation, NOT the shaped reward, so the guidance bonus
     cannot confound them.
  4. Still prints the raw shaped reward, but clearly flagged as a confounded proxy.

Usage:
    python compare_gemini_vs_autonomous.py
    python compare_gemini_vs_autonomous.py <gemini_file.json> <autonomous_file.json>
"""
import json
import os
import sys
import glob
import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

BASE = os.path.dirname(os.path.abspath(__file__))


def _newest(mode: str):
    """Newest reward_history_<mode>[_<timestamp>].json, or None."""
    cands = glob.glob(os.path.join(BASE, f"reward_history_{mode}.json")) + \
            glob.glob(os.path.join(BASE, f"reward_history_{mode}_*.json"))
    # exclude gemini_assisted when asking for gemini_guided prefix-match
    cands = [c for c in cands if "assisted" not in os.path.basename(c)]
    return max(cands, key=os.path.getmtime) if cands else None


def load(path):
    with open(path) as f:
        return json.load(f)


def circle_metric(data, key):
    """Per-circle physical metric array from outer_episode_details."""
    oed = data.get("outer_episode_details", [])
    return np.array([d.get(key, np.nan) for d in oed], dtype=float)


def welch(a, b):
    """Return (mean_a, mean_b, diff, p, cohen_d). p is None without scipy."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.mean(a) if len(a) else 0.0, np.mean(b) if len(b) else 0.0, 0.0, None, 0.0
    ma, mb = float(np.mean(a)), float(np.mean(b))
    diff = ma - mb
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = diff / pooled if pooled > 0 else 0.0
    p = float(stats.ttest_ind(a, b, equal_var=False)[1]) if HAVE_SCIPY else None
    return ma, mb, diff, p, d


def sig(p):
    if p is None:
        return "  ?"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def main():
    if len(sys.argv) >= 3:
        gem_path, aut_path = sys.argv[1], sys.argv[2]
    else:
        gem_path = _newest("gemini_guided")
        aut_path = _newest("autonomous")

    if not gem_path or not os.path.exists(gem_path):
        print("ERROR: no gemini_guided reward-history file found.")
        return
    if not aut_path or not os.path.exists(aut_path):
        print("ERROR: no autonomous reward-history file found.")
        print("  Generate one on the CURRENT code with:")
        print("    FORCE_AUTONOMOUS_RL=1 python enhanced_integrated_evcs_system.py")
        return

    gem, aut = load(gem_path), load(aut_path)

    print("=" * 100)
    print("GEMINI-GUIDED vs AUTONOMOUS — clean physical-outcome comparison")
    print("=" * 100)
    print(f"  Gemini file    : {os.path.basename(gem_path)}  (mode={gem.get('mode')})")
    print(f"  Autonomous file: {os.path.basename(aut_path)}  (mode={aut.get('mode')})")
    if not HAVE_SCIPY:
        print("  [scipy not found → p-values omitted; means/effect sizes still valid]")

    # --- era-mismatch guard ---
    ge, ae = gem.get("outer_episodes"), aut.get("outer_episodes")
    if gem.get("mode") != "gemini_guided":
        print(f"\n  ##  WARNING: 'gemini' file mode is {gem.get('mode')!r}, not 'gemini_guided'.")
    if aut.get("mode") != "autonomous":
        print(f"\n  ##  WARNING: 'autonomous' file mode is {aut.get('mode')!r}, not 'autonomous'."
              f"  You are NOT comparing against a real autonomous run.")
    if ge != ae:
        print(f"\n  ##  WARNING: circle counts differ (gemini={ge}, autonomous={ae}).")
        print("      These are likely different eras (e.g. 30-circle 10-system vs 18-circle")
        print("      6-system) and are NOT directly comparable. Regenerate a matched autonomous")
        print("      baseline: FORCE_AUTONOMOUS_RL=1 python enhanced_integrated_evcs_system.py")

    # --- clean physical metrics (guidance-free) ---
    print("\n" + "-" * 100)
    print("CLEAN metrics (per outer circle, from co-simulation — NOT the shaped reward)")
    print("-" * 100)
    print(f"{'Metric':<26} {'Gemini':>12} {'Autonomous':>12} {'Diff':>12} {'p':>8} {'sig':>5} {'d':>7}  winner")
    metrics = [
        ("aggregate_impact",       "Impact (higher=better)",        False),
        ("aggregate_success_rate", "Success rate",                  False),
        ("aggregate_detection_risk", "Detection risk (lower=bttr)", True),
    ]
    g_imp = circle_metric(gem, "aggregate_impact")
    a_imp = circle_metric(aut, "aggregate_impact")
    g_det = circle_metric(gem, "aggregate_detection_risk")
    a_det = circle_metric(aut, "aggregate_detection_risk")

    for key, label, lower_better in metrics:
        g = circle_metric(gem, key)
        a = circle_metric(aut, key)
        ma, mb, diff, p, d = welch(g, a)
        better_gem = (ma < mb) if lower_better else (ma > mb)
        winner = "GEMINI" if better_gem else "auto"
        print(f"{label:<26} {ma:>12.3f} {mb:>12.3f} {diff:>+12.3f} "
              f"{('%.4f' % p) if p is not None else '  n/a':>8} {sig(p):>5} {d:>+7.2f}  {winner}")

    # --- composite objective: impact x stealth (strong AND stealthy) ---
    g_q = g_imp * (1.0 - g_det)
    a_q = a_imp * (1.0 - a_det)
    ma, mb, diff, p, d = welch(g_q, a_q)
    winner = "GEMINI" if ma > mb else "auto"
    print(f"{'Impact x Stealth (OBJ)':<26} {ma:>12.3f} {mb:>12.3f} {diff:>+12.3f} "
          f"{('%.4f' % p) if p is not None else '  n/a':>8} {sig(p):>5} {d:>+7.2f}  {winner}")

    print("\n  => This 'Impact x Stealth' row is the hypothesis test: does LLM guidance")
    print("     yield attacks that are simultaneously more impactful AND more stealthy?")

    # --- raw shaped reward, flagged as confounded ---
    print("\n" + "-" * 100)
    print("Shaped training reward (CONFOUNDED — for reference only)")
    print("  Dominated by mission/streak accumulation, bimodal by attack type (~1e5 vs ~0),")
    print("  and includes a guidance-alignment bonus. Do NOT use as a quality measure.")
    print("-" * 100)
    g_out = np.array(gem.get("outer_episode_rewards", []), dtype=float)
    a_out = np.array(aut.get("outer_episode_rewards", []), dtype=float)
    ma, mb, diff, p, d = welch(g_out, a_out)
    print(f"{'Outer reward (raw)':<26} {ma:>12.0f} {mb:>12.0f} {diff:>+12.0f} "
          f"{('%.4f' % p) if p is not None else '  n/a':>8} {sig(p):>5} {d:>+7.2f}")

    # --- per-attack CLEAN breakdown (needs per-attack impact/detection in the file) ---
    # Older runs only stored per-attack reward; newer runs (after the measurement fix)
    # also store best_impact / success_rate / avg_detection_risk per attack per circle.
    ATTACKS = ["voltage_manipulation", "current_injection", "power_disruption",
               "communication_spoofing", "data_injection", "protocol_manipulation"]

    def per_attack_clean(data, attack, field):
        inner = data.get("inner_episode_rewards", {}).get(attack, {})
        vals = [c.get(field) for c in inner.values() if c.get(field) is not None]
        return np.array(vals, dtype=float) if vals else np.array([])

    gem_has = any(per_attack_clean(gem, a, "best_impact").size for a in ATTACKS)
    aut_has = any(per_attack_clean(aut, a, "best_impact").size for a in ATTACKS)
    if gem_has and aut_has:
        print("\n" + "-" * 100)
        print("PER-ATTACK clean quality (best_impact / success / stealth) — the bimodal picture")
        print("-" * 100)
        print(f"{'Attack':<22} {'Gem impact':>11} {'Aut impact':>11} {'Gem succ':>9} {'Aut succ':>9} "
              f"{'Gem stlth':>10} {'Aut stlth':>10}")
        for a in ATTACKS:
            gi, ai = per_attack_clean(gem, a, "best_impact"), per_attack_clean(aut, a, "best_impact")
            gs, as_ = per_attack_clean(gem, a, "success_rate"), per_attack_clean(aut, a, "success_rate")
            gd, ad = per_attack_clean(gem, a, "avg_detection_risk"), per_attack_clean(aut, a, "avg_detection_risk")
            def m(x):
                return float(np.mean(x)) if x.size else float("nan")
            print(f"{a:<22} {m(gi):>11.3f} {m(ai):>11.3f} {m(gs):>9.2f} {m(as_):>9.2f} "
                  f"{1-m(gd):>10.2f} {1-m(ad):>10.2f}")
        print("\n  => Attacks stuck at impact~0 / success~0 (current_injection, comm_spoofing)")
        print("     cannot evade regardless of mode — that ceiling is the real limiter, not Gemini.")
    else:
        print("\n  [per-attack clean metrics not in these files — regenerate runs after the")
        print("   measurement fix to populate best_impact/success/detection per attack]")

    print("\n" + "=" * 100)
    print("Interpretation guide:")
    print("  - Judge the hypothesis on 'Impact x Stealth' and the clean metrics, not raw reward.")
    print("  - 'ns' with |d| < 0.2 => gemini and autonomous are statistically indistinguishable.")
    print("  - A real Gemini benefit shows up as GEMINI winning Impact x Stealth with p<0.05.")
    print("  ##  CAVEAT: the p / d here come from CIRCLES WITHIN ONE run, which are")
    print("     correlated (shared replay buffer) — NOT independent samples. For a")
    print("     referee-proof effect size, run several seeds and use the ACROSS-SEED,")
    print("     pooled-SD Cohen's d from spear_sample_efficiency.py:")
    print("       python spear_multiseed_runner.py --seeds 0 1 2 3 4")
    print("       python spear_sample_efficiency.py")
    print("  - SPEAR's committed claim is SAMPLE EFFICIENCY + per-vector semantic impact,")
    print("    NOT a uniform mean-reward win — see spear_sample_efficiency.py for that.")
    print("=" * 100)


if __name__ == "__main__":
    main()
