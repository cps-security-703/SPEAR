import json
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

spear_path = os.path.join(BASE, "reward_history_gemini_guided_20260815_001640.json")
auto_path  = os.path.join(BASE, "reward_history_autonomous_20260816_024035.json")


with open(spear_path) as f:
    spear = json.load(f)
with open(auto_path) as f:
    auto = json.load(f)

attack_types = [
    "voltage_manipulation",
    "current_injection",
    "power_disruption",
    "communication_spoofing",
    "data_injection",
    "protocol_manipulation",
]
short = {
    "voltage_manipulation": "Vol. Manip.",
    "current_injection": "Current Inj.",
    "power_disruption": "Power Disr.",
    "communication_spoofing": "Com. Spoof.",
    "data_injection": "Data Inj.",
    "protocol_manipulation": "Prot. Manip.",
}


def get_circle_means(data, agent_key, atk):

    inner = data.get("inner_episode_rewards", {})
    atk_data = inner.get(atk, {})
    circle_keys = sorted(atk_data.keys(), key=lambda x: int(x))
    means = []
    for ck in circle_keys:
        rews = atk_data[ck].get(agent_key, [])
        if not isinstance(rews, list):
            rews = [rews]
        if len(rews) == 0:
            continue
        means.append(np.mean(rews))
    return np.array(means)


def compute_learning_metrics(circle_means):

    n = len(circle_means)
    if n == 0:
        return {}

    peak = float(np.max(circle_means))
    mean = float(np.mean(circle_means))
    sigma = float(np.std(circle_means, ddof=1)) if n > 1 else 0.0


    threshold_95 = 0.95 * peak
    circles_at_95 = np.where(circle_means >= threshold_95)[0]
    time_to_95 = int(circles_at_95[0]) + 1 if len(circles_at_95) > 0 else n


    auc = float(np.trapezoid(circle_means))


    cv = sigma / abs(mean) if abs(mean) > 1e-10 else float('inf')


    early_n = max(1, int(np.ceil(0.3 * n)))
    early_mean = float(np.mean(circle_means[:early_n]))


    late_n = max(1, int(np.ceil(0.3 * n)))
    late_mean = float(np.mean(circle_means[-late_n:]))


    if abs(early_mean) > 1e-10:
        ee_ratio = late_mean / early_mean
    else:
        ee_ratio = float('inf')


    threshold_90 = 0.90 * peak
    consistency = float(np.mean(circle_means >= threshold_90) * 100)


    if n > 1:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, circle_means, 1)
        slope = float(slope)
    else:
        slope = 0.0


    if n > 2:
        preds = slope * np.arange(n) + (circle_means[0] - slope * 0) if n > 1 else circle_means
        ss_res = np.sum((circle_means - preds) ** 2)
        ss_tot = np.sum((circle_means - np.mean(circle_means)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared = float(r_squared)
    else:
        r_squared = 0.0


    if n > 1:
        diffs = np.diff(circle_means)
        max_jump = float(np.max(diffs))
        max_drop = float(np.min(diffs))
    else:
        max_jump = 0.0
        max_drop = 0.0

    return {
        'peak': peak,
        'mean': mean,
        'sigma': sigma,
        'cv': cv,
        'time_to_95': time_to_95,
        'n_circles': n,
        'auc': auc,
        'early_mean': early_mean,
        'late_mean': late_mean,
        'ee_ratio': ee_ratio,
        'consistency_pct': consistency,
        'slope': slope,
        'r_squared': r_squared,
        'max_jump': max_jump,
        'max_drop': max_drop,
    }


results = {}

for atk in attack_types:
    results[atk] = {}
    for agent_label, agent_key in [("SAC", "sac_rewards"), ("DQN", "dqn_rewards")]:
        s_means = get_circle_means(spear, agent_key, atk)
        a_means = get_circle_means(auto, agent_key, atk)

        s_metrics = compute_learning_metrics(s_means)
        a_metrics = compute_learning_metrics(a_means)

        results[atk][agent_label] = {
            'spear': s_metrics,
            'auto': a_metrics,
        }


def print_metric_table(metric_key, title, fmt=".1f", higher_is_better=True):

    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(f"{'Attack':<18} | {'SAC-SPEAR':>12} {'SAC-Auto':>12} {'SAC-Delta':>10} | {'DQN-SPEAR':>12} {'DQN-Auto':>12} {'DQN-Delta':>10}")
    print("-" * 100)

    sac_s_vals = []
    sac_a_vals = []
    dqn_s_vals = []
    dqn_a_vals = []

    for atk in attack_types:
        s_s = results[atk]["SAC"]["spear"].get(metric_key, 0)
        s_a = results[atk]["SAC"]["auto"].get(metric_key, 0)
        d_s = results[atk]["DQN"]["spear"].get(metric_key, 0)
        d_a = results[atk]["DQN"]["auto"].get(metric_key, 0)

        sac_s_vals.append(s_s)
        sac_a_vals.append(s_a)
        dqn_s_vals.append(d_s)
        dqn_a_vals.append(d_a)


        sac_winner = "S" if (s_s > s_a if higher_is_better else s_s < s_a) else "A"
        dqn_winner = "S" if (d_s > d_a if higher_is_better else d_s < d_a) else "A"

        if isinstance(s_s, float) and abs(s_s) > 1e6:
            fmt_use = ".0f"
        else:
            fmt_use = fmt

        sac_delta = s_s - s_a
        dqn_delta = d_s - d_a

        print(f"{short[atk]:<18} | {s_s:>12{fmt_use}} {s_a:>12{fmt_use}} {sac_delta:>+10{fmt_use}} | {d_s:>12{fmt_use}} {d_a:>12{fmt_use}} {dqn_delta:>+10{fmt_use}}")


    sac_s_overall = np.mean(sac_s_vals)
    sac_a_overall = np.mean(sac_a_vals)
    dqn_s_overall = np.mean(dqn_s_vals)
    dqn_a_overall = np.mean(dqn_a_vals)
    print("-" * 100)
    print(f"{'Overall':<18} | {sac_s_overall:>12{fmt}} {sac_a_overall:>12{fmt}} {sac_s_overall-sac_a_overall:>+10{fmt}} | {dqn_s_overall:>12{fmt}} {dqn_a_overall:>12{fmt}} {dqn_s_overall-dqn_a_overall:>+10{fmt}}")


print("=" * 100)
print("LEARNING & EXPLORATION METRICS: SPEAR vs Autonomous")
print("S = SPEAR wins, A = Auto wins for each metric")
print("=" * 100)


print_metric_table('peak', "1. Peak Reward (max circle mean) - higher = better", ".0f", True)


print_metric_table('time_to_95', "2. Time to 95% Peak (circle #) - lower = faster convergence", ".0f", False)


print_metric_table('auc', "3. AUC (area under circle-mean curve) - higher = more total learning", ".0f", True)


print_metric_table('cv', "4. Coefficient of Variation (sigma/|mean|) - lower = more consistent", ".3f", False)


print_metric_table('early_mean', "5. Early-Phase Mean (first 30% circles) - higher = better initial exploration", ".0f", True)


print_metric_table('late_mean', "6. Late-Phase Mean (last 30% circles) - higher = better exploitation", ".0f", True)


print_metric_table('ee_ratio', "7. Exploration-to-Exploitation Ratio (late/early) - >1 = improved over time", ".2f", True)


print_metric_table('consistency_pct', "8. Consistency Score (% circles within 90% of peak) - higher = more stable", ".1f", True)


print_metric_table('slope', "9. Learning Slope (Delta/circle via linear regression) - higher = faster improvement", ".1f", True)


print_metric_table('r_squared', "10. R-squared of Linear Fit - higher = more predictable/linear learning", ".3f", True)


print_metric_table('max_jump', "11. Max Single-Circle Improvement - higher = bigger breakthrough", ".0f", True)


print_metric_table('max_drop', "12. Max Single-Circle Drop - less negative = more stable (fewer collapses)", ".0f", False)


print(f"\n{'='*100}")
print("SUMMARY: Win counts (SPEAR vs Auto) per metric")
print(f"{'='*100}")
print(f"{'Metric':<45} | {'SAC':>8} {'S/A':>6} | {'DQN':>8} {'S/A':>6}")
print("-" * 100)

metric_info = [
    ('peak', 'Peak Reward', True),
    ('time_to_95', 'Time to 95% Peak (lower=better)', False),
    ('auc', 'AUC (total learning)', True),
    ('cv', 'Coef. of Variation (lower=better)', False),
    ('early_mean', 'Early-Phase Mean (exploration)', True),
    ('late_mean', 'Late-Phase Mean (exploitation)', True),
    ('ee_ratio', 'E-to-E Ratio (improvement factor)', True),
    ('consistency_pct', 'Consistency Score', True),
    ('slope', 'Learning Slope', True),
    ('r_squared', 'R-squared (linearity of learning)', True),
    ('max_jump', 'Max Breakthrough Jump', True),
    ('max_drop', 'Max Drop (fewer collapses)', False),
]

for mk, mlabel, hib in metric_info:
    sac_s_wins = 0
    sac_a_wins = 0
    dqn_s_wins = 0
    dqn_a_wins = 0
    for atk in attack_types:
        s_s = results[atk]["SAC"]["spear"].get(mk, 0)
        s_a = results[atk]["SAC"]["auto"].get(mk, 0)
        d_s = results[atk]["DQN"]["spear"].get(mk, 0)
        d_a = results[atk]["DQN"]["auto"].get(mk, 0)

        if hib:
            if s_s > s_a: sac_s_wins += 1
            elif s_a > s_s: sac_a_wins += 1
            if d_s > d_a: dqn_s_wins += 1
            elif d_a > d_s: dqn_a_wins += 1
        else:
            if s_s < s_a: sac_s_wins += 1
            elif s_a < s_s: sac_a_wins += 1
            if d_s < d_a: dqn_s_wins += 1
            elif d_a < d_s: dqn_a_wins += 1

    print(f"{mlabel:<45} | S={sac_s_wins} A={sac_a_wins} {f'{sac_s_wins}/{sac_a_wins}':>6} | S={dqn_s_wins} A={dqn_a_wins} {f'{dqn_s_wins}/{dqn_a_wins}':>6}")
