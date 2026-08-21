import json
import numpy as np
from scipy import stats


spear_path = r'reward_history_gemini_guided_20260815_001640.json'
auto_path  = r'reward_history_autonomous_20260816_024035.json'


with open(spear_path) as f:
    spear = json.load(f)
with open(auto_path) as f:
    auto = json.load(f)

attack_types = [
    'voltage_manipulation',
    'current_injection',
    'power_disruption',
    'communication_spoofing',
    'data_injection',
    'protocol_manipulation',
]


_system_to_attack = {f'system_{i+1}': atk for i, atk in enumerate(attack_types)}

def collect_sac_rewards(data, attack_type):

    rewards = []


    inner = data.get('inner_episode_rewards', {})
    if attack_type in inner:
        for ep_key, ep_data in inner[attack_type].items():
            sac = ep_data.get('sac_rewards', [])
            rewards.extend(sac)
        return np.array(rewards)


    pre = data.get('pretraining_rewards', {})
    for sys_key, sys_data in pre.items():
        if _system_to_attack.get(sys_key) == attack_type:
            rewards.extend(sys_data.get('sac_episode_rewards', []))
    return np.array(rewards)

def collect_all_sac(data):

    rewards = []
    for atk in attack_types:
        r = collect_sac_rewards(data, atk)
        rewards.extend(r)
    return np.array(rewards)

def welch_t_test(a, b):

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    diff = mean_a - mean_b

    se = np.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))
    df_num = (np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))**2
    df_den = (np.var(a, ddof=1)/len(a))**2 / (len(a)-1) + (np.var(b, ddof=1)/len(b))**2 / (len(b)-1)
    df = df_num / df_den
    t_crit = stats.t.ppf(0.975, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    d = diff / pooled_std if pooled_std > 0 else 0.0
    return mean_a, mean_b, diff, ci_low, ci_high, t_stat, p_value, d

def sig_label(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

labels = {
    'voltage_manipulation': 'Vol. Manip.',
    'current_injection': 'Current Inj.',
    'power_disruption': 'Power Disr.',
    'communication_spoofing': 'Com. Spoof.',
    'data_injection': 'Data Inj.',
    'protocol_manipulation': 'Prot. Manip.',
}

print("=" * 120)
print("SAC Mean Reward Differences and Statistical Significance (SPEAR - Autonomous) via Welch's t-Test")
print("=" * 120)
print(f"{'Attack Type':<18} {'Mean(SPEAR)':>12} {'Mean(Auto)':>12} {'Diff':>10} {'95% CI':>26} {'t':>10} {'p':>10} {'sig':>5} {'d':>8}")
print("-" * 120)

all_spear = []
all_auto = []

for atk in attack_types:
    s = collect_sac_rewards(spear, atk)
    a = collect_sac_rewards(auto, atk)
    all_spear.append(s)
    all_auto.append(a)

    m_s, m_a, diff, ci_lo, ci_hi, t_stat, p_val, d = welch_t_test(s, a)
    sig = sig_label(p_val)

    ci_str = f"[{ci_lo:+,.1f}, {ci_hi:+,.1f}]"
    print(f"{labels[atk]:<18} {m_s:>12.1f} {m_a:>12.1f} {diff:>+10.1f} {ci_str:>26} {t_stat:>+10.3f} {p_val:>10.4f} {sig:>5} {d:>+8.3f}")


all_s = np.concatenate(all_spear)
all_a = np.concatenate(all_auto)
m_s, m_a, diff, ci_lo, ci_hi, t_stat, p_val, d = welch_t_test(all_s, all_a)
sig = sig_label(p_val)
ci_str = f"[{ci_lo:+,.1f}, {ci_hi:+,.1f}]"
print("-" * 120)
print(f"{'Overall SAC':<18} {m_s:>12.1f} {m_a:>12.1f} {diff:>+10.1f} {ci_str:>26} {t_stat:>+10.3f} {p_val:>10.4f} {sig:>5} {d:>+8.3f}")


print("\nSample sizes:")
for atk in attack_types:
    s = collect_sac_rewards(spear, atk)
    a = collect_sac_rewards(auto, atk)
    print(f"  {labels[atk]:<18}  SPEAR n={len(s)}, Auto n={len(a)}")
print(f"  {'Overall':<18}  SPEAR n={len(all_s)}, Auto n={len(all_a)}")


print("\n\nOuter Episode Rewards:")
s_outer = np.array(spear.get('outer_episode_rewards', spear.get('episode_rewards', [])))
a_outer = np.array(auto.get('outer_episode_rewards', auto.get('episode_rewards', [])))
print(f"  SPEAR: mean={np.mean(s_outer):.1f}, std={np.std(s_outer, ddof=1):.1f}, n={len(s_outer)}")
print(f"  Auto:  mean={np.mean(a_outer):.1f}, std={np.std(a_outer, ddof=1):.1f}, n={len(a_outer)}")
m_s, m_a, diff, ci_lo, ci_hi, t_stat, p_val, d = welch_t_test(s_outer, a_outer)
sig = sig_label(p_val)
ci_str = f"[{ci_lo:+,.1f}, {ci_hi:+,.1f}]"
print(f"  Diff={diff:+.1f}, 95% CI={ci_str}, t={t_stat:+.3f}, p={p_val:.4f}, sig={sig}, d={d:+.3f}")
