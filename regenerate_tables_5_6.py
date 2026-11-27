#!/usr/bin/env python3
"""Regenerate the corrected SAC (Table 5) and DQN (Table 6) significance tables.

Fixes vs the previously-shipped tables:
  * Uses ONLY the per-attack schema (inner_episode_rewards[attack][ep]{sac,dqn}),
    so SPEAR and Auto are compared on the identical structure.
  * Correct, consistent sample size (per-step n=900/attack = 18 circles x 50 steps),
    NOT the mislabelled 1800; t and d are therefore mutually consistent (d=t/sqrt(n/2)).
  * Honest signs: diff = SPEAR - Auto (positive => SPEAR larger reward).
  * Also reports the per-episode-mean unit (n=18/attack) that addresses the
    "may not be i.i.d." reviewer concern (each outer episode = one observation).
Emits LaTeX for both tables.
"""
import json, numpy as np
from scipy import stats

# Config-matched pair the paper referenced (adaptive-bandit autonomous baseline).
SPEAR_FILE = 'reward_history_gemini_guided_20260815_001640.json'
AUTO_FILE  = 'reward_history_autonomous_20260816_024035.json'

ATTACKS = ['voltage_manipulation','current_injection','power_disruption',
           'communication_spoofing','data_injection','protocol_manipulation']
LAB = {'voltage_manipulation':'Vol.\\ Manip.','current_injection':'Current Inj.',
       'power_disruption':'Power Disr.','communication_spoofing':'Com.\\ Spoof.',
       'data_injection':'Data Inj.','protocol_manipulation':'Prot.\\ Manip.'}

def per_step(d, atk, key):
    out=[]
    for ed in d.get('inner_episode_rewards',{}).get(atk,{}).values():
        out.extend(ed.get(key,[]))
    return np.array(out,float)

def per_episode(d, atk, key):
    return np.array([np.mean(ed.get(key,[np.nan]))
                     for ed in d.get('inner_episode_rewards',{}).get(atk,{}).values()],float)

def welch(a,b):
    a=a[~np.isnan(a)]; b=b[~np.isnan(b)]
    t,p=stats.ttest_ind(a,b,equal_var=False)
    diff=a.mean()-b.mean()
    se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    dfn=(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))**2
    dfd=(a.var(ddof=1)/len(a))**2/(len(a)-1)+(b.var(ddof=1)/len(b))**2/(len(b)-1)
    tc=stats.t.ppf(0.975,dfn/dfd)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return dict(mS=a.mean(),mA=b.mean(),diff=diff,lo=diff-tc*se,hi=diff+tc*se,
                t=t,p=p,d=diff/sp if sp>0 else 0.0,nS=len(a),nA=len(b))

def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

def latex_table(engine, label, rows, n_per_group, unit_note):
    cap = (f"{engine} Mean Reward Differences and Statistical Significance "
           f"(\\textsc{{SPEAR}} $-$ Autonomous) via Welch's $t$-Test. "
           f"Positive Diff/$d$ favour \\textsc{{SPEAR}}. {unit_note}")
    L=[]
    L.append("\\begin{table*}[ht]\n\\centering")
    L.append(f"\\caption{{\\small {cap}}}")
    L.append(f"\\label{{{label}}}")
    L.append("\\begin{tabular}{p{0.20\\columnwidth}p{0.22\\columnwidth}p{0.20\\columnwidth}"
             "p{0.12\\columnwidth}p{0.36\\columnwidth}p{0.10\\columnwidth}p{0.10\\columnwidth}"
             "p{0.06\\columnwidth}p{0.10\\columnwidth}}")
    L.append("\\toprule")
    L.append("\\textbf{Attack Type} & \\small \\textbf{SPEAR (Mean)} & \\small \\textbf{Auto (Mean)} & "
             "\\textbf{Diff} & \\textbf{95\\% CI} & \\textbf{$t$} & \\textbf{$p$} & \\textbf{sig} & \\textbf{$d$} \\\\")
    L.append("\\midrule")
    for atk,r in rows:
        ci=f"[{r['lo']:+,.1f}, {r['hi']:+,.1f}]"
        pstr = "<0.001" if r['p']<0.001 else f"{r['p']:.3f}"
        L.append(f"{LAB[atk]} & {r['mS']:,.1f} & {r['mA']:,.1f} & {r['diff']:+,.1f} & {ci} & "
                 f"{r['t']:+.3f} & {pstr} & {sig(r['p'])} & {r['d']:+.3f} \\\\")
    L.append("\\bottomrule")
    L.append(f"\\end{{tabular}}\n\\footnotesize $n={n_per_group}$ per group.\n\\end{{table*}}")
    return "\n".join(L)

S=json.load(open(SPEAR_FILE)); A=json.load(open(AUTO_FILE))
print(f"SPEAR = {SPEAR_FILE}\nAuto  = {AUTO_FILE}\n")

for engine,key,lbl in [('SAC','sac_rewards','tab:significance_sac'),
                       ('DQN','dqn_rewards','tab:significance_dqn')]:
    for unit,fn,n in [('per-STEP',per_step,900),('per-EPISODE',per_episode,18)]:
        rows=[(atk,welch(fn(S,atk,key),fn(A,atk,key))) for atk in ATTACKS]
        wins=sum(r['diff']>0 for _,r in rows)
        print("="*118)
        print(f"{engine}  [{unit}, n={n}/group]   SPEAR wins {wins}/6")
        print(f"{'Attack':<14}{'SPEAR':>11}{'Auto':>11}{'Diff':>11}{'95% CI':>26}{'t':>9}{'p':>9}{'sig':>5}{'d':>8}")
        for atk,r in rows:
            ci=f"[{r['lo']:+,.1f},{r['hi']:+,.1f}]"
            print(f"{LAB[atk].replace(chr(92),''):<14}{r['mS']:>11.1f}{r['mA']:>11.1f}{r['diff']:>+11.1f}"
                  f"{ci:>26}{r['t']:>+9.2f}{r['p']:>9.3f}{sig(r['p']):>5}{r['d']:>+8.3f}")
        # overall
        allS=np.concatenate([fn(S,a,key) for a in ATTACKS]); allA=np.concatenate([fn(A,a,key) for a in ATTACKS])
        ro=welch(allS,allA)
        print(f"{'OVERALL':<14}{ro['mS']:>11.1f}{ro['mA']:>11.1f}{ro['diff']:>+11.1f}"
              f"{'['+format(ro['lo'],'+,.1f')+','+format(ro['hi'],'+,.1f')+']':>26}{ro['t']:>+9.2f}{ro['p']:>9.3f}{sig(ro['p']):>5}{ro['d']:>+8.3f}")
        print()
        if unit=='per-STEP':
            note=("Per-step rewards aggregated over 18 outer circles $\\times$ 50 inner steps.")
            tex=latex_table(engine,lbl,rows,n,note)
            out=f"corrected_{lbl.split(':')[1]}.tex"
            open(out,'w').write(tex); print(f"   LaTeX -> {out}\n")
