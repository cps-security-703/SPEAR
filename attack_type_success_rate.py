
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List


def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern}")
    return matches[-1]


def per_type_stats(path: str) -> List[Dict]:

    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)

    stats = defaultdict(lambda: {'n': 0, 'success': 0, 'detected': 0,
                                  'impact_sum': 0.0, 'anomaly_sum': 0.0})
    for ep in d.get('episode_results', []):
        for a in ep.get('attacks_detected', []):
            if a.get('is_benign') or a.get('attack_type') == 'benign_normal':
                continue
            t = a['attack_type']
            s = stats[t]
            s['n'] += 1
            s['success'] += int(bool(a.get('attack_success', a.get('success', False))))
            s['detected'] += int(bool(a.get('detected', False)))
            s['impact_sum'] += float(a.get('attack_impact', a.get('impact', 0.0)) or 0.0)
            s['anomaly_sum'] += float(a.get('anomaly_score', 0.0) or 0.0)

    rows = []
    for t, s in sorted(stats.items(), key=lambda kv: -kv[1]['n']):
        n = s['n']
        rows.append({
            'attack_type': t,
            'n': n,
            'success_rate': round(s['success'] / n, 4) if n else 0.0,
            'detect_rate': round(s['detected'] / n, 4) if n else 0.0,
            'avg_impact': round(s['impact_sum'] / n, 4) if n else 0.0,
            'avg_anomaly': round(s['anomaly_sum'] / n, 4) if n else 0.0,
        })
    return rows


def print_table(rows: List[Dict], label: str, source: str) -> None:
    print(f"\n=== {label} ({os.path.basename(source)}) ===")
    header = f"{'attack_type':<26}{'n':>5}{'success_rate':>14}{'detect_rate':>13}{'avg_impact':>12}{'avg_anomaly':>13}"
    print(header)
    print("-" * len(header))
    total_n = sum(r['n'] for r in rows)
    for r in rows:
        print(f"{r['attack_type']:<26}{r['n']:>5}{r['success_rate']:>14.3f}"
              f"{r['detect_rate']:>13.3f}{r['avg_impact']:>12.3f}{r['avg_anomaly']:>13.3f}")
    if total_n:
        total_s = sum(r['success_rate'] * r['n'] for r in rows) / total_n
        total_d = sum(r['detect_rate'] * r['n'] for r in rows) / total_n
        print("-" * len(header))
        print(f"{'TOTAL':<26}{total_n:>5}{total_s:>14.3f}{total_d:>13.3f}")


def main():
    parser = argparse.ArgumentParser(description="Per-attack-type success/detection breakdown")
    parser.add_argument('--report', type=str, default=None,
                         help='ids_detection_report_*.json (RL-coordinated). Defaults to the latest in detection_results/')
    parser.add_argument('--baseline', type=str, default=None,
                         help='baseline_actual_system_*.json (random, non-RL). Optional.')
    parser.add_argument('--output', type=str, default='detection_results/attack_type_success_rate.csv')
    args = parser.parse_args()

    report_path = args.report or _latest('detection_results/ids_detection_report_*.json')
    rl_rows = per_type_stats(report_path)
    print_table(rl_rows, "RL-coordinated attacks", report_path)

    baseline_rows = []
    baseline_path = args.baseline
    if baseline_path is None:

        stem = os.path.basename(report_path).replace('ids_detection_report_', 'baseline_actual_system_')
        candidate = os.path.join(os.path.dirname(report_path), stem)
        if os.path.exists(candidate):
            baseline_path = candidate
    if baseline_path and os.path.exists(baseline_path):
        baseline_rows = per_type_stats(baseline_path)
        print_table(baseline_rows, "Baseline (random, non-RL) attacks", baseline_path)
        print("\nNOTE: baseline 'success' = impact > 0.3 only (no detection condition), while RL "
              "'success' = impact > 0.01 AND not detected -- the two success_rate columns are NOT "
              "directly comparable until the definitions are harmonized. detect_rate is comparable.")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'attack_type', 'n', 'success_rate',
                                                'detect_rate', 'avg_impact', 'avg_anomaly'])
        writer.writeheader()
        for r in rl_rows:
            writer.writerow({'source': 'rl', **r})
        for r in baseline_rows:
            writer.writerow({'source': 'baseline', **r})
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
