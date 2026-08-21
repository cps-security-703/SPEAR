#!/usr/bin/env python3


import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from datetime import datetime


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RewardHistoryAnalyzer:


    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = self._load_data()
        self.attack_types = list(self.data['inner_episode_rewards'].keys())
        self.num_systems = 6
        self.outer_episodes = self.data['outer_episodes']


        self.attack_system_performance = self._compute_attack_system_performance()

    def _load_data(self) -> Dict:

        print(f" Loading reward history from: {self.json_file}")
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        print(f" Loaded {data['mode']} training data")
        print(f"   Outer episodes: {data['outer_episodes']}")
        print(f"   Inner episodes: {data['inner_episodes']}")
        print(f"   Attack types: {len(data['inner_episode_rewards'])}")
        return data

    def _compute_attack_system_performance(self) -> Dict[str, Dict[int, List[float]]]:

        performance = {}

        for attack_type in self.attack_types:
            performance[attack_type] = {sys_id: [] for sys_id in range(1, self.num_systems + 1)}


            for outer_ep_str, ep_data in self.data['inner_episode_rewards'][attack_type].items():
                system_id = ep_data['assigned_system']
                sac_rewards = ep_data['sac_rewards']

                if sac_rewards:
                    mean_reward = float(np.mean(sac_rewards))
                    performance[attack_type][system_id].append(mean_reward)

        return performance

    def create_attack_system_heatmap(self, save_path: str = None):

        print("\n Creating Attack-System Performance Heatmap...")


        heatmap_data = np.zeros((len(self.attack_types), self.num_systems))

        for i, attack_type in enumerate(self.attack_types):
            for sys_id in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    heatmap_data[i, sys_id - 1] = np.mean(rewards)


        fig, ax = plt.subplots(figsize=(12, 8))


        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=heatmap_data.min(), vmax=heatmap_data.max())


        ax.set_xticks(np.arange(self.num_systems))
        ax.set_yticks(np.arange(len(self.attack_types)))
        ax.set_xticklabels([f'System {i+1}' for i in range(self.num_systems)])
        ax.set_yticklabels(self.attack_types)


        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")


        for i in range(len(self.attack_types)):
            for j in range(self.num_systems):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')


        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean SAC Reward (Higher = More Effective/Evasive)', rotation=270, labelpad=20)


        ax.set_xlabel('Target System', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        ax.set_title(f'Attack-System Performance Matrix\n({self.data["mode"]} training, {self.outer_episodes} outer circles)',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def create_reward_evolution_plot(self, save_path: str = None):

        print("\n Creating Reward Evolution Plot...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, attack_type in enumerate(self.attack_types):
            ax = axes[idx]


            outer_circles = []
            dqn_means = []
            sac_means = []
            dqn_stds = []
            sac_stds = []

            for outer_ep_str in sorted(self.data['inner_episode_rewards'][attack_type].keys(), key=int):
                ep_data = self.data['inner_episode_rewards'][attack_type][outer_ep_str]
                outer_circles.append(int(outer_ep_str) + 1)

                dqn_rewards = ep_data['dqn_rewards']
                sac_rewards = ep_data['sac_rewards']

                dqn_means.append(np.mean(dqn_rewards) if dqn_rewards else 0)
                sac_means.append(np.mean(sac_rewards) if sac_rewards else 0)
                dqn_stds.append(np.std(dqn_rewards) if dqn_rewards else 0)
                sac_stds.append(np.std(sac_rewards) if sac_rewards else 0)


            ax.plot(outer_circles, dqn_means, 'o-', label='DQN Agent', linewidth=2, markersize=6)
            ax.fill_between(outer_circles,
                           np.array(dqn_means) - np.array(dqn_stds),
                           np.array(dqn_means) + np.array(dqn_stds),
                           alpha=0.2)

            ax.plot(outer_circles, sac_means, 's-', label='SAC Agent', linewidth=2, markersize=6)
            ax.fill_between(outer_circles,
                           np.array(sac_means) - np.array(sac_stds),
                           np.array(sac_means) + np.array(sac_stds),
                           alpha=0.2)

            ax.set_xlabel('Outer Circle', fontsize=10, fontweight='bold')
            ax.set_ylabel('Mean Reward', fontsize=10, fontweight='bold')
            ax.set_title(f'{attack_type}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Agent Reward Evolution Across Outer Circles\n({self.data["mode"]} training)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def create_system_ranking_plot(self, save_path: str = None):

        print("\n Creating System-wise Attack Ranking...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for sys_id in range(1, self.num_systems + 1):
            ax = axes[sys_id - 1]


            attack_rewards = {}
            for attack_type in self.attack_types:
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    attack_rewards[attack_type] = np.mean(rewards)
                else:
                    attack_rewards[attack_type] = 0


            sorted_attacks = sorted(attack_rewards.items(), key=lambda x: x[1], reverse=True)
            attacks = [a[0] for a in sorted_attacks]
            rewards = [a[1] for a in sorted_attacks]


            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(attacks)))
            bars = ax.barh(attacks, rewards, color=colors)


            for i, (attack, reward) in enumerate(zip(attacks, rewards)):
                ax.text(reward + 50, i, f'{reward:.0f}', va='center', fontsize=9, fontweight='bold')

            ax.set_xlabel('Mean SAC Reward', fontsize=10, fontweight='bold')
            ax.set_title(f'System {sys_id} - Attack Effectiveness Ranking', fontsize=11, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

        plt.suptitle(f'Attack Effectiveness Rankings per System\n(Higher reward = More suitable/evasive)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def create_inner_circle_convergence_plot(self, save_path: str = None):

        print("\n Creating Inner Circle Convergence Plot...")

        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, attack_type in enumerate(self.attack_types):
            ax = axes[idx]


            for outer_ep_str in sorted(self.data['inner_episode_rewards'][attack_type].keys(), key=int):
                outer_ep = int(outer_ep_str)
                ep_data = self.data['inner_episode_rewards'][attack_type][outer_ep_str]

                dqn_rewards = ep_data['dqn_rewards']
                sac_rewards = ep_data['sac_rewards']
                system_id = ep_data['assigned_system']


                if dqn_rewards:
                    episodes = list(range(1, len(dqn_rewards) + 1))
                    ax.plot(episodes, dqn_rewards,
                           linestyle='--', linewidth=1, alpha=0.6,
                           label=f'Circle {outer_ep+1} DQN (Sys {system_id})')


                if sac_rewards:
                    episodes = list(range(1, len(sac_rewards) + 1))
                    ax.plot(episodes, sac_rewards,
                           linestyle='-', linewidth=1.5, alpha=0.8,
                           label=f'Circle {outer_ep+1} SAC (Sys {system_id})')

            ax.set_xlabel('Inner Episode', fontsize=10, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=10, fontweight='bold')
            ax.set_title(f'{attack_type}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Inner Circle Episode Rewards (Convergence within each Outer Circle)\n'
                    f'Dashed = DQN, Solid = SAC | {self.data["mode"]} training',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def create_inner_circle_boxplot(self, save_path: str = None):

        print("\n Creating Inner Circle Reward Distribution Boxplot...")

        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, attack_type in enumerate(self.attack_types):
            ax = axes[idx]


            outer_circles = []
            sac_reward_data = []
            labels = []

            for outer_ep_str in sorted(self.data['inner_episode_rewards'][attack_type].keys(), key=int):
                outer_ep = int(outer_ep_str)
                ep_data = self.data['inner_episode_rewards'][attack_type][outer_ep_str]
                sac_rewards = ep_data['sac_rewards']
                system_id = ep_data['assigned_system']

                if sac_rewards:
                    outer_circles.append(outer_ep + 1)
                    sac_reward_data.append(sac_rewards)
                    labels.append(f'C{outer_ep+1}\nS{system_id}')


            bp = ax.boxplot(sac_reward_data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)


            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sac_reward_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)


            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], linewidth=1.5)

            ax.set_xlabel('Outer Circle (C) | Target System (S)', fontsize=10, fontweight='bold')
            ax.set_ylabel('SAC Reward Distribution', fontsize=10, fontweight='bold')
            ax.set_title(f'{attack_type}', fontsize=11, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

        plt.suptitle(f'Inner Circle Reward Distribution per Outer Circle\n'
                    f'(Box = IQR, Line = Median, Dashed = Mean)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def create_attack_ranking_plot(self, save_path: str = None):

        print("\n Creating Attack-wise System Ranking...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, attack_type in enumerate(self.attack_types):
            ax = axes[idx]


            system_rewards = {}
            for sys_id in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    system_rewards[f'System {sys_id}'] = np.mean(rewards)
                else:
                    system_rewards[f'System {sys_id}'] = 0


            sorted_systems = sorted(system_rewards.items(), key=lambda x: x[1], reverse=True)
            systems = [s[0] for s in sorted_systems]
            rewards = [s[1] for s in sorted_systems]


            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(systems)))
            bars = ax.barh(systems, rewards, color=colors)


            for i, (system, reward) in enumerate(zip(systems, rewards)):
                ax.text(reward + 50, i, f'{reward:.0f}', va='center', fontsize=9, fontweight='bold')

            ax.set_xlabel('Mean SAC Reward', fontsize=10, fontweight='bold')
            ax.set_title(f'{attack_type}', fontsize=11, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

        plt.suptitle(f'System Vulnerability Rankings per Attack\n(Higher reward = More vulnerable to this attack)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

        return fig

    def generate_summary_report(self, save_path: str = None):

        print("\n Generating Summary Report...")

        report = []
        report.append("=" * 80)
        report.append(f"TWO-LEVEL RL TRAINING REWARD ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Training Mode: {self.data['mode']}")
        report.append(f"Outer Episodes: {self.data['outer_episodes']}")
        report.append(f"Inner Episodes: {self.data['inner_episodes']}")
        report.append(f"Timestamp: {self.data.get('timestamp', 'N/A')}")
        report.append("")


        report.append("=" * 80)
        report.append("OUTER EPISODE REWARDS (Aggregate)")
        report.append("=" * 80)
        outer_rewards = self.data['outer_episode_rewards']
        report.append(f"Mean: {np.mean(outer_rewards):.2f}")
        report.append(f"Std:  {np.std(outer_rewards):.2f}")
        report.append(f"Min:  {np.min(outer_rewards):.2f} (Circle {np.argmin(outer_rewards) + 1})")
        report.append(f"Max:  {np.max(outer_rewards):.2f} (Circle {np.argmax(outer_rewards) + 1})")
        report.append(f"Trend: {self.data['summary']['reward_trend']}")
        report.append("")


        report.append("=" * 80)
        report.append("TOP 10 ATTACK-SYSTEM COMBINATIONS (Highest Mean Reward)")
        report.append("=" * 80)

        combinations = []
        for attack_type in self.attack_types:
            for sys_id in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    mean_reward = np.mean(rewards)
                    combinations.append((attack_type, sys_id, mean_reward, len(rewards)))

        combinations.sort(key=lambda x: x[2], reverse=True)

        for rank, (attack, sys_id, reward, num_circles) in enumerate(combinations[:10], 1):
            report.append(f"{rank:2d}. {attack:30s}  System {sys_id}  |  Reward: {reward:7.1f}  |  Circles: {num_circles}")
        report.append("")


        report.append("=" * 80)
        report.append("BEST SYSTEM FOR EACH ATTACK TYPE")
        report.append("=" * 80)

        for attack_type in self.attack_types:
            best_sys = None
            best_reward = -float('inf')

            for sys_id in range(1, self.num_systems + 1):
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    mean_reward = np.mean(rewards)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_sys = sys_id

            report.append(f"{attack_type:30s}  System {best_sys}  (Reward: {best_reward:.1f})")
        report.append("")


        report.append("=" * 80)
        report.append("BEST ATTACK FOR EACH SYSTEM")
        report.append("=" * 80)

        for sys_id in range(1, self.num_systems + 1):
            best_attack = None
            best_reward = -float('inf')

            for attack_type in self.attack_types:
                rewards = self.attack_system_performance[attack_type][sys_id]
                if rewards:
                    mean_reward = np.mean(rewards)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_attack = attack_type

            report.append(f"System {sys_id}: {best_attack:30s} (Reward: {best_reward:.1f})")
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\n Saved report to: {save_path}")

        return report_text

    def generate_all_visualizations(self, output_dir: str = "reward_analysis"):

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REWARD HISTORY VISUALIZATIONS")
        print("=" * 80)


        self.create_attack_system_heatmap(
            save_path=output_path / f"attack_system_heatmap_{timestamp}.png"
        )


        self.create_reward_evolution_plot(
            save_path=output_path / f"reward_evolution_{timestamp}.png"
        )


        self.create_inner_circle_convergence_plot(
            save_path=output_path / f"inner_circle_convergence_{timestamp}.png"
        )


        self.create_inner_circle_boxplot(
            save_path=output_path / f"inner_circle_distribution_{timestamp}.png"
        )


        self.create_system_ranking_plot(
            save_path=output_path / f"system_rankings_{timestamp}.png"
        )


        self.create_attack_ranking_plot(
            save_path=output_path / f"attack_rankings_{timestamp}.png"
        )


        self.generate_summary_report(
            save_path=output_path / f"summary_report_{timestamp}.txt"
        )

        print("\n" + "=" * 80)
        print(" ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f" Output directory: {output_path.absolute()}")
        print(f" Files generated:")
        print(f"   • attack_system_heatmap_{timestamp}.png")
        print(f"   • reward_evolution_{timestamp}.png (outer circles)")
        print(f"   • inner_circle_convergence_{timestamp}.png (NEW - episode-level learning)")
        print(f"   • inner_circle_distribution_{timestamp}.png (NEW - reward variance)")
        print(f"   • system_rankings_{timestamp}.png")
        print(f"   • attack_rankings_{timestamp}.png")
        print(f"   • summary_report_{timestamp}.txt")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Two-Level RL Training Reward History'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='reward_history_gemini_guided.json',
        help='Path to reward history JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reward_analysis',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots interactively (in addition to saving)'
    )

    args = parser.parse_args()


    if not Path(args.file).exists():
        stem = Path(args.file).stem
        matches = sorted(Path('.').glob(f'{stem}_*.json'),
                         key=lambda p: p.stat().st_mtime)
        if matches:
            args.file = str(matches[-1])
            print(f"ℹ  Using most recent reward history: {args.file}")

    if not Path(args.file).exists():
        print(f" Error: File not found: {args.file}")
        print("\nAvailable reward history files:")
        for f in Path('.').glob('reward_history*.json'):
            print(f"   • {f}")
        return


    analyzer = RewardHistoryAnalyzer(args.file)


    analyzer.generate_all_visualizations(output_dir=args.output)


    if args.show:
        print("\n Displaying plots...")
        plt.show()


if __name__ == "__main__":
    main()
