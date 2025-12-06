#!/usr/bin/env python3
"""
RL Attack Data Visualizer
=========================
Visualizes RL agent attack data from the feedback files sent to Gemini.
Shows system-wise attack timelines, durations, types, and patterns.

Author: Enhanced EVCS System
Date: 2025-10-11
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import re
import os
from collections import defaultdict, Counter
import argparse

class RLAttackVisualizer:
    def __init__(self, file_path=None):
        """Initialize the visualizer with optional file path"""
        self.file_path = file_path
        self.attacks_data = []
        self.parsed_data = None
        
        # Attack type color mapping
        self.attack_colors = {
            'communication_spoofing': '#BB8FCE', # Purple
            'data_injection': '#45B7D1',       # Blue
            'protocol_manipulation': '#EC7063',  # Pink
            'voltage_manipulation': '#4ECDC4',   # Teal
            'power_disruption': '#FFA07A',       # Light Salmon
            'current_injection': '#F7DC6F'       # Yellow
        }
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def find_latest_rl_feedback_file(self):
        """Find the latest RL feedback file in attack_scenarios_logs directory"""
        logs_dir = "attack_scenarios_logs"
        if not os.path.exists(logs_dir):
            print(f"❌ Directory {logs_dir} not found!")
            return None
            
        # Find all RL feedback files
        rl_files = [f for f in os.listdir(logs_dir) if f.startswith('rl_feedback_to_gemini_') and f.endswith('.txt')]
        
        if not rl_files:
            print(f"❌ No RL feedback files found in {logs_dir}")
            return None
            
        # Sort by timestamp in filename
        rl_files.sort(reverse=True)  # Latest first
        latest_file = os.path.join(logs_dir, rl_files[0])
        print(f"📁 Using latest RL feedback file: {latest_file}")
        return latest_file
    
    def parse_rl_feedback_file(self, file_path):
        """Parse the RL feedback text file and extract attack data"""
        print(f"📖 Parsing RL feedback file: {file_path}")
        
        attacks = []
        current_attack = {}
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            attack_section = False
            for line in lines:
                line = line.strip()
                
                # Start of attack data section
                if line == "RL AGENT ATTACK DATA":
                    attack_section = True
                    continue
                    
                # End of attack data section
                if attack_section and line.startswith("RL EPISODE PERFORMANCE METRICS"):
                    break
                    
                if attack_section:
                    # New attack
                    if line.startswith("ATTACK #"):
                        if current_attack:  # Save previous attack
                            attacks.append(current_attack.copy())
                        current_attack = {'attack_id': line}
                        
                    # Parse attack attributes
                    elif ':' in line and current_attack:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Special handling for TYPE - only keep the FIRST occurrence (primary attack type)
                        if key == 'TYPE' and 'TYPE' in current_attack:
                            continue  # Skip subsequent TYPE fields
                        
                        # Convert numeric values
                        try:
                            if key in ['START_TIME', 'DURATION', 'MAGNITUDE', 'STEALTH_LEVEL', 
                                     'IMPACT_FACTOR', 'SUCCESS_RATE', 'VOLTAGE_DEVIATION',
                                     'FREQUENCY_DEVIATION', 'POWER_LOSS', 'LOAD_DISRUPTION']:
                                value = float(value)
                            elif key == 'TARGET_SYSTEM':
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string
                            
                        current_attack[key] = value
                        
            # Add the last attack
            if current_attack:
                attacks.append(current_attack)
                
            print(f"✅ Parsed {len(attacks)} attacks from RL feedback file")
            return attacks
            
        except Exception as e:
            print(f"❌ Error parsing file: {str(e)}")
            return []
    
    def process_attack_data(self, attacks):
        """Process and clean the attack data for visualization"""
        processed_attacks = []
        
        for attack in attacks:
            # Skip if missing essential data
            if not all(key in attack for key in ['TYPE', 'TARGET_SYSTEM', 'START_TIME', 'DURATION']):
                continue
                
            # Standardize attack type
            raw_type = attack['TYPE']
            
            # Mapping for legacy/non-standard names
            type_mapping = {
                'cyber_attack': 'communication_spoofing',
                'soc_spoofing': 'communication_spoofing',
                'frequency_manipulation': 'data_injection',
                'frequency_attack': 'data_injection',
                'thermal_attack': 'protocol_manipulation',
                'load_manipulation': 'power_disruption',
                'model_poisoning': 'data_injection',
                'power_manipulation': 'power_disruption'
            }
            
            attack_type = type_mapping.get(raw_type, raw_type)
            
            processed_attack = {
                'attack_type': attack_type,
                'target_system': attack['TARGET_SYSTEM'],
                'start_time': attack['START_TIME'],
                'duration': attack['DURATION'],
                'end_time': attack['START_TIME'] + attack['DURATION'],
                'magnitude': attack.get('MAGNITUDE', 1.0),
                'stealth_level': attack.get('STEALTH_LEVEL', 0.5),
                'impact_factor': attack.get('IMPACT_FACTOR', 0.5),
                'success_rate': attack.get('SUCCESS_RATE', 1.0)
            }
            processed_attacks.append(processed_attack)
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(processed_attacks)
        print(f"📊 Processed {len(df)} valid attacks")
        return df
    
    def create_timeline_visualization(self, df, save_path=None):
        """Create a comprehensive timeline visualization"""
        if df.empty:
            print("❌ No data to visualize")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main timeline plot
        ax1 = plt.subplot(3, 2, (1, 2))
        self._plot_attack_timeline(df, ax1)
        
        # Attack type distribution
        ax2 = plt.subplot(3, 2, 3)
        self._plot_attack_type_distribution(df, ax2)
        
        # System load distribution
        ax3 = plt.subplot(3, 2, 4)
        self._plot_system_attack_distribution(df, ax3)
        
        # Duration analysis
        ax4 = plt.subplot(3, 2, 5)
        self._plot_duration_analysis(df, ax4)
        
        # Impact vs Stealth scatter
        ax5 = plt.subplot(3, 2, 6)
        self._plot_impact_stealth_analysis(df, ax5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Timeline visualization saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"rl_attack_timeline_{timestamp}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"💾 Timeline visualization saved to: {default_path}")
            
        plt.show()
    
    def _plot_attack_timeline(self, df, ax):
        """Plot the main attack timeline"""
        ax.set_title("RL Agent Attack Timeline by System", fontsize=16, fontweight='bold')
        
        systems = sorted(df['target_system'].unique())
        y_positions = {system: i for i, system in enumerate(systems)}
        
        # Plot each attack as a horizontal bar
        for _, attack in df.iterrows():
            y_pos = y_positions[attack['target_system']]
            color = self.attack_colors.get(attack['attack_type'], '#CCCCCC')
            
            # Create rectangle for attack duration
            rect = patches.Rectangle(
                (attack['start_time'], y_pos - 0.3),
                attack['duration'],
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add attack type label if duration is long enough
            if attack['duration'] > 50:  # Only label longer attacks
                ax.text(
                    attack['start_time'] + attack['duration']/2,
                    y_pos,
                    attack['attack_type'].replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold'
                )
        
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Target System", fontsize=12)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([f"System {s}" for s in systems])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, df['end_time'].max() * 1.1)
        
        # Create legend
        legend_elements = [patches.Patch(color=color, label=attack_type.replace('_', ' ').title()) 
                          for attack_type, color in self.attack_colors.items() 
                          if attack_type in df['attack_type'].values]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _plot_attack_type_distribution(self, df, ax):
        """Plot attack type distribution"""
        attack_counts = df['attack_type'].value_counts()
        colors = [self.attack_colors.get(attack_type, '#CCCCCC') for attack_type in attack_counts.index]
        
        bars = ax.bar(range(len(attack_counts)), attack_counts.values, color=colors, alpha=0.7)
        ax.set_title("Attack Type Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attack Type", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xticks(range(len(attack_counts)))
        ax.set_xticklabels([t.replace('_', '\n') for t in attack_counts.index], rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, attack_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_system_attack_distribution(self, df, ax):
        """Plot attacks per system"""
        system_counts = df['target_system'].value_counts().sort_index()
        
        bars = ax.bar(system_counts.index, system_counts.values, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_title("Attacks per System", fontsize=14, fontweight='bold')
        ax.set_xlabel("Target System", fontsize=10)
        ax.set_ylabel("Number of Attacks", fontsize=10)
        
        # Add count labels
        for bar, count in zip(bars, system_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_duration_analysis(self, df, ax):
        """Plot attack duration analysis"""
        durations_by_type = df.groupby('attack_type')['duration'].mean().sort_values(ascending=False)
        colors = [self.attack_colors.get(attack_type, '#CCCCCC') for attack_type in durations_by_type.index]
        
        bars = ax.bar(range(len(durations_by_type)), durations_by_type.values, color=colors, alpha=0.7)
        ax.set_title("Average Attack Duration by Type", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attack Type", fontsize=10)
        ax.set_ylabel("Average Duration (seconds)", fontsize=10)
        ax.set_xticks(range(len(durations_by_type)))
        ax.set_xticklabels([t.replace('_', '\n') for t in durations_by_type.index], rotation=45, ha='right')
        
        # Add duration labels
        for bar, duration in zip(bars, durations_by_type.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{duration:.1f}s", ha='center', va='bottom', fontweight='bold')
    
    def _plot_impact_stealth_analysis(self, df, ax):
        """Plot impact vs stealth analysis"""
        scatter = ax.scatter(df['stealth_level'], df['impact_factor'], 
                           c=[self.attack_colors.get(t, '#CCCCCC') for t in df['attack_type']],
                           s=df['duration']*2, alpha=0.6, edgecolors='black')
        
        ax.set_title("Impact vs Stealth Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Stealth Level", fontsize=10)
        ax.set_ylabel("Impact Factor", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.05, 0.95, "Bubble size = Duration", transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def create_system_gantt_chart(self, df, save_path=None):
        """Create a detailed Gantt chart for system-wise attack visualization"""
        if df.empty:
            print("❌ No data to visualize")
            return
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        systems = sorted(df['target_system'].unique())
        y_positions = {system: i for i, system in enumerate(systems)}
        
        # Plot each attack
        for _, attack in df.iterrows():
            y_pos = y_positions[attack['target_system']]
            color = self.attack_colors.get(attack['attack_type'], '#CCCCCC')
            
            # Create rectangle
            rect = patches.Rectangle(
                (attack['start_time'], y_pos - 0.35),
                attack['duration'],
                0.7,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add detailed labels
            label_text = f"{attack['attack_type']}\nMag:{attack['magnitude']:.1f}\nStealth:{attack['stealth_level']:.2f}"
            ax.text(
                attack['start_time'] + attack['duration']/2,
                y_pos,
                label_text,
                ha='center', va='center',
                fontsize=24, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
        
        # Customize the plot
        # ax.set_title("RL Agent Attacks: System-wise Gantt Chart", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=24)
        ax.set_ylabel("Target Systems", fontsize=24)
        # ax.set_yticks()
        
        # ax.set_yticklabels([f"EVCS System {s}" for s in systems], fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, df['end_time'].max() * 1.05)
        ax.set_ylim(-0.5, len(systems) - 0.5)
        
        # Add time markers
        time_markers = np.arange(0, df['end_time'].max(), 300)  # Every 5 minutes
        # for marker in time_markers:
        #     ax.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        #     ax.text(marker, len(systems)-0.2, f"{marker/60:.0f}min", 
        #            rotation=90, ha='right', va='top', fontsize=18)
        
        # Create detailed legend
        legend_elements = [patches.Patch(color=color, label=f"{attack_type.replace('_', ' ').title()}") 
                          for attack_type, color in self.attack_colors.items() 
                          if attack_type in df['attack_type'].values]
        ax.legend(handles=legend_elements, fontsize=18)

        ax.tick_params(axis='both', labelsize=18)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gantt chart saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"rl_attack_gantt_{timestamp}.pdf"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gantt chart saved to: {default_path}")
            
        plt.show()
    
    def generate_attack_summary_report(self, df):
        """Generate a comprehensive text summary report"""
        if df.empty:
            print("❌ No data to analyze")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"rl_attack_analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("RL AGENT ATTACK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Attacks Analyzed: {len(df)}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Simulation Duration: {df['end_time'].max():.1f} seconds ({df['end_time'].max()/60:.1f} minutes)\n")
            f.write(f"Number of Target Systems: {df['target_system'].nunique()}\n")
            f.write(f"Attack Types Used: {df['attack_type'].nunique()}\n")
            f.write(f"Average Attack Duration: {df['duration'].mean():.2f} seconds\n")
            f.write(f"Total Attack Time: {df['duration'].sum():.1f} seconds\n\n")
            
            # System-wise analysis
            f.write("SYSTEM-WISE ATTACK ANALYSIS\n")
            f.write("-" * 35 + "\n")
            for system in sorted(df['target_system'].unique()):
                system_data = df[df['target_system'] == system]
                f.write(f"System {system}:\n")
                f.write(f"  - Total Attacks: {len(system_data)}\n")
                f.write(f"  - Attack Types: {', '.join(system_data['attack_type'].unique())}\n")
                f.write(f"  - Total Duration: {system_data['duration'].sum():.1f}s\n")
                f.write(f"  - Avg Magnitude: {system_data['magnitude'].mean():.2f}\n")
                f.write(f"  - Avg Stealth: {system_data['stealth_level'].mean():.2f}\n\n")
            
            # Attack type analysis
            f.write("ATTACK TYPE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            for attack_type in df['attack_type'].unique():
                type_data = df[df['attack_type'] == attack_type]
                f.write(f"{attack_type.replace('_', ' ').title()}:\n")
                f.write(f"  - Count: {len(type_data)}\n")
                f.write(f"  - Avg Duration: {type_data['duration'].mean():.2f}s\n")
                f.write(f"  - Systems Targeted: {sorted(type_data['target_system'].unique())}\n")
                f.write(f"  - Avg Impact: {type_data['impact_factor'].mean():.3f}\n\n")
            
            # Timing patterns
            f.write("TIMING PATTERNS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Earliest Attack: {df['start_time'].min():.1f}s\n")
            f.write(f"Latest Attack Start: {df['start_time'].max():.1f}s\n")
            f.write(f"Latest Attack End: {df['end_time'].max():.1f}s\n")
            
            # Most attacked periods
            time_bins = pd.cut(df['start_time'], bins=10)
            time_counts = time_bins.value_counts().sort_index()
            f.write(f"\nMost Active Time Periods:\n")
            for interval, count in time_counts.head(3).items():
                f.write(f"  - {interval.left:.0f}s - {interval.right:.0f}s: {count} attacks\n")
        
        print(f"📄 Analysis report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self, file_path=None):
        """Run complete analysis and visualization"""
        print("🚀 Starting RL Attack Data Analysis...")
        
        # Use provided file or find latest
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = self.find_latest_rl_feedback_file()
            
        if not self.file_path:
            print("❌ No RL feedback file found!")
            return
            
        # Parse the data
        attacks = self.parse_rl_feedback_file(self.file_path)
        if not attacks:
            print("❌ No attack data found!")
            return
            
        # Process the data
        df = self.process_attack_data(attacks)
        if df.empty:
            print("❌ No valid attack data to visualize!")
            return
            
        self.parsed_data = df
        
        # Generate visualizations
        print("\n📊 Creating timeline visualization...")
        self.create_timeline_visualization(df)
        
        print("\n📊 Creating Gantt chart...")
        self.create_system_gantt_chart(df)
        
        print("\n📄 Generating analysis report...")
        self.generate_attack_summary_report(df)
        
        print("\n✅ Complete analysis finished!")
        return df

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Visualize RL Agent Attack Data")
    parser.add_argument("--file", "-f", help="Path to RL feedback file")
    parser.add_argument("--latest", "-l", action="store_true", help="Use latest RL feedback file")
    
    args = parser.parse_args()
    
    visualizer = RLAttackVisualizer()
    
    if args.file:
        visualizer.run_complete_analysis(args.file)
    else:
        visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()
