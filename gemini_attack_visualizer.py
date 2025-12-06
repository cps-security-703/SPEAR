#!/usr/bin/env python3
"""
Gemini Attack Scenarios Visualizer
==================================
Visualizes Gemini-optimized attack scenarios from hierarchical co-simulation files.
Shows strategic attack waves, coordination patterns, and multi-system impacts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
import re

class GeminiAttackVisualizer:
    def __init__(self):
        """Initialize the Gemini attack visualizer"""
        self.scenarios_data = []
        
        # Attack type colors - same as RL visualizer
        self.attack_colors = {
            'communication_spoofing': '#BB8FCE', # Purple
            'data_injection': '#45B7D1',       # Blue
            'protocol_manipulation': '#EC7063',  # Pink
            'voltage_manipulation': '#4ECDC4',   # Teal
            'power_disruption': '#FFA07A',       # Light Salmon
            'current_injection': '#F7DC6F',      # Yellow
            'default': '#CCCCCC'
        }
        
        # Coordination type patterns
        self.coord_patterns = {
            'simultaneous': '||||',
            'sequential': '----',
            'staggered': '....',
            'overlapping': '////',
            'continuous': '====',
            'cyclical': 'oooo'
        }
        
    def find_latest_gemini_file(self):
        """Find the latest hierarchical cosim attack scenarios file"""
        logs_dir = "attack_scenarios_logs"
        if not os.path.exists(logs_dir):
            print(f"❌ Directory {logs_dir} not found!")
            return None
            
        # Find all hierarchical cosim files
        cosim_files = [f for f in os.listdir(logs_dir) 
                      if f.startswith('hierarchical_cosim_attack_scenarios_') and f.endswith('.txt')]
        
        if not cosim_files:
            print(f"❌ No hierarchical cosim files found in {logs_dir}")
            return None
            
        # Sort by timestamp
        cosim_files.sort(reverse=True)
        latest_file = os.path.join(logs_dir, cosim_files[0])
        print(f"📁 Using latest Gemini scenarios file: {latest_file}")
        return latest_file
    
    def parse_gemini_scenarios(self, file_path):
        """Parse the Gemini attack scenarios file"""
        print(f"📖 Parsing Gemini scenarios: {file_path}")
        
        scenarios = []
        current_scenario = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Split by attack scenarios
            scenario_blocks = re.split(r'ATTACK SCENARIO #\d+', content)[1:]  # Skip header
            
            for block in scenario_blocks:
                scenario = {}
                lines = block.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if ':' in line and not line.startswith('=') and not line.startswith('-'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse arrays
                        if value.startswith('[') and value.endswith(']'):
                            # Handle multi-line arrays
                            array_content = value[1:-1].strip()
                            if ',' in array_content:
                                scenario[key] = [item.strip().strip('"') for item in array_content.split(',')]
                            else:
                                scenario[key] = [array_content.strip().strip('"')] if array_content else []
                        else:
                            # Convert numeric values
                            try:
                                if key in ['START_TIME', 'DURATION', 'ATTACK_MAGNITUDE', 'IMPACT_FACTOR', 
                                         'SUCCESS_RATE', 'VOLTAGE_DEVIATION', 'FREQUENCY_DEVIATION']:
                                    value = float(value)
                                elif key in ['TARGET_SYSTEM']:
                                    value = int(value)
                            except ValueError:
                                pass
                            scenario[key] = value
                
                if scenario:
                    # Standardize attack type
                    raw_type = scenario.get('TYPE', scenario.get('ATTACK_TYPE', 'unknown'))
                    
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
                    
                    standardized_type = type_mapping.get(raw_type, raw_type)
                    scenario['TYPE'] = standardized_type
                    if 'ATTACK_TYPE' in scenario:
                        scenario['ATTACK_TYPE'] = standardized_type
                        
                    scenarios.append(scenario)
                    
            print(f"✅ Parsed {len(scenarios)} Gemini attack scenarios")
            return scenarios
            
        except Exception as e:
            print(f"❌ Error parsing Gemini scenarios: {str(e)}")
            return []
    
    def create_strategic_timeline(self, scenarios, save_path=None):
        """Create strategic attack timeline visualization using same style as RL visualizer"""
        if not scenarios:
            print("❌ No scenarios to visualize")
            return
            
        # Create figure with subplots - same layout as RL visualizer
        fig = plt.figure(figsize=(20, 12))
        
        # Main timeline plot
        ax1 = plt.subplot(3, 2, (1, 2))
        self._plot_strategic_timeline(scenarios, ax1)
        
        # Wave type distribution
        ax2 = plt.subplot(3, 2, 3)
        self._plot_wave_type_distribution(scenarios, ax2)
        
        # System coordination distribution
        ax3 = plt.subplot(3, 2, 4)
        self._plot_system_coordination_distribution(scenarios, ax3)
        
        # Duration analysis
        ax4 = plt.subplot(3, 2, 5)
        self._plot_duration_analysis(scenarios, ax4)
        
        # Impact vs Success scatter
        ax5 = plt.subplot(3, 2, 6)
        self._plot_impact_success_analysis(scenarios, ax5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Strategic timeline saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"gemini_strategic_timeline_{timestamp}.pdf"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"💾 Strategic timeline saved to: {default_path}")
            
        plt.show()
    
    def _plot_strategic_timeline(self, scenarios, ax):
        """Plot the main strategic timeline - same style as RL visualizer"""
        ax.set_title("Gemini Strategic Attack Timeline by Type", fontsize=16, fontweight='bold')
        
        # Plot each scenario as a horizontal bar
        for i, scenario in enumerate(scenarios):
            start_time = scenario.get('START_TIME', 0)
            duration = scenario.get('DURATION', 100)
            attack_type = scenario.get('TYPE', scenario.get('ATTACK_TYPE', 'unknown'))
            target_systems = scenario.get('TARGET_SYSTEMS', [scenario.get('TARGET_SYSTEM', 1)])
            
            # Determine attack color - same as RL visualizer
            color = self.attack_colors.get(attack_type, self.attack_colors['default'])
            
            # Create rectangle for attack duration - same style as RL visualizer
            rect = patches.Rectangle(
                (start_time, i - 0.3),
                duration,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add attack type label if duration is long enough - same logic as RL visualizer
            if duration > 50:  # Only label longer attacks
                ax.text(
                    start_time + duration/2,
                    i,
                    attack_type.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold'
                )
        
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Attack Scenarios", fontsize=12)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([f"Scenario {i+1}" for i in range(len(scenarios))])
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit - same style as RL visualizer
        max_end_time = max(s.get('START_TIME', 0) + s.get('DURATION', 0) for s in scenarios)
        ax.set_xlim(0, max_end_time * 1.1)
        
        # Create legend - same style as RL visualizer
        legend_elements = [patches.Patch(color=color, label=attack_type.replace('_', ' ').title()) 
                          for attack_type, color in self.attack_colors.items() 
                          if attack_type != 'default' and any(s.get('TYPE', s.get('ATTACK_TYPE', '')) == attack_type for s in scenarios)]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _plot_wave_type_distribution(self, scenarios, ax):
        """Plot attack type distribution - same style as RL attack type distribution"""
        attack_types = [s.get('TYPE', s.get('ATTACK_TYPE', 'unknown')) for s in scenarios]
        attack_counts = pd.Series(attack_types).value_counts()
        colors = [self.attack_colors.get(attack_type, self.attack_colors['default']) for attack_type in attack_counts.index]
        
        bars = ax.bar(range(len(attack_counts)), attack_counts.values, color=colors, alpha=0.7)
        ax.set_title("Attack Type Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attack Type", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xticks(range(len(attack_counts)))
        ax.set_xticklabels([t.replace('_', '\n') for t in attack_counts.index], rotation=45, ha='right')
        
        # Add count labels on bars - same style as RL visualizer
        for bar, count in zip(bars, attack_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_system_coordination_distribution(self, scenarios, ax):
        """Plot system coordination distribution - same style as RL system distribution"""
        # Count total systems targeted across all scenarios
        all_systems = []
        for scenario in scenarios:
            target_systems = scenario.get('TARGET_SYSTEMS', [scenario.get('TARGET_SYSTEM', 1)])
            all_systems.extend(target_systems)
        
        system_counts = pd.Series(all_systems).value_counts().sort_index()
        
        bars = ax.bar(system_counts.index, system_counts.values, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_title("Systems Targeted by Gemini Waves", fontsize=14, fontweight='bold')
        ax.set_xlabel("Target System", fontsize=10)
        ax.set_ylabel("Number of Wave Attacks", fontsize=10)
        
        # Add count labels - same style as RL visualizer
        for bar, count in zip(bars, system_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_duration_analysis(self, scenarios, ax):
        """Plot duration analysis - same style as RL duration analysis"""
        attack_types = [s.get('TYPE', s.get('ATTACK_TYPE', 'unknown')) for s in scenarios]
        durations = [s.get('DURATION', 0) for s in scenarios]
        
        # Create duration by attack type
        attack_duration_data = {}
        for attack_type, duration in zip(attack_types, durations):
            if attack_type not in attack_duration_data:
                attack_duration_data[attack_type] = []
            attack_duration_data[attack_type].append(duration)
        
        # Calculate average durations
        avg_durations = {attack_type: np.mean(durs) for attack_type, durs in attack_duration_data.items()}
        sorted_attacks = sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)
        
        attack_types_sorted = [item[0] for item in sorted_attacks]
        duration_values = [item[1] for item in sorted_attacks]
        colors = [self.attack_colors.get(attack_type, self.attack_colors['default']) for attack_type in attack_types_sorted]
        
        bars = ax.bar(range(len(attack_types_sorted)), duration_values, color=colors, alpha=0.7)
        ax.set_title("Average Attack Duration by Type", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attack Type", fontsize=10)
        ax.set_ylabel("Average Duration (seconds)", fontsize=10)
        ax.set_xticks(range(len(attack_types_sorted)))
        ax.set_xticklabels([t.replace('_', '\n') for t in attack_types_sorted], rotation=45, ha='right')
        
        # Add duration labels - same style as RL visualizer
        for bar, duration in zip(bars, duration_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f"{duration:.0f}s", ha='center', va='bottom', fontweight='bold')
    
    def _plot_impact_success_analysis(self, scenarios, ax):
        """Plot impact vs success analysis - same style as RL impact vs stealth"""
        impact_factors = [s.get('IMPACT_FACTOR', 0) for s in scenarios]
        success_rates = [s.get('SUCCESS_RATE', 0) for s in scenarios]
        durations = [s.get('DURATION', 0) for s in scenarios]
        attack_types = [s.get('TYPE', s.get('ATTACK_TYPE', 'unknown')) for s in scenarios]
        
        colors = [self.attack_colors.get(attack_type, self.attack_colors['default']) for attack_type in attack_types]
        
        scatter = ax.scatter(success_rates, impact_factors, 
                           c=colors, s=[d/5 for d in durations], alpha=0.6, edgecolors='black')
        
        ax.set_title("Impact vs Success Rate Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Success Rate", fontsize=10)
        ax.set_ylabel("Impact Factor", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation - same style as RL visualizer
        ax.text(0.05, 0.95, "Bubble size = Duration", transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_wave_progression(self, scenarios, ax):
        """Plot wave progression metrics"""
        waves = []
        impacts = []
        magnitudes = []
        success_rates = []
        
        for scenario in scenarios:
            wave_name = scenario.get('SCENARIO_NAME', 'Unknown').split(':')[0]
            waves.append(wave_name)
            impacts.append(scenario.get('IMPACT_FACTOR', 0))
            magnitudes.append(scenario.get('ATTACK_MAGNITUDE', 0))
            success_rates.append(scenario.get('SUCCESS_RATE', 0))
        
        x = range(len(waves))
        width = 0.25
        
        ax.bar([i - width for i in x], impacts, width, label='Impact Factor', alpha=0.7)
        ax.bar(x, magnitudes, width, label='Attack Magnitude', alpha=0.7)
        ax.bar([i + width for i in x], success_rates, width, label='Success Rate', alpha=0.7)
        
        ax.set_title("Wave Progression Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attack Waves", fontsize=10)
        ax.set_ylabel("Metric Values", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(waves, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_system_coordination(self, scenarios, ax):
        """Plot multi-system coordination patterns"""
        # Create coordination matrix
        all_systems = set()
        for scenario in scenarios:
            target_systems = scenario.get('TARGET_SYSTEMS', [scenario.get('TARGET_SYSTEM', 1)])
            all_systems.update(target_systems)
        
        all_systems = sorted(list(all_systems))
        coord_matrix = np.zeros((len(scenarios), len(all_systems)))
        
        for i, scenario in enumerate(scenarios):
            target_systems = scenario.get('TARGET_SYSTEMS', [scenario.get('TARGET_SYSTEM', 1)])
            for sys in target_systems:
                if sys in all_systems:
                    j = all_systems.index(sys)
                    coord_matrix[i, j] = scenario.get('IMPACT_FACTOR', 0.5)
        
        im = ax.imshow(coord_matrix, cmap='Reds', aspect='auto')
        ax.set_title("Multi-System Coordination Matrix", fontsize=14, fontweight='bold')
        ax.set_xlabel("Target Systems", fontsize=10)
        ax.set_ylabel("Attack Waves", fontsize=10)
        ax.set_xticks(range(len(all_systems)))
        ax.set_xticklabels([f"Sys {s}" for s in all_systems])
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([f"Wave {i+1}" for i in range(len(scenarios))])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Impact Factor')
    
    def _plot_impact_escalation(self, scenarios, ax):
        """Plot impact escalation over time"""
        times = []
        impacts = []
        wave_names = []
        
        for scenario in scenarios:
            start_time = scenario.get('START_TIME', 0)
            duration = scenario.get('DURATION', 100)
            impact = scenario.get('IMPACT_FACTOR', 0)
            wave_name = scenario.get('SCENARIO_NAME', 'Unknown').split(':')[0]
            
            # Plot impact over duration
            times.extend([start_time, start_time + duration])
            impacts.extend([impact, impact])
            wave_names.extend([wave_name, wave_name])
        
        # Create step plot
        for i in range(0, len(times), 2):
            if i + 1 < len(times):
                wave_key = wave_names[i]
                color = self.wave_colors.get(wave_key, self.wave_colors['default'])
                ax.plot([times[i], times[i+1]], [impacts[i], impacts[i+1]], 
                       color=color, linewidth=4, alpha=0.7, label=wave_key)
        
        ax.set_title("Impact Escalation Timeline", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Impact Factor", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    def create_strategic_gantt_chart(self, scenarios, save_path=None):
        """Create a detailed Gantt chart for strategic waves - same style as RL Gantt chart"""
        if not scenarios:
            print("❌ No scenarios to visualize")
            return
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot each strategic attack
        for i, scenario in enumerate(scenarios):
            start_time = scenario.get('START_TIME', 0)
            duration = scenario.get('DURATION', 100)
            attack_type = scenario.get('TYPE', scenario.get('ATTACK_TYPE', 'unknown'))
            target_systems = scenario.get('TARGET_SYSTEMS', [scenario.get('TARGET_SYSTEM', 1)])
            impact_factor = scenario.get('IMPACT_FACTOR', 0)
            success_rate = scenario.get('SUCCESS_RATE', 0)
            
            # Determine attack color - same as RL visualizer
            color = self.attack_colors.get(attack_type, self.attack_colors['default'])
            
            # Create rectangle - same style as RL visualizer
            rect = patches.Rectangle(
                (start_time, i - 0.35),
                duration,
                0.7,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add detailed labels - same style as RL visualizer
            label_text = f"{attack_type}\nImpact:{impact_factor:.2f}\nSuccess:{success_rate:.2f}"
            ax.text(
                start_time + duration/2,
                i,
                label_text,
                ha='center', va='center',
                fontsize=24, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
        
        # Customize the plot - same style as RL visualizer
        # ax.set_title("Gemini Strategic Waves: Detailed Gantt Chart", fontsize=18, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=24)
        ax.set_ylabel("Strategic Waves", fontsize=24)
        ax.tick_params(axis='both', labelsize=18)
        # ax.set_yticks(range(len(scenarios)))
        # ax.set_yticklabels([f"Wave {i+1}" for i in range(len(scenarios))], fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        max_end_time = max(s.get('START_TIME', 0) + s.get('DURATION', 0) for s in scenarios)
        ax.set_xlim(0, max_end_time * 1.05)
        ax.set_ylim(-0.5, len(scenarios) - 0.5)
        
        # Add time markers - same style as RL visualizer
        # time_markers = np.arange(0, max_end_time, 300)  # Every 5 minutes
        # for marker in time_markers:
        #     ax.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        #     ax.text(marker, len(scenarios)-0.2, f"{marker/60:.0f}min", 
        #            rotation=90, ha='right', va='top', fontsize=10)
        
        # Create detailed legend - same style as RL visualizer
        legend_elements = [patches.Patch(color=color, label=f"{attack_type.replace('_', ' ').title()}") 
                          for attack_type, color in self.attack_colors.items() 
                          if attack_type != 'default' and any(s.get('TYPE', s.get('ATTACK_TYPE', '')) == attack_type for s in scenarios)]
        if legend_elements:
            ax.legend(handles=legend_elements, fontsize=18)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Strategic Gantt chart saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"gemini_strategic_gantt_{timestamp}.pdf"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"💾 Strategic Gantt chart saved to: {default_path}")
            
        plt.show()
    
    def create_coordination_analysis(self, scenarios, save_path=None):
        """Create detailed coordination analysis"""
        if not scenarios:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coordination types distribution
        coord_types = [s.get('COORDINATION_TYPE', 'unknown') for s in scenarios]
        coord_counts = pd.Series(coord_types).value_counts()
        
        ax1.pie(coord_counts.values, labels=coord_counts.index, autopct='%1.1f%%')
        ax1.set_title("Coordination Types Distribution", fontsize=14, fontweight='bold')
        
        # Attack types combination
        all_attack_types = []
        for scenario in scenarios:
            combined_types = scenario.get('COMBINED_ATTACK_TYPES', [])
            all_attack_types.extend(combined_types)
        
        attack_counts = pd.Series(all_attack_types).value_counts()
        ax2.bar(range(len(attack_counts)), attack_counts.values)
        ax2.set_title("Combined Attack Types Usage", fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(attack_counts)))
        ax2.set_xticklabels(attack_counts.index, rotation=45, ha='right')
        
        # Success rate vs Impact
        success_rates = [s.get('SUCCESS_RATE', 0) for s in scenarios]
        impact_factors = [s.get('IMPACT_FACTOR', 0) for s in scenarios]
        attack_types = [s.get('TYPE', s.get('ATTACK_TYPE', 'unknown')) for s in scenarios]
        
        colors = [self.attack_colors.get(attack_type, self.attack_colors['default']) for attack_type in attack_types]
        ax3.scatter(success_rates, impact_factors, c=colors, s=100, alpha=0.7)
        ax3.set_xlabel("Success Rate")
        ax3.set_ylabel("Impact Factor")
        ax3.set_title("Success Rate vs Impact Analysis", fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Duration vs Magnitude
        durations = [s.get('DURATION', 0) for s in scenarios]
        magnitudes = [s.get('ATTACK_MAGNITUDE', 0) for s in scenarios]
        
        ax4.scatter(durations, magnitudes, c=colors, s=100, alpha=0.7)
        ax4.set_xlabel("Duration (seconds)")
        ax4.set_ylabel("Attack Magnitude")
        ax4.set_title("Duration vs Magnitude Analysis", fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"gemini_coordination_analysis_{timestamp}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def generate_strategic_report(self, scenarios):
        """Generate strategic analysis report"""
        if not scenarios:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"gemini_strategic_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("GEMINI STRATEGIC ATTACK ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Strategic Scenarios: {len(scenarios)}\n\n")
            
            # Strategic overview
            f.write("STRATEGIC OVERVIEW\n")
            f.write("-" * 30 + "\n")
            total_duration = sum(s.get('DURATION', 0) for s in scenarios)
            avg_impact = sum(s.get('IMPACT_FACTOR', 0) for s in scenarios) / len(scenarios)
            avg_success = sum(s.get('SUCCESS_RATE', 0) for s in scenarios) / len(scenarios)
            
            f.write(f"Total Campaign Duration: {total_duration} seconds ({total_duration/60:.1f} minutes)\n")
            f.write(f"Average Impact Factor: {avg_impact:.3f}\n")
            f.write(f"Average Success Rate: {avg_success:.3f}\n\n")
            
            # Wave-by-wave analysis
            f.write("WAVE-BY-WAVE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for i, scenario in enumerate(scenarios, 1):
                f.write(f"Wave {i}: {scenario.get('SCENARIO_NAME', 'Unknown')}\n")
                f.write(f"  Strategic Goal: {scenario.get('STRATEGIC_GOAL', 'Not specified')}\n")
                f.write(f"  Coordination: {scenario.get('COORDINATION_TYPE', 'Unknown')}\n")
                f.write(f"  Target Systems: {scenario.get('TARGET_SYSTEMS', [])}\n")
                f.write(f"  Duration: {scenario.get('DURATION', 0)}s\n")
                f.write(f"  Impact Factor: {scenario.get('IMPACT_FACTOR', 0):.3f}\n")
                f.write(f"  Success Rate: {scenario.get('SUCCESS_RATE', 0):.3f}\n")
                f.write(f"  Combined Attacks: {scenario.get('COMBINED_ATTACK_TYPES', [])}\n\n")
        
        print(f"📄 Strategic report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self, file_path=None):
        """Run complete Gemini scenarios analysis"""
        print("🧠 Starting Gemini Strategic Attack Analysis...")
        
        # Find or use provided file
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = self.find_latest_gemini_file()
            
        if not self.file_path:
            print("❌ No Gemini scenarios file found!")
            return
            
        # Parse scenarios
        scenarios = self.parse_gemini_scenarios(self.file_path)
        if not scenarios:
            print("❌ No scenarios found!")
            return
            
        # Generate visualizations
        print("\n📊 Creating strategic timeline...")
        self.create_strategic_timeline(scenarios)
        
        print("\n📊 Creating strategic Gantt chart...")
        self.create_strategic_gantt_chart(scenarios)
        
        print("\n📊 Creating coordination analysis...")
        self.create_coordination_analysis(scenarios)
        
        print("\n📄 Generating strategic report...")
        self.generate_strategic_report(scenarios)
        
        print("\n✅ Gemini analysis complete!")
        return scenarios

def main():
    """Main function"""
    print("🧠 Gemini Attack Scenarios Visualizer")
    print("=" * 50)
    
    visualizer = GeminiAttackVisualizer()
    visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()
