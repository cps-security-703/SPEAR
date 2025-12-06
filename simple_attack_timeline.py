#!/usr/bin/env python3
"""
Simple RL Attack Timeline Visualizer
====================================
A focused visualization showing system-wise attack timelines and durations.

Usage: python simple_attack_timeline.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_latest_rl_file():
    """Find and parse the latest RL feedback file"""
    logs_dir = "attack_scenarios_logs"
    
    # Find latest RL feedback file
    rl_files = [f for f in os.listdir(logs_dir) if f.startswith('rl_feedback_to_gemini_')]
    if not rl_files:
        print("❌ No RL feedback files found!")
        return None
        
    latest_file = os.path.join(logs_dir, sorted(rl_files)[-1])
    print(f"📁 Using: {latest_file}")
    
    # Parse attacks
    attacks = []
    current_attack = {}
    
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        
    in_attack_section = False
    for line in lines:
        line = line.strip()
        
        if line == "RL AGENT ATTACK DATA":
            in_attack_section = True
            continue
        elif line.startswith("RL EPISODE PERFORMANCE METRICS"):
            break
            
        if in_attack_section and line.startswith("ATTACK #"):
            if current_attack:
                attacks.append(current_attack.copy())
            current_attack = {}
            
        elif ':' in line and current_attack is not None:
            key, value = line.split(':', 1)
            key, value = key.strip(), value.strip()
            
            # Special handling for TYPE - only keep the FIRST occurrence (primary attack type)
            if key == 'TYPE' and 'TYPE' in current_attack:
                continue  # Skip subsequent TYPE fields
            
            # Convert key values
            if key in ['START_TIME', 'DURATION', 'MAGNITUDE', 'STEALTH_LEVEL']:
                try:
                    value = float(value)
                except:
                    pass
            elif key == 'TARGET_SYSTEM':
                try:
                    value = int(value)
                except:
                    pass
                    
            current_attack[key] = value
    
    if current_attack:
        attacks.append(current_attack)
        
    # Filter valid attacks
    valid_attacks = []
    for attack in attacks:
        if all(k in attack for k in ['TYPE', 'TARGET_SYSTEM', 'START_TIME', 'DURATION']):
            valid_attacks.append({
                'attack_type': attack['TYPE'],
                'target_system': attack['TARGET_SYSTEM'],
                'start_time': attack['START_TIME'],
                'duration': attack['DURATION'],
                'magnitude': attack.get('MAGNITUDE', 1.0),
                'stealth': attack.get('STEALTH_LEVEL', 1.0)
            })
    
    print(f"✅ Parsed {len(valid_attacks)} valid attacks")
    return valid_attacks

def create_system_timeline(attacks):
    """Create a clean system-wise timeline visualization"""
    if not attacks:
        print("❌ No attacks to visualize")
        return
        
    df = pd.DataFrame(attacks)
    
    # Attack type colors
    colors = {
        'charging_hijacking': '#FF6B6B',
        'voltage_manipulation': '#4ECDC4', 
        'frequency_attack': '#45B7D1',
        'power_disruption': '#FFA07A',
        'load_manipulation': '#98D8C8',
        'power_manipulation': '#85C1E9'
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Main timeline
    systems = sorted(df['target_system'].unique())
    y_positions = {sys: i for i, sys in enumerate(systems)}
    
    for _, attack in df.iterrows():
        y_pos = y_positions[attack['target_system']]
        color = colors.get(attack['attack_type'], '#CCCCCC')
        
        # Draw attack bar
        rect = patches.Rectangle(
            (attack['start_time'], y_pos - 0.3),
            attack['duration'], 0.6,
            facecolor=color, alpha=0.8,
            edgecolor='black', linewidth=1
        )
        ax1.add_patch(rect)
        
        # Add label for longer attacks
        if attack['duration'] > 30:
            ax1.text(
                attack['start_time'] + attack['duration']/2, y_pos,
                attack['attack_type'].replace('_', '\n'),
                ha='center', va='center', fontsize=8, fontweight='bold'
            )
    
    # Customize timeline
    ax1.set_title("RL Agent Attacks: System-wise Timeline", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Target System", fontsize=12)
    ax1.set_yticks(range(len(systems)))
    ax1.set_yticklabels([f"System {s}" for s in systems])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, df['start_time'].max() + df['duration'].max())
    
    # Add time markers (every 5 minutes)
    max_time = df['start_time'].max() + df['duration'].max()
    time_markers = np.arange(0, max_time, 300)
    for t in time_markers:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.4)
        ax1.text(t, len(systems)-0.1, f"{t/60:.0f}min", rotation=90, ha='right', fontsize=9)
    
    # Legend
    legend_elements = [patches.Patch(color=color, label=attack_type.replace('_', ' ').title()) 
                      for attack_type, color in colors.items() 
                      if attack_type in df['attack_type'].values]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Attack summary bar chart
    attack_counts = df.groupby(['target_system', 'attack_type']).size().unstack(fill_value=0)
    attack_counts.plot(kind='bar', ax=ax2, color=[colors.get(col, '#CCCCCC') for col in attack_counts.columns])
    ax2.set_title("Attack Count by System and Type", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Target System", fontsize=12)
    ax2.set_ylabel("Number of Attacks", fontsize=12)
    ax2.legend(title="Attack Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simple_attack_timeline_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Timeline saved to: {filename}")
    
    plt.show()
    
    # Print summary
    print("\n📊 ATTACK SUMMARY:")
    print("-" * 40)
    print(f"Total Attacks: {len(df)}")
    print(f"Target Systems: {sorted(df['target_system'].unique())}")
    print(f"Attack Types: {list(df['attack_type'].unique())}")
    print(f"Time Range: {df['start_time'].min():.0f}s - {(df['start_time'] + df['duration']).max():.0f}s")
    print(f"Total Duration: {df['duration'].sum():.1f}s")
    
    print("\nAttacks per System:")
    for sys in sorted(df['target_system'].unique()):
        sys_attacks = df[df['target_system'] == sys]
        print(f"  System {sys}: {len(sys_attacks)} attacks, {sys_attacks['duration'].sum():.1f}s total")

def main():
    """Main function"""
    print("🎯 Simple RL Attack Timeline Visualizer")
    print("=" * 50)
    
    # Parse data
    attacks = parse_latest_rl_file()
    if not attacks:
        return
        
    # Create visualization
    create_system_timeline(attacks)

if __name__ == "__main__":
    main()
