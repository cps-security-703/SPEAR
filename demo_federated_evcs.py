#!/usr/bin/env python3

"""
Demonstration of Federated EVCS System with:
1. Distributed PINN optimization (each system has its own trained model)
2. Constrained RL attacks (realistic load injection with physical limits)
3. Global federated optimizer (customer redirection and load balancing)
"""

import numpy as np
import matplotlib.pyplot as plt
from federated_evcs_integration import FederatedEVCSSystem
from global_federated_optimizer import CustomerRequest
import time
import warnings
warnings.filterwarnings('ignore')

def demonstrate_federated_evcs():
    """Demonstrate the complete federated EVCS system"""
    print("=" * 80)
    print(" FEDERATED EVCS SYSTEM DEMONSTRATION")
    print("=" * 80)
    print(" Features:")
    print(" 1. Federated PINN Training (6 distribution systems)")
    print(" 2. Constrained RL Attacks (realistic load injection)")
    print(" 3. Global Optimization (customer redirection)")
    print(" 4. Anomaly Detection (physical constraint validation)")
    print("=" * 80)
    
    # Initialize federated system
    print("\n🏗️ Initializing Federated EVCS System...")
    federated_system = FederatedEVCSSystem(num_distribution_systems=6)
    
    # Demonstrate federated features
    federated_system.demonstrate_federated_features()
    
    # User choice for training
    print("\n" + "="*60)
    print(" TRAINING OPTIONS")
    print("="*60)
    print("1 - Quick demo (reduced training for fast demonstration)")
    print("2 - Full training (comprehensive federated PINN training)")
    print("3 - Skip training (use pre-trained models if available)")
    
    choice = input("\nSelect option (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n🚀 Quick Demo Mode - Reduced Training")
        training_results = federated_system.train_federated_pinn_models(n_samples_per_system=500)
        
    elif choice == "2":
        print("\n🔬 Full Training Mode - Comprehensive Training")
        training_results = federated_system.train_federated_pinn_models(n_samples_per_system=2000)
        
    elif choice == "3":
        print("\n📂 Attempting to load pre-trained models...")
        try:
            success = federated_system.federated_pinn_manager.load_federated_models('federated_models')
            if success:
                federated_system.federated_training_complete = True
                print("✅ Pre-trained models loaded successfully")
                training_results = {"loaded_from_file": True}
            else:
                print("❌ No pre-trained models found. Running quick training...")
                training_results = federated_system.train_federated_pinn_models(n_samples_per_system=500)
        except:
            print("❌ Failed to load pre-trained models. Running quick training...")
            training_results = federated_system.train_federated_pinn_models(n_samples_per_system=500)
    else:
        print("Invalid choice. Running quick demo...")
        training_results = federated_system.train_federated_pinn_models(n_samples_per_system=500)
    
    # Setup co-simulation
    print("\n🔧 Setting up hierarchical co-simulation...")
    setup_success = federated_system.setup_hierarchical_cosimulation()
    
    if not setup_success:
        print("❌ Failed to setup co-simulation")
        return
    
    # Generate load profile for simulation
    print("\n📊 Generating daily load profile...")
    times = np.linspace(0, 480, 481)  # 8 minutes, 1-second resolution
    
    # Create realistic daily load variation
    load_multipliers = []
    for t in times:
        # Convert to hour of day (8 minutes = 0.133 hours, scale to 24 hours)
        hour_of_day = (t / 480.0) * 24
        
        # Multi-peak load profile
        morning_peak = 0.8 * np.exp(-((hour_of_day - 8) / 2)**2)
        evening_peak = 1.0 * np.exp(-((hour_of_day - 19) / 2.5)**2)
        night_base = 0.3
        
        load_mult = morning_peak + evening_peak + night_base + 0.1 * np.sin(hour_of_day * np.pi / 12)
        load_multipliers.append(max(0.2, load_mult))
    
    load_multipliers = np.array(load_multipliers)
    
    # Add some customer requests
    print("\n👥 Adding customer charging requests...")
    sample_customers = [
        CustomerRequest(
            customer_id="CUST_001",
            requested_power=45.0,
            requested_duration=1.5,
            soc_current=0.25,
            soc_target=0.85,
            urgency_level=4,
            max_travel_distance=8.0
        ),
        CustomerRequest(
            customer_id="CUST_002", 
            requested_power=30.0,
            requested_duration=1.0,
            soc_current=0.40,
            soc_target=0.90,
            urgency_level=2,
            max_travel_distance=5.0
        ),
        CustomerRequest(
            customer_id="CUST_003",
            requested_power=60.0,
            requested_duration=2.0,
            soc_current=0.15,
            soc_target=0.80,
            urgency_level=5,
            max_travel_distance=12.0
        )
    ]
    
    for customer in sample_customers:
        federated_system.global_optimizer.add_customer_request(customer)
    
    print(f"✅ Added {len(sample_customers)} customer requests")
    
    # Run federated simulation
    print("\n🚀 Running federated simulation with constrained RL attacks...")
    simulation_results = federated_system.run_federated_simulation_with_attacks(
        load_profile_data=(times, load_multipliers)
    )
    
    # Analyze results
    print("\n📊 Analyzing simulation results...")
    analyze_federated_results(simulation_results, federated_system)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_federated_visualizations(simulation_results, federated_system, times, load_multipliers)
    
    print("\n✅ Federated EVCS demonstration complete!")
    return simulation_results, federated_system

def analyze_federated_results(simulation_results: dict, federated_system: FederatedEVCSSystem):
    """Analyze and print federated simulation results"""
    print("\n" + "="*60)
    print(" FEDERATED SIMULATION ANALYSIS")
    print("="*60)
    
    # Training results
    if 'training_time' in simulation_results:
        print(f"📚 Training Time: {simulation_results['training_time']:.1f} seconds")
    
    # Customer allocation results
    allocations = simulation_results.get('customer_allocations', [])
    successful_allocations = sum(1 for alloc in allocations if alloc['success'])
    
    print(f"\n👥 Customer Allocations:")
    print(f"   Total requests: {len(allocations)}")
    print(f"   Successful allocations: {successful_allocations}")
    print(f"   Success rate: {successful_allocations/max(1, len(allocations))*100:.1f}%")
    
    for alloc in allocations:
        if alloc['success']:
            details = alloc['allocation_details']
            station_info = details['station_info']
            print(f"   Customer {alloc['customer_id']}: System {alloc['allocated_system']}")
            print(f"     Station: {alloc['allocated_station']}")
            print(f"     Distance: {station_info['distance']:.1f} km")
            print(f"     Wait time: {station_info['wait_time']:.1f} min")
            if details['pinn_success']:
                pinn_opt = details['pinn_optimization']
                print(f"     PINN: {pinn_opt.get('voltage_ref', 0):.0f}V, "
                      f"{pinn_opt.get('current_ref', 0):.0f}A, {pinn_opt.get('power_ref', 0):.0f}kW")
    
    # Attack analysis
    attack_events = simulation_results.get('attack_events', [])
    constraint_violations = simulation_results.get('constraint_violations', [])
    
    print(f"\n⚡ RL Attack Analysis:")
    print(f"   Attack events: {len(attack_events)}")
    print(f"   Constraint violations: {len(constraint_violations)}")
    print(f"   Constraint compliance: {(len(attack_events)/(len(attack_events)+len(constraint_violations))*100) if (len(attack_events)+len(constraint_violations)) > 0 else 100:.1f}%")
    
    for event in attack_events:
        print(f"   System {event['system_id']}: {event['total_steps']} steps, "
              f"stealth score: {event['stealth_score']:.2f}")
    
    for violation in constraint_violations:
        print(f"   ❌ System {violation['system_id']}: {violation['reason']}")
    
    # System performance
    performance = simulation_results.get('system_performance', {})
    if performance:
        print(f"\n📊 System Performance:")
        print(f"   Load balance score: {performance.get('load_balance_score', 0):.2f}")
        print(f"   Customer satisfaction: {performance.get('customer_satisfaction_score', 0):.2f}")
        print(f"   Grid stability: {performance.get('grid_stability_score', 0):.2f}")
    
    # Federated PINN status
    federated_status = federated_system.get_comprehensive_status()
    print(f"\n🔬 Federated PINN Status:")
    print(f"   Training complete: {federated_status['federated_training_complete']}")
    print(f"   Communication rounds: {federated_status['federated_pinn_status']['communication_rounds']}")
    print(f"   Active attacks: {federated_status['attack_system_status']['num_systems']} systems monitored")

def create_federated_visualizations(simulation_results: dict, federated_system: FederatedEVCSSystem, 
                                  times: np.ndarray, load_multipliers: np.ndarray):
    """Create visualizations for federated simulation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Federated EVCS System Analysis', fontsize=16, fontweight='bold')
    
    # 1. Load Profile with Attack Events
    axes[0, 0].plot(times/60, load_multipliers, 'b-', linewidth=2, label='Daily Load Profile')
    
    # Mark attack events
    attack_events = simulation_results.get('attack_events', [])
    for i, event in enumerate(attack_events):
        attack_time = 60 + i * 30  # Attacks start at 60s, spaced 30s apart
        axes[0, 0].axvline(x=attack_time/60, color='red', linestyle='--', alpha=0.7, 
                          label=f"Attack System {event['system_id']}" if i == 0 else "")
    
    axes[0, 0].set_title('Load Profile with Attack Events')
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Load Multiplier')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Customer Allocation Distribution
    allocations = simulation_results.get('customer_allocations', [])
    if allocations:
        system_counts = {}
        for alloc in allocations:
            if alloc['success']:
                sys_id = alloc['allocated_system']
                system_counts[sys_id] = system_counts.get(sys_id, 0) + 1
        
        if system_counts:
            systems = list(system_counts.keys())
            counts = list(system_counts.values())
            
            axes[0, 1].bar(systems, counts, color='green', alpha=0.7)
            axes[0, 1].set_title('Customer Allocation by System')
            axes[0, 1].set_xlabel('Distribution System')
            axes[0, 1].set_ylabel('Number of Customers')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Attack Stealth Scores
    if attack_events:
        stealth_scores = [event.get('stealth_score', 0.5) for event in attack_events]
        system_ids = [event['system_id'] for event in attack_events]
        
        axes[0, 2].scatter(system_ids, stealth_scores, c=stealth_scores, 
                          cmap='RdYlGn', s=100, alpha=0.7)
        axes[0, 2].set_title('Attack Stealth Scores')
        axes[0, 2].set_xlabel('Distribution System')
        axes[0, 2].set_ylabel('Stealth Score (0-1)')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. System Utilization
    global_status = federated_system.global_optimizer.get_global_system_status()
    system_summary = global_status.get('system_summary', {})
    
    if system_summary:
        systems = list(system_summary.keys())
        utilizations = [system_summary[sys]['utilization'] for sys in systems]
        
        colors = ['red' if u > 0.8 else 'yellow' if u > 0.6 else 'green' for u in utilizations]
        axes[1, 0].bar(systems, utilizations, color=colors, alpha=0.7)
        axes[1, 0].set_title('System Utilization')
        axes[1, 0].set_xlabel('Distribution System')
        axes[1, 0].set_ylabel('Utilization (0-1)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Constraint Violations
    constraint_violations = simulation_results.get('constraint_violations', [])
    violation_types = {}
    
    for violation in constraint_violations:
        reason = violation['reason']
        violation_types[reason] = violation_types.get(reason, 0) + 1
    
    if violation_types:
        reasons = list(violation_types.keys())
        counts = list(violation_types.values())
        
        axes[1, 1].pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Constraint Violation Types')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Constraint\nViolations', 
                       ha='center', va='center', fontsize=14, color='green')
        axes[1, 1].set_title('Constraint Violations')
    
    # 6. Performance Metrics
    performance = simulation_results.get('system_performance', {})
    if performance:
        metrics = ['Load Balance', 'Customer Satisfaction', 'Grid Stability']
        values = [
            performance.get('load_balance_score', 0),
            performance.get('customer_satisfaction_score', 0),
            performance.get('grid_stability_score', 0)
        ]
        
        bars = axes[1, 2].bar(metrics, values, color=['blue', 'orange', 'purple'], alpha=0.7)
        axes[1, 2].set_title('System Performance Metrics')
        axes[1, 2].set_ylabel('Score (0-1)')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create federated training visualization
    create_federated_training_visualization(federated_system)

def create_federated_training_visualization(federated_system: FederatedEVCSSystem):
    """Create visualization for federated training process"""
    
    federated_status = federated_system.federated_pinn_manager.get_federated_status()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Federated PINN Training Analysis', fontsize=14, fontweight='bold')
    
    # 1. Training Status by System
    systems = list(range(1, federated_system.num_distribution_systems + 1))
    training_status = [1 if sys in federated_system.federated_pinn_manager.local_models else 0 
                      for sys in systems]
    
    colors = ['green' if status else 'red' for status in training_status]
    axes[0].bar(systems, training_status, color=colors, alpha=0.7)
    axes[0].set_title('Federated PINN Training Status')
    axes[0].set_xlabel('Distribution System')
    axes[0].set_ylabel('Training Complete (1=Yes, 0=No)')
    axes[0].set_ylim(0, 1.2)
    axes[0].grid(True, alpha=0.3)
    
    # Add text labels
    for i, (sys, status) in enumerate(zip(systems, training_status)):
        axes[0].text(sys, status + 0.05, 'Trained' if status else 'Not Trained',
                    ha='center', va='bottom', fontweight='bold')
    
    # 2. Communication Rounds
    comm_rounds = federated_status.get('communication_rounds', 0)
    max_rounds = federated_system.federated_pinn_manager.config.global_rounds
    
    axes[1].pie([comm_rounds, max_rounds - comm_rounds], 
               labels=[f'Completed ({comm_rounds})', f'Remaining ({max_rounds - comm_rounds})'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'Federated Communication Progress\n({comm_rounds}/{max_rounds} rounds)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the demonstration
    results, system = demonstrate_federated_evcs()
    
    print("\n" + "="*80)
    print(" DEMONSTRATION COMPLETE")
    print("="*80)
    print(" Key Improvements Implemented:")
    print(" ✅ Federated PINN: Each system trains its own model")
    print(" ✅ Constrained RL: Realistic attacks (max 50kW vs 3000-10000MW)")
    print(" ✅ Gradual Injection: Step-by-step attack progression")
    print(" ✅ Anomaly Detection: Physical constraint validation")
    print(" ✅ Global Optimization: Customer redirection across systems")
    print(" ✅ Load Balancing: Federated model coordination")
    print("="*80)
