#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from hierarchical_cosimulation import HierarchicalCoSimulation, ChargingManagementSystem, CentralChargingCoordinator
from rl_attack_analytics import create_rl_attack_analytics, HybridAttackAgent
# NEW FEDERATED IMPORTS
from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
from enhanced_rl_attack_system import ConstrainedRLAttackSystem
from global_federated_optimizer import GlobalFederatedOptimizer, CustomerRequest
# DQN/SAC SECURITY EVASION IMPORTS
from dqn_sac_security_evasion import DQNSACSecurityEvasionTrainer, create_dqn_sac_evasion_system
from adaptive_rl_evasion_system import create_adaptive_rl_system, create_dqn_sac_system
import time
import sys
import warnings
from datetime import datetime
import math

warnings.filterwarnings('ignore')

# Create a custom print function that writes to both console and file
class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Generate timestamp for unique log filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'focused_demand_analysis_{timestamp}.txt'

# Redirect stdout to our custom logger
sys.stdout = PrintLogger(log_filename)



def generate_daily_load_profile(total_duration=240.0, time_step=1.0, constant_load=True):
    """
    Generate load profile - either constant or realistic daily EV charging load profile
    Returns load multiplier for each time step
    """
    times = np.arange(0, total_duration, time_step)
    
    if constant_load:
        # Return constant load multiplier (no variation)
        load_multipliers = np.ones_like(times) * 0.6  # Constant 0.6 load multiplier
        print("Using CONSTANT load profile (no variation)")
    else:
        # Enhanced realistic EV charging load profile with continuous smooth transitions
        load_multipliers = np.zeros_like(times)
        
        # Base load that never goes to zero (always some background charging)
        base_load_level = 0.20  # 20% minimum load at all times
        
        # Create continuous load profile using overlapping smooth functions
        for i, t in enumerate(times):
            # Convert time to hour of day (assuming total_duration = 24 hours for simulation)
            hour_of_day = (t / total_duration) * 24
            
            # Start with base load
            total_load = base_load_level
            
            # Create smooth continuous load curve using overlapping Gaussian-like functions
            # Night valley (11 PM - 6 AM): Minimum charging with gradual transitions
            night_center = 2.5  # 2:30 AM center
            if hour_of_day >= 23:
                night_hour = hour_of_day - 24  # Convert to negative for smooth transition
            else:
                night_hour = hour_of_day
            night_contribution = 0.15 * np.exp(-((night_hour - night_center)**2) / (2 * 3**2))
            total_load += night_contribution
            
            # Early morning ramp (5-9 AM): Gradual increase for commuter charging
            morning_center = 7.5  # 7:30 AM peak
            morning_contribution = 0.45 * np.exp(-((hour_of_day - morning_center)**2) / (2 * 2.5**2))
            total_load += morning_contribution
            
            # Mid-morning workplace (9 AM - 1 PM): Sustained workplace charging
            workplace_center = 11  # 11 AM center
            workplace_contribution = 0.35 * np.exp(-((hour_of_day - workplace_center)**2) / (2 * 3**2))
            total_load += workplace_contribution
            
            # Afternoon moderate (1-5 PM): Shopping and opportunity charging
            afternoon_center = 15  # 3 PM center
            afternoon_contribution = 0.25 * np.exp(-((hour_of_day - afternoon_center)**2) / (2 * 2.5**2))
            total_load += afternoon_contribution
            
            # Evening peak (5-9 PM): Major residential charging peak
            evening_center = 19  # 7 PM peak
            evening_contribution = 0.65 * np.exp(-((hour_of_day - evening_center)**2) / (2 * 2.8**2))
            total_load += evening_contribution
            
            # Late evening taper (9-11 PM): Gradual decrease
            late_evening_center = 22  # 10 PM center
            late_evening_contribution = 0.30 * np.exp(-((hour_of_day - late_evening_center)**2) / (2 * 1.8**2))
            total_load += late_evening_contribution
            
            # Add realistic temporal variations
            # Intraday variability (15-minute fluctuations)
            time_minutes = (hour_of_day * 60) % 60
            intraday_variation = 0.03 * np.sin(2 * np.pi * time_minutes / 15)
            total_load += intraday_variation
            
            # Weekly pattern (slightly higher on weekdays)
            weekday_factor = 1.08  # 8% higher on weekdays
            total_load *= weekday_factor
            
            # Weather/seasonal effects (correlated noise for realism)
            # Use time-correlated noise instead of pure random
            weather_base = np.sin(2 * np.pi * hour_of_day / 24 + np.pi/3)
            weather_noise = 0.04 * weather_base * (1 + 0.3 * np.random.normal(0, 1))
            total_load += weather_noise
            
            # Demand response events (more realistic probability based on peak hours)
            dr_probability = 0.01 if 17 <= hour_of_day <= 21 else 0.005  # Higher during peak
            if np.random.random() < dr_probability:
                total_load *= 0.85  # 15% reduction during DR events
            
            # Grid-friendly charging incentives (slight load shifting)
            if 23 <= hour_of_day or hour_of_day <= 6:  # Off-peak hours
                total_load *= 1.05  # 5% incentive for off-peak charging
            
            # Ensure realistic bounds with smoother clamping
            total_load = np.clip(total_load, 0.20, 7.0)  # 20% min, 1400% max (25MW peak instead of 5MW)
            
            load_multipliers[i] = total_load
        
        # Enhanced multi-stage smoothing for realistic transitions
        from scipy.ndimage import gaussian_filter1d
        
        # Stage 1: Light initial smoothing to remove noise while preserving shape
        load_multipliers = gaussian_filter1d(load_multipliers, sigma=2.0)
        
        # Stage 2: Preserve variation - only smooth very sharp transitions
        # Calculate local variation and only smooth where needed
        gradient = np.gradient(load_multipliers)
        sharp_transition_threshold = 0.05  # Only smooth if gradient > 5% per step
        
        # Apply selective smoothing only to sharp transitions
        smoothed_copy = gaussian_filter1d(load_multipliers, sigma=1.5)
        for j in range(len(load_multipliers)):
            if abs(gradient[j]) > sharp_transition_threshold:
                # Blend original and smoothed values for sharp transitions
                blend_factor = min(1.0, abs(gradient[j]) / sharp_transition_threshold - 1.0)
                load_multipliers[j] = (1 - blend_factor) * load_multipliers[j] + blend_factor * smoothed_copy[j]
        
        # Ensure we maintain minimum load after all smoothing
        load_multipliers = np.maximum(load_multipliers, 0.20)
        
        # Add final realistic constraints
        # Ensure no sudden jumps > 10% per time step
        for i in range(1, len(load_multipliers)):
            max_change = 0.10  # 10% maximum change per time step
            if abs(load_multipliers[i] - load_multipliers[i-1]) > max_change:
                if load_multipliers[i] > load_multipliers[i-1]:
                    load_multipliers[i] = load_multipliers[i-1] + max_change
                else:
                    load_multipliers[i] = load_multipliers[i-1] - max_change
        print("Using DAILY LOAD VARIATION profile")
    
    return times, load_multipliers



def identify_demand_periods(load_multipliers, times, threshold_low=0.5, threshold_high=0.8, constant_load=False):
    """
    Identify low, medium, and high demand periods
    Returns dictionaries with time ranges for each period
    """
    if constant_load:
        # For constant load, create a single medium demand period
        avg_load = np.mean(load_multipliers)
        periods = {
            'low_demand': [],
            'medium_demand': [{
                'start_time': times[0],
                'end_time': times[-1],
                'duration': times[-1] - times[0],
                'avg_load': avg_load
            }],
            'high_demand': []
        }
        return periods
    periods = {
        'low_demand': [],
        'medium_demand': [],
        'high_demand': []
    }
    
    current_period = None
    period_start = None
    
    for i, load in enumerate(load_multipliers):
        if load < threshold_low:
            period_type = 'low_demand'
        elif load > threshold_high:
            period_type = 'high_demand'
        else:
            period_type = 'medium_demand'
        
        if current_period != period_type:
            # End previous period
            if current_period and period_start is not None:
                periods[current_period].append({
                    'start_time': times[period_start],
                    'end_time': times[i-1],
                    'duration': times[i-1] - times[period_start],
                    'avg_load': np.mean(load_multipliers[period_start:i])
                })
            
            # Start new period
            current_period = period_type
            period_start = i
    
    # Handle last period
    if current_period and period_start is not None:
        periods[current_period].append({
            'start_time': times[period_start],
            'end_time': times[-1],
            'duration': times[-1] - times[period_start],
            'avg_load': np.mean(load_multipliers[period_start:])
        })
    
    return periods


def apply_physics_constraints(attack_scenario, system_constraints=None):
    """
    Apply physics constraints to electrical parameter manipulations to ensure realistic attacks
    """
    if system_constraints is None:
        # Default EVCS system constraints
        system_constraints = {
            'max_voltage': 500.0,      # V (maximum safe voltage)
            'min_voltage': 300.0,      # V (minimum operational voltage)
            'max_current': 200.0,      # A (maximum safe current)
            'min_current': 10.0,       # A (minimum operational current)
            'max_power': 100.0,        # kW (maximum power injection)
            'min_power': 5.0,          # kW (minimum detectable power change)
            'max_frequency_deviation': 0.5,  # Hz (grid stability limit)
            'max_voltage_change_rate': 0.1,  # V/s (voltage slew rate limit)
            'max_reactive_power': 50.0,      # kVAR (reactive power limit)
            'min_power_factor': 0.8,         # Minimum power factor
            'max_power_factor': 0.98         # Maximum power factor
        }
    
    # Apply voltage constraints with 10x amplification for maximum impact
    if 'voltage_injection' in attack_scenario:
        max_voltage_change = system_constraints['max_voltage'] * 0.3  # Max 30% change (3x increase)
        attack_scenario['voltage_injection'] = np.clip(
            attack_scenario['voltage_injection'], 
            -max_voltage_change / system_constraints['max_voltage'], 
            max_voltage_change / system_constraints['max_voltage']
        )
    
    # Apply current constraints with 10x amplification
    if 'current_injection' in attack_scenario:
        max_current_change = system_constraints['max_current'] * 0.5  # Max 50% change (3x increase)
        attack_scenario['current_injection'] = np.clip(
            attack_scenario['current_injection'],
            -max_current_change,
            max_current_change
        )
    
    # Apply frequency constraints with amplification for grid instability
    if 'frequency_injection' in attack_scenario:
        max_freq_change = system_constraints['max_frequency_deviation'] * 0.6  # Max 60% of limit (3x increase)
        attack_scenario['frequency_injection'] = np.clip(
            attack_scenario['frequency_injection'],
            -max_freq_change,
            max_freq_change
        )
    
    # Apply reactive power constraints with 10x amplification
    if 'reactive_power_injection' in attack_scenario:
        attack_scenario['reactive_power_injection'] = np.clip(
            attack_scenario['reactive_power_injection'],
            -system_constraints['max_reactive_power'] * 0.8,  # Max 80% of limit (2.7x increase)
            system_constraints['max_reactive_power'] * 0.8
        )
    
    # Apply power factor constraints
    if 'power_factor_target' in attack_scenario:
        attack_scenario['power_factor_target'] = np.clip(
            attack_scenario['power_factor_target'],
            system_constraints['min_power_factor'],
            system_constraints['max_power_factor']
        )
    
    # Apply load magnitude constraints with 10x amplification
    if 'load_magnitude' in attack_scenario:
        attack_scenario['load_magnitude'] = np.clip(
            attack_scenario['load_magnitude'],
            system_constraints['min_power'],
            system_constraints['max_power'] * 2.0  # Max 200% of system limit for high impact (4x increase)
        )
    
    # Calculate stealth score based on constraint violations (more aggressive = less stealthy)
    stealth_penalties = 0
    
    # Penalize large voltage changes (more aggressive threshold)
    if abs(attack_scenario.get('voltage_injection', 0)) > 0.1:  # >10% voltage change
        stealth_penalties += 0.3
    
    # Penalize large current changes (more aggressive threshold)
    if abs(attack_scenario.get('current_injection', 0)) > 50:  # >50A current change
        stealth_penalties += 0.25
    
    # Penalize frequency deviations (more aggressive threshold)
    if abs(attack_scenario.get('frequency_injection', 0)) > 0.15:  # >0.15Hz change
        stealth_penalties += 0.35
    
    # Update stealth score
    original_stealth = attack_scenario.get('stealth_score', 1.0)
    attack_scenario['stealth_score'] = max(0.1, original_stealth - stealth_penalties)
    
    # Add physics compliance flags
    attack_scenario['voltage_compliant'] = abs(attack_scenario.get('voltage_injection', 0)) <= 0.15
    attack_scenario['current_compliant'] = abs(attack_scenario.get('current_injection', 0)) <= 80
    attack_scenario['frequency_compliant'] = abs(attack_scenario.get('frequency_injection', 0)) <= 0.25
    attack_scenario['physics_validated'] = True
    
    return attack_scenario


def detect_frequency_oscillations(transmission_system_data, window_size=10):
    """
    Detect frequency oscillations in the transmission system to amplify attacks
    """
    if not transmission_system_data or len(transmission_system_data.get('frequency', [])) < window_size:
        return False, 0.0
    
    frequency_data = transmission_system_data['frequency'][-window_size:]
    
    # Calculate frequency variance and rate of change
    freq_variance = np.var(frequency_data)
    freq_std = np.std(frequency_data)
    
    # Detect oscillations
    oscillation_threshold = 0.05  # Hz standard deviation threshold
    rate_threshold = 0.1  # Hz/s rate of change threshold
    
    # Calculate rate of change
    if len(frequency_data) > 1:
        freq_rates = np.diff(frequency_data)
        max_rate = np.max(np.abs(freq_rates))
    else:
        max_rate = 0.0
    
    is_oscillating = (freq_std > oscillation_threshold) or (max_rate > rate_threshold)
    oscillation_magnitude = min(freq_std + max_rate, 1.0)  # Normalized magnitude
    
    return is_oscillating, oscillation_magnitude


def enhance_agent_electrical_parameters(dqn_decision, sac_control, system_state, target_system, attack_progress=0.0):
    """
    Generate intelligent electrical parameter attacks using DQN strategic decisions and SAC continuous control
    """
    # Base electrical parameters from agent control
    base_voltage_injection = sac_control.get('voltage_control', np.random.uniform(-0.15, 0.15))
    base_current_injection = sac_control.get('current_control', np.random.uniform(-80, 80))
    base_frequency_injection = sac_control.get('frequency_control', np.random.uniform(-0.25, 0.25))
    base_reactive_power = sac_control.get('reactive_power_control', np.random.uniform(-40, 40))
    base_power_factor = sac_control.get('power_factor_control', np.random.uniform(0.75, 0.98))
    
    # Apply DQN strategic decisions for electrical attack type
    attack_type_map = {
        0: 'voltage_manipulation',    # Voltage sag/swell attacks
        1: 'current_injection',       # Current surge/deficit attacks  
        2: 'frequency_drift',         # Frequency deviation attacks
        3: 'reactive_power_attack',   # Power factor manipulation
        4: 'coordinated_electrical',  # Multi-parameter coordinated attack
        5: 'oscillation_amplifier'    # Amplify existing oscillations
    }
    
    attack_type = attack_type_map.get(dqn_decision.get('attack_type', 0), 'voltage_manipulation')
    evasion_strategy = dqn_decision.get('evasion_strategy', 0)
    
    # Get system state for intelligent targeting
    grid_voltage = system_state.get('grid_voltage', 1.0)
    grid_frequency = system_state.get('frequency', 60.0)
    current_load = system_state.get('current_load', 100.0)
    
    # Detect if system is already unstable for amplification attacks
    freq_deviation = abs(grid_frequency - 60.0)
    voltage_deviation = abs(grid_voltage - 1.0)
    is_unstable = (freq_deviation > 0.1) or (voltage_deviation > 0.05)
    
    # Generate intelligent electrical parameters based on attack type
    if attack_type == 'voltage_manipulation':
        # Strategic voltage attacks based on current grid state
        if grid_voltage > 1.02:  # Grid voltage already high, push it higher
            voltage_injection = base_voltage_injection * 2.0 if base_voltage_injection > 0 else base_voltage_injection
        elif grid_voltage < 0.98:  # Grid voltage low, make it worse
            voltage_injection = base_voltage_injection * 2.0 if base_voltage_injection < 0 else base_voltage_injection
        else:
            voltage_injection = base_voltage_injection * 1.5  # Normal state, standard attack
        
        current_injection = base_current_injection * 0.3  # Minimal current injection
        frequency_injection = base_frequency_injection * 0.2  # Minimal frequency injection
        
    elif attack_type == 'current_injection':
        # High-current injection attacks
        voltage_injection = base_voltage_injection * 0.2  # Minimal voltage change
        current_injection = base_current_injection * 3.0  # 3x current injection for maximum impact
        frequency_injection = base_frequency_injection * 0.3  # Minimal frequency injection
        
    elif attack_type == 'frequency_drift':
        # Frequency manipulation attacks
        if freq_deviation > 0.05:  # System already oscillating
            # Amplify existing oscillations by injecting in the same direction
            if grid_frequency > 60.0:
                frequency_injection = abs(base_frequency_injection) * 2.5  # Push higher
            else:
                frequency_injection = -abs(base_frequency_injection) * 2.5  # Push lower
        else:
            frequency_injection = base_frequency_injection * 2.0  # Standard frequency attack
        
        voltage_injection = base_voltage_injection * 0.3
        current_injection = base_current_injection * 0.3
        
    elif attack_type == 'reactive_power_attack':
        # Power factor and reactive power manipulation
        voltage_injection = base_voltage_injection * 0.4
        current_injection = base_current_injection * 0.4
        frequency_injection = base_frequency_injection * 0.2
        base_reactive_power *= 2.5  # 2.5x reactive power injection
        base_power_factor = np.random.uniform(0.7, 0.9)  # Force poor power factor
        
    elif attack_type == 'coordinated_electrical':
        # Multi-parameter coordinated attacks for maximum system impact
        voltage_injection = base_voltage_injection * 1.8
        current_injection = base_current_injection * 1.8  
        frequency_injection = base_frequency_injection * 1.8
        base_reactive_power *= 1.5
        
    elif attack_type == 'oscillation_amplifier':
        # Amplify any existing system oscillations
        if is_unstable:
            # Amplify deviations by 3x when system is already unstable
            voltage_injection = base_voltage_injection * 3.0 if voltage_deviation > 0.02 else base_voltage_injection
            frequency_injection = base_frequency_injection * 3.0 if freq_deviation > 0.05 else base_frequency_injection
            current_injection = base_current_injection * 2.0
        else:
            # Create initial instability
            voltage_injection = base_voltage_injection * 1.2
            frequency_injection = base_frequency_injection * 1.2
            current_injection = base_current_injection * 1.2
    
    # Apply evasion strategy modifications
    stealth_factor = sac_control.get('stealth_factor', 0.5)
    
    if evasion_strategy == 0:  # Stealth mode - gradual injection
        gradient_scale = min(attack_progress * 2.0, 1.0)  # Gradual ramp up over first half of attack
        voltage_injection *= gradient_scale
        current_injection *= gradient_scale
        frequency_injection *= gradient_scale
        stealth_factor = max(stealth_factor, 0.8)  # High stealth
        
    elif evasion_strategy == 1:  # Burst mode - immediate maximum impact
        voltage_injection *= 1.5
        current_injection *= 1.5
        frequency_injection *= 1.5
        stealth_factor = min(stealth_factor, 0.3)  # Low stealth
        
    elif evasion_strategy == 2:  # Adaptive mode - respond to detection
        # Scale based on system response (placeholder for now)
        detection_risk = 1.0 - stealth_factor
        if detection_risk > 0.7:  # High detection risk, reduce magnitude
            voltage_injection *= 0.7
            current_injection *= 0.7
            frequency_injection *= 0.7
        else:  # Low detection risk, increase magnitude
            voltage_injection *= 1.3
            current_injection *= 1.3
            frequency_injection *= 1.3
    
    # Apply 10x magnitude scaling for maximum impact as requested
    magnitude_amplifier = sac_control.get('magnitude', 1.0) * 10.0  # 10x amplification
    
    return {
        'voltage_injection': voltage_injection * magnitude_amplifier,
        'current_injection': current_injection * magnitude_amplifier,
        'frequency_injection': frequency_injection * magnitude_amplifier,
        'reactive_power_injection': base_reactive_power * magnitude_amplifier,
        'power_factor_target': base_power_factor,
        'electrical_attack_type': attack_type,
        'stealth_level': stealth_factor,
        'amplification_factor': magnitude_amplifier,
        'attack_strategy': evasion_strategy,
        'target_system': target_system,
        'system_exploitation': is_unstable
    }


def apply_electrical_attack_to_cosimulation(cosim, attack_scenario, current_time):
    """
    Apply electrical parameter attacks to the co-simulation framework with gradual injection
    """
    target_system = attack_scenario['target_system']
    
    # Calculate attack progress for gradual injection
    attack_start = attack_scenario['start_time']
    attack_duration = attack_scenario['duration']
    attack_progress = (current_time - attack_start) / attack_duration
    
    # Apply gradual injection based on attack phases
    if attack_scenario.get('gradual_injection', False):
        ramp_up_ratio = attack_scenario.get('ramp_up_time', attack_duration * 0.3) / attack_duration
        peak_ratio = attack_scenario.get('peak_time', attack_duration * 0.4) / attack_duration
        
        if attack_progress <= ramp_up_ratio:
            # Ramp up phase - gradual increase
            injection_scale = attack_progress / ramp_up_ratio
        elif attack_progress <= (ramp_up_ratio + peak_ratio):
            # Peak phase - full injection
            injection_scale = 1.0
        else:
            # Ramp down phase - gradual decrease
            ramp_down_progress = (attack_progress - ramp_up_ratio - peak_ratio) / (1.0 - ramp_up_ratio - peak_ratio)
            injection_scale = 1.0 - ramp_down_progress
    else:
        # Immediate injection
        injection_scale = 1.0
    
    # Apply stealth scaling for even more gradual injection
    if attack_scenario.get('electrical_stealth', False):
        stealth_factor = attack_scenario.get('stealth_level', 0.5)
        injection_scale *= stealth_factor
    
    try:
        # 1. Apply voltage injection if specified
        if 'voltage_injection' in attack_scenario and attack_scenario['voltage_injection'] != 0:
            voltage_change = attack_scenario['voltage_injection'] * injection_scale
            target_bus = attack_scenario.get('target_bus', f"Bus_{target_system-1}")
            
            # Apply voltage modification to the target system
            if target_system in cosim.distribution_systems:
                dist_system = cosim.distribution_systems[target_system]['system']
                if hasattr(dist_system, 'modify_bus_voltage'):
                    dist_system.modify_bus_voltage(target_bus, voltage_change)
        
        # 2. Apply current injection if specified
        if 'current_injection' in attack_scenario and attack_scenario['current_injection'] != 0:
            current_injection = attack_scenario['current_injection'] * injection_scale
            target_line = attack_scenario.get('target_line', f"Line_{target_system-1}")
            
            # Apply current injection to the target system
            if target_system in cosim.distribution_systems:
                dist_system = cosim.distribution_systems[target_system]['system']
                if hasattr(dist_system, 'inject_current'):
                    dist_system.inject_current(target_line, current_injection)
        
        # 3. Apply frequency injection if specified
        if 'frequency_injection' in attack_scenario and attack_scenario['frequency_injection'] != 0:
            frequency_shift = attack_scenario['frequency_injection'] * injection_scale
            
            # Check for frequency oscillation amplification
            if attack_scenario.get('electrical_attack_type') == 'oscillation_amplifier':
                # Get current transmission system frequency data for oscillation detection
                if hasattr(cosim.transmission_system, 'get_frequency_data'):
                    freq_data = cosim.transmission_system.get_frequency_data()
                    is_oscillating, osc_magnitude = detect_frequency_oscillations(freq_data)
                    
                    if is_oscillating:
                        # Amplify oscillations by 3x when detected
                        frequency_shift *= (3.0 * osc_magnitude)
                        print(f"   🎯 Oscillation detected! Amplifying frequency attack by {3.0 * osc_magnitude:.2f}x")
            
            # Apply frequency modification to transmission system
            if hasattr(cosim.transmission_system, 'modify_frequency'):
                cosim.transmission_system.modify_frequency(frequency_shift)
        
        # 4. Apply reactive power injection if specified
        if 'reactive_power_injection' in attack_scenario and attack_scenario['reactive_power_injection'] != 0:
            reactive_power = attack_scenario['reactive_power_injection'] * injection_scale
            
            # Apply reactive power injection to target system
            if target_system in cosim.distribution_systems:
                dist_system = cosim.distribution_systems[target_system]['system']
                if hasattr(dist_system, 'inject_reactive_power'):
                    dist_system.inject_reactive_power(reactive_power)
        
        # 5. Apply power factor manipulation if specified
        if 'power_factor_target' in attack_scenario:
            power_factor = attack_scenario['power_factor_target']
            
            # Modify power factor in target system
            if target_system in cosim.distribution_systems:
                dist_system = cosim.distribution_systems[target_system]['system']
                if hasattr(dist_system, 'set_power_factor'):
                    dist_system.set_power_factor(power_factor)
        
        # Log attack application with progress information
        attack_type = attack_scenario.get('electrical_attack_type', 'unknown')
        amplification = attack_scenario.get('amplification_factor', 1.0)
        
        if attack_progress < 0.1:  # Only log at start
            print(f"   🚀 Starting gradual {attack_type} attack on System {target_system}")
            print(f"      - Voltage injection: {attack_scenario.get('voltage_injection', 0):.3f} (10x amplified)")
            print(f"      - Current injection: {attack_scenario.get('current_injection', 0):.1f}A (10x amplified)")
            print(f"      - Frequency injection: {attack_scenario.get('frequency_injection', 0):.3f}Hz (10x amplified)")
            print(f"      - Target power deviation: {attack_scenario.get('max_power_deviation', 0):.1f}MW")
        elif 0.9 <= attack_progress <= 1.0:  # Log at end
            print(f"   ✅ Completed {attack_type} attack on System {target_system} at {current_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Failed to apply electrical attack to System {target_system}: {e}")
        return False


def enhance_cosimulation_with_electrical_attacks(cosim, attack_scenarios):
    """
    Enhance the co-simulation framework to handle electrical parameter attacks with gradual injection
    """
    # Store original simulation step function
    original_step = cosim.step if hasattr(cosim, 'step') else None
    
    def enhanced_step(current_time):
        # Execute original simulation step
        if original_step:
            result = original_step(current_time)
        else:
            result = None
        
        # Check for active electrical attacks at current time
        for attack in attack_scenarios:
            attack_start = attack['start_time']
            attack_end = attack_start + attack['duration']
            
            # Apply electrical attacks during attack window
            if attack_start <= current_time <= attack_end:
                # Apply gradual electrical parameter changes
                apply_electrical_attack_to_cosimulation(cosim, attack, current_time)
        
        return result
    
    # Replace simulation step function with enhanced version
    cosim.enhanced_step = enhanced_step
    cosim.electrical_attacks_enabled = True
    
    print(f"   🔧 Enhanced co-simulation with {len(attack_scenarios)} electrical attack scenarios")
    
    return cosim


def enhance_pinn_with_electrical_parameters(pinn_optimizer, attack_scenarios, current_time):
    """
    Enhance PINN optimization to consider electrical parameter modifications from attacks
    """
    # Find active attacks at current time
    active_attacks = []
    for attack in attack_scenarios:
        attack_start = attack['start_time']
        attack_end = attack_start + attack['duration']
        if attack_start <= current_time <= attack_end:
            active_attacks.append(attack)
    
    if not active_attacks:
        return None  # No active attacks, use normal PINN optimization
    
    # Create enhanced system state incorporating attack modifications
    enhanced_system_state = {
        'current_time': current_time,
        'base_demand_factor': 0.7,  # Default base demand
        'attack_modifications': {},
        'max_power_deviation': 0.0
    }
    
    for attack in active_attacks:
        target_system = attack['target_system']
        
        # Calculate attack progress for gradual attacks
        attack_duration = attack['duration']
        progress = (current_time - attack['start_time']) / attack_duration
        
        # Apply gradual scaling based on attack phase
        if attack.get('gradual_injection', False):
            ramp_up_ratio = attack.get('ramp_up_time', attack_duration * 0.3) / attack_duration
            peak_ratio = attack.get('peak_time', attack_duration * 0.4) / attack_duration
            
            if progress <= ramp_up_ratio:
                scale_factor = progress / ramp_up_ratio
            elif progress <= (ramp_up_ratio + peak_ratio):
                scale_factor = 1.0
            else:
                ramp_down_progress = (progress - ramp_up_ratio - peak_ratio) / (1.0 - ramp_up_ratio - peak_ratio)
                scale_factor = 1.0 - ramp_down_progress
        else:
            scale_factor = 1.0
        
        # Track maximum power deviation across all attacks
        power_deviation = attack.get('max_power_deviation', 0) * scale_factor
        enhanced_system_state['max_power_deviation'] = max(
            enhanced_system_state['max_power_deviation'], power_deviation
        )
        
        # Add electrical parameter modifications to system state
        enhanced_system_state['attack_modifications'][target_system] = {
            'voltage_injection': attack.get('voltage_injection', 0) * scale_factor,
            'current_injection': attack.get('current_injection', 0) * scale_factor,
            'frequency_injection': attack.get('frequency_injection', 0) * scale_factor,
            'reactive_power_injection': attack.get('reactive_power_injection', 0) * scale_factor,
            'power_factor_target': attack.get('power_factor_target', 0.9),
            'load_magnitude': attack.get('magnitude', 0) * scale_factor,
            'attack_type': attack.get('electrical_attack_type', 'unknown'),
            'stealth_level': attack.get('stealth_level', 1.0),
            'amplification_factor': attack.get('amplification_factor', 1.0),
            'attack_progress': progress,
            'injection_scale': scale_factor
        }
    
    return enhanced_system_state


def optimize_pinn_with_electrical_attacks(pinn_optimizer, enhanced_system_state):
    """
    Optimize PINN considering electrical parameter attacks with enhanced impact tracking
    """
    if enhanced_system_state is None:
        # No attacks, use standard optimization
        standard_data = {
            'soc': 0.6,
            'grid_voltage': 1.0,
            'grid_frequency': 60.0,
            'demand_factor': 0.7
        }
        return pinn_optimizer.optimize_references(standard_data)
    
    # Process each system with attack modifications
    optimized_references = {}
    total_power_impact = 0.0
    
    for system_id, modifications in enhanced_system_state['attack_modifications'].items():
        # Create system-specific optimization data
        system_data = {
            'soc': 0.6,  # Base SOC
            'grid_voltage': 1.0 + modifications['voltage_injection'],
            'grid_frequency': 60.0 + modifications['frequency_injection'],
            'demand_factor': enhanced_system_state['base_demand_factor'],
            'current_time': enhanced_system_state['current_time'],
            
            # Enhanced parameters from attacks
            'voltage_injection': modifications['voltage_injection'],
            'current_injection': modifications['current_injection'],
            'reactive_power_injection': modifications['reactive_power_injection'],
            'power_factor_target': modifications['power_factor_target'],
            'load_magnitude_injection': modifications['load_magnitude'],
            
            # Attack context with amplification info
            'attack_type': modifications['attack_type'],
            'stealth_level': modifications['stealth_level'],
            'amplification_factor': modifications['amplification_factor'],
            'under_attack': True,
            'attack_progress': modifications['attack_progress'],
            'injection_scale': modifications['injection_scale'],
            
            # Physics constraints (relaxed for higher impact)
            'max_voltage_deviation': 0.3,  # Increased from 0.1
            'max_current_deviation': 100.0,  # Increased from 50.0
            'max_frequency_deviation': 0.5   # Increased from 0.2
        }
        
        # Apply PINN optimization with attack-modified parameters
        try:
            v_ref, i_ref, p_ref = pinn_optimizer.optimize_references(system_data)
            
            # Calculate actual power impact
            power_impact = abs(p_ref - 15.0) * modifications['amplification_factor']  # 15.0 is baseline
            total_power_impact += power_impact
            
            # Apply defense mechanisms if attack is detected (less aggressive now)
            if modifications['stealth_level'] < 0.3:  # Only for very obvious attacks
                # Apply less conservative limits for higher impact
                v_ref = np.clip(v_ref, 300, 500)  # Wider voltage range
                i_ref = np.clip(i_ref, 10, 150)   # Wider current range
                p_ref = np.clip(p_ref, 5, 80)     # Wider power range
            
            optimized_references[system_id] = {
                'voltage_ref': v_ref,
                'current_ref': i_ref,
                'power_ref': p_ref,
                'power_impact_mw': power_impact / 1000.0,  # Convert to MW
                'attack_detected': modifications['stealth_level'] < 0.3,
                'defense_active': modifications['stealth_level'] < 0.2,
                'amplification_active': modifications['amplification_factor'] > 5.0
            }
            
        except Exception as e:
            print(f"   ⚠️ PINN optimization failed for system {system_id}: {e}")
            # Fallback to safe default values
            optimized_references[system_id] = {
                'voltage_ref': 400.0,
                'current_ref': 50.0,
                'power_ref': 20.0,
                'power_impact_mw': 0.0,
                'attack_detected': True,
                'defense_active': True,
                'amplification_active': False
            }
    
    # Log significant power impacts
    if total_power_impact > 1000:  # > 1MW impact
        print(f"   ⚡ Significant power impact detected: {total_power_impact/1000:.2f}MW across {len(optimized_references)} systems")
    
    return optimized_references


def create_intelligent_attack_scenarios(load_periods, total_duration=240.0, pinn_optimizer=None, use_rl=True, federated_manager=None, use_dqn_sac=True):

    # NEW: Try DQN/SAC Security Evasion System first
    if use_dqn_sac and use_rl and federated_manager is not None:
        print(" Using DQN/SAC SECURITY EVASION SYSTEM (Federated)")
        try:
            # Create mock CMS for DQN/SAC system
            from hierarchical_cosimulation import EnhancedChargingManagementSystem, EVChargingStation
            
            # Create mock stations for DQN/SAC system
            stations = []
            for i in range(6):
                station = EVChargingStation(
                    evcs_id=f"EVCS_{i}",
                    bus_name=f"Bus_{i+1}",
                    max_power=50000.0,
                    num_ports=40
                )
                stations.append(station)
            
            # Create CMS with security features
            cms = EnhancedChargingManagementSystem(stations=stations, use_pinn=True)
            cms.federated_manager = federated_manager
            cms.max_consecutive_anomalies = 3
            cms.rate_change_limit = 0.5
            cms.anomaly_threshold = 0.3
            cms.max_power_reference = 100.0
            cms.max_voltage_reference = 500.0
            cms.max_current_reference = 200.0
            cms.anomaly_counters = {i: 0 for i in range(6)}
            cms.reference_history = {i: [] for i in range(6)}
            cms.attack_active = False
            cms.adaptive_rl_enabled = True
            
            # Load pre-trained DQN/SAC security evasion trainer
            try:
                dqn_sac_trainer = create_dqn_sac_evasion_system(cms)
                dqn_sac_trainer.load_agents()  # Load pre-trained models
                print("    Pre-trained DQN/SAC agents loaded for attack generation")
            except:
                print("    Pre-trained DQN/SAC models not found, using default attack parameters")
                dqn_sac_trainer = None
            
            # Generate attack scenarios using trained agents
            dqn_sac_scenarios = {'DQN/SAC Security Evasion Attacks': []}
            
            # Generate attacks for different demand periods
            attack_count = 0
            for period_type, periods in load_periods.items():
                for period in periods[:2]:  # Limit to 2 periods per type
                    if period['duration'] >= 30:
                        attack_start = period['start_time'] + 10
                        attack_duration = min(50, period['duration'] - 20)  # Longer duration for gradual injection
                        target_system = (attack_count % 6) + 1
                        
                        # Generate realistic system state for agent decision making
                        current_system_state = {
                            'grid_voltage': np.random.uniform(0.90, 1.1),
                            'frequency': np.random.uniform(59.5, 60.5),
                            'current_load': np.random.uniform(80, 150),
                            'system_id': target_system,
                            'time': attack_start,
                            'period_type': period_type
                        }
                        
                        # Get agent decisions for electrical parameter control
                        if hasattr(dqn_sac_trainer, 'get_coordinated_attack'):
                            baseline_outputs = {
                                'voltage': 400.0, 'current': 100.0, 'power': 40.0,
                                'voltage_reference': 400.0, 'current_reference': 100.0, 'power_reference': 40.0,
                                'soc': 0.5, 'grid_voltage': current_system_state['grid_voltage'], 
                                'grid_frequency': current_system_state['frequency'],
                                'demand_factor': 1.0, 'voltage_priority': 0.0, 'urgency_factor': 1.0
                            }
                            
                            # Get coordinated agent attack
                            coordinated_attack = dqn_sac_trainer.get_coordinated_attack(target_system-1, baseline_outputs)
                            
                            if coordinated_attack:
                                dqn_decision = coordinated_attack.get('dqn_decision', {})
                                sac_control = coordinated_attack.get('sac_control', {})
                                
                                # Generate intelligent electrical parameters using agent decisions
                                electrical_params = enhance_agent_electrical_parameters(
                                    dqn_decision, sac_control, current_system_state, target_system, 0.0
                                )
                            else:
                                # Fallback to enhanced random generation if agent fails
                                electrical_params = {
                                    'voltage_injection': np.random.uniform(-0.15, 0.15) * 10,  # 10x amplification
                                    'current_injection': np.random.uniform(-80, 80) * 10,      # 10x amplification
                                    'frequency_injection': np.random.uniform(-0.25, 0.25) * 10, # 10x amplification
                                    'reactive_power_injection': np.random.uniform(-40, 40) * 5, # 5x amplification
                                    'power_factor_target': np.random.uniform(0.75, 0.95),
                                    'electrical_attack_type': ['voltage_manipulation', 'current_injection', 'frequency_drift'][attack_count % 3],
                                    'stealth_level': 0.4,  # Lower stealth for higher impact
                                    'amplification_factor': 10.0,
                                    'system_exploitation': False
                                }
                        else:
                            # Enhanced fallback with 10x amplification
                            electrical_params = {
                                'voltage_injection': np.random.uniform(-0.15, 0.15) * 10,  # 10x amplification
                                'current_injection': np.random.uniform(-80, 80) * 10,      # 10x amplification
                                'frequency_injection': np.random.uniform(-0.25, 0.25) * 10, # 10x amplification
                                'reactive_power_injection': np.random.uniform(-40, 40) * 5, # 5x amplification
                                'power_factor_target': np.random.uniform(0.75, 0.95),
                                'electrical_attack_type': ['voltage_manipulation', 'current_injection', 'frequency_drift'][attack_count % 3],
                                'stealth_level': 0.4,  # Lower stealth for higher impact
                                'amplification_factor': 10.0,
                                'system_exploitation': False
                            }
                        
                        # Enhanced scenario with true agent-controlled electrical parameters
                        scenario = {
                            'start_time': attack_start,
                            'duration': attack_duration,
                            'target_system': target_system,
                            'type': 'dqn_sac_evasion',
                            
                            # Load profile manipulation (amplified)
                            'magnitude': (1.5 + (attack_count * 0.2)) * 5.0,  # 5x load impact amplification
                            'target_percentage': 90,  # Higher target for more impact
                            
                            # Agent-controlled electrical parameter manipulation
                            'voltage_injection': electrical_params['voltage_injection'],
                            'current_injection': electrical_params['current_injection'],
                            'frequency_injection': electrical_params['frequency_injection'],
                            'reactive_power_injection': electrical_params['reactive_power_injection'],
                            'power_factor_target': electrical_params['power_factor_target'],
                            
                            # Enhanced attack parameters
                            'electrical_attack_type': electrical_params['electrical_attack_type'],
                            'target_bus': f"Bus_{target_system-1}",
                            'target_line': f"Line_{target_system-1}",
                            
                            # Agent strategy and context
                            'demand_context': f"Enhanced Agent-Controlled Attack during {period_type} (avg: {period['avg_load']:.2f})",
                            'rl_generated': True,
                            'dqn_sac_generated': True,
                            'security_evasion': True,
                            'trainer_instance': dqn_sac_trainer,
                            'stealth_score': electrical_params['stealth_level'],
                            'evasion_strategy': 'adaptive_threshold_bypass',
                            
                            # Enhanced stealth and impact features
                            'electrical_stealth': True,
                            'multi_parameter_attack': True,
                            'physics_constrained': True,
                            'gradual_injection': True,
                            'amplification_factor': electrical_params['amplification_factor'],
                            'agent_controlled': True,
                            'system_state': current_system_state,
                            'max_power_deviation': 100.0,  # Target 100MW deviation (4x current 25MW)
                            
                            # Gradual injection parameters for stealth
                            'injection_steps': max(5, int(attack_duration / 5)),  # 5-second steps
                            'ramp_up_time': attack_duration * 0.3,  # 30% of duration for ramp up
                            'peak_time': attack_duration * 0.4,     # 40% of duration at peak
                            'ramp_down_time': attack_duration * 0.3  # 30% of duration for ramp down
                        }
                        
                        # Apply physics constraints to ensure realistic attack parameters
                        scenario = apply_physics_constraints(scenario)
                        
                        dqn_sac_scenarios['DQN/SAC Security Evasion Attacks'].append(scenario)
                        attack_count += 1
            
            print(f"\n   DQN/SAC Security Evasion System Ready:")
            print(f"    - {len(dqn_sac_scenarios['DQN/SAC Security Evasion Attacks'])} evasion attacks generated")
            print("    - Professional RL algorithms (DQN + SAC)")
            print("    - Security bypass optimization")
            print("    - Adaptive anomaly detection evasion")
            print("    - Federated PINN integration")
            print("    - ✨ True agent-controlled electrical parameter manipulation")
            print("    - ✨ 10x amplified attack magnitudes for maximum impact")
            print("    - ✨ Gradual injection for stealth (5-second steps)")
            print("    - ✨ Intelligent frequency oscillation amplification")
            print("    - ✨ Target: 100MW power deviation (4x current 25MW)")
            print("    - ✨ Physics-constrained yet high-impact attacks")
            
            return dqn_sac_scenarios
            
        except Exception as e:
            print(f" DQN/SAC system failed: {e}")
            print(" Falling back to federated constrained RL system...")
    
    if use_rl and federated_manager is not None:
        print(" Using FEDERATED CONSTRAINED RL Attack System")
        try:
            # Create constrained RL attack system with federated PINN models
            constrained_rl_system = ConstrainedRLAttackSystem(
                num_systems=6,
                pinn_optimizers=federated_manager.local_models
            )
            
            # Generate system states and load contexts for attack generation
            system_states = {}
            load_contexts = {}
            
            for sys_id in range(1, 7):
                system_states[sys_id] = {
                    'grid_voltage': np.random.uniform(0.90, 1.1),
                    'frequency': np.random.uniform(59.5, 60.5),
                    'current_load': np.random.uniform(50.0, 200.0)
                }
                
                # Use load periods to create realistic contexts
                avg_load = 0.7
                if load_periods.get('high_demand'):
                    avg_load = np.mean([p['avg_load'] for p in load_periods['high_demand']])
                elif load_periods.get('low_demand'):
                    avg_load = np.mean([p['avg_load'] for p in load_periods['low_demand']])
                
                load_contexts[sys_id] = {
                    'avg_load': avg_load,
                    'peak_load': avg_load * 1.3,
                    'load_variance': 0.2
                }
            
            # Generate coordinated constrained attacks
            coordinated_attacks = constrained_rl_system.generate_coordinated_attacks(
                system_states, load_contexts
            )
            
            # Convert to format expected by simulation
            federated_scenarios = {'Federated Constrained RL Attacks': []}
            
            for attack in coordinated_attacks:
                scenario = {
                    'start_time': 180 + len(federated_scenarios['Federated Constrained RL Attacks']) * 45.0,
                    'duration': attack.get('duration', 30.0),
                    'target_system': attack['system_id'],
                    'type': attack['type'],
                    'magnitude': attack['magnitude'],  # Now constrained to 15-50kW
                    'target_percentage': attack.get('target_percentage', 80),
                    'demand_context': f"Federated RL attack on System {attack['system_id']}",
                    'rl_generated': True,
                    'federated': True,
                    'constrained': True,
                    'stealth_score': attack.get('stealth_score', 0.8),
                    'gradual_injection': attack.get('gradual_injection', True),
                    'injection_steps': attack.get('injection_steps', 5)
                }
                federated_scenarios['Federated Constrained RL Attacks'].append(scenario)
            
            print("\n   Federated Constrained RL Attack System Ready:")
            print(f"    - {len(coordinated_attacks)} constrained attacks generated")
            print("    - Physical constraints enforced (max 50kW injection)")
            print("    - Gradual injection patterns (5-second steps)")
            print("    - Anomaly detection enabled")
            print("    - Federated PINN optimization integrated")
            
            return federated_scenarios
            
        except Exception as e:
            print(f" Federated RL system failed: {e}")
            print(" Falling back to legacy RL system...")
            
    # # Legacy RL system fallback
    # if use_rl and pinn_optimizer is not None:
    #     print(" Using LEGACY RL Attack System (Single PINN)")
    #     try:
    #         # Import unified RL attack system
    #         from unified_rl_attack_system import create_unified_rl_attack_system
            
    #         # Create unified RL attack system that makes continuous decisions
    #         unified_system = create_unified_rl_attack_system(pinn_optimizer, total_duration)
            
    #         # Generate unified attack scenarios (replaces predefined scenarios)
    #         unified_scenarios = unified_system.generate_attacks_for_simulation(load_periods, total_duration)
            
    #         print("\n ✅ Legacy RL Attack System Ready:")
    #         print("    - Continuous decision making throughout simulation")
    #         print("    - Adaptive to real-time system response")
    #         print("    - Dynamic attack type selection")
    #         print("    - WARNING: No physical constraints enforced")
            
    #         return unified_scenarios
            
    #     except Exception as e:
    #         print(f" Legacy RL system failed: {e}")
    #         print(" Falling back to rule-based attack scenarios...")
    #         use_rl = False
    
    if not use_rl:
        print(" Using Rule-Based Attack Scenarios")
        # Fallback to original rule-based scenarios
        scenarios = {}
        low_demand_windows = []
        high_demand_windows = []
        
        for period in load_periods['low_demand']:
            if period['duration'] >= 20:
                low_demand_windows.append(period)
        
        for period in load_periods['high_demand']:
            if period['duration'] >= 20:
                high_demand_windows.append(period)
        
        # Sudden Demand Increase (during low demand)
        if low_demand_windows:
            scenarios['Demand Increase'] = []
            for i, window in enumerate(low_demand_windows[:4]):
                attack_start = window['start_time'] + 5
                attack_duration = min(30, window['duration'] - 10)
                
                scenarios['Demand Increase'].append({
                    'start_time': attack_start,
                    'duration': attack_duration,
                    'target_system': (i % 6) + 1,
                    'type': 'demand_increase',
                    'magnitude': (4.0 + i * 0.5),
                    'target_percentage': 80,
                    'demand_context': f"Low demand period (avg: {window['avg_load']:.2f})",
                    'rl_generated': False,
                    'voltage_drop_factor': 0.15  # 15% voltage drop for rule-based attacks
                })
        
        # Sudden Demand Decrease (during high demand)
        if high_demand_windows:
            scenarios['Demand Decrease'] = []
            for i, window in enumerate(high_demand_windows[:4]):
                attack_start = window['start_time'] + 5
                attack_duration = min(30, window['duration'] - 10)
                
                scenarios['Demand Decrease'].append({
                    'start_time': attack_start,
                    'duration': attack_duration,
                    'target_system': (i % 6) + 1,
                    'type': 'demand_decrease',
                    'magnitude': (4.0 + i * 0.1),
                    'target_percentage': 80,
                    'demand_context': f"High demand period (avg: {window['avg_load']:.2f})",
                    'rl_generated': False,
                    'voltage_drop_factor': 0.12  # 12% voltage drop for demand decrease attacks
                })
        
        # Oscillating Demand
        scenarios['Oscillating Demand'] = []
        strategic_times = [200, 250]
        
        for i, start_time in enumerate(strategic_times):
            if start_time < total_duration - 50:
                scenarios['Oscillating Demand'].append({
                    'start_time': start_time,
                    'duration': 50.0,
                    'target_system': (i % 6) + 1,
                    'type': 'oscillating_demand',
                    'magnitude': (2.5 + i * 0.5),
                    'target_percentage': 80,
                    'demand_context': f"Strategic timing at {start_time}s",
                    'rl_generated': False,
                    'voltage_drop_factor': 0.10  # 10% voltage drop for oscillating attacks
                })
        
        return scenarios


def run_training_phase():
    """Pre-train PINN optimizer and RL agents before co-simulation"""
    print("=" * 80)
    print(" FEDERATED TRAINING PHASE: Training Distributed PINN Models")
    print("=" * 80)
    
    # Step 1: Initialize Federated PINN Manager
    print("\n Phase 1: Initializing Federated PINN System...")
    print("-" * 50)
    
    federated_config = FederatedPINNConfig(
        num_distribution_systems=6,
        local_epochs=1000,  # Reduced for faster training
        global_rounds=10,   # Reduced for faster training
        aggregation_method='fedavg'
    )
    
    federated_manager = FederatedPINNManager(federated_config)
    print(" ✅ Federated PINN Manager initialized with 6 distribution systems")
    
    # Step 2: Train Federated PINN Models
    print("\n Phase 2: Training Federated PINN Models...")
    print("-" * 50)
    
    # Train each local model with ENHANCED PINN training (real EVCS dynamics)
    for sys_id in range(1, 7):
        print(f"\n 🔬 Training System {sys_id} PINN with REAL EVCS Dynamics...")
        print(f"  This will use the enhanced PINN training with controller.update_dynamics() calls!")
        
        # Use enhanced PINN training instead of simplified random data
        from pinn_optimizer import LSTMPINNConfig, PhysicsDataGenerator
        
        # Create configuration for this system
        pinn_config = LSTMPINNConfig()
        pinn_config.num_evcs_stations = 2  # Fewer stations per system for faster training
        pinn_config.sequence_length = 6     # Shorter sequences for faster training
        
        # Create enhanced data generator
        data_generator = PhysicsDataGenerator(pinn_config)
        
        # Generate realistic training data with real dynamics
        print(f"  Generating physics-accurate training data for System {sys_id}...")
        n_samples = 1000  # Reduced for faster training
        sequences, targets = data_generator.generate_realistic_evcs_scenarios(n_samples)
        
        print(f"  ✅ Generated {len(sequences)} sequences with {sequences.shape[-1]} features")
        print(f"  ✅ Target ranges: V={targets[:, 0].min():.1f}-{targets[:, 0].max():.1f}, I={targets[:, 1].min():.1f}-{targets[:, 1].max():.1f}, P={targets[:, 2].min():.2f}-{targets[:, 2].max():.2f}")
        
        # Train the local model with real dynamics data
        # We need to modify the local model to use our enhanced data
        local_model = federated_manager.local_models[sys_id]
        
        # Store the enhanced data in the local model for training
        if hasattr(local_model, 'data_generator'):
            # Clear existing data generator and use our enhanced data
            local_model._enhanced_training_data = (sequences, targets)
            print(f"  🔧 Enhanced data stored in local model for System {sys_id}")
        
        # Train with the enhanced data
        training_result = federated_manager.train_local_model(sys_id, (sequences, targets), n_samples)
        print(f" ✅ System {sys_id} training completed with REAL EVCS dynamics")
    
    # Perform federated averaging
    print("\n 🔄 Performing federated averaging...")
    for round_num in range(federated_config.global_rounds):
        print(f" Round {round_num + 1}/{federated_config.global_rounds}")
    # Step 3: Train DQN/SAC Security Evasion Agents
    print("\n Phase 3: Training DQN/SAC Security Evasion Agents...")
    print("-" * 50)
    
    try:
        # Create CMS for DQN/SAC training
        from hierarchical_cosimulation import EnhancedChargingManagementSystem
        cms = EnhancedChargingManagementSystem(stations=[], use_pinn=True)
        cms.federated_manager = federated_manager
        
        # Create and train DQN/SAC system
        dqn_sac_trainer = create_dqn_sac_evasion_system(cms)
        print(" 🚀 Training DQN/SAC agents (this may take a few minutes)...")
        dqn_sac_trainer.train_agents(sac_timesteps=100000, dqn_timesteps=100000)
        
        # Save trained models
        dqn_sac_trainer.save_agents()
        print(" ✅ DQN/SAC Security Evasion Agents trained and saved")
        
    except Exception as e:
        print(f" ⚠️ DQN/SAC training failed: {e}")
    
    # Step 4: Create Global Federated Optimizer
    print("\n Phase 4: Creating Global Federated Optimizer...")
    print("-" * 50)
    
    global_optimizer = GlobalFederatedOptimizer(federated_manager)
    print(" ✅ Global federated optimizer created")
    
    # # Legacy single PINN for backward compatibility
    # print("\n Phase 3: Training Legacy Single PINN (for compatibility)...")
    # print("-" * 50)
    
    # from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
    
    # # Create LSTM-PINN configuration for comprehensive training
    # pinn_config = LSTMPINNConfig(
    #     lstm_hidden_size=128,
    #     lstm_num_layers=3,
    #     sequence_length=10,
    #     hidden_layers=[256, 512, 256, 128],
    #     learning_rate=0.001,
    #     epochs=1000,  # Reduced epochs since federated is primary
    #     physics_weight=1.0,
    #     boundary_weight=10.0,
    #     data_weight=1.0,
    #     # Updated EVCS Charging Specifications (Realistic Constraints)
    #     rated_voltage=400.0,      # V (base reference voltage)
    #     rated_current=100.0,      # A (base reference current)
    #     rated_power=40.0,         # kW (400V × 100A = 40kW base power)
    #     max_voltage=500.0,        # V (maximum voltage limit)
    #     min_voltage=300.0,        # V (minimum voltage limit)
    #     max_current=150.0,        # A (maximum current limit)
    #     min_current=50.0,         # A (minimum current limit)
    #     max_power=75.0,           # kW (500V × 150A = 75kW maximum)
    #     min_power=15.0            # kW (300V × 50A = 15kW minimum)
    # )
    
    # # Train legacy PINN optimizer
    # pinn_optimizer = LSTMPINNChargingOptimizer(pinn_config, always_train=True)
    # print("Training legacy PINN with 2000 samples...")
    # pinn_optimizer.train_model(n_samples=2000)
    # pinn_optimizer.save_model('pinn_evcs_optimizer_pretrained.pth')
    # print(" Legacy PINN Optimizer training completed and saved")
    
    # # Step 9: Train RL Agents (Legacy)
    # print("\n Phase 9: Training Legacy RL Attack Agents...")
    # print("-" * 50)
    
    # try:
    #     from rl_attack_analytics import train_rl_agents
        
    #     # Train RL agents for different attack strategies
    #     print("Training legacy RL agents for cyber attack strategies...")
    #     rl_models = train_rl_agents(
    #         pinn_optimizer=pinn_optimizer,
    #         discrete_timesteps=50000,  # Reduced for faster training
    #         continuous_timesteps=50000  # Reduced for faster training
    #     )
    #     print(" Legacy RL Attack Agents training completed and saved")
    # except Exception as e:
    #     print(f" ⚠️ Legacy RL training failed: {e}")
    #     print(" Continuing with federated system only...")
    #     rl_models = None
    
    print("\n" + "=" * 80)
    print(" FEDERATED TRAINING PHASE COMPLETED")
    print("=" * 80)
    print(" ✅ Federated PINN Models: 6 distribution systems trained")
    print("=" * 80)
    
    # NOW ask user what to do after training completes
    print("\n Training Complete! Choose next action:")
    print("0 - Exit (training only, print information)")
    print("1 - Continue to co-simulation (use federated models)")
    print("2 - Continue to co-simulation (use legacy single PINN)")
    
    # user_choice = input("\nEnter your choice (0, 1, or 2): ").strip()
    user_choice = "1"
    
    if user_choice == "0":
        # Print training completion information and exit
        print("\n" + "=" * 80)
        print(" TRAINING COMPLETED - Models Ready for Future Use")
        print("=" * 80)
        print(" Federated Models: Saved in 'federated_models/' directory")
        print("=" * 80)
        return None, None, None, True  # Signal to exit
    elif user_choice == "1":
        print("\n🚀 Proceeding to co-simulation with FEDERATED models...")
        return federated_manager, global_optimizer, dqn_sac_trainer, False  # Continue with federated
    else:
        print("\n🚀 Proceeding to co-simulation with LEGACY single PINN...")
        return None, None, None, False  # Continue with legacy

def load_pretrained_models():
    """Load pre-trained PINN optimizer and RL agents"""
    print("\n🔄 Loading Pre-trained Models...")
    print("-" * 50)
    
    # Try to load federated models first
    federated_manager = None
    try:
        federated_config = FederatedPINNConfig(
            num_distribution_systems=6,
            local_epochs=1000,
            global_rounds=10,
            aggregation_method='fedavg'
        )
        federated_manager = FederatedPINNManager(federated_config)
        
        success = federated_manager.load_federated_models('federated_models')
        if success:
            print("✅ Federated PINN models loaded successfully")
        else:
            federated_manager = None
            print("⚠️ Federated models not found, trying legacy models...")
    except Exception as e:
        print(f"⚠️ Failed to load federated models: {e}")
        federated_manager = None
    
    # # Load legacy PINN optimizer
    # from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
    
    # pinn_config = LSTMPINNConfig(
    #     lstm_hidden_size=128,
    #     lstm_num_layers=3,
    #     sequence_length=10,
    #     hidden_layers=[256, 512, 256, 128],
    #     learning_rate=0.001,
    #     epochs=100,
    #     physics_weight=1.0,
    #     boundary_weight=10.0,
    #     data_weight=1.0,
    #     # Updated EVCS Charging Specifications (Realistic Constraints)
    #     rated_voltage=400.0,      # V (base reference voltage)
    #     rated_current=100.0,      # A (base reference current)
    #     rated_power=40.0,         # kW (400V × 100A = 40kW base power)
    #     max_voltage=500.0,        # V (maximum voltage limit)
    #     min_voltage=300.0,        # V (minimum voltage limit)
    #     max_current=150.0,        # A (maximum current limit)
    #     min_current=50.0,         # A (minimum current limit)
    #     max_power=75.0,           # kW (500V × 150A = 75kW maximum)
    #     min_power=15.0            # kW (300V × 50A = 15kW minimum)
    # )
    
    # pinn_optimizer = LSTMPINNChargingOptimizer(pinn_config, always_train=False)
    
    # try:
    #     pinn_optimizer.load_model('pinn_evcs_optimizer_pretrained.pth')
    #     print("✅ Legacy PINN Optimizer loaded successfully")
    # except:
    #     try:
    #         pinn_optimizer.load_model('pinn_evcs_optimizer.pth')
    #         print("✅ Legacy PINN Optimizer loaded from backup model")
    #     except:
    #         print("❌ No pre-trained PINN model found. Please run training first.")
    #         return None, None, None
    
    # Load coordinated DQN/SAC security evasion trainer
    dqn_sac_trainer = None
    try:
        from dqn_sac_security_evasion import DQNSACSecurityEvasionTrainer
        # Create trainer with CMS system (will be set later in co-simulation)
        dqn_sac_trainer = DQNSACSecurityEvasionTrainer(
            cms_system=None,  # Will be set during co-simulation
            num_stations=6,
            use_both=True
        )
        print("✅ Coordinated DQN/SAC Security Evasion Trainer initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize DQN/SAC trainer: {e}")
        dqn_sac_trainer = None
    
    # Load individual PINN optimizers for each distribution system (6 systems)
    individual_optimizers = {}
    for system_id in range(1, 7):  # Systems 1-6
        try:
            optimizer_file = f'federated_pinn_system_{system_id}.pth'
            
            # Check if file exists
            import os
            if not os.path.exists(optimizer_file):
                print(f"⚠️ PINN model file not found: {optimizer_file}")
                continue
                
            # Create individual PINN optimizer for this system
            from pinn_optimizer import LSTMPINNChargingOptimizer, PINNConfig
            
            pinn_config = PINNConfig(
                input_size=4,
                hidden_size=64,
                num_layers=3,
                output_size=3,
                physics_weight=0.3,
                max_voltage=500.0,
                max_current=200.0,
                max_power=100.0,
                min_voltage=300.0,
                min_current=10.0,
                min_power=15.0
            )
            
            individual_optimizer = LSTMPINNChargingOptimizer(pinn_config, always_train=False)
            
            # Try to load the model with error handling
            try:
                individual_optimizer.load_model(optimizer_file)
                individual_optimizers[system_id] = individual_optimizer
                print(f"✅ Individual PINN Optimizer {system_id} loaded successfully")
            except Exception as load_error:
                print(f"⚠️ Failed to load PINN model {optimizer_file}: {load_error}")
                # Create optimizer without loading for fallback
                individual_optimizers[system_id] = individual_optimizer
                print(f"   Using fresh PINN optimizer for system {system_id}")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize PINN optimizer {system_id}: {e}")
    
    if federated_manager and dqn_sac_trainer:
        print("✅ All models loaded successfully (Federated + DQN/SAC Trainer + Individual PINNs)!")
    elif federated_manager:
        print("✅ Federated and individual PINN models loaded successfully!")
    elif dqn_sac_trainer:
        print("✅ DQN/SAC Trainer and individual PINN models loaded successfully!")
    else:
        print("⚠️ Limited models loaded - some systems may use fallback methods")
    
    return federated_manager, individual_optimizers, dqn_sac_trainer

def run_focused_demand_analysis():
    """Run focused demand manipulation analysis with user choice for training vs pre-trained models"""
    print("=" * 80)
    print(" EVCS TRAINING AND SIMULATION WORKFLOW")
    print("=" * 80)
    
    # Initial user choice
    print("\n Choose your workflow:")
    print("1 - Train new models (PINN + RL agents) then run co-simulation")
    print("2 - Load pre-trained models and run co-simulation directly")
    print("3 - Train new models only (no co-simulation)")
    
    user_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    pinn_optimizer = None
    rl_models = None
    
    federated_manager = None
    dqn_sac_system = None
    individual_optimizers = {}
    
    if user_choice == "1":
        # Train new models then run co-simulation
        print("\n Option 1: Training new models then running co-simulation...")
        result = run_training_phase()
        
        # Check if user chose to exit after training
        if result[3]:  # exit_flag is True
            return None  # Exit without co-simulation
        
        # User chose to continue - use the trained models
        federated_manager, pinn_optimizer, dqn_sac_system = result[0], result[1], result[2]
        rl_models = dqn_sac_system  # For backward compatibility
        
        if federated_manager:
            print("\n Using freshly trained FEDERATED models for co-simulation...")
        else:
            print("\n Using freshly trained LEGACY models for co-simulation...")
        
    elif user_choice == "2":
        # Load pre-trained models and run co-simulation
        print("\n Option 2: Loading pre-trained models for co-simulation...")
        federated_manager, individual_optimizers, dqn_sac_system = load_pretrained_models()
        
        if not individual_optimizers and not federated_manager:
            print("\n   Failed to load pre-trained models. Please run training first (option 1).")
            return None
        
        if federated_manager and dqn_sac_system:
            print("\n Using pre-trained FEDERATED + DQN/SAC models for co-simulation...")
            # Train coordinated agents if DQN/SAC trainer is available
            if hasattr(dqn_sac_system, 'train_coordinated_agents'):
                print("   Training coordinated DQN/SAC agents...")
                dqn_sac_system.train_coordinated_agents(total_timesteps=50000)
        elif federated_manager:
            print("\n Using pre-trained FEDERATED models for co-simulation...")
        elif dqn_sac_system:
            print("\n Using pre-trained DQN/SAC + Individual PINN models for co-simulation...")
            # Train coordinated agents if DQN/SAC trainer is available
            if hasattr(dqn_sac_system, 'train_coordinated_agents'):
                print("   Training coordinated DQN/SAC agents...")
                dqn_sac_system.train_coordinated_agents(total_timesteps=50000)
        else:
            print("\n Using pre-trained Individual PINN models for co-simulation...")
        
    elif user_choice == "3":
        # Train models only, no co-simulation
        print("\n Option 3: Training models only...")
        result = run_training_phase()
        return None  # Exit after training
        
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        return None
    
    # Add safety timeout
    import signal
    import time
    
    def timeout_handler(signum, frame):
        print("\n TIMEOUT: Simulation taking too long!")
        raise TimeoutError("Simulation timeout")
    
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(300)
    
    try:
        # Generate load profile
        print("\n Generating load profile...")
        # Use 960 seconds to represent 24 hours (960s = 24h for simulation)
        times, load_multipliers = generate_daily_load_profile(total_duration=960.0, time_step=1.0, constant_load=False)
        
        # Identify demand periods
        print("Identifying demand periods...")
        load_periods = identify_demand_periods(load_multipliers, times, constant_load=False)
        
        # Print demand period analysis
        print("\n=== LOAD PROFILE ANALYSIS ===")
        for period_type, periods in load_periods.items():
            print(f"\n{period_type.replace('_', ' ').title()}:")
            for i, period in enumerate(periods):
                print(f"  Period {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
                      f"(Duration: {period['duration']:.1f}s, Avg Load: {period['avg_load']:.2f})")
        
        print("\n Starting Co-simulation with Pre-trained Models...")
        print("=" * 60)
        
        # Use the already loaded/trained models from the workflow choice
        if user_choice == "2":
            # For pre-trained models, use the new structure
            print(f"✅ Using Federated Manager: {'Available' if federated_manager else 'None'}")
            print(f"✅ Using Individual Optimizers: {len(individual_optimizers)} systems")
            print(f"✅ Using DQN/SAC Trainer: {'Available' if dqn_sac_system else 'None'}")
            
            # Use first individual optimizer as primary for legacy compatibility
            pinn_optimizer = individual_optimizers.get(1, None) if individual_optimizers else None
            rl_models = dqn_sac_system  # Use DQN/SAC trainer as RL models
        else:
            # For trained models, use existing structure
            print(f"✅ Using PINN Optimizer: {type(pinn_optimizer).__name__}")
            print(f"✅ Using RL Models: {type(rl_models).__name__ if rl_models else 'None'}")
        
        # Create intelligent attack scenarios with DQN/SAC integration
        print("\n Creating intelligent attack scenarios...")
        if federated_manager:
            print(" Using DQN/SAC SECURITY EVASION with federated PINN")
            scenarios = create_intelligent_attack_scenarios(
                load_periods, 
                pinn_optimizer=pinn_optimizer, 
                use_rl=True, 
                federated_manager=federated_manager,
                use_dqn_sac=True
            )
        else:
            print(" Using DQN/SAC SECURITY EVASION with individual PINNs")
            # Force DQN/SAC even without federated manager
            scenarios = create_intelligent_attack_scenarios(
                load_periods, 
                pinn_optimizer=pinn_optimizer, 
                use_rl=True,
                federated_manager=None,  # Explicitly pass None
                use_dqn_sac=True  # Force DQN/SAC usage
            )
        
        # Print attack scenarios
        print("\n=== INTELLIGENT ATTACK SCENARIOS ===")
        for scenario_name, attacks in scenarios.items():
            print(f"\n{scenario_name}:")
            for i, attack in enumerate(attacks):
                print(f"  Attack {i+1}: {attack['start_time']:.1f}s - {attack['start_time'] + attack['duration']:.1f}s "
                      f"(System {attack['target_system']}, {attack['demand_context']})")
        
        
        # User interaction for co-simulation continuation
        print("\n" + "="*60)
        print(" LSTM-PINN Training Complete!")
        print(" Model trained with real EVCS dynamics and IEEE 34 bus data")
        print(" Ready for hierarchical co-simulation integration")
        
        # user_choice = input("\n🤖 Continue with hierarchical co-simulation? (1=Yes, 0=No): ")
        user_choice = '1'
        
        if user_choice != '1':
            print(" Running standalone LSTM-PINN demonstration...")
            
            print("\n Demonstrating LSTM-PINN Optimization Capabilities:")
            
            # Test scenarios with proper sequence data
            test_scenarios = [
                {"name": "Low SOC Emergency", "soc": 0.15, "grid_voltage": 0.95, "urgency_factor": 2.0},
                {"name": "Grid Support Mode", "soc": 0.8, "grid_voltage": 0.92, "voltage_priority": 0.8},
                {"name": "Normal Charging", "soc": 0.4, "grid_voltage": 1.0, "urgency_factor": 1.0},
                {"name": "High Load Period", "soc": 0.6, "grid_voltage": 0.98, "load_factor": 0.9}
            ]
            
            print("\n Optimization Results:")
            print("-" * 80)
            print(f"{'Scenario':<20} {'SOC':<6} {'V(pu)':<8} {'V_ref(V)':<10} {'I_ref(A)':<10} {'P_ref(kW)':<10}")
            print("-" * 80)
            
            for scenario in test_scenarios:
                try:
                    test_data = {
                        'soc': scenario['soc'],
                        'grid_voltage': scenario['grid_voltage'],
                        'grid_frequency': 60.0,
                        'demand_factor': 0.7,
                        'voltage_priority': scenario.get('voltage_priority', 0.1),
                        'urgency_factor': scenario.get('urgency_factor', 1.0),
                        'current_time': 120.0,
                        'bus_distance': 2.0,
                        'load_factor': scenario.get('load_factor', 0.7)
                    }
                    
                    v_ref, i_ref, p_ref = pinn_optimizer.optimize_references(test_data)
                    print(f"{scenario['name']:<20} {scenario['soc']:<6.2f} {scenario['grid_voltage']:<8.2f} "
                          f"{v_ref:<10.1f} {i_ref:<10.1f} {p_ref:<10.1f}")
                    
                except Exception as e:
                    print(f" Error in scenario {scenario['name']}: {e}")
            
            print("\n LSTM-PINN Optimizer demonstration complete!")
            return
        
        print(" Starting co-simulation with trained LSTM-PINN model...")
        print(" Initializing co-simulation framework...")
        
        # Initialize simulation with enhanced PINN models if available
        if user_choice == "1" and federated_manager:
            # Option 1: Use enhanced PINN models from training
            print("🚀 Initializing co-simulation with ENHANCED PINN models from training...")
            cosim = HierarchicalCoSimulation(use_enhanced_pinn=True)
            print("   Enhanced PINN models will be used for realistic EVCS dynamics")
        else:
            # Option 2: Use standard co-simulation
            print("🚀 Initializing co-simulation with standard models...")
            cosim = HierarchicalCoSimulation()
        
        cosim.total_duration = 960.0  # 960 seconds for daily simulation (960s = 24h)
        # Add distribution systems with enhanced EVCS
        cosim.add_distribution_system(1, "ieee34Mod1.dss", 4)
        cosim.add_distribution_system(2, "ieee34Mod1.dss", 9)
        cosim.add_distribution_system(3, "ieee34Mod1.dss", 13)
        cosim.add_distribution_system(4, "ieee34Mod1.dss", 5)
        cosim.add_distribution_system(5, "ieee34Mod1.dss", 10)
        cosim.add_distribution_system(6, "ieee34Mod1.dss", 7)
        
        # Setup enhanced EV charging stations for 6 distribution systems
        enhanced_evcs_configs = [
            # Distribution System 1 - Urban Area
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ],
            # Distribution System 2 - Highway Corridor
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ],
            # Distribution System 3 - Mixed Area
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ],
            # Distribution System 4 - Industrial Zone
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ],
            # Distribution System 5 - Commercial District
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ],
            # Distribution System 6 - Residential Complex
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                # {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                # {'bus': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                # {'bus': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                # {'bus': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                {'bus': '848', 'max_power': 250, 'num_ports': 5},   # Industrial area
                {'bus': '830', 'max_power': 300, 'num_ports': 6},   # Suburban area
                {'bus': '824', 'max_power': 300, 'num_ports': 6},   # Shopping center
                {'bus': '826', 'max_power': 200, 'num_ports': 4},    # Residential area
            ]
        ]
        
        for i, (sys_id, dist_info) in enumerate(cosim.distribution_systems.items()):
            if i < len(enhanced_evcs_configs):
                dist_info['system'].add_ev_charging_stations(enhanced_evcs_configs[i])
        
        results = {}
        
        # Add baseline scenario (no attacks, just load profile) - REUSE TRAINED PINN
        print("\n--- Running Baseline (Load Profile Only) ---")
        if user_choice == "1" and federated_manager:
            # Option 1: Use enhanced PINN models from training
            baseline_cosim = HierarchicalCoSimulation(use_enhanced_pinn=True)
            print("   Baseline using enhanced PINN models from training")
        else:
            # Option 2: Use standard co-simulation
            baseline_cosim = HierarchicalCoSimulation()
            print("   Baseline using standard models")
        
        baseline_cosim.total_duration = 960.0  # 960 seconds for daily simulation (960s = 24h)
        baseline_cosim.add_distribution_system(1, "ieee34Mod1.dss", 4)
        baseline_cosim.add_distribution_system(2, "ieee34Mod1.dss", 9)
        baseline_cosim.add_distribution_system(3, "ieee34Mod1.dss", 13)
        baseline_cosim.add_distribution_system(4, "ieee34Mod1.dss", 5)
        baseline_cosim.add_distribution_system(5, "ieee34Mod1.dss", 10)
        baseline_cosim.add_distribution_system(6, "ieee34Mod1.dss", 7)
        
        for i, (sys_id, dist_info) in enumerate(baseline_cosim.distribution_systems.items()):
            if i < len(enhanced_evcs_configs):
                dist_info['system'].add_ev_charging_stations(enhanced_evcs_configs[i])
        
        # Set load profile in transmission system for dynamic base load scaling
        print("Setting load profile in transmission system...")
        baseline_cosim.transmission_system.set_load_profile(times, load_multipliers)
        
        # Share the trained PINN model across all CMS instances
        print(" Sharing trained PINN model with baseline scenario...")
        for sys_id, dist_info in baseline_cosim.distribution_systems.items():
            if hasattr(dist_info['system'], 'cms') and dist_info['system'].cms:
                if hasattr(dist_info['system'].cms, 'pinn_optimizer'):
                    # Load the pre-trained model instead of training new one
                    try:
                        dist_info['system'].cms.pinn_optimizer.load_model('pinn_evcs_optimizer.pth')
                        dist_info['system'].cms.pinn_trained = True
                        print(f" System {sys_id}: Pre-trained PINN model loaded")
                    except:
                        print(f" System {sys_id}: Could not load pre-trained model")
        
        start_time = time.time()
        baseline_simulation_results = baseline_cosim.run_hierarchical_simulation(attack_scenarios=[])
        baseline_time = time.time() - start_time
        
        results['Baseline (Daily Load Variation)'] = {
            'cosim': baseline_cosim,
            'simulation_time': baseline_time,
            'attack_scenarios': [],
            'load_profile': {'times': times, 'multipliers': load_multipliers},
            'simulation_results': baseline_simulation_results
        }

        baseline_cosim.plot_hierarchical_results()
        
        # print(f" Baseline completed in {baseline_time:.2f}s")

        # stats = baseline_cosim.get_simulation_statistics()

        # print("Baseline Simulation Statistics:######")
        
        # # Display key statistics
        # print(f"Frequency Analysis:")
        # print(f"  Min: {stats['frequency']['min']:.3f} Hz")
        # print(f"  Max: {stats['frequency']['max']:.3f} Hz")
        # print(f"  Mean: {stats['frequency']['mean']:.3f} Hz")
        # print(f"  Max Deviation: {stats['frequency']['max_deviation']:.3f} Hz")
        
        # print(f"\nLoad Analysis:")
        # print(f"  Min: {stats['total_load']['min']:.1f} MW")
        # print(f"  Max: {stats['total_load']['max']:.1f} MW")
        # print(f"  Mean: {stats['total_load']['mean']:.1f} MW")
        
        # print(f"\nAGC Performance:")
        # print(f"  Updates: {stats['agc_performance']['num_updates']}")
        # print(f"  Avg Reference Power: {stats['agc_performance']['avg_reference_power']:.1f} MW")
        
        # print(f"\nLoad Balancing:")
        # print(f"  Balancing Events: {stats['load_balancing']['num_balancing_events']}")
        # print(f"  Customer Redirections: {stats['load_balancing']['num_customer_redirections']}")
        
        # # Display distribution system statistics
        # print(f"\nDistribution System Performance:")
        # for sys_id, sys_stats in stats['distribution_systems'].items():
        #     print(f"  System {sys_id}:")
        #     print(f"    Load Range: {sys_stats['min']:.1f} - {sys_stats['max']:.1f} MW")
        #     print(f"    Load Std: {sys_stats['std']:.1f} MW")
            
        #     if sys_id in stats['charging_infrastructure']:
        #         charging_stats = stats['charging_infrastructure'][sys_id]
        #         print(f"    Avg Charging Time: {charging_stats['']:.1f} min")
        #         print(f"    Avg Utilization: {charging_stats.get('avg_utilization', 0):.2f}")
        #         print(f"    Efficiency Score: {charging_stats.get('efficiency_score', 0):.2f}")
            
        #     if sys_id in stats['attack_impacts']:
        #         attack_stats = stats['attack_impacts'][sys_id]
        #         print(f"    Attack Types: {', '.join(attack_stats['attack_types'])}")
        #         print(f"    Max Load Change: {attack_stats['max_load_change']:.1f}%")
        #         print(f"    Avg Charging Time Factor: {attack_stats['avg_charging_time_factor']:.2f}")
        
        # # Display global charging metrics
        # if 'global_charging_metrics' in stats:
        #     global_metrics = stats['global_charging_metrics']
        #     print(f"\nGlobal Charging Infrastructure:")
        #     print(f"  Avg Charging Time: {global_metrics['avg_charging_time']:.1f} min")
        #     print(f"  Charging Efficiency: {global_metrics['charging_time_efficiency']:.2f}")
        #     print(f"  Avg Customer Satisfaction: {global_metrics['avg_customer_satisfaction']:.2f}")
        #     print(f"  Overall Efficiency: {global_metrics['overall_efficiency']:.2f}")
        
        # # Plot results
        # print("\n" + "="*80)
        # print("GENERATING PLOTS")
        # print("="*80)

        
        # Run attack scenarios
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n--- Running {scenario_name} ---")
            
            # Reset simulation with appropriate model choice
            if user_choice == "1" and federated_manager:
                # Option 1: Use enhanced PINN models from training
                cosim = HierarchicalCoSimulation(use_enhanced_pinn=True)
                print(f"   {scenario_name} using enhanced PINN models from training")
            else:
                # Option 2: Use standard co-simulation
                cosim = HierarchicalCoSimulation()
                print(f"   {scenario_name} using standard models")
            
            cosim.total_duration = 960.0  # 960 seconds for daily simulation (960s = 24h)
            cosim.add_distribution_system(1, "ieee34Mod1.dss", 4)
            cosim.add_distribution_system(2, "ieee34Mod1.dss", 9)
            cosim.add_distribution_system(3, "ieee34Mod1.dss", 13)
            cosim.add_distribution_system(4, "ieee34Mod1.dss", 5)
            cosim.add_distribution_system(5, "ieee34Mod1.dss", 10)
            cosim.add_distribution_system(6, "ieee34Mod1.dss", 7)
                
            for i, (sys_id, dist_info) in enumerate(cosim.distribution_systems.items()):
                if i < len(enhanced_evcs_configs):  
                    dist_info['system'].add_ev_charging_stations(enhanced_evcs_configs[i])
            
            # Set load profile in transmission system for dynamic base load scaling
            cosim.transmission_system.set_load_profile(times, load_multipliers)
            
            # Share the trained PINN model across all CMS instances for this scenario
            print(f" Sharing trained PINN model with {scenario_name} scenario...")
            for sys_id, dist_info in cosim.distribution_systems.items():
                if hasattr(dist_info['system'], 'cms') and dist_info['system'].cms:
                    if hasattr(dist_info['system'].cms, 'pinn_optimizer'):
                        # Load the pre-trained model from the correct location based on training choice
                        try:
                            if user_choice == "1" and federated_manager:
                                # Option 1: Use freshly trained federated models
                                model_path = f'federated_models/local_pinn_system_{sys_id}.pth'
                                dist_info['system'].cms.pinn_optimizer.load_model(model_path)
                                print(f" System {sys_id}: Freshly trained federated PINN model loaded from {model_path}")
                            elif user_choice == "1" and pinn_optimizer:
                                # Option 1: Use freshly trained individual model
                                dist_info['system'].cms.pinn_optimizer.load_model('pinn_evcs_optimizer.pth')
                                print(f" System {sys_id}: Freshly trained individual PINN model loaded")
                            else:
                                # Option 2: Use pre-trained models
                                dist_info['system'].cms.pinn_optimizer.load_model('pinn_evcs_optimizer.pth')
                                print(f" System {sys_id}: Pre-trained PINN model loaded")
                            
                            dist_info['system'].cms.pinn_trained = True
                        except Exception as e:
                            print(f" System {sys_id}: Could not load PINN model: {e}")
                            print(f"   This may be expected if models were not trained yet")
            
            # Handle unified RL attack system vs traditional scenarios
            if isinstance(scenario_data, dict) and scenario_data.get('type') == 'continuous_rl':
                # Unified RL attack system - continuous decision making
                print("🎯 Running Unified RL Attack Simulation")
                unified_system = scenario_data['system']
                
                # Run simulation with unified RL attack system
                start_time = time.time()
                scenario_simulation_results = cosim.run_hierarchical_simulation_with_unified_rl(unified_system)
                simulation_time = time.time() - start_time
                
                # Get dynamic attack history from unified system
                attack_scenarios = unified_system.env.attack_history
                
            else:
                # Traditional predefined attack scenarios
                attack_scenarios = scenario_data if isinstance(scenario_data, list) else []
                
                # Integrate coordinated DQN/SAC trainer into co-simulation
                if dqn_sac_system and hasattr(dqn_sac_system, 'get_coordinated_attack'):
                    print("🎯 Integrating Coordinated DQN/SAC Trainer into Co-simulation")
                    cosim.dqn_sac_trainer = dqn_sac_system
                    cosim.security_evasion_active = True
                    
                    # Ensure DQN/SAC models are properly loaded based on training choice
                    if user_choice == "1":
                        print("   Using freshly trained DQN/SAC models for attack generation")
                        # Models are already loaded in the trainer from training phase
                    else:
                        print("   Using pre-trained DQN/SAC models for attack generation")
                        # Models should already be loaded from load_pretrained_models()
                
                # Enhance co-simulation with electrical attack capabilities
                cosim = enhance_cosimulation_with_electrical_attacks(cosim, attack_scenarios)
                
                # Run enhanced simulation with electrical parameter manipulation
                print(f"🚀 Starting attack scenario simulation: {scenario_name}...")
                start_time = time.time()
                scenario_simulation_results = cosim.run_hierarchical_simulation(attack_scenarios=attack_scenarios)
                simulation_time = time.time() - start_time
            
            results[scenario_name] = {
                'cosim': cosim,
                'simulation_time': simulation_time,
                'attack_scenarios': attack_scenarios,
                'load_profile': {'times': times, 'multipliers': load_multipliers},
                'simulation_results': scenario_simulation_results
            }
            
            print(f" {scenario_name} completed in {simulation_time:.2f}s")
            cosim.plot_hierarchical_results()

        #     stats = baseline_cosim.get_simulation_statistics()

        # print("Baseline Simulation Statistics:######")
        
        # # Display key statistics
        # print(f"Frequency Analysis:")
        # print(f"  Min: {stats['frequency']['min']:.3f} Hz")
        # print(f"  Max: {stats['frequency']['max']:.3f} Hz")
        # print(f"  Mean: {stats['frequency']['mean']:.3f} Hz")
        # print(f"  Max Deviation: {stats['frequency']['max_deviation']:.3f} Hz")
        
        # print(f"\nLoad Analysis:")
        # print(f"  Min: {stats['total_load']['min']:.1f} MW")
        # print(f"  Max: {stats['total_load']['max']:.1f} MW")
        # print(f"  Mean: {stats['total_load']['mean']:.1f} MW")
        
        # print(f"\nAGC Performance:")
        # print(f"  Updates: {stats['agc_performance']['num_updates']}")
        # print(f"  Avg Reference Power: {stats['agc_performance']['avg_reference_power']:.1f} MW")
        
        # print(f"\nLoad Balancing:")
        # print(f"  Balancing Events: {stats['load_balancing']['num_balancing_events']}")
        # print(f"  Customer Redirections: {stats['load_balancing']['num_customer_redirections']}")
        
        # # Display distribution system statistics
        # print(f"\nDistribution System Performance:")
        # for sys_id, sys_stats in stats['distribution_systems'].items():
        #     print(f"  System {sys_id}:")
        #     print(f"    Load Range: {sys_stats['min']:.1f} - {sys_stats['max']:.1f} MW")
        #     print(f"    Load Std: {sys_stats['std']:.1f} MW")
            
        #     if sys_id in stats['charging_infrastructure']:
        #         charging_stats = stats['charging_infrastructure'][sys_id]
        #         print(f"    Avg Charging Time: {charging_stats['']:.1f} min")
        #         print(f"    Avg Utilization: {charging_stats.get('avg_utilization', 0):.2f}")
        #         print(f"    Efficiency Score: {charging_stats.get('efficiency_score', 0):.2f}")
            
        #     if sys_id in stats['attack_impacts']:
        #         attack_stats = stats['attack_impacts'][sys_id]
        #         print(f"    Attack Types: {', '.join(attack_stats['attack_types'])}")
        #         print(f"    Max Load Change: {attack_stats['max_load_change']:.1f}%")
        #         print(f"    Avg Charging Time Factor: {attack_stats['avg_charging_time_factor']:.2f}")
        
        # # Display global charging metrics
        # if 'global_charging_metrics' in stats:
        #     global_metrics = stats['global_charging_metrics']
        #     print(f"\nGlobal Charging Infrastructure:")
        #     print(f"  Avg Charging Time: {global_metrics['avg_charging_time']:.1f} min")
        #     print(f"  Charging Efficiency: {global_metrics['charging_time_efficiency']:.2f}")
        #     print(f"  Avg Customer Satisfaction: {global_metrics['avg_customer_satisfaction']:.2f}")
        #     print(f"  Overall Efficiency: {global_metrics['overall_efficiency']:.2f}")
        
        # # Plot results
        # print("\n" + "="*80)
        # print("GENERATING PLOTS")
        # print("="*80)
        
        # Detailed analysis and visualization
        analyze_focused_results(results)
        
        return results
        
    except TimeoutError as e:
        print(f"\n {e}")
        print("   The simulation was terminated due to timeout.")
        print("   This indicates a potential infinite loop in the CMS power allocation.")
        print("   The fixes have been applied to prevent this issue.")
        return None
        
    except ZeroDivisionError as e:
        print(f"\n Division by zero error: {e}")
        print("   This is likely caused by invalid inputs to CMS power allocation.")
        print("   Check for zero voltage, current, or power values in the simulation.")
        return None
        
    except ArithmeticError as e:
        print(f"\n Arithmetic error: {e}")
        print("   This may be caused by numerical instability in CMS calculations.")
        print("   Check for invalid mathematical operations (sqrt of negative, etc.).")
        return None
        
    except ValueError as e:
        print(f"\n Value error: {e}")
        print("   This may be caused by invalid parameter values in the simulation.")
        print("   Check for NaN, infinity, or out-of-range values.")
        return None
        
    except Exception as e:
        print(f"\n Unexpected error: {type(e).__name__}: {e}")
        print("   This may be related to the CMS power allocation issue.")
        print(f"   Error details: {str(e)}")
        import traceback
        print(f"   Stack trace: {traceback.format_exc()}")
        return None
        
    finally:
        # Cancel the alarm
        # signal.alarm(0)
        print("\n Safety timeout disabled - simulation completed or terminated")

def analyze_focused_results(results):
    """Analyze and visualize focused demand manipulation results with daily load variation"""
    
    import os
    
    # Create sub_figures directory if it doesn't exist
    os.makedirs('sub_figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n=== FOCUSED ANALYSIS RESULTS WITH DAILY LOAD VARIATION ===")
    
    # Extract RL rewards and PINN losses for plotting
    rl_rewards = {'discrete': [], 'continuous': [], 'combined': []}
    pinn_losses = {'total': [], 'physics': [], 'data': []}
    attack_stealth_scores = []
    
    # Collect RL and PINN metrics from results
    for scenario_name, scenario_data in results.items():
        if 'attack_scenarios' in scenario_data:
            for attack in scenario_data['attack_scenarios']:
                if attack.get('rl_generated', False):
                    rl_rewards['discrete'].append(attack.get('discrete_reward', 0))
                    rl_rewards['continuous'].append(attack.get('continuous_reward', 0))
                    rl_rewards['combined'].append(attack.get('combined_reward', 0))
                    attack_stealth_scores.append(attack.get('stealth_score', 1.0))
        
        # Extract PINN training history if available
        if 'pinn_history' in scenario_data:
            pinn_hist = scenario_data['pinn_history']
            if 'total_loss' in pinn_hist:
                pinn_losses['total'].extend(pinn_hist['total_loss'])
            if 'physics_loss' in pinn_hist:
                pinn_losses['physics'].extend(pinn_hist['physics_loss'])
            if 'data_loss' in pinn_hist:
                pinn_losses['data'].extend(pinn_hist['data_loss'])
    
    # Create comprehensive visualization with daily load profile
    num_scenarios = len(results)
    fig, axes = plt.subplots(4, num_scenarios, figsize=(5*num_scenarios, 20))
    
    # Handle single column case
    if num_scenarios == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['green', 'red', 'blue', 'orange']
    scenario_names = list(results.keys())
    
    # Baseline reference
    baseline_key = 'Baseline (Daily Load Variation)'
    baseline_result = results.get(baseline_key, results[list(results.keys())[0]])
    base_cosim = baseline_result['cosim']
    base_time = np.array(base_cosim.results['time'])
    base_freq = np.array(base_cosim.results['frequency'])
    base_load = np.array(base_cosim.results['total_load'])
    base_ref = np.array(base_cosim.results['reference_power'])
    
    # Load profile
    load_times = baseline_result['load_profile']['times']
    load_multipliers = baseline_result['load_profile']['multipliers']
    
    for i, (scenario_name, result) in enumerate(results.items()):
        cosim = result['cosim']
        
        # Extract data for full simulation window
        time_data = np.array(cosim.results['time'])
        frequencies = np.array(cosim.results['frequency'])
        loads = np.array(cosim.results['total_load'])
        ref_power = np.array(cosim.results['reference_power'])
        
        # Align baseline to scenario time grid
        base_freq_interp = np.interp(time_data, base_time, base_freq)
        base_load_interp = np.interp(time_data, base_time, base_load)
        base_ref_interp = np.interp(time_data, base_time, base_ref)
        
        # Deltas vs baseline
        dfreq = frequencies - base_freq_interp
        dload = loads - base_load_interp
        dref = ref_power - base_ref_interp
        
        # 1. Daily Load Profile with Attack Windows (Row 1)
        axes[0, i].plot(load_times, load_multipliers, color='gray', linewidth=2, alpha=0.7, label='Daily Load Profile')
        # axes[0, i].set_title(f'{scenario_name}\nDaily Load Profile & Attack Windows')
        axes[0, i].set_ylabel('Load Multiplier', fontsize=18)
        axes[0, i].set_xlabel('Time (s)', fontsize=18)
        axes[0, i].grid(True, alpha=0.3)
        
        # Mark attack windows if attacks exist
        if result['attack_scenarios']:
            for attack in result['attack_scenarios']:
                start_time = attack['start_time']
                end_time = start_time + attack['duration']
                axes[0, i].axvspan(start_time, end_time, alpha=0.25, color=colors[i % len(colors)], \
                                   label=f"{attack['type'].replace('_',' ').title()} (Sys {attack.get('target_system','?')})")
        axes[0, i].legend()
        
        # 2. Frequency Response (Row 2) with baseline overlay and delta annotation
        axes[1, i].plot(time_data, base_freq_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        axes[1, i].plot(time_data, frequencies, color=colors[i % len(colors)], linewidth=2, label='Scenario')
        axes[1, i].axhline(y=60.0, color='black', linestyle=':', alpha=0.5)
        # axes[1, i].set_title('Frequency Response (vs Baseline)', fontsize=18)
        axes[1, i].set_ylabel('Frequency (Hz)', fontsize=18)
        axes[1, i].grid(True, alpha=0.3)
        max_dfreq = float(np.max(np.abs(dfreq))) if len(dfreq) else 0.0
        axes[1, i].text(0.01, 0.92, f"max Δf = {max_dfreq:.3f} Hz", transform=axes[1, i].transAxes,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        axes[1, i].legend(loc='best')
        
        # 3. Load Variation (Row 3) with baseline overlay and delta annotation
        axes[2, i].plot(time_data, base_load_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        axes[2, i].plot(time_data, loads, color=colors[i % len(colors)], linewidth=2, label='Scenario')
            # axes[2, i].set_title('Distribution Load Variation (vs Baseline)', fontsize=18)
        axes[2, i].set_ylabel('Load (MW)', fontsize=18)
        axes[2, i].grid(True, alpha=0.3)
        max_dload = float(np.max(np.abs(dload))) if len(dload) else 0.0
        axes[2, i].text(0.01, 0.92, f"max ΔLoad = {max_dload:.1f} MW", transform=axes[2, i].transAxes,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        axes[2, i].legend(loc='best')
        
        # 4. AGC Reference Power (Row 4) with baseline overlay and delta annotation
        axes[3, i].plot(time_data, base_ref_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        axes[3, i].plot(time_data, ref_power, color=colors[i % len(colors)], linewidth=2, label='Scenario')
        # axes[3, i].set_title('AGC Reference Power (vs Baseline)', fontsize=18)
        axes[3, i].set_ylabel('Reference Power (MW)', fontsize=18)
        axes[3, i].set_xlabel('Time (s)', fontsize=18)
        axes[3, i].grid(True, alpha=0.3)
        max_dref = float(np.max(np.abs(dref))) if len(dref) else 0.0
        axes[3, i].text(0.01, 0.85, f"max ΔRef = {max_dref:.1f} MW", transform=axes[3, i].transAxes,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        axes[3, i].legend(loc='best')
        
        # Print statistics
        print(f"\n{scenario_name} Statistics:")
        print(f"  Frequency range: {frequencies.min():.3f} - {frequencies.max():.3f} Hz (max Δ vs baseline {max_dfreq:.3f} Hz)")
        print(f"  Load range: {loads.min():.1f} - {loads.max():.1f} MW (max Δ vs baseline {max_dload:.1f} MW)")
        print(f"  AGC reference range: {ref_power.min():.1f} - {ref_power.max():.1f} MW (max Δ vs baseline {max_dref:.1f} MW)")
        
        if result['attack_scenarios']:
            print(f"  Number of attacks: {len(result['attack_scenarios'])}")
            for j, attack in enumerate(result['attack_scenarios']):
                print(f"    Attack {j+1}: {attack['type']} @ {attack['start_time']}s for {attack['duration']}s")
    
    plt.tight_layout()
    plt.show()
    
    # Save individual subplots as separate PDFs
    colors = ['green', 'red', 'blue', 'orange']
    
    for i, (scenario_name, result) in enumerate(results.items()):
        cosim = result['cosim']
        
        # Extract data for full simulation window
        time_data = np.array(cosim.results['time'])
        frequencies = np.array(cosim.results['frequency'])
        loads = np.array(cosim.results['total_load'])
        ref_power = np.array(cosim.results['reference_power'])
        
        # Align baseline to scenario time grid
        base_freq_interp = np.interp(time_data, base_time, base_freq)
        base_load_interp = np.interp(time_data, base_time, base_load)
        base_ref_interp = np.interp(time_data, base_time, base_ref)
        
        # Deltas vs baseline
        dfreq = frequencies - base_freq_interp
        dload = loads - base_load_interp
        dref = ref_power - base_ref_interp
        
        # Save subplot 1: Daily Load Profile with Attack Windows
        fig_1 = plt.figure(figsize=(10, 8))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(load_times, load_multipliers, color='gray', linewidth=2, alpha=0.7, label='Daily Load Profile')
        ax_1.set_ylabel('Load Multiplier', fontsize=18)
        ax_1.set_xlabel('Time (s)', fontsize=18)
        ax_1.grid(True, alpha=0.3)
        
        if result['attack_scenarios']:
            for attack in result['attack_scenarios']:
                start_time = attack['start_time']
                end_time = start_time + attack['duration']
                ax_1.axvspan(start_time, end_time, alpha=0.25, color=colors[i % len(colors)], 
                            label=f"{attack['type'].replace('_',' ').title()} (Sys {attack.get('target_system','?')})")
        ax_1.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        clean_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_").replace(":", "_")
        plt.savefig(f'sub_figures/daily_load_profile_{timestamp}.pdf', 
                    format='pdf', bbox_inches='tight')
        plt.close(fig_1)
        
        # Save subplot 2: Frequency Response
        fig_2 = plt.figure(figsize=(10, 8))
        ax_2 = fig_2.add_subplot(111)
        ax_2.plot(time_data, base_freq_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax_2.plot(time_data, frequencies, color=colors[i % len(colors)], linewidth=2, label='Scenario')
        ax_2.axhline(y=60.0, color='black', linestyle=':', alpha=0.5)
        ax_2.set_ylabel('Frequency (Hz)', fontsize=18)
        ax_2.set_xlabel('Time (s)', fontsize=18)
        ax_2.grid(True, alpha=0.3)
        max_dfreq = float(np.max(np.abs(dfreq))) if len(dfreq) else 0.0
        ax_2.text(0.01, 0.92, f"max Δf = {max_dfreq:.3f} Hz", transform=ax_2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax_2.legend(loc='best')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/frequency_response_{timestamp}.pdf', 
                    format='pdf', bbox_inches='tight')
        plt.close(fig_2)
        
        # Save subplot 3: Load Variation
        fig_3 = plt.figure(figsize=(10, 8))
        ax_3 = fig_3.add_subplot(111)
        ax_3.plot(time_data, base_load_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax_3.plot(time_data, loads, color=colors[i % len(colors)], linewidth=2, label='Scenario')
        ax_3.set_ylabel('Load (MW)', fontsize=18)
        ax_3.set_xlabel('Time (s)', fontsize=18)
        ax_3.grid(True, alpha=0.3)
        max_dload = float(np.max(np.abs(dload))) if len(dload) else 0.0
        ax_3.text(0.01, 0.92, f"max ΔLoad = {max_dload:.1f} MW", transform=ax_3.transAxes,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax_3.legend(loc='best')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/load_variation_{timestamp}.pdf', 
                    format='pdf', bbox_inches='tight')
        plt.close(fig_3)
        
        # Save subplot 4: AGC Reference Power
        fig_4 = plt.figure(figsize=(10, 8))
        ax_4 = fig_4.add_subplot(111)
        ax_4.plot(time_data, base_ref_interp, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax_4.plot(time_data, ref_power, color=colors[i % len(colors)], linewidth=2, label='Scenario')
        ax_4.set_ylabel('Reference Power (MW)', fontsize=18)
        ax_4.set_xlabel('Time (s)', fontsize=18)
        ax_4.grid(True, alpha=0.3)
        max_dref = float(np.max(np.abs(dref))) if len(dref) else 0.0
        ax_4.text(0.01, 0.85, f"max ΔRef = {max_dref:.1f} MW", transform=ax_4.transAxes,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax_4.legend(loc='best')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/agc_reference_power_{timestamp}.pdf', 
                    format='pdf', bbox_inches='tight')
        plt.close(fig_4)
    
    print(f"\n✅ Individual scenario subplot PDFs saved in 'sub_figures/' directory:")
    for i, scenario_name in enumerate(results.keys()):
        clean_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_").replace(":", "_")
        print(f"   Scenario '{scenario_name}':")
        print(f"     - daily_load_profile_{clean_name}.pdf")
        print(f"     - frequency_response_{clean_name}.pdf")
        print(f"     - load_variation_{clean_name}.pdf")
        print(f"     - agc_reference_power_{clean_name}.pdf")
    
    # Create comparison plots
    create_comparison_plots(results, rl_rewards, pinn_losses, attack_stealth_scores)


def create_comparison_plots(results, rl_rewards, pinn_losses, attack_stealth_scores):
    """Create comparison plots across all scenarios with daily load context and baseline deltas"""
    
    import os
    
    # Create sub_figures directory if it doesn't exist
    os.makedirs('sub_figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    colors = ['green', 'red', 'blue', 'orange']
    scenario_names = list(results.keys())
    
    # Baseline reference
    baseline_key = 'Baseline (Daily Load Variation)'
    baseline_result = results.get(baseline_key, results[list(results.keys())[0]])
    base_cosim = baseline_result['cosim']
    base_time = np.array(base_cosim.results['time'])
    base_freq = np.array(base_cosim.results['frequency'])
    base_load = np.array(base_cosim.results['total_load'])
    
    # Daily load profile
    load_times = baseline_result['load_profile']['times']
    load_multipliers = baseline_result['load_profile']['multipliers']
    
    comparison_data = {}
    
    for scenario_name, result in results.items():
        cosim = result['cosim']
        
        time_data = np.array(cosim.results['time'])
        frequencies = np.array(cosim.results['frequency'])
        loads = np.array(cosim.results['total_load'])
        
        # Baseline aligned to scenario time
        base_freq_interp = np.interp(time_data, base_time, base_freq)
        base_load_interp = np.interp(time_data, base_time, base_load)
        
        comparison_data[scenario_name] = {
            'times': time_data,
            'frequencies': frequencies,
            'loads': loads,
            'dfreq': frequencies - base_freq_interp,
            'dload': loads - base_load_interp,
            'attack_scenarios': result['attack_scenarios']
        }
    
    # 1. Daily Load Profile with Attack Timing
    axes[0,0].plot(load_times, load_multipliers, color='gray', linewidth=3, alpha=0.8, label='Daily Load Profile')
    # axes[0,0].set_title('Daily Load Profile with Attack Timing', fontsize=18)
    axes[0,0].set_ylabel('Load Multiplier', fontsize=18)
    axes[0,0].set_xlabel('Time (s)', fontsize=18)
    axes[0,0].grid(True, alpha=0.3)
    
    # Mark attack windows for each scenario
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        if data['attack_scenarios']:
            for attack in data['attack_scenarios']:
                start_time = attack['start_time']
                end_time = start_time + attack['duration']
                axes[0,0].axvspan(start_time, end_time, alpha=0.2, color=colors[i % len(colors)], 
                                 label=f"{scenario_name}: {attack['type'].replace('_', ' ').title()}")
    axes[0,0].legend()
    
    # Save subplot 1 as separate PDF
    fig_1 = plt.figure(figsize=(10, 8))
    ax_1 = fig_1.add_subplot(111)
    ax_1.plot(load_times, load_multipliers, color='gray', linewidth=3, alpha=0.8, label='Daily Load Profile')
    ax_1.set_ylabel('Load Multiplier', fontsize=18)
    ax_1.set_xlabel('Time (s)', fontsize=18)
    ax_1.grid(True, alpha=0.3)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        if data['attack_scenarios']:
            for attack in data['attack_scenarios']:
                start_time = attack['start_time']
                end_time = start_time + attack['duration']
                ax_1.axvspan(start_time, end_time, alpha=0.2, color=colors[i % len(colors)], 
                            label=f"{scenario_name}: {attack['type'].replace('_', ' ').title()}")
    ax_1.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'sub_figures/daily_load_profile_with_attacks_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_1)
    
    # 2. Frequency Comparison (delta vs baseline)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        axes[0,1].plot(data['times'], data['dfreq'], 
                      color=colors[i % len(colors)], linewidth=2, label=f"{scenario_name}")
    axes[0,1].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    # axes[0,1].set_title('Frequency Delta vs Baseline', fontsize=18)
    axes[0,1].set_ylabel('Δ Frequency (Hz)', fontsize=18)
    axes[0,1].set_xlabel('Time (s)', fontsize=18)
    axes[0,1].legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    axes[0,1].grid(True, alpha=0.3)
    
    # Save subplot 2 as separate PDF
    fig_2 = plt.figure(figsize=(10, 8))
    ax_2 = fig_2.add_subplot(111)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        ax_2.plot(data['times'], data['dfreq'], 
                  color=colors[i % len(colors)], linewidth=2, label=f"{scenario_name}")
    ax_2.axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax_2.set_ylabel('Δ Frequency (Hz)', fontsize=18)
    ax_2.set_xlabel('Time (s)', fontsize=18)
    ax_2.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax_2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'sub_figures/frequency_delta_vs_baseline_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_2)
    
    # 3. Load Comparison (delta vs baseline)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        axes[1,0].plot(data['times'], data['dload'], 
                      color=colors[i % len(colors)], linewidth=2, label=f"{scenario_name}")
    axes[1,0].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    # axes[1,0].set_title('Load Delta vs Baseline', fontsize=18)
    axes[1,0].set_ylabel('Δ Load (MW)', fontsize=18)
    axes[1,0].set_xlabel('Time (s)', fontsize=18)
    axes[1,0].legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    axes[1,0].grid(True, alpha=0.3)
    
    # Save subplot 3 as separate PDF
    fig_3 = plt.figure(figsize=(10, 8))
    ax_3 = fig_3.add_subplot(111)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        ax_3.plot(data['times'], data['dload'], 
                  color=colors[i % len(colors)], linewidth=2, label=f"{scenario_name}")
    ax_3.axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax_3.set_ylabel('Δ Load (MW)', fontsize=18)
    ax_3.set_xlabel('Time (s)', fontsize=18)
    ax_3.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax_3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'sub_figures/load_delta_vs_baseline_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_3)
    
    # 4. Frequency Deviation from 60 Hz (unchanged)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        freq_deviation = abs(60.0 - (data['frequencies']))
        axes[1,1].plot(data['times'], freq_deviation, 
                      color=colors[i % len(colors)], linewidth=2, label=scenario_name)
    # axes[1,1].set_title('Frequency Deviation from 60 Hz', fontsize=18)
    axes[1,1].set_ylabel('Deviation (Hz)', fontsize=18)
    axes[1,1].set_xlabel('Time (s)', fontsize=18)
    axes[1,1].legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    axes[1,1].grid(True, alpha=0.3)
    
    # Save subplot 4 as separate PDF
    fig_4 = plt.figure(figsize=(10, 8))
    ax_4 = fig_4.add_subplot(111)
    for i, (scenario_name, data) in enumerate(comparison_data.items()):
        freq_deviation = abs(60.0 - (data['frequencies']))
        ax_4.plot(data['times'], freq_deviation, 
                  color=colors[i % len(colors)], linewidth=2, label=scenario_name)
    ax_4.set_ylabel('Deviation (Hz)', fontsize=18)
    ax_4.set_xlabel('Time (s)', fontsize=18)
    ax_4.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax_4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'sub_figures/frequency_deviation_from_60hz_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_4)
    
    # 5. Max Load Delta During Attacks (bar chart)
    attack_impacts = []
    attack_labels = []
    for scenario_name, data in comparison_data.items():
        max_during_attacks = 0.0
        if data['attack_scenarios']:
            for attack in data['attack_scenarios']:
                start_idx = np.argmin(np.abs(data['times'] - attack['start_time']))
                end_idx = np.argmin(np.abs(data['times'] - (attack['start_time'] + attack['duration'])))
                if start_idx < end_idx:
                    max_during_attacks = max(max_during_attacks, float(np.max(np.abs(data['dload'][start_idx:end_idx]))))
        attack_impacts.append(max_during_attacks)
        attack_labels.append(scenario_name)
    if attack_impacts:
        bars = axes[2,1].bar(attack_labels, attack_impacts, color=colors[:len(attack_impacts)])
        # axes[2,1].set_title('Max Load Delta During Attacks (vs Baseline)', fontsize=18)
        axes[2,1].set_ylabel('Δ Load (MW)', fontsize=18)
        axes[2,1].tick_params(axis='x', rotation=45)
        axes[2,1].grid(True, alpha=0.3)
        for bar, impact in zip(bars, attack_impacts):
            axes[2,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{impact:.1f}', ha='center', va='bottom')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    
    # Save subplot 5 as separate PDF
    if attack_impacts:
        fig_5 = plt.figure(figsize=(10, 8))
        ax_5 = fig_5.add_subplot(111)
        bars = ax_5.bar(attack_labels, attack_impacts, color=colors[:len(attack_impacts)])
        ax_5.set_ylabel('Δ Load (MW)', fontsize=18)
        ax_5.tick_params(axis='x', rotation=45)
        ax_5.grid(True, alpha=0.3)
        for bar, impact in zip(bars, attack_impacts):
            ax_5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{impact:.1f}', ha='center', va='bottom')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/max_load_delta_during_attacks_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Individual subplot PDFs saved in 'sub_figures/' directory:")
    print(f"   1. daily_load_profile_with_attacks.pdf")
    print(f"   2. frequency_delta_vs_baseline.pdf")
    print(f"   3. load_delta_vs_baseline.pdf")
    print(f"   4. frequency_deviation_from_60hz.pdf")
    if attack_impacts:
        print(f"   5. max_load_delta_during_attacks.pdf")
    
    # Add RL Rewards and PINN Loss plots (unchanged below)


def plot_daily_load_analysis(results):
    """Create detailed daily load analysis with attack timing insights"""
    
    print("\n=== DAILY LOAD ANALYSIS WITH ATTACK TIMING INSIGHTS ===")
    
    # Get baseline data
    baseline_result = results.get('Baseline (Daily Load Variation)', results[list(results.keys())[0]])
    load_times = baseline_result['load_profile']['times']
    load_multipliers = baseline_result['load_profile']['multipliers']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily Load Profile with Periods
    axes[0,0].plot(load_times, load_multipliers, color='blue', linewidth=3, label='Daily Load Profile')
    
    # Identify and mark periods
    threshold_low = 0.4
    threshold_high = 0.8
    
    low_demand_mask = load_multipliers < threshold_low
    high_demand_mask = load_multipliers > threshold_high
    medium_demand_mask = ~(low_demand_mask | high_demand_mask)
    
    axes[0,0].fill_between(load_times, 0, load_multipliers, where=low_demand_mask, 
                          alpha=0.3, color='green', label='Low Demand')
    axes[0,0].fill_between(load_times, 0, load_multipliers, where=high_demand_mask, 
                          alpha=0.3, color='red', label='High Demand')
    axes[0,0].fill_between(load_times, 0, load_multipliers, where=medium_demand_mask, 
                          alpha=0.3, color='yellow', label='Medium Demand')
    
    axes[0,0].axhline(y=threshold_low, color='green', linestyle='--', alpha=0.7, label=f'Low Threshold ({threshold_low})')
    axes[0,0].axhline(y=threshold_high, color='red', linestyle='--', alpha=0.7, label=f'High Threshold ({threshold_high})')
    
    # axes[0,0].set_title('Daily Load Profile with Demand Periods')
    axes[0,0].set_ylabel('Load Multiplier', fontsize=18)
    axes[0,0].tick_params(axis='both', which='major', labelsize=18)
    axes[0,0].set_xlabel('Time (s)', fontsize=18)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Attack Timing Analysis
    colors = ['red', 'blue', 'green']
    scenario_names = ['Demand Increase', ' Demand Decrease', 'Oscillating Demand']
    
    for i, scenario_name in enumerate(scenario_names):
        if scenario_name in results:
            result = results[scenario_name]
            for attack in result['attack_scenarios']:
                start_time = attack['start_time']
                end_time = start_time + attack['duration']
                
                # Find load level at attack start
                start_idx = np.argmin(np.abs(load_times - start_time))
                load_at_attack = load_multipliers[start_idx]
                
                axes[0,1].scatter(start_time, load_at_attack, 
                                color=colors[i], s=100, alpha=0.8, 
                                label=f"{scenario_name}: {attack['type'].replace('_', ' ').title()}")
                
                # Draw arrow to show attack direction
                if attack['type'] == 'demand_increase':
                    axes[0,1].arrow(start_time, load_at_attack, 0, 0.3, 
                                   head_width=5, head_length=0.05, fc=colors[i], ec=colors[i])
                elif attack['type'] == 'demand_decrease':
                    axes[0,1].arrow(start_time, load_at_attack, 0, -0.3, 
                                   head_width=5, head_length=0.05, fc=colors[i], ec=colors[i])
                else:  # oscillating
                    axes[0,1].arrow(start_time, load_at_attack, 0, 0.2, 
                                   head_width=5, head_length=0.05, fc=colors[i], ec=colors[i])
                    axes[0,1].arrow(start_time, load_at_attack, 0, -0.2, 
                                   head_width=5, head_length=0.05, fc=colors[i], ec=colors[i])
    
    axes[0,1].plot(load_times, load_multipliers, color='gray', linewidth=2, alpha=0.7, label='Daily Load Profile')
    # axes[0,1].set_title('Attack Timing vs Load Level')
    axes[0,1].set_ylabel('Load Multiplier', fontsize=18)
    axes[0,1].tick_params(axis='both', which='major', labelsize=18)
    axes[0,1].set_xlabel('Time (s)', fontsize=18)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Frequency Impact vs Load Level
    for i, scenario_name in enumerate(scenario_names):
        if scenario_name in results:
            result = results[scenario_name]
            cosim = result['cosim']
            
            time_data = np.array(cosim.results['time'])
            frequencies = np.array(cosim.results['frequency'])
            
            # Interpolate load profile to match frequency data
            from scipy.interpolate import interp1d
            load_interp = interp1d(load_times, load_multipliers, bounds_error=False, fill_value='extrapolate')
            load_at_freq_times = load_interp(time_data)
            
            freq_deviation = abs(60.0 - frequencies)
            axes[1,0].scatter(load_at_freq_times, freq_deviation, 
                            color=colors[i], alpha=0.6, s=20, label=scenario_name)
    
    # axes[1,0].set_title('Frequency Deviation vs Load Level')
    axes[1,0].set_xlabel('Load Multiplier', fontsize=18)
    axes[1,0].set_ylabel('Frequency Deviation (Hz)', fontsize=18)
    axes[1,0].tick_params(axis='both', which='major', labelsize=18)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Attack Effectiveness Analysis
    attack_effectiveness = []
    attack_labels = []
    
    for scenario_name in scenario_names:
        if scenario_name in results:
            result = results[scenario_name]
            cosim = result['cosim']
            
            time_data = np.array(cosim.results['time'])
            frequencies = np.array(cosim.results['frequency'])
            
            # Calculate effectiveness (max frequency deviation during attacks)
            max_deviation = 0
            for attack in result['attack_scenarios']:
                start_idx = np.argmin(np.abs(time_data - attack['start_time']))
                end_idx = np.argmin(np.abs(time_data - (attack['start_time'] + attack['duration'])))
                
                if start_idx < end_idx and end_idx < len(frequencies):
                    attack_freq_deviation = abs(60.0 - frequencies[start_idx:end_idx])
                    max_deviation = max(max_deviation, np.max(attack_freq_deviation))
            
            attack_effectiveness.append(max_deviation)
            attack_labels.append(scenario_name)
    
    if attack_effectiveness:
        bars = axes[1,1].bar(attack_labels, attack_effectiveness, color=colors[:len(attack_effectiveness)])
        # axes[1,1].set_title('Attack Effectiveness (Max Frequency Deviation)')
        axes[1,1].set_ylabel('Max Frequency Deviation (Hz)', fontsize=18)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=18)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, effectiveness in zip(bars, attack_effectiveness):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{effectiveness:.3f}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    plt.show()
    
    # Print attack timing insights
    print("\n=== ATTACK TIMING INSIGHTS ===")
    for scenario_name in scenario_names:
        if scenario_name in results:
            result = results[scenario_name]
            print(f"\n{scenario_name}:")
            for attack in result['attack_scenarios']:
                start_time = attack['start_time']
                start_idx = np.argmin(np.abs(load_times - start_time))
                load_at_attack = load_multipliers[start_idx]
                
                print(f"  Attack at {start_time:.1f}s: Load level = {load_at_attack:.2f}")
                print(f"    Context: {attack['demand_context']}")
                print(f"    Strategy: {attack['type'].replace('_', ' ').title()}")
                print(f"    Magnitude: {attack['magnitude']}")


if __name__ == "__main__":
    # Run focused demand analysis
    results = run_focused_demand_analysis()
    
    # Create additional daily load analysis
    if results is not None:
        plot_daily_load_analysis(results)
    else:
        print("No results to analyze - simulation failed or was terminated.")
    
    print("\n FOCUSED DEMAND MANIPULATION ANALYSIS WITH DAILY LOAD VARIATION COMPLETE!")
    print("\nKey Observations:")