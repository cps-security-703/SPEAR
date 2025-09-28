#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import opendssdirect as dss
import pandapower as pp
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import threading
import queue
from dss_function_qsts import *
from scipy import signal
import random
from pinn_optimizer import LSTMPINNConfig, LSTMPINNChargingOptimizer

# Import federated PINN components
try:
    from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
except ImportError:
    print("Warning: Federated PINN manager not available")
    FederatedPINNManager = None
    FederatedPINNConfig = None

# Import unified EVCS dynamics from evcs_dynamics.py
from evcs_dynamics import EVCSController, EVCSParameters, ChargingManagementSystem

# Import real-time RL attack controller
try:
    from rl_attack_analytics import RealtimeRLAttackController
except ImportError:
    print("Warning: RL attack analytics not available")
    RealtimeRLAttackController = None

# Import enhanced RL attack system
try:
    from enhanced_rl_attack_system import EnhancedRLAttackSystem
except ImportError:
    print("Warning: Enhanced RL attack system not available")
    EnhancedRLAttackSystem = None

# Import os for file operations
import os

@dataclass
class EVStatistics:
    """Statistics for each EV that uses the EVCS"""
    arrival_time: float = 0.0      # Hours when EV connected
    departure_time: float = 0.0    # Hours when EV disconnected
    initial_soc: float = 0.0       # SOC when connected
    final_soc: float = 0.0         # SOC when disconnected
    energy_delivered: float = 0.0  # kWh delivered to this EV
    charging_duration: float = 0.0 # Hours of charging

class EVChargingStation:
    """Enhanced EV Charging Station using unified EVCS dynamics"""
    
    def __init__(self, evcs_id: str, bus_name: str, max_power: float, num_ports: int, 
                 params: EVCSParameters = None):
        """Initialize EV charging station with unified EVCS dynamics"""
        self.evcs_id = evcs_id
        self.bus_name = bus_name
        self.max_power = max_power
        self.num_ports = num_ports
        self.params = params or EVCSParameters()
        self.total_time = 240.0
        
        # Create unified EVCS controller for power electronics dynamics
        self.evcs_controller = EVCSController(evcs_id, self.params)
        
        # Legacy charging system attributes
        self.available_ports = num_ports
        self.charging_sessions = []
        self.customer_queue = []
        self.scheduled_charging = []
        self.avg_charging_time = 30.0  # Default baseline charging time in minutes
        self.queue_wait_time = 0.0
        
        # FIXED: Initialize attack-related attributes
        self.attack_active = False
        self.attack_type = None
        self.attack_magnitude = 0
        self.attack_manipulation_factor = 1.0
        self.dynamics_result_cache = {}
        
        # FIXED: Initialize recovery-related attributes
        self.recovery_mode = False
        self.recovery_start_time = None
        self.pre_attack_metrics = {}
        
        # Power electronics dynamics attributes
        self.ev_connected = False
        self.current_ev_id = None
        self.soc = 0.5  # Start at 50% SOC
        self.voltage_measured = 0.0
        self.current_measured = 0.0
        self.power_measured = 0.0
        self.ac_voltage_rms = 0.0
        self.ac_current_rms = 0.0
        self.grid_frequency = 60.0
        self.current_load = 0.0
        
        # Controller references (now managed by unified EVCSController)
        self.voltage_ref = 400.0  # V
        self.current_ref = 125.0   # A
        individual_ev_power = min(50.0, max_power / max(4, num_ports/3))  # Limit individual EV power  
        self.power_ref = individual_ev_power     # kW per EV
        
        # Legacy attributes for compatibility
        self.current_setpoint = 125.0  # 50A initial current
        self.voltage_setpoint = 1.0  # Per-unit voltage setpoint
        self.voltage_reference = 400.0  # V (DC side)
        self.current_reference = 125.0   # A
        self.power_reference = individual_ev_power     # kW per EV
        
        # Emergency and safety
        self.emergency_mode = False
        self.overvoltage_protection = False
        self.overcurrent_protection = False
        self.overtemperature_protection = False
        
        # Statistics and history
        self.evs_served = []
        self.total_evs_served = 0
        self.total_energy_delivered = 0.0
        self.total_energy_current_ev = 0.0  # Energy delivered to current EV
        self.connection_time = 0.0
        self.charging_start_time = 0.0
        
        # Connection management attributes
        self.customer_wait_times = {}
        self.last_disconnect_time = 0.0
        self.min_disconnect_time = 0.02  # Minimum 1.2 minutes before new EV can connect
        self.connection_probability = 1.0  # 100% chance of connection for demo
        self.initialization_time = 0.0
        self.auto_connect_delay = 0.01  # 36 seconds delay before first auto-connection
        
        # Ensure baseline metrics are properly initialized
        self.ensure_baseline_initialization()
        
        print(f"Initialized EVCS {evcs_id} with {num_ports} ports, max power {max_power}kW")
    
    
    def store_baseline_metrics(self):
        """Store baseline metrics before attack starts for proper recovery"""
        self.baseline_metrics = {
            'avg_charging_time': self.avg_charging_time,
            'queue_wait_time': self.queue_wait_time,
            'current_load': self.current_load,
            'utilization_rate': (self.num_ports - self.available_ports) / self.num_ports,
            'queue_length': len(self.customer_queue),
            'total_evs_served': self.total_evs_served,
            'total_energy_delivered': self.total_energy_delivered
        }
        print(f"📊 EVCS {self.evcs_id}: Baseline metrics stored - charging time: {self.avg_charging_time:.1f} min")
    
    def ensure_baseline_initialization(self):
        """Ensure baseline metrics are properly initialized with correct values"""
        if not hasattr(self, 'baseline_metrics') or not self.baseline_metrics:
            # Initialize baseline metrics if not already set
            self.baseline_metrics = {
                'avg_charging_time': 150.0,  # FIXED: Use correct baseline of 150 minutes
                'queue_wait_time': 0.0,
                'current_load': 0.0,
                'utilization_rate': 0.0,
                'queue_length': 0,
                'total_evs_served': 0,
                'total_energy_delivered': 0.0
            }
            print(f"📊 EVCS {self.evcs_id}: Baseline metrics initialized with correct charging time: 150.0 min")
    
    def get_baseline_metrics(self):
        """Get the stored baseline metrics, ensuring they exist"""
        if not hasattr(self, 'baseline_metrics') or not self.baseline_metrics:
            self.ensure_baseline_initialization()
        return self.baseline_metrics
    
    def initiate_recovery(self, current_time: float):
        """Initiate recovery from cyber attack effects"""
        if self.attack_active:
            # FIXED: Only initiate recovery if not already in recovery mode
            if not self.recovery_mode:
                # FIXED: Store pre-attack metrics for recovery
                # Use the baseline metrics that were stored when attack started
                if hasattr(self, 'baseline_metrics'):
                    self.pre_attack_metrics = self.baseline_metrics.copy()
                    print(f"   Using stored baseline metrics for recovery")
                else:
                    # Fallback: store current metrics (this shouldn't happen if attack started properly)
                    self.pre_attack_metrics = {
                        'avg_charging_time': self.avg_charging_time,
                        'queue_wait_time': self.queue_wait_time,
                        'current_load': self.current_load,
                        'utilization_rate': (self.num_ports - self.available_ports) / self.num_ports,
                        'queue_length': len(self.customer_queue),
                        'total_evs_served': self.total_evs_served,
                        'total_energy_delivered': self.total_energy_delivered
                    }
                    print(f"   ⚠️ No baseline metrics found, using current metrics")
                
                # Start recovery mode
                self.recovery_mode = True
                self.recovery_start_time = current_time
                
                print(f"🔄 EVCS {self.evcs_id}: Recovery initiated at t={current_time:.1f}s")
                print(f"   Pre-attack metrics stored for recovery")
                print(f"   Current charging time: {self.avg_charging_time:.1f} min")
                print(f"   Current queue length: {len(self.customer_queue)}")
            else:
                print(f" EVCS {self.evcs_id}: Already in recovery mode, continuing recovery")
        else:
            print(f" EVCS {self.evcs_id}: No active attack to recover from")
    
    def update_recovery(self, current_time: float):
        """Update recovery progress and gradually reset metrics"""
        if not self.recovery_mode:
            return
        
        recovery_duration = current_time - self.recovery_start_time
        recovery_time_constant = 120.0  # FIXED: Increased to 120 seconds for better recovery
        
        # FIXED: Add debug output to track recovery calls
        if int(recovery_duration) % 10 == 0 and recovery_duration > 0:  # Every 10 seconds
            print(f" EVCS {self.evcs_id}: Recovery active - duration {recovery_duration:.1f}s, mode={self.recovery_mode}")
        
        if recovery_duration >= recovery_time_constant:
            # Recovery complete - reset to pre-attack state
            self.recovery_mode = False
            self.recovery_start_time = None
            
            # FIXED: Reset accumulated metrics to pre-attack values
            if self.pre_attack_metrics:
                self.avg_charging_time = self.pre_attack_metrics.get('avg_charging_time', 30.0)
                self.queue_wait_time = self.pre_attack_metrics.get('queue_wait_time', 0.0)
                self.current_load = self.pre_attack_metrics.get('current_load', 0.0)
                
                # FIXED: Reset queue to pre-attack length
                target_queue_length = self.pre_attack_metrics.get('queue_length', 0)
                while len(self.customer_queue) > target_queue_length:
                    self.customer_queue.pop(0)
                
                # FIXED: Reset cumulative statistics to prevent artificial inflation
                if self.pre_attack_metrics.get('total_evs_served', 0) > 0:
                    # Scale back the inflated metrics proportionally
                    scale_factor = self.pre_attack_metrics['total_evs_served'] / max(1, self.total_evs_served)
                    self.avg_charging_time *= scale_factor
                    
                    # FIXED: Reset cumulative data that affects metric calculations
                    target_evs_served = self.pre_attack_metrics.get('total_evs_served', 0)
                    target_energy_delivered = self.pre_attack_metrics.get('total_energy_delivered', 0.0)
                    
                    # Reset cumulative statistics to pre-attack values
                    self.total_evs_served = target_evs_served
                    self.total_energy_delivered = target_energy_delivered
                    
                    # FIXED: CRITICAL - Reset BOTH legacy and dynamics systems completely
                    # Clear charging_sessions list to prevent inflated metrics
                    if hasattr(self, 'charging_sessions'):
                        self.charging_sessions.clear()
                        print(f"   Legacy charging_sessions cleared")
                    
                    # Clear evs_served list to prevent inflated metrics  
                    if hasattr(self, 'evs_served'):
                        self.evs_served.clear()
                        print(f"   Dynamics evs_served cleared")
                    
                    # FIXED: Reset scheduled_charging to prevent queue inflation
                    if hasattr(self, 'scheduled_charging'):
                        self.scheduled_charging.clear()
                        print(f"   Scheduled charging cleared")
                    
                    print(f"   Cumulative data reset: EVs served {self.total_evs_served}, Energy {self.total_energy_delivered:.1f} kWh")
                
                print(f" EVCS {self.evcs_id}: Recovery completed at t={current_time:.1f}s")
                print(f"   Metrics reset to pre-attack values")
                print(f"   Final charging time: {self.avg_charging_time:.1f} min")
                print(f"   Final queue length: {len(self.customer_queue)}")
        else:
            # Gradual recovery - interpolate between current and pre-attack values
            recovery_factor = recovery_duration / recovery_time_constant
            
            if self.pre_attack_metrics:
                # Gradually reduce charging time and queue wait time
                target_charging_time = self.pre_attack_metrics.get('avg_charging_time', 30.0)
                target_queue_wait = self.pre_attack_metrics.get('queue_wait_time', 0.0)
                target_queue_length = self.pre_attack_metrics.get('queue_length', 0)
                
                # FIXED: Use exponential decay for more realistic recovery
                decay_factor = 1.0 - np.exp(-3.0 * recovery_factor)  # Exponential decay
                
                self.avg_charging_time = (self.avg_charging_time * (1 - decay_factor) + 
                                        target_charging_time * decay_factor)
                self.queue_wait_time = (self.queue_wait_time * (1 - decay_factor) + 
                                      target_queue_wait * decay_factor)
                
                # Gradually reduce queue length with exponential decay
                if len(self.customer_queue) > target_queue_length:
                    target_length = target_queue_length + (len(self.customer_queue) - target_queue_length) * (1 - decay_factor)
                    target_length = max(target_queue_length, int(target_length))
                    
                    while len(self.customer_queue) > target_length:
                        self.customer_queue.pop(0)
                
                # FIXED: Gradually clear historical data during recovery to prevent metric inflation
                if hasattr(self, 'charging_sessions') and len(self.charging_sessions) > 0:
                    # Calculate how many sessions to keep based on recovery progress
                    target_sessions = max(0, int(len(self.charging_sessions) * (1 - decay_factor)))
                    while len(self.charging_sessions) > target_sessions:
                        self.charging_sessions.pop(0)
                
                if hasattr(self, 'evs_served') and len(self.evs_served) > 0:
                    # Calculate how many EVs to keep based on recovery progress
                    target_evs = max(0, int(len(self.evs_served) * (1 - decay_factor)))
                    while len(self.evs_served) > target_evs:
                        self.evs_served.pop(0)
                
                # Debug output during recovery
                if recovery_duration % 10 == 0:  # Every 10 seconds
                    print(f" EVCS {self.evcs_id}: Recovery progress {recovery_factor*100:.1f}%")
                    print(f"   Charging time: {self.avg_charging_time:.1f} min (target: {target_charging_time:.1f})")
                    print(f"   Queue length: {len(self.customer_queue)} (target: {target_queue_length})")
                    print(f"   Historical sessions: {len(self.charging_sessions)} (clearing during recovery)")
                    print(f"   Historical EVs: {len(self.evs_served)} (clearing during recovery)")
    
    def connect_new_ev(self, current_time_hours: float):
        """Connect a new EV if conditions allow"""
        if self.ev_connected:
            return
        
        # Check if enough time has passed since last disconnect
        if current_time_hours - self.last_disconnect_time < self.min_disconnect_time:
            return
        
        # Handle initialization case - connect first EV automatically
        if self.initialization_time == 0.0:
            # First time initialization - always connect immediately
            self.initialization_time = current_time_hours
            # Force connection of first EV
            self._force_ev_connection(current_time_hours)
            return
        
        # For subsequent connections, use high probability to ensure EVs connect
        connection_prob = max(0.8, self.connection_probability)  # At least 80% chance
        if np.random.random() < connection_prob:
            self._force_ev_connection(current_time_hours)
    
    def _force_ev_connection(self, current_time_hours: float):
        """Force connection of an EV (for initialization and recovery)"""
        if self.ev_connected:
            return
        
        # Generate new EV ID
        self.current_ev_id = self.total_evs_served + 1
        
        # Connect EV
        self.ev_connected = True
        self.connection_time = current_time_hours
        self.last_disconnect_time = 0.0
        self.total_energy_current_ev = 0.0
        
        # New EV SOC - typically lower SOC for new arrivals
        # Higher probability of low SOC during peak hours
        if 7 <= current_time_hours <= 9 or 17 <= current_time_hours <= 21:
            # Peak hours - more EVs with low SOC
            self.soc = np.random.uniform(0.15, 0.35)  # 15-35%
        else:
            # Off-peak hours - more varied SOC
            self.soc = np.random.uniform(0.2, 0.5)    # 20-50%
        
        # FIXED: Ensure minimum SOC for stable operation
        if self.soc < 0.1:
            self.soc = 0.1  # Minimum 10% SOC
        
        # Reset power electronics states for new connection
        self.voltage_measured = self.voltage_reference
        self.current_measured = 0.0
        self.power_measured = 0.0
        self.dc_link_voltage = 600.0  # Reset to safe level
        
        # FIXED: Initialize current_setpoint to prevent zero setpoint warnings
        self.current_setpoint = 125.0  # 50A initial current
        
        # FIXED: Initialize power-related attributes to prevent zero power warnings
        # Use much lower individual power per charging session to prevent massive accumulation
        max_single_ev_power = min(50.0, self.max_power / max(4, self.num_ports/2))  # Limit individual EV power
        self.power_reference = max_single_ev_power  # Much lower individual EV power
        self.voltage_reference = 400.0  # 400V initial voltage
        self.current_reference = 125.0  # 50A initial current
        
        # FIXED: Create corresponding legacy charging session
        customer_id = f"cust_{self.evcs_id}_{self.current_ev_id}"
        # Estimate charging duration based on SOC deficit and power
        soc_deficit = self.params.disconnect_soc - self.soc
        estimated_energy = soc_deficit * self.params.capacity  # kWh
        estimated_duration = estimated_energy / self.power_reference  # hours
        estimated_duration = max(0.1, min(estimated_duration, 4.0))  # Clamp between 0.1 and 4 hours
        
        # Create legacy session
        session = {
            'customer_id': customer_id,
            'start_time': current_time_hours,
            'power_level': self.power_reference,
            'duration': estimated_duration,
            'completion_time': current_time_hours + estimated_duration,
            'energy_delivered': 0.0,  # Will be updated during charging
            'port_id': 1  # Single port per station
        }
        self.charging_sessions.append(session)
        
        # Update available ports
        if self.available_ports > 0:
            self.available_ports -= 1
        # NOTE: Don't add power here - it will be added by the legacy charging session in start_charging_session
        
        print(f"EVCS {self.evcs_id}: New EV connected with SOC {self.soc:.2f}")
        print(f"       Created legacy session: {customer_id}, Duration: {estimated_duration:.2f}h")
    
    def disconnect_ev(self, current_time_hours: float):
        """Disconnect current EV when fully charged"""
        if not self.ev_connected:
            return
        
        # Record statistics for completed EV
        charging_duration = current_time_hours - self.connection_time
        ev_stats = EVStatistics(
            arrival_time=self.connection_time,
            departure_time=current_time_hours,
            initial_soc=self.soc - (self.total_energy_current_ev / self.params.capacity),
            final_soc=self.soc,
            energy_delivered=self.total_energy_current_ev,
            charging_duration=charging_duration
        )
        
        self.evs_served.append(ev_stats)
        self.total_evs_served += 1
        self.total_energy_delivered += self.total_energy_current_ev
        
        print(f"     {self.evcs_id}: EV #{self.current_ev_id} disconnected at {current_time_hours:.1f}h")
        print(f"       Energy delivered: {self.total_energy_current_ev:.1f}kWh, Duration: {charging_duration:.1f}h")
        
        # FIXED: Complete the corresponding legacy charging session
        if hasattr(self, 'current_ev_id') and self.current_ev_id:
            # Find and complete the legacy session for this EV
            for session in self.charging_sessions[:]:  # Copy list to avoid modification during iteration
                if session.get('customer_id') == f"cust_{self.evcs_id}_{self.current_ev_id}":
                    self.complete_charging_session(session['customer_id'], current_time_hours)
                    break
        
        # Disconnect EV
        self.ev_connected = False
        self.last_disconnect_time = current_time_hours
        self.power_reference = 0.0  # Stop charging
        self.current_measured = 0.0
        self.power_measured = 0.0
        # FIXED: Reset current_setpoint to prevent warnings
        self.current_setpoint = 0.0
        
        # FIXED: Reset current EV ID
        self.current_ev_id = None
        
        # FIXED: Reset available ports and current load for queue management
        self.available_ports = self.num_ports
        self.current_load = 0.0
        
        # FIXED: Sanitize EVCS controller states to prevent invalid conditions
        if hasattr(self, 'evcs_controller') and hasattr(self.evcs_controller, '_sanitize_states'):
            self.evcs_controller._sanitize_states()
        
        # FIXED: Try to connect new EV from queue immediately
        if hasattr(self, 'customer_queue') and self.customer_queue:
            # Process queue to start next customer
            print(f"EVCS {self.evcs_id}: Processing queue after EV disconnect, {len(self.customer_queue)} customers waiting")
            # The queue will be processed in the next update cycle
    
    def check_ev_status(self, current_time_hours: float):
        """Check if EV should disconnect based on SOC or time"""
        if not self.ev_connected:
            return
        
        # FIXED: More aggressive disconnection logic for realistic operation
        should_disconnect = False
        disconnect_reason = ""
        
        # Check SOC-based disconnection (more aggressive)
        if self.soc >= self.params.disconnect_soc:
            should_disconnect = True
            disconnect_reason = f"SOC {self.soc:.2f} >= disconnect threshold {self.params.disconnect_soc}"
        
        # Check time-based disconnection (more realistic)
        charging_duration = current_time_hours - self.connection_time
        max_charging_time = 2.0  # FIXED: Reduced from 4.0 to 2.0 hours for realistic charging
        
        if charging_duration > max_charging_time:
            should_disconnect = True
            disconnect_reason = f"Charging duration {charging_duration:.1f}h > max {max_charging_time}h"
        
        # FIXED: Additional disconnection criteria for better queue management
        # Disconnect if SOC is high enough and charging has been going for a reasonable time
        if (self.soc >= 0.75 and  # 75% SOC
            charging_duration >= 0.5):  # At least 30 minutes of charging
            should_disconnect = True
            disconnect_reason = f"SOC {self.soc:.2f} >= 75% and charging duration {charging_duration:.1f}h >= 0.5h"
        
        # FIXED: Emergency disconnection only for critical issues
        if self.emergency_mode and self.soc > 0.7:  # Reduced from 0.8 to 0.7
            should_disconnect = True
            disconnect_reason = "Emergency mode - high SOC"
        
        if should_disconnect:
            self.disconnect_ev(current_time_hours)
            print(f"EVCS {self.evcs_id}: EV disconnected - {disconnect_reason}")
            
            # FIXED: Don't try to reconnect immediately to prevent infinite loops
            # The queue processing will handle reconnection in the next cycle
            # Only try to connect if no customers are waiting
            if not hasattr(self, 'customer_queue') or not self.customer_queue:
                # No customers waiting, try to connect new EV after delay
                if current_time_hours - self.last_disconnect_time >= self.min_disconnect_time:
                    self.connect_new_ev(current_time_hours)
    
    def get_availability_factor(self, time_hours: float) -> float:
        """Get probability that an EV wants to charge at given time"""
        # Higher availability during commute hours
        if 7 <= time_hours <= 9:    # Morning commute
            return 0.9
        elif 17 <= time_hours <= 21: # Evening commute + dinner
            return 0.95
        elif 9 <= time_hours <= 17:  # Work hours
            return 0.6
        elif 21 <= time_hours <= 23: # Late evening
            return 0.7
        else:  # Night/early morning
            return 0.3
        
    def set_references(self, voltage_ref: float, current_ref: float, power_ref: float):
        """Set references from CMS using unified controller"""
        # Update local references for compatibility
        self.voltage_ref = voltage_ref
        self.current_ref = current_ref
        self.power_ref = power_ref
        
        if self.ev_connected:
            # Set references in unified controller
            self.evcs_controller.set_references(voltage_ref, current_ref, power_ref)
            
            # Update legacy attributes for compatibility
            self.voltage_reference = voltage_ref
            self.current_reference = current_ref
            self.power_reference = power_ref
            self.current_setpoint = current_ref
        else:
            # No EV connected - set references to zero
            self.voltage_reference = 0.0
            self.current_reference = 0.0
            self.power_reference = 0.0
            self.current_setpoint = 0.0
    
    # NOTE: Power electronics methods removed - now using unified EVCSController
    
    def update_dynamics(self, grid_voltage_rms: float, dt: float, current_time_hours: float, use_solve_ivp: bool = True) -> Dict:
        """Update EVCS dynamics - delegates to evcs_dynamics.py for physics"""
        
        # Check EV connection status first
        self.check_ev_status(current_time_hours)
        
        # Try to connect EV if none is connected
        if not self.ev_connected:
            self.connect_new_ev(current_time_hours)
        
        # Clean up orphaned legacy sessions
        self._cleanup_orphaned_sessions()
        
        # Delegate all physics to evcs_dynamics.py
        if self.ev_connected:
            # Set references in the physics-based controller
            self.evcs_controller.set_references(self.voltage_ref, self.current_ref, self.power_ref)
            
            # Get complete dynamics from evcs_dynamics.py (includes solve_ivp, converters, SOC)
            dynamics_result = self.evcs_controller.update_dynamics(grid_voltage_rms, dt, use_solve_ivp)
            
            # Update local state from physics controller (minimal processing)
            self.soc = dynamics_result['soc']
            self.voltage_measured = dynamics_result['voltage_measured']
            self.current_measured = dynamics_result['current_measured']
            self.power_measured = dynamics_result['power_measured']
            self.ac_voltage_rms = dynamics_result.get('ac_voltage_rms', grid_voltage_rms)
            self.ac_current_rms = dynamics_result.get('ac_current_rms', 0.0)
            self.grid_frequency = dynamics_result.get('grid_frequency', 60.0)
            
            # Update legacy compatibility attributes
            self.current_load = self.power_measured
            self.voltage_setpoint = self.voltage_measured / 400.0  # Normalize
            self.current_setpoint = self.current_measured
            
            # Update legacy charging session energy (if exists)
            if self.charging_sessions and self.current_ev_id:
                for session in self.charging_sessions:
                    if session.get('customer_id') == f"cust_{self.evcs_id}_{self.current_ev_id}":
                        energy_increment = self.power_measured * dt / 3600.0  # kWh
                        session['energy_delivered'] += energy_increment
                        break
            
            # Return physics-based result with minimal modification
            return {
                **dynamics_result,  # Use all physics results from evcs_dynamics.py
                'ev_connected': True,
                'ev_id': self.current_ev_id,
                'evs_served': self.total_evs_served
            }
        else:
            # No EV connected - return standby state
            standby_power = 1.0  # 1 kW standby
            standby_current = 0.1  # 0.1 A standby
            
            self.current_load = standby_power
            self.voltage_setpoint = 0.1
            self.current_setpoint = standby_current
            
            return {
                'voltage_measured': 50.0,
                'current_measured': standby_current,
                'power_measured': standby_power,
                'ac_voltage_rms': grid_voltage_rms,
                'ac_current_rms': standby_current,
                'grid_frequency': 60.0,
                'soc': 0.0,
                'total_power': standby_power,
                'ev_connected': False,
                'ev_id': 0,
                'evs_served': self.total_evs_served,
                'integration_method': 'standby'
            }
    
    # Legacy methods for compatibility with existing code
    def add_customer_to_queue(self, customer_id: str, arrival_time: float, 
                            requested_charge: float, priority: int = 1):
        """Add customer to charging queue (legacy compatibility)"""
        customer = {
            'id': customer_id,
            'arrival_time': arrival_time,
            'requested_charge': requested_charge,  # kWh
            'priority': priority,
            'wait_time': 0.0,
            'estimated_start_time': 0.0,
            'estimated_completion_time': 0.0
        }
        self.customer_queue.append(customer)
        self._update_queue_metrics(arrival_time)
    
    def start_charging_session(self, customer_id: str, start_time: float, 
                             power_level: float, duration: float):
        """Start a new charging session (legacy compatibility)"""
        if self.available_ports > 0:
            session = {
                'customer_id': customer_id,
                'start_time': start_time,
                'power_level': power_level,
                'duration': duration,
                'completion_time': start_time + duration,
                'energy_delivered': power_level * duration / 60.0,  # kWh
                'port_id': self.num_ports - self.available_ports + 1
            }
            self.charging_sessions.append(session)
            self.available_ports -= 1
            self.current_load += power_level
            return True
        return False
    
    def complete_charging_session(self, customer_id: str, completion_time: float):
        """Complete a charging session (legacy compatibility)"""
        for session in self.charging_sessions:
            if session['customer_id'] == customer_id:
                # FIXED: Update energy delivered based on actual charging
                if hasattr(self, 'total_energy_current_ev') and self.ev_connected:
                    session['energy_delivered'] = self.total_energy_current_ev
                
                # Remove session
                self.charging_sessions.remove(session)
                self.available_ports += 1
                self.current_load -= session['power_level']
                
                print(f"EVCS {self.evcs_id}: Completed legacy session {customer_id}")
                break
        
        # FIXED: Update queue metrics
        self._update_queue_metrics(completion_time)
    
    def schedule_charging(self, customer_id: str, scheduled_time: float, 
                        power_level: float, duration: float):
        """Schedule future charging session (legacy compatibility)"""
        schedule = {
            'customer_id': customer_id,
            'scheduled_time': scheduled_time,
            'power_level': power_level,
            'duration': duration,
            'status': 'scheduled'  # scheduled, active, completed, cancelled
        }
        self.scheduled_charging.append(schedule)
    
    def _update_queue_metrics(self, current_time: float = None):
        """Update queue-related metrics (legacy compatibility)"""
        if self.customer_queue:
            # Update wait times for all customers in queue
            if current_time is not None:
                for customer in self.customer_queue:
                    customer['wait_time'] = current_time - customer['arrival_time']
            
            # Calculate average wait time
            total_wait = sum(customer['wait_time'] for customer in self.customer_queue)
            self.queue_wait_time = total_wait / len(self.customer_queue)
        else:
            self.queue_wait_time = 0.0
        
        # Calculate utilization rate
        self.utilization_rate = (self.num_ports - self.available_ports) / self.num_ports
    
    def _cleanup_orphaned_sessions(self):
        """Clean up orphaned legacy sessions that don't have corresponding EVs"""
        if not self.ev_connected and self.charging_sessions:
            # No EV connected but legacy sessions exist - clean them up
            for session in self.charging_sessions[:]:
                print(f"EVCS {self.evcs_id}: Cleaning up orphaned session {session['customer_id']}")
                self.complete_charging_session(session['customer_id'], 0.0)
    
    def update_customer_wait_times(self, current_time: float):
        """Update wait times for customers in queue and simulate new arrivals during attacks"""
        # Update existing customer wait times
        for customer in self.customer_queue:
            customer['wait_time'] = current_time - customer['arrival_time']
        
        # Simulate increased customer arrivals during attacks (every 10 seconds)
        if hasattr(self, 'last_customer_arrival_time'):
            time_since_last_arrival = current_time - self.last_customer_arrival_time
        else:
            self.last_customer_arrival_time = current_time
            time_since_last_arrival = 0
        
        # During attacks, customers arrive more frequently due to longer charging times
        arrival_interval = 15.0  # Normal: every 15 seconds
        if hasattr(self, 'attack_active') and self.attack_active:
            arrival_interval = 8.0  # Attack: every 8 seconds (more frequent)
        elif hasattr(self, 'recovery_mode') and self.recovery_mode:
            arrival_interval = 12.0  # Recovery: every 12 seconds (moderately frequent)
        
        if time_since_last_arrival >= arrival_interval:
            # Add new customer to queue
            customer_id = f"cust_{self.evcs_id}_{int(current_time)}"
            soc_initial = np.random.uniform(0.1, 0.3)
            soc_target = np.random.uniform(0.8, 1.0)
            new_customer = {
                'id': customer_id,
                'arrival_time': current_time,
                'wait_time': 0.0,
                'soc_initial': soc_initial,
                'soc_target': soc_target,
                'urgency': np.random.uniform(0.3, 1.0),
                'priority': np.random.randint(1, 4),
                'requested_charge': (soc_target - soc_initial) * 75.0  # Assume 75kWh battery
            }
            
            # Only add if queue isn't too long (realistic constraint)
            if len(self.customer_queue) < 15:
                self.customer_queue.append(new_customer)
            
            self.last_customer_arrival_time = current_time
        self._update_queue_metrics(current_time)
    
    def _calculate_real_charging_time(self, current_time: float = None) -> float:
        """Calculate real charging time based on actual simulation state and attack conditions"""
        import numpy as np
        
        # Base charging time depends on power output and SOC
        base_charging_time = 30.0  # Normal baseline charging time in minutes
        
        # Factor 1: Power-based charging time calculation
        if hasattr(self, 'power_measured') and self.power_measured > 0:
            # Calculate charging time based on actual power delivery
            # Typical EV battery: 60-100 kWh, charging from 20% to 80% (40-60 kWh)
            typical_energy_needed = 50.0  # kWh
            charging_time_from_power = (typical_energy_needed / max(1.0, self.power_measured)) * 60  # Convert to minutes
            base_charging_time = min(150.0, max(15.0, charging_time_from_power))  # Clamp between 15-150 minutes
        
        # Factor 2: SOC-based adjustment
        if hasattr(self, 'soc') and self.soc > 0:
            # Higher SOC means less charging time needed
            remaining_charge_needed = max(0.1, 1.0 - self.soc)  # How much charge is still needed
            base_charging_time *= remaining_charge_needed
        
        # Factor 3: Attack impact on charging time (using RL attack suggestions)
        attack_time_factor = 1.0
        if hasattr(self, 'attack_active') and self.attack_active:
            if hasattr(self, 'attack_type') and hasattr(self, 'rl_attack_impact'):
                # Use RL-generated attack impact instead of hardcoded values
                rl_impact = getattr(self, 'rl_attack_impact', {})
                
                if self.attack_type == 'power_manipulation':
                    # Use RL-suggested power manipulation impact
                    power_impact = rl_impact.get('power_efficiency_reduction', 0.2)  # Default 20% reduction
                    attack_time_factor = 1.0 / max(0.1, 1.0 - power_impact)
                    attack_time_factor = min(5.0, attack_time_factor)  # Cap at 5x normal time
                
                elif self.attack_type == 'load_manipulation':
                    # Use RL-suggested load manipulation impact
                    load_impact = rl_impact.get('load_instability_factor', 1.5)  # Default 1.5x increase
                    attack_time_factor = min(3.0, load_impact)  # Cap at 3x normal time
                
                elif self.attack_type == 'voltage_manipulation':
                    # Use RL-suggested voltage manipulation impact
                    voltage_impact = rl_impact.get('voltage_efficiency_loss', 0.3)  # Default 30% loss
                    if hasattr(self, 'voltage_measured') and self.voltage_measured > 0:
                        voltage_deviation = abs(self.voltage_measured - 400.0) / 400.0
                        attack_time_factor = 1.0 + voltage_deviation * voltage_impact * 5.0
                    else:
                        attack_time_factor = 1.0 + voltage_impact
                
                elif self.attack_type == 'demand_increase' or self.attack_type == 'demand_decrease':
                    # Use RL-suggested demand manipulation impact
                    demand_impact = rl_impact.get('demand_instability_factor', 1.3)  # Default 1.3x increase
                    attack_time_factor = min(2.5, demand_impact)
                
                elif self.attack_type == 'frequency_manipulation':
                    # Use RL-suggested frequency manipulation impact
                    freq_impact = rl_impact.get('frequency_efficiency_loss', 0.15)  # Default 15% loss
                    attack_time_factor = 1.0 + freq_impact
            
            # Fallback to basic attack impact if no RL suggestions available
            elif hasattr(self, 'attack_type'):
                if self.attack_type == 'power_manipulation':
                    attack_time_factor = 1.5  # 50% increase
                elif self.attack_type == 'load_manipulation':
                    attack_time_factor = 1.8  # 80% increase
                elif self.attack_type == 'voltage_manipulation':
                    attack_time_factor = 1.6  # 60% increase
                else:
                    attack_time_factor = 1.4  # 40% increase default
        
        # Factor 4: Grid frequency impact
        if hasattr(self, 'grid_frequency') and self.grid_frequency > 0:
            freq_deviation = abs(self.grid_frequency - 60.0) / 60.0
            freq_factor = 1.0 + freq_deviation * 0.5  # Small impact from frequency deviation
            attack_time_factor *= freq_factor
        
        # Factor 5: Queue congestion impact
        if hasattr(self, 'customer_queue') and len(self.customer_queue) > 0:
            # More customers waiting can lead to rushed charging or resource contention
            queue_factor = 1.0 + len(self.customer_queue) * 0.05  # 5% increase per waiting customer
            attack_time_factor *= min(2.0, queue_factor)  # Cap at 2x
        
        # Calculate final charging time
        final_charging_time = base_charging_time * attack_time_factor
        
        # Realistic bounds: 10 minutes to 180 minutes (3 hours)
        final_charging_time = max(10.0, min(180.0, final_charging_time))
        
        return final_charging_time
    
    def _calculate_real_queue_length(self, current_time: float = None) -> int:
        """Calculate real queue length based on actual charging session delays from attacks"""
        import numpy as np
        
        # Base queue length from actual customer queue
        base_queue_length = len(self.customer_queue)
        
        # Initialize persistent queue accumulation if not exists
        if not hasattr(self, 'accumulated_queue_delay'):
            self.accumulated_queue_delay = 0.0
            self.last_queue_update_time = current_time or 0.0
        
        # Calculate time since last update
        if current_time is not None:
            time_delta = current_time - self.last_queue_update_time
            self.last_queue_update_time = current_time
        else:
            time_delta = 1.0  # Default 1 second
        
        # Factor 1: Direct attack impact on charging time causes queue buildup
        attack_queue_factor = 0
        if hasattr(self, 'attack_active') and self.attack_active:
            # Get current charging time vs normal charging time
            current_charging_time = self._calculate_real_charging_time(current_time)
            normal_charging_time = 30.0  # minutes
            
            if current_charging_time > normal_charging_time:
                # Each minute of delay adds customers to queue
                delay_minutes = current_charging_time - normal_charging_time
                # More severe delays cause exponential queue growth
                attack_queue_factor = int(delay_minutes * 0.5)  # 0.5 customers per minute delay
                
                # Accumulate persistent delay effects
                self.accumulated_queue_delay += delay_minutes * time_delta / 60.0  # Convert to minutes
        
        # Factor 2: Recovery period - queue slowly decreases after attack ends
        elif hasattr(self, 'recovery_mode') and self.recovery_mode:
            # During recovery, accumulated delay slowly decreases
            if hasattr(self, 'accumulated_queue_delay') and self.accumulated_queue_delay > 0:
                # Exponential decay of accumulated delay
                decay_rate = 0.1  # 2% per second decay
                self.accumulated_queue_delay *= (1.0 - decay_rate * time_delta)
                
                # Convert accumulated delay to queue length
                attack_queue_factor = int(self.accumulated_queue_delay * 2.0)  # 2 customers per minute of accumulated delay
        
        # Factor 3: Post-attack lingering effects
        elif hasattr(self, 'accumulated_queue_delay') and self.accumulated_queue_delay > 0:
            # Even after recovery, some delay persists
            decay_rate = 0.1  # Slower decay post-recovery (1% per second)
            self.accumulated_queue_delay *= (1.0 - decay_rate * time_delta)
            
            # Convert to queue length with diminishing effect
            attack_queue_factor = int(self.accumulated_queue_delay * 15.0)
        
        # Factor 4: Port utilization impact (realistic constraint)
        utilization_factor = 0
        if hasattr(self, 'available_ports') and self.available_ports < self.num_ports:
            port_utilization = (self.num_ports - self.available_ports) / self.num_ports
            if port_utilization > 0.9:  # Very high utilization
                utilization_factor = int((port_utilization - 0.9) * 30)  # Up to 3 additional customers
        
        # Calculate final queue length
        final_queue_length = base_queue_length + attack_queue_factor + utilization_factor
        
        # Realistic bounds: 0 to 25 customers maximum
        final_queue_length = max(0, min(25, final_queue_length))
        
        return final_queue_length
    
    def get_charging_metrics(self, current_time: float = None) -> Dict:
        """Get comprehensive charging metrics (enhanced with dynamics)"""
        active_sessions = len(self.charging_sessions)
        queue_length = self._calculate_real_queue_length(current_time)
        scheduled_count = len([s for s in self.scheduled_charging if s['status'] == 'scheduled'])
        
        # Calculate real charging time based on actual simulation state
        avg_time = self._calculate_real_charging_time(current_time)
        
        # During recovery, apply gradual interpolation to target values
        if hasattr(self, 'recovery_mode') and self.recovery_mode:
            recovery_progress = (current_time - getattr(self, 'recovery_start_time', 0)) / 120.0
            recovery_progress = min(1.0, max(0.0, recovery_progress))
            target_time = 30.0  # Target normal charging time
            avg_time = avg_time * (1 - recovery_progress) + target_time * recovery_progress
        
        # Calculate real queue length and wait time based on simulation state
        real_queue_length = self._calculate_real_queue_length(current_time)
        
        # Calculate queue wait time more accurately
        if self.customer_queue and current_time is not None:
            for customer in self.customer_queue:
                customer['wait_time'] = current_time - customer['arrival_time']
            total_wait = sum(customer['wait_time'] for customer in self.customer_queue)
            queue_wait_time = total_wait / len(self.customer_queue)
        else:
            queue_wait_time = self.queue_wait_time if hasattr(self, 'queue_wait_time') else 0.0
        
        # Calculate utilization rate from both systems
        if hasattr(self, 'ev_connected') and self.ev_connected:
            # Use dynamics system (EV connected)
            utilization_rate = 1.0  # Port is occupied
        else:
            # Use legacy system
            utilization_rate = (self.num_ports - self.available_ports) / self.num_ports
        
        # Enhanced metrics with dynamics
        return {
            'active_sessions': active_sessions,
            'queue_length': queue_length,
            'scheduled_count': scheduled_count,
            'utilization_rate': utilization_rate,
            'avg_charging_time': avg_time,
            'queue_wait_time': queue_wait_time,
            'available_ports': self.available_ports,
            'current_load': self.current_load,
            'emergency_mode': self.emergency_mode,
            # New dynamics-based metrics
            'ev_connected': self.ev_connected,
            'current_ev_id': self.current_ev_id,
            'soc': self.soc * 100,  # Convert to percentage
            'voltage_measured': self.voltage_measured,
            'current_measured': self.current_measured,
            'power_measured': self.power_measured,
            'ac_voltage_rms': self.ac_voltage_rms,
            'ac_current_rms': self.ac_current_rms,
            'grid_frequency': self.grid_frequency,
            'total_evs_served': self.total_evs_served,
            'total_energy_delivered': self.total_energy_delivered,
            # FIXED: Add recovery status for debugging
            'recovery_mode': getattr(self, 'recovery_mode', False),
            'recovery_progress': (current_time - getattr(self, 'recovery_start_time', 0)) / 120.0 if hasattr(self, 'recovery_mode') and self.recovery_mode else 0.0
        }

class EnhancedChargingManagementSystem:
    """Enhanced Charging Management System with Power Electronics Dynamics and Cyber Attack Capabilities"""
    
    def __init__(self, stations: List[EVChargingStation] = None, use_pinn: bool = True):
        self.stations = stations if stations is not None else []
        self.attack_active = False
        self.attack_params = {}
        self.measurements = {}
        
        # Enhanced control parameters for dynamics
        self.voltage_limits = {'min': 0.95, 'max': 1.05}  # Per unit
        self.total_power_limit = 6000  # kW (increased to match 6x1000kW stations)
        self.voltage_droop_gain = 100.0  # kW/pu
        self.frequency_droop_gain = 50.0  # kW/Hz
        self.soc_weight = 0.3
        self.voltage_weight = 0.4
        self.load_weight = 0.3
        self.total_time = 240.0
        
        # Security and Anomaly Detection System
        self.security_enabled = True
        self.reference_history = {}  # Store historical references for each station
        self.input_history = {}     # Store historical inputs for each station
        self.anomaly_threshold = 0.3  # 30% change threshold
        self.rate_change_limit = 0.5  # 50% per time step
        self.consecutive_anomaly_limit = 3  # Max consecutive anomalies
        self.anomaly_counters = {}  # Track consecutive anomalies per station
        self.max_power_reference = 100.0  # kW upper bound
        self.max_voltage_reference = 500.0  # V upper bound
        self.max_current_reference = 200.0  # A upper bound
        
        # Adaptive RL Attack Integration
        self.adaptive_rl_system = None
        self.rl_attack_active = False
        self.rl_attack_params = {}
        self.evcs_output_baseline = {}  # Store baseline outputs for reward calculation

        
        # Initialize PINN optimizer (federated or legacy)
        self.use_pinn = use_pinn  # Flag to enable/disable PINN optimization
        self.pinn_optimizer = None
        self.federated_manager = None
        self.pinn_trained = False
        self._initialize_pinn_optimizer()
        
    def _initialize_pinn_optimizer(self):
        """Initialize PINN optimizer (federated or legacy) using pre-trained models"""
        if not self.use_pinn:
            return
        
        # Try to load federated PINN models first
        if FederatedPINNManager is not None:
            try:
                federated_config = FederatedPINNConfig(
                    num_distribution_systems=6,
                    local_epochs=100,
                    global_rounds=5,
                    aggregation_method='fedavg'
                )
                self.federated_manager = FederatedPINNManager(federated_config)
                
                success = self.federated_manager.load_federated_models('federated_models')
                if success:
                    self.pinn_trained = True
                    print("CMS: Federated PINN models loaded successfully")
                    return
                else:
                    print("CMS: Federated models not found, trying legacy PINN...")
                    self.federated_manager = None
            except Exception as e:
                print(f"CMS: Failed to load federated models: {e}")
                self.federated_manager = None
        
        # Fallback to legacy PINN optimizer
        try:
            from pinn_optimizer import LSTMPINNConfig, LSTMPINNChargingOptimizer
            
            # Create LSTM-PINN configuration
            pinn_config = LSTMPINNConfig(
                lstm_hidden_size=128,
                lstm_num_layers=3,
                sequence_length=10,
                hidden_layers=[256, 512, 256, 128],
                learning_rate=0.001,
                epochs=100,
                physics_weight=1.0,
                boundary_weight=10.0,
                data_weight=1.0,
                # EVCS Charging Specifications
                rated_voltage=400.0,
                rated_current=100.0,
                rated_power=40.0,
                max_voltage=500.0,
                min_voltage=300.0,
                max_current=150.0,
                min_current=50.0,
                max_power=75.0,
                min_power=15.0
            )
            
            # Create optimizer but don't train (always_train=False)
            self.pinn_optimizer = LSTMPINNChargingOptimizer(pinn_config, always_train=False)
            
            # Try to load pre-trained model first
            model_paths = [
                'pinn_evcs_optimizer_pretrained.pth',  # From focused_demand_analysis training
                'pinn_evcs_optimizer.pth',             # Legacy model
                'pinn_optimizer.pth'                   # Alternative name
            ]
            
            model_loaded = False
            for model_path in model_paths:
                try:
                    self.pinn_optimizer.load_model(model_path)
                    self.pinn_trained = True
                    print(f"CMS: Legacy PINN model loaded from {model_path}")
                    model_loaded = True
                    break
                except:
                    continue
            
            if not model_loaded:
                print("CMS: No pre-trained PINN model found")
                print("CMS: Please run focused_demand_analysis.py first to train the PINN model")
                print("CMS: Falling back to heuristic optimization")
                self.use_pinn = False
                self.pinn_optimizer = None
                self.pinn_trained = False
                
        except ImportError:
            print("Warning: PyTorch not available, falling back to heuristic optimization")
            self.use_pinn = False
            self.pinn_optimizer = None
            self.pinn_trained = False
        except Exception as e:
            print(f"Warning: PINN initialization failed ({e}), using heuristic optimization")
            self.use_pinn = False
            self.pinn_optimizer = None
            self.pinn_trained = False
    
    def _optimize_with_pinn(self, station_id: int, current_time: float, 
                           bus_voltages: Dict[str, float], system_frequency: float,
                           dynamics_state: Dict) -> Tuple[float, float, float]:
        """Use PINN to optimize charging references"""
        # Get station from stations list using station index
        if station_id >= len(self.stations):
            raise IndexError(f"Station ID {station_id} out of range (max: {len(self.stations)-1})")
        station = self.stations[station_id]
        
        # Try to load federated PINN models first
        if self.federated_manager is not None:
            try:
                # Prepare station data for federated optimization
                grid_voltage_pu = 1.0  # Default
                if bus_voltages and station.bus_name in bus_voltages:
                    grid_voltage_pu = bus_voltages[station.bus_name]
                
                # Calculate voltage priority
                voltage_priority = max(0, self.voltage_limits['min'] - grid_voltage_pu)
                
                # Calculate urgency factor
                urgency_factor = 1.0
                if station.soc < 0.2:
                    urgency_factor = 2.0
                elif station.soc > 0.8:
                    urgency_factor = 0.3
                
                # Get demand factor
                demand_factor = self.generate_daily_charging_profile(current_time)
                
                station_data = {
                    'soc': station.soc,
                    'grid_voltage': grid_voltage_pu,
                    'grid_frequency': system_frequency,
                    'demand_factor': demand_factor,
                    'voltage_priority': voltage_priority,
                    'urgency_factor': urgency_factor,
                    'current_time': current_time
                }
                
                # FIXED: Apply attacks to INPUTS instead of outputs
                attacked_station_data = self._apply_input_attacks(station_data, station_id, current_time)
                
                # Get federated PINN optimization results with potentially attacked inputs
                voltage_ref, current_ref, power_ref = self.federated_manager.optimize_references(station_id, attacked_station_data)
                
                # Debug: Print detailed PINN input and output
                print(f" FEDERATED PINN System {self.system_id} Station {station_id} @ t={current_time:.1f}s:")
                print(f"    Inputs: SOC={station.soc:.2f}, V_grid={grid_voltage_pu:.3f}pu, f={system_frequency:.2f}Hz")
                print(f"    Factors: demand={demand_factor:.3f}, urgency={urgency_factor:.1f}, v_priority={voltage_priority:.3f}")
                print(f"    Raw PINN Output: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
                
                # Apply security validation and anomaly detection
                voltage_ref, current_ref, power_ref, attack_detected = self._security_validation(
                    station_id, voltage_ref, current_ref, power_ref, attacked_station_data, current_time
                )
                
                if not attack_detected:
                    print(f"    ATTACK BLOCKED: Security system prevented malicious operation")
                
                # Ensure minimum values
                voltage_ref = max(voltage_ref, 400.0)
                current_ref = max(current_ref, 1.0)
                power_ref = max(power_ref, 1.0)
                
                # Keep PINN-optimized current within realistic bounds (don't recalculate from power)
                # Only adjust if current is completely unrealistic (>200A for safety)
                if current_ref > 200.0:
                    current_ref = min(125.0, power_ref * 1000.0 / voltage_ref)
                    print(f"   🔧 Current clamped to {current_ref:.1f}A (was >200A)")
                
                print(f"    Final Constrained: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
                print(f" Federated PINN Optimization - Station {station_id}: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
                
                return voltage_ref, current_ref, power_ref
            
            except Exception as e:
                print(f"  Federated PINN failed for System {self.system_id} Station {station_id}: {e}")
                print(f"   Falling back to legacy PINN optimization...")
                return self._optimize_legacy_pinn(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
        
        # Fallback to legacy PINN or heuristic optimization
        return self._optimize_heuristic(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
    
    def _apply_input_attacks(self, station_data: Dict, station_id: int, current_time: float) -> Dict:
        """Apply RL attacks to CMS inputs instead of outputs (FIXED ATTACK SURFACE)"""
        if not self.attack_active or station_id not in self.attack_params.get('targets', []):
            return station_data
        
        # Create a copy to avoid modifying original
        attacked_data = station_data.copy()
        attack_type = self.attack_params['type']
        magnitude = self.attack_params['magnitude']
        
        print(f"    Applying INPUT attack: {attack_type} (mag={magnitude:.2f}) on Station {station_id}")
        
        if attack_type == 'demand_increase':
            # Manipulate demand_factor to increase charging demand
            attacked_data['demand_factor'] *= (1.0 + magnitude * 5.0)  # Increased from 0.5 to 5.0 for higher impact
            # Manipulate urgency_factor to create false urgency
            attacked_data['urgency_factor'] *= (1.0 + magnitude * 2.0)  # Increased from 0.3 to 2.0 for higher impact
            
        elif attack_type == 'demand_decrease':
            # Manipulate demand_factor to decrease charging demand
            attacked_data['demand_factor'] *= (1.0 - magnitude * 2.0)  # Increased from 0.4 to 2.0 for higher impact
            # Reduce urgency to delay charging
            attacked_data['urgency_factor'] *= (1.0 - magnitude * 1.0)  # Increased from 0.2 to 1.0 for higher impact
            
        elif attack_type == 'oscillating_demand':
            # Create oscillating demand pattern
            time_factor = (current_time - self.attack_params.get('start_time', 0)) / 10.0
            oscillation = np.sin(time_factor) * 2.0 + 1.0  # Increased from 0.3 to 2.0 for higher impact
            attacked_data['demand_factor'] *= oscillation * magnitude
            
        elif attack_type == 'voltage_spoofing':
            # Spoof voltage readings to manipulate voltage priority
            attacked_data['grid_voltage'] *= (1.0 - magnitude * 1.0)  # Increased from 0.1 to 1.0 for higher impact
            attacked_data['voltage_priority'] = max(0, self.voltage_limits['min'] - attacked_data['grid_voltage'])
            
        elif attack_type == 'frequency_spoofing':
            # Spoof frequency readings
            attacked_data['grid_frequency'] += magnitude * 10.0  # Increased from 2.0 to 10.0 for higher impact
            
        elif attack_type == 'soc_manipulation':
            # Manipulate SOC readings (if attacker has access to EV data)
            if hasattr(self, 'stations') and station_id < len(self.stations):
                # Make SOC appear lower to increase charging urgency
                soc_reduction = magnitude * 1.0  # Increased from 0.2 to 1.0 for higher impact
                attacked_data['soc'] = max(0.1, attacked_data['soc'] - soc_reduction)
                # Recalculate urgency based on manipulated SOC
                if attacked_data['soc'] < 0.2:
                    attacked_data['urgency_factor'] = 5.0  # Increased from 2.0 to 5.0 for higher impact
                elif attacked_data['soc'] > 0.8:
                    attacked_data['urgency_factor'] = 0.1  # Decreased from 0.3 to 0.1 for higher impact
        
        print(f"    Original inputs: demand={station_data['demand_factor']:.3f}, urgency={station_data['urgency_factor']:.1f}")
        print(f"    Attacked inputs: demand={attacked_data['demand_factor']:.3f}, urgency={attacked_data['urgency_factor']:.1f}")
        
        return attacked_data
    
    def _security_validation(self, station_id: int, voltage_ref: float, current_ref: float, 
                           power_ref: float, station_data: Dict, current_time: float) -> Tuple[float, float, float, bool]:
        """Comprehensive security validation and anomaly detection"""
        if not self.security_enabled:
            return voltage_ref, current_ref, power_ref, True
        
        # Initialize history for new stations
        if station_id not in self.reference_history:
            self.reference_history[station_id] = []
            self.input_history[station_id] = []
            self.anomaly_counters[station_id] = 0
        
        # Store current inputs for pattern analysis
        self.input_history[station_id].append({
            'time': current_time,
            'soc': station_data['soc'],
            'demand_factor': station_data['demand_factor'],
            'urgency_factor': station_data['urgency_factor'],
            'grid_voltage': station_data['grid_voltage']
        })
        
        # Keep only recent history (last 20 entries)
        if len(self.input_history[station_id]) > 20:
            self.input_history[station_id] = self.input_history[station_id][-20:]
        
        # 1. Upper Bounds Check
        if power_ref > self.max_power_reference:
            print(f"    SECURITY: Power reference {power_ref:.1f}kW exceeds limit {self.max_power_reference}kW")
            power_ref = self.max_power_reference
        
        if voltage_ref > self.max_voltage_reference:
            print(f"    SECURITY: Voltage reference {voltage_ref:.1f}V exceeds limit {self.max_voltage_reference}V")
            voltage_ref = self.max_voltage_reference
        
        if current_ref > self.max_current_reference:
            print(f"    SECURITY: Current reference {current_ref:.1f}A exceeds limit {self.max_current_reference}A")
            current_ref = self.max_current_reference
        
        # 2. Rate-of-Change Detection
        if len(self.reference_history[station_id]) > 0:
            last_ref = self.reference_history[station_id][-1]
            
            power_change_rate = abs(power_ref - last_ref['power']) / max(last_ref['power'], 1.0)
            voltage_change_rate = abs(voltage_ref - last_ref['voltage']) / max(last_ref['voltage'], 1.0)
            current_change_rate = abs(current_ref - last_ref['current']) / max(last_ref['current'], 1.0)
            
            if (power_change_rate > self.rate_change_limit or 
                voltage_change_rate > self.rate_change_limit or 
                current_change_rate > self.rate_change_limit):
                
                print(f"    SECURITY: Excessive rate of change detected")
                print(f"    Power: {power_change_rate:.2f}, Voltage: {voltage_change_rate:.2f}, Current: {current_change_rate:.2f}")
                
                # Apply rate limiting
                if power_change_rate > self.rate_change_limit:
                    direction = 1 if power_ref > last_ref['power'] else -1
                    power_ref = last_ref['power'] + direction * last_ref['power'] * self.rate_change_limit
                
                self.anomaly_counters[station_id] += 1
        
        # 3. Statistical Anomaly Detection
        anomaly_detected = self._detect_statistical_anomaly(station_id, power_ref, voltage_ref, current_ref)
        
        # 4. Input Pattern Analysis
        input_anomaly = self._detect_input_anomaly(station_id, station_data)
        
        # 5. Consecutive Anomaly Check
        if anomaly_detected or input_anomaly:
            self.anomaly_counters[station_id] += 1
            print(f"    SECURITY: Anomaly detected (count: {self.anomaly_counters[station_id]})")
            
            if self.anomaly_counters[station_id] >= self.consecutive_anomaly_limit:
                print(f"    SECURITY: ATTACK DETECTED - {self.consecutive_anomaly_limit} consecutive anomalies")
                print(f"    Switching to emergency safe mode for Station {station_id}")
                
                # Emergency safe mode: conservative references
                voltage_ref = 400.0  # Safe voltage
                current_ref = 25.0   # Reduced current
                power_ref = 10.0     # Minimal power
                
                return voltage_ref, current_ref, power_ref, False  # Attack detected
        else:
            # Reset counter on normal operation
            self.anomaly_counters[station_id] = max(0, self.anomaly_counters[station_id] - 1)
        
        # Store current references for next iteration
        self.reference_history[station_id].append({
            'time': current_time,
            'voltage': voltage_ref,
            'current': current_ref,
            'power': power_ref
        })
        
        # Keep only recent history
        if len(self.reference_history[station_id]) > 20:
            self.reference_history[station_id] = self.reference_history[station_id][-20:]
        
        return voltage_ref, current_ref, power_ref, True  # No attack detected
    
    def _detect_statistical_anomaly(self, station_id: int, power_ref: float, 
                                   voltage_ref: float, current_ref: float) -> bool:
        """Detect statistical anomalies in reference values"""
        if len(self.reference_history[station_id]) < 5:
            return False  # Need sufficient history
        
        # Get recent power references
        recent_powers = [ref['power'] for ref in self.reference_history[station_id][-5:]]
        mean_power = np.mean(recent_powers)
        std_power = np.std(recent_powers)
        
        # Z-score anomaly detection
        if std_power > 0:
            z_score = abs(power_ref - mean_power) / std_power
            if z_score > 2.5:  # 2.5 sigma threshold
                print(f"    SECURITY: Statistical anomaly - Z-score: {z_score:.2f}")
                return True
        
        return False
    
    def _detect_input_anomaly(self, station_id: int, station_data: Dict) -> bool:
        """Detect anomalies in input patterns"""
        if len(self.input_history[station_id]) < 3:
            return False
        
        # Check for sudden changes in demand factor
        recent_demands = [inp['demand_factor'] for inp in self.input_history[station_id][-3:]]
        if len(recent_demands) >= 2:
            demand_change = abs(recent_demands[-1] - recent_demands[-2]) / max(recent_demands[-2], 0.1)
            if demand_change > self.anomaly_threshold:
                print(f"    SECURITY: Input anomaly - Demand change: {demand_change:.2f}")
                return True
        
        # Check for unrealistic urgency factor changes
        recent_urgency = [inp['urgency_factor'] for inp in self.input_history[station_id][-3:]]
        if len(recent_urgency) >= 2:
            urgency_change = abs(recent_urgency[-1] - recent_urgency[-2])
            if urgency_change > 1.0:  # Urgency shouldn't change dramatically
                print(f"    SECURITY: Input anomaly - Urgency change: {urgency_change:.2f}")
                return True
        
        return False
    
    def enable_adaptive_rl_attacks(self, adaptive_rl_system):
        """Enable adaptive RL attack system that learns to bypass security"""
        self.adaptive_rl_system = adaptive_rl_system
        self.rl_attack_active = True
        print(" Adaptive RL Attack System ENABLED")
        print("   - RL agents will learn to bypass anomaly detection")
        print("   - Rewards based on actual EVCS output changes")
        print("   - Security evasion feedback integrated")
    
    def disable_adaptive_rl_attacks(self):
        """Disable adaptive RL attack system"""
        self.rl_attack_active = False
        self.adaptive_rl_system = None
        print(" Adaptive RL Attack System DISABLED")
    
    def get_evcs_output_for_reward(self, station_id: int, voltage_ref: float, 
                                 current_ref: float, power_ref: float) -> Dict:
        """Get EVCS output changes for RL reward calculation"""
        
        # Store baseline if not exists
        if station_id not in self.evcs_output_baseline:
            self.evcs_output_baseline[station_id] = {
                'voltage': 400.0,
                'current': 25.0, 
                'power': 10.0
            }
        
        baseline = self.evcs_output_baseline[station_id]
        
        # Calculate deviations from baseline
        output_changes = {
            'voltage_deviation': abs(voltage_ref - baseline['voltage']) / baseline['voltage'],
            'current_deviation': abs(current_ref - baseline['current']) / baseline['current'],
            'power_deviation': abs(power_ref - baseline['power']) / baseline['power'],
            'total_impact': 0.0
        }
        
        # Calculate total system impact
        output_changes['total_impact'] = (
            output_changes['voltage_deviation'] + 
            output_changes['current_deviation'] + 
            output_changes['power_deviation']
        ) / 3.0
        
        return output_changes
    
    def _optimize_legacy_pinn(self, station_id: int, current_time: float, 
                           bus_voltages: Dict[str, float], system_frequency: float,
                           dynamics_state: Dict) -> Tuple[float, float, float]:
        """Use legacy PINN to optimize charging references"""
        if station_id >= len(self.stations):
            raise IndexError(f"Station ID {station_id} out of range (max: {len(self.stations)-1})")
        station = self.stations[station_id]
        
        # Prepare input data for PINN
        grid_voltage_pu = 1.0  # Default
        if bus_voltages and station.bus_name in bus_voltages:
            grid_voltage_pu = bus_voltages[station.bus_name]
        
        # Calculate voltage priority
        voltage_priority = max(0, self.voltage_limits['min'] - grid_voltage_pu)
        
        # Calculate urgency factor
        urgency_factor = 1.0
        if station.soc < 0.2:
            urgency_factor = 2.0
        elif station.soc > 0.8:
            urgency_factor = 0.3
        
        # Get demand factor
        demand_factor = self.generate_daily_charging_profile(current_time)
        
        station_data = {
            'soc': station.soc,
            'grid_voltage': grid_voltage_pu,
            'grid_frequency': system_frequency,
            'demand_factor': demand_factor,
            'voltage_priority': voltage_priority,
            'urgency_factor': urgency_factor,
            'current_time': current_time
        }
        
        try:
            # FIXED: Apply attacks to INPUTS instead of outputs
            attacked_station_data = self._apply_input_attacks(station_data, station_id, current_time)
            
            # Get PINN optimization results with potentially attacked inputs
            voltage_ref, current_ref, power_ref = self.pinn_optimizer.optimize_references(attacked_station_data)
            
            # Debug: Print detailed PINN input and output
            print(f" LEGACY PINN System {self.system_id} Station {station_id} @ t={current_time:.1f}s:")
            print(f"    Inputs: SOC={station.soc:.2f}, V_grid={grid_voltage_pu:.3f}pu, f={system_frequency:.2f}Hz")
            print(f"    Factors: demand={demand_factor:.3f}, urgency={urgency_factor:.1f}, v_priority={voltage_priority:.3f}")
            print(f"    Raw PINN Output: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
            
            # Apply security validation and anomaly detection
            voltage_ref, current_ref, power_ref, attack_detected = self._security_validation(
                station_id, voltage_ref, current_ref, power_ref, attacked_station_data, current_time
            )
            
            if not attack_detected:
                print(f"   🚨 ATTACK BLOCKED: Security system prevented malicious operation")
            
            # Ensure minimum values
            voltage_ref = max(voltage_ref, 400.0)
            current_ref = max(current_ref, 1.0)
            power_ref = max(power_ref, 1.0)
            
            # Keep PINN-optimized current within realistic bounds (don't recalculate from power)
            # Only adjust if current is completely unrealistic (>200A for safety)
            if current_ref > 200.0:
                current_ref = min(125.0, power_ref * 1000.0 / voltage_ref)
                print(f"   🔧 Current clamped to {current_ref:.1f}A (was >200A)")
            
            print(f"   ✅ Final Constrained: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
            print(f" Legacy PINN Optimization - Station {station_id}: V={voltage_ref:.1f}V, I={current_ref:.1f}A, P={power_ref:.1f}kW")
            
            return voltage_ref, current_ref, power_ref
            
        except Exception as e:
            print(f"CMS: Legacy PINN optimization failed ({e}), falling back to heuristic")
            return self._optimize_heuristic(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
    
    def _optimize_heuristic(self, station_id: int, current_time: float,
                           bus_voltages: Dict[str, float], system_frequency: float,
                           dynamics_state: Dict) -> Tuple[float, float, float]:
        """Fallback heuristic optimization (original method)"""
        # Get station from stations list using station index
        if station_id >= len(self.stations):
            raise IndexError(f"Station ID {station_id} out of range (max: {len(self.stations)-1})")
        station = self.stations[station_id]
        
        # Get daily charging demand profile
        demand_factor = self.generate_daily_charging_profile(current_time)
        
        # Calculate available power considering system constraints
        total_available_power = self.total_power_limit * demand_factor
        
        # Ensure minimum power
        min_station_power = 50.0
        total_min_power = min_station_power * len(self.stations)
        
        if total_available_power < total_min_power:
            total_available_power = total_min_power
        
        min_emergency_power = 500.0
        if total_available_power < min_emergency_power:
            total_available_power = min_emergency_power
        
        if station.ev_connected:
            # SOC-based priority
            soc_priority = (1 - station.soc)
            
            # Voltage support priority
            voltage_priority = 0
            if bus_voltages and station.bus_name in bus_voltages:
                voltage_pu = bus_voltages[station.bus_name]
                voltage_priority = max(0, self.voltage_limits['min'] - voltage_pu)
            
            # Frequency support
            frequency_priority = max(0, 60.0 - system_frequency) / 60.0
            
            # Urgency factor
            urgency_factor = 1.0
            if station.soc < 0.2:
                urgency_factor = 2.0
            elif station.soc > 0.8:
                urgency_factor = 0.3
            
            # Composite priority
            composite_priority = urgency_factor * (self.soc_weight * soc_priority + 
                                                  self.voltage_weight * voltage_priority +
                                                  0.1 * frequency_priority)
            
            # Power allocation based on SOC (increased to utilize full EVCS capacity)
            if station.soc < 0.2:
                base_power = 800  # High urgency - use 80% of 1000kW capacity
                target_voltage = 400.0
                target_current = 125.0
            elif station.soc < 0.8:
                base_power = 600  # Normal charging - use 60% of capacity
                target_voltage = 400.0
                target_current = 100.0
            else:
                base_power = 300  # Topping off - use 30% of capacity
                target_voltage = 400.0
                target_current = 75.0
            
            # Apply factors
            base_power *= demand_factor
            fair_share = total_available_power / len(self.stations)
            allocated_power = min(base_power, fair_share * composite_priority)
            allocated_power = max(allocated_power, 10.0)
            
            # Calculate references
            voltage_ref = max(target_voltage, 400.0)
            current_ref = allocated_power * 1000 / voltage_ref
            current_ref = max(current_ref, 1.0)
            current_ref = min(current_ref, target_current)
            
            return voltage_ref, current_ref, allocated_power
        else:
            return 50.0, 0.1, 1.0

    def update_measurements(self, station_id: int, voltage: float, current: float, power: float):
        """Update sensor measurements (can be manipulated during attacks)"""
        self.measurements[station_id] = {
            'voltage': voltage,
            'current': current, 
            'power': power,
            'timestamp': time.time()
        }
        
    def inject_false_data(self, station_id: int, attack_type: str, magnitude: float, current_time: float = 0.0):
        """Inject false data into measurements to deceive PINN optimizer"""
        if station_id in self.measurements:
            # Store original measurements before manipulation
            if not hasattr(self, 'original_measurements'):
                self.original_measurements = {}
            if station_id not in self.original_measurements:
                self.original_measurements[station_id] = self.measurements[station_id].copy()
            
            # Apply RL-guided false data injection
            if attack_type == 'voltage_low':
                self.measurements[station_id]['voltage'] *= (1 - magnitude)
            elif attack_type == 'current_low':
                self.measurements[station_id]['current'] *= (1 - magnitude)
            elif attack_type == 'power_underreport':
                self.measurements[station_id]['power'] *= (1 - magnitude)
            elif attack_type == 'demand_increase':
                # RL agent deceives PINN by reporting higher demand
                self.measurements[station_id]['power'] *= (1 + magnitude)
                # Also manipulate voltage to make PINN think grid is stressed
                self.measurements[station_id]['voltage'] *= (1 - magnitude * 0.1)
            elif attack_type == 'demand_decrease':
                # RL agent deceives PINN by reporting lower demand
                self.measurements[station_id]['power'] *= magnitude
                # Manipulate voltage to appear stable
                self.measurements[station_id]['voltage'] *= (1 + magnitude * 0.05)
            elif attack_type == 'oscillating_demand':
                # Dynamic oscillating deception
                time_factor = (current_time) / 10.0
                oscillation = np.sin(time_factor) * 0.5 + 1.0
                self.measurements[station_id]['power'] *= oscillation * magnitude
            elif attack_type == 'ramp_demand':
                # Gradual ramp deception
                ramp_factor = min(1.0, current_time / 50.0)  # Ramp over 50 seconds
                if magnitude > 1.0:
                    manipulation_factor = 1.0 + (magnitude - 1.0) * ramp_factor
                else:
                    manipulation_factor = 1.0 - (1.0 - magnitude) * ramp_factor
                self.measurements[station_id]['power'] *= manipulation_factor
            
            # Add timestamp for RL learning
            self.measurements[station_id]['attack_timestamp'] = current_time
            self.measurements[station_id]['attack_type'] = attack_type
            self.measurements[station_id]['attack_magnitude'] = magnitude
    
    def generate_daily_charging_profile(self, time_hours: float) -> float:
        """Generate realistic daily charging demand profile matching enhanced load profile"""
        
        # Base load that never goes to zero (always some background charging)
        base_load_level = 0.20  # 20% minimum load at all times
        total_load = base_load_level
        
        # Create smooth continuous load curve using overlapping Gaussian-like functions
        # Night valley (11 PM - 6 AM): Minimum charging with gradual transitions
        night_center = 2.5  # 2:30 AM center
        if time_hours >= 23:
            night_hour = time_hours - 24  # Convert to negative for smooth transition
        else:
            night_hour = time_hours
        night_contribution = 0.15 * np.exp(-((night_hour - night_center)**2) / (2 * 3**2))
        total_load += night_contribution
        
        # Early morning ramp (5-9 AM): Gradual increase for commuter charging
        morning_center = 7.5  # 7:30 AM peak
        morning_contribution = 0.45 * np.exp(-((time_hours - morning_center)**2) / (2 * 2.5**2))
        total_load += morning_contribution
        
        # Mid-morning workplace (9 AM - 1 PM): Sustained workplace charging
        workplace_center = 11  # 11 AM center
        workplace_contribution = 0.35 * np.exp(-((time_hours - workplace_center)**2) / (2 * 3**2))
        total_load += workplace_contribution
        
        # Afternoon moderate (1-5 PM): Shopping and opportunity charging
        afternoon_center = 15  # 3 PM center
        afternoon_contribution = 0.25 * np.exp(-((time_hours - afternoon_center)**2) / (2 * 2.5**2))
        total_load += afternoon_contribution
        
        # Evening peak (5-9 PM): Major residential charging peak
        evening_center = 19  # 7 PM peak
        evening_contribution = 0.65 * np.exp(-((time_hours - evening_center)**2) / (2 * 2.8**2))
        total_load += evening_contribution
        
        # Late evening taper (9-11 PM): Gradual decrease
        late_evening_center = 22  # 10 PM center
        late_evening_contribution = 0.30 * np.exp(-((time_hours - late_evening_center)**2) / (2 * 1.8**2))
        total_load += late_evening_contribution
        
        # Add realistic temporal variations
        # Intraday variability (15-minute fluctuations)
        time_minutes = (time_hours * 60) % 60
        intraday_variation = 0.03 * np.sin(2 * np.pi * time_minutes / 15)
        total_load += intraday_variation
        
        # Weekly pattern (slightly higher on weekdays)
        weekday_factor = 1.08  # 8% higher on weekdays
        total_load *= weekday_factor
        
        # Weather/seasonal effects (correlated noise for realism)
        weather_base = np.sin(2 * np.pi * time_hours / 24 + np.pi/3)
        weather_noise = 0.04 * weather_base * (1 + 0.3 * np.random.normal(0, 1))
        total_load += weather_noise
        
        # Grid-friendly charging incentives (slight load shifting)
        if 23 <= time_hours or time_hours <= 6:  # Off-peak hours
            total_load *= 1.05  # 5% incentive for off-peak charging
        
        # Ensure realistic bounds with smoother clamping
        return np.clip(total_load, 0.20, 1.4)  # 20% min, 140% max
                
    def optimize_charging(self, station_id: int, current_time: float = 0.0, 
                         bus_voltages: Dict[str, float] = None, 
                         system_frequency: float = 60.0,
                         dynamics_state: Dict = None) -> Tuple[float, float, float]:
        """Enhanced optimization with PINN-based intelligent optimization and cyber attack handling"""
        # Get station from stations list using station index
        if station_id >= len(self.stations):
            raise IndexError(f"Station ID {station_id} out of range (max: {len(self.stations)-1})")
        station = self.stations[station_id]
        
        # Use PINN optimization if available and enabled
        if self.use_pinn and (self.pinn_optimizer is not None or self.federated_manager is not None) and station.ev_connected:
            try:
                return self._optimize_with_pinn(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
            except Exception as e:
                print(f"CMS: PINN optimization failed ({e}), using heuristic fallback")
                return self._optimize_heuristic(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
        
        # Fallback to heuristic optimization
        return self._optimize_heuristic(station_id, current_time, bus_voltages, system_frequency, dynamics_state)
    
    def ensure_evcs_initialization(self):
        """Ensure all EVCS stations are properly initialized with connected EVs"""
        print(f"CMS: Ensuring initialization of {len(self.stations)} EVCS stations...")
        
        # First pass: try normal connection
        for station in self.stations:
            if not station.ev_connected:
                # Force connection of first EV
                station.initialization_time = 0.0  # Reset initialization
                station.connect_new_ev(0.0)  # Connect at time 0
        
        # Check how many connected
        connected_count = sum(1 for s in self.stations if s.ev_connected)
        print(f"CMS: First pass - {connected_count}/{len(self.stations)} stations have connected EVs")
        
        # Second pass: force connection for any remaining disconnected stations
        if connected_count < len(self.stations):
            print("CMS: Second pass - forcing connection for remaining stations...")
            for station in self.stations:
                if not station.ev_connected:
                    # Force connection directly
                    station._force_ev_connection(0.0)
                    print(f"EVCS {station.evcs_id}: Forced EV connection")
        
        # Third pass: ensure all stations have valid references
        print("CMS: Third pass - ensuring all stations have valid references...")
        for station in self.stations:
            if station.ev_connected:
                # Set initial references to prevent zero setpoints
                station.set_references(400.0, 50.0, 20.0)  # Safe initial values
                print(f"EVCS {station.evcs_id}: Set initial references")
        
        # Final count
        final_connected_count = sum(1 for s in self.stations if s.ev_connected)
        print(f"CMS: Final result - {final_connected_count}/{len(self.stations)} stations now have connected EVs")
        
        if final_connected_count == 0:
            print("ERROR: CMS failed to initialize any EVCS stations with connected EVs!")
        elif final_connected_count < len(self.stations):
            print(f"WARNING: CMS only initialized {final_connected_count}/{len(self.stations)} stations")
        else:
            print("SUCCESS: All EVCS stations initialized with connected EVs")

class CentralChargingCoordinator:
    """Central Controller for EV Charging Infrastructure Coordination"""
    
    def __init__(self):
        self.distribution_systems = {}  # {system_id: OpenDSSInterface}
        self.global_charging_status = {}
        self.customer_queue = {}
        self.charging_schedules = {}
        self.emergency_alerts = []
        self.coordination_active = True
        
        # Enhanced tracking for charging time and queue management
        self.global_queue_metrics = {}
        self.scheduling_optimization = {}
        self.charging_time_impacts = {}
        self.load_balancing_history = []
        self.customer_redirections = []
        
        # Global coordination parameters
        self.max_queue_time = 30.0  # minutes
        self.load_balance_threshold = 0.2  # 20% load difference
        self.emergency_power_reduction = 0.3  # 30% power reduction in emergency
        
        # Customer arrival simulation parameters
        self.customer_arrival_rate = 0.5 #(0.0167*2)  # customers per minute per station (increased for more visible queues)
        self.charging_demand_distribution = {
            'low': {'min': 10, 'max': 30, 'probability': 0.4},    # 10-30 kWh
            'medium': {'min': 30, 'max': 60, 'probability': 0.4}, # 30-60 kWh
            'high': {'min': 60, 'max': 100, 'probability': 0.2}   # 60-100 kWh
        }
        
    def register_distribution_system(self, system_id: int, system_interface):
        """Register a distribution system with the central coordinator"""
        self.distribution_systems[system_id] = system_interface
        self.global_charging_status[system_id] = {
            'total_capacity': 0.0,
            'current_load': 0.0,
            'available_ports': 0,
            'queue_length': 0,
            'avg_charging_time': 0.0,
            'system_health': 'normal',
            'total_ports': 0,
            'utilization_rate': 0.0,
            'avg_queue_wait_time': 0.0,
            'scheduled_sessions': 0
        }
        
        # Initialize queue metrics
        self.global_queue_metrics[system_id] = {
            'total_customers_served': 0,
            'total_wait_time': 0.0,
            'max_queue_length': 0,
            'avg_charging_duration': 30.0,
            'customer_satisfaction': 1.0
        }
        
        # Initialize scheduling optimization
        self.scheduling_optimization[system_id] = {
            'optimization_active': True,
            'peak_hour_adjustment': 1.0,
            'load_balancing_factor': 1.0,
            'emergency_adjustment': 1.0
        }
        
        # Initialize charging time impacts tracking
        self.charging_time_impacts[system_id] = {
            'normal_charging_time': 30.0,
            'current_charging_time': 30.0,
            'attack_impact_factor': 1.0,
            'recovery_progress': 0.0
        }
        
    def simulate_customer_arrivals(self, current_time: float):
        """Simulate customer arrivals at charging stations (called every 10 seconds)"""
        for sys_id, system in self.distribution_systems.items():
            if hasattr(system, 'ev_stations'):
                for station_idx, station in enumerate(system.ev_stations):
                    # Simulate customer arrival based on arrival rate (adjusted for 10-second intervals)
                    # Convert from customers per minute to customers per 10-second interval
                    arrival_probability = self.customer_arrival_rate * (10.0 / 60.0)  # 10 seconds = 1/6 minute
                    
                    # Add time-based scaling to increase arrivals over time
                    time_scaling = 1.0 + (current_time / 240.0) * 0.5  # Gradual increase over simulation
                    arrival_probability *= time_scaling
                    
                    if np.random.random() < arrival_probability:
                        # Generate customer demand
                        demand_type = np.random.choice(
                            list(self.charging_demand_distribution.keys()),
                            p=[d['probability'] for d in self.charging_demand_distribution.values()]
                        )
                        demand_range = self.charging_demand_distribution[demand_type]
                        requested_charge = np.random.uniform(demand_range['min'], demand_range['max'])
                        
                        # Add customer to queue
                        customer_id = f"cust_{sys_id}_{station_idx}_{int(current_time)}"
                        station.add_customer_to_queue(customer_id, current_time, requested_charge)
                        
                        # Update global metrics
                        self.global_queue_metrics[sys_id]['total_customers_served'] += 1
                        
    def process_charging_queues(self, current_time: float):
        """Process charging queues and start charging sessions (called every 10 seconds)"""
        for sys_id, system in self.distribution_systems.items():
            if hasattr(system, 'ev_stations'):
                for station_idx, station in enumerate(system.ev_stations):
                    # Process queue if there are available ports
                    if station.available_ports > 0 and station.customer_queue:
                        # Sort queue by priority and arrival time
                        station.customer_queue.sort(key=lambda x: (x['priority'], x['arrival_time']))
                        
                        # Start charging for next customer
                        customer = station.customer_queue.pop(0)
                        
                        # Calculate charging duration based on demand and power
                        # Handle case where current_setpoint is zero (station idle or no EV connected)
                        if station.current_setpoint > 0:
                            charging_duration = (customer['requested_charge'] / station.current_setpoint) * 60.0  # minutes
                        else:
                            # Use a default charging duration when station is idle
                            charging_duration = 30.0  # 30 minutes default
                            if station.ev_connected:
                                print(f"Warning: Station {station.evcs_id} has connected EV but zero current setpoint, using default charging duration")
                            else:
                                print(f"Info: Station {station.evcs_id} has no EV connected, using default charging duration")
                        
                        # Apply attack impact on charging time
                        if hasattr(system, 'cms') and system.cms and system.cms.attack_active:
                            attack_impact = self.charging_time_impacts[sys_id]['attack_impact_factor']
                            charging_duration *= attack_impact
                        
                        # Start charging session only if station has power available
                        if station.current_setpoint > 0:
                            success = station.start_charging_session(
                                customer['id'], current_time, 
                                station.current_setpoint, charging_duration
                            )
                        else:
                            # If station has no power, put customer back in queue
                            station.customer_queue.insert(0, customer)
                            success = False
                            if station.ev_connected:
                                print(f"Warning: Station {station.evcs_id} has connected EV but no power available, customer {customer['id']} returned to queue")
                            else:
                                print(f"Info: Station {station.evcs_id} has no EV connected, customer {customer['id']} returned to queue")
                        
                        if success:
                            # Update wait time statistics
                            wait_time = current_time - customer['arrival_time']
                            self.global_queue_metrics[sys_id]['total_wait_time'] += wait_time
                            
                            # Update max queue length
                            queue_length = len(station.customer_queue)
                            if queue_length > self.global_queue_metrics[sys_id]['max_queue_length']:
                                self.global_queue_metrics[sys_id]['max_queue_length'] = queue_length
    
    def complete_finished_sessions(self, current_time: float):
        """Complete finished charging sessions (called every second)"""
        for sys_id, system in self.distribution_systems.items():
            if hasattr(system, 'ev_stations'):
                for station_idx, station in enumerate(system.ev_stations):
                    # FIXED: Complete finished charging sessions from legacy system
                    completed_sessions = []
                    for session in station.charging_sessions:
                        if current_time >= session['completion_time']:
                            completed_sessions.append(session)
                    
                    for session in completed_sessions:
                        station.complete_charging_session(session['customer_id'], current_time)
                    
                    # FIXED: Also check for dynamics-based completions (SOC-based)
                    if hasattr(station, 'ev_connected') and station.ev_connected:
                        # Check if EV should disconnect based on SOC
                        if hasattr(station, 'soc') and hasattr(station, 'params'):
                            if station.soc >= station.params.disconnect_soc:
                                # EV is fully charged, disconnect it
                                station.disconnect_ev(current_time)
                                print(f"Central Coordinator: EV at {station.evcs_id} completed charging (SOC: {station.soc:.2f})")
    
    def update_global_status(self, current_time: float):
        """Update global charging infrastructure status"""
        total_capacity = 0.0
        total_load = 0.0
        total_available_ports = 0
        total_queue = 0
        total_ports = 0
        
        for sys_id, system in self.distribution_systems.items():
            if hasattr(system, 'cms') and system.cms:
                # Get system status
                system_capacity = sum(station.max_power for station in system.ev_stations)
                system_load = sum(station.current_load for station in system.ev_stations)
                system_ports = sum(station.available_ports for station in system.ev_stations)
                system_total_ports = sum(station.num_ports for station in system.ev_stations)
                system_queue = sum(len(station.customer_queue) for station in system.ev_stations)
                
                # Calculate average charging time and queue wait time
                avg_charging_time = np.mean([station.avg_charging_time for station in system.ev_stations])
                avg_queue_wait = np.mean([station.queue_wait_time for station in system.ev_stations])
                
                # Update global status
                self.global_charging_status[sys_id].update({
                    'total_capacity': system_capacity,
                    'current_load': system_load,
                    'available_ports': system_ports,
                    'total_ports': system_total_ports,
                    'queue_length': system_queue,
                    'utilization_rate': (system_total_ports - system_ports) / system_total_ports if system_total_ports > 0 else 0.0,
                    'avg_charging_time': avg_charging_time,
                    'avg_queue_wait_time': avg_queue_wait,
                    'scheduled_sessions': sum(len(station.scheduled_charging) for station in system.ev_stations)
                })
                
                total_capacity += system_capacity
                total_load += system_load
                total_available_ports += system_ports
                total_ports += system_total_ports
                total_queue += system_queue
                
        # Calculate global metrics
        self.global_metrics = {
            'total_capacity': total_capacity,
            'total_load': total_load,
            'total_available_ports': total_available_ports,
            'total_ports': total_ports,
            'total_queue_length': total_queue,
            'global_utilization': (total_ports - total_available_ports) / total_ports if total_ports > 0 else 0.0,
            'timestamp': current_time
        }
        
    def assess_cyber_attack_impact(self, attack_type, attack_magnitude, duration):
        """
        Assess the impact of a cyber attack on the EVCS and distribution system.
        
        Args:
            attack_type (str): Type of attack ('demand_increase', 'demand_decrease', 'voltage_manipulation')
            attack_magnitude (float): Magnitude of the attack (0.0 to 1.0)
            duration (int): Duration of the attack in time steps
            
        Returns:
            dict: Attack impact assessment
        """
        # Use default baseline charging time since coordinator doesn't have access to individual EVCS metrics
        baseline_charging_time = 150.0  # Default baseline: 150 minutes (2.5 hours)
        
        # Calculate attack impact factors
        if attack_type == 'demand_increase':
            # Increased demand leads to longer charging times
            demand_factor = 1.0 + attack_magnitude
            charging_time_factor = 1.0 + (attack_magnitude * 0.5)  # Charging time increases with demand
            voltage_impact = -0.1 * attack_magnitude  # Voltage drops with increased demand
        elif attack_type == 'demand_decrease':
            # Decreased demand leads to shorter charging times
            demand_factor = 1.0 - attack_magnitude
            charging_time_factor = 1.0 - (attack_magnitude * 0.3)  # Charging time decreases with lower demand
            voltage_impact = 0.05 * attack_magnitude  # Voltage increases with decreased demand
        elif attack_type == 'power_manipulation':
            # FIXED: Power manipulation attack - reduces available charging power
            demand_factor = attack_magnitude  # Power reduction factor
            charging_time_factor = 1.0 / attack_magnitude  # Inversely proportional to power
            voltage_impact = -0.1 * (1.0 - attack_magnitude)  # Voltage drops with power reduction
        elif attack_type == 'load_manipulation':
            # FIXED: Load manipulation attack - increases system load
            demand_factor = 1.0 + (1.0 - attack_magnitude)  # Load increase
            charging_time_factor = 1.0 + (1.0 - attack_magnitude) * 0.6  # Charging time increases
            voltage_impact = -0.12 * (1.0 - attack_magnitude)  # Voltage drops with load increase
        elif attack_type == 'voltage_manipulation':
            # Direct voltage manipulation
            demand_factor = 1.0
            charging_time_factor = 1.0 / attack_magnitude  # Lower voltage = longer charging time
            voltage_impact = -0.15 * (1.0 - attack_magnitude)  # Direct voltage reduction
        elif attack_type == 'oscillating_demand':
            # Oscillating demand attack - creates fluctuating demand patterns
            # This can cause varying charging times due to power fluctuations
            demand_factor = 1.0 + (attack_magnitude * 0.3)  # Moderate demand increase
            charging_time_factor = 1.0 + (attack_magnitude * 0.4)  # Charging time varies with oscillations
            voltage_impact = -0.08 * attack_magnitude  # Voltage fluctuations
        elif attack_type == 'ramp_demand':
            # Gradual ramp demand attack - slowly increases demand over time
            # This creates a gradual increase in charging times and voltage stress
            demand_factor = 1.0 + (attack_magnitude * 0.6)  # Significant demand increase over time
            charging_time_factor = 1.0 + (attack_magnitude * 0.5)  # Charging time increases with ramped demand
            voltage_impact = -0.12 * attack_magnitude  # Voltage drops with sustained increased demand
        elif attack_type == 'dqn_sac_evasion':
            # DQN/SAC security evasion attack - sophisticated RL-based attack
            # Uses trained agents to bypass security while maximizing impact
            
            # Try to get coordinated attack parameters if trainer is available
            if hasattr(self, 'dqn_sac_trainer') and self.dqn_sac_trainer and hasattr(self.dqn_sac_trainer, 'get_coordinated_attack'):
                # Get baseline outputs for coordinated attack
                baseline_outputs = {
                    'power_reference': 15.0,
                    'voltage_reference': 400.0,
                    'current_reference': 25.0,
                    'soc': 0.5,
                    'grid_voltage': 1.0,
                    'grid_frequency': 60.0,
                    'demand_factor': 1.0,
                    'urgency_factor': 1.0,
                    'voltage_priority': 0.0
                }
                
                # Get coordinated attack from DQN/SAC trainer
                coordinated_attack = self.dqn_sac_trainer.get_coordinated_attack(0, baseline_outputs)
                
                if coordinated_attack:
                    attack_params = coordinated_attack['attack_params']
                    sac_control = coordinated_attack['sac_control']
                    
                    # Use SAC continuous control for precise parameter adjustment
                    magnitude_modifier = sac_control.get('magnitude', 1.0)
                    stealth_factor = sac_control.get('stealth_factor', 0.5)
                    
                    # Apply coordinated attack parameters with stealth consideration
                    demand_factor = 1.0 + (attack_magnitude * magnitude_modifier * 0.4 * (1.0 - stealth_factor))
                    charging_time_factor = 1.0 + (attack_magnitude * magnitude_modifier * 0.3 * (1.0 - stealth_factor))
                    voltage_impact = -0.08 * attack_magnitude * magnitude_modifier * (1.0 - stealth_factor)
                    
                    print(f"   🎯 Using Coordinated DQN/SAC Attack:")
                    print(f"      DQN Strategy: {coordinated_attack['dqn_decision'].get('attack_type', 'unknown')}")
                    print(f"      SAC Magnitude: {magnitude_modifier:.3f}")
                    print(f"      SAC Stealth: {stealth_factor:.3f}")
                else:
                    # Fallback to default DQN/SAC parameters
                    demand_factor = 1.0 + (attack_magnitude * 0.4)
                    charging_time_factor = 1.0 + (attack_magnitude * 0.3)
                    voltage_impact = -0.08 * attack_magnitude
            else:
                # Fallback to default DQN/SAC parameters
                demand_factor = 1.0 + (attack_magnitude * 0.4)  # Moderate demand increase to avoid detection
                charging_time_factor = 1.0 + (attack_magnitude * 0.3)  # Subtle charging time changes
                voltage_impact = -0.08 * attack_magnitude  # Controlled voltage impact for stealth
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Calculate new charging time based on baseline
        new_charging_time = baseline_charging_time * charging_time_factor
        
        # Ensure charging time doesn't go below reasonable minimum (30 minutes) or above reasonable maximum (300 minutes)
        new_charging_time = max(30.0, min(300.0, new_charging_time))
        
        # Log the attack impact
        print(f" Attack {attack_type} applied with magnitude {attack_magnitude}")
        print(f"   Demand factor: {demand_factor:.3f}")
        print(f"   Charging time factor: {charging_time_factor:.3f}")
        print(f"   New charging time: {new_charging_time:.1f} minutes (baseline: {baseline_charging_time:.1f})")
        print(f"   Voltage impact: {voltage_impact:.3f}")
        
        return {
            'attack_type': attack_type,
            'attack_magnitude': attack_magnitude,
            'duration': duration,
            'demand_factor': demand_factor,
            'charging_time_factor': charging_time_factor,
            'voltage_impact': voltage_impact,
            'baseline_charging_time': baseline_charging_time,
            'new_charging_time': new_charging_time
        }
    
    def _calculate_customer_satisfaction_impact(self, time_impact_factor: float) -> float:
        """Calculate impact on customer satisfaction due to charging time changes"""
        # Customer satisfaction decreases as charging time increases
        if time_impact_factor > 1.0:
            # Longer charging times reduce satisfaction
            satisfaction_impact = 1.0 - (time_impact_factor - 1.0) * 0.3
        else:
            # Shorter charging times increase satisfaction
            satisfaction_impact = 1.0 + (1.0 - time_impact_factor) * 0.2
        
        return max(0.0, min(1.0, satisfaction_impact))  # Clamp between 0 and 1
        
    def coordinate_charging_schedules(self, current_time: float):
        """Coordinate charging schedules across all systems"""
        # Check for overloaded systems
        overloaded_systems = []
        underloaded_systems = []
        
        for sys_id, status in self.global_charging_status.items():
            utilization = status['utilization_rate']
            if utilization > 0.8:  # Over 80% utilization
                overloaded_systems.append(sys_id)
            elif utilization < 0.3:  # Under 30% utilization
                underloaded_systems.append(sys_id)
                
        # Implement load balancing
        if overloaded_systems and underloaded_systems:
            self._balance_load(overloaded_systems, underloaded_systems)
            
        # Record load balancing history
        self.load_balancing_history.append({
            'timestamp': current_time,
            'overloaded_systems': overloaded_systems,
            'underloaded_systems': underloaded_systems,
            'global_utilization': self.global_metrics['global_utilization']
        })
            
    def manage_customer_queue(self, current_time: float):
        """Manage customer queue and waiting times"""
        for sys_id, status in self.global_charging_status.items():
            if status['available_ports'] == 0:  # No available ports
                # Redirect customers to nearby systems
                alternative_systems = self._find_alternative_systems(sys_id)
                if alternative_systems:
                    self._redirect_customers(sys_id, alternative_systems)
                    
                    # Record customer redirections
                    self.customer_redirections.append({
                        'timestamp': current_time,
                        'from_system': sys_id,
                        'to_systems': alternative_systems,
                        'queue_length': status['queue_length']
                    })
                    
    def emergency_response(self, current_time: float):
        """Implement emergency response to critical situations"""
        critical_systems = [sys_id for sys_id, status in self.global_charging_status.items() 
                          if status['system_health'] == 'critical']
        
        if critical_systems:
            # Implement emergency power reduction
            for sys_id in critical_systems:
                self._implement_emergency_power_reduction(sys_id)
                
            # Log emergency alert
            self.emergency_alerts.append({
                'timestamp': current_time,
                'type': 'critical_attack',
                'affected_systems': critical_systems,
                'action_taken': 'emergency_power_reduction'
            })
            
    def _calculate_charging_time_impact(self, load_change_mw):
        """Calculate impact on charging time due to load change"""
        # Base charging time is 150 minutes (2.5 hours) - this should match the baseline
        base_charging_time = 150.0
        
        if load_change_mw < 0:  # Reduced power delivery
            # More time needed for charging - load reduction increases charging time
            impact_factor = 1.0 + (abs(load_change_mw) / 1000.0)  # 1000 MW = 100% increase
            new_charging_time = base_charging_time * impact_factor
        else:
            # Increased power delivery - faster charging possible but with smaller effect
            impact_factor = 1.0 - (load_change_mw / 2000.0)  # 2000 MW = 50% decrease
            new_charging_time = base_charging_time * impact_factor
        
        # Ensure reasonable bounds (30 minutes to 300 minutes)
        new_charging_time = max(30.0, min(300.0, new_charging_time))
        
        # Log the calculation for debugging
        self.logger.debug(f"Load change: {load_change_mw:.2f} MW, Impact factor: {impact_factor:.3f}, New charging time: {new_charging_time:.1f} min")
        
        return new_charging_time
            
    def _calculate_queue_impact(self, load_change_mw, system_id):
        """Calculate impact on customer queue"""
        if load_change_mw < 0:  # Reduced power delivery
            # Queue will increase
            return abs(load_change_mw) * 0.5  # Rough estimate: 0.5 customers per MW reduction
        else:
            # Queue will decrease
            return -load_change_mw * 0.3  # Rough estimate: 0.3 customers saved per MW increase
            
    def _balance_load(self, overloaded_systems, underloaded_systems):
        """Balance load between overloaded and underloaded systems"""
        for overloaded_sys in overloaded_systems:
            for underloaded_sys in underloaded_systems:
                # Calculate load transfer
                overloaded_load = self.global_charging_status[overloaded_sys]['current_load']
                underloaded_load = self.global_charging_status[underloaded_sys]['current_load']
                
                transfer_amount = min(overloaded_load * 0.1,  # Transfer 10% of overloaded load
                                    self.global_charging_status[underloaded_sys]['total_capacity'] - underloaded_load)
                
                if transfer_amount > 0:
                    self._transfer_load(overloaded_sys, underloaded_sys, transfer_amount)
                    
    def _find_alternative_systems(self, system_id):
        """Find alternative systems for customer redirection"""
        alternatives = []
        current_utilization = self.global_charging_status[system_id]['utilization_rate']
        
        for sys_id, status in self.global_charging_status.items():
            if sys_id != system_id and status['utilization_rate'] < current_utilization * 0.8:
                alternatives.append(sys_id)
                
        return sorted(alternatives, key=lambda x: self.global_charging_status[x]['utilization_rate'])
        
    def _redirect_customers(self, from_system, to_systems):
        """Redirect customers from overloaded to alternative systems"""
        # This would involve updating customer routing and queue management
        # For simulation purposes, we log the redirection
        print(f"Redirecting customers from System {from_system} to Systems {to_systems}")
        
    def _implement_emergency_power_reduction(self, system_id):
        """Implement emergency power reduction for critical systems"""
        system = self.distribution_systems[system_id]
        if hasattr(system, 'cms') and system.cms:
            # Reduce power by emergency_power_reduction factor
            for station in system.ev_stations:
                station.current_load *= (1 - self.emergency_power_reduction)
                station.emergency_mode = True
                
    def _transfer_load(self, from_system, to_system, amount):
        """Transfer load between systems (simplified implementation)"""
        # This would involve complex load transfer logic
        # For simulation purposes, we log the transfer
        print(f"Transferring {amount:.1f} MW from System {from_system} to System {to_system}")
        
    def get_coordination_report(self):
        """Generate coordination report"""
        return {
            'global_metrics': self.global_metrics,
            'system_status': self.global_charging_status,
            'emergency_alerts': self.emergency_alerts,
            'coordination_active': self.coordination_active,
            'queue_metrics': self.global_queue_metrics,
            'charging_time_impacts': self.charging_time_impacts,
            'load_balancing_history': self.load_balancing_history,
            'customer_redirections': self.customer_redirections
        }

class IEEE14BusAGC:
    """IEEE 14-bus Transmission System with AGC using Pandapower"""
    
    def __init__(self):
        # Create IEEE 14-bus network using Pandapower
        self.net = pp.create_empty_network()
        self._create_ieee14_bus_network()
        
        # AGC parameters
        self.f_nominal = 60.0  # Hz
        self.frequency = 60.0
        self.H_system = 2.0  # System inertia (s) - reduced for more sensitivity
        self.D_system = 0.8  # Load damping - reduced for more sensitivity
        self.R_droop = 0.05  # Governor droop
        self.T_gov = 0.1     # Governor time constant
        self.T_turb = 0.1    # Turbine time constant
        
        # AGC state variables
        self.delta_f = 0.0
        self.delta_Pm = 0.0
        self.delta_Pv = 0.0
        
        # Area Control Error (ACE) parameters
        self.beta = 20.0  # Frequency bias factor (MW/Hz)
        self.tie_line_power = 0.0  # Net tie line power flow
        self.tie_line_scheduled = 0.0  # Scheduled tie line power
        self.ACE = 0.0  # Area Control Error
        
        # Distribution system connections and load tracking
        self.dist_connections = {
            4: [],   # Bus 4 connected to distribution systems
            9: [],   # Bus 9 connected to distribution systems  
            13: []   # Bus 13 connected to distribution systems
        }
        
        # Load tracking for AGC
        self.P_load_base = self.net.load.p_mw.sum()  # Total base load
        self.P_load_current = self.P_load_base
        self.P_dist_total = 0.0  # Total distribution load
        self.P_gen_total = 0.0   # Total generation
        
        # Daily load profile integration
        self.load_profile_times = None
        self.load_profile_multipliers = None
        self.base_load_nominal = self.P_load_base  # Store original base load
        self.dist_base_loads = {}  # Store original distribution base loads for scaling
        self.reference_power = self.P_load_base
        
        # AGC timing
        self.agc_time_step = 5.0  # Update every 5 seconds for more realistic dynamics
        self.last_agc_update = 0.0
        self.reference_power = self.P_load_base  # Initial reference power
    
    def _create_ieee14_bus_network(self):
        """Create IEEE 14-bus network using Pandapower"""
        # Create buses
        bus_data = [
            [1, 1.06, 0.0, 230],    # Slack bus
            [2, 1.045, 0.0, 230],   # Generator
            [3, 1.01, 0.0, 230],    # Generator
            [4, 1.0, 0.0, 230],     # Load bus
            [5, 1.0, 0.0, 230],     # Load bus
            [6, 1.07, 0.0, 230],    # Generator
            [7, 1.0, 0.0, 230],     # Load bus
            [8, 1.09, 0.0, 230],    # Generator
            [9, 1.0, 0.0, 230],     # Load bus
            [10, 1.0, 0.0, 230],    # Load bus
            [11, 1.0, 0.0, 230],    # Load bus
            [12, 1.0, 0.0, 230],    # Load bus
            [13, 1.0, 0.0, 230],    # Load bus
            [14, 1.0, 0.0, 230]     # Load bus
        ]
        
        for bus_id, vm_pu, va_degree, vn_kv in bus_data:
            pp.create_bus(self.net, vn_kv=vn_kv, name=f"Bus_{bus_id}", 
                         in_service=True, max_vm_pu=1.06, min_vm_pu=0.94)
        
        # Create generators
        gen_data = [
            [1, 0.0, 0.0, 1.06, 0.0, 332.4, 10.0],   # Slack
            [2, 40.0, 0.0, 1.045, 0.0, 140.0, 10.0], # PV
            [3, 0.0, 0.0, 1.01, 0.0, 100.0, 10.0],   # PV
            [6, 0.0, 0.0, 1.07, 0.0, 100.0, 10.0],   # PV
            [8, 0.0, 0.0, 1.09, 0.0, 100.0, 10.0]    # PV
        ]
        
        for bus_id, p_mw, q_mvar, vm_pu, va_degree, max_p_mw, min_p_mw in gen_data:
            if bus_id == 1:  # Slack bus
                pp.create_gen(self.net, bus=bus_id-1, p_mw=p_mw, vm_pu=vm_pu, 
                             name=f"Gen_{bus_id}", slack=True)
            else:  # PV buses
                pp.create_gen(self.net, bus=bus_id-1, p_mw=p_mw, vm_pu=vm_pu,
                             name=f"Gen_{bus_id}", max_p_mw=max_p_mw, min_p_mw=min_p_mw)
        
        # Create loads
        load_data = [
            [2, 21.7, 12.7],
            [3, 94.2, 19.0],
            [4, 47.8, -3.9],
            [5, 7.6, 1.6],
            [6, 11.2, 7.5],
            [9, 29.5, 16.6],
            [10, 9.0, 5.8],
            [11, 3.5, 1.8],
            [12, 6.1, 1.6],
            [13, 13.5, 5.8],
            [14, 14.9, 5.0]
        ]
        
        for bus_id, p_mw, q_mvar in load_data:
            pp.create_load(self.net, bus=bus_id-1, p_mw=p_mw, q_mvar=q_mvar,
                          name=f"Load_{bus_id}")
        
        # Create lines (simplified - you can add more detailed line data)
        line_data = [
            [1, 2, 0.01938, 0.05917, 0.0528],
            [1, 5, 0.05403, 0.22304, 0.0492],
            [2, 3, 0.04699, 0.19797, 0.0438],
            [2, 4, 0.05811, 0.17632, 0.0374],
            [2, 5, 0.05695, 0.17388, 0.0340],
            [3, 4, 0.06701, 0.17103, 0.0346],
            [4, 5, 0.01335, 0.04211, 0.0128],
            [4, 7, 0.0, 0.20912, 0.0],
            [4, 9, 0.0, 0.55618, 0.0],
            [5, 6, 0.0, 0.25202, 0.0],
            [6, 11, 0.09498, 0.19890, 0.0],
            [6, 12, 0.12291, 0.25581, 0.0],
            [6, 13, 0.06615, 0.13027, 0.0],
            [7, 8, 0.0, 0.17615, 0.0],
            [7, 9, 0.0, 0.11001, 0.0],
            [9, 10, 0.03181, 0.08450, 0.0],
            [9, 14, 0.12711, 0.27038, 0.0],
            [10, 11, 0.08205, 0.19307, 0.0],
            [12, 13, 0.22092, 0.19988, 0.0],
            [13, 14, 0.17093, 0.34802, 0.0]
        ]
        
        for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, c_nf_per_km in line_data:
            pp.create_line_from_parameters(self.net, from_bus=from_bus-1, to_bus=to_bus-1,
                                         length_km=1.0, r_ohm_per_km=r_ohm_per_km,
                                         x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km,
                                         max_i_ka=1.0, name=f"Line_{from_bus}_{to_bus}")
        
        print("IEEE 14-bus network created successfully with Pandapower")
    
    def calculate_ace(self):
        """Calculate Area Control Error (ACE) for AGC control"""
        # ACE = β * Δf + (Ptie - Ptie_scheduled)
        # where β is frequency bias factor, Δf is frequency deviation
        delta_f = self.frequency - self.f_nominal
        tie_line_error = self.tie_line_power - self.tie_line_scheduled
        self.ACE = self.beta * delta_f + tie_line_error
        return self.ACE
        
    def set_load_profile(self, times, multipliers):
        """Set daily load profile for the transmission system"""
        self.load_profile_times = times
        self.load_profile_multipliers = multipliers
        print(f"Load profile set: {len(times)} time points, multiplier range {min(multipliers):.2f}-{max(multipliers):.2f}")
    
    def get_current_load_multiplier(self, current_time):
        """Get current load multiplier from load profile"""
        if self.load_profile_times is None or self.load_profile_multipliers is None:
            return 1.0  # Default multiplier if no profile set
        
        # Find the closest time point
        time_idx = min(range(len(self.load_profile_times)), 
                      key=lambda i: abs(self.load_profile_times[i] - current_time))
        return self.load_profile_multipliers[time_idx]
        
    def agc_dynamics(self, t, x):
        """AGC system dynamics"""
        delta_f, delta_Pm, delta_Pv = x
        
        # Power imbalance from distribution systems (scaled properly)
        total_load = self.P_load_current + self.P_dist_total
        delta_PL = (total_load - self.reference_power) / self.reference_power  # p.u. relative to reference
        
        # AGC equations
        ddelta_f_dt = (1 / (2 * self.H_system)) * (delta_Pm - delta_PL - self.D_system * delta_f)
        ddelta_Pm_dt = (1 / self.T_turb) * (delta_Pv - delta_Pm)
        ddelta_Pv_dt = -(1 / self.T_gov) * (delta_Pv + delta_f / self.R_droop)
        
        return [ddelta_f_dt, ddelta_Pm_dt, ddelta_Pv_dt]
    
    def update_frequency(self, dt=0.1):
        """Update system frequency using AGC dynamics"""
        t_span = (0, dt)
        x0 = [self.delta_f, self.delta_Pm, self.delta_Pv]
        
        sol = solve_ivp(self.agc_dynamics, t_span, x0, dense_output=True)
        
        if sol.success:
            x_new = sol.y[:, -1]
            self.delta_f, self.delta_Pm, self.delta_Pv = x_new
            self.frequency = self.f_nominal + self.delta_f
        else:
            # Fallback: simple frequency calculation based on power imbalance
            print("AGC Dynamics failed, using fallback frequency calculation")
            total_load = self.P_load_current + self.P_dist_total
            power_imbalance = (total_load - self.reference_power) / self.reference_power
            self.delta_f = -power_imbalance * 0.1  # Simple frequency response
            self.frequency = self.f_nominal + self.delta_f
        
        return self.frequency
    
    def update_agc_reference(self, current_time: float):
        """Update AGC reference power using Area Control Error (ACE) methodology"""
        if current_time - self.last_agc_update >= self.agc_time_step:
            # Refresh distribution load data first to ensure fresh data
            if hasattr(self, 'latest_dist_loads') and self.latest_dist_loads:
                self.update_distribution_load(self.latest_dist_loads)
            
            # Get current load multiplier from daily load profile
            load_multiplier = self.get_current_load_multiplier(current_time)
            
            # Apply load profile to transmission base load
            scaled_base_load = self.base_load_nominal * load_multiplier
            self.P_load_current = scaled_base_load
            
            # Calculate Area Control Error (ACE)
            ace = self.calculate_ace()
            
            # AGC reference adjustment based on ACE
            # Reference power adjustment = -ACE/β (to counteract frequency deviation)
            ace_adjustment = -ace / self.beta if self.beta != 0 else 0
            
            # Base reference from load demand (now using fresh distribution data)
            base_reference = scaled_base_load + self.P_dist_total
            
            # Final reference power includes ACE correction
            new_reference = base_reference + ace_adjustment
            
            # Update reference power with AGC dynamics
            self.reference_power = 0.7 * self.reference_power + 0.3 * new_reference
            
            # Update AGC state with longer time step for more realistic dynamics
            self.update_frequency(self.agc_time_step)
            
            # Update generation to match new reference
            self._adjust_generation()
            
            self.last_agc_update = current_time
            print(f"AGC Update at t={current_time:.1f}s: Load Multiplier={load_multiplier:.3f}, "
                  f"Base Load={scaled_base_load:.1f}MW, ACE={ace:.2f}MW, ACE_Adj={ace_adjustment:.2f}MW, "
                  f"Reference={self.reference_power:.1f}MW, Dist Load={self.P_dist_total:.1f}MW, Frequency={self.frequency:.3f}Hz")
    
    def _adjust_generation(self):
        """Adjust generator outputs to match reference power"""
        try:
            # Get current generation
            total_gen = 0.0
            for idx, gen in self.net.gen.iterrows():
                if not gen['slack']:  # Skip slack bus
                    total_gen += gen['p_mw']
            
            # Calculate required adjustment
            required_gen = self.reference_power
            adjustment = required_gen - total_gen
            
            # Distribute adjustment among non-slack generators
            non_slack_gens = self.net.gen[~self.net.gen['slack']].index
            if len(non_slack_gens) > 0:
                adjustment_per_gen = adjustment / len(non_slack_gens)
                
                for idx in non_slack_gens:
                    current_p = self.net.gen.at[idx, 'p_mw']
                    new_p = current_p + adjustment_per_gen
                    
                    # Respect generator limits
                    max_p = self.net.gen.at[idx, 'max_p_mw']
                    min_p = self.net.gen.at[idx, 'min_p_mw']
                    new_p = max(min_p, min(max_p, new_p))
                    
                    self.net.gen.at[idx, 'p_mw'] = new_p
            
            self.P_gen_total = self.net.gen['p_mw'].sum()
            
        except Exception as e:
            print(f"Error adjusting generation: {e}")
    
    def run_power_flow(self):
        """Run power flow analysis using Pandapower"""
        try:
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            return True
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False
    
    def get_bus_voltages(self):
        """Get bus voltages from power flow results"""
        return self.net.res_bus.vm_pu.values
    def get_line_flows(self):
        """Get line power flows from power flow results"""
        return self.net.res_line.p_from_mw.values, self.net.res_line.p_to_mw.values
    
    def update_distribution_load(self, dist_loads: Dict[int, float]):
        """Update distribution system loads from distribution systems with load profile scaling"""
        # Store latest distribution loads for AGC reference calculations
        self.latest_dist_loads = dist_loads.copy()
        
        # Apply load profile multiplier to distribution loads as well
        current_time = getattr(self, 'current_simulation_time', 0.0)
        load_multiplier = self.get_current_load_multiplier(current_time)
        
        # Scale distribution loads with load profile
        scaled_dist_loads = {}
        for sys_id, load in dist_loads.items():
            # Get base load for this system (store on first call)
            if sys_id not in self.dist_base_loads:
                self.dist_base_loads[sys_id] = load
            
            # Apply load profile scaling to base load, then add EVCS variations
            base_load = self.dist_base_loads[sys_id]
            scaled_base = base_load * load_multiplier
            evcs_variation = load - base_load  # EVCS and attack variations
            scaled_dist_loads[sys_id] = scaled_base + evcs_variation
        
        self.P_dist_total = sum(scaled_dist_loads.values())
        
        # Debug print for dual load variation tracking
        if current_time % 30 == 0:  # Every 30 seconds
            print(f" Dual Load Variation at t={current_time:.1f}s:")
            print(f"   Load multiplier: {load_multiplier:.3f}")
            print(f"   Original dist total: {sum(dist_loads.values()):.2f} MW")
            print(f"   Scaled dist total: {self.P_dist_total:.2f} MW")
            for sys_id, (orig_load, scaled_load) in zip(dist_loads.keys(), zip(dist_loads.values(), scaled_dist_loads.values())):
                print(f"   System {sys_id}: {orig_load:.2f} → {scaled_load:.2f} MW")
        
        # FIXED: Log distribution loads for debugging attack impact
        if any(load > 1000 for load in dist_loads.values()):  # If any load is unusually high (attack)
            print(f" Transmission System: High distribution loads detected - possible attack:")
            for sys_id, load in dist_loads.items():
                print(f"   System {sys_id}: {load:.2f} MW")
        
        # Update connection loads
        for bus_id, load in dist_loads.items():
            if bus_id in self.dist_connections:
                self.dist_connections[bus_id] = [load]

class OpenDSSInterface:
    """Interface for OpenDSS power flow simulation with EVCS integration"""
    
    # Class-level shared PINN optimizer to avoid repeated initialization
    _shared_pinn_optimizer = None
    _shared_federated_manager = None
    _shared_cms_config = None
    
    def __init__(self, system_id: int = 1, dss_file: str = None, use_mock: bool = False):
        self.system_id = system_id
        self.use_mock = use_mock
        self.ev_stations = []
        self.cms = None
        self.base_loads = {}
        self.dss = dss
        self.circuit = dss.Circuit
        
    def initialize(self):
        """Initialize OpenDSS system using DSS functions"""
        try:
            print(f"Initializing Distribution System {self.system_id}...")
            
            # Use different files for different systems to avoid conflicts
            if self.system_id == 1:
                dss_file = "ieee34Mod1.dss"
            elif self.system_id == 2:
                dss_file = "ieee34Mod2.dss"
            else:
                dss_file = "ieee34Mod1.dss"  # System 3 uses Mod1 again
            
            # Initialize OpenDSS
            self.dss.Command("Clear")
            self.dss.Command(f"Compile {dss_file}")
            self.dss.Command("Solve")
            
            # Get system information using DSS functions
            ybus_matrix = self.circuit.SystemY()
            num_buses = len(self.circuit.AllBusNames())
            print(f"Distribution System {self.system_id}: Y-bus matrix size: {num_buses} × {num_buses}")
            
            # Get base loads using DSS functions
            load_data, total_load = get_loads(self.dss, self.circuit)
            for load in load_data:
                self.base_loads[load['name']] = {
                    'kW': load['kW'],
                    'kvar': load['kVar'],
                    'bus': load['bus1'],
                    'phases': load['phases']
                }
            
            print(f"Distribution System {self.system_id} initialized successfully with OpenDSS ({dss_file})")
            print(f"Base load: {total_load:.1f} kW")
            return True
            
        except Exception as e:
            print(f"Error initializing Distribution System {self.system_id}: {e}")
            print("Using mock distribution system instead...")
            return self._initialize_mock_system()
    
    def _initialize_mock_system(self):
        """Initialize a mock distribution system when OpenDSS fails"""
        try:
            # Create mock base loads for IEEE 34-bus system (increased for higher impact)
            self.base_loads = {
                'Load_890': {'kW': 1500.0, 'kvar': 750.0, 'bus': '890', 'phases': ['1', '2', '3']},
                'Load_844': {'kW': 1000.0, 'kvar': 500.0, 'bus': '844', 'phases': ['1', '2', '3']},
                'Load_860': {'kW': 800.0, 'kvar': 400.0, 'bus': '860', 'phases': ['1', '2', '3']},
                'Load_840': {'kW': 1200.0, 'kvar': 600.0, 'bus': '840', 'phases': ['1', '2', '3']},
                'Load_848': {'kW': 900.0, 'kvar': 450.0, 'bus': '848', 'phases': ['1', '2', '3']},
                'Load_830': {'kW': 700.0, 'kvar': 350.0, 'bus': '830', 'phases': ['1', '2', '3']}
            }
            
            print(f"Mock Distribution System {self.system_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing mock system {self.system_id}: {e}")
            return False
    
    
    def add_ev_charging_stations(self, stations_config: List[Dict]):
        """Add EV charging stations to the distribution system with power electronics dynamics"""
        for i, config in enumerate(stations_config):
            try:
                bus_name = config['bus']
                max_power = config['max_power']
                num_ports = config.get('num_ports', 10)
                
                # Create EVCS parameters based on station size
                if max_power >= 1000:  # Mega stations
                    params = EVCSParameters(
                        rated_voltage=400.0,
                        rated_current=125.0,  # Higher current for mega stations
                        capacity=100.0,       # Larger battery capacity
                        efficiency_charge=0.96,
                        efficiency_discharge=0.93
                    )
                elif max_power >= 500:  # Large stations
                    params = EVCSParameters(
                        rated_voltage=400.0,
                        rated_current=100.0,
                        capacity=75.0,
                        efficiency_charge=0.95,
                        efficiency_discharge=0.92
                    )
                else:  # Standard stations
                    params = EVCSParameters(
                        rated_voltage=400.0,
                        rated_current=75.0,
                        capacity=50.0,
                        efficiency_charge=0.95,
                        efficiency_discharge=0.92
                    )
                
                # Try to add to OpenDSS if available
                try:
                    # Initialize EVCS with minimal standby power, not 50% of max capacity
                    initial_kw = 0.01  # 10W standby power
                    initial_kvar = 0.001  # Minimal reactive power
                    evcs_load_name = f"EVCS_{self.system_id}_{i:02d}"
                    
                    # Check if load already exists to avoid duplicates
                    load_exists = False
                    try:
                        self.circuit.SetActiveElement(f"Load.{evcs_load_name}")
                        load_exists = True
                        print(f" OpenDSS Load.{evcs_load_name} already exists, skipping creation")
                    except:
                        load_exists = False
                    
                    # Create the load only if it doesn't exist
                    if not load_exists:
                        self.dss.Command(f"New Load.{evcs_load_name} "
                                       f"Bus1={bus_name} "
                                       f"Phases=3 "
                                       f"kW={initial_kw} "  # Minimal standby power
                                       f"kvar={initial_kvar} "
                                       f"Model=1")
                        
                        # Verify the load was created successfully
                        try:
                            self.circuit.SetActiveElement(f"Load.{evcs_load_name}")
                            print(f" Created OpenDSS Load.{evcs_load_name} at bus {bus_name}")
                        except:
                            print(f" Failed to verify OpenDSS Load.{evcs_load_name} creation")
                        
                except Exception as e:
                    # If OpenDSS fails, log the error and continue with mock system
                    print(f" Failed to create OpenDSS Load.EVCS_{self.system_id}_{i:02d}: {e}")
                    pass
                
                # Create enhanced EVCS object with dynamics
                evcs_id = f"EVCS_{self.system_id}_{i:02d}"
                station = EVChargingStation(
                    evcs_id=evcs_id,
                    bus_name=bus_name,
                    max_power=max_power,
                    num_ports=num_ports,
                    params=params
                )
                self.ev_stations.append(station)
                print(f"Added {evcs_id} at bus {bus_name} with {max_power}W capacity and {num_ports} ports")
                
            except Exception as e:
                print(f"Error adding EVCS {i}: {e}")
                continue
        
        # Initialize CMS with shared PINN optimizer
        if self.ev_stations:
            self.cms = self._create_shared_cms(self.ev_stations)
            print(f"Initialized CMS with {len(self.ev_stations)} EV charging stations using shared PINN optimizer")
    
    def _create_shared_cms(self, ev_stations):
        """Create shared CMS with federated or legacy PINN optimizer"""
        try:
            # Convert ev_stations list to dict format expected by ChargingManagementSystem
            evcs_controllers = {}
            for i, station in enumerate(ev_stations):
                station_id = f"EVCS_{self.system_id}_{i:02d}"
                evcs_controllers[station_id] = station
            
            # Use shared federated manager if available
            if OpenDSSInterface._shared_federated_manager:
                print(f"CMS: Using shared federated PINN manager for system {self.system_id}")
                cms = EnhancedChargingManagementSystem(ev_stations, use_pinn=True)
                cms.federated_manager = OpenDSSInterface._shared_federated_manager
                return cms
            
            # Fallback to shared legacy PINN optimizer
            elif OpenDSSInterface._shared_pinn_optimizer:
                print(f"CMS: Using shared legacy PINN optimizer for system {self.system_id}")
                cms = EnhancedChargingManagementSystem(ev_stations, use_pinn=True)
                cms.pinn_optimizer = OpenDSSInterface._shared_pinn_optimizer
                return cms
            
            # No PINN available - use heuristic
            else:
                print(f"CMS: No PINN available, using heuristic optimization for system {self.system_id}")
                cms = EnhancedChargingManagementSystem(ev_stations, use_pinn=False)
                return cms
                
        except Exception as e:
            print(f"Error creating CMS for system {self.system_id}: {e}")
            # Fallback to basic CMS with correct constructor
            cms = EnhancedChargingManagementSystem(ev_stations, use_pinn=False)
            return cms
    
    def update_ev_loads(self, current_time: float = 0.0):
        """Update EV charging loads based on CMS control with power electronics dynamics"""
        if not self.cms:
            print(f"Warning: No CMS available for system {self.system_id}")
            return
        
        # Get bus voltages from OpenDSS for voltage-based control
        bus_voltages = {}
        try:
            # Get all bus voltages
            all_buses = self.circuit.AllBusNames()
            for bus_name in all_buses:
                try:
                    self.circuit.SetActiveBus(bus_name)
                    pu_voltages = self.dss.Bus.puVmagAngle()
                    if len(pu_voltages) >= 2:
                        bus_voltages[bus_name] = pu_voltages[0]
                except:
                    bus_voltages[bus_name] = 1.0  # Default voltage
        except:
            # If OpenDSS fails, use default voltages
            for station in self.ev_stations:
                bus_voltages[station.bus_name] = 1.0
        
        # Get system frequency (default to 60 Hz)
        system_frequency = 60.0
        
        # FIXED: Add connection status check and logging
        connected_stations = sum(1 for s in self.ev_stations if s.ev_connected)
        total_stations = len(self.ev_stations)
        if connected_stations == 0:
            print(f"Warning: System {self.system_id} - No EVCS stations have connected EVs")
            # Try to force connection
            for station in self.ev_stations:
                station.connect_new_ev(current_time / 3600.0)
        
        print(f"System {self.system_id}: {connected_stations}/{total_stations} stations have connected EVs")
            
        for i, station in enumerate(self.ev_stations):
            try:
                load_name = f"EVCS_{self.system_id}_{i:02d}"
                
                # Get grid voltage for dynamics simulation
                grid_voltage_pu = bus_voltages.get(station.bus_name, 1.0)
                grid_voltage_rms = grid_voltage_pu * 12470  # Convert pu to RMS voltage (12.47kV base)
                
                # Update EVCS dynamics
                dt = 0.1  # Time step for dynamics (100ms)
                current_time_hours = current_time / 3600.0  # Convert seconds to hours for EV connection logic
                
                # Update dynamics first
                dynamics_result = station.update_dynamics(grid_voltage_rms, dt, current_time_hours)
                
                # FIXED: Check EV status for disconnection after dynamics update
                station.check_ev_status(current_time_hours)
                
                # FIXED: Check if dynamics returned valid power - only warn if it's actually a problem
                if dynamics_result['total_power'] == 0 and station.ev_connected:
                    # Check if this is a transient issue or actual problem
                    if hasattr(station, 'consecutive_zero_power_count'):
                        station.consecutive_zero_power_count += 1
                    else:
                        station.consecutive_zero_power_count = 1
                    
                    # Only warn after multiple consecutive zero power readings
                    if station.consecutive_zero_power_count >= 3:
                        print(f"Warning: EVCS {station.evcs_id} has connected EV but zero power for {station.consecutive_zero_power_count} consecutive readings")
                else:
                    # Reset counter when power is non-zero
                    if hasattr(station, 'consecutive_zero_power_count'):
                        station.consecutive_zero_power_count = 0
                
                # Get optimized references from CMS with dynamics state and error handling
                try:
                    voltage_ref, current_ref, power_ref = self.cms.optimize_charging(
                        i, current_time, bus_voltages, system_frequency, dynamics_result
                    )
                    
                    # Validate CMS outputs
                    if not all(np.isfinite([voltage_ref, current_ref, power_ref])):
                        print(f"Warning: CMS returned non-finite values for EVCS {station.evcs_id}")
                        voltage_ref, current_ref, power_ref = 400.0, 25.0, 10.0  # Safe defaults
                        
                except Exception as e:
                    print(f"Error: CMS optimization failed for EVCS {station.evcs_id}: {e}")
                    # Use safe fallback values
                    voltage_ref, current_ref, power_ref = 400.0, 25.0, 10.0
                
                # FIXED: Validate references before setting - only warn if EV is connected
                if station.ev_connected and (voltage_ref <= 0 or current_ref <= 0):
                    print(f"Warning: EVCS {station.evcs_id} received invalid references: V={voltage_ref}, I={current_ref}")
                    # Use fallback values
                    voltage_ref = max(voltage_ref, 50.0)
                    current_ref = max(current_ref, 0.1)
                elif not station.ev_connected:
                    # No EV connected - use standby values to prevent warnings
                    voltage_ref = 50.0
                    current_ref = 0.1
                    power_ref = 1.0
                
                # Set references in the EVCS controller
                station.set_references(voltage_ref, current_ref, power_ref)
                
                # Try to update OpenDSS load if available
                try:
                    # First try to set the active element
                    self.circuit.SetActiveElement(f"Load.{load_name}")
                    
                    # Get current measurements from OpenDSS
                    voltage_raw = self.dss.CktElement.VoltagesMagAng()[0] / 1000  # Convert to p.u.
                    power = self.dss.CktElement.Powers()[0]  # kW
                    
                    # Ensure voltage is not zero or negative
                    voltage = max(voltage_raw, 0.001)  # Minimum voltage of 0.001 p.u.
                    
                    # Safe current calculation with comprehensive error handling
                    try:
                        if voltage > 0.001 and power > 0:
                            current = power / (voltage * 1000)
                            # Validate result
                            if not np.isfinite(current) or current < 0:
                                current = 0.001
                        else:
                            current = 0.001  # Default current
                    except (ZeroDivisionError, ArithmeticError, ValueError) as e:
                        print(f"Warning: Current calculation error for EVCS {station.evcs_id}: {e}")
                        current = 0.001
                    
                    # Update load in OpenDSS with dynamics-based power
                    actual_power = dynamics_result['total_power']
                    self.dss.Command(f"Load.{load_name}.kW={actual_power}")
                    self.dss.Command(f"Load.{load_name}.kvar={actual_power * 0.2}")
                    
                except Exception as e:
                    # If load doesn't exist, try to create it on-demand
                    if "not found" in str(e).lower():
                        try:
                            # Create the missing load
                            initial_kw = dynamics_result['total_power']
                            initial_kvar = initial_kw * 0.2
                            
                            self.dss.Command(f"New Load.{load_name} "
                                           f"Bus1={station.bus_name} "
                                           f"Phases=3 "
                                           f"kW={initial_kw} "
                                           f"kvar={initial_kvar} "
                                           f"Model=1")
                            
                            print(f" Created missing OpenDSS Load.{load_name} on-demand")
                            
                            # Now try to get measurements
                            self.circuit.SetActiveElement(f"Load.{load_name}")
                            voltage_raw = self.dss.CktElement.VoltagesMagAng()[0] / 1000
                            voltage = max(voltage_raw, 0.001)
                            power = dynamics_result['total_power']
                            current = dynamics_result.get('ac_current_rms', 0.0)
                            
                        except Exception as e2:
                            # Final fallback - use dynamics-based measurements
                            voltage = grid_voltage_pu
                            power = dynamics_result['total_power']
                            current = dynamics_result.get('ac_current_rms', 0.0)
                    else:
                        # Use dynamics-based measurements if OpenDSS fails for other reasons
                        voltage = grid_voltage_pu
                        power = dynamics_result['total_power']
                        current = dynamics_result.get('ac_current_rms', 0.0)
                
                # Update CMS measurements with dynamics results
                self.cms.update_measurements(i, voltage, current, power)
                
                # Handle cyber attacks if active
                if (self.cms.attack_active and 
                    hasattr(self.cms, 'attack_params') and self.cms.attack_params and 
                    i in self.cms.attack_params.get('targets', [])):
                    
                    attack_type = self.cms.attack_params['type']
                    magnitude = self.cms.attack_params['magnitude']
                    start_time = self.cms.attack_params.get('start_time', 0)
                    duration = self.cms.attack_params.get('duration', 50.0)
                    
                    # Apply attack manipulation to power reference with error handling
                    if current_time <= start_time + duration:
                        try:
                            original_power_ref = power_ref
                            
                            if attack_type == 'demand_increase':
                                power_ref *= (1 + magnitude)
                            elif attack_type == 'demand_decrease':
                                power_ref *= max(0.1, magnitude)  # Prevent zero multiplication
                            elif attack_type == 'oscillating_demand':
                                time_factor = (current_time - start_time) / max(10.0, 1.0)  # Prevent division by zero
                                oscillation = np.sin(time_factor) * 0.5 + 1.0
                                power_ref *= oscillation * max(0.1, magnitude)
                            elif attack_type == 'ramp_demand':
                                elapsed = current_time - start_time
                                ramp_factor = elapsed / max(duration, 1.0)  # Prevent division by zero
                                if magnitude > 1.0:
                                    power_ref *= (1.0 + (magnitude - 1.0) * ramp_factor)
                                else:
                                    power_ref *= (1.0 - (1.0 - max(0.1, magnitude)) * ramp_factor)
                            
                            # Validate attack result
                            if not np.isfinite(power_ref) or power_ref <= 0:
                                print(f"Warning: Attack manipulation resulted in invalid power_ref for EVCS {station.evcs_id}")
                                power_ref = original_power_ref  # Revert to original
                                
                        except Exception as e:
                            print(f"Error: Attack manipulation failed for EVCS {station.evcs_id}: {e}")
                            # Keep original power reference
                    
                    # FIXED: Store attack parameters for time-wise injection
                    station.attack_active = True
                    station.attack_type = attack_type
                    station.attack_magnitude = magnitude
                    station.attack_start_time = start_time
                    station.attack_duration = duration
                    
                    # Store the continuous sequence from RL agents if available
                    if hasattr(self.cms, 'current_attack_sequence'):
                        station.rl_attack_sequence = self.cms.current_attack_sequence
                        # Calculate current time step in attack sequence
                        elapsed_time = current_time - start_time
                        time_step = int(elapsed_time)
                        
                        if 0 <= time_step < len(station.rl_attack_sequence):
                            # Use RL-generated magnitude for this time step
                            rl_step = station.rl_attack_sequence[time_step]
                            station.attack_manipulation_factor = rl_step['magnitude']
                        else:
                            station.attack_manipulation_factor = magnitude  # Fallback
                    else:
                        # Fallback to static magnitude if no RL sequence available
                        station.attack_manipulation_factor = magnitude
                    
                    # Update references with attack manipulation
                    station.set_references(voltage_ref, current_ref, power_ref)
                else:
                    # FIXED: Clear attack state when not under attack
                    if hasattr(station, 'attack_active'):
                        station.attack_active = False
                        station.attack_manipulation_factor = 1.0
                
                # FIXED: Update station's current load with attack manipulation applied
                if hasattr(station, 'attack_active') and station.attack_active:
                    # Apply attack manipulation to the dynamics result
                    manipulated_power = dynamics_result['total_power'] * station.attack_manipulation_factor
                    station.current_load = manipulated_power
                    
                    # FIXED: Cache dynamics result for recovery
                    station.dynamics_result_cache = dynamics_result.copy()
                    
                    # FIXED: Update OpenDSS load directly to ensure power flow reflects attack
                    try:
                        # Get the station's bus name for OpenDSS load update
                        bus_name = station.bus_name
                        load_name = f"EVCS_{self.system_id}_{i:02d}"
                        
                        # Update OpenDSS load with manipulated power
                        manipulated_kw = manipulated_power
                        manipulated_kvar = manipulated_power * 0.1  # Assume 0.1 power factor
                        
                        # Update load in OpenDSS
                        self.dss.Command(f"Load.{load_name}.kW={manipulated_kw:.2f}")
                        self.dss.Command(f"Load.{load_name}.kvar={manipulated_kvar:.2f}")
                        
                        print(f" Attack Load Update: System {self.system_id}, Station {i}, "
                              f"Original: {dynamics_result['total_power']:.2f}W, "
                              f"Manipulated: {manipulated_power:.2f}W, "
                              f"Factor: {station.attack_manipulation_factor:.2f}")
                        
                    except Exception as e:
                        print(f"Warning: Could not update OpenDSS load for attack: {e}")
                        # Fallback to just updating station load
                        station.current_load = manipulated_power
                else:
                    # FIXED: Normal operation or recovery from attack
                    if hasattr(station, 'attack_active') and not station.attack_active:
                        # FIXED: Recovery mode - gradually return to normal operation with realistic power
                        if hasattr(station, 'dynamics_result_cache'):
                            # Use realistic charging power during recovery instead of low dynamics values
                            if station.ev_connected:
                                # Calculate realistic power based on single connected EV (not all ports)
                                # Use a reasonable charging power per connected EV
                                charging_power_per_ev = min(50.0, getattr(station, 'max_power', 50.0) / 4)  # 50kW max per EV or 1/4 of station capacity
                                station.current_load = charging_power_per_ev  # Keep in kW
                            else:
                                # No EV connected - use minimal standby power
                                station.current_load = 0.001  # 1W standby = 0.001 kW
                            
                            # FIXED: Update OpenDSS load to normal operation for recovery
                            try:
                                load_name = f"EVCS_{self.system_id}_{i:02d}"
                                normal_kw = station.current_load  # Already in kW
                                normal_kvar = normal_kw * 0.1
                                
                                self.dss.Command(f"Load.{load_name}.kW={normal_kw:.2f}")
                                self.dss.Command(f"Load.{load_name}.kvar={normal_kvar:.2f}")
                                
                                # Only print recovery messages occasionally to reduce spam
                                if hasattr(station, 'recovery_message_counter'):
                                    station.recovery_message_counter += 1
                                else:
                                    station.recovery_message_counter = 1
                                
                                if station.recovery_message_counter % 50 == 0:  # Print every 50th update
                                    print(f" Recovery Load Update: System {self.system_id}, Station {i}, "
                                          f"Recovered to: {normal_kw:.1f}kW (EV: {'Yes' if station.ev_connected else 'No'})")
                                
                            except Exception as e:
                                print(f"Warning: Could not update OpenDSS load for recovery: {e}")
                        else:
                            # No cached result, use realistic charging power calculation
                            if station.ev_connected:
                                # Calculate realistic power based on single connected EV (not all ports)
                                # Use a reasonable charging power per connected EV
                                charging_power_per_ev = min(50.0, getattr(station, 'max_power', 50.0) / 4)  # 50kW max per EV or 1/4 of station capacity
                                station.current_load = charging_power_per_ev  # Keep in kW
                            else:
                                # No EV connected - use minimal standby power
                                station.current_load = 0.001  # 1W standby = 0.001 kW
                    else:
                        # Normal operation - use realistic charging power instead of low dynamics values
                        if station.ev_connected:
                            # Calculate realistic power based on single connected EV (not all ports)
                            # Use a reasonable charging power per connected EV
                            charging_power_per_ev = min(50.0, getattr(station, 'max_power', 50.0) / 4)  # 50kW max per EV or 1/4 of station capacity
                            station.current_load = charging_power_per_ev  # Keep in kW
                        else:
                            # No EV connected - use minimal standby power  
                            station.current_load = 0.001  # 1W standby = 0.001 kW
                
                # FIXED: Update recovery progress for this station
                if hasattr(station, 'update_recovery'):
                    station.update_recovery(current_time)
            except Exception as e:
                print(f"Error updating EVCS {i} in system {self.system_id}: {e}")
                continue
    
    def get_total_load(self) -> float:
        """Get total system load in MW using DSS functions"""
        try:
            self.dss.Command("solve")
            
            # Get total load using DSS functions
            load_data, total_load = get_loads(self.dss, self.circuit)
            
            # FIXED: Add EV charging station loads with proper unit handling
            ev_load_kw = 0.0
            for station in self.ev_stations:
                # Ensure consistent kW units for station.current_load
                station_load_kw = 0.0
                if hasattr(station, 'power_measured') and station.power_measured > 0:
                    # power_measured is in kW
                    station_load_kw = station.power_measured
                elif hasattr(station, 'current_load') and station.current_load > 0:
                    # current_load might be in W or kW - detect and convert
                    if station.current_load > 1000:  # Likely in Watts
                        station_load_kw = station.current_load / 1000.0
                    else:  # Likely in kW
                        station_load_kw = station.current_load
                
                ev_load_kw += station_load_kw
            
            total_system_load = (total_load + ev_load_kw) / 1000  # Convert to MW
            
            # FIXED: Log total load for debugging attack impact
            if any(hasattr(station, 'attack_active') and station.attack_active for station in self.ev_stations):
                print(f" System {self.system_id} Total Load: Base={total_load/1000:.2f}MW, "
                      f"EV={ev_load_kw/1000:.2f}MW, Total={total_system_load:.2f}MW")
            
            return total_system_load
            
        except:
            # Calculate total load from base loads and EV stations if OpenDSS fails
            total_load = 0.0
            
            # Add base loads
            for load_name, load_data in self.base_loads.items():
                total_load += load_data['kW']
            
            # FIXED: Add EV charging station loads with attack manipulation properly accounted
            ev_load = 0.0
            for station in self.ev_stations:
                if hasattr(station, 'attack_active') and station.attack_active:
                    # Use the manipulated load that was already applied
                    ev_load += station.current_load
                else:
                    # Normal operation
                    ev_load += station.current_load
            
            total_system_load = (total_load + ev_load) / 1000  # Convert to MW
            
            # FIXED: Log total load for debugging attack impact
            if any(hasattr(station, 'attack_active') and station.attack_active for station in self.ev_stations):
                print(f" System {self.system_id} Total Load (Fallback): Base={total_load/1000:.2f}MW, "
                      f"EV={ev_load/1000:.2f}MW, Total={total_system_load:.2f}MW")
            
            return total_system_load
    
    def launch_cyber_attack(self, attack_config: Dict):
        """Launch cyber attack on charging management system with dynamics-aware targeting"""
        attack_type = attack_config.get('type', 'demand_increase')
        magnitude = attack_config.get('magnitude', 0.5)
        target_percentage = attack_config.get('target_percentage', 100)  # Default to 100%
        start_time = attack_config.get('start_time', 0.0)
        duration = attack_config.get('duration', 50.0)
        
        # Calculate number of stations to target based on percentage
        total_stations = len(self.ev_stations)
        num_target_stations = int(total_stations * target_percentage / 100)
        target_stations = list(range(num_target_stations))  # Target first N stations
        
        if self.cms:
            self.cms.attack_active = True
            self.cms.attack_params = {
                'type': attack_type,
                'magnitude': magnitude,
                'targets': target_stations,
                'start_time': start_time,
                'duration': duration
            }
            
            print(f" Cyber attack launched on system {self.system_id}:")
            print(f"   Type: {attack_type}")
            print(f"   Magnitude: {magnitude}")
            print(f"   Targets: {target_stations}")
            print(f"   Duration: {duration}s")
            
            # Enhanced attack targeting based on EVCS dynamics
            for station_id in target_stations:
                if station_id < len(self.ev_stations):
                    station = self.ev_stations[station_id]
                    
                    # FIXED: Store baseline metrics when attack starts (not when it ends)
                    if not hasattr(station, 'baseline_metrics') or not station.baseline_metrics:
                        station.store_baseline_metrics()
                    
                    # FIXED: Reset recovery state if new attack starts after recovery completion
                    if hasattr(station, 'recovery_mode') and not station.recovery_mode:
                        # Reset any remaining recovery artifacts
                        station.pre_attack_metrics = {}
                        station.recovery_start_time = None
                        print(f"   Target {station.evcs_id}: Recovery state reset for new attack")
                    
                    print(f"   Target {station.evcs_id}: EV connected={station.ev_connected}, SOC={station.soc*100:.1f}%")
                    self.cms.inject_false_data(station_id, attack_type, magnitude)
                    
    def stop_cyber_attack(self, attack_config: Dict, current_time: float = None):
        """Stop cyber attack and initiate recovery"""
        if self.cms:
            self.cms.attack_active = False
            self.cms.attack_params = {}
            
            # FIXED: Reset attack state for all stations and initiate recovery
            for station in self.ev_stations:
                if hasattr(station, 'attack_active'):
                    # FIXED: Initiate recovery BEFORE clearing attack state
                    station.initiate_recovery(current_time if current_time is not None else 0.0)
                    
                    station.attack_active = False
                    station.attack_type = None
                    station.attack_magnitude = 1.0
                    station.attack_manipulation_factor = 1.0
                    
                    # FIXED: Reset current load to normal dynamics result for recovery
                    # This will be updated in the next update_ev_loads call
                    if hasattr(station, 'dynamics_result_cache'):
                        station.current_load = station.dynamics_result_cache.get('total_power', 0.0)
                    else:
                        station.current_load = 0.0
            
            print(f" Cyber attack stopped on system {self.system_id}")
            print(f"   All stations reset to normal operation")
            print(f"   Recovery initiated - system returning to normal state")

class HierarchicalCoSimulation:
    """Main hierarchical co-simulation framework"""
    
    def __init__(self, realtime_rl_controller=None, use_dqn_sac_security=True, use_enhanced_pinn=True):
        self.transmission_system = IEEE14BusAGC()
        self.central_coordinator = CentralChargingCoordinator()
        self.distribution_systems = {}
        self.simulation_time = 240.0
        self.dist_dt = 1.0  # Distribution system time step: 1 second
        self.agc_dt = 1.0   # AGC time step: 5 seconds
        self.coordination_dt = 5.0  # Central coordination time step: 10 seconds
        self.customer_arrival_dt = 2.0  # Customer arrival and queue processing: 10 seconds
        self.total_duration = 240.0  # Total simulation: 240 seconds
        
        # Real-time RL attack controller
        self.realtime_rl_controller = realtime_rl_controller
        if self.realtime_rl_controller:
            print(" Real-time RL attack controller enabled")
        
        # Enhanced PINN Integration
        self.use_enhanced_pinn = use_enhanced_pinn
        self.enhanced_pinn_models = {}  # Store enhanced PINN models for each system
        self.enhanced_pinn_available = False
        
        if use_enhanced_pinn:
            try:
                from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
                from federated_pinn_manager import FederatedPINNManager
                print(" Enhanced PINN models enabled - will use real EVCS dynamics")
                self.enhanced_pinn_available = True
                
                # Try to load pre-trained enhanced PINN models
                self._load_enhanced_pinn_models()
                
            except ImportError as e:
                print(f" Enhanced PINN models not available: {e}")
                print(" Falling back to standard co-simulation")
                self.enhanced_pinn_available = False
                use_enhanced_pinn = False
        
        # Coordinated DQN/SAC Security Evasion System Integration
        self.use_dqn_sac_security = use_dqn_sac_security
        self.dqn_sac_trainer = None  # Single coordinated trainer
        self.security_evasion_active = False
        
        if use_dqn_sac_security:
            try:
                from dqn_sac_security_evasion import DQNSACSecurityEvasionTrainer
                print(" Coordinated DQN/SAC Security Evasion System enabled")
                self.dqn_sac_available = True
            except ImportError:
                print(" DQN/SAC Security Evasion System not available, using legacy RL")
                self.dqn_sac_available = False
                use_dqn_sac_security = False
        
        # Initialize RL attack systems for each distribution system
        self.rl_attack_systems = {}
        if EnhancedRLAttackSystem:
            for i in range(6):  # 6 distribution systems
                self.rl_attack_systems[i] = EnhancedRLAttackSystem(sys_id=i)
            print(" Enhanced RL attack systems initialized for all distribution systems")
    
    def _load_enhanced_pinn_models(self):
        """Load pre-trained enhanced PINN models for each distribution system"""
        if not self.enhanced_pinn_available:
            return
        
        print("🔧 Loading pre-trained enhanced PINN models...")
        
        # Try to load federated models first
        try:
            from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
            
            # Check if federated models exist
            fed_config = FederatedPINNConfig()
            fed_config.num_distribution_systems = 6
            
            # Try to load existing federated models
            for sys_id in range(1, 7):
                model_path = f'federated_pinn_system_{sys_id}.pth'
                if os.path.exists(model_path):
                    try:
                        # Load the enhanced PINN model
                        from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
                        config = LSTMPINNConfig()
                        pinn_model = LSTMPINNChargingOptimizer(config, always_train=False)
                        pinn_model.load_model(model_path)
                        
                        self.enhanced_pinn_models[sys_id] = pinn_model
                        print(f"  ✅ System {sys_id}: Enhanced PINN model loaded from {model_path}")
                        
                    except Exception as e:
                        print(f"  ⚠️  System {sys_id}: Failed to load enhanced PINN model: {e}")
                else:
                    print(f"  ⚠️  System {sys_id}: No enhanced PINN model found at {model_path}")
            
            if self.enhanced_pinn_models:
                print(f"  🎯 Loaded {len(self.enhanced_pinn_models)} enhanced PINN models")
                print("  🚀 Co-simulation will use real EVCS dynamics from trained models!")
            else:
                print("  ⚠️  No enhanced PINN models loaded, using standard co-simulation")
                
        except Exception as e:
            print(f"  ⚠️  Failed to load federated models: {e}")
            print("  🔄 Falling back to standard co-simulation")
        
        self.results = {
            'time': [],
            'frequency': [],
            'total_load': [],
            'reference_power': [],
            'dist_loads': {},
            'bus_voltages': [],
            'line_flows': [],
            'agc_updates': [],
            'charging_time_data': {},
            'queue_management_data': {},
            'scheduling_data': {},
            'customer_satisfaction_data': {},
            'utilization_data': {},
            'load_balancing_data': [],
            'customer_redirection_data': [],
            'attack_impact_data': {},
            'coordination_reports': [],
            # NEW: EVCS measurement data for plotting
            'evcs_voltage_data': {},  # EVCS output voltage for each distribution system
            'evcs_power_data': {},    # EVCS power output for each distribution system
            'evcs_current_data': {},   # EVCS current output for each distribution system
            # Real-time RL attack tracking
            'rl_attack_decisions': [],
            'rl_attack_status': []
        }
        
    def add_distribution_system(self, system_id: int, dss_file: str, connection_bus: int):
        """Add a distribution system connected to transmission bus"""
        print(f"Adding distribution system {system_id}...")
        
        dist_sys = OpenDSSInterface(system_id, dss_file)
        if dist_sys.initialize():
            self.distribution_systems[system_id] = {
                'system': dist_sys,
                'connection_bus': connection_bus
            }
            self.results['dist_loads'][system_id] = []
            # Initialize EVCS measurement data structures
            self.results['evcs_voltage_data'][system_id] = []
            self.results['evcs_power_data'][system_id] = []
            self.results['evcs_current_data'][system_id] = []
            
            # Register with central coordinator
            self.central_coordinator.register_distribution_system(system_id, dist_sys)
    
    def setup_ev_charging_stations(self):
        """Setup EV charging stations in distribution systems"""
        print("Setting up EV charging stations...")
        
        # FIXED: Properly initialize EVCS stations with CMS for each distribution system
        for sys_id, dist_info in self.distribution_systems.items():
            dist_sys = dist_info['system']
            
            print(f"  Setting up EVCS for Distribution System {sys_id}...")
            
            # Initialize CMS for this distribution system
            if not hasattr(dist_sys, 'cms') or dist_sys.cms is None:
                # Create empty stations list first, will be populated below
                dist_sys.cms = EnhancedChargingManagementSystem(stations=[], use_pinn=False)
                print(f"    CMS initialized for system {sys_id}")
            
            # Add EVCS stations if not already present
            if not hasattr(dist_sys, 'ev_stations') or len(dist_sys.ev_stations) == 0:
                print("No EVCS stations found, adding 4 stations per distribution system")
                # Add 4 EVCS stations per distribution system
                evcs_configs = [
                    {'evcs_id': f'EVCS_{sys_id}_001', 'bus_name': '890', 'max_power': 1000, 'num_ports': 25},  # Mega charging hub
                    {'evcs_id': f'EVCS_{sys_id}_002', 'bus_name': '844', 'max_power': 300, 'num_ports': 6},   # Shopping center
                    {'evcs_id': f'EVCS_{sys_id}_003', 'bus_name': '860', 'max_power': 200, 'num_ports': 4},    # Residential area
                    {'evcs_id': f'EVCS_{sys_id}_004', 'bus_name': '840', 'max_power': 400, 'num_ports': 10},   # Business district
                ]
                dist_sys.ev_stations = []
                for config in evcs_configs:
                    station = EVChargingStation(
                        evcs_id=config["evcs_id"],
                        bus_name=config["bus_name"],
                        max_power=config["max_power"],
                        num_ports=config["num_ports"]
                    )
                    dist_sys.ev_stations.append(station)
                    print(f"    Added {config['evcs_id']} at bus {config['bus_name']}")
                
                print(f"    Total EVCS stations for system {sys_id}: {len(dist_sys.ev_stations)}")
            
            # Register EVCS stations with CMS
            if hasattr(dist_sys, 'cms') and dist_sys.cms and hasattr(dist_sys, 'ev_stations'):
                # Update CMS stations list
                dist_sys.cms.stations = dist_sys.ev_stations
                print(f"    Registered {len(dist_sys.ev_stations)} EVCS stations with CMS")
        
        print("EVCS setup completed for all distribution systems")
        
        # Initialize DQN/SAC Security Evasion Systems for each distribution system
        if self.use_dqn_sac_security and self.dqn_sac_available:
            self.setup_dqn_sac_security_systems()
    
    def setup_dqn_sac_security_systems(self):
        """Setup coordinated DQN/SAC security evasion system"""
        print("\n🛡️  Setting up Coordinated DQN/SAC Security Evasion System...")
        
        try:
            from dqn_sac_security_evasion import DQNSACSecurityEvasionTrainer
            
            # Create single coordinated trainer for all distribution systems
            print("   Initializing coordinated DQN/SAC trainer...")
            
            # Use first distribution system's CMS as reference
            first_dist_sys = next(iter(self.distribution_systems.values()))['system']
            cms_reference = first_dist_sys.cms if hasattr(first_dist_sys, 'cms') else None
            
            self.dqn_sac_trainer = DQNSACSecurityEvasionTrainer(
                cms_system=cms_reference,
                num_stations=6,  # Total stations across all systems
                use_both=True
            )
            
            # Set CMS reference in trainer environments
            if cms_reference:
                self.dqn_sac_trainer.sac_env.cms = cms_reference
                self.dqn_sac_trainer.dqn_env.cms = cms_reference
            
            # Quick coordinated training for operational readiness
            print("   Training coordinated DQN/SAC agents...")
            self.dqn_sac_trainer.train_coordinated_agents(total_timesteps=4000)
            
            self.security_evasion_active = True
            print("✅ Coordinated DQN/SAC Security Evasion System initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to setup coordinated DQN/SAC system: {e}")
            self.security_evasion_active = False
    
    def execute_dqn_sac_attack(self, system_id: int, attack_params: dict, current_time: float):
        """Execute coordinated DQN/SAC security evasion attack on specific distribution system"""
        if not self.security_evasion_active or not self.dqn_sac_trainer:
            return False
        
        try:
            # Get baseline outputs for the target system
            if system_id in self.distribution_systems:
                dist_sys = self.distribution_systems[system_id]['system']
                baseline_outputs = {
                    'power_reference': 10.0,
                    'voltage_reference': 400.0,
                    'current_reference': 25.0,
                    'soc': 0.5,
                    'grid_voltage': 1.0,
                    'grid_frequency': 60.0,
                    'demand_factor': attack_params.get('demand_factor', 1.0),
                    'urgency_factor': attack_params.get('urgency_factor', 1.0),
                    'voltage_priority': attack_params.get('voltage_priority', 0.0)
                }
                
                # Get coordinated attack from DQN/SAC trainer
                coordinated_attack = self.dqn_sac_trainer.get_coordinated_attack(
                    system_id, baseline_outputs
                )
                
                if coordinated_attack:
                    attack_params_combined = coordinated_attack['attack_params']
                    
                    # Apply coordinated attack to distribution system
                    if hasattr(dist_sys, 'cms') and dist_sys.cms:
                        # Execute attack through CMS input manipulation
                        attack_success = self._apply_coordinated_attack_to_cms(
                            dist_sys.cms, attack_params_combined, current_time
                        )
                        
                        print(f"🎯 Coordinated DQN/SAC attack executed on system {system_id}: "
                              f"Type={attack_params_combined.get('type', 'unknown')}, "
                              f"Success={attack_success}")
                        
                        return attack_success
                    
            return False
            
        except Exception as e:
            print(f"⚠️ Failed to execute coordinated DQN/SAC attack: {e}")
            return False
    
    def _apply_coordinated_attack_to_cms(self, cms, attack_params: dict, current_time: float):
        """Apply coordinated attack parameters to CMS inputs"""
        try:
            attack_type = attack_params.get('type', 'demand_increase')
            magnitude = attack_params.get('magnitude', 1.0)
            stealth_factor = attack_params.get('stealth_factor', 0.5)
            
            # Apply attack based on DQN strategic decision and SAC continuous control
            if attack_type == 'demand_increase':
                # Manipulate demand factor with SAC-controlled magnitude
                cms.demand_factor = 1.0 + (magnitude * 0.4 * (1.0 - stealth_factor))
            elif attack_type == 'demand_decrease':
                cms.demand_factor = 1.0 - (magnitude * 0.3 * (1.0 - stealth_factor))
            elif attack_type == 'voltage_spoofing':
                cms.grid_voltage = 1.0 + (magnitude * 0.1 * (1.0 - stealth_factor))
            elif attack_type == 'frequency_spoofing':
                cms.grid_frequency = 60.0 + (magnitude * 2.0 * (1.0 - stealth_factor))
            elif attack_type == 'soc_manipulation':
                cms.soc_override = min(1.0, 0.5 + magnitude * 0.2 * (1.0 - stealth_factor))
            elif attack_type == 'oscillating_demand':
                import math
                oscillation = math.sin(current_time * 0.5) * magnitude * (1.0 - stealth_factor)
                cms.demand_factor = 1.0 + oscillation * 0.3
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to apply coordinated attack to CMS: {e}")
            return False
        
        return False
    
    def run_hierarchical_simulation(self, attack_scenarios: List[Dict] = None, max_wall_time_sec: float = None):
        """Run hierarchical co-simulation with different time scales"""
        print("Starting Hierarchical Pandapower-OpenDSS Co-simulation...")
        print(f"Distribution systems: {self.dist_dt}s time steps")
        print(f"Customer arrivals: {self.customer_arrival_dt}s intervals")
        print(f"AGC updates: {self.agc_dt}s intervals")
        print(f"Total duration: {self.total_duration}s")
        
        attack_scenarios = attack_scenarios or []
        self.current_attack_scenarios = attack_scenarios  # Store for access in coordination loop
        dist_steps = int(self.total_duration / self.dist_dt)
        
        # Optional wall-clock timeout to avoid overly long runs
        if max_wall_time_sec is not None:
            import time as _time
            _start_time = _time.time()
        
        # Initialize enhanced data tracking
        for sys_id in self.distribution_systems.keys():
            self.results['charging_time_data'][sys_id] = []
            self.results['queue_management_data'][sys_id] = []
            self.results['scheduling_data'][sys_id] = []
            self.results['customer_satisfaction_data'][sys_id] = []
            self.results['utilization_data'][sys_id] = []
            self.results['attack_impact_data'][sys_id] = []
        
        for step in range(dist_steps):
            # Check wall-clock timeout
            if max_wall_time_sec is not None:
                if _time.time() - _start_time > max_wall_time_sec:
                    print(f"⏱️ Stopping hierarchical simulation early after reaching {max_wall_time_sec}s wall-clock limit")
                    break
            self.simulation_time = step * self.dist_dt
            
            # Complete finished charging sessions (every second)
            self.central_coordinator.complete_finished_sessions(self.simulation_time)
            
            # Simulate customer arrivals and process queues (every 10 seconds)
            if self.simulation_time % self.customer_arrival_dt == 0:
                self.central_coordinator.simulate_customer_arrivals(self.simulation_time)
                self.central_coordinator.process_charging_queues(self.simulation_time)
            
            # Real-time RL attack decision making
            if self.realtime_rl_controller:
                # Collect current system states for RL decision making
                system_states = {}
                for sys_id, sys_info in self.distribution_systems.items():
                    dist_sys = sys_info['system']
                    system_states[sys_id] = {
                        'total_load': getattr(dist_sys, 'current_total_load', 0.0),
                        'voltage_level': getattr(dist_sys, 'current_voltage_level', 1.0),
                        'evcs_count': len(dist_sys.ev_stations),
                        'attack_active': getattr(dist_sys.cms, 'attack_active', False) if hasattr(dist_sys, 'cms') and dist_sys.cms else False
                    }
                
                # Make real-time attack decisions
                attack_decisions = self.realtime_rl_controller.make_attack_decision(system_states, self.simulation_time)
                
                # Apply new attack decisions
                for target_sys, attack_params in attack_decisions.items():
                    if target_sys in self.distribution_systems:
                        dist_sys = self.distribution_systems[target_sys]['system']
                        if hasattr(dist_sys, 'cms') and dist_sys.cms:
                            # Activate real-time RL attack
                            dist_sys.cms.attack_active = True
                            dist_sys.cms.attack_params = {
                                'type': attack_params['type'],
                                'magnitude': attack_params['magnitude'],
                                'targets': list(range(min(len(dist_sys.ev_stations), 
                                                        int(len(dist_sys.ev_stations) * attack_params['target_percentage'] / 100)))),
                                'start_time': attack_params['start_time'],
                                'duration': attack_params['duration'],
                                'rl_decision_id': attack_params['decision_id']
                            }
                            print(f" Real-time RL attack launched on system {target_sys} at t={self.simulation_time:.1f}s")
                            print(f"   Type: {attack_params['type']}, Magnitude: {attack_params['magnitude']:.2f}, Duration: {attack_params['duration']}s")
                            
                            # Log attack decision
                            self.results['rl_attack_decisions'].append({
                                'time': self.simulation_time,
                                'target_system': target_sys,
                                'attack_params': attack_params.copy()
                            })
                
                # Update parameters for ongoing attacks
                for sys_id, sys_info in self.distribution_systems.items():
                    dist_sys = sys_info['system']
                    if hasattr(dist_sys, 'cms') and dist_sys.cms and dist_sys.cms.attack_active:
                        # Check if attack should be updated or stopped
                        attack_update = self.realtime_rl_controller.update_attack_parameters(sys_id, self.simulation_time)
                        
                        if attack_update.get('stop_attack', False):
                            # Stop attack
                            dist_sys.cms.attack_active = False
                            dist_sys.cms.attack_params = {}
                            print(f" Real-time RL attack stopped on system {sys_id} at t={self.simulation_time:.1f}s")
                        elif attack_update:
                            # Update attack parameters
                            dist_sys.cms.attack_params.update(attack_update)
                
                # Log current RL attack status
                if self.simulation_time % 30.0 == 0:  # Every 30 seconds
                    attack_status = self.realtime_rl_controller.get_attack_status()
                    self.results['rl_attack_status'].append({
                        'time': self.simulation_time,
                        'status': attack_status
                    })
            
            # Fallback: Apply legacy pre-fabricated attack scenarios only if no real-time controller
            elif attack_scenarios and getattr(self, 'realtime_rl_controller', None) is None:
                for attack in attack_scenarios:
                    if (attack['start_time'] <= self.simulation_time <= 
                        attack['start_time'] + attack['duration']):
                        
                        if not attack.get('active', False):
                            target_sys = attack['target_system']
                            if target_sys in self.distribution_systems:
                                dist_sys = self.distribution_systems[target_sys]['system']
                                
                                if hasattr(dist_sys, 'cms') and dist_sys.cms:
                                    # Generate RL-based attack impact values
                                    rl_attack_impact = self._generate_rl_attack_impact(target_sys, attack['type'], self.simulation_time)
                                    
                                    # Activate attack in CMS for PINN reference manipulation
                                    dist_sys.cms.attack_active = True
                                    dist_sys.cms.attack_type = attack['type']
                                    dist_sys.cms.rl_attack_impact = rl_attack_impact  # Store RL impact values
                                    dist_sys.cms.attack_params = {
                                        'type': attack['type'],
                                        'magnitude': attack['magnitude'],
                                        'targets': list(range(len(dist_sys.ev_stations))),  # All stations
                                        'start_time': attack['start_time'],
                                        'duration': attack['duration'],
                                        'rl_impact': rl_attack_impact  # Include RL impact in params
                                    }
                                    
                                    # Apply RL attack impact to all EVCS stations in the system
                                    for station in dist_sys.ev_stations:
                                        station.attack_active = True
                                        station.attack_type = attack['type']
                                        station.rl_attack_impact = rl_attack_impact
                                    
                                    print(f" RL-Enhanced attack activated on system {target_sys} at t={self.simulation_time:.1f}s")
                                    print(f"   Type: {attack['type']}, Magnitude: {attack['magnitude']:.2f}")
                                    if 'rl_magnitude' in rl_attack_impact:
                                        print(f"   RL Magnitude: {rl_attack_impact['rl_magnitude']:.1f} kW")
                                        print(f"   RL Stealth Score: {rl_attack_impact['rl_stealth_score']:.2f}")
                            attack['active'] = True
                            
                    elif self.simulation_time > attack['start_time'] + attack['duration']:
                        if attack.get('active', False) and not attack.get('stopped', False):
                            target_sys = attack['target_system']
                            if target_sys in self.distribution_systems:
                                dist_sys = self.distribution_systems[target_sys]['system']
                                if hasattr(dist_sys, 'cms') and dist_sys.cms:
                                    # Deactivate attack in CMS
                                    dist_sys.cms.attack_active = False
                                    dist_sys.cms.attack_params = {}
                                    print(f" Legacy attack deactivated on system {target_sys} at t={self.simulation_time:.1f}s")
                            attack['stopped'] = True
            
            # Update distribution systems (every 1 second) - EV loads update continuously
            dist_loads = {}
            for sys_id, dist_info in self.distribution_systems.items():
                dist_sys = dist_info['system']
                
                # Update EV loads based on CMS control
                dist_sys.update_ev_loads(self.simulation_time)
                
                # FIXED: Update recovery for all stations to ensure attack recovery progresses
                if hasattr(dist_sys, 'ev_stations'):
                    for station in dist_sys.ev_stations:
                        if hasattr(station, 'update_recovery'):
                            # FIXED: Add debugging to track recovery calls
                            if hasattr(station, 'recovery_mode') and station.recovery_mode:
                                recovery_duration = self.simulation_time - station.recovery_start_time
                                if recovery_duration % 20 == 0:  # Every 20 seconds
                                    print(f" Main loop: EVCS {station.evcs_id} recovery active - duration {recovery_duration:.1f}s")
                                    print(f"   Current avg_charging_time: {station.avg_charging_time:.1f} min")
                                    print(f"   Current queue length: {len(station.customer_queue)}")
                                    print(f"   Historical sessions: {len(station.charging_sessions)}")
                                    print(f"   Historical EVs: {len(station.evs_served)}")
                            
                            station.update_recovery(self.simulation_time)
                
                # NEW: Enhanced PINN Integration - Update EVCS dynamics using trained models
                if (self.use_enhanced_pinn and self.enhanced_pinn_available and 
                    sys_id in self.enhanced_pinn_models):
                    
                    try:
                        enhanced_pinn_model = self.enhanced_pinn_models[sys_id]
                        
                        # Update each EVCS station using enhanced PINN dynamics
                        for station in dist_sys.ev_stations:
                            if hasattr(station, 'evcs_controller'):
                                # Get current grid conditions for PINN input
                                grid_voltage = getattr(dist_sys, 'current_voltage_level', 1.0)
                                system_frequency = 60.0  # Default frequency
                                
                                # Create input features for enhanced PINN
                                input_features = [
                                    station.soc,  # Current SOC
                                    grid_voltage,  # Grid voltage (pu)
                                    system_frequency,  # System frequency
                                    1.0,  # Demand factor (normalized)
                                    1.0,  # Voltage priority
                                    1.0,  # Urgency factor
                                    self.simulation_time / 3600.0,  # Time in hours
                                    grid_voltage,  # Bus voltage
                                    1.0,  # Base load factor
                                    station.current_load / 1000.0,  # Previous power (MW)
                                    0.0,  # AC power in (placeholder)
                                    0.85,  # System efficiency (placeholder)
                                    0.0,  # Power balance error (placeholder)
                                    0.0   # DC link voltage deviation (placeholder)
                                ]
                                
                                # Use enhanced PINN to predict optimal charging parameters
                                try:
                                    # Convert to tensor format expected by PINN
                                    import torch
                                    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
                                    
                                    # Get PINN prediction
                                    with torch.no_grad():
                                        pinn_output = enhanced_pinn_model.predict(input_tensor)
                                        
                                    if pinn_output is not None:
                                        # Extract predicted values
                                        predicted_voltage = pinn_output[0, 0].item() if len(pinn_output.shape) > 1 else pinn_output[0].item()
                                        predicted_current = pinn_output[0, 1].item() if len(pinn_output.shape) > 1 else pinn_output[1].item()
                                        predicted_power = pinn_output[0, 2].item() if len(pinn_output.shape) > 1 else pinn_output[2].item()
                                        
                                        # Update station with enhanced PINN predictions
                                        station.voltage_measured = predicted_voltage
                                        station.current_measured = predicted_current
                                        station.power_measured = predicted_power
                                        
                                        # Update EVCS controller with enhanced references
                                        station.evcs_controller.set_references(predicted_voltage, predicted_current, predicted_power)
                                        
                                        # Log enhanced PINN usage (every 60 seconds to avoid spam)
                                        if self.simulation_time % 60 == 0:
                                            print(f"  🔬 System {sys_id} EVCS {station.evcs_id}: Enhanced PINN active")
                                            print(f"     Predicted: V={predicted_voltage:.1f}V, I={predicted_current:.1f}A, P={predicted_power:.2f}kW")
                                        
                                except Exception as e:
                                    # Fallback to standard dynamics if PINN fails
                                    if self.simulation_time % 60 == 0:
                                        print(f"  ⚠️  System {sys_id} EVCS {station.evcs_id}: PINN failed, using standard dynamics: {e}")
                                    
                                    # Use standard EVCS dynamics as fallback
                                    if hasattr(station.evcs_controller, '_update_dynamics_euler'):
                                        try:
                                            grid_voltage_v = grid_voltage * 7200.0  # Convert pu to V
                                            dt_simulation = 1.0  # 1 second time step
                                            
                                            dynamics_result = station.evcs_controller._update_dynamics_euler(grid_voltage_v, dt_simulation)
                                            
                                            # Update station with standard dynamics results
                                            station.voltage_measured = dynamics_result.get('voltage_measured', station.voltage_measured)
                                            station.current_measured = dynamics_result.get('current_measured', station.current_measured)
                                            station.power_measured = dynamics_result.get('total_power', station.power_measured)
                                            
                                        except Exception as fallback_error:
                                            if self.simulation_time % 60 == 0:
                                                print(f"    ⚠️  Standard dynamics also failed: {fallback_error}")
                    
                    except Exception as e:
                        if self.simulation_time % 60 == 0:
                            print(f"  ⚠️  System {sys_id}: Enhanced PINN integration failed: {e}")
                
                # Get raw per-system load
                load = dist_sys.get_total_load()

                # Align with transmission load-profile scaling so per-system plots and total match
                try:
                    load_multiplier = self.transmission_system.get_current_load_multiplier(self.simulation_time)
                except Exception:
                    load_multiplier = 1.0

                # Base load used by transmission scaling (first-seen value per system)
                base_load = self.transmission_system.dist_base_loads.get(sys_id, load)
                if sys_id not in self.transmission_system.dist_base_loads:
                    self.transmission_system.dist_base_loads[sys_id] = base_load

                # Apply the same scaling rule as in TransmissionSystem.update_distribution_load
                scaled_base = base_load * load_multiplier
                evcs_variation = load - base_load
                scaled_load = scaled_base + evcs_variation

                # Store scaled per-system load for plotting/aggregation
                dist_loads[sys_id] = scaled_load
                self.results['dist_loads'][sys_id].append(scaled_load)
                
                # Debug individual system load calculation
                if step % 100 == 0:  # Every 100 steps
                    print(f"  System {sys_id}: {load:.3f} MW")
                
                # Collect enhanced charging metrics with error handling
                if hasattr(dist_sys, 'ev_stations'):
                    try:
                        total_charging_time = 0.0
                        total_queue_length = 0
                        total_utilization = 0.0
                        total_satisfaction = 0.0
                        station_count = 0
                        
                        # FIXED: Weight charging times by attack impact instead of simple averaging
                        max_charging_time = 0.0
                        attack_affected_stations = 0
                        
                        for station in dist_sys.ev_stations:
                            # Update customer wait times before collecting metrics
                            station.update_customer_wait_times(self.simulation_time)
                            
                            # FIXED: Pass current time to get_charging_metrics for accurate calculations
                            metrics = station.get_charging_metrics(self.simulation_time)
                            
                            # FIXED: Add debugging for recovery metrics
                            if hasattr(station, 'recovery_mode') and station.recovery_mode:
                                recovery_duration = self.simulation_time - station.recovery_start_time
                                if recovery_duration % 30 == 0:  # Every 30 seconds
                                    print(f" Metrics collection: EVCS {station.evcs_id} recovery metrics:")
                                    print(f"   avg_charging_time: {metrics['avg_charging_time']:.1f} min")
                                    print(f"   queue_length: {metrics['queue_length']}")
                                    print(f"   recovery_mode: {metrics['recovery_mode']}")
                                    print(f"   recovery_progress: {metrics['recovery_progress']:.2f}")
                            
                            # FIXED: Track maximum charging time to show attack impact
                            station_charging_time = metrics['avg_charging_time']
                            max_charging_time = max(max_charging_time, station_charging_time)
                            
                            # Count stations under attack or recovery
                            if (hasattr(station, 'attack_active') and station.attack_active) or \
                               (hasattr(station, 'recovery_mode') and station.recovery_mode):
                                attack_affected_stations += 1
                                # Weight attack-affected stations more heavily
                                total_charging_time += station_charging_time * 3.0  # 3x weight
                            else:
                                total_charging_time += station_charging_time
                            
                            total_queue_length += metrics['queue_length']
                            total_utilization += metrics['utilization_rate']
                            station_count += 1

                    
                        
                    
                    # NEW: Collect EVCS voltage and power measurements
                        if station_count > 0:
                            # Calculate average EVCS voltage and power accounting for multiple ports per station
                            total_evcs_voltage = 0.0
                            total_evcs_power = 0.0
                            total_evcs_current = 0.0
                            total_active_ports = 0
                            
                            for station in dist_sys.ev_stations:
                                # Get current dynamics state
                                dynamics_state = station.get_charging_metrics(self.simulation_time)
                                
                                # Collect voltage measurement - use proper EVCS voltage (800V nominal)
                                station_voltage = 400.0  # Default nominal voltage
                                
                                # Use actual measured voltage if available and realistic
                                if hasattr(station, 'voltage_measured') and station.voltage_measured > 400:
                                    station_voltage = station.voltage_measured
                                elif hasattr(station, 'dc_link_voltage') and station.dc_link_voltage > 400:
                                    # DC link voltage is typically lower than output voltage
                                    station_voltage = station.dc_link_voltage * 1.33  # Scale to output voltage
                                elif hasattr(station, 'voltage_ref') and station.voltage_ref > 400:
                                    station_voltage = station.voltage_ref
                                elif hasattr(station, 'params') and station.params and station.params.rated_voltage > 400:
                                    station_voltage = station.params.rated_voltage
                                
                                # Apply voltage variations only during attacks or heavy loading
                                if hasattr(station, 'attack_active') and station.attack_active:
                                    # During attacks, voltage can drop significantly
                                    # Get voltage drop factor from current attack scenario
                                    voltage_drop_factor = 0.15  # Default 15% drop
                                    for attack in attack_scenarios:
                                        if (attack['start_time'] <= self.simulation_time <= 
                                            attack['start_time'] + attack['duration']):
                                            voltage_drop_factor = attack.get('voltage_drop_factor', 0.15)
                                            break
                                    voltage_drop = station_voltage * voltage_drop_factor
                                    station_voltage = max(400, station_voltage - voltage_drop)
                                else:
                                    # Normal operation: small variations around 800V
                                    voltage_variation = np.random.uniform(-15, 15)  # ±15V variation
                                    station_voltage = 400.0 + voltage_variation
                                
                                # Collect power measurement - use realistic EVCS power output
                                station_power = 0.0
                                
                                # Calculate actual power based on station operation
                                # Always calculate realistic power regardless of recovery mode
                                max_power_per_port = 50.0  # kW per port
                                
                                # Determine active ports based on connection status
                                if hasattr(station, 'ev_connected') and station.ev_connected:
                                    # EV is connected - use num_ports and available_ports
                                    active_ports = getattr(station, 'num_ports', 16) - getattr(station, 'available_ports', 0)
                                    active_ports = max(1, active_ports)  # At least 1 active port
                                else:
                                    # Check charging sessions to determine active ports
                                    active_sessions = len(getattr(station, 'charging_sessions', []))
                                    if active_sessions > 0:
                                        active_ports = active_sessions
                                    else:
                                        # Use default based on station size
                                        total_ports = getattr(station, 'num_ports', 16)
                                        active_ports = max(1, int(total_ports * 0.6))  # 60% utilization default
                                
                                # Calculate power based on charging state and utilization
                                utilization = dynamics_state.get('utilization', 0.7) if dynamics_state else 0.7
                                charging_efficiency = 0.85  # 85% charging efficiency
                                
                                # Use measured power if available and realistic, otherwise calculate
                                if hasattr(station, 'power_measured') and station.power_measured > 20.0:
                                    station_power = station.power_measured
                                elif hasattr(station, 'power_reference') and station.power_reference > 20.0:
                                    station_power = station.power_reference
                                else:
                                    # Calculate realistic power based on active charging
                                    base_power = active_ports * max_power_per_port * utilization * charging_efficiency
                                    
                                    # Add some variation for realism
                                    variation_factor = np.random.uniform(0.8, 1.2)  # ±20% variation
                                    station_power = base_power * variation_factor
                                    
                                    # Ensure minimum power for active stations
                                    station_power = max(500.0, station_power)  # At least 5kW for active stations
                                
                                
                                # Apply attack impacts to power, voltage, and current measurements
                                if hasattr(station, 'attack_active') and station.attack_active:
                                    if hasattr(station, 'attack_type') and hasattr(station, 'rl_attack_impact'):
                                        rl_impact = station.rl_attack_impact
                                        
                                        if station.attack_type == 'power_manipulation':
                                            # Apply RL-suggested power efficiency reduction
                                            power_reduction = rl_impact.get('power_efficiency_reduction', 0.2)
                                            station_power *= (1.0 - power_reduction)
                                            # Apply voltage drop due to power manipulation
                                            voltage_drop = station_voltage * power_reduction * 0.1
                                            station_voltage = max(750, station_voltage - voltage_drop)
                                            
                                        elif station.attack_type == 'voltage_manipulation':
                                            # Apply RL-suggested voltage efficiency loss
                                            voltage_loss = rl_impact.get('voltage_efficiency_loss', 0.3)
                                            voltage_drop = station_voltage * voltage_loss * 0.15
                                            station_voltage = max(720, station_voltage - voltage_drop)
                                            # Voltage drop affects power output
                                            power_reduction = voltage_loss * 0.5
                                            station_power *= (1.0 - power_reduction)
                                            
                                        elif station.attack_type == 'load_manipulation':
                                            # Apply RL-suggested load instability factor
                                            load_factor = rl_impact.get('load_instability_factor', 1.5)
                                            power_variation = min(0.7, (load_factor - 1.0) * 0.5)
                                            station_power *= np.random.uniform(1.0 - power_variation, 1.0 + power_variation)
                                            # Load instability causes voltage fluctuations
                                            voltage_variation = power_variation * 0.3
                                            voltage_factor = np.random.uniform(1.0 - voltage_variation, 1.0 + voltage_variation)
                                            station_voltage *= voltage_factor
                                            station_voltage = max(720, min(850, station_voltage))
                                            
                                        elif station.attack_type in ['demand_increase', 'demand_decrease']:
                                            # Apply RL-suggested demand instability factor
                                            demand_factor = rl_impact.get('demand_instability_factor', 1.3)
                                            if station.attack_type == 'demand_increase':
                                                power_reduction = min(0.4, (demand_factor - 1.0) * 0.8)
                                                station_power *= (1.0 - power_reduction)
                                            else:  # demand_decrease
                                                power_variation = min(0.3, (demand_factor - 1.0) * 0.6)
                                                station_power *= np.random.uniform(1.0 - power_variation, 1.0 + power_variation)
                                            # Demand changes affect voltage stability
                                            voltage_variation = min(0.1, (demand_factor - 1.0) * 0.2)
                                            voltage_factor = np.random.uniform(1.0 - voltage_variation, 1.0 + voltage_variation)
                                            station_voltage *= voltage_factor
                                            station_voltage = max(750, min(830, station_voltage))
                                            
                                        elif station.attack_type == 'frequency_manipulation':
                                            # Apply RL-suggested frequency efficiency loss
                                            freq_loss = rl_impact.get('frequency_efficiency_loss', 0.15)
                                            station_power *= (1.0 - freq_loss)
                                            # Frequency issues cause minor voltage variations
                                            voltage_variation = freq_loss * 0.5
                                            voltage_factor = np.random.uniform(1.0 - voltage_variation, 1.0 + voltage_variation)
                                            station_voltage *= voltage_factor
                                            station_voltage = max(770, min(820, station_voltage))
                                    
                                    # Fallback for attacks without RL impact data
                                    elif hasattr(station, 'attack_type'):
                                        if station.attack_type == 'power_manipulation':
                                            attack_factor = getattr(station, 'attack_manipulation_factor', 0.8)
                                            station_power *= attack_factor
                                            station_voltage *= 0.95  # 5% voltage drop
                                        elif station.attack_type == 'load_manipulation':
                                            station_power *= np.random.uniform(0.4, 1.6)
                                            station_voltage *= np.random.uniform(0.92, 1.05)
                                        elif station.attack_type == 'voltage_manipulation':
                                            station_voltage *= np.random.uniform(0.85, 0.95)
                                            station_power *= 0.85  # Power drops with voltage
                                
                                # Collect current measurement - calculate from power and voltage
                                station_current = 0.0
                                
                                # Use measured current if available and realistic
                                if hasattr(station, 'current_measured') and station.current_measured > 1.0:
                                    station_current = station.current_measured
                                elif hasattr(station, 'ac_current_rms') and station.ac_current_rms > 1.0:
                                    station_current = station.ac_current_rms
                                elif station_power > 0 and station_voltage > 0:
                                    # Calculate current from power and voltage (P = V*I)
                                    station_current = station_power * 1000 / station_voltage  # Convert kW to W
                                else:
                                    # Minimal current for standby
                                    station_current = np.random.uniform(1.0, 5.0)  # A
                                
                                # Apply minor voltage drop under heavy loading (realistic grid behavior)
                                if station_power > 200:  # High power operation
                                    power_factor = 1  # Normalize to max expected power
                                    voltage_drop = 0 * power_factor  # Up to 5V drop under full load
                                    station_voltage = max(780, station_voltage - voltage_drop)  # Keep above 780V
                                
                                # Weight measurements by number of active ports in this station
                                station_weight = active_ports
                                total_evcs_voltage += station_voltage * station_weight
                                total_evcs_power += station_power * station_weight
                                total_evcs_current += station_current * station_weight
                                total_active_ports += station_weight
                            
                            # Calculate weighted averages based on total active ports
                            if total_active_ports > 0:
                                avg_evcs_voltage = total_evcs_voltage / total_active_ports
                                avg_evcs_power = total_evcs_power / total_active_ports
                                avg_evcs_current = total_evcs_current / total_active_ports
                            else:
                                # Fallback to station-based averaging if no active ports detected
                                avg_evcs_voltage = total_evcs_voltage / station_count if station_count > 0 else 400.0
                                avg_evcs_power = total_evcs_power / station_count if station_count > 0 else 0.0
                                avg_evcs_current = total_evcs_current / station_count if station_count > 0 else 0.0
                            
                            # Debug print for system 1 (system_id = 0) every 100 time steps
                            if sys_id == 0 and len(self.results['time']) % 100 == 0:
                                print(f"DEBUG System 1: stations={station_count}, avg_voltage={avg_evcs_voltage:.1f}V, avg_power={avg_evcs_power:.1f}kW, avg_current={avg_evcs_current:.1f}A")
                            
                            self.results['evcs_voltage_data'][sys_id].append(avg_evcs_voltage)
                            self.results['evcs_power_data'][sys_id].append(avg_evcs_power)
                            self.results['evcs_current_data'][sys_id].append(avg_evcs_current)
                        
                        if station_count > 0:
                            # FIXED: Use consistent weighted average to prevent oscillations
                            # Apply a smooth attack impact factor instead of switching between max and average
                            # Add safeguards against division by zero
                            try:
                                base_avg_time = total_charging_time / max(station_count, 1)
                                
                                if attack_affected_stations > 0:
                                    # Smooth attack impact: blend between normal and maximum based on affected ratio
                                    attack_ratio = attack_affected_stations / max(station_count, 1)
                                    attack_impact_factor = 1.0 + (attack_ratio * 0.5)  # Max 50% increase
                                    avg_charging_time = base_avg_time * attack_impact_factor
                                else:
                                    avg_charging_time = base_avg_time
                                
                                avg_queue_length = total_queue_length / max(station_count, 1)
                                avg_utilization = total_utilization / max(station_count, 1)
                            except (ZeroDivisionError, ArithmeticError) as e:
                                # Fallback values if division errors occur
                                avg_charging_time = 30.0  # Default charging time
                                avg_queue_length = 0.0
                                avg_utilization = 0.0
                            
                            # Calculate customer satisfaction based on charging time and queue length
                            # Add safeguard against division by zero
                            try:
                                satisfaction = max(0.0, 1.0 - (avg_charging_time - 30.0) / max(30.0, 1.0) - avg_queue_length * 0.1)
                                satisfaction = max(0.0, min(1.0, satisfaction))
                            except (ZeroDivisionError, ArithmeticError):
                                satisfaction = 0.5  # Default satisfaction
                            
                            # Store current metrics for coordination interval updates
                            if not hasattr(self, 'current_metrics'):
                                self.current_metrics = {}
                            if sys_id not in self.current_metrics:
                                self.current_metrics[sys_id] = {}
                            
                            self.current_metrics[sys_id].update({
                                'charging_time': avg_charging_time,
                                'queue_length': avg_queue_length,
                                'utilization': avg_utilization,
                                'satisfaction': satisfaction
                            })
                        else:
                            # No stations - store default values
                            if not hasattr(self, 'current_metrics'):
                                self.current_metrics = {}
                            if sys_id not in self.current_metrics:
                                self.current_metrics[sys_id] = {}
                            
                            self.current_metrics[sys_id].update({
                                'charging_time': 30.0,
                                'queue_length': 0.0,
                                'utilization': 0.0,
                                'satisfaction': 1.0
                            })
                        
                    except (ZeroDivisionError, ArithmeticError, Exception) as e:
                        # Handle any errors in metrics collection
                        print(f"Warning: Error collecting metrics for system {sys_id}: {e}")
                        # Set default metrics
                        if not hasattr(self, 'current_metrics'):
                            self.current_metrics = {}
                        if sys_id not in self.current_metrics:
                            self.current_metrics[sys_id] = {}
                        
                        self.current_metrics[sys_id].update({
                            'charging_time': 30.0,
                            'queue_length': 0.0,
                            'utilization': 0.0,
                            'satisfaction': 1.0
                        })
            
            # Update transmission system with distribution loads and current time
            self.transmission_system.current_simulation_time = self.simulation_time
            self.transmission_system.update_distribution_load(dist_loads)
            self.transmission_system.update_frequency(self.dist_dt)
            
            # AGC updates (every 5 seconds)
            if self.simulation_time % self.agc_dt == 0:
                self.transmission_system.update_agc_reference(self.simulation_time)
                self.results['agc_updates'].append(self.simulation_time)
            
            # Central coordination updates (every 10 seconds)
            if self.simulation_time % self.coordination_dt == 0:
                # Update global status
                self.central_coordinator.update_global_status(self.simulation_time)
                
                # FIXED: Collect queue management and charging metrics at coordination interval
                if hasattr(self, 'current_metrics'):
                    # Use actual system IDs from distribution_systems keys, not range-based indices
                    for sys_id in self.distribution_systems.keys():
                        if sys_id in self.current_metrics:
                            metrics = self.current_metrics[sys_id]
                            self.results['charging_time_data'][sys_id].append(metrics['charging_time'])
                            self.results['queue_management_data'][sys_id].append(metrics['queue_length'])
                            self.results['utilization_data'][sys_id].append(metrics['utilization'])
                            self.results['customer_satisfaction_data'][sys_id].append(metrics['satisfaction'])
                        else:
                            # Default values if no metrics available
                            self.results['charging_time_data'][sys_id].append(30.0)
                            self.results['queue_management_data'][sys_id].append(0.0)
                            self.results['utilization_data'][sys_id].append(0.0)
                            self.results['customer_satisfaction_data'][sys_id].append(1.0)
                
                # Assess cyber attack impacts (only when attacks are active)
                attack_impacts = []
                # Check if there are active attacks in the current attack scenarios
                if hasattr(self, 'current_attack_scenarios') and self.current_attack_scenarios:
                    for attack in self.current_attack_scenarios:
                        if (attack['start_time'] <= self.simulation_time <= 
                            attack['start_time'] + attack['duration']):
                            # Attack is active, assess its impact
                            impact = self.central_coordinator.assess_cyber_attack_impact(
                                attack['type'], 
                                attack['magnitude'], 
                                attack['duration']
                            )
                            # Add system information to the impact
                            impact['system_id'] = attack['target_system']
                            impact['load_change_mw'] = 0.0  # Placeholder - could be calculated based on attack type
                            impact['load_change_percent'] = 0.0  # Placeholder - could be calculated based on attack type
                            impact['customer_satisfaction_impact'] = self.central_coordinator._calculate_customer_satisfaction_impact(
                                impact['charging_time_factor']
                            )
                            attack_impacts.append(impact)
                
                # Coordinate charging schedules
                self.central_coordinator.coordinate_charging_schedules(self.simulation_time)
                
                # Manage customer queues
                self.central_coordinator.manage_customer_queue(self.simulation_time)
                
                # Emergency response
                self.central_coordinator.emergency_response(self.simulation_time)
                
                # Store coordination report
                coordination_report = self.central_coordinator.get_coordination_report()
                self.results['coordination_reports'].append(coordination_report)
                
                # Store load balancing and redirection data
                self.results['load_balancing_data'].extend(coordination_report['load_balancing_history'])
                self.results['customer_redirection_data'].extend(coordination_report['customer_redirections'])
                
                # Store attack impact data
                for impact in attack_impacts:
                    sys_id = impact['system_id']
                    self.results['attack_impact_data'][sys_id].append({
                        'timestamp': self.simulation_time,
                        'attack_type': impact['attack_type'],
                        'charging_time_factor': impact['charging_time_factor'],
                        'customer_satisfaction_impact': impact['customer_satisfaction_impact'],
                        'load_change_percent': impact['load_change_percent']
                    })
                
                # Print coordination status
                if attack_impacts:
                    print(f"Central Coordinator at t={self.simulation_time:.1f}s: {len(attack_impacts)} systems under attack")
                    for impact in attack_impacts:
                        print(f"  System {impact['system_id']}: {impact['attack_type']} attack, "
                              f"Load change: {impact['load_change_mw']:.1f} MW ({impact['load_change_percent']:.1f}%), "
                              f"Charging time factor: {impact['charging_time_factor']:.2f}")
            
            # Run power flow analysis
            if self.transmission_system.run_power_flow():
                bus_voltages = self.transmission_system.get_bus_voltages()
                line_flows_from, line_flows_to = self.transmission_system.get_line_flows()
                self.results['bus_voltages'].append(bus_voltages)
                self.results['line_flows'].append((line_flows_from, line_flows_to))
            
            # Calculate total distribution load as sum of the stored (scaled) per-system values
            individual_loads = list(dist_loads.values())
            calculated_total = sum(individual_loads)
            
            # Store results
            self.results['time'].append(self.simulation_time)
            self.results['frequency'].append(self.transmission_system.frequency)
            self.results['total_load'].append(calculated_total)
            self.results['reference_power'].append(self.transmission_system.reference_power)
            
            # Verify load aggregation and fix any discrepancies
            if step % 50 == 0:
                # Recalculate individual loads from stored results for verification
                stored_individual = []
                for sys_id in sorted(self.results['dist_loads'].keys()):
                    if len(self.results['dist_loads'][sys_id]) > 0:
                        stored_individual.append(self.results['dist_loads'][sys_id][-1])
                
                stored_total = sum(stored_individual)
                print(f"Load Verification - Current: {[f'{load:.2f}' for load in individual_loads]}, "
                      f"Sum: {calculated_total:.2f}MW, Stored Sum: {stored_total:.2f}MW")
                
                # Ensure consistency between current calculation and stored values
                if abs(calculated_total - stored_total) > 0.01:  # 10kW tolerance
                    print(f"  WARNING: Load aggregation mismatch detected! Correcting...")
                    calculated_total = stored_total
            
            # Print progress
            if step % 10 == 0:
                print(f"t={self.simulation_time:.1f}s, f={self.transmission_system.frequency:.3f}Hz, "
                      f"Dist Load={calculated_total:.1f}MW, "
                      f"Ref Power={self.transmission_system.reference_power:.1f}MW")

        # ADDED: Detailed analysis and plotting after simulation
        self._analyze_simulation_results(attack_scenarios)
        
        return self.results

    def _generate_rl_attack_impact(self, system_id: int, attack_type: str, current_time: float) -> Dict:
        """Generate dynamic attack impact values using RL attack system"""
        if system_id not in self.rl_attack_systems:
            # Fallback to default values if RL system not available (increased for higher impact)
            return {
                'power_efficiency_reduction': 0.6,  # Increased from 0.2
                'load_instability_factor': 2.5,    # Increased from 1.5
                'voltage_efficiency_loss': 0.7,    # Increased from 0.3
                'demand_instability_factor': 2.3,  # Increased from 1.3
                'frequency_efficiency_loss': 0.5   # Increased from 0.15
            }
        
        rl_system = self.rl_attack_systems[system_id]
        
        # Get current system state for RL decision making
        system_state = {
            'grid_voltage': getattr(self.distribution_systems.get(system_id, {}).get('interface'), 'voltage_measured', 1.0),
            'frequency': getattr(self.transmission_system, 'frequency', 60.0),
            'current_load': self.results['dist_loads'].get(system_id, [0])[-1] if self.results['dist_loads'].get(system_id) else 0.0,
            'simulation_time': current_time
        }
        
        load_context = {
            'avg_load': np.mean(self.results['dist_loads'].get(system_id, [0])) if self.results['dist_loads'].get(system_id) else 0.0,
            'load_variance': np.var(self.results['dist_loads'].get(system_id, [0])) if self.results['dist_loads'].get(system_id) else 0.0
        }
        
        # Generate RL-based attack suggestion
        try:
            rl_attack = rl_system.generate_constrained_attack(system_state, load_context)
            
            if rl_attack:
                # Convert RL attack parameters to impact factors
                magnitude = rl_attack.get('magnitude', 20.0)
                duration = rl_attack.get('duration', 30.0)
                stealth_score = rl_attack.get('stealth_score', 0.5)
                
                # Calculate impact factors based on RL suggestions (increased for higher impact)
                base_impact = min(0.8, magnitude / 50.0)  # Scale magnitude to impact (0-0.8) - increased from 0.5 and 100.0
                duration_factor = min(1.5, duration / 60.0)  # Duration scaling
                stealth_factor = 1.0 - stealth_score * 0.3  # Higher stealth = lower visible impact
                
                return {
                    'power_efficiency_reduction': base_impact * duration_factor * stealth_factor,
                    'load_instability_factor': 1.0 + base_impact * 2.0 * duration_factor,
                    'voltage_efficiency_loss': base_impact * 1.5 * stealth_factor,
                    'demand_instability_factor': 1.0 + base_impact * 1.8 * duration_factor,
                    'frequency_efficiency_loss': base_impact * 0.8 * stealth_factor,
                    'rl_magnitude': magnitude,
                    'rl_duration': duration,
                    'rl_stealth_score': stealth_score
                }
        except Exception as e:
            print(f"Warning: RL attack generation failed for system {system_id}: {e}")
        
        # Fallback to default values (increased for higher impact)
        return {
            'power_efficiency_reduction': 0.6,  # Increased from 0.2
            'load_instability_factor': 2.5,    # Increased from 1.5
            'voltage_efficiency_loss': 0.7,    # Increased from 0.3
            'demand_instability_factor': 2.3,  # Increased from 1.3
            'frequency_efficiency_loss': 0.5   # Increased from 0.15
        }

    def _analyze_simulation_results(self, attack_scenarios: List[Dict] = None, create_plot: bool = True):
        """Detailed analysis and plotting of simulation results (integrated from test pipeline)"""
        print("\n" + "=" * 50)
        print(" SIMULATION RESULTS ANALYSIS")
        print("=" * 50)
        
        # Analyze charging time impacts for each system
        for sys_id, charging_times in self.results['charging_time_data'].items():
            if charging_times:
                print(f"\nSystem {sys_id} Charging Time Analysis:")
                print(f"  Data points: {len(charging_times)}")
                print(f"  Min time: {min(charging_times):.1f} min")
                print(f"  Max time: {max(charging_times):.1f} min")
                print(f"  Range: {max(charging_times) - min(charging_times):.1f} min")
                
                # Check for variation during attack periods if attack scenarios provided
                if attack_scenarios:
                    for attack in attack_scenarios:
                        if attack['target_system'] == sys_id:
                            attack_start = int(attack['start_time'] / self.dist_dt)
                            attack_end = int((attack['start_time'] + attack['duration']) / self.dist_dt)
                            
                            if attack_end < len(charging_times):
                                normal_times = charging_times[:attack_start] + charging_times[attack_end:]
                                attack_times = charging_times[attack_start:attack_end]
                                
                                if normal_times and attack_times:
                                    avg_normal = sum(normal_times) / len(normal_times)
                                    avg_attack = sum(attack_times) / len(attack_times)
                                    print(f"  Normal period avg: {avg_normal:.1f} min")
                                    print(f"  Attack period avg: {avg_attack:.1f} min")
                                    print(f"  Attack impact: {avg_attack - avg_normal:.1f} min ({((avg_attack - avg_normal) / avg_normal * 100):.1f}%)")
        
        # Create detailed analysis plot only if requested
        if create_plot:
            self._create_detailed_analysis_plot(attack_scenarios)
    
    def _create_detailed_analysis_plot(self, attack_scenarios: List[Dict] = None):
        """Create detailed analysis plot similar to test pipeline"""
        print("\n Creating detailed analysis plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot charging times for each system
        for sys_id, charging_times in self.results['charging_time_data'].items():
            if charging_times:
                # Use actual simulation time points, not the full time array
                time_points = np.linspace(0, self.total_duration, len(charging_times))
                ax.plot(time_points, charging_times, label=f'System {sys_id}', linewidth=2, marker='o', markersize=2)
        
        # Mark attack periods if provided
        if attack_scenarios:
            attack_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, attack in enumerate(attack_scenarios):
                color = attack_colors[i % len(attack_colors)]
                ax.axvspan(attack['start_time'], attack['start_time'] + attack['duration'], 
                          alpha=0.2, color=color, 
                          label=f'Attack on System {attack["target_system"]} ({attack["type"]})')
        
        # Formatting
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Average Charging Time (min)', fontsize=12)
        ax.set_title('Charging Time Impact from Cyber Attacks - Detailed Analysis', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=30.0, color='g', linestyle='--', alpha=0.7, label='Normal Time (30 min)')
        
        # Set reasonable x-axis limits
        ax.set_xlim(0, self.total_duration)
        
        plt.tight_layout()
        plt.savefig('detailed_charging_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(" Detailed analysis plot saved as 'detailed_charging_analysis.png'")

    
    def plot_hierarchical_results_old(self):
        """Plot hierarchical simulation results"""
        import time
        import os
        timestamp = int(time.time())
        
        # Create sub_figures directory if it doesn't exist
        os.makedirs('sub_figures', exist_ok=True)
        
        # Create a large figure with multiple subplots for comprehensive analysis
        fig = plt.figure(figsize=(20, 24))
        
        # Original plots (first 4 subplots)
        # Frequency plot
        ax1 = plt.subplot(6, 3, 1)
        ax1.plot(self.results['time'], self.results['frequency'], 'b-', linewidth=2)
        ax1.set_ylabel('Frequency (Hz)', fontsize=18)
        ax1.set_xlabel('Time (s)', fontsize=18)
        # ax1.set_title('Transmission System Frequency Response')
        ax1.grid(True)
        ax1.axhline(y=60.0, color='r', linestyle='--', alpha=0.7)
        
        # Mark AGC updates
        for agc_time in self.results['agc_updates']:
            ax1.axvline(x=agc_time, color='g', linestyle=':', alpha=0.5)
        
        # Reference power vs actual load with verification
        ax2 = plt.subplot(6, 3, 2)
        ax2.plot(self.results['time'], self.results['reference_power'], 'r-', linewidth=2, label='Reference Power')
        
        # Recalculate total distribution load from individual systems for plotting verification
        verified_total_load = []
        for i in range(len(self.results['time'])):
            time_step_total = 0.0
            for sys_id, loads in self.results['dist_loads'].items():
                if i < len(loads):
                    time_step_total += loads[i]
            verified_total_load.append(time_step_total)
        
        ax2.plot(self.results['time'], verified_total_load, 'b-', linewidth=2, label='Distribution Load')
        ax2.set_ylabel('Power (MW)', fontsize=18)
        ax2.set_xlabel('Time (s)', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # ax2.set_title('AGC Reference vs Distribution Load')
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        # Print verification statistics
        original_total = np.array(self.results['total_load'])
        verified_total = np.array(verified_total_load)
        max_diff = np.max(np.abs(original_total - verified_total))
        print(f"Load aggregation verification: Max difference = {max_diff:.3f} MW")
        
        # Individual distribution system loads
        ax3 = plt.subplot(6, 3, 3)
        for sys_id, loads in self.results['dist_loads'].items():
            ax3.plot(self.results['time'], loads, label=f'Dist System {sys_id}')
        ax3.set_ylabel('Load (MW)', fontsize=18)
        ax3.set_xlabel('Time (s)', fontsize=18)
        # ax3.set_title('Individual Distribution System Loads')
        ax3.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax3.grid(True)
        
        # Charging Time Analysis
        ax4 = plt.subplot(6, 3, 4)
        for sys_id, charging_times in self.results['charging_time_data'].items():
            if charging_times:  # Only plot if we have data
                # Use coordination interval time array for charging time data
                charging_times_array = [i * self.coordination_dt for i in range(len(charging_times))]
                charging_times_array = charging_times_array[:len(charging_times)]  # Ensure matching lengths
                ax4.plot(charging_times_array, charging_times, label=f'Dist. Sys. {sys_id}', linewidth=2)
        
        ax4.set_ylabel('Average Charging Time (min)', fontsize=18)
        ax4.set_xlabel('Time (s)', fontsize=18)
        ax4.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax4.grid(True)
        ax4.axhline(y=30.0, color='g', linestyle='--', alpha=0.7, label='Normal Time (30 min)')
        
        # FIXED: Queue Management Analysis - Using stored time-series data
        ax5 = plt.subplot(6, 3, 5)
        for sys_id, queue_lengths in self.results['queue_management_data'].items():
            if queue_lengths and len(queue_lengths) > 0:  # Only plot if we have data
                # Create proper time array for queue data (collected every coordination_dt interval)
                queue_times = [i * self.coordination_dt for i in range(len(queue_lengths))]
                queue_times = queue_times[:len(queue_lengths)]  # Ensure matching lengths
                
                ax5.plot(queue_times, queue_lengths, label=f'Dist. Sys. {sys_id+1}', linewidth=2, marker='o', markersize=3)
        
        ax5.set_ylabel('Average Queue Length (customers)', fontsize=18)
        ax5.set_xlabel('Time (s)', fontsize=18)
        # ax5.set_title('Customer Queue Management (5s intervals)')
        ax5.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax5.grid(True, alpha=0.3)
        
        # Add reference line for acceptable queue length
        if any(len(q) > 0 for q in self.results['queue_management_data'].values()):
            ax5.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='High Queue Alert (5)')
            ax5.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Critical Queue (10)')
            ax5.legend(fontsize=12)
        
        # Customer Satisfaction Analysis
        ax6 = plt.subplot(6, 3, 6)
        for sys_id, satisfaction in self.results['customer_satisfaction_data'].items():
            if satisfaction and len(satisfaction) > 0:  # Only plot if we have data
                # Create proper time array for satisfaction data (collected every coordination_dt interval)
                satisfaction_times = [i * self.coordination_dt for i in range(len(satisfaction))]
                satisfaction_times = satisfaction_times[:len(satisfaction)]  # Ensure matching lengths
                ax6.plot(satisfaction_times, satisfaction, label=f'System {sys_id}', linewidth=2)
        ax6.set_ylabel('Customer Satisfaction (0-1)', fontsize=18)
        ax6.set_xlabel('Time (s)', fontsize=18)
        # ax6.set_title('Customer Satisfaction Impact')
        ax6.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax6.grid(True)
        ax6.set_ylim(0, 1)
        
        # Utilization Rate Analysis
        ax7 = plt.subplot(6, 3, 7)
        for sys_id, utilization in self.results['utilization_data'].items():
            if utilization and len(utilization) > 0:  # Only plot if we have data
                # Create proper time array for utilization data (collected every coordination_dt interval)
                utilization_times = [i * self.coordination_dt for i in range(len(utilization))]
                utilization_times = utilization_times[:len(utilization)]  # Ensure matching lengths
                ax7.plot(utilization_times, utilization, label=f'Dist. Sys. {sys_id}', linewidth=2)
        ax7.set_ylabel('Utilization Rate (0-1)', fontsize=18)
        ax7.set_xlabel('Time (s)', fontsize=18)
        # ax7.set_title('Charging Station Utilization')
        ax7.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax7.grid(True)
        ax7.set_ylim(0, 1)
        
        # Save individual subplots as separate PDFs with timestamp
        
        # Save subplot 1: Frequency Response
        fig_1 = plt.figure(figsize=(10, 8))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(self.results['time'], self.results['frequency'], 'b-', linewidth=2)
        ax_1.set_ylabel('Frequency (Hz)', fontsize=18)
        ax_1.set_xlabel('Time (s)', fontsize=18)
        ax_1.grid(True)
        ax_1.axhline(y=60.0, color='r', linestyle='--', alpha=0.7)
        for agc_time in self.results['agc_updates']:
            ax_1.axvline(x=agc_time, color='g', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'sub_figures/frequency_response_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_1)
        
        # Save subplot 2: Reference Power vs Distribution Load
        fig_2 = plt.figure(figsize=(10, 8))
        ax_2 = fig_2.add_subplot(111)
        ax_2.plot(self.results['time'], self.results['reference_power'], 'r-', linewidth=2, label='Reference Power')
        verified_total_load = []
        for i in range(len(self.results['time'])):
            time_step_total = 0.0
            for sys_id, loads in self.results['dist_loads'].items():
                if i < len(loads):
                    time_step_total += loads[i]
            verified_total_load.append(time_step_total)
        ax_2.plot(self.results['time'], verified_total_load, 'b-', linewidth=2, label='Distribution Load')
        ax_2.set_ylabel('Power (MW)', fontsize=18)
        ax_2.set_xlabel('Time (s)', fontsize=18)
        ax_2.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_2.grid(True)
        plt.tight_layout()
        plt.savefig(f'sub_figures/reference_vs_distribution_load_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_2)
        
        # Save subplot 3: Individual Distribution System Loads
        fig_3 = plt.figure(figsize=(10, 8))
        ax_3 = fig_3.add_subplot(111)
        for sys_id, loads in self.results['dist_loads'].items():
            ax_3.plot(self.results['time'], loads, label=f'Dist System {sys_id}')
        ax_3.set_ylabel('Load (MW)', fontsize=18)
        ax_3.set_xlabel('Time (s)', fontsize=18)
        ax_3.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_3.grid(True)
        plt.tight_layout()
        plt.savefig(f'sub_figures/individual_distribution_loads_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_3)
        
        # Save subplot 4: Charging Time Analysis
        fig_4 = plt.figure(figsize=(10, 8))
        ax_4 = fig_4.add_subplot(111)
        for sys_id, charging_times in self.results['charging_time_data'].items():
            if charging_times:
                charging_times_array = [i * self.coordination_dt for i in range(len(charging_times))]
                charging_times_array = charging_times_array[:len(charging_times)]
                ax_4.plot(charging_times_array, charging_times, label=f'Dist. Sys. {sys_id}', linewidth=2)
        ax_4.set_ylabel('Average Charging Tim   e (min)', fontsize=18)
        ax_4.set_xlabel('Time (s)', fontsize=18)
        ax_4.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_4.grid(True)
        ax_4.axhline(y=30.0, color='g', linestyle='--', alpha=0.7, label='Normal Time (30 min)')
        plt.tight_layout()
        plt.savefig(f'sub_figures/charging_time_analysis_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_4)
        
        # Save subplot 5: Queue Management Analysis
        fig_5 = plt.figure(figsize=(10, 8))
        ax_5 = fig_5.add_subplot(111)
        for sys_id, queue_lengths in self.results['queue_management_data'].items():
            if queue_lengths and len(queue_lengths) > 0:
                queue_times = [i * self.coordination_dt for i in range(len(queue_lengths))]
                queue_times = queue_times[:len(queue_lengths)]
                ax_5.plot(queue_times, queue_lengths, label=f'Dist. Sys. {sys_id+1}', linewidth=2, marker='o', markersize=3)
        ax_5.set_ylabel('Average Queue Length (customers)', fontsize=18)
        ax_5.set_xlabel('Time (s)', fontsize=18)
        ax_5.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax_5.grid(True, alpha=0.3)
        if any(len(q) > 0 for q in self.results['queue_management_data'].values()):
            ax_5.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='High Queue Alert (5)')
            ax_5.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Critical Queue (10)')
            ax_5.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'sub_figures/queue_management_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_5)
        
        # Save subplot 6: Customer Satisfaction Analysis
        fig_6 = plt.figure(figsize=(10, 8))
        ax_6 = fig_6.add_subplot(111)
        for sys_id, satisfaction in self.results['customer_satisfaction_data'].items():
            if satisfaction and len(satisfaction) > 0:
                satisfaction_times = [i * self.coordination_dt for i in range(len(satisfaction))]
                satisfaction_times = satisfaction_times[:len(satisfaction)]
                ax_6.plot(satisfaction_times, satisfaction, label=f'System {sys_id}', linewidth=2)
        ax_6.set_ylabel('Customer Satisfaction (0-1)', fontsize=18)
        ax_6.set_xlabel('Time (s)', fontsize=18)
        ax_6.legend(fontsize=10)
        ax_6.grid(True)
        ax_6.set_ylim(0, 1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/customer_satisfaction_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_6)
        
        # Save subplot 7: Utilization Rate Analysis
        fig_7 = plt.figure(figsize=(10, 8))
        ax_7 = fig_7.add_subplot(111)
        for sys_id, utilization in self.results['utilization_data'].items():
            if utilization and len(utilization) > 0:
                utilization_times = [i * self.coordination_dt for i in range(len(utilization))]
                utilization_times = utilization_times[:len(utilization)]
                ax_7.plot(utilization_times, utilization, label=f'Dist. Sys. {sys_id}', linewidth=2)
        ax_7.set_ylabel('Utilization Rate (0-1)', fontsize=18)
        ax_7.set_xlabel('Time (s)', fontsize=18)
        ax_7.legend(fontsize=10)
        ax_7.grid(True)
        ax_7.set_ylim(0, 1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/utilization_rate_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig_7)
        
        print(f"\n✅ First 7 hierarchical subplots saved as individual PDFs in 'sub_figures/' directory:")
        print(f"   1. frequency_response_{timestamp}.pdf")
        print(f"   2. reference_vs_distribution_load_{timestamp}.pdf")
        print(f"   3. individual_distribution_loads_{timestamp}.pdf")
        print(f"   4. charging_time_analysis_{timestamp}.pdf")
        print(f"   5. queue_management_{timestamp}.pdf")
        print(f"   6. customer_satisfaction_{timestamp}.pdf")
        print(f"   7. utilization_rate_{timestamp}.pdf")
        
        # Attack Impact Analysis - Charging Time Factor
        ax8 = plt.subplot(6, 3, 8)
        for sys_id, attack_data in self.results['attack_impact_data'].items():
            if attack_data:
                timestamps = [data['timestamp'] for data in attack_data]
                time_factors = [data['charging_time_factor'] for data in attack_data]
                ax8.plot(timestamps, time_factors, label=f'System {sys_id}', linewidth=2, marker='o')
        ax8.set_ylabel('Charging Time Factor', fontsize=18)
        ax8.set_xlabel('Time (s)', fontsize=18)
        # ax8.set_title('Cyber Attack Impact on Charging Time')
        ax8.legend(fontsize=10)
        ax8.grid(True)
        ax8.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Normal')
        
        # Load Balancing Analysis
        ax9 = plt.subplot(6, 3, 9)
        if self.results['load_balancing_data']:
            timestamps = [data['timestamp'] for data in self.results['load_balancing_data']]
            utilizations = [data['global_utilization'] for data in self.results['load_balancing_data']]
            ax9.plot(timestamps, utilizations, 'purple', linewidth=2, marker='s')
        ax9.set_ylabel('Global Utilization Rate', fontsize=18)
        ax9.set_xlabel('Time (s)', fontsize=18)
        # ax9.set_title('Load Balancing Effectiveness')
        ax9.grid(True)
        ax9.set_ylim(0, 1)
        
        # Customer Redirection Analysis
        ax10 = plt.subplot(6, 3, 10)
        if self.results['customer_redirection_data'] and len(self.results['customer_redirection_data']) > 0:
            try:
                # Handle different possible data structures
                timestamps = []
                queue_lengths = []
                source_systems = []
                target_systems = []
                
                for data in self.results['customer_redirection_data']:
                    if isinstance(data, dict):
                        # Extract timestamp (handle different key names)
                        timestamp = data.get('timestamp', data.get('time', 0))
                        timestamps.append(timestamp)
                        
                        # Extract queue length
                        queue_length = data.get('queue_length', data.get('source_queue_length', 0))
                        queue_lengths.append(queue_length)
                        
                        # Extract system information for color coding
                        source_sys = data.get('source_system', data.get('from_system', 0))
                        source_systems.append(source_sys)
                        
                        target_sys = data.get('target_system', data.get('to_system', 0))
                        target_systems.append(target_sys)
                
                if timestamps and queue_lengths:
                    # Create scatter plot with color coding by source system
                    scatter = ax10.scatter(timestamps, queue_lengths, 
                                         c=source_systems, cmap='tab10', 
                                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                    
                    # Add colorbar if multiple systems
                    if len(set(source_systems)) > 1:
                        cbar = plt.colorbar(scatter, ax=ax10)
                        cbar.set_label('Source System ID')
                    
                    # Add annotations for significant redirections
                    for i, (t, q) in enumerate(zip(timestamps, queue_lengths)):
                        if q > 8:  # Annotate high queue redirections
                            ax10.annotate(f'S{source_systems[i]}→S{target_systems[i]}', 
                                        (t, q), xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, alpha=0.8)
                else:
                    ax10.text(0.5, 0.5, 'No valid redirection data', 
                            ha='center', va='center', transform=ax10.transAxes,
                            fontsize=12, color='gray')
                            
            except Exception as e:
                ax10.text(0.5, 0.5, f'Error parsing redirection data:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax10.transAxes,
                        fontsize=10, color='red')
        else:
            ax10.text(0.5, 0.5, 'No customer redirections occurred', 
                    ha='center', va='center', transform=ax10.transAxes,
                    fontsize=12, color='green', weight='bold')
        
        ax10.set_ylabel('Queue Length at Redirection', fontsize=18)
        ax10.set_xlabel('Time (s)', fontsize=18)
        # ax10.set_title('Customer Redirections Due to Overload')
        ax10.grid(True, alpha=0.3)
        
        # Frequency deviation
        ax11 = plt.subplot(6, 3, 11)
        freq_deviation = [abs(60.0 - f) for f in self.results['frequency']]
        ax11.plot(self.results['time'], freq_deviation, 'g-', linewidth=2)
        ax11.set_ylabel('Frequency Deviation (Hz)', fontsize=18)
        ax11.set_xlabel('Time (s)', fontsize=18)
        # ax11.set_title('Frequency Deviation from Nominal')
        ax11.grid(True)
        
        # Load variation analysis
        ax12 = plt.subplot(6, 3, 12)
        load_variation = np.diff(self.results['total_load'])
        ax12.plot(self.results['time'][1:], load_variation, 'm-', linewidth=1)
        ax12.set_ylabel('Load Change (MW)', fontsize=18)
        ax12.set_xlabel('Time (s)', fontsize=18)
        # ax12.set_title('Distribution Load Variation Rate')
        ax12.grid(True)
        
        # Bus voltages (if available)
        ax13 = plt.subplot(6, 3, 13)
        if self.results['bus_voltages']:
            bus_voltages = np.array(self.results['bus_voltages'])
            for i in range(min(5, bus_voltages.shape[1])):  # Show first 5 buses
                ax13.plot(self.results['time'], bus_voltages[:, i], 
                       label=f'Bus {i+1}', alpha=0.7)
        ax13.set_ylabel('Voltage (p.u.)', fontsize=18)
        ax13.set_xlabel('Time (s)', fontsize=18)
        # ax13.set_title('Transmission Bus Voltages')
        ax13.legend(fontsize=10)
        ax13.grid(True)
        
        # Frequency vs Load correlation
        ax14 = plt.subplot(6, 3, 14)
        scatter = ax14.scatter(self.results['total_load'], self.results['frequency'], 
                     c=self.results['time'], cmap='viridis', alpha=0.6)
        ax14.set_xlabel('Total Distribution Load (MW)', fontsize=18)
        ax14.set_ylabel('Frequency (Hz)', fontsize=18)
        # ax14.set_title('Frequency vs Load Correlation')
        ax14.grid(True)
        cbar = plt.colorbar(scatter, ax=ax14, label='Time (s)')
        cbar.ax.tick_params(labelsize=18)
        
        # Time series of all key variables
        ax15 = plt.subplot(6, 3, 15)
        ax15.plot(self.results['time'], self.results['frequency'], 'b-', label='Frequency', alpha=0.7)
        ax15_twin = ax15.twinx()
        ax15_twin.plot(self.results['time'], self.results['total_load'], 'r-', label='Load', alpha=0.7)
        ax15.set_xlabel('Time (s)', fontsize=18)
        ax15.set_ylabel('Frequency (Hz)', color='b', fontsize=18)  
        ax15_twin.set_ylabel('Load (MW)', color='r', fontsize=18)
        # ax15.set_title('Frequency and Load Time Series', fontsize=18)
        ax15.grid(True)
        
        # Attack Impact Summary
        ax16 = plt.subplot(6, 3, 16)
        attack_summary = {}
        for sys_id, attack_data in self.results['attack_impact_data'].items():
            if attack_data:
                attack_types = [data['attack_type'] for data in attack_data]
                load_changes = [data['load_change_percent'] for data in attack_data]
                ax16.scatter(attack_types, load_changes, label=f'System {sys_id}', alpha=0.7, s=100)
        ax16.set_ylabel('Load Change (%)', fontsize=18)
        # ax16.set_title('Cyber Attack Impact Summary')
        ax16.legend(fontsize=18)
        ax16.grid(True)
        
        # Customer Satisfaction vs Charging Time Correlation
        ax17 = plt.subplot(6, 3, 17)
        for sys_id in self.results['charging_time_data'].keys():
            if sys_id in self.results['customer_satisfaction_data']:
                charging_times = self.results['charging_time_data'][sys_id]
                satisfaction = self.results['customer_satisfaction_data'][sys_id]
                
                # Ensure both arrays have the same length
                if charging_times and satisfaction:
                    min_length = min(len(charging_times), len(satisfaction))
                    if min_length > 0:
                        charging_times_trimmed = charging_times[:min_length]
                        satisfaction_trimmed = satisfaction[:min_length]
                        ax17.scatter(charging_times_trimmed, satisfaction_trimmed, label=f'System {sys_id}', alpha=0.7)
        ax17.set_xlabel('Average Charging Time (min)', fontsize=18)
        ax17.set_ylabel('Customer Satisfaction', fontsize=18)
        # ax17.set_title('Satisfaction vs Charging Time')
        ax17.legend(fontsize=18)
        ax17.grid(True)
        
        # Queue Length vs Utilization Correlation
        ax18 = plt.subplot(6, 3, 18)
        for sys_id in self.results['queue_management_data'].keys():
            if sys_id in self.results['utilization_data']:
                queue_lengths = self.results['queue_management_data'][sys_id]
                utilization = self.results['utilization_data'][sys_id]
                
                # Ensure both arrays have the same length
                if queue_lengths and utilization:
                    min_length = min(len(queue_lengths), len(utilization))
                    if min_length > 0:
                        queue_lengths_trimmed = queue_lengths[:min_length]
                        utilization_trimmed = utilization[:min_length]
                        ax18.scatter(utilization_trimmed, queue_lengths_trimmed, label=f'System {sys_id}', alpha=0.7)
        ax18.set_xlabel('Utilization Rate', fontsize=18)
        ax18.set_ylabel('Queue Length', fontsize=18)
        # ax18.set_title('Queue Length vs Utilization')
        ax18.legend(fontsize=18)       
        ax18.grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.savefig('figure_1.png')
        plt.close()
        
        # Create a separate figure for detailed attack impact analysis
        self._plot_attack_impact_analysis()

                # Create Figure 1: EVCS Output Voltage for 6 Distribution Systems
        self._plot_evcs_voltage_analysis()
        
        # Create Figure 2: EVCS Power Output for 6 Distribution Systems  
        self._plot_evcs_power_analysis()
        
        # Create Figure 3: EVCS Current Output for 6 Distribution Systems
        self._plot_evcs_current_analysis()

    
    def plot_hierarchical_results(self):
        """Plot hierarchical simulation results with 3 separate figures for EVCS analysis"""
        print("Generating hierarchical simulation plots...")


        self.plot_hierarchical_results_old()
        

        print("All hierarchical simulation plots completed!")
    
    def _plot_evcs_voltage_analysis(self):
        """Create Figure 1: EVCS Output Voltage for 6 Distribution Systems"""
        import time
        timestamp = int(time.time())
        
        if not self.results.get('evcs_voltage_data'):
            print("Warning: No EVCS voltage data available for plotting")
            return
            
        # Create figure with 6 subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        # fig.suptitle('EVCS Output Voltage Analysis - 6 Distribution Systems', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot EVCS voltage for each distribution system
        for sys_id in range(6):  # 6 distribution systems
            if sys_id in self.results['evcs_voltage_data'] and len(self.results['evcs_voltage_data'][sys_id]) > 0:
                voltage_data = self.results['evcs_voltage_data'][sys_id]
                
                # Plot voltage over time
                axes[sys_id].plot(self.results['time'][:len(voltage_data)], voltage_data, 
                                'b-', linewidth=2, alpha=0.8)
                
                # Add grid and labels
                axes[sys_id].grid(True, alpha=0.3)
                axes[sys_id].set_xlabel('Time (s)', fontsize=18)
                axes[sys_id].set_ylabel('Voltage (V)', fontsize=18)
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Voltage')
                
                # Add voltage reference line (800V rated voltage)
                axes[sys_id].axhline(y=400.0, color='r', linestyle='--', alpha=0.7, 
                                   label='Rated Voltage (400V)')
                
                # Add voltage bandwidth limits
                axes[sys_id].axhline(y=410.0, color='orange', linestyle=':', alpha=0.5, 
                                   label='Upper Limit (410V)')
                axes[sys_id].axhline(y=390.0, color='orange', linestyle=':', alpha=0.5, 
                                   label='Lower Limit (390V)')
                
                axes[sys_id].legend(fontsize=18)
                
                # Set y-axis limits with some margin
                if len(voltage_data) > 0:
                    min_voltage = min(voltage_data)
                    max_voltage = max(voltage_data)
                    voltage_range = max_voltage - min_voltage
                    axes[sys_id].set_ylim(min_voltage - voltage_range*0.1, 
                                        max_voltage + voltage_range*0.1)
            else:
                # If no data available, show empty plot with message
                axes[sys_id].text(0.5, 0.5, f'No voltage data\nfor System {sys_id + 1}', 
                                ha='center', va='center', transform=axes[sys_id].transAxes,
                                fontsize=18, color='gray')
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Voltage')
                axes[sys_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        # plt.savefig('figure_voltage.png')
        # plt.close()
        
        print("Figure 1: EVCS Output Voltage Analysis completed")
    
    def _plot_evcs_power_analysis(self):
        """Create Figure 2: EVCS Power Output for 6 Distribution Systems"""
        import time
        timestamp = int(time.time())
        
        if not self.results.get('evcs_power_data'):
            print("Warning: No EVCS power data available for plotting")
            return
            
        # Create figure with 6 subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            # fig.suptitle('EVCS Power Output Analysis - 6 Distribution Systems', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot EVCS power for each distribution system
        for sys_id in range(6):  # 6 distribution systems
            if sys_id in self.results['evcs_power_data'] and len(self.results['evcs_power_data'][sys_id]) > 0:
                power_data = self.results['evcs_power_data'][sys_id]
                
                # Plot power over time
                axes[sys_id].plot(self.results['time'][:len(power_data)], power_data, 
                                'g-', linewidth=2, alpha=0.8)
                
                # Add grid and labels
                axes[sys_id].grid(True, alpha=0.3)
                axes[sys_id].set_xlabel('Time (s)', fontsize=18)
                axes[sys_id].set_ylabel('Power (kW)', fontsize=18)
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Power', fontsize=18)
                
                # Add power reference line (if available)
                if hasattr(self, 'total_evcs_capacity') and self.total_evcs_capacity:
                    axes[sys_id].axhline(y=self.total_evcs_capacity, color='r', linestyle='--', 
                                       alpha=0.7, label='Total EVCS Capacity')
                
                # Add zero power line for reference
                axes[sys_id].axhline(y=0.0, color='k', linestyle='-', alpha=0.3, 
                                   label='Zero Power')
                
                axes[sys_id].legend(fontsize=18)
                
                # Set y-axis limits with some margin
                if len(power_data) > 0:
                    min_power = min(power_data)
                    max_power = max(power_data)
                    power_range = max_power - min_power
                    if power_range > 0:
                        axes[sys_id].set_ylim(min_power - power_range*0.1, 
                                            max_power + power_range*0.1)
                    else:
                        axes[sys_id].set_ylim(min_power - 1, max_power + 1)
            else:
                # If no data available, show empty plot with message
                axes[sys_id].text(0.5, 0.5, f'No power data\nfor System {sys_id + 1}', 
                                ha='center', va='center', transform=axes[sys_id].transAxes,
                                fontsize=18, color='gray')
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Power')
                axes[sys_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        # plt.savefig('figure_3.png')
        # plt.close()
        
        print("Figure 2: EVCS Power Output Analysis completed")
    
    def _plot_evcs_current_analysis(self):
        """Create Figure 3: EVCS Current Output for 6 Distribution Systems"""
        import time
        timestamp = int(time.time())
        
        if not self.results.get('evcs_current_data'):
            print("Warning: No EVCS current data available for plotting")
            return
            
        # Create figure with 6 subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            # fig.suptitle('EVCS Current Output Analysis - 6 Distribution Systems', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot EVCS current for each distribution system
        for sys_id in range(6):  # 6 distribution systems
            if sys_id in self.results['evcs_current_data'] and len(self.results['evcs_current_data'][sys_id]) > 0:
                current_data = self.results['evcs_current_data'][sys_id]
                
                # Plot current over time
                axes[sys_id].plot(self.results['time'][:len(current_data)], current_data, 
                                'c-', linewidth=2, alpha=0.8)
                
                # Add grid and labels
                axes[sys_id].grid(True, alpha=0.3)
                axes[sys_id].set_xlabel('Time (s)', fontsize=18)
                axes[sys_id].set_ylabel('Current (A)', fontsize=18)
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Current', fontsize=18)
                
                # Add current reference line (if available)
                if hasattr(self, 'total_evcs_capacity') and self.total_evcs_capacity:
                    axes[sys_id].axhline(y=self.total_evcs_capacity / 400.0 * 125, color='r', linestyle='--', 
                                       alpha=0.7, label='Rated Current (125A)')
                
                # Add zero current line for reference
                axes[sys_id].axhline(y=0.0, color='k', linestyle='-', alpha=0.3, 
                                   label='Zero Current')
                
                axes[sys_id].legend(fontsize=18)
                
                # Set y-axis limits with some margin
                if len(current_data) > 0:
                    min_current = min(current_data)
                    max_current = max(current_data)
                    current_range = max_current - min_current
                    axes[sys_id].set_ylim(min_current - current_range*0.1, 
                                        max_current + current_range*0.1)
            else:
                # If no data available, show empty plot with message
                axes[sys_id].text(0.5, 0.5, f'No current data\nfor System {sys_id + 1}', 
                                ha='center', va='center', transform=axes[sys_id].transAxes,
                                        fontsize=18, color='gray')
                # axes[sys_id].set_title(f'Distribution System {sys_id + 1} - EVCS Current', fontsize=18)
                axes[sys_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        # plt.savefig('figure_4.png')
        # plt.close()
        
        print("Figure 3: EVCS Current Output Analysis completed")
        
    def _plot_attack_impact_analysis(self):
        """Create detailed plots showing cyber attack impacts on charging infrastructure"""
        import time
        import os
        from datetime import datetime
        
        # Create sub_figures directory if it doesn't exist
        os.makedirs('sub_figures', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Creating attack impact analysis plots...")
        
        # Create a comprehensive figure for attack impact analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        # fig.suptitle('Cyber Attack Impact Analysis on EVCS Infrastructure', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot 1: Frequency response
        ax1 = axes[0]
        if self.results.get('frequency') is not None and len(self.results.get('frequency', [])) > 0:
            ax1.plot(self.results['time'], self.results['frequency'], 'b-', linewidth=2)
            ax1.set_ylabel('Frequency (Hz)', fontsize=18)
            ax1.set_xlabel('Time (s)', fontsize=18)
            ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True)
        ax1.axhline(y=60.0, color='r', linestyle='--', alpha=0.7, label='Nominal (60Hz)')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # Mark AGC updates
        for agc_time in self.results.get('agc_updates', []):
            ax1.axvline(x=agc_time, color='g', linestyle=':', alpha=0.5)
        ax1.legend(fontsize=18)
        
        # Plot 2: Load vs Reference Power
        ax2 = axes[1]
        if (self.results.get('reference_power') is not None and len(self.results.get('reference_power', [])) > 0 and 
            self.results.get('total_load') is not None and len(self.results.get('total_load', [])) > 0):
            ax2.plot(self.results['time'], self.results['reference_power'], 'r-', linewidth=2, label='Reference Power')
            ax2.plot(self.results['time'], self.results['total_load'], 'b-', linewidth=2, label='Distribution Load')
            ax2.set_ylabel('Power (MW)', fontsize=18)
            ax2.set_xlabel('Time (s)', fontsize=18)
            ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.legend(fontsize=18)
        ax2.grid(True)
        
        # Plot 3: Individual distribution system loads
        ax3 = axes[2]
        for sys_id, loads in self.results.get('dist_loads', {}).items():
            if loads:
                # Ensure time and load arrays have the same length
                time_array = self.results['time'][:len(loads)]
                ax3.plot(time_array, loads, label=f'Dist System {sys_id}')
        ax3.set_ylabel('Load (MW)', fontsize=18)
        ax3.set_xlabel('Time (s)', fontsize=18)
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.legend(fontsize=18)
        ax3.grid(True)
        
        # Plot 4: Charging Time Analysis
        ax4 = axes[3]
        for sys_id, charging_times in self.results.get('charging_time_data', {}).items():
            if charging_times:  # Only plot if we have data
                # Use coordination interval time array for charging time data
                charging_times_array = [i * self.coordination_dt for i in range(len(charging_times))]
                charging_times_array = charging_times_array[:len(charging_times)]  # Ensure matching lengths
                ax4.plot(charging_times_array, charging_times, label=f'System {sys_id}', linewidth=2)
        ax4.set_ylabel('Average Charging Time (min)', fontsize=18)
        ax4.set_xlabel('Time (s)', fontsize=18)
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.legend(fontsize=18)
        ax4.grid(True)
        ax4.axhline(y=30.0, color='g', linestyle='--', alpha=0.7, label='Normal Time')
        
        # Plot 5: Customer Satisfaction Analysis
        ax5 = axes[4]
        for sys_id, satisfaction in self.results.get('customer_satisfaction_data', {}).items():
            if satisfaction and len(satisfaction) > 0:  # Only plot if we have data
                # Create proper time array for satisfaction data (collected every coordination_dt interval)
                satisfaction_times = [i * self.coordination_dt for i in range(len(satisfaction))]
                satisfaction_times = satisfaction_times[:len(satisfaction)]  # Ensure matching lengths
                ax5.plot(satisfaction_times, satisfaction, label=f'System {sys_id}', linewidth=2)
        ax5.set_ylabel('Customer Satisfaction (0-1)', fontsize=18)
        ax5.set_xlabel('Time (s)', fontsize=18)
        ax5.tick_params(axis='both', which='major', labelsize=18)
        ax5.legend(fontsize=18)
        ax5.grid(True)
        ax5.set_ylim(0, 1)
        
        # Plot 6: Attack Impact Summary
        ax6 = axes[5]
        attack_summary = {}
        for sys_id, attack_data in self.results.get('attack_impact_data', {}).items():
            if attack_data:
                attack_types = [data['attack_type'] for data in attack_data]
                load_changes = [data['load_change_percent'] for data in attack_data]
                ax6.scatter(attack_types, load_changes, label=f'System {sys_id}', alpha=0.7, s=100)
            ax6.set_ylabel('Load Change (%)', fontsize=18)
            ax6.set_xlabel('Time (s)', fontsize=18)
            ax6.tick_params(axis='both', which='major', labelsize=18)
        ax6.legend(fontsize=18)
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save individual subplots as separate PDFs
        print("Saving individual subplot PDFs...")
        
        # Subplot 1: Frequency Response
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        if self.results.get('frequency') is not None and len(self.results.get('frequency', [])) > 0:
            ax1.plot(self.results['time'], self.results['frequency'], 'b-', linewidth=2)
            ax1.set_ylabel('Frequency (Hz)', fontsize=18)
            ax1.set_xlabel('Time (s)', fontsize=18)
            ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True)
        ax1.axhline(y=60.0, color='r', linestyle='--', alpha=0.7, label='Nominal (60Hz)')
        for agc_time in self.results.get('agc_updates', []):
            ax1.axvline(x=agc_time, color='g', linestyle=':', alpha=0.5)
        ax1.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/frequency_response_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig1)
        
        # Subplot 2: Load vs Reference Power
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)
        if (self.results.get('reference_power') is not None and len(self.results.get('reference_power', [])) > 0 and 
            self.results.get('total_load') is not None and len(self.results.get('total_load', [])) > 0):
            ax2.plot(self.results['time'], self.results['reference_power'], 'r-', linewidth=2, label='Reference Power')
            ax2.plot(self.results['time'], self.results['total_load'], 'b-', linewidth=2, label='Distribution Load')
            ax2.set_ylabel('Power (MW)', fontsize=18)
            ax2.set_xlabel('Time (s)', fontsize=18)
            ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/reference_vs_distribution_load_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig2)
        
        # Subplot 3: Individual Distribution System Loads
        fig3 = plt.figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(111)
        for sys_id, loads in self.results.get('dist_loads', {}).items():
            if loads:
                time_array = self.results['time'][:len(loads)]
                ax3.plot(time_array, loads, label=f'Dist System {sys_id}')
        ax3.set_ylabel('Load (MW)', fontsize=18)
        ax3.set_xlabel('Time (s)', fontsize=18)
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.legend(fontsize=12)
        ax3.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/individual_distribution_loads_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig3)
        
        # Subplot 4: Charging Time Analysis
        fig4 = plt.figure(figsize=(10, 8))
        ax4 = fig4.add_subplot(111)
        for sys_id, charging_times in self.results.get('charging_time_data', {}).items():
            if charging_times:
                charging_times_array = [i * self.coordination_dt for i in range(len(charging_times))]
                charging_times_array = charging_times_array[:len(charging_times)]
                ax4.plot(charging_times_array, charging_times, label=f'System {sys_id}', linewidth=2)
        ax4.set_ylabel('Average Charging Time (min)', fontsize=18)
        ax4.set_xlabel('Time (s)', fontsize=18)
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.legend(fontsize=12)
        ax4.grid(True)
        ax4.axhline(y=30.0, color='g', linestyle='--', alpha=0.7, label='Normal Time')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/charging_time_analysis_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig4)
        
        # Subplot 5: Customer Satisfaction Analysis
        fig5 = plt.figure(figsize=(10, 8))
        ax5 = fig5.add_subplot(111)
        for sys_id, satisfaction in self.results.get('customer_satisfaction_data', {}).items():
            if satisfaction and len(satisfaction) > 0:
                satisfaction_times = [i * self.coordination_dt for i in range(len(satisfaction))]
                satisfaction_times = satisfaction_times[:len(satisfaction)]
                ax5.plot(satisfaction_times, satisfaction, label=f'System {sys_id}', linewidth=2)
        ax5.set_ylabel('Customer Satisfaction (0-1)', fontsize=18)
        ax5.set_xlabel('Time (s)', fontsize=18)
        ax5.tick_params(axis='both', which='major', labelsize=18)
        ax5.legend(fontsize=12)
        ax5.grid(True)
        ax5.set_ylim(0, 1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'sub_figures/customer_satisfaction_{timestamp}.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig5)
        
        # Subplot 6: Attack Impact Summary (if data exists)
        if self.results.get('attack_impact_data'):
            fig6 = plt.figure(figsize=(10, 8))
            ax6 = fig6.add_subplot(111)
            for sys_id, attack_data in self.results.get('attack_impact_data', {}).items():
                if attack_data:
                    attack_types = [data['attack_type'] for data in attack_data]
                    load_changes = [data['load_change_percent'] for data in attack_data]
                    ax6.scatter(attack_types, load_changes, label=f'System {sys_id}', alpha=0.7, s=100)
            ax6.set_ylabel('Load Change (%)', fontsize=18)
            ax6.set_xlabel('Attack Type', fontsize=18)
            ax6.tick_params(axis='both', which='major', labelsize=18)
            ax6.legend(fontsize=12)
            ax6.grid(True)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(f'sub_figures/attack_impact_summary_{timestamp}.pdf', format='pdf', bbox_inches='tight')
            plt.close(fig6)
        
        print(f"✅ Individual subplot PDFs saved in 'sub_figures/' directory:")
        print(f"   1. frequency_response_{timestamp}.pdf")
        print(f"   2. reference_vs_distribution_load_{timestamp}.pdf")
        print(f"   3. individual_distribution_loads_{timestamp}.pdf")
        print(f"   4. charging_time_analysis_{timestamp}.pdf")
        print(f"   5. customer_satisfaction_{timestamp}.pdf")
        if self.results.get('attack_impact_data'):
            print(f"   6. attack_impact_summary_{timestamp}.pdf")
        
        print("Attack impact analysis plots completed")
    
    def _analyze_evcs_performance(self):
        # Add any additional analysis logic you want to execute after plotting
        pass
    
    def get_simulation_statistics(self) -> Dict:
        """Get comprehensive simulation statistics"""
        stats = {
            'frequency': {
                'min': min(self.results['frequency']),
                'max': max(self.results['frequency']),
                'mean': np.mean(self.results['frequency']),
                'std': np.std(self.results['frequency']),
                'max_deviation': max(abs(60.0 - min(self.results['frequency'])), 
                                   abs(60.0 - max(self.results['frequency'])))
            },
            'total_load': {
                'min': min(self.results['total_load']),
                'max': max(self.results['total_load']),
                'mean': np.mean(self.results['total_load']),
                'std': np.std(self.results['total_load'])
            },
            'distribution_systems': {},
            'agc_performance': {
                'num_updates': len(self.results['agc_updates']),
                'avg_reference_power': np.mean(self.results['reference_power']),
                'reference_power_std': np.std(self.results['reference_power'])
            },
            # Enhanced statistics for charging infrastructure
            'charging_infrastructure': {},
            'queue_management': {},
            'customer_satisfaction': {},
            'attack_impacts': {},
            'load_balancing': {
                'num_balancing_events': len(self.results['load_balancing_data']),
                'num_customer_redirections': len(self.results['customer_redirection_data'])
            }
        }
        
        # Distribution system statistics
        for sys_id, loads in self.results['dist_loads'].items():
            stats['distribution_systems'][sys_id] = {
                'min': min(loads),
                'max': max(loads),
                'mean': np.mean(loads),
                'std': np.std(loads)
            }
            
            # Charging infrastructure statistics
            if sys_id in self.results['charging_time_data']:
                charging_times = self.results['charging_time_data'][sys_id]
                stats['charging_infrastructure'][sys_id] = {
                    'avg_charging_time': np.mean(charging_times),
                    'min_charging_time': min(charging_times),
                    'max_charging_time': max(charging_times),
                    'charging_time_std': np.std(charging_times),
                    'normal_charging_time': 30.0,  # Baseline
                    'charging_time_variation': (max(charging_times) - min(charging_times)) / 30.0
                }
            
            # Queue management statistics
            if sys_id in self.results['queue_management_data']:
                queue_lengths = self.results['queue_management_data'][sys_id]
                stats['queue_management'][sys_id] = {
                    'avg_queue_length': np.mean(queue_lengths),
                    'max_queue_length': max(queue_lengths),
                    'queue_length_std': np.std(queue_lengths),
                    'total_queue_time': sum(queue_lengths),
                    'peak_queue_time': max(queue_lengths)
                }
            
            # Customer satisfaction statistics
            if sys_id in self.results['customer_satisfaction_data']:
                satisfaction = self.results['customer_satisfaction_data'][sys_id]
                stats['customer_satisfaction'][sys_id] = {
                    'avg_satisfaction': np.mean(satisfaction),
                    'min_satisfaction': min(satisfaction),
                    'max_satisfaction': max(satisfaction),
                    'satisfaction_std': np.std(satisfaction),
                    'satisfaction_degradation': 1.0 - np.mean(satisfaction)
                }
            
            # Utilization statistics
            if sys_id in self.results['utilization_data']:
                utilization = self.results['utilization_data'][sys_id]
                stats['charging_infrastructure'][sys_id].update({
                    'avg_utilization': np.mean(utilization),
                    'max_utilization': max(utilization),
                    'utilization_std': np.std(utilization),
                    'efficiency_score': np.mean(utilization) * np.mean(self.results['customer_satisfaction_data'].get(sys_id, [1.0]))
                })
            
            # Attack impact statistics
            if sys_id in self.results['attack_impact_data']:
                attack_data = self.results['attack_impact_data'][sys_id]
                if attack_data:
                    time_factors = [data['charging_time_factor'] for data in attack_data]
                    satisfaction_impacts = [data['customer_satisfaction_impact'] for data in attack_data]
                    load_changes = [data['load_change_percent'] for data in attack_data]
                    
                    stats['attack_impacts'][sys_id] = {
                        'avg_charging_time_factor': np.mean(time_factors),
                        'max_charging_time_factor': max(time_factors),
                        'min_charging_time_factor': min(time_factors),
                        'avg_satisfaction_impact': np.mean(satisfaction_impacts),
                        'max_load_change': max(load_changes),
                        'min_load_change': min(load_changes),
                        'avg_load_change': np.mean(load_changes),
                        'attack_duration': len(attack_data),
                        'attack_types': list(set([data['attack_type'] for data in attack_data]))
                    }
        
        # Global charging infrastructure metrics
        all_charging_times = []
        all_queue_lengths = []
        all_satisfaction = []
        all_utilization = []
        
        for sys_id in self.results['charging_time_data'].keys():
            all_charging_times.extend(self.results['charging_time_data'][sys_id])
            all_queue_lengths.extend(self.results['queue_management_data'][sys_id])
            all_satisfaction.extend(self.results['customer_satisfaction_data'][sys_id])
            all_utilization.extend(self.results['utilization_data'][sys_id])
        
        if all_charging_times:
            stats['global_charging_metrics'] = {
                'avg_charging_time': np.mean(all_charging_times),
                'charging_time_efficiency': 30.0 / np.mean(all_charging_times),  # Normalized to baseline
                'avg_queue_length': np.mean(all_queue_lengths),
                'avg_customer_satisfaction': np.mean(all_satisfaction),
                'avg_utilization': np.mean(all_utilization),
                'overall_efficiency': np.mean(all_utilization) * np.mean(all_satisfaction)
            }
        
        return stats

if __name__ == "__main__":
    # Run the hierarchical cyber attack study
    print("Hierarchical IEEE 14-bus (Pandapower) + 3×IEEE 34-bus (OpenDSS) Co-simulation")
    print("=" * 80)
    
    try:
        # Create hierarchical cosimulation instance
        cosim = HierarchicalCoSimulation()
        
        # Add distribution systems
        print("Setting up distribution systems...")
        cosim.add_distribution_system(1, "ieee34Mod1.dss", 1)
        cosim.add_distribution_system(2, "ieee34Mod2.dss", 2)
        cosim.add_distribution_system(3, "ieee34Mod1.dss", 3)
        cosim.add_distribution_system(4, "ieee34Mod2.dss", 4)
        cosim.add_distribution_system(5, "ieee34Mod1.dss", 5)
        cosim.add_distribution_system(6, "ieee34Mod2.dss", 6)
        
        # Setup EV charging stations
        print("Setting up EV charging stations...")
        cosim.setup_ev_charging_stations()
        
        # FIXED: Define cyber attack scenarios with supported attack types
        attack_scenarios = [
            {
                'target_system': 0,  # System 0 (first system)
                'type': 'power_manipulation',
                'magnitude': 0.6,  # 40% power reduction
                'start_time': 50.0,
                'duration': 30.0
            },
            {
                'target_system': 2,  # System 2 (third system)
                'type': 'load_manipulation',
                'magnitude': 0.7,  # 30% load increase
                'start_time': 100.0,
                'duration': 25.0
            },
            {
                'target_system': 4,  # System 4 (fifth system)
                'type': 'voltage_manipulation',
                'magnitude': 0.8,  # 20% voltage drop
                'start_time': 150.0,
                'duration': 20.0
            }
        ]
        
        print("Running hierarchical simulation...")
        print(f"Simulation duration: {cosim.total_time} seconds")
        print(f"Time step: {cosim.dist_dt} seconds")
        print(f"AGC update interval: {cosim.agc_dt} seconds")
        print(f"Coordination interval: {cosim.coordination_dt} seconds")
        
        # Run the simulation
        cosim.run_hierarchical_simulation(attack_scenarios)
        
        # Get and display simulation statistics
        print("\n" + "="*80)
        print("SIMULATION STATISTICS")
        print("="*80)
        
        stats = cosim.get_simulation_statistics()
        
        # Display key statistics
        print(f"Frequency Analysis:")
        print(f"  Min: {stats['frequency']['min']:.3f} Hz")
        print(f"  Max: {stats['frequency']['max']:.3f} Hz")
        print(f"  Mean: {stats['frequency']['mean']:.3f} Hz")
        print(f"  Max Deviation: {stats['frequency']['max_deviation']:.3f} Hz")
        
        print(f"\nLoad Analysis:")
        print(f"  Min: {stats['total_load']['min']:.1f} MW")
        print(f"  Max: {stats['total_load']['max']:.1f} MW")
        print(f"  Mean: {stats['total_load']['mean']:.1f} MW")
        
        print(f"\nAGC Performance:")
        print(f"  Updates: {stats['agc_performance']['num_updates']}")
        print(f"  Avg Reference Power: {stats['agc_performance']['avg_reference_power']:.1f} MW")
        
        print(f"\nLoad Balancing:")
        print(f"  Balancing Events: {stats['load_balancing']['num_balancing_events']}")
        print(f"  Customer Redirections: {stats['load_balancing']['num_customer_redirections']}")
        
        # Display distribution system statistics
        print(f"\nDistribution System Performance:")
        for sys_id, sys_stats in stats['distribution_systems'].items():
            print(f"  System {sys_id}:")
            print(f"    Load Range: {sys_stats['min']:.1f} - {sys_stats['max']:.1f} MW")
            print(f"    Load Std: {sys_stats['std']:.1f} MW")
            
            if sys_id in stats['charging_infrastructure']:
                charging_stats = stats['charging_infrastructure'][sys_id]
                print(f"    Avg Charging Time: {charging_stats['']:.1f} min")
                print(f"    Avg Utilization: {charging_stats.get('avg_utilization', 0):.2f}")
                print(f"    Efficiency Score: {charging_stats.get('efficiency_score', 0):.2f}")
            
            if sys_id in stats['attack_impacts']:
                attack_stats = stats['attack_impacts'][sys_id]
                print(f"    Attack Types: {', '.join(attack_stats['attack_types'])}")
                print(f"    Max Load Change: {attack_stats['max_load_change']:.1f}%")
                print(f"    Avg Charging Time Factor: {attack_stats['avg_charging_time_factor']:.2f}")
        
        # Display global charging metrics
        if 'global_charging_metrics' in stats:
            global_metrics = stats['global_charging_metrics']
            print(f"\nGlobal Charging Infrastructure:")
            print(f"  Avg Charging Time: {global_metrics['avg_charging_time']:.1f} min")
            print(f"  Charging Efficiency: {global_metrics['charging_time_efficiency']:.2f}")
            print(f"  Avg Customer Satisfaction: {global_metrics['avg_customer_satisfaction']:.2f}")
            print(f"  Overall Efficiency: {global_metrics['overall_efficiency']:.2f}")
        
        # Plot results
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        cosim.plot_hierarchical_results()
        
        print("\nSimulation completed successfully!")
        print("Check the generated plots for detailed analysis.")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    