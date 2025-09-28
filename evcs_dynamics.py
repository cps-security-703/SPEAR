import opendssdirect as dss
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple

@dataclass
class EVCSParameters:
    """EVCS specifications - Updated for Realistic Constraints"""
    # Base Reference Values
    rated_voltage: float = 400.0  # V (base reference voltage)
    rated_current: float = 100.0  # A (base reference current)
    rated_power: float = 50.0     # kW (400V × 100A = 40kW base power)
    
    # Voltage and Current Limits
    max_voltage: float = 500.0    # V (maximum voltage limit)
    min_voltage: float = 300.0    # V (minimum voltage limit)
    max_current: float = 150.0    # A (maximum current limit)
    min_current: float = 50.0     # A (minimum current limit)
    
    # Power Limits
    max_power: float = 75.0       # kW (500V × 150A = 75kW maximum)
    min_power: float = 15.0       # kW (300V × 50A = 15kW minimum)
    
    # Battery and Efficiency Parameters
    capacity: float = 50.0        # kWh
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.92
    min_soc: float = 0.1         # 10%
    max_soc: float = 0.9         # 90%
    disconnect_soc: float = 0.90  # 90% - SOC threshold for EV disconnection
    voltage_bandwidth: float = 10.0  # V, for voltage regulation
    current_bandwidth: float = 5.0   # A, for current regulation

class EVCSController:
    """Individual EVCS Controller with power electronics dynamics"""
    
    def __init__(self, evcs_id: str, params: EVCSParameters):
        self.evcs_id = evcs_id
        self.params = params
        self.pinn_training_mode = False  # Flag to reduce logging during PINN training
        self.soc = np.random.uniform(0.2, 0.4)  # Random initial SOC 20-40%
        
        # Control parameters for voltage and current loops
        self.kp_voltage = 1.2
        self.ki_voltage = 0.15
        self.kp_current = 0.8
        self.ki_current = 0.25
        self.kp_power = 0.5
        self.ki_power = 0.1
        
        # Internal states
        self.voltage_error_integral = 0.0
        self.current_error_integral = 0.0
        self.power_error_integral = 0.0
        
        # References from CMS
        self.voltage_reference = 400.0  # V (DC side) - Updated to rated voltage
        self.current_reference = 0.0    # A
        self.power_reference = 0.0      # kW
        
        # Measured values (simulated dynamics)
        self.voltage_measured = 400.0   # V - Updated to rated voltage
        self.current_measured = 0.0     # A
        self.power_measured = 0.0       # kW
        
        # AC side measurements
        self.ac_voltage_rms = 7200.0    # V (line-neutral)
        self.ac_current_rms = 0.0       # A
        self.grid_frequency = 60.0      # Hz
        self.pll_angle = 0.0           # radians
        
        # Power electronics states
        self.dc_link_voltage = 400.0    # V - Updated to rated voltage
        self.switching_frequency = 10000 # Hz
        self.filter_time_constant = 0.02 # s (50 Hz cutoff)
        
        # Power flow coupling variables
        self.dc_link_capacitance = 0.1  # F (DC link capacitor)
        self.dc_link_power_demand = 0.0  # kW (power demanded by DC-DC converter)
        self.ac_dc_efficiency = 0.98    # AC-DC converter efficiency
        self.dc_dc_efficiency = 0.96    # DC-DC converter efficiency
        
        # Power balance tracking
        self.power_balance_error = 0.0  # kW
        self.total_efficiency = self.ac_dc_efficiency * self.dc_dc_efficiency
        
    def set_references(self, voltage_ref: float, current_ref: float, power_ref: float):
        """Set references from CMS"""
        self.voltage_reference = voltage_ref
        self.current_reference = current_ref
        self.power_reference = power_ref
    
    def park_transformation(self, va: float, vb: float, vc: float, theta: float) -> Tuple[float, float]:
        """Park transformation for AC-DC conversion"""
        vd = (2/3) * (va * np.cos(theta) + vb * np.cos(theta - 2*np.pi/3) + vc * np.cos(theta + 2*np.pi/3))
        vq = (2/3) * (-va * np.sin(theta) - vb * np.sin(theta - 2*np.pi/3) - vc * np.sin(theta + 2*np.pi/3))
        return vd, vq
    
    def evcs_dynamics_system(self, t, x, grid_voltage_rms):
        """
        Complete EVCS dynamics system for solve_ivp with integrated converter dynamics
        State vector x = [pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage]
        """
        try:
            pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage = x
            
            # Validate inputs to prevent numerical issues
            if not all(np.isfinite([pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage])):
                # Return zero derivatives for invalid states
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Clamp values to realistic EVCS ranges
            pll_angle = pll_angle % (2 * np.pi)
            voltage_integral = np.clip(voltage_integral, -1000, 1000)
            current_integral = np.clip(current_integral, -1000, 1000)
            current_measured = np.clip(current_measured, 0, self.params.max_current)  # Use max_current instead of rated_current * 1.5
            soc = np.clip(soc, 0.0, 1.0)
            dc_link_voltage = np.clip(dc_link_voltage, self.params.min_voltage, self.params.max_voltage)  # 300-500V realistic range
            
            # PLL dynamics
            va = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle)
            vb = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle - 2*np.pi/3)
            vc = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle + 2*np.pi/3)
            
            v_alpha = va
            v_beta = (2/np.sqrt(3)) * (vb - vc/2)
            angle_error = np.arctan2(v_beta, v_alpha) - pll_angle
            
            # PI controller for PLL (reduced gains to prevent instability)
            kp_pll, ki_pll = 10.0, 100.0  # Much lower gains for stability
            frequency_deviation = kp_pll * angle_error + ki_pll * angle_error
            # Limit frequency deviation to reasonable range
            frequency_deviation = np.clip(frequency_deviation, -5.0, 5.0)
            grid_frequency = 60.0 + frequency_deviation
            
            # PLL angle derivative
            dpll_angle_dt = 2 * np.pi * grid_frequency
            
            # AC-DC Converter Dynamics (integrated)
            # Calculate AC side power and current
            ac_power = current_measured * grid_voltage_rms * np.sqrt(3)  # 3-phase power
            ac_current = current_measured
            
            # DC side power (with converter efficiency)
            converter_efficiency = 0.95
            dc_power = ac_power * converter_efficiency
            dc_current = dc_power / max(dc_link_voltage, 100.0)  # Avoid division by zero
            
            # DC link voltage dynamics (capacitor charging/discharging)
            dc_link_capacitance = 0.1  # F (100mF) - larger capacitance for stability
            # Power balance: P_in - P_out = C * V * dV/dt
            power_out = self.power_reference * 1000.0  # Convert kW to W
            power_in = dc_power
            # Add damping to prevent oscillations
            power_balance = power_in - power_out
            # Limit rate of change to prevent numerical instability
            max_voltage_rate = 1000.0  # V/s maximum rate
            ddc_link_voltage_dt = np.clip(power_balance / (dc_link_capacitance * max(dc_link_voltage, self.params.min_voltage)), 
                                        -max_voltage_rate, max_voltage_rate)
            
            # Voltage control dynamics (using actual DC link voltage)
            voltage_error = self.voltage_reference - dc_link_voltage
            dvoltage_integral_dt = voltage_error
            
            # Current control dynamics  
            voltage_control_output = (self.kp_voltage * voltage_error + 
                                     self.ki_voltage * voltage_integral)
            
            # Current limiting with anti-windup - using max_current for realistic constraints
            current_limit = min(self.current_reference, self.params.max_current)
            if voltage_control_output > current_limit:
                voltage_control_output = current_limit
                dvoltage_integral_dt = 0  # Anti-windup
            
            current_error = voltage_control_output - current_measured
            dcurrent_integral_dt = current_error
            
            # First-order current dynamics with converter time constant
            converter_time_constant = 0.1  # 100ms time constant for stability
            dcurrent_measured_dt = (voltage_control_output - current_measured) / converter_time_constant
            # Limit current rate of change
            max_current_rate = 50.0  # A/s maximum rate
            dcurrent_measured_dt = np.clip(dcurrent_measured_dt, -max_current_rate, max_current_rate)
            
            # SOC dynamics (integrated with DC-DC converter efficiency)
            # DC-DC converter from DC link to battery
            battery_voltage = soc * 200.0 + 300.0  # 300-500V based on SOC (realistic EVCS range)
            dcdc_efficiency = 0.95
            battery_power = power_out * dcdc_efficiency / 1000.0  # Convert back to kW
            
            # SOC rate based on actual battery power
            charging_efficiency = self.params.efficiency_charge if battery_power > 0 else self.params.efficiency_discharge
            dsoc_dt = battery_power * charging_efficiency / (self.params.capacity * 3600)
            
            # Clamp SOC rate to prevent unrealistic charging speeds
            max_soc_rate = 0.1 / 3600  # Max 10% per hour
            dsoc_dt = np.clip(dsoc_dt, -max_soc_rate, max_soc_rate)
            
            return [dpll_angle_dt, dvoltage_integral_dt, dcurrent_integral_dt, 
                    dcurrent_measured_dt, dsoc_dt, ddc_link_voltage_dt]
                    
        except Exception as e:
            # Return zero derivatives if any calculation fails
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def pll_update(self, v_abc: List[float], dt: float):
        """Phase-Locked Loop for grid synchronization (legacy method for compatibility)"""
        # This method is kept for backward compatibility but solve_ivp is preferred
        va, vb, vc = v_abc
        v_alpha = va
        v_beta = (2/np.sqrt(3)) * (vb - vc/2)
        
        # Calculate angle
        angle_error = np.arctan2(v_beta, v_alpha) - self.pll_angle
        
        # PI controller for PLL
        kp_pll = 100.0
        ki_pll = 10000.0
        
        frequency_deviation = kp_pll * angle_error + ki_pll * angle_error * dt
        self.grid_frequency = 60.0 + frequency_deviation
        
        # Update PLL angle
        self.pll_angle += 2 * np.pi * self.grid_frequency * dt
        self.pll_angle = self.pll_angle % (2 * np.pi)
    
    def ac_dc_converter_dynamics(self, grid_voltage_rms: float, dt: float) -> Dict:
        """AC-DC converter dynamics with proper DC link power coupling"""
        
        # Update AC voltage measurement
        self.ac_voltage_rms = grid_voltage_rms
        
        # Three-phase voltages (assuming balanced)
        va = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle)
        vb = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle - 2*np.pi/3)
        vc = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle + 2*np.pi/3)
        
        # Update PLL
        self.pll_update([va, vb, vc], dt)
        
        # Park transformation
        vd, vq = self.park_transformation(va, vb, vc, self.pll_angle)
        
        # POWER FLOW COUPLING: AC side drives DC side
        # Current reference calculation based on power demand from DC-DC converter
        v_rms_effective = max(grid_voltage_rms, 100.0)  # Avoid division by zero
        
        # AC current needed to supply DC power demand (considering efficiency)
        required_ac_power = max(0.0, self.dc_link_power_demand / self.ac_dc_efficiency)  # kW
        desired_ac_current = (required_ac_power * 1000) / (3 * v_rms_effective) if required_ac_power > 0 else 0.0  # A (RMS)
        
        # Limit current to reasonable values
        max_ac_current = self.params.max_power * 1000 / (3 * v_rms_effective)  # Max current based on EVCS rating
        desired_ac_current = min(desired_ac_current, max_ac_current)
        
        # Current control loop
        current_error = desired_ac_current - self.ac_current_rms
        self.current_error_integral += current_error * dt
        
        # PI current controller with anti-windup
        current_control_output = (self.kp_current * current_error + 
                                 self.ki_current * self.current_error_integral)
        
        # Anti-windup for current integral
        if current_control_output > max_ac_current:
            current_control_output = max_ac_current
            self.current_error_integral -= current_error * dt  # Prevent windup
        
        # Update measured current (first-order dynamics with improved response)
        time_constant = max(self.filter_time_constant, dt * 2)  # Ensure stability
        current_change = (current_control_output - self.ac_current_rms) * dt / time_constant
        
        # Limit current change rate to prevent oscillation
        max_current_change = max_ac_current * dt / 0.1  # Max change in 100ms
        current_change = np.clip(current_change, -max_current_change, max_current_change)
        
        self.ac_current_rms += current_change
        self.ac_current_rms = max(0, min(self.ac_current_rms, max_ac_current))  # Clamp current
        
        # Calculate actual AC power input
        self.power_measured = 3 * grid_voltage_rms * self.ac_current_rms / 1000  # kW
        
        # Calculate DC link power available (after AC-DC conversion efficiency)
        dc_link_power_available = self.power_measured * self.ac_dc_efficiency  # kW
        
        # Update DC link voltage dynamics (capacitor charging/discharging)
        power_balance = dc_link_power_available - self.dc_link_power_demand  # kW
        
        # DC link voltage dynamics: C * V * dV/dt = P
        if abs(self.dc_link_voltage) > 1.0:  # Avoid division by very small numbers
            ddc_link_voltage_dt = (power_balance * 1000) / (self.dc_link_capacitance * self.dc_link_voltage)  # V/s
            self.dc_link_voltage += ddc_link_voltage_dt * dt
            # Clamp DC link voltage to realistic range
            self.dc_link_voltage = np.clip(self.dc_link_voltage, 
                                         self.params.min_voltage * 0.9,  # Allow 10% below min
                                         self.params.max_voltage * 1.1)  # Allow 10% above max
        
        return {
            'ac_voltage_rms': self.ac_voltage_rms,
            'ac_current_rms': self.ac_current_rms,
            'power_measured': self.power_measured,
            'dc_link_power_available': dc_link_power_available,
            'dc_link_voltage': self.dc_link_voltage,
            'power_balance': power_balance,
            'grid_frequency': self.grid_frequency,
            'pll_angle': np.degrees(self.pll_angle)
        }
    
    def dc_dc_converter_dynamics(self, dt: float) -> Dict:
        """DC-DC converter dynamics with proper DC link power coupling"""
        
        # POWER FLOW COUPLING: Calculate maximum current based on available DC link power
        max_current_from_dc_link = self.params.max_current  # Default to physical limit
        if self.dc_link_voltage > 10.0:  # Avoid division by very small voltage
            # Maximum current limited by DC link power capability and voltage
            max_power_from_dc_link = self.params.max_power  # Max power rating of EVCS
            max_current_from_dc_link = min(
                self.params.max_current,  # Physical current limit
                max_power_from_dc_link * 1000 / self.dc_link_voltage  # Power limit at current DC voltage
            )
        
        # Voltage control loop
        voltage_error = self.voltage_reference - self.voltage_measured
        self.voltage_error_integral += voltage_error * dt
        
        # PI voltage controller
        voltage_control_output = (self.kp_voltage * voltage_error + 
                                 self.ki_voltage * self.voltage_error_integral)
        
        # Current limiting - consider both reference limit, physical limit, and DC link capability
        current_limit = min(
            self.current_reference,
            self.params.max_current,
            max_current_from_dc_link
        )
        
        # Use current reference directly if voltage control is not aggressive enough
        if self.current_reference > 0:
            # Blend voltage control output with current reference for better response
            voltage_control_output = max(voltage_control_output, self.current_reference * 0.8)
        
        # Anti-windup for voltage controller
        if voltage_control_output > current_limit:
            voltage_control_output = current_limit
            self.voltage_error_integral -= voltage_error * dt
        
        # Current limiting based on available DC link voltage
        if self.dc_link_voltage < self.params.min_voltage:
            # Reduce current if DC link voltage is too low
            voltage_ratio = self.dc_link_voltage / self.params.min_voltage
            voltage_control_output *= max(0.1, voltage_ratio)  # Minimum 10% current
        
        # Update measured current (first-order dynamics with faster response)
        time_constant = 0.01  # 10ms time constant for faster response
        self.current_measured += (voltage_control_output - self.current_measured) * dt / time_constant
        self.current_measured = max(0, min(self.current_measured, current_limit))
        
        # If still very low current, boost it towards reference (startup assistance)
        if self.current_measured < self.current_reference * 0.1 and self.current_reference > 1.0:
            self.current_measured = min(self.current_reference * 0.5, current_limit)
        
        # Update measured voltage based on SOC with realistic voltage range (300V-500V)
        # Battery voltage varies with SOC
        battery_voltage_base = self.params.min_voltage + self.soc * (self.params.max_voltage - self.params.min_voltage)
        
        # Add voltage drop due to current (internal resistance effect)
        internal_resistance = 0.1  # Ohms (typical battery internal resistance)
        voltage_drop = self.current_measured * internal_resistance
        self.voltage_measured = battery_voltage_base - voltage_drop
        
        # Ensure voltage stays within bounds
        self.voltage_measured = np.clip(self.voltage_measured, 
                                      self.params.min_voltage, 
                                      self.params.max_voltage)
        
        # Calculate actual delivered power to battery
        dc_power = self.voltage_measured * self.current_measured / 1000  # kW
        
        # UPDATE DC LINK POWER DEMAND (this drives the AC-DC converter)
        # Power demanded from DC link = delivered power / DC-DC efficiency
        self.dc_link_power_demand = dc_power / self.dc_dc_efficiency if dc_power > 0 else 0.0
        
        # Calculate power balance and efficiency
        power_loss_dc_dc = self.dc_link_power_demand - dc_power  # Power lost in DC-DC converter
        
        # FIXED: Efficiency should be between 0 and 1 (0% to 100%)
        if self.dc_link_power_demand > 0.001:
            # Efficiency = output power / input power (should be < 1.0 for losses)
            actual_efficiency = dc_power / self.dc_link_power_demand
            # Clamp efficiency to realistic range [0.5, 0.99] (50% to 99%)
            actual_efficiency = np.clip(actual_efficiency, 0.5, 0.99)
        else:
            actual_efficiency = 0.95  # Default efficiency when no power flow
        
        return {
            'voltage_measured': self.voltage_measured,
            'current_measured': self.current_measured,
            'dc_power': dc_power,
            'dc_link_power_demand': self.dc_link_power_demand,
            'max_current_from_dc_link': max_current_from_dc_link,
            'power_loss_dc_dc': power_loss_dc_dc,
            'dc_dc_efficiency_actual': actual_efficiency,
            'soc': self.soc
        }
    
    def update_soc(self, power_kW: float, dt: float):
        """Update State of Charge based on power flow"""
        if power_kW > 0:  # Charging
            energy_kwh = power_kW * dt / 3600 * self.params.efficiency_charge
        else:  # Discharging
            energy_kwh = power_kW * dt / 3600 / self.params.efficiency_discharge
        
        self.soc += energy_kwh / self.params.capacity
        self.soc = np.clip(self.soc, self.params.min_soc, self.params.max_soc)
    
    def _sanitize_states(self):
        """Sanitize all state variables to prevent numerical issues"""
        # Clamp PLL angle to [0, 2π] range
        if hasattr(self, 'pll_angle'):
            self.pll_angle = self.pll_angle % (2 * np.pi)
            if not np.isfinite(self.pll_angle):
                self.pll_angle = 0.0
        
        # Clamp integral terms to prevent windup
        max_integral = 1000.0
        if hasattr(self, 'voltage_error_integral'):
            self.voltage_error_integral = np.clip(self.voltage_error_integral, -max_integral, max_integral)
            if not np.isfinite(self.voltage_error_integral):
                self.voltage_error_integral = 0.0
                
        if hasattr(self, 'current_error_integral'):
            self.current_error_integral = np.clip(self.current_error_integral, -max_integral, max_integral)
            if not np.isfinite(self.current_error_integral):
                self.current_error_integral = 0.0
        
        # Clamp measured current to realistic range
        if hasattr(self, 'current_measured'):
            self.current_measured = np.clip(self.current_measured, 0.0, 500.0)  # 0-500A range
            if not np.isfinite(self.current_measured):
                self.current_measured = 0.0
        
        # Clamp SOC to valid range [0.01, 0.99] to avoid boundary issues
        if hasattr(self, 'soc'):
            self.soc = np.clip(self.soc, 0.01, 0.99)
            if not np.isfinite(self.soc):
                self.soc = 0.5  # Default to 50%
        
        # Clamp DC link voltage to realistic range
        if hasattr(self, 'dc_link_voltage'):
            self.dc_link_voltage = np.clip(self.dc_link_voltage, 100.0, 1000.0)  # 100V-1000V range
            if not np.isfinite(self.dc_link_voltage):
                self.dc_link_voltage = 400.0  # Default to 400V
    
    def update_dynamics_with_solve_ivp(self, grid_voltage_rms: float, dt: float) -> Dict:
        """Enhanced dynamics update using solve_ivp for accurate integration"""
        
        # Pre-condition states to prevent numerical issues
        self._sanitize_states()
        
        # State vector: [pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage]
        x0 = [self.pll_angle, self.voltage_error_integral, self.current_error_integral, 
              self.current_measured, self.soc, self.dc_link_voltage]
        
        t_span = (0, dt)
        
        try:
            # Enhanced validation with specific error reporting
            if any(not np.isfinite(val) for val in x0):
                state_names = ['pll_angle', 'voltage_integral', 'current_integral', 
                              'current_measured', 'soc', 'dc_link_voltage']
                invalid_states = []
                for name, val in zip(state_names, x0):
                    if not np.isfinite(val):
                        invalid_states.append(f"{name}={val}")
                print(f"EVCS {self.evcs_id}: Invalid states: {', '.join(invalid_states)} - using Euler fallback")
                return self._update_dynamics_euler(grid_voltage_rms, dt)
            
            # Use more robust solver settings to prevent recursion depth errors
            sol = solve_ivp(
                lambda t, x: self.evcs_dynamics_system(t, x, grid_voltage_rms),
                t_span, x0, 
                method='LSODA',  # Adaptive method for stiff/non-stiff systems
                rtol=1e-3,       # More relaxed tolerance
                atol=1e-5,       # More relaxed absolute tolerance
                max_step=dt/5,   # Larger max step size
                first_step=dt/100,  # Small initial step
                dense_output=False  # Disable dense output to save memory
            )
            
            if sol.success:
                # Update states with solved values
                x_new = sol.y[:, -1]
                self.pll_angle, self.voltage_error_integral, self.current_error_integral, \
                self.current_measured, self.soc, self.dc_link_voltage = x_new
                
                # Constrain values to realistic EVCS limits
                self.pll_angle = self.pll_angle % (2 * np.pi)
                self.current_measured = max(self.params.min_current, min(self.current_measured, self.params.max_current))
                self.soc = np.clip(self.soc, self.params.min_soc, self.params.max_soc)
                
                # Calculate derived quantities with realistic voltage range
                # Map SOC to voltage range 300V-500V instead of just using rated_voltage
                self.voltage_measured = self.params.min_voltage + self.soc * (self.params.max_voltage - self.params.min_voltage)
                self.power_measured = self.voltage_measured * self.current_measured / 1000
                self.grid_frequency = 60.0  # Updated within dynamics system
                
                # Calculate AC current from power and grid voltage
                if grid_voltage_rms > 0 and self.power_reference > 0:
                    self.ac_current_rms = (self.power_reference * 1000) / (3 * grid_voltage_rms)
                else:
                    self.ac_current_rms = 0.0
                
                # Update AC voltage
                self.ac_voltage_rms = grid_voltage_rms
                
                return {
                    'voltage_measured': self.voltage_measured,
                    'current_measured': self.current_measured,
                    'power_measured': self.power_measured,
                    'soc': self.soc,
                    'pll_angle': np.degrees(self.pll_angle),
                    'grid_frequency': self.grid_frequency,
                    'ac_voltage_rms': self.ac_voltage_rms,
                    'ac_current_rms': self.ac_current_rms,
                    'total_power': self.power_reference,
                    'integration_method': 'solve_ivp'
                }
            else:
                # Fallback to Euler method if solve_ivp fails
                print(f"EVCS {self.evcs_id}: solve_ivp failed, using Euler fallback")
                return self._update_dynamics_euler(grid_voltage_rms, dt)
                
        except Exception as e:
            print(f"EVCS {self.evcs_id}: solve_ivp error: {e}, using Euler fallback")
            return self._update_dynamics_euler(grid_voltage_rms, dt)
    
    def update_dynamics(self, grid_voltage_rms: float, dt: float, use_solve_ivp: bool = True) -> Dict:
        """Update all EVCS dynamics with option for solve_ivp or Euler method"""
        
        if use_solve_ivp:
            # Use solve_ivp for accurate integration
            return self.update_dynamics_with_solve_ivp(grid_voltage_rms, dt)
        else:
            # Legacy Euler method for fallback
            return self._update_dynamics_euler(grid_voltage_rms, dt)
    
    def _update_dynamics_euler(self, grid_voltage_rms: float, dt: float, simulation_time: float = 0.0) -> Dict:
        """Update all EVCS dynamics with stable power flow coupling (Euler method)"""
        
        # Store simulation time for logging control
        self.simulation_time = simulation_time
        
        # Reset adjustment count periodically to prevent getting stuck
        if not hasattr(self, '_last_reset_time'):
            self._last_reset_time = simulation_time
        elif simulation_time - self._last_reset_time > 10.0:  # Reset every 10 seconds
            if hasattr(self, '_adjustment_count'):
                self._adjustment_count = 0
            self._last_reset_time = simulation_time
        
        # ROBUST APPROACH: Start from power reference and work through the chain with validation
        target_dc_power = min(self.power_reference, self.params.max_power)
        
        # 1. Calculate required DC link power (including DC-DC losses)
        required_dc_link_power = target_dc_power / self.dc_dc_efficiency if target_dc_power > 0 else 0.0
        
        # 2. Calculate required AC power (including AC-DC losses)
        required_ac_power = required_dc_link_power / self.ac_dc_efficiency if required_dc_link_power > 0 else 0.0
        
        # 3. Validate power flow chain consistency
        if target_dc_power > 0.001:
            # Check if the power flow chain makes sense
            expected_total_efficiency = self.ac_dc_efficiency * self.dc_dc_efficiency
            if abs(expected_total_efficiency - 0.94) > 0.1:  # Should be around 94%
                print(f"EVCS {self.evcs_id}: Efficiency mismatch detected: {expected_total_efficiency:.3f}")
                # Reset to default values if efficiency is unrealistic
                self.ac_dc_efficiency = 0.98
                self.dc_dc_efficiency = 0.96
        
        # 3. Set stable DC link power demand with improved coupling
        # Use a moving average to smooth out power demand and reduce oscillations
        if not hasattr(self, '_power_demand_history'):
            self._power_demand_history = []
        
        # Add current demand to history (keep last 5 values)
        self._power_demand_history.append(required_dc_link_power)
        if len(self._power_demand_history) > 5:
            self._power_demand_history.pop(0)
        
        # Use smoothed power demand to reduce oscillations
        smoothed_demand = np.mean(self._power_demand_history)
        self.dc_link_power_demand = smoothed_demand
        
        # 4. Update DC-DC converter with known power target
        dc_results = self.dc_dc_converter_dynamics(dt)
        actual_dc_power = dc_results['dc_power']
        
        # 5. Update AC-DC converter with known AC power target
        ac_results = self.ac_dc_converter_dynamics(grid_voltage_rms, dt)
        actual_ac_power = ac_results['power_measured']
        
        # 6. Calculate actual power flow and efficiency with improved balance
        if actual_ac_power > 0.001:
            # FIXED: System efficiency should be between 0 and 1 (0% to 100%)
            system_efficiency = actual_dc_power / actual_ac_power
            # Clamp efficiency to realistic range [0.4, 0.98] (40% to 98% for total system)
            system_efficiency = np.clip(system_efficiency, 0.4, 0.98)
        else:
            system_efficiency = 0.9  # Default system efficiency when no power flow
            
        total_power_loss = actual_ac_power - actual_dc_power
        
        # IMPROVED: Calculate power balance error with better coupling
        # The error should be small if power flow coupling is working correctly
        power_balance_error = abs(required_dc_link_power - actual_dc_power)
        
        # IMPROVED: More stable power flow coupling with convergence criteria
        adjustment_factor = 0.0  # Initialize adjustment_factor to avoid scope error
        
        # Initialize error tracking if not exists
        if not hasattr(self, '_last_power_error'):
            self._last_power_error = 0.0
            self._error_history = []
            self._adjustment_count = 0
            self._max_adjustments = 10  # Limit adjustments per sample to prevent infinite loops
        
        # Only adjust if error is significant and we haven't exceeded max adjustments
        if power_balance_error > 100.0 and self._adjustment_count < self._max_adjustments:  # Reduced threshold to 100kW
            # Check if error is actually changing (not stuck)
            error_change = abs(power_balance_error - self._last_power_error)
            
            # Only adjust if error is significant and changing
            if error_change > 10.0 or self._adjustment_count == 0:  # Reduced change threshold
                # Calculate adjustment factor based on error magnitude
                if power_balance_error > 500.0:
                    adjustment_factor = 0.05  # 5% for large errors
                elif power_balance_error > 200.0:
                    adjustment_factor = 0.02  # 2% for medium errors
                else:
                    adjustment_factor = 0.01  # 1% for small errors
                
                # Apply adjustment
                if required_dc_link_power > actual_dc_power:
                    self.dc_link_power_demand *= (1.0 + adjustment_factor)
                else:
                    self.dc_link_power_demand *= (1.0 - adjustment_factor)
                
                # Clamp power demand to reasonable bounds
                self.dc_link_power_demand = np.clip(self.dc_link_power_demand, 0.0, self.params.max_power * 1.1)
                
                # Update tracking
                self._adjustment_count += 1
                self._last_power_error = power_balance_error
                self._error_history.append(power_balance_error)
                if len(self._error_history) > 5:  # Keep shorter history
                    self._error_history.pop(0)
                
                # Log adjustments with reduced frequency
                if not self.pinn_training_mode:
                    if self._adjustment_count % 5 == 0:  # Log every 5th adjustment
                        print(f"EVCS {self.evcs_id}: Power coupling adjustment: {adjustment_factor:.3f}, Error: {power_balance_error:.2f}kW")
                else:
                    # During PINN training, suppress most logging
                    if self._adjustment_count % 10 == 0:  # Log every 10th adjustment
                        print(f"EVCS {self.evcs_id}: [PINN Training] Power coupling adjustment: {adjustment_factor:.3f}, Error: {power_balance_error:.2f}kW")
            else:
                # Error is not changing significantly, stop adjusting
                self._adjustment_count = self._max_adjustments
        else:
            # Reset adjustment count if error is small
            if power_balance_error <= 100.0:
                self._adjustment_count = 0
        
        # 7. Update SOC based on actual delivered power
        self.update_soc(actual_dc_power, dt)
        self.power_measured = actual_dc_power
        
        return {
            # AC side results
            **ac_results,
            # DC side results  
            **dc_results,
            # Power flow coupling results
            'total_power': actual_dc_power,
            'ac_power_in': actual_ac_power,
            'dc_power_out': actual_dc_power,
            'total_power_loss': total_power_loss,
            'system_efficiency': system_efficiency,
            'power_balance_error': power_balance_error,
            'total_efficiency_actual': system_efficiency,
            'total_efficiency_design': self.total_efficiency,
            'target_dc_power': target_dc_power,
            'required_dc_link_power': required_dc_link_power,
            'required_ac_power': required_ac_power,
            'integration_method': 'euler_stable'
        }

class ChargingManagementSystem:
    """Advanced Charging Management System for QSTS simulation"""
    
    def __init__(self, evcs_controllers: Dict[str, EVCSController]):
        self.evcs_controllers = evcs_controllers
        self.voltage_limits = {'min': 0.95, 'max': 1.05}  # Per unit
        self.total_power_limit = 300  # kW
        
        # Advanced control parameters
        self.voltage_droop_gain = 100.0  # kW/pu
        self.frequency_droop_gain = 50.0  # kW/Hz
        self.soc_weight = 0.3
        self.voltage_weight = 0.4
        self.load_weight = 0.3
        
    def generate_daily_charging_profile(self, time_hours: float) -> float:
        """Generate realistic daily charging demand profile"""
        
        # Peak charging periods: 7-9 AM, 6-8 PM
        morning_peak = np.exp(-((time_hours - 8)**2) / (2 * 1.5**2))
        evening_peak = np.exp(-((time_hours - 19)**2) / (2 * 2**2))
        
        # Base load throughout the day
        base_load = 0.3 + 0.2 * np.sin(2 * np.pi * time_hours / 24)
        
        # Weekend vs weekday pattern (assuming weekday)
        daily_pattern = 0.4 * morning_peak + 0.6 * evening_peak + base_load
        
        return np.clip(daily_pattern, 0.1, 1.0)
    
    def optimize_charging_qsts(self, time_hours: float, bus_voltages: Dict[str, float], 
                              system_frequency: float = 60.0, pinn_optimizer=None) -> Dict[str, Dict]:
        """Optimize charging using PINN optimizer or fallback to heuristic method"""
        
        if pinn_optimizer is not None:
            return self._optimize_with_pinn(time_hours, bus_voltages, system_frequency, pinn_optimizer)
        else:
            return self._optimize_heuristic(time_hours, bus_voltages, system_frequency)
    
    def _optimize_with_pinn(self, time_hours: float, bus_voltages: Dict[str, float], 
                           system_frequency: float, pinn_optimizer) -> Dict[str, Dict]:
        """PINN-based optimization for intelligent charging control"""
        
        # Get daily charging demand profile
        demand_factor = self.generate_daily_charging_profile(time_hours)
        
        # Updated bus mapping for new EVCS configuration
        evcs_to_bus_mapping = {
            'EVCS1': '890', 'EVCS2': '844', 'EVCS3': '860', 
            'EVCS4': '840', 'EVCS5': '848', 'EVCS6': '830'
        }
        
        references = {}
        
        for evcs_name, controller in self.evcs_controllers.items():
            # Get bus information
            bus_name = evcs_to_bus_mapping.get(evcs_name, '890')
            voltage_pu = bus_voltages.get(bus_name, 1.0)
            
            # Calculate priority factors for PINN input
            voltage_priority = max(0, self.voltage_limits['min'] - voltage_pu)
            urgency_factor = 2.0 - controller.soc  # Higher urgency for low SOC
            
            # Get bus distance (simplified)
            bus_distances = {'890': 0.0, '844': 0.4, '860': 0.7, '840': 1.6, '848': 2.9, '830': 4.0}
            bus_distance = bus_distances.get(bus_name, 1.0)
            
            # Create input sequence for PINN (simplified single-step for real-time)
            input_features = [
                controller.soc,                    # SOC
                voltage_pu,                        # Grid voltage (pu)
                system_frequency,                  # Grid frequency (Hz)
                demand_factor,                     # Demand factor
                voltage_priority,                  # Voltage priority
                urgency_factor,                    # Urgency factor
                time_hours,                        # Time (hours)
                bus_distance,                      # Bus distance (km)
                1.0,                              # Load factor
                controller.power_reference        # Previous power
            ]
            
            # Create sequence for LSTM (repeat current state for sequence length)
            sequence_length = 10  # Match PINN config
            sequence = [input_features] * sequence_length
            
            try:
                # Get PINN optimization
                voltage_ref, current_ref, power_ref = pinn_optimizer.optimize_references_lstm(sequence)
                
                # Apply system constraints
                max_power = getattr(controller.params, 'max_power', 200.0)  # kW
                power_ref = min(power_ref, max_power)
                
                references[evcs_name] = {
                    'power_ref': power_ref,
                    'voltage_ref': voltage_ref,
                    'current_ref': current_ref,
                    'priority': urgency_factor,
                    'voltage_pu': voltage_pu
                }
                
                # Set references in controller
                controller.set_references(voltage_ref, current_ref, power_ref)
                
            except Exception as e:
                print(f"PINN optimization failed for {evcs_name}: {e}, using fallback")
                # Fallback to realistic EVCS heuristic
                if controller.soc < 0.3:
                    power_ref, voltage_ref, current_ref = 60.0, 480.0, 125.0  # Fast charging within limits
                elif controller.soc < 0.7:
                    power_ref, voltage_ref, current_ref = 40.0, 400.0, 100.0  # Rated charging
                else:
                    power_ref, voltage_ref, current_ref = 20.0, 350.0, 60.0   # Trickle charging
                
                references[evcs_name] = {
                    'power_ref': power_ref,
                    'voltage_ref': voltage_ref,
                    'current_ref': current_ref,
                    'priority': urgency_factor,
                    'voltage_pu': voltage_pu
                }
                controller.set_references(voltage_ref, current_ref, power_ref)
        
        return references
    
    def _optimize_heuristic(self, time_hours: float, bus_voltages: Dict[str, float], 
                           system_frequency: float = 60.0) -> Dict[str, Dict]:
        """Legacy heuristic optimization (fallback when PINN not available)"""
        
        # Get daily charging demand profile
        demand_factor = self.generate_daily_charging_profile(time_hours)
        
        # Calculate available power considering system constraints
        total_available_power = self.total_power_limit * demand_factor
        
        # Ensure minimum power for low SOC EVCS even during low demand periods
        min_emergency_power = 100.0  # kW minimum for critical charging
        if total_available_power < min_emergency_power:
            total_available_power = min_emergency_power
        
        # Updated bus mapping for new EVCS configuration
        evcs_to_bus_mapping = {
            'EVCS1': '890', 'EVCS2': '844', 'EVCS3': '860', 
            'EVCS4': '840', 'EVCS5': '848', 'EVCS6': '830'
        }
        
        # Sort EVCS by composite priority
        evcs_priority = []
        for evcs_name, controller in self.evcs_controllers.items():
            # Get correct bus name from mapping
            bus_name = evcs_to_bus_mapping.get(evcs_name, '890')
            
            # Multi-objective priority calculation
            soc_priority = (1 - controller.soc)  # Lower SOC = higher priority
            
            if bus_name in bus_voltages:
                voltage_pu = bus_voltages[bus_name]
                # Voltage support priority (lower voltage = higher priority for grid support)
                voltage_priority = max(0, self.voltage_limits['min'] - voltage_pu)
            else:
                voltage_priority = 0
                voltage_pu = 1.0
            
            # Frequency support
            frequency_priority = max(0, 60.0 - system_frequency) / 60.0
            
            # Composite priority
            composite_priority = (self.soc_weight * soc_priority + 
                                self.voltage_weight * voltage_priority +
                                0.1 * frequency_priority)
            
            evcs_priority.append((evcs_name, controller, composite_priority, bus_name, voltage_pu))
        
        # Sort by priority (highest first)
        evcs_priority.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate power and set references
        references = {}
        remaining_power = total_available_power
        
        for evcs_name, controller, priority, bus_name, voltage_pu in evcs_priority:
            
            # Base power allocation based on SOC and priority - using realistic EVCS constraints
            if controller.soc < 0.2:
                base_power = 70  # kW - Fast charging for low SOC (within 75kW max)
                target_voltage = 500.0  # V (maximum voltage)
                target_current = 140.0  # A
            elif controller.soc < 0.8:
                base_power = 40  # kW - Normal charging (rated power)
                target_voltage = 400.0  # V (rated voltage)
                target_current = 100.0   # A (rated current)
            else:
                base_power = 20  # kW - Trickle charging (above minimum)
                target_voltage = 350.0  # V
                target_current = 60.0   # A
            
            # Voltage-based power modification
            if voltage_pu < self.voltage_limits['min']:
                # Low voltage - reduce charging to support grid
                power_factor = 0.4
            elif voltage_pu > self.voltage_limits['max']:
                # High voltage - can increase charging
                power_factor = 1.3
            else:
                power_factor = 1.0
            
            # Apply demand factor
            base_power *= demand_factor
            
            # Final power allocation
            allocated_power = min(base_power * power_factor, remaining_power)
            allocated_power = max(0, allocated_power)  # No negative power
            
            # Calculate optimal voltage and current references
            if allocated_power > 0:
                # Optimal voltage reference with realistic EVCS range (300V-500V varying with SOC)
                voltage_ref = controller.params.min_voltage + controller.soc * (controller.params.max_voltage - controller.params.min_voltage)
                
                # Current reference based on power and voltage
                if voltage_ref > 0:
                    current_ref = min(allocated_power * 1000 / voltage_ref, target_current)
                else:
                    current_ref = 0
                
                # Ensure current is within realistic EVCS limits
                current_ref = max(controller.params.min_current, min(current_ref, controller.params.max_current))
            else:
                voltage_ref = controller.voltage_measured  # Hold current voltage
                current_ref = 0
                allocated_power = 0
            
            references[evcs_name] = {
                'power_ref': allocated_power,
                'voltage_ref': voltage_ref,
                'current_ref': current_ref,
                'priority': priority,
                'voltage_pu': voltage_pu
            }
            
            remaining_power -= allocated_power
            
            # Set references in controller
            controller.set_references(voltage_ref, current_ref, allocated_power)
            
            if remaining_power <= 0:
                break
        
        return references

def create_daily_load_shapes():
    """Create daily load shapes for QSTS simulation"""
    
    # 288 points for 24 hours at 5-minute intervals
    time_points = np.linspace(0, 24, 288)
    
    # Residential load pattern
    residential_pattern = []
    for hour in time_points:
        if 0 <= hour < 6:  # Night
            load_factor = 0.3 + 0.1 * np.random.normal(0, 0.05)
        elif 6 <= hour < 9:  # Morning peak
            load_factor = 0.7 + 0.2 * np.sin(np.pi * (hour - 6) / 3) + 0.1 * np.random.normal(0, 0.05)
        elif 9 <= hour < 17:  # Day
            load_factor = 0.5 + 0.1 * np.random.normal(0, 0.05)
        elif 17 <= hour < 21:  # Evening peak
            load_factor = 0.8 + 0.2 * np.sin(np.pi * (hour - 17) / 4) + 0.1 * np.random.normal(0, 0.05)
        else:  # Late evening
            load_factor = 0.4 + 0.1 * np.random.normal(0, 0.05)
        
        residential_pattern.append(max(0.2, min(1.0, load_factor)))
    
    return time_points, residential_pattern

def setup_ieee34_with_evcs_qsts():
    """Setup IEEE 34 bus system with EVCS for QSTS simulation"""
    
    # Clear and load IEEE 34 bus system
    dss.Command("Clear")
    
    # Try to compile the IEEE 34 system
    try:
        dss.Command("Compile ieee34Mod1.dss")
        print("IEEE 34 bus system loaded successfully")
    except Exception as e:
        print(f"Error loading IEEE 34 system: {e}")
        print("Trying alternative file names...")
        
        alternative_files = ["IEEE34Mod1.dss", "ieee34mod1.dss", "IEEE34.dss"]
        loaded = False
        
        for filename in alternative_files:
            try:
                dss.Command(f"Compile {filename}")
                print(f"Successfully loaded: {filename}")
                loaded = True
                break
            except:
                continue
        
        if not loaded:
            raise FileNotFoundError("Could not find IEEE 34 bus system file.")
    
    # Use simpler approach - don't rely on OpenDSS daily mode
    print("Setting up for manual time-step simulation...")
    
    # Just set to snapshot mode for more reliable operation
    dss.Command("Set Mode=Snapshot")
    dss.Command("Set ControlMode=Static")
    
    # Create daily load shapes (for reference only)
    time_points, load_pattern = create_daily_load_shapes()
    
    # Try to create and apply load shapes, but don't fail if it doesn't work
    try:
        load_shape_str = "New Loadshape.Daily npts=288 interval=0.0833 mult=["
        load_shape_str += ",".join([f"{val:.3f}" for val in load_pattern])
        load_shape_str += "]"
        dss.Command(load_shape_str)
        print("Daily load shape created successfully")
        
        # Apply load shape to existing loads
        dss.Command("Solve")  # First solve to populate the system
        
        load_names = dss.Loads.AllNames()
        
        if load_names and len(load_names) > 0:
            print(f"Found {len(load_names)} loads in the system")
            
            successful_loads = 0
            for load_name in load_names:
                try:
                    dss.Command(f"Load.{load_name}.Daily=Daily")
                    successful_loads += 1
                except Exception as e:
                    continue
            
            print(f"Daily load shape applied to {successful_loads}/{len(load_names)} loads")
        else:
            print("No loads found in the system")
            
    except Exception as e:
        print(f"Load shape setup failed: {e}")
        print("Will use manual load variation instead")
    
    # EVCS connection buses
    evcs_buses = ['800', '802', '806', '814', '820', '832']
    
    # Check if buses exist
    dss.Command("Solve")
    all_buses = dss.Circuit.AllBusNames()
    print(f"Total buses in system: {len(all_buses)}")
    
    valid_evcs_buses = []
    for bus in evcs_buses:
        if bus in all_buses:
            valid_evcs_buses.append(bus)
            print(f"Bus {bus} found - will add EVCS")
        else:
            print(f"Bus {bus} not found - skipping")
    
    if not valid_evcs_buses:
        raise ValueError("No valid buses found for EVCS placement")
    
    # Add EVCS Storage elements
    evcs_data = []
    for i, bus in enumerate(valid_evcs_buses, 1):
        evcs_name = f"EVCS{i}"
        
        try:
            # Add Storage element with initial charging state
            storage_cmd = f"New Storage.{evcs_name} Bus1={bus} kV=12.47 conn=wye kW=0 kWh=50 %stored=30 %reserve=10 %EffCharge=95 %EffDischarge=92 State=Idling"
            dss.Command(storage_cmd)
            
            # Verify the storage was added
            dss.Circuit.SetActiveElement(f"Storage.{evcs_name}")
            if dss.CktElement.Name().lower() == f"storage.{evcs_name.lower()}":
                print(f"✓ Successfully added {evcs_name} at Bus {bus}")
                
                evcs_data.append({
                    'name': evcs_name,
                    'bus': bus,
                    'kV': 12.47
                })
            else:
                print(f"✗ Failed to verify {evcs_name}")
                
        except Exception as e:
            print(f"✗ Error adding {evcs_name}: {e}")
            continue
    
    if not evcs_data:
        raise ValueError("No EVCS were successfully added")
    
    print(f"Successfully added {len(evcs_data)} EVCS to the system")
    
    # Initial solve and test
    print("\n=== Initial System Test ===")
    dss.Command("Solve")
    if dss.Solution.Converged():
        print("✓ Initial power flow converged")
        
        # Test EVCS control
        test_evcs = evcs_data[0]['name']
        print(f"Testing {test_evcs} control...")
        
        # Set a test power level
        dss.Command(f"Storage.{test_evcs}.State=Charging")
        dss.Command(f"Storage.{test_evcs}.kW=25")
        dss.Command("Solve")
        
        # Check if it worked
        dss.Circuit.SetActiveElement(f"Storage.{test_evcs}")
        test_power = dss.CktElement.Powers()[0]
        print(f"  Set 25kW, Actual: {test_power:.1f}kW")
        
        if abs(test_power - (-25.0)) < 1.0:  # Negative because it's consuming power
            print("✓ EVCS control test passed")
        else:
            print("⚠ EVCS control test failed - values may not update properly")
        
        # Reset to idle
        dss.Command(f"Storage.{test_evcs}.State=Idling")
        dss.Command(f"Storage.{test_evcs}.kW=0")
        
    else:
        print("✗ Initial power flow did not converge")
        print("⚠ Simulation may have issues")
    
    return evcs_data, time_points

def run_qsts_evcs_simulation():
    """Main QSTS simulation function"""
    
    print("Setting up QSTS simulation...")
    evcs_data, time_points = setup_ieee34_with_evcs_qsts()
    
    # Create EVCS controllers
    params = EVCSParameters()
    evcs_controllers = {}
    
    for evcs in evcs_data:
        controller = EVCSController(evcs['name'], params)
        evcs_controllers[evcs['name']] = controller
    
    # Create CMS
    cms = ChargingManagementSystem(evcs_controllers)
    
    # Simulation parameters
    total_steps = 288  # 24 hours at 5-minute intervals
    dt = 300  # 300 seconds = 5 minutes
    
    # Get baseline load for scaling
    try:
        dss.Command("Solve")
        baseline_load = dss.Circuit.TotalPower()[0]  # kW
        print(f"Baseline system load: {baseline_load:.1f} kW")
    except:
        baseline_load = 1000.0  # Default value
        print("Using default baseline load: 1000 kW")
    
    # Data storage
    results = {
        'time_hours': [],
        'step': [],
        'bus_voltages': {evcs['bus']: [] for evcs in evcs_data},
        'evcs_power': {evcs['name']: [] for evcs in evcs_data},
        'evcs_voltage_ref': {evcs['name']: [] for evcs in evcs_data},
        'evcs_current_ref': {evcs['name']: [] for evcs in evcs_data},
        'evcs_voltage_measured': {evcs['name']: [] for evcs in evcs_data},
        'evcs_current_measured': {evcs['name']: [] for evcs in evcs_data},
        'evcs_soc': {evcs['name']: [] for evcs in evcs_data},
        'total_power': [],
        'system_frequency': [],
        'system_load_factor': []
    }
    
    print(f"Starting QSTS simulation: 288 steps (24 hours)")
    
    # QSTS Time loop (Manual approach for reliability)
    for step in range(total_steps):
        current_time_hours = step * 5 / 60.0  # Convert 5-minute steps to hours
        
        # Calculate load factor for current time
        load_factor = cms.generate_daily_charging_profile(current_time_hours)
        results['system_load_factor'].append(load_factor)
        
        # Manual approach: solve at each time step without relying on OpenDSS daily mode
        try:
            # Simple solve for current conditions
            dss.Command("Solve")
            
            converged = dss.Solution.Converged()
            if not converged and step % 50 == 0:
                print(f"Warning: Power flow did not converge at step {step}")
                
        except Exception as e:
            if step % 50 == 0:  # Only print errors occasionally
                print(f"Error solving at step {step}: {e}")
            continue
        
        # Get system frequency
        system_frequency = dss.Solution.Frequency()
        
        # Get bus voltages (Fixed voltage reading and scope issue)
        bus_voltages = {}
        for evcs in evcs_data:
            voltage_pu = 1.0  # Default value
            try:
                dss.Circuit.SetActiveBus(evcs['bus'])
                # Get voltage magnitude in kV and convert to per unit
                voltage_kv = dss.Bus.kVBase()  # Base voltage in kV
                voltages_actual = dss.Bus.VMagAngle()  # Actual voltage in kV
                
                if len(voltages_actual) >= 2 and voltage_kv > 0:
                    voltage_pu = voltages_actual[0] / voltage_kv  # Convert to per unit
                    
                # Debug output every 50 steps
                if step % 50 == 0:
                    print(f"  Bus {evcs['bus']}: {voltages_actual[0]:.1f}kV / {voltage_kv:.1f}kV = {voltage_pu:.3f}pu")
                    
            except Exception as e:
                if step % 50 == 0:
                    print(f"  Error reading Bus {evcs['bus']}: {e}")
            
            bus_voltages[evcs['bus']] = voltage_pu
        
        # CMS optimization - set optimal references
        references = cms.optimize_charging_qsts(current_time_hours, bus_voltages, system_frequency)
        
        # Debug output every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step} (t={current_time_hours:.1f}h):")
            print(f"  Load factor: {load_factor:.3f}")
            print(f"  Bus voltages: {[(bus, f'{v:.3f}pu') for bus, v in bus_voltages.items()]}")
            # print(f"  CMS allocated power: {[(name, f'{ref['power_ref']:.1f}kW') for name, ref in references.items()]}")
            total_allocated = sum([ref["power_ref"] for ref in references.values()])
            print(f"  Total allocated: {total_allocated:.1f}kW")
        
        # Update EVCS dynamics and OpenDSS elements
        total_power = 0
        for evcs_name, controller in evcs_controllers.items():
            
            if evcs_name in references:
                ref_data = references[evcs_name]
                power_kW = ref_data['power_ref']
                
                # Update OpenDSS storage element
                try:
                    if power_kW > 0:
                        dss.Command(f"Storage.{evcs_name}.State=Charging")
                        dss.Command(f"Storage.{evcs_name}.kW={power_kW}")
                        
                        # Debug: verify the command worked
                        if step % 50 == 0:
                            dss.Circuit.SetActiveElement(f"Storage.{evcs_name}")
                            actual_kW = dss.CktElement.Powers()[0]  # Real power
                            print(f"  {evcs_name}: Set {power_kW:.1f}kW, Actual {actual_kW:.1f}kW")
                    else:
                        dss.Command(f"Storage.{evcs_name}.State=Idling")
                        dss.Command(f"Storage.{evcs_name}.kW=0")
                        
                except Exception as e:
                    if step % 50 == 0:
                        print(f"  Error updating {evcs_name}: {e}")
                    continue
                
                # Update controller dynamics
                evcs_num = evcs_name.replace('EVCS', '')
                bus_mapping = {'1': '800', '2': '802', '3': '806', '4': '814', '5': '820', '6': '832'}
                bus_name = bus_mapping.get(evcs_num, '800')
                
                grid_voltage = bus_voltages.get(bus_name, 1.0) * 7200  # Convert to RMS voltage
                
                dynamics_result = controller.update_dynamics(grid_voltage, dt)
                total_power += dynamics_result['total_power']
                
                # Debug controller state every 50 steps
                if step % 50 == 0:
                    print(f"    {evcs_name}: SOC={controller.soc:.3f}, V_ref={ref_data['voltage_ref']:.1f}V, I_ref={ref_data['current_ref']:.1f}A")
            
            else:
                # No reference for this EVCS
                if step % 50 == 0:
                    print(f"    {evcs_name}: No power reference")
        
        # Store results
        results['time_hours'].append(current_time_hours)
        results['step'].append(step)
        results['total_power'].append(total_power)
        results['system_frequency'].append(system_frequency)
        
        # Store bus voltages and EVCS data
        for evcs in evcs_data:
            bus_name = evcs['bus']
            evcs_name = evcs['name']
            controller = evcs_controllers[evcs_name]
            
            # Store bus voltage
            results['bus_voltages'][bus_name].append(bus_voltages.get(bus_name, 1.0))
            
            # Store EVCS data
            if evcs_name in references:
                ref_data = references[evcs_name]
                results['evcs_power'][evcs_name].append(ref_data['power_ref'])
                results['evcs_voltage_ref'][evcs_name].append(ref_data['voltage_ref'])
                results['evcs_current_ref'][evcs_name].append(ref_data['current_ref'])
            else:
                results['evcs_power'][evcs_name].append(0.0)
                results['evcs_voltage_ref'][evcs_name].append(controller.voltage_reference)
                results['evcs_current_ref'][evcs_name].append(controller.current_reference)
            
            # Store measured values
            results['evcs_voltage_measured'][evcs_name].append(controller.voltage_measured)
            results['evcs_current_measured'][evcs_name].append(controller.current_measured)
            results['evcs_soc'][evcs_name].append(controller.soc)
        
        # Progress indicator (more frequent for debugging)
        if step % 24 == 0:  # Every 2 hours
            progress = step / total_steps * 100
            print(f"\n=== QSTS Progress: {progress:.1f}% - Time: {current_time_hours:.1f}h ===")
            print(f"Total EVCS Power: {total_power:.1f}kW, System Frequency: {system_frequency:.3f}Hz")
            
        # Summary every 4 hours
        if step % 48 == 0 and step > 0:
            print(f"\n--- 4-Hour Summary (t={current_time_hours:.1f}h) ---")
            for evcs_name, controller in evcs_controllers.items():
                print(f"  {evcs_name}: SOC={controller.soc*100:.1f}%, Power_ref={controller.power_reference:.1f}kW")
    
    print(f"QSTS simulation completed! Total steps: {len(results['time_hours'])}")
    return results, evcs_data

def plot_qsts_results(results, evcs_data):
    """Plot QSTS simulation results"""
    
    if not results['time_hours']:
        print("No simulation data to plot!")
        return
    
    print(f"Plotting QSTS results for {len(results['time_hours'])} time steps...")
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('24-Hour QSTS EVCS Dynamics - IEEE 34 Bus System', fontsize=16)
    
    time_hours = results['time_hours']
    
    # Plot 1: Bus Voltages over 24 hours
    ax1 = axes[0, 0]
    for bus in results['bus_voltages']:
        ax1.plot(time_hours, results['bus_voltages'][bus], 
                label=f'Bus {bus}', linewidth=2)
    ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='Upper Limit')
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Lower Limit')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Voltage (per unit)')
    ax1.set_title('24-Hour Bus Voltage Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    
    # Plot 2: EVCS Charging Power
    ax2 = axes[0, 1]
    for evcs_name in results['evcs_power']:
        ax2.plot(time_hours, results['evcs_power'][evcs_name], 
                label=evcs_name, linewidth=2)
    ax2.plot(time_hours, results['total_power'], 
            'k--', linewidth=3, label='Total Power')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('24-Hour EVCS Charging Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    
    # Plot 3: Voltage References vs Measured
    ax3 = axes[1, 0]
    for i, evcs_name in enumerate(list(results['evcs_voltage_ref'].keys())[:3]):  # Show first 3
        ax3.plot(time_hours, results['evcs_voltage_ref'][evcs_name], 
                '--', linewidth=2, label=f'{evcs_name} Ref')
        ax3.plot(time_hours, results['evcs_voltage_measured'][evcs_name], 
                '-', linewidth=2, label=f'{evcs_name} Measured')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('DC Voltage (V)')
    ax3.set_title('EVCS Voltage References vs Measured')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    # Plot 4: Current References vs Measured
    ax4 = axes[1, 1]
    for i, evcs_name in enumerate(list(results['evcs_current_ref'].keys())[:3]):  # Show first 3
        ax4.plot(time_hours, results['evcs_current_ref'][evcs_name], 
                '--', linewidth=2, label=f'{evcs_name} Ref')
        ax4.plot(time_hours, results['evcs_current_measured'][evcs_name], 
                '-', linewidth=2, label=f'{evcs_name} Measured')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('DC Current (A)')
    ax4.set_title('EVCS Current References vs Measured')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)
    
    # Plot 5: SOC Evolution
    ax5 = axes[2, 0]
    for evcs_name in results['evcs_soc']:
        ax5.plot(time_hours, [soc*100 for soc in results['evcs_soc'][evcs_name]], 
                linewidth=2, label=evcs_name)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('State of Charge (%)')
    ax5.set_title('24-Hour SOC Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 24)
    ax5.set_ylim(0, 100)
    
    # Plot 6: System Frequency and Total Power
    ax6 = axes[2, 1]
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(time_hours, results['system_frequency'], 'b-', linewidth=2, label='Frequency')
    line2 = ax6_twin.plot(time_hours, results['total_power'], 'r-', linewidth=2, label='Total Power')
    
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Frequency (Hz)', color='b')
    ax6_twin.set_ylabel('Total Power (kW)', color='r')
    ax6.set_title('System Frequency & Total EVCS Power')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 24)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("24-HOUR QSTS SIMULATION SUMMARY")
    print("="*80)
    
    for evcs in evcs_data:
        bus = evcs['bus']
        evcs_name = evcs['name']
        
        # Bus statistics
        voltages = results['bus_voltages'][bus]
        avg_voltage = np.mean(voltages)
        min_voltage = np.min(voltages)
        max_voltage = np.max(voltages)
        
        # EVCS statistics
        powers = results['evcs_power'][evcs_name]
        total_energy = np.sum(powers) * 5 / 60  # 5-minute intervals to hours
        initial_soc = results['evcs_soc'][evcs_name][0] * 100
        final_soc = results['evcs_soc'][evcs_name][-1] * 100
        soc_change = final_soc - initial_soc
        
        # Control performance
        voltage_refs = results['evcs_voltage_ref'][evcs_name]
        voltage_measured = results['evcs_voltage_measured'][evcs_name]
        voltage_tracking_error = np.mean([abs(r-m) for r,m in zip(voltage_refs, voltage_measured)])
        
        current_refs = results['evcs_current_ref'][evcs_name]
        current_measured = results['evcs_current_measured'][evcs_name]
        current_tracking_error = np.mean([abs(r-m) for r,m in zip(current_refs, current_measured)])
        
        print(f"\n{evcs_name} (Bus {bus}):")
        print(f"  Bus Voltage - Avg: {avg_voltage:.3f}pu, Min: {min_voltage:.3f}pu, Max: {max_voltage:.3f}pu")
        print(f"  Energy Charged: {total_energy:.1f} kWh")
        print(f"  SOC Change: {initial_soc:.1f}% → {final_soc:.1f}% (Δ{soc_change:+.1f}%)")
        print(f"  Voltage Tracking Error: {voltage_tracking_error:.1f}V")
        print(f"  Current Tracking Error: {current_tracking_error:.1f}A")
    
    # System-wide statistics
    total_system_energy = np.sum(results['total_power']) * 5 / 60  # kWh
    avg_frequency = np.mean(results['system_frequency'])
    frequency_deviation = np.std(results['system_frequency'])
    
    print(f"\nSYSTEM SUMMARY:")
    print(f"  Total Energy Delivered: {total_system_energy:.1f} kWh")
    print(f"  Average System Frequency: {avg_frequency:.3f} Hz")
    print(f"  Frequency Std Deviation: {frequency_deviation:.4f} Hz")
    print(f"  Peak Total Power: {max(results['total_power']):.1f} kW")

if __name__ == "__main__":
    print("Starting 24-Hour QSTS EVCS Dynamics Simulation...")
    
    # Check OpenDSS version
    try:
        version = dss.Basic.Version()
        print(f"OpenDSS Version: {version}")
    except:
        print("Could not get OpenDSS version")
    
    # Check if we can access basic OpenDSS functions
    try:
        dss.Command("Clear")
        print("OpenDSS interface working correctly")
    except Exception as e:
        print(f"OpenDSS interface error: {e}")
        exit(1)
    
    try:
        # Run QSTS simulation
        print("Initializing simulation...")
        results, evcs_data = run_qsts_evcs_simulation()
        
        # Plot results
        print("\nGenerating comprehensive plots...")
        plot_qsts_results(results, evcs_data)
        
        print("\n24-Hour QSTS Simulation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
        print("Please ensure the IEEE 34 bus system file (.dss) is in the current directory")
        print("Expected file names: ieee34Mod1.dss, IEEE34Mod1.dss, ieee34mod1.dss, or IEEE34.dss")
        
    except Exception as e:
        print(f"Error during QSTS simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure ieee34Mod1.dss file is in the current directory")
        print("2. Check that OpenDSS is properly installed and accessible")
        print("3. Verify all required Python packages are installed:")
        print("   pip install opendssdirect matplotlib pandas scipy numpy")
        print("4. Try running a simple OpenDSS command first to test the installation")