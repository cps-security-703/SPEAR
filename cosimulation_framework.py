"""
Transmission-Distribution Co-simulation Framework
IEEE 14-bus (MATLAB/Python) + Multiple IEEE 34-bus (OpenDSS) Systems
With EV Charging Station Cyber Attack Simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import opendssdirect as dss
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import threading
import queue

@dataclass
class EVChargingStation:
    """EV Charging Station Model"""
    bus_name: str
    max_power: float  # kW
    num_ports: int
    efficiency: float = 0.95
    current_load: float = 0.0
    available_ports: int = 0
    voltage_setpoint: float = 1.0
    current_setpoint: float = 0.0
    
    def __post_init__(self):
        """Initialize derived attributes"""
        self.available_ports = self.num_ports
        self.current_setpoint = self.max_power * 0.5  # Start at 50% capacity
    
class ChargingManagementSystem:
    """Charging Management System with Cyber Attack Capabilities"""
    
    def __init__(self, stations: List[EVChargingStation]):
        self.stations = stations
        self.attack_active = False
        self.attack_params = {}
        self.measurements = {}
        
    def update_measurements(self, station_id: int, voltage: float, current: float, power: float):
        """Update sensor measurements (can be manipulated during attacks)"""
        self.measurements[station_id] = {
            'voltage': voltage,
            'current': current, 
            'power': power,
            'timestamp': time.time()
        }
        
    def inject_false_data(self, station_id: int, attack_type: str, magnitude: float):
        """Inject false data into measurements"""
        if station_id in self.measurements:
            if attack_type == 'voltage_low':
                self.measurements[station_id]['voltage'] *= (1 - magnitude)
            elif attack_type == 'current_low':
                self.measurements[station_id]['current'] *= (1 - magnitude)
            elif attack_type == 'power_underreport':
                self.measurements[station_id]['power'] *= (1 - magnitude)
                
    def optimize_charging(self, station_id: int) -> Tuple[float, float]:
        """Optimize voltage and current setpoints (vulnerable to false data)"""
        station = self.stations[station_id]
        if station_id in self.measurements:
            meas = self.measurements[station_id]
            
            # Normal control logic (can be fooled by false measurements)
            if meas['voltage'] < 0.95:  # If voltage appears low
                # Increase current to maintain power
                new_current = min(station.current_setpoint * 1.1, station.max_power / (meas['voltage'] * station.efficiency))
            else:
                new_current = station.current_setpoint
                
            return meas['voltage'], new_current
        
        return station.voltage_setpoint, station.current_setpoint

class IEEE14BusAGC:
    """IEEE 14-bus Transmission System with AGC"""
    
    def __init__(self):
        # Bus data (from your file)
        self.busdata = np.array([
            [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.06, 0.0],  # Slack bus
            [2, 2, 21.7, 12.7, 0.0, 0.0, 1, 1.045, 0.0],  # Generator
            [3, 2, 94.2, 19.0, 0.0, 0.0, 1, 1.01, 0.0],   # Generator
            [4, 1, 47.8, -3.9, 0.0, 0.0, 1, 1.0, 0.0],
            [5, 1, 7.6, 1.6, 0.0, 0.0, 1, 1.0, 0.0],
            [6, 2, 11.2, 7.5, 0.0, 0.0, 1, 1.07, 0.0],    # Generator
            [7, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0],
            [8, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.09, 0.0],     # Generator
            [9, 1, 29.5, 16.6, 0.0, 0.0, 1, 1.0, 0.0],
            [10, 1, 9.0, 5.8, 0.0, 0.0, 1, 1.0, 0.0],
            [11, 1, 3.5, 1.8, 0.0, 0.0, 1, 1.0, 0.0],
            [12, 1, 6.1, 1.6, 0.0, 0.0, 1, 1.0, 0.0],
            [13, 1, 13.5, 5.8, 0.0, 0.0, 1, 1.0, 0.0],
            [14, 1, 14.9, 5.0, 0.0, 0.0, 1, 1.0, 0.0]
        ])
        
        # AGC parameters
        self.f_nominal = 60.0  # Hz
        self.frequency = 60.0
        self.H_system = 5.0  # System inertia (s)
        self.D_system = 1.0  # Load damping
        self.R_droop = 0.05  # Governor droop
        self.T_gov = 0.2     # Governor time constant
        self.T_turb = 0.5    # Turbine time constant
        
        # AGC state variables
        self.delta_f = 0.0
        self.delta_Pm = 0.0
        self.delta_Pv = 0.0
        
        # Distribution system connections
        self.dist_connections = {
            4: [],   # Bus 4 connected to distribution systems
            9: [],   # Bus 9 connected to distribution systems  
            13: []   # Bus 13 connected to distribution systems
        }
        
        # Base load
        self.P_load_base = np.sum(self.busdata[:, 2])  # Total base load
        self.P_load_current = self.P_load_base
        
    def agc_dynamics(self, t, x):
        """AGC system dynamics"""
        delta_f, delta_Pm, delta_Pv = x
        
        # Power imbalance from distribution systems
        P_dist_total = sum([sum(loads) for loads in self.dist_connections.values()])
        delta_PL = (self.P_load_current + P_dist_total - self.P_load_base) / 100.0  # p.u.
        
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
        
        return self.frequency

class OpenDSSInterface:
    """Interface to OpenDSS Distribution Systems"""
    
    # Class variable to track if OpenDSS is already initialized
    _opendss_initialized = False
    _current_circuit = None
    
    def __init__(self, dss_file_path: str, system_id: int):
        self.dss_file_path = dss_file_path
        self.system_id = system_id
        self.ev_stations = []
        self.cms = None
        self.base_loads = {}
        
    def initialize(self):
        """Initialize OpenDSS system"""
        try:
            print(f"Initializing Distribution System {self.system_id}...")
            
            # Use different files for different systems to avoid conflicts
            if self.system_id == 1:
                dss_file = "ieee34Mod1.dss"
            elif self.system_id == 2:
                dss_file = "ieee34Mod2.dss"
            else:
                dss_file = "ieee34Mod1.dss"  # System 3 uses Mod1 again
            
            # Check if this is the first system being initialized
            if not OpenDSSInterface._opendss_initialized:
                print(f"First OpenDSS initialization with {dss_file}")
                dss.Command("Clear")
                dss.Command(f"Redirect {dss_file}")
                
                # Add a simple timeout check
                import time
                start_time = time.time()
                dss.Command("solve")
                solve_time = time.time() - start_time
                
                if solve_time > 5.0:  # If solve takes more than 5 seconds
                    print(f"OpenDSS solve took too long ({solve_time:.2f}s), using mock system")
                    return self._initialize_mock_system()
                
                OpenDSSInterface._opendss_initialized = True
                OpenDSSInterface._current_circuit = dss_file
                print(f"OpenDSS initialized successfully with {dss_file}")
            else:
                print(f"OpenDSS already initialized with {OpenDSSInterface._current_circuit}")
                if dss_file != OpenDSSInterface._current_circuit:
                    print(f"Warning: Different file requested ({dss_file}) but using existing circuit")
            
            # Store base loads
            dss.Loads.First()
            while dss.Loads.Name():
                self.base_loads[dss.Loads.Name()] = {
                    'kW': dss.Loads.kW(),
                    'kvar': dss.Loads.kvar()
                }
                dss.Loads.Next()
                
            print(f"Distribution System {self.system_id} initialized successfully with OpenDSS")
            return True
            
        except Exception as e:
            print(f"Error initializing Distribution System {self.system_id}: {e}")
            print("Using mock distribution system instead...")
            return self._initialize_mock_system()
    
    def _initialize_mock_system(self):
        """Initialize a mock distribution system when OpenDSS fails"""
        try:
            # Create mock base loads for IEEE 34-bus system
            self.base_loads = {
                'Load_890': {'kW': 150.0, 'kvar': 75.0},
                'Load_844': {'kW': 100.0, 'kvar': 50.0},
                'Load_860': {'kW': 80.0, 'kvar': 40.0},
                'Load_840': {'kW': 120.0, 'kvar': 60.0},
                'Load_848': {'kW': 90.0, 'kvar': 45.0},
                'Load_830': {'kW': 70.0, 'kvar': 35.0}
            }
            
            print(f"Mock Distribution System {self.system_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing mock system {self.system_id}: {e}")
            return False
    
    def add_ev_charging_stations(self, stations_config: List[Dict]):
        """Add EV charging stations to the distribution system"""
        for i, config in enumerate(stations_config):
            try:
                bus_name = config['bus']
                max_power = config['max_power']
                
                # Try to add to OpenDSS if available
                try:
                    dss.Command(f"New Load.EVCS_{self.system_id}_{i} "
                              f"Bus1={bus_name} "
                              f"Phases=3 "
                              f"kW={max_power * 0.5} "  # Start at 50% capacity
                              f"kvar={max_power * 0.1} "
                              f"Model=1")
                except:
                    # If OpenDSS fails, just continue with mock system
                    pass
                
                # Create EVCS object
                station = EVChargingStation(
                    bus_name=bus_name,
                    max_power=max_power,
                    num_ports=config.get('num_ports', 10),
                    current_load=max_power * 0.5
                )
                self.ev_stations.append(station)
                print(f"Added EVCS {i} at bus {bus_name} with {max_power}kW capacity")
                
            except Exception as e:
                print(f"Error adding EVCS {i}: {e}")
                continue
        
        # Initialize CMS
        if self.ev_stations:
            self.cms = ChargingManagementSystem(self.ev_stations)
            print(f"Initialized CMS with {len(self.ev_stations)} EV charging stations")
        
    def update_ev_loads(self):
        """Update EV charging loads based on CMS control"""
        if not self.cms:
            return
            
        for i, station in enumerate(self.ev_stations):
            try:
                load_name = f"EVCS_{self.system_id}_{i}"
                
                # Try to get measurements from OpenDSS if available
                try:
                    dss.Circuit.SetActiveElement(f"Load.{load_name}")
                    voltage = dss.CktElement.VoltagesMagAng()[0] / 1000  # Convert to p.u.
                    power = dss.CktElement.Powers()[0]  # kW
                    current = power / (voltage * 1000) if voltage > 0 else 0
                    
                    # Update load in OpenDSS
                    dss.Command(f"Load.{load_name}.kW={station.current_load}")
                    dss.Command(f"Load.{load_name}.kvar={station.current_load * 0.2}")
                    
                except:
                    # Use mock measurements if OpenDSS fails
                    voltage = 1.0  # p.u.
                    power = station.current_load
                    current = power / (voltage * 1000) if voltage > 0 else 0
                
                # Update CMS measurements
                self.cms.update_measurements(i, voltage, current, power)
                
                # Get optimized setpoints
                v_set, i_set = self.cms.optimize_charging(i)
                
                # Calculate new power
                new_power = min(v_set * i_set * station.efficiency, station.max_power)
                station.current_load = new_power
                
            except Exception as e:
                print(f"Error updating EVCS {i} in system {self.system_id}: {e}")
                continue
    
    def get_total_load(self) -> float:
        """Get total system load in MW"""
        try:
            dss.Command("solve")
            return dss.Circuit.TotalPower()[0] / 1000  # Convert kW to MW
        except:
            # Calculate total load from base loads and EV stations if OpenDSS fails
            total_load = 0.0
            
            # Add base loads
            for load_name, load_data in self.base_loads.items():
                total_load += load_data['kW']
            
            # Add EV charging station loads
            for station in self.ev_stations:
                total_load += station.current_load
            
            return total_load / 1000  # Convert to MW
    
    def launch_cyber_attack(self, attack_config: Dict):
        """Launch cyber attack on charging management system"""
        attack_type = attack_config.get('type', 'voltage_low')
        magnitude = attack_config.get('magnitude', 0.2)
        target_stations = attack_config.get('targets', list(range(len(self.ev_stations))))
        
        if self.cms:
            self.cms.attack_active = True
            for station_id in target_stations:
                if station_id < len(self.ev_stations):
                    self.cms.inject_false_data(station_id, attack_type, magnitude)
                    
    def stop_cyber_attack(self):
        """Stop cyber attack"""
        if self.cms:
            self.cms.attack_active = False

class CoSimulationFramework:
    """Main co-simulation framework"""
    
    def __init__(self):
        self.transmission_system = IEEE14BusAGC()
        self.distribution_systems = {}
        self.simulation_time = 0.0
        self.dt = 0.1  # 100ms time step
        self.results = {
            'time': [],
            'frequency': [],
            'total_load': [],
            'dist_loads': {}
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
            self.transmission_system.dist_connections[connection_bus] = []
            self.results['dist_loads'][system_id] = []
            
    def setup_ev_charging_stations(self):
        """Setup EV charging stations in distribution systems"""
        # Example configuration for each distribution system

        print("Setting up EV charging stations...")
        evcs_configs = [
            [
                {'bus': '890', 'max_power': 500, 'num_ports': 20},  # Fast charging hub
                {'bus': '844', 'max_power': 150, 'num_ports': 6},   # Shopping center
                {'bus': '860', 'max_power': 100, 'num_ports': 4},   # Residential area
            ],
            [
                {'bus': '890', 'max_power': 750, 'num_ports': 30},  # Highway charging station
                {'bus': '840', 'max_power': 200, 'num_ports': 8},   # Commercial area
            ],
            [
                {'bus': '848', 'max_power': 300, 'num_ports': 12},  # Industrial area
                {'bus': '830', 'max_power': 100, 'num_ports': 4},   # Suburban area
            ]
        ]
        
        for i, (sys_id, dist_info) in enumerate(self.distribution_systems.items()):
            if i < len(evcs_configs):
                dist_info['system'].add_ev_charging_stations(evcs_configs[i])
    
    def run_simulation(self, duration: float = 300.0, attack_scenarios: List[Dict] = None):
        """Run co-simulation"""
        print("Starting Transmission-Distribution Co-simulation...")
        
        steps = int(duration / self.dt)
        attack_scenarios = attack_scenarios or []
        
        for step in range(steps):
            self.simulation_time = step * self.dt
            
            # Check for attack scenarios
            for attack in attack_scenarios:
                if (attack['start_time'] <= self.simulation_time <= 
                    attack['start_time'] + attack['duration']):
                    
                    if not attack.get('launched', False):
                        print(f"Launching cyber attack at t={self.simulation_time:.1f}s")
                        target_sys = attack['target_system']
                        if target_sys in self.distribution_systems:
                            self.distribution_systems[target_sys]['system'].launch_cyber_attack(attack)
                        attack['launched'] = True
                        
                elif self.simulation_time > attack['start_time'] + attack['duration']:
                    if attack.get('launched', False) and not attack.get('stopped', False):
                        print(f"Stopping cyber attack at t={self.simulation_time:.1f}s")
                        target_sys = attack['target_system']
                        if target_sys in self.distribution_systems:
                            self.distribution_systems[target_sys]['system'].stop_cyber_attack()
                        attack['stopped'] = True
            
            # Update distribution systems
            total_dist_load = 0.0
            for sys_id, dist_info in self.distribution_systems.items():
                dist_sys = dist_info['system']
                connection_bus = dist_info['connection_bus']
                
                # Update EV loads based on CMS control
                dist_sys.update_ev_loads()
                
                # Get total load
                load = dist_sys.get_total_load()
                total_dist_load += load
                
                # Update transmission system connection
                self.transmission_system.dist_connections[connection_bus] = [load]
                self.results['dist_loads'][sys_id].append(load)
            
            # Update transmission system frequency
            frequency = self.transmission_system.update_frequency(self.dt)
            
            # Store results
            self.results['time'].append(self.simulation_time)
            self.results['frequency'].append(frequency)
            self.results['total_load'].append(total_dist_load)
            
            # Print progress
            if step % 100 == 0:
                print(f"t={self.simulation_time:.1f}s, f={frequency:.3f}Hz, "
                      f"Load={total_dist_load:.1f}MW")
    
    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Frequency plot
        axes[0].plot(self.results['time'], self.results['frequency'], 'b-', linewidth=2)
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title('Transmission System Frequency Response')
        axes[0].grid(True)
        axes[0].axhline(y=60.0, color='r', linestyle='--', alpha=0.7)
        
        # Total load plot
        axes[1].plot(self.results['time'], self.results['total_load'], 'g-', linewidth=2)
        axes[1].set_ylabel('Total Dist. Load (MW)')
        axes[1].set_title('Distribution Systems Total Load')
        axes[1].grid(True)
        
        # Individual distribution system loads
        for sys_id, loads in self.results['dist_loads'].items():
            axes[2].plot(self.results['time'], loads, label=f'Dist System {sys_id}')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Load (MW)')
        axes[2].set_title('Individual Distribution System Loads')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
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
            'distribution_systems': {}
        }
        
        for sys_id, loads in self.results['dist_loads'].items():
            stats['distribution_systems'][sys_id] = {
                'min': min(loads),
                'max': max(loads),
                'mean': np.mean(loads),
                'std': np.std(loads)
            }
        
        return stats

# Example usage and attack scenarios
def run_cyber_attack_study():
    """Run cyber attack study on transmission-distribution system"""

    print("Starting Transmission-Distribution Co-simulation...")
    
    # Initialize co-simulation framework
    cosim = CoSimulationFramework()
    
    # Add distribution systems (you'll need to have these files in your directory)
    cosim.add_distribution_system(1, "ieee34Mod1.dss", 4)   # Connected to bus 4
    cosim.add_distribution_system(2, "ieee34Mod1.dss", 9)   # Connected to bus 9  
    cosim.add_distribution_system(3, "ieee34Mod1.dss", 13)  # Connected to bus 13
    
    # Setup EV charging stations
    cosim.setup_ev_charging_stations()
    
    # Define cyber attack scenarios
    attack_scenarios = [
        {
            'start_time': 60.0,      # Attack starts at 60s
            'duration': 60.0,        # Attack lasts 60s
            'target_system': 1,      # Target distribution system 1
            'type': 'voltage_low',   # False low voltage readings
            'magnitude': 0.15,       # 15% voltage underreporting
            'targets': [0, 1]        # Target first two EVCS
        },
        {
            'start_time': 150.0,     # Second attack at 150s
            'duration': 45.0,        # Attack lasts 45s
            'target_system': 2,      # Target distribution system 2
            'type': 'power_underreport',  # False low power readings
            'magnitude': 0.25,       # 25% power underreporting
            'targets': [0]           # Target largest EVCS
        }
    ]
    
    # Run simulation
    cosim.run_simulation(duration=300.0, attack_scenarios=attack_scenarios)
    
    # Plot results
    cosim.plot_results()
    
    return cosim

if __name__ == "__main__":
    # Run the cyber attack study
    print("IEEE 14-bus + 3×IEEE 34-bus Co-simulation with Cyber Attacks")
    print("=" * 60)
    
    simulation = run_cyber_attack_study()
    
    # Print summary statistics
    stats = simulation.get_simulation_statistics()
    print(f"\n=== SIMULATION STATISTICS ===")
    print(f"Frequency Range: {stats['frequency']['min']:.3f} - {stats['frequency']['max']:.3f} Hz")
    print(f"Frequency Mean: {stats['frequency']['mean']:.3f} Hz")
    print(f"Maximum Frequency Deviation: {stats['frequency']['max_deviation']:.3f} Hz")
    print(f"Total Load Range: {stats['total_load']['min']:.1f} - {stats['total_load']['max']:.1f} MW")
    print(f"Total Load Mean: {stats['total_load']['mean']:.1f} MW")
    
    print(f"\n=== DISTRIBUTION SYSTEM STATISTICS ===")
    for sys_id, sys_stats in stats['distribution_systems'].items():
        print(f"System {sys_id}: Load {sys_stats['min']:.1f} - {sys_stats['max']:.1f} MW (Mean: {sys_stats['mean']:.1f} MW)")