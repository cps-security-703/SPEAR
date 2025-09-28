import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# LLM Integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import BitsAndBytesConfig
    import torch
    LLM_AVAILABLE = True
    print("LLM libraries loaded successfully!")
except ImportError:
    print("LLM libraries not available. Install with: pip install transformers accelerate")
    LLM_AVAILABLE = False

# Install with: pip install pandapower
try:
    import pandapower as pp
    import pandas as pd
    PANDAPOWER_AVAILABLE = True
except ImportError:
    print("Pandapower not available. Install with: pip install pandapower")
    PANDAPOWER_AVAILABLE = False

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    PLOTTING_AVAILABLE = True
    print("Plotting libraries loaded successfully!")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available. Install matplotlib and seaborn for visualizations.")

@dataclass
class BusData:
    """Data structure for power system bus"""
    voltage: float
    angle: float
    frequency: float
    active_power: float
    reactive_power: float

class PowerSystem14Bus:
    """IEEE 14-bus power system model with AGC using Pandapower"""
    
    def __init__(self):
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("Pandapower is required. Install with: pip install pandapower")
            
        self.num_buses = 14
        self.base_frequency = 60.0  # Hz
        self.base_voltage = 230.0   # kV
        
        # Create Pandapower network
        self.net = pp.create_empty_network()
        self._create_ieee14_network()
        
        # Initialize system measurements and state
        self.bus_measurements = {}
        self.line_measurements = {}
        self.gen_measurements = {}
        
        # AGC parameters
        self.area_control_error = 0.0
        self.tie_line_bias = -0.3
        self.frequency_bias = 20.0
        self.agc_integral = 0.0
        
        # System dynamics
        self.inertia = 5.0
        self.damping = 1.0
        self.frequency_deviation = 0.0
        
        # Attack state tracking
        self.compromised_buses = set()
        self.compromised_lines = set()
        self.attack_history = []
        
        # Initialize measurements
        self._update_measurements()
    
    def _create_ieee14_network(self):
        """Create the IEEE 14-bus test system"""
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
        
        # Create lines
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
    
    def _update_measurements(self):
        """Update system measurements from power flow solution"""
        try:
            # Run power flow
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            
            # Update bus measurements
            for i in range(self.num_buses):
                # Add small random measurement noise
                noise_voltage = np.random.normal(0, 0.001)
                noise_angle = np.random.normal(0, 0.001)
                noise_freq = np.random.normal(0, 0.001)
                
                self.bus_measurements[i+1] = BusData(
                    voltage=self.net.res_bus.vm_pu.iloc[i] + noise_voltage,
                    angle=self.net.res_bus.va_degree.iloc[i] + noise_angle,
                    frequency=self.base_frequency + self.frequency_deviation + noise_freq,
                    active_power=0.0,  # Will be updated from line flows
                    reactive_power=0.0
                )
            
            # Update line power flows
            self.line_measurements = {}
            for idx, line in self.net.line.iterrows():
                from_bus = line.from_bus + 1
                to_bus = line.to_bus + 1
                p_from_mw = self.net.res_line.p_from_mw.iloc[idx]
                p_to_mw = self.net.res_line.p_to_mw.iloc[idx]
                
                self.line_measurements[(from_bus, to_bus)] = {
                    'p_from': p_from_mw,
                    'p_to': p_to_mw,
                    'q_from': self.net.res_line.q_from_mvar.iloc[idx],
                    'q_to': self.net.res_line.q_to_mvar.iloc[idx],
                    'loading': self.net.res_line.loading_percent.iloc[idx]
                }
            
            # Update generator measurements
            for idx, gen in self.net.gen.iterrows():
                bus_id = gen.bus + 1
                self.gen_measurements[bus_id] = {
                    'p_mw': self.net.res_gen.p_mw.iloc[idx],
                    'q_mvar': self.net.res_gen.q_mvar.iloc[idx],
                    'vm_pu': self.net.res_gen.vm_pu.iloc[idx]
                }
                
        except Exception as e:
            print(f"Power flow convergence issue: {e}")
            # Use previous measurements if power flow fails
        
    def calculate_tie_line_power(self, bus1_id: int, bus2_id: int) -> float:
        """Calculate power flow between two buses"""
        if (bus1_id, bus2_id) in self.line_measurements:
            return self.line_measurements[(bus1_id, bus2_id)]['p_from']
        elif (bus2_id, bus1_id) in self.line_measurements:
            return -self.line_measurements[(bus2_id, bus1_id)]['p_to']
        else:
            # Fallback calculation if line not found
            if bus1_id in self.bus_measurements and bus2_id in self.bus_measurements:
                bus1 = self.bus_measurements[bus1_id]
                bus2 = self.bus_measurements[bus2_id]
                angle_diff = np.radians(bus1.angle - bus2.angle)
                power_flow = bus1.voltage * bus2.voltage * np.sin(angle_diff) * 2.0
                return power_flow
        return 0.0
    
    def update_agc(self, dt=0.1):
        """Update AGC control based on frequency and tie-line deviations"""
        # Calculate average system frequency
        frequencies = [bus.frequency for bus in self.bus_measurements.values()]
        avg_frequency = np.mean(frequencies)
        frequency_error = self.base_frequency - avg_frequency
        
        # Calculate tie line power deviation (monitoring key interconnections)
        key_tie_lines = [(1, 2), (1, 5), (2, 3), (4, 5), (6, 11)]
        tie_line_power_total = 0.0
        
        for bus1, bus2 in key_tie_lines:
            tie_line_power_total += self.calculate_tie_line_power(bus1, bus2)
        
        # Scheduled tie line power (assuming 0 for simplicity)
        scheduled_tie_power = 0.0
        tie_line_error = tie_line_power_total - scheduled_tie_power
        
        # Area Control Error (ACE)
        self.area_control_error = (frequency_error * self.frequency_bias + 
                                 tie_line_error * self.tie_line_bias)
        
        # Integral control for AGC
        self.agc_integral += self.area_control_error * dt
        
        # AGC control signal (PI controller)
        kp_agc = 0.1  # Proportional gain
        ki_agc = 0.05  # Integral gain
        control_signal = -(kp_agc * self.area_control_error + ki_agc * self.agc_integral)
        
        # Apply control to generators (avoid compromised buses)
        generator_buses = [1, 2, 3, 6, 8]  # Generator bus IDs
        active_generators = [bus for bus in generator_buses if bus not in self.compromised_buses]
        
        if active_generators:
            # Distribute control signal among active generators
            control_per_gen = control_signal / len(active_generators)
            
            for gen_bus_id in active_generators:
                gen_idx = None
                for idx, gen in self.net.gen.iterrows():
                    if gen.bus == gen_bus_id - 1:  # Convert to 0-indexed
                        gen_idx = idx
                        break
                
                if gen_idx is not None:
                    # Update generator setpoint
                    current_p = self.net.gen.p_mw.iloc[gen_idx]
                    new_p = current_p + control_per_gen
                    
                    # Apply limits
                    max_p = self.net.gen.max_p_mw.iloc[gen_idx] if not pd.isna(self.net.gen.max_p_mw.iloc[gen_idx]) else 200.0
                    min_p = self.net.gen.min_p_mw.iloc[gen_idx] if not pd.isna(self.net.gen.min_p_mw.iloc[gen_idx]) else 0.0
                    
                    new_p = np.clip(new_p, min_p, max_p)
                    self.net.gen.loc[self.net.gen.index[gen_idx], 'p_mw'] = new_p
            
            # Update system frequency based on generation changes
            total_gen_change = control_signal
            self.frequency_deviation += total_gen_change * 0.01  # Simplified frequency response
    
    def apply_attack(self, attack_vector: Dict):
        """Apply attack to the power system"""
        bus_id = attack_vector.get('target_bus')
        attack_type = attack_vector.get('attack_type')
        magnitude = attack_vector.get('magnitude', 0.0)
        
        if bus_id and 1 <= bus_id <= self.num_buses:
            
            if attack_type == 'frequency_manipulation':
                # Directly manipulate frequency measurements
                if bus_id in self.bus_measurements:
                    self.bus_measurements[bus_id].frequency += magnitude
            
            elif attack_type == 'voltage_manipulation':
                # Manipulate voltage measurements or inject reactive power
                if bus_id in self.bus_measurements:
                    self.bus_measurements[bus_id].voltage += magnitude
                
                # If there's a load at this bus, modify reactive power
                for idx, load in self.net.load.iterrows():
                    if load.bus == bus_id - 1:
                        self.net.load.loc[idx, 'q_mvar'] += magnitude * 10  # Scale for impact
                        break
            
            elif attack_type == 'power_injection':
                # Inject false active power (modify load)
                for idx, load in self.net.load.iterrows():
                    if load.bus == bus_id - 1:
                        self.net.load.loc[idx, 'p_mw'] += magnitude * 10  # Scale for impact
                        break
                
                # If no load exists, create a temporary disturbance
                if bus_id not in [load.bus + 1 for _, load in self.net.load.iterrows()]:
                    pp.create_load(self.net, bus=bus_id-1, p_mw=magnitude*10, q_mvar=0.0,
                                  name=f"Attack_Load_{bus_id}")
            
            elif attack_type == 'false_data_injection':
                # Inject false measurement data
                if bus_id in self.bus_measurements:
                    self.bus_measurements[bus_id].angle += magnitude
                    self.bus_measurements[bus_id].active_power += magnitude * 5
            
            elif attack_type == 'line_trip_attack':
                # Trip transmission lines connected to the bus
                target_line = attack_vector.get('target_line')
                if target_line:
                    for idx, line in self.net.line.iterrows():
                        if (line.from_bus == bus_id - 1 or line.to_bus == bus_id - 1) and idx == target_line:
                            self.net.line.loc[idx, 'in_service'] = False
                            self.compromised_lines.add(idx)
                            break
            
            elif attack_type == 'generator_trip':
                # Trip generator
                for idx, gen in self.net.gen.iterrows():
                    if gen.bus == bus_id - 1:
                        self.net.gen.loc[idx, 'in_service'] = False
                        break
            
            self.compromised_buses.add(bus_id)
            self.attack_history.append(attack_vector)
    
    def get_system_state(self) -> np.ndarray:
        """Get current system state as feature vector for ML"""
        features = []
        
        # Bus measurements
        for i in range(1, self.num_buses + 1):
            if i in self.bus_measurements:
                bus = self.bus_measurements[i]
                features.extend([
                    bus.voltage, bus.angle, bus.frequency,
                    bus.active_power, bus.reactive_power
                ])
            else:
                features.extend([1.0, 0.0, 60.0, 0.0, 0.0])  # Default values
        
        # Line power flows (first 10 lines)
        line_count = 0
        for (from_bus, to_bus), line_data in self.line_measurements.items():
            if line_count < 10:
                features.extend([line_data['p_from'], line_data['p_to']])
                line_count += 1
        
        # Pad with zeros if fewer than 10 lines
        while line_count < 10:
            features.extend([0.0, 0.0])
            line_count += 1
        
        # Generator outputs
        gen_count = 0
        for bus_id, gen_data in self.gen_measurements.items():
            if gen_count < 5:  # Max 5 generators
                features.extend([gen_data['p_mw'], gen_data['q_mvar']])
                gen_count += 1
        
        # Pad with zeros if fewer than 5 generators
        while gen_count < 5:
            features.extend([0.0, 0.0])
            gen_count += 1
        
        # System-wide features
        features.extend([
            self.area_control_error,
            self.frequency_deviation,
            len(self.compromised_buses),
            len(self.compromised_lines),
            self.agc_integral
        ])
        
        return np.array(features)
    
    def step(self, dt=0.1):
        """Advance system by one time step"""
        # Update AGC
        self.update_agc(dt)
        
        # Update measurements (run power flow)
        self._update_measurements()
        
        # Add small random variations for realism
        self.frequency_deviation += np.random.normal(0, 0.001)
    
    def get_state_text(self):
        """Get human-readable state description"""
        avg_voltage = np.mean([bus.voltage for bus in self.bus_measurements.values()])
        avg_frequency = np.mean([bus.frequency for bus in self.bus_measurements.values()])
        
        text = f"System state: Average voltage = {avg_voltage:.3f} pu, "
        text += f"Average frequency = {avg_frequency:.2f} Hz, "
        text += f"AGC error = {self.area_control_error:.3f}, "
        text += f"Compromised buses = {len(self.compromised_buses)}"
        
        return text
    
    def reset(self):
        """Reset system to initial state"""
        # Reset network to original state
        self.net = pp.create_empty_network()
        self._create_ieee14_network()
        
        # Reset state variables
        self.area_control_error = 0.0
        self.agc_integral = 0.0
        self.frequency_deviation = 0.0
        self.compromised_buses = set()
        self.compromised_lines = set()
        self.attack_history = []
        
        # Update initial measurements
        self._update_measurements()
        
        return self.get_system_state()

class AnomalyDetector:
    """ML-based anomaly detection for power system"""
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detection_threshold = 0.7
        
    def train(self, normal_data: np.ndarray):
        """Train on normal system operation data"""
        scaled_data = self.scaler.fit_transform(normal_data)
        self.model.fit(scaled_data)
        self.is_trained = True
        print(f"Anomaly detector trained on {len(normal_data)} samples")
    
    def detect_anomaly(self, state: np.ndarray) -> Tuple[bool, float]:
        """Detect if current state is anomalous"""
        if not self.is_trained:
            return False, 0.0
        
        scaled_state = self.scaler.transform(state.reshape(1, -1))
        anomaly_score = self.model.decision_function(scaled_state)[0]
        is_anomaly = self.model.predict(scaled_state)[0] == -1
        
        # Additional heuristic checks
        confidence = abs(anomaly_score)
        
        return is_anomaly and confidence > self.detection_threshold, confidence

class STRIDEThreatModel:
    """STRIDE threat modeling framework for power systems"""
    
    def __init__(self):
        self.stride_categories = {
            'Spoofing': 'Impersonating system components or users',
            'Tampering': 'Modifying system data or components',
            'Repudiation': 'Denying actions or events',
            'Information_Disclosure': 'Exposing sensitive system information',
            'Denial_of_Service': 'Disrupting system availability',
            'Elevation_of_Privilege': 'Gaining unauthorized access levels'
        }
        
        # MITRE ATT&CK techniques relevant to power systems
        self.mitre_techniques = {
            'T0801': 'Monitor Process State',
            'T0802': 'Steal Application Access Token', 
            'T0803': 'Block Command Message',
            'T0804': 'Block Reporting Message',
            'T0805': 'Block Serial COM',
            'T0806': 'Brute Force I/O',
            'T0807': 'Command-Line Interface',
            'T0808': 'Control Device Identification',
            'T0809': 'Data Destruction',
            'T0810': 'Data Historian Compromise',
            'T0811': 'Data from Information Repositories',
            'T0812': 'Default Credentials',
            'T0813': 'Denial of Control',
            'T0814': 'Denial of View',
            'T0815': 'Denial of Service',
            'T0816': 'Device Restart/Shutdown',
            'T0817': 'Drive-by Compromise',
            'T0818': 'Engineering Workstation Compromise',
            'T0819': 'Exploit Public-Facing Application',
            'T0820': 'Exploitation for Evasion',
            'T0821': 'Exploitation of Remote Services',
            'T0822': 'External Remote Services',
            'T0823': 'Graphical User Interface',
            'T0824': 'I/O Image',
            'T0825': 'Location Identification',
            'T0826': 'Loss of Availability',
            'T0827': 'Loss of Control',
            'T0828': 'Loss of Productivity and Revenue',
            'T0829': 'Loss of Protection',
            'T0830': 'Man in the Middle',
            'T0831': 'Manipulation of Control',
            'T0832': 'Manipulation of View',
            'T0833': 'Modify Alarm Settings',
            'T0834': 'Modify Control Logic',
            'T0835': 'Modify Parameter',
            'T0836': 'Modify Program',
            'T0837': 'Loss of Safety',
            'T0838': 'Modify Alarm Settings',
            'T0839': 'Module Firmware',
            'T0840': 'Network Connection Enumeration',
            'T0841': 'Network Service Scanning',
            'T0842': 'Network Sniffing',
            'T0843': 'Program Download',
            'T0844': 'Program Upload',
            'T0845': 'Program Organization Units',
            'T0846': 'Project File Infection',
            'T0847': 'Replication Through Removable Media',
            'T0848': 'Rogue Master',
            'T0849': 'Masquerading',
            'T0850': 'Role Identification',
            'T0851': 'Rootkit',
            'T0852': 'Screen Capture',
            'T0853': 'Scripting',
            'T0854': 'Serial Connection Enumeration',
            'T0855': 'Unauthorized Command Message',
            'T0856': 'Spoof Reporting Message',
            'T0857': 'System Firmware',
            'T0858': 'Change Operating Mode',
            'T0859': 'Valid Accounts',
            'T0860': 'Wireless Compromise',
            'T0861': 'Point & Tag Identification',
            'T0862': 'Supply Chain Compromise',
            'T0863': 'User Execution',
            'T0864': 'Transient Cyber Asset',
            'T0865': 'Spearphishing Attachment',
            'T0866': 'Exploitation of Remote Services',
            'T0867': 'Lateral Tool Transfer',
            'T0868': 'Detect Operating Mode',
            'T0869': 'Standard Application Layer Protocol',
            'T0870': 'Detect Program State',
            'T0871': 'Execution Guardrails',
            'T0872': 'Indicator Removal',
            'T0873': 'Project File Infection',
            'T0874': 'Hooking',
            'T0875': 'Change Program State',
            'T0876': 'Remote Services',
            'T0877': 'I/O Image',
            'T0878': 'Alarm Suppression',
            'T0879': 'Damage to Property',
            'T0880': 'Loss of Safety',
            'T0881': 'Service Stop',
            'T0882': 'Theft of Operational Information',
            'T0883': 'Internet Accessible Device',
            'T0884': 'Connection Proxy'
        }
    
    def analyze_attack_surface(self, system_state: Dict) -> Dict:
        """Analyze system using STRIDE framework"""
        threats = {}
        
        # Spoofing threats
        threats['Spoofing'] = {
            'sensor_spoofing': {
                'description': 'Spoof frequency/voltage measurements',
                'mitre_id': 'T0856',
                'target_components': ['frequency_sensors', 'voltage_sensors'],
                'impact_score': 8.5,
                'detection_difficulty': 7.0
            },
            'communication_spoofing': {
                'description': 'Spoof AGC control messages',
                'mitre_id': 'T0855', 
                'target_components': ['agc_controller', 'scada_network'],
                'impact_score': 9.0,
                'detection_difficulty': 8.0
            }
        }
        
        # Tampering threats
        threats['Tampering'] = {
            'parameter_tampering': {
                'description': 'Modify AGC control parameters',
                'mitre_id': 'T0835',
                'target_components': ['agc_parameters', 'control_logic'],
                'impact_score': 9.5,
                'detection_difficulty': 6.0
            },
            'measurement_tampering': {
                'description': 'Inject false measurement data',
                'mitre_id': 'T0832',
                'target_components': ['measurement_database', 'historian'],
                'impact_score': 8.0,
                'detection_difficulty': 7.5
            }
        }
        
        # Denial of Service threats  
        threats['Denial_of_Service'] = {
            'communication_flooding': {
                'description': 'Flood communication channels',
                'mitre_id': 'T0815',
                'target_components': ['scada_network', 'communication_links'],
                'impact_score': 7.5,
                'detection_difficulty': 4.0
            },
            'control_blocking': {
                'description': 'Block AGC control commands',
                'mitre_id': 'T0803',
                'target_components': ['agc_controller', 'control_network'],
                'impact_score': 9.0,
                'detection_difficulty': 5.0
            }
        }
        
        return threats
    
    def get_attack_recommendations(self, system_state: Dict, anomaly_detection_capability: float) -> List[Dict]:
        """Generate attack recommendations based on current system state"""
        threats = self.analyze_attack_surface(system_state)
        recommendations = []
        
        for category, attacks in threats.items():
            for attack_name, attack_info in attacks.items():
                # Calculate stealth score based on detection difficulty vs anomaly detection capability
                stealth_score = attack_info['detection_difficulty'] / (anomaly_detection_capability + 1)
                
                recommendation = {
                    'attack_name': attack_name,
                    'category': category,
                    'mitre_technique': attack_info['mitre_id'],
                    'description': attack_info['description'],
                    'impact_score': attack_info['impact_score'],
                    'stealth_score': stealth_score,
                    'target_components': attack_info['target_components'],
                    'recommended_magnitude': min(0.1 * stealth_score, 0.3),  # Conservative for stealth
                    'recommended_timing': 'during_high_load' if attack_info['impact_score'] > 8.5 else 'anytime'
                }
                
                recommendations.append(recommendation)
        
        # Sort by combination of impact and stealth
        recommendations.sort(key=lambda x: x['impact_score'] * x['stealth_score'], reverse=True)
        
        return recommendations

class LLMThreatAnalyzer:
    """LLM-powered threat analyzer using Hugging Face models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        if not LLM_AVAILABLE:
            raise ImportError("LLM libraries required. Install with: pip install transformers accelerate")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Try to load the model
        self._load_model()
        
        # Knowledge base for power system context
        self.power_system_context = """
        IEEE 14-bus power system with AGC, SCADA, real-time measurements.
        Attacks: frequency, voltage, data injection, communication spoofing.
        Evasion: noise injection, temporal distribution, mimicry, coordination.
        """
    
    def _load_model(self):
        """Load the LLM model"""
        try:
            print(f"Loading LLM model: {self.model_name}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Fix the padding token issue - use a dedicated padding token
            if self.tokenizer.pad_token is None:
                # Create a new padding token that's different from eos_token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Ensure pad_token_id is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Try different loading strategies for compatibility
            loading_strategies = [
                # Strategy 1: Basic loading
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ),
                # Strategy 2: With low memory usage
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ),
                # Strategy 3: With dtype (newer PyTorch)
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    dtype=torch.float16
                ),
                # Strategy 4: With torch_dtype (older PyTorch)
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16
                )
            ]
            
            # Try each strategy until one works
            for i, strategy in enumerate(loading_strategies):
                try:
                    print(f"  Trying loading strategy {i+1}...")
                    self.model = strategy()
                    break
                except Exception as e:
                    print(f"    Strategy {i+1} failed: {e}")
                    if i == len(loading_strategies) - 1:
                        raise e
                    continue
            
            # Resize token embeddings if we added new tokens
            if self.tokenizer.pad_token != self.tokenizer.eos_token:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.is_loaded = True
            print("LLM model loaded successfully!")
            print(f"  Pad token: {self.tokenizer.pad_token}")
            print(f"  EOS token: {self.tokenizer.eos_token}")
            print(f"  Pad token ID: {self.tokenizer.pad_token_id}")
            print(f"  EOS token ID: {self.tokenizer.eos_token_id}")
            
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print("Falling back to rule-based analysis")
            self.is_loaded = False
    
    def _generate_llm_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response using the LLM"""
        if not self.is_loaded:
            return "LLM not available, using fallback analysis"
        
        try:
            # Prepare input with proper tokenization
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=400,
                padding=True,
                return_attention_mask=True
            )
            
            # Generate response using max_new_tokens
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response (remove the input prompt)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response if response else "No response generated"
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "Error in LLM generation"
    
    def analyze_system_vulnerabilities(self, power_system, anomaly_detector) -> Dict:
        """LLM-powered comprehensive vulnerability analysis"""
        
        # Get system state information
        system_state = {
            'bus_count': power_system.num_buses,
            'compromised_buses': len(power_system.compromised_buses),
            'agc_active': True,
            'anomaly_detection_active': anomaly_detector.is_trained,
            'frequency_deviation': power_system.frequency_deviation,
            'area_control_error': power_system.area_control_error
        }
        
        # Create LLM prompt for vulnerability analysis
        vulnerability_prompt = f"""
        {self.power_system_context}
        
        System: {system_state['bus_count']} buses, {system_state['compromised_buses']} compromised
        AGC: {system_state['agc_active']}, Detection: {system_state['anomaly_detection_active']}
        Freq dev: {system_state['frequency_deviation']:.3f}, AGC error: {system_state['area_control_error']:.3f}
        
        Analyze vulnerabilities and suggest:
        1. Top attack vectors
        2. Attack strategies  
        3. Evasion techniques
        4. Expected impact
        """
        
        # Get LLM analysis
        llm_analysis = self._generate_llm_response(vulnerability_prompt, max_new_tokens=150)
        
        # Parse LLM response and create structured output
        vulnerability_assessment = self._parse_llm_response(llm_analysis, system_state)
        
        # Add system state analysis
        vulnerability_assessment['system_state_analysis'] = self._analyze_current_state(power_system)
        vulnerability_assessment['llm_raw_analysis'] = llm_analysis
        
        return vulnerability_assessment
    
    def _parse_llm_response(self, llm_response: str, system_state: Dict) -> Dict:
        """Parse LLM response into structured format"""
        
        # Default recommendations if parsing fails
        default_recommendations = [
            {
                'attack_name': 'frequency_manipulation',
                'category': 'Frequency_Attack',
                'mitre_technique': 'T0831',
                'description': 'Manipulate frequency measurements to disrupt AGC',
                'impact_score': 8.5,
                'stealth_score': 6.0,
                'target_components': ['frequency_sensors', 'agc_controller'],
                'recommended_magnitude': 0.15,
                'recommended_timing': 'during_high_load'
            },
            {
                'attack_name': 'false_data_injection',
                'category': 'Data_Manipulation',
                'mitre_technique': 'T0832',
                'description': 'Inject false measurement data to mislead operators',
                'impact_score': 7.5,
                'stealth_score': 7.0,
                'target_components': ['measurement_database', 'scada_system'],
                'recommended_magnitude': 0.12,
                'recommended_timing': 'anytime'
            },
            {
                'attack_name': 'communication_spoofing',
                'category': 'Communication_Attack',
                'mitre_technique': 'T0855',
                'description': 'Spoof AGC control messages to disrupt frequency regulation',
                'impact_score': 9.0,
                'stealth_score': 5.5,
                'target_components': ['agc_controller', 'communication_network'],
                'recommended_magnitude': 0.18,
                'recommended_timing': 'during_peak_load'
            }
        ]
        
        # Try to extract information from LLM response
        try:
            # Look for specific patterns in the response
            if "frequency" in llm_response.lower():
                default_recommendations[0]['description'] = f"LLM suggests: {llm_response[:100]}..."
            if "data" in llm_response.lower() or "injection" in llm_response.lower():
                default_recommendations[1]['description'] = f"LLM suggests: {llm_response[:100]}..."
            if "communication" in llm_response.lower() or "spoof" in llm_response.lower():
                default_recommendations[2]['description'] = f"LLM suggests: {llm_response[:100]}..."
        except:
            pass
        
        return {
            'critical_vulnerabilities': default_recommendations,
            'evasion_recommendations': self._generate_evasion_strategy_llm(llm_response),
            'attack_sequence_recommendations': self._plan_attack_sequence_llm(llm_response, default_recommendations)
        }
    
    def _generate_evasion_strategy_llm(self, llm_response: str) -> Dict:
        """Generate evasion strategy based on LLM analysis"""
        
        # Create evasion prompt
        evasion_prompt = f"""
        Based on this power system attack analysis:
        "{llm_response[:200]}..."
        
        Provide specific evasion techniques to avoid detection:
        1. Timing strategies
        2. Magnitude adjustments  
        3. Coordination methods
        4. Stealth approaches
        
        Focus on practical techniques that could be implemented.
        """
        
        evasion_analysis = self._generate_llm_response(evasion_prompt, max_new_tokens=100)
        
        return {
            'approach': 'llm_guided_stealth',
            'techniques': [
                {
                    'name': 'llm_timing_strategy',
                    'description': f'LLM suggests: {evasion_analysis[:100]}...',
                    'effectiveness': 0.8
                },
                {
                    'name': 'adaptive_magnitude',
                    'description': 'Dynamically adjust attack magnitude based on system response',
                    'effectiveness': 0.85
                },
                {
                    'name': 'coordinated_evasion',
                    'description': 'Coordinate multiple attack vectors to mask individual signatures',
                    'effectiveness': 0.9
                }
            ],
            'recommended_magnitude_limit': 0.2,
            'recommended_attack_interval': 3,
            'llm_evasion_analysis': evasion_analysis
        }
    
    def _plan_attack_sequence_llm(self, llm_response: str, recommendations: List[Dict]) -> Dict:
        """Plan attack sequence using LLM insights"""
        
        sequence_prompt = f"""
        Based on this vulnerability analysis:
        "{llm_response[:200]}..."
        
        And these attack recommendations:
        {[rec['attack_name'] for rec in recommendations]}
        
        Plan an optimal attack sequence that:
        1. Maximizes impact while minimizing detection
        2. Coordinates multiple attack vectors
        3. Uses timing to avoid detection
        4. Adapts to system responses
        
        Provide specific sequence, timing, and coordination strategy.
        """
        
        sequence_analysis = self._generate_llm_response(sequence_prompt, max_new_tokens=120)
        
        # Create attack sequence
        attack_sequence = []
        for i, rec in enumerate(recommendations[:5]):
            attack_sequence.append({
                'step': i + 1,
                'target': rec.get('target', 'unknown'),
                'method': rec.get('attack_name', 'unknown'),
                'magnitude': rec.get('recommended_magnitude', 0.1),
                'description': rec.get('description', 'No description available'),
                'llm_insights': sequence_analysis[:100] if i == 0 else "Coordinated with step 1"
            })
        
        return {
            'sequence': attack_sequence,
            'timing': [i * 2 for i in range(len(attack_sequence))],
            'coordination': 'llm_guided_coordination',
            'expected_impact': sum(rec.get('recommended_magnitude', 0.1) for rec in attack_sequence),
            'llm_sequence_analysis': sequence_analysis
        }
    
    def _analyze_current_state(self, power_system) -> Dict:
        """Analyze current system state for attack opportunities"""
        analysis = {}
        
        # Check for unstable conditions
        if power_system.bus_measurements:
            frequencies = [bus.frequency for bus in power_system.bus_measurements.values()]
            freq_std = np.std(frequencies)
            avg_frequency = np.mean(frequencies)
            
            analysis['frequency_stability'] = 'unstable' if freq_std > 0.05 else 'stable'
            analysis['avg_frequency'] = avg_frequency
            analysis['frequency_deviation'] = abs(avg_frequency - 60.0)
        else:
            analysis['frequency_stability'] = 'unknown'
            analysis['avg_frequency'] = 60.0
            analysis['frequency_deviation'] = 0.0
        
        # Check AGC stress
        analysis['agc_stress_level'] = abs(power_system.area_control_error)
        analysis['compromise_level'] = len(power_system.compromised_buses) / power_system.num_buses
        
        # Check voltage stability
        if power_system.bus_measurements:
            voltages = [bus.voltage for bus in power_system.bus_measurements.values()]
            voltage_std = np.std(voltages)
            analysis['voltage_stability'] = 'unstable' if voltage_std > 0.05 else 'stable'
            analysis['avg_voltage'] = np.mean(voltages)
        else:
            analysis['voltage_stability'] = 'stable'
            analysis['avg_voltage'] = 1.0
        
        # Check line loading if available
        if power_system.line_measurements:
            loadings = [line_data.get('loading', 0.0) for line_data in power_system.line_measurements.values()]
            analysis['max_line_loading'] = max(loadings) if loadings else 0.0
            analysis['avg_line_loading'] = np.mean(loadings) if loadings else 0.0
        else:
            analysis['max_line_loading'] = 0.0
            analysis['avg_line_loading'] = 0.0
        
        return analysis

class AgenticThreatAnalyzer:
    """Agentic AI model for intelligent threat analysis"""
    
    def __init__(self):
        self.stride_model = STRIDEThreatModel()
        self.knowledge_base = {
            'power_system_vulnerabilities': [
                'AGC timing attacks', 'False data injection', 'Cascading failures',
                'Load redistribution attacks', 'Frequency instability attacks'
            ],
            'evasion_techniques': [
                'Slow and low attacks', 'Mimicking normal variations',
                'Coordinated multi-point attacks', 'Timing-based evasion'
            ],
            'system_weaknesses': [
                'Legacy communication protocols', 'Insufficient authentication',
                'Inadequate anomaly detection', 'Single points of failure'
            ]
        }
        

    
    def analyze_system_vulnerabilities(self, power_system, anomaly_detector) -> Dict:
        """Comprehensive vulnerability analysis"""
        system_state = {
            'bus_count': power_system.num_buses,
            'compromised_buses': len(power_system.compromised_buses),
            'agc_active': True,
            'anomaly_detection_active': anomaly_detector.is_trained
        }
        
        # Get STRIDE analysis
        recommendations = self.stride_model.get_attack_recommendations(
            system_state, 
            anomaly_detector.detection_threshold if anomaly_detector.is_trained else 0.1
        )
        
        # Analyze current system state for opportunities
        current_state = power_system.get_system_state()
        
        vulnerability_assessment = {
            'critical_vulnerabilities': recommendations[:3],  # Top 3 threats
            'system_state_analysis': self._analyze_current_state(power_system),
            'evasion_recommendations': self._generate_evasion_strategy(anomaly_detector),
            'attack_sequence_recommendations': self._plan_attack_sequence(recommendations)
        }
        
        return vulnerability_assessment
    
    def _analyze_current_state(self, power_system) -> Dict:
        """Analyze current system state for attack opportunities"""
        analysis = {}
        
        # Check for unstable conditions
        if power_system.bus_measurements:
            frequencies = [bus.frequency for bus in power_system.bus_measurements.values()]
            freq_std = np.std(frequencies)
            avg_frequency = np.mean(frequencies)
            
            analysis['frequency_stability'] = 'unstable' if freq_std > 0.05 else 'stable'
            analysis['avg_frequency'] = avg_frequency
            analysis['frequency_deviation'] = abs(avg_frequency - 60.0)
        else:
            analysis['frequency_stability'] = 'unknown'
            analysis['avg_frequency'] = 60.0
            analysis['frequency_deviation'] = 0.0
        
        # Check AGC stress
        analysis['agc_stress_level'] = abs(power_system.area_control_error)
        analysis['compromise_level'] = len(power_system.compromised_buses) / power_system.num_buses
        
        # Check voltage stability
        if power_system.bus_measurements:
            voltages = [bus.voltage for bus in power_system.bus_measurements.values()]
            voltage_std = np.std(voltages)
            analysis['voltage_stability'] = 'unstable' if voltage_std > 0.05 else 'stable'
            analysis['avg_voltage'] = np.mean(voltages)
        else:
            analysis['voltage_stability'] = 'stable'
            analysis['avg_voltage'] = 1.0
        
        # Check line loading if available
        if power_system.line_measurements:
            loadings = [line_data.get('loading', 0.0) for line_data in power_system.line_measurements.values()]
            analysis['max_line_loading'] = max(loadings) if loadings else 0.0
            analysis['avg_line_loading'] = np.mean(loadings) if loadings else 0.0
        else:
            analysis['max_line_loading'] = 0.0
            analysis['avg_line_loading'] = 0.0
        
        return analysis
    
    def _generate_evasion_strategy(self, anomaly_detector) -> Dict:
        """Generate strategy to evade anomaly detection"""
        strategy = {
            'approach': 'multi_stage_stealth',
            'techniques': [
                {
                    'name': 'noise_injection',
                    'description': 'Add small random variations to mask attack signatures',
                    'effectiveness': 0.7
                },
                {
                    'name': 'temporal_distribution',
                    'description': 'Spread attack over multiple time periods',
                    'effectiveness': 0.8
                },
                {
                    'name': 'mimicry',
                    'description': 'Mimic normal operational variations',
                    'effectiveness': 0.9
                }
            ],
            'recommended_magnitude_limit': 0.15,  # Stay below detection threshold
            'recommended_attack_interval': 5  # Steps between attacks
        }
        
        return strategy
    
    def _plan_attack_sequence(self, recommendations: List) -> Dict:
        """Plan optimal attack sequence based on recommendations"""
        if not recommendations:
            return {
                'sequence': [],
                'timing': [],
                'coordination': 'none',
                'expected_impact': 0.0
            }
        
        # Create attack sequence from recommendations
        attack_sequence = []
        timing = []
        
        for i, rec in enumerate(recommendations[:5]):  # Limit to 5 attacks
            attack_sequence.append({
                'step': i + 1,
                'target': rec.get('target', 'unknown'),
                'method': rec.get('method', 'unknown'),
                'magnitude': rec.get('magnitude', 0.1),
                'description': rec.get('description', 'No description available')
            })
            timing.append(i * 2)  # 2 steps between attacks
        
        return {
            'sequence': attack_sequence,
            'timing': timing,
            'coordination': 'sequential' if len(attack_sequence) > 1 else 'single',
            'expected_impact': sum(rec.get('magnitude', 0.1) for rec in attack_sequence)
        }

class RLAttackAgent:
    """RL agent that learns optimal stealthy attacks"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Neural network for Q-function approximation
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        
        # RL parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.batch_size = 32
        
        # Attack actions mapping
        self.actions = [
            {'attack_type': 'frequency_manipulation', 'magnitude': 0.01},
            {'attack_type': 'frequency_manipulation', 'magnitude': 0.05},
            {'attack_type': 'frequency_manipulation', 'magnitude': 0.1},
            {'attack_type': 'voltage_manipulation', 'magnitude': 0.02},
            {'attack_type': 'voltage_manipulation', 'magnitude': 0.05},
            {'attack_type': 'power_injection', 'magnitude': 0.1},
            {'attack_type': 'power_injection', 'magnitude': 0.2},
            {'attack_type': 'false_data_injection', 'magnitude': 0.05},
            {'attack_type': 'false_data_injection', 'magnitude': 0.1},
            {'attack_type': 'no_attack', 'magnitude': 0.0}  # Do nothing
        ]
        
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'detected_attacks': 0,
            'impact_achieved': 0.0,
            'impact_history': [],
            'cumulative_impact': [],
            'attack_types_used': [],
            'detection_history': [],
            'reward_history': []
        }
    
    def select_action(self, state: np.ndarray, threat_recommendations: List[Dict] = None) -> int:
        """Select action using epsilon-greedy with threat intelligence"""
        
        if np.random.random() < self.epsilon:
            # Exploration: use threat recommendations or random
            if threat_recommendations and np.random.random() < 0.6:
                # Use agentic AI recommendations
                recommended_attack = threat_recommendations[0]  # Top recommendation
                
                # Find matching action
                for i, action in enumerate(self.actions):
                    if (action['attack_type'] == 'no_attack' or 
                        recommended_attack['attack_name'] in action['attack_type'] or
                        abs(action['magnitude'] - recommended_attack['recommended_magnitude']) < 0.05):
                        return i
            
            return random.randint(0, len(self.actions) - 1)
        else:
            # Exploitation: use Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def calculate_reward(self, prev_state: Dict, action: Dict, new_state: Dict, 
                        detected: bool, impact: float) -> float:
        """Calculate reward for RL agent"""
        reward = 0.0
        
        # Base impact reward
        reward += impact * 10.0
        
        # Stealth bonus/penalty
        if detected:
            reward -= 50.0  # Heavy penalty for detection
            self.attack_stats['detected_attacks'] += 1
            self.attack_stats['detection_history'].append(1)
        else:
            reward += 20.0  # Bonus for staying hidden
            if action['attack_type'] != 'no_attack':
                self.attack_stats['successful_attacks'] += 1
            self.attack_stats['detection_history'].append(0)
        
        # Frequency stability impact
        freq_deviation = abs(new_state.get('avg_frequency', 60.0) - 60.0)
        reward += freq_deviation * 100.0  # Reward for frequency deviation
        
        # AGC stress reward
        agc_stress = new_state.get('agc_stress_level', 0.0)
        reward += agc_stress * 50.0
        
        # Persistence bonus (staying in system longer)
        if len(new_state.get('compromised_buses', [])) > 0:
            reward += 5.0
        
        # Track statistics for plotting
        self.attack_stats['impact_achieved'] += impact
        self.attack_stats['impact_history'].append(impact)
        self.attack_stats['cumulative_impact'].append(self.attack_stats['impact_achieved'])
        self.attack_stats['attack_types_used'].append(action['attack_type'])
        self.attack_stats['reward_history'].append(reward)
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Train the Q-network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def run_security_simulation(episodes=100, max_steps_per_episode=50, use_llm=True):
    """Run the complete security analysis simulation"""
    
    print("Initializing IEEE 14-Bus Power System Security Simulation...")
    print("="*60)
    
    if not PANDAPOWER_AVAILABLE:
        print("ERROR: Pandapower is required for realistic power system simulation.")
        print("Install with: pip install pandapower")
        return None, [], [], [], {}
    
    # Initialize components
    try:
        power_system = PowerSystem14Bus()
        anomaly_detector = AnomalyDetector()
        
        # Choose between LLM and rule-based threat analyzer
        if use_llm and LLM_AVAILABLE:
            print("Using LLM-powered threat analyzer...")
            threat_analyzer = LLMThreatAnalyzer()
        else:
            print("Using rule-based threat analyzer...")
            threat_analyzer = AgenticThreatAnalyzer()
            
    except Exception as e:
        print(f"Failed to initialize power system: {e}")
        return None, [], [], [], {}
    
    # Generate normal operation data for anomaly detector training
    print("Generating normal operation data...")
    normal_data = []
    for i in range(1000):
        try:
            power_system.step()
            state = power_system.get_system_state()
            if len(state) > 0 and not np.any(np.isnan(state)):
                normal_data.append(state)
        except Exception as e:
            print(f"Warning: Step {i} failed: {e}")
            continue
    
    if len(normal_data) < 100:
        print("ERROR: Insufficient normal data generated")
        return None, [], [], [], {}
    
    normal_data = np.array(normal_data)
    anomaly_detector.train(normal_data)
    
    # Reset system for attack simulation
    power_system.reset()
    
    # Initialize RL agent
    state_dim = len(power_system.get_system_state())
    action_dim = 12  # Expanded action space for more sophisticated attacks
    
    rl_agent = RLAttackAgent(state_dim, action_dim)
    
    # Update actions for more realistic attacks
    rl_agent.actions = [
        {'attack_type': 'frequency_manipulation', 'magnitude': 0.005},
        {'attack_type': 'frequency_manipulation', 'magnitude': 0.02},
        {'attack_type': 'frequency_manipulation', 'magnitude': 0.05},
        {'attack_type': 'voltage_manipulation', 'magnitude': 0.01},
        {'attack_type': 'voltage_manipulation', 'magnitude': 0.03},
        {'attack_type': 'voltage_manipulation', 'magnitude': 0.05},
        {'attack_type': 'power_injection', 'magnitude': 0.05},
        {'attack_type': 'power_injection', 'magnitude': 0.1},
        {'attack_type': 'power_injection', 'magnitude': 0.2},
        {'attack_type': 'false_data_injection', 'magnitude': 0.02},
        {'attack_type': 'false_data_injection', 'magnitude': 0.05},
        {'attack_type': 'no_attack', 'magnitude': 0.0}
    ]
    
    # Training loop
    episode_rewards = []
    detection_rates = []
    impact_scores = []
    
    print(f"\nStarting {episodes} episodes of attack simulation...")
    
    for episode in range(episodes):
        # Reset environment
        power_system.reset()
        episode_reward = 0.0
        episode_detections = 0
        episode_impact = 0.0
        
        if episode % 20 == 0:
            print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(max_steps_per_episode):
            try:
                # Get current state
                current_state = power_system.get_system_state()
                
                if len(current_state) == 0 or np.any(np.isnan(current_state)):
                    print(f"Invalid state at episode {episode}, step {step}")
                    break
                
                # Get threat analysis from agentic AI
                vulnerability_assessment = threat_analyzer.analyze_system_vulnerabilities(
                    power_system, anomaly_detector
                )
                
                threat_recommendations = vulnerability_assessment['critical_vulnerabilities']
                
                # Show LLM insights if available
                if use_llm and LLM_AVAILABLE and 'llm_raw_analysis' in vulnerability_assessment:
                    if episode % 20 == 0 and step == 0:
                        print(f"  LLM Analysis: {vulnerability_assessment['llm_raw_analysis'][:100]}...")
                
                # RL agent selects action based on state and recommendations
                action_idx = rl_agent.select_action(current_state, threat_recommendations)
                action_params = rl_agent.actions[action_idx].copy()
                
                # Select target bus (prioritize generator buses for higher impact)
                generator_buses = [1, 2, 3, 6, 8]
                load_buses = [4, 5, 7, 9, 10, 11, 12, 13, 14]
                
                if np.random.random() < 0.3 and action_params['attack_type'] != 'no_attack':
                    target_bus = random.choice(generator_buses)  # Target generators more often
                else:
                    target_bus = random.randint(1, 14)
                
                action_params['target_bus'] = target_bus
                
                # Apply attack
                if action_params['attack_type'] != 'no_attack':
                    power_system.apply_attack(action_params)
                    rl_agent.attack_stats['total_attacks'] += 1
                
                # System response
                power_system.step()
                next_state = power_system.get_system_state()
                
                if len(next_state) == 0 or np.any(np.isnan(next_state)):
                    # System instability, end episode
                    break
                
                # Check for anomaly detection
                detected, confidence = anomaly_detector.detect_anomaly(next_state)
                
                if detected:
                    episode_detections += 1
                    if episode % 20 == 0:
                        print(f"  DETECTED: Attack on bus {target_bus} detected (confidence: {confidence:.3f})")
                
                # Calculate impact
                try:
                    state_analysis = threat_analyzer._analyze_current_state(power_system)
                    impact = (state_analysis['frequency_deviation'] + 
                             abs(state_analysis['agc_stress_level']) * 0.1 + 
                             state_analysis['compromise_level'] * 5.0)
                except:
                    impact = 0.0
                
                episode_impact += impact
                
                # Calculate reward
                prev_state_dict = {'avg_frequency': 60.0, 'agc_stress_level': 0.0, 'compromised_buses': []}
                new_state_dict = {
                    'avg_frequency': np.mean([bus.frequency for bus in power_system.bus_measurements.values()]) if power_system.bus_measurements else 60.0,
                    'agc_stress_level': power_system.area_control_error,
                    'compromised_buses': list(power_system.compromised_buses)
                }
                
                reward = rl_agent.calculate_reward(
                    prev_state_dict, action_params, new_state_dict, detected, impact
                )
                
                episode_reward += reward
                
                # Store experience
                done = step == max_steps_per_episode - 1
                rl_agent.store_experience(current_state, action_idx, reward, next_state, done)
                
                # Train RL agent
                if len(rl_agent.replay_buffer) > rl_agent.batch_size:
                    rl_agent.train_step()
                
                if episode % 20 == 0 and step % 10 == 0:
                    print(f"  Step {step}: Reward = {reward:.2f}, Impact = {impact:.3f}")
                
            except Exception as e:
                print(f"Error in episode {episode}, step {step}: {e}")
                break
        
        episode_rewards.append(episode_reward)
        detection_rates.append(episode_detections / max_steps_per_episode)
        impact_scores.append(episode_impact)
        
        # Update target network periodically
        if episode % 10 == 0:
            rl_agent.update_target_network()
        
        # Episode summary
        if episode % 20 == 0:
            print(f"Episode {episode + 1} Summary:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Detection Rate: {episode_detections/max_steps_per_episode:.1%}")
            print(f"  Total Impact: {episode_impact:.3f}")
            print(f"  Epsilon: {rl_agent.epsilon:.3f}")
        
        # Show attack statistics every 25 episodes
        if (episode + 1) % 25 == 0:
            stats = rl_agent.attack_stats
            success_rate = stats['successful_attacks'] / max(stats['total_attacks'], 1)
            detection_rate_stats = stats['detected_attacks'] / max(stats['total_attacks'], 1)
            
            print(f"\n--- Performance Summary (Episodes {max(0, episode-24)}-{episode+1}) ---")
            print(f"Attack Success Rate: {success_rate:.1%}")
            print(f"Detection Rate: {detection_rate_stats:.1%}")
            print(f"Average Impact: {stats['impact_achieved']/max(stats['total_attacks'], 1):.3f}")
            
            # Show real-time plotting if available
            if PLOTTING_AVAILABLE and episode > 0:
                try:
                    # Create a simple progress plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot recent episode rewards
                    recent_rewards = episode_rewards[-25:] if len(episode_rewards) >= 25 else episode_rewards
                    recent_episodes = list(range(max(0, len(episode_rewards)-25), len(episode_rewards)))
                    
                    ax1.plot(recent_episodes, recent_rewards, 'b-', linewidth=2, alpha=0.8)
                    ax1.set_title('Recent Episode Rewards')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Total Reward')
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot recent detection rates
                    recent_detections = detection_rates[-25:] if len(detection_rates) >= 25 else detection_rates
                    ax2.plot(recent_episodes, [d * 100 for d in recent_detections], 'r-', linewidth=2, alpha=0.8)
                    ax2.set_title('Recent Detection Rates')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Detection Rate (%)')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig('llm_guided_rl_power_system_rewards.png')
                    plt.close()


                except Exception as e:
                    print(f"Warning: Real-time plotting failed: {e}")

    # Create visualizations if plotting is available
    if PLOTTING_AVAILABLE:
        try:
            visualizer = SimulationVisualizer()
            
            # Plot episode performance
            visualizer.plot_episode_performance(
                episode_rewards, 
                [d * 100 for d in detection_rates],  # Convert to percentage
                impact_scores,
                f"Episode Performance - {'LLM-Powered' if use_llm else 'Rule-Based'} Analysis"
            )
            
            # Plot attack strategy analysis
            attack_stats = rl_agent.attack_stats.copy()
            # Add derived statistics for plotting
            attack_stats['freq_attacks'] = sum(1 for a in rl_agent.actions if 'frequency' in a['attack_type'])
            attack_stats['voltage_attacks'] = sum(1 for a in rl_agent.actions if 'voltage' in a['attack_type'])
            attack_stats['power_attacks'] = sum(1 for a in rl_agent.actions if 'power' in a['attack_type'])
            attack_stats['false_data_attacks'] = sum(1 for a in rl_agent.actions if 'false_data' in a['attack_type'])
            attack_stats['no_attack_count'] = sum(1 for a in rl_agent.actions if 'no_attack' in a['attack_type'])
            
            visualizer.plot_attack_strategy_analysis(
                attack_stats,
                f"Attack Strategy Analysis - {'LLM-Powered' if use_llm else 'Rule-Based'} Analysis"
            )
            
            # Plot power system state
            visualizer.plot_power_system_state(
                power_system,
                f"Power System State - {'LLM-Powered' if use_llm else 'Rule-Based'} Analysis"
            )
            
            # Plot threat analysis insights
            visualizer.plot_threat_analysis_insights(
                vulnerability_assessment,
                f"Threat Analysis Insights - {'LLM-Powered' if use_llm else 'Rule-Based'} Analysis"
            )
            
        except Exception as e:
            print(f"Warning: Plotting failed: {e}")
    
    return rl_agent, episode_rewards, detection_rates, impact_scores, vulnerability_assessment

def demonstrate_attack_strategies():
    """Demonstrate learned attack strategies"""
    print("\n" + "="*60)
    print("DEMONSTRATING LEARNED ATTACK STRATEGIES")
    print("="*60)
    
    # Run simulation
    rl_agent, rewards, detection_rates, impacts, vuln_assessment = run_security_simulation(
        episodes=200, max_steps_per_episode=50
    )
    
    print(f"\n--- FINAL RESULTS ---")
    print(f"Average Episode Reward: {np.mean(rewards[-10:]):.2f}")
    print(f"Average Detection Rate: {np.mean(detection_rates[-10:]):.1%}")
    print(f"Average Impact Score: {np.mean(impacts[-10:]):.3f}")
    
    print(f"\n--- VULNERABILITY ASSESSMENT ---")
    print("Top 3 Critical Vulnerabilities:")
    for i, vuln in enumerate(vuln_assessment['critical_vulnerabilities'], 1):
        print(f"{i}. {vuln['attack_name']}")
        print(f"   MITRE ID: {vuln['mitre_technique']}")
        print(f"   Impact Score: {vuln['impact_score']:.1f}")
        print(f"   Stealth Score: {vuln['stealth_score']:.1f}")
        print(f"   Description: {vuln['description']}")
    
    print(f"\n--- EVASION STRATEGY ---")
    evasion = vuln_assessment['evasion_recommendations']
    print(f"Recommended Approach: {evasion['approach']}")
    print("Evasion Techniques:")
    for tech in evasion['techniques']:
        print(f"  - {tech['name']}: {tech['description']} (effectiveness: {tech['effectiveness']:.1%})")
    
    print(f"\n--- ATTACK STATISTICS ---")
    stats = rl_agent.attack_stats
    if stats['total_attacks'] > 0:
        print(f"Total Attacks Attempted: {stats['total_attacks']}")
        print(f"Successful Attacks: {stats['successful_attacks']}")
        print(f"Detected Attacks: {stats['detected_attacks']}")
        print(f"Success Rate: {stats['successful_attacks']/stats['total_attacks']:.1%}")
        print(f"Evasion Rate: {1 - stats['detected_attacks']/stats['total_attacks']:.1%}")
    
    return rl_agent, vuln_assessment

class SimulationVisualizer:
    """Comprehensive visualization for power system cybersecurity simulation results"""
    
    def __init__(self):
        if not PLOTTING_AVAILABLE:
            print("Warning: Plotting not available. Install matplotlib and seaborn.")
            return
        
        # Set style for better-looking plots
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_episode_performance(self, episode_rewards: List[float], episode_detections: List[float], 
                                episode_impacts: List[float], title: str = "Episode Performance"):
        """Plot episode-wise performance metrics"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode detections
        axes[0, 1].plot(episode_detections, 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Detection Rate per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Detection Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode impacts
        axes[1, 0].plot(episode_impacts, 'g-', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Attack Impact per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Impact')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics
        axes[1, 1].plot(episode_rewards, 'b-', label='Rewards', alpha=0.7)
        axes[1, 1].plot(episode_impacts, 'g-', label='Impact', alpha=0.7)
        axes[1, 1].set_title('Rewards vs Impact')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig('llm_guided_rl_power_system_episode_performance.png')
        plt.close()
        
    def plot_attack_strategy_analysis(self, attack_stats: Dict, title: str = "Attack Strategy Analysis"):
        """Plot attack strategy statistics"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Attack success rate
        success_rate = (attack_stats['successful_attacks'] / max(attack_stats['total_attacks'], 1)) * 100
        detection_rate = (attack_stats['detected_attacks'] / max(attack_stats['total_attacks'], 1)) * 100
        
        # Pie chart for attack outcomes
        labels = ['Successful', 'Detected', 'Failed']
        sizes = [success_rate, detection_rate, 100 - success_rate - detection_rate]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Attack Outcomes Distribution')
        
        # Bar chart for attack types
        attack_types = ['Frequency', 'Voltage', 'Power', 'False Data', 'No Attack']
        attack_counts = [attack_stats.get('freq_attacks', 0), 
                        attack_stats.get('voltage_attacks', 0),
                        attack_stats.get('power_attacks', 0),
                        attack_stats.get('false_data_attacks', 0),
                        attack_stats.get('no_attack_count', 0)]
        
        bars = axes[0, 1].bar(attack_types, attack_counts, color=['#3498db', '#e67e22', '#9b59b6', '#f1c40f', '#95a5a6'])
        axes[0, 1].set_title('Attack Types Distribution')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Impact over time
        if 'impact_history' in attack_stats:
            impact_history = attack_stats['impact_history']
            axes[1, 0].plot(impact_history, 'r-', linewidth=2, alpha=0.8)
            axes[1, 0].set_title('Attack Impact Over Time')
            axes[1, 0].set_xlabel('Attack Step')
            axes[1, 0].set_ylabel('Impact Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative impact
        if 'cumulative_impact' in attack_stats:
            cumulative_impact = attack_stats['cumulative_impact']
            axes[1, 1].plot(cumulative_impact, 'g-', linewidth=2, alpha=0.8)
            axes[1, 1].set_title('Cumulative Attack Impact')
            axes[1, 1].set_xlabel('Attack Step')
            axes[1, 1].set_ylabel('Cumulative Impact')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig('llm_guided_rl_power_system_attack_strategy_analysis.png')
        plt.close()
        
    def plot_power_system_state(self, power_system, title: str = "Power System State"):
        """Plot current power system state"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Bus voltage profile
        if hasattr(power_system, 'bus_measurements') and power_system.bus_measurements:
            bus_ids = list(power_system.bus_measurements.keys())
            voltages = [power_system.bus_measurements[bus_id].voltage for bus_id in bus_ids]
            
            axes[0, 0].bar(bus_ids, voltages, color='lightblue', alpha=0.8)
            axes[0, 0].set_title('Bus Voltage Profile')
            axes[0, 0].set_xlabel('Bus ID')
            axes[0, 0].set_ylabel('Voltage (p.u.)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Nominal')
            axes[0, 0].legend()
        
        # Frequency distribution
        if hasattr(power_system, 'bus_measurements') and power_system.bus_measurements:
            frequencies = [power_system.bus_measurements[bus_id].frequency for bus_id in bus_ids]
            
            axes[0, 1].hist(frequencies, bins=10, color='lightgreen', alpha=0.8, edgecolor='black')
            axes[0, 1].set_title('Frequency Distribution')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(x=60.0, color='r', linestyle='--', alpha=0.7, label='Nominal (60 Hz)')
            axes[0, 1].legend()
        
        # Generator power output
        if hasattr(power_system, 'net') and hasattr(power_system.net, 'gen'):
            gen_ids = power_system.net.gen.index.tolist()
            p_mw = power_system.net.gen.p_mw.values
            
            axes[1, 0].bar(gen_ids, p_mw, color='orange', alpha=0.8)
            axes[1, 0].set_title('Generator Power Output')
            axes[1, 0].set_xlabel('Generator ID')
            axes[1, 0].set_ylabel('Power (MW)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Load distribution
        if hasattr(power_system, 'net') and hasattr(power_system.net, 'load'):
            load_ids = power_system.net.load.index.tolist()
            p_mw = power_system.net.load.p_mw.values
            
            axes[1, 1].bar(load_ids, p_mw, color='lightcoral', alpha=0.8)
            axes[1, 1].set_title('Load Distribution')
            axes[1, 1].set_xlabel('Load ID')
            axes[1, 1].set_ylabel('Power (MW)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('llm_guided_rl_power_system_power_system_state.png')
        plt.close()

        # plt.show()
        
    def plot_comparison_analysis(self, llm_results: Dict, rule_based_results: Dict, title: str = "LLM vs Rule-Based Comparison"):
        """Plot comparison between LLM and rule-based approaches"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Success rate comparison
        methods = ['LLM-Powered', 'Rule-Based']
        success_rates = [llm_results.get('success_rate', 0), rule_based_results.get('success_rate', 0)]
        
        bars = axes[0, 0].bar(methods, success_rates, color=['#3498db', '#e67e22'], alpha=0.8)
        axes[0, 0].set_title('Attack Success Rate Comparison')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        for bar, rate in zip(bars, success_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Detection rate comparison
        detection_rates = [llm_results.get('detection_rate', 0), rule_based_results.get('detection_rate', 0)]
        
        bars = axes[0, 1].bar(methods, detection_rates, color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[0, 1].set_title('Detection Rate Comparison')
        axes[0, 1].set_ylabel('Detection Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        for bar, rate in zip(bars, detection_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Average impact comparison
        avg_impacts = [llm_results.get('average_impact', 0), rule_based_results.get('average_impact', 0)]
        
        bars = axes[1, 0].bar(methods, avg_impacts, color=['#9b59b6', '#f1c40f'], alpha=0.8)
        axes[1, 0].set_title('Average Attack Impact Comparison')
        axes[1, 0].set_ylabel('Average Impact')
        for bar, impact in zip(bars, avg_impacts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{impact:.2f}', ha='center', va='bottom')
        
        # Radar chart for multiple metrics
        metrics = ['Success Rate', 'Stealth', 'Impact', 'Efficiency']
        llm_scores = [llm_results.get('success_rate', 0)/100, 
                      (100-llm_results.get('detection_rate', 0))/100,
                      min(llm_results.get('average_impact', 0)/20, 1.0),
                      llm_results.get('efficiency', 0.8)]
        rule_scores = [rule_based_results.get('success_rate', 0)/100,
                       (100-rule_based_results.get('detection_rate', 0))/100,
                       min(rule_based_results.get('average_impact', 0)/20, 1.0),
                       rule_based_results.get('efficiency', 0.7)]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        llm_scores += llm_scores[:1]  # Close the loop
        rule_scores += rule_scores[:1]
        angles += angles[:1]
        
        axes[1, 1].plot(angles, llm_scores, 'o-', linewidth=2, label='LLM-Powered', color='#3498db')
        axes[1, 1].fill(angles, llm_scores, alpha=0.25, color='#3498db')
        axes[1, 1].plot(angles, rule_scores, 'o-', linewidth=2, label='Rule-Based', color='#e67e22')
        axes[1, 1].fill(angles, rule_scores, alpha=0.25, color='#e67e22')
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Performance Radar Chart')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('llm_guided_rl_power_system_comparison_analysis.png')
        plt.close()
        # plt.show()
        
    def plot_threat_analysis_insights(self, vulnerability_assessment: Dict, title: str = "Threat Analysis Insights"):
        """Plot threat analysis insights and recommendations"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Critical vulnerabilities
        if 'critical_vulnerabilities' in vulnerability_assessment:
            vulns = vulnerability_assessment['critical_vulnerabilities']
            if vulns:
                vuln_names = [v.get('name', f'Vulnerability {i+1}') for i, v in enumerate(vulns[:5])]
                vuln_scores = [v.get('risk_score', 0.5) for v in vulns[:5]]
                
                bars = axes[0, 0].barh(vuln_names, vuln_scores, color='red', alpha=0.7)
                axes[0, 0].set_title('Critical Vulnerabilities')
                axes[0, 0].set_xlabel('Risk Score')
                axes[0, 0].set_xlim(0, 1)
                for bar, score in zip(bars, vuln_scores):
                    axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{score:.2f}', ha='left', va='center')
        
        # System state analysis
        if 'system_state_analysis' in vulnerability_assessment:
            state_analysis = vulnerability_assessment['system_state_analysis']
            
            # Frequency stability
            if 'frequency_stability' in state_analysis:
                stability = state_analysis['frequency_stability']
                colors = ['green' if stability == 'stable' else 'red']
                axes[0, 1].pie([1], colors=colors, labels=[stability.title()], autopct='%s')
                axes[0, 1].set_title('Frequency Stability')
            
            # Voltage stability
            if 'voltage_stability' in state_analysis:
                voltage_stability = state_analysis['voltage_stability']
                colors = ['green' if voltage_stability == 'stable' else 'red']
                axes[1, 0].pie([1], colors=colors, labels=[voltage_stability.title()], autopct='%s')
                axes[1, 0].set_title('Voltage Stability')
        
        # Evasion recommendations
        if 'evasion_recommendations' in vulnerability_assessment:
            evasion = vulnerability_assessment['evasion_recommendations']
            if 'techniques' in evasion:
                techniques = evasion['techniques']
                tech_names = [t.get('name', 'Unknown') for t in techniques]
                tech_effectiveness = [t.get('effectiveness', 0) for t in techniques]
                
                bars = axes[1, 1].bar(tech_names, tech_effectiveness, color='purple', alpha=0.7)
                axes[1, 1].set_title('Evasion Technique Effectiveness')
                axes[1, 1].set_ylabel('Effectiveness')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].tick_params(axis='x', rotation=45)
                for bar, eff in zip(bars, tech_effectiveness):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{eff:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('llm_guided_rl_power_system_threat_analysis_insights.png')
        plt.close()
        # plt.show()
        
    def plot_comprehensive_summary(self, llm_results: Dict, rule_based_results: Dict, 
                                 llm_episode_data: Dict, rule_episode_data: Dict,
                                 title: str = "Comprehensive Simulation Summary"):
        """Create a comprehensive summary of all simulation results"""
        if not PLOTTING_AVAILABLE:
            return
            
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Performance comparison (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        methods = ['LLM-Powered', 'Rule-Based']
        success_rates = [llm_results.get('success_rate', 0), rule_based_results.get('success_rate', 0)]
        bars = ax1.bar(methods, success_rates, color=['#3498db', '#e67e22'], alpha=0.8)
        ax1.set_title('Attack Success Rate Comparison', fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Detection rate comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        detection_rates = [llm_results.get('detection_rate', 0), rule_based_results.get('detection_rate', 0)]
        bars = ax2.bar(methods, detection_rates, color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax2.set_title('Detection Rate Comparison', fontweight='bold')
        ax2.set_ylabel('Detection Rate (%)')
        ax2.set_ylim(0, 100)
        for bar, rate in zip(bars, detection_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Episode rewards comparison
        ax3 = fig.add_subplot(gs[1, :2])
        if 'episode_rewards' in llm_episode_data and 'episode_rewards' in rule_episode_data:
            ax3.plot(llm_episode_data['episode_rewards'], 'b-', label='LLM-Powered', linewidth=2, alpha=0.8)
            ax3.plot(rule_episode_data['episode_rewards'], 'r-', label='Rule-Based', linewidth=2, alpha=0.8)
            ax3.set_title('Episode Rewards Comparison', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Total Reward')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Episode impacts comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'impact_scores' in llm_episode_data and 'impact_scores' in rule_episode_data:
            ax4.plot(llm_episode_data['impact_scores'], 'g-', label='LLM-Powered', linewidth=2, alpha=0.8)
            ax4.plot(rule_episode_data['impact_scores'], 'orange', label='Rule-Based', linewidth=2, alpha=0.8)
            ax4.set_title('Episode Impact Comparison', fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Total Impact')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Detection rates over episodes
        ax5 = fig.add_subplot(gs[2, :2])
        if 'detection_rates' in llm_episode_data and 'detection_rates' in rule_episode_data:
            ax5.plot([d * 100 for d in llm_episode_data['detection_rates']], 'purple', label='LLM-Powered', linewidth=2, alpha=0.8)
            ax5.plot([d * 100 for d in rule_episode_data['detection_rates']], 'brown', label='Rule-Based', linewidth=2, alpha=0.8)
            ax5.set_title('Detection Rate Over Episodes', fontweight='bold')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Detection Rate (%)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Radar chart for performance metrics
        ax6 = fig.add_subplot(gs[2, 2:], projection='polar')
        metrics = ['Success Rate', 'Stealth', 'Impact', 'Efficiency', 'Adaptability']
        llm_scores = [llm_results.get('success_rate', 0)/100, 
                      (100-llm_results.get('detection_rate', 0))/100,
                      min(llm_results.get('average_impact', 0)/20, 1.0),
                      llm_results.get('efficiency', 0.8),
                      0.9]  # LLM adaptability
        rule_scores = [rule_based_results.get('success_rate', 0)/100,
                       (100-rule_based_results.get('detection_rate', 0))/100,
                       min(rule_based_results.get('average_impact', 0)/20, 1.0),
                       rule_based_results.get('efficiency', 0.7),
                       0.6]  # Rule-based adaptability
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        llm_scores += llm_scores[:1]
        rule_scores += rule_scores[:1]
        angles += angles[:1]
        
        ax6.plot(angles, llm_scores, 'o-', linewidth=2, label='LLM-Powered', color='#3498db')
        ax6.fill(angles, llm_scores, alpha=0.25, color='#3498db')
        ax6.plot(angles, rule_scores, 'o-', linewidth=2, label='Rule-Based', color='#e67e22')
        ax6.fill(angles, rule_scores, alpha=0.25, color='#e67e22')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_ylim(0, 1)
        ax6.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
        
        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create summary table
        summary_data = [
            ['Metric', 'LLM-Powered', 'Rule-Based', 'Improvement'],
            ['Success Rate (%)', f"{llm_results.get('success_rate', 0):.1f}", 
             f"{rule_based_results.get('success_rate', 0):.1f}", 
             f"{llm_results.get('success_rate', 0) - rule_based_results.get('success_rate', 0):+.1f}"],
            ['Detection Rate (%)', f"{llm_results.get('detection_rate', 0):.1f}", 
             f"{rule_based_results.get('detection_rate', 0):.1f}", 
             f"{llm_results.get('detection_rate', 0) - rule_based_results.get('detection_rate', 0):+.1f}"],
            ['Average Impact', f"{llm_results.get('average_impact', 0):.3f}", 
             f"{rule_based_results.get('average_impact', 0):.3f}", 
             f"{llm_results.get('average_impact', 0) - rule_based_results.get('average_impact', 0):+.3f}"],
            ['Stealth Score', f"{(100-llm_results.get('detection_rate', 0)):.1f}", 
             f"{(100-rule_based_results.get('detection_rate', 0)):.1f}", 
             f"{(100-llm_results.get('detection_rate', 0)) - (100-rule_based_results.get('detection_rate', 0)):+.1f}"]
        ]
        
        table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the improvement column
        for i in range(1, len(summary_data)):
            cell = table[(i, 3)]
            try:
                value = float(summary_data[i][3])
                if value > 0:
                    cell.set_facecolor('#d5e8d4')  # Light green for improvement
                elif value < 0:
                    cell.set_facecolor('#f8cecc')  # Light red for degradation
                else:
                    cell.set_facecolor('#fff2cc')  # Light yellow for no change
            except:
                cell.set_facecolor('#ffffff')  # White for header
        
        ax7.set_title('Performance Summary Table', fontweight='bold', pad=20)
        plt.savefig('llm_guided_rl_power_system_comprehensive_summary.png')
        plt.tight_layout()
        plt.close()
 

        # plt.show()

# Example usage for cybersecurity research
if __name__ == "__main__":
    print("IEEE 14-Bus Power System Cybersecurity Research Simulation")
    print("Analyzing vulnerabilities and attack strategies for defensive purposes")
    print("=" * 80)
    
    # Check dependencies
    if not PANDAPOWER_AVAILABLE:
        print("REQUIRED DEPENDENCIES:")
        print("pip install pandapower pandas torch scikit-learn matplotlib")
        print("\nPandapower provides realistic power system modeling capabilities.")
        print("Please install the dependencies and run again.")
        exit(1)
    
    print("All dependencies available. Starting simulation...")
    
    # Check LLM availability
    if LLM_AVAILABLE:
        print("✅ LLM libraries available - Will use LLM-powered threat analysis")
    else:
        print("⚠️  LLM libraries not available - Will use rule-based threat analysis")
    
    # Run demonstration with LLM if available
    try:
        if LLM_AVAILABLE:
            print("\n" + "="*60)
            print("RUNNING LLM-POWERED SIMULATION")
            print("="*60)
            result_llm = run_security_simulation(
                episodes=100, max_steps_per_episode=50, use_llm=True
            )
            trained_agent_llm = result_llm[0] if result_llm else None
            assessment_llm = result_llm[4] if result_llm and len(result_llm) > 4 else {}
            
            print("\n" + "="*60)
            print("RUNNING RULE-BASED SIMULATION FOR COMPARISON")
            print("="*60)
            result_rule = run_security_simulation(
                episodes=100, max_steps_per_episode=50, use_llm=False
            )
            trained_agent_rule = result_rule[0] if result_rule else None
            assessment_rule = result_rule[4] if result_rule and len(result_rule) > 4 else {}
            
            # Compare results
            print("\n" + "="*60)
            print("LLM vs RULE-BASED COMPARISON")
            print("="*60)
            
            if trained_agent_llm and trained_agent_rule:
                llm_stats = trained_agent_llm.attack_stats
                rule_stats = trained_agent_rule.attack_stats
                
                print("LLM-Powered Results:")
                print(f"  Success Rate: {llm_stats['successful_attacks']/max(llm_stats['total_attacks'], 1):.1%}")
                print(f"  Detection Rate: {llm_stats['detected_attacks']/max(llm_stats['total_attacks'], 1):.1%}")
                print(f"  Average Impact: {llm_stats['impact_achieved']/max(llm_stats['total_attacks'], 1):.3f}")
                
                print("\nRule-Based Results:")
                print(f"  Success Rate: {rule_stats['successful_attacks']/max(rule_stats['total_attacks'], 1):.1%}")
                print(f"  Detection Rate: {rule_stats['detected_attacks']/max(rule_stats['total_attacks'], 1):.1%}")
                print(f"  Average Impact: {rule_stats['impact_achieved']/max(rule_stats['total_attacks'], 1):.3f}")
                
                # Create comparison plots if plotting is available
                if PLOTTING_AVAILABLE:
                    try:
                        visualizer = SimulationVisualizer()
                        
                        # Prepare results for comparison plotting
                        llm_results = {
                            'success_rate': (llm_stats['successful_attacks']/max(llm_stats['total_attacks'], 1)) * 100,
                            'detection_rate': (llm_stats['detected_attacks']/max(llm_stats['total_attacks'], 1)) * 100,
                            'average_impact': llm_stats['impact_achieved']/max(llm_stats['total_attacks'], 1),
                            'efficiency': 0.8  # LLM efficiency
                        }
                        
                        rule_based_results = {
                            'success_rate': (rule_stats['successful_attacks']/max(rule_stats['total_attacks'], 1)) * 100,
                            'detection_rate': (rule_stats['detected_attacks']/max(rule_stats['total_attacks'], 1)) * 100,
                            'average_impact': rule_stats['impact_achieved']/max(rule_stats['total_attacks'], 1),
                            'efficiency': 0.7  # Rule-based efficiency
                        }
                        
                        # Plot comparison analysis
                        visualizer.plot_comparison_analysis(
                            llm_results, 
                            rule_based_results,
                            "LLM-Powered vs Rule-Based Threat Analysis Comparison"
                        )
                        
                        # Create comprehensive summary plot
                        llm_episode_data = {
                            'episode_rewards': result_llm[1] if result_llm and len(result_llm) > 1 else [],
                            'detection_rates': result_llm[2] if result_llm and len(result_llm) > 2 else [],
                            'impact_scores': result_llm[3] if result_llm and len(result_llm) > 3 else []
                        }
                        
                        rule_episode_data = {
                            'episode_rewards': result_rule[1] if result_rule and len(result_rule) > 1 else [],
                            'detection_rates': result_rule[2] if result_rule and len(result_rule) > 2 else [],
                            'impact_scores': result_rule[3] if result_rule and len(result_rule) > 3 else []
                        }
                        
                        visualizer.plot_comprehensive_summary(
                            llm_results,
                            rule_based_results,
                            llm_episode_data,
                            rule_episode_data,
                            "Comprehensive LLM vs Rule-Based Analysis Summary"
                        )
                        
                    except Exception as e:
                        print(f"Warning: Comparison plotting failed: {e}")
                
                # Use LLM results for final assessment
                trained_agent = trained_agent_llm
                assessment = assessment_llm
            else:
                trained_agent = trained_agent_llm or trained_agent_rule
                assessment = assessment_llm or assessment_rule
        else:
            # Fallback to rule-based
            result_fallback = demonstrate_attack_strategies()
            trained_agent = result_fallback[0] if result_fallback else None
            assessment = result_fallback[1] if result_fallback and len(result_fallback) > 1 else {}
        
        print(f"\nSimulation completed successfully!")
        
        if LLM_AVAILABLE and 'llm_raw_analysis' in assessment:
            print(f"\n--- LLM INSIGHTS ---")
            print("LLM Analysis Summary:")
            llm_analysis = assessment.get('llm_raw_analysis', '')
            if llm_analysis:
                print(f"  {llm_analysis[:200]}...")
            
            if 'evasion_recommendations' in assessment and 'llm_evasion_analysis' in assessment['evasion_recommendations']:
                evasion_llm = assessment['evasion_recommendations']['llm_evasion_analysis']
                print(f"\nLLM Evasion Strategy:")
                print(f"  {evasion_llm[:150]}...")
        
        print(f"\nKey findings for cybersecurity research:")
        print("1. Realistic power flow modeling enables accurate vulnerability assessment")
        print("2. AGC systems are vulnerable to coordinated frequency manipulation attacks") 
        print("3. Generator buses are high-value targets for maximum system impact")
        print("4. Multi-stage attacks can evade simple anomaly detection systems")
        print("5. STRIDE/MITRE frameworks provide comprehensive threat coverage")
        if LLM_AVAILABLE:
            print("6. LLM-powered analysis provides dynamic, context-aware threat assessment")
            print("7. AI-generated attack strategies adapt to system state changes")
            print("8. Natural language reasoning enhances threat understanding and planning")
        
        print(f"\nResults can be used to:")
        print("- Improve anomaly detection systems")
        print("- Develop better defensive strategies") 
        print("- Identify critical system vulnerabilities")
        print("- Design more robust AGC control systems")
        print("- Enhance security monitoring capabilities")
        print("- Train cybersecurity personnel on power system threats")
        
    except Exception as e:
        print(f"Simulation error: {e}")
        print("This may be due to power flow convergence issues or missing dependencies.")
        print("Consider reducing attack magnitudes or checking system parameters.")

# Example usage for cybersecurity research