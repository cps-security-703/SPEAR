#!/usr/bin/env python3
"""
Complete Power Grid Cybersecurity Demonstration
Integration of OpenDSS, SCADA Control Center, Dashboard, and STRIDE/MITRE Testing
"""

import asyncio
import json
import time
import threading
import random
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Import required libraries (install with: pip install opendssdirect dash plotly pymodbus flask-socketio)
try:
    import opendssdirect as dss
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objs as go
    import plotly.express as px
    from pymodbus.server import StartTcpServer
    from pymodbus.device import ModbusDeviceIdentification
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
    from pymodbus.client import ModbusTcpClient
    import flask_socketio as socketio
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install opendssdirect dash plotly pymodbus flask-socketio")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Data structure for sensor readings"""
    sensor_id: str
    timestamp: datetime
    voltage: float
    current: float
    power: float
    frequency: float
    status: str

@dataclass
class ControlCommand:
    """Data structure for control commands"""
    command_id: str
    timestamp: datetime
    target_device: str
    command_type: str
    value: float
    operator_id: str

class PowerGridSimulator:
    """OpenDSS-based power grid simulator"""
    
    def __init__(self):
        self.circuit_name = "IEEE13Node_Security_Demo"
        self.buses = []
        self.loads = []
        self.generators = []
        self.is_running = False
        self.simulation_data = {}
        
    def initialize_grid(self):
        """Initialize IEEE 13-node test system with modifications for cybersecurity demo"""
        
        # Clear any existing circuit
        dss.Command("Clear")
        
        # Create new circuit with proper voltage source
        dss.Command(f"New Circuit.{self.circuit_name} basekv=4.16 pu=1.0 phases=3 bus1=sourcebus")
        
        # Define the circuit (simplified IEEE 13-node system)
        circuit_commands = [
            # Set voltage base
            "Set DefaultBaseFreq=60",
            "Set voltagebases=[4.16, 0.48]",
            "Calcvoltagebases",
            
            # Define Line codes
            "New LineCode.601 nphases=3 basefreq=60 r1=0.3465 x1=1.0179 r0=1.3238 x0=4.1711",
            "New LineCode.602 nphases=3 basefreq=60 r1=0.7526 x1=1.1814 r0=1.9890 x0=4.7950", 
            "New LineCode.603 nphases=2 basefreq=60 r1=1.3238 x1=1.3569 r0=1.9890 x0=4.4132",
            "New LineCode.604 nphases=2 basefreq=60 r1=1.3238 x1=1.3569 r0=1.9890 x0=4.4132",
            "New LineCode.605 nphases=1 basefreq=60 r1=1.3292 x1=1.3475 r0=1.9890 x0=4.4132",
            
            # Define Lines
            "New Line.650632 phases=3 bus1=sourcebus.1.2.3 bus2=632.1.2.3 linecode=601 length=2000",
            "New Line.632670 phases=3 bus1=632.1.2.3 bus2=670.1.2.3 linecode=601 length=667",
            "New Line.670671 phases=3 bus1=670.1.2.3 bus2=671.1.2.3 linecode=601 length=1333",
            "New Line.671680 phases=3 bus1=671.1.2.3 bus2=680.1.2.3 linecode=601 length=1000",
            "New Line.632633 phases=3 bus1=632.1.2.3 bus2=633.1.2.3 linecode=602 length=500",
            
            # Define Loads (with varying characteristics for demo)
            "New Load.671 phases=3 bus1=671.1.2.3 kw=1155 kvar=660 model=1 vminpu=0.9 vmaxpu=1.1",
            "New Load.634a phases=1 bus1=634.1 kw=160 kvar=110 model=1",
            "New Load.634b phases=1 bus1=634.2 kw=120 kvar=90 model=1", 
            "New Load.634c phases=1 bus1=634.3 kw=120 kvar=90 model=1",
            "New Load.645 phases=1 bus1=645.2 kw=170 kvar=125 model=1",
            "New Load.646 phases=1 bus1=646.2.3 kw=230 kvar=132 model=1",
            "New Load.692 phases=3 bus1=692.3.1.2 kw=170 kvar=151 model=1",
            "New Load.675a phases=1 bus1=675.1 kw=485 kvar=190 model=1",
            "New Load.675b phases=1 bus1=675.2 kw=68 kvar=60 model=1",
            "New Load.675c phases=1 bus1=675.3 kw=290 kvar=212 model=1",
            "New Load.611 phases=1 bus1=611.3 kw=170 kvar=80 model=1",
            "New Load.652 phases=1 bus1=652.1 kw=128 kvar=86 model=1",
            
            # Define Generators (distributed resources)
            "New Generator.DG1 phases=3 bus1=675 kw=500 kv=4.16 pf=0.9",
            "New Generator.DG2 phases=1 bus1=611 kw=100 kv=4.16 pf=0.85",
            
            # Define monitoring points (our "sensors")
            "New Monitor.Mon_671 element=Load.671 terminal=1 mode=1",
            "New Monitor.Mon_675 element=Load.675a terminal=1 mode=1", 
            "New Monitor.Mon_DG1 element=Generator.DG1 terminal=1 mode=1",
            
            # Set solution parameters
            "Set tolerance=0.000001",
            "Set maxiterations=20"
        ]
        
        # Execute all commands
        for cmd in circuit_commands:
            dss.Command(cmd)
            
        # Solve initial power flow
        dss.Command("Solve")
        
        # Get bus names and other info for our simulation
        self.buses = dss.Circuit.AllBusNames()
        self.loads = dss.Loads.AllNames()
        self.generators = dss.Generators.AllNames()
        
        logger.info(f"Grid initialized with {len(self.buses)} buses, {len(self.loads)} loads, {len(self.generators)} generators")
        
    def run_simulation(self):
        """Run continuous power flow simulation"""
        self.is_running = True
        logger.info("Starting power grid simulation...")
        
        while self.is_running:
            try:
                # Vary loads slightly to simulate real conditions
                self.vary_system_conditions()
                
                # Solve power flow
                dss.Command("Solve")
                
                # Collect system data
                self.collect_system_data()
                
                # Wait for next iteration
                time.sleep(2)  # 2-second intervals
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                break
    
    def vary_system_conditions(self):
        """Add realistic variations to system conditions"""
        # Vary loads by ±5%
        for load_name in self.loads:
            dss.Loads.Name(load_name)
            current_kw = dss.Loads.kW()
            variation = random.uniform(-0.05, 0.05)
            new_kw = current_kw * (1 + variation)
            dss.Command(f"Load.{load_name}.kw={new_kw:.2f}")
    
    def collect_system_data(self):
        """Collect system measurements for SCADA"""
        timestamp = datetime.now()
        
        # Get bus voltages
        bus_voltages = {}
        for bus_name in self.buses[:10]:  # Limit for demo
            dss.Circuit.SetActiveBus(bus_name)
            voltages = dss.Bus.puVmagAngle()
            if voltages:
                bus_voltages[bus_name] = {
                    'voltage_mag': voltages[0] if len(voltages) > 0 else 1.0,
                    'voltage_angle': voltages[1] if len(voltages) > 1 else 0.0
                }
        
        # Get load powers
        load_powers = {}
        for load_name in self.loads:
            dss.Loads.Name(load_name)
            load_powers[load_name] = {
                'kw': dss.Loads.kW(),
                'kvar': dss.Loads.kvar()
            }
        
        # Get generator outputs
        gen_outputs = {}
        for gen_name in self.generators:
            dss.Generators.Name(gen_name)
            gen_outputs[gen_name] = {
                'kw': dss.Generators.kW(),
                'kvar': dss.Generators.kvar()
            }
        
        # Store simulation data
        self.simulation_data = {
            'timestamp': timestamp.isoformat(),
            'buses': bus_voltages,
            'loads': load_powers,
            'generators': gen_outputs,
            'total_losses': dss.Circuit.Losses()[0],  # Real power losses in W
            'convergence': dss.Solution.Converged()
        }
    
    def get_sensor_readings(self) -> List[SensorReading]:
        """Convert simulation data to sensor readings"""
        readings = []
        
        if not self.simulation_data:
            return readings
            
        timestamp = datetime.fromisoformat(self.simulation_data['timestamp'])
        
        # Create sensor readings from bus data
        for bus_name, bus_data in self.simulation_data['buses'].items():
            reading = SensorReading(
                sensor_id=f"BUS_{bus_name}",
                timestamp=timestamp,
                voltage=bus_data['voltage_mag'] * 4160,  # Convert to actual voltage
                current=random.uniform(50, 200),  # Simulated current
                power=random.uniform(100, 500),   # Simulated power
                frequency=60.0 + random.uniform(-0.1, 0.1),  # Slight frequency variation
                status="NORMAL"
            )
            readings.append(reading)
        
        return readings
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        logger.info("Power grid simulation stopped")

class SCADAControlCenter:
    """SCADA Control Center with Modbus TCP interface"""
    
    def __init__(self, grid_simulator: PowerGridSimulator):
        self.grid_simulator = grid_simulator
        self.modbus_server = None
        self.sensor_readings = []
        self.control_commands = []
        self.alarms = []
        self.datastore = None
        
    def initialize_modbus_server(self, host='127.0.0.1', port=502):
        """Initialize Modbus TCP server"""
        try:
            # Create data blocks
            holding_registers = ModbusSequentialDataBlock(0, [0]*100)
            input_registers = ModbusSequentialDataBlock(0, [0]*100) 
            coils = ModbusSequentialDataBlock(0, [False]*100)
            discrete_inputs = ModbusSequentialDataBlock(0, [False]*100)
            
            # Create slave context
            slave_context = ModbusSlaveContext(
                di=discrete_inputs,
                co=coils,
                hr=holding_registers,
                ir=input_registers
            )
            
            # Create server context
            self.datastore = ModbusServerContext(slaves=slave_context, single=True)
            
            # Device information
            identity = ModbusDeviceIdentification()
            identity.VendorName = 'PowerGrid Security Demo'
            identity.ProductCode = 'SCADA-001'
            identity.VendorUrl = 'https://powergrid-security.demo'
            identity.ProductName = 'Demo SCADA System'
            identity.ModelName = 'PowerGrid SCADA v1.0'
            identity.MajorMinorRevision = '1.0'
            
            logger.info(f"Starting Modbus TCP server on {host}:{port}")
            
            # Start server in separate thread
            server_thread = threading.Thread(
                target=lambda: StartTcpServer(context=self.datastore, 
                                            identity=identity,
                                            address=(host, port))
            )
            server_thread.daemon = True
            server_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Modbus server: {e}")
            return False
    
    def update_modbus_registers(self):
        """Update Modbus registers with sensor data"""
        if not self.datastore:
            return
            
        readings = self.grid_simulator.get_sensor_readings()
        
        if not readings:
            return
            
        try:
            # Update input registers with sensor data
            context = self.datastore[0]  # Unit ID 0
            
            for i, reading in enumerate(readings[:10]):  # Limit to first 10 sensors
                base_addr = i * 4
                
                # Store voltage (scaled to integer)
                voltage_int = int(reading.voltage * 10)  # Scale by 10 for precision
                context.setValues(3, base_addr, [voltage_int])  # Input registers
                
                # Store current (scaled to integer)
                current_int = int(reading.current * 10)
                context.setValues(3, base_addr + 1, [current_int])
                
                # Store power (scaled to integer)
                power_int = int(reading.power)
                context.setValues(3, base_addr + 2, [power_int])
                
                # Store frequency (scaled to integer)
                freq_int = int(reading.frequency * 100)  # Scale by 100 for precision
                context.setValues(3, base_addr + 3, [freq_int])
            
            # Update system status in coils
            system_healthy = all(r.status == "NORMAL" for r in readings)
            context.setValues(1, 0, [system_healthy])  # Coil 0 = System Health
            
        except Exception as e:
            logger.error(f"Failed to update Modbus registers: {e}")
    
    def process_control_command(self, command: ControlCommand):
        """Process control commands from operators"""
        logger.info(f"Processing control command: {command.command_type} for {command.target_device}")
        
        # Log the command for audit trail
        self.control_commands.append(command)
        
        # Execute command based on type
        if command.command_type == "LOAD_SHED":
            self.execute_load_shed(command.target_device, command.value)
        elif command.command_type == "GEN_DISPATCH":
            self.execute_generator_dispatch(command.target_device, command.value)
        elif command.command_type == "BREAKER_OPEN":
            self.execute_breaker_operation(command.target_device, "OPEN")
        elif command.command_type == "BREAKER_CLOSE":
            self.execute_breaker_operation(command.target_device, "CLOSE")
    
    def execute_load_shed(self, load_name, amount):
        """Execute load shedding command"""
        try:
            dss.Loads.Name(load_name)
            current_kw = dss.Loads.kW()
            new_kw = max(0, current_kw - amount)
            dss.Command(f"Load.{load_name}.kw={new_kw}")
            logger.info(f"Load shed: {load_name} reduced by {amount}kW")
        except Exception as e:
            logger.error(f"Failed to execute load shed: {e}")
    
    def execute_generator_dispatch(self, gen_name, setpoint):
        """Execute generator dispatch command"""
        try:
            dss.Generators.Name(gen_name)
            dss.Command(f"Generator.{gen_name}.kw={setpoint}")
            logger.info(f"Generator dispatch: {gen_name} set to {setpoint}kW")
        except Exception as e:
            logger.error(f"Failed to execute generator dispatch: {e}")
    
    def execute_breaker_operation(self, breaker_name, operation):
        """Execute breaker operation"""
        # In a real system, this would control actual breakers
        # For demo, we'll just log the operation
        logger.info(f"Breaker operation: {breaker_name} {operation}")
        
        # Update system model if needed
        if operation == "OPEN":
            # Simulate opening a line (simplified)
            pass
    
    def run_control_loop(self):
        """Main SCADA control loop"""
        logger.info("Starting SCADA control loop...")
        
        while self.grid_simulator.is_running:
            try:
                # Update sensor readings
                self.sensor_readings = self.grid_simulator.get_sensor_readings()
                
                # Update Modbus registers
                self.update_modbus_registers()
                
                # Check for alarms
                self.check_alarms()
                
                # Process any pending commands
                # (Commands would come from HMI/dashboard)
                
                time.sleep(1)  # 1-second control loop
                
            except Exception as e:
                logger.error(f"SCADA control loop error: {e}")
                break
    
    def check_alarms(self):
        """Check for alarm conditions"""
        for reading in self.sensor_readings:
            # Voltage alarm
            if reading.voltage < 3800 or reading.voltage > 4500:  # ±10% of 4160V
                alarm = {
                    'timestamp': reading.timestamp.isoformat(),
                    'type': 'VOLTAGE_ALARM',
                    'sensor_id': reading.sensor_id,
                    'value': reading.voltage,
                    'severity': 'HIGH'
                }
                self.alarms.append(alarm)
                logger.warning(f"VOLTAGE ALARM: {reading.sensor_id} = {reading.voltage:.2f}V")

class CyberSecurityTester:
    """Implements STRIDE and MITRE ATT&CK testing"""
    
    def __init__(self, target_host='127.0.0.1', target_port=502):
        self.target_host = target_host
        self.target_port = target_port
        self.client = None
        self.attack_results = {
            'stride': {},
            'mitre': {}
        }
        
    def connect_to_scada(self):
        """Connect to SCADA system via Modbus"""
        try:
            self.client = ModbusTcpClient(self.target_host, port=self.target_port)
            connected = self.client.connect()
            if connected:
                logger.info(f"Connected to SCADA system at {self.target_host}:{self.target_port}")
                return True
            else:
                logger.error("Failed to connect to SCADA system")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def stride_spoofing_test(self):
        """STRIDE - Spoofing Identity Test"""
        logger.info("=== STRIDE SPOOFING TEST ===")
        
        if not self.client:
            return False
            
        try:
            # Attempt to read system data without authentication
            result = self.client.read_input_registers(0, 10, unit=1)
            
            if not result.isError():
                logger.warning("VULNERABILITY: System allows unauthenticated access")
                logger.info(f"Read sensor data: {result.registers[:5]}...")
                
                # Try to impersonate legitimate operator
                fake_command = self.client.write_register(50, 999, unit=1)  # Fake setpoint
                if not fake_command.isError():
                    logger.warning("CRITICAL: Successfully spoofed operator command!")
                    self.attack_results['stride']['spoofing'] = {
                        'success': True,
                        'impact': 'HIGH',
                        'details': 'Unauthenticated access and command injection possible'
                    }
                    return True
            
            self.attack_results['stride']['spoofing'] = {
                'success': False,
                'impact': 'LOW',
                'details': 'System properly rejects unauthenticated access'
            }
            return False
            
        except Exception as e:
            logger.error(f"Spoofing test error: {e}")
            return False
    
    def stride_tampering_test(self):
        """STRIDE - Data Tampering Test"""
        logger.info("=== STRIDE TAMPERING TEST ===")
        
        if not self.client:
            return False
            
        try:
            # Read original sensor values
            original_values = self.client.read_input_registers(0, 20, unit=1)
            if original_values.isError():
                logger.info("Cannot read values for tampering test")
                return False
                
            logger.info(f"Original sensor readings: {original_values.registers[:5]}...")
            
            # Attempt to modify critical system parameters
            tampering_attempts = [
                {'register': 0, 'value': 9999, 'description': 'Voltage sensor'},
                {'register': 1, 'value': 0, 'description': 'Current sensor'}, 
                {'register': 4, 'value': 5500, 'description': 'Frequency sensor'}
            ]
            
            successful_tampering = 0
            for attempt in tampering_attempts:
                result = self.client.write_register(attempt['register'], attempt['value'], unit=1)
                if not result.isError():
                    logger.warning(f"VULNERABILITY: Successfully tampered with {attempt['description']}")
                    successful_tampering += 1
                else:
                    logger.info(f"Tampering blocked for {attempt['description']}")
            
            if successful_tampering > 0:
                self.attack_results['stride']['tampering'] = {
                    'success': True,
                    'impact': 'CRITICAL',
                    'details': f'Successfully tampered with {successful_tampering} sensors'
                }
                return True
            else:
                self.attack_results['stride']['tampering'] = {
                    'success': False,
                    'impact': 'LOW',
                    'details': 'System protected against data tampering'
                }
                return False
                
        except Exception as e:
            logger.error(f"Tampering test error: {e}")
            return False
    
    def stride_information_disclosure_test(self):
        """STRIDE - Information Disclosure Test"""
        logger.info("=== STRIDE INFORMATION DISCLOSURE TEST ===")
        
        if not self.client:
            return False
            
        disclosed_info = []
        
        try:
            # Attempt to read device identification
            device_info = self.client.read_device_information(0x00, 0x00, unit=1)
            if not device_info.isError():
                logger.warning("VULNERABILITY: Device information disclosed")
                for obj_id, value in device_info.information.items():
                    logger.info(f"Device Info {obj_id}: {value}")
                    disclosed_info.append(f"{obj_id}: {value}")
            
            # Attempt to read all accessible registers
            register_ranges = [(0, 20), (50, 10), (100, 20)]
            accessible_registers = 0
            
            for start, count in register_ranges:
                try:
                    result = self.client.read_input_registers(start, count, unit=1)
                    if not result.isError():
                        accessible_registers += count
                        logger.warning(f"VULNERABILITY: Registers {start}-{start+count-1} accessible")
                except:
                    continue
            
            # Try to read holding registers (configuration data)
            config_result = self.client.read_holding_registers(0, 10, unit=1)
            if not config_result.isError():
                logger.warning("VULNERABILITY: Configuration data accessible")
                disclosed_info.append(f"Configuration registers: {config_result.registers}")
            
            if disclosed_info or accessible_registers > 0:
                self.attack_results['stride']['information_disclosure'] = {
                    'success': True,
                    'impact': 'HIGH',
                    'details': f'Disclosed: {len(disclosed_info)} device info items, {accessible_registers} registers'
                }
                return True
            else:
                self.attack_results['stride']['information_disclosure'] = {
                    'success': False,
                    'impact': 'LOW', 
                    'details': 'System properly protects information'
                }
                return False
                
        except Exception as e:
            logger.error(f"Information disclosure test error: {e}")
            return False
    
    def stride_denial_of_service_test(self):
        """STRIDE - Denial of Service Test"""
        logger.info("=== STRIDE DENIAL OF SERVICE TEST ===")
        
        # Connection flooding test
        connections = []
        try:
            # Create multiple connections rapidly
            for i in range(20):
                client = ModbusTcpClient(self.target_host, port=self.target_port)
                if client.connect():
                    connections.append(client)
                else:
                    break
            
            logger.info(f"Created {len(connections)} simultaneous connections")
            
            # Test if system still responds
            test_client = ModbusTcpClient(self.target_host, port=self.target_port)
            if test_client.connect():
                response = test_client.read_input_registers(0, 1, unit=1)
                if response.isError():
                    logger.warning("VULNERABILITY: DoS attack successful - system unresponsive")
                    dos_success = True
                else:
                    logger.info("System still responsive during connection flood")
                    dos_success = False
                test_client.close()
            else:
                logger.warning("VULNERABILITY: Cannot establish new connections")
                dos_success = True
            
            # Clean up connections
            for client in connections:
                try:
                    client.close()
                except:
                    pass
                    
            self.attack_results['stride']['denial_of_service'] = {
                'success': dos_success,
                'impact': 'HIGH' if dos_success else 'LOW',
                'details': f'Connection flood test with {len(connections)} connections'
            }
            
            return dos_success
            
        except Exception as e:
            logger.error(f"DoS test error: {e}")
            return False
    
    def mitre_t1046_network_service_scanning(self):
        """MITRE T1046 - Network Service Scanning"""
        logger.info("=== MITRE T1046 - Network Service Scanning ===")
        
        import socket
        
        # Scan common industrial ports
        target_ports = [102, 502, 2404, 20000, 44818, 1911, 9600]
        open_ports = []
        services_identified = []
        
        for port in target_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.target_host, port))
            
            if result == 0:
                open_ports.append(port)
                service = self.identify_service(port)
                services_identified.append(service)
                logger.warning(f"DISCOVERY: Port {port} open - {service}")
            
            sock.close()
        
        self.attack_results['mitre']['T1046'] = {
            'technique': 'Network Service Scanning',
            'success': len(open_ports) > 0,
            'details': f'Found {len(open_ports)} open ports: {open_ports}',
            'services': services_identified
        }
        
        return len(open_ports) > 0
    
    def mitre_t0801_monitor_process_state(self):
        """MITRE T0801 - Monitor Process State"""
        logger.info("=== MITRE T0801 - Monitor Process State ===")
        
        if not self.client:
            return False
            
        monitoring_data = []
        
        try:
            # Monitor system state over time
            for i in range(10):  # 10 samples
                timestamp = datetime.now()
                
                # Read various system parameters
                sensors = self.client.read_input_registers(0, 20, unit=1)
                status = self.client.read_coils(0, 10, unit=1)
                
                if not sensors.isError() and not status.isError():
                    sample = {
                        'timestamp': timestamp.isoformat(),
                        'sensors': sensors.registers,
                        'status': status.bits
                    }
                    monitoring_data.append(sample)
                    logger.info(f"Sample {i+1}: Sensors={sensors.registers[:3]}, Status={status.bits[:3]}")
                
                time.sleep(1)  # 1-second intervals
            
            # Analyze patterns
            if monitoring_data:
                logger.warning(f"SURVEILLANCE: Successfully monitored system for {len(monitoring_data)} samples")
                
                # Detect operational patterns
                voltage_values = [s['sensors'][0] for s in monitoring_data if s['sensors']]
                avg_voltage = sum(voltage_values) / len(voltage_values) if voltage_values else 0
                
                logger.info(f"Pattern Analysis: Average voltage reading = {avg_voltage}")
                
                self.attack_results['mitre']['T0801'] = {
                    'technique': 'Monitor Process State',
                    'success': True,
                    'details': f'Monitored system for {len(monitoring_data)} samples',
                    'patterns_discovered': f'Average voltage: {avg_voltage}'
                }
                return True
            
        except Exception as e:
            logger.error(f"Process monitoring error: {e}")
        
        self.attack_results['mitre']['T0801'] = {
            'technique': 'Monitor Process State', 
            'success': False,
            'details': 'Unable to monitor system state'
        }
        return False
    
    def mitre_t0836_modify_parameter(self):
        """MITRE T0836 - Modify Parameter"""
        logger.info("=== MITRE T0836 - Modify Parameter ===")
        
        if not self.client:
            return False
            
        try:
            # Read current parameters
            current_params = self.client.read_holding_registers(0, 10, unit=1)
            if current_params.isError():
                logger.info("Cannot read current parameters")
                return False
                
            logger.info(f"Current parameters: {current_params.registers}")
            
            # Attempt parameter modification
            modifications = [
                {'register': 0, 'original': current_params.registers[0] if len(current_params.registers) > 0 else 0, 'new': 1337},
                {'register': 1, 'original': current_params.registers[1] if len(current_params.registers) > 1 else 0, 'new': 2022},
                {'register': 5, 'original': current_params.registers[5] if len(current_params.registers) > 5 else 0, 'new': 9999}
            ]
            
            successful_mods = 0
            for mod in modifications:
                result = self.client.write_register(mod['register'], mod['new'], unit=1)
                if not result.isError():
                    logger.warning(f"PARAMETER MODIFIED: Register {mod['register']}: {mod['original']} → {mod['new']}")
                    successful_mods += 1
                else:
                    logger.info(f"Parameter modification blocked for register {mod['register']}")
            
            if successful_mods > 0:
                self.attack_results['mitre']['T0836'] = {
                    'technique': 'Modify Parameter',
                    'success': True,
                    'details': f'Successfully modified {successful_mods} parameters',
                    'impact': 'CRITICAL'
                }
                return True
            else:
                self.attack_results['mitre']['T0836'] = {
                    'technique': 'Modify Parameter',
                    'success': False,
                    'details': 'All parameter modifications blocked'
                }
                return False
                
        except Exception as e:
            logger.error(f"Parameter modification error: {e}")
            return False
    
    def identify_service(self, port):
        """Identify service based on port number"""
        service_map = {
            102: 'IEC 61850 MMS',
            502: 'Modbus TCP',
            2404: 'IEC 61850 GOOSE',
            20000: 'DNP3',
            44818: 'EtherNet/IP',
            1911: 'Tridium Fox',
            9600: 'OMRON FINS'
        }
        return service_map.get(port, f'Unknown service on port {port}')
    
    def run_comprehensive_test(self):
        """Run complete STRIDE and MITRE testing"""
        logger.info("Starting comprehensive cybersecurity assessment...")
        
        if not self.connect_to_scada():
            return False
        
        # STRIDE Tests
        logger.info("\n" + "="*50)
        logger.info("EXECUTING STRIDE THREAT MODEL TESTS")
        logger.info("="*50)
        
        stride_tests = [
            ('Spoofing', self.stride_spoofing_test),
            ('Tampering', self.stride_tampering_test), 
            ('Information Disclosure', self.stride_information_disclosure_test),
            ('Denial of Service', self.stride_denial_of_service_test)
        ]
        
        for test_name, test_func in stride_tests:
            logger.info(f"\n--- {test_name} Test ---")
            test_func()
        
        # MITRE ATT&CK Tests
        logger.info("\n" + "="*50)
        logger.info("EXECUTING MITRE ATT&CK TECHNIQUE TESTS") 
        logger.info("="*50)
        
        mitre_tests = [
            ('T1046 - Network Service Scanning', self.mitre_t1046_network_service_scanning),
            ('T0801 - Monitor Process State', self.mitre_t0801_monitor_process_state),
            ('T0836 - Modify Parameter', self.mitre_t0836_modify_parameter)
        ]
        
        for test_name, test_func in mitre_tests:
            logger.info(f"\n--- {test_name} ---")
            test_func()
        
        # Generate final report
        self.generate_security_report()
        
        self.client.close()
        return True
    
    def generate_security_report(self):
        """Generate comprehensive security assessment report"""
        logger.info("\n" + "="*60)
        logger.info("CYBERSECURITY ASSESSMENT REPORT")
        logger.info("="*60)
        
        # STRIDE Summary
        logger.info("\nSTRIDE THREAT MODEL RESULTS:")
        logger.info("-" * 30)
        
        stride_score = 0
        total_stride_tests = len(self.attack_results['stride'])
        
        for threat_type, result in self.attack_results['stride'].items():
            status = "VULNERABLE" if result['success'] else "PROTECTED"
            impact = result.get('impact', 'UNKNOWN')
            logger.info(f"{threat_type.upper()}: {status} (Impact: {impact})")
            logger.info(f"  Details: {result['details']}")
            
            if result['success']:
                stride_score += 1
        
        # MITRE ATT&CK Summary
        logger.info(f"\nMITRE ATT&CK TECHNIQUE RESULTS:")
        logger.info("-" * 35)
        
        mitre_score = 0
        total_mitre_tests = len(self.attack_results['mitre'])
        
        for technique, result in self.attack_results['mitre'].items():
            status = "SUCCESSFUL" if result['success'] else "BLOCKED"
            logger.info(f"{technique} ({result['technique']}): {status}")
            logger.info(f"  Details: {result['details']}")
            
            if result['success']:
                mitre_score += 1
        
        # Overall Assessment
        logger.info(f"\nOVERALL ASSESSMENT:")
        logger.info("-" * 20)
        logger.info(f"STRIDE Vulnerabilities: {stride_score}/{total_stride_tests}")
        logger.info(f"MITRE Techniques Successful: {mitre_score}/{total_mitre_tests}")
        
        overall_risk = "HIGH" if (stride_score + mitre_score) > 3 else "MEDIUM" if (stride_score + mitre_score) > 1 else "LOW"
        logger.info(f"Overall Risk Level: {overall_risk}")
        
        # Recommendations
        logger.info(f"\nRECOMMENDATIONS:")
        logger.info("-" * 15)
        recommendations = [
            "1. Implement strong authentication and authorization",
            "2. Deploy network segmentation and firewalls", 
            "3. Enable comprehensive logging and monitoring",
            "4. Implement data integrity checks and validation",
            "5. Regular security assessments and penetration testing",
            "6. Incident response procedures for ICS environments",
            "7. Security awareness training for operators"
        ]
        
        for rec in recommendations:
            logger.info(rec)

class PowerGridDashboard:
    """Real-time dashboard for monitoring grid and security"""
    
    def __init__(self, grid_simulator: PowerGridSimulator, scada_center: SCADAControlCenter):
        self.grid_simulator = grid_simulator
        self.scada_center = scada_center
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Power Grid Cybersecurity Monitoring Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Real-time metrics row
            html.Div([
                html.Div([
                    html.H3("System Status", style={'textAlign': 'center'}),
                    html.Div(id='system-status', style={'fontSize': 24, 'textAlign': 'center'})
                ], className='four columns'),
                
                html.Div([
                    html.H3("Active Alarms", style={'textAlign': 'center'}),
                    html.Div(id='alarm-count', style={'fontSize': 24, 'textAlign': 'center'})
                ], className='four columns'),
                
                html.Div([
                    html.H3("Grid Frequency", style={'textAlign': 'center'}),
                    html.Div(id='frequency-display', style={'fontSize': 24, 'textAlign': 'center'})
                ], className='four columns')
            ], className='row', style={'marginBottom': 30}),
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(id='voltage-chart')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='power-chart') 
                ], className='six columns')
            ], className='row'),
            
            # Security monitoring row
            html.Div([
                html.H2("Security Monitoring", style={'textAlign': 'center', 'marginTop': 30}),
                html.Div([
                    html.Div([
                        html.H4("Network Traffic"),
                        dcc.Graph(id='network-traffic-chart')
                    ], className='six columns'),
                    
                    html.Div([
                        html.H4("Security Events"),
                        html.Div(id='security-events')
                    ], className='six columns')
                ], className='row')
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            )
        ])
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates"""
        
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('alarm-count', 'children'), 
             Output('frequency-display', 'children'),
             Output('voltage-chart', 'figure'),
             Output('power-chart', 'figure'),
             Output('network-traffic-chart', 'figure'),
             Output('security-events', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get current sensor readings
            readings = self.scada_center.sensor_readings
            
            if not readings:
                return "No Data", "0", "0.0 Hz", {}, {}, {}, "No Events"
            
            # System status
            system_status = "NORMAL" if all(r.status == "NORMAL" for r in readings) else "ALARM"
            status_color = "green" if system_status == "NORMAL" else "red"
            
            # Alarm count
            alarm_count = len(self.scada_center.alarms)
            
            # Frequency
            avg_frequency = sum(r.frequency for r in readings) / len(readings)
            frequency_display = f"{avg_frequency:.2f} Hz"
            
            # Voltage chart
            voltage_data = []
            timestamps = []
            for reading in readings[-10:]:  # Last 10 readings
                voltage_data.append(reading.voltage)
                timestamps.append(reading.timestamp.strftime('%H:%M:%S'))
            
            voltage_fig = {
                'data': [
                    go.Scatter(
                        x=timestamps,
                        y=voltage_data,
                        mode='lines+markers',
                        name='Voltage (V)',
                        line={'color': 'blue'}
                    )
                ],
                'layout': {
                    'title': 'Bus Voltages',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Voltage (V)'},
                    'height': 300
                }
            }
            
            # Power chart
            power_data = []
            for reading in readings[-10:]:
                power_data.append(reading.power)
            
            power_fig = {
                'data': [
                    go.Scatter(
                        x=timestamps,
                        y=power_data,
                        mode='lines+markers',
                        name='Power (kW)',
                        line={'color': 'red'}
                    )
                ],
                'layout': {
                    'title': 'System Power',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Power (kW)'},
                    'height': 300
                }
            }
            
            # Network traffic chart (simulated)
            network_fig = {
                'data': [
                    go.Bar(
                        x=['Modbus', 'DNP3', 'IEC61850'],
                        y=[random.randint(50, 200), random.randint(20, 80), random.randint(10, 50)],
                        name='Packets/sec'
                    )
                ],
                'layout': {
                    'title': 'Protocol Traffic',
                    'height': 300
                }
            }
            
            # Security events
            recent_alarms = self.scada_center.alarms[-5:] if self.scada_center.alarms else []
            security_events = html.Ul([
                html.Li(f"{alarm['timestamp'][:19]}: {alarm['type']} - {alarm['sensor_id']}")
                for alarm in recent_alarms
            ])
            
            return (
                html.Span(system_status, style={'color': status_color}),
                html.Span(str(alarm_count), style={'color': 'red' if alarm_count > 0 else 'green'}),
                frequency_display,
                voltage_fig,
                power_fig, 
                network_fig,
                security_events if recent_alarms else "No recent events"
            )
    
    def run_dashboard(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting dashboard server on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug, use_reloader=False)

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("POWER GRID CYBERSECURITY DEMONSTRATION")
    print("OpenDSS + SCADA + Dashboard + STRIDE/MITRE Testing")
    print("="*70)
    
    # Initialize components
    grid_sim = PowerGridSimulator()
    scada_center = SCADAControlCenter(grid_sim)
    security_tester = CyberSecurityTester()
    dashboard = PowerGridDashboard(grid_sim, scada_center)
    
    try:
        # Step 1: Initialize power grid simulation
        print("\n1. Initializing Power Grid Simulation...")
        grid_sim.initialize_grid()
        
        # Step 2: Start SCADA system
        print("\n2. Starting SCADA Control Center...")
        scada_center.initialize_modbus_server()
        
        # Step 3: Start grid simulation in background
        print("\n3. Starting Grid Simulation...")
        grid_thread = threading.Thread(target=grid_sim.run_simulation)
        grid_thread.daemon = True
        grid_thread.start()
        
        # Step 4: Start SCADA control loop
        print("\n4. Starting SCADA Control Loop...")
        scada_thread = threading.Thread(target=scada_center.run_control_loop)
        scada_thread.daemon = True
        scada_thread.start()
        
        # Step 5: Wait for systems to stabilize
        print("\n5. Waiting for systems to stabilize...")
        time.sleep(5)
        
        # Step 6: Start dashboard in background
        print("\n6. Starting Real-time Dashboard...")
        dashboard_thread = threading.Thread(
            target=lambda: dashboard.run_dashboard(debug=False)
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        print("\n✓ Dashboard available at: http://127.0.0.1:8050")
        print("✓ SCADA Modbus server running on port 502")
        
        # Step 7: Wait a bit more for full initialization
        time.sleep(3)
        
        # Step 8: Run cybersecurity assessment
        print("\n7. Running Cybersecurity Assessment...")
        print("   This will test STRIDE threats and MITRE ATT&CK techniques")
        input("   Press Enter to start security testing...")
        
        security_tester.run_comprehensive_test()
        
        # Step 9: Keep system running for demonstration
        print("\n8. System Running - Press Ctrl+C to stop")
        print("   - View dashboard at http://127.0.0.1:8050")
        print("   - Monitor logs for real-time events")
        print("   - SCADA system continues operating")
        
        # Keep main thread alive
        while True:
            time.sleep(10)
            print(f"System operational - {datetime.now().strftime('%H:%M:%S')}")
            
    except KeyboardInterrupt:
        print("\n\nShutting down system...")
        grid_sim.stop_simulation()
        print("✓ Power grid simulation stopped")
        print("✓ SCADA system stopped") 
        print("✓ Dashboard stopped")
        print("System shutdown complete.")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        grid_sim.stop_simulation()

if __name__ == "__main__":
    main()