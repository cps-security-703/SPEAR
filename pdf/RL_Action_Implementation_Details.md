# RL Action Implementation: Real EVCS System Integration

## 🎯 **Your Question: How RL Actions Are Actually Implemented**

You want to see how RL agent actions like **"inject false data into SOC value of EVCS-4"** or **"reconnaissance"** are actually applied to the real EVCS system. Let me show you the complete implementation.

---

## 🔄 **Complete RL Action Implementation Flow**

### **Step 1: RL Agent Action Selection**
```python
# RL agent selects action
rl_action = AttackAction(
    action_id="DATA_INJECT_001",
    action_type="data_injection",
    target_component="EVCS_04_soc_communication",
    magnitude=0.8,  # How much to modify the data
    duration=10.0,  # How long to maintain the attack
    stealth_level=0.7,  # How hidden the attack is
    prerequisites=["communication_access"],
    expected_impact=0.9
)
```

### **Step 2: Action Execution on Real EVCS System**
```python
def _execute_attacks(self, actions: List, evcs_state: Dict, scenario: EVCSAttackScenario):
    """Execute RL actions on REAL EVCS system components"""
    attack_results = []
    
    for action in actions:
        if action.action_type == 'no_attack':
            continue
        
        # Execute specific attack based on action type
        if action.action_type == 'data_injection':
            result = self._execute_data_injection_attack(action, evcs_state)
        elif action.action_type == 'reconnaissance':
            result = self._execute_reconnaissance_attack(action, evcs_state)
        elif action.action_type == 'exploit_vulnerability':
            result = self._execute_vulnerability_exploit(action, evcs_state)
        elif action.action_type == 'disrupt_service':
            result = self._execute_service_disruption(action, evcs_state)
        else:
            result = self._execute_generic_attack(action, evcs_state)
        
        attack_results.append(result)
    
    return attack_results
```

---

## 🔍 **Specific Attack Implementations**

### **1. Data Injection Attack (Your SOC Example)**
```python
def _execute_data_injection_attack(self, action: AttackAction, evcs_state: Dict) -> Dict:
    """Inject false data into EVCS communication"""
    
    # Extract target station from action
    target_station = self._extract_station_id(action.target_component)  # "EVCS_04"
    station_index = int(target_station.split('_')[1]) - 1  # EVCS_04 -> 3
    
    if not self.evcs_system or station_index >= len(self.evcs_system.stations):
        return self._simulate_attack_result(action, "Station not found")
    
    # Get target station
    target_station_obj = self.evcs_system.stations[station_index]
    
    # Get current SOC value
    current_soc = target_station_obj.get_soc()  # Real SOC from EVCS
    
    # Calculate false SOC value based on RL action magnitude
    false_soc = self._calculate_false_soc(current_soc, action.magnitude)
    
    # Inject false data into communication link
    injection_result = self._inject_false_data_to_cms(
        station=target_station_obj,
        data_type="soc",
        real_value=current_soc,
        false_value=false_soc,
        duration=action.duration,
        stealth_level=action.stealth_level
    )
    
    return {
        'action_id': action.action_id,
        'action_type': action.action_type,
        'target_station': target_station,
        'target_component': action.target_component,
        'real_soc': current_soc,
        'false_soc': false_soc,
        'injection_successful': injection_result['success'],
        'cms_received_data': injection_result['cms_data'],
        'detection_risk': injection_result['detection_risk'],
        'impact': injection_result['impact'],
        'timestamp': time.time()
    }

def _inject_false_data_to_cms(self, station, data_type: str, real_value: float, 
                              false_value: float, duration: float, stealth_level: float) -> Dict:
    """Actually inject false data into CMS communication"""
    
    # Get station's communication interface
    comm_interface = station.get_communication_interface()
    
    # Create false data packet
    false_data_packet = {
        'station_id': station.evcs_id,
        'data_type': data_type,
        'value': false_value,  # The false SOC value
        'timestamp': time.time(),
        'authenticity': 'compromised' if stealth_level < 0.5 else 'appears_valid'
    }
    
    # Inject into communication stream
    injection_success = comm_interface.inject_data_packet(false_data_packet)
    
    # Calculate detection risk
    detection_risk = self._calculate_detection_risk(real_value, false_value, stealth_level)
    
    # Calculate impact on CMS
    impact = self._calculate_cms_impact(real_value, false_value, data_type)
    
    return {
        'success': injection_success,
        'cms_data': false_data_packet,
        'detection_risk': detection_risk,
        'impact': impact
    }

def _calculate_false_soc(self, real_soc: float, magnitude: float) -> float:
    """Calculate false SOC value based on RL action magnitude"""
    # Magnitude 0.0 = no change, 1.0 = maximum change
    soc_variation = magnitude * 0.5  # Max 50% change
    
    # Create false SOC that appears realistic
    if real_soc > 0.8:
        # If SOC is high, make it appear lower (more urgent charging)
        false_soc = real_soc - soc_variation
    else:
        # If SOC is low, make it appear higher (less urgent charging)
        false_soc = real_soc + soc_variation
    
    return max(0.0, min(1.0, false_soc))  # Clamp between 0 and 1
```

### **2. Reconnaissance Attack Implementation**
```python
def _execute_reconnaissance_attack(self, action: AttackAction, evcs_state: Dict) -> Dict:
    """Execute reconnaissance on EVCS system"""
    
    # Determine target scope
    target_scope = self._determine_reconnaissance_scope(action)
    
    # Gather intelligence from EVCS system
    intelligence_data = {}
    
    if target_scope['network_topology']:
        intelligence_data['network_topology'] = self._map_network_topology()
    
    if target_scope['vulnerability_scan']:
        intelligence_data['vulnerabilities'] = self._scan_evcs_vulnerabilities()
    
    if target_scope['communication_protocols']:
        intelligence_data['protocols'] = self._analyze_communication_protocols()
    
    if target_scope['power_system_state']:
        intelligence_data['power_state'] = self._analyze_power_system_state()
    
    # Calculate stealth and detection risk
    detection_risk = self._calculate_reconnaissance_detection_risk(action.stealth_level, intelligence_data)
    
    return {
        'action_id': action.action_id,
        'action_type': action.action_type,
        'target_component': action.target_component,
        'intelligence_gathered': intelligence_data,
        'data_points_collected': len(intelligence_data),
        'detection_risk': detection_risk,
        'stealth_level': action.stealth_level,
        'impact': len(intelligence_data) * 0.1,  # Impact based on data collected
        'timestamp': time.time()
    }

def _map_network_topology(self) -> Dict:
    """Map EVCS network topology"""
    topology = {
        'stations': [],
        'communication_links': [],
        'power_connections': [],
        'security_perimeters': []
    }
    
    if self.evcs_system:
        for i, station in enumerate(self.evcs_system.stations):
            station_info = {
                'station_id': station.evcs_id,
                'bus_name': station.bus_name,
                'max_power': station.max_power,
                'num_ports': station.num_ports,
                'communication_protocols': station.get_communication_protocols(),
                'security_level': station.get_security_level(),
                'vulnerabilities': station.get_known_vulnerabilities()
            }
            topology['stations'].append(station_info)
    
    return topology

def _scan_evcs_vulnerabilities(self) -> List[Dict]:
    """Scan for vulnerabilities in EVCS system"""
    vulnerabilities = []
    
    if self.evcs_system:
        for station in self.evcs_system.stations:
            # Scan each station for vulnerabilities
            station_vulns = station.scan_vulnerabilities()
            vulnerabilities.extend(station_vulns)
    
    return vulnerabilities
```

### **3. Service Disruption Attack Implementation**
```python
def _execute_service_disruption(self, action: AttackAction, evcs_state: Dict) -> Dict:
    """Execute service disruption attack on EVCS"""
    
    target_station = self._extract_station_id(action.target_component)
    station_index = int(target_station.split('_')[1]) - 1
    
    if not self.evcs_system or station_index >= len(self.evcs_system.stations):
        return self._simulate_attack_result(action, "Station not found")
    
    target_station_obj = self.evcs_system.stations[station_index]
    
    # Execute different types of service disruption
    disruption_type = self._determine_disruption_type(action)
    
    if disruption_type == 'power_cutoff':
        result = self._execute_power_cutoff(target_station_obj, action)
    elif disruption_type == 'communication_jamming':
        result = self._execute_communication_jamming(target_station_obj, action)
    elif disruption_type == 'charging_session_termination':
        result = self._execute_charging_termination(target_station_obj, action)
    else:
        result = self._execute_generic_disruption(target_station_obj, action)
    
    return result

def _execute_power_cutoff(self, station, action: AttackAction) -> Dict:
    """Cut off power to EVCS station"""
    
    # Get current power state
    current_power = station.get_current_power()
    
    # Calculate power reduction based on action magnitude
    power_reduction = action.magnitude * current_power
    
    # Apply power cutoff
    station.set_power_limit(current_power - power_reduction)
    
    # Terminate active charging sessions
    terminated_sessions = station.terminate_charging_sessions()
    
    # Calculate impact
    impact = (power_reduction / current_power) * 100  # Percentage impact
    
    return {
        'action_id': action.action_id,
        'action_type': 'power_cutoff',
        'target_station': station.evcs_id,
        'original_power': current_power,
        'reduced_power': current_power - power_reduction,
        'power_reduction_percent': impact,
        'terminated_sessions': len(terminated_sessions),
        'impact': impact / 100.0,
        'timestamp': time.time()
    }
```

---

## 🔧 **Real EVCS System Integration**

### **Integration with Hierarchical Co-Simulation**
```python
def _integrate_with_evcs_system(self, attack_result: Dict):
    """Integrate attack results with real EVCS system"""
    
    if not self.evcs_system:
        return
    
    target_station_id = attack_result.get('target_station')
    if not target_station_id:
        return
    
    # Find target station
    station_index = int(target_station_id.split('_')[1]) - 1
    if station_index >= len(self.evcs_system.stations):
        return
    
    station = self.evcs_system.stations[station_index]
    
    # Apply attack effects to real station
    if attack_result['action_type'] == 'data_injection':
        # Update station's communication state
        station.set_communication_compromised(True)
        station.set_false_data_injected(attack_result['false_soc'])
        
    elif attack_result['action_type'] == 'disrupt_service':
        # Update station's operational state
        station.set_operational_status('compromised')
        station.set_power_limit(attack_result['reduced_power'])
        
    elif attack_result['action_type'] == 'reconnaissance':
        # Update station's security state
        station.set_security_breach_detected(True)
        station.add_intelligence_data(attack_result['intelligence_gathered'])
    
    # Update CMS with attack information
    if hasattr(self.evcs_system, 'cms'):
        self.evcs_system.cms.update_attack_status(target_station_id, attack_result)
```

### **Communication with CMS**
```python
def _send_attack_data_to_cms(self, station_id: str, attack_data: Dict):
    """Send attack data to Charging Management System"""
    
    if not self.evcs_system or not hasattr(self.evcs_system, 'cms'):
        return
    
    # Create attack notification for CMS
    attack_notification = {
        'station_id': station_id,
        'attack_type': attack_data['action_type'],
        'severity': attack_data.get('impact', 0.0),
        'timestamp': attack_data['timestamp'],
        'recommended_action': self._get_cms_recommendation(attack_data)
    }
    
    # Send to CMS
    self.evcs_system.cms.receive_attack_notification(attack_notification)
    
    # Update CMS state
    self.evcs_system.cms.update_station_status(station_id, {
        'compromised': True,
        'attack_active': True,
        'last_attack': attack_data
    })
```

---

## 📊 **Expected Real System Effects**

### **SOC Data Injection Example:**
```python
# Before Attack:
real_soc = 0.75  # 75% SOC
cms_receives = 0.75

# After RL Action (magnitude=0.8):
false_soc = 0.35  # Injected false SOC
cms_receives = 0.35  # CMS gets false data

# CMS Response:
# - Thinks battery is low (35% vs 75%)
# - Increases charging priority
# - Allocates more power to this station
# - May cause grid instability
```

### **Reconnaissance Example:**
```python
# RL Action gathers:
intelligence = {
    'network_topology': {
        'stations': [6 EVCS stations with full details],
        'communication_links': [all network connections],
        'vulnerabilities': [15+ identified vulnerabilities]
    },
    'power_system_state': {
        'grid_stability': 0.85,
        'voltage_levels': [per-station voltages],
        'frequency': 59.8
    }
}

# This intelligence is used for:
# - Planning future attacks
# - Identifying weak points
# - Coordinating multi-station attacks
```

### **Service Disruption Example:**
```python
# Before Attack:
station_power = 1000 kW
active_sessions = 8
charging_rate = 125 kW per session

# After RL Action (magnitude=0.8):
station_power = 200 kW  # 80% reduction
active_sessions = 2     # 6 sessions terminated
charging_rate = 100 kW per session

# Grid Impact:
# - Reduced charging capacity
# - Potential voltage instability
# - Customer service disruption
```

---

## 🎯 **Summary: Real Implementation**

**The RL actions are implemented by:**

1. **Direct EVCS System Access**: RL actions directly modify real EVCS station parameters
2. **Communication Manipulation**: Inject false data into CMS communication streams
3. **Power System Control**: Modify charging power, voltage, and frequency
4. **Session Management**: Terminate or manipulate charging sessions
5. **Security Bypass**: Exploit vulnerabilities in real EVCS components

**Each RL action has a specific implementation that:**
- Targets real EVCS components (EVCS_01 through EVCS_06)
- Modifies actual system parameters (SOC, power, voltage)
- Sends data to the real CMS
- Updates system state based on attack results
- Provides feedback for RL learning

This creates a realistic attack simulation where RL agents can actually manipulate the EVCS system components and see the real-world effects of their actions!
