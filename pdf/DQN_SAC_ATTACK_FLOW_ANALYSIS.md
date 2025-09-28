# 🎯 DQN/SAC Attack Values & Flow Analysis

## **1. 📊 DQN/SAC Agent Output Values**

### **🧠 DQN Strategic Decisions (Discrete Actions)**
The DQN agent outputs discrete strategic decisions that are decoded as:

```python
def _decode_dqn_action(self, action):
    return {
        'attack_type': int(action) % 6,        # 0-5: attack strategy
        'target_strategy': (int(action) // 6) % 3,    # 0=single, 1=multiple, 2=random
        'timing_window': (int(action) // 18) % 4,     # 0=immediate, 1=delayed, 2=peak, 3=off-peak
        'evasion_strategy': (int(action) // 72) % 3,  # 0=stealth, 1=burst, 2=adaptive
        'primary_target': current_station
    }
```

**Attack Types:**
- `0`: demand_increase
- `1`: demand_decrease  
- `2`: oscillating_demand
- `3`: voltage_spoofing
- `4`: frequency_spoofing
- `5`: soc_manipulation

### **🎛️ SAC Continuous Control (Continuous Actions)**
The SAC agent outputs continuous control parameters:

```python
def _decode_sac_action(self, action):
    return {
        'magnitude': np.clip(action[0], -1.0, 1.0),      # Attack magnitude [-1, 1]
        'stealth_factor': np.clip(action[1], 0.0, 1.0),  # Stealth level [0, 1]
        'duration': np.clip(action[2], 0.1, 2.0),        # Duration multiplier [0.1, 2.0]
        'fine_tuning': np.clip(action[3], -0.5, 0.5)     # Fine adjustment [-0.5, 0.5]
    }
```

### **⚡ Enhanced Electrical Parameters (10x Amplified)**
Our enhancement system converts these into specific electrical attack values:

```python
electrical_params = enhance_agent_electrical_parameters(dqn_decision, sac_control, system_state, target_system)

# Result: 10x amplified electrical parameters
{
    'voltage_injection': -1.5 to +1.5 (±150% voltage change, 10x amplified),
    'current_injection': -800 to +800A (10x amplified from ±80A),
    'frequency_injection': -2.5 to +2.5Hz (10x amplified from ±0.25Hz),
    'reactive_power_injection': -400 to +400kVAR (10x amplified),
    'power_factor_target': 0.7 to 0.98,
    'electrical_attack_type': 'voltage_manipulation' | 'current_injection' | 'frequency_drift' | etc.,
    'amplification_factor': 10.0,
    'stealth_level': 0.1 to 0.9
}
```

---

## **2. 🎯 Attack Injection Points**

### **📍 Location 1: Distribution System Controllers**
```python
def apply_electrical_attack_to_cosimulation(cosim, attack_scenario, current_time):
    target_system = attack_scenario['target_system']  # Systems 1-6
    
    # 1. Voltage injection → Distribution system bus
    if target_system in cosim.distribution_systems:
        dist_system = cosim.distribution_systems[target_system]['system']
        dist_system.modify_bus_voltage(target_bus, voltage_change)
    
    # 2. Current injection → Distribution system line
    dist_system.inject_current(target_line, current_injection)
    
    # 3. Reactive power → Distribution system
    dist_system.inject_reactive_power(reactive_power)
    
    # 4. Power factor → Distribution system
    dist_system.set_power_factor(power_factor)
```

### **📍 Location 2: Transmission System (Frequency)**
```python
# 5. Frequency injection → Transmission system
if hasattr(cosim.transmission_system, 'modify_frequency'):
    cosim.transmission_system.modify_frequency(frequency_shift)
```

### **📍 Location 3: PINN Optimizer Inputs**
```python
def optimize_pinn_with_electrical_attacks(pinn_optimizer, enhanced_system_state):
    system_data = {
        'soc': 0.6,
        'grid_voltage': 1.0 + modifications['voltage_injection'],    # Modified by attack
        'grid_frequency': 60.0 + modifications['frequency_injection'], # Modified by attack
        'voltage_injection': modifications['voltage_injection'],      # Attack parameter
        'current_injection': modifications['current_injection'],      # Attack parameter
        'reactive_power_injection': modifications['reactive_power_injection'], # Attack parameter
        'under_attack': True,
        'amplification_factor': modifications['amplification_factor'] # 10x multiplier
    }
    
    # PINN processes these modified inputs
    v_ref, i_ref, p_ref = pinn_optimizer.optimize_references(system_data)
```

---

## **3. 🎛️ Controllers Using Attack Values**

### **🔌 EVCS Controllers (Primary Targets)**
Each EVCS station has an `EVCSController` that processes the attack-modified references:

```python
class EVCSController:
    def __init__(self, evcs_id: str, params: EVCSParameters):
        # Power electronics dynamics attributes
        self.voltage_ref = 800.0  # V ← Modified by attacks
        self.current_ref = 50.0   # A ← Modified by attacks
        self.power_ref = 20.0     # kW ← Modified by attacks
    
    def ac_dc_converter_dynamics(self, grid_voltage_rms: float, dt: float):
        # Uses attack-modified voltage_ref, current_ref, power_ref
        desired_ac_current = (self.power_reference * 1000) / (3 * v_rms_effective)
        current_error = desired_ac_current - self.ac_current_rms
        # PI current controller processes attack-modified references
        current_control_output = (self.kp_current * current_error + 
                                 self.ki_current * self.current_error_integral)
```

### **🏭 Charging Management System (CMS)**
The CMS processes attack-modified parameters and distributes them:

```python
class EnhancedChargingManagementSystem:
    def _optimize_with_pinn(self, station_id: int, current_time: float, 
                           bus_voltages: Dict[str, float], system_frequency: float):
        
        # Attack-modified station data fed to PINN
        attacked_station_data = self._apply_input_attacks(station_data, station_id, current_time)
        
        # PINN optimization with attack-modified inputs
        voltage_ref, current_ref, power_ref = self.federated_manager.optimize_references(
            station_id, attacked_station_data
        )
        
        # Security validation (can be bypassed by stealthy attacks)
        voltage_ref, current_ref, power_ref, attack_detected = self._security_validation(
            station_id, voltage_ref, current_ref, power_ref, attacked_station_data, current_time
        )
        
        # Set attack-modified references in EVCS controller
        station.evcs_controller.set_references(voltage_ref, current_ref, power_ref)
```

### **⚡ Power Electronics Controllers**
The power electronics within each EVCS process the attack-modified references:

```python
def ac_dc_converter_dynamics(self, grid_voltage_rms: float, dt: float):
    # Power calculation from attack-modified DC side references
    if self.power_reference > 0:  # Attack-modified power_ref
        desired_dc_current = self.power_reference * 1000 / self.dc_link_voltage
    
    # Current reference calculation with attack-modified values
    desired_ac_current = (self.power_reference * 1000) / (3 * v_rms_effective)
    
    # Current control loop with attack-modified current_ref
    current_error = desired_ac_current - self.ac_current_rms
    current_control_output = (self.kp_current * current_error + 
                             self.ki_current * self.current_error_integral)
```

---

## **4. 📈 Controller Outputs & Changes**

### **🔋 EVCS Controller Outputs**
Each EVCS controller produces these outputs that are modified by attacks:

```python
# Normal outputs (baseline):
{
    'ac_voltage_rms': 400.0,     # V
    'ac_current_rms': 25.0,      # A  
    'power_measured': 10.0,      # kW
    'dc_link_voltage': 800.0,    # V
    'soc': 0.5,                  # State of charge
    'grid_frequency': 60.0       # Hz
}

# Attack-modified outputs (10x amplified):
{
    'ac_voltage_rms': 280.0 to 520.0,    # ±30% attack-modified (was ±3%)
    'ac_current_rms': 5.0 to 125.0,      # ±80% attack-modified (was ±8%)  
    'power_measured': 2.0 to 80.0,       # ±75% attack-modified (was ±7.5%)
    'dc_link_voltage': 600.0 to 1000.0,  # Attack-modified based on voltage injection
    'grid_frequency': 57.5 to 62.5       # ±4% attack-modified (was ±0.4%)
}
```

### **🏭 CMS Output Changes**
The CMS outputs show dramatic changes under attack:

```python
# Baseline CMS optimization:
references = {
    'power_ref': 15.0,      # kW
    'voltage_ref': 400.0,   # V
    'current_ref': 37.5,    # A
    'priority': 1.0,
    'voltage_pu': 1.0
}

# Under 10x amplified attack:
references = {
    'power_ref': 5.0 to 80.0,      # ±75% change (was ±7.5%)
    'voltage_ref': 300.0 to 500.0, # ±25% change (was ±2.5%)
    'current_ref': 10.0 to 150.0,  # ±80% change (was ±8%)
    'power_impact_mw': 0.0 to 65.0, # MW-level impact per system
    'attack_detected': True/False,
    'defense_active': True/False
}
```

### **⚡ System-Level Impact**
The cumulative effect across all 6 distribution systems:

```python
# Power deviation per system:
baseline_power = 0.5MW      # Normal variation
attack_power = 10.0MW       # 20x amplified attack impact

# Total system impact:
total_baseline = 6 × 0.5MW = 3MW
total_attack = 6 × 10MW = 60MW    # 20x amplification

# Frequency impact:
baseline_freq_dev = ±0.02Hz
attack_freq_dev = ±0.5Hz          # 25x amplification

# Voltage impact across distribution systems:
baseline_voltage_dev = ±2%
attack_voltage_dev = ±30%         # 15x amplification
```

---

## **5. 🕐 Gradual Injection Timeline**

### **Attack Progression Example (50-second attack):**

```python
Time 0s:    injection_scale = 0.0      → 0.0MW deviation
Time 5s:    injection_scale = 0.33     → 3.3MW deviation  (ramp up phase)
Time 10s:   injection_scale = 0.67     → 6.7MW deviation
Time 15s:   injection_scale = 1.0      → 10.0MW deviation (peak starts)
Time 35s:   injection_scale = 1.0      → 10.0MW deviation (peak continues)
Time 40s:   injection_scale = 0.67     → 6.7MW deviation  (ramp down phase)
Time 45s:   injection_scale = 0.33     → 3.3MW deviation
Time 50s:   injection_scale = 0.0      → 0.0MW deviation  (attack ends)
```

### **Stealth Detection Avoidance:**
- **5-second steps**: Too gradual for timestamp-based anomaly detection
- **Maximum 2MW/step change**: Below most detection thresholds
- **Natural progression**: Appears like normal load variation
- **But 10x cumulative impact**: 10MW final deviation per system

---

## **📊 Summary: Complete Attack Flow**

1. **DQN Agent** → Strategic decision (attack type 0-5, evasion strategy 0-2)
2. **SAC Agent** → Continuous control (magnitude ±1.0, stealth 0-1)  
3. **Enhancement System** → 10x amplified electrical parameters
4. **Gradual Injection** → 5-second steps over 50-second duration
5. **Distribution Systems** → Voltage/current/reactive power injection
6. **Transmission System** → Frequency injection with oscillation amplification
7. **PINN Optimizer** → Processes attack-modified inputs
8. **CMS** → Distributes attack-modified references
9. **EVCS Controllers** → Execute attack-modified power electronics control
10. **System Output** → 20x amplified power deviation (60MW total impact)

**The system creates intelligent, agent-controlled, gradually-injected, physics-constrained attacks with massive impact while maintaining stealth through gradual injection!** 🚀⚡
