# 🛡️ CMS Security & Anomaly Detection Analysis

## 🎯 **Overview**

The CMS (Charging Management System) has a **multi-layered security architecture** that monitors, detects, and responds to cyber attacks in real-time. Here's how it works during simulation:

---

## 🔍 **1. Security Validation Layers**

### **🚨 Layer 1: Upper Bounds Check**
**Purpose**: Prevent dangerous operating conditions

```python
# Hard limits enforcement
if power_ref > self.max_power_reference:        # Default: 100kW
    power_ref = self.max_power_reference
    
if voltage_ref > self.max_voltage_reference:    # Default: 500V
    voltage_ref = self.max_voltage_reference
    
if current_ref > self.max_current_reference:    # Default: 200A
    current_ref = self.max_current_reference
```

**Action**: **Clamps values** to safe limits, **allows operation** with reduced parameters

---

### **🚨 Layer 2: Rate-of-Change Detection**
**Purpose**: Detect sudden parameter changes that indicate attacks

```python
# Calculate rate of change from previous timestep
power_change_rate = abs(power_ref - last_power) / max(last_power, 1.0)
voltage_change_rate = abs(voltage_ref - last_voltage) / max(last_voltage, 1.0)
current_change_rate = abs(current_ref - last_current) / max(last_current, 1.0)

# Threshold check
if (power_change_rate > self.rate_change_limit):  # Default: 0.5 (50% change)
    # Apply rate limiting
    power_ref = last_power + direction * last_power * self.rate_change_limit
    self.anomaly_counters[station_id] += 1
```

**Action**: **Rate-limits changes**, **increments anomaly counter**

---

### **🚨 Layer 3: Statistical Anomaly Detection**
**Purpose**: Detect values that deviate significantly from normal patterns

```python
def _detect_statistical_anomaly(self, station_id, power_ref, voltage_ref, current_ref):
    # Get recent 5 power references
    recent_powers = [ref['power'] for ref in self.reference_history[station_id][-5:]]
    mean_power = np.mean(recent_powers)
    std_power = np.std(recent_powers)
    
    # Z-score anomaly detection
    if std_power > 0:
        z_score = abs(power_ref - mean_power) / std_power
        if z_score > 2.5:  # 2.5 sigma threshold (99.4% confidence)
            return True  # Anomaly detected
```

**Action**: **Flags statistical outliers**, **increments anomaly counter**

---

### **🚨 Layer 4: Input Pattern Analysis**
**Purpose**: Detect anomalous input patterns that suggest manipulation

```python
def _detect_input_anomaly(self, station_id, station_data):
    # Check demand factor changes
    recent_demands = [inp['demand_factor'] for inp in self.input_history[station_id][-3:]]
    demand_change = abs(recent_demands[-1] - recent_demands[-2]) / max(recent_demands[-2], 0.1)
    
    if demand_change > self.anomaly_threshold:  # Default: 0.3 (30% change)
        return True
    
    # Check urgency factor changes
    recent_urgency = [inp['urgency_factor'] for inp in self.input_history[station_id][-3:]]
    urgency_change = abs(recent_urgency[-1] - recent_urgency[-2])
    
    if urgency_change > 1.0:  # Urgency shouldn't change dramatically
        return True
```

**Action**: **Detects input manipulation**, **increments anomaly counter**

---

### **🚨 Layer 5: Physical Constraint Validation**
**Purpose**: Ensure inputs comply with physical laws and system limits

```python
def validate_physical_constraints(self, inputs):
    violations = {}
    
    # SOC constraints
    if not (0.0 <= soc <= 1.0):
        violations['soc'] = "SOC outside valid range [0.0, 1.0]"
    
    # Voltage constraints (per unit)
    if not (0.85 <= grid_voltage <= 1.15):
        violations['voltage'] = "Grid voltage outside safe range [0.85, 1.15] pu"
    
    # Frequency constraints
    if not (59.0 <= frequency <= 61.0):
        violations['frequency'] = "Frequency outside normal range [59.0, 61.0] Hz"
    
    # Demand factor constraints
    if not (0.0 <= demand_factor <= 2.0):
        violations['demand'] = "Demand factor outside realistic range [0.0, 2.0]"
```

**Action**: **Validates physical realism**, **prevents impossible values**

---

### **🚨 Layer 6: Attack Pattern Detection**
**Purpose**: Detect coordinated attack patterns across time

```python
def detect_attack_patterns(self, current_load, system_id):
    # Detect sudden large load changes
    if load_change > self.load_change_threshold:  # Default: 25kW
        return True, "Suspicious load change detected"
    
    # Detect unrealistic load injection
    if current_load > self.max_system_load:  # Default: 500MW
        return True, "Unrealistic load injection exceeds system capacity"
    
    # Detect oscillating patterns (potential attack)
    if len(recent_loads) >= 4:
        changes = [recent_loads[i+1] - recent_loads[i] for i in range(len(recent_loads)-1)]
        sign_changes = sum(1 for i in range(len(changes)-1) if changes[i] * changes[i+1] < 0)
        if sign_changes >= 2:
            return True, "Oscillating load pattern detected (potential attack)"
```

**Action**: **Detects coordinated attacks**, **identifies attack signatures**

---

## ⚖️ **2. Decision Logic: Allow vs Discard**

### **🟢 ALLOW Operation (Modified Parameters)**
The CMS **rarely discards completely** - instead it **modifies/clamps** values to safe ranges:

```python
# Example: Upper bounds exceeded
if power_ref > 100kW:
    power_ref = 100kW        # Clamp to safe limit
    # Continue operation with reduced power
    
# Example: Rate limiting
if change_rate > 50%:
    power_ref = last_power * 1.5  # Limit to 50% change
    # Continue with rate-limited change
```

### **🔴 DISCARD/Emergency Mode (Attack Detected)**
Only when **consecutive anomalies** exceed the limit:

```python
if self.anomaly_counters[station_id] >= self.consecutive_anomaly_limit:  # Default: 3
    print("ATTACK DETECTED - Switching to emergency safe mode")
    
    # Emergency safe mode: conservative references
    voltage_ref = 400.0  # Safe voltage
    current_ref = 25.0   # Reduced current  
    power_ref = 10.0     # Minimal power
    
    return voltage_ref, current_ref, power_ref, False  # Attack detected = False
```

---

## 🔧 **3. Security Configuration Parameters**

### **Detection Thresholds:**
```python
self.anomaly_threshold = 0.3           # 30% change threshold for inputs
self.rate_change_limit = 0.5           # 50% max change per timestep
self.consecutive_anomaly_limit = 3     # 3 consecutive anomalies trigger emergency
self.max_power_reference = 100.0       # 100kW upper bound
self.max_voltage_reference = 500.0     # 500V upper bound
self.max_current_reference = 200.0     # 200A upper bound
```

### **Statistical Detection:**
```python
z_score_threshold = 2.5                # 2.5 sigma (99.4% confidence)
history_window = 20                    # Keep last 20 entries for analysis
min_history = 5                        # Need 5 samples for statistical analysis
```

### **Physical Constraints:**
```python
soc_range = [0.0, 1.0]                # SOC must be 0-100%
voltage_range = [0.85, 1.15]          # Grid voltage ±15% per unit
frequency_range = [59.0, 61.0]        # Frequency ±1Hz from nominal
demand_range = [0.0, 2.0]             # Demand factor 0-200%
```

---

## 🎭 **4. How Agents Bypass Security**

### **🕵️ Stealth Strategies:**
1. **Gradual Injection**: Changes below rate limits (< 50% per step)
2. **Statistical Evasion**: Stay within 2.5 sigma bounds
3. **Pattern Mimicking**: Avoid oscillating signatures
4. **Physical Compliance**: Keep values within realistic ranges

### **🎯 Agent Learning:**
```python
# DQN learns strategic evasion
'evasion_strategy': 0=stealth, 1=burst, 2=adaptive

# SAC learns continuous stealth factors
'stealth_factor': 0.0-1.0  # Higher = more stealthy
'magnitude': -1.0 to 1.0   # Attack intensity
```

---

## 🚀 **5. Runtime Behavior During Simulation**

### **Normal Operation:**
```
✅ All checks pass → Use PINN-optimized references
✅ Minor violations → Apply clamping/rate limiting
✅ Continue normal charging operations
```

### **Stealth Attack (High stealth_factor > 0.7):**
```
⚠️ Small gradual changes → Rate limiting applied
⚠️ Statistical detection may miss → Changes within 2.5 sigma
⚠️ Input patterns normal → Below 30% threshold
✅ Operation continues with attack-modified values
```

### **Obvious Attack (Low stealth_factor < 0.3):**
```
🚨 Large sudden changes → Rate limiting + anomaly counter++
🚨 Statistical outliers → Z-score > 2.5 + anomaly counter++
🚨 Dramatic input changes → Pattern detection + anomaly counter++
🚨 3rd consecutive anomaly → EMERGENCY SAFE MODE ACTIVATED
❌ Attack blocked, minimal power/voltage/current references
```

---

## 📊 **6. Attack Success vs Detection Trade-off**

### **High Impact + High Detection Risk:**
- **10x amplified attacks** without stealth
- **Immediate detection** and emergency mode
- **Minimal actual impact** due to blocking

### **Moderate Impact + Low Detection Risk:**
- **Gradual 10x amplified attacks** with high stealth
- **Rate limiting reduces** but doesn't eliminate impact
- **Cumulative effect** over attack duration creates significant deviation

### **Optimal Attack Strategy:**
- **Gradual injection** over 50-second duration
- **Stay below rate limits** (50% per step)
- **High stealth factor** (> 0.8) from SAC agent
- **Result**: **5-10MW impact** while evading detection

---

## 🎯 **Summary**

The CMS security system is **sophisticated but not impenetrable**. It **prioritizes system safety** over attack prevention, meaning:

1. **Dangerous values** are **clamped to safe ranges**
2. **Rapid changes** are **rate-limited**
3. **Statistical outliers** are **flagged but allowed**
4. **Only sustained attacks** trigger **emergency shutdown**

**Your enhanced agents exploit this by:**
- **Gradual injection** to avoid rate limits
- **High stealth factors** to stay within statistical bounds
- **10x amplification** applied gradually for maximum cumulative impact
- **Result**: **Massive power deviations** while appearing normal to security systems! 🚀⚡
