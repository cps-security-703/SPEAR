# 🔍 EVCS Values Source Analysis

## **📍 Source of Values in Terminal Output**

### **1. 🏗️ Max Power: 50000.0kW**

**Source Location**: `focused_demand_analysis.py` - Lines 833-838

```python
# Create mock stations for DQN/SAC system
stations = []
for i in range(6):
    station = EVChargingStation(
        evcs_id=f"EVCS_{i}",
        bus_name=f"Bus_{i+1}",
        max_power=50000.0,        # ← This is the source of "50000.0kW"
        num_ports=4              # ← This is the source of "4 ports"
    )
    stations.append(station)
```

**Context**: This code creates 6 mock EVCS stations for the DQN/SAC security evasion system initialization.

### **2. ⏱️ Charging Time: 150.0 min**

**Source Location**: `hierarchical_cosimulation.py` - Lines 162-170

```python
# In EVChargingStation.__init__ method
if not hasattr(self, 'baseline_metrics') or not self.baseline_metrics:
    # Initialize baseline metrics if not already set
    self.baseline_metrics = {
        'avg_charging_time': 150.0,  # ← This is the source of "150.0 min"
        'queue_wait_time': 0.0,
        'current_load': 0.0,
        'utilization_rate': 0.0,
        'efficiency_score': 1.0,
        'customer_satisfaction': 1.0,
        'total_evs_served': 0,
        'total_energy_delivered': 0.0
    }
    print(f"📊 EVCS {self.evcs_id}: Baseline metrics initialized with correct charging time: 150.0 min")
    #                                                                                    ↑ This print statement
```

**Context**: This is set in the baseline metrics initialization for each EVCS station.

---

## **🏭 Different EVCS Configurations in the System**

Your system actually uses **two different EVCS configurations**:

### **🎯 Mock Stations (DQN/SAC System Initialization)**
- **Purpose**: Initial DQN/SAC trainer setup
- **Count**: 6 stations
- **Configuration**: 
  ```python
  max_power=50000.0  # 50MW (unrealistically high for testing)
  num_ports=4
  ```

### **🏗️ Real Stations (Actual Simulation)**
- **Purpose**: Actual hierarchical co-simulation
- **Count**: 60 stations (10 per distribution system × 6 systems)
- **Configuration**: Varies by system type

**Real Station Configurations** (from `federated_evcs_integration.py` and other config files):

```python
enhanced_evcs_configs = [
    # System 1 - Urban (High capacity)
    [
        {'bus': '890', 'max_power': 1000, 'num_ports': 25},  # 1MW mega hub
        {'bus': '844', 'max_power': 300, 'num_ports': 6},    # 300kW shopping
        {'bus': '860', 'max_power': 200, 'num_ports': 4},    # 200kW residential
        {'bus': '840', 'max_power': 400, 'num_ports': 10},   # 400kW business
        {'bus': '848', 'max_power': 250, 'num_ports': 5},    # 250kW industrial
        {'bus': '830', 'max_power': 300, 'num_ports': 6},    # 300kW suburban
        # ... more stations
    ],
    # System 2-6 with different configurations...
]
```

---

## **📊 Value Usage Flow**

### **🔄 50MW Mock Stations:**
1. **Creation**: `focused_demand_analysis.py:836` - Creates 6 mock stations with 50MW each
2. **Purpose**: Initialize DQN/SAC trainer with extreme values for testing
3. **Usage**: DQN/SAC agent normalization and training setup
4. **Terminal Output**: "Initialized EVCS EVCS_0 with 4 ports, max power 50000.0kW"

### **⏰ 150-minute Charging Time:**
1. **Definition**: `hierarchical_cosimulation.py:162` - Sets baseline charging time
2. **Trigger**: Called during `EVChargingStation.__init__()` 
3. **Purpose**: Realistic baseline for attack impact calculation
4. **Terminal Output**: "📊 EVCS EVCS_0: Baseline metrics initialized with correct charging time: 150.0 min"

### **🏗️ Real Station Values:**
1. **Definition**: Multiple config files with realistic values (200kW-1MW)
2. **Usage**: Actual co-simulation with realistic power ratings
3. **Context**: 60 stations across 6 distribution systems

---

## **🎯 Why These Specific Values?**

### **50MW Mock Stations:**
- **Reason**: Extreme values for DQN/SAC agent testing
- **Impact**: Forces agents to handle wide parameter ranges
- **Normalization**: Values are normalized in agent training (`max_power_norm = 50000.0 / 100000.0`)

### **150-minute Charging Time:**
- **Reason**: Realistic EV charging duration (2.5 hours)
- **Context**: Typical Level 2/3 charging for 40-60kWh battery (20% → 80%)
- **Usage**: Baseline for attack impact measurement

### **Real Station Values (200kW-1MW):**
- **Reason**: Industry-standard EVCS power ratings
- **Range**: 
  - 200kW: Residential/small commercial
  - 300kW: Shopping centers
  - 400-500kW: Business districts
  - 1000kW: Mega charging hubs

---

## **🔧 File Locations Summary**

| Value | Source File | Line | Purpose |
|-------|-------------|------|---------|
| `max_power=50000.0` | `focused_demand_analysis.py` | 836 | Mock DQN/SAC stations |
| `num_ports=4` | `focused_demand_analysis.py` | 837 | Mock DQN/SAC stations |
| `avg_charging_time: 150.0` | `hierarchical_cosimulation.py` | 162 | Baseline metrics |
| Print statement | `hierarchical_cosimulation.py` | 170 | Debug output |
| Real EVCS configs | `federated_evcs_integration.py` | 191-228 | Actual simulation |
| Alternative configs | `pinn_optimizer.py` | 582-591 | PINN training data |
| Global configs | `global_federated_optimizer.py` | 65-85 | Federated optimization |

---

## **💡 Key Insight**

The **50MW values** you see in the terminal are from **mock stations** used for DQN/SAC initialization, not the real simulation values. The actual simulation uses **realistic 200kW-1MW stations** with proper power ratings for each distribution system type.

**The 150-minute charging time** is a realistic baseline used across all stations for attack impact calculation and represents typical EV charging duration.
