# System Indexing Fix Summary

## Problem Identified
The error "Unexpected error: 0" was caused by a **system indexing mismatch** between:
- **focused_demand_analysis.py**: Using 1-indexed systems (1-6)
- **hierarchical_cosimulation.py**: Expecting 0-indexed systems (0-5)

## Root Cause
Looking at `hierarchical_cosimulation.py`, the example attack scenarios use:
```python
attack_scenarios = [
    {'target_system': 0, ...},  # System 0 (first system)
    {'target_system': 2, ...},  # System 2 (third system)  
    {'target_system': 4, ...}   # System 4 (fifth system)
]
```

But our code was generating:
```python
'target_system': (attack_count % 6) + 1,  # Generated 1-6 instead of 0-5
```

## Fixes Applied

### 1. **Attack Scenario Generation**
```python
# BEFORE (1-indexed)
'target_system': (attack_count % 6) + 1,

# AFTER (0-indexed)  
'target_system': attack_count % 6,  # 0-indexed for hierarchical_cosimulation
```

### 2. **System States Generation**
```python
# BEFORE (1-indexed)
for sys_id in range(1, 7):

# AFTER (0-indexed)
for sys_id in range(6):  # 0-indexed for hierarchical_cosimulation
```

### 3. **Federated Attack Conversion**
```python
# BEFORE
'target_system': attack['system_id'],

# AFTER
'target_system': attack['system_id'] - 1,  # Convert to 0-indexed for hierarchical_cosimulation
```

### 4. **Electrical Attack Application**
```python
def apply_electrical_attack_to_cosimulation(cosim, attack_scenario, current_time):
    target_system = attack_scenario['target_system']
    # Convert 0-indexed attack target to 1-indexed distribution_systems key
    system_key = target_system + 1 if target_system in range(6) else target_system
    
    # Use system_key for accessing cosim.distribution_systems
    if system_key in cosim.distribution_systems:
        dist_system = cosim.distribution_systems[system_key]['system']
```

### 5. **Bus and Line Naming**
```python
# BEFORE
'target_bus': f"Bus_{(attack_count % 10) + 1}",
'target_line': f"Line_{(attack_count % 15) + 1}",

# AFTER  
'target_bus': f"Bus_{(attack_count % 10)}",  # 0-indexed bus names
'target_line': f"Line_{(attack_count % 15)}",  # 0-indexed line names
```

## Key Changes Summary

| Component | Before | After | Reason |
|-----------|--------|--------|---------|
| Attack target_system | 1-6 | 0-5 | Match hierarchical_cosimulation expectations |
| System states loop | range(1,7) | range(6) | Generate 0-indexed system IDs |
| Bus/Line names | Bus_1, Line_1 | Bus_0, Line_0 | Consistent 0-indexed naming |
| Distribution system access | target_system | target_system + 1 | Convert for 1-indexed distribution_systems dict |

## Verification Points

### ✅ **Attack Scenarios Now Generate:**
- target_system: 0, 1, 2, 3, 4, 5 (instead of 1, 2, 3, 4, 5, 6)
- Bus names: Bus_0, Bus_1, etc. (instead of Bus_1, Bus_2, etc.)
- Line names: Line_0, Line_1, etc. (instead of Line_1, Line_2, etc.)

### ✅ **System Access:**
- Attack target_system=0 → Access cosim.distribution_systems[1]
- Attack target_system=1 → Access cosim.distribution_systems[2]
- Attack target_system=5 → Access cosim.distribution_systems[6]

### ✅ **Debug Information Added:**
```
- 🔧 Using 0-indexed target systems (0-5) for hierarchical_cosimulation compatibility
```

## Expected Resolution
This indexing fix should resolve the "Unexpected error: 0" by ensuring that:
1. Attack scenarios target valid systems (0-5)
2. System access uses correct keys (1-6 for distribution_systems dict)
3. All electrical parameter attacks are applied to existing systems
4. Co-simulation can properly handle the attack scenarios

The error was likely occurring because the co-simulation framework couldn't find distribution systems with IDs 1-6 when it was expecting 0-5, or vice versa, leading to a division by zero or null reference error.
