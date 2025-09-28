#!/usr/bin/env python3
"""
Attack Magnitude Configuration
Easily adjust attack magnitudes for higher deviations
"""

# 🎯 ATTACK MAGNITUDE MULTIPLIERS
# Increased for massive impacts with 25MW system loads

# 1. ELECTRICAL PARAMETER AMPLIFICATION
ELECTRICAL_AMPLIFICATION = {
    'voltage_injection': 200,     # Was 50 - 200x amplification for voltage
    'current_injection': 200,     # Was 50 - 200x amplification for current  
    'frequency_injection': 200,   # Was 50 - 200x amplification for frequency
    'reactive_power_injection': 100,  # Was 25 - 100x amplification for reactive power
    'power_factor_target': 0.5,   # Target power factor (0.5 = 50% - more extreme)
}

# 2. DEMAND FACTOR SCALING
DEMAND_SCALING = {
    'demand_increase': 20.0,      # Was 5.0 - 20x scaling for demand increase
    'demand_decrease': 15.0,      # Was 3.0 - 15x scaling for demand decrease
    'oscillating_demand': 15.0,   # Was 3.0 - 15x scaling for oscillating demand
}

# 3. ATTACK SCENARIO BASE MAGNITUDES
SCENARIO_MAGNITUDES = {
    'demand_increase_base': 100.0,  # Was 20.0 - base magnitude for demand increase
    'demand_increase_increment': 10.0,  # Was 2.0 - increment per scenario
    'demand_decrease_base': 75.0,   # Was 15.0 - base magnitude for demand decrease
    'demand_decrease_increment': 10.0,  # Was 2.0 - increment per scenario
}

# 4. RL ATTACK IMPACT SCALING
RL_IMPACT_SCALING = {
    'magnitude_divisor': 10.0,    # Was 25.0 - lower = higher impact
    'max_impact': 0.95,          # Was 0.9 - maximum impact factor
    'power_efficiency_reduction': 5.0,  # Was 2.0 - power efficiency impact
    'voltage_efficiency_loss': 5.0,    # Was 2.0 - voltage efficiency impact
    'frequency_efficiency_loss': 5.0,  # Was 2.0 - frequency efficiency impact
}

# 5. VOLTAGE DROP FACTORS
VOLTAGE_DROP_FACTORS = {
    'power_manipulation': 0.5,    # Was 0.2 - voltage drop during power attacks
    'voltage_manipulation': 0.6,  # Was 0.3 - voltage drop during voltage attacks
    'load_manipulation': 0.7,     # Was 0.4 - voltage drop during load attacks
    'demand_attacks': 0.6,        # Was 0.25 - voltage drop during demand attacks
    'frequency_attacks': 0.4,     # Was 0.15 - voltage drop during frequency attacks
}

# 6. CHARGING TIME IMPACT
CHARGING_TIME_IMPACT = {
    'demand_increase': 5.0,       # Was 2.0 - charging time factor for demand increase
    'demand_decrease': 3.0,       # Was 1.5 - charging time factor for demand decrease
    'ramped_attacks': 5.0,        # Was 2.0 - charging time factor for ramped attacks
}

# 7. STEALTH VS IMPACT TRADE-OFF
STEALTH_IMPACT_TRADE_OFF = {
    'stealth_level': 0.1,         # Was 0.2 - lower = higher impact, less stealth
    'amplification_factor': 200,   # Was 50 - overall amplification factor
    'system_exploitation': True,   # Was True - enable system exploitation
}

def get_high_impact_config():
    """Get configuration for maximum attack impact"""
    return {
        'electrical': ELECTRICAL_AMPLIFICATION,
        'demand': DEMAND_SCALING,
        'scenarios': SCENARIO_MAGNITUDES,
        'rl_impact': RL_IMPACT_SCALING,
        'voltage_drop': VOLTAGE_DROP_FACTORS,
        'charging_time': CHARGING_TIME_IMPACT,
        'stealth': STEALTH_IMPACT_TRADE_OFF
    }

def get_moderate_impact_config():
    """Get configuration for moderate attack impact"""
    return {
        'electrical': {k: v // 2 for k, v in ELECTRICAL_AMPLIFICATION.items()},
        'demand': {k: v // 2 for k, v in DEMAND_SCALING.items()},
        'scenarios': {k: v // 2 for k, v in SCENARIO_MAGNITUDES.items()},
        'rl_impact': {k: v // 2 if isinstance(v, (int, float)) else v for k, v in RL_IMPACT_SCALING.items()},
        'voltage_drop': {k: v // 2 for k, v in VOLTAGE_DROP_FACTORS.items()},
        'charging_time': {k: v // 2 for k, v in CHARGING_TIME_IMPACT.items()},
        'stealth': {k: v if k == 'system_exploitation' else v * 2 for k, v in STEALTH_IMPACT_TRADE_OFF.items()}
    }

def print_current_config():
    """Print current attack magnitude configuration"""
    print("🔥 CURRENT ATTACK MAGNITUDE CONFIGURATION:")
    print("=" * 60)
    
    config = get_high_impact_config()
    
    print(f"⚡ Electrical Amplification:")
    for param, value in config['electrical'].items():
        print(f"   {param}: {value}x")
    
    print(f"\n📊 Demand Scaling:")
    for param, value in config['demand'].items():
        print(f"   {param}: {value}x")
    
    print(f"\n🎯 Scenario Magnitudes:")
    for param, value in config['scenarios'].items():
        print(f"   {param}: {value}")
    
    print(f"\n🤖 RL Impact Scaling:")
    for param, value in config['rl_impact'].items():
        print(f"   {param}: {value}")
    
    print(f"\n⚡ Voltage Drop Factors:")
    for param, value in config['voltage_drop'].items():
        print(f"   {param}: {value}")
    
    print(f"\n⏱️ Charging Time Impact:")
    for param, value in config['charging_time'].items():
        print(f"   {param}: {value}x")
    
    print(f"\n🕵️ Stealth vs Impact:")
    for param, value in config['stealth'].items():
        print(f"   {param}: {value}")

if __name__ == "__main__":
    print_current_config()
    
    print("\n" + "=" * 60)
    print("💡 TO INCREASE ATTACK DEVIATIONS:")
    print("1. Increase ELECTRICAL_AMPLIFICATION values")
    print("2. Increase DEMAND_SCALING values") 
    print("3. Increase SCENARIO_MAGNITUDES values")
    print("4. Decrease RL_IMPACT_SCALING['magnitude_divisor']")
    print("5. Increase VOLTAGE_DROP_FACTORS values")
    print("6. Decrease STEALTH_IMPACT_TRADE_OFF['stealth_level']")
    print("=" * 60)
