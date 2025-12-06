import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import json
import traceback
import math


import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC, DQN

from DiscreteHybridEnv import DiscreteHybridEnv
from combined_pinn import CompetingHybridEnv

import sys
import os
import tempfile
import json
import traceback

import time
from datetime import timedelta



'''
if you train this with longer amount of episode , the defender will win most of the time. 
But if you train it with less amount of episode more than 5 or 10 then the attacker will win most of the time.
'''


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
torch.set_num_threads(1)

from datetime import datetime  # Change this line

# ... existing code ...
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 

log_file = f"logs/training_log_{current_time}.txt"

# Create a custom logger class
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and file
sys.stdout = Logger(log_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# System Constants
NUM_BUSES = 33
NUM_EVCS = 5

# Base Values
S_BASE = 10e6      # VA
V_BASE_HV_AC = 12660  # V
V_BASE_LV_AC = 480   # V

V_BASE_DC = 800    # V
V_OUT_EVCS = 550
I_OUT_EVCS = 100

ATTACK_WEIGHT = 1.0

modulation_index_system= V_OUT_EVCS/V_BASE_DC

m_buck_vdc_to_v_out = 550/800

# Calculate base currents
I_BASE_HV_AC = S_BASE / (torch.sqrt(torch.tensor(3)) * V_BASE_HV_AC)  # HV AC base current
I_BASE_LV_AC = S_BASE / (torch.sqrt(torch.tensor(3)) * V_BASE_LV_AC)  # LV AC base current
I_BASE_DC = S_BASE / V_BASE_DC                       # DC base current

I_BASE_EVCS = I_OUT_EVCS / I_OUT_EVCS

# Calculate base impedances
Z_BASE_HV_AC = V_BASE_HV_AC**2 / S_BASE              # HV AC base impedance
Z_BASE_LV_AC = V_BASE_LV_AC**2 / S_BASE              # LV AC base impedance
Z_BASE_DC = V_BASE_DC**2 / S_BASE                    # DC base impedance

# EVCS Parameters
EVCS_POWER = 50    # EVCS power rating in kW
EVCS_POWER_PU = EVCS_POWER * 1000 / S_BASE     # Convert kW to p.u.
EVCS_CAPACITY = EVCS_POWER * 1000 / S_BASE     # EVCS capacity in p.u. (same as power rating)
EVCS_EFFICIENCY = 0.98                         # EVCS conversion efficiency
EVCS_VOLTAGE = V_OUT_EVCS / V_BASE_DC          # Nominal voltage ratio

# Voltage Limits
MAX_VOLTAGE_PU = 1.05               # Maximum allowable voltage in per unit (reduced from 1.2)
MIN_VOLTAGE_PU = 0.95               # Minimum allowable voltage in per unit (increased from 0.8)
VOLTAGE_VIOLATION_PENALTY = 1000.0  # Penalty for voltage violations


V_OUT_NOMINAL = EVCS_VOLTAGE  # Nominal output voltage in p.u.
V_OUT_VARIATION = 0.1  # 5% allowed variation (reduced from 10%)

MAX_CURRENT = 100.0/I_BASE_DC

# Controller Parameters (in p.u.)
EVCS_PLL_KP = 0.1
EVCS_PLL_KI = 0.5
MAX_PLL_ERROR = 10.0

EVCS_OUTER_KP = 1
EVCS_OUTER_KI = 0.5

EVCS_INNER_KP = 1
EVCS_INNER_KI = 0.5
OMEGA_N = 2 * torch.pi * 60         # Nominal angular frequency (60 Hz)

# Wide Area Controller Parameters
# WAC_KP_VDC = 1.0
    # WAC_KI_VDC = 0.5

# WAC_KP_VOUT =[1, 1, 1, 1, 1]
# WAC_KI_VOUT =[0.5, 0.5, 0.5, 0.5, 0.5]

# WAC_KP_VOUT = torch.tensor(WAC_KP_VOUT)
# WAC_KI_VOUT = torch.tensor(WAC_KI_VOUT)


VARS_PER_BUS = 2  # voltage magnitude and angle
VARS_PER_EVCS = 2  # v_out and i_out
START_EVCS_VARS = NUM_BUSES * VARS_PER_BUS  # Start after bus variables



PWM_SWITCHING_FREQ = 0.2  # 10kHz switching frequency
SQRT_2 = torch.sqrt(torch.tensor(2.0))
PI = torch.tensor(math.pi)

evcs_count = [1, 1, 1, 1, 1]
evcs_count = torch.tensor(evcs_count, dtype=torch.float32)




# WAC_KP_VOUT = 1.0
# WAC_KI_VOUT = 0.5

# WAC_KP_VOUT =[1, 1, 1, 1, 1]
# WAC_KI_VOUT =[0.5, 0.5, 0.5, 0.5, 0.5]

WAC_DC_LINK_VOLTAGE_SETPOINT = V_BASE_DC / V_BASE_DC  # Desired DC voltage in p.u.

v_out_ref= WAC_VOUT_SETPOINT = V_OUT_EVCS / V_BASE_DC     # Desired output voltage in p.u.

# Circuit Parameters (convert to p.u.)
CONSTRAINT_WEIGHT = 1.0
LCL_L1 = 0.01/ Z_BASE_LV_AC     # LCL filter inductor 1
LCL_L2 = 0.02 / Z_BASE_LV_AC     # LCL filter inductor 2
LCL_CF = 0.02 * S_BASE / (V_BASE_LV_AC**2)  # LCL filter capacitor
R      = 0.02/ Z_BASE_LV_AC           # Resistance
C_dc = 0.01 * S_BASE / (V_BASE_DC**2)  # DC-link capacitor (modified to use DC base)
L_dc = 0.001 / Z_BASE_DC      # DC inductor (modified to use DC base)
v_battery = 800 / V_BASE_DC   # Battery voltage in p.u.
R_battery = 0.01 / Z_BASE_DC   # Battery resistance (modified to use DC base)

# Time parameters
TIME_STEP = 1/(PWM_SWITCHING_FREQ*10)/2  # 1 ms
TOTAL_TIME = 10000*TIME_STEP  # 100 seconds

POWER_BALANCE_WEIGHT = 1.0
RATE_OF_CHANGE_LIMIT = 0.05  # Maximum 5% change per time step
VOLTAGE_STABILITY_WEIGHT = 2.0
POWER_FLOW_WEIGHT = 1.5
MIN_VOLTAGE_LIMIT = 0.85  # Minimum allowable voltage
THERMAL_LIMIT_WEIGHT = 1.0
COORDINATION_WEIGHT = 0.8  # Weight for coordinated attack impact


# Add these constants to your parameters section
WAC_KP_BATT = 0.1  # Gain for modulation index adjustment
WAC_KI_BATT = 0.2  # Gain for current setpoint adjustment
I_BAT_NOMINAL = 1.0  # Nominal battery charging current in p.u.
M0 = 0.8  # Initial modulation index



EVCS_BUSES = [7, 18, 10, 25, 30]           # Location of EVCS units

# Load IEEE 33-bus system data
line_data = [
    (1, 2, 0.0922, 0.0477), (2, 3, 0.493, 0.2511), (3, 4, 0.366, 0.1864), (4, 5, 0.3811, 0.1941),
    (5, 6, 0.819, 0.707), (6, 7, 0.1872, 0.6188), (7, 8, 1.7114, 1.2351), (8, 9, 1.03, 0.74),
    (9, 10, 1.04, 0.74), (10, 11, 0.1966, 0.065), (11, 12, 0.3744, 0.1238), (12, 13, 1.468, 1.155),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.591, 0.526), (15, 16, 0.7463, 0.545), (16, 17, 1.289, 1.721),
    (17, 18, 0.732, 0.574), (2, 19, 0.164, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.898, 0.7091), (24, 25, 0.896, 0.7011),
    (6, 26, 0.203, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.059, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.963), (31, 32, 0.31, 0.3619), (32, 33, 0.341, 0.5302)
]


# Add this before creating Y_bus
def convert_line_impedance_to_pu(r, x, z_base):
    """Convert line impedance to per unit."""
    return r / z_base, x / z_base


# Convert line impedances to p.u.
# line_data_pu = []
# for line in line_data:
    # from_bus, to_bus, r, x = line
    # r_pu, x_pu = convert_line_impedance_to_pu(r, x, Z_BASE_HV_AC)
    # line_data_pu.append((from_bus, to_bus, r_pu, x_pu))

line_data = torch.tensor(line_data, dtype=torch.float32)

bus_data = np.array([
    [1, 0, 0, 0], [2, 100, 60, 0], [3, 70, 40, 0], [4, 120, 80, 0], [5, 80, 30, 0],
    [6, 60, 20, 0], [7, 145, 100, 0], [8, 160, 100, 0], [9, 160, 20, 0], [10, 160, 120, 0],
    [11, 100, 30, 0], [12, 160, 35, 0], [13, 60, 35, 0], [14, 80, 80, 0], [15, 100, 10, 0],
    [16, 100, 20, 0], [17, 60, 20, 0], [18, 90, 40, 0], [19, 90, 40, 0], [20, 90, 40, 0],
    [21, 90, 40, 0], [22, 90, 40, 0], [23, 90, 40, 0], [24, 420, 200, 0], [25, 380, 200, 0],
    [26, 100, 25, 0], [27, 60, 25, 0], [28, 60, 20, 0], [29, 120, 70, 0], [30, 200, 600, 0],
    [31, 150, 70, 0], [32, 210, 100, 0], [33, 60, 40, 0]
])

# line_data = torch.tensor(line_data, dtype=torch.float32)
bus_data = torch.tensor(bus_data, dtype=torch.float32)

# Convert bus data to per-unit
bus_data[:, 1:3] = bus_data[:, 1:3] * 1e3 / S_BASE

# Initialize Y-bus matrix
Y_bus = torch.zeros((NUM_BUSES, NUM_BUSES), dtype=torch.complex64)



# Fill Y-bus matrix
for line in line_data:
    from_bus, to_bus, r, x = line
    from_bus, to_bus = int(from_bus)-1 , int(to_bus)-1  # Convert to 0-based index
    y = 1/complex(r, x)  # Creates complex admittance
    Y_bus[from_bus, from_bus] += y
    Y_bus[to_bus, to_bus] += y
    Y_bus[from_bus, to_bus] -= y
    Y_bus[to_bus, from_bus] -= y

# Convert to TensorFlow constant
if isinstance(Y_bus, torch.Tensor):
    Y_bus_tf = Y_bus.clone().detach().to(dtype=torch.complex64)
else:
    Y_bus_tf = torch.tensor(Y_bus, dtype=torch.complex64)


G_d = None
G_q = None

def initialize_conductance_matrices():
    """Initialize conductance matrices from Y-bus matrix"""
    global G_d, G_q, B_d, B_q
    # Extract G (conductance) and B (susceptance) matrices
    G_d = torch.real(Y_bus_tf)  # Real part for d-axis
    G_q = torch.real(Y_bus_tf)  # Real part for q-axis
    B_d = torch.imag(Y_bus_tf)  # Imaginary part for d-axis
    B_q = torch.imag(Y_bus_tf)  # Imaginary part for q-axis
    return G_d, G_q, B_d, B_q

# Call this function before training starts
G_d, G_q, B_d, B_q = initialize_conductance_matrices()

# For individual elements (if needed)
G_d_kh = torch.diag(G_d)  # Diagonal elements for d-axis conductance
G_q_kh = torch.diag(G_q)  # Diagonal elements for q-axis conductance
B_d_kh = torch.diag(B_d)  # Diagonal elements for d-axis susceptance
B_q_kh = torch.diag(B_q)  # Diagonal elements for q-axis susceptance



class SACWrapper(gym.Env):
    def __init__(self, env, agent_type, dqn_agent=None, sac_defender=None, sac_attacker=None):
        super(SACWrapper, self).__init__()
        
        self.env = env
        self.agent_type = agent_type
        self.dqn_agent = dqn_agent
        self.sac_defender = sac_defender
        self.sac_attacker = sac_attacker
        self.NUM_EVCS = env.NUM_EVCS
        self.TIME_STEP = env.TIME_STEP  # Add this line to fix the missing attribute
        
        # Initialize tracking variables
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, dtype=torch.float32)
        self.cumulative_deviation = 0.0
        self.attack_active = False
        self.target_evcs = torch.zeros(self.NUM_EVCS)
        self.attack_duration = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rewards = 0.0
        
        # Define observation and action spaces using numpy dtypes
        self.observation_space = env.observation_space
        if agent_type == 'attacker':
            self.action_space = gym.spaces.Box(
                low=np.float32(-0.1),  # Changed from torch.tensor to np.float32
                high=np.float32(0.1),  # Changed from torch.tensor to np.float32
                shape=(self.NUM_EVCS * 2,),
                dtype=np.float32  # Changed from torch.float32 to np.float32
            )
        else:  # defender
            self.action_space = gym.spaces.Box(
                low=np.float32(-0.1),  # Changed from torch.tensor to np.float32
                high=np.float32(0.1),  # Changed from torch.tensor to np.float32
                shape=(self.NUM_EVCS * 2,),
                dtype=np.float32  # Changed from torch.float32 to np.float32
            )
        
        # Initialize state
        self.state = None
        self.reset()

    def step(self, action):
        try:
            # Convert input action to proper shape tensor
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif isinstance(action, list):
                action = torch.tensor(action, dtype=torch.float32)
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor([action], dtype=torch.float32)
            
            # Use coordinated DQN/SAC attack system if available
            if hasattr(self, 'dqn_sac_trainer') and self.dqn_sac_trainer:
                # Get baseline outputs for current station
                baseline_outputs = {
                    'power_reference': float(action[0]) if len(action) > 0 else 10.0,
                    'voltage_reference': float(action[1]) if len(action) > 1 else 400.0,
                    'current_reference': float(action[2]) if len(action) > 2 else 25.0,
                    'soc': 0.5,
                    'grid_voltage': 1.0,
                    'grid_frequency': 60.0,
                    'demand_factor': 1.0,
                    'urgency_factor': 1.0,
                    'voltage_priority': 0.0
                }
                
                # Get coordinated attack from DQN/SAC system
                current_station = getattr(self, 'current_station', 0)
                coordinated_attack = self.dqn_sac_trainer.get_coordinated_attack(
                    current_station, baseline_outputs
                )
                
                if coordinated_attack:
                    combined_action = {
                        'coordinated_attack': coordinated_attack,
                        'baseline_action': action.numpy() if isinstance(action, torch.Tensor) else action
                    }
                else:
                    # Fallback to legacy system
                    combined_action = {'legacy_action': action.numpy() if isinstance(action, torch.Tensor) else action}
            else:
                # Legacy DQN/SAC processing
                dqn_state = torch.as_tensor(self.state, dtype=torch.float32).reshape(1, -1)
                dqn_raw = self.dqn_agent.predict(dqn_state.numpy(), deterministic=True) if self.dqn_agent else [0]
                dqn_action = torch.tensor(dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw, dtype=torch.int64)
                
                # Process actions based on agent type
                if self.agent_type == 'attacker':
                    attacker_action = action
                    if self.sac_defender is not None:
                        defender_state = torch.as_tensor(self.state, dtype=torch.float32)
                        defender_raw = self.sac_defender.predict(defender_state.numpy(), deterministic=True)
                        defender_action = torch.tensor(defender_raw[0], dtype=torch.float32)
                    else:
                        defender_action = torch.zeros(self.NUM_EVCS * 2, dtype=torch.float32)
                else:  # defender
                    defender_action = action
                    if self.sac_attacker is not None:
                        attacker_state = torch.as_tensor(self.state, dtype=torch.float32)
                        attacker_raw = self.sac_attacker.predict(attacker_state.numpy(), deterministic=True)
                        attacker_action = torch.tensor(attacker_raw[0], dtype=torch.float32)
                    else:
                        attacker_action = torch.zeros(self.NUM_EVCS * 2, dtype=torch.float32)
                
                # Combine actions into dictionary
                combined_action = {
                    'dqn': dqn_action.numpy(),
                    'attacker': attacker_action.numpy(),
                    'defender': defender_action.numpy()
                }
            
            # Take step in environment
            next_state, rewards, done, truncated, info = self.env.step(combined_action)

            # Handle rewards based on agent type
            if isinstance(rewards, dict):
                # Extract reward for current agent type
                reward = float(rewards.get(self.agent_type, 0.0))
            elif isinstance(rewards, (int, float)):
                reward = float(rewards)
            else:
                reward = 0.0  # Default reward if invalid type
                print(f"Warning: Unexpected reward type: {type(rewards)}")

            # Convert state to numpy array for return
            if isinstance(next_state, torch.Tensor):
                state_np = next_state.detach().numpy()
            else:
                state_np = np.array(next_state, dtype=np.float32)

            return state_np, reward, bool(done), bool(truncated), dict(info)

        except Exception as e:
            print(f"Error in SACWrapper step: {str(e)}")
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,  # End episode on error
                False,
                {'error': str(e)}
            )

    def reset(self, seed=None, options=None):
        try:
            # Reset tracking variables
            self.voltage_deviations = torch.zeros(self.NUM_EVCS, dtype=torch.float32)
            self.cumulative_deviation = 0.0
            self.attack_active = False
            self.target_evcs = torch.zeros(self.NUM_EVCS)
            self.attack_duration = 0.0
            self.rewards = 0.0
            # Reset environment
            obs_info = self.env.reset(seed=seed)
            
            # Handle different return types
            if isinstance(obs_info, tuple):
                obs, info = obs_info
            else:
                obs = obs_info
                info = {}
            
            # Convert observation to proper format
            if isinstance(obs, np.ndarray):
                self.state = torch.from_numpy(obs).float()
            else:
                self.state = torch.tensor(obs, dtype=torch.float32)
            
            # Return numpy array for SB3 compatibility
            return self.state.numpy(), dict(info)
            
        except Exception as e:
            print(f"Error in SACWrapper reset: {str(e)}")
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                {'error': str(e)}
            )

    def update_agents(self, dqn_agent= None, sac_defender=None, sac_attacker= None):
        """Update the agents used by the wrapper."""
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
            print("Updated DQN agent")
        if sac_defender is not None:
            self.sac_defender = sac_defender
            print("Updated SAC defender")
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
            print("Updated SAC attacker")

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh()):
        super().__init__()
        
        # Wider intermediate layers with skip connections
        self.block1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            activation,
            nn.LayerNorm(dim * 4),  # Changed from BatchNorm1d to LayerNorm
            nn.Linear(dim * 4, dim),
            activation,
            nn.LayerNorm(dim)  # Changed from BatchNorm1d to LayerNorm
        )
        
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.activation = activation
        
    def forward(self, x):
        identity = x
        # Change this line from self.block to self.block1
        x = self.block1(x)
        return self.activation(x + self.skip_scale * identity)

class EVCS_PowerSystem_PINN(nn.Module):
    def __init__(self, num_buses=NUM_BUSES, num_evcs=NUM_EVCS, hidden_dim=1024, num_blocks=6):
        super(EVCS_PowerSystem_PINN, self).__init__()
        
        self.num_buses = num_buses
        self.num_evcs = num_evcs
        self.hidden_dim = hidden_dim
        
        # Calculate output dimension
        self.output_dim = num_buses * 3 + num_evcs * 18
        
        # Changed BatchNorm to LayerNorm in input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),  # Changed from BatchNorm1d
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)  # Changed from BatchNorm1d
        )
        
        # Residual blocks with dropout
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_dim, activation=nn.SiLU()),
                nn.Dropout(0.1 + i * 0.05)  # Progressive dropout
            ) for i in range(num_blocks)
        ])
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Gating layer for skip connection
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Improved output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )
        
        # Learnable scaling factors
        self.voltage_scale = nn.Parameter(torch.ones(1))
        self.angle_scale = nn.Parameter(torch.ones(1))
        self.evcs_scale = nn.Parameter(torch.ones(1))
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Enhanced initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        self.apply(init_weights)
    
    def forward(self, t):
        # Input processing
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        
        t = t.to(self.device)
        
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        
        # Initial embedding
        x = self.input_embedding(t)
        initial_features = x
        
        # Process through residual blocks with gradient checkpointing
        residual_sum = 0
        for i, block in enumerate(self.res_blocks):
            x = torch.utils.checkpoint.checkpoint(block, x) if self.training else block(x)
            residual_sum = residual_sum + x * (1.0 / (i + 1))  # Weighted residual accumulation
        
        # Apply attention mechanismwac_integral_i_bat
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        
        # Add skip connection with gating
        gate = torch.sigmoid(self.gate_layer(x))
        x = gate * x + (1 - gate) + initial_features
        
        # Output processing with scaling
        output = self.output_network(x)
        
        # Split and process with learned scaling
        voltage_magnitude = self.voltage_scale * F.softplus(output[:, :self.num_buses])
        voltage_angle = self.angle_scale * torch.tanh(output[:, self.num_buses:2*self.num_buses]) * math.pi
        evcs_outputs = self.evcs_scale * torch.tanh(output[:, 2*self.num_buses:])
        
        return torch.cat([
            voltage_magnitude,
            voltage_angle,
            evcs_outputs
        ], dim=1)

    def get_state(self, t):
        outputs = self.forward(t)
        
        v_d = outputs[:, :self.num_buses]
        v_q = outputs[:, self.num_buses:2*self.num_buses]
        evcs_vars = outputs[:, 2*self.num_buses:]
        
        state = torch.cat([
            v_d,
            v_q,
            torch.sqrt(v_d**2 + v_q**2),
            evcs_vars
        ], dim=1)
        
        return state
    
    @property
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_gradient_norm(self):
        total_norm = 0
        for p in self.trainable_parameters:
            if p.grad is not None:
                total_norm =total_norm+ p.grad.data.norm(2).item() ** 2
        return torch.sqrt(torch.tensor(total_norm))

def get_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )

def get_scheduler(optimizer, num_epochs):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=num_epochs,
        steps_per_epoch=1,
        pct_start=0.3,
        anneal_strategy='cos'
    )

def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    

class SafeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        # Return x where finite, small value where not
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x) + 1e-30)

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved input
        x, = ctx.saved_tensors
        # Return gradient where finite, zero where not
        return torch.where(torch.isfinite(grad_output), grad_output, torch.zeros_like(grad_output) + 1e-30)
    
def safe_op(x):
    """Safely perform tensor operations with proper gradient handling."""
    return SafeOp.apply(x)

def calculate_gradient(x, spacing, eps=1e-3):
    """Calculate gradient using finite differences with proper handling of small tensors."""
    try:
        # Ensure input is a tensor and detach from computation graph
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Create a new tensor instead of modifying in place
        x_padded = x.clone().detach()
        
        # Add batch dimension if needed
        if len(x_padded.shape) == 1:
            x_padded = x_padded.unsqueeze(0)
        
        # If tensor is too small, pad it safely
        if x_padded.shape[0] < 2:
            x_padded = torch.cat([x_padded, x_padded.clone()], dim=0)
        
        # Calculate gradient using forward difference with numerical stability
        diff = x_padded[1:] - x_padded[:-1]
        spacing_tensor = torch.tensor(spacing, dtype=torch.float32) + eps
        grad = diff / spacing_tensor
        
        # Handle NaN values
        grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
                # If original input was single value, return single value gradient
        if len(grad) == 0:
            return torch.zeros_like(x_padded[0])
        
        return grad.clone()  # Return a clone to prevent in-place modifications
        
    except Exception as e:
        print(f"Error in calculate_gradient: {e}")
        return torch.zeros_like(x[0] if len(x) > 0 else x)
    

def safe_matrix_operations(func):
    """Decorator for safe matrix operations with logging"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Handle tuple return type properly
            if isinstance(result, tuple):
                nan_check = any(torch.isnan(r).any() for r in result if isinstance(r, torch.Tensor))
                if nan_check:
                    print(f"Warning: NaN detected in {func.__name__}")
                    print(f"Input shapes: {[arg.shape if isinstance(arg, torch.Tensor) else None for arg in args]}")
                    return tuple(torch.zeros_like(r) if isinstance(r, torch.Tensor) else r for r in result)
                return result
            else:
                if torch.isnan(result).any():
                    print(f"Warning: NaN detected in {func.__name__}")
                    print(f"Input shapes: {[arg.shape if isinstance(arg, torch.Tensor) else None for arg in args]}")
                    return torch.zeros_like(result)
                return result
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            # Get shapes from first argument assuming it's a tensor
            if isinstance(args[0], torch.Tensor):
                batch_size = args[0].shape[0]
                num_buses = args[0].shape[1]
                return (torch.zeros(batch_size, num_buses), 
                       torch.zeros(batch_size, num_buses), 
                       {})
            else:
                raise ValueError("First argument must be a tensor")
    return wrapper

def calculate_power_flow_base(v_d, v_q, G, B, mask, eps=1e-6):
    """Base power flow calculation with numerical stability."""
    try:
        # Ensure proper shapes for matrix multiplication
        # Reshape v_d and v_q to be [batch_size, num_buses]
        if len(v_d.shape) == 1:
            v_d = v_d.unsqueeze(0)  # Add batch dimension
        if len(v_q.shape) == 1:
            v_q = v_q.unsqueeze(0)  # Add batch dimension
            
        # Ensure G and B have correct shape [num_buses, num_buses]
        if len(G.shape) != 2:
            G = G.view(NUM_BUSES, NUM_BUSES)
        if len(B.shape) != 2:
            B = B.view(NUM_BUSES, NUM_BUSES)
            
        # Transpose v_d and v_q for matrix multiplication
        v_d_t = v_d.transpose(-1, -2)  # Shape: [batch_size, num_buses, 1]
        v_q_t = v_q.transpose(-1, -2)  # Shape: [batch_size, num_buses, 1]
        
        # Calculate currents with numerical stability
        i_d = torch.matmul(G, v_d_t).squeeze(-1) + torch.matmul(B, v_q_t).squeeze(-1)
        i_q = torch.matmul(G, v_q_t).squeeze(-1) - torch.matmul(B, v_d_t).squeeze(-1)
        
        # Safe magnitude calculations
        V = safe_sqrt(v_d**2 + v_q**2)
        I = safe_sqrt(i_d**2 + i_q**2)
        
        # Calculate power with masking
        P = (v_d * i_d + v_q * i_q) * mask
        Q = (v_q * i_d - v_d * i_q) * mask
        
        # Apply mask to other quantities
        V = V * mask
        I = I * mask
        theta = torch.atan2(v_q, v_d) * mask
        
        return {
            'P': P,
            'Q': Q,
            'V': V,
            'I': I,
            'theta': theta
        }
    except Exception as e:
        print(f"Error in power flow calculation: {e}")
        return {
            'P': torch.zeros_like(v_d),
            'Q': torch.zeros_like(v_d),
            'V': torch.zeros_like(v_d),
            'I': torch.zeros_like(v_d),
            'theta': torch.zeros_like(v_d)
        }

# Update the other power flow functions to use the new return format
def calculate_power_flow_pcc(v_d, v_q, G, B):
    """PCC power flow calculation."""
    num_buses = v_d.shape[-1]
    mask = torch.cat([torch.tensor([1.0]), torch.zeros(num_buses - 1)])
    mask = mask.unsqueeze(0)
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

def calculate_power_flow_load(v_d, v_q, G, B):
    """Load bus power flow calculation."""
    num_buses = v_d.shape[-1]
    mask = torch.ones(1, num_buses)
    mask[0, 0] = 0.0  # Zero out PCC bus
    for bus in EVCS_BUSES:
        mask[0, bus] = 0.0
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

def calculate_power_flow_ev(v_d, v_q, G, B):
    """EV bus power flow calculation."""
    num_buses = v_d.shape[-1]
    mask = torch.zeros(1, num_buses)
    for bus in EVCS_BUSES:
        mask[0, bus] = 1.0
    return calculate_power_flow_base(v_d, v_q, G, B, mask)



def safe_sqrt(x, eps=1e-8):
    """Safe sqrt operation that prevents NaN gradients"""
    return torch.sqrt(torch.clamp(x, min=eps))

def physics_loss(model, t, Y_bus_tf, bus_data, attack_actions, defend_actions):
    try:
        # Constants
        BATTERY_VOLTAGE_SETPOINT = 550.0/V_BASE_DC  # V
        TIME_STEP = 1/(PWM_SWITCHING_FREQ*5)/2  # Simulation time step
        CONTROL_INTERVAL = 0.2  # Control update interval
        i_ref_target = 100.0/I_BASE_DC
        
        # Initialize persistent variables if not already done
        if not hasattr(physics_loss, 'last_execution_time'):
            physics_loss.last_execution_time = 0.0
            physics_loss.prev_errors = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.integral_terms = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.integral_bat = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.Kp = torch.ones(NUM_EVCS, dtype=torch.float32) * 0.8
            physics_loss.Ki = torch.ones(NUM_EVCS, dtype=torch.float32) * 0.5
            physics_loss.prev_v_bat = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.last_m = torch.ones(NUM_EVCS, dtype=torch.float32) * 0.5  # Initialize with default value
            physics_loss.last_iref = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.i_ref = torch.zeros(NUM_EVCS, dtype=torch.float32)



        if not hasattr(physics_loss, 'i_integral'):
            physics_loss.i_integral = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.i_error = torch.zeros(NUM_EVCS, dtype=torch.float32)
            physics_loss.Kp_i = torch.ones(NUM_EVCS, dtype=torch.float32) * 0.8
            physics_loss.Ki_i = torch.ones(NUM_EVCS, dtype=torch.float32) * 0.5
            physics_loss.last_iref = torch.zeros(NUM_EVCS, dtype=torch.float32)

        # Initialize loss components
        power_flow_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        evcs_total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        wac_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        V_regulation_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)



        # Ensure t has correct shape [batch_size, 1]
        if isinstance(t, (int, float)):
            t = torch.tensor([[t]], dtype=torch.float32)
        elif isinstance(t, torch.Tensor):
            t = t.reshape(-1, 1)

        # Convert Y_bus_tf properly
        WAC_KP_VOUT = 1.0
        WAC_KI_VOUT = 0.5

        # Convert inputs properly
        if isinstance(t, torch.Tensor):
            t = t.clone().detach().reshape(-1, 1)
        else:
            t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        # Initialize variables
        wac_error_vout = torch.zeros_like(t)
        wac_integral_vout = torch.zeros_like(t)
        
        # Fix WAC control calculation
        # wac_control = (WAC_KP_VOUT * wac_eError processing EVCSrror_vout + WAC_KI_VOUT * wac_integral_vout).clone().detach()
        # modulation_index_vout = safe_op(torch.clamp(wac_control, min=0.0, max=1.0))

        # Ensure Y_bus_tf is complex
        if not Y_bus_tf.is_complex():
            Y_bus_tf = Y_bus_tf.to(dtype=torch.complex64)
    
        # Convert inputs to proper tensors
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        attack_actions = torch.tensor(attack_actions, dtype=torch.float32, requires_grad=True)
        defend_actions = torch.tensor(defend_actions, dtype=torch.float32, requires_grad=True)

        # ... existing code ...
        t = t.clone().detach().to(dtype=torch.float32).reshape(-1, 1)
        attack_actions = attack_actions.clone().detach().requires_grad_(True).to(dtype=torch.float32)
        defend_actions = defend_actions.clone().detach().requires_grad_(True).to(dtype=torch.float32)
# ... existing code ...



                  # Extract attack and defense actions
        fdi_voltage = attack_actions[:, :NUM_EVCS].reshape(-1, NUM_EVCS)
        fdi_current = attack_actions[:, NUM_EVCS:].reshape(-1, NUM_EVCS)
        
        KP = defend_actions[:, :NUM_EVCS].reshape(-1, NUM_EVCS)
        KI = defend_actions[:, NUM_EVCS:].reshape(-1, NUM_EVCS)
        
        # Extract real and imaginary parts of Y_bus
        G = Y_bus_tf.real.to(torch.float32)
        B = Y_bus_tf.imag.to(torch.float32)
        
        # Initialize loss components
        evcs_loss = []
        attack_loss = torch.tensor(0.0, dtype=torch.float32)
        voltage_violation_loss = torch.tensor(0.0, dtype=torch.float32)
        voltage_regulation_loss = torch.tensor(0.0, dtype=torch.float32)
  


        with torch.enable_grad():
                # Get predictions and ensure proper shapes
    # Ensure t has correct shape [batch_size, 1]
            if isinstance(t, (int, float)):
                t = torch.tensor([[t]], dtype=torch.float32)
            elif isinstance(t, torch.Tensor):
                if t.dim() == 0:
                    t = t.unsqueeze(0).unsqueeze(0)
                elif t.dim() == 1:
                    t = t.unsqueeze(1)
                
            # # Ensure attack_actions and defend_actions have correct shape [batch_size, num_actions]
            # if isinstance(attack_actions, torch.Tensor):
            #     if attack_actions.dim() == 1:
            #         attack_actions = attack_actions.unsqueeze(0)
            # else:
            #     attack_actions = torch.tensor(attack_actions, dtype=torch.float32).unsqueeze(0)
                
            # if isinstance(defend_actions, torch.Tensor):
            #     if defend_actions.dim() == 1:
            #         defend_actions = defend_actions.unsqueeze(0)
            # else:
            #     defend_actions = torch.tensor(defend_actions, dtype=torch.float32).unsqueeze(0)

            # Get predictions from model
            predictions = model(t)  # Shape: [batch_size, output_dim]
        
            
            # Extract predictions with explicit shapes
            V = safe_op(torch.exp(predictions[:, :NUM_BUSES]))  # [batch_size, NUM_BUSES]
            theta = safe_op(torch.atan(predictions[:, NUM_BUSES:2*NUM_BUSES]))  # [batch_size, NUM_BUSES]
            evcs_vars = predictions[:, 2*NUM_BUSES:]    
            
            # Calculate voltage components
            v_d = V * torch.cos(theta)
            v_q = V * torch.sin(theta)
            
            # NEW code:            pcc_results = calculate_power_flow_pcc(v_d, v_q, G, B)            load_results = calculate_power_flow_load(v_d, v_q, G, B)            ev_results = calculate_power_flow_ev(v_d, v_q, G, B)            # Access P and Q from the tuples (not dictionaries)            P_g_pcc, Q_g_pcc = pcc_results  # Unpack tuple directly            P_g_load, Q_g_load = load_results  # Unpack tuple directly            P_g_ev_load, Q_g_ev_load = ev_results  # Unpack tuple directly
            # Calculate power mismatches
                        # NEW code:
            pcc_results = calculate_power_flow_pcc(v_d, v_q, G, B)
            load_results = calculate_power_flow_load(v_d, v_q, G, B)
            ev_results = calculate_power_flow_ev(v_d, v_q, G, B)

            # Access P and Q from the dictionaries
            P_g_pcc = pcc_results['P']
            Q_g_pcc = pcc_results['Q']
            P_g_load = load_results['P']
            Q_g_load = load_results['Q']
            P_g_ev_load = ev_results['P']
            Q_g_ev_load = ev_results['Q']
            

            
            
            
            P_mismatch = P_g_pcc - (P_g_load + P_g_ev_load)
            Q_mismatch = Q_g_pcc - (Q_g_load + Q_g_ev_load)
            
            # Calculate power flow loss
            power_flow_loss = safe_op(torch.mean(torch.square(P_mismatch) + torch.square(Q_mismatch)))
            
            # Initialize EVCS losses list and WAC variables
            evcs_loss = []
            wac_error_vdc = torch.zeros_like(t)
            wac_integral_vdc = torch.zeros_like(t)
            wac_error_vout = torch.zeros_like(t)
            wac_integral_vout = torch.zeros_like(t)
            wac_error_i_bat = torch.zeros_like(t)
            wac_integral_i_bat = torch.zeros_like(t)

            power_flow_results = calculate_power_flow_ev(v_d, v_q, G, B)
            
            # Process each EVCS with proper indexing
            for i, bus in enumerate(EVCS_BUSES):
                try:
                    bus_voltage = power_flow_results['V'][:, bus:bus+1]
                    bus_current = power_flow_results['I'][:, bus:bus+1]
                    bus_theta = power_flow_results['theta'][:, bus:bus+1]

                    v_ac = bus_voltage * (V_BASE_LV_AC/ V_BASE_HV_AC)
                    i_ac = bus_current * (V_BASE_LV_AC/ V_BASE_HV_AC)
                    delta = bus_theta

                    # Ensure evcs_vars has proper batch dimension
                    evcs = evcs_vars[:, i*18:(i+1)*18]  # Shape should be [batch_size, 18]
                    
                    # Instead of using split, directly index the tensor
                    v_ac = v_ac 
                    i_ac = i_ac 

                    v_out = evcs[:, 2:3] # need to predict by model
                    i_out = evcs[:, 3:4] # need to predict by model

                    v_dc = evcs[:, 4:5] # need to predict by model
                    i_dc = evcs[:, 5:6] # need to predict by model

                    i_L1 = evcs[:, 6:7]
                    i_L2 = evcs[:, 7:8]
                    v_c = evcs[:, 8:9]
                    soc = evcs[:, 9:10]

                    delta = delta
                    omega = evcs[:, 11:12]
                    phi_d = evcs[:, 12:13]
                    phi_q = evcs[:, 13:14]

                    gamma_d = evcs[:, 14:15]
                    gamma_q = evcs[:, 15:16]

                    i_d = evcs[:, 16:17]
                    i_q = evcs[:, 17:18]

                    # Clarke and Park Transformations
                    v_ac = safe_op(v_ac)
                    i_ac = safe_op(i_ac)

                    v_dc = safe_op(v_dc)
                    i_dc = safe_op(i_dc)

                    v_out = safe_op(v_out)
                    i_out = safe_op(i_out)

                    i_L1 = safe_op(i_L1)
                    i_L2 = safe_op(i_L2)
                    v_c = safe_op(v_c)

                    #System Dynamics
                    v_alpha = v_ac
                    v_beta = torch.zeros_like(v_ac)

                    v_ac_mag = safe_sqrt(v_alpha**2 + v_beta**2)

                    v_dc_ref = 1
                    # Apply FDI attack
                    v_out_attacked = v_out 
                    i_out_attacked = i_out

                    v_out_attacked = v_out + fdi_voltage[:, i:i+1]
                    i_out_attacked = i_out + fdi_current[:, i:i+1]    

                    i_alpha = i_ac
                    i_beta = torch.zeros_like(i_ac)
                    
                    # v_d_evcs = safe_op(v_alpha * torch.cos(delta) + v_beta * torch.sin(delta))
                    # v_q_evcs = safe_op(-v_alpha * torch.sin(delta) + v_beta * torch.cos(delta))


                    i_d_evcs = safe_op(i_alpha * torch.cos(delta) + i_beta * torch.sin(delta))
                    i_q_evcs = safe_op(-i_alpha * torch.sin(delta) + i_beta * torch.cos(delta))

                                        # Converter Outer Loop
                    i_d_ref = safe_op(EVCS_OUTER_KP * (v_dc - v_dc_ref) + EVCS_OUTER_KI * gamma_d)
                    i_q_ref = safe_op(EVCS_OUTER_KP * (0 - v_q) + EVCS_OUTER_KI * gamma_q)

                    # Converter Inner Loop
                    v_d_ref = safe_op(EVCS_INNER_KP * (i_d_ref - i_d_evcs) + EVCS_INNER_KI * phi_d - omega * LCL_L1 * i_q_evcs + v_d)
                    v_q_ref = safe_op(EVCS_INNER_KP * (i_q_ref - i_q_evcs) + EVCS_INNER_KI * phi_q + omega * LCL_L1 * i_d_evcs + v_q)

                      # PLL Dynamics
                    v_q_normalized = torch.tanh(safe_op(v_q_ref))
                    pll_error = safe_op(EVCS_PLL_KP * v_q_normalized + EVCS_PLL_KI * phi_q)
                    pll_error = torch.clamp(torch.tensor(pll_error, dtype=torch.float32), torch.tensor(-MAX_PLL_ERROR, dtype=torch.float32), torch.tensor(MAX_PLL_ERROR, dtype=torch.float32))

                    v_ref_mag = safe_sqrt(v_d_ref**2 + v_q_ref**2)


                    WAC_KI = KI[:,i:i+1]
                    WAC_KP = KP[:,i:i+1]

                
                    # Get local copies of persistent variables at the start of EVCS processing
                    physics_loss.Kp[i] = WAC_KP
                    physics_loss.Ki[i] = WAC_KI
                    physics_loss.Kp_i[i] = WAC_KP
                    physics_loss.Ki_i[i] = WAC_KI
                    integral_bat= physics_loss.integral_bat[i].clone()
                    prev_errors = physics_loss.prev_errors[i].clone()
                    prev_v_bat = physics_loss.prev_v_bat[i].clone()
                    last_m = physics_loss.last_m[i].clone()
                    last_iref = physics_loss.last_iref[i].clone()
                    i_ref = physics_loss.i_ref[i].clone()

                    new_Kp = WAC_KP
                    new_Ki = WAC_KI
                    new_Kp_i = WAC_KP
                    new_Ki_i = WAC_KI


                    current_time = t.item() if isinstance(t, torch.Tensor) else t
                    time_elapsed = current_time - physics_loss.last_execution_time
                    update_needed = time_elapsed >= CONTROL_INTERVAL



                    if update_needed:
                        # Calculate voltage errors
                        error_bat = BATTERY_VOLTAGE_SETPOINT - v_out_attacked
                        i_error = i_ref_target - i_out_attacked  
                        
                        # Calculate error derivatives
                        d_error = (error_bat - prev_errors) / TIME_STEP
                        
                        # # Adaptive learning parameters
                        # alpha_p = 0.01  # Learning rate for Kp
                        # alpha_i = 0.005  # Learning rate for Ki
                        
                        # # Adapt controller gains
                        # new_Kp = adapt_gain(Kp, error_bat, d_error, alpha_p)
                        # new_Ki = adapt_gain(Ki, error_bat, integral_bat, alpha_i)

                        # new_Kp_i = adapt_gain(Kp_i, error_bat, d_error, alpha_p)
                        # new_Ki_i = adapt_gain(Ki_i, error_bat, integral_bat, alpha_i)
                        
                        # Calculate modulation index with adaptive PI control
                        m = new_Kp * error_bat + new_Ki * integral_bat
                        m = safe_op(m)
                        # Apply cross-coupling
                        other_indices = [j for j in range(NUM_EVCS) if j != i]
                        if other_indices:
                            cross_coupling = 0.1 * torch.exp(-0.1 * torch.abs(error_bat))
                            other_m = physics_loss.last_m[other_indices]
                            if len(other_m) > 0:  # Check if there are other EVCSs
                                m = m + cross_coupling * torch.mean(other_m)
                        
                        # Dynamic modulation index limiting
                        max_m = 0.25 * (1 + 0.1 * torch.exp(-torch.abs(error_bat)))
                        min_m = -max_m
                        m = torch.clamp(m, min_m, max_m)
                        
                        # Adaptive current reference update
                        i_ref = adaptive_current_ref(
                            i_ref,
                            v_out_attacked,
                            prev_v_bat,
                            i_out_attacked,
                            error_bat,
                            evcs_count[i],
                            MAX_CURRENT,
                            new_Kp_i,
                            new_Ki_i
                        )
                        
                        # Update states for next iteration (create new tensors)
                        new_integral_bat = adaptive_integral(
                            integral_bat,
                            error_bat,
                            TIME_STEP,
                            m
                        )
                    
                        integral_bat = new_integral_bat
                        prev_errors = error_bat
                        prev_v_bat = v_out_attacked
                        last_m = m
                        last_iref = i_ref

                        # Inside physics_loss function, where we update tracking variables:
    # Create new detached tensors:
                        physics_loss.integral_bat = physics_loss.integral_bat.clone().detach()
                        physics_loss.integral_bat[i] = integral_bat.clone().detach()
                        
                        physics_loss.prev_errors = physics_loss.prev_errors.clone().detach()
                        physics_loss.prev_errors[i] = prev_errors.clone().detach()
                        
                        physics_loss.prev_v_bat = physics_loss.prev_v_bat.clone().detach()
                        physics_loss.prev_v_bat[i] = prev_v_bat.clone().detach()
                        
                        physics_loss.last_m = physics_loss.last_m.clone().detach()
                        physics_loss.last_m[i] = last_m.clone().detach()
                        
                        physics_loss.last_iref = physics_loss.last_iref.clone().detach()
                        physics_loss.last_iref[i] = last_iref.clone().detach()
                        
                        # Update time separately since it's not part of gradient computation
                        physics_loss.last_execution_time = current_time



                    
                    else:
                        # Use previous values without modification
                        m = last_m
                        i_ref = last_iref

             # Calculate modulation indices from vd and vq first
                    m_d = safe_op(2 * v_d_ref / (v_dc + 1e-8))  # Add small epsilon to prevent division by zero
                    m_q = safe_op(2 * v_q_ref / (v_dc + 1e-8))

                    # Calculate total modulation index
                    m_system = safe_sqrt(m_d**2 + m_q**2)
                    m_system = torch.clamp(m_system, -1.0, 1.0)  # Ensure modulation index is between 0 and 1
                        
                        # Apply the control outputs
                    corrected_modulation_index = m

                    overall_modulation_index = safe_op(m_system + corrected_modulation_index)
                    overall_modulation_index = torch.clamp(overall_modulation_index, -1.0, 1.0)


                    wac_error_i_bat = (physics_loss.i_ref[i] - i_out_attacked)
                    wac_integral_i_bat = wac_integral_i_bat + (wac_error_i_bat * TIME_STEP)



                    # a. Current Setpoint Adjustment
                    i_bat_set = safe_op(physics_loss.i_ref[i] + new_Kp_i * wac_error_i_bat + new_Ki_i * wac_integral_i_bat)
                    i_bat_set = torch.clamp(i_bat_set, -3.0, 3.0)  # Limit charging current

                    i_out = safe_op(i_bat_set)  # Use the adjusted current setpoint

                    i_dc = torch.clamp(i_dc.clone(), -i_bat_set, i_bat_set)


                    

                    

                    v_dc = safe_op((2 * SQRT_2 / PI) * (v_ref_mag / (overall_modulation_index + 1e-6)))

                    v_out = safe_op(m_buck_vdc_to_v_out * v_dc)

                    # v_out = safe_op(v_dc - i_dc * R_battery)
                    
                    v_out_loss = safe_op(torch.mean(torch.square(v_out_ref - v_out)))

                    # Calculate voltage deviation from nominal
                    voltage_deviation = torch.abs(v_out_ref- v_out)
                    impact_scale = torch.exp(voltage_deviation * 2.0) 

                    attack_loss = attack_loss+ ( -ATTACK_WEIGHT * torch.mean(
                    v_out_loss * impact_scale
                    )) # Penalize attack magnitude

                    # Penalize voltages outside acceptable range
                    zero_tensor = torch.zeros_like(v_out)
                    upper_violation = torch.maximum(zero_tensor, v_out - torch.tensor(MAX_VOLTAGE_PU))
                    lower_violation = torch.maximum(zero_tensor, torch.tensor(MIN_VOLTAGE_PU) - v_out)

                    voltage_violation_loss = voltage_violation_loss + safe_op(torch.mean(
                        VOLTAGE_VIOLATION_PENALTY * (torch.square(upper_violation) + torch.square(lower_violation))
                    ))

                    zero_tensor = torch.zeros_like(v_out)
                    v_out_regulation_loss = safe_op(torch.mean(
                        torch.square(torch.maximum(zero_tensor, torch.tensor(0.95) - v_out)) + 
                        torch.square(torch.maximum(zero_tensor, v_out - torch.tensor(1.05)))
                    ))      

                    VOLTAGE_REG_WEIGHT = 1.0 # Weight for voltage regulation

                    voltage_regulation_loss = voltage_regulation_loss + safe_op(VOLTAGE_REG_WEIGHT * v_out_regulation_loss)

                    converter_efficiency = 0.95

                    # Calculate losses
                    ddelta_dt = calculate_gradient(delta, TIME_STEP)
                    domega_dt = calculate_gradient(omega, TIME_STEP)
                    dphi_d_dt = calculate_gradient(phi_d, TIME_STEP)
                    dphi_q_dt = calculate_gradient(phi_q, TIME_STEP)
                    di_d_dt = calculate_gradient(i_d, TIME_STEP)
                    di_q_dt = calculate_gradient(i_q, TIME_STEP)
                    di_L1_dt = calculate_gradient(i_L1, TIME_STEP)
                    di_L2_dt = calculate_gradient(i_L2, TIME_STEP)
                    dv_c_dt = calculate_gradient(v_c, TIME_STEP)
                    dv_dc_dt = calculate_gradient(v_dc, TIME_STEP)
                    di_out_dt = calculate_gradient(i_out, TIME_STEP)

                    P_ac = safe_op(v_d_ref * i_d_evcs + v_q_ref * i_q_evcs)
                    i_dc = safe_op(P_ac * converter_efficiency / v_dc)


                    v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
                    v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
                    zero_tensor = torch.zeros_like(v_out)
                    v_out_constraint = safe_op(torch.mean(torch.square(
                        torch.maximum(zero_tensor, v_out_lower - v_out) + 
                        torch.maximum(zero_tensor, v_out - v_out_upper)
                    )))

                    P_dc = safe_op(v_dc * i_dc)
                    P_out = safe_op(v_out * i_out)
                    DC_DC_EFFICIENCY = 0.98

                    di_d_dt_loss = safe_op(torch.mean(torch.square(di_d_dt - (1/LCL_L1) * (v_d_ref - R * i_d - v_d + omega * LCL_L1 * i_q))))
                    di_q_dt_loss = safe_op(torch.mean(torch.square(di_q_dt - (1/LCL_L1) * (v_q_ref - R * i_q - v_q - omega * LCL_L1 * i_d))))

                    di_L1_dt_loss = safe_op(torch.mean(torch.square(di_L1_dt - (1/LCL_L1) * (v_d_ref - v_c - R * i_L1))))
                    di_L2_dt_loss = safe_op(torch.mean(torch.square(di_L2_dt - (1/LCL_L2) * (v_c - v_ac - R * i_L2))))
                    dv_c_dt_loss = safe_op(torch.mean(torch.square(dv_c_dt - (1/LCL_CF) * (i_L1 - i_L2))))

                    # Calculate EVCS losses with safe handling
                    evcs_losses = [
                        # safe_op(torch.mean(torch.square(ddelta_dt - omega))),
                        # safe_op(torch.mean(torch.square(domega_dt - pll_error))),
                        safe_op(torch.mean(torch.square(dphi_d_dt - v_d_ref))),
                        safe_op(torch.mean(torch.square(dphi_q_dt - v_q_ref))),
                        safe_op(torch.mean(torch.square(di_d_dt_loss))),
                        safe_op(torch.mean(torch.square(di_q_dt_loss))),
                        safe_op(torch.mean(torch.square(di_L1_dt_loss))),
                        safe_op(torch.mean(torch.square(di_L2_dt_loss))),
                        safe_op(torch.mean(torch.square(dv_c_dt_loss))),
                        safe_op(torch.mean(torch.square(v_out - v_out_attacked))),
                        safe_op(torch.mean(torch.square(v_out_constraint))),
                        safe_op(torch.mean(torch.square(di_out_dt - (1/L_dc) * (v_out - v_battery - R_battery * i_out)))),                      
                        safe_op(torch.mean(torch.square(P_dc - P_ac) + torch.square(P_out - P_dc * DC_DC_EFFICIENCY))),
                        safe_op(torch.mean(torch.square(i_ac - i_L2) + torch.square(i_d - i_d_evcs) + torch.square(i_q - i_q_evcs)))
                    ]
                    
                    evcs_loss.extend(evcs_losses)


                except Exception as e:
                    print(f"Error processing EVCS {i}: {str(e)}")
                    continue

            # Calculate final losses
            V_regulation_loss = safe_op(voltage_regulation_loss)
            if len(evcs_loss) > 0:
                # Filter out None values and convert to tensors if needed
                valid_losses = []
                for loss in evcs_loss:
                    if loss is not None:
                        if not isinstance(loss, torch.Tensor):
                            loss = torch.tensor(loss, dtype=torch.float32)
                        valid_losses.append(loss)
                
                if len(valid_losses) > 0:
                    evcs_total_loss = safe_op(torch.sum(torch.stack(valid_losses)))
                else:
                    evcs_total_loss = torch.tensor(0.0, dtype=torch.float32)
            else:
                   evcs_total_loss = torch.tensor(0.0, dtype=torch.float32)
            wac_loss = safe_op(torch.mean(torch.square(wac_error_vdc) + torch.square(wac_error_vout)))






            # Calculate final total loss only after all components are computed
            total_loss = safe_op(power_flow_loss + evcs_total_loss + wac_loss + V_regulation_loss)

  
            return (
                total_loss,
                power_flow_loss,
                evcs_total_loss,
                wac_loss,
                V_regulation_loss
            )
            
    except Exception as e:
        print(f"Error in physics loss: {e}")
        return (
            torch.tensor(float('inf')), 
            torch.tensor(0.0), 
            torch.tensor(0.0), 
            torch.tensor(0.0), 
            torch.tensor(0.0)
        )

def adapt_gain(current_gain, error, rate, alpha):
    """Adapt controller gain based on error and learning rate."""
    gain = current_gain + alpha * error * rate
    return torch.clamp(gain, 0.1, 1.5)

def adaptive_integral(current_integral, error, dt, m):
    """Update integral term with adaptive anti-windup."""
    MAX_INTEGRAL = 10.0 * (1 - torch.abs(m)/0.25)
    integral = current_integral + error * dt
    return torch.clamp(integral, -MAX_INTEGRAL, MAX_INTEGRAL)

def adaptive_current_ref(current_ref, v_bat, prev_v_bat, i_bat, error, evcs_count, max_current, kp, ki):
    """Update current reference using PI control.
    
    Args:
        current_ref (float): Current reference value
        v_bat (float): Battery voltage
        prev_v_bat (float): Previous battery voltage 
        i_bat (float): Battery current
        error (float): Voltage error
        evcs_count (int): Number of active EVCS
        max_current (float): Maximum allowable current
        kp (float): Proportional gain
        ki (float): Integral gain
        
    Returns:
        float: Updated current reference
    """
    try:
        # Initialize static variables if not already set
        if not hasattr(adaptive_current_ref, 'integral_error'):
            adaptive_current_ref.integral_error = 0.0
            adaptive_current_ref.prev_error = 0.0
            adaptive_current_ref.dt = 0.001  # Time step
            
        # Update integral error with anti-windup
        MAX_INTEGRAL = 10.0
        adaptive_current_ref.integral_error = torch.clamp(
            adaptive_current_ref.integral_error + error * adaptive_current_ref.dt,
            -MAX_INTEGRAL,
            MAX_INTEGRAL
        )
        
        # Calculate PI control output
        control_output = kp * error + ki * adaptive_current_ref.integral_error
        
        # Determine current limits based on EVCS count
        if evcs_count == 1:
            current_limit = 1.0
        elif evcs_count == 2:
            current_limit = 2.0
        elif evcs_count == 3:
            current_limit = 3.0
        else:
            current_limit = 1.0
            
        # Clamp output to limits
        new_current_ref = torch.clamp(
            current_ref + control_output,
            0.0,  # Minimum current
            current_limit  # Maximum current based on EVCS count
        )
        
        # Store error for next iteration
        adaptive_current_ref.prev_error = error
        
        return new_current_ref
        
    except Exception as e:
        print(f"Error in adaptive_current_ref: {str(e)}")
        return current_ref  # Return unchanged reference on error





def calculate_gradient(x, spacing):
    """Calculate gradient using finite differences with proper handling of small tensors."""
    try:
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            



        # If tensor is too small, pad it by repeating the last value
        if x.shape[0] < 2:
            x = torch.cat([x, x], dim=0)
            
        # Calculate gradient using forward difference
        # (x[1:] - x[:-1]) / spacing for each batch
        grad = (x[1:] - x[:-1]) / spacing
        
        # If original input was single value, return single value gradient
        if grad.shape[0] == 0:
            return torch.zeros_like(x[0])
            
        return grad
        
    except Exception as e:
        print(f"Error in calculate_gradient: {e}")
        return torch.zeros_like(x[0] if len(x) > 0 else x)

def train_step(model, optimizer, bus_data_batch, Y_bus_tf, bus_data_tf, attack_actions, defend_actions):
    """Performs a single training step with proper tensor handling"""
    try:
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Calculate all losses
        total_loss, power_flow_loss, evcs_loss, wac_loss, v_reg_loss = physics_loss(
            model, Y_bus_tf, bus_data_tf,
            attack_actions, defend_actions
        )
        
        # Skip gradient update if we got error values
        if torch.abs(total_loss) >= 1e6:  # Check for error condition
            print("Skipping gradient update due to error in physics_loss")
            return torch.tensor(1e6, dtype=torch.float32)
        
        # Skip if loss is too high
        if torch.abs(total_loss) >= 1e6:
            return torch.tensor(1e6, dtype=torch.float32)
        
        # Backward pass with retain_graph=True
        total_loss.backward(retain_graph=True)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return total_loss, {
            'power_flow_loss': power_flow_loss,
            'evcs_loss': evcs_loss,
            'wac_loss': wac_loss,
            'v_reg_loss': v_reg_loss
        }
    except Exception as e:
        print(f"Error in training step: {e}")
        return torch.tensor(1e6, dtype=torch.float32), {}




def train_model(initial_model, dqn_agent, sac_attacker, sac_defender, Y_bus_tf, bus_data, epochs=1500, batch_size=256):
    torch.autograd.set_detect_anomaly(True)
    try:
        model = initial_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Move model to device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create environment with necessary data
        env = CompetingHybridEnv(
            pinn_model=model,
            y_bus_tf=Y_bus_tf,
            bus_data=bus_data,
            v_base_lv=V_BASE_DC,
            dqn_agent=dqn_agent,
            num_evcs=NUM_EVCS,
            num_buses=NUM_BUSES,
            time_step=TIME_STEP
        )
        
        # Convert bus data to PyTorch tensors and move to device
        bus_data_tf = torch.tensor(bus_data, dtype=torch.float32).to(device)
        Y_bus_tf = torch.tensor(Y_bus_tf, dtype=torch.float32).to(device)    
        
        history = {
            'total_loss': [],
            'power_flow_loss': [],
            'evcs_loss': [],
            'wac_loss': [],
            'v_reg_loss': []
        }
        
        current_loss = None  # Initialize loss variable in outer scope
        
        for epoch in range(epochs):
            try:
                # Set model to training mode
                model.train()
                
                # Reset environment and get initial state
                reset_result = env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                
                if state is None:
                    print(f"Error: Invalid state in epoch {epoch}")
                    continue
                
                # Ensure state is properly shaped and on correct device
                state = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(device)
                
                try:
                    # Get actions from agents
                    with torch.no_grad():  # No need to track gradients for action prediction
                        # DQN action
                        dqn_prediction = dqn_agent.predict(state.cpu().numpy(), deterministic=True)
                        dqn_action = dqn_prediction[0] if isinstance(dqn_prediction, tuple) else dqn_prediction
                        
                        # SAC actions
                        attack_action = sac_attacker.predict(state.cpu().numpy(), deterministic=True)[0]
                        defend_action = sac_defender.predict(state.cpu().numpy(), deterministic=True)[0]
                    
                    # Convert actions to tensors and move to device
                    attack_tensor = torch.tensor(attack_action, dtype=torch.float32).reshape(1, -1).to(device)
                    defend_tensor = torch.tensor(defend_action, dtype=torch.float32).reshape(1, -1).to(device)
                    
                except Exception as e:
                    print(f"Error in action prediction: {str(e)}")
                    continue
                
                # Calculate losses
                try:
                    # Clear gradients at the start
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass and loss calculation
                    losses = physics_loss(
                        model=model,
                        t=torch.tensor([[epoch * TIME_STEP]], dtype=torch.float32).to(device),
                        Y_bus_tf=Y_bus_tf,
                        bus_data=bus_data_tf,
                        attack_actions=attack_tensor,
                        defend_actions=defend_tensor
                    )
                    
                    if not isinstance(losses, tuple) or len(losses) != 5:
                        print(f"Invalid losses returned in epoch {epoch}")
                        continue
                    
                    total_loss, pf_loss, ev_loss, wac_loss, v_loss = losses
                    current_loss = total_loss.detach().clone()  # Store current loss
                    
                    # Check for invalid loss values
                    if not torch.isfinite(total_loss):
                        print(f"Non-finite loss detected in epoch {epoch}")
                        continue
                    
                    # Backward pass with retain_graph=True
                    total_loss.backward(retain_graph=True)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Explicitly detach losses before adding to history
                    with torch.no_grad():
                        history['total_loss'].append(float(total_loss.detach().cpu().item()))
                        history['power_flow_loss'].append(float(pf_loss.detach().cpu().item()))
                        history['evcs_loss'].append(float(ev_loss.detach().cpu().item()))
                        history['wac_loss'].append(float(wac_loss.detach().cpu().item()))
                        history['v_reg_loss'].append(float(v_loss.detach().cpu().item()))
                    
                    # Free up memory
                    del losses, total_loss, pf_loss, ev_loss, wac_loss, v_loss
                    torch.cuda.empty_cache()  # If using GPU
                    
                    # Print progress every N epochs
                    if epoch % 100 == 0:
                        print(f"Epoch {epoch}/{epochs}, Total Loss: {history['total_loss'][-1]:.4f}")
                    
                except Exception as e:
                    print(f"\nDetailed Error Information for epoch {epoch}:")
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # Take environment step
                try:
                    next_state, rewards, done, truncated, info = env.step({
                        'dqn': dqn_action,
                        'attacker': attack_action,
                        'defender': defend_action
                    })
                    
                    if done or truncated:
                        break
                        
                except Exception as e:
                    print(f"Error in environment step for epoch {epoch}: {str(e)}")
                    continue
                
                # Save checkpoint with the stored loss value
                if epoch % 500 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss,  # Use stored loss value
                    }, f'checkpoint_epoch_{epoch}.pt')
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        return model, history
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return initial_model, None


def evaluate_model_with_three_agents(env, dqn_agent, sac_attacker, sac_defender, num_steps=1500):
    """Evaluate the environment with DQN, SAC attacker, and SAC defender agents."""
    try:
        state, _ = env.reset()
        done = False
        time_step = env.TIME_STEP if hasattr(env, 'TIME_STEP') else TIME_STEP  # Add fallback

        # Initialize tracking variables as lists
        tracking_data = {
            'time_steps': [],
            'cumulative_deviations': [],
            'voltage_deviations': [],
            'attack_active_states': [],
            'target_evcs_history': [],
            'attack_durations': [],
            'dqn_actions': [],
            'sac_attacker_actions': [],
            'sac_defender_actions': [],
            'observations': [],
            'evcs_attack_durations': {i: [] for i in range(env.NUM_EVCS)},
            'attack_counts': {i: 0 for i in range(env.NUM_EVCS)},
            'total_durations': {i: 0 for i in range(env.NUM_EVCS)},
            'rewards': []
        }

        for step in range(num_steps):
            current_time = step * time_step
            
            try:
                # Convert state to numpy if it's a tensor
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().numpy()
                else:
                    state_np = np.array(state)

                # Get DQN action
                dqn_raw = dqn_agent.predict(state_np, deterministic=True)
                dqn_action = dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw
                
                # Convert DQN action to proper format
                if isinstance(dqn_action, np.ndarray):
                    dqn_action = torch.from_numpy(dqn_action).long()
                elif not isinstance(dqn_action, torch.Tensor):
                    dqn_action = torch.tensor(dqn_action, dtype=torch.long)

                # Get SAC actions
                sac_attacker_action, _ = sac_attacker.predict(state_np, deterministic=True)
                sac_defender_action, _ = sac_defender.predict(state_np, deterministic=True)
                
                # Convert SAC actions to numpy arrays
                if isinstance(sac_attacker_action, torch.Tensor):
                    sac_attacker_action = sac_attacker_action.detach().numpy()
                if isinstance(sac_defender_action, torch.Tensor):
                    sac_defender_action = sac_defender_action.detach().numpy()

                # Combine actions
                action = {
                    'dqn': dqn_action,
                    'attacker': sac_attacker_action,
                    'defender': sac_defender_action
                }

                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Handle rewards properly
                if isinstance(rewards, dict):
                    # Sum up all rewards from all agents
                    reward_value = sum(value for value in rewards.values() if isinstance(value, (int, float)))
                else:
                    reward_value = float(rewards) if isinstance(rewards, (int, float)) else 0.0

                # Update tracking data
                tracking_data['rewards'].append(reward_value)
                
                # Ensure next_state is numpy array
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.detach().numpy()

                # Store data with proper type conversion
                tracking_data['time_steps'].append(float(current_time))
                tracking_data['cumulative_deviations'].append(float(info.get('cumulative_deviation', 0)))
                tracking_data['voltage_deviations'].append(
                    np.array(info.get('voltage_deviations', [0] * env.NUM_EVCS), dtype=np.float32)
                )
                tracking_data['attack_active_states'].append(bool(info.get('attack_active', False)))
                tracking_data['target_evcs_history'].append(
                    np.array(info.get('target_evcs', [0] * env.NUM_EVCS), dtype=np.float32)
                )
                tracking_data['attack_durations'].append(float(info.get('attack_duration', 0)))
                tracking_data['dqn_actions'].append(dqn_action.cpu().numpy() if isinstance(dqn_action, torch.Tensor) else dqn_action)
                tracking_data['sac_attacker_actions'].append(sac_attacker_action.tolist())
                tracking_data['sac_defender_actions'].append(sac_defender_action.tolist())
                tracking_data['observations'].append(next_state.tolist())

                # Track EVCS-specific attack data
                target_evcs = np.array(info.get('target_evcs', [0] * env.NUM_EVCS))
                attack_duration = float(info.get('attack_duration', 0))
                for i in range(env.NUM_EVCS):
                    if target_evcs[i] == 1:
                        tracking_data['evcs_attack_durations'][i].append(attack_duration)
                        tracking_data['attack_counts'][i] =tracking_data['attack_counts'][i]+ 1
                        tracking_data['total_durations'][i] = tracking_data['total_durations'][i]+ attack_duration
                
                state = next_state
                if done or truncated:
                    break

            except Exception as e:
                print(f"Error in evaluation step {step}: {str(e)}")
                continue

        # Calculate average attack durations
        avg_attack_durations = []
        for i in range(env.NUM_EVCS):
            if tracking_data['attack_counts'][i] > 0:
                avg_duration = tracking_data['total_durations'][i] / tracking_data['attack_counts'][i]
            else:
                avg_duration = 0
            avg_attack_durations.append(float(avg_duration))

        # Convert lists to numpy arrays
        processed_data = {}
        for key, value in tracking_data.items():
            try:
                if isinstance(value, dict):
                    processed_data[key] = value
                elif key in ['time_steps', 'cumulative_deviations', 'attack_durations']:
                    processed_data[key] = np.array(value, dtype=np.float32)
                elif key in ['voltage_deviations', 'sac_attacker_actions', 'sac_defender_actions']:
                    processed_data[key] = np.array(value, dtype=np.float32)
                elif key == 'attack_active_states':
                    processed_data[key] = np.array(value, dtype=bool)
                else:
                    processed_data[key] = value
            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                processed_data[key] = value

        processed_data['avg_attack_durations'] = np.array(avg_attack_durations, dtype=np.float32)
        return processed_data

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None

def check_constraints(state, info):
        """Helper function to check individual constraints."""
        violations = []
        
        # Extract relevant state components
        # Assuming state structure matches your environment's observation space
        voltage_indices = slice(0, NUM_BUSES)  # Adjust based on your state structure
        current_indices = slice(NUM_BUSES, 2*NUM_BUSES)  # Adjust as needed
        
        # Check voltage constraints (0.9 to 1.1 p.u.)
        voltages = state[voltage_indices]
        if torch.any(voltages < 0.8) or torch.any(voltages > 1.2):
            violations.append({
                'type': 'Voltage',
                'values': voltages,
                'limits': (0.8, 1.2),
                'violated_indices': torch.where((voltages < 0.8) | (voltages > 1.2))[0]
            })

        # Check current constraints (-1.0 to 1.0 p.u.)
        currents = state[current_indices]
        if torch.any(torch.abs(currents) > 1.0):
            violations.append({
                'type': 'Current',
                'values': currents,
                'limits': (-3.0, 3.0),
                'violated_indices': torch.where(torch.abs(currents) > 3.0)[0]
            })

        # Check power constraints if available in state
        if 'power_output' in info:
            power = info['power_output']
            if torch.any(torch.abs(power) > 3.0):
                violations.append({
                    'type': 'Power',
                    'values': power,
                    'limits': (-3.0, 3.0),
                    'violated_indices': torch.where(torch.abs(power) > 3.0)[0]
                })

        # # Check SOC constraints if available
        # if 'soc' in info:
        #     soc = info['soc']
        #     if torch.any((soc < 0.1) | (soc > 0.9)):
        #         violations.append({
        #             'type': 'State of Charge',
        #             'values': soc,
        #             'limits': (0.1, 0.9),
        #             'violated_indices': torch.where((soc < 0.1) | (soc > 0.9))[0]
        #         })

        return violations, info

def validate_physics_constraints(env, dqn_agent, sac_attacker, sac_defender, num_episodes=5):
    """Validate that the agents respect physics constraints with detailed reporting."""


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            try:
                # Get actions from all agents
                dqn_action_scalar = dqn_agent.predict(state, deterministic=True)[0]
                dqn_action = env.decode_dqn_action(dqn_action_scalar)
                attacker_action = sac_attacker.predict(state, deterministic=True)[0]
                defender_action = sac_defender.predict(state, deterministic=True)[0]
                
                # Combine actions
                action = {
                    'dqn': dqn_action,
                    'attacker': attacker_action,
                    'defender': defender_action
                }
                
                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Check for physics violations
                violations = check_constraints(next_state, info)
                
                if violations:
                    print(f"\nPhysics constraints violated in episode {episode}, step {step_count}:")
                    for violation in violations:
                        print(f"\nViolation Type: {violation['type']}")
                        print(f"Limits: {violation['limits']}")
                        # print(f"Violated at indices: {violation['violated_indices']}")
                        # print(f"Values at violated indices: {violation['values'][violation['violated_indices']]}")
                    return False
                
                state = next_state
                step_count = step_count+ 1
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return False
            
    print("All physics constraints validated successfully!")
    return True, info


def convert_to_serializable(obj):
    """Convert numpy arrays and tensors to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj



def plot_evaluation_results(results, save_dir="./figures"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data from results and properly scale time
    time_steps = torch.tensor(results['time_steps']) * 10  # Multiply by 10 to correct scaling        
    cumulative_deviations = torch.tensor(results['cumulative_deviations'])
    voltage_deviations = torch.tensor(results['voltage_deviations'])
    attack_active_states = torch.tensor(results['attack_active_states'])
    avg_attack_durations = torch.tensor(results['avg_attack_durations'])

    #
    attacker_actions = torch.tensor(results['sac_attacker_actions'])  # Shape: [timesteps, NUM_EVCS*2]
    defender_actions = torch.tensor(results['sac_defender_actions'])  # Shape: [timesteps, NUM_EVCS*2]

    num_evcs = attacker_actions.shape[1] // 2


    # Plot cumulative deviations over time
    plt.figure()
    plt.plot(time_steps, cumulative_deviations, label='Cumulative Deviations')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Cumulative Deviations', fontsize = "20")
    # plt.title('Cumulative Deviations Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.savefig(f"{save_dir}/cumulative_deviations_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot total rewards over time
    plt.figure()
    # Convert rewards to total numerical values if they're dictionaries
    total_rewards = []
    for reward in results['rewards']:
        if isinstance(reward, dict):
            total_rewards.append(reward.get('attacker', 0) + reward.get('defender', 0))
        else:
            total_rewards.append(float(reward))
    
    plt.plot(time_steps, total_rewards, label='Total Rewards')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Total Rewards', fontsize = "20")
    # plt.title('Total Rewards Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot voltage deviations for each EVCS over time
    plt.figure()
    for i in range(voltage_deviations.shape[1]):
        plt.plot(time_steps, voltage_deviations[:, i], label=f'EVCS {i+1} Voltage Deviation')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Voltage Deviation (p.u.)', fontsize = "20")
    # plt.title('Voltage Deviations Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.savefig(f"{save_dir}/voltage_deviations_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot attack active states over time
    plt.figure()
    plt.plot(time_steps, attack_active_states, label='Attack Active State')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Attack Active State', fontsize = "20")
    # plt.title('Attack Active State Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.savefig(f"{save_dir}/attack_states_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


    # Plot average attack durations for each EVCS
    plt.figure()
    plt.bar(range(len(avg_attack_durations)), avg_attack_durations*2, 
            tick_label=[f'EVCS {i+1}' for i in range(len(avg_attack_durations))])
    plt.xlabel('EVCS', fontsize = "20")
    plt.ylim(0, 100)
    plt.ylabel('Average Attack Duration (s)', fontsize = "20")
    # plt.title('Average Attack Duration for Each EVCS', fontsize = "20")
    plt.grid(True)
    plt.savefig(f"{save_dir}/avg_attack_durations_{timestamp}.pdf", dpi=500, bbox_inches='tight')
    plt.close()



    # Plot defender actions for each EVCS
    plt.figure()
    for i in range(num_evcs):
        # Plot Kp control actions
        plt.plot(time_steps, defender_actions[:, i], 
                label=f'EVCS {i+1} Kp Control', linestyle='-')
        # Plot Ki control actions
        plt.plot(time_steps, defender_actions[:, i+num_evcs], 
                label=f'EVCS {i+1} Ki Control', linestyle='--')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Defense Action Value', fontsize = "20")
    # plt.title('Defender Actions Over Time', fontsize = "20")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = "18")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/defender_actions_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


    # # Plot voltage attacks for each EVCS
    # plt.figure()
    # attacked_evcs = [i for i in range(num_evcs) if avg_attack_durations[i] > 0.01]  # Use threshold
    # if attacked_evcs:  # Only create plot if there were attacks
    #     for i in attacked_evcs:
    #         plt.plot(time_steps, attacker_actions[:, i], 
    #                 label=f'EVCS {i+1}', linestyle='-')
    #     plt.xlabel('Time (s)', fontsize = "20")
    #     plt.ylabel('Voltage Attack Value', fontsize = "20")
    #     plt.legend(fontsize = "18")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/voltage_attacks_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    # plt.close()

    # # Plot current attacks for each EVCS
    # plt.figure()
    # if attacked_evcs:  # Only create plot if there were attacks
    #     for i in attacked_evcs:
    #         plt.plot(time_steps, attacker_actions[:, i+num_evcs], 
    #                 label=f'EVCS {i+1}', linestyle='-')
    #     plt.xlabel('Time (s)', fontsize = "20")
    #     plt.ylabel('Current Attack Value', fontsize = "20")
    #     plt.legend(fontsize = "18")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/current_attacks_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    # plt.close()


    attacker_data = {}
    for i in range(num_evcs):
        attacker_data[f'EVCS_{i+1}_Voltage_Attack'] = attacker_actions[:, i]
        attacker_data[f'EVCS_{i+1}_Current_Attack'] = attacker_actions[:, i+num_evcs]
    attacker_data['Time'] = time_steps
    attacker_df = pd.DataFrame(attacker_data)
    attacker_df.to_csv(f"{save_dir}/attacker_actions_{timestamp}.csv", index=False)

    # Create DataFrames for defender actions
    defender_data = {}
    for i in range(num_evcs):
        defender_data[f'EVCS_{i+1}_Kp_Control'] = defender_actions[:, i]
        defender_data[f'EVCS_{i+1}_Ki_Control'] = defender_actions[:, i+num_evcs]
    defender_data['Time'] = time_steps
    defender_df = pd.DataFrame(defender_data)
    defender_df.to_csv(f"{save_dir}/defender_actions_{timestamp}.csv", index=False)

    # Plot voltage attacks for each EVCS
    plt.figure()
    for i in range(num_evcs):
        plt.plot(time_steps, attacker_actions[:, i], 
                label=f'EVCS {i+1}', linestyle='-')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Voltage Attack Value', fontsize = "20")
    # plt.title('Voltage Attacks Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/voltage_attacks_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot current attacks for each EVCS
    plt.figure()
    for i in range(num_evcs):
        plt.plot(time_steps, attacker_actions[:, i+num_evcs], 
                label=f'EVCS {i+1}', linestyle='-')
    plt.xlabel('Time (s)', fontsize = "20")
    plt.ylabel('Current Attack Value', fontsize = "20")
    # plt.title('Current Attacks Over Time', fontsize = "20")
    plt.legend(fontsize = "18")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/current_attacks_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def pretrain_model(initial_model, Y_bus_tf, bus_data, epochs=100, batch_size=128):
    """Pretrain the model without using RL agents"""
    try:
        model = initial_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Move model to device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Convert data to PyTorch tensors and move to device
        bus_data_tf = torch.tensor(bus_data, dtype=torch.float32).to(device)
        Y_bus_tf = torch.tensor(Y_bus_tf, dtype=torch.float32).to(device)
        
        history = {
            'total_loss': [],
            'power_flow_loss': [],
            'evcs_loss': [],
            'wac_loss': [],
            'v_reg_loss': []
        }
        
        # Create dummy attack and defend actions (zeros)
        attack_actions = torch.zeros((1, 2 * NUM_EVCS), device=device)
        defend_actions = torch.zeros((1, 2 * NUM_EVCS), device=device)
        
        for epoch in range(epochs):
            try:
                # Set model to training mode
                model.train()
                
                # Clear gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Calculate time step
                t = torch.tensor([[epoch * TIME_STEP]], dtype=torch.float32).to(device)
                
                # Calculate losses without attack/defend actions
                losses = physics_loss(
                    model=model,
                    t=t,
                    Y_bus_tf=Y_bus_tf,
                    bus_data=bus_data_tf,
                    attack_actions=attack_actions,
                    defend_actions=defend_actions
                )
                
                if not isinstance(losses, tuple) or len(losses) != 5:
                    print(f"Invalid losses returned in epoch {epoch}")
                    continue
                
                total_loss, pf_loss, ev_loss, wac_loss, v_loss = losses
                
                # Check for invalid loss values
                if not torch.isfinite(total_loss):
                    print(f"Non-finite loss detected in epoch {epoch}")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Store losses
                history['total_loss'].append(float(total_loss.detach().cpu()))
                history['power_flow_loss'].append(float(pf_loss.detach().cpu()))
                history['evcs_loss'].append(float(ev_loss.detach().cpu()))
                history['wac_loss'].append(float(wac_loss.detach().cpu()))
                history['v_reg_loss'].append(float(v_loss.detach().cpu()))
                
                # Print progress
                if epoch % 100 == 0:
                    print(f"Pretraining Epoch {epoch}/{epochs}, Total Loss: {history['total_loss'][-1]:.4f}")
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
                
        return model, history
        
    except Exception as e:
        print(f"Pretraining error: {str(e)}")
        return initial_model, None

if __name__ == '__main__':
    # Define physics parameters
    print("Starting program execution...")
    start_time = time.time()

    physics_params = {
        'voltage_limits': (0.8, 1.2),
        'v_out_nominal': 1.0,
        'current_limits': (-3.0, 3.0),
        'i_rated': 1.0,
        'attack_magnitude': 0.04,
        'current_magnitude': 0.03,
        'wac_kp_limits': (0.0, 0.5),
        'wac_ki_limits': (0.0, 0.5),
        'control_saturation': 0.3,
        'power_limits': (0.5, 1.5),
        'power_ramp_rate': 0.1,
        'evcs_efficiency': 0.98,
        'soc_limits': (0.1, 0.9),
        'modulation_index_system': (modulation_index_system*0.5, modulation_index_system*1.5)
    }

    # Initialize the PINN model
    initial_pinn_model = EVCS_PowerSystem_PINN(        num_buses=NUM_BUSES,
        num_evcs=NUM_EVCS,
        hidden_dim=2048,  # You can adjust this
        num_blocks=4 )


    optimizer = torch.optim.Adam(initial_pinn_model.parameters(), lr=1e-3)



    print("Pretraining the initial PINN model...")
    initial_pinn_model, pre_training_history = pretrain_model(
        initial_model=initial_pinn_model,
        Y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        epochs=1000,
        batch_size=128
    )

    print("Pretraining completed")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{timestamp}"
    model_dir = f"./models/{timestamp}"
    for dir_path in [log_dir, model_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create the Discrete Environment for DQN Agent
    print("Creating the DiscreteHybridEnv environment...")
    discrete_env = DiscreteHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    # Initialize callbacks
    
    dqn_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/dqn_checkpoints/",
        name_prefix="dqn"
    )
    
    # Initialize the DQN Agent with improved parameters
    print("Initializing the DQN agent...")
    dqn_agent = DQN(
        'MlpPolicy',
        discrete_env,
        verbose=1,
        learning_rate=4e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.2,
        train_freq=4,
        batch_size=32,
        gamma=0.99,
        device='cuda',
        tensorboard_log=f"{log_dir}/dqn/"
    )

    # Train DQN with monitoring
    print("Training DQN agent...")
    dqn_agent.learn(
        total_timesteps=5000,
        callback=dqn_checkpoint,
        progress_bar=True
    )
    dqn_agent.save(f"{model_dir}/dqn_final")

    # Create the CompetingHybridEnv
    print("Creating the CompetingHybridEnv environment...")
    combined_env = CompetingHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    print("Creating SAC Wrapper environments...")
    sac_attacker_env = SACWrapper(
        env=combined_env,
        agent_type='attacker',
        dqn_agent=dqn_agent
    )
    # Initialize SAC Attacker
    print("Initializing SAC Attacker...")
    sac_attacker = SAC(
        'MlpPolicy',
        sac_attacker_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_attacker/"
    )

    # Create defender wrapper environment with the trained attacker
    print("Creating SAC Defender environment...")
    sac_defender_env = SACWrapper(
        env=combined_env,
        agent_type='defender',
        dqn_agent=dqn_agent
    )

    # Initialize SAC Defender
    print("Initializing SAC Defender...")
    sac_defender = SAC(
        'MlpPolicy',
        sac_defender_env,
        verbose=1,
        learning_rate=1e-5,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_defender/"
    )

    # Update wrapper environments with both agents
    sac_attacker_env.sac_defender = sac_defender
    sac_defender_env.sac_attacker = sac_attacker

    # Create callbacks for monitoring
    sac_attacker_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_attacker_checkpoints/",
        name_prefix="attacker"
    )
    
    sac_defender_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_defender_checkpoints/",
        name_prefix="defender"
    )
# New Addition 
    print("Training the SAC Attacker agent...")
    sac_attacker.learn(
        total_timesteps=5000,   
        callback=sac_attacker_checkpoint,
        progress_bar=True
    )
    sac_attacker.save(f"{model_dir}/sac_attacker_final")

    print("Training the SAC Defender agent...")
    sac_defender.learn(
        total_timesteps=5000,
        callback=sac_defender_checkpoint,
        progress_bar=True
    )
    sac_defender.save(f"{model_dir}/sac_defender_final")

    num_iterations = 5
    # Joint training loop with validation
    print("Starting joint training...")
    for iteration in range(num_iterations):
        print(f"\nJoint training iteration {iteration + 1}/{num_iterations}")
        
        # Train agents with progress monitoring
        for agent, name, callback, env in [
            (dqn_agent, "DQN", dqn_checkpoint, discrete_env),
            (sac_attacker, "SAC Attacker", sac_attacker_checkpoint, sac_attacker_env),
            (sac_defender, "SAC Defender", sac_defender_checkpoint, sac_defender_env)
        ]:
            print(f"\nTraining {name}...")
            if name == "SAC Defender":
                total_timesteps=2500
            else:
                total_timesteps=5000
            agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            agent.save(f"{model_dir}/{name.lower()}_iter_{iteration + 1}")

            # Update environment references after each agent training
            combined_env.update_agents(dqn_agent, sac_attacker, sac_defender)
            sac_attacker_env.update_agents(sac_defender=sac_defender, dqn_agent=dqn_agent)
            sac_defender_env.update_agents(sac_attacker=sac_attacker, dqn_agent=dqn_agent)

        # # Validate physics constraints
        # print("\nValidating physics constraints...")
        # validation_success = validate_physics_constraints(
        #     combined_env,
        #     dqn_agent,
        #     sac_attacker,
        #     sac_defender,
        #     num_episodes=3
        # )
        # print(f"Physics validation: {'Passed' if validation_success else 'Failed'}")
    epochs = 2500
    print("Training the PINN model with the hybrid RL agents (DQN for target, SAC Attacker for FDI, and SAC Defender for stabilization)...")
    trained_pinn_model, training_history = train_model(
        initial_model=initial_pinn_model,
        dqn_agent=dqn_agent,
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        Y_bus_tf=Y_bus_tf,  # Your Y-bus matrix
        bus_data=bus_data,  # Your bus data
        epochs=epochs,
        batch_size=128
    )

    # Optionally plot training history
    if training_history is not None:
        for epoch in range(0, epochs, 100):  # Print every 100 epochs
            print(f"\nEpoch {epoch}:")
            print(f"Total Loss: {training_history['total_loss'][epoch]:.4f}")
            print(f"Power Flow Loss: {training_history['power_flow_loss'][epoch]:.4f}")
            print(f"EVCS Loss: {training_history['evcs_loss'][epoch]:.4f}")
            print(f"WAC Loss: {training_history['wac_loss'][epoch]:.4f}")
            print(f"Voltage Regulation Loss: {training_history['v_reg_loss'][epoch]:.4f}")
    

        # After training the PINN model, create a new environment using the trained model
    print("Creating a new CompetingHybridEnv environment with the trained PINN model...")
    trained_combined_env = CompetingHybridEnv(
        pinn_model=trained_pinn_model,  # Use the trained PINN model here
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,  # Use the trained agents
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP
    )

    # Save the trained model if needed
    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(trained_pinn_model.state_dict(), f'models/pinn_model_{current_time}.pth')
        print(f"\nModel saved successfully as: pinn_model_{current_time}.pth")
        
        # Save training history
        import json
        with open(f'models/training_history_{current_time}.json', 'w') as f:
            json.dump(training_history, f)
        print(f"Training history saved as: training_history_{current_time}.json")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")

        # Update the environment's agent references if necessary
    trained_combined_env.sac_attacker = sac_attacker
    trained_combined_env.sac_defender = sac_defender
    trained_combined_env.dqn_agent = dqn_agent

    # Update the main evaluation and saving code
    try:
        print("Running final evaluation...")
        # Change this line:
        # results = evaluate_model_with_three_agents(env, dqn_agent, sac_attacker, sac_defender)
        
        # To this:
        results = evaluate_model_with_three_agents(trained_combined_env, dqn_agent, sac_attacker, sac_defender)
        
        if results is not None:
            # Convert results to serializable format
            save_results = convert_to_serializable(results)
            
            # Save to file
            with open('evaluation_results.json', 'w') as f:
                json.dump(save_results, f, indent=4)
            print("Evaluation results saved successfully")
            
            # Prepare data for plotting
            plot_data = {
                'time_steps': save_results['time_steps'],
                'cumulative_deviations': save_results['cumulative_deviations'],
                'voltage_deviations': save_results['voltage_deviations'],
                'attack_active_states': save_results['attack_active_states'],
                'target_evcs_history': save_results['target_evcs_history'],
                'attack_durations': save_results['attack_durations'],
                'observations': save_results['observations'],
                'avg_attack_durations': save_results['avg_attack_durations'],
                'rewards': save_results['rewards'],
                'sac_attacker_actions': save_results['sac_attacker_actions'],
                'sac_defender_actions': save_results['sac_defender_actions']
            }
            
            # Plot the results
            plot_evaluation_results(plot_data)
        else:
            print("Evaluation failed to produce results")
            
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        traceback.print_exc()


    # Plot the results
    # plot_evaluation_results(plot_data)

    print("\nTraining completed successfully!")

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format the time nicely
    time_delta = timedelta(seconds=execution_time)
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    
    print("\nProgram Execution Summary:")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")




