import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

class CompetingHybridEnv(gym.Env):
    """Custom environment for joint training of DQN and SAC agents."""
    def __init__(self, pinn_model, y_bus_tf, bus_data, v_base_lv, dqn_agent, num_evcs=5, num_buses=33, time_step=0.1, **physics_params):
        super(CompetingHybridEnv, self).__init__()
        
        # System parameters
        self.NUM_EVCS = num_evcs
        self.NUM_BUSES = num_buses
        self.TIME_STEP = time_step
        self.V_BASE_LV = v_base_lv

        # Extract physics parameters with defaults
        self.voltage_limits = physics_params.get('voltage_limits', (0.8, 1.2))
        self.v_out_nominal = physics_params.get('v_out_nominal', 1.0)
        self.current_limits = physics_params.get('current_limits', (-4.0, 4.0))
        self.i_rated = physics_params.get('i_rated', 1.0)
        self.attack_magnitude = physics_params.get('attack_magnitude', 0.02)
        self.current_magnitude = physics_params.get('current_magnitude', 0.01)
        self.wac_kp_limits = physics_params.get('wac_kp_limits', (0.0, 0.5))
        self.wac_ki_limits = physics_params.get('wac_ki_limits', (0.0, 0.5))
        self.control_saturation = physics_params.get('control_saturation', 0.5)
        self.power_limits = physics_params.get('power_limits', (0.5, 1.5))
        self.power_ramp_rate = physics_params.get('power_ramp_rate', 0.1)
        self.evcs_efficiency = physics_params.get('evcs_efficiency', 0.98)
        self.soc_limits = physics_params.get('soc_limits', (0.1, 0.9))
        self.modulation_index_system = physics_params.get('modulation_index_system', (0.0,0.0))

        self.attack_rejections = {i: 0 for i in range(self.NUM_EVCS)}
        self.attack_attempts = {i: 0 for i in range(self.NUM_EVCS)}
        
        # WAC parameters
        self.WAC_VOUT_SETPOINT = 1.0  # Nominal voltage in p.u.
        self.WAC_VDC_SETPOINT = 1.0  # Nominal voltage in p.u.
        self.WAC_KP_VOUT_DEFAULT = 0.3
        self.WAC_KI_VOUT_DEFAULT = 0.2
        self.WAC_KP_VDC_DEFAULT = 0.3
        self.WAC_KI_VDC_DEFAULT = 0.2
        
        # Voltage limits
        self.V_OUT_NOMINAL = 1.0  # Nominal voltage in p.u.
        self.V_OUT_VARIATION = 0.05  # 5% allowed variation
        self.V_OUT_MIN = self.V_OUT_NOMINAL - self.V_OUT_VARIATION
        self.V_OUT_MAX = self.V_OUT_NOMINAL + self.V_OUT_VARIATION
        
        # Initialize models and data
        self.pinn_model = pinn_model
        self.y_bus_tf = torch.tensor(y_bus_tf, dtype=torch.float32)
        self.bus_data = bus_data
        self.dqn_agent = dqn_agent
        
        # Initialize state variables
    
        self.time_step_counter = 0
        self.current_time = 0.0
        self.cumulative_deviation = 0.0
        self.modulation_index = np.zeros(self.NUM_EVCS)
        
        # Initialize attack-related variables as numpy arrays
        self.target_evcs = np.zeros(self.NUM_EVCS)  # Use int dtype
        self.attack_active = False
        self.attack_start_time = 0
        self.attack_end_time = 0
        self.attack_duration = 0
        self.voltage_deviations = np.zeros(self.NUM_EVCS)
        
        # WAC control variablesf
        self.wac_integral = np.zeros(self.NUM_EVCS)
        self.wac_error = np.zeros(self.NUM_EVCS)
        self.wac_control = np.zeros(self.NUM_EVCS)
        self.voltage_error = np.zeros(self.NUM_EVCS)
        self.kp_vout = np.ones(self.NUM_EVCS) * self.WAC_KP_VOUT_DEFAULT
        self.ki_vout = np.ones(self.NUM_EVCS) * self.WAC_KI_VOUT_DEFAULT

        self.NUM_DURATION = 50
        
        # Define action spaces
        # self.dqn_action_space = gym.spaces.Discrete(self.NUM_EVCS * 3)

        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [self.NUM_DURATION])
        
        self.sac_attacker_action_space = gym.spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        self.sac_defender_action_space = gym.spaces.Box(
            low=0.0,
            high=2.0,
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # Define observation space (25 dimensions for state variables)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

        self.state = np.zeros(self.observation_space.shape[0])

        # Control limits
        self.CONTROL_MAX = 1.0
        self.CONTROL_MIN = -1.0
        self.INTEGRAL_MAX = 10.0
        self.INTEGRAL_MIN = -10.0

        # Add voltage and current limits
        self.voltage_limits = (0.8, 1.2)  # ±15% of nominal voltage
        self.current_limits = (-3.0, 3.0)   # Normalized current limits

        self.reset_state()

    def _setup_action_spaces(self):
        """Setup action spaces for all agents."""
        # DQN action space
        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [self.NUM_DURATION])
        self.total_dqn_actions = int(np.prod([2] * self.NUM_EVCS + [self.NUM_DURATION]))
        
        # SAC Attacker action space (using attack_magnitude from physics params)
        self.sac_attacker_action_space = gym.spaces.Box(
            low=-self.attack_magnitude * np.ones(self.NUM_EVCS * 2),
            high=self.attack_magnitude * np.ones(self.NUM_EVCS * 2),
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # SAC Defender action space (using WAC limits from physics params)
        self.sac_defender_action_space = gym.spaces.Box(
            low=np.zeros(self.NUM_EVCS * 2),
            high=np.array([self.wac_kp_limits[1]] * self.NUM_EVCS + 
                         [self.wac_ki_limits[1]] * self.NUM_EVCS),
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # Combined action space
        self.action_space = gym.spaces.Dict({
            'dqn': self.dqn_action_space,
            'attacker': self.sac_attacker_action_space,
            'defender': self.sac_defender_action_space
        })


    def update_agents(self, dqn_agent=None, sac_attacker=None, sac_defender=None):
        """Update agent references in the environment."""
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
        if sac_defender is not None:
            self.sac_defender = sac_defender
        
        # Validate that required agents are present
        if self.dqn_agent is None:
            print("Warning: DQN agent is not set")
        if self.sac_attacker is None:
            print("Warning: SAC attacker is not set")
        if self.sac_defender is None:
            print("Warning: SAC defender is not set")


    def validate_agents(self):
        """Validate that all required agents are properly initialized."""
        agents_valid = True
        
        if self.dqn_agent is None:
            print("Error: DQN agent is required but not set")
            agents_valid = False
            
        if self.sac_attacker is None:
            print("Warning: SAC attacker is not set, will use default actions")
            
        if self.sac_defender is None:
            print("Warning: SAC defender is not set, will use default actions")
            
        return agents_valid
            
        return self.validate_agents()
    def reset_state(self):
        """Reset all state variables."""
        self.time_step_counter = 0
        self.current_time = 0.0
        self.cumulative_deviation = 0.0
        self.wac_integral = np.zeros(self.NUM_EVCS)
        self.wac_error = np.zeros(self.NUM_EVCS)
        self.wac_control = np.zeros(self.NUM_EVCS)
        self.voltage_error = np.zeros(self.NUM_EVCS)
        self.kp_vout = np.zeros(self.NUM_EVCS)
        self.ki_vout = np.zeros(self.NUM_EVCS)
        self.fdi_v = np.zeros(self.NUM_EVCS)
        self.fdi_i_d = np.zeros(self.NUM_EVCS)

        self.target_evcs = np.zeros(self.NUM_EVCS)  # Use int dtype
        self.attack_active = False
        self.attack_start_time = 0
        self.attack_end_time = 0
        self.attack_duration = 0  
        self.voltage_deviations = np.zeros(self.NUM_EVCS)

        self.attack_rejections = {i: 0 for i in range(self.NUM_EVCS)}
        self.attack_attempts = {i: 0 for i in range(self.NUM_EVCS)} 
        
        # Initialize state with zeros
        self.state = np.zeros(25)  # Assuming 25 is the observation space size

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            # Reset time counter
            self.time_step_counter = 0
            self.current_time = 0.0
            
            # Reset attack parameters
            self.target_evcs = np.zeros(self.NUM_EVCS, dtype=np.int32)
            self.attack_duration = 0
            self.attack_start_time = 0
            self.attack_end_time = 0
            
            # Reset WAC parameters
            self.kp_vout = np.full(self.NUM_EVCS, self.WAC_KP_VOUT_DEFAULT)
            self.ki_vout = np.full(self.NUM_EVCS, self.WAC_KI_VOUT_DEFAULT)
            
            # Get initial state from PINN model
            initial_prediction = self.pinn_model(torch.tensor([[0.0]], dtype=torch.float32))
            if initial_prediction is None:
                print("Error: Initial prediction is None")
                return np.zeros(self.observation_space.shape[0]), {}
                
            evcs_vars = initial_prediction[:, 2 * self.NUM_BUSES:].detach().numpy()[0]
            initial_state = self.get_observation(evcs_vars)
            
            # Super call if needed
            if hasattr(super(), 'reset'):
                super().reset(seed=seed)
            
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
            
            return initial_state, {}  # Return state and empty info dict
            
        except Exception as e:
            print(f"Error in environment reset: {str(e)}")
            # Return zero state and empty info dict as fallback
            return np.zeros(self.observation_space.shape[0]), {}

    def apply_wac_control(self,voltage_error,kp_vout,ki_vout):
        """Apply Wide Area Control with anti-windup protection."""
        self.voltage_error = voltage_error
        self.kp_vout = kp_vout
        self.ki_vout = ki_vout

        self.wac_control = np.zeros(self.NUM_EVCS)
        for i in range(self.NUM_EVCS):
            # Update integral term with anti-windup
            self.wac_integral[i] = np.clip(
                self.wac_integral[i] + self.voltage_error[i] * self.TIME_STEP,
                -self.control_saturation, self.control_saturation  # Using physics params
            )
            
            # Calculate control action
            self.modulation_index[i] = (
                self.kp_vout[i] * self.voltage_error[i] 
                + self.ki_vout[i] * self.wac_integral[i]
            )
            
            # Clip control action using physics params
            self.modulation_index[i] = np.clip(
                self.modulation_index[i],0,1 )

        return self.modulation_index

    def validate_physics(self, new_state):
        """Validate physics constraints."""
        try:
            # Handle scalar input
            if np.isscalar(new_state):
                return True  # or handle differently based on your requirements
                
            v_out = new_state[:self.NUM_EVCS]
            i_out = new_state[3*self.NUM_EVCS:4*self.NUM_EVCS]
            i_dc = new_state[4*self.NUM_EVCS:5*self.NUM_EVCS]
            
            voltage_valid = np.all((v_out >= self.voltage_limits[0]) & 
                                 (v_out <= self.voltage_limits[1]))
            current_valid = np.all((i_out >= self.current_limits[0]) & 
                                 (i_out <= self.current_limits[1]) &
                                 (i_dc >= self.current_limits[0]) & 
                                 (i_dc <= self.current_limits[1]))
            
            return voltage_valid and current_valid
            
        except Exception as e:
            print(f"Error in validate_physics: {e}")
            return False

    def calculate_rewards(self, voltage_deviations):
        """Calculate rewards for all agents."""
        try:
            attack_rewards = []
            defender_rewards = []
            
            # Calculate rewards for each EVCS
            for i, deviation in enumerate(voltage_deviations):
                if self.target_evcs[i] == 1:  # For targeted EVCSs
                    if deviation > 0.05:  # Significant deviation
                        # Attacker gets positive reward for successful attack
                        attack_reward = 40.0 * deviation - 0.1 * self.current_time
                        # Defender gets negative reward proportional to deviation
                        defend_reward = -10.0 * deviation - 0.01 * self.current_time
                    else:  # Small deviation
                        # Attacker gets small reward for attempting
                        attack_reward = 10 * deviation
                        # Defender gets positive reward for maintaining stability
                        defend_reward = 1.0 * (0.05 - deviation) + 0.01 * self.current_time
                else:  # For non-targeted EVCSs
                    attack_reward = 0.0
                    # Defender gets small positive reward for maintaining stability
                    defend_reward = 1.0 * (0.1 - deviation) if deviation < 0.1 else -5.0 * deviation
                    
                attack_rewards.append(attack_reward)
                defender_rewards.append(defend_reward)
            
            # Sum rewards for each agent
            total_attack_reward = float(np.sum(attack_rewards))
            total_defend_reward = float(np.sum(defender_rewards))
            
            # Add time-based penalties to discourage prolonged attacks
            if self.attack_active:
                time_penalty = 0.1 * self.current_time
                total_attack_reward -= time_penalty
                total_defend_reward += 0.02 * self.current_time  # Small bonus for defender over time
            
            # print(f"Attack reward: {total_attack_reward} and Defender reward: {total_defend_reward}")
            
            return {
                'attacker': total_attack_reward,
                'defender': total_defend_reward
            }
            
        except Exception as e:
            print(f"Error in calculate_rewards: {e}")
            return {
                'attacker': 0.0,
                'defender': 0.0
        }


    def calculate_rewardssss(self, voltage_deviations):
        """Calculate rewards for all agents."""
        try:
            attack_rewards = []
            defender_rewards = []
            
            # Calculate rewards for each EVCS
            for i, deviation in enumerate(voltage_deviations):
                if self.target_evcs[i] == 1:  # For targeted EVCSs
                    if deviation > 0.1:  # Significant deviation
                        # Attacker gets positive reward for successful attack
                        attack_reward = 20.0 * deviation - 0.1 * self.current_time
                        # Defender gets negative reward proportional to deviation
                        defend_reward = -10.0 * deviation - 0.01 * self.current_time
                    else:  # Small deviation
                        # Attacker gets small reward for attempting
                        attack_reward = 0.01 * deviation
                        # Defender gets positive reward for maintaining stability
                        defend_reward = 1.0 * (0.05 - deviation) + 0.01 * self.current_time
                else:  # For non-targeted EVCSs
                    attack_reward = 0.0
                    # Defender gets small positive reward for maintaining stability
                    defend_reward = 1.0 * (0.1 - deviation) if deviation < 0.1 else -5.0 * deviation
                    
                attack_rewards.append(attack_reward)
                defender_rewards.append(defend_reward)
            
            # Sum rewards for each agent
            total_attack_reward = float(np.sum(attack_rewards))
            total_defend_reward = float(np.sum(defender_rewards))
            
            # Add time-based penalties to discourage prolonged attacks
            if self.attack_active:
                time_penalty = 0.1 * self.current_time
                total_attack_reward -= time_penalty
                total_defend_reward += 0.02 * self.current_time  # Small bonus for defender over time
            
            # print(f"Attack reward: {total_attack_reward} and Defender reward: {total_defend_reward}")
            
            return {
                'attacker': total_attack_reward,
                'defender': total_defend_reward
            }
            
        except Exception as e:
            print(f"Error in calculate_rewards: {e}")
            return {
                'attacker': 0.0,
                'defender': 0.0
        }

    
    def prepare_defender_actions_for_pinn(self, defender_action):
        """Prepare defender actions in the format expected by PINN model."""
        try:
            # Split defender action into Kp and Ki adjustments
            kp_adjustments = defender_action[:self.NUM_EVCS]
            ki_adjustments = defender_action[self.NUM_EVCS:]

            # Scale the defender actions to appropriate ranges
            kp_vout = self.control_saturation(
                kp_adjustments,
                0.0,  # WAC parameters should be positive
                0.5
            )
            ki_vout = self.control_saturation(
                ki_adjustments,
                0.0,
                0.5
            )

            # Convert to tensor and reshape for PINN model
            wac_params = torch.tensor(
                np.concatenate([kp_vout, ki_vout])[np.newaxis, :],
                dtype=torch.float32
            )
            
            return wac_params
            
        except Exception as e:
            print(f"Error in prepare_defender_actions_for_pinn: {e}")
            return torch.zeros((1, self.NUM_EVCS * 2), dtype=torch.float32)

    def step(self, action):
        """Execute one time step within the environment."""
        try:
            # Update time
            self.time_step_counter += 1
            self.current_time += self.TIME_STEP

            # Ensure action is a dictionary with the right keys
            if not isinstance(action, dict):
                raise ValueError(f"Expected dict action, got {type(action)}")
            
            if not all(k in action for k in ['dqn', 'attacker', 'defender']):
                raise ValueError("Action dict missing required keys")

            # Get actions and ensure they are numpy arrays
            dqn_action = np.array(action['dqn'])
            dqn_action = self.decode_action(dqn_action)


            attacker_action = np.array(action['attacker'], dtype=np.float32)
            defender_action = np.array(action['defender'], dtype=np.float32)

            # Process DQN action
            # First NUM_EVCS elements are target selection (0 or 1)
            # Last element is attack duration (0-9)
            self.target_evcs = dqn_action[:-1].astype(int) # Convert to int array
            self.attack_duration = int(dqn_action[-1])  # Scale duration

            # Set attack parameters
            if np.any(self.target_evcs > 0):  # Check if any EVCS is targeted
                self.attack_active = True
                if self.attack_start_time == 0:  # Only set start time if not already set
                    self.attack_start_time = self.time_step_counter
                    self.attack_end_time = np.clip(self.attack_start_time + self.attack_duration, 0, 1000)
            else:
                self.attack_active = False
                self.attack_start_time = 0
                self.attack_end_time = 0


            # Get PINN prediction (without passing actions directly)
            current_time_torch = torch.tensor([[self.current_time]], dtype=torch.float32)
            with torch.no_grad():
                prediction = self.pinn_model(current_time_torch)
            evcs_vars = prediction[:, 2 * self.NUM_BUSES:].detach().numpy().flatten()
            
            # Get current state observation
            current_state = self.get_observation(evcs_vars)
            # print("Current state: 1 ", current_state.shape)

            # Apply actions
            if self.attack_active and self.attack_start_time <= self.time_step_counter <= self.attack_end_time:
                # print("Attack active with Defender")
                for t in range(self.attack_start_time, self.attack_end_time):
                    current_state = self.apply_defender_actions(current_state, defender_action)
                    current_state = self.apply_attack_effects(current_state, attacker_action, self.target_evcs, self.attack_duration)
                    
            # print("Current state: 2 ", current_state.shape)

            
            # Update state
            self.state = current_state

            self.voltage_deviations = np.abs(self.state[:self.NUM_EVCS] - self.WAC_VOUT_SETPOINT)

            max_deviations= np.max(self.voltage_deviations)

            self.rewards = self.calculate_rewards(self.voltage_deviations)

            
            # Check if episode is done
            done = self.time_step_counter >= 1000 or np.sum(self.voltage_deviations >= 0.3) >= self.target_evcs.sum()
            
            # Get info
            info = self.get_info(self.voltage_deviations, self.target_evcs, self.attack_duration, self.rewards)
            
            return self.state, self.rewards, done, False, info

        except Exception as e:
            print(f"Error in step: {e}")
            return (
                np.zeros(25, dtype=np.float32),
                {'dqn': 0.0, 'attacker': 0.0, 'defender': 0.0},
                True,
                False,
                self.get_info(self.voltage_deviations, self.target_evcs, self.attack_duration, self.rewards)
            )

    def process_dqn_action(self, dqn_action):
        """Process and validate DQN action."""
        if dqn_action is not None and not isinstance(dqn_action, list):
            dqn_action = self.decode_action(dqn_action)
        else:
            dqn_action = [0] * (self.NUM_EVCS + 1)
        return dqn_action
    

    def decode_dqn_action(self, action_scalar):
        """Decode DQN action scalar into target EVCSs and duration."""
        try:
            # Handle action_scalar if it's a list or array
            if isinstance(action_scalar, (list, np.ndarray)):
                action_scalar = action_scalar[0]
            
            # Convert to integer
            action_idx = int(float(action_scalar))
            
            # Initialize output array
            action = np.zeros(self.NUM_EVCS + 1, dtype=np.int32)
            
            # Extract target selection and duration
            target_value = action_idx // self.NUM_DURATION
            duration_value = action_idx % self.NUM_DURATION
            
            # Convert target_value to binary representation
            for i in range(self.NUM_EVCS):
                action[i] = (target_value >> i) & 1
            
            # Set duration
            action[-1] = duration_value
            
            # Debug output without using self.debug
            # print(f"Decoded action {action_scalar}:")
            # print(f"Target value: {target_value}")
            # print(f"Duration value: {duration_value}")
            # print(f"Target EVCSs: {action[:-1]}")
            # print(f"Duration: {action[-1]}")
            
            return action.astype(np.int32)  # Ensure int32 type
            
        except Exception as e:
            print(f"Error decoding DQN action: {e}")
            print(f"Action scalar: {action_scalar}, Type: {type(action_scalar)}")
            return np.zeros(self.NUM_EVCS + 1, dtype=np.int32)

    def decode_action(self, action_array):
        """Convert DQN action array directly into target format."""
        try:
            # Convert input to numpy array
            action_array = np.asarray(action_array)
            
            # If it's a single-element array or list
            if action_array.size == 1:
                return self.decode_dqn_action(action_array.item())
            
            # If it's already the right shape
            elif action_array.shape == (self.NUM_EVCS + 1,):
                return action_array.astype(np.int32)
                
            else:
                print(f"Warning: Unexpected action shape: {action_array.shape}")
                return np.zeros(self.NUM_EVCS + 1, dtype=np.int32)
                
        except Exception as e:
            print(f"Error in decode_action: {e}")
            print(f"Input action: {action_array}, Type: {type(action_array)}")
            return np.zeros(self.NUM_EVCS + 1, dtype=np.int32)

    def update_attack_parameters(self, dqn_action):
        """Update attack-related parameters."""
        self.target_evcs = dqn_action[:self.NUM_EVCS]
        self.attack_duration = dqn_action[-1] 
        self.attack_start_time = self.time_step_counter

    # def apply_actions(self, dqn_action, sac_attacker_action, sac_defender_action):
    #     """Apply all agent actions and return new state."""
    #     # Get PINN prediction
    #     prediction = self.pinn_model(tf.constant([[self.current_time]], dtype=tf.float32))
    #     evcs_vars = prediction[:, 2 * self.NUM_BUSES:].numpy()[0]
    #     new_state = self.get_observation(evcs_vars)
        
    #     # Apply attack if active
    #     if self.attack_start_time <= self.time_step_counter <= self.attack_end_time:
    #         new_state = self.apply_attack_effects(new_state, sac_attacker_action, self.target_evcs, self.attack_duration)
        
    #     # Apply defender actions
    #     new_state = self.apply_defender_actions(new_state, sac_defender_action)
        
    #     return new_state

    def apply_attack_effects(self, state, attack_action, target_evcs, attack_duration):
        """Apply attack effects with proper shape handling."""
        try:
            # Ensure inputs are numpy arrays
            state = np.array(state)
            attack_action = np.array(attack_action).flatten()
            target_evcs = np.array(target_evcs, dtype=bool)
            
            # Ensure attack_action has correct shape (NUM_EVCS * 2)
            if attack_action.shape[0] != self.NUM_EVCS * 2:
                attack_action = np.resize(attack_action, self.NUM_EVCS * 2)
            
            # First, clip the attack actions to the allowed magnitude
            attack_action = np.clip(attack_action, -self.attack_magnitude, self.attack_magnitude)
            
            # Split attack actions for voltage and current
            voltage_attacks = attack_action[:self.NUM_EVCS]
            current_attacks = attack_action[self.NUM_EVCS:]
            
            # Apply attacks only to targeted EVCSs
            state_copy = state.copy()
            for i in range(self.NUM_EVCS):
                if target_evcs[i]:

                    state_copy[i] = np.clip(
                        state[i] + voltage_attacks[i],
                        self.voltage_limits[0],  # Using system voltage limits
                        self.voltage_limits[1]
                    )

                                        # Apply current attack (with proper clipping)
                    current_idx = 3 * self.NUM_EVCS + i
                    state_copy[current_idx] = np.clip(
                        state[current_idx] + current_attacks[i],
                        self.current_limits[0],
                        self.current_limits[1]
                    )

                                    # Validate physics constraints before applying
                    if self.validate_physics(state_copy):
                        self.state = state_copy
                    else:
                        self.attack_rejections[i] += 1
                        print(f"Attack on EVCS {i} rejected: Physics constraints violated")
                        print(f"#Rejection rate for EVCS {i}: {self.attack_rejections[i]}/{self.attack_attempts[i]} " 
                            f"({(self.attack_rejections[i]/self.attack_attempts[i]*100):.2f}%)")
            
                    # Apply voltage attack (with proper clipping)

                    

            
            return state_copy
            
        except Exception as e:
            print(f"Error in apply_attack_effects: {e}")
            return state

    def apply_defender_actions(self, state, defender_action):
        """Apply defender actions (WAC parameter adjustments)."""
        try:
            # Ensure inputs are numpy arrays
            state = np.array(state)
            defender_action = np.array(defender_action)
            
            # Reshape defender_action if it's not the right shape
            if defender_action.size != self.NUM_EVCS * 2:
                print(f"Warning: Reshaping defender_action from shape {defender_action.shape}")
                # Ensure we have enough values by repeating or truncating
                defender_action = np.resize(defender_action, (self.NUM_EVCS * 2,))
            
            # Split defender action into Kp and Ki adjustments
            kp_adjustments = defender_action[:self.NUM_EVCS]
            ki_adjustments = defender_action[self.NUM_EVCS:self.NUM_EVCS * 2]

            # Ensure adjustments have correct shapes
            kp_adjustments = np.resize(kp_adjustments, (self.NUM_EVCS,))
            ki_adjustments = np.resize(ki_adjustments, (self.NUM_EVCS,))

            # Update WAC parameters with saturation
            self.kp_vout = np.clip(
                self.WAC_KP_VOUT_DEFAULT + kp_adjustments,
                0.0,
                0.5
            )
            self.ki_vout = np.clip(
                self.WAC_KI_VOUT_DEFAULT + ki_adjustments,
                0.0,
                0.5
            )

            # Extract voltages from state
            v_out = state[:self.NUM_EVCS]
            v_dc = state[self.NUM_EVCS:2*self.NUM_EVCS]
            
            # Calculate voltage error
            self.voltage_error = self.WAC_VDC_SETPOINT - v_dc

            # Apply WAC control
            self.modulation_index = self.apply_wac_control(
                self.voltage_error, 
                self.kp_vout, 
                self.ki_vout
            )

            # Ensure modulation_index_system is properly shaped
            modulation_system = np.zeros(self.NUM_EVCS)
            if isinstance(self.modulation_index_system, tuple):
                modulation_system += self.modulation_index_system[0]
            else:
                modulation_system += self.modulation_index_system

            # Calculate new voltage
            v_out = v_dc * (self.modulation_index + modulation_system)

            # Apply control effect to state
            state_copy = state.copy()
            state_copy[:self.NUM_EVCS] = np.clip(
                v_out,
                0.8,
                1.2
            )

            return state_copy

        except Exception as e:
            print(f"Error in apply_defender_actions: {e}")
            print(f"Defender action shape: {defender_action.shape}")
            print(f"State shape: {state.shape}")
            return state

    def get_info(self, voltage_deviations, target_evcs, attack_duration, rewards):
        """Get current environment info."""
        try:
            # Calculate voltage deviations
            # v_out = self.state[:self.NUM_EVCS]  # Output voltages
            # voltage_deviations = np.abs(v_out - self.WAC_VOUT_SETPOINT)
            self.cumulative_deviation = np.sum(voltage_deviations)
            self.voltage_deviations = voltage_deviations
            
            # Calculate rewards
            # rewards = self.calculate_rewards(self.state)
            
            return {
                'time_step': self.time_step_counter,
                'current_time': self.current_time,
                'attack_active': self.attack_active,
                'cumulative_deviation': float(self.cumulative_deviation),
                'target_evcs': target_evcs.astype(int),
                'attack_duration': int(attack_duration),
                'voltage_deviations': self.voltage_deviations,
                'rewards': rewards
            }
            
        except Exception as e:
            print(f"Error in get_info: {e}")
            return {
                'time_step': self.time_step_counter,
                'current_time': self.current_time,
                'attack_active': False,
                'cumulative_deviation': 0.0,
                'target_evcs': [0] * self.NUM_EVCS,
                'attack_duration': 0,
                'voltage_deviations': [0.0] * self.NUM_EVCS,
                'rewards': 0.0
            }

    
    def get_observation(self, evcs_vars):
        """Get observation from EVCS variables."""
        v_out_values = []
        soc_values   = []
        v_dc_values  = []
        i_out_values = []
        i_dc_values  = []
        for i in range(self.NUM_EVCS):
            v_dc = np.exp(evcs_vars[i * 18 + 2])  # DC link voltage
            v_out = np.exp(evcs_vars[i * 18 + 4])  # Output voltage
            soc = evcs_vars[i * 18 + 9]  # State of Charge
            i_out = evcs_vars[i * 18 + 16]  # Output current
            i_dc = evcs_vars[i * 18 + 17]  # DC current

            v_dc_values.append(v_dc)
            v_out_values.append(v_out)
            soc_values.append(soc)
            i_out_values.append(i_out)
            i_dc_values.append(i_dc)

        return np.concatenate([v_out_values, soc_values, v_dc_values, i_out_values, i_dc_values])

    def control_saturation(self, value, v_min, v_max):
        """
        Saturate control signal between minimum and maximum values.

        Parameters:
        value (float): The control signal value to be saturated.
        v_min (float): The minimum allowable value for the control signal.
        v_max (float): The maximum allowable value for the control signal.

        Returns:
        float: The saturated control signal value, constrained between v_min and v_max.
               Returns 0.0 if an error occurs during the operation.
        """
        """Saturate control signal between minimum and maximum values."""
        try:
            return np.clip(value, v_min, v_max)
        except Exception as e:
            print(f"Error in control_saturation: {e}")
            return 0.0

    # def update_wac_parameters(self, defender_action):
    #     """Update WAC parameters from defender actions."""
    #     try:
    #         # Split defender action into Kp and Ki adjustments
    #         kp_adjustments = defender_action[:self.NUM_EVCS]
    #         ki_adjustments = defender_action[self.NUM_EVCS:]

    #         # Update WAC parameters with saturation
    #         self.kp_vout = self.control_saturation(
    #             kp_adjustments,
    #             0.0,  # WAC parameters should be positive
    #             self.CONTROL_MAX
    #         )
    #         self.ki_vout = self.control_saturation(
    #             ki_adjustments,
    #             0.0,
    #             self.CONTROL_MAX
    #         )
            
    #     except Exception as e:
    #         print(f"Error in update_wac_parameters: {e}")
