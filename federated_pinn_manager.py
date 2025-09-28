#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import copy
import time
from pinn_optimizer import LSTMPINNChargingOptimizer, LSTMPINNConfig
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FederatedPINNConfig:
    """Configuration for federated PINN training"""
    # Federated learning parameters
    num_distribution_systems: int = 6
    local_epochs: int = 50  # Local training epochs per round
    global_rounds: int = 20  # Number of federated rounds
    aggregation_method: str = 'fedavg'  # 'fedavg', 'weighted_avg', 'median'
    communication_rounds: int = 0  # Track completed communication rounds
    
    # Local training parameters
    local_batch_size: int = 32
    local_learning_rate: float = 0.001
    
    # Privacy and security
    differential_privacy: bool = True
    noise_multiplier: float = 0.1
    max_grad_norm: float = 1.0
    
    # Model sharing frequency
    share_frequency: int = 5  # Share models every N local epochs

class AnomalyDetector:
    """Anomaly detection for EVCS inputs and RL attack patterns"""
    
    def __init__(self, config: LSTMPINNConfig):
        self.config = config
        
        # Physical constraint thresholds
        self.max_realistic_power = 100.0  # kW per EVCS station
        self.max_realistic_load_change = 50.0  # kW per time step
        self.max_system_load = 500.0  # MW total system load
        
        # Attack detection parameters
        self.attack_detection_window = 10  # Time steps to analyze
        self.load_change_threshold = 25.0  # kW sudden change threshold
        self.frequency_deviation_threshold = 0.5  # Hz
        
        # Historical data for anomaly detection
        self.load_history = []
        self.power_history = []
        self.voltage_history = []
        
    def validate_physical_constraints(self, inputs: Dict) -> Tuple[bool, Dict]:
        """Validate inputs against physical constraints"""
        violations = {}
        is_valid = True
        
        # SOC constraints
        soc = inputs.get('soc', 0.5)
        if not (0.0 <= soc <= 1.0):
            violations['soc'] = f"SOC {soc:.3f} outside valid range [0.0, 1.0]"
            is_valid = False
        
        # Voltage constraints (per unit)
        grid_voltage = inputs.get('grid_voltage', 1.0)
        if not (0.85 <= grid_voltage <= 1.15):
            violations['voltage'] = f"Grid voltage {grid_voltage:.3f} pu outside safe range [0.85, 1.15]"
            is_valid = False
        
        # Frequency constraints
        frequency = inputs.get('grid_frequency', 60.0)
        if not (59.0 <= frequency <= 61.0):
            violations['frequency'] = f"Frequency {frequency:.3f} Hz outside normal range [59.0, 61.0]"
            is_valid = False
        
        # Power demand constraints
        demand_factor = inputs.get('demand_factor', 0.7)
        if not (0.0 <= demand_factor <= 2.0):
            violations['demand'] = f"Demand factor {demand_factor:.3f} outside realistic range [0.0, 2.0]"
            is_valid = False
        
        # Load factor constraints
        load_factor = inputs.get('load_factor', 0.7)
        if not (0.1 <= load_factor <= 1.5):
            violations['load_factor'] = f"Load factor {load_factor:.3f} outside realistic range [0.1, 1.5]"
            is_valid = False
        
        return is_valid, violations
    
    def detect_attack_patterns(self, current_load: float, system_id: int) -> Tuple[bool, str]:
        """Detect potential attack patterns in load injection"""
        # Update history
        self.load_history.append((time.time(), current_load, system_id))
        
        # Keep only recent history
        current_time = time.time()
        self.load_history = [(t, load, sys_id) for t, load, sys_id in self.load_history 
                           if current_time - t < 60.0]  # Keep last 60 seconds
        
        # Check for sudden large load changes
        if len(self.load_history) >= 2:
            recent_loads = [load for _, load, sys_id in self.load_history[-5:] if sys_id == system_id]
            
            if len(recent_loads) >= 2:
                load_change = abs(recent_loads[-1] - recent_loads[-2])
                
                # Detect unrealistic load injection
                if current_load > self.max_system_load:
                    return True, f"Unrealistic load injection: {current_load:.1f} MW exceeds system capacity"
                
                # Detect sudden large changes
                if load_change > self.load_change_threshold:
                    return True, f"Suspicious load change: {load_change:.1f} kW in single step"
                
                # Detect oscillating patterns (potential attack)
                if len(recent_loads) >= 4:
                    changes = [recent_loads[i+1] - recent_loads[i] for i in range(len(recent_loads)-1)]
                    if all(abs(change) > 10.0 for change in changes):
                        sign_changes = sum(1 for i in range(len(changes)-1) 
                                         if changes[i] * changes[i+1] < 0)
                        if sign_changes >= 2:
                            return True, "Oscillating load pattern detected (potential attack)"
        
        return False, "Normal operation"
    
    def sanitize_inputs(self, inputs: Dict) -> Dict:
        """Sanitize inputs to prevent extreme values"""
        sanitized = inputs.copy()
        
        # Clamp values to safe ranges
        sanitized['soc'] = np.clip(inputs.get('soc', 0.5), 0.05, 0.95)
        sanitized['grid_voltage'] = np.clip(inputs.get('grid_voltage', 1.0), 0.90, 1.10)
        sanitized['grid_frequency'] = np.clip(inputs.get('grid_frequency', 60.0), 59.5, 60.5)
        sanitized['demand_factor'] = np.clip(inputs.get('demand_factor', 0.7), 0.1, 1.5)
        sanitized['load_factor'] = np.clip(inputs.get('load_factor', 0.7), 0.2, 1.2)
        sanitized['urgency_factor'] = np.clip(inputs.get('urgency_factor', 1.0), 0.5, 2.0)
        
        return sanitized

class GradualAttackController:
    """Controller for gradual, stealthy attack injection"""
    
    def __init__(self, max_attack_magnitude: float = 50.0):
        self.max_attack_magnitude = max_attack_magnitude
        self.attack_step_size = 2.0  # kW per step
        self.attack_delay = 5.0  # seconds between steps
        
        # Attack state
        self.current_attack_level = 0.0
        self.target_attack_level = 0.0
        self.last_attack_time = 0.0
        self.attack_active = False
        
    def start_gradual_attack(self, target_magnitude: float, attack_type: str = 'increase'):
        """Start a gradual attack with specified target magnitude"""
        # Limit attack magnitude to realistic values
        self.target_attack_level = np.clip(target_magnitude, 0.0, self.max_attack_magnitude)
        self.attack_active = True
        self.last_attack_time = time.time()
        
        if attack_type == 'decrease':
            self.target_attack_level = -self.target_attack_level
        
        print(f"🎯 Starting gradual {attack_type} attack: target {self.target_attack_level:.1f} kW")
    
    def update_attack_level(self) -> float:
        """Update attack level gradually"""
        if not self.attack_active:
            return 0.0
        
        current_time = time.time()
        
        # Check if enough time has passed for next step
        if current_time - self.last_attack_time >= self.attack_delay:
            # Calculate step direction
            if abs(self.current_attack_level - self.target_attack_level) > self.attack_step_size:
                if self.current_attack_level < self.target_attack_level:
                    self.current_attack_level += self.attack_step_size
                else:
                    self.current_attack_level -= self.attack_step_size
                
                self.last_attack_time = current_time
            else:
                # Attack target reached
                self.current_attack_level = self.target_attack_level
                self.attack_active = False
                print(f"✅ Gradual attack completed: {self.current_attack_level:.1f} kW")
        
        return self.current_attack_level
    
    def stop_attack(self):
        """Stop current attack"""
        self.attack_active = False
        self.current_attack_level = 0.0
        self.target_attack_level = 0.0

class FederatedPINNManager:
    """Manager for federated PINN training across distribution systems"""
    
    def __init__(self, config: FederatedPINNConfig):
        self.config = config
        self.pinn_config = LSTMPINNConfig()
        
        # Local PINN models for each distribution system
        self.local_models: Dict[int, LSTMPINNChargingOptimizer] = {}
        self.global_model: Optional[LSTMPINNChargingOptimizer] = None
        
        # Anomaly detectors for each system
        self.anomaly_detectors: Dict[int, AnomalyDetector] = {}
        
        # Attack controllers for each system
        self.attack_controllers: Dict[int, GradualAttackController] = {}
        
        # Training metrics
        self.training_history = {
            'local_losses': {},
            'global_losses': [],
            'communication_rounds': 0
        }
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize local models and detectors for each distribution system"""
        print(f"🏗️ Initializing {self.config.num_distribution_systems} federated PINN systems...")
        
        for sys_id in range(1, self.config.num_distribution_systems + 1):
            # Create local PINN model
            local_config = copy.deepcopy(self.pinn_config)
            local_config.epochs = self.config.local_epochs
            local_config.learning_rate = self.config.local_learning_rate
            
            self.local_models[sys_id] = LSTMPINNChargingOptimizer(local_config, always_train=False)
            
            # Create anomaly detector
            self.anomaly_detectors[sys_id] = AnomalyDetector(self.pinn_config)
            
            # Create attack controller
            self.attack_controllers[sys_id] = GradualAttackController()
            
            # Initialize training history
            self.training_history['local_losses'][sys_id] = []
            
            print(f"  ✅ System {sys_id}: Local PINN + Anomaly Detector + Attack Controller")
        
        # Initialize global model
        self.global_model = LSTMPINNChargingOptimizer(self.pinn_config, always_train=False)
        print("✅ Global federated model initialized")
    
    def train_local_model(self, sys_id: int, local_data: Union[np.ndarray, Tuple], 
                         n_samples: int = 1000) -> Dict:
        """Train local PINN model for specific distribution system with enhanced data"""
        if sys_id not in self.local_models:
            raise ValueError(f"System {sys_id} not initialized")
        
        print(f"🔬 Training local PINN for Distribution System {sys_id}...")
        
        # Get local model
        local_model = self.local_models[sys_id]
        
        # Check if we have enhanced data (sequences, targets) or simplified data
        if isinstance(local_data, tuple) and len(local_data) == 2:
            # Enhanced data: (sequences, targets) from PhysicsDataGenerator
            sequences, targets = local_data
            print(f"  📊 Using ENHANCED training data: {len(sequences)} sequences with {sequences.shape[-1]} features")
            print(f"  🎯 Target ranges: V={targets[:, 0].min():.1f}-{targets[:, 0].max():.1f}, I={targets[:, 1].min():.1f}-{targets[:, 1].max():.1f}, P={targets[:, 2].min():.2f}-{targets[:, 2].max():.2f}")
            
            # Train with enhanced data using the data generator
            if hasattr(local_model, 'data_generator'):
                # Update the data generator with our enhanced data
                local_model.data_generator = None  # Clear existing generator
                local_model._training_data = (sequences, targets)  # Store enhanced data
                training_metrics = local_model.train_model(n_samples=len(sequences))
            else:
                # Fallback to regular training
                training_metrics = local_model.train_model(n_samples=n_samples)
        else:
            # Simplified data: use regular training
            print(f"  📊 Using simplified training data: {local_data.shape}")
            training_metrics = local_model.train_model(n_samples=n_samples)
        
        # Store training history
        self.training_history['local_losses'][sys_id].append(training_metrics)
        
        # Save local model
        model_path = f'federated_pinn_system_{sys_id}.pth'
        local_model.save_model(model_path)
        
        print(f"  ✅ System {sys_id} training completed, model saved to {model_path}")
        return training_metrics
    
    def federated_averaging(self) -> Dict:
        """Perform federated averaging of local models"""
        print("🔄 Performing federated averaging...")
        
        if not self.local_models:
            raise ValueError("No local models to aggregate")
        
        # Get state dictionaries from all local models
        local_state_dicts = []
        for sys_id, model in self.local_models.items():
            local_state_dicts.append(model.model.state_dict())
        
        # Perform federated averaging
        global_state_dict = {}
        
        for key in local_state_dicts[0].keys():
            # Get all tensors for this parameter
            tensors = [state_dict[key] for state_dict in local_state_dicts]
            
            # Check if all tensors have the same shape and are numeric
            if all(t.shape == tensors[0].shape for t in tensors):
                # Convert to float if needed for averaging
                if tensors[0].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    # For integer tensors (like indices), take the first one without averaging
                    global_state_dict[key] = tensors[0].clone()
                else:
                    # Average parameters across all local models for float tensors
                    if self.config.aggregation_method == 'fedavg':
                        # Ensure tensors are float for averaging
                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = torch.stack(float_tensors).mean(dim=0)
                        # Convert back to original dtype if needed
                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)
                    
                    elif self.config.aggregation_method == 'weighted_avg':
                        # Weight by number of samples (simplified - equal weights for now)
                        weights = torch.ones(len(local_state_dicts)) / len(local_state_dicts)
                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = sum(
                            w * tensor for w, tensor in zip(weights, float_tensors)
                        )
                        # Convert back to original dtype if needed
                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)
                    
                    elif self.config.aggregation_method == 'median':
                        float_tensors = [t.float() if t.dtype != torch.float32 else t for t in tensors]
                        global_state_dict[key] = torch.median(torch.stack(float_tensors), dim=0)[0]
                        # Convert back to original dtype if needed
                        if tensors[0].dtype != torch.float32:
                            global_state_dict[key] = global_state_dict[key].to(tensors[0].dtype)
            else:
                # If shapes don't match, use the first tensor
                global_state_dict[key] = tensors[0].clone()
        
        # Update global model
        self.global_model.model.load_state_dict(global_state_dict)
        
        # Update communication rounds
        self.config.communication_rounds += 1
        
        print(f"  ✅ Federated averaging completed (Round {self.config.communication_rounds})")
        
        return {
            'round': self.config.communication_rounds,
            'aggregation_method': self.config.aggregation_method,
            'num_participants': len(local_state_dicts)
        }
    
    def distribute_global_model(self):
        """Distribute global model to all local systems"""
        print("📡 Distributing global model to local systems...")
        
        global_state_dict = self.global_model.model.state_dict()
        
        for sys_id, local_model in self.local_models.items():
            local_model.model.load_state_dict(copy.deepcopy(global_state_dict))
            print(f"  ✅ System {sys_id}: Global model distributed")
    
    def optimize_with_constraints(self, sys_id: int, inputs: Dict) -> Tuple[Dict, bool, str]:
        """Optimize charging parameters with anomaly detection and constraints"""
        if sys_id not in self.local_models:
            raise ValueError(f"System {sys_id} not initialized")
        
        # Step 1: Validate physical constraints
        detector = self.anomaly_detectors[sys_id]
        is_valid, violations = detector.validate_physical_constraints(inputs)
        
        if not is_valid:
            violation_msg = "; ".join(violations.values())
            return {}, False, f"Physical constraint violations: {violation_msg}"
        
        # Step 2: Detect attack patterns
        current_load = inputs.get('demand_factor', 0.7) * 100.0  # Convert to kW
        is_attack, attack_msg = detector.detect_attack_patterns(current_load, sys_id)
        
        if is_attack:
            return {}, False, f"Attack detected: {attack_msg}"
        
        # Step 3: Sanitize inputs
        sanitized_inputs = detector.sanitize_inputs(inputs)
        
        # Step 4: Apply gradual attack if active
        attack_controller = self.attack_controllers[sys_id]
        attack_level = attack_controller.update_attack_level()
        
        if attack_level != 0.0:
            # Apply gradual attack to demand factor
            sanitized_inputs['demand_factor'] += attack_level / 100.0  # Convert kW to factor
            sanitized_inputs['demand_factor'] = np.clip(sanitized_inputs['demand_factor'], 0.1, 1.5)
        
        # Step 5: Optimize using local PINN model
        local_model = self.local_models[sys_id]
        
        try:
            v_ref, i_ref, p_ref = local_model.optimize_references(sanitized_inputs)
            
            # Validate output constraints
            if not (300.0 <= v_ref <= 500.0 and 50.0 <= i_ref <= 150.0 and 15.0 <= p_ref <= 75.0):
                return {}, False, f"Output constraints violated: V={v_ref:.1f}V, I={i_ref:.1f}A, P={p_ref:.1f}kW"
            
            results = {
                'voltage_ref': v_ref,
                'current_ref': i_ref,
                'power_ref': p_ref,
                'system_id': sys_id,
                'attack_level': attack_level,
                'sanitized': sanitized_inputs != inputs
            }
            
            return results, True, "Optimization successful"
            
        except Exception as e:
            return {}, False, f"Optimization failed: {str(e)}"
    
    def start_coordinated_attack(self, target_systems: List[int], attack_magnitude: float, 
                                attack_type: str = 'increase'):
        """Start coordinated gradual attack across multiple systems"""
        print(f"🎯 Starting coordinated {attack_type} attack on systems {target_systems}")
        print(f"   Target magnitude: {attack_magnitude:.1f} kW per system")
        
        for sys_id in target_systems:
            if sys_id in self.attack_controllers:
                self.attack_controllers[sys_id].start_gradual_attack(attack_magnitude, attack_type)
    
    def stop_all_attacks(self):
        """Stop all active attacks"""
        print("🛑 Stopping all active attacks...")
        for sys_id, controller in self.attack_controllers.items():
            controller.stop_attack()
        print("✅ All attacks stopped")
    
    def get_federated_status(self) -> Dict:
        """Get status of federated training and attacks"""
        status = {
            'num_systems': len(self.local_models),
            'communication_rounds': self.config.communication_rounds,
            'active_attacks': {},
            'anomaly_detections': {},
            'training_status': {}
        }
        
        for sys_id in self.local_models.keys():
            # Attack status
            controller = self.attack_controllers[sys_id]
            status['active_attacks'][sys_id] = {
                'active': controller.attack_active,
                'current_level': controller.current_attack_level,
                'target_level': controller.target_attack_level
            }
            
            # Training status
            if sys_id in self.training_history['local_losses']:
                losses = self.training_history['local_losses'][sys_id]
                status['training_status'][sys_id] = {
                    'training_rounds': len(losses),
                    'last_loss': losses[-1] if losses else None
                }
        
        return status
    
    def save_federated_models(self, base_path: str = 'federated_models'):
        """Save all federated models"""
        print(f"💾 Saving federated models to {base_path}/...")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save global model
        if self.global_model:
            self.global_model.save_model(f'{base_path}/global_federated_pinn.pth')
        
        # Save local models
        for sys_id, model in self.local_models.items():
            model.save_model(f'{base_path}/local_pinn_system_{sys_id}.pth')
        
        print("✅ All federated models saved")
    
    def load_federated_models(self, base_path: str = 'federated_models'):
        """Load all federated models"""
        print(f"📂 Loading federated models from {base_path}/...")
        
        try:
            # Load global model
            if self.global_model:
                self.global_model.load_model(f'{base_path}/global_federated_pinn.pth')
                print("  ✅ Global model loaded")
            
            # Load local models
            for sys_id, model in self.local_models.items():
                model.load_model(f'{base_path}/local_pinn_system_{sys_id}.pth')
                print(f"  ✅ System {sys_id} model loaded")
            
            print("✅ All federated models loaded successfully")
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Failed to load federated models: {e}")
            return False
