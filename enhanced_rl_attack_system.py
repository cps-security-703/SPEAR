#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import time
from federated_pinn_manager import GradualAttackController, AnomalyDetector
from pinn_optimizer import LSTMPINNConfig
import warnings
warnings.filterwarnings('ignore')

class PhysicalConstraintValidator:
    """Enhanced physical constraint validation for RL attacks"""
    
    def __init__(self):
        # Realistic system limits
        self.max_single_load_injection = 50.0  # kW per injection
        self.max_system_load = 500.0  # MW total system load
        self.max_load_change_rate = 10.0  # kW per second
        self.max_frequency_deviation = 0.5  # Hz
        self.max_voltage_deviation = 0.1  # pu
        
        # Attack detection thresholds
        self.suspicious_load_threshold = 100.0  # kW
        self.rapid_change_threshold = 25.0  # kW in single step
        self.oscillation_detection_window = 10  # time steps
        
        # Historical tracking
        self.load_history = {}  # sys_id -> [(time, load), ...]
        self.attack_history = {}  # sys_id -> [attack_events, ...]
        
    def validate_attack_parameters(self, sys_id: int, attack_params: Dict) -> Tuple[bool, Dict, str]:
        """Validate RL attack parameters against physical constraints"""
        violations = {}
        
        # Extract attack parameters
        magnitude = attack_params.get('magnitude', 0.0)
        attack_type = attack_params.get('type', 'demand_increase')
        duration = attack_params.get('duration', 30.0)
        target_percentage = attack_params.get('target_percentage', 80)
        
        # Validate magnitude
        if magnitude > self.max_single_load_injection:
            violations['magnitude'] = f"Attack magnitude {magnitude:.1f} kW exceeds limit {self.max_single_load_injection:.1f} kW"
        
        # Validate rate of change
        if duration > 0:
            change_rate = magnitude / duration
            if change_rate > self.max_load_change_rate:
                violations['change_rate'] = f"Change rate {change_rate:.2f} kW/s exceeds limit {self.max_load_change_rate:.2f} kW/s"
        
        # Validate target percentage
        if target_percentage > 90:
            violations['target_percentage'] = f"Target percentage {target_percentage}% too high (max 90%)"
        
        # Check historical patterns
        is_suspicious, suspicion_reason = self._check_attack_history(sys_id, attack_params)
        if is_suspicious:
            violations['pattern'] = suspicion_reason
        
        is_valid = len(violations) == 0
        message = "Valid attack parameters" if is_valid else f"Constraint violations: {'; '.join(violations.values())}"
        
        return is_valid, violations, message
    
    def _check_attack_history(self, sys_id: int, attack_params: Dict) -> Tuple[bool, str]:
        """Check if attack pattern is suspicious based on history"""
        current_time = time.time()
        
        if sys_id not in self.attack_history:
            self.attack_history[sys_id] = []
        
        # Add current attack to history
        self.attack_history[sys_id].append({
            'time': current_time,
            'magnitude': attack_params.get('magnitude', 0.0),
            'type': attack_params.get('type', 'unknown')
        })
        
        # Keep only recent history (last 5 minutes)
        self.attack_history[sys_id] = [
            attack for attack in self.attack_history[sys_id]
            if current_time - attack['time'] < 300.0
        ]
        
        recent_attacks = self.attack_history[sys_id]
        
        # Check for rapid succession attacks
        if len(recent_attacks) > 5:
            return True, "Too many attacks in short period"
        
        # Check for oscillating pattern
        if len(recent_attacks) >= 3:
            types = [attack['type'] for attack in recent_attacks[-3:]]
            if len(set(types)) == 1 and types[0] in ['demand_increase', 'demand_decrease']:
                return True, "Repetitive attack pattern detected"
        
        # Check for escalating magnitude
        if len(recent_attacks) >= 2:
            magnitudes = [attack['magnitude'] for attack in recent_attacks]
            if all(magnitudes[i] < magnitudes[i+1] for i in range(len(magnitudes)-1)):
                if magnitudes[-1] > magnitudes[0] * 2:
                    return True, "Escalating attack magnitude detected"
        
        return False, "Normal pattern"
    
    def sanitize_attack_parameters(self, attack_params: Dict) -> Dict:
        """Sanitize attack parameters to meet physical constraints"""
        sanitized = attack_params.copy()
        
        # Clamp magnitude to realistic values
        sanitized['magnitude'] = np.clip(attack_params.get('magnitude', 0.0), 0.0, self.max_single_load_injection)
        
        # Ensure minimum duration for gradual attacks
        sanitized['duration'] = max(attack_params.get('duration', 30.0), 15.0)
        
        # Limit target percentage
        sanitized['target_percentage'] = min(attack_params.get('target_percentage', 80), 90)
        
        # Add gradual injection flag
        sanitized['gradual_injection'] = True
        sanitized['injection_steps'] = max(3, int(sanitized['duration'] / 5.0))  # 5-second steps
        
        return sanitized

class EnhancedRLAttackAgent:
    """Enhanced RL attack agent with physical constraints and gradual injection"""
    
    def __init__(self, sys_id: int, pinn_optimizer=None):
        self.sys_id = sys_id
        self.pinn_optimizer = pinn_optimizer
        
        # Physical constraint validator
        self.constraint_validator = PhysicalConstraintValidator()
        
        # Gradual attack controller
        self.attack_controller = GradualAttackController(max_attack_magnitude=50.0)
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(LSTMPINNConfig())
        
        # RL agent state
        self.current_state = None
        self.action_history = []
        self.reward_history = []
        
        # Attack strategy parameters
        self.stealth_factor = 0.8  # Higher = more stealthy
        self.effectiveness_factor = 0.6  # Higher = more effective
        self.learning_rate = 0.001
        
    def generate_constrained_attack(self, system_state: Dict, load_context: Dict) -> Dict:
        """Generate attack that respects physical constraints"""
        
        # Analyze current system state
        grid_voltage = system_state.get('grid_voltage', 1.0)
        frequency = system_state.get('frequency', 60.0)
        current_load = system_state.get('current_load', 0.0)
        load_factor = load_context.get('avg_load', 0.7)
        
        # Determine attack strategy based on system state
        attack_strategy = self._select_attack_strategy(system_state, load_context)
        
        # Generate base attack parameters
        base_attack = self._generate_base_attack(attack_strategy, system_state)
        
        # Validate against physical constraints
        is_valid, violations, message = self.constraint_validator.validate_attack_parameters(
            self.sys_id, base_attack
        )
        
        if not is_valid:
            print(f"  System {self.sys_id}: Attack constrained - {message}")
            # Sanitize parameters to meet constraints
            base_attack = self.constraint_validator.sanitize_attack_parameters(base_attack)
        
        # Convert to gradual attack
        gradual_attack = self._convert_to_gradual_attack(base_attack, system_state)
        
        # Add stealth features
        stealthy_attack = self._add_stealth_features(gradual_attack, system_state)
        
        # Validate final attack
        final_is_valid, final_violations, final_message = self.constraint_validator.validate_attack_parameters(
            self.sys_id, stealthy_attack
        )
        
        if final_is_valid:
            print(f"✅ System {self.sys_id}: Generated constrained attack - {stealthy_attack['type']}")
            print(f"   Magnitude: {stealthy_attack['magnitude']:.1f} kW over {stealthy_attack['duration']:.1f}s")
            print(f"   Stealth score: {stealthy_attack.get('stealth_score', 0.5):.2f}")
        else:
            print(f"❌ System {self.sys_id}: Attack generation failed - {final_message}")
            return {}
        
        return stealthy_attack
    
    def _select_attack_strategy(self, system_state: Dict, load_context: Dict) -> str:
        """Select attack strategy based on system conditions"""
        grid_voltage = system_state.get('grid_voltage', 1.0)
        frequency = system_state.get('frequency', 60.0)
        load_factor = load_context.get('avg_load', 0.7)
        
        # Low load periods - increase demand to stress system
        if load_factor < 0.5:
            return 'demand_increase'
        
        # High load periods - decrease demand to cause instability
        elif load_factor > 0.8:
            return 'demand_decrease'
        
        # Voltage issues - exploit voltage instability
        elif abs(grid_voltage - 1.0) > 0.05:
            return 'voltage_manipulation'
        
        # Frequency issues - exploit frequency instability
        elif abs(frequency - 60.0) > 0.2:
            return 'frequency_manipulation'
        
        # Default - oscillating demand
        else:
            return 'oscillating_demand'
    
    def _generate_base_attack(self, strategy: str, system_state: Dict) -> Dict:
        """Generate base attack parameters for strategy"""
        current_load = system_state.get('current_load', 0.0)
        
        if strategy == 'demand_increase':
            magnitude = np.random.uniform(15.0, 35.0)  # Realistic increase
            duration = np.random.uniform(20.0, 45.0)
            attack_type = 'demand_increase'
            
        elif strategy == 'demand_decrease':
            magnitude = np.random.uniform(10.0, 25.0)  # Smaller decrease
            duration = np.random.uniform(15.0, 30.0)
            attack_type = 'demand_decrease'
            
        elif strategy == 'voltage_manipulation':
            magnitude = np.random.uniform(20.0, 40.0)
            duration = np.random.uniform(25.0, 50.0)
            attack_type = 'voltage_manipulation'
            
        elif strategy == 'frequency_manipulation':
            magnitude = np.random.uniform(18.0, 35.0)
            duration = np.random.uniform(30.0, 60.0)
            attack_type = 'frequency_manipulation'
            
        else:  # oscillating_demand
            magnitude = np.random.uniform(12.0, 28.0)
            duration = np.random.uniform(40.0, 80.0)
            attack_type = 'oscillating_demand'
        
        return {
            'type': attack_type,
            'magnitude': magnitude,
            'duration': duration,
            'target_percentage': np.random.randint(70, 85),
            'strategy': strategy
        }
    
    def _convert_to_gradual_attack(self, base_attack: Dict, system_state: Dict) -> Dict:
        """Convert attack to gradual injection pattern"""
        gradual_attack = base_attack.copy()
        
        # Calculate injection steps
        total_duration = base_attack['duration']
        step_duration = 5.0  # 5-second steps
        num_steps = max(3, int(total_duration / step_duration))
        
        # Calculate magnitude per step
        total_magnitude = base_attack['magnitude']
        step_magnitude = total_magnitude / num_steps
        
        # Ensure step magnitude is within limits
        max_step = self.constraint_validator.max_single_load_injection / num_steps
        step_magnitude = min(step_magnitude, max_step)
        
        gradual_attack.update({
            'gradual_injection': True,
            'injection_steps': num_steps,
            'step_duration': step_duration,
            'step_magnitude': step_magnitude,
            'total_steps': num_steps,
            'current_step': 0
        })
        
        return gradual_attack
    
    def _add_stealth_features(self, attack: Dict, system_state: Dict) -> Dict:
        """Add stealth features to make attack less detectable"""
        stealthy_attack = attack.copy()
        
        # Add random delays between steps
        base_delay = attack.get('step_duration', 5.0)
        random_delay = np.random.uniform(0.8, 1.2) * base_delay
        stealthy_attack['step_duration'] = random_delay
        
        # Add magnitude variation to appear natural
        base_magnitude = attack.get('step_magnitude', 5.0)
        magnitude_variation = np.random.uniform(0.9, 1.1)
        stealthy_attack['step_magnitude'] = base_magnitude * magnitude_variation
        
        # Calculate stealth score
        stealth_score = self._calculate_stealth_score(stealthy_attack, system_state)
        stealthy_attack['stealth_score'] = stealth_score
        
        # Add timing randomization
        stealthy_attack['timing_jitter'] = np.random.uniform(0.5, 2.0)
        
        # Add load masking (blend with normal load variations)
        stealthy_attack['load_masking'] = True
        stealthy_attack['masking_factor'] = np.random.uniform(0.7, 0.9)
        
        return stealthy_attack
    
    def _calculate_stealth_score(self, attack: Dict, system_state: Dict) -> float:
        """Calculate stealth score (0-1, higher = more stealthy)"""
        score = 1.0
        
        # Penalize large magnitudes
        magnitude = attack.get('step_magnitude', 0.0)
        if magnitude > 20.0:
            score -= 0.3
        elif magnitude > 10.0:
            score -= 0.1
        
        # Reward gradual injection
        if attack.get('gradual_injection', False):
            score += 0.2
        
        # Reward longer durations (more gradual)
        duration = attack.get('duration', 30.0)
        if duration > 60.0:
            score += 0.2
        elif duration > 30.0:
            score += 0.1
        
        # Penalize rapid changes
        num_steps = attack.get('injection_steps', 1)
        if num_steps < 3:
            score -= 0.2
        
        # Consider system state
        current_load = system_state.get('current_load', 0.0)
        if current_load > 100.0:  # High load - easier to hide
            score += 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def execute_gradual_attack(self, attack: Dict) -> Dict:
        """Execute gradual attack with step-by-step injection"""
        if not attack.get('gradual_injection', False):
            return {'success': False, 'message': 'Not a gradual attack'}
        
        # Start gradual attack using controller
        self.attack_controller.start_gradual_attack(
            attack['magnitude'], 
            attack['type']
        )
        
        execution_log = {
            'attack_id': f"attack_{self.sys_id}_{int(time.time())}",
            'system_id': self.sys_id,
            'start_time': time.time(),
            'total_steps': attack['injection_steps'],
            'step_magnitude': attack['step_magnitude'],
            'stealth_score': attack.get('stealth_score', 0.5),
            'execution_steps': [],
            'success': True
        }
        
        print(f"🎯 System {self.sys_id}: Starting gradual attack execution")
        print(f"   Type: {attack['type']}")
        print(f"   Total magnitude: {attack['magnitude']:.1f} kW")
        print(f"   Steps: {attack['injection_steps']}")
        print(f"   Stealth score: {attack.get('stealth_score', 0.5):.2f}")
        
        return execution_log
    
    def update_attack_progress(self) -> Tuple[float, bool]:
        """Update attack progress and return current injection level"""
        current_level = self.attack_controller.update_attack_level()
        is_active = self.attack_controller.attack_active
        
        return current_level, is_active
    
    def stop_attack(self):
        """Stop current attack"""
        self.attack_controller.stop_attack()
        print(f"🛑 System {self.sys_id}: Attack stopped")
    
    def get_attack_status(self) -> Dict:
        """Get current attack status"""
        return {
            'system_id': self.sys_id,
            'attack_active': self.attack_controller.attack_active,
            'current_level': self.attack_controller.current_attack_level,
            'target_level': self.attack_controller.target_attack_level,
            'action_history_length': len(self.action_history),
            'reward_history_length': len(self.reward_history)
        }

class ConstrainedRLAttackSystem:
    """Coordinated RL attack system with physical constraints"""
    
    def __init__(self, num_systems: int = 6, pinn_optimizers: Dict = None):
        self.num_systems = num_systems
        self.pinn_optimizers = pinn_optimizers or {}
        
        # Create enhanced RL agents for each system
        self.attack_agents: Dict[int, EnhancedRLAttackAgent] = {}
        for sys_id in range(1, num_systems + 1):
            pinn_opt = self.pinn_optimizers.get(sys_id, None)
            self.attack_agents[sys_id] = EnhancedRLAttackAgent(sys_id, pinn_opt)
        
        # Global constraint validator
        self.global_validator = PhysicalConstraintValidator()
        
        # Coordination parameters
        self.max_concurrent_attacks = 3
        self.coordination_delay = 10.0  # seconds between coordinated attacks
        self.last_coordination_time = 0.0
        
    def generate_coordinated_attacks(self, system_states: Dict, load_contexts: Dict) -> List[Dict]:
        """Generate coordinated attacks across multiple systems"""
        current_time = time.time()
        
        # Check coordination timing
        if current_time - self.last_coordination_time < self.coordination_delay:
            return []
        
        print(f"\n🎯 Generating coordinated constrained attacks...")
        
        # Select systems for attack (limit concurrent attacks)
        available_systems = list(range(1, self.num_systems + 1))
        target_systems = np.random.choice(
            available_systems, 
            size=min(self.max_concurrent_attacks, len(available_systems)), 
            replace=False
        )
        
        coordinated_attacks = []
        
        for sys_id in target_systems:
            if sys_id in self.attack_agents:
                system_state = system_states.get(sys_id, {})
                load_context = load_contexts.get(sys_id, {})
                
                # Generate constrained attack for this system
                attack = self.attack_agents[sys_id].generate_constrained_attack(
                    system_state, load_context
                )
                
                if attack:  # Only add valid attacks
                    attack['system_id'] = sys_id
                    attack['coordination_time'] = current_time
                    coordinated_attacks.append(attack)
        
        self.last_coordination_time = current_time
        
        print(f"✅ Generated {len(coordinated_attacks)} coordinated attacks")
        return coordinated_attacks
    
    def execute_coordinated_attacks(self, attacks: List[Dict]) -> Dict:
        """Execute coordinated attacks with constraint validation"""
        execution_results = {
            'total_attacks': len(attacks),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'execution_logs': [],
            'constraint_violations': []
        }
        
        for attack in attacks:
            sys_id = attack['system_id']
            
            if sys_id in self.attack_agents:
                try:
                    # Execute gradual attack
                    execution_log = self.attack_agents[sys_id].execute_gradual_attack(attack)
                    
                    if execution_log.get('success', False):
                        execution_results['successful_attacks'] += 1
                        execution_results['execution_logs'].append(execution_log)
                    else:
                        execution_results['failed_attacks'] += 1
                        execution_results['constraint_violations'].append({
                            'system_id': sys_id,
                            'reason': execution_log.get('message', 'Unknown failure')
                        })
                        
                except Exception as e:
                    execution_results['failed_attacks'] += 1
                    execution_results['constraint_violations'].append({
                        'system_id': sys_id,
                        'reason': f"Execution error: {str(e)}"
                    })
        
        print(f"⚡ Attack execution complete:")
        print(f"   Successful: {execution_results['successful_attacks']}")
        print(f"   Failed: {execution_results['failed_attacks']}")
        
        return execution_results
    
    def update_all_attacks(self) -> Dict:
        """Update all active attacks and return status"""
        attack_status = {
            'active_attacks': 0,
            'total_injection': 0.0,
            'system_status': {}
        }
        
        for sys_id, agent in self.attack_agents.items():
            current_level, is_active = agent.update_attack_progress()
            
            attack_status['system_status'][sys_id] = {
                'active': is_active,
                'injection_level': current_level,
                'stealth_score': agent.get_attack_status().get('stealth_score', 0.5)
            }
            
            if is_active:
                attack_status['active_attacks'] += 1
                attack_status['total_injection'] += abs(current_level)
        
        return attack_status
    
    def stop_all_attacks(self):
        """Stop all active attacks"""
        print("🛑 Stopping all coordinated attacks...")
        
        for sys_id, agent in self.attack_agents.items():
            agent.stop_attack()
        
        print("✅ All attacks stopped")
    
    def get_global_attack_status(self) -> Dict:
        """Get comprehensive attack system status"""
        status = {
            'num_systems': self.num_systems,
            'max_concurrent_attacks': self.max_concurrent_attacks,
            'last_coordination_time': self.last_coordination_time,
            'system_agents': {}
        }
        
        for sys_id, agent in self.attack_agents.items():
            status['system_agents'][sys_id] = agent.get_attack_status()
        
        return status
