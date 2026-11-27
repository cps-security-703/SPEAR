#!/usr/bin/env python3

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
from global_federated_optimizer import GlobalFederatedOptimizer, CustomerRequest, EVCSStationStatus
from enhanced_rl_attack_system import ConstrainedRLAttackSystem
from hierarchical_cosimulation import HierarchicalCoSimulation
import warnings
warnings.filterwarnings('ignore')

class FederatedEVCSSystem:


    def __init__(self, num_distribution_systems: int = 6):
        self.num_distribution_systems = num_distribution_systems


        federated_config = FederatedPINNConfig(
            num_distribution_systems=num_distribution_systems,
            local_epochs=50,
            global_rounds=10,
            aggregation_method='fedavg'
        )
        self.federated_pinn_manager = FederatedPINNManager(federated_config)


        self.global_optimizer = GlobalFederatedOptimizer(self.federated_pinn_manager)


        self.rl_attack_system = ConstrainedRLAttackSystem(
            num_systems=num_distribution_systems,
            pinn_optimizers=self.federated_pinn_manager.local_models
        )


        self.cosim = None


        self.federated_training_complete = False
        self.system_ready = False

    def train_federated_pinn_models(self, n_samples_per_system: int = 1000) -> Dict:

        print(" Starting Federated PINN Training...")
        print(f"   Training {self.num_distribution_systems} local models")
        print(f"   {n_samples_per_system} samples per system")

        training_results = {
            'local_training_results': {},
            'federated_rounds': [],
            'global_model_performance': {},
            'training_time': 0.0
        }

        start_time = time.time()


        print("\n Phase 1: Local PINN Training")
        for sys_id in range(1, self.num_distribution_systems + 1):
            print(f"\n Training System {sys_id} PINN...")


            local_data = self._generate_system_specific_data(sys_id, n_samples_per_system)


            local_result = self.federated_pinn_manager.train_local_model(
                sys_id, local_data, n_samples_per_system
            )

            training_results['local_training_results'][sys_id] = local_result
            print(f"##System {sys_id} training completed")


        print("\n## Phase 2: Federated Averaging")
        for round_num in range(self.federated_pinn_manager.config.global_rounds):
            print(f"\n## Federated Round {round_num + 1}/{self.federated_pinn_manager.config.global_rounds}")


            round_result = self.federated_pinn_manager.federated_averaging()


            self.federated_pinn_manager.distribute_global_model()


            if round_num < self.federated_pinn_manager.config.global_rounds - 1:
                print("   ## Additional local training with global model...")
                for sys_id in range(1, self.num_distribution_systems + 1):

                    local_data = self._generate_system_specific_data(sys_id, n_samples_per_system // 2)
                    self.federated_pinn_manager.train_local_model(
                        sys_id, local_data, n_samples_per_system // 2
                    )

            training_results['federated_rounds'].append(round_result)
            print(f"##Round {round_num + 1} completed")


        print("\n## Phase 3: Model Evaluation")
        global_performance = self._evaluate_federated_models()
        training_results['global_model_performance'] = global_performance

        training_time = time.time() - start_time
        training_results['training_time'] = training_time


        self.federated_pinn_manager.save_federated_models('federated_models')

        self.federated_training_complete = True
        print(f"\n##Federated PINN Training Complete!")
        print(f"   Total training time: {training_time:.1f} seconds")
        print(f"   Models saved to 'federated_models/' directory")

        return training_results

    def _generate_system_specific_data(self, sys_id: int, n_samples: int) -> np.ndarray:


        system_characteristics = {
            1: {'load_factor': 1.2, 'voltage_stability': 0.95, 'area_type': 'urban'},
            2: {'load_factor': 0.9, 'voltage_stability': 0.92, 'area_type': 'highway'},
            3: {'load_factor': 1.0, 'voltage_stability': 0.94, 'area_type': 'mixed'},
            4: {'load_factor': 1.3, 'voltage_stability': 0.91, 'area_type': 'industrial'},
            5: {'load_factor': 1.1, 'voltage_stability': 0.96, 'area_type': 'commercial'},
            6: {'load_factor': 0.8, 'voltage_stability': 0.97, 'area_type': 'residential'}
        }

        char = system_characteristics.get(sys_id, system_characteristics[1])


        data = np.random.randn(n_samples, 10)


        data[:, 3] *= char['load_factor']
        data[:, 1] = np.clip(data[:, 1] * 0.05 + char['voltage_stability'], 0.9, 1.1)

        return data

    def _evaluate_federated_models(self) -> Dict:

        evaluation_results = {}

        for sys_id in range(1, self.num_distribution_systems + 1):

            test_inputs = {
                'soc': 0.5,
                'grid_voltage': 0.98,
                'grid_frequency': 60.0,
                'demand_factor': 0.7,
                'voltage_priority': 0.1,
                'urgency_factor': 1.0,
                'current_time': 12.0,
                'bus_distance': 2.0,
                'load_factor': 0.8
            }

            result, success, message = self.federated_pinn_manager.optimize_with_constraints(
                sys_id, test_inputs
            )

            evaluation_results[sys_id] = {
                'optimization_success': success,
                'message': message,
                'sample_output': result if success else {}
            }

        return evaluation_results

    def setup_hierarchical_cosimulation(self) -> bool:

        if not self.federated_training_complete:
            print("## Federated training must be completed first")
            return False

        print(" Setting up Hierarchical Co-simulation with Federated PINN...")


        self.cosim = HierarchicalCoSimulation()
        self.cosim.total_duration = 480.0


        dss_files = ["ieee34Mod1.dss"] * 6
        bus_configs = [4, 9, 13, 5, 10, 7]

        for i in range(self.num_distribution_systems):
            sys_id = i + 1
            self.cosim.add_distribution_system(sys_id, dss_files[i], bus_configs[i])


        enhanced_evcs_configs = [

            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25},
                {'bus': '844', 'max_power': 300, 'num_ports': 6},
                {'bus': '860', 'max_power': 200, 'num_ports': 4},
                {'bus': '840', 'max_power': 400, 'num_ports': 10},
            ],

            [
                {'bus': '890', 'max_power': 800, 'num_ports': 20},
                {'bus': '844', 'max_power': 350, 'num_ports': 8},
                {'bus': '860', 'max_power': 250, 'num_ports': 5},
                {'bus': '840', 'max_power': 300, 'num_ports': 6},
            ],

            [
                {'bus': '890', 'max_power': 600, 'num_ports': 15},
                {'bus': '844', 'max_power': 250, 'num_ports': 5},
                {'bus': '860', 'max_power': 200, 'num_ports': 4},
                {'bus': '840', 'max_power': 350, 'num_ports': 7},
            ],

            [
                {'bus': '890', 'max_power': 1200, 'num_ports': 30},
                {'bus': '844', 'max_power': 400, 'num_ports': 8},
                {'bus': '860', 'max_power': 300, 'num_ports': 6},
                {'bus': '840', 'max_power': 500, 'num_ports': 10},
            ],

            [
                {'bus': '890', 'max_power': 700, 'num_ports': 18},
                {'bus': '844', 'max_power': 280, 'num_ports': 6},
                {'bus': '860', 'max_power': 220, 'num_ports': 4},
                {'bus': '840', 'max_power': 320, 'num_ports': 6},
            ],

            [
                {'bus': '890', 'max_power': 400, 'num_ports': 10},
                {'bus': '844', 'max_power': 200, 'num_ports': 4},
                {'bus': '860', 'max_power': 150, 'num_ports': 3},
                {'bus': '840', 'max_power': 250, 'num_ports': 5},
            ]
        ]


        for i, (sys_id, dist_info) in enumerate(self.cosim.distribution_systems.items()):
            if i < len(enhanced_evcs_configs):
                dist_info['system'].add_ev_charging_stations(enhanced_evcs_configs[i])


                if hasattr(dist_info['system'], 'cms') and dist_info['system'].cms:
                    if hasattr(dist_info['system'].cms, 'pinn_optimizer'):

                        federated_model = self.federated_pinn_manager.local_models[sys_id]
                        dist_info['system'].cms.pinn_optimizer = federated_model
                        dist_info['system'].cms.pinn_trained = True
                        print(f"##System {sys_id}: Federated PINN model integrated")

        self.system_ready = True
        print("##Hierarchical co-simulation setup complete with federated PINN models")
        return True

    def run_federated_simulation_with_attacks(self, load_profile_data: Tuple = None) -> Dict:

        if not self.system_ready:
            print("## System not ready. Complete setup first.")
            return {}

        print("\n##Starting Federated EVCS Simulation with Constrained RL Attacks...")


        if load_profile_data:
            times, load_multipliers = load_profile_data
            self.cosim.transmission_system.set_load_profile(times, load_multipliers)
            print(f"##Load profile set: {len(times)} time points")


        simulation_results = {
            'federated_optimization_calls': 0,
            'attack_events': [],
            'constraint_violations': [],
            'customer_allocations': [],
            'system_performance': {},
            'simulation_time': 0.0
        }

        start_time = time.time()


        customer_requests = self._generate_sample_customer_requests()
        for request in customer_requests:
            self.global_optimizer.add_customer_request(request)


        print("\n Processing customer requests with federated optimization...")
        allocation_results = self.global_optimizer.process_customer_queue()
        simulation_results['customer_allocations'] = allocation_results

        print(f"##Processed {len(allocation_results)} customer allocations")


        print("\n Generating constrained RL attacks...")


        system_states = {}
        load_contexts = {}

        for sys_id in range(1, self.num_distribution_systems + 1):
            system_states[sys_id] = {
                'grid_voltage': np.random.uniform(0.95, 1.05),
                'frequency': np.random.uniform(59.8, 60.2),
                'current_load': np.random.uniform(50.0, 200.0)
            }

            load_contexts[sys_id] = {
                'avg_load': np.random.uniform(0.5, 0.9),
                'peak_load': np.random.uniform(0.8, 1.2),
                'load_variance': np.random.uniform(0.1, 0.3)
            }


        coordinated_attacks = self.rl_attack_system.generate_coordinated_attacks(
            system_states, load_contexts
        )

        if coordinated_attacks:
            print(f"##Generated {len(coordinated_attacks)} coordinated attacks")


            execution_results = self.rl_attack_system.execute_coordinated_attacks(coordinated_attacks)
            simulation_results['attack_events'] = execution_results['execution_logs']
            simulation_results['constraint_violations'] = execution_results['constraint_violations']

            print(f" Attack execution: {execution_results['successful_attacks']} successful, "
                  f"{execution_results['failed_attacks']} failed")


        print("\n## Running hierarchical co-simulation...")


        attack_scenarios = []
        for attack_event in simulation_results['attack_events']:
            attack_scenario = {
                'start_time': 60.0,
                'duration': 30.0,
                'target_system': attack_event.get('system_id', 1),
                'type': 'demand_increase',
                'magnitude': 25.0,
                'target_percentage': 80,
                'rl_generated': True,
                'constrained': True,
                'stealth_score': attack_event.get('stealth_score', 0.8)
            }
            attack_scenarios.append(attack_scenario)


        cosim_results = self.cosim.run_hierarchical_simulation(attack_scenarios=attack_scenarios)


        for _ in range(10):
            attack_status = self.rl_attack_system.update_all_attacks()
            time.sleep(0.1)


        self.rl_attack_system.stop_all_attacks()


        simulation_results['system_performance'] = self._collect_performance_metrics()

        simulation_time = time.time() - start_time
        simulation_results['simulation_time'] = simulation_time

        print(f"\n##Federated simulation complete!")
        print(f"   Simulation time: {simulation_time:.1f} seconds")
        print(f"   Customer allocations: {len(allocation_results)}")
        print(f"   Attack events: {len(simulation_results['attack_events'])}")
        print(f"   Constraint violations: {len(simulation_results['constraint_violations'])}")

        return simulation_results

    def _generate_sample_customer_requests(self) -> List[CustomerRequest]:

        requests = []

        for i in range(5):
            request = CustomerRequest(
                customer_id=f"CUST_{i+1:03d}",
                requested_power=np.random.uniform(20.0, 60.0),
                requested_duration=np.random.uniform(0.5, 2.0),
                soc_current=np.random.uniform(0.2, 0.8),
                soc_target=np.random.uniform(0.8, 0.95),
                urgency_level=np.random.randint(1, 6),
                max_travel_distance=np.random.uniform(2.0, 10.0),
                arrival_time=np.random.uniform(0.0, 1.0)
            )
            requests.append(request)

        return requests

    def _collect_performance_metrics(self) -> Dict:

        metrics = {
            'federated_pinn_status': self.federated_pinn_manager.get_federated_status(),
            'global_optimizer_status': self.global_optimizer.get_global_system_status(),
            'attack_system_status': self.rl_attack_system.get_global_attack_status(),
            'load_balance_score': 0.0,
            'customer_satisfaction_score': 0.0,
            'grid_stability_score': 0.0
        }


        global_status = self.global_optimizer.get_global_system_status()
        metrics['load_balance_score'] = global_status['global_metrics'].get('load_balance_score', 0.0)


        total_allocations = len(self.global_optimizer.customer_queue)
        successful_allocations = sum(1 for result in self.global_optimizer.customer_queue if True)
        metrics['customer_satisfaction_score'] = successful_allocations / max(1, total_allocations)


        active_attacks = sum(1 for sys_id, agent in self.rl_attack_system.attack_agents.items()
                           if agent.attack_controller.attack_active)
        metrics['grid_stability_score'] = max(0.0, 1.0 - (active_attacks / self.num_distribution_systems))

        return metrics

    def get_comprehensive_status(self) -> Dict:

        return {
            'federated_training_complete': self.federated_training_complete,
            'system_ready': self.system_ready,
            'num_distribution_systems': self.num_distribution_systems,
            'federated_pinn_status': self.federated_pinn_manager.get_federated_status(),
            'global_optimizer_status': self.global_optimizer.get_global_system_status(),
            'attack_system_status': self.rl_attack_system.get_global_attack_status(),
            'cosim_ready': self.cosim is not None
        }

    def demonstrate_federated_features(self):

        print("\n## Demonstrating Federated EVCS Features...")


        print("\n## Federated PINN Training:")
        print("   ##Each distribution system trains its own PINN model")
        print("   ##Models share knowledge through federated averaging")
        print("   ##Privacy-preserving distributed learning")


        print("\n## Constrained RL Attacks:")
        print("   ##Physical constraint validation (max 50kW injection)")
        print("   ##Gradual attack injection (5-second steps)")
        print("   ##Anomaly detection and stealth scoring")
        print("   ##Realistic attack patterns instead of 3000-10000MW")


        print("\n## Global Federated Optimization:")
        print("   ##Customer redirection across distribution systems")
        print("   ##Load balancing using federated PINN insights")
        print("   ##Queue management and wait time optimization")


        print("\n## System Integration:")
        print("   ##Hierarchical co-simulation with federated models")
        print("   ##Real-time constraint validation")
        print("   ##Coordinated attack detection and mitigation")

        print("\n## Federated EVCS System Ready for Advanced Simulations!")
