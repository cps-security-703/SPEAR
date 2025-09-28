#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from federated_pinn_manager import FederatedPINNManager, FederatedPINNConfig
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CustomerRequest:
    """Customer charging request"""
    customer_id: str
    requested_power: float  # kW
    requested_duration: float  # hours
    soc_current: float  # 0.0 - 1.0
    soc_target: float  # 0.0 - 1.0
    urgency_level: int  # 1-5 (5 = highest urgency)
    preferred_location: Optional[int] = None  # Preferred distribution system
    max_travel_distance: float = 10.0  # km
    arrival_time: float = 0.0  # hours from now

@dataclass
class EVCSStationStatus:
    """EVCS station status"""
    system_id: int
    station_id: str
    available_ports: int
    total_ports: int
    current_load: float  # kW
    max_capacity: float  # kW
    queue_length: int
    average_wait_time: float  # minutes
    location_distance: float  # km from customer
    voltage_stability: float  # 0.0 - 1.0
    grid_health: float  # 0.0 - 1.0

class GlobalFederatedOptimizer:
    """Global optimizer that combines federated PINN models for system-wide optimization"""
    
    def __init__(self, federated_manager: FederatedPINNManager):
        self.federated_manager = federated_manager
        self.customer_queue: List[CustomerRequest] = []
        self.station_status: Dict[int, List[EVCSStationStatus]] = {}
        
        # Global optimization parameters
        self.load_balancing_weight = 0.4
        self.customer_satisfaction_weight = 0.3
        self.grid_stability_weight = 0.2
        self.economic_efficiency_weight = 0.1
        
        # System-wide constraints
        self.max_system_load = 500.0  # MW
        self.min_voltage_stability = 0.9
        self.max_queue_length = 10
        
        # Initialize station status
        self._initialize_station_status()
    
    def _initialize_station_status(self):
        """Initialize EVCS station status for all distribution systems"""
        # EVCS configuration from the original system
        evcs_configs = [
            # Distribution System 1 - Urban Area
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS1_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS1_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS1_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS1_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS1_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS1_ST6'},
            ],
            # Distribution System 2 - Highway Corridor
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS2_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS2_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS2_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS2_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS2_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS2_ST6'},
            ],
            # Distribution System 3 - Mixed Area
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS3_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS3_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS3_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS3_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS3_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS3_ST6'},
            ],
            # Distribution System 4 - Industrial Zone
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS4_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS4_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS4_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS4_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS4_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS4_ST6'},
            ],
            # Distribution System 5 - Commercial District
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS5_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS5_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS5_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS5_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS5_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS5_ST6'},
            ],
            # Distribution System 6 - Residential Complex
            [
                {'bus': '890', 'max_power': 1000, 'num_ports': 25, 'station_id': 'DS6_ST1'},
                {'bus': '844', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS6_ST2'},
                {'bus': '860', 'max_power': 200, 'num_ports': 4, 'station_id': 'DS6_ST3'},
                {'bus': '840', 'max_power': 400, 'num_ports': 10, 'station_id': 'DS6_ST4'},
                {'bus': '848', 'max_power': 250, 'num_ports': 5, 'station_id': 'DS6_ST5'},
                {'bus': '830', 'max_power': 300, 'num_ports': 6, 'station_id': 'DS6_ST6'},
            ]
        ]
        
        # Initialize station status for each distribution system
        for sys_id in range(1, 7):  # 6 distribution systems
            self.station_status[sys_id] = []
            
            if sys_id <= len(evcs_configs):
                config_list = evcs_configs[sys_id - 1]
                
                for i, config in enumerate(config_list):
                    # Simulate realistic station status
                    utilization = np.random.uniform(0.3, 0.8)  # 30-80% utilization
                    
                    status = EVCSStationStatus(
                        system_id=sys_id,
                        station_id=config['station_id'],
                        available_ports=int(config['num_ports'] * (1 - utilization)),
                        total_ports=config['num_ports'],
                        current_load=config['max_power'] * utilization,
                        max_capacity=config['max_power'],
                        queue_length=int(np.random.poisson(2)),  # Poisson distribution for queue
                        average_wait_time=np.random.uniform(5, 30),  # 5-30 minutes
                        location_distance=np.random.uniform(0.5, 8.0),  # 0.5-8 km
                        voltage_stability=np.random.uniform(0.92, 0.98),
                        grid_health=np.random.uniform(0.90, 0.99)
                    )
                    
                    self.station_status[sys_id].append(status)
    
    def add_customer_request(self, request: CustomerRequest):
        """Add customer charging request to queue"""
        self.customer_queue.append(request)
        print(f"📋 Customer {request.customer_id} added to queue")
        print(f"   Requested: {request.requested_power:.1f} kW for {request.requested_duration:.1f} hours")
        print(f"   SOC: {request.soc_current:.1%} → {request.soc_target:.1%}")
        print(f"   Urgency: {request.urgency_level}/5")
    
    def optimize_customer_allocation(self, request: CustomerRequest) -> Tuple[int, str, Dict]:
        """Optimize customer allocation across federated systems"""
        print(f"\n🎯 Optimizing allocation for customer {request.customer_id}...")
        
        # Get all available stations across all systems
        candidate_stations = []
        
        for sys_id, stations in self.station_status.items():
            for station in stations:
                if (station.available_ports > 0 and 
                    station.location_distance <= request.max_travel_distance and
                    station.current_load + request.requested_power <= station.max_capacity):
                    
                    candidate_stations.append((sys_id, station))
        
        if not candidate_stations:
            return -1, "", {"error": "No available stations found"}
        
        # Score each candidate station using federated PINN insights
        best_score = -1
        best_system = -1
        best_station = ""
        allocation_details = {}
        
        for sys_id, station in candidate_stations:
            score = self._calculate_allocation_score(request, sys_id, station)
            
            if score > best_score:
                best_score = score
                best_system = sys_id
                best_station = station.station_id
                
                # Get federated PINN optimization for this allocation
                pinn_inputs = {
                    'soc': request.soc_current,
                    'grid_voltage': station.voltage_stability,
                    'grid_frequency': 60.0,
                    'demand_factor': (station.current_load + request.requested_power) / station.max_capacity,
                    'voltage_priority': 1.0 - station.voltage_stability,
                    'urgency_factor': request.urgency_level / 2.5,  # Scale to 0.4-2.0
                    'current_time': time.time() / 3600.0 % 24,  # Hour of day
                    'bus_distance': station.location_distance,
                    'load_factor': station.current_load / station.max_capacity
                }
                
                # Get optimization from federated PINN
                pinn_result, success, message = self.federated_manager.optimize_with_constraints(
                    sys_id, pinn_inputs
                )
                
                allocation_details = {
                    'score': score,
                    'station_info': {
                        'system_id': sys_id,
                        'station_id': station.station_id,
                        'available_ports': station.available_ports,
                        'current_load': station.current_load,
                        'max_capacity': station.max_capacity,
                        'distance': station.location_distance,
                        'wait_time': station.average_wait_time,
                        'voltage_stability': station.voltage_stability
                    },
                    'pinn_optimization': pinn_result if success else {},
                    'pinn_success': success,
                    'pinn_message': message,
                    'estimated_charging_time': self._estimate_charging_time(request, pinn_result if success else {}),
                    'total_cost': self._estimate_cost(request, station, pinn_result if success else {})
                }
        
        if best_system != -1:
            # Update station status (simulate allocation)
            self._update_station_allocation(best_system, best_station, request)
            
            print(f"✅ Optimal allocation found:")
            print(f"   System: {best_system}, Station: {best_station}")
            print(f"   Score: {best_score:.3f}")
            print(f"   Distance: {allocation_details['station_info']['distance']:.1f} km")
            print(f"   Wait time: {allocation_details['station_info']['wait_time']:.1f} min")
            
            if allocation_details['pinn_success']:
                pinn_opt = allocation_details['pinn_optimization']
                print(f"   PINN Optimization: {pinn_opt.get('voltage_ref', 0):.1f}V, "
                      f"{pinn_opt.get('current_ref', 0):.1f}A, {pinn_opt.get('power_ref', 0):.1f}kW")
        
        return best_system, best_station, allocation_details
    
    def _calculate_allocation_score(self, request: CustomerRequest, sys_id: int, 
                                  station: EVCSStationStatus) -> float:
        """Calculate allocation score for customer-station pair"""
        score = 0.0
        
        # Distance factor (closer is better)
        distance_score = max(0, 1 - station.location_distance / request.max_travel_distance)
        score += distance_score * 0.25
        
        # Availability factor (more available ports is better)
        availability_score = station.available_ports / station.total_ports
        score += availability_score * 0.20
        
        # Load factor (less loaded is better for faster charging)
        load_score = 1 - (station.current_load / station.max_capacity)
        score += load_score * 0.20
        
        # Queue factor (shorter queue is better)
        queue_score = max(0, 1 - station.queue_length / self.max_queue_length)
        score += queue_score * 0.15
        
        # Voltage stability factor
        voltage_score = station.voltage_stability
        score += voltage_score * 0.10
        
        # Grid health factor
        grid_score = station.grid_health
        score += grid_score * 0.10
        
        # Power matching factor (can the station handle the request?)
        power_match_score = min(1.0, station.max_capacity / request.requested_power)
        score += power_match_score * 0.10
        
        # Urgency factor (high urgency customers get priority at better stations)
        if request.urgency_level >= 4:
            score += 0.1  # Bonus for high urgency
        
        return score
    
    def _estimate_charging_time(self, request: CustomerRequest, pinn_result: Dict) -> float:
        """Estimate charging time based on PINN optimization"""
        if not pinn_result:
            # Fallback estimation
            energy_needed = (request.soc_target - request.soc_current) * 50.0  # Assume 50 kWh battery
            return energy_needed / request.requested_power
        
        # Use PINN optimized power
        optimized_power = pinn_result.get('power_ref', request.requested_power)
        energy_needed = (request.soc_target - request.soc_current) * 50.0
        
        # Account for charging curve (slower at high SOC)
        if request.soc_target > 0.8:
            optimized_power *= 0.7  # Reduce power for high SOC
        
        return energy_needed / optimized_power
    
    def _estimate_cost(self, request: CustomerRequest, station: EVCSStationStatus, 
                      pinn_result: Dict) -> float:
        """Estimate charging cost"""
        # Base electricity rate ($/kWh)
        base_rate = 0.15
        
        # Peak hour multiplier
        current_hour = time.time() / 3600.0 % 24
        if 17 <= current_hour <= 21:  # Peak hours
            rate_multiplier = 1.5
        elif 22 <= current_hour <= 6:  # Off-peak hours
            rate_multiplier = 0.8
        else:
            rate_multiplier = 1.0
        
        # Station utilization multiplier
        utilization = station.current_load / station.max_capacity
        utilization_multiplier = 1.0 + (utilization * 0.3)  # Up to 30% increase for high utilization
        
        # Energy cost
        energy_needed = (request.soc_target - request.soc_current) * 50.0  # kWh
        energy_cost = energy_needed * base_rate * rate_multiplier * utilization_multiplier
        
        # Service fee
        service_fee = 2.0  # Base service fee
        
        # Urgency premium
        if request.urgency_level >= 4:
            urgency_premium = energy_cost * 0.2  # 20% premium for high urgency
        else:
            urgency_premium = 0.0
        
        return energy_cost + service_fee + urgency_premium
    
    def _update_station_allocation(self, sys_id: int, station_id: str, request: CustomerRequest):
        """Update station status after allocation"""
        for station in self.station_status[sys_id]:
            if station.station_id == station_id:
                station.available_ports -= 1
                station.current_load += request.requested_power
                station.queue_length += 1
                break
    
    def process_customer_queue(self) -> List[Dict]:
        """Process all customers in queue and return allocation results"""
        print(f"\n🔄 Processing {len(self.customer_queue)} customers in queue...")
        
        allocation_results = []
        
        # Sort queue by urgency and arrival time
        sorted_queue = sorted(self.customer_queue, 
                            key=lambda x: (-x.urgency_level, x.arrival_time))
        
        for request in sorted_queue:
            sys_id, station_id, details = self.optimize_customer_allocation(request)
            
            result = {
                'customer_id': request.customer_id,
                'allocated_system': sys_id,
                'allocated_station': station_id,
                'allocation_details': details,
                'success': sys_id != -1
            }
            
            allocation_results.append(result)
        
        # Clear processed queue
        self.customer_queue.clear()
        
        return allocation_results
    
    def rebalance_load_across_systems(self) -> Dict:
        """Rebalance load across federated systems using global optimization"""
        print("\n⚖️ Performing global load rebalancing...")
        
        # Calculate current load distribution
        system_loads = {}
        system_capacities = {}
        
        for sys_id, stations in self.station_status.items():
            total_load = sum(station.current_load for station in stations)
            total_capacity = sum(station.max_capacity for station in stations)
            
            system_loads[sys_id] = total_load
            system_capacities[sys_id] = total_capacity
        
        # Identify overloaded and underloaded systems
        overloaded_systems = []
        underloaded_systems = []
        
        for sys_id in system_loads:
            utilization = system_loads[sys_id] / system_capacities[sys_id]
            
            if utilization > 0.85:  # Over 85% utilization
                overloaded_systems.append((sys_id, utilization))
            elif utilization < 0.50:  # Under 50% utilization
                underloaded_systems.append((sys_id, utilization))
        
        # Generate rebalancing recommendations
        rebalancing_actions = []
        
        for overloaded_sys, over_util in overloaded_systems:
            for underloaded_sys, under_util in underloaded_systems:
                # Calculate potential load transfer
                excess_load = system_loads[overloaded_sys] - (system_capacities[overloaded_sys] * 0.75)
                available_capacity = (system_capacities[underloaded_sys] * 0.75) - system_loads[underloaded_sys]
                
                transfer_amount = min(excess_load, available_capacity, 50.0)  # Max 50 kW transfer
                
                if transfer_amount > 10.0:  # Only transfer if significant
                    action = {
                        'from_system': overloaded_sys,
                        'to_system': underloaded_sys,
                        'transfer_amount': transfer_amount,
                        'from_utilization': over_util,
                        'to_utilization': under_util,
                        'estimated_benefit': self._calculate_transfer_benefit(
                            overloaded_sys, underloaded_sys, transfer_amount
                        )
                    }
                    rebalancing_actions.append(action)
        
        # Sort by estimated benefit
        rebalancing_actions.sort(key=lambda x: x['estimated_benefit'], reverse=True)
        
        rebalancing_result = {
            'total_actions': len(rebalancing_actions),
            'actions': rebalancing_actions[:5],  # Top 5 recommendations
            'system_utilizations': {
                sys_id: system_loads[sys_id] / system_capacities[sys_id] 
                for sys_id in system_loads
            },
            'global_utilization': sum(system_loads.values()) / sum(system_capacities.values())
        }
        
        print(f"✅ Rebalancing analysis complete:")
        print(f"   Global utilization: {rebalancing_result['global_utilization']:.1%}")
        print(f"   Recommended actions: {len(rebalancing_actions)}")
        
        return rebalancing_result
    
    def _calculate_transfer_benefit(self, from_sys: int, to_sys: int, amount: float) -> float:
        """Calculate benefit score for load transfer"""
        # Simplified benefit calculation
        from_stations = self.station_status[from_sys]
        to_stations = self.station_status[to_sys]
        
        # Voltage stability improvement
        from_voltage_avg = np.mean([s.voltage_stability for s in from_stations])
        to_voltage_avg = np.mean([s.voltage_stability for s in to_stations])
        voltage_benefit = (to_voltage_avg - from_voltage_avg) * amount
        
        # Queue reduction benefit
        from_queue_avg = np.mean([s.queue_length for s in from_stations])
        to_queue_avg = np.mean([s.queue_length for s in to_stations])
        queue_benefit = max(0, from_queue_avg - to_queue_avg) * 10
        
        # Grid health benefit
        from_grid_avg = np.mean([s.grid_health for s in from_stations])
        to_grid_avg = np.mean([s.grid_health for s in to_stations])
        grid_benefit = (to_grid_avg - from_grid_avg) * amount
        
        return voltage_benefit + queue_benefit + grid_benefit
    
    def get_global_system_status(self) -> Dict:
        """Get comprehensive status of the global federated system"""
        status = {
            'federated_status': self.federated_manager.get_federated_status(),
            'customer_queue_length': len(self.customer_queue),
            'system_summary': {},
            'global_metrics': {}
        }
        
        # System-wise summary
        total_available_ports = 0
        total_ports = 0
        total_current_load = 0.0
        total_capacity = 0.0
        total_queue_length = 0
        
        for sys_id, stations in self.station_status.items():
            system_available = sum(s.available_ports for s in stations)
            system_total = sum(s.total_ports for s in stations)
            system_load = sum(s.current_load for s in stations)
            system_capacity = sum(s.max_capacity for s in stations)
            system_queue = sum(s.queue_length for s in stations)
            
            status['system_summary'][sys_id] = {
                'available_ports': system_available,
                'total_ports': system_total,
                'utilization': system_load / system_capacity if system_capacity > 0 else 0,
                'current_load': system_load,
                'max_capacity': system_capacity,
                'queue_length': system_queue,
                'avg_voltage_stability': np.mean([s.voltage_stability for s in stations]),
                'avg_grid_health': np.mean([s.grid_health for s in stations])
            }
            
            total_available_ports += system_available
            total_ports += system_total
            total_current_load += system_load
            total_capacity += system_capacity
            total_queue_length += system_queue
        
        # Global metrics
        status['global_metrics'] = {
            'total_available_ports': total_available_ports,
            'total_ports': total_ports,
            'global_utilization': total_current_load / total_capacity if total_capacity > 0 else 0,
            'total_current_load': total_current_load,
            'total_capacity': total_capacity,
            'total_queue_length': total_queue_length,
            'average_system_utilization': np.mean([
                s['utilization'] for s in status['system_summary'].values()
            ]),
            'load_balance_score': self._calculate_load_balance_score()
        }
        
        return status
    
    def _calculate_load_balance_score(self) -> float:
        """Calculate how well balanced the load is across systems (0-1, 1 = perfectly balanced)"""
        utilizations = []
        
        for sys_id, stations in self.station_status.items():
            system_load = sum(s.current_load for s in stations)
            system_capacity = sum(s.max_capacity for s in stations)
            utilization = system_load / system_capacity if system_capacity > 0 else 0
            utilizations.append(utilization)
        
        if not utilizations:
            return 1.0
        
        # Calculate coefficient of variation (lower = more balanced)
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        
        if mean_util == 0:
            return 1.0
        
        cv = std_util / mean_util
        balance_score = max(0, 1 - cv)  # Convert to 0-1 scale where 1 = perfectly balanced
        
        return balance_score
