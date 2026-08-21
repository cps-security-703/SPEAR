import opendssdirect as dss
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple

@dataclass
class EVCSParameters:


    rated_voltage: float = 240.0
    rated_current: float = 24.0
    rated_power:   float = 5.76


    max_voltage: float = 240.0
    min_voltage: float = 208.0
    max_current: float = 32.0
    min_current: float = 6.0


    max_power: float = 7.68
    min_power: float = 1.44


    capacity: float = 60.0
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.92
    min_soc: float = 0.1
    max_soc: float = 0.9
    disconnect_soc: float = 0.80
    voltage_bandwidth: float = 5.0
    current_bandwidth: float = 2.0

class EVCSController:


    def __init__(self, evcs_id: str, params: EVCSParameters):
        self.evcs_id = evcs_id
        self.params = params
        self.pinn_training_mode = False
        self.soc = np.random.uniform(0.2, 0.4)


        self.kp_voltage = 1.2
        self.ki_voltage = 0.15
        self.kp_current = 0.8
        self.ki_current = 0.25
        self.kp_power = 0.5
        self.ki_power = 0.1


        self.voltage_error_integral = 0.0
        self.current_error_integral = 0.0
        self.power_error_integral = 0.0


        self.voltage_reference = 240.0
        self.current_reference = 24.0
        self.power_reference   = 5.76


        self.voltage_measured = 240.0
        self.current_measured = 24.0
        self.power_measured   = 5.76


        self.ac_voltage_rms = 240.0
        self.ac_current_rms = 24.0
        self.grid_frequency = 60.0
        self.pll_angle = 0.0


        self.dc_link_voltage = 240.0
        self.switching_frequency = 10000
        self.filter_time_constant = 0.02


        self.dc_link_capacitance = 0.1
        self.dc_link_power_demand = 5.76
        self.ac_dc_efficiency = 0.95
        self.dc_dc_efficiency = 1.00


        self.power_balance_error = 0.0
        self.total_efficiency = self.ac_dc_efficiency * self.dc_dc_efficiency

    def set_references(self, voltage_ref: float, current_ref: float, power_ref: float):

        self.voltage_reference = float(np.clip(voltage_ref, 0.0, 240.0))
        self.current_reference = float(np.clip(current_ref, 0.0, 32.0))
        self.power_reference = float(np.clip(power_ref, 0.0, 7.68))

    def park_transformation(self, va: float, vb: float, vc: float, theta: float) -> Tuple[float, float]:

        vd = (2/3) * (va * np.cos(theta) + vb * np.cos(theta - 2*np.pi/3) + vc * np.cos(theta + 2*np.pi/3))
        vq = (2/3) * (-va * np.sin(theta) - vb * np.sin(theta - 2*np.pi/3) - vc * np.sin(theta + 2*np.pi/3))
        return vd, vq

    def evcs_dynamics_system(self, t, x, grid_voltage_rms):

        try:
            pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage = x


            if not all(np.isfinite([pll_angle, voltage_integral, current_integral, current_measured, soc, dc_link_voltage])):

                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


            pll_angle = pll_angle % (2 * np.pi)
            voltage_integral = np.clip(voltage_integral, -1000, 1000)
            current_integral = np.clip(current_integral, -1000, 1000)
            current_measured = np.clip(current_measured, 0, self.params.max_current)
            soc = np.clip(soc, 0.0, 1.0)
            dc_link_voltage = np.clip(dc_link_voltage, self.params.min_voltage, self.params.max_voltage)


            va = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle)
            vb = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle - 2*np.pi/3)
            vc = grid_voltage_rms * np.sqrt(2) * np.cos(pll_angle + 2*np.pi/3)

            v_alpha = va
            v_beta = (2/np.sqrt(3)) * (vb - vc/2)
            angle_error = np.arctan2(v_beta, v_alpha) - pll_angle


            kp_pll, ki_pll = 10.0, 100.0
            frequency_deviation = kp_pll * angle_error + ki_pll * angle_error

            frequency_deviation = np.clip(frequency_deviation, -5.0, 5.0)
            grid_frequency = 60.0 + frequency_deviation


            dpll_angle_dt = 2 * np.pi * grid_frequency


            ac_power = current_measured * grid_voltage_rms * np.sqrt(3)
            ac_current = current_measured


            converter_efficiency = 0.95
            dc_power = ac_power * converter_efficiency
            dc_current = dc_power / max(dc_link_voltage, 100.0)


            dc_link_capacitance = 0.1

            power_out = self.power_reference * 1000.0
            power_in = dc_power

            power_balance = power_in - power_out

            max_voltage_rate = 1000.0
            ddc_link_voltage_dt = np.clip(power_balance / (dc_link_capacitance * max(dc_link_voltage, self.params.min_voltage)),
                                        -max_voltage_rate, max_voltage_rate)


            voltage_error = self.voltage_reference - dc_link_voltage
            dvoltage_integral_dt = voltage_error


            voltage_control_output = (self.kp_voltage * voltage_error +
                                     self.ki_voltage * voltage_integral)


            current_limit = min(self.current_reference, self.params.max_current)
            if voltage_control_output > current_limit:
                voltage_control_output = current_limit
                dvoltage_integral_dt = 0

            current_error = voltage_control_output - current_measured
            dcurrent_integral_dt = current_error


            converter_time_constant = 0.1
            dcurrent_measured_dt = (voltage_control_output - current_measured) / converter_time_constant

            max_current_rate = 50.0
            dcurrent_measured_dt = np.clip(dcurrent_measured_dt, -max_current_rate, max_current_rate)


            battery_voltage = soc * 200.0 + 300.0
            dcdc_efficiency = 0.95
            battery_power = power_out * dcdc_efficiency / 1000.0


            charging_efficiency = self.params.efficiency_charge if battery_power > 0 else self.params.efficiency_discharge
            dsoc_dt = battery_power * charging_efficiency / (self.params.capacity * 3600)


            max_soc_rate = 0.1 / 3600
            dsoc_dt = np.clip(dsoc_dt, -max_soc_rate, max_soc_rate)

            return [dpll_angle_dt, dvoltage_integral_dt, dcurrent_integral_dt,
                    dcurrent_measured_dt, dsoc_dt, ddc_link_voltage_dt]

        except Exception as e:

            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def pll_update(self, v_abc: List[float], dt: float):


        va, vb, vc = v_abc
        v_alpha = va
        v_beta = (2/np.sqrt(3)) * (vb - vc/2)


        angle_error = np.arctan2(v_beta, v_alpha) - self.pll_angle


        kp_pll = 100.0
        ki_pll = 10000.0

        frequency_deviation = kp_pll * angle_error + ki_pll * angle_error * dt
        self.grid_frequency = 60.0 + frequency_deviation


        self.pll_angle += 2 * np.pi * self.grid_frequency * dt
        self.pll_angle = self.pll_angle % (2 * np.pi)

    def ac_dc_converter_dynamics(self, grid_voltage_rms: float, dt: float) -> Dict:


        self.ac_voltage_rms = grid_voltage_rms


        va = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle)
        vb = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle - 2*np.pi/3)
        vc = grid_voltage_rms * np.sqrt(2) * np.cos(self.pll_angle + 2*np.pi/3)


        self.pll_update([va, vb, vc], dt)


        vd, vq = self.park_transformation(va, vb, vc, self.pll_angle)


        v_rms_effective = max(grid_voltage_rms, 100.0)


        required_ac_power = max(0.0, self.dc_link_power_demand / self.ac_dc_efficiency)
        desired_ac_current = (required_ac_power * 1000) / (3 * v_rms_effective) if required_ac_power > 0 else 0.0


        max_ac_current = self.params.max_power * 1000 / (3 * v_rms_effective)
        desired_ac_current = min(desired_ac_current, max_ac_current)


        current_error = desired_ac_current - self.ac_current_rms
        self.current_error_integral += current_error * dt


        current_control_output = (self.kp_current * current_error +
                                 self.ki_current * self.current_error_integral)


        if current_control_output > max_ac_current:
            current_control_output = max_ac_current
            self.current_error_integral -= current_error * dt


        time_constant = max(self.filter_time_constant, dt * 2)
        current_change = (current_control_output - self.ac_current_rms) * dt / time_constant


        max_current_change = max_ac_current * dt / 0.1
        current_change = np.clip(current_change, -max_current_change, max_current_change)

        self.ac_current_rms += current_change
        self.ac_current_rms = max(0, min(self.ac_current_rms, max_ac_current))


        self.power_measured = 3 * grid_voltage_rms * self.ac_current_rms / 1000


        dc_link_power_available = self.power_measured * self.ac_dc_efficiency


        power_balance = dc_link_power_available - self.dc_link_power_demand


        if abs(self.dc_link_voltage) > 1.0:
            ddc_link_voltage_dt = (power_balance * 1000) / (self.dc_link_capacitance * self.dc_link_voltage)
            self.dc_link_voltage += ddc_link_voltage_dt * dt

            self.dc_link_voltage = np.clip(self.dc_link_voltage,
                                         self.params.min_voltage * 0.9,
                                         self.params.max_voltage * 1.1)

        return {
            'ac_voltage_rms': self.ac_voltage_rms,
            'ac_current_rms': self.ac_current_rms,
            'power_measured': self.power_measured,
            'dc_link_power_available': dc_link_power_available,
            'dc_link_voltage': self.dc_link_voltage,
            'power_balance': power_balance,
            'grid_frequency': self.grid_frequency,
            'pll_angle': np.degrees(self.pll_angle)
        }

    def dc_dc_converter_dynamics(self, dt: float) -> Dict:


        max_current_from_dc_link = self.params.max_current
        if self.dc_link_voltage > 10.0:

            max_power_from_dc_link = self.params.max_power
            max_current_from_dc_link = min(
                self.params.max_current,
                max_power_from_dc_link * 1000 / self.dc_link_voltage
            )


        voltage_error = self.voltage_reference - self.voltage_measured
        self.voltage_error_integral += voltage_error * dt


        voltage_control_output = (self.kp_voltage * voltage_error +
                                 self.ki_voltage * self.voltage_error_integral)


        current_limit = min(
            self.current_reference,
            self.params.max_current,
            max_current_from_dc_link
        )


        if self.current_reference > 0:

            voltage_control_output = max(voltage_control_output, self.current_reference * 0.8)


        if voltage_control_output > current_limit:
            voltage_control_output = current_limit
            self.voltage_error_integral -= voltage_error * dt


        if self.dc_link_voltage < self.params.min_voltage:

            voltage_ratio = self.dc_link_voltage / self.params.min_voltage
            voltage_control_output *= max(0.1, voltage_ratio)


        time_constant = 0.01
        self.current_measured += (voltage_control_output - self.current_measured) * dt / time_constant
        self.current_measured = max(0, min(self.current_measured, current_limit))


        if self.current_measured < self.current_reference * 0.1 and self.current_reference > 1.0:
            self.current_measured = min(self.current_reference * 0.5, current_limit)


        battery_voltage_base = self.params.min_voltage + self.soc * (self.params.max_voltage - self.params.min_voltage)


        internal_resistance = 0.1
        voltage_drop = self.current_measured * internal_resistance
        self.voltage_measured = battery_voltage_base - voltage_drop


        self.voltage_measured = np.clip(self.voltage_measured,
                                      self.params.min_voltage,
                                      self.params.max_voltage)


        dc_power = self.voltage_measured * self.current_measured / 1000


        self.dc_link_power_demand = dc_power / self.dc_dc_efficiency if dc_power > 0 else 0.0


        power_loss_dc_dc = self.dc_link_power_demand - dc_power


        if self.dc_link_power_demand > 0.001:

            actual_efficiency = dc_power / self.dc_link_power_demand

            actual_efficiency = np.clip(actual_efficiency, 0.5, 0.99)
        else:
            actual_efficiency = 0.95

        return {
            'voltage_measured': self.voltage_measured,
            'current_measured': self.current_measured,
            'dc_power': dc_power,
            'dc_link_power_demand': self.dc_link_power_demand,
            'max_current_from_dc_link': max_current_from_dc_link,
            'power_loss_dc_dc': power_loss_dc_dc,
            'dc_dc_efficiency_actual': actual_efficiency,
            'soc': self.soc
        }

    def update_soc(self, power_kW: float, dt: float):

        if power_kW > 0:
            energy_kwh = power_kW * dt / 3600 * self.params.efficiency_charge
        else:
            energy_kwh = power_kW * dt / 3600 / self.params.efficiency_discharge

        self.soc += energy_kwh / self.params.capacity
        self.soc = np.clip(self.soc, self.params.min_soc, self.params.max_soc)

    def _sanitize_states(self):


        if hasattr(self, 'pll_angle'):
            self.pll_angle = self.pll_angle % (2 * np.pi)
            if not np.isfinite(self.pll_angle):
                self.pll_angle = 0.0


        max_integral = 1000.0
        if hasattr(self, 'voltage_error_integral'):
            self.voltage_error_integral = np.clip(self.voltage_error_integral, -max_integral, max_integral)
            if not np.isfinite(self.voltage_error_integral):
                self.voltage_error_integral = 0.0

        if hasattr(self, 'current_error_integral'):
            self.current_error_integral = np.clip(self.current_error_integral, -max_integral, max_integral)
            if not np.isfinite(self.current_error_integral):
                self.current_error_integral = 0.0


        if hasattr(self, 'current_measured'):
            self.current_measured = np.clip(self.current_measured, 0.0, 32.0)
            if not np.isfinite(self.current_measured):
                self.current_measured = 0.0


        if hasattr(self, 'soc'):
            self.soc = np.clip(self.soc, 0.01, 0.99)
            if not np.isfinite(self.soc):
                self.soc = 0.5


        if hasattr(self, 'dc_link_voltage'):
            self.dc_link_voltage = np.clip(self.dc_link_voltage, 208.0, 240.0)
            if not np.isfinite(self.dc_link_voltage):
                self.dc_link_voltage = 240.0

    def update_dynamics_with_solve_ivp(self, grid_voltage_rms: float, dt: float,
                                       system_frequency: Optional[float] = None) -> Dict:


        self._sanitize_states()


        x0 = [self.pll_angle, self.voltage_error_integral, self.current_error_integral,
              self.current_measured, self.soc, self.dc_link_voltage]

        t_span = (0, dt)

        try:

            if any(not np.isfinite(val) for val in x0):
                state_names = ['pll_angle', 'voltage_integral', 'current_integral',
                              'current_measured', 'soc', 'dc_link_voltage']
                invalid_states = []
                for name, val in zip(state_names, x0):
                    if not np.isfinite(val):
                        invalid_states.append(f"{name}={val}")
                print(f"EVCS {self.evcs_id}: Invalid states: {', '.join(invalid_states)} - using Euler fallback")
                return self._update_dynamics_euler(grid_voltage_rms, dt)


            sol = solve_ivp(
                lambda t, x: self.evcs_dynamics_system(t, x, grid_voltage_rms),
                t_span, x0,
                method='LSODA',
                rtol=1e-3,
                atol=1e-5,
                max_step=dt/5,
                first_step=dt/100,
                dense_output=False
            )

            if sol.success:

                x_new = sol.y[:, -1]
                self.pll_angle, self.voltage_error_integral, self.current_error_integral, \
                self.current_measured, self.soc, self.dc_link_voltage = x_new


                self.pll_angle = self.pll_angle % (2 * np.pi)
                self.current_measured = max(self.params.min_current, min(self.current_measured, self.params.max_current))
                self.soc = np.clip(self.soc, self.params.min_soc, self.params.max_soc)


                self.voltage_measured = self.params.min_voltage + self.soc * (self.params.max_voltage - self.params.min_voltage)
                self.power_measured = self.voltage_measured * self.current_measured / 1000

                if system_frequency is not None and np.isfinite(system_frequency):
                    self.grid_frequency = float(system_frequency)
                else:
                    self.grid_frequency = getattr(self, 'grid_frequency', 60.0)


                if grid_voltage_rms > 0 and self.power_reference > 0:
                    self.ac_current_rms = (self.power_reference * 1000) / (3 * grid_voltage_rms)
                else:
                    self.ac_current_rms = 0.0


                self.ac_voltage_rms = grid_voltage_rms

                return {
                    'voltage_measured': self.voltage_measured,
                    'current_measured': self.current_measured,
                    'power_measured': self.power_measured,
                    'soc': self.soc,
                    'pll_angle': np.degrees(self.pll_angle),
                    'grid_frequency': self.grid_frequency,
                    'ac_voltage_rms': self.ac_voltage_rms,
                    'ac_current_rms': self.ac_current_rms,
                    'total_power': self.power_reference,
                    'integration_method': 'solve_ivp'
                }
            else:

                print(f"EVCS {self.evcs_id}: solve_ivp failed, using Euler fallback")
                return self._update_dynamics_euler(grid_voltage_rms, dt, system_frequency=system_frequency)

        except Exception as e:
            print(f"EVCS {self.evcs_id}: solve_ivp error: {e}, using Euler fallback")
            return self._update_dynamics_euler(grid_voltage_rms, dt, system_frequency=system_frequency)

    def update_dynamics(self, grid_voltage_rms: float, dt: float,
                        use_solve_ivp: bool = True, system_frequency: float = None) -> Dict:


        if use_solve_ivp:

            return self.update_dynamics_with_solve_ivp(grid_voltage_rms, dt, system_frequency=system_frequency)
        else:

            return self._update_dynamics_euler(grid_voltage_rms, dt, system_frequency=system_frequency)

    def _update_dynamics_euler(self, grid_voltage_rms: float, dt: float,
                               simulation_time: float = 0.0, system_frequency: float = None) -> Dict:


        self.simulation_time = simulation_time


        if not hasattr(self, '_last_reset_time'):
            self._last_reset_time = simulation_time
        elif simulation_time - self._last_reset_time > 10.0:
            if hasattr(self, '_adjustment_count'):
                self._adjustment_count = 0
            self._last_reset_time = simulation_time


        target_dc_power = min(self.power_reference, self.params.max_power)


        required_dc_link_power = target_dc_power / self.dc_dc_efficiency if target_dc_power > 0 else 0.0


        required_ac_power = required_dc_link_power / self.ac_dc_efficiency if required_dc_link_power > 0 else 0.0


        if target_dc_power > 0.001:

            expected_total_efficiency = self.ac_dc_efficiency * self.dc_dc_efficiency
            if abs(expected_total_efficiency - 0.94) > 0.1:
                print(f"EVCS {self.evcs_id}: Efficiency mismatch detected: {expected_total_efficiency:.3f}")

                self.ac_dc_efficiency = 0.98
                self.dc_dc_efficiency = 0.96


        if not hasattr(self, '_power_demand_history'):
            self._power_demand_history = []


        self._power_demand_history.append(required_dc_link_power)
        if len(self._power_demand_history) > 5:
            self._power_demand_history.pop(0)


        smoothed_demand = np.mean(self._power_demand_history)
        self.dc_link_power_demand = smoothed_demand


        dc_results = self.dc_dc_converter_dynamics(dt)
        actual_dc_power = dc_results['dc_power']


        ac_results = self.ac_dc_converter_dynamics(grid_voltage_rms, dt)
        actual_ac_power = ac_results['power_measured']


        if actual_ac_power > 0.001:

            system_efficiency = actual_dc_power / actual_ac_power

            system_efficiency = np.clip(system_efficiency, 0.4, 0.98)
        else:
            system_efficiency = 0.9

        total_power_loss = actual_ac_power - actual_dc_power


        power_balance_error = abs(required_dc_link_power - actual_dc_power)


        adjustment_factor = 0.0


        if not hasattr(self, '_last_power_error'):
            self._last_power_error = 0.0
            self._error_history = []
            self._adjustment_count = 0
            self._max_adjustments = 10


        if power_balance_error > 100.0 and self._adjustment_count < self._max_adjustments:

            error_change = abs(power_balance_error - self._last_power_error)


            if error_change > 10.0 or self._adjustment_count == 0:

                if power_balance_error > 500.0:
                    adjustment_factor = 0.05
                elif power_balance_error > 200.0:
                    adjustment_factor = 0.02
                else:
                    adjustment_factor = 0.01


                if required_dc_link_power > actual_dc_power:
                    self.dc_link_power_demand *= (1.0 + adjustment_factor)
                else:
                    self.dc_link_power_demand *= (1.0 - adjustment_factor)


                self.dc_link_power_demand = np.clip(self.dc_link_power_demand, 0.0, self.params.max_power * 1.1)


                self._adjustment_count += 1
                self._last_power_error = power_balance_error
                self._error_history.append(power_balance_error)
                if len(self._error_history) > 5:
                    self._error_history.pop(0)


                if not self.pinn_training_mode:
                    if self._adjustment_count % 5 == 0:
                        print(f"EVCS {self.evcs_id}: Power coupling adjustment: {adjustment_factor:.3f}, Error: {power_balance_error:.2f}kW")
                else:

                    if self._adjustment_count % 10 == 0:
                        print(f"EVCS {self.evcs_id}: [PINN Training] Power coupling adjustment: {adjustment_factor:.3f}, Error: {power_balance_error:.2f}kW")
            else:

                self._adjustment_count = self._max_adjustments
        else:

            if power_balance_error <= 100.0:
                self._adjustment_count = 0


        self.update_soc(actual_dc_power, dt)
        self.power_measured = actual_dc_power


        if system_frequency is not None and np.isfinite(system_frequency):
            self.grid_frequency = float(system_frequency)
        else:
            self.grid_frequency = ac_results.get('grid_frequency', self.grid_frequency)
        ac_results['grid_frequency'] = self.grid_frequency

        return {

            **ac_results,

            **dc_results,

            'total_power': actual_dc_power,
            'ac_power_in': actual_ac_power,
            'dc_power_out': actual_dc_power,
            'total_power_loss': total_power_loss,
            'system_efficiency': system_efficiency,
            'power_balance_error': power_balance_error,
            'total_efficiency_actual': system_efficiency,
            'total_efficiency_design': self.total_efficiency,
            'target_dc_power': target_dc_power,
            'required_dc_link_power': required_dc_link_power,
            'required_ac_power': required_ac_power,
            'integration_method': 'euler_stable'
        }

class ChargingManagementSystem:


    def __init__(self, evcs_controllers: Dict[str, EVCSController]):
        self.evcs_controllers = evcs_controllers
        self.voltage_limits = {'min': 0.95, 'max': 1.05}
        self.total_power_limit = 300


        self.voltage_droop_gain = 100.0
        self.frequency_droop_gain = 50.0
        self.soc_weight = 0.3
        self.voltage_weight = 0.4
        self.load_weight = 0.3

    def generate_daily_charging_profile(self, time_hours: float) -> float:


        morning_peak = np.exp(-((time_hours - 8)**2) / (2 * 1.5**2))
        evening_peak = np.exp(-((time_hours - 19)**2) / (2 * 2**2))


        base_load = 0.3 + 0.2 * np.sin(2 * np.pi * time_hours / 24)


        daily_pattern = 0.4 * morning_peak + 0.6 * evening_peak + base_load

        return np.clip(daily_pattern, 0.1, 1.0)

    def optimize_charging_qsts(self, time_hours: float, bus_voltages: Dict[str, float],
                              system_frequency: float = 60.0, pinn_optimizer=None) -> Dict[str, Dict]:


        if pinn_optimizer is not None:
            return self._optimize_with_pinn(time_hours, bus_voltages, system_frequency, pinn_optimizer)
        else:
            return self._optimize_heuristic(time_hours, bus_voltages, system_frequency)

    def _optimize_with_pinn(self, time_hours: float, bus_voltages: Dict[str, float],
                           system_frequency: float, pinn_optimizer) -> Dict[str, Dict]:


        demand_factor = self.generate_daily_charging_profile(time_hours)


        evcs_to_bus_mapping = {
            'EVCS1': '890', 'EVCS2': '844', 'EVCS3': '860',
            'EVCS4': '840', 'EVCS5': '848', 'EVCS6': '830'
        }

        references = {}

        for evcs_name, controller in self.evcs_controllers.items():

            bus_name = evcs_to_bus_mapping.get(evcs_name, '890')
            voltage_pu = bus_voltages.get(bus_name, 1.0)


            voltage_priority = max(0, self.voltage_limits['min'] - voltage_pu)
            urgency_factor = 2.0 - controller.soc


            bus_distances = {'890': 0.0, '844': 0.4, '860': 0.7, '840': 1.6, '848': 2.9, '830': 4.0}
            bus_distance = bus_distances.get(bus_name, 1.0)


            input_features = [
                controller.soc,
                voltage_pu,
                system_frequency,
                demand_factor,
                voltage_priority,
                urgency_factor,
                time_hours,
                bus_distance,
                1.0,
                controller.power_reference
            ]


            sequence_length = 10
            sequence = [input_features] * sequence_length

            try:

                voltage_ref, current_ref, power_ref = pinn_optimizer.optimize_references_lstm(sequence)


                max_power = getattr(controller.params, 'max_power', 200.0)
                power_ref = min(power_ref, max_power)

                references[evcs_name] = {
                    'power_ref': power_ref,
                    'voltage_ref': voltage_ref,
                    'current_ref': current_ref,
                    'priority': urgency_factor,
                    'voltage_pu': voltage_pu
                }


                controller.set_references(voltage_ref, current_ref, power_ref)

            except Exception as e:
                print(f"PINN optimization failed for {evcs_name}: {e}, using fallback")

                if controller.soc < 0.3:
                    power_ref, voltage_ref, current_ref = 7.68, 240.0, 32.0
                elif controller.soc < 0.7:
                    power_ref, voltage_ref, current_ref = 5.76, 240.0, 24.0
                else:
                    power_ref, voltage_ref, current_ref = 2.88, 240.0, 12.0

                references[evcs_name] = {
                    'power_ref': power_ref,
                    'voltage_ref': voltage_ref,
                    'current_ref': current_ref,
                    'priority': urgency_factor,
                    'voltage_pu': voltage_pu
                }
                controller.set_references(voltage_ref, current_ref, power_ref)

        return references

    def _optimize_heuristic(self, time_hours: float, bus_voltages: Dict[str, float],
                           system_frequency: float = 60.0) -> Dict[str, Dict]:


        demand_factor = self.generate_daily_charging_profile(time_hours)


        total_available_power = self.total_power_limit * demand_factor


        min_emergency_power = 100.0
        if total_available_power < min_emergency_power:
            total_available_power = min_emergency_power


        evcs_to_bus_mapping = {
            'EVCS1': '890', 'EVCS2': '844', 'EVCS3': '860',
            'EVCS4': '840', 'EVCS5': '848', 'EVCS6': '830'
        }


        evcs_priority = []
        for evcs_name, controller in self.evcs_controllers.items():

            bus_name = evcs_to_bus_mapping.get(evcs_name, '890')


            soc_priority = (1 - controller.soc)

            if bus_name in bus_voltages:
                voltage_pu = bus_voltages[bus_name]

                voltage_priority = max(0, self.voltage_limits['min'] - voltage_pu)
            else:
                voltage_priority = 0
                voltage_pu = 1.0


            frequency_priority = max(0, 60.0 - system_frequency) / 60.0


            composite_priority = (self.soc_weight * soc_priority +
                                self.voltage_weight * voltage_priority +
                                0.1 * frequency_priority)

            evcs_priority.append((evcs_name, controller, composite_priority, bus_name, voltage_pu))


        evcs_priority.sort(key=lambda x: x[2], reverse=True)


        references = {}
        remaining_power = total_available_power

        for evcs_name, controller, priority, bus_name, voltage_pu in evcs_priority:


            if controller.soc < 0.2:
                base_power = 70
                target_voltage = 500.0
                target_current = 140.0
            elif controller.soc < 0.8:
                base_power = 40
                target_voltage = 400.0
                target_current = 100.0
            else:
                base_power = 20
                target_voltage = 350.0
                target_current = 60.0


            if voltage_pu < self.voltage_limits['min']:

                power_factor = 0.4
            elif voltage_pu > self.voltage_limits['max']:

                power_factor = 1.3
            else:
                power_factor = 1.0


            base_power *= demand_factor


            allocated_power = min(base_power * power_factor, remaining_power)
            allocated_power = max(0, allocated_power)


            if allocated_power > 0:

                voltage_ref = controller.params.min_voltage + controller.soc * (controller.params.max_voltage - controller.params.min_voltage)


                if voltage_ref > 0:
                    current_ref = min(allocated_power * 1000 / voltage_ref, target_current)
                else:
                    current_ref = 0


                current_ref = max(controller.params.min_current, min(current_ref, controller.params.max_current))
            else:
                voltage_ref = controller.voltage_measured
                current_ref = 0
                allocated_power = 0

            references[evcs_name] = {
                'power_ref': allocated_power,
                'voltage_ref': voltage_ref,
                'current_ref': current_ref,
                'priority': priority,
                'voltage_pu': voltage_pu
            }

            remaining_power -= allocated_power


            controller.set_references(voltage_ref, current_ref, allocated_power)

            if remaining_power <= 0:
                break

        return references

def create_daily_load_shapes():


    time_points = np.linspace(0, 24, 288)


    residential_pattern = []
    for hour in time_points:
        if 0 <= hour < 6:
            load_factor = 0.3 + 0.1 * np.random.normal(0, 0.05)
        elif 6 <= hour < 9:
            load_factor = 0.7 + 0.2 * np.sin(np.pi * (hour - 6) / 3) + 0.1 * np.random.normal(0, 0.05)
        elif 9 <= hour < 17:
            load_factor = 0.5 + 0.1 * np.random.normal(0, 0.05)
        elif 17 <= hour < 21:
            load_factor = 0.8 + 0.2 * np.sin(np.pi * (hour - 17) / 4) + 0.1 * np.random.normal(0, 0.05)
        else:
            load_factor = 0.4 + 0.1 * np.random.normal(0, 0.05)

        residential_pattern.append(max(0.2, min(1.0, load_factor)))

    return time_points, residential_pattern

def setup_ieee34_with_evcs_qsts():


    dss.Command("Clear")


    try:
        dss.Command("Compile ieee34Mod1.dss")
        print("IEEE 34 bus system loaded successfully")
    except Exception as e:
        print(f"Error loading IEEE 34 system: {e}")
        print("Trying alternative file names...")

        alternative_files = ["IEEE34Mod1.dss", "ieee34mod1.dss", "IEEE34.dss"]
        loaded = False

        for filename in alternative_files:
            try:
                dss.Command(f"Compile {filename}")
                print(f"Successfully loaded: {filename}")
                loaded = True
                break
            except:
                continue

        if not loaded:
            raise FileNotFoundError("Could not find IEEE 34 bus system file.")


    print("Setting up for manual time-step simulation...")


    dss.Command("Set Mode=Snapshot")
    dss.Command("Set ControlMode=Static")


    time_points, load_pattern = create_daily_load_shapes()


    try:
        load_shape_str = "New Loadshape.Daily npts=288 interval=0.0833 mult=["
        load_shape_str += ",".join([f"{val:.3f}" for val in load_pattern])
        load_shape_str += "]"
        dss.Command(load_shape_str)
        print("Daily load shape created successfully")


        dss.Command("Solve")

        load_names = dss.Loads.AllNames()

        if load_names and len(load_names) > 0:
            print(f"Found {len(load_names)} loads in the system")

            successful_loads = 0
            for load_name in load_names:
                try:
                    dss.Command(f"Load.{load_name}.Daily=Daily")
                    successful_loads += 1
                except Exception as e:
                    continue

            print(f"Daily load shape applied to {successful_loads}/{len(load_names)} loads")
        else:
            print("No loads found in the system")

    except Exception as e:
        print(f"Load shape setup failed: {e}")
        print("Will use manual load variation instead")


    evcs_buses = ['800', '802', '806', '814', '820', '832']


    dss.Command("Solve")
    all_buses = dss.Circuit.AllBusNames()
    print(f"Total buses in system: {len(all_buses)}")

    valid_evcs_buses = []
    for bus in evcs_buses:
        if bus in all_buses:
            valid_evcs_buses.append(bus)
            print(f"Bus {bus} found - will add EVCS")
        else:
            print(f"Bus {bus} not found - skipping")

    if not valid_evcs_buses:
        raise ValueError("No valid buses found for EVCS placement")


    evcs_data = []
    for i, bus in enumerate(valid_evcs_buses, 1):
        evcs_name = f"EVCS{i}"

        try:

            storage_cmd = f"New Storage.{evcs_name} Bus1={bus} kV=12.47 conn=wye kW=0 kWh=50 %stored=30 %reserve=10 %EffCharge=95 %EffDischarge=92 State=Idling"
            dss.Command(storage_cmd)


            dss.Circuit.SetActiveElement(f"Storage.{evcs_name}")
            if dss.CktElement.Name().lower() == f"storage.{evcs_name.lower()}":
                print(f" Successfully added {evcs_name} at Bus {bus}")

                evcs_data.append({
                    'name': evcs_name,
                    'bus': bus,
                    'kV': 12.47
                })
            else:
                print(f" Failed to verify {evcs_name}")

        except Exception as e:
            print(f" Error adding {evcs_name}: {e}")
            continue

    if not evcs_data:
        raise ValueError("No EVCS were successfully added")

    print(f"Successfully added {len(evcs_data)} EVCS to the system")


    print("\n=== Initial System Test ===")
    dss.Command("Solve")
    if dss.Solution.Converged():
        print(" Initial power flow converged")


        test_evcs = evcs_data[0]['name']
        print(f"Testing {test_evcs} control...")


        dss.Command(f"Storage.{test_evcs}.State=Charging")
        dss.Command(f"Storage.{test_evcs}.kW=25")
        dss.Command("Solve")


        dss.Circuit.SetActiveElement(f"Storage.{test_evcs}")
        test_power = dss.CktElement.Powers()[0]
        print(f"  Set 25kW, Actual: {test_power:.1f}kW")

        if abs(test_power - (-25.0)) < 1.0:
            print(" EVCS control test passed")
        else:
            print("# EVCS control test failed - values may not update properly")


        dss.Command(f"Storage.{test_evcs}.State=Idling")
        dss.Command(f"Storage.{test_evcs}.kW=0")

    else:
        print(" Initial power flow did not converge")
        print("# Simulation may have issues")

    return evcs_data, time_points

def run_qsts_evcs_simulation():


    print("Setting up QSTS simulation...")
    evcs_data, time_points = setup_ieee34_with_evcs_qsts()


    params = EVCSParameters()
    evcs_controllers = {}

    for evcs in evcs_data:
        controller = EVCSController(evcs['name'], params)
        evcs_controllers[evcs['name']] = controller


    cms = ChargingManagementSystem(evcs_controllers)


    total_steps = 288
    dt = 300


    try:
        dss.Command("Solve")
        baseline_load = dss.Circuit.TotalPower()[0]
        print(f"Baseline system load: {baseline_load:.1f} kW")
    except:
        baseline_load = 1000.0
        print("Using default baseline load: 1000 kW")


    results = {
        'time_hours': [],
        'step': [],
        'bus_voltages': {evcs['bus']: [] for evcs in evcs_data},
        'evcs_power': {evcs['name']: [] for evcs in evcs_data},
        'evcs_voltage_ref': {evcs['name']: [] for evcs in evcs_data},
        'evcs_current_ref': {evcs['name']: [] for evcs in evcs_data},
        'evcs_voltage_measured': {evcs['name']: [] for evcs in evcs_data},
        'evcs_current_measured': {evcs['name']: [] for evcs in evcs_data},
        'evcs_soc': {evcs['name']: [] for evcs in evcs_data},
        'total_power': [],
        'system_frequency': [],
        'system_load_factor': []
    }

    print(f"Starting QSTS simulation: 288 steps (24 hours)")


    for step in range(total_steps):
        current_time_hours = step * 5 / 60.0


        load_factor = cms.generate_daily_charging_profile(current_time_hours)
        results['system_load_factor'].append(load_factor)


        try:

            dss.Command("Solve")

            converged = dss.Solution.Converged()
            if not converged and step % 50 == 0:
                print(f"Warning: Power flow did not converge at step {step}")

        except Exception as e:
            if step % 50 == 0:
                print(f"Error solving at step {step}: {e}")
            continue


        system_frequency = dss.Solution.Frequency()


        bus_voltages = {}
        for evcs in evcs_data:
            voltage_pu = 1.0
            try:
                dss.Circuit.SetActiveBus(evcs['bus'])

                voltage_kv = dss.Bus.kVBase()
                voltages_actual = dss.Bus.VMagAngle()

                if len(voltages_actual) >= 2 and voltage_kv > 0:
                    voltage_pu = voltages_actual[0] / voltage_kv


                if step % 50 == 0:
                    print(f"  Bus {evcs['bus']}: {voltages_actual[0]:.1f}kV / {voltage_kv:.1f}kV = {voltage_pu:.3f}pu")

            except Exception as e:
                if step % 50 == 0:
                    print(f"  Error reading Bus {evcs['bus']}: {e}")

            bus_voltages[evcs['bus']] = voltage_pu


        references = cms.optimize_charging_qsts(current_time_hours, bus_voltages, system_frequency)


        if step % 50 == 0:
            print(f"\nStep {step} (t={current_time_hours:.1f}h):")
            print(f"  Load factor: {load_factor:.3f}")
            print(f"  Bus voltages: {[(bus, f'{v:.3f}pu') for bus, v in bus_voltages.items()]}")

            total_allocated = sum([ref["power_ref"] for ref in references.values()])
            print(f"  Total allocated: {total_allocated:.1f}kW")


        total_power = 0
        for evcs_name, controller in evcs_controllers.items():

            if evcs_name in references:
                ref_data = references[evcs_name]
                power_kW = ref_data['power_ref']


                try:
                    if power_kW > 0:
                        dss.Command(f"Storage.{evcs_name}.State=Charging")
                        dss.Command(f"Storage.{evcs_name}.kW={power_kW}")


                        if step % 50 == 0:
                            dss.Circuit.SetActiveElement(f"Storage.{evcs_name}")
                            actual_kW = dss.CktElement.Powers()[0]
                            print(f"  {evcs_name}: Set {power_kW:.1f}kW, Actual {actual_kW:.1f}kW")
                    else:
                        dss.Command(f"Storage.{evcs_name}.State=Idling")
                        dss.Command(f"Storage.{evcs_name}.kW=0")

                except Exception as e:
                    if step % 50 == 0:
                        print(f"  Error updating {evcs_name}: {e}")
                    continue


                evcs_num = evcs_name.replace('EVCS', '')
                bus_mapping = {'1': '800', '2': '802', '3': '806', '4': '814', '5': '820', '6': '832'}
                bus_name = bus_mapping.get(evcs_num, '800')

                grid_voltage = bus_voltages.get(bus_name, 1.0) * 7200

                dynamics_result = controller.update_dynamics(grid_voltage, dt)
                total_power += dynamics_result['total_power']


                if step % 50 == 0:
                    print(f"    {evcs_name}: SOC={controller.soc:.3f}, V_ref={ref_data['voltage_ref']:.1f}V, I_ref={ref_data['current_ref']:.1f}A")

            else:

                if step % 50 == 0:
                    print(f"    {evcs_name}: No power reference")


        results['time_hours'].append(current_time_hours)
        results['step'].append(step)
        results['total_power'].append(total_power)
        results['system_frequency'].append(system_frequency)


        for evcs in evcs_data:
            bus_name = evcs['bus']
            evcs_name = evcs['name']
            controller = evcs_controllers[evcs_name]


            results['bus_voltages'][bus_name].append(bus_voltages.get(bus_name, 1.0))


            if evcs_name in references:
                ref_data = references[evcs_name]
                results['evcs_power'][evcs_name].append(ref_data['power_ref'])
                results['evcs_voltage_ref'][evcs_name].append(ref_data['voltage_ref'])
                results['evcs_current_ref'][evcs_name].append(ref_data['current_ref'])
            else:
                results['evcs_power'][evcs_name].append(0.0)
                results['evcs_voltage_ref'][evcs_name].append(controller.voltage_reference)
                results['evcs_current_ref'][evcs_name].append(controller.current_reference)


            results['evcs_voltage_measured'][evcs_name].append(controller.voltage_measured)
            results['evcs_current_measured'][evcs_name].append(controller.current_measured)
            results['evcs_soc'][evcs_name].append(controller.soc)


        if step % 24 == 0:
            progress = step / total_steps * 100
            print(f"\n=== QSTS Progress: {progress:.1f}% - Time: {current_time_hours:.1f}h ===")
            print(f"Total EVCS Power: {total_power:.1f}kW, System Frequency: {system_frequency:.3f}Hz")


        if step % 48 == 0 and step > 0:
            print(f"\n--- 4-Hour Summary (t={current_time_hours:.1f}h) ---")
            for evcs_name, controller in evcs_controllers.items():
                print(f"  {evcs_name}: SOC={controller.soc*100:.1f}%, Power_ref={controller.power_reference:.1f}kW")

    print(f"QSTS simulation completed! Total steps: {len(results['time_hours'])}")
    return results, evcs_data

def plot_qsts_results(results, evcs_data):


    if not results['time_hours']:
        print("No simulation data to plot!")
        return

    print(f"Plotting QSTS results for {len(results['time_hours'])} time steps...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('24-Hour QSTS EVCS Dynamics - IEEE 34 Bus System', fontsize=16)

    time_hours = results['time_hours']


    ax1 = axes[0, 0]
    for bus in results['bus_voltages']:
        ax1.plot(time_hours, results['bus_voltages'][bus],
                label=f'Bus {bus}', linewidth=2)
    ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='Upper Limit')
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Lower Limit')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Voltage (per unit)')
    ax1.set_title('24-Hour Bus Voltage Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)


    ax2 = axes[0, 1]
    for evcs_name in results['evcs_power']:
        ax2.plot(time_hours, results['evcs_power'][evcs_name],
                label=evcs_name, linewidth=2)
    ax2.plot(time_hours, results['total_power'],
            'k--', linewidth=3, label='Total Power')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('24-Hour EVCS Charging Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)


    ax3 = axes[1, 0]
    for i, evcs_name in enumerate(list(results['evcs_voltage_ref'].keys())[:3]):
        ax3.plot(time_hours, results['evcs_voltage_ref'][evcs_name],
                '--', linewidth=2, label=f'{evcs_name} Ref')
        ax3.plot(time_hours, results['evcs_voltage_measured'][evcs_name],
                '-', linewidth=2, label=f'{evcs_name} Measured')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('DC Voltage (V)')
    ax3.set_title('EVCS Voltage References vs Measured')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)


    ax4 = axes[1, 1]
    for i, evcs_name in enumerate(list(results['evcs_current_ref'].keys())[:3]):
        ax4.plot(time_hours, results['evcs_current_ref'][evcs_name],
                '--', linewidth=2, label=f'{evcs_name} Ref')
        ax4.plot(time_hours, results['evcs_current_measured'][evcs_name],
                '-', linewidth=2, label=f'{evcs_name} Measured')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('DC Current (A)')
    ax4.set_title('EVCS Current References vs Measured')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)


    ax5 = axes[2, 0]
    for evcs_name in results['evcs_soc']:
        ax5.plot(time_hours, [soc*100 for soc in results['evcs_soc'][evcs_name]],
                linewidth=2, label=evcs_name)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('State of Charge (%)')
    ax5.set_title('24-Hour SOC Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 24)
    ax5.set_ylim(0, 100)


    ax6 = axes[2, 1]
    ax6_twin = ax6.twinx()

    line1 = ax6.plot(time_hours, results['system_frequency'], 'b-', linewidth=2, label='Frequency')
    line2 = ax6_twin.plot(time_hours, results['total_power'], 'r-', linewidth=2, label='Total Power')

    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Frequency (Hz)', color='b')
    ax6_twin.set_ylabel('Total Power (kW)', color='r')
    ax6.set_title('System Frequency & Total EVCS Power')


    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')

    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 24)

    plt.tight_layout()
    plt.show()


    print("\n" + "="*80)
    print("24-HOUR QSTS SIMULATION SUMMARY")
    print("="*80)

    for evcs in evcs_data:
        bus = evcs['bus']
        evcs_name = evcs['name']


        voltages = results['bus_voltages'][bus]
        avg_voltage = np.mean(voltages)
        min_voltage = np.min(voltages)
        max_voltage = np.max(voltages)


        powers = results['evcs_power'][evcs_name]
        total_energy = np.sum(powers) * 5 / 60
        initial_soc = results['evcs_soc'][evcs_name][0] * 100
        final_soc = results['evcs_soc'][evcs_name][-1] * 100
        soc_change = final_soc - initial_soc


        voltage_refs = results['evcs_voltage_ref'][evcs_name]
        voltage_measured = results['evcs_voltage_measured'][evcs_name]
        voltage_tracking_error = np.mean([abs(r-m) for r,m in zip(voltage_refs, voltage_measured)])

        current_refs = results['evcs_current_ref'][evcs_name]
        current_measured = results['evcs_current_measured'][evcs_name]
        current_tracking_error = np.mean([abs(r-m) for r,m in zip(current_refs, current_measured)])

        print(f"\n{evcs_name} (Bus {bus}):")
        print(f"  Bus Voltage - Avg: {avg_voltage:.3f}pu, Min: {min_voltage:.3f}pu, Max: {max_voltage:.3f}pu")
        print(f"  Energy Charged: {total_energy:.1f} kWh")
        print(f"  SOC Change: {initial_soc:.1f}%  {final_soc:.1f}% (Δ{soc_change:+.1f}%)")
        print(f"  Voltage Tracking Error: {voltage_tracking_error:.1f}V")
        print(f"  Current Tracking Error: {current_tracking_error:.1f}A")


    total_system_energy = np.sum(results['total_power']) * 5 / 60
    avg_frequency = np.mean(results['system_frequency'])
    frequency_deviation = np.std(results['system_frequency'])

    print(f"\nSYSTEM SUMMARY:")
    print(f"  Total Energy Delivered: {total_system_energy:.1f} kWh")
    print(f"  Average System Frequency: {avg_frequency:.3f} Hz")
    print(f"  Frequency Std Deviation: {frequency_deviation:.4f} Hz")
    print(f"  Peak Total Power: {max(results['total_power']):.1f} kW")

if __name__ == "__main__":
    print("Starting 24-Hour QSTS EVCS Dynamics Simulation...")


    try:
        version = dss.Basic.Version()
        print(f"OpenDSS Version: {version}")
    except:
        print("Could not get OpenDSS version")


    try:
        dss.Command("Clear")
        print("OpenDSS interface working correctly")
    except Exception as e:
        print(f"OpenDSS interface error: {e}")
        exit(1)

    try:

        print("Initializing simulation...")
        results, evcs_data = run_qsts_evcs_simulation()


        print("\nGenerating comprehensive plots...")
        plot_qsts_results(results, evcs_data)

        print("\n24-Hour QSTS Simulation completed successfully!")

    except FileNotFoundError as e:
        print(f"File error: {e}")
        print("Please ensure the IEEE 34 bus system file (.dss) is in the current directory")
        print("Expected file names: ieee34Mod1.dss, IEEE34Mod1.dss, ieee34mod1.dss, or IEEE34.dss")

    except Exception as e:
        print(f"Error during QSTS simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure ieee34Mod1.dss file is in the current directory")
        print("2. Check that OpenDSS is properly installed and accessible")
        print("3. Verify all required Python packages are installed:")
        print("   pip install opendssdirect matplotlib pandas scipy numpy")
        print("4. Try running a simple OpenDSS command first to test the installation")
