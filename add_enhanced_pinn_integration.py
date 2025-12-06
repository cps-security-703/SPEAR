#!/usr/bin/env python3
"""
Script to add enhanced PINN integration to hierarchical_cosimulation.py
"""

def add_enhanced_pinn_integration():
    """Add enhanced PINN integration after the recovery update"""
    
    # Read the file
    with open('hierarchical_cosimulation.py', 'r') as f:
        content = f.read()
    
    # Define the enhanced PINN integration code
    enhanced_pinn_code = """ station.update_recovery(self.simulation_time)
                
                # NEW: Enhanced PINN Integration - Update EVCS dynamics using trained models
                if (self.use_enhanced_pinn and self.enhanced_pinn_available and 
                    sys_id in self.enhanced_pinn_models):
                    
                    try:
                        enhanced_pinn_model = self.enhanced_pinn_models[sys_id]
                        
                        # Update each EVCS station using enhanced PINN dynamics
                        for station in dist_sys.ev_stations:
                            if hasattr(station, 'evcs_controller'):
                                # Get current grid conditions for PINN input
                                grid_voltage = getattr(dist_sys, 'current_voltage_level', 1.0)
                                system_frequency = 60.0  # Default frequency
                                
                                # Create input features for enhanced PINN
                                input_features = [
                                    station.soc,  # Current SOC
                                    grid_voltage,  # Grid voltage (pu)
                                    system_frequency,  # System frequency
                                    1.0,  # Demand factor (normalized)
                                    1.0,  # Voltage priority
                                    1.0,  # Urgency factor
                                    self.simulation_time / 3600.0,  # Time in hours
                                    grid_voltage,  # Bus voltage
                                    1.0,  # Base load factor
                                    station.current_load / 1000.0,  # Previous power (MW)
                                    0.0,  # AC power in (placeholder)
                                    0.85,  # System efficiency (placeholder)
                                    0.0,  # Power balance error (placeholder)
                                    0.0   # DC link voltage deviation (placeholder)
                                ]
                                
                                # Use enhanced PINN to predict optimal charging parameters
                                try:
                                    # Convert to tensor format expected by PINN
                                    import torch
                                    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
                                    
                                    # Get PINN prediction
                                    with torch.no_grad():
                                        pinn_output = enhanced_pinn_model.predict(input_tensor)
                                        
                                    if pinn_output is not None:
                                        # Extract predicted values
                                        predicted_voltage = pinn_output[0, 0].item() if len(pinn_output.shape) > 1 else pinn_output[0].item()
                                        predicted_current = pinn_output[0, 1].item() if len(pinn_output.shape) > 1 else pinn_output[1].item()
                                        predicted_power = pinn_output[0, 2].item() if len(pinn_output.shape) > 1 else pinn_output[2].item()
                                        
                                        # Update station with enhanced PINN predictions
                                        station.voltage_measured = predicted_voltage
                                        station.current_measured = predicted_current
                                        station.power_measured = predicted_power
                                        
                                        # Update EVCS controller with enhanced references
                                        station.evcs_controller.set_references(predicted_voltage, predicted_current, predicted_power)
                                        
                                        # Log enhanced PINN usage (every 60 seconds to avoid spam)
                                        if self.simulation_time % 60 == 0:
                                            print(f"  🔬 System {sys_id} EVCS {station.evcs_id}: Enhanced PINN active")
                                            print(f"     Predicted: V={predicted_voltage:.1f}V, I={predicted_current:.1f}A, P={predicted_power:.2f}kW")
                                        
                                except Exception as e:
                                    # Fallback to standard dynamics if PINN fails
                                    if self.simulation_time % 60 == 0:
                                        print(f"  ⚠️  System {sys_id} EVCS {station.evcs_id}: PINN failed, using standard dynamics: {e}")
                                    
                                    # Use standard EVCS dynamics as fallback
                                    if hasattr(station.evcs_controller, '_update_dynamics_euler'):
                                        try:
                                            grid_voltage_v = grid_voltage * 7200.0  # Convert pu to V
                                            dt_simulation = 1.0  # 1 second time step
                                            
                                            dynamics_result = station.evcs_controller._update_dynamics_euler(grid_voltage_v, dt_simulation)
                                            
                                            # Update station with standard dynamics results
                                            station.voltage_measured = dynamics_result.get('voltage_measured', station.voltage_measured)
                                            station.current_measured = dynamics_result.get('current_measured', station.current_measured)
                                            station.power_measured = dynamics_result.get('total_power', station.power_measured)
                                            
                                        except Exception as fallback_error:
                                            if self.simulation_time % 60 == 0:
                                                print(f"    ⚠️  Standard dynamics also failed: {fallback_error}")
                    
                    except Exception as e:
                        if self.simulation_time % 60 == 0:
                            print(f"  ⚠️  System {sys_id}: Enhanced PINN integration failed: {e}")
                
                # Get total load"""
    
    # Find the old section to replace
    old_section = """                            station.update_recovery(self.simulation_time)
                
                # Get total load"""
    
    if old_section in content:
        new_content = content.replace(old_section, enhanced_pinn_code)
        
        # Write back to file
        with open('hierarchical_cosimulation.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully added enhanced PINN integration to hierarchical_cosimulation.py")
        return True
    else:
        print("❌ Could not find the section to modify")
        return False

if __name__ == "__main__":
    add_enhanced_pinn_integration()
