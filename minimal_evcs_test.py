#!/usr/bin/env python3
"""
Minimal EVCS test to identify and fix the evcs_id issue
"""

def test_evcs_creation():
    """Test EVCS creation with different parameter combinations"""
    print("🧪 Testing EVCS Creation")
    print("=" * 40)
    
    try:
        from hierarchical_cosimulation import EVChargingStation, EnhancedChargingManagementSystem
        
        # Test 1: Basic EVChargingStation creation
        print("1. Testing EVChargingStation creation...")
        
        # Try different parameter combinations
        test_cases = [
            {
                'name': 'With evcs_id parameter',
                'params': {
                    'evcs_id': 'TEST_001',
                    'bus_name': 'Bus_1',
                    'max_power': 1000,
                    'num_ports': 4
                }
            },
            {
                'name': 'Without evcs_id parameter',
                'params': {
                    'bus_name': 'Bus_1',
                    'max_power': 1000,
                    'num_ports': 4
                }
            },
            {
                'name': 'With all parameters',
                'params': {
                    'evcs_id': 'TEST_002',
                    'bus_name': 'Bus_2',
                    'max_power': 2000,
                    'num_ports': 8,
                    'current_load': 500
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"   Test {i}: {test_case['name']}")
                station = EVChargingStation(**test_case['params'])
                print(f"   ✅ Success: {station.evcs_id if hasattr(station, 'evcs_id') else 'No evcs_id'}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Test 2: EnhancedChargingManagementSystem creation
        print("\n2. Testing EnhancedChargingManagementSystem creation...")
        
        try:
            # Create a working station first
            station = EVChargingStation(
                evcs_id='WORKING_001',
                bus_name='Bus_1',
                max_power=1000,
                num_ports=4
            )
            
            # Create CMS
            cms = EnhancedChargingManagementSystem(
                stations=[station],
                use_pinn=True
            )
            print(f"   ✅ CMS created with {len(cms.stations)} stations")
            
        except Exception as e:
            print(f"   ❌ CMS creation failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_evcs_constructor():
    """Check the EVChargingStation constructor signature"""
    print("\n🔍 Checking EVChargingStation Constructor")
    print("=" * 40)
    
    try:
        from hierarchical_cosimulation import EVChargingStation
        import inspect
        
        # Get constructor signature
        sig = inspect.signature(EVChargingStation.__init__)
        print(f"EVChargingStation constructor parameters:")
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                print(f"   {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
                if param.default != inspect.Parameter.empty:
                    print(f"      Default: {param.default}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect constructor: {e}")
        return False

if __name__ == "__main__":
    print("🚀 EVCS Creation Test")
    print("=" * 50)
    
    # Check constructor signature
    check_evcs_constructor()
    
    # Test EVCS creation
    success = test_evcs_creation()
    
    if success:
        print("\n✅ EVCS creation test completed")
    else:
        print("\n❌ EVCS creation test failed")
    
    print("\n📋 If EVCS creation fails:")
    print("1. Check the EVChargingStation constructor in hierarchical_cosimulation.py")
    print("2. Make sure evcs_id parameter is properly handled")
    print("3. Check if there are any missing imports or dependencies")
