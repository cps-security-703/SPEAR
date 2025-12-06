#!/usr/bin/env python3
"""
Script to add missing OS import to hierarchical_cosimulation.py
"""

def add_os_import():
    """Add missing OS import to the file"""
    
    # Read the file
    with open('hierarchical_cosimulation.py', 'r') as f:
        content = f.read()
    
    # Find the right place to add the import
    import_section = """# Import enhanced RL attack system
try:
    from enhanced_rl_attack_system import EnhancedRLAttackSystem
except ImportError:
    print("Warning: Enhanced RL attack system not available")
    EnhancedRLAttackSystem = None

# Import os for file operations
import os"""
    
    # Replace the section
    old_section = """# Import enhanced RL attack system
try:
    from enhanced_rl_attack_system import EnhancedRLAttackSystem
except ImportError:
    print("Warning: Enhanced RL attack system not available")
    EnhancedRLAttackSystem = None"""
    
    if old_section in content:
        new_content = content.replace(old_section, import_section)
        
        # Write back to file
        with open('hierarchical_cosimulation.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully added OS import to hierarchical_cosimulation.py")
        return True
    else:
        print("❌ Could not find the import section to modify")
        return False

if __name__ == "__main__":
    add_os_import()
