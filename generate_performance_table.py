
import os
import runpy
import sys

print("=" * 88)
print("DEPRECATED: generate_performance_table.py compared gemini-vs-gemini on confounded reward.")
print("Redirecting to compare_gemini_vs_autonomous.py (clean impact/stealth metrics).")
print("Original preserved at deprecated_analysis/generate_performance_table.py")
print("=" * 88)

_here = os.path.dirname(os.path.abspath(__file__))
_new = os.path.join(_here, "compare_gemini_vs_autonomous.py")
if os.path.exists(_new):
    sys.argv = [_new]
    runpy.run_path(_new, run_name="__main__")
else:
    print("ERROR: compare_gemini_vs_autonomous.py not found next to this script.")
