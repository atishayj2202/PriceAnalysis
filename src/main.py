import os
import sys
import subprocess

# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_tests import run_all_tests

def main():
    print("==================================================")
    print("📈 STARTING DEMAND APPROXIMATION SYSTEM")
    print("==================================================")
    
    # 1. First run testing
    print("\n[STEP 1/2] Running automated regression and agent tests...")
    try:
        tests_passed = run_all_tests()
    except Exception as e:
        print(f"❌ Testing failed with error: {str(e)}")
        tests_passed = False
        
    if not tests_passed:
        print("\n❌ System tests failed. Halting startup. Please fix regression tests before launching.")
        sys.exit(1)
        
    # 2. Start Streamlit dashboard
    print("\n[STEP 2/2] Tests passed successfully! Launching Streamlit dashboard...")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Command to run streamlit
    cmd = ["streamlit", "run", app_path]
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run streamlit as a subprocess
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"\n❌ Failed to run Streamlit: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
