"""Debugging Tutorial for Medical Physics Programming.

This module teaches you how to find and fix bugs in your code.
Debugging is a critical skill for any programmer!
"""

import pdb  # Python debugger - built into Python


def calculate_dose_rate(activity_bq, distance_cm):
    """Calculate dose rate from a radioactive source.
    
    This function has a deliberate bug for teaching purposes!
    
    Args:
        activity_bq: Activity in Becquerels
        distance_cm: Distance from source in centimeters
        
    Returns:
        Dose rate in mSv/hour
    """
    # BUG ALERT: This calculation has an error!
    # Can you spot it before running the debugger?
    dose_constant = 1.3e-7  # Simplified dose constant for Cs-137
    distance_m = distance_cm / 100  # Convert cm to meters
    
    # The inverse square law: dose rate ∝ 1/distance²
    # But there's a bug here...
    dose_rate = (activity_bq * dose_constant) / distance_m  # Missing square!
    
    return dose_rate


def main():
    """Demonstrate debugging techniques."""
    print("Debugging Tutorial - Finding Bugs in Dose Calculations")
    print("=" * 55)
    
    # Test data
    cs137_activity = 3.7e9  # 3.7 GBq Cs-137 source
    distances = [50, 100, 200]  # cm
    
    print("\nCalculating dose rates at different distances:")
    print("Distance (cm) | Dose Rate (mSv/h)")
    print("-" * 35)
    
    for distance in distances:
        dose_rate = calculate_dose_rate(cs137_activity, distance)
        print(f"{distance:>8} cm   | {dose_rate:>10.3f}")
    
    print("\n🔍 DEBUGGING EXERCISE:")
    print("1. Look at the results above - do they make sense?")
    print("2. The dose rate should DECREASE as distance increases")
    print("3. But our results might show the opposite!")
    print("4. Let's debug this...")
    
    # DEBUGGING TECHNIQUE 1: Add print statements
    print("\n--- Debug Method 1: Print Statements ---")
    distance_test = 100
    activity_test = cs137_activity
    
    print(f"Input: activity = {activity_test:.2e} Bq, distance = {distance_test} cm")
    
    # Let's trace through the calculation step by step
    dose_constant = 1.3e-7
    distance_m = distance_test / 100
    print(f"Distance in meters: {distance_m}")
    
    # The buggy calculation
    dose_rate_buggy = (activity_test * dose_constant) / distance_m
    print(f"Buggy result: {dose_rate_buggy:.3f} mSv/h")
    
    # The correct calculation (inverse square law)
    dose_rate_correct = (activity_test * dose_constant) / (distance_m ** 2)
    print(f"Correct result: {dose_rate_correct:.3f} mSv/h")
    
    print("\n--- Debug Method 2: Interactive Debugger ---")
    print("Uncomment the line below and run this script to use the debugger:")
    print("# pdb.set_trace()  # This starts the interactive debugger")
    print("\nDebugger commands:")
    print("  n = next line")
    print("  s = step into function")
    print("  l = list current code")
    print("  p variable_name = print variable value")
    print("  c = continue execution")
    print("  q = quit debugger")
    
    # Uncomment this line to start the debugger:
    # pdb.set_trace()
    
    print("\n🎓 You've learned:")
    print("  ✓ How to spot logical errors in calculations")
    print("  ✓ Using print statements for debugging")
    print("  ✓ How to use Python's built-in debugger (pdb)")
    print("  ✓ The importance of testing with realistic data")
    print("  ✓ Physics knowledge helps catch programming bugs!")


if __name__ == "__main__":
    main() 