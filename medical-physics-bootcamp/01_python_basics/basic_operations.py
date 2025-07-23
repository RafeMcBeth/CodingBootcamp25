"""Demonstrates basic Python syntax and functions.

This file teaches fundamental Python concepts through a medical physics example.
We'll work with radiation dose measurements to learn programming basics.
"""


def calculate_mean(values):
    """Return the average of a list of numbers.
    
    This is a function definition. Functions are reusable blocks of code.
    
    Args:
        values: A list of numbers to calculate the average from
        
    Returns:
        The arithmetic mean (average) of the input values
    """
    # sum() adds all numbers in the list
    # len() returns how many items are in the list
    # Division (/) gives us the average
    return sum(values) / len(values)


def main():
    """Main function that demonstrates basic Python concepts."""
    
    # VARIABLES: Store data in memory with descriptive names
    # This is a list - a collection of values in square brackets
    doses = [1.2, 2.5, 3.1, 4.0]  # Radiation doses in Gray (Gy) units
    print("Demonstration of basic Python concepts with dose data:")
    print("-" * 50)  # Print a line separator
    
    # FOR LOOP: Repeat code for each item in a collection
    # 'd' is a temporary variable that holds each dose value one at a time
    print("\n1. LOOP EXAMPLE - Processing each dose:")
    for d in doses:
        # f-strings (f"...") let us insert variables into text
        # The {d} gets replaced with the actual dose value
        print(f"   Measured dose: {d} Gy")
    
    # CONDITIONAL STATEMENTS: Make decisions based on conditions
    # 'if' statements run code only when a condition is True
    print("\n2. CONDITIONAL EXAMPLE - Checking for high doses:")
    if max(doses) > 3:  # max() finds the largest value in the list
        print("   ⚠️  High dose detected!")
        print(f"   Maximum dose is {max(doses)} Gy")
    else:
        print("   ✓ All doses are within safe range")
    
    # FUNCTION USAGE: Call our custom function to do calculations
    print("\n3. FUNCTION EXAMPLE - Calculating statistics:")
    mean_dose = calculate_mean(doses)  # Call our function and store the result
    
    # The :.2f means "format as a number with 2 decimal places"
    print(f"   Average dose: {mean_dose:.2f} Gy")
    
    # ADDITIONAL EXAMPLES for learning
    print(f"   Total of all doses: {sum(doses):.2f} Gy")
    print(f"   Number of measurements: {len(doses)}")
    print(f"   Highest dose: {max(doses)} Gy")
    print(f"   Lowest dose: {min(doses)} Gy")


# This special condition runs main() only when the script is executed directly
# (not when imported as a module in another file)
if __name__ == "__main__":
    main() 