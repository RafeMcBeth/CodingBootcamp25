"""Hello World example for medical physics bootcamp.

This is your very first Python program! We'll start simple and build up to complex applications.
"""

# This is a comment - Python ignores everything after the # symbol
# Comments help explain what our code does

# BASIC OUTPUT: The print() function displays text to the screen
print("Hello Medical Physics World!")

# VARIABLES: Store information in memory with descriptive names
student_name = "Future Medical Physicist"  # This is a string (text)
dose_limit = 20.0  # This is a number (float)

# Using variables in our output
print(f"Welcome {student_name}!")
print(f"Annual dose limit: {dose_limit} mSv")

# BASIC CALCULATION: Python can do math
daily_background = 0.01  # mSv per day
days_in_year = 365
annual_background = daily_background * days_in_year

print(f"Annual background radiation: {annual_background:.2f} mSv")

# SIMPLE COMPARISON: Is background dose safe?
if annual_background < dose_limit:
    print("✓ Background radiation is well within safe limits!")
else:
    print("⚠️ Background radiation exceeds limits")

print("\nCongratulations! You've just run your first medical physics Python program! 🎉")
