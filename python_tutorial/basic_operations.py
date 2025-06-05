"""Demonstrates basic Python syntax and functions."""


def calculate_mean(values):
    """Return the average of a list of numbers."""
    return sum(values) / len(values)


def main():
    # Variables
    doses = [1.2, 2.5, 3.1, 4.0]

    # Loop
    for d in doses:
        print(f"Measured dose: {d} Gy")

    # Conditional
    if max(doses) > 3:
        print("High dose detected!")

    # Function usage
    mean_dose = calculate_mean(doses)
    print(f"Average dose: {mean_dose:.2f} Gy")


if __name__ == "__main__":
    main()
