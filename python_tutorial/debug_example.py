"""Example script to demonstrate the use of the pdb debugger."""


def compute_dose_rate(dose, time):
    """Return dose rate as dose divided by time."""
    return dose / time


def main():
    dose = 5
    time = 0

    import pdb; pdb.set_trace()  # Set a breakpoint
    rate = compute_dose_rate(dose, time)  # This will raise ZeroDivisionError
    print(rate)


if __name__ == "__main__":
    main()
