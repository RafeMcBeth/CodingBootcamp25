"""Command-line interface for dose calculator."""

import argparse
from dose_calculator import calculate_dose, safety_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute radiation dose")
    parser.add_argument("dose_rate", type=float, help="Dose rate in Gy/min")
    parser.add_argument("time", type=float, help="Exposure time in minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_dose = calculate_dose(args.dose_rate, args.time)
    msg, _ = safety_check(total_dose)
    print(f"Total dose: {total_dose:.2f} Gy")
    print(msg)


if __name__ == "__main__":
    main()
