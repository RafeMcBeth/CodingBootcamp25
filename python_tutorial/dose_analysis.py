"""Compute basic statistics for dose measurements stored in a CSV file."""

import csv
from statistics import mean, stdev


def read_doses(file_path):
    """Return a list of float dose values from ``file_path``."""
    with open(file_path) as f:
        reader = csv.reader(f)
        return [float(row[0]) for row in reader if row]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation of dose measurements"
    )
    parser.add_argument(
        "csv_file",
        help="CSV file containing one dose value per line",
    )
    args = parser.parse_args()

    doses = read_doses(args.csv_file)
    print(f"Number of measurements: {len(doses)}")
    print(f"Mean dose: {mean(doses):.2f} Gy")
    if len(doses) > 1:
        print(f"Standard deviation: {stdev(doses):.2f} Gy")


if __name__ == "__main__":
    main()
