"""Calculate the decayed activity of a radioisotope using its half-life."""

from dataclasses import dataclass
import argparse
import math


@dataclass
class Radioisotope:
    name: str
    half_life_hours: float
    initial_activity_bq: float

    def activity_at(self, elapsed_hours: float) -> float:
        """Return activity after ``elapsed_hours`` hours."""
        decay_constant = math.log(2) / self.half_life_hours
        return self.initial_activity_bq * math.exp(-decay_constant * elapsed_hours)


def main():
    parser = argparse.ArgumentParser(description="Decay calculator")
    parser.add_argument("name", help="Isotope name")
    parser.add_argument("half_life", type=float, help="Half-life in hours")
    parser.add_argument("activity", type=float, help="Initial activity in Bq")
    parser.add_argument("hours", type=float, help="Elapsed time in hours")
    args = parser.parse_args()

    iso = Radioisotope(args.name, args.half_life, args.activity)
    remaining = iso.activity_at(args.hours)
    print(
        f"{iso.name} activity after {args.hours} h: {remaining:.2e} Bq"
    )


if __name__ == "__main__":
    main()
