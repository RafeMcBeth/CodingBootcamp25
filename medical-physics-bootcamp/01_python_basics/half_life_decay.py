"""Calculate the decayed activity of a radioisotope using its half-life.

This advanced example demonstrates:
- Dataclasses for clean object modeling
- Command-line argument parsing
- Mathematical modeling of radioactive decay
- Professional Python code structure
"""

from dataclasses import dataclass
import argparse
import math


@dataclass
class Radioisotope:
    """A dataclass representing a radioactive isotope.
    
    Dataclasses automatically generate __init__, __repr__, and other methods.
    This is a modern Python way to create simple classes that hold data.
    """
    name: str
    half_life_hours: float
    initial_activity_bq: float

    def activity_at(self, elapsed_hours: float) -> float:
        """Calculate activity after elapsed time using the decay law.
        
        Uses the radioactive decay equation: A(t) = A₀ * e^(-λt)
        where λ = ln(2) / t½ is the decay constant.
        
        Args:
            elapsed_hours: Time elapsed since initial measurement
            
        Returns:
            Activity in Becquerels after decay
        """
        decay_constant = math.log(2) / self.half_life_hours
        return self.initial_activity_bq * math.exp(-decay_constant * elapsed_hours)
    
    def time_to_activity(self, target_activity_bq: float) -> float:
        """Calculate time needed to decay to a target activity.
        
        Solves: target = initial * e^(-λt) for t
        Result: t = ln(initial/target) / λ
        
        Args:
            target_activity_bq: Desired activity level
            
        Returns:
            Time in hours to reach target activity
        """
        if target_activity_bq >= self.initial_activity_bq:
            return 0.0
        
        decay_constant = math.log(2) / self.half_life_hours
        return math.log(self.initial_activity_bq / target_activity_bq) / decay_constant
    
    def fraction_remaining(self, elapsed_hours: float) -> float:
        """Calculate what fraction of the original activity remains.
        
        Args:
            elapsed_hours: Time elapsed since initial measurement
            
        Returns:
            Fraction between 0 and 1
        """
        return self.activity_at(elapsed_hours) / self.initial_activity_bq


def demonstrate_common_isotopes():
    """Demonstrate decay calculations for common medical isotopes."""
    print("\n🔬 Common Medical Physics Isotopes:")
    print("=" * 45)
    
    # Common isotopes used in medical physics
    isotopes = [
        Radioisotope("Co-60", 5.27 * 365 * 24, 3.7e10),     # Cobalt-60: 5.27 years
        Radioisotope("Cs-137", 30.17 * 365 * 24, 3.7e9),     # Cesium-137: 30.17 years  
        Radioisotope("I-125", 59.4 * 24, 1e8),               # Iodine-125: 59.4 days
        Radioisotope("Tc-99m", 6.01, 1e9),                   # Technetium-99m: 6.01 hours
    ]
    
    # Show activity after 1 year for each
    hours_in_year = 365 * 24
    
    for iso in isotopes:
        activity_after_year = iso.activity_at(hours_in_year)
        fraction_remaining = iso.fraction_remaining(hours_in_year)
        
        print(f"{iso.name:8}: {iso.initial_activity_bq:.1e} Bq → "
              f"{activity_after_year:.1e} Bq ({fraction_remaining:.1%} remaining)")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Calculate radioactive decay for medical physics applications",
        epilog="Example: python half_life_decay.py Co-60 46000 3.7e10 8760\n"
               "Run without arguments to see a demonstration with common isotopes."
    )
    parser.add_argument("name", nargs='?', help="Isotope name (e.g., Co-60, Cs-137)")
    parser.add_argument("half_life", nargs='?', type=float, help="Half-life in hours")
    parser.add_argument("activity", nargs='?', type=float, help="Initial activity in Bq")
    parser.add_argument("hours", nargs='?', type=float, help="Elapsed time in hours")
    args = parser.parse_args()

    # If no arguments provided, show demonstration
    if not all([args.name, args.half_life, args.activity, args.hours]):
        print("🎯 No arguments provided - showing demonstration with common medical isotopes!\n")
        print("💡 To calculate for a specific isotope, use:")
        print("   python half_life_decay.py <name> <half_life_hours> <activity_bq> <elapsed_hours>")
        print("   Example: python half_life_decay.py Co-60 46000 3.7e10 8760\n")
        demonstrate_common_isotopes()
        return

    # Create isotope object and calculate decay
    iso = Radioisotope(args.name, args.half_life, args.activity)
    remaining = iso.activity_at(args.hours)
    fraction = iso.fraction_remaining(args.hours)
    
    print(f"\n☢️  RADIOACTIVE DECAY CALCULATION")
    print("=" * 40)
    print(f"Isotope: {iso.name}")
    print(f"Half-life: {args.half_life:,.1f} hours ({args.half_life/24:.1f} days)")
    print(f"Initial activity: {iso.initial_activity_bq:.2e} Bq")
    print(f"Time elapsed: {args.hours:,.1f} hours ({args.hours/24:.1f} days)")
    print(f"Current activity: {remaining:.2e} Bq")
    print(f"Fraction remaining: {fraction:.1%}")
    
    # Additional useful information
    if fraction > 0.5:
        print("✓ More than half the activity remains")
    elif fraction > 0.25:
        print("⚠️  Between 25% and 50% activity remains")
    else:
        print("❗ Less than 25% activity remains")
    
    # When will it reach 1% of original activity?
    time_to_1_percent = iso.time_to_activity(iso.initial_activity_bq * 0.01)
    print(f"Time to reach 1% activity: {time_to_1_percent:.1f} hours ({time_to_1_percent/24:.1f} days)")
    
    # Show additional demonstration
    print("\n" + "="*50)
    demonstrate_common_isotopes()


if __name__ == "__main__":
    main() 