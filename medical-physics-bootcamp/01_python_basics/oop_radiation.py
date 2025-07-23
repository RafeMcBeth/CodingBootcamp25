"""Object-Oriented Programming with Radiation Concepts.

This demonstrates how to create classes and objects using medical physics examples.
Classes are blueprints for creating objects that bundle data and functionality together.
"""


class RadioactiveSource:
    """A class representing a radioactive source.
    
    Classes let us create custom types that bundle related data and functions.
    This class models radioactive decay and dose calculations.
    """
    
    def __init__(self, isotope, half_life_days, initial_activity):
        """Initialize a new radioactive source.
        
        __init__ is a special method called when creating a new object.
        It sets up the initial state of our radioactive source.
        
        Args:
            isotope: Name of the radioactive isotope (e.g., "Co-60")
            half_life_days: Half-life in days
            initial_activity: Starting activity in Becquerels (Bq)
        """
        # Instance variables - each source object has its own values
        self.isotope = isotope
        self.half_life_days = half_life_days
        self.initial_activity = initial_activity
        
    def current_activity(self, days_elapsed):
        """Calculate current activity after radioactive decay.
        
        This is a method - a function that belongs to the class.
        Methods can access the object's data using 'self'.
        
        Args:
            days_elapsed: Number of days since initial measurement
            
        Returns:
            Current activity in Becquerels
        """
        # Radioactive decay formula: A(t) = A₀ * (1/2)^(t/t_half)
        decay_factor = (0.5) ** (days_elapsed / self.half_life_days)
        return self.initial_activity * decay_factor
        
    def dose_rate_at_distance(self, distance_cm, days_elapsed=0):
        """Calculate dose rate at a given distance.
        
        Simplified dose rate calculation for educational purposes.
        
        Args:
            distance_cm: Distance from source in centimeters
            days_elapsed: Days since initial measurement (default: 0)
            
        Returns:
            Dose rate in mSv/hour (simplified calculation)
        """
        current_act = self.current_activity(days_elapsed)
        # Simplified: dose rate decreases with square of distance
        # This is just for demonstration - real calculations are more complex
        dose_rate = (current_act / 1e9) / (distance_cm ** 2) * 1000
        return dose_rate
        
    def __str__(self):
        """Return a human-readable description of the source.
        
        __str__ is a special method that defines how the object appears
        when converted to a string (e.g., when using print()).
        """
        return f"{self.isotope} source: {self.initial_activity:.2e} Bq, t½={self.half_life_days} days"


def main():
    """Demonstrate object-oriented programming concepts."""
    print("Object-Oriented Programming with Radioactive Sources")
    print("=" * 55)
    
    # CREATE OBJECTS: Make instances of our RadioactiveSource class
    # Each object has its own independent data
    co60_source = RadioactiveSource("Co-60", 1925, 3.7e10)  # Cobalt-60 source
    cs137_source = RadioactiveSource("Cs-137", 11000, 1e9)  # Cesium-137 source
    
    # USE OBJECT METHODS: Call functions that belong to the objects
    print(f"\nSource 1: {co60_source}")  # Uses __str__ method
    print(f"Source 2: {cs137_source}")
    
    # Calculate activities after 5 years (1825 days)
    days_elapsed = 1825
    print(f"\nAfter {days_elapsed} days ({days_elapsed/365:.1f} years):")
    
    co60_activity = co60_source.current_activity(days_elapsed)
    cs137_activity = cs137_source.current_activity(days_elapsed)
    
    print(f"Co-60 activity: {co60_activity:.2e} Bq")
    print(f"Cs-137 activity: {cs137_activity:.2e} Bq")
    
    # Calculate dose rates at 1 meter distance
    distance = 100  # cm
    print(f"\nDose rates at {distance} cm distance:")
    
    co60_dose_rate = co60_source.dose_rate_at_distance(distance, days_elapsed)
    cs137_dose_rate = cs137_source.dose_rate_at_distance(distance, days_elapsed)
    
    print(f"Co-60: {co60_dose_rate:.3f} mSv/hour")
    print(f"Cs-137: {cs137_dose_rate:.3f} mSv/hour")
    
    print("\n🎓 You've learned:")
    print("  ✓ How to define classes with __init__ and methods")
    print("  ✓ How to create objects from classes")
    print("  ✓ How objects store their own data")
    print("  ✓ How to call methods on objects")
    print("  ✓ Real applications in medical physics!")


if __name__ == "__main__":
    main() 