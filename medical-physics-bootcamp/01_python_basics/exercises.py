"""Interactive Python Exercises for Medical Physics Students.

This file contains hands-on exercises to practice the concepts you've learned.
Run this script and follow the prompts to test your understanding!
"""

import math
import random


def exercise_1_variables():
    """Exercise 1: Working with Variables and Basic Math."""
    print("🧮 EXERCISE 1: Variables and Basic Calculations")
    print("=" * 50)
    
    print("Calculate the half-value layer (HVL) for X-rays in aluminum.")
    print("Formula: HVL = ln(2) / μ, where μ is the linear attenuation coefficient")
    
    # Give student the values
    mu_aluminum = 0.435  # cm⁻¹ for 100 keV X-rays
    print(f"Given: μ (aluminum) = {mu_aluminum} cm⁻¹")
    
    # Let them calculate
    print("\nYour turn! Calculate the HVL:")
    
    # The correct answer
    correct_hvl = math.log(2) / mu_aluminum
    print(f"Correct answer: HVL = {correct_hvl:.3f} cm")
    
    # Interactive challenge
    print("\n💡 CHALLENGE: Try different energies!")
    energies = [50, 100, 150, 200]  # keV
    mu_values = [1.525, 0.435, 0.204, 0.136]  # cm⁻¹
    
    for energy, mu in zip(energies, mu_values):
        hvl = math.log(2) / mu
        print(f"   {energy} keV: μ = {mu:.3f} cm⁻¹, HVL = {hvl:.3f} cm")
    
    return correct_hvl


def exercise_2_lists_loops():
    """Exercise 2: Lists and Loops with Dose Measurements."""
    print("\n🔄 EXERCISE 2: Lists and Loops")
    print("=" * 40)
    
    print("Process a series of dose rate measurements from a linear accelerator.")
    
    # Realistic dose rate data (cGy/min)
    dose_rates = [598, 602, 595, 604, 590, 607, 593, 601, 599, 596]
    print(f"Dose rate measurements: {dose_rates} cGy/min")
    
    # Calculate statistics using loops
    total = 0
    count = 0
    max_dose = dose_rates[0]
    min_dose = dose_rates[0]
    
    print("\nProcessing measurements...")
    for i, dose_rate in enumerate(dose_rates):
        print(f"  Measurement {i+1}: {dose_rate} cGy/min")
        total += dose_rate
        count += 1
        
        if dose_rate > max_dose:
            max_dose = dose_rate
        if dose_rate < min_dose:
            min_dose = dose_rate
    
    average = total / count
    
    print(f"\n📊 RESULTS:")
    print(f"  Average dose rate: {average:.1f} cGy/min")
    print(f"  Maximum: {max_dose} cGy/min")
    print(f"  Minimum: {min_dose} cGy/min")
    print(f"  Range: {max_dose - min_dose} cGy/min")
    
    # Quality check
    tolerance = 5  # ±5 cGy/min
    if max_dose - min_dose <= tolerance:
        print("  ✅ PASS: Dose rate stability within tolerance")
    else:
        print("  ⚠️  FAIL: Dose rate variation exceeds tolerance")
    
    return average


def exercise_3_functions():
    """Exercise 3: Creating and Using Functions."""
    print("\n🔧 EXERCISE 3: Functions")
    print("=" * 30)
    
    print("Create functions for common medical physics calculations.")
    
    def percent_depth_dose(depth_cm, energy_mv=6):
        """Calculate percent depth dose for photon beams.
        
        Simplified exponential model for demonstration.
        """
        # Simplified parameters for 6 MV beam
        d_max = 1.5  # cm, depth of maximum dose
        mu_eff = 0.065  # cm⁻¹, effective attenuation coefficient
        
        if depth_cm <= d_max:
            # Build-up region
            pdd = (depth_cm / d_max) * 100
        else:
            # Exponential falloff
            pdd = 100 * math.exp(-mu_eff * (depth_cm - d_max))
        
        return pdd
    
    def tissue_air_ratio(depth_cm, field_size_cm=10):
        """Calculate tissue-air ratio for dose calculations."""
        # Simplified TAR calculation
        mu_eff = 0.065  # cm⁻¹
        scatter_factor = 1 + 0.001 * field_size_cm**2
        
        tar = scatter_factor * math.exp(-mu_eff * depth_cm)
        return tar
    
    # Test the functions
    print("\nTesting percent depth dose function:")
    depths = [0, 1.5, 5, 10, 15, 20]
    
    print("Depth (cm) | PDD (%)")
    print("-" * 20)
    for depth in depths:
        pdd = percent_depth_dose(depth)
        print(f"{depth:>6} cm   | {pdd:>6.1f}%")
    
    print("\nTesting tissue-air ratio function:")
    print("Depth (cm) | TAR")
    print("-" * 18)
    for depth in depths[1:]:  # Skip surface
        tar = tissue_air_ratio(depth)
        print(f"{depth:>6} cm   | {tar:>6.3f}")
    
    return percent_depth_dose, tissue_air_ratio


def exercise_4_conditionals():
    """Exercise 4: Conditional Logic for Safety Checks."""
    print("\n⚡ EXERCISE 4: Conditional Logic")
    print("=" * 35)
    
    print("Implement safety checks for radiation therapy delivery.")
    
    def safety_check(patient_id, prescribed_dose, delivered_dose, field_size):
        """Perform comprehensive safety checks before beam delivery."""
        
        print(f"\n🏥 SAFETY CHECK for Patient {patient_id}")
        print("-" * 40)
        
        # Dose tolerance check
        dose_tolerance = 0.05  # ±5%
        dose_difference = abs(delivered_dose - prescribed_dose) / prescribed_dose
        
        print(f"Prescribed dose: {prescribed_dose:.2f} cGy")
        print(f"Delivered dose:  {delivered_dose:.2f} cGy")
        print(f"Difference: {dose_difference*100:.1f}%")
        
        if dose_difference <= dose_tolerance:
            print("✅ PASS: Dose within tolerance")
            dose_ok = True
        else:
            print("❌ FAIL: Dose exceeds tolerance")
            dose_ok = False
        
        # Field size check
        min_field = 3.0  # cm
        max_field = 40.0  # cm
        
        print(f"\nField size: {field_size:.1f} cm")
        
        if min_field <= field_size <= max_field:
            print("✅ PASS: Field size within limits")
            field_ok = True
        else:
            print("❌ FAIL: Field size outside limits")
            field_ok = False
        
        # Overall safety verdict
        if dose_ok and field_ok:
            print("\n🟢 TREATMENT APPROVED")
            return True
        else:
            print("\n🔴 TREATMENT BLOCKED - REVIEW REQUIRED")
            return False
    
    # Test scenarios
    scenarios = [
        ("001", 200.0, 201.0, 10.0),  # Good case
        ("002", 180.0, 190.0, 8.5),   # Dose too high
        ("003", 220.0, 218.0, 45.0),  # Field too large
        ("004", 200.0, 175.0, 5.0),   # Dose too low
    ]
    
    passed = 0
    for patient_id, prescribed, delivered, field in scenarios:
        if safety_check(patient_id, prescribed, delivered, field):
            passed += 1
    
    print(f"\n📊 SUMMARY: {passed}/{len(scenarios)} treatments approved")
    
    return passed


def main():
    """Run all exercises interactively."""
    print("🎓 Medical Physics Python Exercises")
    print("=" * 60)
    print("Practice the concepts you've learned with real medical physics problems!")
    
    # Run all exercises
    hvl = exercise_1_variables()
    avg_dose = exercise_2_lists_loops()
    pdd_func, tar_func = exercise_3_functions()
    safety_passed = exercise_4_conditionals()
    
    # Final summary
    print("\n" + "="*60)
    print("🏆 EXERCISE SUMMARY")
    print("="*60)
    print(f"✓ Calculated HVL: {hvl:.3f} cm")
    print(f"✓ Processed dose measurements: {avg_dose:.1f} cGy/min average")
    print(f"✓ Created dose calculation functions")
    print(f"✓ Safety checks: {safety_passed}/4 treatments approved")
    
    print("\n🎉 Congratulations! You've completed all exercises!")
    print("You're ready to move on to building real applications!")
    
    # Bonus challenge
    print("\n💪 BONUS CHALLENGE:")
    print("Try modifying the functions to handle different beam energies,")
    print("add more sophisticated dose models, or create your own safety checks!")


if __name__ == "__main__":
    main() 