"""Advanced Dose Data Analysis with Command Line Arguments.

This script demonstrates how to process CSV data files and accept command line parameters.
Perfect for automating dose analysis workflows!
"""

import csv
import sys
from statistics import mean, stdev


def load_dose_data(filename):
    """Load dose measurements from a CSV file.
    
    Args:
        filename: Path to CSV file with dose data
        
    Returns:
        List of dose values (floats)
    """
    doses = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            
            for row in reader:
                if row:  # Skip empty rows
                    dose = float(row[0])  # Assume dose is in first column
                    doses.append(dose)
                    
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data in file - {e}")
        sys.exit(1)
        
    return doses


def analyze_doses(doses):
    """Perform statistical analysis on dose measurements.
    
    Args:
        doses: List of dose values
        
    Returns:
        Dictionary with analysis results
    """
    if not doses:
        return None
        
    results = {
        'count': len(doses),
        'mean': mean(doses),
        'std_dev': stdev(doses) if len(doses) > 1 else 0,
        'min': min(doses),
        'max': max(doses),
        'range': max(doses) - min(doses)
    }
    
    return results


def check_dose_limits(doses, limit=20.0):
    """Check which doses exceed safety limits.
    
    Args:
        doses: List of dose values
        limit: Maximum allowed dose (default: 20.0 mSv)
        
    Returns:
        List of doses that exceed the limit
    """
    return [dose for dose in doses if dose > limit]


def main():
    """Main function demonstrating command line argument processing."""
    
    # COMMAND LINE ARGUMENTS: Get filename from user
    if len(sys.argv) != 2:
        print("Usage: python dose_analysis.py <csv_filename>")
        print("Example: python dose_analysis.py sample_doses.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"Analyzing dose data from: {filename}")
    print("=" * 50)
    
    # Load and analyze data
    doses = load_dose_data(filename)
    
    if not doses:
        print("No dose data found in file.")
        return
    
    # Statistical analysis
    stats = analyze_doses(doses)
    
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"  Number of measurements: {stats['count']}")
    print(f"  Mean dose: {stats['mean']:.2f} mSv")
    print(f"  Standard deviation: {stats['std_dev']:.2f} mSv")
    print(f"  Minimum dose: {stats['min']:.2f} mSv")
    print(f"  Maximum dose: {stats['max']:.2f} mSv")
    print(f"  Range: {stats['range']:.2f} mSv")
    
    # Safety analysis
    dose_limit = 20.0  # mSv annual limit for general public
    excessive_doses = check_dose_limits(doses, dose_limit)
    
    print(f"\n⚡ SAFETY ANALYSIS:")
    print(f"  Dose limit: {dose_limit} mSv")
    
    if excessive_doses:
        print(f"  ⚠️  {len(excessive_doses)} measurements exceed limit!")
        print(f"  Excessive doses: {excessive_doses}")
    else:
        print(f"  ✅ All measurements within safe limits")
    
    # Data quality checks
    print(f"\n🔍 DATA QUALITY:")
    if stats['std_dev'] > stats['mean'] * 0.5:
        print("  ⚠️  High variability in measurements - check calibration")
    else:
        print("  ✅ Measurement variability appears normal")
    
    print(f"\n🎓 This script demonstrates:")
    print(f"  ✓ Reading CSV files with error handling")
    print(f"  ✓ Command line argument processing")
    print(f"  ✓ Statistical analysis with built-in functions")
    print(f"  ✓ Automated safety limit checking")
    print(f"  ✓ Professional error reporting")


if __name__ == "__main__":
    main() 