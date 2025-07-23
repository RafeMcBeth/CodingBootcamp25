"""CT Image Analysis for Medical Physics Students.

This script demonstrates medical image processing concepts using simulated CT data.
You'll learn about Hounsfield units, image thresholding, and tissue segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def generate_synthetic_ct_slice():
    """Generate a synthetic CT slice for educational purposes.
    
    Returns:
        2D numpy array representing a CT slice with realistic Hounsfield values
    """
    print("🖼️  Generating synthetic CT slice...")
    
    # Create a 256x256 pixel CT slice
    size = 256
    ct_slice = np.full((size, size), -1000)  # Start with air (-1000 HU)
    
    # Add patient outline (soft tissue ~40 HU)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    
    # Outer body contour (elliptical)
    body_mask = ((x - center) / 80)**2 + ((y - center) / 100)**2 <= 1
    ct_slice[body_mask] = 40  # Soft tissue
    
    # Add lungs (air-filled, ~-800 HU)
    lung_left = ((x - center + 40) / 30)**2 + ((y - center) / 40)**2 <= 1
    lung_right = ((x - center - 40) / 30)**2 + ((y - center) / 40)**2 <= 1
    ct_slice[lung_left | lung_right] = -800
    
    # Add heart (muscle tissue, ~50 HU)
    heart = ((x - center - 10) / 20)**2 + ((y - center + 20) / 25)**2 <= 1
    ct_slice[heart] = 50
    
    # Add spine (bone, ~400 HU)
    spine = ((x - center) / 8)**2 + ((y - center - 60) / 15)**2 <= 1
    ct_slice[spine] = 400
    
    # Add ribs (bone, ~300 HU)
    for angle in np.linspace(0, 2*np.pi, 12):
        rib_x = center + 70 * np.cos(angle)
        rib_y = center + 85 * np.sin(angle)
        if 0 <= rib_x < size and 0 <= rib_y < size:
            rib_mask = ((x - rib_x) / 3)**2 + ((y - rib_y) / 8)**2 <= 1
            ct_slice[rib_mask] = 300
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, (size, size))
    ct_slice = ct_slice + noise
    
    print(f"✓ Created {size}x{size} synthetic CT slice")
    print(f"  HU range: {ct_slice.min():.0f} to {ct_slice.max():.0f}")
    
    return ct_slice


def display_ct_slice(ct_data, title="CT Slice"):
    """Display CT data with appropriate windowing.
    
    Args:
        ct_data: 2D numpy array of CT data
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Standard soft tissue window (W=400, L=40)
    window_width = 400
    window_level = 40
    
    vmin = window_level - window_width / 2
    vmax = window_level + window_width / 2
    
    plt.imshow(ct_data, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Hounsfield Units (HU)')
    plt.title(f'{title}\nSoft Tissue Window (W={window_width}, L={window_level})')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    
    return plt.gcf()


def analyze_hounsfield_distribution(ct_data):
    """Analyze the distribution of Hounsfield units in the image.
    
    Args:
        ct_data: 2D numpy array of CT data
    """
    print("\n📊 HOUNSFIELD UNIT ANALYSIS")
    print("=" * 40)
    
    # Basic statistics
    print(f"Image dimensions: {ct_data.shape}")
    print(f"Total pixels: {ct_data.size:,}")
    print(f"HU range: {ct_data.min():.1f} to {ct_data.max():.1f}")
    print(f"Mean HU: {ct_data.mean():.1f}")
    print(f"Standard deviation: {ct_data.std():.1f}")
    
    # Tissue type analysis based on HU ranges
    tissue_ranges = {
        'Air': (-1000, -500),
        'Lung': (-500, -100),
        'Fat': (-100, -50),
        'Water': (-50, 50),
        'Soft Tissue': (50, 150),
        'Bone': (150, 3000)
    }
    
    print(f"\n🔬 TISSUE SEGMENTATION:")
    total_pixels = ct_data.size
    
    for tissue, (min_hu, max_hu) in tissue_ranges.items():
        mask = (ct_data >= min_hu) & (ct_data < max_hu)
        pixel_count = np.sum(mask)
        percentage = (pixel_count / total_pixels) * 100
        
        print(f"  {tissue:12}: {pixel_count:6,} pixels ({percentage:5.1f}%) "
              f"[{min_hu:4.0f} to {max_hu:4.0f} HU]")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(ct_data.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Hounsfield Units (HU)')
    plt.ylabel('Pixel Count')
    plt.title('HU Distribution - Full Range')
    plt.grid(True, alpha=0.3)
    
    # Zoomed histogram (excluding air)
    plt.subplot(1, 2, 2)
    tissue_data = ct_data[ct_data > -500]  # Exclude air
    plt.hist(tissue_data.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Hounsfield Units (HU)')
    plt.ylabel('Pixel Count')
    plt.title('HU Distribution - Tissue Only')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()


def apply_threshold_segmentation(ct_data):
    """Demonstrate image thresholding for tissue segmentation.
    
    Args:
        ct_data: 2D numpy array of CT data
        
    Returns:
        Dictionary of segmented tissue masks
    """
    print("\n🎯 THRESHOLD SEGMENTATION")
    print("=" * 35)
    
    # Define thresholds for different tissues
    thresholds = {
        'bone': 150,      # Bone vs soft tissue
        'soft_tissue': 0, # Soft tissue vs fat/air
        'lung': -500,     # Lung vs air
    }
    
    # Create binary masks
    masks = {}
    masks['bone'] = ct_data >= thresholds['bone']
    masks['soft_tissue'] = (ct_data >= thresholds['soft_tissue']) & (ct_data < thresholds['bone'])
    masks['lung'] = (ct_data >= thresholds['lung']) & (ct_data < thresholds['soft_tissue'])
    masks['air'] = ct_data < thresholds['lung']
    
    # Print segmentation results
    for tissue, mask in masks.items():
        pixel_count = np.sum(mask)
        percentage = (pixel_count / ct_data.size) * 100
        print(f"  {tissue.title():12}: {pixel_count:6,} pixels ({percentage:5.1f}%)")
    
    # Create color-coded segmentation map
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(ct_data, cmap='gray', vmin=-200, vmax=400)
    plt.title('Original CT Image')
    plt.colorbar(label='HU')
    
    # Binary bone mask
    plt.subplot(1, 3, 2)
    plt.imshow(masks['bone'], cmap='Reds')
    plt.title('Bone Segmentation\n(HU ≥ 150)')
    plt.colorbar()
    
    # Multi-class segmentation
    plt.subplot(1, 3, 3)
    segmentation = np.zeros_like(ct_data)
    segmentation[masks['air']] = 1
    segmentation[masks['lung']] = 2
    segmentation[masks['soft_tissue']] = 3
    segmentation[masks['bone']] = 4
    
    colors = ['black', 'blue', 'cyan', 'green', 'red']
    cmap = ListedColormap(colors)
    
    plt.imshow(segmentation, cmap=cmap)
    plt.title('Multi-class Segmentation')
    plt.colorbar(ticks=[0, 1, 2, 3, 4], 
                label='Air | Lung | Soft Tissue | Bone')
    
    plt.tight_layout()
    
    return masks, plt.gcf()


def calculate_dose_statistics(ct_data, dose_grid=None):
    """Calculate dose-related statistics from CT data.
    
    Args:
        ct_data: 2D numpy array of CT data
        dose_grid: Optional dose distribution (if None, creates synthetic)
    """
    print("\n☢️  DOSE ANALYSIS")
    print("=" * 25)
    
    if dose_grid is None:
        # Create a synthetic dose distribution (simplified beam)
        print("Creating synthetic dose distribution...")
        
        center_x, center_y = ct_data.shape[1] // 2, ct_data.shape[0] // 2
        y, x = np.ogrid[:ct_data.shape[0], :ct_data.shape[1]]
        
        # Gaussian beam profile
        beam_width = 30
        dose_grid = 100 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * beam_width**2))
        
        # Apply tissue-dependent dose modification
        # Higher density tissues (bone) get slightly higher dose
        density_factor = (ct_data + 1000) / 1000  # Convert HU to relative density
        dose_grid = dose_grid * density_factor
        
        # Ensure dose is only delivered to patient (not air)
        patient_mask = ct_data > -500
        dose_grid[~patient_mask] = 0
    
    # Calculate dose-volume statistics
    tissue_masks = {
        'All Tissue': ct_data > -500,
        'Soft Tissue': (ct_data >= 0) & (ct_data < 150),
        'Bone': ct_data >= 150,
        'Lung': (ct_data >= -500) & (ct_data < 0)
    }
    
    print("Dose-Volume Statistics:")
    print("-" * 25)
    
    for tissue_name, mask in tissue_masks.items():
        if np.any(mask):
            tissue_dose = dose_grid[mask]
            mean_dose = np.mean(tissue_dose)
            max_dose = np.max(tissue_dose)
            volume_ml = np.sum(mask) * 0.1  # Assume 1mm³ voxels = 0.001 ml
            
            print(f"  {tissue_name:12}: Mean={mean_dose:5.1f}% Max={max_dose:5.1f}% "
                  f"Volume={volume_ml:6.1f}ml")
    
    # Visualize dose distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(ct_data, cmap='gray', alpha=0.7)
    plt.title('CT Image')
    plt.colorbar(label='HU')
    
    plt.subplot(1, 3, 2)
    plt.imshow(dose_grid, cmap='hot')
    plt.title('Dose Distribution')
    plt.colorbar(label='Dose (%)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ct_data, cmap='gray', alpha=0.5)
    plt.imshow(dose_grid, cmap='hot', alpha=0.7)
    plt.title('CT + Dose Overlay')
    plt.colorbar(label='Dose (%)')
    
    plt.tight_layout()
    
    return dose_grid, plt.gcf()


def main():
    """Main function demonstrating CT image analysis workflow."""
    print("🏥 CT Image Analysis for Medical Physics")
    print("=" * 50)
    print("Learn medical image processing with realistic examples!")
    
    # Step 1: Generate synthetic CT data
    ct_data = generate_synthetic_ct_slice()
    
    # Step 2: Display the image
    print("\n📱 Displaying CT image...")
    fig1 = display_ct_slice(ct_data, "Synthetic Chest CT Slice")
    plt.show()
    
    # Step 3: Analyze Hounsfield units
    fig2 = analyze_hounsfield_distribution(ct_data)
    plt.show()
    
    # Step 4: Threshold segmentation
    masks, fig3 = apply_threshold_segmentation(ct_data)
    plt.show()
    
    # Step 5: Dose analysis
    dose_grid, fig4 = calculate_dose_statistics(ct_data)
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("🎓 LEARNING SUMMARY")
    print("="*50)
    print("✓ Generated synthetic CT data with realistic anatomy")
    print("✓ Applied proper medical image display windowing")
    print("✓ Analyzed Hounsfield unit distributions")
    print("✓ Performed threshold-based tissue segmentation")
    print("✓ Calculated dose-volume statistics")
    print("✓ Created overlay visualizations")
    
    print("\n💡 Next Steps:")
    print("  • Try different window/level settings")
    print("  • Experiment with advanced segmentation algorithms")
    print("  • Load real DICOM images")
    print("  • Implement dose-volume histogram analysis")
    
    print("\n🔧 Required packages: numpy, matplotlib")
    print("   Install with: pip install -r requirements.txt")
    print("   Or individually: pip install numpy matplotlib")


if __name__ == "__main__":
    main() 