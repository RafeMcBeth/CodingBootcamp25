"""Interactive app for exploring contour comparison metrics.

This advanced application displays two shapes (circles or ellipses) that can be adjusted 
using sliders for size, position, and orientation. It calculates multiple metrics including 
IoU, Dice coefficient, Hausdorff distance, and centroid distance to illustrate how changes 
in geometry affect these important medical physics metrics.

This demonstrates advanced Python concepts:
- Interactive matplotlib widgets
- Real-time data visualization 
- Medical image analysis metrics
- Object-oriented GUI design

Requires ``matplotlib``, ``numpy``, and ``scipy``. Run with:
    python contour_metrics_app.py

This is an advanced example showing how Python can create sophisticated 
interactive applications for medical physics research and education.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Circle
from scipy.spatial.distance import directed_hausdorff
from collections import deque
import matplotlib.patches as patches


GRID_SIZE = 200
HISTORY_LENGTH = 100


def circle_mask(cx: float, cy: float, radius: float) -> np.ndarray:
    """Return a boolean mask with a circle of ``radius`` centered at (cx, cy)."""
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def ellipse_mask(cx: float, cy: float, a: float, b: float, angle: float) -> np.ndarray:
    """Return a boolean mask with an ellipse centered at (cx, cy), axes a, b, rotated by angle (degrees)."""
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    angle_rad = np.deg2rad(angle)
    x_shift = x - cx
    y_shift = y - cy
    x_rot = x_shift * np.cos(angle_rad) + y_shift * np.sin(angle_rad)
    y_rot = -x_shift * np.sin(angle_rad) + y_shift * np.cos(angle_rad)
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1


def get_contour_points(mask: np.ndarray) -> np.ndarray:
    """Extract contour points from a boolean mask."""
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask)
    contour = mask & ~eroded
    return np.column_stack(np.where(contour))


def intersection_over_union(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU (Jaccard index) between two masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    This is a fundamental metric in medical image segmentation
    used to evaluate organ contour accuracy.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Dice similarity coefficient.
    
    Dice = 2|A ∩ B| / (|A| + |B|)
    
    Also known as F1 score or Sørensen-Dice coefficient.
    Commonly used in medical image analysis for contour comparison.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    return 2 * intersection / total if total > 0 else 0.0


def hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Hausdorff distance between contours.
    
    Measures the maximum distance between nearest points of two contours.
    Important for assessing boundary accuracy in medical imaging.
    """
    contour1 = get_contour_points(mask1)
    contour2 = get_contour_points(mask2)
    
    if len(contour1) == 0 or len(contour2) == 0:
        return float('inf')
    
    # Hausdorff distance is the maximum of directed distances
    dist1 = directed_hausdorff(contour1, contour2)[0]
    dist2 = directed_hausdorff(contour2, contour1)[0]
    return max(dist1, dist2)


def centroid_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate distance between centroids of two masks.
    
    Useful for measuring organ displacement in medical imaging.
    """
    def centroid(mask):
        if mask.sum() == 0:
            return np.array([0, 0])
        coords = np.where(mask)
        return np.array([coords[0].mean(), coords[1].mean()])
    
    c1 = centroid(mask1)
    c2 = centroid(mask2)
    return np.linalg.norm(c1 - c2)


class ContourMetricsApp:
    """Interactive application for exploring contour comparison metrics."""
    
    def __init__(self):
        """Initialize the interactive application."""
        print("🔬 Contour Metrics Explorer for Medical Physics")
        print("=" * 50)
        print("This interactive tool demonstrates key metrics used in")
        print("medical image analysis and radiation therapy planning:")
        print("• IoU (Intersection over Union)")
        print("• Dice Coefficient") 
        print("• Hausdorff Distance")
        print("• Centroid Distance")
        print("\nAdjust the sliders to see how shape changes affect metrics!")
        
        # Initialize data storage
        self.history = {
            'iou': deque(maxlen=HISTORY_LENGTH),
            'dice': deque(maxlen=HISTORY_LENGTH),
            'hausdorff': deque(maxlen=HISTORY_LENGTH),
            'centroid': deque(maxlen=HISTORY_LENGTH)
        }
        
        # Setup the matplotlib figure
        self.setup_figure()
        self.setup_sliders()
        self.setup_radio_buttons()
        
        # Initial calculation
        self.update_display()
        
    def setup_figure(self):
        """Setup the matplotlib figure layout."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Contour Metrics Explorer - Medical Physics Tool', fontsize=16)
        
        # Main visualization area
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        self.ax_main.set_title('Contour Comparison')
        self.ax_main.set_xlim(0, GRID_SIZE)
        self.ax_main.set_ylim(0, GRID_SIZE)
        self.ax_main.set_aspect('equal')
        
        # Metrics display
        self.ax_metrics = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        self.ax_metrics.set_title('Current Metrics')
        self.ax_metrics.axis('off')
        
        # History plots
        self.ax_history = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        self.ax_history.set_title('Metrics History')
        self.ax_history.set_xlabel('Update #')
        self.ax_history.set_ylabel('Metric Value')
        
        # Slider areas
        self.slider_area = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        self.slider_area.axis('off')
        
    def setup_sliders(self):
        """Setup interactive sliders for shape parameters."""
        # Define slider positions (left, bottom, width, height)
        slider_height = 0.03
        slider_width = 0.15
        
        # Shape 1 sliders (Reference - usually ground truth)
        ax_r1 = plt.axes([0.1, 0.15, slider_width, slider_height])
        ax_x1 = plt.axes([0.1, 0.11, slider_width, slider_height])
        ax_y1 = plt.axes([0.1, 0.07, slider_width, slider_height])
        
        self.slider_r1 = Slider(ax_r1, 'Ref Radius', 10, 50, valinit=30)
        self.slider_x1 = Slider(ax_x1, 'Ref X', 20, GRID_SIZE-20, valinit=70)
        self.slider_y1 = Slider(ax_y1, 'Ref Y', 20, GRID_SIZE-20, valinit=100)
        
        # Shape 2 sliders (Test - usually segmentation result)
        ax_r2 = plt.axes([0.3, 0.15, slider_width, slider_height])
        ax_x2 = plt.axes([0.3, 0.11, slider_width, slider_height])
        ax_y2 = plt.axes([0.3, 0.07, slider_width, slider_height])
        
        self.slider_r2 = Slider(ax_r2, 'Test Radius', 10, 50, valinit=25)
        self.slider_x2 = Slider(ax_x2, 'Test X', 20, GRID_SIZE-20, valinit=90)
        self.slider_y2 = Slider(ax_y2, 'Test Y', 20, GRID_SIZE-20, valinit=110)
        
        # Additional sliders for ellipse mode
        ax_a2 = plt.axes([0.5, 0.15, slider_width, slider_height])
        ax_b2 = plt.axes([0.5, 0.11, slider_width, slider_height])
        ax_angle2 = plt.axes([0.5, 0.07, slider_width, slider_height])
        
        self.slider_a2 = Slider(ax_a2, 'Test A-axis', 10, 50, valinit=25)
        self.slider_b2 = Slider(ax_b2, 'Test B-axis', 10, 50, valinit=20)
        self.slider_angle2 = Slider(ax_angle2, 'Test Angle', 0, 180, valinit=0)
        
        # Connect sliders to update function
        for slider in [self.slider_r1, self.slider_x1, self.slider_y1,
                       self.slider_r2, self.slider_x2, self.slider_y2,
                       self.slider_a2, self.slider_b2, self.slider_angle2]:
            slider.on_changed(self.update_display)
    
    def setup_radio_buttons(self):
        """Setup radio buttons for shape selection."""
        ax_radio = plt.axes([0.7, 0.05, 0.1, 0.15])
        self.radio = RadioButtons(ax_radio, ('Circle', 'Ellipse'))
        self.radio.on_clicked(self.update_display)
    
    def update_display(self, val=None):
        """Update the display with current parameters."""
        # Get current shape type
        shape_type = self.radio.value_selected
        
        # Generate masks based on current parameters
        if shape_type == 'Circle':
            mask1 = circle_mask(self.slider_x1.val, self.slider_y1.val, self.slider_r1.val)
            mask2 = circle_mask(self.slider_x2.val, self.slider_y2.val, self.slider_r2.val)
        else:  # Ellipse
            mask1 = circle_mask(self.slider_x1.val, self.slider_y1.val, self.slider_r1.val)
            mask2 = ellipse_mask(self.slider_x2.val, self.slider_y2.val, 
                                self.slider_a2.val, self.slider_b2.val, self.slider_angle2.val)
        
        # Calculate metrics
        iou = intersection_over_union(mask1, mask2)
        dice = dice_coefficient(mask1, mask2)
        hausdorff = hausdorff_distance(mask1, mask2)
        centroid_dist = centroid_distance(mask1, mask2)
        
        # Store in history
        self.history['iou'].append(iou)
        self.history['dice'].append(dice)
        self.history['hausdorff'].append(hausdorff)
        self.history['centroid'].append(centroid_dist)
        
        # Update main visualization
        self.ax_main.clear()
        self.ax_main.set_title('Contour Comparison (Red=Reference, Blue=Test, Purple=Overlap)')
        
        # Create colored overlay
        overlay = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        overlay[mask1, 0] = 0.7  # Red for reference
        overlay[mask2, 2] = 0.7  # Blue for test
        overlay[mask1 & mask2, :] = [0.7, 0, 0.7]  # Purple for overlap
        
        self.ax_main.imshow(overlay, origin='lower', alpha=0.8)
        self.ax_main.set_xlim(0, GRID_SIZE)
        self.ax_main.set_ylim(0, GRID_SIZE)
        
        # Update metrics display
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        # Format metrics with clinical context
        metrics_text = f"""
        📊 QUANTITATIVE METRICS:
        
        IoU (Jaccard): {iou:.3f}
        {'✓ Excellent' if iou > 0.8 else '⚠️ Good' if iou > 0.6 else '❌ Poor'} overlap
        
        Dice Coefficient: {dice:.3f}
        {'✓ Excellent' if dice > 0.9 else '⚠️ Good' if dice > 0.7 else '❌ Poor'} similarity
        
        Hausdorff Distance: {hausdorff:.1f} pixels
        {'✓ Excellent' if hausdorff < 5 else '⚠️ Acceptable' if hausdorff < 10 else '❌ Poor'} boundary match
        
        Centroid Distance: {centroid_dist:.1f} pixels
        {'✓ Well aligned' if centroid_dist < 10 else '⚠️ Slight shift' if centroid_dist < 20 else '❌ Major displacement'}
        
        🏥 CLINICAL RELEVANCE:
        • IoU > 0.8: Clinically acceptable
        • Dice > 0.9: High quality segmentation
        • Hausdorff < 5mm: Precise boundaries
        """
        
        self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Update history plot
        self.ax_history.clear()
        if len(self.history['iou']) > 1:
            x = range(len(self.history['iou']))
            self.ax_history.plot(x, self.history['iou'], 'r-', label='IoU', linewidth=2)
            self.ax_history.plot(x, self.history['dice'], 'g-', label='Dice', linewidth=2)
            
            # Scale Hausdorff and centroid for display
            hausdorff_scaled = [h/50 for h in self.history['hausdorff']]  # Scale to 0-1 range
            centroid_scaled = [c/50 for c in self.history['centroid']]
            
            self.ax_history.plot(x, hausdorff_scaled, 'b-', label='Hausdorff/50', linewidth=2)
            self.ax_history.plot(x, centroid_scaled, 'm-', label='Centroid/50', linewidth=2)
            
            self.ax_history.set_ylim(0, 1)
            self.ax_history.legend()
            self.ax_history.grid(True, alpha=0.3)
        
        self.ax_history.set_title('Metrics Trend (Lower Hausdorff/Centroid = Better)')
        self.ax_history.set_xlabel('Adjustment #')
        self.ax_history.set_ylabel('Normalized Metric Value')
        
        # Refresh display
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive application."""
        plt.show()


def main():
    """Main function to run the contour metrics explorer."""
    print("🚀 Starting Contour Metrics Explorer...")
    print("\nThis tool helps understand metrics used in:")
    print("• Organ segmentation evaluation")
    print("• Treatment planning quality assurance") 
    print("• Automated contouring validation")
    print("• Research in medical image analysis")
    
    try:
        app = ContourMetricsApp()
        app.show()
    except ImportError as e:
        print(f"\n❌ Missing required package: {e}")
        print("Install with: pip install -r requirements.txt")
        print("Or individually: pip install matplotlib numpy scipy")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Make sure you have a display available for matplotlib.")


if __name__ == "__main__":
    main() 