"""Interactive app for exploring contour comparison metrics.

This script displays two circles that can be adjusted using sliders for radius and position.
It calculates multiple metrics including IoU, Dice coefficient, Hausdorff distance, and centroid distance
to illustrate how changes in size and overlap affect these metrics.

Requires ``matplotlib`` and ``numpy``. Run with:

    python3 contour_metrics_app.py

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


def compute_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """Return comprehensive metrics for the two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Basic metrics
    iou = intersection / union if union else 0.0
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) else 0.0
    
    # Centroid distance
    y1, x1 = np.where(mask1)
    y2, x2 = np.where(mask2)
    
    if len(x1) > 0 and len(x2) > 0:
        centroid1 = np.array([x1.mean(), y1.mean()])
        centroid2 = np.array([x2.mean(), y2.mean()])
        centroid_dist = np.linalg.norm(centroid1 - centroid2)
    else:
        centroid_dist = 0.0
    
    # Hausdorff distance
    try:
        contour1 = get_contour_points(mask1)
        contour2 = get_contour_points(mask2)
        
        if len(contour1) > 0 and len(contour2) > 0:
            hausdorff_dist = max(
                directed_hausdorff(contour1, contour2)[0],
                directed_hausdorff(contour2, contour1)[0]
            )
        else:
            hausdorff_dist = 0.0
    except:
        hausdorff_dist = 0.0
    
    # Area ratio
    area1, area2 = mask1.sum(), mask2.sum()
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0
    
    return {
        'iou': iou,
        'dice': dice,
        'centroid_dist': centroid_dist,
        'hausdorff_dist': hausdorff_dist,
        'area_ratio': area_ratio,
        'intersection': intersection,
        'union': union,
        'area1': area1,
        'area2': area2
    }


def setup_dark_mode():
    """Configure matplotlib for dark mode."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    plt.rcParams['axes.facecolor'] = '#2d2d2d'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#ffffff'
    plt.rcParams['xtick.color'] = '#ffffff'
    plt.rcParams['ytick.color'] = '#ffffff'


def main() -> None:
    setup_dark_mode()
    
    # Initial parameters
    r1 = 40
    r2 = 40
    offset = 20
    cx1 = GRID_SIZE // 2
    cy1 = GRID_SIZE // 2
    ellipse_a1, ellipse_b1, ellipse_angle1 = 40, 30, 0
    ellipse_a2, ellipse_b2, ellipse_angle2 = 40, 30, 0
    
    # Shape selection
    shape_types = ['Circle', 'Ellipse']
    shape1 = 'Circle'
    shape2 = 'Circle'

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Main visualization
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    ax_main.set_facecolor('#2d2d2d')
    
    # Metrics history plots (2x2 grid)
    ax_hist_iou = plt.subplot2grid((3, 4), (0, 2))
    ax_hist_dice = plt.subplot2grid((3, 4), (0, 3))
    ax_hist_centroid = plt.subplot2grid((3, 4), (1, 2))
    ax_hist_hausdorff = plt.subplot2grid((3, 4), (1, 3))
    for ax, title in zip([ax_hist_iou, ax_hist_dice, ax_hist_centroid, ax_hist_hausdorff],
                         ["IoU", "Dice", "Centroid Dist", "Hausdorff Dist"]):
        ax.set_facecolor('#2d2d2d')
        ax.set_title(title, color='white', fontsize=11)
        ax.grid(True, alpha=0.3, color='#666666')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    # Remove old ax_history
    # Metrics display
    ax_metrics = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax_metrics.set_facecolor('#2d2d2d')
    ax_metrics.axis('off')
    
    # Initialize history tracking
    history = {
        'iou': deque(maxlen=HISTORY_LENGTH),
        'dice': deque(maxlen=HISTORY_LENGTH),
        'centroid_dist': deque(maxlen=HISTORY_LENGTH),
        'hausdorff_dist': deque(maxlen=HISTORY_LENGTH)
    }
    
    # Initialize history lines for each subplot
    history_lines = {
        'iou': ax_hist_iou.plot([], [], color='#ff6b6b', linewidth=2)[0],
        'dice': ax_hist_dice.plot([], [], color='#4ecdc4', linewidth=2)[0],
        'centroid_dist': ax_hist_centroid.plot([], [], color='#45b7d1', linewidth=2)[0],
        'hausdorff_dist': ax_hist_hausdorff.plot([], [], color='#96ceb4', linewidth=2)[0],
    }
    
    # Initial masks and visualization
    mask1 = circle_mask(cx1, cy1, r1)
    mask2 = circle_mask(cx1 + offset, cy1, r2)
    
    # Create enhanced visualization
    overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # RGBA
    overlay[mask1] = [1, 0.3, 0.3, 0.7]  # Red for circle 1
    overlay[mask2] = [0.3, 0.3, 1, 0.7]  # Blue for circle 2
    overlay[np.logical_and(mask1, mask2)] = [1, 0.3, 1, 0.9]  # Purple for intersection
    
    im = ax_main.imshow(overlay, extent=[0, GRID_SIZE, GRID_SIZE, 0])
    ax_main.set_xlim(0, GRID_SIZE)
    ax_main.set_ylim(0, GRID_SIZE)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3, color='#666666')
    ax_main.set_title('Interactive Contour Comparison', color='white', fontsize=14, pad=20)
    
    # Add circle outlines
    circle1 = Circle((cx1, cy1), r1, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
    circle2 = Circle((cx1 + offset, cy1), r2, fill=False, edgecolor='blue', linewidth=2, alpha=0.8)
    ax_main.add_patch(circle1)
    ax_main.add_patch(circle2)
    
    # Initialize metrics display
    metrics_text = ax_metrics.text(0.02, 0.5, '', transform=ax_metrics.transAxes, 
                                  fontsize=10, color='white', fontfamily='monospace',
                                  verticalalignment='center')
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.3, wspace=0.3)
    
    # Slider axes with dark theme
    axcolor = '#404040'
    ax_r1 = plt.axes([0.25, 0.08, 0.65, 0.02], facecolor=axcolor)
    ax_r2 = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor)
    ax_offset = plt.axes([0.25, 0.02, 0.65, 0.02], facecolor=axcolor)

    s_r1 = Slider(ax_r1, "Radius 1", 5, 80, valinit=r1, color='#ff6b6b')
    s_r2 = Slider(ax_r2, "Radius 2", 5, 80, valinit=r2, color='#4ecdc4')
    s_offset = Slider(ax_offset, "Offset", -80, 80, valinit=offset, color='#45b7d1')

    # Add radio buttons for shape selection
    ax_shape1 = plt.axes([0.05, 0.02, 0.08, 0.10], facecolor='#404040')
    ax_shape2 = plt.axes([0.15, 0.02, 0.08, 0.10], facecolor='#404040')
    radio_shape1 = RadioButtons(ax_shape1, shape_types, active=0)
    radio_shape2 = RadioButtons(ax_shape2, shape_types, active=0)
    # for circle in radio_shape1.circles + radio_shape2.circles:
    #     circle.set_edgecolor('white')
    #     circle.set_linewidth(1.5)
    for label in radio_shape1.labels + radio_shape2.labels:
        label.set_color('white')

    # Add ellipse sliders (hidden by default)
    ax_a1 = plt.axes([0.25, 0.11, 0.65, 0.02], facecolor='#404040')
    ax_b1 = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor='#404040')
    ax_angle1 = plt.axes([0.25, 0.07, 0.65, 0.02], facecolor='#404040')
    s_a1 = Slider(ax_a1, "Ellipse a1", 5, 80, valinit=ellipse_a1, color='#ff6b6b')
    s_b1 = Slider(ax_b1, "Ellipse b1", 5, 80, valinit=ellipse_b1, color='#ff6b6b')
    s_angle1 = Slider(ax_angle1, "Angle 1", -90, 90, valinit=ellipse_angle1, color='#ff6b6b')
    ax_a1.set_visible(False)
    ax_b1.set_visible(False)
    ax_angle1.set_visible(False)

    ax_a2 = plt.axes([0.25, 0.06, 0.65, 0.02], facecolor='#404040')
    ax_b2 = plt.axes([0.25, 0.04, 0.65, 0.02], facecolor='#404040')
    ax_angle2 = plt.axes([0.25, 0.02, 0.65, 0.02], facecolor='#404040')
    s_a2 = Slider(ax_a2, "Ellipse a2", 5, 80, valinit=ellipse_a2, color='#4ecdc4')
    s_b2 = Slider(ax_b2, "Ellipse b2", 5, 80, valinit=ellipse_b2, color='#4ecdc4')
    s_angle2 = Slider(ax_angle2, "Angle 2", -90, 90, valinit=ellipse_angle2, color='#4ecdc4')
    ax_a2.set_visible(False)
    ax_b2.set_visible(False)
    ax_angle2.set_visible(False)

    def get_mask1():
        if shape1 == 'Circle':
            return circle_mask(cx1, cy1, s_r1.val)
        else:
            return ellipse_mask(cx1, cy1, s_a1.val, s_b1.val, s_angle1.val)
    def get_mask2():
        if shape2 == 'Circle':
            return circle_mask(cx1 + s_offset.val, cy1, s_r2.val)
        else:
            return ellipse_mask(cx1 + s_offset.val, cy1, s_a2.val, s_b2.val, s_angle2.val)

    def update(val):
        nonlocal shape1, shape2
        # Update shape selection
        shape1 = radio_shape1.value_selected
        shape2 = radio_shape2.value_selected
        # Show/hide sliders
        if shape1 == 'Circle':
            ax_r1.set_visible(True)
            ax_a1.set_visible(False)
            ax_b1.set_visible(False)
            ax_angle1.set_visible(False)
        else:
            ax_r1.set_visible(False)
            ax_a1.set_visible(True)
            ax_b1.set_visible(True)
            ax_angle1.set_visible(True)
        if shape2 == 'Circle':
            ax_r2.set_visible(True)
            ax_a2.set_visible(False)
            ax_b2.set_visible(False)
            ax_angle2.set_visible(False)
        else:
            ax_r2.set_visible(False)
            ax_a2.set_visible(True)
            ax_b2.set_visible(True)
            ax_angle2.set_visible(True)
        # Update masks
        m1 = get_mask1()
        m2 = get_mask2()
        # Update visualization
        overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        overlay[m1] = [1, 0.3, 0.3, 0.7]
        overlay[m2] = [0.3, 0.3, 1, 0.7]
        overlay[np.logical_and(m1, m2)] = [1, 0.3, 1, 0.9]
        im.set_data(overlay)
        # Update outlines
        if shape1 == 'Circle':
            circle1.set_visible(True)
            circle1.radius = s_r1.val
            circle1.center = (cx1, cy1)
        else:
            circle1.set_visible(False)
        if shape2 == 'Circle':
            circle2.set_visible(True)
            circle2.radius = s_r2.val
            circle2.center = (cx1 + s_offset.val, cy1)
        else:
            circle2.set_visible(False)
        # Compute metrics
        metrics = compute_metrics(m1, m2)
        # Update metrics display
        metrics_str = f"""IoU: {metrics['iou']:.3f} | Dice: {metrics['dice']:.3f} | Centroid Dist: {metrics['centroid_dist']:.1f} | Hausdorff: {metrics['hausdorff_dist']:.1f} | Area Ratio: {metrics['area_ratio']:.3f}"""
        metrics_text.set_text(metrics_str)
        # Update history
        for metric in history:
            history[metric].append(metrics[metric])
        # Update history plots (each metric in its own subplot)
        for metric, line in history_lines.items():
            if len(history[metric]) > 1:
                x_data = list(range(len(history[metric])))
                line.set_data(x_data, list(history[metric]))
                ax = line.axes
                ax.set_xlim(0, len(history[metric]) - 1)
                values = list(history[metric])
                if values:
                    minv, maxv = min(values), max(values)
                    if minv == maxv:
                        minv, maxv = minv - 0.1, maxv + 0.1
                    ax.set_ylim(minv - 0.1 * abs(minv), maxv + 0.1 * abs(maxv))
        fig.canvas.draw_idle()

    # Connect update to all widgets
    s_r1.on_changed(update)
    s_r2.on_changed(update)
    s_offset.on_changed(update)
    s_a1.on_changed(update)
    s_b1.on_changed(update)
    s_angle1.on_changed(update)
    s_a2.on_changed(update)
    s_b2.on_changed(update)
    s_angle2.on_changed(update)
    radio_shape1.on_clicked(update)
    radio_shape2.on_clicked(update)
    
    # Initial update
    update(None)
    
    # Add some cool features
    ax_main.text(0.02, 0.98, 'Red Circle', transform=ax_main.transAxes, 
                color='red', fontsize=10, verticalalignment='top')
    ax_main.text(0.02, 0.95, 'Blue Circle', transform=ax_main.transAxes, 
                color='blue', fontsize=10, verticalalignment='top')
    ax_main.text(0.02, 0.92, 'Purple = Overlap', transform=ax_main.transAxes, 
                color='magenta', fontsize=10, verticalalignment='top')
    
    plt.show()


if __name__ == "__main__":
    main()
