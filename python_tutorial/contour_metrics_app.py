"""Interactive app for exploring contour comparison metrics.

This script displays two circles that can be adjusted using sliders for radius and position.
It calculates Intersection over Union (IoU) and Dice coefficient to illustrate how
changes in size and overlap affect these metrics.

Requires ``matplotlib`` and ``numpy``. Run with:

    python3 contour_metrics_app.py

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


GRID_SIZE = 200


def circle_mask(cx: float, cy: float, radius: float) -> np.ndarray:
    """Return a boolean mask with a circle of ``radius`` centered at (cx, cy)."""
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def compute_metrics(mask1: np.ndarray, mask2: np.ndarray) -> tuple[float, float]:
    """Return (IoU, Dice coefficient) for the two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union else 0.0
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) else 0.0
    return iou, dice


def main() -> None:
    # Initial parameters
    r1 = 40
    r2 = 40
    offset = 20

    cx1 = GRID_SIZE // 2
    cy1 = GRID_SIZE // 2

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    mask1 = circle_mask(cx1, cy1, r1)
    mask2 = circle_mask(cx1 + offset, cy1, r2)
    overlay = mask1.astype(int) + mask2.astype(int)
    im = ax.imshow(overlay, cmap="Reds", vmin=0, vmax=2)
    iou, dice = compute_metrics(mask1, mask2)
    title = ax.set_title(f"IoU: {iou:.2f}  Dice: {dice:.2f}")
    ax.axis("off")

    # Slider axes
    axcolor = "lightgoldenrodyellow"
    ax_r2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_offset = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    s_r2 = Slider(ax_r2, "Radius 2", 5, 80, valinit=r2)
    s_offset = Slider(ax_offset, "Offset", -80, 80, valinit=offset)

    def update(val):
        new_r2 = s_r2.val
        new_offset = s_offset.val
        m1 = circle_mask(cx1, cy1, r1)
        m2 = circle_mask(cx1 + new_offset, cy1, new_r2)
        im.set_data(m1.astype(int) + m2.astype(int))
        iou_v, dice_v = compute_metrics(m1, m2)
        title.set_text(f"IoU: {iou_v:.2f}  Dice: {dice_v:.2f}")
        fig.canvas.draw_idle()

    s_r2.on_changed(update)
    s_offset.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
