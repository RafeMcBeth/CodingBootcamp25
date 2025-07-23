"""Dose calculation utilities."""

from typing import Union


def calculate_dose(dose_rate: Union[int, float], time: Union[int, float]) -> float:
    """Calculate total radiation dose.

    Parameters
    ----------
    dose_rate : float
        Dose rate in Gy/min.
    time : float
        Exposure time in minutes.

    Returns
    -------
    float
        Total dose in Gray.
    """
    return dose_rate * time


def safety_check(total_dose: float, threshold: float = 2.0) -> tuple[str, str]:
    """Return a message and status indicating if dose exceeds threshold."""
    if total_dose > threshold:
        return "WARNING: dose exceeds safe limit", "warning"
    return "OK", "ok"
