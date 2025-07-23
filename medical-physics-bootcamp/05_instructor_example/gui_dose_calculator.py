"""Example GUI dose calculator using Tkinter."""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from ..02_dose_calculator.dose_calculator import calculate_dose, safety_check


def compute_and_display(result_label: ttk.Label, rate_entry: ttk.Entry, time_entry: ttk.Entry) -> None:
    """Compute dose and update the result label."""
    try:
        rate = float(rate_entry.get())
        time = float(time_entry.get())
    except ValueError:
        result_label.config(text="Please enter valid numbers")
        return

    dose = calculate_dose(rate, time)
    msg, _ = safety_check(dose)
    result_label.config(text=f"Dose: {dose:.2f} Gy\n{msg}")


def main() -> None:
    root = tk.Tk()
    root.title("Dose Calculator")

    ttk.Label(root, text="Dose rate (Gy/min)").grid(row=0, column=0, padx=5, pady=5)
    rate_entry = ttk.Entry(root)
    rate_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Time (min)").grid(row=1, column=0, padx=5, pady=5)
    time_entry = ttk.Entry(root)
    time_entry.grid(row=1, column=1, padx=5, pady=5)

    result_label = ttk.Label(root, text="")
    result_label.grid(row=3, column=0, columnspan=2, pady=10)

    calc_button = ttk.Button(
        root,
        text="Calculate",
        command=lambda: compute_and_display(result_label, rate_entry, time_entry),
    )
    calc_button.grid(row=2, column=0, columnspan=2, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
