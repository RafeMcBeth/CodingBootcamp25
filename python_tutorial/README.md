# Python Basics for Medical Physics

This directory contains simple Python scripts demonstrating fundamental programming concepts. They are designed for graduate medical physics students who are new to Python and GitHub.

## Contents

- `basic_operations.py` – Variables, loops, functions, and how to run a script.
- `oop_radiation.py` – Example of object-oriented programming with simple radiation calculations.
- `debug_example.py` – Shows how to use the built-in debugger (`pdb`).
- `dose_analysis.py` – Compute statistics for dose measurements stored in a CSV file.
- `half_life_decay.py` – Command-line tool demonstrating dataclasses for decay calculations.

## Running Examples

Use the provided scripts with Python 3.

```bash
python3 basic_operations.py
python3 oop_radiation.py
python3 dose_analysis.py sample_doses.csv
python3 half_life_decay.py Co60 1925 3.7e10 24
python3 debug_example.py
```

`sample_doses.csv` contains example measurements for use with `dose_analysis.py`.

`debug_example.py` intentionally triggers a division by zero so you can experiment with stepping through code using `pdb`.

## Further Reading

- [Python documentation](https://docs.python.org/3/)
- [GitHub getting started guide](https://docs.github.com/en/get-started/using-git)
- Introductory use cases of Python in medical physics include data analysis, dose calculations, and automation of imaging workflows.
