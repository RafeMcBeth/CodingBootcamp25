# Medical Physics Bootcamp

This repo contains starter materials for a short course in Python and AI for medical physicists. Follow the setup guide below to get started.

## Setup

Create the Conda environment and install dependencies:

```bash
conda env create -f environment.yml
```

Activate the environment each time:

```bash
conda activate medphys-bootcamp
```

## Instructor Example

The `05_instructor_example` folder contains a fully working GUI dose calculator implemented with Tkinter. Run it with:

```bash
python medical-physics-bootcamp/05_instructor_example/gui_dose_calculator.py
```

A skeleton script `gui_template.py` is provided in the same folder for students to extend.

## Project Ideas

See `PROJECT_IDEAS.md` for ten small concepts that can be expanded into final projects.

## Streamlit UI

Run the visual dose calculator with:

```bash
conda activate medphys-bootcamp
streamlit run 02_dose_calculator/app.py
```

An optional PyScript mini-game can be found in `extras/pyscript_beam_balance/index.html`.
