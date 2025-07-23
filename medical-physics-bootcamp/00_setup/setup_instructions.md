# Setup Instructions

## Quick Start with Python Virtual Environments

**Prerequisites:** Python 3.8 or newer installed on your system.
- Download from [python.org](https://www.python.org/downloads/) if needed
- Verify installation: `python --version` or `python3 --version`

### Step-by-Step Setup

1. **Clone this repository:**
   ```bash
   git clone <repo_url>
   cd CodingBootcamp25/medical-physics-bootcamp
   ```

2. **Create a virtual environment:**
   ```bash
   # On Windows:
   python -m venv medphys-env
   
   # On macOS/Linux:
   python3 -m venv medphys-env
   ```

3. **Activate the virtual environment:**
   ```bash
   # On Windows:
   medphys-env\Scripts\activate
   
   # On macOS/Linux:
   source medphys-env/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib, pandas; print('✅ All packages installed successfully!')"
   ```

6. **Start coding:**
   ```bash
   cd 01_python_basics
   python hello_world.py
   ```

## Troubleshooting

### Common Issues:

**"python command not found"**
- Try `python3` instead of `python`
- Make sure Python is installed and in your PATH

**Permission errors on Windows**
- Run Command Prompt as Administrator
- Or use PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Package installation fails**
- Upgrade pip: `pip install --upgrade pip`
- If specific packages fail, install individually: `pip install numpy matplotlib`

### Daily Workflow:

1. **Activate environment** (every new terminal session):
   ```bash
   # Windows: medphys-env\Scripts\activate
   # macOS/Linux: source medphys-env/bin/activate
   ```

2. **Run Python scripts:**
   ```bash
   python script_name.py
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

## Why Virtual Environments?

- **Isolation**: Keep project dependencies separate
- **Reproducibility**: Same environment on any machine
- **Industry Standard**: Used in professional Python development
- **No Conflicts**: Different projects can use different package versions

🎉 **You're ready to start the bootcamp!**
