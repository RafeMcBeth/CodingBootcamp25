# CodingBootcamp25 - From Zero to Medical Physics Applications 🚀

**Go from "Hello World" to building real medical physics tools in just a few days!**

This intensive coding bootcamp is designed specifically for graduate medical physics students. You'll start with basic Python setup and rapidly progress to building functional applications that solve real problems in medical physics - from dose calculators to AI-powered image analysis tools.

## 🎯 What You'll Build

By the end of this bootcamp, you'll have created:
- **Radiation Dose Calculator**: A GUI application for clinical dose calculations
- **CT Image Analysis Tool**: Process and analyze medical images with thresholding
- **AI-Powered Inference**: Run machine learning models for medical image analysis  
- **Interactive Web Apps**: Browser-based physics simulations
- **Command-line Tools**: Automate common medical physics calculations

## 📈 Your Learning Journey - One Clear Path

All content is organized with a logical progression:

### 🌱 **Day 1: Python Foundations**
**`00_setup/`** - Set up Python and your development environment  
**`01_python_basics/`** - From "Hello World" to advanced programming concepts
```bash
# Start here - your first program!
python hello_world.py           # Basic syntax and variables
python basic_operations.py      # Functions, loops, conditionals  
python oop_radiation.py         # Object-oriented programming
python debug_example.py         # Professional debugging skills
python exercises.py             # Interactive practice exercises
```

### 🔧 **Day 2: Build Your First App**  
**`02_dose_calculator/`** - Create a working dose calculator with professional structure
```bash
python app.py                   # Launch your first medical physics app!
python dose_analysis.py sample_doses.csv  # Command-line data analysis
python -m pytest tests/        # Run professional unit tests
```

### 📊 **Day 3: Medical Image Processing**
**`03_image_basics/`** - Analyze CT scans and apply image processing techniques
```bash
python ct_image_analysis.py     # Complete CT analysis workflow
# Learn: Hounsfield units, thresholding, segmentation, dose overlays
```

### 🤖 **Day 4: AI Integration**
**`04_ai_demo/`** - Run machine learning inference on medical images
```bash
python ai_medical_inference.py  # Complete AI/ML demonstration
# Learn: Classification, regression, anomaly detection, real-time inference
```

### 🖥️ **Day 5: Professional GUIs**
**`05_instructor_example/`** - Create polished applications with graphical interfaces

## 🚀 Quick Start - Single Path Setup

Get up and running in under 5 minutes:

```bash
# 1. Clone the repository
git clone <repo_url>
cd CodingBootcamp25/medical-physics-bootcamp

# 2. Set up your Python virtual environment
python -m venv medphys-env

# 3. Activate the virtual environment
# Windows:
medphys-env\Scripts\activate
# macOS/Linux:
source medphys-env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start your journey!
cd 01_python_basics
python hello_world.py

# 6. Progress through each module
cd ../02_dose_calculator
python app.py
```

## 📚 Learning Modules - Clear Progression

| Day | Module | What You'll Learn | Time | Key Files |
|-----|--------|-------------------|------|-----------|
| 1 | **Python Basics** | Variables, functions, OOP, debugging | 4-5 hours | `hello_world.py`, `basic_operations.py`, `oop_radiation.py`, `exercises.py` |
| 2 | **Dose Calculator** | App structure, testing, CLI, data analysis | 4-5 hours | `app.py`, `dose_calculator.py`, `dose_analysis.py` |
| 3 | **Image Processing** | CT analysis, Hounsfield units, segmentation | 4-5 hours | `ct_image_analysis.py` |
| 4 | **AI Integration** | ML models, classification, prediction | 3-4 hours | `ai_medical_inference.py` |
| 5 | **GUI Development** | Professional interfaces | 3-4 hours | `gui_dose_calculator.py` |

## 🎓 Why This Works for Medical Physics Students

- **Single clear path**: No confusion about where to start or what's next
- **Pure Python**: No Jupyter notebook dependencies - just run Python files directly
- **Virtual environments**: Industry-standard Python setup (no conda required)
- **Real-world relevance**: Every example uses actual medical physics concepts
- **Rapid progression**: See immediate results that matter to your field
- **Practical focus**: Build tools you can actually use in your research/clinic
- **Industry-standard practices**: Learn testing, documentation, and code organization

## 🛠️ What's Included - Everything in One Place

### Complete Learning Path
- **00_setup/**: Environment setup instructions
- **01_python_basics/**: Core Python concepts with medical physics examples
  - `hello_world.py` - Your first program with radiation safety calculations
  - `basic_operations.py` - Functions, loops, conditionals with dose data
  - `oop_radiation.py` - Object-oriented programming with radioactive sources
  - `debug_example.py` - Professional debugging with dose rate calculations
  - `exercises.py` - Interactive practice problems
  - `half_life_decay.py` - Advanced dataclasses and command-line arguments
- **02_dose_calculator/**: Complete application with tests and data analysis
  - `app.py` - Main dose calculator application
  - `dose_calculator.py` - Core calculation logic
  - `dose_analysis.py` - CSV data processing with command-line interface
  - `sample_doses.csv` - Test data for analysis
- **03_image_basics/**: Medical image processing
  - `ct_image_analysis.py` - Complete CT workflow: windowing, segmentation, dose overlay
- **04_ai_demo/**: AI/ML integration examples
  - `ai_medical_inference.py` - Organ classification, dose prediction, QA anomaly detection
- **05_instructor_example/**: Professional GUI applications
- **extras/**: Bonus applications and advanced topics
  - `contour_metrics_app.py` - Interactive medical image analysis tool

## 💡 Project Ideas

Once you've completed the bootcamp, try these challenges:
- **Treatment Planning Tool**: Calculate isodose curves
- **QA Dashboard**: Automate daily QA checks
- **Research Pipeline**: Process large datasets automatically
- **Web Portal**: Share calculations with your team

## 🎯 Success Metrics

After this bootcamp, you'll be able to:
- ✅ Set up a Python development environment
- ✅ Write clean, documented, tested code
- ✅ Build GUI applications for medical physics
- ✅ Process and analyze medical images
- ✅ Integrate AI/ML into your workflow
- ✅ Automate repetitive calculations
- ✅ Share your tools with colleagues

## 🔧 Setup Instructions

### Prerequisites
Python 3.8 or newer installed on your system.
- Download from [python.org](https://www.python.org/downloads/) if needed
- Verify installation: `python --version` or `python3 --version`

### Step-by-Step Setup

1. **Create a virtual environment:**
   ```bash
   # On Windows:
   python -m venv medphys-env
   
   # On macOS/Linux:
   python3 -m venv medphys-env
   ```

2. **Activate the virtual environment:**
   ```bash
   # On Windows:
   medphys-env\Scripts\activate
   
   # On macOS/Linux:
   source medphys-env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib, pandas; print('✅ All packages installed successfully!')"
   ```

### Daily Workflow

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

---

**Ready to transform your coding skills?** Start with `00_setup/setup_instructions.md` and follow the single clear path from beginner to building professional medical physics applications!
