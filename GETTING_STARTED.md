# Getting Started Guide

This guide will help you set up and run the On-Demand Sleep-Based Aloha Simulator.

## Prerequisites

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** (Python package manager)

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).

## Installation

### Step 1: Navigate to Project Directory

```bash
cd fyp
```

### Step 2: (Optional but Recommended) Create Virtual Environment

Creating a virtual environment keeps your project dependencies isolated:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when the virtual environment is active.

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `simpy` - Discrete-event simulation framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `pandas` - Data analysis
- `tqdm` - Progress bars

**Installation time:** ~1-2 minutes depending on your internet connection.

## Verify Installation

Run the quick start script to verify everything is working:

```bash
python quick_start.py
```

You should see:
1. Simulation parameters printed
2. A progress indication during simulation
3. Results summary with metrics
4. A visualization window with plots (close it to continue)

**Expected output:**
```
======================================================================
ON-DEMAND SLEEP-BASED ALOHA SIMULATOR - Quick Start
======================================================================
...
Lifetime:
  Average: 3000.00 slots
...
```

## Basic Usage

### 1. Run the Basic Simulator

```bash
python simulator.py
```

This runs a single simulation with default parameters and prints results.

### 2. Run Quick Start

```bash
python quick_start.py
```

This runs a simulation and generates visualizations.

### 3. Explore Examples

```bash
python example_usage.py
```

This file contains 6 examples. By default, only Example 1 runs. To run others:

1. Open `example_usage.py` in a text editor
2. Uncomment the example you want (remove the `#` at the start of the line)
3. Save and run again

**Examples available:**
- Example 1: Basic simulation (fast, ~10 seconds)
- Example 2: Parameter sweep (moderate, ~2 minutes)
- Example 3: Optimization (moderate, ~2 minutes)
- Example 4: Tradeoff analysis (slow, ~5-10 minutes)
- Example 5: Multi-parameter sweep (moderate, ~3 minutes)
- Example 6: Configuration comparison (moderate, ~2 minutes)

## Quick Customization

### Change Simulation Parameters

Edit any Python script and modify the parameters:

```python
# In quick_start.py or your own script
N_NODES = 20           # Try 5, 10, 20, 50
LAMBDA_ARRIVAL = 0.15  # Try 0.05, 0.1, 0.2
Q_TRANSMIT = 0.08      # Try 0.01 to 0.3
TS_IDLE = 10           # Try 1, 5, 10, 20
SIM_TIME = 5000        # Try 1000, 3000, 5000
```

### Find Optimal Parameters

```python
from optimizer import find_optimal_q

# Copy base_params from example_usage.py
base_params = {
    'n_nodes': 10,
    'lambda_arrival': 0.1,
    # ... other params
}

# Find best q for maximum lifetime
q_opt, metrics = find_optimal_q(
    base_params=base_params,
    q_range=(0.01, 0.3),
    n_samples=20,
    objective='lifetime'
)

print(f"Optimal q: {q_opt:.4f}")
```

## Common Issues

### Issue: `ModuleNotFoundError`

**Problem:** Missing required packages

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Visualization window doesn't appear

**Problem:** Running in headless environment (no display)

**Solution:** 
- The simulation will still run and print text results
- To generate plot files instead of displaying, modify visualizer calls:
  ```python
  plot_metrics_summary(metrics, save_path='results.png')
  ```

### Issue: Simulation is too slow

**Solutions:**
1. Reduce `simulation_time` (e.g., from 5000 to 1000 slots)
2. Reduce number of nodes
3. Use fewer samples in parameter sweeps
4. Enable parallel processing:
   ```python
   results = parameter_sweep(..., parallel=True)
   ```

### Issue: Unicode errors on Windows

**Problem:** Console doesn't support Unicode characters

**Solution:** Already fixed in the code (we use ASCII characters only)

## Next Steps

1. **Read the README.md** for complete documentation
2. **Try the examples** in `example_usage.py`
3. **Modify parameters** to explore different scenarios
4. **Create your own scripts** using the provided modules

## Project Structure

```
fyp/
├── simulator.py         # Core simulation engine
├── optimizer.py         # Parameter optimization tools
├── visualizer.py        # Plotting functions
├── example_usage.py     # Comprehensive examples
├── quick_start.py       # Quick demonstration
├── requirements.txt     # Python dependencies
├── README.md           # Full documentation
├── GETTING_STARTED.md  # This file
└── prompts/            # Original design documents
    ├── grok1.md
    └── grok2.md
```

## Getting Help

1. Check **README.md** for detailed documentation
2. Read the docstrings in the code:
   ```bash
   python -c "from simulator import Simulator; help(Simulator)"
   ```
3. Review the example scripts for usage patterns

## Learning Path

**Beginner:**
1. Run `quick_start.py` ✓
2. Modify parameters in `quick_start.py`
3. Run Example 1 in `example_usage.py`

**Intermediate:**
1. Run Examples 2-3 (parameter sweeps and optimization)
2. Create your own simulation script
3. Try different parameter ranges

**Advanced:**
1. Run Examples 4-6 (tradeoff analysis, multi-parameter)
2. Implement custom metrics or modifications
3. Compare with analytical results from the paper

## Troubleshooting Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Reinstall requirements
pip install --upgrade -r requirements.txt

# Test individual modules
python -c "import simpy; print('SimPy OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

## Performance Tips

- **First run is slower:** Python compiles and caches modules
- **Subsequent runs are faster:** Use the same parameter values
- **Large simulations:** Use `simulation_time < 10000` for quick testing
- **Many runs:** Enable `parallel=True` in optimizer functions

---

**You're ready to go!** Start with `python quick_start.py` 🚀
