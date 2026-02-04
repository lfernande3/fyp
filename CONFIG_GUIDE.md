# Configuration Guide

## Quick Start

The easiest way to run simulations is by editing **`config.py`** and running **`run_simulation.py`**.

### Step 1: Edit Parameters

Open `config.py` and modify the parameters you want:

```python
# In config.py
n_nodes = 20           # Change number of nodes
lambda_arrival = 0.15  # Change traffic load
q_transmit = 0.08      # Change transmission probability
ts_idle = 10           # Change idle timeout
simulation_time = 3000 # Change simulation duration
```

### Step 2: Run Simulation

```bash
python run_simulation.py
```

That's it! The simulation will use your parameters from `config.py`.

## Parameter Reference

### System Parameters (Most Important)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_nodes` | 10 | 5-50 | Number of devices in network |
| `lambda_arrival` | 0.1 | 0.01-0.5 | Packet arrival probability per slot |
| `q_transmit` | 0.05 | 0.01-0.3 | Transmission probability per slot |
| `ts_idle` | 5 | 1-20 | Slots to wait before sleeping |
| `tw_wakeup` | 2 | 1-5 | Slots needed to wake up |

### Energy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E_initial` | 10000 | Battery capacity |
| `PS_sleep` | 0.1 | Power in sleep mode |
| `PW_wakeup` | 1.0 | Power during wake-up |
| `PT_transmit` | 5.0 | Power when transmitting |
| `PB_busy` | 0.5 | Power when active/idle |

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulation_time` | 5000 | Duration in time slots |
| `random_seed` | 42 | Seed for reproducibility (None=random) |

## Using Presets

Instead of editing `config.py`, use built-in presets:

### Available Presets

```bash
# Low traffic (λ=0.05)
python run_simulation.py --preset low

# Medium traffic (λ=0.1, default)
python run_simulation.py --preset medium

# High traffic (λ=0.2)
python run_simulation.py --preset high

# Large network (50 nodes)
python run_simulation.py --preset large

# Quick test (fast, for testing)
python run_simulation.py --preset quick

# Energy-optimized (maximize lifetime)
python run_simulation.py --preset energy

# Delay-optimized (minimize delay)
python run_simulation.py --preset delay
```

## Common Use Cases

### 1. Test Different Traffic Loads

**Edit config.py:**
```python
lambda_arrival = 0.05  # Low traffic
lambda_arrival = 0.1   # Medium traffic
lambda_arrival = 0.2   # High traffic
```

**Or use presets:**
```bash
python run_simulation.py --preset low
python run_simulation.py --preset high
```

### 2. Find Best Transmission Probability

**Option A: Manual testing**

Edit `config.py`:
```python
q_transmit = 0.03  # Conservative
# Run simulation
q_transmit = 0.05  # Moderate
# Run simulation
q_transmit = 0.1   # Aggressive
# Run simulation
```

**Option B: Automatic optimization**

Use `example_usage.py` Example 3:
```python
from optimizer import find_optimal_q
import config

params = config.get_base_params()
q_opt, metrics = find_optimal_q(
    base_params=params,
    objective='lifetime'  # or 'delay'
)
print(f"Optimal q: {q_opt:.4f}")
```

### 3. Analyze Lifetime vs Delay Tradeoff

**Vary ts_idle in config.py:**
```python
ts_idle = 1   # Sleep immediately (best energy)
ts_idle = 5   # Moderate
ts_idle = 20  # Stay awake long (best delay)
```

**Or use automatic analysis:**
```python
from optimizer import analyze_tradeoff
import config

tradeoff_df = analyze_tradeoff(
    base_params=config.get_base_params(),
    ts_values=[1, 3, 5, 10, 15, 20]
)
```

### 4. Test Large Networks

**Edit config.py:**
```python
n_nodes = 50           # Larger network
q_transmit = 0.02      # Lower q to reduce collisions
simulation_time = 3000 # May need shorter sim (slower with more nodes)
```

**Or use preset:**
```bash
python run_simulation.py --preset large
```

### 5. Quick Testing During Development

**Edit config.py:**
```python
n_nodes = 5            # Fewer nodes
simulation_time = 1000 # Shorter duration
```

**Or use preset:**
```bash
python run_simulation.py --preset quick
```

## Validation

Check if your configuration is valid:

```bash
python run_simulation.py --validate
```

This will:
- Print your current configuration
- Check for parameter range issues
- Warn about potential problems (e.g., network overload)

## Print Current Configuration

```bash
python run_simulation.py --print-config
```

Or directly:

```bash
python config.py
```

## Using Config in Your Own Scripts

### Method 1: Import get_base_params()

```python
import simpy
from simulator import Simulator
import config

# Get current configuration
params = config.get_base_params()

# Modify specific parameters if needed
params['n_nodes'] = 20
params['lambda_arrival'] = 0.15

# Run simulation
env = simpy.Environment()
sim = Simulator(env=env, **params)
sim.run()
```

### Method 2: Import Individual Parameters

```python
from config import (
    n_nodes, lambda_arrival, q_transmit,
    ts_idle, tw_wakeup, E_initial,
    PS_sleep, PW_wakeup, PT_transmit, PB_busy,
    simulation_time, random_seed
)

# Use directly in Simulator
sim = Simulator(
    env=env,
    n_nodes=n_nodes,
    lambda_arrival=lambda_arrival,
    # ... etc
)
```

### Method 3: Use Presets

```python
from config import Presets

# Use a preset
params = Presets.LOW_TRAFFIC

# Or modify it
params = Presets.MEDIUM_TRAFFIC.copy()
params['simulation_time'] = 10000

# Run simulation
sim = Simulator(env=env, **params)
```

## Understanding Parameter Effects

### Network Load

**Network load** = `n_nodes × lambda_arrival × q_transmit`

- **Load < 0.1**: Light traffic, few collisions
- **Load ≈ 0.2-0.3**: Moderate traffic, good throughput
- **Load > 0.368**: **Overload!** Exceeds Aloha capacity

**Example:**
```python
n_nodes = 10
lambda_arrival = 0.1
q_transmit = 0.05
# Load = 10 × 0.1 × 0.05 = 0.05 (light)
```

### Transmission Probability (q)

**Higher q:**
- ✅ Faster packet transmission
- ✅ Less time in buffer → more sleep → better lifetime
- ❌ More collisions (if load is high)

**Lower q:**
- ✅ Fewer collisions
- ✅ Higher success rate per attempt
- ❌ Packets wait longer → less sleep → worse lifetime

**Optimal q:**
- Depends on network load
- Use `find_optimal_q()` to find it automatically

### Idle Timeout (ts)

**Lower ts (e.g., 1-3):**
- ✅ Sleep quickly after buffer empty → better energy
- ❌ Wake-up penalty if packet arrives soon → worse delay

**Higher ts (e.g., 15-20):**
- ✅ Stay awake longer → no wake-up delay → better delay
- ❌ Less sleep → worse energy

**Tradeoff:**
- ts controls the **fundamental lifetime vs delay tradeoff**
- Use `analyze_tradeoff()` to explore the curve

## Tips and Best Practices

### 1. Start Simple
```python
# Use defaults first
python run_simulation.py

# Then modify one parameter at a time
```

### 2. Watch for Warnings
```bash
python run_simulation.py --validate

# Fix any warnings before running long simulations
```

### 3. Use Quick Tests First
```python
# In config.py
simulation_time = 1000  # Quick test
# ... test different parameters ...

simulation_time = 5000  # Final run
```

### 4. Check Network Load
```python
# Calculate manually:
load = n_nodes * lambda_arrival * q_transmit
print(f"Load: {load:.4f}")

# Should be < 0.368 for stable Aloha
```

### 5. Enable Parallel for Sweeps
```python
# In config.py
enable_parallel_processing = True  # Much faster for parameter sweeps
```

### 6. Save Results
```python
# In config.py
save_plots = True
output_directory = "results"
```

## Troubleshooting

### Simulation Too Slow?
```python
simulation_time = 1000  # Reduce duration
n_nodes = 5             # Fewer nodes
```

### Too Many Collisions?
```python
q_transmit = 0.02      # Lower q
# or
lambda_arrival = 0.05  # Reduce traffic
```

### Nodes Die Too Quickly?
```python
E_initial = 50000      # More battery
# or
PS_sleep = 0.05        # Lower sleep power
ts_idle = 1            # Sleep more aggressively
```

### Delay Too High?
```python
ts_idle = 15           # Stay awake longer
q_transmit = 0.08      # Transmit more aggressively
```

## Example Workflow

### Finding Best Configuration

1. **Start with defaults:**
   ```bash
   python run_simulation.py
   ```

2. **Try different presets:**
   ```bash
   python run_simulation.py --preset energy
   python run_simulation.py --preset delay
   ```

3. **Find optimal q:**
   ```python
   # In your script
   from optimizer import find_optimal_q
   import config
   
   q_opt, _ = find_optimal_q(
       base_params=config.get_base_params(),
       objective='lifetime'
   )
   ```

4. **Update config.py with optimal value:**
   ```python
   q_transmit = 0.0782  # Found by optimizer
   ```

5. **Run final simulation:**
   ```bash
   python run_simulation.py
   ```

---

**Next Steps:**
- Edit `config.py` with your parameters
- Run `python run_simulation.py`
- Check `GETTING_STARTED.md` for more examples
