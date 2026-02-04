# On-Demand Sleep-Based Aloha Simulator

A Python-based discrete-event simulator for **Machine-to-Machine (M2M) communication** using **slotted Aloha** with an **on-demand sleep mechanism**. This implementation is based on the paper:

> "On-Demand-Sleep-Based Aloha for M2M Communication: Modeling, Optimization, and Tradeoff Between Lifetime and Delay"

## Features

- **Discrete-event simulation** using SimPy
- **Slotted Aloha protocol** with collision detection
- **On-demand sleep mechanism** (buffer-based, event-triggered sleep)
- **Energy consumption modeling** (sleep, wake-up, transmit, idle states)
- **Comprehensive metrics**: lifetime, delay, throughput, collision rate, energy efficiency
- **Parameter optimization**: find optimal transmission probability (q) and idle timeout (ts)
- **Tradeoff analysis**: lifetime vs delay curves
- **Rich visualizations**: metrics plots, state timelines, energy consumption, tradeoff curves

## System Model

### Protocol Overview

- **n nodes** (MTDs) transmit to a single receiver
- **Time is slotted**: all operations occur in discrete time slots
- **Bernoulli packet arrivals**: each node generates packets with probability λ per slot
- **Slotted Aloha access**: each active node transmits with probability q per slot
- **Collision model**: successful transmission only if exactly one node transmits

### On-Demand Sleep Mechanism

Nodes transition through four states:

1. **Active**: Has packets in buffer, attempting transmission with probability q
2. **Idle**: Buffer empty, waiting for ts slots before sleeping
3. **Sleep**: Deep sleep mode (very low power PS)
4. **Wake-up**: Transitioning from sleep to active (takes tw slots, power PW)

Key behavior:
- Node sleeps **only after** buffer is empty for ts slots (on-demand/event-triggered)
- New packet arrival during sleep triggers immediate wake-up
- More aggressive than duty-cycling (no forced sleep while packets waiting)

### Energy Model

Power consumption per slot:
- **PS**: Sleep power (very low, ~0.1)
- **PW**: Wake-up power (moderate, ~1.0)
- **PT**: Transmit power (high, ~5.0)
- **PB**: Busy/idle power (low, ~0.5)

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone or download the repository
cd fyp

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Method 1: Using Configuration File (Easiest)

**Step 1:** Edit `config.py` to set your parameters:
```python
# In config.py
n_nodes = 10
lambda_arrival = 0.1
q_transmit = 0.05
# ... etc
```

**Step 2:** Run the simulation:
```bash
python run_simulation.py
```

See `CONFIG_GUIDE.md` for detailed configuration instructions.

### Method 2: Using Presets

```bash
# Try different preset configurations
python run_simulation.py --preset low     # Low traffic
python run_simulation.py --preset high    # High traffic
python run_simulation.py --preset energy  # Energy-optimized
python run_simulation.py --preset delay   # Delay-optimized
```

### Method 3: Direct Python Code

```python
import simpy
from simulator import Simulator

# Create environment
env = simpy.Environment()

# Create simulator
sim = Simulator(
    env=env,
    n_nodes=10,           # Number of nodes
    lambda_arrival=0.1,   # Packet arrival rate
    q_transmit=0.05,      # Transmission probability
    ts_idle=5,            # Idle timeout (slots)
    tw_wakeup=2,          # Wake-up duration (slots)
    E_initial=10000,      # Initial energy
    PS=0.1, PW=1.0,       # Sleep/wake-up power
    PT=5.0, PB=0.5,       # Transmit/busy power
    simulation_time=5000, # Duration (slots)
    seed=42
)

# Run simulation
sim.run()

# Get results
metrics = sim.collect_metrics()
print(f"Average Lifetime: {metrics['avg_lifetime']:.2f} slots")
print(f"Average Delay: {metrics['avg_delay']:.2f} slots")
print(f"Throughput: {metrics['total_throughput']:.4f} packets/slot")
```

### Run Examples

```bash
# Quick start with visualization
python quick_start.py

# Run with current config.py settings
python run_simulation.py

# Run with preset configuration
python run_simulation.py --preset quick

# Run comprehensive examples
python example_usage.py
```

The example script demonstrates:
1. Basic simulation and visualization
2. Parameter sweeps (varying q or ts)
3. Optimization (finding optimal q)
4. Tradeoff analysis (lifetime vs delay)
5. Multi-parameter sweeps (2D grid search)
6. Configuration comparisons

## Module Overview

### `config.py`

**Centralized configuration file** for all simulation parameters:

- System parameters (n_nodes, λ, q, ts, tw)
- Energy parameters (E_initial, PS, PW, PT, PB)
- Simulation parameters (time, seed)
- Optimization parameters (ranges, samples)
- Preset configurations (7 built-in scenarios)
- Helper functions (`get_base_params()`, `print_config()`, `validate_config()`)

**Usage:** Edit `config.py` then run `python run_simulation.py`

See `CONFIG_GUIDE.md` for complete configuration documentation.

### `run_simulation.py`

**Main simulation runner** that uses `config.py`:

```bash
python run_simulation.py                 # Use config.py
python run_simulation.py --preset low    # Use preset
python run_simulation.py --validate      # Check config
python run_simulation.py --print-config  # Show config
```

### `simulator.py`

Core simulation engine with three main classes:

- **`Packet`**: Data packet with arrival time (for delay tracking)
- **`Node`**: Machine-type device with buffer, state machine, energy tracking
- **`Simulator`**: Network coordinator managing all nodes and collision detection

Key methods:
- `Node.generate_traffic()`: Bernoulli packet generation
- `Node.process_slot()`: State machine and energy consumption
- `Simulator.slot_coordinator()`: Collision resolution for slotted Aloha

### `optimizer.py`

Parameter optimization and analysis tools:

- **`parameter_sweep()`**: Sweep a single parameter over a range
- **`multi_parameter_sweep()`**: Grid search over multiple parameters
- **`find_optimal_q()`**: Find optimal q for max lifetime or min delay
- **`analyze_tradeoff()`**: Generate lifetime vs delay tradeoff curves
- **`sensitivity_analysis()`**: Measure parameter sensitivity
- **`compare_configurations()`**: Compare different protocol configurations

### `visualizer.py`

Visualization functions using matplotlib and seaborn:

- **`plot_metrics_summary()`**: Comprehensive metric overview (6-panel plot)
- **`plot_tradeoff_curve()`**: Lifetime vs delay tradeoff
- **`plot_parameter_sweep()`**: Multi-panel sweep results
- **`plot_energy_timeline()`**: Energy consumption over time
- **`plot_state_timeline()`**: State transitions over time
- **`plot_node_comparison()`**: Compare performance across nodes
- **`plot_3d_tradeoff()`**: 3D surface plot for multi-parameter analysis

## Key Parameters

### Protocol Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `n_nodes` | n | Number of nodes | 5-50 |
| `lambda_arrival` | λ | Packet arrival probability per slot | 0.01-0.5 |
| `q_transmit` | q | Transmission probability per slot | 0.01-0.3 |
| `ts_idle` | ts | Idle timeout before sleep (slots) | 1-20 |
| `tw_wakeup` | tw | Wake-up duration (slots) | 1-5 |

### Energy Parameters

| Parameter | Symbol | Description | Typical Value |
|-----------|--------|-------------|---------------|
| `E_initial` | E | Initial energy per node | 10000 |
| `PS` | PS | Sleep power | 0.1 |
| `PW` | PW | Wake-up power | 1.0 |
| `PT` | PT | Transmit power | 5.0 |
| `PB` | PB | Busy/idle power | 0.5 |

### Simulation Parameters

| Parameter | Description |
|-----------|-------------|
| `simulation_time` | Duration in time slots (e.g., 5000) |
| `seed` | Random seed for reproducibility |

## Performance Metrics

The simulator tracks:

**Lifetime Metrics:**
- Average/min/max node lifetime (slots until energy depletion)
- Energy consumption ratio

**Delay Metrics:**
- Average/max queueing delay (arrival to successful transmission)

**Throughput Metrics:**
- Total network throughput (packets/slot)
- Per-node throughput

**Transmission Statistics:**
- Total transmissions, collisions, delivery ratio
- Collision rate

## Usage Examples

### Example 1: Find Optimal q

```python
from optimizer import find_optimal_q

base_params = {
    'n_nodes': 10, 'lambda_arrival': 0.1, 'ts_idle': 5, 'tw_wakeup': 2,
    'E_initial': 10000, 'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
    'simulation_time': 3000, 'q_transmit': 0.05  # Will be optimized
}

# Optimize for maximum lifetime
q_opt, metrics = find_optimal_q(
    base_params=base_params,
    q_range=(0.01, 0.3),
    n_samples=20,
    objective='lifetime',
    n_runs=3
)

print(f"Optimal q: {q_opt:.4f}")
print(f"Lifetime: {metrics['avg_lifetime']:.2f} slots")
```

### Example 2: Tradeoff Analysis

```python
from optimizer import analyze_tradeoff
from visualizer import plot_tradeoff_curve

# Analyze lifetime vs delay for different ts values
tradeoff_df = analyze_tradeoff(
    base_params=base_params,
    ts_values=np.array([1, 3, 5, 10, 15, 20]),
    q_range=(0.01, 0.3),
    n_q_samples=15
)

# Plot results
plot_tradeoff_curve(
    results=tradeoff_df.to_dict('records'),
    x_param='min_delay',
    y_param='max_lifetime',
    color_param='ts_idle'
)
```

### Example 3: Parameter Sweep

```python
from optimizer import parameter_sweep
from visualizer import plot_parameter_sweep

# Sweep q from 0.01 to 0.3
results = parameter_sweep(
    base_params=base_params,
    sweep_param='q_transmit',
    sweep_values=np.linspace(0.01, 0.3, 20),
    n_runs=2
)

# Visualize
plot_parameter_sweep(
    results=results,
    sweep_param='q_transmit',
    metrics_to_plot=['avg_lifetime', 'avg_delay', 'collision_rate']
)
```

## Implementation Notes

### Discrete-Event Simulation

- Uses **SimPy** for discrete-event scheduling
- All time advances in discrete slots (1 time unit = 1 slot)
- Events: packet arrivals, transmissions, state transitions, energy updates

### Collision Detection

- Global slot coordinator tracks all transmission attempts
- Success only if exactly 1 node transmits in a slot
- Collisions require retransmission (packet stays in buffer)

### State Machine

Each node implements a 4-state machine:
```
    [Active] <--> [Idle] --> [Sleep]
       ^                       |
       |---- [Wake-up] <-------|
```

Transitions:
- Active → Idle: Buffer empty
- Idle → Sleep: Timer expires (ts slots)
- Sleep → Wake-up: Packet arrives
- Wake-up → Active: Timer completes (tw slots)

### Energy Tracking

- Energy decreases by power consumption per slot
- Node "dies" when energy ≤ 0 (lifetime recorded)
- Different power for each state

## Validation

The simulator can be validated against:

1. **Analytical expressions** from the paper (service rate μ, average power)
2. **Known Aloha properties** (throughput vs load curves)
3. **Energy balance** (total consumed = sum of per-state consumption)

## Advanced Usage

### Parallel Execution

```python
from optimizer import parameter_sweep

# Enable parallel processing for faster sweeps
results = parameter_sweep(
    base_params=base_params,
    sweep_param='q_transmit',
    sweep_values=np.linspace(0.01, 0.3, 50),
    n_runs=10,
    parallel=True  # Use all CPU cores
)
```

### Custom Metrics

Access detailed per-node data:

```python
node_metrics = sim.get_detailed_node_metrics()
for node in node_metrics:
    print(f"{node['name']}: lifetime={node['lifetime']}, "
          f"delay={node['avg_delay']:.2f}")
```

### Save Results

```python
from visualizer import create_results_table

# Create and save results table
df = create_results_table(results, save_path='results.csv')

# Save plots
plot_metrics_summary(metrics, save_path='summary.png')
```

## Paper Replication

To replicate the paper's key results:

1. **Optimal q for different λ** (Section V):
```python
lambda_values = [0.05, 0.1, 0.15, 0.2]
for lam in lambda_values:
    params['lambda_arrival'] = lam
    q_opt, metrics = find_optimal_q(params, objective='lifetime')
    print(f"λ={lam}: optimal q={q_opt:.4f}")
```

2. **Lifetime vs delay tradeoff** (Figure 4):
```python
tradeoff_df = analyze_tradeoff(
    base_params=params,
    ts_values=np.linspace(1, 20, 10)
)
```

3. **Comparison with duty-cycling**: Implement duty-cycling variant and use `compare_configurations()`

## License

This simulator is provided for educational and research purposes.

## References

Based on concepts from:
- Slotted Aloha protocol
- M2M/IoT communication
- On-demand sleep mechanisms
- Queueing theory for wireless networks

## Contact

For questions or issues, please refer to the original paper or open an issue in the repository.
