# Sleep-Based Low-Latency Access for M2M Communications Simulator

## Getting started (after cloning)

1. **Install dependencies** (from the repo root):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run notebooks**: Open any notebook in `examples/` (e.g. in Jupyter or VS Code). The first cell adds the project root to `sys.path` so `from src...` works whether the kernel’s working directory is the repo root or `examples/`.
3. **Run tests** (from repo root):
   ```bash
   pytest tests/
   ```
   Or run the validation script: `python run_validation.py`

## Overview

This project implements a discrete-event simulation framework for sleep-based random access schemes in Machine-to-Machine (M2M) communications. The simulator focuses on slotted Aloha with on-demand sleep to analyze the trade-offs between battery lifetime and low-latency access in massive IoT networks with battery-powered Machine-Type Devices (MTDs).

## Project Objectives

The simulator addresses four main objectives:

1. **O1:** Set up a discrete-event simulation framework for sleep-based random access schemes (slotted Aloha with on-demand sleep as baseline)
2. **O2:** Quantify impact of key parameters (ts, tw, q, λ, n, traffic models) via simulation
3. **O3:** Optimize sleep and access parameters for latency-longevity trade-offs using simulation results
4. **O4:** Validate results against 3GPP mMTC parameters (e.g., RA-SDT, MICO mode, T3324 timer)

## Current Implementation Status

### 🎉 Objective O1: COMPLETE (100%)
### 🎉 Objective O2: COMPLETE (100%)

**45 tests passing | 38 days ahead of schedule | ~4,500 lines of code**

---

### ✅ Completed: Task 1.1 - Node Class (Feb 10, 2026)

The MTD Node class has been fully implemented with the following features:

#### Node States
- **ACTIVE**: Node has packets and is actively contending for transmission
- **IDLE**: Node has no packets, idle timer running before sleep
- **SLEEP**: Deep sleep mode for energy conservation
- **WAKEUP**: Transitioning from sleep to active state

#### Key Features
- **Packet Queue**: FIFO queue with arrival timestamps for delay tracking
- **Energy Tracking**: Realistic power model with 5 power states (PT, PB, PI, PW, PS)
- **State Management**: Automatic state transitions based on queue status and timers
- **Statistics**: Comprehensive tracking of delays, energy consumption, and state occupancy

#### Methods Implemented
- `arrive_packet()`: Bernoulli packet arrival process
- `update_state()`: State transition logic for on-demand sleep
- `attempt_transmit()`: Probabilistic transmission attempts (parameter q)
- `handle_success()`: Process successful transmissions and record delays
- `consume_energy()`: State-based energy consumption
- `get_statistics()`: Comprehensive metrics collection

#### Power Model
The implementation supports configurable power consumption rates:
- **PT**: Transmit power (highest)
- **PB**: Busy/collision power
- **PI**: Idle power
- **PW**: Wake-up transition power
- **PS**: Sleep power (lowest)

### ✅ Completed: Task 1.2 - Simulator Class (Feb 10, 2026)

The Simulator class has been fully implemented, providing discrete-event simulation capabilities:

**Key Features:**
- ✓ Multi-node network management (supports 10-10,000 nodes)
- ✓ Slotted time loop with configurable max slots
- ✓ Collision detection (success if exactly 1 transmission)
- ✓ Comprehensive metrics collection
- ✓ Time series tracking (queue, energy, states)
- ✓ Reproducible simulations with seed control

**Classes Implemented:**

1. `SimulationConfig` - Configuration dataclass for simulation parameters
2. `SimulationResults` - Results dataclass with comprehensive metrics
3. `Simulator` - Main discrete-event simulator
4. `BatchSimulator` - Batch processing with parameter sweeps

**Metrics Tracked:**
- Network: throughput, arrivals, deliveries, collisions
- Delay: mean, 95th/99th percentile
- Energy: consumption by state, lifetime in years
- States: time fractions in each state
- Performance: empirical success probability (p), service rate (μ)

**Batch Processing:**
- Multiple replications with different seeds
- Parameter sweeps (q, ts, n, λ)
- Statistical aggregation with confidence intervals

### ✅ Completed: Task 1.3 - Power Model (Feb 10, 2026)

The PowerModel module provides realistic 3GPP-inspired power consumption profiles:

**Key Features:**
- ✓ 6 predefined power profiles based on 3GPP specifications
- ✓ Realistic power values (mW) for MT communications
- ✓ Battery configuration with 5 common battery types
- ✓ Lifetime estimation utilities
- ✓ Custom profile creation support

**Power Profiles:**

1. **LoRa** - Low-power long-range (PT: 120mW, PS: 1μW)
2. **NB-IoT** - 3GPP NB-IoT Release 13+ (PT: 220mW, PS: 15μW)
3. **LTE-M** - eMTC (PT: 250mW, PS: 20μW)
4. **5G NR mMTC** - With MICO mode (PT: 200mW, PS: 10μW)
5. **Generic Low** - Generic low-power IoT (PT: 100mW, PS: 5μW)
6. **Generic High** - Generic high-power (PT: 500mW, PS: 100μW)

**Battery Types:**
- AA, AAA, Coin Cell, LiPo (small/large)
- Automatic energy calculations (mWh, Joules)
- Conversion to simulation units

**Features:**
- Power normalization for simulation
- Lifetime estimation (years, days, slots)
- Custom profile creation with typical ratios

### ✅ Completed: Task 1.4 - Basic Testing & Debugging (Feb 10, 2026)

The Validation module provides comprehensive testing and debugging utilities:

**Key Features:**
- ✓ Trace logging for slot-by-slot debugging
- ✓ Analytical validation against theoretical models
- ✓ Sanity checks for expected behaviors
- ✓ Small-scale integration tests

**Components:**

1. **TraceLogger** - Detailed slot-by-slot state tracking
   - Records node states, queues, energy per slot
   - Save/load trace data to JSON
   - Print summaries for debugging

2. **AnalyticalValidator** - Compare simulation to theory
   - Success probability: p = q(1-q)^(n-1)
   - Service rate: μ calculations
   - Validation against paper's analytical models

3. **SanityChecker** - Automated sanity checks
   - No-sleep (ts=∞) matches standard Aloha
   - Immediate sleep (ts=0) increases sleep fraction
   - Higher q increases collisions

**Integration Tests:**
- Small-scale test (n=5, 1000 slots)
- Comprehensive validation suite
- Standalone validation script

---

### ✅ Completed: Objective O2 - Parameter Impact Quantification (Feb 10, 2026)

**All three O2 tasks completed 38 days ahead of schedule!**

#### Task 2.1: Metrics Calculation ✅

**Comprehensive Metrics:**
- Delay: mean, 95th/99th percentiles
- Throughput and success probability
- Energy consumption and lifetime estimation
- State fractions (active, idle, sleep, wakeup)
- Queue length tracking
- Network statistics (arrivals, deliveries, collisions)

**Analytical Validation:**
- `AnalyticalValidator` class
- Success probability: p = q(1-q)^(n-1)
- Service rate: μ = p / (1 + tw·λ/(1-λ))
- Automatic validation against theoretical formulas

#### Task 2.2: Parameter Sweep Experiments ✅

**Traffic Models Module** (`src/traffic_models.py` - 400+ lines):
- **Poisson (Bernoulli):** Independent arrivals with probability λ
- **Bursty:** Batch/correlated arrivals for event-driven traffic
- **Periodic:** Deterministic arrivals with jitter
- **Mixed:** Heterogeneous networks with multiple traffic types
- Effective arrival rate calculation
- Reproducibility with seed control

**Experiments Module** (`src/experiments.py` - 550+ lines):
- `ExperimentSuite` class with pre-configured experiments:
  - Transmission probability (q) sweep [0.01-0.5]
  - Idle timer (ts) sweep [0-100]
  - Arrival rate (λ) sweep [0.001-0.1]
  - Number of nodes (n) sweep [10-500]
  - Scenario comparisons (low-latency vs. battery-life)
- Batch replication support (20-50 runs)
- Statistical aggregation
- JSON result export

**Test Suite:**
- 9 comprehensive traffic model tests (all passing)
- Coverage: Poisson, bursty, periodic, mixed traffic
- Reproducibility and effective rate validation

#### Task 2.3: Visualization Integration ✅

**Visualization Module** (`src/visualization.py` - 650+ lines):

**9 Plot Types:**
1. **Delay vs. Lifetime** - Trade-off curves with Pareto frontier
2. **Parameter Impact** - Multi-metric analysis
3. **State Fractions** - Pie charts of node state occupancy
4. **Energy Breakdown** - Pie + bar charts
5. **Time Series** - Queue and energy evolution
6. **Scenario Comparison** - Multi-scenario overlays
7. **2D Heatmaps** - Parameter interaction visualization
8. **Summary Figures** - 4-panel comprehensive views
9. **Custom Plots** - Publication-quality configurability

**Interactive Widgets:**
- ipywidgets interface for real-time exploration
- Sliders for q, ts, n, λ parameters
- Live plot updates
- Demo-ready interface

**Publication Quality:**
- 300 DPI output
- Configurable styling
- Automatic figure saving
- Professional formatting

**Demo Notebook:**
- `examples/objective_o2_demo.ipynb` - Complete O2 demonstration
- All tasks showcased with interactive examples
- Publication-ready figures

---

## Project Structure

```
fyp/
├── docs/
│   ├── prd.md                          # Product Requirements Document
│   ├── task.md                         # Task breakdown (updated)
│   ├── task_1_1_completion_summary.md  # Task 1.1 report
│   ├── task_1_2_completion_summary.md  # Task 1.2 report
│   ├── task_1_3_completion_summary.md  # Task 1.3 report
│   ├── task_1_4_completion_summary.md  # Task 1.4 report
│   ├── task_2_completion_summary.md    # Task 2 (O2) report ← NEW
│   └── PROGRESS.md                     # Progress tracking
├── src/
│   ├── __init__.py           # Package initialization
│   ├── node.py               # MTD Node class (510 lines)
│   ├── simulator.py          # Simulator classes (650+ lines)
│   ├── power_model.py        # Power profiles (410+ lines)
│   ├── validation.py         # Validation utilities (560+ lines)
│   ├── traffic_models.py     # Traffic generation (400+ lines) ← NEW
│   ├── visualization.py      # Plotting & widgets (650+ lines) ← NEW
│   └── experiments.py        # Experiment suite (550+ lines) ← NEW
├── tests/
│   ├── __init__.py
│   ├── test_node.py          # Node tests (8 tests)
│   ├── test_simulator.py     # Simulator tests (10 tests)
│   ├── test_power_model.py   # Power model tests (11 tests)
│   ├── test_validation.py    # Validation tests (7 tests)
│   └── test_traffic_models.py # Traffic tests (9 tests) ← NEW
├── examples/
│   ├── node_demo.ipynb       # Node class demo
│   ├── simulator_demo.ipynb  # Simulator demo with visualizations
│   ├── power_model_demo.ipynb # Power model demo
│   └── objective_o2_demo.ipynb # O2 complete demo ← NEW
├── run_validation.py         # Standalone validation script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

**Total:** ~4,500 lines of production code | 45 tests passing

## Installation

1. Clone or navigate to the repository:
```bash
cd c:\Users\lsfer\StudioProjects\ipynb\fyp\fyp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing

Run all tests with pytest:

```bash
# Run all tests (45 tests)
pytest tests/ -v

# Run specific test modules
pytest tests/test_node.py         # 8 tests
pytest tests/test_simulator.py    # 10 tests
pytest tests/test_power_model.py  # 11 tests
pytest tests/test_validation.py   # 7 tests
pytest tests/test_traffic_models.py # 9 tests ← NEW

# Run validation script
python run_validation.py
```

**Current Status: 45/45 tests passing ✅**

**Node Tests:**
- Initialization, packet arrival, state transitions
- Energy consumption, transmission attempts
- Delay calculation, statistics, depletion

**Simulator Tests:**
- Initialization, collision detection, simulation run
- Reproducibility, energy depletion, history tracking
- Batch replications, parameter sweeps, aggregation
- Transmission probability impact

## Usage Example

### Basic Simulation

```python
from src.simulator import Simulator, SimulationConfig

# Configure simulation
power_rates = {
    'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 
    'PW': 2.0, 'PS': 0.1
}

config = SimulationConfig(
    n_nodes=10,
    arrival_rate=0.01,         # λ
    transmission_prob=0.1,     # q
    idle_timer=10,             # ts
    wakeup_time=5,             # tw
    initial_energy=5000.0,
    power_rates=power_rates,
    max_slots=10000,
    seed=42
)

# Run simulation
sim = Simulator(config)
result = sim.run_simulation(track_history=True, verbose=True)

# Print results
print(f"Mean delay: {result.mean_delay:.2f} slots")
print(f"Throughput: {result.throughput:.4f} pkts/slot")
print(f"Lifetime: {result.mean_lifetime_years:.4f} years")
```

### Parameter Sweep

```python
from src.simulator import BatchSimulator

# Create batch simulator
batch_sim = BatchSimulator(config)

# Sweep transmission probability
q_values = [0.05, 0.1, 0.15, 0.2]
sweep_results = batch_sim.parameter_sweep(
    param_name='transmission_prob',
    param_values=q_values,
    n_replications=20,
    verbose=True
)

# Aggregate results
for q, results in sweep_results.items():
    aggregated = BatchSimulator.aggregate_results(results)
    mean_delay, std_delay = aggregated['mean_delay']
    print(f"q={q}: delay={mean_delay:.2f}±{std_delay:.2f}")
```

## Next Steps

### Upcoming Tasks (from task.md)

- **Task 1.3** (Due Feb 25, 2026): Integrate Power Model
  - Configurable power consumption with 3GPP-inspired values
  - Battery lifetime estimation in years

- **Task 1.4** (Due Feb 28, 2026): Basic Testing & Debugging
  - Small-scale tests for state transitions and collisions
  - Sanity checks against no-sleep Aloha baseline

## Key Parameters

- **n**: Number of MTD nodes (100-10,000)
- **λ**: Packet arrival rate per slot (Bernoulli parameter)
- **q**: Transmission probability per slot
- **ts**: Idle timer (slots before sleep)
- **tw**: Wake-up time (slots to transition from sleep to active)
- **Slot duration**: ~6 ms (3GPP NR based)

## References

- Wang et al. (2024): Sleep-Based Low-Latency Access for M2M Communications
- 3GPP specifications for mMTC (RA-SDT, MICO mode, T3324 timer)
- FYP Progress Report (2026)

## License

Academic project for Final Year Project.

---

**Last Updated:** February 10, 2026  
**Status:** Objective O1 COMPLETE - All 4 Tasks Finished ✓  
**Progress:** Baseline simulator production-ready with 3GPP power models and validation (100%)  
**Total Tests:** 36/36 passing ✅
