"""
Script to create a comprehensive Jupyter notebook with all simulator functionality
"""

import json

def create_cell(cell_type, content, execution_count=None):
    """Create a notebook cell"""
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': content.split('\n')
    }
    if cell_type == 'code':
        cell['execution_count'] = execution_count
        cell['outputs'] = []
    return cell

# Read the source files
with open('simulator.py', 'r', encoding='utf-8') as f:
    simulator_code = f.read()

with open('config.py', 'r', encoding='utf-8') as f:
    config_code = f.read()

# Create notebook
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Add cells
cells = []

# Title cell
cells.append(create_cell('markdown', '''# On-Demand Sleep-Based Aloha Simulator

**Final Year Project**  
**Date:** February 2026

## Overview

Complete implementation of a discrete-event simulator for M2M communication using Slotted Aloha with on-demand sleep.

### Features
- Slotted Aloha with collision detection
- On-demand sleep mechanism
- Energy tracking (4 power states)
- Performance metrics (lifetime, delay, throughput)
- Parameter optimization
- Visualization

### How to Use This Notebook
1. Run cells sequentially from top to bottom
2. Edit configuration in Section 2
3. View results and plots
4. Experiment with parameters

**All code is self-contained in this notebook!**'''))

# Setup cell
cells.append(create_cell('markdown', '''## 1. Import Dependencies'''))

cells.append(create_cell('code', '''# Install if needed (uncomment):
# !pip install simpy numpy matplotlib seaborn pandas

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Dependencies loaded!")'''))

# Configuration
cells.append(create_cell('markdown', '''## 2. Configuration Parameters

**Edit these to customize your simulation:**'''))

cells.append(create_cell('code', '''# ==================================================
# SIMULATION CONFIGURATION - Edit These Values
# ==================================================

# System Parameters
N_NODES = 10              # Number of nodes
LAMBDA_ARRIVAL = 0.1      # Packet arrival rate (λ)
Q_TRANSMIT = 0.05         # Transmission probability (q)
TS_IDLE = 5               # Idle timeout (slots)
TW_WAKEUP = 2             # Wake-up duration (slots)

# Energy Parameters  
E_INITIAL = 10000         # Initial energy
PS_SLEEP = 0.1            # Sleep power
PW_WAKEUP = 1.0           # Wake-up power
PT_TRANSMIT = 5.0         # Transmit power
PB_BUSY = 0.5             # Busy power

# Simulation
SIMULATION_TIME = 5000    # Duration (slots)
RANDOM_SEED = 42          # For reproducibility

print("="*70)
print(f"Configuration: {N_NODES} nodes, λ={LAMBDA_ARRIVAL}, q={Q_TRANSMIT}")
print(f"Sleep: ts={TS_IDLE}, tw={TW_WAKEUP}, E={E_INITIAL}")
print(f"Duration: {SIMULATION_TIME} slots")
print("="*70)'''))

# Core classes
cells.append(create_cell('markdown', '''## 3. Core Simulator Classes

### 3.1 Packet Class'''))

cells.append(create_cell('code', '''@dataclass
class Packet:
    """Data packet"""
    size: int
    src: str
    dest: str
    arrival_time: float
    packet_id: int = 0
    
    def __repr__(self):
        return f"Packet(id={self.packet_id}, src={self.src})"

print("Packet class defined")'''))

cells.append(create_cell('markdown', '''### 3.2 Node Class (MTD)

Implements 4-state machine: active → idle → sleep → wake-up'''))

cells.append(create_cell('code', simulator_code.split('class Node:')[1].split('class Simulator:')[0].strip()))

cells.append(create_cell('markdown', '''### 3.3 Simulator Class

Coordinates all nodes and handles collision detection.'''))

cells.append(create_cell('code', 'class Simulator:\n' + simulator_code.split('class Simulator:')[1].split('if __name__')[0].strip() + '\nprint("Simulator class defined")'))

# Visualization
cells.append(create_cell('markdown', '''## 4. Visualization Functions'''))

cells.append(create_cell('code', '''def plot_results(metrics):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Simulation Results', fontsize=16, fontweight='bold')
    
    # Lifetime
    ax = axes[0, 0]
    data = [metrics['min_lifetime'], metrics['avg_lifetime'], metrics['max_lifetime']]
    ax.bar(['Min', 'Avg', 'Max'], data, color=['red', 'green', 'blue'])
    ax.set_ylabel('Lifetime (slots)')
    ax.set_title('Node Lifetime')
    ax.grid(True, alpha=0.3)
    
    # Delay
    ax = axes[0, 1]
    ax.bar(['Avg', 'Max'], [metrics['avg_delay'], metrics['max_delay']], 
           color=['orange', 'red'])
    ax.set_ylabel('Delay (slots)')
    ax.set_title('Packet Delay')
    ax.grid(True, alpha=0.3)
    
    # Throughput
    ax = axes[0, 2]
    ax.bar(['Total', 'Per Node'], 
           [metrics['total_throughput'], metrics['avg_node_throughput']])
    ax.set_ylabel('Throughput (packets/slot)')
    ax.set_title('Throughput')
    ax.grid(True, alpha=0.3)
    
    # Transmissions (pie)
    ax = axes[1, 0]
    ax.pie([metrics['total_packets_sent'], metrics['total_collisions']], 
           labels=['Success', 'Collision'], autopct='%1.1f%%',
           colors=['green', 'red'], startangle=90)
    ax.set_title('Transmissions')
    
    # Energy (pie)
    ax = axes[1, 1]
    consumed = metrics['energy_consumed_ratio'] * 100
    ax.pie([consumed, 100-consumed], labels=['Used', 'Remaining'],
           autopct='%1.1f%%', colors=['orange', 'blue'], startangle=90)
    ax.set_title('Energy')
    
    # Delivery ratio
    ax = axes[1, 2]
    delivered = metrics['packet_delivery_ratio'] * 100
    bars = ax.bar(['Delivered', 'Buffered'], [delivered, 100-delivered],
                   color=['green', 'yellow'])
    ax.set_ylabel('%')
    ax.set_title('Packet Delivery')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

print("Visualization functions defined")'''))

# Run simulation
cells.append(create_cell('markdown', '''## 5. Run Simulation

Execute the simulation with configured parameters.'''))

cells.append(create_cell('code', '''print("Running simulation...")

env = simpy.Environment()
sim = Simulator(
    env=env,
    n_nodes=N_NODES,
    lambda_arrival=LAMBDA_ARRIVAL,
    q_transmit=Q_TRANSMIT,
    ts_idle=TS_IDLE,
    tw_wakeup=TW_WAKEUP,
    E_initial=E_INITIAL,
    PS=PS_SLEEP,
    PW=PW_WAKEUP,
    PT=PT_TRANSMIT,
    PB=PB_BUSY,
    simulation_time=SIMULATION_TIME,
    seed=RANDOM_SEED
)

sim.run()
metrics = sim.collect_metrics()

print("Simulation complete!")'''))

# Display results
cells.append(create_cell('markdown', '''## 6. Results'''))

cells.append(create_cell('code', '''print("="*70)
print("SIMULATION RESULTS")
print("="*70)

print(f"\\nLifetime:")
print(f"  Average: {metrics['avg_lifetime']:.2f} slots")
print(f"  Range: [{metrics['min_lifetime']:.0f}, {metrics['max_lifetime']:.0f}]")

print(f"\\nDelay:")
print(f"  Average: {metrics['avg_delay']:.2f} slots")
print(f"  Maximum: {metrics['max_delay']:.2f} slots")

print(f"\\nThroughput:")
print(f"  Total: {metrics['total_throughput']:.4f} packets/slot")
print(f"  Per node: {metrics['avg_node_throughput']:.4f} packets/slot")

print(f"\\nTransmissions:")
print(f"  Sent: {metrics['total_packets_sent']}")
print(f"  Arrived: {metrics['total_packets_arrived']}")
print(f"  Delivery ratio: {metrics['packet_delivery_ratio']:.1%}")
print(f"  Collisions: {metrics['total_collisions']}")
print(f"  Collision rate: {metrics['collision_rate']:.1%}")

print(f"\\nEnergy:")
print(f"  Remaining: {metrics['avg_energy_remaining']:.2f}")
print(f"  Consumed: {metrics['energy_consumed_ratio']:.1%}")

load = N_NODES * LAMBDA_ARRIVAL * Q_TRANSMIT
print(f"\\nNetwork Load: {load:.4f} (Aloha capacity: 0.368)")

print("="*70)'''))

cells.append(create_cell('code', '''# Visualize
plot_results(metrics)'''))

# Parameter sweep
cells.append(create_cell('markdown', '''## 7. Parameter Sweep (Optional)

Explore how q affects performance.'''))

cells.append(create_cell('code', '''# Sweep q from 0.01 to 0.2
q_values = np.linspace(0.01, 0.2, 10)
results = []

print(f"Sweeping {len(q_values)} q values...")
for q in q_values:
    env = simpy.Environment()
    sim = Simulator(env, N_NODES, LAMBDA_ARRIVAL, q, TS_IDLE, TW_WAKEUP,
                    E_INITIAL, PS_SLEEP, PW_WAKEUP, PT_TRANSMIT, PB_BUSY,
                    3000, RANDOM_SEED)  # Shorter sim for speed
    sim.run()
    results.append(sim.collect_metrics())

print("Sweep complete!")'''))

cells.append(create_cell('code', '''# Plot sweep results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Parameter Sweep: q', fontsize=14, fontweight='bold')

q_vals = [r['q_transmit'] for r in results]

# Lifetime
ax = axes[0, 0]
lifetimes = [r['avg_lifetime'] for r in results]
ax.plot(q_vals, lifetimes, 'o-', linewidth=2)
ax.plot(q_vals[np.argmax(lifetimes)], max(lifetimes), 'r*', markersize=15)
ax.set_xlabel('q')
ax.set_ylabel('Lifetime (slots)')
ax.set_title('Lifetime vs q')
ax.grid(True, alpha=0.3)

# Delay
ax = axes[0, 1]
delays = [r['avg_delay'] for r in results]
ax.plot(q_vals, delays, 'o-', linewidth=2, color='orange')
ax.plot(q_vals[np.argmin(delays)], min(delays), 'r*', markersize=15)
ax.set_xlabel('q')
ax.set_ylabel('Delay (slots)')
ax.set_title('Delay vs q')
ax.grid(True, alpha=0.3)

# Throughput
ax = axes[1, 0]
throughputs = [r['total_throughput'] for r in results]
ax.plot(q_vals, throughputs, 'o-', linewidth=2, color='green')
ax.plot(q_vals[np.argmax(throughputs)], max(throughputs), 'r*', markersize=15)
ax.set_xlabel('q')
ax.set_ylabel('Throughput')
ax.set_title('Throughput vs q')
ax.grid(True, alpha=0.3)

# Collision rate
ax = axes[1, 1]
coll_rates = [r['collision_rate'] for r in results]
ax.plot(q_vals, coll_rates, 'o-', linewidth=2, color='red')
ax.set_xlabel('q')
ax.set_ylabel('Collision Rate')
ax.set_title('Collision Rate vs q')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Show optimal values
best_life_idx = np.argmax(lifetimes)
best_delay_idx = np.argmin(delays)

print("\\nOptimal Values:")
print(f"  For max lifetime: q = {q_vals[best_life_idx]:.4f}")
print(f"  For min delay: q = {q_vals[best_delay_idx]:.4f}")'''))

# Conclusion
cells.append(create_cell('markdown', '''## 8. Conclusions

### Key Findings

1. **On-demand sleep** saves significant energy compared to always-active
2. **Transmission probability (q)** affects lifetime and delay
3. **Idle timeout (ts)** controls lifetime vs delay tradeoff
4. **Network load** should stay below 0.368 (Aloha capacity)

### Parameters Effect

- **Higher q**: Faster transmission, more collisions
- **Lower q**: Fewer collisions, longer delays
- **Higher ts**: Better delay, worse lifetime
- **Lower ts**: Better lifetime, worse delay

### Applications

This simulator can optimize:
- IoT network parameters
- M2M communication protocols
- Energy-constrained systems
- Massive device deployments

---

**Thank you for reviewing this simulator!**

For more details, see the complete documentation in the project files.'''))

notebook['cells'] = cells

# Save
with open('FYP_Simulator_Complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Complete notebook created: FYP_Simulator_Complete.ipynb")
print(f"Total cells: {len(cells)}")
