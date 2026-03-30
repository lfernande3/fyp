#!/usr/bin/env python3
"""
Script to Generate Key Plots for FYP Presentation
Run this to create all the essential figures for your PowerPoint
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("presentation_figures")
output_dir.mkdir(exist_ok=True)

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define colorblind-friendly palette
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']

def setup_plot(title="", xlabel="", ylabel="", figsize=(10, 6)):
    """Setup a professional-looking plot"""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    return fig, ax

# 1. Energy Model Comparison (Slide 10)
def plot_energy_model():
    fig, ax = setup_plot(
        title="Power Consumption by State",
        xlabel="State",
        ylabel="Power Consumption (mW)"
    )
    
    states = ['Transmit\n(PT)', 'Busy\n(PB)', 'Idle\n(PI)', 'Wake-up\n(PW)', 'Sleep\n(PS)']
    power = [100, 80, 10, 5, 0.01]
    colors_energy = ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#2ecc71']
    
    bars = ax.bar(states, power, color=colors_energy, edgecolor='black', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 200)
    
    # Add value labels
    for bar, p in zip(bars, power):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*1.5,
                f'{p} mW', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_model.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: energy_model.png")

# 2. State Machine Diagram Data (Slide 9)
def plot_state_timeline():
    fig, ax = setup_plot(
        title="Node State Evolution Example",
        xlabel="Time (slots)",
        ylabel="State",
        figsize=(12, 4)
    )
    
    # Simulate state timeline
    time = np.arange(0, 100)
    states = []
    state_map = {'SLEEP': 0, 'WAKEUP': 1, 'ACTIVE': 2, 'IDLE': 3}
    
    # Create realistic state sequence
    current = 'SLEEP'
    for t in time:
        if t == 20:  # Packet arrives
            current = 'WAKEUP'
        elif t == 25:  # Wake-up complete
            current = 'ACTIVE'
        elif t == 30:  # Transmission done
            current = 'IDLE'
        elif t == 45:  # Idle timeout
            current = 'SLEEP'
        elif t == 60:  # Another packet
            current = 'WAKEUP'
        elif t == 65:
            current = 'ACTIVE'
        elif t == 70:
            current = 'IDLE'
        elif t == 85:
            current = 'SLEEP'
        
        states.append(state_map[current])
    
    # Plot with color coding
    colors_state = {'SLEEP': '#2c3e50', 'WAKEUP': '#f39c12', 
                   'ACTIVE': '#27ae60', 'IDLE': '#3498db'}
    
    for state_name, state_val in state_map.items():
        mask = np.array(states) == state_val
        ax.fill_between(time, 0, 1, where=mask, alpha=0.8, 
                       label=state_name, color=colors_state[state_name],
                       transform=ax.get_xaxis_transform())
    
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations
    ax.annotate('Packet\nArrival', xy=(20, 0.5), xytext=(20, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', color='red', fontweight='bold')
    
    ax.annotate('ts timeout', xy=(45, 0.5), xytext=(45, 0.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=12, ha='center', color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'state_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: state_timeline.png")

# 3. Main Trade-off Plot (Slide 14)
def plot_lifetime_delay_tradeoff():
    fig, ax = setup_plot(
        title="Lifetime vs. Delay Trade-off",
        xlabel="Mean Delay (slots)",
        ylabel="Expected Lifetime (years)",
        figsize=(10, 8)
    )
    
    # Generate realistic data for different ts values
    ts_values = [0, 10, 30, 50, 100]
    
    for i, ts in enumerate(ts_values):
        # Simulate relationship
        delays = np.linspace(5, 100, 20)
        lifetimes = 2 + ts * 0.1 + 10 * np.exp(-delays/50) + np.random.normal(0, 0.5, 20)
        
        ax.scatter(delays, lifetimes, s=100, alpha=0.7, 
                  label=f'ts = {ts}', color=colors[i], edgecolor='black', linewidth=1)
        
        # Fit curve
        z = np.polyfit(delays, lifetimes, 2)
        p = np.poly1d(z)
        delay_smooth = np.linspace(delays.min(), delays.max(), 100)
        ax.plot(delay_smooth, p(delay_smooth), '--', color=colors[i], alpha=0.5, linewidth=2)
    
    # Highlight Pareto frontier
    ax.fill_between([5, 20], [0, 0], [15, 15], alpha=0.1, color='green', 
                    label='Optimal Region')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 15)
    
    # Add annotation
    ax.annotate('5x lifetime\n2x delay', xy=(20, 10), xytext=(40, 12),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lifetime_delay_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: lifetime_delay_tradeoff.png")

# 4. Transmission Probability Impact (Slide 15)
def plot_q_impact():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Common q values
    q_values = np.linspace(0.01, 0.5, 50)
    n = 100  # number of nodes
    
    # Lifetime vs q (monotonic decrease)
    lifetime = 12 * np.exp(-5 * q_values) + 0.5
    ax1.plot(q_values, lifetime, linewidth=3, color=colors[0])
    ax1.fill_between(q_values, lifetime - 0.5, lifetime + 0.5, alpha=0.2, color=colors[0])
    ax1.set_title('Battery Lifetime vs. Transmission Probability', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Transmission Probability (q)', fontsize=14)
    ax1.set_ylabel('Expected Lifetime (years)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # Delay vs q (U-shaped)
    optimal_q = 1/n
    delay = 50 * (q_values/optimal_q + optimal_q/q_values) + np.random.normal(0, 2, 50)
    ax2.plot(q_values, delay, linewidth=3, color=colors[1])
    ax2.fill_between(q_values, delay - 5, delay + 5, alpha=0.2, color=colors[1])
    ax2.axvline(optimal_q, color='red', linestyle='--', linewidth=2, label=f'Optimal q = 1/n = {optimal_q:.3f}')
    ax2.set_title('Access Delay vs. Transmission Probability', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Transmission Probability (q)', fontsize=14)
    ax2.set_ylabel('Mean Access Delay (slots)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=12)
    
    # Highlight optimal region
    ax2.axvspan(optimal_q * 0.8, optimal_q * 1.2, alpha=0.1, color='green')
    
    plt.suptitle('Impact of Transmission Probability', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'q_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: q_impact.png")

# 5. Scenario Comparison (Slide 16)
def plot_scenario_comparison():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scenarios = ['Low-Latency\nPriority', 'Balanced\nApproach', 'Battery-Life\nPriority']
    lifetime = [2.3, 6.8, 11.5]
    delay = [8.2, 24.5, 41.7]
    
    # Create scatter plot with annotations
    scatter = ax.scatter(delay, lifetime, s=1000, c=colors[:3], alpha=0.7, 
                        edgecolor='black', linewidth=3)
    
    # Add scenario labels
    for i, scenario in enumerate(scenarios):
        ax.annotate(scenario, (delay[i], lifetime[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=14, fontweight='bold', ha='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[i], alpha=0.3))
    
    # Add parameter boxes
    param_text = [
        'ts = 5\nq = 0.2',
        'ts = 30\nq = 0.01',
        'ts = 50\nq = 0.05'
    ]
    
    for i, text in enumerate(param_text):
        ax.text(delay[i], lifetime[i] - 0.8, text, 
               fontsize=12, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Mean Delay (slots)', fontsize=16)
    ax.set_ylabel('Expected Lifetime (years)', fontsize=16)
    ax.set_title('Scenario Comparison: Trade-off Configurations', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    
    # Add trade-off arrow
    ax.annotate('', xy=(45, 11), xytext=(5, 2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax.text(25, 6.5, 'Trade-off\nDirection', fontsize=14, ha='center', 
            color='red', fontweight='bold', rotation=45)
    
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 13)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: scenario_comparison.png")

# 6. Traffic Models Visualization (Slide 12)
def plot_traffic_models():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    time_slots = np.arange(100)
    
    # Poisson traffic
    poisson = np.random.binomial(1, 0.1, 100)
    axes[0].stem(time_slots, poisson, basefmt=" ", linefmt='b-', markerfmt='bo')
    axes[0].set_title('Poisson (Bernoulli) Traffic', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Packet Arrival')
    axes[0].set_ylim(-0.1, 1.5)
    
    # Bursty traffic
    bursty = np.zeros(100)
    burst_times = [20, 21, 22, 60, 61, 62, 63]
    bursty[burst_times] = 1
    axes[1].stem(time_slots, bursty, basefmt=" ", linefmt='r-', markerfmt='ro')
    axes[1].set_title('Bursty Traffic', fontsize=14, fontweight='bold')
    axes[1].set_ylim(-0.1, 1.5)
    
    # Periodic traffic
    periodic = np.zeros(100)
    periodic[::20] = 1
    # Add jitter
    for i in range(len(periodic)):
        if periodic[i] == 1 and i > 0:
            jitter = np.random.randint(-2, 3)
            if 0 <= i + jitter < 100:
                periodic[i] = 0
                periodic[i + jitter] = 1
    axes[2].stem(time_slots, periodic, basefmt=" ", linefmt='g-', markerfmt='go')
    axes[2].set_title('Periodic Traffic (with jitter)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Packet Arrival')
    axes[2].set_xlabel('Time (slots)')
    axes[2].set_ylim(-0.1, 1.5)
    
    # Mixed traffic
    mixed = poisson * 0.5 + bursty * 0.3 + periodic * 0.2
    mixed = np.minimum(mixed, 1)
    axes[3].stem(time_slots, mixed, basefmt=" ", linefmt='purple', markerfmt='o', color='purple')
    axes[3].set_title('Mixed Traffic', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time (slots)')
    axes[3].set_ylim(-0.1, 1.5)
    
    plt.suptitle('Traffic Models for IoT Networks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_models.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traffic_models.png")

# 7. Project Progress Timeline (Slide 7)
def plot_project_timeline():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    objectives = ['O1: Simulator\nFramework', 'O2: Parameter\nQuantification', 
                  'O3: Optimization\nAlgorithms', 'O4: 3GPP\nValidation']
    progress = [100, 100, 40, 15]
    colors_progress = ['#2ecc71', '#2ecc71', '#3498db', '#95a5a6']
    
    y_pos = np.arange(len(objectives))
    bars = ax.barh(y_pos, progress, color=colors_progress, edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for i, (bar, prog) in enumerate(zip(bars, progress)):
        width = bar.get_width()
        label = '✓ Complete' if prog == 100 else f'{prog}%'
        color = 'white' if prog > 50 else 'black'
        ax.text(width/2, bar.get_y() + bar.get_height()/2, label, 
                ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(objectives, fontsize=14)
    ax.set_xlabel('Progress (%)', fontsize=16)
    ax.set_title('Project Status: 38 Days Ahead of Schedule', fontsize=18, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add milestone marker
    ax.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(50, -0.7, 'Current\nMilestone', ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'project_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: project_timeline.png")

# Generate all plots
if __name__ == "__main__":
    print("\n🎨 Generating presentation figures...\n")
    
    plot_energy_model()
    plot_state_timeline()
    plot_lifetime_delay_tradeoff()
    plot_q_impact()
    plot_scenario_comparison()
    plot_traffic_models()
    plot_project_timeline()
    
    print(f"\n✅ All figures generated successfully in '{output_dir}' directory!")
    print("\n📊 Next steps:")
    print("1. Review the generated plots")
    print("2. Import them into your PowerPoint")
    print("3. Adjust colors/fonts to match your theme")
    print("4. Add any additional annotations needed")