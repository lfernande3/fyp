"""
Visualization module for On-Demand Sleep-Based Aloha Simulator

Provides plotting functions for:
- Lifetime vs delay tradeoff curves
- Energy consumption over time
- State transitions
- Performance metrics comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_metrics_summary(metrics: Dict, save_path: Optional[str] = None):
    """
    Create a comprehensive summary plot of simulation metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('On-Demand Sleep-Based Aloha - Simulation Summary', 
                 fontsize=16, fontweight='bold')
    
    # 1. Lifetime statistics
    ax = axes[0, 0]
    lifetime_data = [metrics['min_lifetime'], metrics['avg_lifetime'], metrics['max_lifetime']]
    ax.bar(['Min', 'Average', 'Max'], lifetime_data, color=['#d62728', '#2ca02c', '#1f77b4'])
    ax.set_ylabel('Lifetime (slots)')
    ax.set_title('Node Lifetime Statistics')
    ax.grid(True, alpha=0.3)
    
    # 2. Delay metrics
    ax = axes[0, 1]
    ax.bar(['Avg Delay', 'Max Delay'], 
           [metrics['avg_delay'], metrics['max_delay']], 
           color=['#ff7f0e', '#d62728'])
    ax.set_ylabel('Delay (slots)')
    ax.set_title('Packet Delay Statistics')
    ax.grid(True, alpha=0.3)
    
    # 3. Throughput
    ax = axes[0, 2]
    ax.bar(['Total', 'Per Node'], 
           [metrics['total_throughput'], metrics['avg_node_throughput']], 
           color=['#9467bd', '#8c564b'])
    ax.set_ylabel('Throughput (packets/slot)')
    ax.set_title('Network Throughput')
    ax.grid(True, alpha=0.3)
    
    # 4. Transmission statistics (pie chart)
    ax = axes[1, 0]
    successful = metrics['total_packets_sent']
    collided = metrics['total_collisions']
    ax.pie([successful, collided], 
           labels=['Successful', 'Collisions'],
           autopct='%1.1f%%',
           colors=['#2ca02c', '#d62728'],
           startangle=90)
    ax.set_title('Transmission Outcomes')
    
    # 5. Energy consumption
    ax = axes[1, 1]
    consumed = metrics['energy_consumed_ratio'] * 100
    remaining = 100 - consumed
    ax.pie([consumed, remaining],
           labels=['Consumed', 'Remaining'],
           autopct='%1.1f%%',
           colors=['#ff7f0e', '#1f77b4'],
           startangle=90)
    ax.set_title('Energy Status')
    
    # 6. Packet delivery
    ax = axes[1, 2]
    delivered = metrics['packet_delivery_ratio'] * 100
    buffered = 100 - delivered
    bars = ax.bar(['Delivered', 'In Buffer'], [delivered, buffered],
                   color=['#2ca02c', '#bcbd22'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Packet Delivery Ratio')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
    
    plt.show()


def plot_tradeoff_curve(results: List[Dict], x_param: str, y_param: str,
                        color_param: Optional[str] = None,
                        save_path: Optional[str] = None):
    """
    Plot tradeoff curves between two parameters (e.g., lifetime vs delay).
    
    Args:
        results: List of metric dictionaries from multiple simulation runs
        x_param: Parameter name for x-axis (e.g., 'avg_delay')
        y_param: Parameter name for y-axis (e.g., 'avg_lifetime')
        color_param: Parameter to use for color coding (e.g., 'q_transmit')
    """
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color_param and color_param in df.columns:
        # Color by parameter
        scatter = ax.scatter(df[x_param], df[y_param], 
                           c=df[color_param], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=1)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_param)
        
        # Also draw lines connecting points with same color_param
        for val in df[color_param].unique():
            subset = df[df[color_param] == val].sort_values(x_param)
            ax.plot(subset[x_param], subset[y_param], '--', alpha=0.3)
    else:
        ax.scatter(df[x_param], df[y_param], s=100, alpha=0.6, 
                  edgecolors='black', linewidth=1)
        ax.plot(df[x_param], df[y_param], '--', alpha=0.3)
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_title(f'Tradeoff: {y_param} vs {x_param}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved tradeoff plot to {save_path}")
    
    plt.show()


def plot_energy_timeline(node, save_path: Optional[str] = None):
    """
    Plot energy consumption over time for a single node.
    """
    if not node.energy_history:
        print("No energy history available")
        return
    
    times, energies = zip(*node.energy_history)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, energies, linewidth=1.5, color='#1f77b4')
    ax.fill_between(times, 0, energies, alpha=0.3, color='#1f77b4')
    
    ax.set_xlabel('Time (slots)')
    ax.set_ylabel('Remaining Energy')
    ax.set_title(f'Energy Consumption Timeline - {node.name}')
    ax.grid(True, alpha=0.3)
    
    # Mark lifetime if depleted
    if node.lifetime:
        ax.axvline(node.lifetime, color='red', linestyle='--', 
                  label=f'Lifetime: {node.lifetime:.0f} slots')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy timeline to {save_path}")
    
    plt.show()


def plot_state_timeline(node, max_slots: Optional[int] = None, 
                        save_path: Optional[str] = None):
    """
    Plot state transitions over time for a single node.
    """
    if not node.state_history:
        print("No state history available")
        return
    
    # Convert state history to plottable format
    times, states = zip(*node.state_history)
    
    if max_slots:
        # Limit to first max_slots
        times = [t for t in times if t <= max_slots]
        states = states[:len(times)]
    
    # Map states to numeric values
    state_map = {'sleep': 0, 'wakeup': 1, 'idle': 2, 'active': 3, 'dead': -1}
    state_values = [state_map.get(s, 0) for s in states]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create step plot
    ax.step(times, state_values, where='post', linewidth=1.5)
    ax.fill_between(times, state_values, step='post', alpha=0.3)
    
    ax.set_xlabel('Time (slots)')
    ax.set_ylabel('State')
    ax.set_yticks(list(state_map.values()))
    ax.set_yticklabels(list(state_map.keys()))
    ax.set_title(f'State Timeline - {node.name}')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved state timeline to {save_path}")
    
    plt.show()


def plot_parameter_sweep(results: List[Dict], sweep_param: str,
                        metrics_to_plot: List[str] = None,
                        save_path: Optional[str] = None):
    """
    Plot multiple metrics as a function of a swept parameter.
    
    Args:
        results: List of metric dictionaries
        sweep_param: Parameter that was varied (e.g., 'q_transmit', 'ts_idle')
        metrics_to_plot: List of metric names to plot
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['avg_lifetime', 'avg_delay', 'total_throughput', 'collision_rate']
    
    df = pd.DataFrame(results)
    df = df.sort_values(sweep_param)
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle(f'Parameter Sweep: {sweep_param}', fontsize=14, fontweight='bold')
    
    colors = sns.color_palette("husl", n_metrics)
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.plot(df[sweep_param], df[metric], marker='o', 
               linewidth=2, markersize=8, color=colors[idx])
        ax.set_xlabel(sweep_param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal point (depends on metric)
        if 'lifetime' in metric or 'throughput' in metric:
            # Higher is better
            best_idx = df[metric].idxmax()
        else:
            # Lower is better (delay, collision rate)
            best_idx = df[metric].idxmin()
        
        best_x = df.loc[best_idx, sweep_param]
        best_y = df.loc[best_idx, metric]
        ax.plot(best_x, best_y, 'r*', markersize=15, 
               label=f'Optimal: {best_x:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter sweep plot to {save_path}")
    
    plt.show()


def plot_node_comparison(node_metrics: List[Dict], save_path: Optional[str] = None):
    """
    Compare performance across different nodes.
    """
    df = pd.DataFrame(node_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Node Performance Comparison', fontsize=14, fontweight='bold')
    
    # 1. Lifetime comparison
    ax = axes[0, 0]
    ax.bar(df['name'], df['lifetime'], color='steelblue', alpha=0.7)
    ax.set_ylabel('Lifetime (slots)')
    ax.set_title('Node Lifetimes')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Packets sent comparison
    ax = axes[0, 1]
    ax.bar(df['name'], df['packets_sent'], color='seagreen', alpha=0.7)
    ax.set_ylabel('Packets Sent')
    ax.set_title('Successful Transmissions per Node')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Average delay comparison
    ax = axes[1, 0]
    ax.bar(df['name'], df['avg_delay'], color='orange', alpha=0.7)
    ax.set_ylabel('Average Delay (slots)')
    ax.set_title('Average Packet Delay per Node')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Energy remaining
    ax = axes[1, 1]
    ax.bar(df['name'], df['energy_remaining'], color='crimson', alpha=0.7)
    ax.set_ylabel('Energy Remaining')
    ax.set_title('Remaining Energy per Node')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved node comparison plot to {save_path}")
    
    plt.show()


def create_results_table(results: List[Dict], 
                        columns: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a formatted table of results.
    
    Args:
        results: List of metric dictionaries
        columns: Columns to include (None = all)
        save_path: Path to save CSV file
    
    Returns:
        DataFrame with results
    """
    df = pd.DataFrame(results)
    
    if columns:
        df = df[columns]
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved results table to {save_path}")
    
    return df


def plot_3d_tradeoff(results: List[Dict], x_param: str, y_param: str, z_param: str,
                     save_path: Optional[str] = None):
    """
    Create a 3D plot showing tradeoffs between three parameters.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    df = pd.DataFrame(results)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df[x_param], df[y_param], df[z_param],
                        c=df[z_param], cmap='viridis', s=100,
                        alpha=0.6, edgecolors='black', linewidth=1)
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_zlabel(z_param.replace('_', ' ').title())
    ax.set_title(f'3D Tradeoff Surface')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(z_param.replace('_', ' ').title())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D tradeoff plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example: Generate sample data and create plots
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    # Create sample metrics
    sample_metrics = {
        'avg_lifetime': 4523.5,
        'min_lifetime': 4200.0,
        'max_lifetime': 4800.0,
        'std_lifetime': 150.2,
        'avg_delay': 45.3,
        'max_delay': 120.5,
        'total_throughput': 0.0892,
        'avg_node_throughput': 0.00892,
        'total_packets_sent': 446,
        'total_packets_arrived': 500,
        'packet_delivery_ratio': 0.892,
        'total_transmission_attempts': 650,
        'total_collisions': 204,
        'collision_rate': 0.314,
        'avg_energy_remaining': 1250.5,
        'energy_consumed_ratio': 0.875,
    }
    
    print("\nGenerating sample summary plot...")
    plot_metrics_summary(sample_metrics)
    
    print("\nVisualization examples complete!")
