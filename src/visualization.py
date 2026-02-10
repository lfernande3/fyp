"""
Visualization Utilities for M2M Sleep-Based Random Access Simulator

This module provides comprehensive plotting and interactive visualization
capabilities including:
- Trade-off plots (delay vs. lifetime)
- Parameter sweep visualization
- Interactive widgets
- Publication-quality figures

Date: February 10, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

from .simulator import SimulationResults, SimulationConfig


# Set publication-quality style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[float, float] = (10, 6)
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = 'husl'
    save_format: str = 'png'
    save_dpi: int = 300


class ResultsVisualizer:
    """
    Comprehensive visualization tools for simulation results.
    
    Provides methods for plotting trade-offs, parameter sweeps,
    time series, and comparative analyses.
    """
    
    def __init__(self, config: PlotConfig = PlotConfig()):
        """
        Initialize visualizer.
        
        Args:
            config: Plot configuration
        """
        self.config = config
        sns.set_palette(config.color_palette)
    
    def plot_delay_vs_lifetime(
        self,
        results_list: List[SimulationResults],
        param_name: str = "Parameter",
        param_values: Optional[List[Any]] = None,
        save_path: Optional[str] = None,
        show_percentiles: bool = True
    ):
        """
        Plot delay vs. lifetime trade-off curve.
        
        Classic trade-off visualization showing Pareto frontier.
        
        Args:
            results_list: List of simulation results
            param_name: Name of swept parameter
            param_values: Values of swept parameter
            save_path: Path to save figure
            show_percentiles: Show tail delay percentiles
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        delays = [r.mean_delay for r in results_list]
        lifetimes = [r.mean_lifetime_years for r in results_list]
        
        # Main scatter plot
        scatter = ax.scatter(delays, lifetimes, s=100, alpha=0.7,
                            c=range(len(results_list)), cmap='viridis')
        
        # Connect points to show progression
        ax.plot(delays, lifetimes, 'k--', alpha=0.3, linewidth=1)
        
        # Add colorbar if param values provided
        if param_values is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(param_name, fontsize=self.config.label_fontsize)
        
        # Labels and title
        ax.set_xlabel('Mean Delay (slots)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Mean Lifetime (years)', fontsize=self.config.label_fontsize)
        ax.set_title(f'Delay vs. Lifetime Trade-off\n({param_name} sweep)',
                    fontsize=self.config.title_fontsize)
        
        ax.grid(True, alpha=0.3)
        
        # Add 95th percentile if requested
        if show_percentiles:
            ax2 = ax.twiny()
            delays_95 = [r.tail_delay_95 for r in results_list]
            ax2.plot(delays_95, lifetimes, 'r:', alpha=0.5, linewidth=2)
            ax2.set_xlabel('95th Percentile Delay (slots)', 
                          fontsize=self.config.label_fontsize, color='r')
            ax2.tick_params(axis='x', labelcolor='r')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_impact(
        self,
        param_values: List[Any],
        results_list: List[SimulationResults],
        param_name: str,
        metrics: List[str] = ['mean_delay', 'mean_lifetime_years', 'throughput'],
        save_path: Optional[str] = None
    ):
        """
        Plot impact of parameter on multiple metrics.
        
        Args:
            param_values: Parameter values swept
            results_list: Corresponding results
            param_name: Parameter name for labeling
            metrics: List of metrics to plot
            save_path: Path to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.config.figsize[0], 
                                                        4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        metric_labels = {
            'mean_delay': 'Mean Delay (slots)',
            'tail_delay_95': '95th Percentile Delay (slots)',
            'tail_delay_99': '99th Percentile Delay (slots)',
            'mean_lifetime_years': 'Mean Lifetime (years)',
            'throughput': 'Throughput (packets/slot)',
            'empirical_success_prob': 'Success Probability',
            'mean_queue_length': 'Mean Queue Length',
            'total_collisions': 'Total Collisions'
        }
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [getattr(r, metric) for r in results_list]
            
            ax.plot(param_values, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(param_name, fontsize=self.config.label_fontsize)
            ax.set_ylabel(metric_labels.get(metric, metric),
                         fontsize=self.config.label_fontsize)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Impact of {param_name} on Performance Metrics',
                    fontsize=self.config.title_fontsize, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_state_fractions(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None
    ):
        """
        Plot state fraction pie chart.
        
        Args:
            results: Simulation results
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        states = list(results.state_fractions.keys())
        fractions = list(results.state_fractions.values())
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        explode = [0.05 if f == max(fractions) else 0 for f in fractions]
        
        ax.pie(fractions, labels=states, autopct='%1.1f%%',
              colors=colors[:len(states)], explode=explode,
              textprops={'fontsize': self.config.label_fontsize})
        
        ax.set_title('Node State Occupancy',
                    fontsize=self.config.title_fontsize)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_energy_breakdown(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None
    ):
        """
        Plot energy consumption breakdown by state.
        
        Args:
            results: Simulation results
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Energy fractions pie chart
        states = list(results.energy_fractions_by_state.keys())
        fractions = list(results.energy_fractions_by_state.values())
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f']
        ax1.pie(fractions, labels=states, autopct='%1.1f%%',
               colors=colors[:len(states)],
               textprops={'fontsize': self.config.label_fontsize})
        ax1.set_title('Energy Consumption by State',
                     fontsize=self.config.title_fontsize)
        
        # State time vs energy consumption bar chart
        state_times = [results.state_fractions.get(s, 0) for s in states]
        
        x = np.arange(len(states))
        width = 0.35
        
        ax2.bar(x - width/2, state_times, width, label='Time Fraction',
               alpha=0.7)
        ax2.bar(x + width/2, fractions, width, label='Energy Fraction',
               alpha=0.7)
        
        ax2.set_xlabel('State', fontsize=self.config.label_fontsize)
        ax2.set_ylabel('Fraction', fontsize=self.config.label_fontsize)
        ax2.set_title('Time vs. Energy by State',
                     fontsize=self.config.title_fontsize)
        ax2.set_xticks(x)
        ax2.set_xticklabels(states)
        ax2.legend(fontsize=self.config.legend_fontsize)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_time_series(
        self,
        results: SimulationResults,
        metrics: List[str] = ['queue_length', 'energy'],
        max_slots: int = 1000,
        save_path: Optional[str] = None
    ):
        """
        Plot time series evolution.
        
        Args:
            results: Simulation results
            metrics: Metrics to plot
            max_slots: Maximum slots to show
            save_path: Path to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.config.figsize[0],
                                                        4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        metric_data = {
            'queue_length': results.queue_length_history,
            'energy': results.energy_history
        }
        
        metric_labels = {
            'queue_length': 'Mean Queue Length',
            'energy': 'Mean Energy Remaining'
        }
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            data = metric_data.get(metric)
            
            if data is not None:
                slots = list(range(min(len(data), max_slots)))
                values = data[:max_slots]
                
                ax.plot(slots, values, linewidth=1.5, alpha=0.8)
                ax.set_xlabel('Slot', fontsize=self.config.label_fontsize)
                ax.set_ylabel(metric_labels.get(metric, metric),
                             fontsize=self.config.label_fontsize)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}',
                       ha='center', va='center',
                       transform=ax.transAxes)
        
        plt.suptitle('Time Series Evolution',
                    fontsize=self.config.title_fontsize, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison(
        self,
        results_dict: Dict[str, List[SimulationResults]],
        param_values: List[Any],
        param_name: str,
        metric: str = 'mean_delay',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of different scenarios.
        
        Args:
            results_dict: Dictionary mapping scenario names to results lists
            param_values: Parameter values
            param_name: Parameter name
            metric: Metric to compare
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        metric_labels = {
            'mean_delay': 'Mean Delay (slots)',
            'mean_lifetime_years': 'Mean Lifetime (years)',
            'throughput': 'Throughput (packets/slot)',
            'empirical_success_prob': 'Success Probability'
        }
        
        for scenario_name, results_list in results_dict.items():
            values = [getattr(r, metric) for r in results_list]
            ax.plot(param_values, values, 'o-', linewidth=2, 
                   markersize=8, label=scenario_name, alpha=0.7)
        
        ax.set_xlabel(param_name, fontsize=self.config.label_fontsize)
        ax.set_ylabel(metric_labels.get(metric, metric),
                     fontsize=self.config.label_fontsize)
        ax.set_title(f'{metric_labels.get(metric, metric)} vs. {param_name}\nScenario Comparison',
                    fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def plot_heatmap(
        self,
        x_values: List[Any],
        y_values: List[Any],
        z_values: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        cmap: str = 'YlOrRd',
        save_path: Optional[str] = None
    ):
        """
        Plot 2D parameter heatmap.
        
        Args:
            x_values: X-axis parameter values
            y_values: Y-axis parameter values
            z_values: 2D array of metric values
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            cmap: Colormap name
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(z_values, cmap=cmap, aspect='auto',
                      origin='lower', interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(range(len(x_values)))
        ax.set_yticks(range(len(y_values)))
        ax.set_xticklabels([f'{x:.3f}' for x in x_values])
        ax.set_yticklabels([f'{y:.1f}' for y in y_values])
        
        # Labels
        ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel(y_label, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()
    
    def create_summary_figure(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive 4-panel summary figure.
        
        Args:
            results: Simulation results
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: State fractions
        ax1 = fig.add_subplot(gs[0, 0])
        states = list(results.state_fractions.keys())
        fractions = list(results.state_fractions.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax1.pie(fractions, labels=states, autopct='%1.1f%%',
               colors=colors[:len(states)])
        ax1.set_title('State Occupancy', fontsize=self.config.title_fontsize)
        
        # Panel 2: Energy breakdown
        ax2 = fig.add_subplot(gs[0, 1])
        energy_states = list(results.energy_fractions_by_state.keys())
        energy_fractions = list(results.energy_fractions_by_state.values())
        ax2.pie(energy_fractions, labels=energy_states, autopct='%1.1f%%',
               colors=colors[:len(energy_states)])
        ax2.set_title('Energy Consumption', fontsize=self.config.title_fontsize)
        
        # Panel 3: Key metrics
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_text = (
            f"Performance Metrics:\n\n"
            f"Mean Delay: {results.mean_delay:.2f} slots\n"
            f"95th %ile Delay: {results.tail_delay_95:.2f} slots\n"
            f"99th %ile Delay: {results.tail_delay_99:.2f} slots\n"
            f"Throughput: {results.throughput:.4f} packets/slot\n"
            f"Success Prob: {results.empirical_success_prob:.4f}\n\n"
            f"Energy & Lifetime:\n\n"
            f"Mean Energy: {results.mean_energy_consumed:.2f} units\n"
            f"Mean Lifetime: {results.mean_lifetime_years:.6f} years\n"
            f"Mean Lifetime: {results.mean_lifetime_years * 365.25 * 24:.2f} hours\n\n"
            f"Network Stats:\n\n"
            f"Arrivals: {results.total_arrivals}\n"
            f"Deliveries: {results.total_deliveries}\n"
            f"Collisions: {results.total_collisions}"
        )
        ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')
        ax3.axis('off')
        
        # Panel 4: Configuration
        ax4 = fig.add_subplot(gs[1, 1])
        config_text = (
            f"Simulation Configuration:\n\n"
            f"Nodes (n): {results.config.n_nodes}\n"
            f"Arrival Rate (λ): {results.config.arrival_rate:.4f}\n"
            f"Transmission Prob (q): {results.config.transmission_prob:.3f}\n"
            f"Idle Timer (ts): {results.config.idle_timer} slots\n"
            f"Wakeup Time (tw): {results.config.wakeup_time} slots\n"
            f"Initial Energy: {results.config.initial_energy:.1f} units\n"
            f"Total Slots: {results.total_slots}\n"
            f"Seed: {results.config.seed}"
        )
        ax4.text(0.1, 0.95, config_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')
        ax4.axis('off')
        
        fig.suptitle('Simulation Summary', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi,
                       bbox_inches='tight')
        
        plt.show()


def create_interactive_widget(
    simulator_class,
    config_base: SimulationConfig
):
    """
    Create interactive ipywidgets interface for parameter exploration.
    
    Args:
        simulator_class: Simulator class to use
        config_base: Base configuration to modify
        
    Returns:
        Interactive widget
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return None
    
    # Create sliders
    q_slider = widgets.FloatSlider(
        value=0.1, min=0.01, max=0.5, step=0.01,
        description='q (trans prob):', continuous_update=False
    )
    
    ts_slider = widgets.IntSlider(
        value=10, min=0, max=100, step=5,
        description='ts (idle timer):', continuous_update=False
    )
    
    n_slider = widgets.IntSlider(
        value=10, min=5, max=100, step=5,
        description='n (nodes):', continuous_update=False
    )
    
    lambda_slider = widgets.FloatLogSlider(
        value=0.01, base=10, min=-3, max=-1, step=0.1,
        description='λ (arrival):', continuous_update=False
    )
    
    # Output widget
    output = widgets.Output()
    
    def update_plot(q, ts, n, lambda_rate):
        """Update plot with new parameters."""
        with output:
            output.clear_output(wait=True)
            
            # Create new config
            from copy import deepcopy
            config = deepcopy(config_base)
            config.transmission_prob = q
            config.idle_timer = ts
            config.n_nodes = n
            config.arrival_rate = lambda_rate
            config.max_slots = 5000  # Shorter for interactive response
            
            # Run simulation
            print(f"Running simulation with q={q:.3f}, ts={ts}, n={n}, λ={lambda_rate:.4f}...")
            sim = simulator_class(config)
            results = sim.run_simulation(verbose=False)
            
            # Plot results
            visualizer = ResultsVisualizer()
            visualizer.create_summary_figure(results)
    
    # Link widgets
    widgets.interact(
        update_plot,
        q=q_slider,
        ts=ts_slider,
        n=n_slider,
        lambda_rate=lambda_slider
    )
    
    return output
