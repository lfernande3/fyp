"""
Visualization Module for M2M Sleep-Based Simulator

This module provides comprehensive visualization utilities for simulation results,
including static plots, interactive widgets, and trade-off analysis.

Implements Task 2.3: Use matplotlib for post-sim plot selection in Jupyter with
interactive ipywidgets sliders.

Date: February 10, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import warnings

from .simulator import SimulationResults, SimulationConfig
from .metrics import MetricsCalculator, analyze_batch_results


@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    figsize: Tuple[int, int] = (10, 6)
    style: str = 'seaborn-v0_8-darkgrid'
    dpi: int = 100
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid: bool = True
    colormap: str = 'viridis'


class SimulationVisualizer:
    """
    Comprehensive visualization utilities for simulation results.
    
    Provides static plots for:
    - Lifetime vs. delay scatter plots
    - Lifetime vs. q curves
    - Delay vs. q curves
    - Queue evolution over time
    - Energy consumption breakdown (pie charts)
    - State occupation pie charts
    - Energy depletion curves
    - Trade-off curves
    """
    
    def __init__(self, plot_config: PlotConfig = None):
        """
        Initialize visualizer.
        
        Args:
            plot_config: PlotConfig for appearance settings
        """
        self.plot_config = plot_config or PlotConfig()
        
        # Try to set style
        try:
            plt.style.use(self.plot_config.style)
        except:
            # Fallback to default style
            pass
    
    def plot_lifetime_vs_delay_scatter(
        self,
        results_dict: Dict[Any, List[SimulationResults]],
        param_name: str = "ts",
        title: str = None,
        xlabel: str = "Mean Delay (slots)",
        ylabel: str = "Mean Lifetime (years)",
        ax: plt.Axes = None,
        show_legend: bool = True,
        **kwargs
    ) -> plt.Axes:
        """
        Create scatter plot of lifetime vs. delay for different parameter values.
        
        This is one of the key trade-off visualizations showing the Pareto frontier
        between latency and battery life.
        
        Args:
            results_dict: Dictionary mapping parameter values to lists of results
            param_name: Name of the parameter being varied (for legend)
            title: Plot title (auto-generated if None)
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Matplotlib axes (creates new if None)
            show_legend: Whether to show legend
            **kwargs: Additional arguments for plt.scatter
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize, 
                                   dpi=self.plot_config.dpi)
        
        # Plot each parameter value
        cmap = plt.colormaps.get_cmap(self.plot_config.colormap)
        colors = cmap(np.linspace(0, 1, len(results_dict)))
        
        for i, (param_val, results_list) in enumerate(sorted(results_dict.items())):
            delays = [r.mean_delay for r in results_list]
            lifetimes = [r.mean_lifetime_years for r in results_list]
            
            ax.scatter(delays, lifetimes, 
                      label=f"{param_name}={param_val}",
                      color=colors[i],
                      alpha=0.6,
                      s=80,
                      **kwargs)
        
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        
        if title is None:
            title = f"Lifetime vs. Delay Trade-off (varying {param_name})"
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        if show_legend:
            ax.legend(fontsize=self.plot_config.legend_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def plot_lifetime_vs_parameter(
        self,
        results_dict: Dict[Any, List[SimulationResults]],
        param_name: str = "q",
        title: str = None,
        xlabel: str = None,
        ylabel: str = "Mean Lifetime (years)",
        ax: plt.Axes = None,
        show_ci: bool = True,
        **kwargs
    ) -> plt.Axes:
        """
        Plot lifetime vs. parameter value with confidence intervals.
        
        Args:
            results_dict: Dictionary mapping parameter values to lists of results
            param_name: Name of the parameter
            title: Plot title
            xlabel: X-axis label (auto-generated if None)
            ylabel: Y-axis label
            ax: Matplotlib axes
            show_ci: Whether to show confidence intervals
            **kwargs: Additional arguments for plt.plot
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize,
                                   dpi=self.plot_config.dpi)
        
        # Extract data
        param_values = sorted(results_dict.keys())
        means = []
        stds = []
        
        for param_val in param_values:
            lifetimes = [r.mean_lifetime_years for r in results_dict[param_val]]
            means.append(np.mean(lifetimes))
            stds.append(np.std(lifetimes))
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot mean
        ax.plot(param_values, means, 'o-', linewidth=2, markersize=8, **kwargs)
        
        # Plot confidence interval
        if show_ci:
            ax.fill_between(param_values, means - 1.96 * stds, means + 1.96 * stds,
                           alpha=0.2)
        
        if xlabel is None:
            xlabel = f"Parameter: {param_name}"
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        
        if title is None:
            title = f"Lifetime vs. {param_name}"
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def plot_delay_vs_parameter(
        self,
        results_dict: Dict[Any, List[SimulationResults]],
        param_name: str = "q",
        delay_type: str = "mean",  # "mean", "tail_95", or "tail_99"
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        ax: plt.Axes = None,
        show_ci: bool = True,
        **kwargs
    ) -> plt.Axes:
        """
        Plot delay vs. parameter value with confidence intervals.
        
        Args:
            results_dict: Dictionary mapping parameter values to lists of results
            param_name: Name of the parameter
            delay_type: Type of delay ("mean", "tail_95", "tail_99")
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Matplotlib axes
            show_ci: Whether to show confidence intervals
            **kwargs: Additional arguments for plt.plot
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize,
                                   dpi=self.plot_config.dpi)
        
        # Extract data based on delay type
        param_values = sorted(results_dict.keys())
        means = []
        stds = []
        
        for param_val in param_values:
            if delay_type == "mean":
                delays = [r.mean_delay for r in results_dict[param_val]]
            elif delay_type == "tail_95":
                delays = [r.tail_delay_95 for r in results_dict[param_val]]
            elif delay_type == "tail_99":
                delays = [r.tail_delay_99 for r in results_dict[param_val]]
            else:
                raise ValueError(f"Unknown delay_type: {delay_type}")
            
            means.append(np.mean(delays))
            stds.append(np.std(delays))
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot mean
        ax.plot(param_values, means, 'o-', linewidth=2, markersize=8, **kwargs)
        
        # Plot confidence interval
        if show_ci:
            ax.fill_between(param_values, means - 1.96 * stds, means + 1.96 * stds,
                           alpha=0.2)
        
        if xlabel is None:
            xlabel = f"Parameter: {param_name}"
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        
        if ylabel is None:
            ylabel_map = {
                "mean": "Mean Delay (slots)",
                "tail_95": "95th Percentile Delay (slots)",
                "tail_99": "99th Percentile Delay (slots)"
            }
            ylabel = ylabel_map.get(delay_type, "Delay (slots)")
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        
        if title is None:
            title = f"{ylabel} vs. {param_name}"
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def plot_queue_evolution(
        self,
        result: SimulationResults,
        sample_rate: int = 1,
        title: str = "Queue Length Evolution Over Time",
        xlabel: str = "Time Slot",
        ylabel: str = "Mean Queue Length",
        ax: plt.Axes = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot queue length evolution over time.
        
        Args:
            result: Single simulation result with queue_length_history
            sample_rate: Sample every N slots for cleaner visualization
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Matplotlib axes
            **kwargs: Additional arguments for plt.plot
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize,
                                   dpi=self.plot_config.dpi)
        
        if result.queue_length_history is None:
            raise ValueError("No queue_length_history in result. "
                           "Enable time series tracking in simulation.")
        
        # Sample data
        history = result.queue_length_history[::sample_rate]
        time_slots = np.arange(0, len(result.queue_length_history), sample_rate)
        
        ax.plot(time_slots, history, linewidth=1.5, **kwargs)
        
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def plot_energy_breakdown_pie(
        self,
        result: SimulationResults,
        title: str = "Energy Consumption Breakdown by State",
        ax: plt.Axes = None,
        autopct: str = '%1.1f%%',
        **kwargs
    ) -> plt.Axes:
        """
        Plot energy consumption breakdown as pie chart.
        
        Args:
            result: Single simulation result
            title: Plot title
            ax: Matplotlib axes
            autopct: Format string for percentages
            **kwargs: Additional arguments for plt.pie
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.plot_config.dpi)
        
        # Extract energy fractions
        labels = []
        sizes = []
        for state, fraction in result.energy_fractions_by_state.items():
            if fraction > 0.001:  # Only show states with >0.1% energy
                labels.append(state.replace('_', ' ').title())
                sizes.append(fraction * 100)
        
        # Create pie chart
        cmap = plt.colormaps.get_cmap(self.plot_config.colormap)
        colors = cmap(np.linspace(0, 1, len(labels)))
        
        ax.pie(sizes, labels=labels, autopct=autopct, colors=colors,
              startangle=90, **kwargs)
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        plt.tight_layout()
        return ax
    
    def plot_state_occupation_pie(
        self,
        result: SimulationResults,
        title: str = "State Occupation Fractions",
        ax: plt.Axes = None,
        autopct: str = '%1.1f%%',
        **kwargs
    ) -> plt.Axes:
        """
        Plot state occupation fractions as pie chart.
        
        Args:
            result: Single simulation result
            title: Plot title
            ax: Matplotlib axes
            autopct: Format string for percentages
            **kwargs: Additional arguments for plt.pie
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.plot_config.dpi)
        
        # Extract state fractions
        labels = []
        sizes = []
        for state, fraction in result.state_fractions.items():
            if fraction > 0.001:  # Only show states with >0.1%
                labels.append(state.replace('_', ' ').title())
                sizes.append(fraction * 100)
        
        # Create pie chart
        cmap = plt.colormaps.get_cmap(self.plot_config.colormap)
        colors = cmap(np.linspace(0, 1, len(labels)))
        
        ax.pie(sizes, labels=labels, autopct=autopct, colors=colors,
              startangle=90, **kwargs)
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        plt.tight_layout()
        return ax
    
    def plot_energy_depletion(
        self,
        result: SimulationResults,
        sample_rate: int = 1,
        title: str = "Energy Depletion Over Time",
        xlabel: str = "Time Slot",
        ylabel: str = "Mean Remaining Energy",
        ax: plt.Axes = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot energy depletion over time.
        
        Args:
            result: Single simulation result with energy_history
            sample_rate: Sample every N slots
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Matplotlib axes
            **kwargs: Additional arguments for plt.plot
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize,
                                   dpi=self.plot_config.dpi)
        
        if result.energy_history is None:
            raise ValueError("No energy_history in result. "
                           "Enable time series tracking in simulation.")
        
        # Sample data
        history = result.energy_history[::sample_rate]
        time_slots = np.arange(0, len(result.energy_history), sample_rate)
        
        ax.plot(time_slots, history, linewidth=1.5, **kwargs)
        
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return ax
    
    def plot_tradeoff_comparison(
        self,
        scenario_results: Dict[str, List[SimulationResults]],
        metric_x: str = "mean_delay",
        metric_y: str = "mean_lifetime_years",
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        ax: plt.Axes = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot trade-off comparison between different scenarios.
        
        Useful for comparing low-latency vs. battery-life prioritization.
        
        Args:
            scenario_results: Dict mapping scenario names to result lists
            metric_x: Metric for x-axis
            metric_y: Metric for y-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Matplotlib axes
            **kwargs: Additional arguments for plt.scatter
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize,
                                   dpi=self.plot_config.dpi)
        
        # Plot each scenario
        cmap = plt.colormaps.get_cmap(self.plot_config.colormap)
        colors = cmap(np.linspace(0, 1, len(scenario_results)))
        
        for i, (scenario_name, results_list) in enumerate(scenario_results.items()):
            x_values = [getattr(r, metric_x) for r in results_list]
            y_values = [getattr(r, metric_y) for r in results_list]
            
            ax.scatter(x_values, y_values,
                      label=scenario_name,
                      color=colors[i],
                      alpha=0.6,
                      s=100,
                      **kwargs)
        
        if xlabel is None:
            xlabel = metric_x.replace('_', ' ').title()
        if ylabel is None:
            ylabel = metric_y.replace('_', ' ').title()
        
        ax.set_xlabel(xlabel, fontsize=self.plot_config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.plot_config.label_fontsize)
        
        if title is None:
            title = "Trade-off Comparison Across Scenarios"
        ax.set_title(title, fontsize=self.plot_config.title_fontsize)
        
        ax.legend(fontsize=self.plot_config.legend_fontsize)
        
        if self.plot_config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def create_summary_dashboard(
        self,
        result: SimulationResults,
        include_time_series: bool = True,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            result: Single simulation result
            include_time_series: Whether to include time series plots
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        if include_time_series and (result.queue_length_history is None or 
                                    result.energy_history is None):
            warnings.warn("Time series data not available, setting include_time_series=False")
            include_time_series = False
        
        if include_time_series:
            fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=self.plot_config.dpi)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=self.plot_config.dpi)
            axes = axes.flatten()
        
        # Plot 1: Energy breakdown pie
        self.plot_energy_breakdown_pie(result, ax=axes[0])
        
        # Plot 2: State occupation pie
        self.plot_state_occupation_pie(result, ax=axes[1])
        
        if include_time_series:
            # Plot 3: Queue evolution
            self.plot_queue_evolution(result, ax=axes[2], sample_rate=100)
            
            # Plot 4: Energy depletion
            self.plot_energy_depletion(result, ax=axes[3], sample_rate=100)
            
            # Plot 5 & 6: Summary statistics (text)
            axes[4].axis('off')
            axes[5].axis('off')
            
            # Add summary text
            summary_text = self._create_summary_text(result)
            axes[4].text(0.1, 0.9, summary_text, transform=axes[4].transAxes,
                        fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        return fig
    
    def _create_summary_text(self, result: SimulationResults) -> str:
        """Create summary text for dashboard."""
        text = f"""Simulation Summary
{'='*40}
Configuration:
  Nodes: {result.config.n_nodes}
  Arrival rate (λ): {result.config.arrival_rate:.4f}
  Transmission prob (q): {result.config.transmission_prob:.4f}
  Idle timer (ts): {result.config.idle_timer}
  Wake-up time (tw): {result.config.wakeup_time}
  
Performance:
  Mean lifetime: {result.mean_lifetime_years:.2f} years
  Mean delay: {result.mean_delay:.2f} slots
  Tail delay (95%): {result.tail_delay_95:.2f} slots
  Throughput: {result.throughput:.4f}
  
Network:
  Total arrivals: {result.total_arrivals}
  Total deliveries: {result.total_deliveries}
  Total collisions: {result.total_collisions}
  Success prob: {result.empirical_success_prob:.4f}
"""
        return text


class InteractiveVisualizer:
    """
    Interactive visualization utilities using ipywidgets.
    
    Provides interactive parameter exploration with sliders for q, ts, n.
    """
    
    def __init__(self, base_config: SimulationConfig):
        """
        Initialize interactive visualizer.
        
        Args:
            base_config: Base simulation configuration
        """
        self.base_config = base_config
        self.visualizer = SimulationVisualizer()
    
    def create_interactive_explorer(
        self,
        q_range: Tuple[float, float] = (0.01, 0.5),
        ts_range: Tuple[int, int] = (1, 100),
        n_range: Tuple[int, int] = (10, 200),
        metrics_to_plot: List[str] = None
    ):
        """
        Create interactive parameter explorer with sliders.
        
        Requires ipywidgets and must be run in Jupyter notebook.
        
        Args:
            q_range: Range for transmission probability slider
            ts_range: Range for idle timer slider
            n_range: Range for number of nodes slider
            metrics_to_plot: List of metrics to display
            
        Returns:
            Interactive widget
        """
        try:
            from ipywidgets import interact, FloatSlider, IntSlider
            from IPython.display import display
        except ImportError:
            raise ImportError("ipywidgets required for interactive visualization. "
                            "Install with: pip install ipywidgets")
        
        if metrics_to_plot is None:
            metrics_to_plot = ['mean_lifetime_years', 'mean_delay', 'throughput']
        
        def update_plot(q, ts, n):
            """Update plot based on slider values."""
            from .simulator import Simulator
            
            # Create config with new parameters
            config = SimulationConfig(
                n_nodes=n,
                arrival_rate=self.base_config.arrival_rate,
                transmission_prob=q,
                idle_timer=ts,
                wakeup_time=self.base_config.wakeup_time,
                initial_energy=self.base_config.initial_energy,
                power_rates=self.base_config.power_rates,
                max_slots=self.base_config.max_slots,
                seed=42
            )
            
            # Run simulation
            sim = Simulator(config)
            result = sim.run_simulation(track_time_series=False)
            
            # Display results
            print(f"\nSimulation Results:")
            print(f"{'='*50}")
            for metric in metrics_to_plot:
                value = getattr(result, metric)
                print(f"{metric}: {value:.4f}")
            print(f"{'='*50}")
            
            # Create simple visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Energy breakdown
            self.visualizer.plot_energy_breakdown_pie(result, ax=axes[0])
            
            # Plot 2: State occupation
            self.visualizer.plot_state_occupation_pie(result, ax=axes[1])
            
            plt.show()
        
        # Create sliders
        q_slider = FloatSlider(
            value=(q_range[0] + q_range[1]) / 2,
            min=q_range[0],
            max=q_range[1],
            step=0.01,
            description='q:',
            continuous_update=False
        )
        
        ts_slider = IntSlider(
            value=(ts_range[0] + ts_range[1]) // 2,
            min=ts_range[0],
            max=ts_range[1],
            step=1,
            description='ts:',
            continuous_update=False
        )
        
        n_slider = IntSlider(
            value=(n_range[0] + n_range[1]) // 2,
            min=n_range[0],
            max=n_range[1],
            step=10,
            description='n:',
            continuous_update=False
        )
        
        # Create interactive widget
        return interact(update_plot, q=q_slider, ts=ts_slider, n=n_slider)


def plot_parameter_sweep_summary(
    results_dict: Dict[Any, List[SimulationResults]],
    param_name: str,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comprehensive summary plot for a parameter sweep.
    
    Args:
        results_dict: Dictionary mapping parameter values to result lists
        param_name: Name of the parameter
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    visualizer = SimulationVisualizer()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Lifetime vs. parameter
    visualizer.plot_lifetime_vs_parameter(results_dict, param_name, ax=axes[0])
    
    # Plot 2: Mean delay vs. parameter
    visualizer.plot_delay_vs_parameter(results_dict, param_name, 
                                       delay_type="mean", ax=axes[1])
    
    # Plot 3: Tail delay (95%) vs. parameter
    visualizer.plot_delay_vs_parameter(results_dict, param_name,
                                       delay_type="tail_95", ax=axes[2])
    
    # Plot 4: Lifetime vs. delay scatter
    visualizer.plot_lifetime_vs_delay_scatter(results_dict, param_name, ax=axes[3])
    
    # Plot 5: Throughput vs. parameter
    param_values = sorted(results_dict.keys())
    throughputs = [np.mean([r.throughput for r in results_dict[p]]) 
                   for p in param_values]
    axes[4].plot(param_values, throughputs, 'o-', linewidth=2, markersize=8)
    axes[4].set_xlabel(f"Parameter: {param_name}")
    axes[4].set_ylabel("Throughput")
    axes[4].set_title(f"Throughput vs. {param_name}")
    axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Energy per packet vs. parameter
    energy_per_packet = []
    for p in param_values:
        results = results_dict[p]
        epp = [r.mean_energy_consumed / max(r.total_deliveries, 1) for r in results]
        energy_per_packet.append(np.mean(epp))
    
    axes[5].plot(param_values, energy_per_packet, 'o-', linewidth=2, markersize=8)
    axes[5].set_xlabel(f"Parameter: {param_name}")
    axes[5].set_ylabel("Energy per Packet")
    axes[5].set_title(f"Energy Efficiency vs. {param_name}")
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, formats: List[str] = None, dpi: int = 300):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        formats: List of formats to save (default: ['png', 'pdf'])
        dpi: DPI for raster formats
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    for fmt in formats:
        filepath = f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
