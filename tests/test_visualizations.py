"""
Tests for Visualization Module

Date: February 10, 2026
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from typing import Dict, List

from src.visualizations import (
    SimulationVisualizer, InteractiveVisualizer, PlotConfig,
    plot_parameter_sweep_summary, save_figure
)
from src.simulator import Simulator, SimulationConfig, SimulationResults
from src.power_model import PowerModel, PowerProfile


# Fixtures
@pytest.fixture
def sample_config():
    """Create sample simulation configuration."""
    power_rates = PowerModel.get_profile(PowerProfile.NR_MMTC)
    return SimulationConfig(
        n_nodes=10,
        arrival_rate=0.01,
        transmission_prob=0.1,
        idle_timer=10,
        wakeup_time=2,
        initial_energy=1000,
        power_rates=power_rates,
        max_slots=1000,
        seed=42
    )


@pytest.fixture
def sample_result(sample_config):
    """Create sample simulation result."""
    sim = Simulator(sample_config)
    return sim.run_simulation(track_history=True)


@pytest.fixture
def sample_results_dict(sample_config):
    """Create sample results dictionary for parameter sweep."""
    results_dict = {}
    
    for q in [0.05, 0.1, 0.2]:
        results_list = []
        for seed in range(3):
            config = SimulationConfig(
                n_nodes=sample_config.n_nodes,
                arrival_rate=sample_config.arrival_rate,
                transmission_prob=q,
                idle_timer=sample_config.idle_timer,
                wakeup_time=sample_config.wakeup_time,
                initial_energy=sample_config.initial_energy,
                power_rates=sample_config.power_rates,
                max_slots=1000,
                seed=seed
            )
            sim = Simulator(config)
            result = sim.run_simulation(track_history=True)
            results_list.append(result)
        
        results_dict[q] = results_list
    
    return results_dict


@pytest.fixture
def visualizer():
    """Create visualizer instance."""
    return SimulationVisualizer()


# Test PlotConfig
def test_plot_config_defaults():
    """Test PlotConfig default values."""
    config = PlotConfig()
    assert config.figsize == (10, 6)
    assert config.dpi == 100
    assert config.grid is True
    assert config.colormap == 'viridis'


def test_plot_config_custom():
    """Test PlotConfig with custom values."""
    config = PlotConfig(
        figsize=(12, 8),
        dpi=150,
        grid=False,
        colormap='plasma'
    )
    assert config.figsize == (12, 8)
    assert config.dpi == 150
    assert config.grid is False
    assert config.colormap == 'plasma'


# Test SimulationVisualizer initialization
def test_visualizer_init():
    """Test visualizer initialization."""
    viz = SimulationVisualizer()
    assert viz.plot_config is not None
    assert isinstance(viz.plot_config, PlotConfig)


def test_visualizer_init_custom_config():
    """Test visualizer with custom config."""
    config = PlotConfig(dpi=200)
    viz = SimulationVisualizer(config)
    assert viz.plot_config.dpi == 200


# Test lifetime vs delay scatter plot
def test_plot_lifetime_vs_delay_scatter(visualizer, sample_results_dict):
    """Test lifetime vs delay scatter plot."""
    ax = visualizer.plot_lifetime_vs_delay_scatter(
        sample_results_dict,
        param_name="q"
    )
    
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Mean Delay (slots)"
    assert ax.get_ylabel() == "Mean Lifetime (years)"
    plt.close('all')


def test_plot_lifetime_vs_delay_scatter_custom_labels(visualizer, sample_results_dict):
    """Test scatter plot with custom labels."""
    ax = visualizer.plot_lifetime_vs_delay_scatter(
        sample_results_dict,
        param_name="q",
        xlabel="Custom X",
        ylabel="Custom Y",
        title="Custom Title"
    )
    
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    assert ax.get_title() == "Custom Title"
    plt.close('all')


# Test lifetime vs parameter plot
def test_plot_lifetime_vs_parameter(visualizer, sample_results_dict):
    """Test lifetime vs parameter plot."""
    ax = visualizer.plot_lifetime_vs_parameter(
        sample_results_dict,
        param_name="q"
    )
    
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert "Lifetime" in ax.get_ylabel()
    plt.close('all')


def test_plot_lifetime_vs_parameter_no_ci(visualizer, sample_results_dict):
    """Test lifetime plot without confidence intervals."""
    ax = visualizer.plot_lifetime_vs_parameter(
        sample_results_dict,
        param_name="q",
        show_ci=False
    )
    
    assert ax is not None
    plt.close('all')


# Test delay vs parameter plot
def test_plot_delay_vs_parameter_mean(visualizer, sample_results_dict):
    """Test mean delay vs parameter plot."""
    ax = visualizer.plot_delay_vs_parameter(
        sample_results_dict,
        param_name="q",
        delay_type="mean"
    )
    
    assert ax is not None
    assert "Mean Delay" in ax.get_ylabel()
    plt.close('all')


def test_plot_delay_vs_parameter_tail_95(visualizer, sample_results_dict):
    """Test 95th percentile delay plot."""
    ax = visualizer.plot_delay_vs_parameter(
        sample_results_dict,
        param_name="q",
        delay_type="tail_95"
    )
    
    assert ax is not None
    assert "95th" in ax.get_ylabel()
    plt.close('all')


def test_plot_delay_vs_parameter_tail_99(visualizer, sample_results_dict):
    """Test 99th percentile delay plot."""
    ax = visualizer.plot_delay_vs_parameter(
        sample_results_dict,
        param_name="q",
        delay_type="tail_99"
    )
    
    assert ax is not None
    assert "99th" in ax.get_ylabel()
    plt.close('all')


def test_plot_delay_vs_parameter_invalid_type(visualizer, sample_results_dict):
    """Test delay plot with invalid delay type."""
    with pytest.raises(ValueError, match="Unknown delay_type"):
        visualizer.plot_delay_vs_parameter(
            sample_results_dict,
            param_name="q",
            delay_type="invalid"
        )


# Test queue evolution plot
def test_plot_queue_evolution(visualizer, sample_result):
    """Test queue evolution plot."""
    ax = visualizer.plot_queue_evolution(sample_result)
    
    assert ax is not None
    assert "Queue" in ax.get_ylabel()
    plt.close('all')


def test_plot_queue_evolution_sampled(visualizer, sample_result):
    """Test queue evolution with sampling."""
    ax = visualizer.plot_queue_evolution(sample_result, sample_rate=10)
    
    assert ax is not None
    plt.close('all')


def test_plot_queue_evolution_no_history(visualizer, sample_config):
    """Test queue evolution without time series data."""
    sim = Simulator(sample_config)
    result = sim.run_simulation(track_history=False)
    
    with pytest.raises(ValueError, match="No queue_length_history"):
        visualizer.plot_queue_evolution(result)


# Test energy breakdown pie chart
def test_plot_energy_breakdown_pie(visualizer, sample_result):
    """Test energy breakdown pie chart."""
    ax = visualizer.plot_energy_breakdown_pie(sample_result)
    
    assert ax is not None
    plt.close('all')


def test_plot_energy_breakdown_pie_custom_format(visualizer, sample_result):
    """Test energy pie with custom format."""
    ax = visualizer.plot_energy_breakdown_pie(
        sample_result,
        autopct='%1.2f%%',
        title="Custom Energy Title"
    )
    
    assert ax is not None
    assert ax.get_title() == "Custom Energy Title"
    plt.close('all')


# Test state occupation pie chart
def test_plot_state_occupation_pie(visualizer, sample_result):
    """Test state occupation pie chart."""
    ax = visualizer.plot_state_occupation_pie(sample_result)
    
    assert ax is not None
    plt.close('all')


# Test energy depletion plot
def test_plot_energy_depletion(visualizer, sample_result):
    """Test energy depletion plot."""
    ax = visualizer.plot_energy_depletion(sample_result)
    
    assert ax is not None
    assert "Energy" in ax.get_ylabel()
    plt.close('all')


def test_plot_energy_depletion_sampled(visualizer, sample_result):
    """Test energy depletion with sampling."""
    ax = visualizer.plot_energy_depletion(sample_result, sample_rate=10)
    
    assert ax is not None
    plt.close('all')


def test_plot_energy_depletion_no_history(visualizer, sample_config):
    """Test energy depletion without time series data."""
    sim = Simulator(sample_config)
    result = sim.run_simulation(track_history=False)
    
    with pytest.raises(ValueError, match="No energy_history"):
        visualizer.plot_energy_depletion(result)


# Test trade-off comparison plot
def test_plot_tradeoff_comparison(visualizer, sample_results_dict):
    """Test trade-off comparison plot."""
    scenario_results = {
        "Low Latency": sample_results_dict[0.2],
        "Battery Life": sample_results_dict[0.05]
    }
    
    ax = visualizer.plot_tradeoff_comparison(scenario_results)
    
    assert ax is not None
    assert len(ax.get_legend().get_texts()) == 2
    plt.close('all')


def test_plot_tradeoff_comparison_custom_metrics(visualizer, sample_results_dict):
    """Test trade-off plot with custom metrics."""
    scenario_results = {
        "Scenario A": sample_results_dict[0.1],
        "Scenario B": sample_results_dict[0.2]
    }
    
    ax = visualizer.plot_tradeoff_comparison(
        scenario_results,
        metric_x="throughput",
        metric_y="mean_delay",
        xlabel="Custom X",
        ylabel="Custom Y"
    )
    
    assert ax is not None
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    plt.close('all')


# Test summary dashboard
def test_create_summary_dashboard_with_timeseries(visualizer, sample_result):
    """Test summary dashboard with time series."""
    fig = visualizer.create_summary_dashboard(sample_result, include_time_series=True)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6
    plt.close('all')


def test_create_summary_dashboard_without_timeseries(visualizer, sample_result):
    """Test summary dashboard without time series."""
    fig = visualizer.create_summary_dashboard(sample_result, include_time_series=False)
    
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close('all')


def test_create_summary_dashboard_no_history(visualizer, sample_config):
    """Test dashboard with no time series data."""
    sim = Simulator(sample_config)
    result = sim.run_simulation(track_history=False)
    
    # Should warn and create dashboard without time series
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = visualizer.create_summary_dashboard(result, include_time_series=True)
    
    assert fig is not None
    plt.close('all')


# Test summary text creation
def test_create_summary_text(visualizer, sample_result):
    """Test summary text creation."""
    text = visualizer._create_summary_text(sample_result)
    
    assert "Simulation Summary" in text
    assert "Configuration:" in text
    assert "Performance:" in text
    assert "Network:" in text
    assert str(sample_result.config.n_nodes) in text


# Test InteractiveVisualizer
def test_interactive_visualizer_init(sample_config):
    """Test interactive visualizer initialization."""
    viz = InteractiveVisualizer(sample_config)
    
    assert viz.base_config == sample_config
    assert isinstance(viz.visualizer, SimulationVisualizer)


# Test parameter sweep summary plot
def test_plot_parameter_sweep_summary(sample_results_dict):
    """Test parameter sweep summary plot."""
    fig = plot_parameter_sweep_summary(sample_results_dict, "q")
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6
    plt.close('all')


def test_plot_parameter_sweep_summary_custom_size(sample_results_dict):
    """Test sweep summary with custom size."""
    fig = plot_parameter_sweep_summary(sample_results_dict, "q", figsize=(20, 12))
    
    assert fig is not None
    plt.close('all')


# Test save figure
def test_save_figure_png(sample_result, tmp_path):
    """Test saving figure as PNG."""
    visualizer = SimulationVisualizer()
    fig, ax = plt.subplots()
    visualizer.plot_energy_breakdown_pie(sample_result, ax=ax)
    
    filename = tmp_path / "test_plot"
    save_figure(fig, str(filename), formats=['png'])
    
    assert (tmp_path / "test_plot.png").exists()
    plt.close('all')


def test_save_figure_multiple_formats(sample_result, tmp_path):
    """Test saving figure in multiple formats."""
    visualizer = SimulationVisualizer()
    fig, ax = plt.subplots()
    visualizer.plot_energy_breakdown_pie(sample_result, ax=ax)
    
    filename = tmp_path / "test_plot"
    save_figure(fig, str(filename), formats=['png', 'pdf'])
    
    assert (tmp_path / "test_plot.png").exists()
    assert (tmp_path / "test_plot.pdf").exists()
    plt.close('all')


def test_save_figure_custom_dpi(sample_result, tmp_path):
    """Test saving figure with custom DPI."""
    visualizer = SimulationVisualizer()
    fig, ax = plt.subplots()
    visualizer.plot_energy_breakdown_pie(sample_result, ax=ax)
    
    filename = tmp_path / "test_plot"
    save_figure(fig, str(filename), formats=['png'], dpi=150)
    
    assert (tmp_path / "test_plot.png").exists()
    plt.close('all')


# Integration test: Create multiple plots
def test_multiple_plots_integration(sample_results_dict, sample_result):
    """Test creating multiple plots in sequence."""
    visualizer = SimulationVisualizer()
    
    # Create multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    visualizer.plot_lifetime_vs_parameter(sample_results_dict, "q", ax=axes[0, 0])
    visualizer.plot_delay_vs_parameter(sample_results_dict, "q", ax=axes[0, 1])
    visualizer.plot_energy_breakdown_pie(sample_result, ax=axes[1, 0])
    visualizer.plot_state_occupation_pie(sample_result, ax=axes[1, 1])
    
    assert fig is not None
    plt.close('all')


# Test edge cases
def test_empty_results_dict():
    """Test with empty results dictionary."""
    visualizer = SimulationVisualizer()
    results_dict = {}
    
    # Should handle gracefully
    ax = visualizer.plot_lifetime_vs_delay_scatter(results_dict, "q")
    assert ax is not None
    plt.close('all')


def test_single_result_in_dict(sample_result):
    """Test with single result in dictionary."""
    visualizer = SimulationVisualizer()
    results_dict = {0.1: [sample_result]}
    
    ax = visualizer.plot_lifetime_vs_parameter(results_dict, "q")
    assert ax is not None
    plt.close('all')


# Performance test
def test_large_parameter_sweep(sample_config):
    """Test visualization with larger parameter sweep."""
    visualizer = SimulationVisualizer()
    results_dict = {}
    
    # Create results for 5 different q values
    for q in [0.05, 0.1, 0.15, 0.2, 0.25]:
        results_list = []
        for seed in range(2):  # Only 2 replications for speed
            config = SimulationConfig(
                n_nodes=10,
                arrival_rate=sample_config.arrival_rate,
                transmission_prob=q,
                idle_timer=sample_config.idle_timer,
                wakeup_time=sample_config.wakeup_time,
                initial_energy=500,  # Lower energy for faster runs
                power_rates=sample_config.power_rates,
                max_slots=500,
                seed=seed
            )
            sim = Simulator(config)
            result = sim.run_simulation(track_history=False)
            results_list.append(result)
        
        results_dict[q] = results_list
    
    # Create comprehensive summary
    fig = plot_parameter_sweep_summary(results_dict, "q")
    
    assert fig is not None
    assert len(fig.axes) == 6
    plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
