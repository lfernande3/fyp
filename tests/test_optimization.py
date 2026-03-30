"""
Test Suite for Optimization Module (Objective O3)

Tests cover:
- ParameterOptimizer: grid_search_q, tradeoff_analysis, grid_search_q_ts
- DutyCycleSimulator: run_duty_cycle_simulation, compare_with_on_demand
- PrioritizationAnalyzer: run_scenario_comparison, gains computation
- OptimizationVisualizer: plot methods (non-interactive, using Agg backend)
- run_optimization_experiments: end-to-end quick-mode smoke test

Date: March 2026
"""

import pytest
import numpy as np
import sys
import os

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization import (
    ParameterOptimizer,
    DutyCycleSimulator,
    PrioritizationAnalyzer,
    OptimizationVisualizer,
    OptimizationResult,
    TradeoffPoint,
    PrioritizationComparison,
    run_optimization_experiments,
)
from src.simulator import SimulationConfig, SimulationResults
from src.power_model import PowerModel, PowerProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_base_config(n_nodes: int = 10, max_slots: int = 5_000, seed: int = 42) -> SimulationConfig:
    return SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.01,
        transmission_prob=0.05,
        idle_timer=10,
        wakeup_time=3,
        initial_energy=5_000,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=max_slots,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# ParameterOptimizer
# ---------------------------------------------------------------------------

class TestGridSearchQ:
    def test_returns_optimization_result(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=[0.05, 0.1, 0.2], objective="lifetime",
            n_replications=3, verbose=False,
        )
        assert isinstance(result, OptimizationResult)

    def test_param_name_is_q(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=[0.05, 0.1], objective="lifetime",
            n_replications=2, verbose=False,
        )
        assert result.param_name == "q"

    def test_lifetime_objective(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=[0.05, 0.1, 0.2], objective="lifetime",
            n_replications=3, verbose=False,
        )
        assert result.objective == "lifetime"
        assert result.optimal_value in [0.05, 0.1, 0.2]

    def test_delay_objective(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=[0.05, 0.1, 0.2], objective="delay",
            n_replications=3, verbose=False,
        )
        assert result.objective == "delay"
        assert result.optimal_value in [0.05, 0.1, 0.2]

    def test_output_lengths_match(self):
        cfg = make_base_config()
        q_vals = [0.05, 0.1, 0.2, 0.3]
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=q_vals, objective="lifetime",
            n_replications=2, verbose=False,
        )
        assert len(result.all_values) == len(q_vals)
        assert len(result.all_lifetimes) == len(q_vals)
        assert len(result.all_delays) == len(q_vals)
        assert len(result.all_metrics) == len(q_vals)

    def test_optimal_value_in_range(self):
        cfg = make_base_config()
        q_vals = list(np.linspace(0.01, 0.4, 8))
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=q_vals, objective="lifetime",
            n_replications=2, verbose=False,
        )
        assert min(q_vals) <= result.optimal_value <= max(q_vals)

    def test_config_at_optimum_has_correct_q(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q(
            cfg, q_values=[0.05, 0.1, 0.2], objective="lifetime",
            n_replications=2, verbose=False,
        )
        assert result.config_at_optimum.transmission_prob == result.optimal_value

    def test_default_q_values_used_when_none(self):
        cfg = make_base_config(max_slots=3_000)
        result = ParameterOptimizer.grid_search_q(
            cfg, objective="lifetime", n_replications=2, verbose=False,
        )
        assert result.optimal_value is not None
        assert 0.01 <= result.optimal_value <= 0.5


class TestTradeoffAnalysis:
    def test_returns_list_of_tradeoff_points(self):
        cfg = make_base_config()
        points = ParameterOptimizer.tradeoff_analysis(
            cfg, q_values=[0.05, 0.1, 0.2], ts_values=[1, 10],
            n_replications=3, verbose=False,
        )
        assert isinstance(points, list)
        for pt in points:
            assert isinstance(pt, TradeoffPoint)

    def test_one_point_per_ts_at_most(self):
        cfg = make_base_config()
        ts_vals = [1, 10, 50]
        points = ParameterOptimizer.tradeoff_analysis(
            cfg, q_values=[0.05, 0.1], ts_values=ts_vals,
            n_replications=2, verbose=False,
        )
        assert len(points) <= len(ts_vals)

    def test_tradeoff_point_fields(self):
        cfg = make_base_config()
        points = ParameterOptimizer.tradeoff_analysis(
            cfg, q_values=[0.05, 0.15], ts_values=[5],
            n_replications=3, verbose=False,
        )
        if points:
            pt = points[0]
            assert pt.ts == 5
            assert pt.q in [0.05, 0.15]
            assert pt.lifetime_years >= 0
            assert pt.delay_slots >= 0
            assert pt.delay_ms == pytest.approx(pt.delay_slots * 6.0)

    def test_ts_values_assigned_correctly(self):
        cfg = make_base_config()
        points = ParameterOptimizer.tradeoff_analysis(
            cfg, q_values=[0.05, 0.1, 0.2], ts_values=[1, 20],
            n_replications=2, verbose=False,
        )
        point_ts = {pt.ts for pt in points}
        assert point_ts.issubset({1, 20})


class TestGridSearchQTs:
    def test_returns_dict_with_expected_keys(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q_ts(
            cfg, q_values=[0.05, 0.1], ts_values=[1, 10],
            n_replications=2, verbose=False,
        )
        for key in ("q_values", "ts_values", "lifetime_matrix", "delay_matrix",
                    "lifetime_std_matrix", "delay_std_matrix"):
            assert key in result

    def test_matrix_shapes(self):
        cfg = make_base_config()
        q_vals, ts_vals = [0.05, 0.1, 0.2], [1, 10]
        result = ParameterOptimizer.grid_search_q_ts(
            cfg, q_values=q_vals, ts_values=ts_vals,
            n_replications=2, verbose=False,
        )
        assert result["lifetime_matrix"].shape == (len(ts_vals), len(q_vals))
        assert result["delay_matrix"].shape == (len(ts_vals), len(q_vals))

    def test_delay_matrix_non_negative(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q_ts(
            cfg, q_values=[0.05, 0.1], ts_values=[1, 10],
            n_replications=2, verbose=False,
        )
        assert np.all(result["delay_matrix"] >= 0)

    def test_lifetime_matrix_non_negative(self):
        cfg = make_base_config()
        result = ParameterOptimizer.grid_search_q_ts(
            cfg, q_values=[0.05, 0.1], ts_values=[1, 10],
            n_replications=2, verbose=False,
        )
        assert np.all(result["lifetime_matrix"] >= 0)


# ---------------------------------------------------------------------------
# DutyCycleSimulator
# ---------------------------------------------------------------------------

class TestDutyCycleSimulator:
    def test_run_returns_simulation_results(self):
        cfg = make_base_config()
        result = DutyCycleSimulator.run_duty_cycle_simulation(
            cfg, cycle_period=20, awake_fraction=0.5, seed=42,
        )
        assert isinstance(result, SimulationResults)

    def test_delay_non_negative(self):
        cfg = make_base_config()
        result = DutyCycleSimulator.run_duty_cycle_simulation(
            cfg, cycle_period=20, awake_fraction=0.5, seed=42,
        )
        assert result.mean_delay >= 0

    def test_lifetime_non_negative(self):
        cfg = make_base_config()
        result = DutyCycleSimulator.run_duty_cycle_simulation(
            cfg, cycle_period=20, awake_fraction=0.5, seed=42,
        )
        assert result.mean_lifetime_years >= 0 or result.mean_lifetime_years == float("inf")

    def test_awake_fraction_extremes(self):
        cfg = make_base_config()
        r_low = DutyCycleSimulator.run_duty_cycle_simulation(
            cfg, cycle_period=20, awake_fraction=0.1, seed=0,
        )
        r_high = DutyCycleSimulator.run_duty_cycle_simulation(
            cfg, cycle_period=20, awake_fraction=0.9, seed=0,
        )
        assert isinstance(r_low, SimulationResults)
        assert isinstance(r_high, SimulationResults)

    def test_compare_with_on_demand_structure(self):
        cfg = make_base_config()
        cmp = DutyCycleSimulator.compare_with_on_demand(
            cfg, ts_values=[1, 10], awake_fractions=[0.3, 0.7],
            cycle_period=20, n_replications=2, verbose=False,
        )
        assert "on_demand" in cmp
        assert "duty_cycle" in cmp

    def test_compare_on_demand_lengths(self):
        cfg = make_base_config()
        ts_vals = [1, 10, 20]
        cmp = DutyCycleSimulator.compare_with_on_demand(
            cfg, ts_values=ts_vals, awake_fractions=[0.3],
            cycle_period=20, n_replications=2, verbose=False,
        )
        assert len(cmp["on_demand"]["lifetimes"]) == len(ts_vals)
        assert len(cmp["on_demand"]["delays"]) == len(ts_vals)

    def test_compare_duty_cycle_lengths(self):
        cfg = make_base_config()
        fracs = [0.2, 0.5, 0.8]
        cmp = DutyCycleSimulator.compare_with_on_demand(
            cfg, ts_values=[10], awake_fractions=fracs,
            cycle_period=20, n_replications=2, verbose=False,
        )
        assert len(cmp["duty_cycle"]["lifetimes"]) == len(fracs)
        assert len(cmp["duty_cycle"]["delays"]) == len(fracs)


# ---------------------------------------------------------------------------
# PrioritizationAnalyzer
# ---------------------------------------------------------------------------

class TestPrioritizationAnalyzer:
    def test_returns_prioritization_comparison(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, arrival_rate=0.01,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        assert isinstance(cmp, PrioritizationComparison)

    def test_three_scenarios(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        assert len(cmp.scenario_names) == 3
        assert len(cmp.mean_delays) == 3
        assert len(cmp.mean_lifetimes) == 3
        assert len(cmp.delay_stds) == 3
        assert len(cmp.lifetime_stds) == 3
        assert len(cmp.configs) == 3

    def test_gains_computed_for_all_scenarios(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        for name in cmp.scenario_names:
            assert name in cmp.gains_vs_baseline
            g = cmp.gains_vs_baseline[name]
            assert "delay_change_pct" in g
            assert "lifetime_change_pct" in g
            assert "delay_slots" in g
            assert "lifetime_years" in g

    def test_balanced_baseline_has_zero_gains(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=3, verbose=False,
        )
        balanced_name = next(n for n in cmp.scenario_names if "balanced" in n.lower())
        g = cmp.gains_vs_baseline[balanced_name]
        assert g["delay_change_pct"] == pytest.approx(0.0, abs=1e-9)
        assert g["lifetime_change_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_delays_non_negative(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        for d in cmp.mean_delays:
            assert d >= 0

    def test_print_summary_runs(self, capsys):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        PrioritizationAnalyzer.print_comparison_summary(cmp)
        out = capsys.readouterr().out
        assert "PRIORITIZATION SCENARIO COMPARISON SUMMARY" in out
        assert "Balanced" in out

    def test_low_latency_scenario_small_ts(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        ll_cfg = cmp.configs[0]   # First scenario = low-latency
        assert ll_cfg.idle_timer == 1

    def test_battery_scenario_large_ts(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        bat_cfg = cmp.configs[2]  # Third scenario = battery-life
        assert bat_cfg.idle_timer == 50


# ---------------------------------------------------------------------------
# OptimizationVisualizer
# ---------------------------------------------------------------------------

class TestOptimizationVisualizer:
    @pytest.fixture(autouse=True)
    def use_agg(self):
        """Use non-interactive backend for all visualizer tests."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_plot_q_sweep_returns_figure(self):
        fig, ax_l, ax_r = OptimizationVisualizer.plot_q_sweep(
            q_values=[0.05, 0.1, 0.2],
            lifetimes=[2.0, 3.0, 2.5],
            delays=[10.0, 8.0, 9.0],
        )
        assert fig is not None
        assert ax_l is not None
        assert ax_r is not None

    def test_plot_q_sweep_with_optima(self):
        fig, ax_l, ax_r = OptimizationVisualizer.plot_q_sweep(
            q_values=[0.05, 0.1, 0.2],
            lifetimes=[2.0, 3.0, 2.5],
            delays=[10.0, 8.0, 9.0],
            optimal_q_lifetime=0.1,
            optimal_q_delay=0.2,
        )
        assert fig is not None

    def test_plot_pareto_frontier(self):
        points = [
            TradeoffPoint(q=0.05, ts=1, lifetime_years=2.0, delay_slots=5.0, delay_ms=30.0),
            TradeoffPoint(q=0.1, ts=10, lifetime_years=3.0, delay_slots=8.0, delay_ms=48.0),
            TradeoffPoint(q=0.05, ts=50, lifetime_years=4.0, delay_slots=12.0, delay_ms=72.0),
        ]
        fig, ax = OptimizationVisualizer.plot_pareto_frontier(points)
        assert fig is not None
        assert ax is not None

    def test_plot_tradeoff_heatmap_lifetime(self):
        grid = {
            "q_values": [0.05, 0.1, 0.2],
            "ts_values": [1, 10],
            "lifetime_matrix": np.array([[2.0, 3.0, 2.5], [3.0, 4.0, 3.5]]),
            "delay_matrix": np.array([[10.0, 8.0, 9.0], [12.0, 9.0, 11.0]]),
            "lifetime_std_matrix": np.zeros((2, 3)),
            "delay_std_matrix": np.zeros((2, 3)),
        }
        fig, ax = OptimizationVisualizer.plot_tradeoff_heatmap(grid, metric="lifetime")
        assert fig is not None

    def test_plot_tradeoff_heatmap_delay(self):
        grid = {
            "q_values": [0.05, 0.1],
            "ts_values": [1, 10, 50],
            "lifetime_matrix": np.ones((3, 2)) * 2.0,
            "delay_matrix": np.array([[10.0, 8.0], [12.0, 9.0], [15.0, 11.0]]),
            "lifetime_std_matrix": np.zeros((3, 2)),
            "delay_std_matrix": np.zeros((3, 2)),
        }
        fig, ax = OptimizationVisualizer.plot_tradeoff_heatmap(grid, metric="delay")
        assert fig is not None

    def test_plot_prioritization_comparison(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        fig, ax = OptimizationVisualizer.plot_prioritization_comparison(cmp)
        assert fig is not None

    def test_plot_gains_losses_bar(self):
        cmp = PrioritizationAnalyzer.run_scenario_comparison(
            n_nodes=10, max_slots=5_000, n_replications=2, verbose=False,
        )
        fig, ax = OptimizationVisualizer.plot_gains_losses_bar(cmp)
        assert fig is not None

    def test_plot_duty_cycle_comparison(self):
        cfg = make_base_config()
        cmp_data = DutyCycleSimulator.compare_with_on_demand(
            cfg, ts_values=[1, 10], awake_fractions=[0.3, 0.7],
            cycle_period=20, n_replications=2, verbose=False,
        )
        fig, ax = OptimizationVisualizer.plot_duty_cycle_comparison(cmp_data)
        assert fig is not None


# ---------------------------------------------------------------------------
# End-to-end quick-mode smoke test
# ---------------------------------------------------------------------------

class TestRunOptimizationExperiments:
    def test_quick_mode_runs_and_returns_all_keys(self):
        results = run_optimization_experiments(quick_mode=True)
        expected_keys = {
            "opt_q_lifetime",
            "opt_q_delay",
            "tradeoff_points",
            "grid_results",
            "prioritization_comparison",
            "duty_cycle_comparison",
        }
        assert expected_keys.issubset(results.keys())

    def test_quick_mode_opt_q_lifetime_is_result(self):
        results = run_optimization_experiments(quick_mode=True)
        assert isinstance(results["opt_q_lifetime"], OptimizationResult)

    def test_quick_mode_tradeoff_points_list(self):
        results = run_optimization_experiments(quick_mode=True)
        assert isinstance(results["tradeoff_points"], list)

    def test_quick_mode_grid_has_matrices(self):
        results = run_optimization_experiments(quick_mode=True)
        grid = results["grid_results"]
        assert "lifetime_matrix" in grid
        assert "delay_matrix" in grid

    def test_quick_mode_prioritization_comparison(self):
        results = run_optimization_experiments(quick_mode=True)
        assert isinstance(results["prioritization_comparison"], PrioritizationComparison)

    def test_quick_mode_duty_cycle_comparison(self):
        results = run_optimization_experiments(quick_mode=True)
        dc = results["duty_cycle_comparison"]
        assert "on_demand" in dc
        assert "duty_cycle" in dc
