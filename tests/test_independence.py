"""
Test Suite for Independence Analysis Module (Objective O5)

Tests cover:
- IndependenceAnalyzer: compute_analytical_quantities, run_factorial_sweep,
  run_regression_analysis, find_optimal_q_per_ts
- IndependenceVisualizer: plot methods (non-interactive, using Agg backend)

Date: April 2026
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

from src.independence import (
    IndependenceAnalyzer,
    IndependenceVisualizer,
    run_o5_experiments,
)
from src.simulator import SimulationConfig
from src.power_model import PowerModel, PowerProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_base_config(n_nodes=10, max_slots=5_000, seed=42):
    return SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.01,
        transmission_prob=0.05,
        idle_timer=10,
        wakeup_time=2,
        initial_energy=5_000,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=max_slots,
        seed=seed,
    )


def make_small_sweep():
    """Run a tiny factorial sweep for use in multiple tests."""
    cfg = make_base_config()
    return IndependenceAnalyzer.run_factorial_sweep(
        cfg,
        q_values=[0.05, 0.1, 0.2],
        ts_values=[1, 5, 10],
        n_replications=3,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# compute_analytical_quantities
# ---------------------------------------------------------------------------

class TestComputeAnalytical:
    def test_p_at_q_equals_1_over_n(self):
        n = 10
        q = 1.0 / n
        result = IndependenceAnalyzer.compute_analytical_quantities(
            [q], [1], tw=2, n=n,
        )
        expected_p = q * (1 - q) ** (n - 1)
        assert abs(result["p_matrix"][0, 0] - expected_p) < 1e-10

    def test_mu_decreases_with_ts(self):
        result = IndependenceAnalyzer.compute_analytical_quantities(
            [0.1], [1, 5, 10, 50], tw=2, n=10,
        )
        mu_vals = result["mu_matrix"][:, 0]
        for i in range(len(mu_vals) - 1):
            assert mu_vals[i] > mu_vals[i + 1], (
                f"mu should decrease as ts increases: mu[ts={i}]={mu_vals[i]}"
            )

    def test_kappa_is_p_times_ts(self):
        result = IndependenceAnalyzer.compute_analytical_quantities(
            [0.05, 0.1], [5, 10], tw=2, n=20,
        )
        p = result["p_matrix"]
        ts_arr = np.array(result["ts_values"])
        for i, ts in enumerate(ts_arr):
            for j in range(len(result["q_values"])):
                assert abs(result["kappa_matrix"][i, j] - p[i, j] * ts) < 1e-12

    def test_output_shapes(self):
        q_vals = [0.01, 0.05, 0.1]
        ts_vals = [1, 10, 50]
        result = IndependenceAnalyzer.compute_analytical_quantities(
            q_vals, ts_vals, tw=2, n=10,
        )
        assert result["p_matrix"].shape == (3, 3)
        assert result["mu_matrix"].shape == (3, 3)
        assert result["kappa_matrix"].shape == (3, 3)

    def test_kappa_nonnegative(self):
        result = IndependenceAnalyzer.compute_analytical_quantities(
            [0.01, 0.05, 0.1, 0.2], [1, 2, 5, 10, 20], tw=2, n=50,
        )
        assert np.all(result["kappa_matrix"] >= 0)


# ---------------------------------------------------------------------------
# run_factorial_sweep
# ---------------------------------------------------------------------------

class TestFactorialSweep:
    def test_matrix_shape(self):
        sweep = make_small_sweep()
        assert sweep["lifetime_matrix"].shape == (3, 3)
        assert sweep["delay_matrix"].shape == (3, 3)

    def test_dataframe_rows(self):
        sweep = make_small_sweep()
        df = sweep["df"]
        assert len(df) == 9  # 3 q x 3 ts

    def test_kappa_in_dataframe(self):
        sweep = make_small_sweep()
        df = sweep["df"]
        assert "kappa" in df.columns
        assert all(df["kappa"] >= 0)

    def test_stable_column_is_bool(self):
        sweep = make_small_sweep()
        df = sweep["df"]
        assert df["stable"].dtype == bool


# ---------------------------------------------------------------------------
# run_regression_analysis
# ---------------------------------------------------------------------------

class TestRegression:
    def test_returns_expected_keys(self):
        sweep = make_small_sweep()
        reg = IndependenceAnalyzer.run_regression_analysis(sweep["df"])
        if "error" in reg:
            pytest.skip("Not enough stable points for regression")
        for metric in ["delay", "lifetime"]:
            assert metric in reg
            r = reg[metric]
            assert "F_statistic" in r
            assert "p_value" in r
            assert "R2_additive" in r
            assert "R2_interaction" in r
            assert "interaction_coeffs" in r
            assert "log_q_x_log_ts" in r["interaction_coeffs"]

    def test_f_stat_nonnegative(self):
        sweep = make_small_sweep()
        reg = IndependenceAnalyzer.run_regression_analysis(sweep["df"])
        if "error" in reg:
            pytest.skip("Not enough stable points")
        for metric in ["delay", "lifetime"]:
            assert reg[metric]["F_statistic"] >= 0

    def test_r2_between_0_and_1(self):
        sweep = make_small_sweep()
        reg = IndependenceAnalyzer.run_regression_analysis(sweep["df"])
        if "error" in reg:
            pytest.skip("Not enough stable points")
        for metric in ["delay", "lifetime"]:
            assert 0 <= reg[metric]["R2_additive"] <= 1.0 + 1e-6
            assert 0 <= reg[metric]["R2_interaction"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# find_optimal_q_per_ts
# ---------------------------------------------------------------------------

class TestFindOptimalQ:
    def test_returns_ts_values(self):
        sweep = make_small_sweep()
        opt = IndependenceAnalyzer.find_optimal_q_per_ts(
            sweep["df"], lifetime_constraints=[0.001], n_nodes=10,
        )
        assert "ts_values" in opt
        assert len(opt["ts_values"]) > 0

    def test_q_star_within_grid(self):
        sweep = make_small_sweep()
        q_vals = sweep["q_values"]
        opt = IndependenceAnalyzer.find_optimal_q_per_ts(
            sweep["df"], lifetime_constraints=[0.001], n_nodes=10,
        )
        for data in opt["constraints"].values():
            for q_star in data["q_stars"]:
                if q_star is not None:
                    assert q_star in q_vals


# ---------------------------------------------------------------------------
# Visualizers (smoke tests — just check they return fig without error)
# ---------------------------------------------------------------------------

class TestVisualizers:
    @pytest.fixture(autouse=True)
    def _setup(self):
        import matplotlib.pyplot as plt
        self.sweep = make_small_sweep()
        self.df = self.sweep["df"]
        self.analytical = self.sweep["analytical"]
        yield
        plt.close("all")

    def test_plot_interaction_plots(self):
        fig, axes = IndependenceVisualizer.plot_interaction_plots(self.df)
        assert fig is not None
        assert axes.shape == (2, 2)

    def test_plot_coupling_heatmap(self):
        fig, ax = IndependenceVisualizer.plot_coupling_heatmap(self.analytical)
        assert fig is not None

    def test_plot_regime_map(self):
        fig, ax = IndependenceVisualizer.plot_regime_map(self.df, self.analytical)
        assert fig is not None

    def test_plot_regression_summary(self):
        reg = IndependenceAnalyzer.run_regression_analysis(self.df)
        fig, axes = IndependenceVisualizer.plot_regression_summary(reg)
        assert fig is not None

    def test_plot_optimal_q_shift(self):
        opt = IndependenceAnalyzer.find_optimal_q_per_ts(
            self.df, lifetime_constraints=[0.001], n_nodes=10,
        )
        fig, ax = IndependenceVisualizer.plot_optimal_q_shift(opt)
        assert fig is not None

    def test_plot_iso_contours(self):
        fig, axes = IndependenceVisualizer.plot_iso_contours(self.df)
        assert fig is not None

    def test_plot_kappa_vs_outputs(self):
        fig, axes = IndependenceVisualizer.plot_kappa_vs_outputs(self.df)
        assert fig is not None
        assert axes.shape == (2,)

    def test_plot_pareto_surface(self):
        fig, ax = IndependenceVisualizer.plot_pareto_surface(self.df)
        assert fig is not None

    def test_plot_delay_lifetime_tradeoff_by_kappa(self):
        fig, ax = IndependenceVisualizer.plot_delay_lifetime_tradeoff_by_kappa(self.df)
        assert fig is not None

    def test_plot_marginal_effects(self):
        fig, axes = IndependenceVisualizer.plot_marginal_effects(
            self.df, self.analytical
        )
        assert fig is not None
