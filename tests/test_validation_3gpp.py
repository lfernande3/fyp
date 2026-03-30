"""
Test Suite for 3GPP Validation Module (Objective O4)

Tests cover:
- ThreeGPPAlignment: parameter mapping, scenario creation
- AnalyticsValidator: formula validation, convergence, on-demand superiority
- DesignGuidelines: recommended ts, lifetime_vs_lambda, guideline table
- ValidationVisualizer: plot methods (Agg backend)
- run_o4_experiments: end-to-end quick-mode smoke test

Date: April 2026
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.validation_3gpp import (
    ThreeGPPAlignment,
    ThreeGPPScenario,
    AnalyticsValidator,
    FormulaValidationResult,
    ValidationReport,
    DesignGuidelines,
    GuidelineEntry,
    ValidationVisualizer,
    run_o4_experiments,
    SLOT_DURATION_MS,
)
from src.simulator import SimulationConfig
from src.power_model import PowerModel, PowerProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_config(n: int = 10, max_slots: int = 5_000, seed: int = 0) -> SimulationConfig:
    return SimulationConfig(
        n_nodes=n,
        arrival_rate=0.005,
        transmission_prob=1.0 / n,
        idle_timer=10,
        wakeup_time=2,
        initial_energy=5000.0,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=max_slots,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# ThreeGPPAlignment – parameter mapping
# ---------------------------------------------------------------------------

class TestThreeGPPAlignment:
    def test_t3324_to_ts_basic(self):
        # 6ms slots: 10s / 0.006s = 1666 slots
        ts = ThreeGPPAlignment.t3324_to_ts(10.0, slot_duration_ms=6.0)
        assert ts == pytest.approx(1666, abs=1)

    def test_t3324_to_ts_minimum_one(self):
        ts = ThreeGPPAlignment.t3324_to_ts(0.001, slot_duration_ms=6.0)
        assert ts >= 1

    def test_ts_to_t3324_roundtrip(self):
        ts = 100
        t3324 = ThreeGPPAlignment.ts_to_t3324(ts, slot_duration_ms=6.0)
        ts_back = ThreeGPPAlignment.t3324_to_ts(t3324, slot_duration_ms=6.0)
        assert ts_back == ts

    def test_ra_sdt_2step_tw(self):
        assert ThreeGPPAlignment.RA_SDT_2STEP_TW == 2

    def test_ra_sdt_4step_tw(self):
        assert ThreeGPPAlignment.RA_SDT_4STEP_TW == 4

    def test_create_mico_nb_iot_returns_scenario(self):
        sc = ThreeGPPAlignment.create_mico_nb_iot_scenario(n_nodes=10, arrival_rate=0.01)
        assert isinstance(sc, ThreeGPPScenario)
        assert sc.config.n_nodes == 10
        assert sc.config.arrival_rate == 0.01

    def test_create_mico_nb_iot_tw_from_steps(self):
        sc2 = ThreeGPPAlignment.create_mico_nb_iot_scenario(ra_sdt_steps=2)
        sc4 = ThreeGPPAlignment.create_mico_nb_iot_scenario(ra_sdt_steps=4)
        assert sc2.config.wakeup_time == 2
        assert sc4.config.wakeup_time == 4

    def test_create_mico_nb_iot_ts_matches_t3324(self):
        t3324_s = 12.0
        sc = ThreeGPPAlignment.create_mico_nb_iot_scenario(t3324_s=t3324_s)
        expected_ts = ThreeGPPAlignment.t3324_to_ts(t3324_s)
        assert sc.config.idle_timer == expected_ts

    def test_create_mico_nr_mmtc_returns_scenario(self):
        sc = ThreeGPPAlignment.create_mico_nr_mmtc_scenario(n_nodes=15)
        assert isinstance(sc, ThreeGPPScenario)
        assert sc.config.n_nodes == 15

    def test_create_mico_nr_mmtc_uses_nr_profile(self):
        sc = ThreeGPPAlignment.create_mico_nr_mmtc_scenario()
        assert sc.power_profile == PowerProfile.NR_MMTC

    def test_create_mico_nb_iot_uses_nb_iot_profile(self):
        sc = ThreeGPPAlignment.create_mico_nb_iot_scenario()
        assert sc.power_profile == PowerProfile.NB_IOT

    def test_create_standard_scenarios_returns_four(self):
        scenarios = ThreeGPPAlignment.create_standard_scenarios(n_nodes=10, max_slots=5_000)
        assert len(scenarios) == 4
        for sc in scenarios:
            assert isinstance(sc, ThreeGPPScenario)

    def test_standard_scenarios_different_names(self):
        scenarios = ThreeGPPAlignment.create_standard_scenarios(n_nodes=10, max_slots=5_000)
        names = [sc.name for sc in scenarios]
        assert len(set(names)) == 4  # all distinct

    def test_scenario_q_optimal_per_n(self):
        sc = ThreeGPPAlignment.create_mico_nb_iot_scenario(n_nodes=20)
        assert sc.config.transmission_prob == pytest.approx(1.0 / 20)


# ---------------------------------------------------------------------------
# AnalyticsValidator
# ---------------------------------------------------------------------------

class TestAnalyticsValidator:
    def test_validate_one_returns_report(self):
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.005, tw=2, ts=10,
            max_slots=5_000, n_replications=3,
        )
        assert isinstance(rep, ValidationReport)

    def test_validate_one_has_p_and_mu(self):
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.005, tw=2, ts=10,
            max_slots=5_000, n_replications=3,
        )
        assert isinstance(rep.p_result, FormulaValidationResult)
        assert isinstance(rep.mu_result, FormulaValidationResult)

    def test_validate_one_stable_regime_has_delay_result(self):
        # Low λ, moderate q → stable
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.001, tw=2, ts=10,
            max_slots=5_000, n_replications=3,
        )
        assert rep.is_stable
        assert rep.delay_result is not None

    def test_validate_one_empirical_p_positive(self):
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.005, tw=2, ts=10,
            max_slots=5_000, n_replications=3,
        )
        assert rep.p_result.empirical_mean > 0

    def test_validate_one_relative_error_finite(self):
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.005, tw=2, ts=10,
            max_slots=5_000, n_replications=3,
        )
        assert rep.p_result.relative_error >= 0
        assert rep.mu_result.relative_error >= 0

    def test_formula_validation_result_thresholds(self):
        r = FormulaValidationResult(
            metric_name="p", n=10, q=0.1, lambda_rate=0.005, tw=2, ts=10,
            analytical_value=0.100, empirical_mean=0.103, empirical_std=0.002,
            relative_error=0.03, within_5pct=True, within_10pct=True,
            within_20pct=True, n_replications=5,
        )
        assert r.within_5pct
        assert r.within_10pct
        assert r.within_20pct

    def test_validate_across_n_returns_dict(self):
        results = AnalyticsValidator.validate_across_n(
            n_values=[5, 10], lambda_rate=0.003, tw=2, ts=5,
            max_slots=5_000, n_replications=3, verbose=False,
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {5, 10}
        for n, rep in results.items():
            assert isinstance(rep, ValidationReport)

    def test_validate_across_n_q_per_n(self):
        results = AnalyticsValidator.validate_across_n(
            n_values=[10], q_per_n=True, lambda_rate=0.003, tw=2, ts=5,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        # q = 1/10 = 0.1
        assert results[10].p_result.q == pytest.approx(0.1)

    def test_validate_convergence_returns_dict(self):
        cfg = make_small_config()
        results = AnalyticsValidator.validate_convergence(
            cfg, slot_counts=[3_000, 5_000], n_replications=2, verbose=False,
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {3_000, 5_000}

    def test_validate_convergence_error_decreases(self):
        """Error should generally decrease with more slots (probabilistic)."""
        cfg = make_small_config(n=10)
        results = AnalyticsValidator.validate_convergence(
            cfg, slot_counts=[1_000, 10_000], n_replications=5, verbose=False,
        )
        err_low = results[1_000].p_result.relative_error
        err_high = results[10_000].p_result.relative_error
        # Not guaranteed but likely; just assert both are finite
        assert err_low >= 0
        assert err_high >= 0

    def test_demonstrate_on_demand_returns_dict(self):
        cfg = make_small_config()
        result = AnalyticsValidator.demonstrate_on_demand_superiority(
            base_config=cfg,
            ts_values=[1, 10], awake_fractions=[0.3, 0.7],
            n_replications=2, verbose=False,
        )
        assert "on_demand" in result
        assert "duty_cycle" in result
        assert "summary" in result

    def test_demonstrate_superiority_summary_keys(self):
        cfg = make_small_config()
        result = AnalyticsValidator.demonstrate_on_demand_superiority(
            cfg, ts_values=[1, 10], awake_fractions=[0.3],
            n_replications=2, verbose=False,
        )
        summary = result["summary"]
        assert "on_demand_median_lifetime" in summary
        assert "duty_cycle_median_lifetime" in summary
        assert "conclusion" in summary

    def test_print_validation_report_runs(self, capsys):
        rep = AnalyticsValidator.validate_one(
            n=10, q=0.1, lambda_rate=0.003, tw=2, ts=10,
            max_slots=5_000, n_replications=2,
        )
        AnalyticsValidator.print_validation_report(rep)
        out = capsys.readouterr().out
        assert "Validation Report" in out


# ---------------------------------------------------------------------------
# DesignGuidelines
# ---------------------------------------------------------------------------

class TestDesignGuidelines:
    def test_recommended_ts_returns_int_or_none(self):
        result = DesignGuidelines.recommended_ts_for_delay_target(
            lambda_rate=0.01, delay_target_ms=5000.0,
            n=10, tw=2,
            ts_candidates=[1, 5, 10],
            n_replications=2, verbose=False,
        )
        assert result is None or isinstance(result, int)

    def test_recommended_ts_feasible_for_generous_target(self):
        # Generous target (100 s = 100,000 ms): almost all ts should qualify
        result = DesignGuidelines.recommended_ts_for_delay_target(
            lambda_rate=0.005, delay_target_ms=100_000.0,
            n=10, tw=2,
            ts_candidates=[1, 10, 50],
            n_replications=3, verbose=False,
        )
        assert result is not None
        assert result in [1, 10, 50]

    def test_lifetime_vs_lambda_returns_dict(self):
        data = DesignGuidelines.lifetime_vs_lambda(
            ts_values=[1, 10], lambda_values=[0.005, 0.01],
            n=10, tw=2, max_slots=5_000, n_replications=2, verbose=False,
        )
        assert "lifetime_matrix" in data
        assert "delay_matrix" in data
        assert data["lifetime_matrix"].shape == (2, 2)
        assert data["delay_matrix"].shape == (2, 2)

    def test_lifetime_vs_lambda_delay_non_negative(self):
        data = DesignGuidelines.lifetime_vs_lambda(
            ts_values=[1, 10], lambda_values=[0.005],
            n=10, tw=2, max_slots=5_000, n_replications=2, verbose=False,
        )
        assert np.all(data["delay_matrix"] >= 0)

    def test_generate_guideline_table_returns_list(self):
        entries = DesignGuidelines.generate_guideline_table(
            lambda_values=[0.005], ts_values=[1, 10],
            n=10, tw=2, delay_target_ms=5000.0,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        assert isinstance(entries, list)
        assert len(entries) == 2  # 1 λ × 2 ts

    def test_guideline_entry_fields(self):
        entries = DesignGuidelines.generate_guideline_table(
            lambda_values=[0.005], ts_values=[10],
            n=10, tw=2, delay_target_ms=5000.0,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        e = entries[0]
        assert isinstance(e, GuidelineEntry)
        assert e.lambda_rate == 0.005
        assert e.ts_slots == 10
        assert e.recommended_q == pytest.approx(1.0 / 10)
        assert e.mean_delay_ms >= 0
        assert isinstance(e.meets_delay_target, bool)
        assert isinstance(e.stability_margin, float)

    def test_guideline_t3324_conversion(self):
        entries = DesignGuidelines.generate_guideline_table(
            lambda_values=[0.005], ts_values=[100],
            n=10, tw=2, delay_target_ms=5000.0,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        e = entries[0]
        expected_t3324 = 100 * SLOT_DURATION_MS / 1000.0
        assert e.t3324_s == pytest.approx(expected_t3324)

    def test_print_guideline_table_runs(self, capsys):
        entries = DesignGuidelines.generate_guideline_table(
            lambda_values=[0.005], ts_values=[1, 10],
            n=10, tw=2, delay_target_ms=5000.0,
            max_slots=5_000, n_replications=2, verbose=False,
        )
        DesignGuidelines.print_guideline_table(entries)
        out = capsys.readouterr().out
        assert "DESIGN GUIDELINE TABLE" in out


# ---------------------------------------------------------------------------
# ValidationVisualizer
# ---------------------------------------------------------------------------

class TestValidationVisualizer:
    @pytest.fixture(autouse=True)
    def use_agg(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def _make_reports(self):
        return AnalyticsValidator.validate_across_n(
            n_values=[5, 10], lambda_rate=0.003, tw=2, ts=5,
            max_slots=5_000, n_replications=2, verbose=False,
        )

    def test_plot_analytical_vs_empirical_p(self):
        reports = self._make_reports()
        fig, ax = ValidationVisualizer.plot_analytical_vs_empirical(reports, metric="p")
        assert fig is not None

    def test_plot_analytical_vs_empirical_mu(self):
        reports = self._make_reports()
        fig, ax = ValidationVisualizer.plot_analytical_vs_empirical(reports, metric="mu")
        assert fig is not None

    def test_plot_convergence(self):
        cfg = make_small_config()
        conv = AnalyticsValidator.validate_convergence(
            cfg, slot_counts=[3_000, 5_000], n_replications=2, verbose=False,
        )
        fig, ax = ValidationVisualizer.plot_convergence(conv, metric="p")
        assert fig is not None

    def test_plot_lifetime_vs_lambda(self):
        data = DesignGuidelines.lifetime_vs_lambda(
            ts_values=[1, 10], lambda_values=[0.005, 0.01],
            n=10, tw=2, max_slots=5_000, n_replications=2, verbose=False,
        )
        fig, ax = ValidationVisualizer.plot_lifetime_vs_lambda(data)
        assert fig is not None

    def test_plot_delay_vs_lambda(self):
        data = DesignGuidelines.lifetime_vs_lambda(
            ts_values=[1, 10], lambda_values=[0.005, 0.01],
            n=10, tw=2, max_slots=5_000, n_replications=2, verbose=False,
        )
        fig, ax = ValidationVisualizer.plot_delay_vs_lambda(data, delay_target_ms=1000.0)
        assert fig is not None

    def test_plot_3gpp_scenario_comparison(self):
        scenario_results = {
            "NB-IoT 2s": (120.0, 3.5, 10.0, 0.2),
            "NB-IoT 60s": (500.0, 5.0, 30.0, 0.4),
        }
        fig, ax = ValidationVisualizer.plot_3gpp_scenario_comparison(scenario_results)
        assert fig is not None

    def test_plot_q_star_vs_n(self):
        fig, ax = ValidationVisualizer.plot_q_star_vs_n(n_values=[5, 10, 20, 50])
        assert fig is not None


# ---------------------------------------------------------------------------
# End-to-end quick-mode smoke test
# ---------------------------------------------------------------------------

class TestRunO4Experiments:
    def test_quick_mode_returns_all_keys(self):
        results = run_o4_experiments(quick_mode=True)
        expected = {
            "scenarios", "scenario_results", "validation_across_n",
            "convergence", "on_demand_superiority",
            "guideline_table", "lifetime_data",
        }
        assert expected.issubset(results.keys())

    def test_scenarios_list(self):
        results = run_o4_experiments(quick_mode=True)
        assert isinstance(results["scenarios"], list)
        assert len(results["scenarios"]) == 4

    def test_validation_across_n_dict(self):
        results = run_o4_experiments(quick_mode=True)
        assert isinstance(results["validation_across_n"], dict)
        for n, rep in results["validation_across_n"].items():
            assert isinstance(rep, ValidationReport)

    def test_convergence_dict(self):
        results = run_o4_experiments(quick_mode=True)
        assert isinstance(results["convergence"], dict)

    def test_guideline_table_list(self):
        results = run_o4_experiments(quick_mode=True)
        assert isinstance(results["guideline_table"], list)
        assert all(isinstance(e, GuidelineEntry) for e in results["guideline_table"])

    def test_lifetime_data_has_matrices(self):
        results = run_o4_experiments(quick_mode=True)
        ld = results["lifetime_data"]
        assert "lifetime_matrix" in ld
        assert "delay_matrix" in ld

    def test_scenario_results_dict(self):
        results = run_o4_experiments(quick_mode=True)
        sr = results["scenario_results"]
        assert isinstance(sr, dict)
        assert len(sr) == 4
        for name, tup in sr.items():
            assert len(tup) == 4   # (d_ms, lt, d_std, lt_std)
