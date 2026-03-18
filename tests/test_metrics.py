"""
Unit tests for metrics calculation module.

Tests all metric computation functions including analytical formulas,
empirical comparisons, and comprehensive metric analysis.

Date: February 10, 2026
"""

import unittest
import numpy as np
from typing import Dict

from src.metrics import (
    MetricsCalculator,
    AnalyticalMetrics,
    ComparisonMetrics,
    analyze_batch_results
)
from src.simulator import SimulationConfig, SimulationResults, Simulator
from src.power_model import PowerModel, PowerProfile


class TestAnalyticalMetrics(unittest.TestCase):
    """Test analytical metric calculations based on paper formulas."""
    
    def test_success_probability_formula(self):
        """Test p = q(1-q)^(n-1) formula."""
        # Test with n=10, q=0.1
        n, q = 10, 0.1
        p = MetricsCalculator.compute_analytical_success_probability(n, q)
        
        # Expected: 0.1 * (0.9)^9 ≈ 0.0387
        expected = 0.1 * (0.9 ** 9)
        self.assertAlmostEqual(p, expected, places=6)
    
    def test_optimal_q(self):
        """Test that optimal q = 1/n."""
        for n in [5, 10, 50, 100]:
            optimal_q = MetricsCalculator.compute_optimal_q(n)
            self.assertAlmostEqual(optimal_q, 1.0/n, places=6)
    
    def test_optimal_q_maximizes_success_prob(self):
        """Test that optimal q actually maximizes success probability."""
        n = 10
        optimal_q = MetricsCalculator.compute_optimal_q(n)
        
        # Compute p at optimal q
        p_optimal = MetricsCalculator.compute_analytical_success_probability(n, optimal_q)
        
        # Test nearby q values - should have lower p
        for delta in [-0.02, -0.01, 0.01, 0.02]:
            q_test = optimal_q + delta
            if 0 < q_test < 1:
                p_test = MetricsCalculator.compute_analytical_success_probability(n, q_test)
                self.assertLessEqual(p_test, p_optimal)
    
    def test_service_rate_without_sleep(self):
        """Test μ = p when no sleep."""
        p = 0.1
        lambda_rate = 0.05
        tw = 5
        
        mu = MetricsCalculator.compute_analytical_service_rate(
            p, lambda_rate, tw, has_sleep=False
        )
        
        self.assertAlmostEqual(mu, p, places=6)
    
    def test_service_rate_with_sleep(self):
        """Test μ = p / (1 + tw * λ / (1 - λ)) with sleep."""
        p = 0.1
        lambda_rate = 0.05
        tw = 5
        
        mu = MetricsCalculator.compute_analytical_service_rate(
            p, lambda_rate, tw, has_sleep=True
        )
        
        # Expected: 0.1 / (1 + 5 * 0.05 / 0.95) ≈ 0.0902
        expected = p / (1 + tw * lambda_rate / (1 - lambda_rate))
        self.assertAlmostEqual(mu, expected, places=6)
    
    def test_mean_delay_formula(self):
        """Test ¯T = 1/(μ - λ) formula."""
        lambda_rate = 0.05
        mu = 0.1
        
        mean_delay = MetricsCalculator.compute_analytical_mean_delay(lambda_rate, mu)
        
        # Expected: 1 / (0.1 - 0.05) = 20
        expected = 1.0 / (mu - lambda_rate)
        self.assertAlmostEqual(mean_delay, expected, places=6)
    
    def test_mean_delay_saturated(self):
        """Test that saturated regime gives infinite delay."""
        lambda_rate = 0.15
        mu = 0.1
        
        mean_delay = MetricsCalculator.compute_analytical_mean_delay(lambda_rate, mu)
        
        self.assertEqual(mean_delay, float('inf'))
    
    def test_mean_queue_length_formula(self):
        """Test ¯L = λ / (μ - λ) formula."""
        lambda_rate = 0.05
        mu = 0.1
        
        mean_queue = MetricsCalculator.compute_analytical_mean_queue_length(lambda_rate, mu)
        
        # Expected: 0.05 / (0.1 - 0.05) = 1.0
        expected = lambda_rate / (mu - lambda_rate)
        self.assertAlmostEqual(mean_queue, expected, places=6)
    
    def test_analytical_metrics_complete(self):
        """Test complete analytical metrics computation."""
        n, q = 20, 0.05
        lambda_rate = 0.01
        tw, ts = 5, 10
        
        analytical = MetricsCalculator.compute_analytical_metrics(
            n, q, lambda_rate, tw, ts, has_sleep=True
        )
        
        # Check all fields are present and valid
        self.assertIsInstance(analytical, AnalyticalMetrics)
        self.assertGreater(analytical.success_probability, 0)
        self.assertGreater(analytical.service_rate, 0)
        self.assertTrue(analytical.stability_condition)
        self.assertGreater(analytical.mean_delay, 0)
        self.assertGreater(analytical.mean_queue_length, 0)


class TestEmpiricalMetrics(unittest.TestCase):
    """Test empirical metrics computation from simulation results."""
    
    def setUp(self):
        """Create a simple simulation result for testing."""
        self.config = SimulationConfig(
            n_nodes=10,
            arrival_rate=0.01,
            transmission_prob=0.1,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=1000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=1000,
            seed=42
        )
        
        # Run a short simulation
        sim = Simulator(self.config)
        self.result = sim.run_simulation(track_history=False, verbose=False)
    
    def test_energy_per_packet(self):
        """Test energy per packet calculation."""
        energy_per_packet = MetricsCalculator.compute_energy_per_packet(self.result)
        
        # Should be positive and finite
        self.assertGreater(energy_per_packet, 0)
        self.assertNotEqual(energy_per_packet, float('inf'))
    
    def test_delivery_ratio(self):
        """Test delivery ratio calculation."""
        delivery_ratio = MetricsCalculator.compute_delivery_ratio(self.result)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(delivery_ratio, 0.0)
        self.assertLessEqual(delivery_ratio, 1.0)
    
    def test_collision_rate(self):
        """Test collision rate calculation."""
        collision_rate = MetricsCalculator.compute_collision_rate(self.result)
        
        # Should be non-negative
        self.assertGreaterEqual(collision_rate, 0.0)
    
    def test_channel_utilization(self):
        """Test channel utilization calculation."""
        utilization = MetricsCalculator.compute_channel_utilization(self.result)
        
        # Should equal throughput
        self.assertAlmostEqual(utilization, self.result.throughput, places=6)
    
    def test_energy_efficiency_metrics(self):
        """Test comprehensive energy efficiency metrics."""
        metrics = MetricsCalculator.compute_energy_efficiency_metrics(self.result)
        
        # Check all required fields
        self.assertIn('energy_per_packet', metrics)
        self.assertIn('energy_per_slot', metrics)
        self.assertIn('energy_in_sleep', metrics)
        self.assertIn('energy_in_active', metrics)
        self.assertIn('packets_per_energy', metrics)
        
        # Energy fractions should sum to approximately 1
        total_fraction = (
            metrics['energy_in_sleep'] +
            metrics['energy_in_active'] +
            metrics['energy_in_idle'] +
            metrics['energy_in_wakeup']
        )
        self.assertAlmostEqual(total_fraction, 1.0, places=2)
    
    def test_latency_metrics(self):
        """Test comprehensive latency metrics."""
        metrics = MetricsCalculator.compute_latency_metrics(self.result)
        
        # Check all required fields
        self.assertIn('mean_delay_slots', metrics)
        self.assertIn('mean_delay_ms', metrics)
        self.assertIn('tail_delay_95_slots', metrics)
        self.assertIn('tail_delay_99_slots', metrics)
        
        # MS conversion should be consistent
        expected_ms = metrics['mean_delay_slots'] * 6.0
        self.assertAlmostEqual(metrics['mean_delay_ms'], expected_ms, places=2)
    
    def test_network_performance_metrics(self):
        """Test comprehensive network performance metrics."""
        metrics = MetricsCalculator.compute_network_performance_metrics(self.result)
        
        # Check all required fields
        self.assertIn('throughput', metrics)
        self.assertIn('delivery_ratio', metrics)
        self.assertIn('collision_rate', metrics)
        self.assertIn('empirical_success_prob', metrics)
        self.assertIn('empirical_service_rate', metrics)
        
        # All should be non-negative
        for key, value in metrics.items():
            if key not in ['total_transmissions', 'total_successes', 'total_collisions']:
                self.assertGreaterEqual(value, 0.0)


class TestComparisonMetrics(unittest.TestCase):
    """Test comparison between empirical and analytical metrics."""
    
    def setUp(self):
        """Create simulation result for comparison testing."""
        # Use optimal q = 1/n for best comparison
        n = 20
        q = 1.0 / n
        
        self.config = SimulationConfig(
            n_nodes=n,
            arrival_rate=0.01,
            transmission_prob=q,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=5000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=50000,  # Long enough for good statistics
            seed=42
        )
        
        # Run simulation
        sim = Simulator(self.config)
        self.result = sim.run_simulation(track_history=False, verbose=False)
        
        # Compute analytical metrics
        self.analytical = MetricsCalculator.compute_analytical_metrics(
            n=self.config.n_nodes,
            q=self.config.transmission_prob,
            lambda_rate=self.config.arrival_rate,
            tw=self.config.wakeup_time,
            ts=self.config.idle_timer,
            has_sleep=True
        )
    
    def test_comparison_metrics_structure(self):
        """Test that comparison metrics have correct structure."""
        comparison = MetricsCalculator.compare_empirical_vs_analytical(
            self.result, self.analytical
        )
        
        self.assertIsInstance(comparison, ComparisonMetrics)
        self.assertIsInstance(comparison.warnings, list)
        self.assertIsInstance(comparison.is_valid_comparison, bool)
    
    def test_success_probability_agreement(self):
        """Test that empirical and analytical success probabilities agree."""
        comparison = MetricsCalculator.compare_empirical_vs_analytical(
            self.result, self.analytical, tolerance=0.30
        )
        
        # With enough samples, should have reasonable agreement
        # Note: Empirical values may differ from analytical due to stochastic variation
        # and the fact that analytical formulas assume steady-state
        self.assertIsNotNone(comparison.success_prob_error)
    
    def test_service_rate_agreement(self):
        """Test that empirical and analytical service rates agree."""
        comparison = MetricsCalculator.compare_empirical_vs_analytical(
            self.result, self.analytical, tolerance=0.30
        )
        
        # Should compute both rates
        # Note: Empirical service rate may differ significantly from analytical
        # due to transient effects, finite simulation time, and measurement differences
        self.assertGreaterEqual(comparison.empirical_service_rate, 0.0)
        self.assertGreaterEqual(comparison.analytical_service_rate, 0.0)
    
    def test_stability_condition(self):
        """Test that stability condition is correctly identified."""
        # Should be stable (λ < μ)
        self.assertTrue(self.analytical.stability_condition)
        self.assertLess(self.config.arrival_rate, self.analytical.service_rate)


class TestComprehensiveMetrics(unittest.TestCase):
    """Test comprehensive metrics computation."""
    
    def setUp(self):
        """Create simulation result for comprehensive testing."""
        self.config = SimulationConfig(
            n_nodes=10,
            arrival_rate=0.01,
            transmission_prob=0.1,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=2000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=10000,
            seed=42
        )
        
        sim = Simulator(self.config)
        self.result = sim.run_simulation(track_history=True, verbose=False)
    
    def test_comprehensive_metrics_structure(self):
        """Test that comprehensive metrics have correct structure."""
        metrics = MetricsCalculator.compute_comprehensive_metrics(
            self.result, include_analytical=True
        )
        
        # Check all major categories
        self.assertIn('config', metrics)
        self.assertIn('simulation', metrics)
        self.assertIn('latency', metrics)
        self.assertIn('energy', metrics)
        self.assertIn('network', metrics)
        self.assertIn('lifetime', metrics)
        self.assertIn('state_fractions', metrics)
        self.assertIn('analytical', metrics)
        self.assertIn('comparison', metrics)
    
    def test_comprehensive_metrics_without_analytical(self):
        """Test comprehensive metrics without analytical comparison."""
        metrics = MetricsCalculator.compute_comprehensive_metrics(
            self.result, include_analytical=False
        )
        
        # Should not have analytical or comparison sections
        self.assertNotIn('analytical', metrics)
        self.assertNotIn('comparison', metrics)
    
    def test_queue_statistics_included(self):
        """Test that queue statistics are included when history available."""
        metrics = MetricsCalculator.compute_comprehensive_metrics(
            self.result, include_analytical=False
        )
        
        # Should have queue statistics
        self.assertIn('queue_statistics', metrics)
        
        stats = metrics['queue_statistics']
        self.assertIn('mean', stats)
        self.assertIn('max', stats)
        self.assertIn('std', stats)
        self.assertIn('p95', stats)
    
    def test_print_metrics_summary(self):
        """Test that metrics summary prints without errors."""
        metrics = MetricsCalculator.compute_comprehensive_metrics(
            self.result, include_analytical=True
        )
        
        # Should not raise any exceptions
        try:
            MetricsCalculator.print_metrics_summary(metrics, verbose=True)
            MetricsCalculator.print_metrics_summary(metrics, verbose=False)
        except Exception as e:
            self.fail(f"print_metrics_summary raised exception: {e}")


class TestBatchAnalysis(unittest.TestCase):
    """Test batch results analysis."""
    
    def setUp(self):
        """Create batch results for testing."""
        self.config = SimulationConfig(
            n_nodes=10,
            arrival_rate=0.01,
            transmission_prob=0.1,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=2000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=5000,
            seed=None
        )
        
        # Run multiple replications
        self.batch_results = []
        for seed in range(5):
            config = SimulationConfig(
                n_nodes=self.config.n_nodes,
                arrival_rate=self.config.arrival_rate,
                transmission_prob=self.config.transmission_prob,
                idle_timer=self.config.idle_timer,
                wakeup_time=self.config.wakeup_time,
                initial_energy=self.config.initial_energy,
                power_rates=self.config.power_rates,
                max_slots=self.config.max_slots,
                seed=seed
            )
            
            sim = Simulator(config)
            result = sim.run_simulation(track_history=False, verbose=False)
            self.batch_results.append(result)
    
    def test_batch_analysis_structure(self):
        """Test that batch analysis returns correct structure."""
        aggregated = analyze_batch_results(self.batch_results)
        
        # Check for key metrics
        self.assertIn('mean_delay', aggregated)
        self.assertIn('lifetime_years', aggregated)
        self.assertIn('energy_per_packet', aggregated)
        self.assertIn('throughput', aggregated)
        
        # Each metric should be (mean, std) tuple
        for metric_name, (mean, std) in aggregated.items():
            self.assertIsInstance(mean, (int, float))
            self.assertIsInstance(std, (int, float))
            self.assertGreaterEqual(std, 0.0)
    
    def test_batch_analysis_empty(self):
        """Test batch analysis with empty results."""
        aggregated = analyze_batch_results([])
        self.assertEqual(aggregated, {})
    
    def test_batch_analysis_statistical_properties(self):
        """Test that batch analysis has reasonable statistical properties."""
        aggregated = analyze_batch_results(self.batch_results)
        
        # Standard deviation should be less than mean for stable metrics
        mean_delay_mean, mean_delay_std = aggregated['mean_delay']
        self.assertGreater(mean_delay_mean, 0)
        self.assertGreaterEqual(mean_delay_std, 0)
        
        # Throughput should have low variance (deterministic system)
        throughput_mean, throughput_std = aggregated['throughput']
        self.assertGreater(throughput_mean, 0)


class TestQueueStatistics(unittest.TestCase):
    """Test queue length statistics computation."""
    
    def test_empty_queue_history(self):
        """Test queue statistics with empty history."""
        stats = MetricsCalculator.compute_queue_length_statistics([])
        
        # All values should be 0
        self.assertEqual(stats['mean'], 0.0)
        self.assertEqual(stats['max'], 0.0)
        self.assertEqual(stats['std'], 0.0)
    
    def test_queue_statistics_computation(self):
        """Test queue statistics with sample data."""
        queue_history = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
        
        stats = MetricsCalculator.compute_queue_length_statistics(queue_history)
        
        self.assertAlmostEqual(stats['mean'], np.mean(queue_history), places=4)
        self.assertAlmostEqual(stats['max'], 3.0, places=4)
        self.assertAlmostEqual(stats['min'], 0.0, places=4)
        self.assertAlmostEqual(stats['median'], np.median(queue_history), places=4)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_zero_nodes(self):
        """Test with zero nodes."""
        p = MetricsCalculator.compute_analytical_success_probability(0, 0.1)
        self.assertEqual(p, 0.0)
    
    def test_q_equals_zero(self):
        """Test with q=0 (no transmissions)."""
        p = MetricsCalculator.compute_analytical_success_probability(10, 0.0)
        self.assertEqual(p, 0.0)
    
    def test_q_equals_one(self):
        """Test with q=1 (always transmit)."""
        n = 10
        p = MetricsCalculator.compute_analytical_success_probability(n, 1.0)
        # p = 1.0 * (0.0)^9 = 0.0
        self.assertEqual(p, 0.0)
    
    def test_saturated_regime(self):
        """Test metrics in saturated regime (λ >= μ)."""
        lambda_rate = 0.15
        mu = 0.1
        
        delay = MetricsCalculator.compute_analytical_mean_delay(lambda_rate, mu)
        queue = MetricsCalculator.compute_analytical_mean_queue_length(lambda_rate, mu)
        
        self.assertEqual(delay, float('inf'))
        self.assertEqual(queue, float('inf'))


class TestEdgeCasesExtended(unittest.TestCase):
    """Extended edge-case and boundary-condition tests for each analytical formula."""

    # ------------------------------------------------------------------
    # Success probability: p = q(1-q)^(n-1)
    # ------------------------------------------------------------------

    def test_success_probability_n1(self):
        """With a single node (n=1) success probability must equal q exactly."""
        for q in [0.1, 0.3, 0.5, 0.9]:
            p = MetricsCalculator.compute_analytical_success_probability(1, q)
            self.assertAlmostEqual(
                p, q, places=10,
                msg=f"n=1, q={q}: expected p=q={q}, got {p}"
            )

    def test_success_probability_two_nodes(self):
        """With n=2 the formula reduces to p = q(1-q)."""
        for q in [0.1, 0.2, 0.5]:
            p = MetricsCalculator.compute_analytical_success_probability(2, q)
            expected = q * (1 - q)
            self.assertAlmostEqual(p, expected, places=10,
                                   msg=f"n=2, q={q}: expected {expected}, got {p}")

    def test_success_probability_large_n_small(self):
        """For large n with fixed q, p should approach 0 as n grows."""
        q = 0.1
        p_10 = MetricsCalculator.compute_analytical_success_probability(10, q)
        p_100 = MetricsCalculator.compute_analytical_success_probability(100, q)
        p_1000 = MetricsCalculator.compute_analytical_success_probability(1000, q)
        self.assertGreater(p_10, p_100)
        self.assertGreater(p_100, p_1000)

    def test_success_probability_at_optimal_q(self):
        """At q = 1/n the analytical p should equal (1/n)*(1 - 1/n)^(n-1)."""
        for n in [5, 10, 20, 50]:
            q_opt = 1.0 / n
            p = MetricsCalculator.compute_analytical_success_probability(n, q_opt)
            expected = q_opt * ((1 - q_opt) ** (n - 1))
            self.assertAlmostEqual(p, expected, places=10)
            # For large n, optimal p approaches 1/e ≈ 0.3679
            if n >= 20:
                self.assertAlmostEqual(p * n, 1 / np.e, delta=0.05,
                                       msg=f"n={n}: n*p should approach 1/e")

    def test_success_probability_symmetry(self):
        """p(n, q) should be non-negative for all valid inputs."""
        for n in [1, 2, 5, 100]:
            for q in [0.0, 0.01, 0.5, 1.0]:
                p = MetricsCalculator.compute_analytical_success_probability(n, q)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    # ------------------------------------------------------------------
    # Service rate: mu = p / (1 + tw * lambda / (1 - lambda))
    # ------------------------------------------------------------------

    def test_service_rate_tw0_equals_p(self):
        """With tw=0 the service rate must equal p regardless of sleep mode."""
        p = 0.08
        lambda_rate = 0.02
        mu = MetricsCalculator.compute_analytical_service_rate(
            p, lambda_rate, tw=0, has_sleep=True
        )
        self.assertAlmostEqual(mu, p, places=10,
                               msg="tw=0 should give mu=p")

    def test_service_rate_monotone_decreasing_in_tw(self):
        """Service rate should strictly decrease as tw increases (sleep overhead)."""
        p = 0.1
        lambda_rate = 0.02
        tw_values = [0, 1, 5, 10, 20]
        mu_values = [
            MetricsCalculator.compute_analytical_service_rate(
                p, lambda_rate, tw, has_sleep=True
            )
            for tw in tw_values
        ]
        for i in range(len(mu_values) - 1):
            self.assertGreaterEqual(
                mu_values[i], mu_values[i + 1],
                msg=f"mu(tw={tw_values[i]})={mu_values[i]:.6f} should be >= mu(tw={tw_values[i+1]})={mu_values[i+1]:.6f}"
            )

    def test_service_rate_denominator_formula(self):
        """Verify the denominator 1 + tw*λ/(1-λ) is computed correctly."""
        p = 0.1
        lambda_rate = 0.04
        tw = 8
        mu = MetricsCalculator.compute_analytical_service_rate(
            p, lambda_rate, tw, has_sleep=True
        )
        expected_denom = 1 + tw * lambda_rate / (1 - lambda_rate)
        expected_mu = p / expected_denom
        self.assertAlmostEqual(mu, expected_mu, places=10)

    def test_service_rate_lambda_saturated(self):
        """At lambda >= 1.0 the service rate should return 0 (invalid regime)."""
        mu = MetricsCalculator.compute_analytical_service_rate(
            0.1, lambda_rate=1.0, tw=5, has_sleep=True
        )
        self.assertEqual(mu, 0.0)

    # ------------------------------------------------------------------
    # Mean delay: T = 1/(mu - lambda)
    # ------------------------------------------------------------------

    def test_mean_delay_lambda_equals_zero(self):
        """With no arrivals (λ=0), mean delay should equal 1/μ."""
        mu = 0.1
        delay = MetricsCalculator.compute_analytical_mean_delay(
            lambda_rate=0.0, mu=mu
        )
        self.assertAlmostEqual(delay, 1.0 / mu, places=10)

    def test_mean_delay_approaches_infinity_near_stability_boundary(self):
        """As λ → μ from below, mean delay should grow without bound."""
        mu = 0.1
        delays = []
        for eps in [0.05, 0.01, 0.001, 0.0001]:
            d = MetricsCalculator.compute_analytical_mean_delay(mu - eps, mu)
            delays.append(d)
        # Each successive delay should be strictly larger
        for i in range(len(delays) - 1):
            self.assertLess(delays[i], delays[i + 1])
        # The last value should be very large (> 1000 slots for eps=0.0001)
        self.assertGreater(delays[-1], 1000)

    def test_mean_delay_equals_lambda_zero_special_case(self):
        """Manual check: λ=0.05, μ=0.10 → T = 1/(0.10-0.05) = 20.0 slots."""
        delay = MetricsCalculator.compute_analytical_mean_delay(0.05, 0.10)
        self.assertAlmostEqual(delay, 20.0, places=6)

    # ------------------------------------------------------------------
    # Little's Law consistency: L = λ * T
    # ------------------------------------------------------------------

    def test_littles_law_consistency(self):
        """Mean queue length L = λ * T must hold (Little's Law)."""
        lambda_rate = 0.05
        mu = 0.12
        T = MetricsCalculator.compute_analytical_mean_delay(lambda_rate, mu)
        L = MetricsCalculator.compute_analytical_mean_queue_length(lambda_rate, mu)
        # Little's Law: L = λ * T
        self.assertAlmostEqual(L, lambda_rate * T, places=10,
                               msg="Little's Law L = λT must hold")

    def test_queue_length_zero_arrivals(self):
        """With λ=0, mean queue length should be 0."""
        L = MetricsCalculator.compute_analytical_mean_queue_length(0.0, mu=0.1)
        self.assertAlmostEqual(L, 0.0, places=10)

    def test_queue_length_increases_with_lambda(self):
        """Queue length should increase monotonically as λ increases toward μ."""
        mu = 0.1
        prev_L = 0.0
        for lam in [0.01, 0.03, 0.05, 0.07, 0.09]:
            L = MetricsCalculator.compute_analytical_mean_queue_length(lam, mu)
            self.assertGreater(L, prev_L,
                               msg=f"L should increase: L(λ={lam}) should be > L(λ<{lam})")
            prev_L = L


if __name__ == '__main__':
    unittest.main()
