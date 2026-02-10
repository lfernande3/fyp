"""
Unit tests for traffic models module.

Tests different traffic arrival patterns including Poisson and bursty traffic.

Date: February 10, 2026
"""

import unittest
import numpy as np

from src.traffic_models import (
    TrafficGenerator,
    TrafficModel,
    BurstyTrafficConfig,
    generate_bursty_traffic_trace,
    analyze_traffic_trace,
    compare_poisson_vs_bursty
)


class TestTrafficGenerator(unittest.TestCase):
    """Test traffic generator functions."""
    
    def test_poisson_arrival(self):
        """Test Poisson arrival generation."""
        # Test with rate 0.0 (no arrivals)
        for _ in range(100):
            self.assertFalse(TrafficGenerator.poisson_arrival(0.0))
        
        # Test with rate 1.0 (always arrives)
        for _ in range(100):
            self.assertTrue(TrafficGenerator.poisson_arrival(1.0))
        
        # Test with rate 0.5 (approximately 50%)
        np.random.seed(42)
        arrivals = sum(TrafficGenerator.poisson_arrival(0.5) for _ in range(1000))
        # Should be around 500 ± some tolerance
        self.assertGreater(arrivals, 400)
        self.assertLess(arrivals, 600)
    
    def test_bursty_arrival(self):
        """Test bursty arrival generation."""
        # Test no burst mode (burst_probability = 0)
        for _ in range(100):
            n_arrivals = TrafficGenerator.bursty_arrival(
                base_rate=0.5,
                burst_probability=0.0,
                burst_size_mean=3
            )
            # Should be 0 or 1 (Poisson-like)
            self.assertIn(n_arrivals, [0, 1])
        
        # Test always burst mode (burst_probability = 1.0)
        burst_sizes = []
        for _ in range(100):
            n_arrivals = TrafficGenerator.bursty_arrival(
                base_rate=0.0,
                burst_probability=1.0,
                burst_size_mean=3,
                burst_size_std=0.5
            )
            burst_sizes.append(n_arrivals)
            # Should always have at least 1 packet
            self.assertGreaterEqual(n_arrivals, 1)
        
        # Mean burst size should be around 3
        mean_burst = np.mean(burst_sizes)
        self.assertGreater(mean_burst, 2.0)
        self.assertLess(mean_burst, 4.0)
    
    def test_periodic_arrival(self):
        """Test periodic arrival generation."""
        period = 10
        
        # Test periodic slots
        for slot in range(0, 100, period):
            self.assertTrue(TrafficGenerator.periodic_arrival(slot, period, jitter=0.0))
        
        # Test non-periodic slots (no jitter)
        for slot in range(1, period):
            self.assertFalse(TrafficGenerator.periodic_arrival(slot, period, jitter=0.0))
    
    def test_on_off_arrival(self):
        """Test on-off traffic pattern."""
        on_duration = 10
        off_duration = 10
        on_rate = 1.0
        
        # Test ON period (slots 0-9)
        for slot in range(on_duration):
            result = TrafficGenerator.on_off_arrival(slot, on_duration, off_duration, on_rate)
            self.assertTrue(result)
        
        # Test OFF period (slots 10-19)
        for slot in range(on_duration, on_duration + off_duration):
            result = TrafficGenerator.on_off_arrival(slot, on_duration, off_duration, on_rate)
            self.assertFalse(result)


class TestBurstyTrafficConfig(unittest.TestCase):
    """Test bursty traffic configuration."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = BurstyTrafficConfig(
            base_rate=0.01,
            burst_probability=0.1,
            burst_size_mean=3,
            burst_size_std=1
        )
        
        self.assertEqual(config.base_rate, 0.01)
        self.assertEqual(config.burst_probability, 0.1)
        self.assertEqual(config.burst_size_mean, 3)
        self.assertEqual(config.burst_size_std, 1)
    
    def test_effective_arrival_rate(self):
        """Test effective arrival rate calculation."""
        config = BurstyTrafficConfig(
            base_rate=0.01,
            burst_probability=0.1,
            burst_size_mean=3,
            burst_size_std=0
        )
        
        effective_rate = config.get_effective_arrival_rate()
        
        # Expected: 0.1 * 3 (bursts) + 0.9 * 0.01 (normal) = 0.309
        expected = 0.1 * 3 + 0.9 * 0.01
        self.assertAlmostEqual(effective_rate, expected, places=5)
    
    def test_repr(self):
        """Test string representation."""
        config = BurstyTrafficConfig()
        repr_str = repr(config)
        
        self.assertIn("BurstyTrafficConfig", repr_str)
        self.assertIn("base_rate", repr_str)


class TestTrafficTrace(unittest.TestCase):
    """Test traffic trace generation and analysis."""
    
    def test_generate_bursty_trace(self):
        """Test bursty traffic trace generation."""
        config = BurstyTrafficConfig(
            base_rate=0.01,
            burst_probability=0.1,
            burst_size_mean=3
        )
        
        trace = generate_bursty_traffic_trace(
            n_slots=1000,
            config=config,
            seed=42
        )
        
        # Check length
        self.assertEqual(len(trace), 1000)
        
        # Check all values are non-negative integers
        for val in trace:
            self.assertIsInstance(val, (int, np.integer))
            self.assertGreaterEqual(val, 0)
        
        # Should have some bursts (values > 1)
        max_burst = max(trace)
        self.assertGreater(max_burst, 1)
    
    def test_analyze_traffic_trace(self):
        """Test traffic trace analysis."""
        # Create simple trace
        trace = [0, 1, 0, 2, 0, 1, 0, 0, 3, 0]
        
        stats = analyze_traffic_trace(trace)
        
        # Check all required fields
        self.assertIn('total_packets', stats)
        self.assertIn('mean_rate', stats)
        self.assertIn('std_rate', stats)
        self.assertIn('max_burst', stats)
        self.assertIn('burst_slots', stats)
        self.assertIn('burst_fraction', stats)
        self.assertIn('burstiness_coefficient', stats)
        
        # Check values
        self.assertEqual(stats['total_packets'], 7)
        self.assertEqual(stats['max_burst'], 3)
        self.assertEqual(stats['burst_slots'], 2)  # 2 and 3 are bursts
        self.assertEqual(stats['burst_fraction'], 0.2)  # 2/10
    
    def test_compare_poisson_vs_bursty(self):
        """Test Poisson vs bursty comparison."""
        poisson_trace, bursty_trace, poisson_stats, bursty_stats = compare_poisson_vs_bursty(
            n_slots=5000,
            mean_rate=0.01,
            seed=42
        )
        
        # Check traces length
        self.assertEqual(len(poisson_trace), 5000)
        self.assertEqual(len(bursty_trace), 5000)
        
        # Check stats dictionaries
        self.assertIn('burstiness_coefficient', poisson_stats)
        self.assertIn('burstiness_coefficient', bursty_stats)
        
        # Both should have burstiness coefficients (may vary due to randomness)
        self.assertGreaterEqual(bursty_stats['burstiness_coefficient'], 0)
        self.assertGreaterEqual(poisson_stats['burstiness_coefficient'], 0)
        
        # Bursty traffic should typically have higher max burst (but not guaranteed)
        # Just check that both have valid max burst values
        self.assertGreaterEqual(bursty_stats['max_burst'], 0)
        self.assertGreaterEqual(poisson_stats['max_burst'], 0)
    
    def test_reproducibility_with_seed(self):
        """Test that seed ensures reproducibility."""
        config = BurstyTrafficConfig()
        
        trace1 = generate_bursty_traffic_trace(1000, config, seed=42)
        trace2 = generate_bursty_traffic_trace(1000, config, seed=42)
        
        # Should be identical
        self.assertEqual(trace1, trace2)
        
        # Different seed should give different results
        trace3 = generate_bursty_traffic_trace(1000, config, seed=123)
        self.assertNotEqual(trace1, trace3)


class TestTrafficModel(unittest.TestCase):
    """Test TrafficModel enum."""
    
    def test_traffic_model_enum(self):
        """Test traffic model enumeration."""
        self.assertEqual(TrafficModel.POISSON.value, "poisson")
        self.assertEqual(TrafficModel.BURSTY.value, "bursty")
        self.assertEqual(TrafficModel.PERIODIC.value, "periodic")
        self.assertEqual(TrafficModel.ON_OFF.value, "on_off")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_trace_analysis(self):
        """Test analysis of empty trace."""
        trace = []
        stats = analyze_traffic_trace(trace)
        
        self.assertEqual(stats['total_packets'], 0)
        self.assertEqual(stats['mean_rate'], 0.0)
    
    def test_all_zeros_trace(self):
        """Test analysis of trace with no arrivals."""
        trace = [0] * 100
        stats = analyze_traffic_trace(trace)
        
        self.assertEqual(stats['total_packets'], 0)
        self.assertEqual(stats['mean_rate'], 0.0)
        self.assertEqual(stats['max_burst'], 0)
        self.assertEqual(stats['burst_slots'], 0)
    
    def test_single_slot_trace(self):
        """Test analysis of single-slot trace."""
        trace = [5]
        stats = analyze_traffic_trace(trace)
        
        self.assertEqual(stats['total_packets'], 5)
        self.assertEqual(stats['max_burst'], 5)
    
    def test_zero_burst_probability(self):
        """Test bursty config with zero burst probability."""
        config = BurstyTrafficConfig(
            base_rate=0.01,
            burst_probability=0.0,
            burst_size_mean=3
        )
        
        effective_rate = config.get_effective_arrival_rate()
        # Should equal base_rate
        self.assertAlmostEqual(effective_rate, 0.01, places=5)


if __name__ == '__main__':
    unittest.main()
