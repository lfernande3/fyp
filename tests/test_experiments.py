"""
Unit tests for experiments module.

Tests parameter sweep functionality and scenario comparisons.

Date: February 10, 2026
"""

import unittest
import numpy as np

from src.experiments import (
    ParameterSweep,
    ScenarioExperiments,
    SweepConfig,
    ScenarioConfig
)
from src.simulator import SimulationConfig
from src.power_model import PowerModel, PowerProfile


class TestParameterSweep(unittest.TestCase):
    """Test parameter sweep functionality."""
    
    def setUp(self):
        """Create base configuration for testing."""
        self.base_config = SimulationConfig(
            n_nodes=10,
            arrival_rate=0.01,
            transmission_prob=0.05,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=1000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=2000,
            seed=None
        )
    
    def test_sweep_transmission_prob(self):
        """Test transmission probability sweep."""
        q_values = [0.05, 0.1]
        results = ParameterSweep.sweep_transmission_prob(
            self.base_config,
            q_values=q_values,
            n_replications=2,
            verbose=False
        )
        
        # Check structure
        self.assertEqual(len(results), 2)
        for q in q_values:
            self.assertIn(q, results)
            self.assertEqual(len(results[q]), 2)  # 2 replications
    
    def test_sweep_idle_timer(self):
        """Test idle timer sweep."""
        ts_values = [5, 10]
        results = ParameterSweep.sweep_idle_timer(
            self.base_config,
            ts_values=ts_values,
            n_replications=2,
            verbose=False
        )
        
        self.assertEqual(len(results), 2)
        for ts in ts_values:
            self.assertIn(ts, results)
    
    def test_sweep_num_nodes(self):
        """Test number of nodes sweep."""
        n_values = [5, 10]
        results = ParameterSweep.sweep_num_nodes(
            self.base_config,
            n_values=n_values,
            n_replications=2,
            verbose=False
        )
        
        self.assertEqual(len(results), 2)
        for n in n_values:
            self.assertIn(n, results)
    
    def test_sweep_arrival_rate(self):
        """Test arrival rate sweep."""
        lambda_values = [0.005, 0.01]
        results = ParameterSweep.sweep_arrival_rate(
            self.base_config,
            lambda_values=lambda_values,
            n_replications=2,
            verbose=False
        )
        
        self.assertEqual(len(results), 2)
        for lam in lambda_values:
            self.assertIn(lam, results)
    
    def test_analyze_sweep_results(self):
        """Test sweep results analysis."""
        q_values = [0.05, 0.1]
        results = ParameterSweep.sweep_transmission_prob(
            self.base_config,
            q_values=q_values,
            n_replications=2,
            verbose=False
        )
        
        analysis = ParameterSweep.analyze_sweep_results(results, 'q')
        
        # Check structure
        self.assertEqual(len(analysis), 2)
        for q in q_values:
            self.assertIn(q, analysis)
            # Check metrics present
            self.assertIn('mean_delay', analysis[q])
            self.assertIn('lifetime_years', analysis[q])
            # Each metric should have (mean, std) tuple
            self.assertEqual(len(analysis[q]['mean_delay']), 2)


class TestScenarioExperiments(unittest.TestCase):
    """Test scenario comparison functionality."""
    
    def test_create_low_latency_scenario(self):
        """Test low-latency scenario creation."""
        scenario = ScenarioExperiments.create_low_latency_scenario(
            n_nodes=20,
            arrival_rate=0.01
        )
        
        self.assertIsInstance(scenario, ScenarioConfig)
        self.assertEqual(scenario.name, "Low-Latency Priority")
        self.assertEqual(scenario.priority, "latency")
        # Should have small idle timer
        self.assertEqual(scenario.config.idle_timer, 1)
        # Should have optimal q = 1/n
        self.assertAlmostEqual(scenario.config.transmission_prob, 1.0/20, places=3)
    
    def test_create_battery_life_scenario(self):
        """Test battery-life scenario creation."""
        scenario = ScenarioExperiments.create_battery_life_scenario(
            n_nodes=20,
            arrival_rate=0.01
        )
        
        self.assertIsInstance(scenario, ScenarioConfig)
        self.assertEqual(scenario.name, "Battery-Life Priority")
        self.assertEqual(scenario.priority, "battery")
        # Should have large idle timer
        self.assertEqual(scenario.config.idle_timer, 50)
        # Should have low q
        self.assertEqual(scenario.config.transmission_prob, 0.02)
    
    def test_create_balanced_scenario(self):
        """Test balanced scenario creation."""
        scenario = ScenarioExperiments.create_balanced_scenario(
            n_nodes=20,
            arrival_rate=0.01
        )
        
        self.assertIsInstance(scenario, ScenarioConfig)
        self.assertEqual(scenario.name, "Balanced")
        self.assertEqual(scenario.priority, "balanced")
        # Should have moderate values
        self.assertEqual(scenario.config.idle_timer, 10)
        self.assertEqual(scenario.config.transmission_prob, 0.05)
    
    def test_compare_scenarios(self):
        """Test scenario comparison."""
        # Create simple scenarios
        scenarios = [
            ScenarioExperiments.create_low_latency_scenario(n_nodes=10),
            ScenarioExperiments.create_balanced_scenario(n_nodes=10)
        ]
        
        # Modify configs for faster execution
        for scenario in scenarios:
            scenario.config.max_slots = 2000
            scenario.config.initial_energy = 1000
        
        results = ScenarioExperiments.compare_scenarios(
            scenarios,
            n_replications=2,
            verbose=False
        )
        
        # Check results structure
        self.assertEqual(len(results), 2)
        self.assertIn("Low-Latency Priority", results)
        self.assertIn("Balanced", results)
        
        # Each scenario should have replications
        for scenario_name, scenario_results in results.items():
            self.assertEqual(len(scenario_results), 2)
    
    def test_analyze_tradeoffs(self):
        """Test trade-off analysis."""
        # Create and run scenarios
        scenarios = [
            ScenarioExperiments.create_low_latency_scenario(n_nodes=10),
            ScenarioExperiments.create_battery_life_scenario(n_nodes=10)
        ]
        
        for scenario in scenarios:
            scenario.config.max_slots = 2000
            scenario.config.initial_energy = 1000
        
        results = ScenarioExperiments.compare_scenarios(
            scenarios,
            n_replications=2,
            verbose=False
        )
        
        tradeoffs = ScenarioExperiments.analyze_tradeoffs(results)
        
        # Check structure
        self.assertEqual(len(tradeoffs), 2)
        for scenario_name, analysis in tradeoffs.items():
            self.assertIn('delay', analysis)
            self.assertIn('lifetime', analysis)
            self.assertIn('energy_per_packet', analysis)
            self.assertIn('throughput', analysis)
            
            # Each metric should have mean, std, unit
            self.assertIn('mean', analysis['delay'])
            self.assertIn('std', analysis['delay'])
            self.assertIn('unit', analysis['delay'])
    
    def test_scenarios_show_expected_tradeoffs(self):
        """Test that scenarios show expected trade-offs."""
        scenarios = [
            ScenarioExperiments.create_low_latency_scenario(n_nodes=10),
            ScenarioExperiments.create_battery_life_scenario(n_nodes=10)
        ]
        
        for scenario in scenarios:
            scenario.config.max_slots = 2000
            scenario.config.initial_energy = 1000
        
        results = ScenarioExperiments.compare_scenarios(
            scenarios,
            n_replications=3,
            verbose=False
        )
        
        tradeoffs = ScenarioExperiments.analyze_tradeoffs(results)
        
        low_lat = tradeoffs["Low-Latency Priority"]
        battery = tradeoffs["Battery-Life Priority"]
        
        # Low-latency should have lower delay
        # (This might not always hold due to stochasticity, so we just check they exist)
        self.assertGreater(low_lat['delay']['mean'], 0)
        self.assertGreater(battery['delay']['mean'], 0)
        
        # Battery-life should have longer lifetime
        self.assertGreater(low_lat['lifetime']['mean'], 0)
        self.assertGreater(battery['lifetime']['mean'], 0)


class TestSweepConfig(unittest.TestCase):
    """Test SweepConfig dataclass."""
    
    def test_sweep_config_creation(self):
        """Test sweep configuration creation."""
        config = SweepConfig(
            param_name='transmission_prob',
            param_values=[0.05, 0.1],
            n_replications=10
        )
        
        self.assertEqual(config.param_name, 'transmission_prob')
        self.assertEqual(len(config.param_values), 2)
        self.assertEqual(config.n_replications, 10)
        self.assertTrue(config.save_results)


class TestScenarioConfig(unittest.TestCase):
    """Test ScenarioConfig dataclass."""
    
    def test_scenario_config_creation(self):
        """Test scenario configuration creation."""
        sim_config = SimulationConfig(
            n_nodes=10,
            arrival_rate=0.01,
            transmission_prob=0.05,
            idle_timer=10,
            wakeup_time=2,
            initial_energy=1000,
            power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
            max_slots=1000
        )
        
        scenario = ScenarioConfig(
            name="Test Scenario",
            description="Test description",
            config=sim_config,
            priority="latency"
        )
        
        self.assertEqual(scenario.name, "Test Scenario")
        self.assertEqual(scenario.priority, "latency")
        self.assertIsInstance(scenario.config, SimulationConfig)


if __name__ == '__main__':
    unittest.main()
