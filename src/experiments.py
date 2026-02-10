"""
Parameter Sweep Experiments Module

This module provides utilities for systematic parameter sweeps and scenario
analysis for the M2M sleep-based simulator.

Implements Task 2.2: Run parameter sweeps to show impacts of key parameters
(q, ts, n, λ) on delay, lifetime, and energy consumption.

Date: February 10, 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path

from .simulator import Simulator, SimulationConfig, SimulationResults, BatchSimulator
from .power_model import PowerModel, PowerProfile
from .metrics import MetricsCalculator, analyze_batch_results


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep experiment."""
    param_name: str                      # Parameter to sweep
    param_values: List[Any]              # Values to test
    n_replications: int = 20             # Replications per value
    base_config: SimulationConfig = None # Base configuration
    save_results: bool = True            # Save results to file
    output_dir: str = "results"          # Output directory


@dataclass
class ScenarioConfig:
    """Configuration for a prioritization scenario."""
    name: str                            # Scenario name
    description: str                     # Scenario description
    config: SimulationConfig             # Simulation configuration
    priority: str                        # "latency" or "battery"


class ParameterSweep:
    """
    Systematic parameter sweep experiments.
    
    Provides utilities for sweeping transmission probability (q), idle timer (ts),
    number of nodes (n), and arrival rate (λ) to quantify their impacts on
    system performance.
    """
    
    @staticmethod
    def sweep_transmission_prob(
        base_config: SimulationConfig,
        q_values: List[float] = None,
        n_replications: int = 20,
        verbose: bool = True
    ) -> Dict[float, List[SimulationResults]]:
        """
        Sweep transmission probability q.
        
        Tests impact of q on delay, collisions, and energy consumption.
        Higher q typically reduces delay but increases collisions and energy use.
        
        Args:
            base_config: Base simulation configuration
            q_values: List of q values to test (default: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
            n_replications: Number of replications per value
            verbose: Print progress
            
        Returns:
            Dictionary mapping q values to list of results
        """
        if q_values is None:
            q_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        
        if verbose:
            print(f"Sweeping transmission probability q over {len(q_values)} values")
            print(f"Values: {q_values}")
            print(f"Replications per value: {n_replications}")
        
        batch_sim = BatchSimulator(base_config)
        results = batch_sim.parameter_sweep(
            param_name='transmission_prob',
            param_values=q_values,
            n_replications=n_replications,
            verbose=verbose
        )
        
        return results
    
    @staticmethod
    def sweep_idle_timer(
        base_config: SimulationConfig,
        ts_values: List[int] = None,
        n_replications: int = 20,
        verbose: bool = True
    ) -> Dict[int, List[SimulationResults]]:
        """
        Sweep idle timer ts.
        
        Tests impact of sleep timer on delay and energy consumption.
        Larger ts increases delay but saves energy (less frequent sleep/wakeup).
        
        Args:
            base_config: Base simulation configuration
            ts_values: List of ts values to test (default: [1, 2, 5, 10, 20, 50, 100])
            n_replications: Number of replications per value
            verbose: Print progress
            
        Returns:
            Dictionary mapping ts values to list of results
        """
        if ts_values is None:
            ts_values = [1, 2, 5, 10, 20, 50, 100]
        
        if verbose:
            print(f"Sweeping idle timer ts over {len(ts_values)} values")
            print(f"Values: {ts_values}")
            print(f"Replications per value: {n_replications}")
        
        batch_sim = BatchSimulator(base_config)
        results = batch_sim.parameter_sweep(
            param_name='idle_timer',
            param_values=ts_values,
            n_replications=n_replications,
            verbose=verbose
        )
        
        return results
    
    @staticmethod
    def sweep_num_nodes(
        base_config: SimulationConfig,
        n_values: List[int] = None,
        n_replications: int = 20,
        verbose: bool = True
    ) -> Dict[int, List[SimulationResults]]:
        """
        Sweep number of nodes n.
        
        Tests scalability and impact of population size on contention.
        More nodes increase collisions and delay.
        
        Args:
            base_config: Base simulation configuration
            n_values: List of n values to test (default: [10, 20, 50, 100, 200, 500])
            n_replications: Number of replications per value
            verbose: Print progress
            
        Returns:
            Dictionary mapping n values to list of results
        """
        if n_values is None:
            n_values = [10, 20, 50, 100, 200, 500]
        
        if verbose:
            print(f"Sweeping number of nodes n over {len(n_values)} values")
            print(f"Values: {n_values}")
            print(f"Replications per value: {n_replications}")
        
        batch_sim = BatchSimulator(base_config)
        results = batch_sim.parameter_sweep(
            param_name='n_nodes',
            param_values=n_values,
            n_replications=n_replications,
            verbose=verbose
        )
        
        return results
    
    @staticmethod
    def sweep_arrival_rate(
        base_config: SimulationConfig,
        lambda_values: List[float] = None,
        n_replications: int = 20,
        verbose: bool = True
    ) -> Dict[float, List[SimulationResults]]:
        """
        Sweep arrival rate λ.
        
        Tests impact of traffic load on system performance.
        Higher λ increases delay and energy consumption.
        
        Args:
            base_config: Base simulation configuration
            lambda_values: List of λ values to test (default: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
            n_replications: Number of replications per value
            verbose: Print progress
            
        Returns:
            Dictionary mapping λ values to list of results
        """
        if lambda_values is None:
            lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        if verbose:
            print(f"Sweeping arrival rate λ over {len(lambda_values)} values")
            print(f"Values: {lambda_values}")
            print(f"Replications per value: {n_replications}")
        
        batch_sim = BatchSimulator(base_config)
        results = batch_sim.parameter_sweep(
            param_name='arrival_rate',
            param_values=lambda_values,
            n_replications=n_replications,
            verbose=verbose
        )
        
        return results
    
    @staticmethod
    def analyze_sweep_results(
        sweep_results: Dict[Any, List[SimulationResults]],
        param_name: str
    ) -> Dict[Any, Dict[str, Tuple[float, float]]]:
        """
        Analyze parameter sweep results.
        
        Computes mean and std for all metrics across replications.
        
        Args:
            sweep_results: Dictionary from parameter sweep
            param_name: Name of swept parameter
            
        Returns:
            Dictionary mapping parameter values to aggregated metrics
        """
        analysis = {}
        
        for param_value, results in sweep_results.items():
            agg = analyze_batch_results(results)
            analysis[param_value] = agg
        
        return analysis
    
    @staticmethod
    def save_sweep_results(
        sweep_results: Dict[Any, List[SimulationResults]],
        param_name: str,
        output_dir: str = "results",
        filename: str = None
    ) -> Path:
        """
        Save sweep results to JSON file.
        
        Args:
            sweep_results: Dictionary from parameter sweep
            param_name: Name of swept parameter
            output_dir: Output directory
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"sweep_{param_name}_{timestamp}.json"
        
        filepath = output_path / filename
        
        # Analyze results
        analysis = ParameterSweep.analyze_sweep_results(sweep_results, param_name)
        
        # Convert to serializable format
        data = {
            'param_name': param_name,
            'param_values': [str(k) for k in sweep_results.keys()],
            'n_replications': len(list(sweep_results.values())[0]),
            'analysis': {
                str(param_val): {
                    metric: [float(mean), float(std)]
                    for metric, (mean, std) in metrics.items()
                }
                for param_val, metrics in analysis.items()
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath


class ScenarioExperiments:
    """
    Scenario-based experiments comparing different prioritization strategies.
    
    Compares low-latency priority vs battery-life priority configurations.
    """
    
    @staticmethod
    def create_low_latency_scenario(
        n_nodes: int = 20,
        arrival_rate: float = 0.01,
        power_profile: PowerProfile = PowerProfile.GENERIC_LOW
    ) -> ScenarioConfig:
        """
        Create a low-latency priority scenario.
        
        Configuration optimized for minimal delay:
        - Small idle timer (ts=1) for quick sleep entry
        - Higher transmission probability (q=1/n) for faster access
        - Short wake-up time
        
        Args:
            n_nodes: Number of nodes
            arrival_rate: Arrival rate λ
            power_profile: Power consumption profile
            
        Returns:
            ScenarioConfig for low-latency priority
        """
        optimal_q = 1.0 / n_nodes
        
        config = SimulationConfig(
            n_nodes=n_nodes,
            arrival_rate=arrival_rate,
            transmission_prob=optimal_q,  # Optimal for throughput
            idle_timer=1,                 # Quick sleep entry
            wakeup_time=2,                # Short wake-up
            initial_energy=5000,
            power_rates=PowerModel.get_profile(power_profile),
            max_slots=50000,
            seed=None
        )
        
        return ScenarioConfig(
            name="Low-Latency Priority",
            description="Optimized for minimal delay (small ts, optimal q)",
            config=config,
            priority="latency"
        )
    
    @staticmethod
    def create_battery_life_scenario(
        n_nodes: int = 20,
        arrival_rate: float = 0.01,
        power_profile: PowerProfile = PowerProfile.GENERIC_LOW
    ) -> ScenarioConfig:
        """
        Create a battery-life priority scenario.
        
        Configuration optimized for maximum lifetime:
        - Large idle timer (ts=50) to minimize sleep/wakeup cycles
        - Lower transmission probability (q=0.02) to reduce collisions
        - Longer wake-up acceptable
        
        Args:
            n_nodes: Number of nodes
            arrival_rate: Arrival rate λ
            power_profile: Power consumption profile
            
        Returns:
            ScenarioConfig for battery-life priority
        """
        config = SimulationConfig(
            n_nodes=n_nodes,
            arrival_rate=arrival_rate,
            transmission_prob=0.02,       # Conservative for energy
            idle_timer=50,                # Long idle before sleep
            wakeup_time=5,                # Wake-up time less critical
            initial_energy=5000,
            power_rates=PowerModel.get_profile(power_profile),
            max_slots=50000,
            seed=None
        )
        
        return ScenarioConfig(
            name="Battery-Life Priority",
            description="Optimized for maximum lifetime (large ts, low q)",
            config=config,
            priority="battery"
        )
    
    @staticmethod
    def create_balanced_scenario(
        n_nodes: int = 20,
        arrival_rate: float = 0.01,
        power_profile: PowerProfile = PowerProfile.GENERIC_LOW
    ) -> ScenarioConfig:
        """
        Create a balanced scenario.
        
        Configuration balancing delay and lifetime:
        - Moderate idle timer (ts=10)
        - Moderate transmission probability
        
        Args:
            n_nodes: Number of nodes
            arrival_rate: Arrival rate λ
            power_profile: Power consumption profile
            
        Returns:
            ScenarioConfig for balanced approach
        """
        config = SimulationConfig(
            n_nodes=n_nodes,
            arrival_rate=arrival_rate,
            transmission_prob=0.05,       # Balanced
            idle_timer=10,                # Moderate
            wakeup_time=3,                # Moderate
            initial_energy=5000,
            power_rates=PowerModel.get_profile(power_profile),
            max_slots=50000,
            seed=None
        )
        
        return ScenarioConfig(
            name="Balanced",
            description="Balance between delay and lifetime (moderate ts, q)",
            config=config,
            priority="balanced"
        )
    
    @staticmethod
    def compare_scenarios(
        scenarios: List[ScenarioConfig],
        n_replications: int = 20,
        verbose: bool = True
    ) -> Dict[str, List[SimulationResults]]:
        """
        Compare multiple scenarios.
        
        Args:
            scenarios: List of scenario configurations
            n_replications: Number of replications per scenario
            verbose: Print progress
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario in scenarios:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running scenario: {scenario.name}")
                print(f"Description: {scenario.description}")
                print(f"{'='*60}")
            
            batch_sim = BatchSimulator(scenario.config)
            scenario_results = batch_sim.run_replications(
                n_replications=n_replications,
                verbose=verbose
            )
            
            results[scenario.name] = scenario_results
            
            # Print quick summary
            if verbose:
                agg = analyze_batch_results(scenario_results)
                mean_delay, std_delay = agg['mean_delay']
                mean_lifetime, std_lifetime = agg['lifetime_years']
                
                print(f"\n{scenario.name} Summary:")
                print(f"  Mean delay: {mean_delay:.2f} ± {std_delay:.2f} slots")
                print(f"  Lifetime: {BatchSimulator.format_lifetime(mean_lifetime, std_lifetime)}")
        
        return results
    
    @staticmethod
    def analyze_tradeoffs(
        scenario_results: Dict[str, List[SimulationResults]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze trade-offs between scenarios.
        
        Args:
            scenario_results: Results from compare_scenarios
            
        Returns:
            Dictionary with trade-off analysis
        """
        analysis = {}
        
        for scenario_name, results in scenario_results.items():
            agg = analyze_batch_results(results)
            
            mean_delay, std_delay = agg['mean_delay']
            mean_lifetime, std_lifetime = agg['lifetime_years']
            mean_energy, std_energy = agg['energy_per_packet']
            mean_throughput, std_throughput = agg['throughput']
            
            analysis[scenario_name] = {
                'delay': {
                    'mean': mean_delay,
                    'std': std_delay,
                    'unit': 'slots'
                },
                'lifetime': {
                    'mean': mean_lifetime,
                    'std': std_lifetime,
                    'unit': 'years'
                },
                'energy_per_packet': {
                    'mean': mean_energy,
                    'std': std_energy,
                    'unit': 'energy_units'
                },
                'throughput': {
                    'mean': mean_throughput,
                    'std': std_throughput,
                    'unit': 'packets/slot'
                }
            }
        
        return analysis


def run_comprehensive_experiments(
    output_dir: str = "results",
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive parameter sweep experiments.
    
    This function runs all parameter sweeps and scenario comparisons
    for Task 2.2.
    
    Args:
        output_dir: Directory to save results
        quick_mode: If True, use fewer replications for faster execution
        
    Returns:
        Dictionary with all experimental results
    """
    n_reps = 10 if quick_mode else 20
    
    print("="*80)
    print("COMPREHENSIVE PARAMETER SWEEP EXPERIMENTS")
    print("="*80)
    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Replications per configuration: {n_reps}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Base configuration
    base_config = SimulationConfig(
        n_nodes=20,
        arrival_rate=0.01,
        transmission_prob=0.05,
        idle_timer=10,
        wakeup_time=5,
        initial_energy=5000,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=30000 if quick_mode else 50000,
        seed=None
    )
    
    results = {}
    
    # 1. Sweep transmission probability q
    print("\n" + "="*80)
    print("1. TRANSMISSION PROBABILITY (q) SWEEP")
    print("="*80)
    q_results = ParameterSweep.sweep_transmission_prob(
        base_config,
        q_values=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2] if quick_mode else None,
        n_replications=n_reps,
        verbose=True
    )
    results['q_sweep'] = q_results
    ParameterSweep.save_sweep_results(q_results, 'q', output_dir)
    
    # 2. Sweep idle timer ts
    print("\n" + "="*80)
    print("2. IDLE TIMER (ts) SWEEP")
    print("="*80)
    ts_results = ParameterSweep.sweep_idle_timer(
        base_config,
        ts_values=[1, 5, 10, 20, 50] if quick_mode else None,
        n_replications=n_reps,
        verbose=True
    )
    results['ts_sweep'] = ts_results
    ParameterSweep.save_sweep_results(ts_results, 'ts', output_dir)
    
    # 3. Sweep number of nodes n
    print("\n" + "="*80)
    print("3. NUMBER OF NODES (n) SWEEP")
    print("="*80)
    n_results = ParameterSweep.sweep_num_nodes(
        base_config,
        n_values=[10, 20, 50, 100] if quick_mode else None,
        n_replications=n_reps,
        verbose=True
    )
    results['n_sweep'] = n_results
    ParameterSweep.save_sweep_results(n_results, 'n', output_dir)
    
    # 4. Sweep arrival rate λ
    print("\n" + "="*80)
    print("4. ARRIVAL RATE (λ) SWEEP")
    print("="*80)
    lambda_results = ParameterSweep.sweep_arrival_rate(
        base_config,
        lambda_values=[0.001, 0.005, 0.01, 0.02, 0.05] if quick_mode else None,
        n_replications=n_reps,
        verbose=True
    )
    results['lambda_sweep'] = lambda_results
    ParameterSweep.save_sweep_results(lambda_results, 'lambda', output_dir)
    
    # 5. Scenario comparison
    print("\n" + "="*80)
    print("5. SCENARIO COMPARISON")
    print("="*80)
    scenarios = [
        ScenarioExperiments.create_low_latency_scenario(),
        ScenarioExperiments.create_balanced_scenario(),
        ScenarioExperiments.create_battery_life_scenario()
    ]
    scenario_results = ScenarioExperiments.compare_scenarios(
        scenarios,
        n_replications=n_reps,
        verbose=True
    )
    results['scenarios'] = scenario_results
    
    # Analyze trade-offs
    tradeoffs = ScenarioExperiments.analyze_tradeoffs(scenario_results)
    results['tradeoff_analysis'] = tradeoffs
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    
    return results
