"""
Experiment Management for M2M Sleep-Based Random Access Simulator

This module provides pre-configured experiments for parameter impact analysis,
trade-off studies, and scenario comparisons.

Date: February 10, 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json

from .simulator import Simulator, BatchSimulator, SimulationConfig, SimulationResults
from .traffic_models import TrafficGenerator, TrafficConfig, TrafficModel
from .visualization import ResultsVisualizer
from .power_model import PowerModel, PowerProfile


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    name: str
    description: str
    base_config: SimulationConfig
    n_replications: int = 20
    save_results: bool = True
    output_dir: str = "results"


class ExperimentSuite:
    """
    Suite of pre-configured experiments for parameter impact analysis.
    
    Provides standard experiments for:
    - Single parameter sweeps (q, ts, tw, λ, n)
    - Multi-parameter sweeps (heatmaps)
    - Scenario comparisons (low-latency vs. battery-life)
    - Traffic model comparisons (Poisson vs. bursty)
    """
    
    def __init__(self, base_config: Optional[SimulationConfig] = None):
        """
        Initialize experiment suite.
        
        Args:
            base_config: Base configuration (uses default if None)
        """
        if base_config is None:
            # Default configuration
            power_rates = PowerModel.get_profile(PowerProfile.NB_IOT)
            self.base_config = SimulationConfig(
                n_nodes=20,
                arrival_rate=0.02,
                transmission_prob=0.1,
                idle_timer=10,
                wakeup_time=5,
                initial_energy=10000.0,
                power_rates=power_rates,
                max_slots=10000,
                seed=42
            )
        else:
            self.base_config = base_config
        
        self.batch_sim = BatchSimulator(self.base_config)
        self.visualizer = ResultsVisualizer()
    
    def sweep_transmission_prob(
        self,
        q_values: Optional[List[float]] = None,
        n_replications: int = 20,
        plot: bool = True
    ) -> Dict[float, List[SimulationResults]]:
        """
        Sweep transmission probability q.
        
        Shows impact of transmission aggressiveness on delay and lifetime.
        
        Args:
            q_values: Values to sweep (default: 0.01 to 0.5)
            n_replications: Replications per value
            plot: Whether to plot results
            
        Returns:
            Dictionary mapping q values to results
        """
        if q_values is None:
            q_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Transmission Probability (q) Sweep")
        print(f"{'='*80}")
        print(f"Values: {q_values}")
        print(f"Replications: {n_replications}\n")
        
        start_time = time.time()
        
        results = self.batch_sim.parameter_sweep(
            'transmission_prob',
            q_values,
            n_replications=n_replications,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        
        if plot:
            # Get aggregated results
            aggregated = []
            for q in q_values:
                result_list = results[q]
                # Use first result as representative (or could average)
                aggregated.append(result_list[0])
            
            self.visualizer.plot_delay_vs_lifetime(
                aggregated, 
                param_name="Transmission Probability (q)",
                param_values=q_values,
                save_path=None
            )
            
            self.visualizer.plot_parameter_impact(
                q_values,
                aggregated,
                "Transmission Probability (q)",
                metrics=['mean_delay', 'mean_lifetime_years', 'throughput']
            )
        
        return results
    
    def sweep_idle_timer(
        self,
        ts_values: Optional[List[int]] = None,
        n_replications: int = 20,
        plot: bool = True
    ) -> Dict[int, List[SimulationResults]]:
        """
        Sweep idle timer ts.
        
        Shows impact of sleep aggressiveness on delay and lifetime.
        
        Args:
            ts_values: Values to sweep (default: 0 to 100)
            n_replications: Replications per value
            plot: Whether to plot results
            
        Returns:
            Dictionary mapping ts values to results
        """
        if ts_values is None:
            ts_values = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Idle Timer (ts) Sweep")
        print(f"{'='*80}")
        print(f"Values: {ts_values}")
        print(f"Replications: {n_replications}\n")
        
        start_time = time.time()
        
        results = self.batch_sim.parameter_sweep(
            'idle_timer',
            ts_values,
            n_replications=n_replications,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        
        if plot:
            aggregated = [results[ts][0] for ts in ts_values]
            
            self.visualizer.plot_delay_vs_lifetime(
                aggregated,
                param_name="Idle Timer (ts)",
                param_values=ts_values
            )
            
            self.visualizer.plot_parameter_impact(
                ts_values,
                aggregated,
                "Idle Timer (ts) [slots]",
                metrics=['mean_delay', 'mean_lifetime_years', 'throughput']
            )
        
        return results
    
    def sweep_arrival_rate(
        self,
        lambda_values: Optional[List[float]] = None,
        n_replications: int = 20,
        plot: bool = True
    ) -> Dict[float, List[SimulationResults]]:
        """
        Sweep arrival rate λ.
        
        Shows impact of traffic load on system performance.
        
        Args:
            lambda_values: Values to sweep (default: 0.001 to 0.1)
            n_replications: Replications per value
            plot: Whether to plot results
            
        Returns:
            Dictionary mapping λ values to results
        """
        if lambda_values is None:
            lambda_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Arrival Rate (λ) Sweep")
        print(f"{'='*80}")
        print(f"Values: {lambda_values}")
        print(f"Replications: {n_replications}\n")
        
        start_time = time.time()
        
        results = self.batch_sim.parameter_sweep(
            'arrival_rate',
            lambda_values,
            n_replications=n_replications,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        
        if plot:
            aggregated = [results[lam][0] for lam in lambda_values]
            
            self.visualizer.plot_parameter_impact(
                lambda_values,
                aggregated,
                "Arrival Rate (λ)",
                metrics=['mean_delay', 'mean_queue_length', 'throughput']
            )
        
        return results
    
    def sweep_num_nodes(
        self,
        n_values: Optional[List[int]] = None,
        n_replications: int = 20,
        plot: bool = True
    ) -> Dict[int, List[SimulationResults]]:
        """
        Sweep number of nodes n.
        
        Shows scalability and impact of network size.
        
        Args:
            n_values: Values to sweep (default: 10 to 500)
            n_replications: Replications per value
            plot: Whether to plot results
            
        Returns:
            Dictionary mapping n values to results
        """
        if n_values is None:
            n_values = [10, 20, 50, 100, 200, 500]
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Number of Nodes (n) Sweep")
        print(f"{'='*80}")
        print(f"Values: {n_values}")
        print(f"Replications: {n_replications}\n")
        
        start_time = time.time()
        
        results = self.batch_sim.parameter_sweep(
            'n_nodes',
            n_values,
            n_replications=n_replications,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        
        if plot:
            aggregated = [results[n][0] for n in n_values]
            
            self.visualizer.plot_parameter_impact(
                n_values,
                aggregated,
                "Number of Nodes (n)",
                metrics=['mean_delay', 'throughput', 'total_collisions']
            )
        
        return results
    
    def compare_scenarios(
        self,
        scenarios: Optional[Dict[str, SimulationConfig]] = None,
        param_name: str = 'transmission_prob',
        param_values: Optional[List[Any]] = None,
        n_replications: int = 20,
        plot: bool = True
    ) -> Dict[str, Dict[Any, List[SimulationResults]]]:
        """
        Compare different scenarios (e.g., low-latency vs. battery-life).
        
        Args:
            scenarios: Dictionary mapping scenario names to configs
            param_name: Parameter to sweep
            param_values: Values to sweep
            n_replications: Replications per value
            plot: Whether to plot comparison
            
        Returns:
            Nested dictionary: {scenario_name: {param_value: [results]}}
        """
        if scenarios is None:
            # Default: Compare low-latency vs. battery-life priorities
            power_rates = PowerModel.get_profile(PowerProfile.NB_IOT)
            
            scenarios = {
                'Low-Latency Priority': SimulationConfig(
                    n_nodes=20,
                    arrival_rate=0.02,
                    transmission_prob=0.2,  # Aggressive transmission
                    idle_timer=1,           # Quick sleep
                    wakeup_time=5,
                    initial_energy=10000.0,
                    power_rates=power_rates,
                    max_slots=10000,
                    seed=42
                ),
                'Battery-Life Priority': SimulationConfig(
                    n_nodes=20,
                    arrival_rate=0.02,
                    transmission_prob=0.05,  # Conservative transmission
                    idle_timer=50,           # Slow sleep
                    wakeup_time=5,
                    initial_energy=10000.0,
                    power_rates=power_rates,
                    max_slots=10000,
                    seed=42
                ),
                'Balanced': SimulationConfig(
                    n_nodes=20,
                    arrival_rate=0.02,
                    transmission_prob=0.1,
                    idle_timer=10,
                    wakeup_time=5,
                    initial_energy=10000.0,
                    power_rates=power_rates,
                    max_slots=10000,
                    seed=42
                )
            }
        
        if param_values is None:
            if param_name == 'transmission_prob':
                param_values = [0.05, 0.1, 0.15, 0.2, 0.3]
            elif param_name == 'idle_timer':
                param_values = [1, 5, 10, 20, 50]
            else:
                param_values = list(range(5, 25, 5))
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Scenario Comparison")
        print(f"{'='*80}")
        print(f"Scenarios: {list(scenarios.keys())}")
        print(f"Parameter: {param_name}")
        print(f"Values: {param_values}\n")
        
        all_results = {}
        
        for scenario_name, config in scenarios.items():
            print(f"\n--- {scenario_name} ---")
            batch_sim = BatchSimulator(config)
            results = batch_sim.parameter_sweep(
                param_name,
                param_values,
                n_replications=n_replications,
                verbose=True
            )
            all_results[scenario_name] = results
        
        if plot:
            # Create comparison plot
            results_by_scenario = {}
            for scenario_name in scenarios.keys():
                results_by_scenario[scenario_name] = [
                    all_results[scenario_name][val][0] 
                    for val in param_values
                ]
            
            # Plot delay comparison
            self.visualizer.plot_comparison(
                results_by_scenario,
                param_values,
                param_name.replace('_', ' ').title(),
                metric='mean_delay'
            )
            
            # Plot lifetime comparison
            self.visualizer.plot_comparison(
                results_by_scenario,
                param_values,
                param_name.replace('_', ' ').title(),
                metric='mean_lifetime_years'
            )
        
        return all_results
    
    def run_all_experiments(
        self,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete experimental suite.
        
        Args:
            quick_mode: Use fewer replications and values for faster execution
            
        Returns:
            Dictionary with all experiment results
        """
        n_reps = 10 if quick_mode else 20
        
        print(f"\n{'#'*80}")
        print(f"# RUNNING COMPLETE EXPERIMENTAL SUITE")
        print(f"# Mode: {'QUICK' if quick_mode else 'FULL'}")
        print(f"# Replications: {n_reps}")
        print(f"{'#'*80}\n")
        
        all_results = {}
        
        # Experiment 1: Transmission probability sweep
        all_results['q_sweep'] = self.sweep_transmission_prob(
            q_values=[0.05, 0.1, 0.2, 0.3] if quick_mode else None,
            n_replications=n_reps,
            plot=True
        )
        
        # Experiment 2: Idle timer sweep
        all_results['ts_sweep'] = self.sweep_idle_timer(
            ts_values=[0, 5, 10, 20, 50] if quick_mode else None,
            n_replications=n_reps,
            plot=True
        )
        
        # Experiment 3: Arrival rate sweep
        all_results['lambda_sweep'] = self.sweep_arrival_rate(
            lambda_values=[0.01, 0.02, 0.05] if quick_mode else None,
            n_replications=n_reps,
            plot=True
        )
        
        # Experiment 4: Scenario comparison
        all_results['scenario_comparison'] = self.compare_scenarios(
            param_name='idle_timer',
            param_values=[1, 10, 50] if quick_mode else [1, 5, 10, 20, 50],
            n_replications=n_reps,
            plot=True
        )
        
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENTAL SUITE COMPLETE")
        print(f"{'#'*80}\n")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Save experiment results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        # Convert results to JSON-serializable format
        serializable = {}
        
        for exp_name, exp_results in results.items():
            serializable[exp_name] = {}
            
            if isinstance(exp_results, dict):
                for param_val, result_list in exp_results.items():
                    key = str(param_val)
                    serializable[exp_name][key] = {
                        'mean_delay': [r.mean_delay for r in result_list],
                        'mean_lifetime_years': [r.mean_lifetime_years for r in result_list],
                        'throughput': [r.throughput for r in result_list]
                    }
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to {filename}")


def run_quick_demo():
    """Run quick demonstration of experiments."""
    print("Running quick experiment demonstration...")
    
    suite = ExperimentSuite()
    results = suite.run_all_experiments(quick_mode=True)
    
    print("\nDemo complete!")
    return results


if __name__ == "__main__":
    # Run demo when executed directly
    run_quick_demo()
