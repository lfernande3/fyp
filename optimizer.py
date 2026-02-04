"""
Optimization and Parameter Sweep Module

Provides functions for:
- Parameter sweeps (varying q, ts, lambda, etc.)
- Optimization (finding optimal q for max lifetime or min delay)
- Tradeoff analysis (lifetime vs delay curves)
- Multi-run simulations with statistical analysis
"""

import simpy
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from simulator import Simulator


def run_single_simulation(params: Dict) -> Dict:
    """
    Run a single simulation with given parameters.
    Helper function for parallel execution.
    
    Args:
        params: Dictionary containing all simulation parameters
        
    Returns:
        Dictionary of metrics from the simulation
    """
    env = simpy.Environment()
    sim = Simulator(
        env=env,
        n_nodes=params['n_nodes'],
        lambda_arrival=params['lambda_arrival'],
        q_transmit=params['q_transmit'],
        ts_idle=params['ts_idle'],
        tw_wakeup=params['tw_wakeup'],
        E_initial=params['E_initial'],
        PS=params['PS'],
        PW=params['PW'],
        PT=params['PT'],
        PB=params['PB'],
        simulation_time=params['simulation_time'],
        seed=params.get('seed', None)
    )
    
    sim.run()
    metrics = sim.collect_metrics()
    
    return metrics


def parameter_sweep(base_params: Dict, sweep_param: str, 
                   sweep_values: np.ndarray,
                   n_runs: int = 1,
                   parallel: bool = False,
                   verbose: bool = True) -> List[Dict]:
    """
    Perform a parameter sweep over a single parameter.
    
    Args:
        base_params: Base simulation parameters
        sweep_param: Name of parameter to sweep
        sweep_values: Array of values to test
        n_runs: Number of runs per parameter value (for statistical averaging)
        parallel: Whether to use parallel processing
        verbose: Whether to show progress bar
        
    Returns:
        List of metric dictionaries, one per sweep value
    """
    results = []
    
    # Create list of all simulation configs
    sim_configs = []
    for value in sweep_values:
        for run_idx in range(n_runs):
            params = base_params.copy()
            params[sweep_param] = value
            params['seed'] = run_idx if n_runs > 1 else None
            params['_sweep_value'] = value  # Track original sweep value
            params['_run_idx'] = run_idx
            sim_configs.append(params)
    
    if verbose:
        print(f"Running {len(sim_configs)} simulations...")
    
    # Run simulations
    if parallel:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            if verbose:
                iterator = tqdm(executor.map(run_single_simulation, sim_configs),
                              total=len(sim_configs))
            else:
                iterator = executor.map(run_single_simulation, sim_configs)
            all_results = list(iterator)
    else:
        all_results = []
        iterator = tqdm(sim_configs) if verbose else sim_configs
        for config in iterator:
            result = run_single_simulation(config)
            all_results.append(result)
    
    # Aggregate results by sweep value
    if n_runs > 1:
        # Average over multiple runs
        df = pd.DataFrame(all_results)
        df['sweep_value'] = [r['_sweep_value'] for r in all_results]
        
        # Group by sweep value and compute statistics
        grouped = df.groupby('sweep_value').agg(['mean', 'std'])
        
        for value in sweep_values:
            group_data = grouped.loc[value]
            metrics = {}
            for col in df.columns:
                if col not in ['sweep_value', '_sweep_value', '_run_idx']:
                    metrics[col] = group_data[col]['mean']
                    metrics[f'{col}_std'] = group_data[col]['std']
            metrics[sweep_param] = value
            results.append(metrics)
    else:
        results = all_results
    
    return results


def multi_parameter_sweep(base_params: Dict,
                         sweep_config: Dict[str, np.ndarray],
                         n_runs: int = 1,
                         parallel: bool = False,
                         verbose: bool = True) -> List[Dict]:
    """
    Perform a multi-dimensional parameter sweep (grid search).
    
    Args:
        base_params: Base simulation parameters
        sweep_config: Dict mapping parameter names to arrays of values
        n_runs: Number of runs per parameter combination
        parallel: Whether to use parallel processing
        verbose: Whether to show progress bar
        
    Returns:
        List of metric dictionaries for all combinations
    """
    # Generate all combinations
    param_names = list(sweep_config.keys())
    param_values = list(sweep_config.values())
    
    # Create meshgrid of all combinations
    from itertools import product
    combinations = list(product(*param_values))
    
    if verbose:
        print(f"Testing {len(combinations)} parameter combinations with {n_runs} runs each")
        print(f"Total simulations: {len(combinations) * n_runs}")
    
    # Create simulation configs
    sim_configs = []
    for combo in combinations:
        for run_idx in range(n_runs):
            params = base_params.copy()
            for param_name, param_value in zip(param_names, combo):
                params[param_name] = param_value
            params['seed'] = run_idx if n_runs > 1 else None
            params['_combo'] = combo
            params['_run_idx'] = run_idx
            sim_configs.append(params)
    
    # Run simulations
    if parallel:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            if verbose:
                iterator = tqdm(executor.map(run_single_simulation, sim_configs),
                              total=len(sim_configs))
            else:
                iterator = executor.map(run_single_simulation, sim_configs)
            all_results = list(iterator)
    else:
        all_results = []
        iterator = tqdm(sim_configs) if verbose else sim_configs
        for config in iterator:
            result = run_single_simulation(config)
            all_results.append(result)
    
    # Aggregate if multiple runs
    if n_runs > 1:
        df = pd.DataFrame(all_results)
        for param_name in param_names:
            df[f'_{param_name}'] = [cfg[param_name] for cfg in sim_configs]
        
        group_cols = [f'_{pname}' for pname in param_names]
        grouped = df.groupby(group_cols).agg(['mean', 'std'])
        
        results = []
        for combo in combinations:
            combo_dict = dict(zip(param_names, combo))
            lookup_key = tuple(combo)
            
            try:
                group_data = grouped.loc[lookup_key]
                metrics = combo_dict.copy()
                for col in df.columns:
                    if not col.startswith('_'):
                        metrics[col] = group_data[col]['mean']
                        metrics[f'{col}_std'] = group_data[col]['std']
                results.append(metrics)
            except KeyError:
                continue
    else:
        results = all_results
    
    return results


def find_optimal_q(base_params: Dict,
                   q_range: Tuple[float, float] = (0.01, 0.5),
                   n_samples: int = 20,
                   objective: str = 'lifetime',
                   n_runs: int = 3,
                   verbose: bool = True) -> Tuple[float, Dict]:
    """
    Find optimal transmission probability q that maximizes lifetime or minimizes delay.
    
    Args:
        base_params: Base simulation parameters
        q_range: (min_q, max_q) range to search
        n_samples: Number of q values to test
        objective: 'lifetime' (maximize) or 'delay' (minimize)
        n_runs: Number of runs for averaging
        verbose: Whether to print progress
        
    Returns:
        Tuple of (optimal_q, metrics_at_optimal)
    """
    q_values = np.linspace(q_range[0], q_range[1], n_samples)
    
    if verbose:
        print(f"Searching for optimal q (objective: {objective})")
        print(f"Testing {n_samples} values in range [{q_range[0]:.3f}, {q_range[1]:.3f}]")
    
    results = parameter_sweep(
        base_params=base_params,
        sweep_param='q_transmit',
        sweep_values=q_values,
        n_runs=n_runs,
        parallel=False,
        verbose=verbose
    )
    
    # Find optimal based on objective
    if objective == 'lifetime':
        best_idx = max(range(len(results)), 
                      key=lambda i: results[i]['avg_lifetime'])
        optimal_q = results[best_idx]['q_transmit']
        best_value = results[best_idx]['avg_lifetime']
        if verbose:
            print(f"Optimal q = {optimal_q:.4f} (lifetime = {best_value:.2f} slots)")
    elif objective == 'delay':
        best_idx = min(range(len(results)), 
                      key=lambda i: results[i]['avg_delay'])
        optimal_q = results[best_idx]['q_transmit']
        best_value = results[best_idx]['avg_delay']
        if verbose:
            print(f"Optimal q = {optimal_q:.4f} (delay = {best_value:.2f} slots)")
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    return optimal_q, results[best_idx]


def analyze_tradeoff(base_params: Dict,
                    ts_values: np.ndarray,
                    q_range: Tuple[float, float] = (0.01, 0.5),
                    n_q_samples: int = 15,
                    n_runs: int = 2,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Analyze the fundamental tradeoff between lifetime and delay
    by varying ts (idle timeout) and finding optimal q for each.
    
    This replicates the analysis from Section IV of the paper.
    
    Args:
        base_params: Base simulation parameters
        ts_values: Array of ts (idle timeout) values to test
        q_range: Range of q values to optimize over
        n_q_samples: Number of q values to test for each ts
        n_runs: Number of runs for averaging
        verbose: Whether to print progress
        
    Returns:
        DataFrame with tradeoff curve data
    """
    tradeoff_results = []
    
    for ts in (tqdm(ts_values) if verbose else ts_values):
        params = base_params.copy()
        params['ts_idle'] = ts
        
        # Find optimal q for lifetime
        q_opt_lifetime, metrics_lifetime = find_optimal_q(
            base_params=params,
            q_range=q_range,
            n_samples=n_q_samples,
            objective='lifetime',
            n_runs=n_runs,
            verbose=False
        )
        
        # Find optimal q for delay
        q_opt_delay, metrics_delay = find_optimal_q(
            base_params=params,
            q_range=q_range,
            n_samples=n_q_samples,
            objective='delay',
            n_runs=n_runs,
            verbose=False
        )
        
        tradeoff_results.append({
            'ts_idle': ts,
            'q_opt_lifetime': q_opt_lifetime,
            'max_lifetime': metrics_lifetime['avg_lifetime'],
            'delay_at_max_lifetime': metrics_lifetime['avg_delay'],
            'q_opt_delay': q_opt_delay,
            'min_delay': metrics_delay['avg_delay'],
            'lifetime_at_min_delay': metrics_delay['avg_lifetime'],
        })
        
        if verbose:
            print(f"ts={ts}: q_lifetime={q_opt_lifetime:.4f}, q_delay={q_opt_delay:.4f}")
    
    return pd.DataFrame(tradeoff_results)


def sensitivity_analysis(base_params: Dict,
                        param_ranges: Dict[str, Tuple[float, float]],
                        n_samples: int = 10,
                        baseline_metrics: Optional[Dict] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying each parameter individually
    and measuring impact on key metrics.
    
    Args:
        base_params: Base simulation parameters
        param_ranges: Dict mapping parameter names to (min, max) ranges
        n_samples: Number of samples per parameter
        baseline_metrics: Optional baseline metrics for comparison
        verbose: Whether to print progress
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    if baseline_metrics is None:
        # Run baseline simulation
        baseline_metrics = run_single_simulation(base_params)
    
    sensitivity_results = []
    
    for param_name, (min_val, max_val) in param_ranges.items():
        if verbose:
            print(f"\nAnalyzing sensitivity to {param_name}...")
        
        param_values = np.linspace(min_val, max_val, n_samples)
        results = parameter_sweep(
            base_params=base_params,
            sweep_param=param_name,
            sweep_values=param_values,
            n_runs=1,
            parallel=False,
            verbose=False
        )
        
        for result in results:
            sensitivity_results.append({
                'parameter': param_name,
                'value': result[param_name],
                'lifetime_change': (result['avg_lifetime'] - baseline_metrics['avg_lifetime']) / baseline_metrics['avg_lifetime'],
                'delay_change': (result['avg_delay'] - baseline_metrics['avg_delay']) / baseline_metrics['avg_delay'],
                'throughput_change': (result['total_throughput'] - baseline_metrics['total_throughput']) / baseline_metrics['total_throughput'],
            })
    
    return pd.DataFrame(sensitivity_results)


def compare_configurations(configs: List[Dict],
                          config_names: List[str],
                          n_runs: int = 5,
                          parallel: bool = False,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Compare multiple complete configurations (e.g., different protocols).
    
    Args:
        configs: List of complete parameter dictionaries
        config_names: Names for each configuration
        n_runs: Number of runs per configuration
        parallel: Whether to use parallel processing
        verbose: Whether to print progress
        
    Returns:
        DataFrame comparing configurations
    """
    all_results = []
    
    for config, name in zip(configs, config_names):
        if verbose:
            print(f"\nTesting configuration: {name}")
        
        sim_configs = []
        for run_idx in range(n_runs):
            params = config.copy()
            params['seed'] = run_idx
            params['_config_name'] = name
            sim_configs.append(params)
        
        # Run simulations
        if parallel:
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                results = list(executor.map(run_single_simulation, sim_configs))
        else:
            results = [run_single_simulation(cfg) for cfg in sim_configs]
        
        all_results.extend(results)
    
    # Create DataFrame and compute statistics
    df = pd.DataFrame(all_results)
    df['config_name'] = [r.get('_config_name', '') for r in all_results]
    
    # Group by configuration and compute mean ± std
    grouped = df.groupby('config_name').agg(['mean', 'std'])
    
    comparison = []
    for name in config_names:
        stats = grouped.loc[name]
        row = {'configuration': name}
        for col in ['avg_lifetime', 'avg_delay', 'total_throughput', 
                   'collision_rate', 'energy_consumed_ratio']:
            if col in df.columns:
                row[col] = stats[col]['mean']
                row[f'{col}_std'] = stats[col]['std']
        comparison.append(row)
    
    return pd.DataFrame(comparison)


if __name__ == "__main__":
    print("=" * 70)
    print("Optimizer Module - Example Usage")
    print("=" * 70)
    
    # Base parameters
    base_params = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,  # Will be varied
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1,
        'PW': 1.0,
        'PT': 5.0,
        'PB': 0.5,
        'simulation_time': 3000,
    }
    
    print("\n1. Finding optimal q for maximum lifetime...")
    optimal_q, metrics = find_optimal_q(
        base_params=base_params,
        q_range=(0.01, 0.3),
        n_samples=10,
        objective='lifetime',
        n_runs=2,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Optimal q: {optimal_q:.4f}")
    print(f"  Lifetime: {metrics['avg_lifetime']:.2f} slots")
    print(f"  Delay: {metrics['avg_delay']:.2f} slots")
    
    print("\n" + "=" * 70)
    print("Optimization examples complete!")
