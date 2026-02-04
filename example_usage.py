"""
Example usage script for On-Demand Sleep-Based Aloha Simulator

This script demonstrates:
1. Basic simulation run
2. Parameter sweep (varying q)
3. Finding optimal parameters
4. Tradeoff analysis (lifetime vs delay)
5. Visualization of results
"""

import simpy
import numpy as np
from simulator import Simulator
from optimizer import (
    parameter_sweep, 
    find_optimal_q, 
    analyze_tradeoff,
    multi_parameter_sweep,
    compare_configurations
)
from visualizer import (
    plot_metrics_summary,
    plot_tradeoff_curve,
    plot_parameter_sweep,
    plot_energy_timeline,
    plot_state_timeline,
    plot_node_comparison
)


def example_1_basic_simulation():
    """Example 1: Run a basic simulation and visualize results"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Simulation")
    print("=" * 70)
    
    # Create environment and simulator
    env = simpy.Environment()
    sim = Simulator(
        env=env,
        n_nodes=10,
        lambda_arrival=0.1,
        q_transmit=0.05,
        ts_idle=5,
        tw_wakeup=2,
        E_initial=10000,
        PS=0.1, PW=1.0, PT=5.0, PB=0.5,
        simulation_time=5000,
        seed=42
    )
    
    print("\nRunning simulation...")
    sim.run()
    
    # Collect metrics
    metrics = sim.collect_metrics()
    
    print("\nKey Results:")
    print(f"  Average Lifetime: {metrics['avg_lifetime']:.2f} slots")
    print(f"  Average Delay: {metrics['avg_delay']:.2f} slots")
    print(f"  Total Throughput: {metrics['total_throughput']:.4f} packets/slot")
    print(f"  Collision Rate: {metrics['collision_rate']:.2%}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_metrics_summary(metrics)
    
    # Plot energy and state timeline for first node
    plot_energy_timeline(sim.nodes[0])
    plot_state_timeline(sim.nodes[0], max_slots=500)
    
    # Compare nodes
    node_metrics = sim.get_detailed_node_metrics()
    plot_node_comparison(node_metrics)
    
    return metrics


def example_2_parameter_sweep():
    """Example 2: Sweep transmission probability q"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Parameter Sweep (q_transmit)")
    print("=" * 70)
    
    base_params = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,  # Will be varied
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 3000,
    }
    
    # Sweep q from 0.01 to 0.3
    q_values = np.linspace(0.01, 0.3, 15)
    
    print(f"\nSweeping q_transmit from {q_values[0]:.3f} to {q_values[-1]:.3f}")
    print("This may take a few minutes...")
    
    results = parameter_sweep(
        base_params=base_params,
        sweep_param='q_transmit',
        sweep_values=q_values,
        n_runs=2,  # Average over 2 runs
        parallel=False,
        verbose=True
    )
    
    # Visualize sweep results
    plot_parameter_sweep(
        results=results,
        sweep_param='q_transmit',
        metrics_to_plot=['avg_lifetime', 'avg_delay', 'total_throughput', 'collision_rate']
    )
    
    # Find best q for lifetime
    best_lifetime_idx = max(range(len(results)), 
                           key=lambda i: results[i]['avg_lifetime'])
    best_q = results[best_lifetime_idx]['q_transmit']
    
    print(f"\nBest q for lifetime: {best_q:.4f}")
    print(f"  Lifetime: {results[best_lifetime_idx]['avg_lifetime']:.2f} slots")
    print(f"  Delay: {results[best_lifetime_idx]['avg_delay']:.2f} slots")
    
    return results


def example_3_optimize_q():
    """Example 3: Find optimal q for different objectives"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Optimization (Finding Optimal q)")
    print("=" * 70)
    
    base_params = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 3000,
    }
    
    # Optimize for maximum lifetime
    print("\nOptimizing for MAXIMUM LIFETIME:")
    q_opt_lifetime, metrics_lifetime = find_optimal_q(
        base_params=base_params,
        q_range=(0.01, 0.3),
        n_samples=15,
        objective='lifetime',
        n_runs=3,
        verbose=True
    )
    
    print(f"\nOptimal q for lifetime: {q_opt_lifetime:.4f}")
    print(f"  Lifetime: {metrics_lifetime['avg_lifetime']:.2f} slots")
    print(f"  Delay: {metrics_lifetime['avg_delay']:.2f} slots")
    
    # Optimize for minimum delay
    print("\n" + "-" * 70)
    print("Optimizing for MINIMUM DELAY:")
    q_opt_delay, metrics_delay = find_optimal_q(
        base_params=base_params,
        q_range=(0.01, 0.3),
        n_samples=15,
        objective='delay',
        n_runs=3,
        verbose=True
    )
    
    print(f"\nOptimal q for delay: {q_opt_delay:.4f}")
    print(f"  Lifetime: {metrics_delay['avg_lifetime']:.2f} slots")
    print(f"  Delay: {metrics_delay['avg_delay']:.2f} slots")
    
    # Compare the two configurations
    print("\n" + "-" * 70)
    print("COMPARISON:")
    print(f"  Lifetime optimization gives {metrics_lifetime['avg_lifetime']/metrics_delay['avg_lifetime']:.2f}x lifetime")
    print(f"  Delay optimization gives {metrics_lifetime['avg_delay']/metrics_delay['avg_delay']:.2f}x delay")
    
    return q_opt_lifetime, q_opt_delay


def example_4_tradeoff_analysis():
    """Example 4: Analyze lifetime vs delay tradeoff"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Tradeoff Analysis (Lifetime vs Delay)")
    print("=" * 70)
    
    base_params = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 3000,
    }
    
    # Vary ts (idle timeout) from 1 to 20 slots
    ts_values = np.array([1, 3, 5, 10, 15, 20])
    
    print(f"\nAnalyzing tradeoff for ts values: {ts_values}")
    print("For each ts, finding optimal q for both lifetime and delay...")
    print("This will take several minutes...\n")
    
    tradeoff_df = analyze_tradeoff(
        base_params=base_params,
        ts_values=ts_values,
        q_range=(0.01, 0.3),
        n_q_samples=10,
        n_runs=2,
        verbose=True
    )
    
    print("\n" + "-" * 70)
    print("Tradeoff Results:")
    print(tradeoff_df.to_string(index=False))
    
    # Plot lifetime vs delay tradeoff curve
    print("\nPlotting tradeoff curve...")
    
    # Create combined results for plotting
    lifetime_points = []
    delay_points = []
    for _, row in tradeoff_df.iterrows():
        lifetime_points.append({
            'avg_lifetime': row['max_lifetime'],
            'avg_delay': row['delay_at_max_lifetime'],
            'ts_idle': row['ts_idle'],
            'objective': 'Max Lifetime'
        })
        delay_points.append({
            'avg_lifetime': row['lifetime_at_min_delay'],
            'avg_delay': row['min_delay'],
            'ts_idle': row['ts_idle'],
            'objective': 'Min Delay'
        })
    
    all_points = lifetime_points + delay_points
    plot_tradeoff_curve(
        results=all_points,
        x_param='avg_delay',
        y_param='avg_lifetime',
        color_param='ts_idle'
    )
    
    return tradeoff_df


def example_5_multi_parameter_sweep():
    """Example 5: 2D parameter sweep (q and ts)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Parameter Sweep (q and ts)")
    print("=" * 70)
    
    base_params = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 2000,  # Shorter for faster execution
    }
    
    # Define sweep ranges
    sweep_config = {
        'q_transmit': np.linspace(0.02, 0.15, 5),
        'ts_idle': np.array([1, 5, 10, 15])
    }
    
    print(f"\nSweeping:")
    print(f"  q_transmit: {sweep_config['q_transmit']}")
    print(f"  ts_idle: {sweep_config['ts_idle']}")
    print(f"Total combinations: {len(sweep_config['q_transmit']) * len(sweep_config['ts_idle'])}")
    
    results = multi_parameter_sweep(
        base_params=base_params,
        sweep_config=sweep_config,
        n_runs=1,
        parallel=False,
        verbose=True
    )
    
    # Create 3D plot
    from visualizer import plot_3d_tradeoff
    plot_3d_tradeoff(
        results=results,
        x_param='q_transmit',
        y_param='ts_idle',
        z_param='avg_lifetime'
    )
    
    # Also create 2D tradeoff plot colored by ts
    plot_tradeoff_curve(
        results=results,
        x_param='avg_delay',
        y_param='avg_lifetime',
        color_param='ts_idle'
    )
    
    return results


def example_6_compare_configurations():
    """Example 6: Compare different system configurations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Comparing Different Configurations")
    print("=" * 70)
    
    # Define different configurations to compare
    configs = [
        # Configuration 1: Aggressive transmission (high q)
        {
            'n_nodes': 10,
            'lambda_arrival': 0.1,
            'q_transmit': 0.2,
            'ts_idle': 5,
            'tw_wakeup': 2,
            'E_initial': 10000,
            'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
            'simulation_time': 3000,
        },
        # Configuration 2: Conservative transmission (low q)
        {
            'n_nodes': 10,
            'lambda_arrival': 0.1,
            'q_transmit': 0.03,
            'ts_idle': 5,
            'tw_wakeup': 2,
            'E_initial': 10000,
            'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
            'simulation_time': 3000,
        },
        # Configuration 3: Fast sleep (low ts)
        {
            'n_nodes': 10,
            'lambda_arrival': 0.1,
            'q_transmit': 0.05,
            'ts_idle': 1,
            'tw_wakeup': 2,
            'E_initial': 10000,
            'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
            'simulation_time': 3000,
        },
        # Configuration 4: Slow sleep (high ts)
        {
            'n_nodes': 10,
            'lambda_arrival': 0.1,
            'q_transmit': 0.05,
            'ts_idle': 20,
            'tw_wakeup': 2,
            'E_initial': 10000,
            'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
            'simulation_time': 3000,
        },
    ]
    
    config_names = [
        'Aggressive (q=0.2)',
        'Conservative (q=0.03)',
        'Fast Sleep (ts=1)',
        'Slow Sleep (ts=20)'
    ]
    
    print("\nComparing configurations:")
    for i, name in enumerate(config_names):
        print(f"  {i+1}. {name}")
    
    comparison_df = compare_configurations(
        configs=configs,
        config_names=config_names,
        n_runs=3,
        parallel=False,
        verbose=True
    )
    
    print("\n" + "-" * 70)
    print("Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("ON-DEMAND SLEEP-BASED ALOHA SIMULATOR")
    print("Example Usage Demonstrations")
    print("=" * 70)
    
    # Uncomment the examples you want to run:
    
    # Example 1: Basic simulation (fast)
    example_1_basic_simulation()
    
    # Example 2: Parameter sweep (moderate time)
    # example_2_parameter_sweep()
    
    # Example 3: Optimization (moderate time)
    # example_3_optimize_q()
    
    # Example 4: Tradeoff analysis (slow - takes several minutes)
    # example_4_tradeoff_analysis()
    
    # Example 5: Multi-parameter sweep (moderate time)
    # example_5_multi_parameter_sweep()
    
    # Example 6: Configuration comparison (moderate time)
    # example_6_compare_configurations()
    
    print("\n" + "=" * 70)
    print("Examples complete! Uncomment other examples in main() to run them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
