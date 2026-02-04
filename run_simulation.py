"""
Run Simulation Using Configuration File

This script uses parameters from config.py to run simulations.
Simply edit config.py to change parameters, then run this script.

Usage:
    python run_simulation.py                 # Use default config
    python run_simulation.py --preset low    # Use preset configuration
    python run_simulation.py --validate      # Validate config only
"""

import sys
import argparse
import simpy

# Import configuration
import config
from simulator import Simulator
from visualizer import plot_metrics_summary, plot_node_comparison

try:
    from visualizer import plot_energy_timeline, plot_state_timeline
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def run_with_config(params=None, show_viz=True):
    """
    Run simulation using configuration.
    
    Args:
        params: Parameter dictionary (uses config.py if None)
        show_viz: Whether to show visualizations
    """
    # Use config file if no params provided
    if params is None:
        params = config.get_base_params()
    
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION WITH CONFIGURATION")
    print("=" * 70)
    
    # Display parameters
    print("\nParameters:")
    for key, value in params.items():
        if key != 'seed':
            print(f"  {key:20s}: {value}")
    
    # Validate configuration
    print("\n" + "-" * 70)
    print("Validating configuration...")
    print("-" * 70)
    config.validate_config()
    
    # Create and run simulation
    print("\n" + "-" * 70)
    print("Running simulation...")
    print("-" * 70)
    
    env = simpy.Environment()
    sim = Simulator(env=env, **params)
    
    sim.run()
    
    print("[OK] Simulation complete!")
    
    # Collect and display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    metrics = sim.collect_metrics()
    
    print("\n[LIFETIME METRICS]")
    print(f"  Average lifetime    : {metrics['avg_lifetime']:.2f} slots")
    print(f"  Min/Max lifetime    : [{metrics['min_lifetime']:.0f}, {metrics['max_lifetime']:.0f}]")
    print(f"  Std deviation       : {metrics['std_lifetime']:.2f}")
    
    print("\n[DELAY METRICS]")
    print(f"  Average delay       : {metrics['avg_delay']:.2f} slots")
    print(f"  Maximum delay       : {metrics['max_delay']:.2f} slots")
    
    print("\n[THROUGHPUT METRICS]")
    print(f"  Total throughput    : {metrics['total_throughput']:.4f} packets/slot")
    print(f"  Per-node throughput : {metrics['avg_node_throughput']:.4f} packets/slot")
    
    print("\n[TRANSMISSION STATISTICS]")
    print(f"  Packets sent        : {metrics['total_packets_sent']}")
    print(f"  Packets arrived     : {metrics['total_packets_arrived']}")
    print(f"  Delivery ratio      : {metrics['packet_delivery_ratio']:.2%}")
    print(f"  Total attempts      : {metrics['total_transmission_attempts']}")
    print(f"  Collisions          : {metrics['total_collisions']}")
    print(f"  Collision rate      : {metrics['collision_rate']:.2%}")
    
    print("\n[ENERGY METRICS]")
    print(f"  Avg energy remaining: {metrics['avg_energy_remaining']:.2f}")
    print(f"  Energy consumed     : {metrics['energy_consumed_ratio']:.2%}")
    
    # Calculate network efficiency
    network_load = params['n_nodes'] * params['lambda_arrival'] * params['q_transmit']
    print(f"\n[NETWORK ANALYSIS]")
    print(f"  Network load (G)    : {network_load:.4f}")
    print(f"  Aloha capacity      : 0.368")
    if network_load > 0.368:
        print(f"  WARNING: Load exceeds Aloha capacity!")
    
    # Visualizations
    if show_viz and HAS_VISUALIZER and config.show_plots:
        print("\n" + "-" * 70)
        print("Generating visualizations...")
        print("-" * 70)
        
        # Main metrics summary
        save_path = f"{config.output_directory}/metrics_summary.{config.plot_format}" if config.save_plots else None
        plot_metrics_summary(metrics, save_path=save_path)
        
        # Node comparison
        node_metrics = sim.get_detailed_node_metrics()
        save_path = f"{config.output_directory}/node_comparison.{config.plot_format}" if config.save_plots else None
        plot_node_comparison(node_metrics, save_path=save_path)
        
        # Timeline for first node (if not too long)
        if params['simulation_time'] <= 10000:
            save_path = f"{config.output_directory}/energy_timeline.{config.plot_format}" if config.save_plots else None
            plot_energy_timeline(sim.nodes[0], save_path=save_path)
            
            save_path = f"{config.output_directory}/state_timeline.{config.plot_format}" if config.save_plots else None
            plot_state_timeline(sim.nodes[0], max_slots=min(1000, params['simulation_time']), 
                              save_path=save_path)
        
        print("\n[OK] Visualizations complete!")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    return metrics, sim


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run On-Demand Sleep-Based Aloha Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                    # Use default config
  python run_simulation.py --preset low       # Low traffic preset
  python run_simulation.py --preset quick     # Quick test
  python run_simulation.py --validate         # Validate config only
  python run_simulation.py --print-config     # Print current config
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['low', 'medium', 'high', 'large', 'quick', 'energy', 'delay'],
        help='Use a preset configuration'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print configuration and exit'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualizations'
    )
    
    args = parser.parse_args()
    
    # Handle print-config
    if args.print_config:
        config.print_config()
        return
    
    # Handle validate
    if args.validate:
        config.print_config()
        print("\nValidating configuration...")
        config.validate_config()
        print("\n[OK] Configuration is valid!")
        return
    
    # Select configuration
    if args.preset:
        preset_map = {
            'low': config.Presets.LOW_TRAFFIC,
            'medium': config.Presets.MEDIUM_TRAFFIC,
            'high': config.Presets.HIGH_TRAFFIC,
            'large': config.Presets.LARGE_NETWORK,
            'quick': config.Presets.QUICK_TEST,
            'energy': config.Presets.ENERGY_OPTIMIZED,
            'delay': config.Presets.DELAY_OPTIMIZED,
        }
        params = preset_map[args.preset]
        print(f"\n[CONFIG] Using preset: {args.preset.upper()}")
    else:
        params = config.get_base_params()
        print("\n[CONFIG] Using default configuration from config.py")
    
    # Run simulation
    try:
        run_with_config(params, show_viz=not args.no_viz)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
