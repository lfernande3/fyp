"""
Quick Start Script - Basic demonstration of the simulator

This script runs a simple simulation and generates basic visualizations.
Perfect for testing that everything is installed correctly!
"""

import sys

# Check for required packages
try:
    import simpy
    import numpy as np
    from simulator import Simulator
except ImportError as e:
    print("\n" + "=" * 70)
    print("ERROR: Missing required packages")
    print("=" * 70)
    print(f"\nMissing: {e}")
    print("\nPlease install requirements first:")
    print("  pip install -r requirements.txt")
    print("\n" + "=" * 70)
    sys.exit(1)

# Try to import visualizer (optional for basic run)
try:
    from visualizer import plot_metrics_summary
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    print("\nNote: Visualization packages not found. Will show text results only.")
    print("To enable visualizations, run: pip install -r requirements.txt\n")

def main():
    print("\n" + "=" * 70)
    print("ON-DEMAND SLEEP-BASED ALOHA SIMULATOR - Quick Start")
    print("=" * 70)
    
    print("\nThis script demonstrates a basic simulation run.")
    print("For more examples, see example_usage.py")
    
    # Simulation parameters (optimized for quick demonstration)
    print("\n" + "-" * 70)
    print("Setting up simulation...")
    print("-" * 70)
    
    N_NODES = 10
    LAMBDA_ARRIVAL = 0.1
    Q_TRANSMIT = 0.05
    TS_IDLE = 5
    TW_WAKEUP = 2
    E_INITIAL = 10000
    SIM_TIME = 3000
    
    print(f"  Nodes: {N_NODES}")
    print(f"  Arrival rate: {LAMBDA_ARRIVAL} packets/slot")
    print(f"  Transmission probability: {Q_TRANSMIT}")
    print(f"  Idle timeout: {TS_IDLE} slots")
    print(f"  Simulation time: {SIM_TIME} slots")
    
    # Create and run simulation
    print("\n" + "-" * 70)
    print("Running simulation... (this takes a few seconds)")
    print("-" * 70)
    
    env = simpy.Environment()
    sim = Simulator(
        env=env,
        n_nodes=N_NODES,
        lambda_arrival=LAMBDA_ARRIVAL,
        q_transmit=Q_TRANSMIT,
        ts_idle=TS_IDLE,
        tw_wakeup=TW_WAKEUP,
        E_initial=E_INITIAL,
        PS=0.1,  # Sleep power
        PW=1.0,  # Wake-up power
        PT=5.0,  # Transmit power
        PB=0.5,  # Busy/idle power
        simulation_time=SIM_TIME,
        seed=42  # For reproducibility
    )
    
    sim.run()
    print("✓ Simulation complete!")
    
    # Collect metrics
    print("\n" + "-" * 70)
    print("Results Summary")
    print("-" * 70)
    
    metrics = sim.collect_metrics()
    
    print(f"\nLifetime:")
    print(f"  Average: {metrics['avg_lifetime']:.2f} slots")
    print(f"  Range: [{metrics['min_lifetime']:.0f}, {metrics['max_lifetime']:.0f}]")
    
    print(f"\nDelay:")
    print(f"  Average: {metrics['avg_delay']:.2f} slots")
    print(f"  Maximum: {metrics['max_delay']:.2f} slots")
    
    print(f"\nThroughput:")
    print(f"  Total: {metrics['total_throughput']:.4f} packets/slot")
    print(f"  Per node: {metrics['avg_node_throughput']:.4f} packets/slot")
    
    print(f"\nTransmissions:")
    print(f"  Packets sent: {metrics['total_packets_sent']}")
    print(f"  Packets arrived: {metrics['total_packets_arrived']}")
    print(f"  Delivery ratio: {metrics['packet_delivery_ratio']:.1%}")
    print(f"  Collision rate: {metrics['collision_rate']:.1%}")
    
    print(f"\nEnergy:")
    print(f"  Average remaining: {metrics['avg_energy_remaining']:.2f}")
    print(f"  Consumed: {metrics['energy_consumed_ratio']:.1%}")
    
    # Generate visualization
    if HAS_VISUALIZER:
        print("\n" + "-" * 70)
        print("Generating visualization...")
        print("-" * 70)
        print("\n(Close the plot window to continue)")
        
        try:
            plot_metrics_summary(metrics)
            print("\n✓ Visualization displayed!")
        except Exception as e:
            print(f"\n⚠ Visualization failed: {e}")
            print("  (This might happen in headless environments)")
    else:
        print("\n" + "-" * 70)
        print("Skipping visualization (packages not installed)")
        print("Run 'pip install -r requirements.txt' to enable")
        print("-" * 70)
    
    # Show next steps
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Try different parameters by editing this script")
    print("2. Run example_usage.py for advanced examples:")
    print("   - Parameter sweeps")
    print("   - Optimization")
    print("   - Tradeoff analysis")
    print("\n3. Read README.md for full documentation")
    print("\n" + "=" * 70)
    print("Quick start complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
