"""
Configuration File for On-Demand Sleep-Based Aloha Simulator

Edit the parameters in this file to customize your simulations.
All important variables are centralized here for easy access.

Quick Guide:
- Increase n_nodes for larger networks (5-50)
- Increase lambda_arrival for higher traffic load (0.01-0.5)
- Adjust q_transmit to control transmission aggressiveness (0.01-0.3)
- Increase ts_idle for less frequent sleep (1-20)
- Increase E_initial for longer battery life
- Increase simulation_time for longer simulations (at cost of speed)
"""

# =============================================================================
# SYSTEM PARAMETERS - Network Configuration
# =============================================================================

# Number of nodes (MTDs) in the network
# - Typical range: 5-50
# - More nodes = more collisions but higher total throughput
n_nodes = 10

# Packet arrival rate (Bernoulli probability per slot)
# - Symbol: λ (lambda)
# - Range: 0.01 to 0.5
# - Higher values = more traffic but more collisions
lambda_arrival = 0.1

# Transmission probability per slot (for active nodes)
# - Symbol: q
# - Range: 0.01 to 0.3
# - Higher values = faster transmission but more collisions
# - This is often the parameter to optimize!
q_transmit = 0.05

# Idle timeout before sleeping (in time slots)
# - Symbol: ts
# - Range: 1 to 20 slots
# - Lower values = sleep sooner (better energy, worse delay)
# - Higher values = stay awake longer (worse energy, better delay)
ts_idle = 5

# Wake-up duration (in time slots)
# - Symbol: tw
# - Range: 1 to 5 slots
# - Time needed to transition from sleep to active
tw_wakeup = 2


# =============================================================================
# ENERGY PARAMETERS - Power Consumption Model
# =============================================================================

# Initial energy per node (arbitrary units)
# - Higher values = longer node lifetime
# - Typical: 10000 for moderate simulations
E_initial = 10000

# Power consumption in SLEEP state (very low)
# - Symbol: PS
# - Should be much smaller than other states (e.g., 0.1)
PS_sleep = 0.1

# Power consumption during WAKE-UP transition
# - Symbol: PW
# - Moderate power (e.g., 1.0)
PW_wakeup = 1.0

# Power consumption when TRANSMITTING
# - Symbol: PT
# - Highest power state (e.g., 5.0)
PT_transmit = 5.0

# Power consumption when BUSY/IDLE (active but not transmitting)
# - Symbol: PB
# - Low to moderate power (e.g., 0.5)
PB_busy = 0.5


# =============================================================================
# SIMULATION PARAMETERS - Runtime Configuration
# =============================================================================

# Total simulation time in slots
# - Typical range: 1000-10000
# - Longer simulations = more accurate but slower
# - Start with 3000-5000 for testing
simulation_time = 5000

# Random seed for reproducibility
# - Set to an integer (e.g., 42) for reproducible results
# - Set to None for different results each run
random_seed = 42


# =============================================================================
# OPTIMIZATION PARAMETERS - For Parameter Sweeps and Optimization
# =============================================================================

# Range for q optimization
# - Tuple: (minimum_q, maximum_q)
q_optimization_range = (0.01, 0.3)

# Number of samples for q optimization
# - More samples = better accuracy but slower
# - Typical: 15-20 for quick tests, 30-50 for final results
q_optimization_samples = 20

# Number of runs per configuration (for statistical averaging)
# - More runs = more accurate statistics but slower
# - Typical: 1 for quick tests, 3-5 for final results
n_runs_for_averaging = 3

# ts values for tradeoff analysis
# - Array of idle timeout values to test
# - These define the lifetime vs delay tradeoff curve
ts_tradeoff_values = [1, 3, 5, 10, 15, 20]

# Enable parallel processing for parameter sweeps
# - Set to True to use all CPU cores (faster for many simulations)
# - Set to False for sequential execution (easier debugging)
enable_parallel_processing = False


# =============================================================================
# VISUALIZATION PARAMETERS - Plot Configuration
# =============================================================================

# Whether to save plots to files (in addition to displaying them)
save_plots = False

# Output directory for saved plots and results
output_directory = "results"

# Plot file format
# - Options: 'png', 'pdf', 'svg'
plot_format = 'png'

# Plot DPI (resolution)
# - Higher = better quality but larger files
# - Typical: 150 for screen, 300 for publication
plot_dpi = 150

# Whether to show plots in GUI windows
# - Set to False for headless environments or batch processing
show_plots = True


# =============================================================================
# ADVANCED PARAMETERS - Fine-tuning (usually don't need to change)
# =============================================================================

# Packet size in bytes (currently not used in energy calculation)
packet_size = 1000

# Maximum buffer size per node
# - None = infinite buffer (as in the paper)
# - Set to integer (e.g., 100) for finite buffer with packet dropping
max_buffer_size = None

# Verbose output during simulation
# - True = print progress messages
# - False = quiet mode (only final results)
verbose = True


# =============================================================================
# PRESET CONFIGURATIONS - Quick scenario selection
# =============================================================================

class Presets:
    """
    Predefined configurations for common scenarios.
    
    Usage:
        from config import Presets
        params = Presets.LOW_TRAFFIC
    """
    
    # Low traffic scenario (λ=0.05)
    LOW_TRAFFIC = {
        'n_nodes': 10,
        'lambda_arrival': 0.05,
        'q_transmit': 0.08,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }
    
    # Medium traffic scenario (λ=0.1)
    MEDIUM_TRAFFIC = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }
    
    # High traffic scenario (λ=0.2)
    HIGH_TRAFFIC = {
        'n_nodes': 10,
        'lambda_arrival': 0.2,
        'q_transmit': 0.03,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }
    
    # Large network scenario (50 nodes)
    LARGE_NETWORK = {
        'n_nodes': 50,
        'lambda_arrival': 0.05,
        'q_transmit': 0.02,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }
    
    # Fast testing (short simulation)
    QUICK_TEST = {
        'n_nodes': 5,
        'lambda_arrival': 0.1,
        'q_transmit': 0.05,
        'ts_idle': 5,
        'tw_wakeup': 2,
        'E_initial': 5000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 1000,
    }
    
    # Energy-optimized (aggressive sleep)
    ENERGY_OPTIMIZED = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.08,  # Higher q for faster transmission
        'ts_idle': 1,        # Sleep immediately after buffer empty
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.05,  # Very low sleep power
        'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }
    
    # Delay-optimized (avoid sleep)
    DELAY_OPTIMIZED = {
        'n_nodes': 10,
        'lambda_arrival': 0.1,
        'q_transmit': 0.03,  # Lower q for fewer collisions
        'ts_idle': 20,       # Stay awake long time before sleep
        'tw_wakeup': 2,
        'E_initial': 10000,
        'PS': 0.1, 'PW': 1.0, 'PT': 5.0, 'PB': 0.5,
        'simulation_time': 5000,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_base_params():
    """
    Get current configuration as a parameter dictionary.
    
    Returns:
        dict: Dictionary of all simulation parameters
    """
    return {
        'n_nodes': n_nodes,
        'lambda_arrival': lambda_arrival,
        'q_transmit': q_transmit,
        'ts_idle': ts_idle,
        'tw_wakeup': tw_wakeup,
        'E_initial': E_initial,
        'PS': PS_sleep,
        'PW': PW_wakeup,
        'PT': PT_transmit,
        'PB': PB_busy,
        'simulation_time': simulation_time,
        'seed': random_seed,
    }


def print_config():
    """Print current configuration in a readable format."""
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    
    print("\nSystem Parameters:")
    print(f"  n_nodes         : {n_nodes}")
    print(f"  lambda_arrival  : {lambda_arrival}")
    print(f"  q_transmit      : {q_transmit}")
    print(f"  ts_idle         : {ts_idle} slots")
    print(f"  tw_wakeup       : {tw_wakeup} slots")
    
    print("\nEnergy Parameters:")
    print(f"  E_initial       : {E_initial}")
    print(f"  PS_sleep        : {PS_sleep}")
    print(f"  PW_wakeup       : {PW_wakeup}")
    print(f"  PT_transmit     : {PT_transmit}")
    print(f"  PB_busy         : {PB_busy}")
    
    print("\nSimulation Parameters:")
    print(f"  simulation_time : {simulation_time} slots")
    print(f"  random_seed     : {random_seed}")
    
    print("\nOptimization Parameters:")
    print(f"  q_range         : {q_optimization_range}")
    print(f"  q_samples       : {q_optimization_samples}")
    print(f"  n_runs          : {n_runs_for_averaging}")
    print(f"  parallel        : {enable_parallel_processing}")
    
    print("\n" + "=" * 70)


def validate_config():
    """
    Validate configuration parameters and warn about potential issues.
    
    Returns:
        bool: True if configuration is valid
    """
    valid = True
    warnings = []
    
    # Check ranges
    if not (1 <= n_nodes <= 100):
        warnings.append(f"n_nodes={n_nodes} is unusual (typical: 5-50)")
    
    if not (0.001 <= lambda_arrival <= 0.5):
        warnings.append(f"lambda_arrival={lambda_arrival} outside typical range (0.01-0.5)")
    
    if not (0.001 <= q_transmit <= 0.5):
        warnings.append(f"q_transmit={q_transmit} outside typical range (0.01-0.3)")
    
    if not (1 <= ts_idle <= 50):
        warnings.append(f"ts_idle={ts_idle} outside typical range (1-20)")
    
    if not (1 <= tw_wakeup <= 10):
        warnings.append(f"tw_wakeup={tw_wakeup} outside typical range (1-5)")
    
    # Check power hierarchy
    if not (PS_sleep < PB_busy < PT_transmit):
        warnings.append("Power hierarchy should be: PS < PB < PT")
    
    # Check simulation time
    if simulation_time < 100:
        warnings.append(f"simulation_time={simulation_time} is very short (typical: 1000+)")
    
    if simulation_time > 50000:
        warnings.append(f"simulation_time={simulation_time} is very long (will be slow)")
    
    # Check traffic intensity
    network_load = n_nodes * lambda_arrival * q_transmit
    if network_load > 0.368:  # Aloha capacity
        warnings.append(f"Network load={network_load:.3f} exceeds Aloha capacity (0.368)")
    
    # Print warnings
    if warnings:
        print("\n⚠ Configuration Warnings:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    return valid


# =============================================================================
# AUTO-RUN (when importing this file)
# =============================================================================

if __name__ == "__main__":
    # If run directly, print and validate configuration
    print_config()
    validate_config()
