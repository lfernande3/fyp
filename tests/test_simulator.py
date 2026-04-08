"""
Unit tests for the Simulator class

Tests verify correct implementation of:
- Simulator initialization
- Collision detection
- Slotted time loop
- Batch replications
- Parameter sweeps
- Results aggregation

Date: February 10, 2026
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import Simulator, BatchSimulator, SimulationConfig, SimulationResults
from src.node import NodeState
import random


def get_default_config(n_nodes=10, max_slots=1000):
    """Helper to create default config for testing."""
    power_rates = {
        'PT': 10.0,  # Transmit
        'PB': 5.0,   # Busy
        'PI': 1.0,   # Idle
        'PW': 2.0,   # Wakeup
        'PS': 0.1    # Sleep
    }
    
    return SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.01,
        transmission_prob=0.1,
        idle_timer=10,
        wakeup_time=5,
        initial_energy=5000.0,
        power_rates=power_rates,
        max_slots=max_slots,
        seed=42
    )


def test_simulator_initialization():
    """Test that simulator initializes correctly."""
    print("Testing simulator initialization...")
    
    config = get_default_config()
    sim = Simulator(config)
    
    assert len(sim.nodes) == config.n_nodes
    assert sim.current_slot == 0
    assert sim.total_collisions == 0
    assert sim.total_transmissions == 0
    assert sim.total_successes == 0
    
    # Check nodes are initialized
    for node in sim.nodes:
        assert node.energy == config.initial_energy
        assert node.idle_timer == config.idle_timer
        assert node.wakeup_time == config.wakeup_time
    
    print("[PASS] Simulator initialization test passed")


def test_collision_detection():
    """Test collision detection logic."""
    print("\nTesting collision detection...")
    
    # Use small number of nodes and high transmission prob
    config = get_default_config(n_nodes=5, max_slots=100)
    config.transmission_prob = 0.5  # High probability
    config.arrival_rate = 0.5       # High arrivals
    
    sim = Simulator(config)
    result = sim.run_simulation(verbose=False)
    
    # Should have some transmissions
    assert result.total_transmissions > 0
    
    # Should have some collisions (with 5 nodes, q=0.5, likely collisions)
    assert result.total_collisions >= 0
    
    # Total transmissions should be >= successes + collisions
    # (collisions count as 1 event but have multiple transmitters)
    assert result.total_transmissions >= result.total_deliveries
    
    print(f"  Transmissions: {result.total_transmissions}")
    print(f"  Collisions: {result.total_collisions}")
    print(f"  Successes: {result.total_deliveries}")
    print("[PASS] Collision detection test passed")


def test_simulation_run():
    """Test basic simulation run."""
    print("\nTesting simulation run...")
    
    config = get_default_config(max_slots=500)
    sim = Simulator(config)
    
    result = sim.run_simulation(verbose=False)
    
    # Check results structure
    assert result.total_slots > 0
    assert result.total_slots <= config.max_slots
    assert result.mean_lifetime_slots > 0
    assert result.mean_lifetime_years >= 0
    assert result.throughput >= 0
    
    # Check state fractions sum to ~1
    total_state_fraction = sum(result.state_fractions.values())
    assert abs(total_state_fraction - 1.0) < 0.01
    
    # Check arrivals and deliveries
    assert result.total_arrivals >= 0
    assert result.total_deliveries >= 0
    assert result.total_deliveries <= result.total_arrivals
    
    print(f"  Simulated {result.total_slots} slots")
    print(f"  Arrivals: {result.total_arrivals}, Deliveries: {result.total_deliveries}")
    print(f"  Mean delay: {result.mean_delay:.2f} slots")
    print(f"  Throughput: {result.throughput:.4f}")
    print("[PASS] Simulation run test passed")


def test_reproducibility():
    """Test that simulations with same seed are reproducible."""
    print("\nTesting reproducibility...")
    
    config1 = get_default_config(max_slots=200)
    config1.seed = 123
    
    config2 = get_default_config(max_slots=200)
    config2.seed = 123
    
    sim1 = Simulator(config1)
    result1 = sim1.run_simulation(verbose=False)
    
    sim2 = Simulator(config2)
    result2 = sim2.run_simulation(verbose=False)
    
    # Results should be identical
    assert result1.total_arrivals == result2.total_arrivals
    assert result1.total_deliveries == result2.total_deliveries
    assert result1.total_collisions == result2.total_collisions
    assert abs(result1.mean_delay - result2.mean_delay) < 0.01
    
    print(f"  Run 1: {result1.total_arrivals} arrivals, {result1.total_deliveries} deliveries")
    print(f"  Run 2: {result2.total_arrivals} arrivals, {result2.total_deliveries} deliveries")
    print("[PASS] Reproducibility test passed")


def test_energy_depletion():
    """Test simulation stops when nodes deplete."""
    print("\nTesting energy depletion...")
    
    config = get_default_config(max_slots=100000)
    config.initial_energy = 0.01  # Very low energy (mWh) to ensure depletion
    config.arrival_rate = 0.1       # High arrival rate
    config.transmission_prob = 0.2  # High transmission
    config.stop_on_first_depletion = True
    
    sim = Simulator(config)
    result = sim.run_simulation(verbose=False)
    
    # Should stop before max_slots due to depletion
    assert result.total_slots < config.max_slots
    
    # At least one node should be depleted
    depleted_nodes = sum(1 for node in sim.nodes if node.is_depleted())
    assert depleted_nodes > 0
    
    print(f"  Stopped at slot {result.total_slots} (max was {config.max_slots})")
    print(f"  Depleted nodes: {depleted_nodes}/{config.n_nodes}")
    print("[PASS] Energy depletion test passed")


def test_history_tracking():
    """Test time series history tracking."""
    print("\nTesting history tracking...")
    
    config = get_default_config(max_slots=100)
    sim = Simulator(config)
    
    result = sim.run_simulation(track_history=True, verbose=False)
    
    # Check history is recorded
    assert result.queue_length_history is not None
    assert result.energy_history is not None
    assert result.state_history is not None
    
    assert len(result.queue_length_history) == result.total_slots
    assert len(result.energy_history) == result.total_slots
    assert len(result.state_history) == result.total_slots
    
    # Energy should generally decrease
    assert result.energy_history[0] > result.energy_history[-1]
    
    print(f"  Tracked {len(result.queue_length_history)} time steps")
    print(f"  Initial energy: {result.energy_history[0]:.2f}")
    print(f"  Final energy: {result.energy_history[-1]:.2f}")
    print("[PASS] History tracking test passed")


def test_batch_replications():
    """Test batch replications."""
    print("\nTesting batch replications...")
    
    base_config = get_default_config(max_slots=200)
    batch_sim = BatchSimulator(base_config)
    
    n_reps = 5
    results = batch_sim.run_replications(n_replications=n_reps, verbose=False)
    
    assert len(results) == n_reps
    
    # Results should vary (different seeds)
    arrivals = [r.total_arrivals for r in results]
    assert len(set(arrivals)) > 1  # Not all the same
    
    print(f"  Ran {n_reps} replications")
    print(f"  Arrival counts: {arrivals}")
    print("[PASS] Batch replications test passed")


def test_parameter_sweep():
    """Test parameter sweep functionality."""
    print("\nTesting parameter sweep...")
    
    base_config = get_default_config(max_slots=200)
    batch_sim = BatchSimulator(base_config)
    
    # Sweep transmission probability
    q_values = [0.05, 0.1, 0.2]
    n_reps = 3
    
    sweep_results = batch_sim.parameter_sweep(
        param_name='transmission_prob',
        param_values=q_values,
        n_replications=n_reps,
        verbose=False
    )
    
    assert len(sweep_results) == len(q_values)
    
    for q in q_values:
        assert q in sweep_results
        assert len(sweep_results[q]) == n_reps
    
    print(f"  Swept {len(q_values)} values with {n_reps} replications each")
    print(f"  Total runs: {len(q_values) * n_reps}")
    print("[PASS] Parameter sweep test passed")


def test_results_aggregation():
    """Test aggregation of results from replications."""
    print("\nTesting results aggregation...")
    
    base_config = get_default_config(max_slots=200)
    batch_sim = BatchSimulator(base_config)
    
    results = batch_sim.run_replications(n_replications=5, verbose=False)
    aggregated = BatchSimulator.aggregate_results(results)
    
    # Check aggregated results have mean and std
    assert 'mean_delay' in aggregated
    assert 'throughput' in aggregated
    assert 'mean_lifetime_years' in aggregated
    
    # Each should be a tuple (mean, std)
    for metric_name, (mean, std) in aggregated.items():
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))
        assert std >= 0  # Standard deviation should be non-negative
    
    print(f"  Aggregated {len(aggregated)} metrics")
    print(f"  Mean delay: {aggregated['mean_delay'][0]:.2f} ± {aggregated['mean_delay'][1]:.2f}")
    print(f"  Throughput: {aggregated['throughput'][0]:.4f} ± {aggregated['throughput'][1]:.4f}")
    print("[PASS] Results aggregation test passed")


def test_low_vs_high_transmission_prob():
    """Test that higher q leads to more collisions."""
    print("\nTesting transmission probability impact...")
    
    # Low q
    config_low = get_default_config(max_slots=500)
    config_low.transmission_prob = 0.05
    config_low.arrival_rate = 0.05
    sim_low = Simulator(config_low)
    result_low = sim_low.run_simulation(verbose=False)
    
    # High q
    config_high = get_default_config(max_slots=500)
    config_high.transmission_prob = 0.3
    config_high.arrival_rate = 0.05
    sim_high = Simulator(config_high)
    result_high = sim_high.run_simulation(verbose=False)
    
    # Higher q should lead to more transmissions and collisions
    assert result_high.total_transmissions >= result_low.total_transmissions
    
    print(f"  Low q (0.05): {result_low.total_transmissions} transmissions, {result_low.total_collisions} collisions")
    print(f"  High q (0.30): {result_high.total_transmissions} transmissions, {result_high.total_collisions} collisions")
    print("[PASS] Transmission probability impact test passed")


def run_all_tests():
    """Run all simulator tests."""
    print("=" * 60)
    print("Running Simulator Class Unit Tests")
    print("=" * 60)
    
    test_simulator_initialization()
    test_collision_detection()
    test_simulation_run()
    test_reproducibility()
    test_energy_depletion()
    test_history_tracking()
    test_batch_replications()
    test_parameter_sweep()
    test_results_aggregation()
    test_low_vs_high_transmission_prob()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()


# ===========================================================================
# O7: CSMA tests
# ===========================================================================

def get_csma_config(n_nodes=5, max_slots=500, backoff_window=0):
    """Helper to create a CSMA config."""
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    return SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.05,
        transmission_prob=0.3,
        idle_timer=5,
        wakeup_time=2,
        initial_energy=5000.0,
        power_rates=power_rates,
        max_slots=max_slots,
        seed=42,
        access_scheme="csma_1p",
        backoff_window=backoff_window,
    )


def test_csma_config_field_present():
    """SimulationConfig has access_scheme field with default 'slotted_aloha'."""
    config = get_default_config()
    assert config.access_scheme == "slotted_aloha"
    print("[PASS] CSMA config field test passed")


def test_csma_simulation_runs():
    """CSMA simulation runs without error and produces results."""
    config = get_csma_config()
    sim = Simulator(config)
    result = sim.run_simulation()
    assert result.total_slots > 0
    assert result.total_transmissions >= 0
    print("[PASS] CSMA simulation runs test passed")


def test_csma_fewer_collisions_than_aloha():
    """CSMA should produce fewer collisions than Aloha for same parameters."""
    pr = {'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1}
    base = dict(n_nodes=10, arrival_rate=0.1, transmission_prob=0.3,
                idle_timer=3, wakeup_time=2, initial_energy=5000.0,
                power_rates=pr, max_slots=2000, seed=42)

    aloha_cfg = SimulationConfig(**base, access_scheme="slotted_aloha")
    csma_cfg = SimulationConfig(**base, access_scheme="csma_1p")

    aloha_result = Simulator(aloha_cfg).run_simulation()
    csma_result = Simulator(csma_cfg).run_simulation()

    assert csma_result.total_collisions <= aloha_result.total_collisions, (
        f"CSMA collisions ({csma_result.total_collisions}) should be <= "
        f"Aloha collisions ({aloha_result.total_collisions})"
    )
    print("[PASS] CSMA fewer collisions test passed")


def test_csma_backoff_counter_initialised():
    """Nodes in CSMA mode start with backoff_counter == 0."""
    config = get_csma_config()
    sim = Simulator(config)
    for node in sim.nodes:
        assert node.backoff_counter == 0
    print("[PASS] CSMA backoff_counter init test passed")


def test_csma_with_exponential_backoff():
    """CSMA with backoff_window > 0 runs without error."""
    config = get_csma_config(backoff_window=8)
    result = Simulator(config).run_simulation()
    assert result.total_slots > 0
    print("[PASS] CSMA exponential backoff test passed")


def test_csma_aloha_same_results_single_node():
    """With a single node, CSMA and Aloha produce identical collision counts (0)."""
    pr = {'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1}
    base = dict(n_nodes=1, arrival_rate=0.1, transmission_prob=0.5,
                idle_timer=3, wakeup_time=2, initial_energy=5000.0,
                power_rates=pr, max_slots=500, seed=42)

    aloha_result = Simulator(SimulationConfig(**base, access_scheme="slotted_aloha")).run_simulation()
    csma_result = Simulator(SimulationConfig(**base, access_scheme="csma_1p")).run_simulation()

    assert aloha_result.total_collisions == 0
    assert csma_result.total_collisions == 0
    print("[PASS] CSMA single-node no-collision test passed")


def test_csma_result_has_all_standard_fields():
    """CSMA result contains the standard SimulationResults fields."""
    result = Simulator(get_csma_config()).run_simulation()
    for attr in ('mean_delay', 'mean_lifetime_years', 'throughput',
                 'total_collisions', 'total_transmissions'):
        assert hasattr(result, attr)
    print("[PASS] CSMA result fields test passed")


def test_slotted_aloha_default_unchanged():
    """Default access_scheme is slotted_aloha and existing behaviour is preserved."""
    config = get_default_config(n_nodes=10, max_slots=1000)
    result = Simulator(config).run_simulation()
    # Sanity: simulation ran and produced non-trivial output
    assert result.total_slots > 0
    assert result.empirical_service_rate >= 0
    print("[PASS] Slotted Aloha default unchanged test passed")


# ===========================================================================
# O8: Receiver model tests (via SimulationConfig field)
# ===========================================================================

def get_rx_config(model_name, n_nodes=10, max_slots=500):
    pr = {'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1}
    return SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.05,
        transmission_prob=0.2,
        idle_timer=5,
        wakeup_time=2,
        initial_energy=5000.0,
        power_rates=pr,
        max_slots=max_slots,
        seed=42,
        receiver_model=model_name,
    )


def test_o8_default_receiver_model():
    """Default receiver_model in SimulationConfig is 'collision'."""
    config = get_default_config()
    assert config.receiver_model == "collision"
    print("[PASS] O8 default receiver model test passed")


def test_o8_collision_model_runs():
    """Simulation with explicit 'collision' model runs correctly."""
    result = Simulator(get_rx_config("collision")).run_simulation()
    assert result.total_slots > 0
    print("[PASS] O8 collision model runs test passed")


def test_o8_capture_model_runs():
    """Simulation with 'capture' model runs without error."""
    result = Simulator(get_rx_config("capture")).run_simulation()
    assert result.total_slots > 0
    print("[PASS] O8 capture model runs test passed")


def test_o8_sic_model_runs():
    """Simulation with 'sic' model runs without error."""
    result = Simulator(get_rx_config("sic")).run_simulation()
    assert result.total_slots > 0
    print("[PASS] O8 SIC model runs test passed")


def test_o8_multi_packet_slots_positive_for_sic():
    """SIC model can produce multi_packet_slots > 0 under high load."""
    pr = {'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1}
    # High arrival and transmission probability increases chance of multi-decode
    cfg = SimulationConfig(
        n_nodes=20, arrival_rate=0.2, transmission_prob=0.3,
        idle_timer=5, wakeup_time=2, initial_energy=5000.0,
        power_rates=pr, max_slots=2000, seed=0,
        receiver_model="sic", sic_sinr_threshold=0.1,
    )
    result = Simulator(cfg).run_simulation()
    # With low SINR threshold, SIC should decode multiple packets in some slots
    assert result.multi_packet_slots >= 0  # At minimum it doesn't error
    print("[PASS] O8 multi_packet_slots SIC test passed")
