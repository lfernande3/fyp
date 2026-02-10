"""
Unit tests for traffic models module.

Tests different traffic arrival patterns including Poisson, bursty, and periodic.
"""

import pytest
import numpy as np
from src.traffic_models import (
    TrafficModel, TrafficConfig, TrafficGenerator,
    create_poisson_traffic, create_bursty_traffic, create_periodic_traffic
)


def test_poisson_traffic():
    """Test Poisson traffic generation."""
    print("\nTesting Poisson traffic generation...")
    
    config = TrafficConfig(
        model=TrafficModel.POISSON,
        arrival_rate=0.1
    )
    generator = TrafficGenerator(config, seed=42)
    
    n_nodes = 10
    n_slots = 1000
    
    # Generate arrivals over many slots
    total_arrivals = 0
    for slot in range(n_slots):
        arrivals = generator.generate_arrivals(n_nodes, slot)
        total_arrivals += len(arrivals)
    
    # Expected arrivals = n_nodes * n_slots * arrival_rate
    expected = n_nodes * n_slots * 0.1
    
    # Should be close to expected (within 20% due to randomness)
    assert abs(total_arrivals - expected) / expected < 0.2
    
    print(f"  Generated {total_arrivals} arrivals (expected ~{expected:.0f})")
    print("[PASS] Poisson traffic test passed")


def test_bursty_traffic():
    """Test bursty traffic generation."""
    print("\nTesting bursty traffic generation...")
    
    config = TrafficConfig(
        model=TrafficModel.BURSTY,
        arrival_rate=0.01,
        burst_prob=0.05,
        burst_size_mean=5.0,
        burst_size_std=1.0
    )
    generator = TrafficGenerator(config, seed=42)
    
    n_nodes = 10
    n_slots = 1000
    
    # Track burst events
    burst_slots = []
    total_arrivals = 0
    
    for slot in range(n_slots):
        arrivals = generator.generate_arrivals(n_nodes, slot)
        total_arrivals += len(arrivals)
        if len(arrivals) > 1:  # Likely a burst
            burst_slots.append(slot)
    
    # Should have bursts (multiple arrivals in some slots)
    assert len(burst_slots) > 0
    
    # Arrival rate should be higher than baseline due to bursts
    effective_rate = total_arrivals / (n_nodes * n_slots)
    assert effective_rate > 0.001  # Higher than baseline * 0.1
    
    print(f"  Generated {total_arrivals} arrivals with {len(burst_slots)} burst events")
    print(f"  Effective rate: {effective_rate:.4f}")
    print("[PASS] Bursty traffic test passed")


def test_periodic_traffic():
    """Test periodic traffic generation."""
    print("\nTesting periodic traffic generation...")
    
    period = 50
    jitter = 5
    
    config = TrafficConfig(
        model=TrafficModel.PERIODIC,
        period=period,
        jitter=jitter
    )
    generator = TrafficGenerator(config, seed=42)
    
    n_nodes = 5
    n_slots = 500
    
    # Track arrivals per node
    node_arrivals = {i: [] for i in range(n_nodes)}
    
    for slot in range(n_slots):
        arrivals = generator.generate_arrivals(n_nodes, slot)
        for node_id in arrivals:
            node_arrivals[node_id].append(slot)
    
    # Check each node has periodic arrivals
    for node_id in range(n_nodes):
        arrivals = node_arrivals[node_id]
        
        if len(arrivals) > 1:
            # Calculate inter-arrival times
            inter_arrival = np.diff(arrivals)
            mean_inter_arrival = np.mean(inter_arrival)
            
            # Should be close to period (within jitter tolerance)
            assert abs(mean_inter_arrival - period) < period * 0.3
            
            print(f"  Node {node_id}: {len(arrivals)} arrivals, "
                  f"mean inter-arrival: {mean_inter_arrival:.1f} slots")
    
    print("[PASS] Periodic traffic test passed")


def test_mixed_traffic():
    """Test mixed traffic generation."""
    print("\nTesting mixed traffic generation...")
    
    config = TrafficConfig(
        model=TrafficModel.MIXED,
        arrival_rate=0.05,
        burst_prob=0.03,
        burst_size_mean=4.0,
        bursty_fraction=0.5  # 50% bursty, 50% Poisson
    )
    generator = TrafficGenerator(config, seed=42)
    
    n_nodes = 10
    n_slots = 1000
    
    total_arrivals = 0
    for slot in range(n_slots):
        arrivals = generator.generate_arrivals(n_nodes, slot)
        total_arrivals += len(arrivals)
    
    # Should have arrivals from both types
    assert total_arrivals > 0
    
    print(f"  Generated {total_arrivals} arrivals from mixed traffic")
    print("[PASS] Mixed traffic test passed")


def test_create_poisson_traffic():
    """Test Poisson traffic factory function."""
    print("\nTesting Poisson traffic factory...")
    
    generator = create_poisson_traffic(arrival_rate=0.1, seed=42)
    
    assert generator.config.model == TrafficModel.POISSON
    assert generator.config.arrival_rate == 0.1
    
    # Generate some arrivals
    arrivals = generator.generate_arrivals(10, 0)
    assert isinstance(arrivals, list)
    
    print("[PASS] Poisson traffic factory test passed")


def test_create_bursty_traffic():
    """Test bursty traffic factory function."""
    print("\nTesting bursty traffic factory...")
    
    generator = create_bursty_traffic(
        burst_prob=0.02,
        burst_size_mean=5.0,
        seed=42
    )
    
    assert generator.config.model == TrafficModel.BURSTY
    assert generator.config.burst_prob == 0.02
    assert generator.config.burst_size_mean == 5.0
    
    # Generate some arrivals
    arrivals = generator.generate_arrivals(10, 0)
    assert isinstance(arrivals, list)
    
    print("[PASS] Bursty traffic factory test passed")


def test_create_periodic_traffic():
    """Test periodic traffic factory function."""
    print("\nTesting periodic traffic factory...")
    
    generator = create_periodic_traffic(period=100, jitter=10, seed=42)
    
    assert generator.config.model == TrafficModel.PERIODIC
    assert generator.config.period == 100
    assert generator.config.jitter == 10
    
    # Generate some arrivals
    arrivals = generator.generate_arrivals(10, 0)
    assert isinstance(arrivals, list)
    
    print("[PASS] Periodic traffic factory test passed")


def test_effective_arrival_rate():
    """Test effective arrival rate calculation."""
    print("\nTesting effective arrival rate calculation...")
    
    # Poisson
    config1 = TrafficConfig(model=TrafficModel.POISSON, arrival_rate=0.1)
    gen1 = TrafficGenerator(config1)
    assert abs(gen1.get_effective_arrival_rate(10, 1000) - 0.1) < 1e-6
    
    # Bursty
    config2 = TrafficConfig(
        model=TrafficModel.BURSTY,
        arrival_rate=0.01,
        burst_prob=0.02,
        burst_size_mean=5.0
    )
    gen2 = TrafficGenerator(config2)
    effective_rate = gen2.get_effective_arrival_rate(10, 1000)
    # Should be: burst_prob * burst_size + baseline
    expected = 0.02 * 5.0 + 0.01 * 0.1
    assert abs(effective_rate - expected) < 1e-6
    
    # Periodic
    config3 = TrafficConfig(model=TrafficModel.PERIODIC, period=50)
    gen3 = TrafficGenerator(config3)
    assert abs(gen3.get_effective_arrival_rate(10, 1000) - 1/50) < 1e-6
    
    print("[PASS] Effective arrival rate test passed")


def test_reproducibility():
    """Test that same seed gives same results."""
    print("\nTesting reproducibility with seeds...")
    
    config = TrafficConfig(
        model=TrafficModel.BURSTY,
        burst_prob=0.05,
        burst_size_mean=3.0
    )
    
    # Run twice with same seed
    gen1 = TrafficGenerator(config, seed=123)
    gen2 = TrafficGenerator(config, seed=123)
    
    arrivals1 = []
    arrivals2 = []
    
    for slot in range(100):
        arrivals1.append(gen1.generate_arrivals(10, slot))
        arrivals2.append(gen2.generate_arrivals(10, slot))
    
    # Should be identical
    assert arrivals1 == arrivals2
    
    # Different seed should give different results
    gen3 = TrafficGenerator(config, seed=456)
    arrivals3 = []
    for slot in range(100):
        arrivals3.append(gen3.generate_arrivals(10, slot))
    
    assert arrivals1 != arrivals3
    
    print("[PASS] Reproducibility test passed")
