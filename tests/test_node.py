"""
Unit tests for the Node class

Tests verify correct implementation of:
- State transitions
- Packet arrival and queueing
- Transmission attempts
- Energy consumption
- Statistics tracking

Date: February 10, 2026
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.node import Node, NodeState
import random


def test_node_initialization():
    """Test that node initializes correctly."""
    print("Testing node initialization...")
    
    power_rates = {
        'PT': 10.0,   # Transmit
        'PB': 5.0,    # Busy
        'PI': 1.0,    # Idle
        'PW': 2.0,    # Wakeup
        'PS': 0.1     # Sleep
    }
    
    node = Node(
        node_id=1,
        initial_energy=1000.0,
        idle_timer=10,
        wakeup_time=5,
        power_rates=power_rates
    )
    
    assert node.node_id == 1
    assert node.energy == 1000.0
    assert node.state == NodeState.IDLE
    assert len(node.queue) == 0
    assert node.idle_timer == 10
    assert node.wakeup_time == 5
    
    print("[PASS] Node initialization test passed")


def test_packet_arrival():
    """Test packet arrival mechanism."""
    print("\nTesting packet arrival...")
    
    random.seed(42)  # For reproducibility
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 1000.0, 10, 5, power_rates)
    
    # Force packet arrival (arrival_rate = 1.0)
    arrived = node.arrive_packet(current_slot=0, arrival_rate=1.0)
    
    assert arrived == True
    assert len(node.queue) == 1
    assert node.packets_arrived == 1
    assert node.state == NodeState.ACTIVE  # Should transition from IDLE to ACTIVE
    
    print("[PASS] Packet arrival test passed")


def test_state_transitions():
    """Test state transitions (IDLE -> ACTIVE -> IDLE -> SLEEP)."""
    print("\nTesting state transitions...")
    
    random.seed(42)
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 1000.0, idle_timer=3, wakeup_time=2, power_rates=power_rates)
    
    # Initial state: IDLE
    assert node.state == NodeState.IDLE
    
    # Packet arrives -> ACTIVE
    node.arrive_packet(0, 1.0)
    assert node.state == NodeState.ACTIVE
    
    # Successful transmission -> queue empty -> IDLE
    node.handle_success(1)
    node.update_state(1)
    assert node.state == NodeState.IDLE
    assert len(node.queue) == 0
    
    # Wait idle_timer slots -> SLEEP
    for i in range(3):
        node.update_state(i + 2)
    
    assert node.state == NodeState.SLEEP
    
    # Packet arrives during sleep -> WAKEUP
    node.arrive_packet(5, 1.0)
    assert node.state == NodeState.WAKEUP
    assert node.current_wakeup_count == 2
    
    # Wait wakeup_time slots -> ACTIVE
    node.update_state(6)
    assert node.current_wakeup_count == 1
    node.update_state(7)
    assert node.state == NodeState.ACTIVE
    assert node.current_wakeup_count == 0
    
    print("[PASS] State transitions test passed")


def test_energy_consumption():
    """Test energy consumption in different states."""
    print("\nTesting energy consumption...")
    
    power_rates = {
        'PT': 10.0,   # Transmit
        'PB': 5.0,    # Busy
        'PI': 1.0,    # Idle
        'PW': 2.0,    # Wakeup
        'PS': 0.1     # Sleep
    }
    
    node = Node(1, 1000.0, 10, 5, power_rates)
    initial_energy = node.energy
    
    # Test IDLE energy consumption
    node.state = NodeState.IDLE
    consumed = node.consume_energy(was_transmitting=False, was_collision=False)
    assert consumed == 1.0  # PI
    assert node.energy == initial_energy - 1.0
    
    # Test SLEEP energy consumption
    node.state = NodeState.SLEEP
    consumed = node.consume_energy()
    assert consumed == 0.1  # PS
    assert node.energy == initial_energy - 1.1
    
    # Test TRANSMIT energy consumption
    node.state = NodeState.ACTIVE
    consumed = node.consume_energy(was_transmitting=True)
    assert consumed == 10.0  # PT
    assert node.energy == initial_energy - 11.1
    
    # Test WAKEUP energy consumption
    node.state = NodeState.WAKEUP
    consumed = node.consume_energy()
    assert consumed == 2.0  # PW
    assert node.energy == initial_energy - 13.1
    
    print("[PASS] Energy consumption test passed")


def test_transmission_attempt():
    """Test transmission attempt logic."""
    print("\nTesting transmission attempt...")
    
    random.seed(42)
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 1000.0, 10, 5, power_rates)
    
    # No transmission if not ACTIVE or no packets
    assert node.attempt_transmit(1.0) == False
    
    # Add packet and set to ACTIVE
    node.arrive_packet(0, 1.0)
    node.state = NodeState.ACTIVE
    
    # With q=1.0, should always attempt
    assert node.attempt_transmit(1.0) == True
    
    # With q=0.0, should never attempt
    assert node.attempt_transmit(0.0) == False
    
    print("[PASS] Transmission attempt test passed")


def test_delay_calculation():
    """Test delay calculation on successful transmission."""
    print("\nTesting delay calculation...")
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 1000.0, 10, 5, power_rates)
    
    # Packet arrives at slot 0
    node.queue.append((0, {}))
    node.packets_arrived = 1
    
    # Successfully transmitted at slot 10
    delay = node.handle_success(10)
    
    assert delay == 10  # 10 - 0 = 10 slots delay
    assert len(node.queue) == 0
    assert node.packets_delivered == 1
    assert node.total_delay == 10
    assert node.get_mean_delay() == 10.0
    
    # Add more packets to test mean
    node.queue.append((5, {}))
    node.packets_arrived = 2
    delay2 = node.handle_success(10)
    
    assert delay2 == 5  # 10 - 5 = 5 slots delay
    assert node.packets_delivered == 2
    assert node.get_mean_delay() == 7.5  # (10 + 5) / 2
    
    print("[PASS] Delay calculation test passed")


def test_statistics():
    """Test statistics gathering."""
    print("\nTesting statistics gathering...")
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 1000.0, 10, 5, power_rates)
    
    # Simulate some activity
    node.arrive_packet(0, 1.0)
    node.state = NodeState.ACTIVE
    node.time_in_state[NodeState.ACTIVE] = 50
    node.time_in_state[NodeState.IDLE] = 30
    node.time_in_state[NodeState.SLEEP] = 20
    
    total_slots = 100
    
    # Get statistics
    stats = node.get_statistics(total_slots)
    
    assert stats['node_id'] == 1
    assert stats['packets_arrived'] == 1
    assert stats['packets_in_queue'] == 1
    assert 'energy_remaining' in stats
    assert 'state_fractions' in stats
    
    # Check state fractions
    state_fractions = stats['state_fractions']
    assert state_fractions['active'] == 0.5
    assert state_fractions['idle'] == 0.3
    assert state_fractions['sleep'] == 0.2
    
    print("[PASS] Statistics test passed")


def test_energy_depletion():
    """Test energy depletion detection."""
    print("\nTesting energy depletion...")
    
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    node = Node(1, 50.0, 10, 5, power_rates)  # Low initial energy
    
    assert node.is_depleted() == False
    
    # Consume energy until depleted
    node.state = NodeState.ACTIVE
    for _ in range(6):  # 6 * 10 = 60 > 50
        node.consume_energy(was_transmitting=True)
    
    assert node.energy < 0
    assert node.is_depleted() == True
    
    print("[PASS] Energy depletion test passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Node Class Unit Tests")
    print("=" * 60)
    
    test_node_initialization()
    test_packet_arrival()
    test_state_transitions()
    test_energy_consumption()
    test_transmission_attempt()
    test_delay_calculation()
    test_statistics()
    test_energy_depletion()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
