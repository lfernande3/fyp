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

from src.node import Node, NodeState, _SLOT_DURATION_H
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
    h = _SLOT_DURATION_H  # power (mW) × h → energy (mWh)
    
    # Test IDLE energy consumption
    node.state = NodeState.IDLE
    consumed = node.consume_energy(was_transmitting=False, was_collision=False)
    assert abs(consumed - 1.0 * h) < 1e-18  # PI
    assert abs(node.energy - (initial_energy - 1.0 * h)) < 1e-18
    
    # Test SLEEP energy consumption
    node.state = NodeState.SLEEP
    consumed = node.consume_energy()
    assert abs(consumed - 0.1 * h) < 1e-18  # PS
    assert abs(node.energy - (initial_energy - 1.1 * h)) < 1e-18
    
    # Test TRANSMIT energy consumption
    node.state = NodeState.ACTIVE
    consumed = node.consume_energy(was_transmitting=True)
    assert abs(consumed - 10.0 * h) < 1e-18  # PT
    assert abs(node.energy - (initial_energy - 11.1 * h)) < 1e-18
    
    # Test WAKEUP energy consumption
    node.state = NodeState.WAKEUP
    consumed = node.consume_energy()
    assert abs(consumed - 2.0 * h) < 1e-18  # PW
    assert abs(node.energy - (initial_energy - 13.1 * h)) < 1e-18
    
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
    
    h = _SLOT_DURATION_H
    node = Node(1, 5e-5, 10, 5, power_rates)  # Very low energy (mWh)
    
    assert node.is_depleted() == False
    
    # Consume energy until depleted: 6 × PT(10) × h ≈ 1e-4 > 5e-5
    node.state = NodeState.ACTIVE
    for _ in range(6):
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


# ===========================================================================
# O6: Finite Retry Limits tests
# ===========================================================================

def _make_node(max_retries=None):
    power_rates = {'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1}
    return Node(1, 1000.0, 10, 5, power_rates, max_retries=max_retries)


def test_o6_default_max_retries_is_none():
    """max_retries defaults to None (infinite)."""
    node = _make_node()
    assert node.max_retries is None
    assert node.packets_dropped == 0


def test_o6_max_retries_stored():
    """max_retries parameter is stored on the node."""
    node = _make_node(max_retries=5)
    assert node.max_retries == 5


def test_o6_infinite_retries_never_drops():
    """With max_retries=None, handle_failed_attempt never drops the packet."""
    node = _make_node(max_retries=None)
    node.queue.append((0, {'retry_count': 0}))
    for _ in range(100):
        dropped = node.handle_failed_attempt()
        assert dropped is False
    assert node.packets_dropped == 0
    assert len(node.queue) == 1


def test_o6_packet_dropped_at_limit():
    """Packet is dropped once retry_count reaches max_retries."""
    node = _make_node(max_retries=3)
    node.queue.append((0, {'retry_count': 0}))

    # First 2 failures — no drop yet (retry_count → 1, → 2)
    assert node.handle_failed_attempt() is False  # retry_count=1
    assert node.handle_failed_attempt() is False  # retry_count=2
    assert node.packets_dropped == 0
    assert len(node.queue) == 1

    # Third failure: retry_count=3 >= max_retries=3 → drop
    assert node.handle_failed_attempt() is True
    assert node.packets_dropped == 1
    assert len(node.queue) == 0


def test_o6_drop_rate_in_statistics():
    """Statistics dict includes packets_dropped and drop_rate."""
    node = _make_node(max_retries=1)
    node.queue.append((0, {'retry_count': 0}))
    node.handle_failed_attempt()  # drops the packet

    stats = node.get_statistics(total_slots=100)
    assert 'packets_dropped' in stats
    assert 'drop_rate' in stats
    assert stats['packets_dropped'] == 1
    assert stats['drop_rate'] == 1.0  # 1 dropped / (0 delivered + 1 dropped)


def test_o6_drop_rate_mixed():
    """Drop rate is correct when both deliveries and drops occur."""
    node = _make_node(max_retries=1)

    # Deliver one packet
    node.queue.append((0, {'retry_count': 0}))
    node.handle_success(5)

    # Drop one packet
    node.queue.append((10, {'retry_count': 0}))
    node.handle_failed_attempt()  # retry_count=1 >= 1 → drop

    stats = node.get_statistics(total_slots=100)
    assert stats['packets_dropped'] == 1
    assert stats['packets_delivered'] == 1
    assert abs(stats['drop_rate'] - 0.5) < 1e-9


def test_o6_empty_queue_handle_failed_attempt():
    """handle_failed_attempt on an empty queue returns False safely."""
    node = _make_node(max_retries=2)
    assert node.handle_failed_attempt() is False
    assert node.packets_dropped == 0


def test_o6_retry_count_increments_correctly():
    """retry_count in the HOL packet increments on each failed attempt."""
    node = _make_node(max_retries=5)
    node.queue.append((0, {'retry_count': 0}))

    for expected_count in range(1, 5):
        node.handle_failed_attempt()
        _, pdata = node.queue[0]
        assert pdata['retry_count'] == expected_count


def test_o6_backoff_counter_initialised_to_zero():
    """Node initialises backoff_counter to 0 (used by CSMA O7)."""
    node = _make_node()
    assert node.backoff_counter == 0


# ===========================================================================
# O9: Age of Information tests
# ===========================================================================

def test_o9_initial_aoi_no_delivery():
    """AoI before any delivery equals current_slot + 1."""
    node = _make_node()
    assert node.get_current_aoi(0) == 1.0
    assert node.get_current_aoi(10) == 11.0


def test_o9_aoi_after_delivery():
    """AoI after a delivery equals current_slot - generation_slot + 1."""
    node = _make_node()
    node.queue.append((5, {'retry_count': 0}))  # generated at slot 5
    node.handle_success(15)  # delivered at slot 15

    # At slot 15: aoi = 15 - 5 + 1 = 11
    assert node.get_current_aoi(15) == 11.0
    # At slot 20: aoi = 20 - 5 + 1 = 16 (no new delivery)
    assert node.get_current_aoi(20) == 16.0


def test_o9_aoi_resets_on_fresh_delivery():
    """Second delivery resets AoI based on new generation slot."""
    node = _make_node()

    # First packet: generated at slot 0, delivered at slot 10
    node.queue.append((0, {'retry_count': 0}))
    node.handle_success(10)

    # Second packet: generated at slot 8, delivered at slot 12
    node.queue.append((8, {'retry_count': 0}))
    node.handle_success(12)

    # AoI at slot 12 = 12 - 8 + 1 = 5
    assert node.get_current_aoi(12) == 5.0


def test_o9_last_delivered_generation_slot_updated():
    """handle_success sets last_delivered_generation_slot."""
    node = _make_node()
    assert node.last_delivered_generation_slot is None

    node.queue.append((7, {'retry_count': 0}))
    node.handle_success(20)

    assert node.last_delivered_generation_slot == 7
