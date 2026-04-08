"""
Unit tests for src/receiver_models.py (O8)

Tests cover:
- ReceiverModel enum values
- resolve_transmissions with all three models
- Edge cases (empty input, single transmitter, all-capture failure)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.receiver_models import ReceiverModel, resolve_transmissions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeNode:
    """Minimal stand-in for a Node object."""
    def __init__(self, node_id):
        self.node_id = node_id

    def __repr__(self):
        return f"FakeNode({self.node_id})"


def make_nodes(n):
    return [FakeNode(i) for i in range(n)]


# ---------------------------------------------------------------------------
# ReceiverModel enum tests
# ---------------------------------------------------------------------------

def test_enum_values_exist():
    """ReceiverModel has COLLISION, CAPTURE, and SIC members."""
    assert ReceiverModel.COLLISION.value == "collision"
    assert ReceiverModel.CAPTURE.value == "capture"
    assert ReceiverModel.SIC.value == "sic"


def test_enum_from_string():
    """ReceiverModel can be constructed from its string value."""
    assert ReceiverModel("collision") == ReceiverModel.COLLISION
    assert ReceiverModel("capture") == ReceiverModel.CAPTURE
    assert ReceiverModel("sic") == ReceiverModel.SIC


# ---------------------------------------------------------------------------
# COLLISION model
# ---------------------------------------------------------------------------

def test_collision_empty_input():
    """Empty transmitter list returns empty list."""
    assert resolve_transmissions([], ReceiverModel.COLLISION) == []


def test_collision_single_transmitter_succeeds():
    """Single transmitter always succeeds under COLLISION model."""
    nodes = make_nodes(1)
    result = resolve_transmissions(nodes, ReceiverModel.COLLISION)
    assert result == nodes


def test_collision_two_transmitters_fail():
    """Two simultaneous transmitters → collision → empty result."""
    nodes = make_nodes(2)
    result = resolve_transmissions(nodes, ReceiverModel.COLLISION)
    assert result == []


def test_collision_many_transmitters_fail():
    """Any number > 1 of transmitters results in collision."""
    for n in [3, 5, 10]:
        nodes = make_nodes(n)
        result = resolve_transmissions(nodes, ReceiverModel.COLLISION)
        assert result == [], f"Expected collision with {n} nodes"


# ---------------------------------------------------------------------------
# CAPTURE model
# ---------------------------------------------------------------------------

def test_capture_single_transmitter_always_succeeds():
    """Single transmitter succeeds regardless of threshold."""
    np.random.seed(42)
    nodes = make_nodes(1)
    for _ in range(20):
        result = resolve_transmissions(
            nodes, ReceiverModel.CAPTURE, capture_threshold=1e6
        )
        assert len(result) == 1


def test_capture_empty_input():
    """Empty transmitter list returns empty result under CAPTURE."""
    assert resolve_transmissions([], ReceiverModel.CAPTURE) == []


def test_capture_low_threshold_often_succeeds():
    """With a very low capture threshold (≈0), most slots succeed."""
    np.random.seed(0)
    nodes = make_nodes(3)
    successes = sum(
        1 for _ in range(200)
        if resolve_transmissions(nodes, ReceiverModel.CAPTURE, capture_threshold=0.001)
    )
    # With threshold near 0, the strongest of 3 exponential RVs almost always
    # exceeds the interference → high success rate
    assert successes > 100, f"Expected high success rate, got {successes}/200"


def test_capture_high_threshold_often_fails():
    """With very high capture threshold, collisions rarely resolve."""
    np.random.seed(0)
    nodes = make_nodes(5)
    successes = sum(
        1 for _ in range(200)
        if resolve_transmissions(nodes, ReceiverModel.CAPTURE, capture_threshold=1000.0)
    )
    # With threshold=1000, success requires SINR > 1000 ≈ 30 dB, very rare
    assert successes < 50, f"Expected low success rate, got {successes}/200"


# ---------------------------------------------------------------------------
# SIC model
# ---------------------------------------------------------------------------

def test_sic_empty_input():
    """Empty transmitter list returns empty result under SIC."""
    assert resolve_transmissions([], ReceiverModel.SIC) == []


def test_sic_single_transmitter_succeeds():
    """Single transmitter always succeeds (SINR = ∞ > any threshold)."""
    np.random.seed(42)
    nodes = make_nodes(1)
    for _ in range(20):
        result = resolve_transmissions(nodes, ReceiverModel.SIC)
        assert len(result) == 1


def test_sic_result_subset_of_transmitters():
    """SIC result is always a subset of the input transmitters."""
    np.random.seed(7)
    nodes = make_nodes(4)
    for _ in range(50):
        result = resolve_transmissions(nodes, ReceiverModel.SIC)
        for n in result:
            assert n in nodes


def test_sic_low_threshold_can_decode_multiple():
    """With SINR threshold near 0, SIC can decode all transmitters."""
    np.random.seed(5)
    nodes = make_nodes(3)
    any_multi = False
    for _ in range(100):
        result = resolve_transmissions(
            nodes, ReceiverModel.SIC, sic_sinr_threshold=0.001
        )
        if len(result) > 1:
            any_multi = True
            break
    assert any_multi, "Expected at least one multi-decode event with low threshold"