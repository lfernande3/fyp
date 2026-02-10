"""
Traffic Models for M2M Simulator

This module provides different traffic arrival models beyond the default Poisson
(Bernoulli) process. Includes bursty traffic with batch arrivals.

Date: February 10, 2026
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class TrafficModel(Enum):
    """Traffic model types."""
    POISSON = "poisson"          # Standard Poisson (Bernoulli) arrivals
    BURSTY = "bursty"            # Bursty traffic with batch arrivals
    PERIODIC = "periodic"        # Periodic arrivals
    ON_OFF = "on_off"           # On-off traffic pattern


class TrafficGenerator:
    """
    Base class for traffic generation.
    
    Provides different arrival patterns for simulation.
    """
    
    @staticmethod
    def poisson_arrival(arrival_rate: float) -> bool:
        """
        Standard Poisson (Bernoulli) arrival.
        
        Each slot, a packet arrives with probability λ.
        
        Args:
            arrival_rate: Arrival probability λ per slot
            
        Returns:
            True if packet arrives, False otherwise
        """
        return random.random() < arrival_rate
    
    @staticmethod
    def bursty_arrival(
        base_rate: float,
        burst_probability: float = 0.1,
        burst_size_mean: int = 3,
        burst_size_std: int = 1
    ) -> int:
        """
        Bursty traffic with batch arrivals.
        
        With probability burst_probability, a burst of packets arrives.
        Burst size follows normal distribution (clipped to >= 1).
        Otherwise, standard Poisson arrival.
        
        Args:
            base_rate: Base arrival probability (for non-burst)
            burst_probability: Probability of burst occurring
            burst_size_mean: Mean burst size (number of packets)
            burst_size_std: Standard deviation of burst size
            
        Returns:
            Number of packets arriving (0 or more)
        """
        # Check if burst occurs
        if random.random() < burst_probability:
            # Burst: generate batch of packets
            burst_size = int(np.random.normal(burst_size_mean, burst_size_std))
            burst_size = max(1, burst_size)  # At least 1 packet in burst
            return burst_size
        else:
            # No burst: standard Poisson arrival
            return 1 if random.random() < base_rate else 0
    
    @staticmethod
    def periodic_arrival(slot: int, period: int, jitter: float = 0.0) -> bool:
        """
        Periodic arrivals with optional jitter.
        
        Packets arrive every `period` slots, with some randomness if jitter > 0.
        
        Args:
            slot: Current slot number
            period: Period in slots
            jitter: Jitter as fraction of period (0.0-1.0)
            
        Returns:
            True if packet arrives, False otherwise
        """
        if period <= 0:
            return False
        
        # Check if this is a periodic slot
        if slot % period == 0:
            # Add jitter
            if jitter > 0:
                return random.random() < (1.0 - jitter)
            return True
        
        # Random arrival due to jitter
        if jitter > 0:
            jitter_prob = jitter / period
            return random.random() < jitter_prob
        
        return False
    
    @staticmethod
    def on_off_arrival(
        slot: int,
        on_duration: int,
        off_duration: int,
        on_rate: float
    ) -> bool:
        """
        On-off traffic pattern.
        
        Alternates between ON (high arrival rate) and OFF (no arrivals) periods.
        
        Args:
            slot: Current slot number
            on_duration: Duration of ON period in slots
            off_duration: Duration of OFF period in slots
            on_rate: Arrival rate during ON period
            
        Returns:
            True if packet arrives, False otherwise
        """
        cycle_length = on_duration + off_duration
        position_in_cycle = slot % cycle_length
        
        # Check if in ON period
        if position_in_cycle < on_duration:
            return random.random() < on_rate
        else:
            return False


class BurstyTrafficConfig:
    """Configuration for bursty traffic model."""
    
    def __init__(
        self,
        base_rate: float = 0.01,
        burst_probability: float = 0.1,
        burst_size_mean: int = 3,
        burst_size_std: int = 1
    ):
        """
        Initialize bursty traffic configuration.
        
        Args:
            base_rate: Base arrival probability λ (non-burst)
            burst_probability: Probability of burst per slot
            burst_size_mean: Mean number of packets per burst
            burst_size_std: Standard deviation of burst size
        """
        self.base_rate = base_rate
        self.burst_probability = burst_probability
        self.burst_size_mean = burst_size_mean
        self.burst_size_std = burst_size_std
    
    def get_effective_arrival_rate(self) -> float:
        """
        Calculate effective arrival rate.
        
        Takes into account both normal arrivals and bursts.
        
        Returns:
            Effective average arrival rate
        """
        # Expected arrivals from bursts per slot
        burst_arrivals = self.burst_probability * self.burst_size_mean
        
        # Expected arrivals from normal (non-burst) slots
        normal_prob = 1.0 - self.burst_probability
        normal_arrivals = normal_prob * self.base_rate
        
        return burst_arrivals + normal_arrivals
    
    def __repr__(self) -> str:
        return (f"BurstyTrafficConfig(base_rate={self.base_rate}, "
                f"burst_prob={self.burst_probability}, "
                f"burst_size={self.burst_size_mean}±{self.burst_size_std})")


def generate_bursty_traffic_trace(
    n_slots: int,
    config: BurstyTrafficConfig,
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate a complete traffic trace for bursty traffic.
    
    Useful for analysis and visualization.
    
    Args:
        n_slots: Number of slots to generate
        config: Bursty traffic configuration
        seed: Random seed for reproducibility
        
    Returns:
        List of packet counts per slot
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    trace = []
    for slot in range(n_slots):
        n_arrivals = TrafficGenerator.bursty_arrival(
            config.base_rate,
            config.burst_probability,
            config.burst_size_mean,
            config.burst_size_std
        )
        trace.append(n_arrivals)
    
    return trace


def analyze_traffic_trace(trace: List[int]) -> dict:
    """
    Analyze a traffic trace.
    
    Computes statistics about the traffic pattern.
    
    Args:
        trace: List of packet arrivals per slot
        
    Returns:
        Dictionary with statistics
    """
    trace_array = np.array(trace)
    
    # Handle empty trace
    if len(trace_array) == 0:
        return {
            'total_packets': 0,
            'mean_rate': 0.0,
            'std_rate': 0.0,
            'max_burst': 0,
            'burst_slots': 0,
            'burst_fraction': 0.0,
            'mean_inter_arrival': 0.0,
            'std_inter_arrival': 0.0,
            'burstiness_coefficient': 0.0
        }
    
    # Basic statistics
    total_packets = np.sum(trace_array)
    mean_rate = np.mean(trace_array)
    std_rate = np.std(trace_array)
    max_burst = np.max(trace_array)
    
    # Count bursts (slots with > 1 packet)
    burst_slots = np.sum(trace_array > 1)
    burst_fraction = burst_slots / len(trace) if len(trace) > 0 else 0
    
    # Inter-arrival time statistics
    arrival_slots = np.where(trace_array > 0)[0]
    if len(arrival_slots) > 1:
        inter_arrival = np.diff(arrival_slots)
        mean_inter_arrival = np.mean(inter_arrival)
        std_inter_arrival = np.std(inter_arrival)
    else:
        mean_inter_arrival = 0
        std_inter_arrival = 0
    
    # Burstiness coefficient (std / mean)
    burstiness = std_rate / mean_rate if mean_rate > 0 else 0
    
    return {
        'total_packets': int(total_packets),
        'mean_rate': float(mean_rate),
        'std_rate': float(std_rate),
        'max_burst': int(max_burst),
        'burst_slots': int(burst_slots),
        'burst_fraction': float(burst_fraction),
        'mean_inter_arrival': float(mean_inter_arrival),
        'std_inter_arrival': float(std_inter_arrival),
        'burstiness_coefficient': float(burstiness)
    }


def compare_poisson_vs_bursty(
    n_slots: int = 10000,
    mean_rate: float = 0.01,
    burst_config: Optional[BurstyTrafficConfig] = None,
    seed: int = 42
) -> Tuple[List[int], List[int], dict, dict]:
    """
    Compare Poisson vs bursty traffic patterns.
    
    Args:
        n_slots: Number of slots
        mean_rate: Mean arrival rate
        burst_config: Bursty traffic config (creates default if None)
        seed: Random seed
        
    Returns:
        Tuple of (poisson_trace, bursty_trace, poisson_stats, bursty_stats)
    """
    # Generate Poisson trace
    random.seed(seed)
    np.random.seed(seed)
    poisson_trace = [1 if TrafficGenerator.poisson_arrival(mean_rate) else 0 
                     for _ in range(n_slots)]
    
    # Generate bursty trace
    if burst_config is None:
        burst_config = BurstyTrafficConfig(
            base_rate=mean_rate * 0.5,  # Lower base to maintain similar mean
            burst_probability=0.1,
            burst_size_mean=3
        )
    
    random.seed(seed + 1)
    np.random.seed(seed + 1)
    bursty_trace = generate_bursty_traffic_trace(n_slots, burst_config)
    
    # Analyze both
    poisson_stats = analyze_traffic_trace(poisson_trace)
    bursty_stats = analyze_traffic_trace(bursty_trace)
    
    return poisson_trace, bursty_trace, poisson_stats, bursty_stats
