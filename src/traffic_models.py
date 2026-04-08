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
from dataclasses import dataclass


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


# ---------------------------------------------------------------------------
# O10: Markov-Modulated Bernoulli Process (MMBP)
# ---------------------------------------------------------------------------

@dataclass
class MMBPConfig:
    """
    Parameters for a 2-state Markov-Modulated Bernoulli Process (MMBP).

    States
    ------
    HIGH (H): arrival probability per slot = lambda_H
    LOW  (L): arrival probability per slot = lambda_L

    Transition probabilities
    ------------------------
    p_HH = P(stay in H | currently in H)
    p_LL = P(stay in L | currently in L)

    Therefore:
        P(H → L) = 1 − p_HH
        P(L → H) = 1 − p_LL

    When lambda_H == lambda_L, the process degenerates to a Bernoulli(λ).
    """
    lambda_H: float  # Arrival prob in HIGH state
    lambda_L: float  # Arrival prob in LOW state
    p_HH: float      # P(stay HIGH | HIGH)
    p_LL: float      # P(stay LOW  | LOW)

    def stationary_probs(self) -> Tuple[float, float]:
        """Return stationary probabilities (pi_H, pi_L)."""
        # pi_H / pi_L = (1 - p_LL) / (1 - p_HH)
        denom = (1.0 - self.p_LL) + (1.0 - self.p_HH)
        if denom <= 0:
            return (0.5, 0.5)
        pi_H = (1.0 - self.p_LL) / denom
        pi_L = 1.0 - pi_H
        return (pi_H, pi_L)

    def mean_arrival_rate(self) -> float:
        """Mean arrival rate: λ̄ = π_H · λ_H + π_L · λ_L."""
        pi_H, pi_L = self.stationary_probs()
        return pi_H * self.lambda_H + pi_L * self.lambda_L

    def burstiness_index(self) -> float:
        """
        Asymptotic Fano factor (index of dispersion) for the MMBP.

        F = 1 + 2(λ_H − λ_L)² · π_H · π_L / (λ̄ · (1 − ρ))

        where ρ = p_HH + p_LL − 1 is the eigenvalue controlling mixing speed.
        F = 1 for Bernoulli (no burstiness).
        """
        pi_H, pi_L = self.stationary_probs()
        lambda_bar = self.mean_arrival_rate()
        rho = self.p_HH + self.p_LL - 1.0     # in (-1, 1)
        if abs(1.0 - rho) < 1e-10 or lambda_bar < 1e-10:
            return 1.0
        fano = (1.0 + 2.0 * (self.lambda_H - self.lambda_L) ** 2
                * pi_H * pi_L / (lambda_bar * (1.0 - rho)))
        return max(1.0, float(fano))

    def __post_init__(self):
        if not (0.0 < self.p_HH < 1.0):
            raise ValueError(f"p_HH must be in (0, 1), got {self.p_HH}")
        if not (0.0 < self.p_LL < 1.0):
            raise ValueError(f"p_LL must be in (0, 1), got {self.p_LL}")
        if not (0.0 <= self.lambda_H <= 1.0):
            raise ValueError(f"lambda_H must be in [0, 1], got {self.lambda_H}")
        if not (0.0 <= self.lambda_L <= 1.0):
            raise ValueError(f"lambda_L must be in [0, 1], got {self.lambda_L}")


class MMBPGenerator:
    """
    Generates per-slot packet arrivals according to a 2-state MMBP.

    The Markov chain state is evolved each time :meth:`next_slot` is called.
    Initial state is drawn from the stationary distribution.
    """

    def __init__(self, config: MMBPConfig, seed: Optional[int] = None):
        """
        Args:
            config: MMBP parameters.
            seed:   Random seed for reproducibility (None = use global state).
        """
        self.config = config
        self._rng = random.Random(seed)

        # Initialise state from stationary distribution
        pi_H, _pi_L = config.stationary_probs()
        self._state: int = 1 if self._rng.random() < pi_H else 0  # 1=HIGH, 0=LOW

    @property
    def is_high_state(self) -> bool:
        return self._state == 1

    def next_slot(self) -> int:
        """
        Advance one slot: evolve chain, then generate 0 or 1 arrival.

        Returns:
            1 if a packet arrived, 0 otherwise.
        """
        cfg = self.config
        # Evolve Markov chain
        if self._state == 1:   # HIGH
            self._state = 1 if self._rng.random() < cfg.p_HH else 0
            lam = cfg.lambda_H
        else:                  # LOW
            self._state = 0 if self._rng.random() < cfg.p_LL else 1
            lam = cfg.lambda_L

        return 1 if self._rng.random() < lam else 0

    def generate_trace(self, n_slots: int) -> List[int]:
        """
        Generate a per-slot arrival trace of length ``n_slots``.

        Returns:
            List of 0/1 integers (1 = arrival).
        """
        return [self.next_slot() for _ in range(n_slots)]


def generate_mmbp_traffic_trace(
    n_slots: int,
    config: MMBPConfig,
    seed: Optional[int] = None
) -> List[int]:
    """
    Convenience wrapper: generate an MMBP arrival trace.

    Args:
        n_slots: Number of slots.
        config:  MMBP parameters.
        seed:    Random seed.

    Returns:
        List of per-slot arrival counts (0 or 1).
    """
    gen = MMBPGenerator(config, seed=seed)
    return gen.generate_trace(n_slots)


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
