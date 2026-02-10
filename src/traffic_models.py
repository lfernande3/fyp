"""
Traffic Models for M2M Sleep-Based Random Access Simulator

This module provides different traffic arrival models including:
- Poisson (Bernoulli) arrivals (default)
- Bursty arrivals (batch/correlated traffic)
- Periodic arrivals

Date: February 10, 2026
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass


class TrafficModel(Enum):
    """Traffic arrival model types."""
    POISSON = "poisson"          # Independent Bernoulli arrivals
    BURSTY = "bursty"           # Batch arrivals with correlation
    PERIODIC = "periodic"        # Periodic deterministic arrivals
    MIXED = "mixed"             # Mix of Poisson and bursty


@dataclass
class TrafficConfig:
    """
    Configuration for traffic model.
    
    Attributes:
        model: Type of traffic model
        arrival_rate: Base arrival rate (λ)
        
        # Bursty traffic parameters
        burst_prob: Probability of burst event
        burst_size_mean: Mean number of packets per burst
        burst_size_std: Std dev of burst size
        
        # Periodic traffic parameters
        period: Period between arrivals (in slots)
        jitter: Random jitter around period (±jitter slots)
        
        # Mixed traffic parameters
        bursty_fraction: Fraction of nodes with bursty traffic (0-1)
    """
    model: TrafficModel = TrafficModel.POISSON
    arrival_rate: float = 0.01
    
    # Bursty parameters
    burst_prob: float = 0.01        # Lower probability but larger batches
    burst_size_mean: float = 3.0    # Average 3 packets per burst
    burst_size_std: float = 1.0     # Variation in burst size
    
    # Periodic parameters
    period: int = 100               # Arrive every 100 slots
    jitter: int = 5                 # ±5 slots jitter
    
    # Mixed parameters
    bursty_fraction: float = 0.3    # 30% nodes have bursty traffic


class TrafficGenerator:
    """
    Traffic generator for different arrival models.
    
    Generates packet arrivals according to specified traffic model.
    """
    
    def __init__(self, config: TrafficConfig, seed: Optional[int] = None):
        """
        Initialize traffic generator.
        
        Args:
            config: Traffic configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # For periodic traffic, track last arrival time per node
        self.last_arrival_slot: Dict[int, int] = {}
        
    def generate_arrivals_poisson(
        self,
        n_nodes: int,
        current_slot: int
    ) -> List[int]:
        """
        Generate Poisson (Bernoulli) arrivals.
        
        Each node independently generates packets with probability λ.
        
        Args:
            n_nodes: Number of nodes
            current_slot: Current simulation slot
            
        Returns:
            List of node IDs that have arrivals
        """
        arrivals = []
        for node_id in range(n_nodes):
            if self.rng.random() < self.config.arrival_rate:
                arrivals.append(node_id)
        return arrivals
    
    def generate_arrivals_bursty(
        self,
        n_nodes: int,
        current_slot: int
    ) -> List[int]:
        """
        Generate bursty arrivals with batch traffic.
        
        Nodes experience burst events where multiple packets arrive together.
        This models correlated traffic (e.g., sensor triggered by event).
        
        Model:
        - With probability burst_prob, a burst event occurs
        - Burst generates batch_size packets (Gaussian distributed)
        - Between bursts, low baseline Poisson traffic
        
        Args:
            n_nodes: Number of nodes
            current_slot: Current simulation slot
            
        Returns:
            List of node IDs that have arrivals (may include duplicates for batch)
        """
        arrivals = []
        
        for node_id in range(n_nodes):
            # Check for burst event
            if self.rng.random() < self.config.burst_prob:
                # Generate batch size (Gaussian, minimum 1)
                batch_size = max(1, int(self.rng.normal(
                    self.config.burst_size_mean,
                    self.config.burst_size_std
                )))
                # Add multiple packets for this node
                arrivals.extend([node_id] * batch_size)
            else:
                # Baseline Poisson traffic (at lower rate)
                baseline_rate = self.config.arrival_rate * 0.1  # 10% of base rate
                if self.rng.random() < baseline_rate:
                    arrivals.append(node_id)
        
        return arrivals
    
    def generate_arrivals_periodic(
        self,
        n_nodes: int,
        current_slot: int
    ) -> List[int]:
        """
        Generate periodic arrivals with jitter.
        
        Each node generates packets periodically every 'period' slots,
        with random jitter to avoid synchronization.
        
        Args:
            n_nodes: Number of nodes
            current_slot: Current simulation slot
            
        Returns:
            List of node IDs that have arrivals
        """
        arrivals = []
        
        for node_id in range(n_nodes):
            # Initialize last arrival time if first time
            if node_id not in self.last_arrival_slot:
                # Stagger initial arrivals to avoid all starting together
                self.last_arrival_slot[node_id] = -self.config.period + \
                    self.rng.randint(0, self.config.period)
            
            # Check if it's time for arrival
            slots_since_last = current_slot - self.last_arrival_slot[node_id]
            
            # Period with jitter
            next_arrival_time = self.config.period + \
                self.rng.randint(-self.config.jitter, self.config.jitter + 1)
            
            if slots_since_last >= next_arrival_time:
                arrivals.append(node_id)
                self.last_arrival_slot[node_id] = current_slot
        
        return arrivals
    
    def generate_arrivals_mixed(
        self,
        n_nodes: int,
        current_slot: int,
        node_types: Optional[List[str]] = None
    ) -> List[int]:
        """
        Generate mixed traffic (some nodes Poisson, some bursty).
        
        Simulates heterogeneous network with different traffic patterns.
        
        Args:
            n_nodes: Number of nodes
            current_slot: Current simulation slot
            node_types: Optional list specifying type for each node
                       ('poisson' or 'bursty'). If None, assigned randomly.
            
        Returns:
            List of node IDs that have arrivals
        """
        arrivals = []
        
        # Determine node types if not provided
        if node_types is None:
            node_types = []
            for _ in range(n_nodes):
                if self.rng.random() < self.config.bursty_fraction:
                    node_types.append('bursty')
                else:
                    node_types.append('poisson')
        
        # Generate arrivals based on type
        for node_id in range(n_nodes):
            if node_types[node_id] == 'bursty':
                # Bursty traffic
                if self.rng.random() < self.config.burst_prob:
                    batch_size = max(1, int(self.rng.normal(
                        self.config.burst_size_mean,
                        self.config.burst_size_std
                    )))
                    arrivals.extend([node_id] * batch_size)
            else:
                # Poisson traffic
                if self.rng.random() < self.config.arrival_rate:
                    arrivals.append(node_id)
        
        return arrivals
    
    def generate_arrivals(
        self,
        n_nodes: int,
        current_slot: int,
        node_types: Optional[List[str]] = None
    ) -> List[int]:
        """
        Generate arrivals according to configured model.
        
        Args:
            n_nodes: Number of nodes
            current_slot: Current simulation slot
            node_types: Optional node type specification (for mixed traffic)
            
        Returns:
            List of node IDs that have arrivals
        """
        if self.config.model == TrafficModel.POISSON:
            return self.generate_arrivals_poisson(n_nodes, current_slot)
        elif self.config.model == TrafficModel.BURSTY:
            return self.generate_arrivals_bursty(n_nodes, current_slot)
        elif self.config.model == TrafficModel.PERIODIC:
            return self.generate_arrivals_periodic(n_nodes, current_slot)
        elif self.config.model == TrafficModel.MIXED:
            return self.generate_arrivals_mixed(n_nodes, current_slot, node_types)
        else:
            raise ValueError(f"Unknown traffic model: {self.config.model}")
    
    def get_effective_arrival_rate(self, n_nodes: int, n_slots: int) -> float:
        """
        Estimate effective arrival rate for the configured model.
        
        This is useful for analytical comparisons and validation.
        
        Args:
            n_nodes: Number of nodes
            n_slots: Number of slots to simulate
            
        Returns:
            Estimated arrival rate per node per slot
        """
        if self.config.model == TrafficModel.POISSON:
            return self.config.arrival_rate
        elif self.config.model == TrafficModel.BURSTY:
            # Expected arrivals = burst_prob * burst_size + baseline
            baseline = self.config.arrival_rate * 0.1
            burst_contribution = self.config.burst_prob * self.config.burst_size_mean
            return burst_contribution + baseline
        elif self.config.model == TrafficModel.PERIODIC:
            # Arrivals once per period
            return 1.0 / self.config.period
        elif self.config.model == TrafficModel.MIXED:
            # Weighted average
            poisson_rate = self.config.arrival_rate
            bursty_rate = (self.config.burst_prob * self.config.burst_size_mean +
                          self.config.arrival_rate * 0.1)
            return (1 - self.config.bursty_fraction) * poisson_rate + \
                   self.config.bursty_fraction * bursty_rate
        else:
            return self.config.arrival_rate


def create_poisson_traffic(arrival_rate: float, seed: Optional[int] = None) -> TrafficGenerator:
    """
    Create Poisson (Bernoulli) traffic generator.
    
    Args:
        arrival_rate: Arrival probability per slot (λ)
        seed: Random seed
        
    Returns:
        TrafficGenerator configured for Poisson arrivals
    """
    config = TrafficConfig(
        model=TrafficModel.POISSON,
        arrival_rate=arrival_rate
    )
    return TrafficGenerator(config, seed)


def create_bursty_traffic(
    burst_prob: float = 0.01,
    burst_size_mean: float = 3.0,
    burst_size_std: float = 1.0,
    baseline_rate: float = 0.001,
    seed: Optional[int] = None
) -> TrafficGenerator:
    """
    Create bursty traffic generator.
    
    Args:
        burst_prob: Probability of burst event per slot
        burst_size_mean: Mean packets per burst
        burst_size_std: Std dev of burst size
        baseline_rate: Baseline Poisson rate between bursts
        seed: Random seed
        
    Returns:
        TrafficGenerator configured for bursty arrivals
    """
    config = TrafficConfig(
        model=TrafficModel.BURSTY,
        arrival_rate=baseline_rate * 10,  # Will be scaled to 10%
        burst_prob=burst_prob,
        burst_size_mean=burst_size_mean,
        burst_size_std=burst_size_std
    )
    return TrafficGenerator(config, seed)


def create_periodic_traffic(
    period: int = 100,
    jitter: int = 5,
    seed: Optional[int] = None
) -> TrafficGenerator:
    """
    Create periodic traffic generator.
    
    Args:
        period: Period between arrivals (slots)
        jitter: Random jitter (±jitter slots)
        seed: Random seed
        
    Returns:
        TrafficGenerator configured for periodic arrivals
    """
    config = TrafficConfig(
        model=TrafficModel.PERIODIC,
        period=period,
        jitter=jitter
    )
    return TrafficGenerator(config, seed)
