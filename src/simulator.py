"""
Discrete-Event Simulator for Sleep-Based Random Access

This module implements the Simulator class that manages multiple MTD nodes,
runs slotted time loops, detects collisions, and supports batch parameter sweeps.

Date: February 10, 2026
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import time

from .node import Node, NodeState


@dataclass
class SimulationConfig:
    """Configuration parameters for a simulation run."""
    n_nodes: int                    # Number of MTD nodes
    arrival_rate: float            # λ - Bernoulli arrival probability per slot
    transmission_prob: float       # q - Transmission probability per slot
    idle_timer: int                # ts - Slots before sleep
    wakeup_time: int              # tw - Slots to wake up
    initial_energy: float         # E - Initial energy per node
    power_rates: Dict[str, float] # Power consumption rates
    max_slots: int                # Maximum simulation slots
    seed: Optional[int] = None    # Random seed for reproducibility
    stop_on_first_depletion: bool = False  # Stop when first node depletes


@dataclass
class SimulationResults:
    """Results from a single simulation run."""
    config: SimulationConfig
    total_slots: int
    
    # Aggregate statistics
    mean_lifetime_slots: float
    mean_lifetime_years: float
    mean_delay: float
    tail_delay_95: float
    tail_delay_99: float
    mean_queue_length: float
    throughput: float              # Successful transmissions per slot
    
    # State fractions (averaged across all nodes)
    state_fractions: Dict[str, float]
    
    # Energy statistics
    mean_energy_consumed: float
    energy_fractions_by_state: Dict[str, float]
    
    # Network statistics
    total_arrivals: int
    total_deliveries: int
    total_collisions: int
    total_transmissions: int
    
    # Empirical performance metrics
    empirical_success_prob: float  # p = successful_tx / total_tx
    empirical_service_rate: float  # μ (estimated)
    
    # Per-node results (optional, for detailed analysis)
    node_statistics: Optional[List[Dict]] = None
    
    # Time series data (optional)
    queue_length_history: Optional[List[float]] = None
    energy_history: Optional[List[float]] = None
    state_history: Optional[List[Dict[str, int]]] = None


class Simulator:
    """
    Discrete-event simulator for sleep-based random access.
    
    Manages multiple MTD nodes running slotted Aloha with on-demand sleep.
    Handles collision detection, energy tracking, and comprehensive metrics.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulator.
        
        Args:
            config: SimulationConfig with all parameters
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Create nodes
        self.nodes: List[Node] = []
        for i in range(config.n_nodes):
            node = Node(
                node_id=i,
                initial_energy=config.initial_energy,
                idle_timer=config.idle_timer,
                wakeup_time=config.wakeup_time,
                power_rates=config.power_rates
            )
            self.nodes.append(node)
        
        # Statistics tracking
        self.current_slot = 0
        self.total_collisions = 0
        self.total_transmissions = 0
        self.total_successes = 0
        self.slots_with_transmissions = 0  # Track slots where at least one node transmitted
        
        # Time series tracking (optional, can be large)
        self.track_history = False
        self.queue_history: List[float] = []
        self.energy_history: List[float] = []
        self.state_history: List[Dict[str, int]] = []
    
    def run_simulation(
        self, 
        track_history: bool = False,
        verbose: bool = False,
        progress_interval: int = 10000
    ) -> SimulationResults:
        """
        Run the simulation until max_slots or all nodes depleted.
        
        Main simulation loop:
        1. For each slot:
           a. Packet arrivals (all nodes)
           b. Transmission attempts (active nodes)
           c. Collision detection
           d. Energy consumption
           e. State updates
           f. Statistics collection
        2. Continue until stopping condition
        3. Return results
        
        Args:
            track_history: Whether to track time series data
            verbose: Print progress messages
            progress_interval: Slots between progress messages
            
        Returns:
            SimulationResults with all metrics
        """
        self.track_history = track_history
        start_time = time.time()
        
        if verbose:
            print(f"Starting simulation with {self.config.n_nodes} nodes...")
            print(f"Parameters: λ={self.config.arrival_rate}, q={self.config.transmission_prob}, "
                  f"ts={self.config.idle_timer}, tw={self.config.wakeup_time}")
        
        # Main simulation loop
        for slot in range(self.config.max_slots):
            self.current_slot = slot
            
            # Check stopping condition
            if self.config.stop_on_first_depletion:
                if any(node.is_depleted() for node in self.nodes):
                    if verbose:
                        print(f"First node depleted at slot {slot}")
                    break
            else:
                # Check if all nodes depleted
                if all(node.is_depleted() for node in self.nodes):
                    if verbose:
                        print(f"All nodes depleted at slot {slot}")
                    break
            
            # Progress reporting
            if verbose and slot > 0 and slot % progress_interval == 0:
                elapsed = time.time() - start_time
                print(f"Slot {slot}/{self.config.max_slots} ({slot/self.config.max_slots*100:.1f}%) "
                      f"- {elapsed:.1f}s elapsed")
            
            # 1. Packet arrivals for all non-depleted nodes
            for node in self.nodes:
                if not node.is_depleted():
                    node.arrive_packet(slot, self.config.arrival_rate)
            
            # 2. Transmission attempts (collect which nodes are transmitting)
            transmitting_nodes = []
            for node in self.nodes:
                if not node.is_depleted() and node.state == NodeState.ACTIVE:
                    if node.get_queue_length() > 0:
                        if node.attempt_transmit(self.config.transmission_prob):
                            transmitting_nodes.append(node)
            
            # 3. Collision detection
            n_transmitting = len(transmitting_nodes)
            success = (n_transmitting == 1)
            collision = (n_transmitting > 1)
            
            # Update statistics
            self.total_transmissions += n_transmitting
            if n_transmitting > 0:
                self.slots_with_transmissions += 1
            if collision:
                self.total_collisions += 1
            if success:
                self.total_successes += 1
                # Handle successful transmission
                transmitting_nodes[0].handle_success(slot)
            
            # 4. Energy consumption for all nodes
            for node in self.nodes:
                if not node.is_depleted():
                    was_transmitting = node in transmitting_nodes
                    node.consume_energy(
                        was_transmitting=was_transmitting,
                        was_collision=collision
                    )
            
            # 5. State updates for all nodes
            for node in self.nodes:
                if not node.is_depleted():
                    node.update_state(slot)
            
            # 6. Track history if requested
            if self.track_history:
                self._record_history(slot)
        
        # Simulation complete
        elapsed = time.time() - start_time
        if verbose:
            print(f"\nSimulation complete! Total time: {elapsed:.2f}s")
            print(f"Simulated {self.current_slot + 1} slots")
        
        # Compute and return results
        return self._compute_results()
    
    def _record_history(self, slot: int) -> None:
        """Record time series data for current slot."""
        # Average queue length across all nodes
        total_queue = sum(node.get_queue_length() for node in self.nodes)
        avg_queue = total_queue / self.config.n_nodes
        self.queue_history.append(avg_queue)
        
        # Average remaining energy
        total_energy = sum(node.energy for node in self.nodes)
        avg_energy = total_energy / self.config.n_nodes
        self.energy_history.append(avg_energy)
        
        # State distribution
        state_counts = defaultdict(int)
        for node in self.nodes:
            state_counts[node.state.value] += 1
        self.state_history.append(dict(state_counts))
    
    def _compute_results(self) -> SimulationResults:
        """
        Compute comprehensive results from simulation run.
        
        Returns:
            SimulationResults with all metrics
        """
        total_slots = self.current_slot + 1
        
        # Collect per-node statistics
        node_stats = [node.get_statistics(total_slots) for node in self.nodes]
        
        # Compute aggregate statistics
        
        # Lifetime estimation based on energy consumption rate
        # Calculate mean energy consumed per slot across all nodes
        slot_duration_ms = 6.0
        slot_duration_s = slot_duration_ms / 1000.0
        seconds_per_year = 365.25 * 24 * 3600
        
        # Calculate projected lifetime for each node based on energy consumption rate
        lifetimes_years = []
        for node in self.nodes:
            if node.is_depleted():
                # Node depleted during simulation
                # Find approximate depletion slot based on energy consumed
                energy_consumed = node.initial_energy - node.energy
                if energy_consumed > 0:
                    # Estimate when it depleted
                    energy_per_slot = energy_consumed / total_slots
                    depletion_slot = node.initial_energy / energy_per_slot
                    lifetime_seconds = depletion_slot * slot_duration_s
                    lifetimes_years.append(lifetime_seconds / seconds_per_year)
                else:
                    lifetimes_years.append(0.0)
            else:
                # Node still has energy - project remaining lifetime
                energy_consumed = node.initial_energy - node.energy
                if energy_consumed > 0 and total_slots > 0:
                    energy_per_slot = energy_consumed / total_slots
                    total_lifetime_slots = node.initial_energy / energy_per_slot
                    lifetime_seconds = total_lifetime_slots * slot_duration_s
                    lifetimes_years.append(lifetime_seconds / seconds_per_year)
                else:
                    # No energy consumed - infinite lifetime or just started
                    lifetimes_years.append(float('inf'))
        
        # Filter out infinite lifetimes for mean calculation
        finite_lifetimes = [lt for lt in lifetimes_years if lt != float('inf')]
        if len(finite_lifetimes) > 0:
            mean_lifetime_years = np.mean(finite_lifetimes)
            mean_lifetime_slots = mean_lifetime_years * seconds_per_year / slot_duration_s
        else:
            mean_lifetime_years = float('inf')
            mean_lifetime_slots = float('inf')
        
        # Delay statistics
        all_delays = []
        for node in self.nodes:
            all_delays.extend(node.delays)
        
        if len(all_delays) > 0:
            mean_delay = np.mean(all_delays)
            tail_delay_95 = np.percentile(all_delays, 95)
            tail_delay_99 = np.percentile(all_delays, 99)
        else:
            mean_delay = 0.0
            tail_delay_95 = 0.0
            tail_delay_99 = 0.0
        
        # Queue length (average across all nodes at current time)
        mean_queue_length = sum(node.get_queue_length() for node in self.nodes) / self.config.n_nodes
        
        # Throughput (successful transmissions per slot)
        throughput = self.total_successes / total_slots if total_slots > 0 else 0.0
        
        # State fractions (averaged across all nodes)
        state_fractions = defaultdict(float)
        for node in self.nodes:
            node_fractions = node.get_state_fractions(total_slots)
            for state, fraction in node_fractions.items():
                state_fractions[state] += fraction
        # Average across nodes
        for state in state_fractions:
            state_fractions[state] /= self.config.n_nodes
        
        # Energy statistics
        mean_energy_consumed = np.mean([
            stats['energy_consumed'] for stats in node_stats
        ])
        
        # Energy fractions by state (averaged across nodes)
        energy_fractions = defaultdict(float)
        for node in self.nodes:
            node_fractions = node.get_energy_fraction_by_state()
            for state, fraction in node_fractions.items():
                energy_fractions[state] += fraction
        for state in energy_fractions:
            energy_fractions[state] /= self.config.n_nodes
        
        # Network statistics
        total_arrivals = sum(node.packets_arrived for node in self.nodes)
        total_deliveries = sum(node.packets_delivered for node in self.nodes)
        
        # Empirical performance metrics
        # Success probability: fraction of all slots where exactly one node transmitted
        # This matches the analytical formula: P(exactly 1 transmits in a slot)
        empirical_success_prob = self.total_successes / total_slots if total_slots > 0 else 0.0
        
        # Empirical service rate (throughput / avg_queue_length if queue > 0)
        # More accurately: successful_transmissions / (active_slots * n_nodes)
        total_active_slots = sum(
            node.time_in_state[NodeState.ACTIVE] for node in self.nodes
        )
        empirical_service_rate = (
            self.total_successes / total_active_slots 
            if total_active_slots > 0 else 0.0
        )
        
        # Create results object
        results = SimulationResults(
            config=self.config,
            total_slots=total_slots,
            mean_lifetime_slots=mean_lifetime_slots,
            mean_lifetime_years=mean_lifetime_years,
            mean_delay=mean_delay,
            tail_delay_95=tail_delay_95,
            tail_delay_99=tail_delay_99,
            mean_queue_length=mean_queue_length,
            throughput=throughput,
            state_fractions=dict(state_fractions),
            mean_energy_consumed=mean_energy_consumed,
            energy_fractions_by_state=dict(energy_fractions),
            total_arrivals=total_arrivals,
            total_deliveries=total_deliveries,
            total_collisions=self.total_collisions,
            total_transmissions=self.total_transmissions,
            empirical_success_prob=empirical_success_prob,
            empirical_service_rate=empirical_service_rate,
            node_statistics=node_stats if self.track_history else None,
            queue_length_history=self.queue_history if self.track_history else None,
            energy_history=self.energy_history if self.track_history else None,
            state_history=self.state_history if self.track_history else None
        )
        
        return results


class BatchSimulator:
    """
    Runs batch simulations with parameter sweeps and multiple replications.
    
    Supports:
    - Parameter sweeps (vary q, ts, n, λ)
    - Multiple replications with different seeds
    - Aggregation of results
    - Confidence intervals
    """
    
    def __init__(self, base_config: SimulationConfig):
        """
        Initialize batch simulator.
        
        Args:
            base_config: Base configuration (will be modified for sweeps)
        """
        self.base_config = base_config
        self.results: List[SimulationResults] = []
    
    def run_replications(
        self,
        n_replications: int = 20,
        verbose: bool = False
    ) -> List[SimulationResults]:
        """
        Run multiple replications with different random seeds.
        
        Args:
            n_replications: Number of replications
            verbose: Print progress
            
        Returns:
            List of SimulationResults
        """
        results = []
        
        if verbose:
            print(f"Running {n_replications} replications...")
        
        for rep in range(n_replications):
            # Create config with different seed
            config = SimulationConfig(
                n_nodes=self.base_config.n_nodes,
                arrival_rate=self.base_config.arrival_rate,
                transmission_prob=self.base_config.transmission_prob,
                idle_timer=self.base_config.idle_timer,
                wakeup_time=self.base_config.wakeup_time,
                initial_energy=self.base_config.initial_energy,
                power_rates=self.base_config.power_rates,
                max_slots=self.base_config.max_slots,
                seed=rep,  # Different seed for each replication
                stop_on_first_depletion=self.base_config.stop_on_first_depletion
            )
            
            # Run simulation
            sim = Simulator(config)
            result = sim.run_simulation(
                track_history=False,
                verbose=False
            )
            results.append(result)
            
            if verbose and (rep + 1) % 5 == 0:
                print(f"  Completed {rep + 1}/{n_replications} replications")
        
        self.results = results
        return results
    
    def parameter_sweep(
        self,
        param_name: str,
        param_values: List[Any],
        n_replications: int = 20,
        verbose: bool = False
    ) -> Dict[Any, List[SimulationResults]]:
        """
        Run simulations sweeping a single parameter.
        
        Args:
            param_name: Name of parameter to sweep ('transmission_prob', 'idle_timer', etc.)
            param_values: List of values to try
            n_replications: Replications per parameter value
            verbose: Print progress
            
        Returns:
            Dictionary mapping parameter values to list of results
        """
        sweep_results = {}
        
        if verbose:
            print(f"Parameter sweep: {param_name}")
            print(f"Values: {param_values}")
            print(f"Replications per value: {n_replications}")
        
        for value in param_values:
            if verbose:
                print(f"\n{param_name} = {value}")
            
            # Create modified config
            config_dict = {
                'n_nodes': self.base_config.n_nodes,
                'arrival_rate': self.base_config.arrival_rate,
                'transmission_prob': self.base_config.transmission_prob,
                'idle_timer': self.base_config.idle_timer,
                'wakeup_time': self.base_config.wakeup_time,
                'initial_energy': self.base_config.initial_energy,
                'power_rates': self.base_config.power_rates,
                'max_slots': self.base_config.max_slots,
                'stop_on_first_depletion': self.base_config.stop_on_first_depletion
            }
            
            # Update parameter
            config_dict[param_name] = value
            
            # Run replications
            results = []
            for rep in range(n_replications):
                config_dict['seed'] = rep
                config = SimulationConfig(**config_dict)
                
                sim = Simulator(config)
                result = sim.run_simulation(track_history=False, verbose=False)
                results.append(result)
            
            sweep_results[value] = results
            
            if verbose:
                # Print summary statistics
                mean_delays = [r.mean_delay for r in results]
                mean_lifetimes = [r.mean_lifetime_years for r in results]
                print(f"  Mean delay: {np.mean(mean_delays):.2f} ± {np.std(mean_delays):.2f} slots")
                print(f"  Mean lifetime: {np.mean(mean_lifetimes):.4f} ± {np.std(mean_lifetimes):.4f} years")
        
        return sweep_results
    
    @staticmethod
    def aggregate_results(results: List[SimulationResults]) -> Dict[str, Tuple[float, float]]:
        """
        Aggregate results from multiple replications.
        
        Returns mean and standard deviation for key metrics.
        
        Args:
            results: List of SimulationResults from replications
            
        Returns:
            Dictionary with (mean, std) tuples for each metric
        """
        metrics = {
            'mean_delay': [r.mean_delay for r in results],
            'tail_delay_95': [r.tail_delay_95 for r in results],
            'tail_delay_99': [r.tail_delay_99 for r in results],
            'mean_lifetime_years': [r.mean_lifetime_years for r in results],
            'throughput': [r.throughput for r in results],
            'mean_queue_length': [r.mean_queue_length for r in results],
            'empirical_success_prob': [r.empirical_success_prob for r in results],
            'empirical_service_rate': [r.empirical_service_rate for r in results],
            'total_collisions': [r.total_collisions for r in results],
        }
        
        aggregated = {}
        for metric_name, values in metrics.items():
            aggregated[metric_name] = (np.mean(values), np.std(values))
        
        return aggregated
