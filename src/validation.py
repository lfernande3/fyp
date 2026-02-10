"""
Validation and Debugging Utilities for M2M Sleep-Based Simulator

This module provides trace logging, sanity checks, and analytical validation
for the discrete-event simulator.

Author: Lance Saquilabon (ID: 57848673)
Project: Sleep-Based Low-Latency Access for Machine-to-Machine Communications
Date: February 10, 2026
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json

from .simulator import Simulator, SimulationConfig, SimulationResults
from .node import Node, NodeState


@dataclass
class SlotTrace:
    """Trace information for a single slot."""
    slot: int
    node_states: List[str]           # State of each node
    queue_lengths: List[int]         # Queue length of each node
    energies: List[float]            # Energy of each node
    transmitting_nodes: List[int]    # IDs of transmitting nodes
    collision: bool                  # Whether collision occurred
    success: bool                    # Whether transmission succeeded
    total_arrivals: int              # Total arrivals in this slot
    total_deliveries: int            # Total deliveries in this slot


class TraceLogger:
    """
    Trace logger for detailed slot-by-slot simulation tracking.
    
    Enables debugging and validation by recording complete state
    information for every simulation slot.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize trace logger.
        
        Args:
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.traces: List[SlotTrace] = []
    
    def log_slot(
        self,
        slot: int,
        nodes: List[Node],
        transmitting_nodes: List[int],
        collision: bool,
        success: bool,
        arrivals: int,
        deliveries: int
    ):
        """
        Log complete state for current slot.
        
        Args:
            slot: Current slot number
            nodes: List of all nodes
            transmitting_nodes: IDs of nodes transmitting
            collision: Whether collision occurred
            success: Whether transmission succeeded
            arrivals: Arrivals in this slot
            deliveries: Deliveries in this slot
        """
        if not self.enabled:
            return
        
        trace = SlotTrace(
            slot=slot,
            node_states=[node.state.value for node in nodes],
            queue_lengths=[node.get_queue_length() for node in nodes],
            energies=[node.energy for node in nodes],
            transmitting_nodes=transmitting_nodes,
            collision=collision,
            success=success,
            total_arrivals=arrivals,
            total_deliveries=deliveries
        )
        
        self.traces.append(trace)
    
    def get_trace(self, slot: int) -> Optional[SlotTrace]:
        """Get trace for specific slot."""
        if slot < 0 or slot >= len(self.traces):
            return None
        return self.traces[slot]
    
    def get_traces_range(self, start: int, end: int) -> List[SlotTrace]:
        """Get traces for range of slots."""
        return self.traces[max(0, start):min(len(self.traces), end)]
    
    def save_to_file(self, filename: str):
        """Save traces to JSON file."""
        traces_dict = []
        for trace in self.traces:
            traces_dict.append({
                'slot': trace.slot,
                'node_states': trace.node_states,
                'queue_lengths': trace.queue_lengths,
                'energies': trace.energies,
                'transmitting_nodes': trace.transmitting_nodes,
                'collision': trace.collision,
                'success': trace.success,
                'total_arrivals': trace.total_arrivals,
                'total_deliveries': trace.total_deliveries
            })
        
        with open(filename, 'w') as f:
            json.dump(traces_dict, f, indent=2)
    
    def print_summary(self, start: int = 0, end: Optional[int] = None):
        """
        Print summary of traces.
        
        Args:
            start: Start slot
            end: End slot (None for all)
        """
        if end is None:
            end = len(self.traces)
        
        print("=" * 80)
        print(f"TRACE SUMMARY (Slots {start}-{end-1})")
        print("=" * 80)
        
        for trace in self.traces[start:end]:
            print(f"\nSlot {trace.slot}:")
            print(f"  States: {trace.node_states}")
            print(f"  Queues: {trace.queue_lengths}")
            print(f"  Energies: {[f'{e:.1f}' for e in trace.energies]}")
            
            if len(trace.transmitting_nodes) > 0:
                print(f"  Transmitting: Nodes {trace.transmitting_nodes}")
                print(f"  Collision: {trace.collision}")
                print(f"  Success: {trace.success}")
            
            if trace.total_arrivals > 0:
                print(f"  Arrivals: {trace.total_arrivals}")
            if trace.total_deliveries > 0:
                print(f"  Deliveries: {trace.total_deliveries}")


class AnalyticalValidator:
    """
    Validates simulation results against analytical models.
    
    Compares empirical results from simulation with theoretical
    predictions from the paper's analytical expressions.
    """
    
    @staticmethod
    def compute_success_probability(n: int, q: float) -> float:
        """
        Compute analytical success probability for slotted Aloha.
        
        p = q * (1-q)^(n-1)
        
        This is the probability that exactly one node transmits
        when n nodes are contending with probability q.
        
        Args:
            n: Number of nodes
            q: Transmission probability
            
        Returns:
            Success probability p
        """
        if n <= 0 or q <= 0 or q > 1:
            return 0.0
        
        return q * ((1 - q) ** (n - 1))
    
    @staticmethod
    def compute_service_rate(
        p: float,
        lambda_arrival: float,
        tw: int,
        has_sleep: bool = True
    ) -> float:
        """
        Compute analytical service rate μ.
        
        From paper Eq. 12:
        - Without sleep: μ = p
        - With sleep: μ = p / (1 + tw * λ / (1 - λ))
        
        Args:
            p: Success probability
            lambda_arrival: Arrival rate
            tw: Wake-up time
            has_sleep: Whether sleep is enabled
            
        Returns:
            Service rate μ
        """
        if not has_sleep:
            return p
        
        if lambda_arrival >= 1:
            return 0.0
        
        denominator = 1 + tw * lambda_arrival / (1 - lambda_arrival)
        return p / denominator
    
    @staticmethod
    def validate_results(
        config: SimulationConfig,
        results: SimulationResults,
        tolerance: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Validate simulation results against analytical models.
        
        Args:
            config: Simulation configuration
            results: Simulation results
            tolerance: Acceptable relative error (0.2 = 20%)
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Success probability
        analytical_p = AnalyticalValidator.compute_success_probability(
            config.n_nodes,
            config.transmission_prob
        )
        empirical_p = results.empirical_success_prob
        
        p_error = abs(analytical_p - empirical_p) / max(analytical_p, 1e-10)
        p_valid = p_error < tolerance
        
        validation['success_probability'] = {
            'analytical': analytical_p,
            'empirical': empirical_p,
            'relative_error': p_error,
            'valid': p_valid
        }
        
        # Service rate (approximate, assumes all nodes always active)
        has_sleep = config.idle_timer < float('inf')
        analytical_mu = AnalyticalValidator.compute_service_rate(
            analytical_p,
            config.arrival_rate,
            config.wakeup_time,
            has_sleep
        )
        empirical_mu = results.empirical_service_rate
        
        mu_error = abs(analytical_mu - empirical_mu) / max(analytical_mu, 1e-10)
        mu_valid = mu_error < tolerance * 2  # Allow 2x tolerance for μ
        
        validation['service_rate'] = {
            'analytical': analytical_mu,
            'empirical': empirical_mu,
            'relative_error': mu_error,
            'valid': mu_valid
        }
        
        return validation


class SanityChecker:
    """
    Performs sanity checks on simulation results.
    
    Verifies expected behaviors like:
    - No sleep (ts=∞) should match standard Aloha
    - Immediate sleep (ts=0) should increase delay
    - Higher q should increase collisions
    """
    
    @staticmethod
    def check_no_sleep_vs_standard_aloha(
        base_config: SimulationConfig,
        tolerance: float = 0.1
    ) -> Dict[str, any]:
        """
        Verify that no-sleep mode matches standard slotted Aloha.
        
        With ts=∞, nodes never sleep, so behavior should match
        standard slotted Aloha without sleep-wake transitions.
        
        Args:
            base_config: Base configuration
            tolerance: Acceptable difference
            
        Returns:
            Dictionary with check results
        """
        # Configure no-sleep simulation
        no_sleep_config = SimulationConfig(
            n_nodes=base_config.n_nodes,
            arrival_rate=base_config.arrival_rate,
            transmission_prob=base_config.transmission_prob,
            idle_timer=int(1e9),  # Effectively infinite
            wakeup_time=base_config.wakeup_time,
            initial_energy=base_config.initial_energy,
            power_rates=base_config.power_rates,
            max_slots=5000,
            seed=42
        )
        
        # Run simulation
        sim = Simulator(no_sleep_config)
        result = sim.run_simulation(verbose=False)
        
        # Check: nodes should stay in ACTIVE or IDLE, never SLEEP
        sleep_fraction = result.state_fractions.get('sleep', 0.0)
        wakeup_fraction = result.state_fractions.get('wakeup', 0.0)
        
        no_sleep_check = sleep_fraction < 0.01 and wakeup_fraction < 0.01
        
        # Compare to analytical success probability
        analytical_p = AnalyticalValidator.compute_success_probability(
            no_sleep_config.n_nodes,
            no_sleep_config.transmission_prob
        )
        
        p_matches = abs(result.empirical_success_prob - analytical_p) / analytical_p < tolerance
        
        return {
            'passed': no_sleep_check and p_matches,
            'sleep_fraction': sleep_fraction,
            'wakeup_fraction': wakeup_fraction,
            'empirical_p': result.empirical_success_prob,
            'analytical_p': analytical_p,
            'p_error': abs(result.empirical_success_prob - analytical_p) / analytical_p
        }
    
    @staticmethod
    def check_immediate_sleep_increases_delay(
        base_config: SimulationConfig
    ) -> Dict[str, any]:
        """
        Verify that immediate sleep (ts=0) increases delay.
        
        With ts=0, nodes sleep immediately when idle, requiring
        wake-up time tw before transmitting, which should increase delay.
        
        Args:
            base_config: Base configuration
            
        Returns:
            Dictionary with check results
        """
        # Normal idle timer
        normal_config = SimulationConfig(
            n_nodes=base_config.n_nodes,
            arrival_rate=base_config.arrival_rate,
            transmission_prob=base_config.transmission_prob,
            idle_timer=10,
            wakeup_time=base_config.wakeup_time,
            initial_energy=base_config.initial_energy,
            power_rates=base_config.power_rates,
            max_slots=5000,
            seed=42
        )
        
        # Immediate sleep
        immediate_config = SimulationConfig(
            n_nodes=base_config.n_nodes,
            arrival_rate=base_config.arrival_rate,
            transmission_prob=base_config.transmission_prob,
            idle_timer=0,  # Immediate sleep
            wakeup_time=base_config.wakeup_time,
            initial_energy=base_config.initial_energy,
            power_rates=base_config.power_rates,
            max_slots=5000,
            seed=42
        )
        
        # Run both
        sim_normal = Simulator(normal_config)
        result_normal = sim_normal.run_simulation(verbose=False)
        
        sim_immediate = Simulator(immediate_config)
        result_immediate = sim_immediate.run_simulation(verbose=False)
        
        # Immediate sleep should have higher delay
        delay_increased = result_immediate.mean_delay > result_normal.mean_delay
        
        # Immediate sleep should have higher sleep fraction
        sleep_increased = (
            result_immediate.state_fractions.get('sleep', 0) >
            result_normal.state_fractions.get('sleep', 0)
        )
        
        return {
            'passed': delay_increased and sleep_increased,
            'normal_delay': result_normal.mean_delay,
            'immediate_delay': result_immediate.mean_delay,
            'delay_increase': result_immediate.mean_delay - result_normal.mean_delay,
            'normal_sleep_fraction': result_normal.state_fractions.get('sleep', 0),
            'immediate_sleep_fraction': result_immediate.state_fractions.get('sleep', 0)
        }
    
    @staticmethod
    def check_higher_q_increases_collisions(
        base_config: SimulationConfig
    ) -> Dict[str, any]:
        """
        Verify that higher transmission probability increases collisions.
        
        Args:
            base_config: Base configuration
            
        Returns:
            Dictionary with check results
        """
        # Low q
        low_q_config = SimulationConfig(
            n_nodes=base_config.n_nodes,
            arrival_rate=base_config.arrival_rate,
            transmission_prob=0.05,
            idle_timer=base_config.idle_timer,
            wakeup_time=base_config.wakeup_time,
            initial_energy=base_config.initial_energy,
            power_rates=base_config.power_rates,
            max_slots=5000,
            seed=42
        )
        
        # High q
        high_q_config = SimulationConfig(
            n_nodes=base_config.n_nodes,
            arrival_rate=base_config.arrival_rate,
            transmission_prob=0.3,
            idle_timer=base_config.idle_timer,
            wakeup_time=base_config.wakeup_time,
            initial_energy=base_config.initial_energy,
            power_rates=base_config.power_rates,
            max_slots=5000,
            seed=42
        )
        
        # Run both
        sim_low = Simulator(low_q_config)
        result_low = sim_low.run_simulation(verbose=False)
        
        sim_high = Simulator(high_q_config)
        result_high = sim_high.run_simulation(verbose=False)
        
        # Higher q should have more collisions
        collisions_increased = result_high.total_collisions > result_low.total_collisions
        
        return {
            'passed': collisions_increased,
            'low_q': 0.05,
            'high_q': 0.3,
            'low_q_collisions': result_low.total_collisions,
            'high_q_collisions': result_high.total_collisions,
            'collision_increase': result_high.total_collisions - result_low.total_collisions
        }
    
    @staticmethod
    def run_all_checks(base_config: SimulationConfig) -> Dict[str, Dict]:
        """
        Run all sanity checks.
        
        Args:
            base_config: Base configuration for checks
            
        Returns:
            Dictionary with all check results
        """
        print("Running sanity checks...\n")
        
        results = {}
        
        # Check 1: No sleep vs standard Aloha
        print("1. No-sleep mode vs. standard Aloha...")
        check1 = SanityChecker.check_no_sleep_vs_standard_aloha(base_config)
        results['no_sleep_check'] = check1
        print(f"   {'PASS' if check1['passed'] else 'FAIL'}")
        
        # Check 2: Immediate sleep increases delay
        print("2. Immediate sleep increases delay...")
        check2 = SanityChecker.check_immediate_sleep_increases_delay(base_config)
        results['immediate_sleep_check'] = check2
        print(f"   {'PASS' if check2['passed'] else 'FAIL'}")
        
        # Check 3: Higher q increases collisions
        print("3. Higher q increases collisions...")
        check3 = SanityChecker.check_higher_q_increases_collisions(base_config)
        results['high_q_check'] = check3
        print(f"   {'PASS' if check3['passed'] else 'FAIL'}")
        
        # Overall result
        all_passed = all(r['passed'] for r in results.values())
        print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
        
        return results


def run_small_scale_test(verbose: bool = True) -> Dict:
    """
    Run small-scale integration test (n=5, 1000 slots).
    
    Args:
        verbose: Print detailed output
        
    Returns:
        Dictionary with test results
    """
    if verbose:
        print("=" * 80)
        print("SMALL-SCALE INTEGRATION TEST")
        print("=" * 80)
        print("\nConfiguration: n=5 nodes, 1000 slots, lambda=0.02, q=0.1")
    
    # Configure small-scale test
    power_rates = {
        'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1
    }
    
    config = SimulationConfig(
        n_nodes=5,
        arrival_rate=0.02,
        transmission_prob=0.1,
        idle_timer=10,
        wakeup_time=5,
        initial_energy=5000.0,
        power_rates=power_rates,
        max_slots=1000,
        seed=42
    )
    
    # Run simulation
    sim = Simulator(config)
    result = sim.run_simulation(track_history=True, verbose=False)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Total slots: {result.total_slots}")
        print(f"  Arrivals: {result.total_arrivals}")
        print(f"  Deliveries: {result.total_deliveries}")
        print(f"  Collisions: {result.total_collisions}")
        print(f"  Mean delay: {result.mean_delay:.2f} slots")
        print(f"  Throughput: {result.throughput:.4f}")
        print(f"  Success probability: {result.empirical_success_prob:.4f}")
        
        print(f"\nState fractions:")
        for state, fraction in result.state_fractions.items():
            print(f"  {state.capitalize()}: {fraction*100:.1f}%")
        
        print(f"\nEnergy:")
        print(f"  Mean consumed: {result.mean_energy_consumed:.2f} units")
        print(f"  Mean lifetime: {result.mean_lifetime_years:.4f} years")
    
    # Validate
    validation = AnalyticalValidator.validate_results(config, result)
    
    if verbose:
        print(f"\nValidation:")
        print(f"  Success probability: {validation['success_probability']['valid']}")
        print(f"    Analytical: {validation['success_probability']['analytical']:.4f}")
        print(f"    Empirical: {validation['success_probability']['empirical']:.4f}")
        print(f"    Error: {validation['success_probability']['relative_error']*100:.1f}%")
        
        print("=" * 80)
    
    return {
        'config': config,
        'results': result,
        'validation': validation
    }


if __name__ == "__main__":
    # Run small-scale test
    test_results = run_small_scale_test(verbose=True)
    
    # Run sanity checks
    print("\n")
    base_config = test_results['config']
    sanity_results = SanityChecker.run_all_checks(base_config)
