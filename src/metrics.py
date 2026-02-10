"""
Metrics Calculation Module for M2M Sleep-Based Simulator

This module provides comprehensive metrics calculation and analysis utilities,
including analytical comparisons, post-processing of simulation results, and
design guideline generation.

Implements Task 2.1: Compute all required metrics post-simulation and compare
to paper's analytical expressions.

Date: February 10, 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from .simulator import SimulationResults


@dataclass
class AnalyticalMetrics:
    """Container for analytical (theoretical) metrics based on paper formulas."""
    success_probability: float  # p = q(1-q)^(n-1)
    service_rate: float         # μ with/without sleep
    mean_queue_length: float    # ¯L from paper Sec IV-A
    mean_delay: float           # ¯T from paper Eq. 3
    stability_condition: bool   # λ < μ for unsaturated regime
    

@dataclass
class ComparisonMetrics:
    """Container for comparing empirical simulation results with analytical predictions."""
    empirical_success_prob: float
    analytical_success_prob: float
    success_prob_error: float  # Relative error
    
    empirical_service_rate: float
    analytical_service_rate: float
    service_rate_error: float
    
    empirical_mean_delay: float
    analytical_mean_delay: float
    delay_error: float
    
    # Convergence quality
    is_valid_comparison: bool  # Whether comparison makes sense
    warnings: List[str]


class MetricsCalculator:
    """
    Computes comprehensive metrics from simulation results and analytical formulas.
    
    Provides:
    - Analytical metric calculation based on paper
    - Post-simulation metric processing
    - Empirical vs analytical comparison
    - Time-series analysis
    - Energy efficiency metrics
    """
    
    # Constants
    SLOT_DURATION_MS = 6.0  # 3GPP-inspired slot duration
    SECONDS_PER_YEAR = 365.25 * 24 * 3600
    
    @staticmethod
    def compute_analytical_success_probability(n: int, q: float) -> float:
        """
        Compute analytical success probability from paper.
        
        Formula: p = q(1-q)^(n-1)
        This is the probability that exactly one node transmits in a slot.
        
        Args:
            n: Number of nodes
            q: Transmission probability
            
        Returns:
            Success probability p
        """
        if n <= 0 or q < 0 or q > 1:
            return 0.0
        
        # p = q(1-q)^(n-1)
        p = q * ((1 - q) ** (n - 1))
        return p
    
    @staticmethod
    def compute_optimal_q(n: int) -> float:
        """
        Compute optimal transmission probability that maximizes success probability.
        
        From calculus: optimal q = 1/n maximizes p = q(1-q)^(n-1)
        
        Args:
            n: Number of nodes
            
        Returns:
            Optimal q value
        """
        if n <= 0:
            return 0.0
        return 1.0 / n
    
    @staticmethod
    def compute_analytical_service_rate(
        p: float,
        lambda_rate: float,
        tw: int,
        has_sleep: bool = True
    ) -> float:
        """
        Compute analytical service rate μ from paper.
        
        Without sleep: μ = p
        With sleep (on-demand): μ = p / (1 + tw * λ / (1 - λ))
        
        From paper Eq. 12 and related discussion.
        
        Args:
            p: Success probability
            lambda_rate: Arrival rate λ
            tw: Wake-up time in slots
            has_sleep: Whether on-demand sleep is enabled
            
        Returns:
            Service rate μ
        """
        if not has_sleep or tw == 0:
            # No sleep or instant wake-up
            return p
        
        # With sleep: μ = p / (1 + tw * λ / (1 - λ))
        if lambda_rate >= 1.0:
            # Saturated regime - not valid
            return 0.0
        
        denominator = 1 + tw * lambda_rate / (1 - lambda_rate)
        mu = p / denominator
        return mu
    
    @staticmethod
    def compute_analytical_mean_delay(
        lambda_rate: float,
        mu: float,
        tw: int = 0
    ) -> float:
        """
        Compute analytical mean queueing delay from paper.
        
        Formula from paper Eq. 3 and M/M/1 queue theory:
        ¯T = 1/(μ - λ) for unsaturated regime (λ < μ)
        
        Access delay includes wake-up time: ¯D = ¯T + tw (if applicable)
        
        Args:
            lambda_rate: Arrival rate λ
            mu: Service rate μ
            tw: Wake-up time (for access delay calculation)
            
        Returns:
            Mean queueing delay in slots
        """
        if lambda_rate >= mu:
            # Saturated regime - infinite delay
            return float('inf')
        
        # Mean queueing delay
        mean_delay = 1.0 / (mu - lambda_rate)
        
        return mean_delay
    
    @staticmethod
    def compute_analytical_mean_queue_length(
        lambda_rate: float,
        mu: float
    ) -> float:
        """
        Compute analytical mean queue length from M/M/1 theory.
        
        Formula: ¯L = λ / (μ - λ) for λ < μ
        
        From Little's Law: ¯L = λ * ¯T
        
        Args:
            lambda_rate: Arrival rate λ
            mu: Service rate μ
            
        Returns:
            Mean queue length
        """
        if lambda_rate >= mu:
            return float('inf')
        
        mean_queue_length = lambda_rate / (mu - lambda_rate)
        return mean_queue_length
    
    @staticmethod
    def compute_analytical_metrics(
        n: int,
        q: float,
        lambda_rate: float,
        tw: int,
        ts: int,
        has_sleep: bool = True
    ) -> AnalyticalMetrics:
        """
        Compute all analytical metrics based on paper formulas.
        
        Args:
            n: Number of nodes
            q: Transmission probability
            lambda_rate: Arrival rate λ
            tw: Wake-up time
            ts: Idle timer (affects whether sleep is used)
            has_sleep: Whether on-demand sleep is enabled
            
        Returns:
            AnalyticalMetrics object with all theoretical values
        """
        # Success probability
        p = MetricsCalculator.compute_analytical_success_probability(n, q)
        
        # Service rate (depends on sleep)
        mu = MetricsCalculator.compute_analytical_service_rate(
            p, lambda_rate, tw, has_sleep
        )
        
        # Check stability condition
        is_stable = lambda_rate < mu
        
        # Mean queue length
        if is_stable:
            mean_queue_length = MetricsCalculator.compute_analytical_mean_queue_length(
                lambda_rate, mu
            )
            mean_delay = MetricsCalculator.compute_analytical_mean_delay(
                lambda_rate, mu, tw
            )
        else:
            mean_queue_length = float('inf')
            mean_delay = float('inf')
        
        return AnalyticalMetrics(
            success_probability=p,
            service_rate=mu,
            mean_queue_length=mean_queue_length,
            mean_delay=mean_delay,
            stability_condition=is_stable
        )
    
    @staticmethod
    def compute_energy_per_packet(result: SimulationResults) -> float:
        """
        Compute energy consumed per successfully delivered packet.
        
        Args:
            result: SimulationResults object
            
        Returns:
            Energy per packet (units per packet)
        """
        if result.total_deliveries == 0:
            return float('inf')
        
        total_energy_consumed = result.mean_energy_consumed * result.config.n_nodes
        energy_per_packet = total_energy_consumed / result.total_deliveries
        
        return energy_per_packet
    
    @staticmethod
    def compute_delivery_ratio(result: SimulationResults) -> float:
        """
        Compute packet delivery ratio (delivered / arrived).
        
        Args:
            result: SimulationResults object
            
        Returns:
            Delivery ratio (0.0 to 1.0)
        """
        if result.total_arrivals == 0:
            return 0.0
        
        return result.total_deliveries / result.total_arrivals
    
    @staticmethod
    def compute_collision_rate(result: SimulationResults) -> float:
        """
        Compute collision rate (collisions / total slots).
        
        Args:
            result: SimulationResults object
            
        Returns:
            Collision rate per slot
        """
        if result.total_slots == 0:
            return 0.0
        
        return result.total_collisions / result.total_slots
    
    @staticmethod
    def compute_channel_utilization(result: SimulationResults) -> float:
        """
        Compute channel utilization (successful transmissions / total slots).
        
        This is the same as throughput but expressed as a fraction.
        
        Args:
            result: SimulationResults object
            
        Returns:
            Channel utilization (0.0 to 1.0)
        """
        return result.throughput
    
    @staticmethod
    def compare_empirical_vs_analytical(
        result: SimulationResults,
        analytical: AnalyticalMetrics,
        tolerance: float = 0.15
    ) -> ComparisonMetrics:
        """
        Compare empirical simulation results with analytical predictions.
        
        Args:
            result: SimulationResults from simulation
            analytical: AnalyticalMetrics from theory
            tolerance: Acceptable relative error threshold (default 15%)
            
        Returns:
            ComparisonMetrics with comparison results
        """
        warnings_list = []
        
        # Compare success probability
        success_prob_error = abs(
            result.empirical_success_prob - analytical.success_probability
        ) / max(analytical.success_probability, 1e-10)
        
        if success_prob_error > tolerance:
            warnings_list.append(
                f"Success probability error {success_prob_error:.2%} exceeds tolerance {tolerance:.2%}"
            )
        
        # Compare service rate
        service_rate_error = abs(
            result.empirical_service_rate - analytical.service_rate
        ) / max(analytical.service_rate, 1e-10)
        
        if service_rate_error > tolerance:
            warnings_list.append(
                f"Service rate error {service_rate_error:.2%} exceeds tolerance {tolerance:.2%}"
            )
        
        # Compare mean delay (if finite)
        if analytical.mean_delay != float('inf') and result.mean_delay > 0:
            delay_error = abs(
                result.mean_delay - analytical.mean_delay
            ) / max(analytical.mean_delay, 1e-10)
            
            if delay_error > tolerance:
                warnings_list.append(
                    f"Mean delay error {delay_error:.2%} exceeds tolerance {tolerance:.2%}"
                )
        else:
            delay_error = 0.0
            if analytical.mean_delay == float('inf'):
                warnings_list.append("Analytical delay is infinite (saturated regime)")
        
        # Check if comparison is valid
        is_valid = analytical.stability_condition and result.total_deliveries > 100
        
        if not analytical.stability_condition:
            warnings_list.append("System is unstable (λ >= μ)")
        
        if result.total_deliveries < 100:
            warnings_list.append(
                f"Insufficient deliveries ({result.total_deliveries}) for reliable comparison"
            )
        
        return ComparisonMetrics(
            empirical_success_prob=result.empirical_success_prob,
            analytical_success_prob=analytical.success_probability,
            success_prob_error=success_prob_error,
            empirical_service_rate=result.empirical_service_rate,
            analytical_service_rate=analytical.service_rate,
            service_rate_error=service_rate_error,
            empirical_mean_delay=result.mean_delay,
            analytical_mean_delay=analytical.mean_delay,
            delay_error=delay_error,
            is_valid_comparison=is_valid,
            warnings=warnings_list
        )
    
    @staticmethod
    def compute_queue_length_statistics(
        queue_history: List[float]
    ) -> Dict[str, float]:
        """
        Compute statistics from queue length time series.
        
        Args:
            queue_history: List of average queue lengths per slot
            
        Returns:
            Dictionary with statistics (mean, max, std, percentiles)
        """
        if not queue_history:
            return {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        arr = np.array(queue_history)
        
        return {
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'min': float(np.min(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99))
        }
    
    @staticmethod
    def compute_energy_efficiency_metrics(result: SimulationResults) -> Dict[str, float]:
        """
        Compute comprehensive energy efficiency metrics.
        
        Args:
            result: SimulationResults object
            
        Returns:
            Dictionary with various energy efficiency metrics
        """
        metrics = {}
        
        # Energy per packet
        metrics['energy_per_packet'] = MetricsCalculator.compute_energy_per_packet(result)
        
        # Energy per slot
        if result.total_slots > 0:
            metrics['energy_per_slot'] = result.mean_energy_consumed / result.total_slots
        else:
            metrics['energy_per_slot'] = 0.0
        
        # Fraction of energy in each state
        metrics['energy_in_sleep'] = result.energy_fractions_by_state.get('sleep', 0.0)
        metrics['energy_in_active'] = result.energy_fractions_by_state.get('active', 0.0)
        metrics['energy_in_idle'] = result.energy_fractions_by_state.get('idle', 0.0)
        metrics['energy_in_wakeup'] = result.energy_fractions_by_state.get('wakeup', 0.0)
        
        # Energy efficiency score (deliveries per unit energy)
        if result.mean_energy_consumed > 0:
            metrics['packets_per_energy'] = result.total_deliveries / (
                result.mean_energy_consumed * result.config.n_nodes
            )
        else:
            metrics['packets_per_energy'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_latency_metrics(result: SimulationResults) -> Dict[str, float]:
        """
        Compute comprehensive latency metrics.
        
        Args:
            result: SimulationResults object
            
        Returns:
            Dictionary with various latency metrics
        """
        slot_duration_s = MetricsCalculator.SLOT_DURATION_MS / 1000.0
        
        return {
            'mean_delay_slots': result.mean_delay,
            'mean_delay_ms': result.mean_delay * MetricsCalculator.SLOT_DURATION_MS,
            'mean_delay_s': result.mean_delay * slot_duration_s,
            'tail_delay_95_slots': result.tail_delay_95,
            'tail_delay_95_ms': result.tail_delay_95 * MetricsCalculator.SLOT_DURATION_MS,
            'tail_delay_99_slots': result.tail_delay_99,
            'tail_delay_99_ms': result.tail_delay_99 * MetricsCalculator.SLOT_DURATION_MS,
        }
    
    @staticmethod
    def compute_network_performance_metrics(result: SimulationResults) -> Dict[str, float]:
        """
        Compute comprehensive network performance metrics.
        
        Args:
            result: SimulationResults object
            
        Returns:
            Dictionary with various performance metrics
        """
        return {
            'throughput': result.throughput,
            'delivery_ratio': MetricsCalculator.compute_delivery_ratio(result),
            'collision_rate': MetricsCalculator.compute_collision_rate(result),
            'channel_utilization': MetricsCalculator.compute_channel_utilization(result),
            'empirical_success_prob': result.empirical_success_prob,
            'empirical_service_rate': result.empirical_service_rate,
            'total_transmissions': result.total_transmissions,
            'total_successes': result.total_deliveries,
            'total_collisions': result.total_collisions,
        }
    
    @staticmethod
    def compute_comprehensive_metrics(
        result: SimulationResults,
        include_analytical: bool = True
    ) -> Dict[str, Any]:
        """
        Compute all metrics from a simulation result.
        
        This is the main entry point for comprehensive metrics calculation.
        
        Args:
            result: SimulationResults object
            include_analytical: Whether to compute and compare analytical metrics
            
        Returns:
            Dictionary with all metrics organized by category
        """
        metrics = {
            'config': {
                'n_nodes': result.config.n_nodes,
                'arrival_rate': result.config.arrival_rate,
                'transmission_prob': result.config.transmission_prob,
                'idle_timer': result.config.idle_timer,
                'wakeup_time': result.config.wakeup_time,
                'initial_energy': result.config.initial_energy,
                'max_slots': result.config.max_slots,
                'seed': result.config.seed,
            },
            'simulation': {
                'total_slots': result.total_slots,
                'total_arrivals': result.total_arrivals,
                'total_deliveries': result.total_deliveries,
            },
            'latency': MetricsCalculator.compute_latency_metrics(result),
            'energy': MetricsCalculator.compute_energy_efficiency_metrics(result),
            'network': MetricsCalculator.compute_network_performance_metrics(result),
            'lifetime': {
                'mean_lifetime_slots': result.mean_lifetime_slots,
                'mean_lifetime_years': result.mean_lifetime_years,
                'mean_lifetime_days': result.mean_lifetime_years * 365.25,
                'mean_lifetime_hours': result.mean_lifetime_years * 365.25 * 24,
            },
            'state_fractions': result.state_fractions,
        }
        
        # Add analytical comparison if requested
        if include_analytical:
            has_sleep = result.config.idle_timer < float('inf')
            
            analytical = MetricsCalculator.compute_analytical_metrics(
                n=result.config.n_nodes,
                q=result.config.transmission_prob,
                lambda_rate=result.config.arrival_rate,
                tw=result.config.wakeup_time,
                ts=result.config.idle_timer,
                has_sleep=has_sleep
            )
            
            comparison = MetricsCalculator.compare_empirical_vs_analytical(
                result, analytical
            )
            
            metrics['analytical'] = {
                'success_probability': analytical.success_probability,
                'service_rate': analytical.service_rate,
                'mean_queue_length': analytical.mean_queue_length,
                'mean_delay': analytical.mean_delay,
                'stability_condition': analytical.stability_condition,
            }
            
            metrics['comparison'] = {
                'success_prob_error': comparison.success_prob_error,
                'service_rate_error': comparison.service_rate_error,
                'delay_error': comparison.delay_error,
                'is_valid': comparison.is_valid_comparison,
                'warnings': comparison.warnings,
            }
        
        # Add queue statistics if available
        if result.queue_length_history:
            metrics['queue_statistics'] = MetricsCalculator.compute_queue_length_statistics(
                result.queue_length_history
            )
        
        return metrics
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, Any], verbose: bool = True) -> None:
        """
        Print a formatted summary of computed metrics.
        
        Args:
            metrics: Dictionary from compute_comprehensive_metrics
            verbose: Whether to print detailed information
        """
        print("=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        
        # Configuration
        print("\n📋 Configuration:")
        cfg = metrics['config']
        print(f"  Nodes (n): {cfg['n_nodes']}")
        print(f"  Arrival rate (λ): {cfg['arrival_rate']}")
        print(f"  Transmission prob (q): {cfg['transmission_prob']}")
        print(f"  Idle timer (ts): {cfg['idle_timer']}")
        print(f"  Wake-up time (tw): {cfg['wakeup_time']}")
        print(f"  Initial energy: {cfg['initial_energy']}")
        
        # Simulation summary
        print("\n📊 Simulation:")
        sim = metrics['simulation']
        print(f"  Total slots: {sim['total_slots']}")
        print(f"  Total arrivals: {sim['total_arrivals']}")
        print(f"  Total deliveries: {sim['total_deliveries']}")
        
        # Latency metrics
        print("\n⏱️  Latency:")
        lat = metrics['latency']
        print(f"  Mean delay: {lat['mean_delay_slots']:.2f} slots ({lat['mean_delay_ms']:.2f} ms)")
        print(f"  95th percentile: {lat['tail_delay_95_slots']:.2f} slots ({lat['tail_delay_95_ms']:.2f} ms)")
        print(f"  99th percentile: {lat['tail_delay_99_slots']:.2f} slots ({lat['tail_delay_99_ms']:.2f} ms)")
        
        # Energy metrics
        print("\n🔋 Energy:")
        eng = metrics['energy']
        print(f"  Energy per packet: {eng['energy_per_packet']:.2f} units")
        print(f"  Energy per slot: {eng['energy_per_slot']:.6f} units")
        print(f"  Packets per energy: {eng['packets_per_energy']:.6f}")
        
        # Lifetime
        print("\n⏳ Lifetime:")
        lt = metrics['lifetime']
        if lt['mean_lifetime_years'] < 0.01:
            print(f"  Mean lifetime: {lt['mean_lifetime_hours']:.2f} hours")
        elif lt['mean_lifetime_years'] < 1:
            print(f"  Mean lifetime: {lt['mean_lifetime_days']:.2f} days")
        else:
            print(f"  Mean lifetime: {lt['mean_lifetime_years']:.4f} years")
        
        # Network performance
        print("\n🌐 Network Performance:")
        net = metrics['network']
        print(f"  Throughput: {net['throughput']:.6f} packets/slot")
        print(f"  Delivery ratio: {net['delivery_ratio']:.4f}")
        print(f"  Collision rate: {net['collision_rate']:.6f}")
        print(f"  Success probability: {net['empirical_success_prob']:.6f}")
        print(f"  Service rate: {net['empirical_service_rate']:.6f}")
        
        # Analytical comparison
        if 'analytical' in metrics:
            print("\n📐 Analytical Comparison:")
            ana = metrics['analytical']
            comp = metrics['comparison']
            
            print(f"  Success prob: empirical={net['empirical_success_prob']:.6f}, "
                  f"analytical={ana['success_probability']:.6f} "
                  f"(error: {comp['success_prob_error']:.2%})")
            
            print(f"  Service rate: empirical={net['empirical_service_rate']:.6f}, "
                  f"analytical={ana['service_rate']:.6f} "
                  f"(error: {comp['service_rate_error']:.2%})")
            
            if ana['mean_delay'] != float('inf'):
                print(f"  Mean delay: empirical={lat['mean_delay_slots']:.2f}, "
                      f"analytical={ana['mean_delay']:.2f} "
                      f"(error: {comp['delay_error']:.2%})")
            
            print(f"  Stability: {ana['stability_condition']}")
            print(f"  Valid comparison: {comp['is_valid']}")
            
            if comp['warnings'] and verbose:
                print("\n  ⚠️  Warnings:")
                for warning in comp['warnings']:
                    print(f"    - {warning}")
        
        # State fractions
        if verbose:
            print("\n🔄 State Fractions:")
            states = metrics['state_fractions']
            for state, fraction in states.items():
                print(f"  {state.capitalize()}: {fraction:.4f}")
        
        print("\n" + "=" * 80)


def analyze_batch_results(
    batch_results: List[SimulationResults],
    param_name: str = None,
    param_value: Any = None
) -> Dict[str, Tuple[float, float]]:
    """
    Analyze results from batch simulations (multiple replications).
    
    Computes mean and standard deviation for all key metrics across replications.
    
    Args:
        batch_results: List of SimulationResults from replications
        param_name: Optional name of swept parameter
        param_value: Optional value of swept parameter
        
    Returns:
        Dictionary with (mean, std) tuples for each metric
    """
    if not batch_results:
        return {}
    
    # Collect all metrics
    all_metrics = [
        MetricsCalculator.compute_comprehensive_metrics(result, include_analytical=False)
        for result in batch_results
    ]
    
    # Aggregate key metrics
    aggregated = {}
    
    # Latency metrics
    delays = [m['latency']['mean_delay_slots'] for m in all_metrics]
    tail_95 = [m['latency']['tail_delay_95_slots'] for m in all_metrics]
    tail_99 = [m['latency']['tail_delay_99_slots'] for m in all_metrics]
    
    aggregated['mean_delay'] = (np.mean(delays), np.std(delays))
    aggregated['tail_delay_95'] = (np.mean(tail_95), np.std(tail_95))
    aggregated['tail_delay_99'] = (np.mean(tail_99), np.std(tail_99))
    
    # Lifetime metrics
    lifetimes = [m['lifetime']['mean_lifetime_years'] for m in all_metrics]
    finite_lifetimes = [lt for lt in lifetimes if lt != float('inf')]
    
    if finite_lifetimes:
        aggregated['lifetime_years'] = (np.mean(finite_lifetimes), np.std(finite_lifetimes))
    else:
        aggregated['lifetime_years'] = (float('inf'), 0.0)
    
    # Energy metrics
    energy_per_packet = [m['energy']['energy_per_packet'] for m in all_metrics]
    finite_energy = [e for e in energy_per_packet if e != float('inf')]
    
    if finite_energy:
        aggregated['energy_per_packet'] = (np.mean(finite_energy), np.std(finite_energy))
    else:
        aggregated['energy_per_packet'] = (float('inf'), 0.0)
    
    # Network metrics
    throughputs = [m['network']['throughput'] for m in all_metrics]
    delivery_ratios = [m['network']['delivery_ratio'] for m in all_metrics]
    success_probs = [m['network']['empirical_success_prob'] for m in all_metrics]
    
    aggregated['throughput'] = (np.mean(throughputs), np.std(throughputs))
    aggregated['delivery_ratio'] = (np.mean(delivery_ratios), np.std(delivery_ratios))
    aggregated['success_probability'] = (np.mean(success_probs), np.std(success_probs))
    
    return aggregated
