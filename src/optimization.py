"""
Optimization Module for Sleep-Based Random Access Parameters

Implements Objective O3: Optimize sleep and access parameters for latency-longevity
trade-offs using simulation results.

Tasks:
- 3.1: Grid search / optimization for optimal q and ts values
        (maximize lifetime or minimize delay; Pareto tradeoff curve)
- 3.2: Prioritization scenario comparison (low-latency vs battery-life priority)
        with duty-cycling extension

Date: March 2026
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .simulator import Simulator, BatchSimulator, SimulationConfig, SimulationResults
from .power_model import PowerModel, PowerProfile
from .metrics import MetricsCalculator, analyze_batch_results
from .baselines import GENERIC_LITERATURE_BASELINE, q_one_over_n, seconds_to_slots


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from a single parameter optimization run."""
    param_name: str              # 'q', 'ts', or 'q_ts'
    objective: str               # 'lifetime' or 'delay'
    optimal_value: Any           # Optimal parameter value(s)
    optimal_metric: float        # Metric value at optimum
    all_values: List             # All parameter values tested
    all_metrics: List[float]     # Metric at each tested value
    all_lifetimes: List[float]   # Mean lifetime at each tested value
    all_delays: List[float]      # Mean delay at each tested value
    config_at_optimum: SimulationConfig


@dataclass
class TradeoffPoint:
    """A point on the lifetime-delay Pareto frontier for a given (q*, ts)."""
    q: float
    ts: int
    lifetime_years: float
    delay_slots: float
    delay_ms: float
    lifetime_std: float = 0.0
    delay_std: float = 0.0


@dataclass
class PrioritizationComparison:
    """Aggregated results from comparing prioritization scenarios."""
    scenario_names: List[str]
    mean_delays: List[float]          # Mean delay in slots
    delay_stds: List[float]
    mean_lifetimes: List[float]       # Mean lifetime in years
    lifetime_stds: List[float]
    configs: List[SimulationConfig]
    gains_vs_baseline: Dict[str, Dict[str, float]]   # % gains/losses vs balanced


# ---------------------------------------------------------------------------
# Helper: run replications for a single config dict
# ---------------------------------------------------------------------------

def _run_replications(
    config_dict: Dict[str, Any],
    n_replications: int
) -> Tuple[List[float], List[float]]:
    """Run n_replications with seeds 0..n-1; return (lifetimes, delays)."""
    rep_lifetimes = []
    rep_delays = []
    for rep in range(n_replications):
        config_dict["seed"] = rep
        config = SimulationConfig(**config_dict)
        sim = Simulator(config)
        result = sim.run_simulation(track_history=False, verbose=False)
        rep_lifetimes.append(result.mean_lifetime_years)
        rep_delays.append(result.mean_delay)
    return rep_lifetimes, rep_delays


def _mean_finite(values: List[float]) -> float:
    finite = [v for v in values if v != float("inf")]
    return float(np.mean(finite)) if finite else float("inf")


def _std_finite(values: List[float]) -> float:
    finite = [v for v in values if v != float("inf")]
    return float(np.std(finite)) if len(finite) > 1 else 0.0


def _config_dict(base: SimulationConfig, **overrides) -> Dict[str, Any]:
    """Create a mutable config dict from a base config, applying overrides."""
    d = {
        "n_nodes": base.n_nodes,
        "arrival_rate": base.arrival_rate,
        "transmission_prob": base.transmission_prob,
        "idle_timer": base.idle_timer,
        "wakeup_time": base.wakeup_time,
        "initial_energy": base.initial_energy,
        "power_rates": base.power_rates,
        "max_slots": base.max_slots,
        "stop_on_first_depletion": base.stop_on_first_depletion,
        "seed": None,
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Task 3.1 – Optimization Logic
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """
    Find optimal transmission probability (q) and/or idle timer (ts) via grid search.

    Based on paper Sec IV: lifetime and delay have monotonic properties in q,
    enabling principled optimization.

    Methods
    -------
    grid_search_q        – Sweep q for a fixed ts; find q* for max lifetime or min delay.
    grid_search_q_ts     – 2-D grid search over (q, ts) space.
    tradeoff_analysis    – For each ts, find (max-lifetime q*, min-delay q*) and
                           collect the resulting Pareto points.
    """

    @staticmethod
    def grid_search_q(
        base_config: SimulationConfig,
        q_values: Optional[List[float]] = None,
        objective: str = "lifetime",
        n_replications: int = 10,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Grid search over transmission probability q.

        For each q, runs ``n_replications`` and computes mean lifetime/delay.
        Returns the q* that maximises lifetime or minimises delay.

        Parameters
        ----------
        base_config   : Base configuration (ts, tw, n, λ, energy fixed).
        q_values      : q values to test.  Default: 20 values in [0.01, 0.5].
        objective     : ``'lifetime'`` (maximise) or ``'delay'`` (minimise).
        n_replications: Replications per q value.
        verbose       : Print progress.

        Returns
        -------
        OptimizationResult
        """
        if q_values is None:
            q_values = list(np.linspace(0.01, 0.5, 20))

        if verbose:
            print(
                f"Grid search over q: {len(q_values)} values, "
                f"{n_replications} reps each. Objective: {objective}."
            )

        all_lifetimes: List[float] = []
        all_delays: List[float] = []

        for i, q in enumerate(q_values):
            if verbose and (i % 5 == 0 or i == len(q_values) - 1):
                print(f"  q = {q:.4f} ({i + 1}/{len(q_values)})")

            cd = _config_dict(base_config, transmission_prob=q)
            rep_lt, rep_d = _run_replications(cd, n_replications)

            all_lifetimes.append(_mean_finite(rep_lt))
            all_delays.append(float(np.mean(rep_d)))

        # Determine best index
        if objective == "lifetime":
            all_metrics = all_lifetimes
            finite_pairs = [(i, m) for i, m in enumerate(all_metrics) if m != float("inf")]
            best_idx = max(finite_pairs, key=lambda x: x[1])[0] if finite_pairs else 0
        else:
            all_metrics = all_delays
            nonzero_pairs = [(i, m) for i, m in enumerate(all_metrics) if m > 0]
            best_idx = min(nonzero_pairs, key=lambda x: x[1])[0] if nonzero_pairs else 0

        optimal_q = q_values[best_idx]
        optimal_metric = all_metrics[best_idx]
        config_at_opt = SimulationConfig(**_config_dict(base_config, transmission_prob=optimal_q))

        if verbose:
            if objective == "lifetime":
                print(f"\nOptimal q* = {optimal_q:.4f}  →  max lifetime ≈ {optimal_metric:.4f} years")
            else:
                print(f"\nOptimal q* = {optimal_q:.4f}  →  min delay ≈ {optimal_metric:.2f} slots")

        return OptimizationResult(
            param_name="q",
            objective=objective,
            optimal_value=optimal_q,
            optimal_metric=optimal_metric,
            all_values=list(q_values),
            all_metrics=all_metrics,
            all_lifetimes=all_lifetimes,
            all_delays=all_delays,
            config_at_optimum=config_at_opt,
        )

    @staticmethod
    def tradeoff_analysis(
        base_config: SimulationConfig,
        q_values: Optional[List[float]] = None,
        ts_values: Optional[List[int]] = None,
        n_replications: int = 10,
        verbose: bool = True,
    ) -> List[TradeoffPoint]:
        """
        Tradeoff analysis: for each ts, sweep q and record the Pareto-optimal point
        (the q that maximises lifetime and its corresponding delay).

        This generates the Pareto frontier showing the tension between lifetime
        maximisation and delay minimisation as ts varies.

        Parameters
        ----------
        base_config   : Base configuration.
        q_values      : q values to sweep.  Default: 15 values in [0.01, 0.4].
        ts_values     : ts values to test.  Default: [1, 5, 10, 20, 50].
        n_replications: Replications per (q, ts) combination.
        verbose       : Print progress.

        Returns
        -------
        List of TradeoffPoint objects (one per ts value, at the best-lifetime q*).
        """
        if q_values is None:
            q_values = list(np.linspace(0.01, 0.4, 15))
        if ts_values is None:
            ts_values = [1, 5, 10, 20, 50]

        if verbose:
            total = len(ts_values) * len(q_values) * n_replications
            print(
                f"Tradeoff analysis: {len(ts_values)} ts × {len(q_values)} q = "
                f"{len(ts_values) * len(q_values)} configs, "
                f"{n_replications} reps each ({total} total runs)."
            )

        tradeoff_points: List[TradeoffPoint] = []

        for ts in ts_values:
            if verbose:
                print(f"\n  ts = {ts}:")

            best_lt = -1.0
            best_point: Optional[TradeoffPoint] = None

            for q in q_values:
                cd = _config_dict(base_config, transmission_prob=q, idle_timer=ts)
                rep_lt, rep_d = _run_replications(cd, n_replications)

                mean_lt = _mean_finite(rep_lt)
                std_lt = _std_finite(rep_lt)
                mean_d = float(np.mean(rep_d))
                std_d = float(np.std(rep_d)) if len(rep_d) > 1 else 0.0

                if mean_lt != float("inf") and mean_lt > best_lt:
                    best_lt = mean_lt
                    best_point = TradeoffPoint(
                        q=q,
                        ts=ts,
                        lifetime_years=mean_lt,
                        delay_slots=mean_d,
                        delay_ms=mean_d * 6.0,
                        lifetime_std=std_lt,
                        delay_std=std_d,
                    )

            if best_point is not None:
                tradeoff_points.append(best_point)
                if verbose:
                    print(
                        f"    Max lifetime q* = {best_point.q:.3f}  →  "
                        f"L = {best_point.lifetime_years:.4f} yr, "
                        f"T = {best_point.delay_slots:.1f} slots"
                    )

        return tradeoff_points

    @staticmethod
    def grid_search_q_ts(
        base_config: SimulationConfig,
        q_values: Optional[List[float]] = None,
        ts_values: Optional[List[int]] = None,
        n_replications: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        2-D grid search over (q, ts) space.

        Parameters
        ----------
        base_config   : Base configuration.
        q_values      : q values.  Default: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3].
        ts_values     : ts values. Default: [1, 5, 10, 20, 50].
        n_replications: Replications per grid point.
        verbose       : Print progress.

        Returns
        -------
        Dictionary with keys:
          ``q_values``, ``ts_values``,
          ``lifetime_matrix``  (shape: n_ts × n_q),
          ``delay_matrix``     (shape: n_ts × n_q),
          ``lifetime_std_matrix``, ``delay_std_matrix``.
        """
        if q_values is None:
            q_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        if ts_values is None:
            ts_values = [1, 5, 10, 20, 50]

        n_q = len(q_values)
        n_ts = len(ts_values)
        total = n_q * n_ts

        lifetime_matrix = np.zeros((n_ts, n_q))
        delay_matrix = np.zeros((n_ts, n_q))
        lifetime_std_matrix = np.zeros((n_ts, n_q))
        delay_std_matrix = np.zeros((n_ts, n_q))

        if verbose:
            print(
                f"2-D grid search: {n_q} q × {n_ts} ts = {total} configs, "
                f"{n_replications} reps each."
            )

        done = 0
        for i, ts in enumerate(ts_values):
            for j, q in enumerate(q_values):
                cd = _config_dict(base_config, transmission_prob=q, idle_timer=ts)
                rep_lt, rep_d = _run_replications(cd, n_replications)

                mean_lt = _mean_finite(rep_lt)
                std_lt = _std_finite(rep_lt)
                mean_d = float(np.mean(rep_d))
                std_d = float(np.std(rep_d)) if len(rep_d) > 1 else 0.0

                lifetime_matrix[i, j] = mean_lt if mean_lt != float("inf") else 0.0
                delay_matrix[i, j] = mean_d
                lifetime_std_matrix[i, j] = std_lt
                delay_std_matrix[i, j] = std_d

                done += 1
                if verbose and (done % 5 == 0 or done == total):
                    lt_str = f"{mean_lt:.4f}" if mean_lt != float("inf") else "inf"
                    print(
                        f"  [{done}/{total}] ts={ts}, q={q:.3f}: "
                        f"L={lt_str}y, T={mean_d:.1f}slots"
                    )

        return {
            "q_values": q_values,
            "ts_values": ts_values,
            "lifetime_matrix": lifetime_matrix,
            "delay_matrix": delay_matrix,
            "lifetime_std_matrix": lifetime_std_matrix,
            "delay_std_matrix": delay_std_matrix,
        }


# ---------------------------------------------------------------------------
# Task 3.2 – Duty-Cycle Comparison
# ---------------------------------------------------------------------------

class DutyCycleSimulator:
    """
    Approximate periodic duty-cycling for comparison with on-demand sleep.

    In duty cycling, nodes wake up on a fixed periodic schedule regardless of
    traffic.  This class models that behaviour by setting the idle timer equal
    to the awake-phase length so the node immediately returns to sleep once the
    awake window closes.

    Note: this is an approximation of true duty cycling; it uses the existing
    on-demand Node/Simulator infrastructure with a short ts to mimic fixed
    awake windows.
    """

    @staticmethod
    def run_duty_cycle_simulation(
        base_config: SimulationConfig,
        cycle_period: int,
        awake_fraction: float,
        seed: Optional[int] = None,
    ) -> SimulationResults:
        """
        Run one duty-cycle simulation with a given cycle period and awake fraction.

        Parameters
        ----------
        base_config    : Base simulation configuration.
        cycle_period   : Total cycle length in slots (awake + sleep).
        awake_fraction : Fraction of cycle the node is awake (0 < f ≤ 1).
        seed           : Random seed.

        Returns
        -------
        SimulationResults
        """
        awake_slots = max(1, int(awake_fraction * cycle_period))
        cd = _config_dict(
            base_config,
            idle_timer=awake_slots,
            wakeup_time=1,   # minimal wakeup cost (scheduled wake)
            seed=seed,
        )
        sim = Simulator(SimulationConfig(**cd))
        return sim.run_simulation(track_history=False, verbose=False)

    @staticmethod
    def compare_with_on_demand(
        base_config: SimulationConfig,
        ts_values: Optional[List[int]] = None,
        awake_fractions: Optional[List[float]] = None,
        cycle_period: int = 20,
        n_replications: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare on-demand sleep vs duty cycling on the lifetime-delay plane.

        Parameters
        ----------
        base_config     : Base simulation configuration.
        ts_values       : Idle-timer values for on-demand sleep.
                          Default: [1, 5, 10, 20, 50].
        awake_fractions : Awake fractions for duty cycling.
                          Default: [0.1, 0.2, 0.3, 0.5, 0.7].
        cycle_period    : Cycle period for duty cycling (slots).
        n_replications  : Replications per configuration.
        verbose         : Print progress.

        Returns
        -------
        Dictionary with ``'on_demand'`` and ``'duty_cycle'`` sub-dicts,
        each containing lifetimes, delays, stds, and the respective
        sweep-parameter values.
        """
        if ts_values is None:
            ts_values = [1, 5, 10, 20, 50]
        if awake_fractions is None:
            awake_fractions = [0.1, 0.2, 0.3, 0.5, 0.7]

        if verbose:
            print(
                f"On-demand vs Duty-cycle comparison\n"
                f"  On-demand ts values:     {ts_values}\n"
                f"  Duty-cycle awake fracs:  {awake_fractions}\n"
                f"  Cycle period: {cycle_period} slots"
            )

        # ---- on-demand ----
        od_lifetimes, od_delays = [], []
        od_lt_stds, od_d_stds = [], []

        for ts in ts_values:
            cd = _config_dict(base_config, idle_timer=ts)
            rep_lt, rep_d = _run_replications(cd, n_replications)

            mean_lt = _mean_finite(rep_lt)
            od_lifetimes.append(mean_lt)
            od_lt_stds.append(_std_finite(rep_lt))
            od_delays.append(float(np.mean(rep_d)))
            od_d_stds.append(float(np.std(rep_d)) if len(rep_d) > 1 else 0.0)

            if verbose:
                lt_str = f"{mean_lt:.4f}" if mean_lt != float("inf") else "inf"
                print(f"  On-demand ts={ts}: L={lt_str}y, T={np.mean(rep_d):.1f}s")

        # ---- duty cycle ----
        dc_lifetimes, dc_delays = [], []
        dc_lt_stds, dc_d_stds = [], []

        for frac in awake_fractions:
            rep_lt, rep_d = [], []
            for rep in range(n_replications):
                r = DutyCycleSimulator.run_duty_cycle_simulation(
                    base_config, cycle_period, frac, seed=rep
                )
                rep_lt.append(r.mean_lifetime_years)
                rep_d.append(r.mean_delay)

            mean_lt = _mean_finite(rep_lt)
            dc_lifetimes.append(mean_lt)
            dc_lt_stds.append(_std_finite(rep_lt))
            dc_delays.append(float(np.mean(rep_d)))
            dc_d_stds.append(float(np.std(rep_d)) if len(rep_d) > 1 else 0.0)

            if verbose:
                lt_str = f"{mean_lt:.4f}" if mean_lt != float("inf") else "inf"
                print(f"  Duty-cycle frac={frac:.2f}: L={lt_str}y, T={np.mean(rep_d):.1f}s")

        return {
            "on_demand": {
                "ts_values": ts_values,
                "lifetimes": od_lifetimes,
                "delays": od_delays,
                "lifetime_stds": od_lt_stds,
                "delay_stds": od_d_stds,
            },
            "duty_cycle": {
                "awake_fractions": awake_fractions,
                "cycle_period": cycle_period,
                "lifetimes": dc_lifetimes,
                "delays": dc_delays,
                "lifetime_stds": dc_lt_stds,
                "delay_stds": dc_d_stds,
            },
        }


# ---------------------------------------------------------------------------
# Task 3.2 – Prioritization Scenario Comparison
# ---------------------------------------------------------------------------

class PrioritizationAnalyzer:
    """
    Compare low-latency priority vs battery-life priority configurations.

    Three canonical scenarios are defined:

    * **Low-Latency Priority** – small ts, slightly higher q to reduce contention wait.
    * **Balanced**             – moderate ts, q ≈ 1/n.
    * **Battery-Life Priority**– large ts (less frequent sleep/wakeup cycling), low q.

    The ``run_scenario_comparison`` method runs all three, computes their
    mean delay and lifetime, and quantifies the % gains/losses of each
    vs the balanced baseline.
    """

    SCENARIOS: Dict[str, Dict[str, Any]] = {
        "low_latency": {
            "description": "Low-Latency Priority (ts=1, q=2/n)",
            "ts": 1,
            "q_factor": 2.0,   # q = q_factor / n
            "tw": 2,
        },
        "balanced": {
            "description": "Balanced (ts=10, q=1/n)",
            "ts": 10,
            "q_factor": 1.0,
            "tw": 3,
        },
        "battery_life": {
            "description": "Battery-Life Priority (ts=50, q=0.5/n)",
            "ts": 50,
            "q_factor": 0.5,
            "tw": 5,
        },
    }

    @staticmethod
    def run_scenario_comparison(
        n_nodes: int = 20,
        arrival_rate: float = 0.01,
        initial_energy: float = 5000,
        power_profile: PowerProfile = PowerProfile.GENERIC_LOW,
        max_slots: int = 50000,
        n_replications: int = 20,
        verbose: bool = True,
    ) -> PrioritizationComparison:
        """
        Run the three prioritization scenarios and return comparison results.

        Parameters
        ----------
        n_nodes        : Number of nodes.
        arrival_rate   : Arrival rate λ.
        initial_energy : Initial energy per node.
        power_profile  : Power consumption profile.
        max_slots      : Maximum simulation slots.
        n_replications : Replications per scenario.
        verbose        : Print progress.

        Returns
        -------
        PrioritizationComparison
        """
        power_rates = PowerModel.get_profile(power_profile)

        names: List[str] = []
        mean_delays: List[float] = []
        delay_stds: List[float] = []
        mean_lifetimes: List[float] = []
        lifetime_stds: List[float] = []
        configs: List[SimulationConfig] = []

        for key, params in PrioritizationAnalyzer.SCENARIOS.items():
            q = min(params["q_factor"] / n_nodes, 0.5)
            ts = params["ts"]
            tw = params["tw"]

            if verbose:
                print(f"\nRunning: {params['description']}")
                print(f"  q={q:.4f}, ts={ts}, tw={tw}")

            cd = {
                "n_nodes": n_nodes,
                "arrival_rate": arrival_rate,
                "transmission_prob": q,
                "idle_timer": ts,
                "wakeup_time": tw,
                "initial_energy": initial_energy,
                "power_rates": power_rates,
                "max_slots": max_slots,
                "stop_on_first_depletion": False,
                "seed": None,
            }
            rep_lt, rep_d = _run_replications(cd, n_replications)

            mean_lt = _mean_finite(rep_lt)
            std_lt = _std_finite(rep_lt)
            mean_d = float(np.mean(rep_d))
            std_d = float(np.std(rep_d)) if len(rep_d) > 1 else 0.0

            names.append(params["description"])
            mean_delays.append(mean_d)
            delay_stds.append(std_d)
            mean_lifetimes.append(mean_lt)
            lifetime_stds.append(std_lt)
            configs.append(SimulationConfig(**{**cd, "seed": 0}))

            if verbose:
                lt_str = f"{mean_lt:.4f}" if mean_lt != float("inf") else "inf"
                print(
                    f"  Delay:    {mean_d:.2f} ± {std_d:.2f} slots "
                    f"({mean_d * 6.0:.1f} ms)"
                )
                print(f"  Lifetime: {lt_str} ± {std_lt:.4f} years")

        gains = PrioritizationAnalyzer._compute_gains(names, mean_delays, mean_lifetimes)

        return PrioritizationComparison(
            scenario_names=names,
            mean_delays=mean_delays,
            delay_stds=delay_stds,
            mean_lifetimes=mean_lifetimes,
            lifetime_stds=lifetime_stds,
            configs=configs,
            gains_vs_baseline=gains,
        )

    @staticmethod
    def _compute_gains(
        names: List[str],
        mean_delays: List[float],
        mean_lifetimes: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute % delay / lifetime changes for each scenario vs the balanced baseline.

        Negative delay_change_pct  = less delay  (improvement).
        Positive lifetime_change_pct = more lifetime (improvement).
        """
        # Find balanced baseline index
        balanced_idx = next(
            (i for i, n in enumerate(names) if "balanced" in n.lower()), len(names) // 2
        )
        base_delay = mean_delays[balanced_idx]
        base_lt = mean_lifetimes[balanced_idx]

        gains: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(names):
            d_chg = (
                (mean_delays[i] - base_delay) / base_delay * 100
                if base_delay > 0 and mean_delays[i] > 0
                else 0.0
            )
            lt_chg = (
                (mean_lifetimes[i] - base_lt) / base_lt * 100
                if base_lt not in (float("inf"), 0) and mean_lifetimes[i] not in (float("inf"), 0)
                else 0.0
            )
            gains[name] = {
                "delay_change_pct": d_chg,
                "lifetime_change_pct": lt_chg,
                "delay_slots": mean_delays[i],
                "lifetime_years": mean_lifetimes[i],
            }
        return gains

    @staticmethod
    def print_comparison_summary(comparison: PrioritizationComparison) -> None:
        """Print a formatted summary of scenario comparison results."""
        print("=" * 72)
        print("PRIORITIZATION SCENARIO COMPARISON SUMMARY")
        print("=" * 72)
        print(f"\n{'Scenario':<42} {'Delay (slots)':<16} {'Lifetime (years)'}")
        print("-" * 72)
        for i, name in enumerate(comparison.scenario_names):
            d = comparison.mean_delays[i]
            ds = comparison.delay_stds[i]
            lt = comparison.mean_lifetimes[i]
            lts = comparison.lifetime_stds[i]
            lt_str = f"{lt:.4f} ± {lts:.4f}" if lt != float("inf") else "inf"
            print(f"  {name:<40} {d:.2f} ± {ds:.2f}     {lt_str}")

        print("\n" + "-" * 72)
        print("Gains / Losses vs Balanced Baseline:")
        print("-" * 72)
        for name, g in comparison.gains_vs_baseline.items():
            dc = g["delay_change_pct"]
            lc = g["lifetime_change_pct"]
            dc_sym = "↓" if dc < 0 else "↑"
            lc_sym = "↑" if lc > 0 else "↓"
            print(f"\n  {name}:")
            print(f"    Delay:    {dc_sym} {abs(dc):.1f}%")
            print(f"    Lifetime: {lc_sym} {abs(lc):.1f}%")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Visualization utilities for O3
# ---------------------------------------------------------------------------

class OptimizationVisualizer:
    """
    Visualization helpers for optimization and comparison results.

    All methods return (fig, ax) and close gracefully if axes are provided.
    """

    @staticmethod
    def plot_q_sweep(
        q_values: List[float],
        lifetimes: List[float],
        delays: List[float],
        optimal_q_lifetime: Optional[float] = None,
        optimal_q_delay: Optional[float] = None,
        title: str = "Lifetime and Delay vs Transmission Probability q",
        ax_left: Optional[Any] = None,
        ax_right: Optional[Any] = None,
        ts_label: Optional[str] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Side-by-side plots of lifetime (left) and delay (right) vs q,
        with optional vertical markers at the optimal q values.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax_left is None or ax_right is None:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

        label_sfx = f" (ts={ts_label})" if ts_label else ""
        lt_plot = [lt if lt != float("inf") else np.nan for lt in lifetimes]

        # Left – Lifetime vs q
        ax_left.plot(q_values, lt_plot, "o-", color="#2196F3",
                     label=f"Mean Lifetime{label_sfx}", linewidth=2)
        if optimal_q_lifetime is not None:
            opt_i = min(range(len(q_values)), key=lambda k: abs(q_values[k] - optimal_q_lifetime))
            ax_left.axvline(optimal_q_lifetime, color="green", linestyle="--", alpha=0.7,
                            label=f"q* = {optimal_q_lifetime:.3f}")
            ax_left.scatter([optimal_q_lifetime], [lt_plot[opt_i]], color="green", s=100, zorder=5)
        ax_left.set_xlabel("Transmission Probability q")
        ax_left.set_ylabel("Mean Lifetime (years)")
        ax_left.set_title("Lifetime vs q")
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # Right – Delay vs q
        ax_right.plot(q_values, delays, "s-", color="#F44336",
                      label=f"Mean Delay{label_sfx}", linewidth=2)
        if optimal_q_delay is not None:
            opt_i = min(range(len(q_values)), key=lambda k: abs(q_values[k] - optimal_q_delay))
            ax_right.axvline(optimal_q_delay, color="purple", linestyle="--", alpha=0.7,
                             label=f"q* = {optimal_q_delay:.3f}")
            ax_right.scatter([optimal_q_delay], [delays[opt_i]], color="purple", s=100, zorder=5)
        ax_right.set_xlabel("Transmission Probability q")
        ax_right.set_ylabel("Mean Delay (slots)")
        ax_right.set_title("Delay vs q")
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

        if fig is not None:
            fig.suptitle(title, fontsize=14)
            plt.tight_layout()

        return fig, ax_left, ax_right

    @staticmethod
    def plot_pareto_frontier(
        tradeoff_points: List[TradeoffPoint],
        title: str = "Pareto Frontier: Max Lifetime vs Min Delay for varying ts",
        ax: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Scatter + line plot of (min delay, max lifetime) Pareto points for each ts.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        ts_vals = sorted({pt.ts for pt in tradeoff_points})
        colors = cm.viridis(np.linspace(0.1, 0.9, max(len(ts_vals), 1)))
        ts_color = dict(zip(ts_vals, colors))

        for pt in tradeoff_points:
            c = ts_color[pt.ts]
            ax.scatter(pt.delay_slots, pt.lifetime_years, color=c, s=120, zorder=5,
                       label=f"ts={pt.ts}, q*={pt.q:.3f}")
            if pt.delay_std > 0 or pt.lifetime_std > 0:
                ax.errorbar(pt.delay_slots, pt.lifetime_years,
                            xerr=pt.delay_std, yerr=pt.lifetime_std,
                            color=c, alpha=0.4, capsize=3)

        sorted_pts = sorted(tradeoff_points, key=lambda p: p.delay_slots)
        ax.plot(
            [p.delay_slots for p in sorted_pts],
            [p.lifetime_years for p in sorted_pts],
            "k--", alpha=0.5, linewidth=1.5, label="Tradeoff curve",
        )

        ax.set_xlabel("Min Achievable Delay (slots)", fontsize=12)
        ax.set_ylabel("Max Achievable Lifetime (years)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        if fig is not None:
            plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_tradeoff_heatmap(
        grid_results: Dict[str, Any],
        metric: str = "lifetime",
        title: Optional[str] = None,
        ax: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Heatmap of mean lifetime or mean delay over the (q, ts) grid.

        Parameters
        ----------
        grid_results : Output of ``ParameterOptimizer.grid_search_q_ts``.
        metric       : ``'lifetime'`` or ``'delay'``.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        q_values = grid_results["q_values"]
        ts_values = grid_results["ts_values"]

        if metric == "lifetime":
            data = grid_results["lifetime_matrix"]
            label = "Mean Lifetime (years)"
            cmap = "YlGn"
        else:
            data = grid_results["delay_matrix"]
            label = "Mean Delay (slots)"
            cmap = "YlOrRd_r"

        if title is None:
            title = f"{label} over (q, ts) Grid"

        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap=cmap,
            extent=[min(q_values), max(q_values), -0.5, len(ts_values) - 0.5],
        )
        ax.set_yticks(range(len(ts_values)))
        ax.set_yticklabels([f"ts={ts}" for ts in ts_values])
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("Transmission Probability q")
        ax.set_title(title)

        if fig is not None:
            plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_prioritization_comparison(
        comparison: PrioritizationComparison,
        title: str = "Latency-Lifetime Tradeoffs: Prioritization Scenarios",
        ax: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Scatter plot comparing mean delay vs mean lifetime for each scenario.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        colors = ["#F44336", "#FF9800", "#4CAF50"]
        markers = ["o", "s", "^"]

        for i, name in enumerate(comparison.scenario_names):
            delay = comparison.mean_delays[i]
            lt = comparison.mean_lifetimes[i]
            if lt == float("inf"):
                continue
            c = colors[i % len(colors)]
            ax.scatter(delay, lt, color=c, marker=markers[i % len(markers)], s=200, zorder=5,
                       label=name)
            ax.errorbar(delay, lt,
                        xerr=comparison.delay_stds[i],
                        yerr=comparison.lifetime_stds[i],
                        color=c, alpha=0.4, capsize=4)
            ax.annotate(name, (delay, lt), textcoords="offset points",
                        xytext=(8, 4), fontsize=9, color=c)

        ax.set_xlabel("Mean Delay (slots)", fontsize=12)
        ax.set_ylabel("Mean Lifetime (years)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if fig is not None:
            plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_duty_cycle_comparison(
        comparison_data: Dict[str, Any],
        title: str = "On-Demand Sleep vs Duty Cycling: Lifetime-Delay Tradeoff",
        ax: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Plot on-demand sleep and duty-cycling curves on the lifetime-delay plane.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        od = comparison_data["on_demand"]
        dc = comparison_data["duty_cycle"]

        # On-demand trace
        od_d, od_lt, od_labels = [], [], []
        for i, ts in enumerate(od["ts_values"]):
            lt = od["lifetimes"][i]
            if lt not in (float("inf"),) and lt > 0:
                od_d.append(od["delays"][i])
                od_lt.append(lt)
                od_labels.append(f"ts={ts}")
        if od_d:
            ax.plot(od_d, od_lt, "b-o", linewidth=2, markersize=8,
                    label="On-Demand Sleep", zorder=5)
            for d, lt, lbl in zip(od_d, od_lt, od_labels):
                ax.annotate(lbl, (d, lt), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color="blue")

        # Duty-cycle trace
        dc_d, dc_lt, dc_labels = [], [], []
        for i, frac in enumerate(dc["awake_fractions"]):
            lt = dc["lifetimes"][i]
            if lt not in (float("inf"),) and lt > 0:
                dc_d.append(dc["delays"][i])
                dc_lt.append(lt)
                dc_labels.append(f"frac={frac:.2f}")
        if dc_d:
            ax.plot(dc_d, dc_lt, "r-s", linewidth=2, markersize=8,
                    label="Duty Cycling", zorder=5)
            for d, lt, lbl in zip(dc_d, dc_lt, dc_labels):
                ax.annotate(lbl, (d, lt), textcoords="offset points",
                            xytext=(5, -15), fontsize=8, color="red")

        ax.set_xlabel("Mean Delay (slots)", fontsize=12)
        ax.set_ylabel("Mean Lifetime (years)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if fig is not None:
            plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_gains_losses_bar(
        comparison: PrioritizationComparison,
        title: str = "Gains/Losses vs Balanced Baseline",
        ax: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Grouped bar chart showing % change in delay and lifetime vs balanced baseline.
        Green bars = improvement, red = degradation.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        scenarios, delay_chgs, lt_chgs = [], [], []
        for name, g in comparison.gains_vs_baseline.items():
            if "balanced" not in name.lower():
                scenarios.append(name.split("(")[0].strip())
                delay_chgs.append(g["delay_change_pct"])
                lt_chgs.append(g["lifetime_change_pct"])

        x = np.arange(len(scenarios))
        w = 0.35

        ax.bar(x - w / 2, delay_chgs, w, label="Delay Change (%)",
               color=["#4CAF50" if v < 0 else "#F44336" for v in delay_chgs], alpha=0.75)
        ax.bar(x + w / 2, lt_chgs, w, label="Lifetime Change (%)",
               color=["#4CAF50" if v > 0 else "#F44336" for v in lt_chgs], alpha=0.75)

        for xi, (dc, lc) in enumerate(zip(delay_chgs, lt_chgs)):
            ax.text(xi - w / 2, dc + (1 if dc >= 0 else -3), f"{dc:.1f}%",
                    ha="center", fontsize=10)
            ax.text(xi + w / 2, lc + (1 if lc >= 0 else -3), f"{lc:.1f}%",
                    ha="center", fontsize=10)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel("Change vs Balanced Baseline (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        if fig is not None:
            plt.tight_layout()

        return fig, ax


# ---------------------------------------------------------------------------
# Convenience function – run all O3 experiments
# ---------------------------------------------------------------------------

def run_optimization_experiments(
    output_dir: str = "results",
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run all O3 optimization experiments (Tasks 3.1 and 3.2).

    Parameters
    ----------
    output_dir : Directory to save results (not used internally, passed for context).
    quick_mode : Use fewer values / replications for faster execution.

    Returns
    -------
    Dictionary containing:
      ``opt_q_lifetime``         – OptimizationResult for max-lifetime q*
      ``opt_q_delay``            – OptimizationResult for min-delay q*
      ``tradeoff_points``        – List[TradeoffPoint] Pareto frontier
      ``grid_results``           – 2-D grid search dict
      ``prioritization_comparison`` – PrioritizationComparison
      ``duty_cycle_comparison``  – On-demand vs duty-cycle dict
    """
    n_reps = 5 if quick_mode else 10
    max_slots = 30_000 if quick_mode else 50_000

    print("=" * 80)
    print("O3 OPTIMIZATION EXPERIMENTS")
    print("=" * 80)
    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Replications per configuration: {n_reps}")

    base_config = SimulationConfig(
        n_nodes=20,
        arrival_rate=GENERIC_LITERATURE_BASELINE.arrival_rate,
        transmission_prob=q_one_over_n(20),
        idle_timer=seconds_to_slots(5.0),
        wakeup_time=GENERIC_LITERATURE_BASELINE.wakeup_time,
        initial_energy=GENERIC_LITERATURE_BASELINE.initial_energy_mwh,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=max_slots,
        seed=None,
    )

    results: Dict[str, Any] = {}
    q_vals = list(np.linspace(0.01, 0.4, 12 if quick_mode else 20))

    # 3.1a – Maximise lifetime
    print("\n" + "=" * 80)
    print("TASK 3.1a: GRID SEARCH q – MAXIMISE LIFETIME")
    print("=" * 80)
    results["opt_q_lifetime"] = ParameterOptimizer.grid_search_q(
        base_config, q_vals, "lifetime", n_reps, verbose=True
    )

    # 3.1b – Minimise delay
    print("\n" + "=" * 80)
    print("TASK 3.1b: GRID SEARCH q – MINIMISE DELAY")
    print("=" * 80)
    results["opt_q_delay"] = ParameterOptimizer.grid_search_q(
        base_config, q_vals, "delay", n_reps, verbose=True
    )

    # 3.1c – Pareto tradeoff curve
    print("\n" + "=" * 80)
    print("TASK 3.1c: TRADEOFF ANALYSIS (max L vs min T for varying ts)")
    print("=" * 80)
    ts_vals = [1, 10, 50] if quick_mode else [1, 5, 10, 20, 50]
    q_sw = list(np.linspace(0.01, 0.4, 10 if quick_mode else 15))
    results["tradeoff_points"] = ParameterOptimizer.tradeoff_analysis(
        base_config, q_sw, ts_vals, n_reps, verbose=True
    )

    # 3.1d – 2-D grid
    print("\n" + "=" * 80)
    print("TASK 3.1d: 2-D GRID SEARCH (q, ts)")
    print("=" * 80)
    q_2d = [0.01, 0.05, 0.1, 0.2, 0.3] if quick_mode else [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    ts_2d = [1, 10, 50] if quick_mode else [1, 5, 10, 20, 50]
    results["grid_results"] = ParameterOptimizer.grid_search_q_ts(
        base_config, q_2d, ts_2d, n_reps, verbose=True
    )

    # 3.2a – Prioritization comparison
    print("\n" + "=" * 80)
    print("TASK 3.2a: PRIORITIZATION SCENARIO COMPARISON")
    print("=" * 80)
    comparison = PrioritizationAnalyzer.run_scenario_comparison(
        n_nodes=20, arrival_rate=GENERIC_LITERATURE_BASELINE.arrival_rate,
        max_slots=max_slots,
        n_replications=n_reps * 2,
        verbose=True,
    )
    results["prioritization_comparison"] = comparison
    PrioritizationAnalyzer.print_comparison_summary(comparison)

    # 3.2b – Duty-cycle comparison
    print("\n" + "=" * 80)
    print("TASK 3.2b: ON-DEMAND vs DUTY CYCLING COMPARISON")
    print("=" * 80)
    ts_dc = [1, 10, 50] if quick_mode else [1, 5, 10, 20, 50]
    fracs = [0.1, 0.3, 0.7] if quick_mode else [0.1, 0.2, 0.3, 0.5, 0.7]
    results["duty_cycle_comparison"] = DutyCycleSimulator.compare_with_on_demand(
        base_config, ts_dc, fracs, cycle_period=20, n_replications=n_reps, verbose=True
    )

    print("\n" + "=" * 80)
    print("O3 EXPERIMENTS COMPLETE!")
    print("=" * 80)

    return results
