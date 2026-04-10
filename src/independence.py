"""
Independence Analysis Module for q and t_s Parameters (Objective O5)

Determines whether the transmission probability q and idle timer t_s are
independent parameters with respect to system performance (mean delay T,
lifetime L). The validation baseline for μ lives in `src.metrics`, while this
module uses κ = p*ts as a compact coupling score for the q/ts design space.

This module provides:
- Full factorial sweep over (q, ts) space
- Analytical derivation of the coupling strength kappa = p * ts
- Regression with interaction term (F-test for independence)
- Optimal q* shift analysis per ts
- Six publication-quality visualisation methods

Date: April 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .simulator import Simulator, SimulationConfig, SimulationResults
from .power_model import PowerModel, PowerProfile
from .metrics import MetricsCalculator
from .optimization import _run_replications, _config_dict, _mean_finite, _std_finite
from .baselines import GENERIC_LITERATURE_BASELINE


# ---------------------------------------------------------------------------
# IndependenceAnalyzer
# ---------------------------------------------------------------------------

class IndependenceAnalyzer:
    """
    Static methods to probe whether q and t_s are independent parameters.

    The key heuristic is κ = p*ts, where p = q*(1-q)^(n-1). Because p depends
    on q, κ varies with both q and ts and provides a compact summary of the
    interaction region.
    """

    @staticmethod
    def compute_analytical_quantities(
        q_values: List[float],
        ts_values: List[int],
        tw: int,
        n: int,
        lambda_rate: float = GENERIC_LITERATURE_BASELINE.arrival_rate,
    ) -> Dict[str, Any]:
        """
        Compute p, mu, kappa analytically for every (q, ts) combination.

        Returns dict with 2-D numpy arrays (rows=ts, cols=q) for:
        p_matrix, mu_matrix, kappa_matrix, delay_matrix (analytical T).
        """
        n_q = len(q_values)
        n_ts = len(ts_values)

        p_matrix = np.zeros((n_ts, n_q))
        mu_matrix = np.zeros((n_ts, n_q))
        kappa_matrix = np.zeros((n_ts, n_q))
        delay_matrix = np.zeros((n_ts, n_q))

        for j, q in enumerate(q_values):
            p = q * (1 - q) ** (n - 1)
            for i, ts in enumerate(ts_values):
                mu = MetricsCalculator.compute_analytical_service_rate(
                    p, lambda_rate, tw, has_sleep=True
                )
                kappa = p * ts

                p_matrix[i, j] = p
                mu_matrix[i, j] = mu
                kappa_matrix[i, j] = kappa

                # Analytical mean delay (M/G/1-style from paper)
                if mu > 0 and n * q * (1 - q) ** (n - 1) > 0:
                    lam_per_node = None  # filled by caller
                delay_matrix[i, j] = mu  # placeholder; proper T needs lambda

        return {
            "q_values": list(q_values),
            "ts_values": list(ts_values),
            "tw": tw,
            "n": n,
            "lambda_rate": lambda_rate,
            "p_matrix": p_matrix,
            "mu_matrix": mu_matrix,
            "kappa_matrix": kappa_matrix,
        }

    @staticmethod
    def run_factorial_sweep(
        base_config: SimulationConfig,
        q_values: Optional[List[float]] = None,
        ts_values: Optional[List[int]] = None,
        n_replications: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full factorial sweep over (q, ts), recording metrics per cell.

        Returns a dict containing:
        - 2-D matrices (n_ts x n_q) for lifetime, delay, stds
        - A pandas DataFrame with one row per cell
        - Analytical quantities (p, mu, kappa)
        """
        if q_values is None:
            q_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        if ts_values is None:
            ts_values = [1, 2, 5, 10, 20, 50]

        n_q = len(q_values)
        n_ts = len(ts_values)
        total = n_q * n_ts
        n = base_config.n_nodes
        tw = base_config.wakeup_time
        lam = base_config.arrival_rate

        if verbose:
            print(
                f"O5 factorial sweep: {n_q} q x {n_ts} ts = {total} cells, "
                f"{n_replications} reps each ({total * n_replications} runs)."
            )

        # Analytical quantities
        analytical = IndependenceAnalyzer.compute_analytical_quantities(
            q_values, ts_values, tw, n, lam
        )

        lifetime_matrix = np.zeros((n_ts, n_q))
        delay_matrix = np.zeros((n_ts, n_q))
        lifetime_std_matrix = np.zeros((n_ts, n_q))
        delay_std_matrix = np.zeros((n_ts, n_q))

        rows: List[Dict[str, Any]] = []
        done = 0

        for i, ts in enumerate(ts_values):
            for j, q in enumerate(q_values):
                cd = _config_dict(base_config, transmission_prob=q, idle_timer=ts)
                rep_lt, rep_d = _run_replications(cd, n_replications)

                mean_lt = _mean_finite(rep_lt)
                std_lt = _std_finite(rep_lt)
                mean_d = float(np.mean(rep_d))
                std_d = float(np.std(rep_d)) if len(rep_d) > 1 else 0.0

                p_ana = analytical["p_matrix"][i, j]
                mu_ana = analytical["mu_matrix"][i, j]
                kappa = analytical["kappa_matrix"][i, j]
                stable = lam < mu_ana if mu_ana > 0 else False

                lifetime_matrix[i, j] = mean_lt if mean_lt != float("inf") else np.nan
                delay_matrix[i, j] = mean_d
                lifetime_std_matrix[i, j] = std_lt
                delay_std_matrix[i, j] = std_d

                rows.append({
                    "q": q,
                    "ts": ts,
                    "mean_delay": mean_d,
                    "std_delay": std_d,
                    "mean_lifetime": mean_lt if mean_lt != float("inf") else np.nan,
                    "std_lifetime": std_lt,
                    "p_analytical": p_ana,
                    "mu_analytical": mu_ana,
                    "kappa": kappa,
                    "stable": stable,
                })

                done += 1
                if verbose and (done % 5 == 0 or done == total):
                    lt_s = f"{mean_lt:.4f}" if mean_lt != float("inf") else "inf"
                    print(
                        f"  [{done}/{total}] ts={ts}, q={q:.4f}: "
                        f"L={lt_s}y, T={mean_d:.1f}s, kappa={kappa:.3f}"
                    )

        df = pd.DataFrame(rows)

        return {
            "q_values": list(q_values),
            "ts_values": list(ts_values),
            "lifetime_matrix": lifetime_matrix,
            "delay_matrix": delay_matrix,
            "lifetime_std_matrix": lifetime_std_matrix,
            "delay_std_matrix": delay_std_matrix,
            "analytical": analytical,
            "df": df,
        }

    @staticmethod
    def run_regression_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit additive and interaction regression models in log-log space
        for both mean_delay and mean_lifetime.

        Returns dict with coefficients, R-squared, F-statistic and p-value
        for the interaction term.  A significant interaction term means
        q and ts are NOT independent.
        """
        from scipy.stats import f as f_dist

        # Keep only stable, finite rows
        mask = (
            df["stable"]
            & (df["mean_delay"] > 0)
            & df["mean_lifetime"].notna()
            & (df["mean_lifetime"] > 0)
        )
        sdf = df[mask].copy()

        if len(sdf) < 5:
            return {
                "error": "Not enough stable data points for regression.",
                "filtered_df": sdf,
            }

        log_q = np.log(sdf["q"].values)
        log_ts = np.log(sdf["ts"].values.astype(float))
        n_obs = len(sdf)

        results: Dict[str, Any] = {}

        for metric, col in [("delay", "mean_delay"), ("lifetime", "mean_lifetime")]:
            log_y = np.log(sdf[col].values)

            # Additive model: log_y = a*log_q + b*log_ts + c
            X_add = np.column_stack([np.ones(n_obs), log_q, log_ts])
            beta_add, rss_add_arr, _, _ = np.linalg.lstsq(X_add, log_y, rcond=None)
            pred_add = X_add @ beta_add
            ss_res_add = float(np.sum((log_y - pred_add) ** 2))
            ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
            r2_add = 1 - ss_res_add / ss_tot if ss_tot > 0 else 0.0

            # Interaction model: + d*log_q*log_ts
            interaction = log_q * log_ts
            X_int = np.column_stack([np.ones(n_obs), log_q, log_ts, interaction])
            beta_int, _, _, _ = np.linalg.lstsq(X_int, log_y, rcond=None)
            pred_int = X_int @ beta_int
            ss_res_int = float(np.sum((log_y - pred_int) ** 2))
            r2_int = 1 - ss_res_int / ss_tot if ss_tot > 0 else 0.0

            residuals_add = log_y - pred_add

            # F-test for the interaction term
            df_num = 1  # one extra parameter
            df_denom = n_obs - 4  # interaction model has 4 params
            if df_denom > 0 and ss_res_int > 0:
                f_stat = ((ss_res_add - ss_res_int) / df_num) / (ss_res_int / df_denom)
                p_value = float(f_dist.sf(f_stat, df_num, df_denom))
            else:
                f_stat = 0.0
                p_value = 1.0

            results[metric] = {
                "additive_coeffs": {
                    "intercept": float(beta_add[0]),
                    "log_q": float(beta_add[1]),
                    "log_ts": float(beta_add[2]),
                },
                "interaction_coeffs": {
                    "intercept": float(beta_int[0]),
                    "log_q": float(beta_int[1]),
                    "log_ts": float(beta_int[2]),
                    "log_q_x_log_ts": float(beta_int[3]),
                },
                "R2_additive": r2_add,
                "R2_interaction": r2_int,
                "R2_improvement": r2_int - r2_add,
                "F_statistic": float(f_stat),
                "p_value": p_value,
                "n_observations": n_obs,
                "residuals_additive": residuals_add,
                "log_q": log_q,
                "log_ts": log_ts,
            }

        results["filtered_df"] = sdf
        return results

    @staticmethod
    def find_optimal_q_per_ts(
        df: pd.DataFrame,
        lifetime_constraints: Optional[List[float]] = None,
        n_nodes: int = 100,
    ) -> Dict[str, Any]:
        """
        For each ts, find q* that minimises delay subject to lifetime >= constraint.

        If q and ts were independent, q* would be the same for all ts.
        A monotone shift in q*(ts) proves coupling.
        """
        if lifetime_constraints is None:
            lifetime_constraints = [3.0, 5.0]

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()
        ts_vals = sorted(sdf["ts"].unique())

        result: Dict[str, Any] = {
            "ts_values": ts_vals,
            "n_nodes": n_nodes,
            "q_star_1_over_n": 1.0 / n_nodes,
            "constraints": {},
        }

        for L_min in lifetime_constraints:
            q_stars: List[Optional[float]] = []
            delays_at_opt: List[Optional[float]] = []
            lifetimes_at_opt: List[Optional[float]] = []

            for ts in ts_vals:
                feasible = sdf[(sdf["ts"] == ts) & (sdf["mean_lifetime"] >= L_min)]
                if len(feasible) == 0:
                    q_stars.append(None)
                    delays_at_opt.append(None)
                    lifetimes_at_opt.append(None)
                else:
                    best = feasible.loc[feasible["mean_delay"].idxmin()]
                    q_stars.append(float(best["q"]))
                    delays_at_opt.append(float(best["mean_delay"]))
                    lifetimes_at_opt.append(float(best["mean_lifetime"]))

            result["constraints"][L_min] = {
                "q_stars": q_stars,
                "delays": delays_at_opt,
                "lifetimes": lifetimes_at_opt,
            }

        return result


# ---------------------------------------------------------------------------
# IndependenceVisualizer
# ---------------------------------------------------------------------------

class IndependenceVisualizer:
    """Publication-quality plots for the q / t_s independence analysis."""

    @staticmethod
    def plot_interaction_plots(
        df: pd.DataFrame,
        title: str = "Interaction Plots: Are q and ts Independent?",
    ) -> Tuple[Any, Any]:
        """
        2x2 panel: (T vs q | stratified by ts) and (T vs ts | stratified by q),
        same for lifetime.  Parallel curves => independent; fanning => coupled.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()
        ts_vals = sorted(sdf["ts"].unique())
        q_vals = sorted(sdf["q"].unique())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ts_colors = cm.viridis(np.linspace(0.15, 0.85, len(ts_vals)))
        q_colors = cm.plasma(np.linspace(0.15, 0.85, len(q_vals)))

        # Panel A: delay vs q, stratified by ts
        ax = axes[0, 0]
        for idx, ts in enumerate(ts_vals):
            sub = sdf[sdf["ts"] == ts].sort_values("q")
            if len(sub) > 0:
                ax.errorbar(
                    sub["q"], sub["mean_delay"], yerr=sub["std_delay"],
                    marker="o", color=ts_colors[idx], label=f"ts={ts}",
                    capsize=3, linewidth=1.5, markersize=5,
                )
        ax.set_xlabel("q")
        ax.set_ylabel("Mean Delay (slots)")
        ax.set_title("A: Delay vs q (stratified by ts)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel B: lifetime vs q, stratified by ts
        ax = axes[0, 1]
        for idx, ts in enumerate(ts_vals):
            sub = sdf[sdf["ts"] == ts].sort_values("q")
            if len(sub) > 0:
                ax.errorbar(
                    sub["q"], sub["mean_lifetime"], yerr=sub["std_lifetime"],
                    marker="o", color=ts_colors[idx], label=f"ts={ts}",
                    capsize=3, linewidth=1.5, markersize=5,
                )
        ax.set_xlabel("q")
        ax.set_ylabel("Mean Lifetime (years)")
        ax.set_title("B: Lifetime vs q (stratified by ts)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel C: delay vs ts, stratified by q
        ax = axes[1, 0]
        for idx, q in enumerate(q_vals):
            sub = sdf[sdf["q"] == q].sort_values("ts")
            if len(sub) > 0:
                ax.errorbar(
                    sub["ts"], sub["mean_delay"], yerr=sub["std_delay"],
                    marker="s", color=q_colors[idx], label=f"q={q}",
                    capsize=3, linewidth=1.5, markersize=5,
                )
        ax.set_xlabel("ts")
        ax.set_ylabel("Mean Delay (slots)")
        ax.set_title("C: Delay vs ts (stratified by q)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel D: lifetime vs ts, stratified by q
        ax = axes[1, 1]
        for idx, q in enumerate(q_vals):
            sub = sdf[sdf["q"] == q].sort_values("ts")
            if len(sub) > 0:
                ax.errorbar(
                    sub["ts"], sub["mean_lifetime"], yerr=sub["std_lifetime"],
                    marker="s", color=q_colors[idx], label=f"q={q}",
                    capsize=3, linewidth=1.5, markersize=5,
                )
        ax.set_xlabel("ts")
        ax.set_ylabel("Mean Lifetime (years)")
        ax.set_title("D: Lifetime vs ts (stratified by q)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_coupling_heatmap(
        analytical: Dict[str, Any],
        title: str = "Coupling Strength kappa = p * ts",
    ) -> Tuple[Any, Any]:
        """
        Heatmap of kappa over the (q, ts) grid with threshold contours.
        """
        import matplotlib.pyplot as plt

        q_vals = analytical["q_values"]
        ts_vals = analytical["ts_values"]
        kappa = analytical["kappa_matrix"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            kappa, aspect="auto", origin="lower", cmap="RdYlGn_r",
            extent=[0, len(q_vals) - 1, 0, len(ts_vals) - 1],
        )
        ax.set_xticks(range(len(q_vals)))
        ax.set_xticklabels([f"{q:.3f}" for q in q_vals], rotation=45)
        ax.set_yticks(range(len(ts_vals)))
        ax.set_yticklabels([str(ts) for ts in ts_vals])
        ax.set_xlabel("Transmission Probability q")
        ax.set_ylabel("Idle Timer ts")

        cbar = plt.colorbar(im, ax=ax, label="kappa = p * ts")

        # Annotate cells
        for i in range(len(ts_vals)):
            for j in range(len(q_vals)):
                val = kappa[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_regime_map(
        df: pd.DataFrame,
        analytical: Dict[str, Any],
        title: str = "Regime Map: Independence vs Coupling",
    ) -> Tuple[Any, Any]:
        """
        (q, ts) plane colour-coded by kappa, with labelled regime boundaries
        and stability boundary.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.colors import BoundaryNorm
        from matplotlib.patches import Patch

        q_vals = analytical["q_values"]
        ts_vals = analytical["ts_values"]
        kappa = analytical["kappa_matrix"]
        mu = analytical["mu_matrix"]

        fig, ax = plt.subplots(figsize=(10, 7))

        boundaries = [0, 0.1, 1.0, kappa.max() + 0.1]
        cmap = colormaps["RdYlGn_r"].resampled(len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap.N)

        im = ax.imshow(
            kappa, aspect="auto", origin="lower", cmap=cmap, norm=norm,
            extent=[0, len(q_vals) - 1, 0, len(ts_vals) - 1],
        )
        ax.set_xticks(range(len(q_vals)))
        ax.set_xticklabels([f"{q:.3f}" for q in q_vals], rotation=45)
        ax.set_yticks(range(len(ts_vals)))
        ax.set_yticklabels([str(ts) for ts in ts_vals])
        ax.set_xlabel("Transmission Probability q")
        ax.set_ylabel("Idle Timer ts")

        plt.colorbar(im, ax=ax, label="kappa = p * ts")

        legend_elements = [
            Patch(facecolor=cmap(0), label="Near-independent (kappa < 0.1)"),
            Patch(facecolor=cmap(1), label="Moderate coupling (0.1 <= kappa < 1)"),
            Patch(facecolor=cmap(2), label="Strongly coupled (kappa >= 1)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        # Mark unstable cells
        lam = df["q"].iloc[0] if "arrival_rate" not in df.columns else 0.01
        try:
            lam = df.attrs.get("arrival_rate", 0.01)
        except Exception:
            pass
        for i in range(len(ts_vals)):
            for j in range(len(q_vals)):
                if mu[i, j] > 0 and lam >= mu[i, j]:
                    ax.plot(j, i, "rx", markersize=12, markeredgewidth=2)

        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_iso_contours(
        df: pd.DataFrame,
        title: str = "Iso-Delay and Iso-Lifetime Contours in (q, ts) Space",
    ) -> Tuple[Any, Any]:
        """
        Filled contour of lifetime + dashed delay contours.
        If q and ts were independent, contours would be axis-aligned.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()

        q_vals = sorted(sdf["q"].unique())
        ts_vals = sorted(sdf["ts"].unique())

        q_fine = np.linspace(min(q_vals), max(q_vals), 80)
        ts_fine = np.linspace(min(ts_vals), max(ts_vals), 80)
        Q, TS = np.meshgrid(q_fine, ts_fine)

        points = sdf[["q", "ts"]].values
        lt_grid = griddata(points, sdf["mean_lifetime"].values, (Q, TS), method="cubic")
        d_grid = griddata(points, sdf["mean_delay"].values, (Q, TS), method="cubic")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: lifetime contours
        ax = axes[0]
        cf = ax.contourf(Q, TS, lt_grid, levels=15, cmap="YlGn")
        plt.colorbar(cf, ax=ax, label="Lifetime (years)")
        lt_levels = [v for v in [1, 3, 5, 10] if lt_grid is not None
                     and np.nanmin(lt_grid) < v < np.nanmax(lt_grid)]
        if lt_levels:
            cs = ax.contour(Q, TS, lt_grid, levels=lt_levels, colors="black",
                            linewidths=1.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt="L=%.0f yr")
        ax.set_xlabel("q")
        ax.set_ylabel("ts")
        ax.set_title("Iso-Lifetime Contours")
        ax.annotate(
            "If independent, contours\nwould be horizontal lines",
            xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8,
            va="top", style="italic", color="dimgray",
        )

        # Right: delay contours
        ax = axes[1]
        cf = ax.contourf(Q, TS, d_grid, levels=15, cmap="YlOrRd_r")
        plt.colorbar(cf, ax=ax, label="Delay (slots)")
        d_levels = [v for v in [2, 5, 10, 20] if d_grid is not None
                    and np.nanmin(d_grid) < v < np.nanmax(d_grid)]
        if d_levels:
            cs = ax.contour(Q, TS, d_grid, levels=d_levels, colors="black",
                            linewidths=1.5, linestyles="--")
            ax.clabel(cs, inline=True, fontsize=9, fmt="T=%.0f")
        ax.set_xlabel("q")
        ax.set_ylabel("ts")
        ax.set_title("Iso-Delay Contours")
        ax.annotate(
            "If independent, contours\nwould be vertical lines",
            xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8,
            va="top", style="italic", color="dimgray",
        )

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_optimal_q_shift(
        optimal_q_data: Dict[str, Any],
        title: str = "Optimal q* Shifts with ts (Proof of Coupling)",
    ) -> Tuple[Any, Any]:
        """
        Plot q*(ts) for each lifetime constraint.
        A flat curve would mean independence; a trend confirms coupling.
        """
        import matplotlib.pyplot as plt

        ts_vals = optimal_q_data["ts_values"]
        q_baseline = optimal_q_data["q_star_1_over_n"]

        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ["o", "s", "^", "D"]
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

        for idx, (L_min, data) in enumerate(optimal_q_data["constraints"].items()):
            q_stars = data["q_stars"]
            valid_ts = [ts for ts, q in zip(ts_vals, q_stars) if q is not None]
            valid_q = [q for q in q_stars if q is not None]
            if valid_q:
                ax.plot(
                    valid_ts, valid_q,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2, markersize=8,
                    label=f"q*(ts) | L >= {L_min} yr",
                )

        ax.axhline(
            q_baseline, color="gray", linestyle="--", linewidth=1.5,
            label=f"q* = 1/n = {q_baseline:.4f} (no-sleep baseline)",
        )

        ax.set_xlabel("Idle Timer ts", fontsize=12)
        ax.set_ylabel("Optimal q*", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.annotate(
            "If q and ts were independent,\nthese curves would be flat",
            xy=(0.6, 0.9), xycoords="axes fraction", fontsize=10,
            style="italic", color="dimgray",
        )

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_regression_summary(
        regression_results: Dict[str, Any],
        title: str = "Regression Analysis: Testing q x ts Interaction",
    ) -> Tuple[Any, Any]:
        """
        Residuals from the additive model (coloured by the other variable)
        and a coefficient bar chart.
        """
        import matplotlib.pyplot as plt

        if "error" in regression_results:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, regression_results["error"],
                    transform=ax.transAxes, ha="center")
            return fig, ax

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for col_idx, metric in enumerate(["delay", "lifetime"]):
            res = regression_results[metric]
            residuals = res["residuals_additive"]
            log_q = res["log_q"]
            log_ts = res["log_ts"]

            # Residuals vs log_ts, coloured by log_q
            ax = axes[0, col_idx]
            sc = ax.scatter(log_ts, residuals, c=log_q, cmap="viridis",
                            s=40, edgecolors="gray", linewidths=0.5)
            plt.colorbar(sc, ax=ax, label="log(q)")
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel("log(ts)")
            ax.set_ylabel("Residual (additive model)")
            ax.set_title(f"{metric.title()}: residuals vs log(ts)")
            ax.grid(True, alpha=0.3)

            p_val = res["p_value"]
            f_stat = res["F_statistic"]
            ax.annotate(
                f"F = {f_stat:.2f}, p = {p_val:.4f}\n"
                f"R2 add={res['R2_additive']:.4f}, int={res['R2_interaction']:.4f}",
                xy=(0.02, 0.98), xycoords="axes fraction", fontsize=8,
                va="top", family="monospace",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.7),
            )

        # Bottom row: coefficient comparison bar charts
        for col_idx, metric in enumerate(["delay", "lifetime"]):
            ax = axes[1, col_idx]
            res = regression_results[metric]
            add_c = res["additive_coeffs"]
            int_c = res["interaction_coeffs"]

            labels = ["log_q", "log_ts", "log_q x log_ts"]
            add_vals = [add_c["log_q"], add_c["log_ts"], 0.0]
            int_vals = [int_c["log_q"], int_c["log_ts"], int_c["log_q_x_log_ts"]]

            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, add_vals, w, label="Additive", color="#2196F3", alpha=0.7)
            ax.bar(x + w / 2, int_vals, w, label="Interaction", color="#F44336", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Coefficient")
            ax.set_title(f"{metric.title()}: regression coefficients")
            ax.legend(fontsize=9)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_kappa_vs_outputs(
        df: pd.DataFrame,
        title: str = "Coupling Strength kappa Predicts Both Delay and Lifetime",
    ) -> Tuple[Any, Any]:
        """
        Scatter plots of kappa (= p * ts) vs mean_delay and mean_lifetime.

        A strong correlation confirms kappa is the single quantity that captures
        the joint effect of q and ts.  Points are coloured by q so the reader
        can see how the two parameters combine into kappa.
        """
        import matplotlib.pyplot as plt

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Colour by q (log scale for visibility)
        q_log = np.log10(sdf["q"].values)
        q_norm = (q_log - q_log.min()) / (q_log.max() - q_log.min() + 1e-12)

        cmap = plt.cm.plasma  # noqa: E501

        # Left: kappa vs delay
        ax = axes[0]
        sc = ax.scatter(
            sdf["kappa"], sdf["mean_delay"],
            c=q_norm, cmap=cmap, s=60, edgecolors="gray", linewidths=0.4,
        )
        ax.set_xlabel("kappa = p · ts", fontsize=12)
        ax.set_ylabel("Mean Delay (slots)", fontsize=12)
        ax.set_title("kappa vs Delay")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.axvline(0.1, color="green", linestyle="--", linewidth=1, alpha=0.7,
                   label="kappa = 0.1 (near-independent)")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="kappa = 1 (strongly coupled)")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="log10(q)  [low → high]")

        # Right: kappa vs lifetime
        ax = axes[1]
        sc = ax.scatter(
            sdf["kappa"], sdf["mean_lifetime"],
            c=q_norm, cmap=cmap, s=60, edgecolors="gray", linewidths=0.4,
        )
        ax.set_xlabel("kappa = p · ts", fontsize=12)
        ax.set_ylabel("Mean Lifetime (years)", fontsize=12)
        ax.set_title("kappa vs Lifetime")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.axvline(0.1, color="green", linestyle="--", linewidth=1, alpha=0.7,
                   label="kappa = 0.1")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="kappa = 1")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="log10(q)  [low → high]")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_pareto_surface(
        df: pd.DataFrame,
        title: str = "Lifetime–Delay Tradeoff Surface over (q, ts) Grid",
    ) -> Tuple[Any, Any]:
        """
        2-D heatmap of lifetime over the (q, ts) grid with delay iso-contour
        lines and Pareto-dominant cells marked.  Shows how jointly varying q
        and ts traces out the full achievable frontier.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()

        q_vals = sorted(sdf["q"].unique())
        ts_vals = sorted(sdf["ts"].unique())

        # Build dense grid for smooth rendering
        q_fine = np.linspace(min(q_vals), max(q_vals), 100)
        ts_fine = np.linspace(min(ts_vals), max(ts_vals), 100)
        Q, TS = np.meshgrid(q_fine, ts_fine)
        points = sdf[["q", "ts"]].values

        lt_grid = griddata(points, sdf["mean_lifetime"].values, (Q, TS), method="cubic")
        d_grid = griddata(points, sdf["mean_delay"].values, (Q, TS), method="cubic")

        fig, ax = plt.subplots(figsize=(11, 7))
        cf = ax.contourf(Q, TS, lt_grid, levels=20, cmap="YlGn")
        plt.colorbar(cf, ax=ax, label="Mean Lifetime (years)")

        # Delay iso-contours
        d_levels = [v for v in [2, 5, 10, 20, 50]
                    if d_grid is not None
                    and np.nanmin(d_grid) < v < np.nanmax(d_grid)]
        if d_levels:
            cs = ax.contour(Q, TS, d_grid, levels=d_levels,
                            colors="black", linewidths=1.2, linestyles="--")
            ax.clabel(cs, inline=True, fontsize=9, fmt="T=%.0f slots")

        # Mark Pareto-dominant cells (min delay for each lifetime bucket)
        lifetime_buckets = np.percentile(
            sdf["mean_lifetime"].dropna(), [20, 40, 60, 80, 95]
        )
        pareto_q, pareto_ts = [], []
        for L_thresh in lifetime_buckets:
            feasible = sdf[sdf["mean_lifetime"] >= L_thresh]
            if len(feasible):
                best = feasible.loc[feasible["mean_delay"].idxmin()]
                pareto_q.append(best["q"])
                pareto_ts.append(best["ts"])

        if pareto_q:
            ax.scatter(pareto_q, pareto_ts, marker="*", s=220,
                       color="white", edgecolors="black", linewidths=1,
                       zorder=10, label="Pareto-optimal points")
            # Connect with a dashed line
            idx_sorted = np.argsort(pareto_q)
            ax.plot(
                np.array(pareto_q)[idx_sorted],
                np.array(pareto_ts)[idx_sorted],
                "w--", linewidth=1.5, zorder=9,
            )
            ax.legend(fontsize=10, loc="upper right")

        ax.set_xlabel("Transmission Probability q", fontsize=12)
        ax.set_ylabel("Idle Timer ts", fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_delay_lifetime_tradeoff_by_kappa(
        df: pd.DataFrame,
        title: str = "Delay–Lifetime Tradeoff Coloured by Coupling Strength kappa",
    ) -> Tuple[Any, Any]:
        """
        ¯T vs ¯L scatter coloured by kappa, with marker shapes indicating ts.

        This is the clearest single-plot summary: it shows the achievable
        tradeoff frontier and immediately reveals that operating points cluster
        by kappa regardless of the individual q / ts values chosen.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy()

        fig, ax = plt.subplots(figsize=(11, 7))

        ts_vals = sorted(sdf["ts"].unique())
        markers = ["o", "s", "^", "D", "v", "P", "X", "h"]

        kappa_vals = sdf["kappa"].values
        k_norm = (np.log10(kappa_vals + 1e-9) - np.log10(kappa_vals + 1e-9).min())
        k_range = k_norm.max()
        k_norm = k_norm / k_range if k_range > 0 else k_norm

        cmap = cm.RdYlGn_r
        for idx, ts in enumerate(ts_vals):
            sub = sdf[sdf["ts"] == ts]
            k_sub = (
                np.log10(sub["kappa"].values + 1e-9)
                - np.log10(kappa_vals + 1e-9).min()
            ) / k_range if k_range > 0 else np.zeros(len(sub))

            sc = ax.scatter(
                sub["mean_delay"], sub["mean_lifetime"],
                c=k_sub, cmap=cmap, vmin=0, vmax=1,
                marker=markers[idx % len(markers)], s=90,
                edgecolors="gray", linewidths=0.5,
                label=f"ts={ts}",
            )

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Normalised log(kappa)  [green=low, red=high]")

        ax.set_xlabel("Mean Delay (slots)", fontsize=12)
        ax.set_ylabel("Mean Lifetime (years)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=9, bbox_to_anchor=(1.14, 1), loc="upper left",
                  title="Marker = ts")
        ax.grid(True, alpha=0.3)

        ax.annotate(
            "Bottom-right = good (low delay, long life)\n"
            "Green points: low kappa (near-independent regime)",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=9, va="top", style="italic", color="dimgray",
        )

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_marginal_effects(
        df: pd.DataFrame,
        analytical: Dict[str, Any],
        title: str = "Marginal Effect of q vs ts at Different Operating Points",
    ) -> Tuple[Any, Any]:
        """
        For a representative set of (q, ts) cells, show how much ¯T and ¯L
        change (in %) when each parameter is increased by one step in the grid.
        Side-by-side bars make it easy to see which lever is stronger and how
        the balance shifts across the parameter space.
        """
        import matplotlib.pyplot as plt

        mask = df["stable"] & df["mean_lifetime"].notna() & (df["mean_lifetime"] > 0)
        sdf = df[mask].copy().reset_index(drop=True)

        q_vals = sorted(sdf["q"].unique())
        ts_vals = sorted(sdf["ts"].unique())

        # For each interior cell compute finite-difference marginal effects
        records = []
        for i, ts in enumerate(ts_vals[:-1]):
            for j, q in enumerate(q_vals[:-1]):
                base = sdf[(sdf["q"] == q) & (sdf["ts"] == ts)]
                dq = sdf[(sdf["q"] == q_vals[j + 1]) & (sdf["ts"] == ts)]
                dts = sdf[(sdf["q"] == q) & (sdf["ts"] == ts_vals[i + 1])]
                if base.empty or dq.empty or dts.empty:
                    continue

                T0 = float(base["mean_delay"].iloc[0])
                L0 = float(base["mean_lifetime"].iloc[0])
                if T0 == 0 or L0 == 0:
                    continue

                dT_dq_pct = (float(dq["mean_delay"].iloc[0]) - T0) / T0 * 100
                dT_dts_pct = (float(dts["mean_delay"].iloc[0]) - T0) / T0 * 100
                dL_dq_pct = (float(dq["mean_lifetime"].iloc[0]) - L0) / L0 * 100
                dL_dts_pct = (float(dts["mean_lifetime"].iloc[0]) - L0) / L0 * 100

                records.append({
                    "label": f"q={q}\nts={ts}",
                    "dT_dq": dT_dq_pct,
                    "dT_dts": dT_dts_pct,
                    "dL_dq": dL_dq_pct,
                    "dL_dts": dL_dts_pct,
                    "kappa": float(base["kappa"].iloc[0]),
                })

        if not records:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data for marginal effects.",
                    transform=ax.transAxes, ha="center")
            return fig, ax

        # Pick up to 6 representative cells spanning the kappa range
        records_sorted = sorted(records, key=lambda r: r["kappa"])
        step = max(1, len(records_sorted) // 6)
        selected = records_sorted[::step][:6]

        labels = [r["label"] for r in selected]
        x = np.arange(len(labels))
        w = 0.2

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

        for col_idx, (metric_key_q, metric_key_ts, ylabel, metric_label) in enumerate([
            ("dT_dq", "dT_dts", "% Change in Delay", "Mean delay"),
            ("dL_dq", "dL_dts", "% Change in Lifetime", "Mean lifetime"),
        ]):
            ax = axes[col_idx]
            q_vals_bar = [r[metric_key_q] for r in selected]
            ts_vals_bar = [r[metric_key_ts] for r in selected]

            c_q = ["#4CAF50" if v < 0 else "#F44336" for v in q_vals_bar]
            c_ts = ["#2196F3" if (
                (metric_key_ts == "dT_dts" and v < 0) or
                (metric_key_ts == "dL_dts" and v > 0)
            ) else "#FF9800" for v in ts_vals_bar]

            ax.bar(x - w / 2, q_vals_bar, w, label="Step in q →", color=c_q, alpha=0.8)
            ax.bar(x + w / 2, ts_vals_bar, w, label="Step in ts →", color=c_ts, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Marginal effects on {metric_label}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Annotate kappa for context
            for xi, r in enumerate(selected):
                ax.text(xi, ax.get_ylim()[0] if ax.get_ylim()[0] < -2 else -2,
                        f"κ={r['kappa']:.2f}", ha="center",
                        fontsize=7, color="gray", va="top")

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        return fig, axes


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_o5_experiments(
    output_dir: str = "results",
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run all O5 independence analysis experiments end-to-end.

    Parameters
    ----------
    output_dir : Directory for saving CSV and figures.
    quick_mode : Fewer replications / smaller grid for fast execution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_reps = 10 if quick_mode else 20
    max_slots = 30_000 if quick_mode else 50_000
    n_nodes = 100

    print("=" * 80)
    print("O5 INDEPENDENCE ANALYSIS EXPERIMENTS")
    print("=" * 80)
    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Replications per cell: {n_reps}")

    base_config = SimulationConfig(
        n_nodes=n_nodes,
        arrival_rate=0.01,
        transmission_prob=0.05,
        idle_timer=10,
        wakeup_time=2,
        initial_energy=5000,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=max_slots,
        seed=None,
    )

    q_vals = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    ts_vals = [1, 2, 5, 10, 20, 50]
    if quick_mode:
        q_vals = [0.01, 0.02, 0.05, 0.1]
        ts_vals = [1, 5, 10, 50]

    results: Dict[str, Any] = {}

    # 1. Analytical decomposition
    print("\n" + "=" * 80)
    print("STEP 1: ANALYTICAL DECOMPOSITION")
    print("=" * 80)
    analytical = IndependenceAnalyzer.compute_analytical_quantities(
        q_vals,
        ts_vals,
        tw=base_config.wakeup_time,
        n=n_nodes,
        lambda_rate=base_config.arrival_rate,
    )
    results["analytical"] = analytical

    # 2. Factorial sweep
    print("\n" + "=" * 80)
    print("STEP 2: FACTORIAL SWEEP")
    print("=" * 80)
    sweep = IndependenceAnalyzer.run_factorial_sweep(
        base_config, q_vals, ts_vals, n_reps, verbose=True,
    )
    results["sweep"] = sweep
    df = sweep["df"]

    # Save CSV
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "o5_factorial_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep data to {csv_path}")

    # 3. Regression analysis
    print("\n" + "=" * 80)
    print("STEP 3: REGRESSION ANALYSIS (F-TEST)")
    print("=" * 80)
    regression = IndependenceAnalyzer.run_regression_analysis(df)
    results["regression"] = regression
    if "error" not in regression:
        for metric in ["delay", "lifetime"]:
            r = regression[metric]
            sig = "YES (coupled)" if r["p_value"] < 0.05 else "NO (independent)"
            print(
                f"  {metric}: F={r['F_statistic']:.2f}, p={r['p_value']:.4f} "
                f"-> interaction significant? {sig}"
            )
            print(
                f"    R2 additive={r['R2_additive']:.4f}, "
                f"interaction={r['R2_interaction']:.4f} "
                f"(improvement={r['R2_improvement']:.4f})"
            )

    # 4. Optimal q* shift
    print("\n" + "=" * 80)
    print("STEP 4: OPTIMAL q* SHIFT")
    print("=" * 80)
    optimal_q = IndependenceAnalyzer.find_optimal_q_per_ts(
        df, lifetime_constraints=[3.0, 5.0], n_nodes=n_nodes,
    )
    results["optimal_q"] = optimal_q
    for L_min, data in optimal_q["constraints"].items():
        print(f"  L >= {L_min} yr: q* = {data['q_stars']}")

    # 5. Generate figures
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING FIGURES")
    print("=" * 80)
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = IndependenceVisualizer.plot_coupling_heatmap(analytical)
    fig.savefig(fig_dir / "o5_coupling_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_coupling_heatmap.png")

    fig, _ = IndependenceVisualizer.plot_interaction_plots(df)
    fig.savefig(fig_dir / "o5_interaction_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_interaction_plots.png")

    if "error" not in regression:
        fig, _ = IndependenceVisualizer.plot_regression_summary(regression)
        fig.savefig(fig_dir / "o5_residual_interaction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved o5_residual_interaction.png")

    fig, _ = IndependenceVisualizer.plot_regime_map(df, analytical)
    fig.savefig(fig_dir / "o5_regime_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_regime_map.png")

    fig, _ = IndependenceVisualizer.plot_iso_contours(df)
    fig.savefig(fig_dir / "o5_iso_contours.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_iso_contours.png")

    fig, _ = IndependenceVisualizer.plot_optimal_q_shift(optimal_q)
    fig.savefig(fig_dir / "o5_optimal_q_vs_ts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_optimal_q_vs_ts.png")

    fig, _ = IndependenceVisualizer.plot_kappa_vs_outputs(df)
    fig.savefig(fig_dir / "o5_kappa_vs_outputs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_kappa_vs_outputs.png")

    fig, _ = IndependenceVisualizer.plot_pareto_surface(df)
    fig.savefig(fig_dir / "o5_pareto_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_pareto_surface.png")

    fig, _ = IndependenceVisualizer.plot_delay_lifetime_tradeoff_by_kappa(df)
    fig.savefig(fig_dir / "o5_tradeoff_by_kappa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_tradeoff_by_kappa.png")

    fig, _ = IndependenceVisualizer.plot_marginal_effects(df, analytical)
    fig.savefig(fig_dir / "o5_marginal_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved o5_marginal_effects.png")

    print("\n" + "=" * 80)
    print("O5 EXPERIMENTS COMPLETE!")
    print("=" * 80)

    return results
