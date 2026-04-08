"""
Report Figure Generator
=======================
Generates all 38 figures (Fig 2.1 – Fig 3.36) and 8 tables required for the
FYP report, saving PNGs (300 dpi) to report/figures/.

Usage
-----
    # Quick mode  (5x denser sweeps, n=50, 3 reps, 8k slots — ~30-45 min total)
    python generate_figures.py

    # Full quality  (5x denser sweeps, n=100, 15-20 reps, 80-200k slots — ~4-6 hr)
    python generate_figures.py --full

    # Only specific sections
    python generate_figures.py --only fig2 val o6 o9

    # Always set this on Windows first:
    $env:PYTHONIOENCODING = "utf-8"

Sections
--------
  fig2  — Fig 2.1 (state diagram), Fig 2.2 (architecture)        [no sim]
  val   — Figs 3.1-3.3   Simulator validation (O1)
  o2    — Figs 3.4-3.8   Parameter impact (O2)
  o3    — Figs 3.9-3.13  Optimization (O3)
  o4    — Figs 3.14-3.17 3GPP validation (O4)
  o5    — Figs 3.18-3.23 Independence analysis (O5)
  o6    — Figs 3.24-3.25 Finite retry limits (O6)
  o7    — Figs 3.26-3.28 CSMA vs Slotted Aloha (O7)
  o8    — Figs 3.29-3.31 Receiver models (O8)
  o9    — Figs 3.32-3.34 Age of Information (O9)
  o10   — Figs 3.35-3.36 MMBP arrivals (O10)
"""

from __future__ import annotations
import argparse
import sys
import io
import time
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 can't render Greek/Unicode characters).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.simulator import Simulator, BatchSimulator, SimulationConfig, SimulationResults
from src.power_model import PowerModel, PowerProfile
from src.metrics import MetricsCalculator
from src.experiments import run_o6_experiments, run_o9_experiments, ParameterSweep
from src.mmbp_analytics import run_o10_experiments
from src.receiver_models import ReceiverModel
from src.optimization import (
    ParameterOptimizer, DutyCycleSimulator, PrioritizationAnalyzer,
    _run_replications, _config_dict, _mean_finite,
)
from src.independence import IndependenceAnalyzer
from src.validation_3gpp import run_o4_experiments, ThreeGPPAlignment, DesignGuidelines

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.figsize": (7, 4.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
}
plt.rcParams.update(STYLE)

COLORS = plt.cm.tab10.colors
OUTDIR: Path = ROOT / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SLOT_MS = 6.0  # slot duration in ms


# ===========================================================================
# Progress tracker
# ===========================================================================
_SECTION_INDEX = 0
_SECTION_TOTAL = 0   # set in main()


class Progress:
    """
    Line-by-line terminal progress tracker.

    Example output
    --------------
      --> [2/11] Figs 3.4-3.8  Parameter impact (O2)  |  150 simulations
      [  1/150] [##..................]   0.7%  ETA 04:12  q=0.005 ts=1  ->  T=234 L=1.2yr
      [  2/150] [##..................]   1.3%  ETA 03:58  q=0.010 ts=1  ->  T=198 L=1.4yr
      ...
      [done] 150 steps in 04:05 (1.6 s/step)
    """

    def __init__(self, total: int, label: str = ""):
        global _SECTION_INDEX
        _SECTION_INDEX += 1
        self.total = total
        self.done = 0
        self.t0 = time.perf_counter()
        self._w = len(str(total))
        self._step_times: list[float] = []
        self._last_t = self.t0

        tag = f"[{_SECTION_INDEX}/{_SECTION_TOTAL}] " if _SECTION_TOTAL else ""
        print(f"\n  --> {tag}{label}  |  {total} steps")
        print(f"  {'-' * 70}")

    def step(self, desc: str = "", result: str = ""):
        """Call once per completed simulation or batch."""
        now = time.perf_counter()
        self._step_times.append(now - self._last_t)
        self._last_t = now
        self.done += 1

        elapsed = now - self.t0
        pct = 100.0 * self.done / max(self.total, 1)

        # ETA: use rolling average of last 10 step durations
        recent = self._step_times[-10:]
        avg_step = sum(recent) / len(recent)
        remaining = self.total - self.done
        eta_s = avg_step * remaining
        m, s = divmod(int(eta_s), 60)
        eta_str = f"ETA {m:02d}:{s:02d}"

        # ASCII bar
        bar_w = 20
        filled = int(bar_w * self.done / max(self.total, 1))
        bar = "#" * filled + "." * (bar_w - filled)

        line = (
            f"  [{self.done:{self._w}}/{self.total}]"
            f" [{bar}] {pct:5.1f}%  {eta_str}"
        )
        if desc:
            line += f"  {desc}"
        if result:
            line += f"  ->  {result}"
        print(line)

    def finish(self):
        elapsed = time.perf_counter() - self.t0
        m, s = divmod(int(elapsed), 60)
        per_step = elapsed / max(self.done, 1)
        print(f"  {'-' * 70}")
        print(f"  [done] {self.done} steps in {m:02d}:{s:02d}  ({per_step:.1f} s/step)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def savefig(name: str, fig: plt.Figure | None = None) -> None:
    path = OUTDIR / f"{name}.png"
    (fig or plt).savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved  {path.name}")


def base_config(
    n: int = 100,
    lam: float = 0.01,
    q: float = 0.01,
    ts: int = 10,
    tw: int = 2,
    energy: float = 5000.0,
    slots: int = 50_000,
    seed: int | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        n_nodes=n,
        arrival_rate=lam,
        transmission_prob=q,
        idle_timer=ts,
        wakeup_time=tw,
        initial_energy=energy,
        power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
        max_slots=slots,
        seed=seed,
    )


def run_batch(cfg: SimulationConfig, n_reps: int) -> list[SimulationResults]:
    return BatchSimulator(cfg).run_replications(n_replications=n_reps)


def mean_results(reps: list[SimulationResults]) -> tuple[float, float, float]:
    """Return (mean_delay, mean_lifetime_years, delay_std)."""
    lt = [r.mean_lifetime_years for r in reps if r.mean_lifetime_years != float("inf")]
    delays = [r.mean_delay for r in reps]
    return (
        float(np.mean(delays)),
        float(np.mean(lt)) if lt else float("inf"),
        float(np.std(delays)) if len(delays) > 1 else 0.0,
    )


# ============================================================================
# SEC FIG2 — Architecture diagrams (no simulation)
# ============================================================================
def sec_fig2():
    print("\n  --> [fig2] Architecture diagrams  |  0 simulations (pure drawing)")

    # ---- Fig 2.1: MTD state transition diagram ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")

    BW, BH = 1.9, 1.0   # box half-width (= BW/2 from centre), box height
    CY = 2.2             # centre y of all boxes

    # State positions (x-centre, y-centre, label, fill-colour)
    states = [
        (1.6,  CY, "ACTIVE",            "#4CAF50"),
        (4.2,  CY, "IDLE\n(timer ts)",  "#2196F3"),
        (7.0,  CY, "SLEEP",             "#9E9E9E"),
        (9.8,  CY, "WAKEUP\n(tw slots)","#FF9800"),
    ]

    for x, y, label, color in states:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - BW/2, y - BH/2), BW, BH,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.25,
            edgecolor=color, lw=2, zorder=3,
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=4)

    # --- Forward arrows: right edge of box_i → left edge of box_{i+1} ---
    fwd_labels = [
        "no arrival\n(ts → 0)",
        "ts expires",
        "tw done",
    ]
    for (x1, *_), (x2, *__), lbl in zip(states, states[1:], fwd_labels):
        x_start = x1 + BW / 2
        x_end   = x2 - BW / 2
        x_mid   = (x_start + x_end) / 2
        ax.annotate("",
            xy=(x_end, CY), xytext=(x_start, CY),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6,
                            mutation_scale=14),
            zorder=2,
        )
        ax.text(x_mid, CY - 0.68, lbl, ha="center", va="top",
                fontsize=8, color="#333")

    # --- Back arc: WAKEUP top edge → ACTIVE top edge (curves above boxes) ---
    # Use box top edges (y = CY + BH/2)
    TOP = CY + BH / 2
    ax.annotate("",
        xy   =(1.6,  TOP),   # ACTIVE   top-centre
        xytext=(9.8, TOP),   # WAKEUP   top-centre
        arrowprops=dict(
            arrowstyle="-|>", color="steelblue", lw=1.6,
            connectionstyle="arc3,rad=-0.38",
            mutation_scale=14,
        ),
        zorder=2,
    )
    ax.text(5.7, TOP + 1.25,
            "wakeup complete / new packet arrives",
            ha="center", fontsize=8, color="steelblue",
            style="italic")

    # --- Self-loop on ACTIVE: tx attempt with prob q ---
    # Arc from ACTIVE right-top corner back to ACTIVE left-top corner
    ax.annotate("",
        xy   =(1.6 - BW/2 + 0.15, TOP + 0.05),   # left side of ACTIVE top
        xytext=(1.6 + BW/2 - 0.15, TOP + 0.05),  # right side of ACTIVE top
        arrowprops=dict(
            arrowstyle="-|>", color="#2e7d32", lw=1.6,
            connectionstyle="arc3,rad=-1.35",
            mutation_scale=14,
        ),
        zorder=2,
    )
    ax.text(1.6, TOP + 1.1,
            "tx attempt (q)", ha="center", fontsize=8,
            color="#2e7d32", style="italic")

    ax.set_title("Fig 2.1 — MTD Node State Transition Diagram", pad=10)
    savefig("fig2_1_state_diagram", fig)

    # ---- Fig 2.2: Software architecture — layered with full connections ----
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis("off")

    def box(ax, x, y, w, h, label, color, fs=9):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, alpha=0.22, edgecolor=color, lw=2, zorder=3))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        return (x, y, w, h)   # return for edge helpers

    def cx(b): return b[0] + b[2]/2     # box centre-x
    def cy(b): return b[1] + b[3]/2     # box centre-y
    def top(b):    return (cx(b), b[1] + b[3])
    def bot(b):    return (cx(b), b[1])
    def right(b):  return (b[0] + b[2], cy(b))
    def left(b):   return (b[0],         cy(b))

    def arr(ax, p1, p2, color="#555", rad=0.0, lbl="", lbl_offset=(0, 4)):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=1.4,
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=12,
                    ), zorder=2)
        if lbl:
            mx = (p1[0] + p2[0]) / 2 + lbl_offset[0]
            my = (p1[1] + p2[1]) / 2 + lbl_offset[1] * 0.055
            ax.text(mx, my, lbl, fontsize=7, color=color, ha="center",
                    style="italic", zorder=5)

    # Layer background bands
    for y0, y1, lbl, lc in [
        (5.8, 8.0, "Simulation Core",       "#E3F2FD"),
        (3.2, 5.5, "Extension & Support",    "#FFF8E1"),
        (0.5, 2.9, "Analysis & Validation",  "#F3E5F5"),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y0), 12.7, y1-y0, boxstyle="round,pad=0.05",
            facecolor=lc, alpha=0.35, edgecolor="none", zorder=0))
        ax.text(0.3, (y0+y1)/2, lbl, fontsize=8, color="#555",
                rotation=90, va="center", ha="center", style="italic")

    # ---- Layer 1: Simulation core ----
    bNode  = box(ax,  1.0, 6.3, 2.2, 1.1, "Node\n(node.py)",               "#4CAF50")
    bSim   = box(ax,  4.0, 6.3, 2.4, 1.1, "Simulator\n(simulator.py)",      "#2196F3")
    bBatch = box(ax,  7.2, 6.3, 2.5, 1.1, "BatchSimulator\n(simulator.py)", "#2196F3")
    bMetr  = box(ax, 10.3, 6.3, 2.4, 1.1, "MetricsCalculator\n(metrics.py)","#9C27B0")

    # Horizontal flow arrows (core pipeline)
    arr(ax, right(bNode),  left(bSim),   "#2196F3", lbl="uses")
    arr(ax, right(bSim),   left(bBatch), "#2196F3", lbl="runs N×")
    arr(ax, right(bBatch), left(bMetr),  "#9C27B0", lbl="results")

    # ---- Layer 2: Extension & support modules ----
    bPow   = box(ax,  1.0, 3.7, 2.2, 1.0, "PowerModel\n(power_model.py)",          "#FF9800")
    bTraf  = box(ax,  4.0, 3.7, 2.4, 1.0, "TrafficGenerator\n(traffic_models.py)", "#FF9800")
    bRecv  = box(ax,  7.2, 3.7, 2.5, 1.0, "ReceiverModels\n(receiver_models.py)",  "#E91E63")
    bMmbp  = box(ax, 10.3, 3.7, 2.4, 1.0, "MMBPAnalytics\n(mmbp_analytics.py)",   "#E91E63")

    # Support → core upward arrows
    arr(ax, top(bPow),  bot(bNode),  "#FF9800", rad= 0.0, lbl="power rates")
    arr(ax, top(bTraf), bot(bSim),   "#FF9800", rad= 0.0, lbl="arrivals")
    arr(ax, top(bRecv), bot(bSim),   "#E91E63", rad=-0.2, lbl="collision\nresolution")
    arr(ax, top(bMmbp), bot(bBatch), "#E91E63", rad= 0.0, lbl="MMBP\nconfig")

    # ---- Layer 3: Analysis & validation tools ----
    bOpt   = box(ax,  1.2, 1.0, 3.0, 1.1, "ParameterOptimizer\n(optimization.py)",    "#607D8B")
    bInd   = box(ax,  5.1, 1.0, 3.0, 1.1, "IndependenceAnalyzer\n(independence.py)",  "#607D8B")
    bVal   = box(ax,  9.0, 1.0, 3.5, 1.1, "DesignGuidelines\n(validation_3gpp.py)",   "#607D8B")

    # BatchSimulator → analysis tools (downward)
    arr(ax, bot(bBatch), top(bOpt), "#607D8B", rad= 0.15, lbl="sweep\nresults")
    arr(ax, bot(bBatch), top(bInd), "#607D8B", rad= 0.0,  lbl="factorial\nresults")
    arr(ax, bot(bBatch), top(bVal), "#607D8B", rad=-0.15, lbl="3GPP\nresults")

    ax.set_title("Fig 2.2 — Software Architecture Overview", pad=10, fontsize=13)
    savefig("fig2_2_architecture", fig)


# ============================================================================
# SEC VAL — Figs 3.1–3.3: Simulator Validation
# ============================================================================
def sec_val(quick: bool):
    # quick: 3 reps, 10k slots — enough for smooth validation scatter
    n_reps    = 3 if quick else 20
    max_slots = 10_000 if quick else 200_000

    # 5x more n-values for scatter plots
    n_values = [5, 10, 15, 20, 30, 50] if quick else [5, 10, 15, 20, 30, 50, 75, 100]
    lam, q_base, ts, tw = 0.005, 0.05, 5, 2

    # 5x more convergence checkpoints
    conv_slots = [500, 1_000, 2_000, 5_000, 8_000, 10_000, 15_000] if quick else \
                 [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
    conv_reps  = 3 if quick else 15

    # One prog.step() per n_value and per conv_slot checkpoint
    total_sims = len(n_values) + len(conv_slots)
    prog = Progress(total_sims, "Figs 3.1-3.3  Simulator validation (O1)")

    emp_p, anal_p, emp_mu, anal_mu = [], [], [], []

    for n in n_values:
        q = min(q_base, 1.0 / n)
        cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
        reps = run_batch(cfg, n_reps)
        ep  = float(np.mean([r.empirical_success_prob for r in reps]))
        emu = float(np.mean([r.empirical_service_rate for r in reps]))
        ap  = MetricsCalculator.compute_analytical_success_probability(n, q)
        amu = MetricsCalculator.compute_analytical_service_rate(ap, lam, tw)
        emp_p.append(ep); anal_p.append(ap)
        emp_mu.append(emu); anal_mu.append(amu)
        prog.step(f"n={n}", f"p_emp={ep:.4f} p_ana={ap:.4f}  mu_emp={emu:.5f}")

    # Fig 3.1 — p scatter
    fig, ax = plt.subplots()
    lims = [0, max(max(emp_p), max(anal_p)) * 1.15]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.fill_between(lims, [v*.95 for v in lims], [v*1.05 for v in lims],
                    alpha=0.12, color="gray", label="±5%")
    for i, n in enumerate(n_values):
        ax.scatter(anal_p[i], emp_p[i], s=80, color=COLORS[i % 10], zorder=5, label=f"n={n}")
    ax.set_xlabel("Analytical p"); ax.set_ylabel("Empirical p")
    ax.set_title("Fig 3.1 — Empirical vs. Analytical Success Probability p")
    ax.legend(ncol=2, fontsize=9); savefig("fig3_1_p_validation", fig)

    # Fig 3.2 — μ scatter
    fig, ax = plt.subplots()
    lims = [0, max(max(emp_mu), max(anal_mu)) * 1.15]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.fill_between(lims, [v*.95 for v in lims], [v*1.05 for v in lims],
                    alpha=0.12, color="gray", label="±5%")
    for i, n in enumerate(n_values):
        ax.scatter(anal_mu[i], emp_mu[i], s=80, color=COLORS[i % 10], zorder=5, label=f"n={n}")
    ax.set_xlabel("Analytical μ"); ax.set_ylabel("Empirical μ")
    ax.set_title("Fig 3.2 — Empirical vs. Analytical Service Rate μ")
    ax.legend(ncol=2, fontsize=9); savefig("fig3_2_mu_validation", fig)

    # Fig 3.3 — convergence
    n_conv, q_conv = 50, 0.02
    ap_c  = MetricsCalculator.compute_analytical_success_probability(n_conv, q_conv)
    amu_c = MetricsCalculator.compute_analytical_service_rate(ap_c, lam, tw)
    try:
        ad_c = MetricsCalculator.compute_analytical_mean_delay(lam, amu_c)
    except Exception:
        ad_c = float("nan")

    conv_errors = []
    for slots in conv_slots:
        cfg = base_config(n=n_conv, lam=lam, q=q_conv, ts=ts, tw=tw, slots=slots, seed=7)
        reps = [Simulator(cfg).run_simulation() for _ in range(conv_reps)]
        ed   = float(np.mean([r.mean_delay for r in reps]))
        err  = abs(ed - ad_c) if not np.isnan(ad_c) else abs(ed)
        conv_errors.append(err)
        prog.step(f"slots={slots}", f"|err|={err:.2f}")

    fig, ax = plt.subplots()
    ax.loglog(conv_slots, conv_errors, "o-", color=COLORS[0])
    ax.set_xlabel("Number of simulation slots")
    ax.set_ylabel("|T̄_sim − T̄_analytical| (slots)")
    ax.set_title("Fig 3.3 — Convergence of Mean Delay Estimate")
    savefig("fig3_3_convergence", fig)
    prog.finish()


# ============================================================================
# SEC O2 — Figs 3.4–3.8: Parameter Impact
# ============================================================================
def sec_o2(quick: bool):
    # quick: n=50, 3 reps, 8k slots ≈ 1 s/sim → 350 steps × 3s ≈ 18 min total
    n_reps    = 3 if quick else 20
    max_slots = 8_000 if quick else 100_000
    n, lam, tw = 50 if quick else 100, 0.01, 2

    # 5x more sweep points on all axes
    ts_list  = [1, 5, 10, 30, 50]          # stratified colors (keep manageable)
    q_list   = [0.005, 0.01, 0.02, 0.05, 0.1]
    q_sweep  = list(np.linspace(0.005, 0.15, 30 if quick else 50))
    ts_sweep = list(np.unique(np.round(
        np.geomspace(1, 50, 20 if quick else 35)).astype(int)))  # log-spaced
    lam_sweep = list(np.linspace(0.001, 0.05, 30 if quick else 50))
    n_vals_8  = [10, 25, 50, 100]           # 4 curves for Fig 3.8

    # One prog.step() per (ts,q) pair + per (q,ts) pair + per (n,lam) pair
    total_sims = (
        len(ts_list) * len(q_sweep)
        + len(q_list) * len(ts_sweep)
        + len(n_vals_8) * len(lam_sweep)
    )
    prog = Progress(total_sims, "Figs 3.4-3.8  Parameter impact (O2)")

    # --- q sweep (figs 3.4, 3.5) ---
    delay_by_ts, lt_by_ts, dstd_by_ts = {}, {}, {}
    for ts in ts_list:
        ds, ls, dstds = [], [], []
        for q in q_sweep:
            cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
            d, l, sd = mean_results(run_batch(cfg, n_reps))
            ds.append(d); ls.append(l); dstds.append(sd)
            prog.step(f"ts={ts:2d} q={q:.4f}", f"T={d:.0f}  L={l:.2f}yr")
        delay_by_ts[ts] = ds; lt_by_ts[ts] = ls; dstd_by_ts[ts] = dstds

    # Fig 3.4 — T̄ vs q
    fig, ax = plt.subplots()
    for i, ts in enumerate(ts_list):
        ys = delay_by_ts[ts]; es = dstd_by_ts[ts]
        ax.plot(q_sweep, ys, "-", color=COLORS[i], label=f"ts={ts}", lw=1.6)
        ax.fill_between(q_sweep, [y-1.96*e for y,e in zip(ys,es)],
                        [y+1.96*e for y,e in zip(ys,es)], alpha=0.10, color=COLORS[i])
    ax.set_xlabel("Transmission probability q")
    ax.set_ylabel("Mean queueing delay T̄ (slots)")
    ax.set_title("Fig 3.4 — T̄ vs. q for multiple ts values")
    ax.legend(title="Idle timer ts"); savefig("fig3_4_delay_vs_q", fig)

    # Fig 3.5 — L̄ vs q
    fig, ax = plt.subplots()
    for i, ts in enumerate(ts_list):
        lt = [v if v != float("inf") else np.nan for v in lt_by_ts[ts]]
        ax.plot(q_sweep, lt, "-", color=COLORS[i], label=f"ts={ts}", lw=1.6)
    ax.set_xlabel("Transmission probability q")
    ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.5 — L̄ vs. q for multiple ts values")
    ax.legend(title="Idle timer ts"); savefig("fig3_5_lifetime_vs_q", fig)

    # --- ts sweep (figs 3.6, 3.7) ---
    delay_by_q, lt_by_q = {}, {}
    for q in q_list:
        ds, ls = [], []
        for ts in ts_sweep:
            cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
            d, l, _ = mean_results(run_batch(cfg, n_reps))
            ds.append(d); ls.append(l)
            prog.step(f"q={q:.4f} ts={ts:2d}", f"T={d:.0f}  L={l:.2f}yr")
        delay_by_q[q] = ds; lt_by_q[q] = ls

    # Fig 3.6 — T̄ vs ts
    fig, ax = plt.subplots()
    for i, q in enumerate(q_list):
        ax.plot(ts_sweep, delay_by_q[q], "-", color=COLORS[i], label=f"q={q}", lw=1.6)
    ax.set_xlabel("Idle timer ts (slots)")
    ax.set_ylabel("Mean queueing delay T̄ (slots)")
    ax.set_title("Fig 3.6 — T̄ vs. ts for multiple q values")
    ax.legend(title="Tx prob q"); savefig("fig3_6_delay_vs_ts", fig)

    # Fig 3.7 — L̄ vs ts
    fig, ax = plt.subplots()
    for i, q in enumerate(q_list):
        lt = [v if v != float("inf") else np.nan for v in lt_by_q[q]]
        ax.plot(ts_sweep, lt, "-", color=COLORS[i], label=f"q={q}", lw=1.6)
    ax.set_xlabel("Idle timer ts (slots)")
    ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.7 — L̄ vs. ts for multiple q values")
    ax.legend(title="Tx prob q"); savefig("fig3_7_lifetime_vs_ts", fig)

    # --- λ sweep (fig 3.8) ---
    fig, ax = plt.subplots()
    for i, nn in enumerate(n_vals_8):
        tputs = []
        for lm in lam_sweep:
            cfg = base_config(n=nn, lam=lm, q=1.0/nn, ts=10, tw=2, slots=max_slots)
            reps = run_batch(cfg, n_reps)
            tputs.append(float(np.mean([r.throughput for r in reps])))
            prog.step(f"n={nn} lam={lm:.4f}", f"tput={tputs[-1]:.5f}")
        ax.plot(lam_sweep, tputs, "-", color=COLORS[i], label=f"n={nn}", lw=1.6)
    ax.set_xlabel("Arrival rate λ (packets/slot)")
    ax.set_ylabel("Throughput (packets/slot)")
    ax.set_title("Fig 3.8 — Throughput vs. λ for varying n")
    ax.legend(title="Nodes n"); savefig("fig3_8_throughput_vs_lambda", fig)
    prog.finish()


# ============================================================================
# SEC O3 — Figs 3.9–3.13: Optimization
# ============================================================================
def sec_o3(quick: bool):
    # quick: n=50, 3 reps, 8k slots → grid 9×15 = 135 steps × ~3s ≈ 7 min
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000
    n, lam, tw = 50 if quick else 100, 0.01, 2

    # 5x denser grid (15q × 10ts vs old 6q × 6ts)
    q_sweep  = list(np.linspace(0.005, 0.15, 15 if quick else 25))
    ts_sweep = [1, 2, 5, 8, 10, 15, 20, 30, 50] if quick else \
               [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    ts_compare = [1, 3, 5, 10, 15, 20, 30] if quick else \
                 [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]

    # One prog.step() per grid cell + 3 scenario runs + per (ts × 2 schemes)
    total_sims = (
        len(ts_sweep) * len(q_sweep)
        + 3                        # scenarios: Low-Latency, Balanced, Battery-Life
        + len(ts_compare) * 2      # on-demand + duty-cycle per ts
    )
    prog = Progress(total_sims, "Figs 3.9-3.13  Optimization (O3)")

    # Build 2-D grid
    delay_mat = np.full((len(ts_sweep), len(q_sweep)), np.nan)
    lt_mat    = np.full_like(delay_mat, np.nan)

    for i, ts in enumerate(ts_sweep):
        for j, q in enumerate(q_sweep):
            cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
            d, l, _ = mean_results(run_batch(cfg, n_reps))
            delay_mat[i, j] = d
            lt_mat[i, j]    = l if l != float("inf") else np.nan
            prog.step(f"ts={ts:2d} q={q:.4f}", f"T={d:.0f}  L={l:.2f}yr")

    q_arr  = np.array(q_sweep)
    ts_arr = np.array(ts_sweep)

    # μ matrix for stability contour
    mu_mat = np.zeros_like(delay_mat)
    for i, ts in enumerate(ts_sweep):
        for j, q in enumerate(q_sweep):
            p = q * (1-q)**(n-1)
            denom = 1 + p*ts + p*tw
            mu_mat[i, j] = p / denom if denom > 0 else 0.0

    # Fig 3.9 — T̄ heatmap
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(q_arr, ts_arr, delay_mat, cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=ax, label="T̄ (slots)")
    ax.set_xlabel("q"); ax.set_ylabel("ts")
    ax.set_title("Fig 3.9 — Mean Delay T̄ in (q, ts) Plane")
    savefig("fig3_9_delay_heatmap", fig)

    # Fig 3.10 — L̄ heatmap + stability contour
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(q_arr, ts_arr, lt_mat, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax, label="L̄ (years)")
    CS = ax.contour(q_arr, ts_arr, mu_mat, levels=[lam], colors="white",
                    linestyles="--", linewidths=2)
    ax.clabel(CS, fmt="λ=μ", fontsize=9)
    ax.set_xlabel("q"); ax.set_ylabel("ts")
    ax.set_title("Fig 3.10 — Expected Lifetime L̄ in (q, ts) Plane")
    savefig("fig3_10_lifetime_heatmap", fig)

    # Fig 3.11 — Pareto frontier
    pareto_lt, pareto_d = [], []
    for i, ts in enumerate(ts_sweep):
        j_lt = int(np.nanargmax(lt_mat[i]))
        j_d  = int(np.nanargmin(delay_mat[i]))
        pareto_lt.append((float(delay_mat[i, j_lt]), float(lt_mat[i, j_lt]), ts))
        pareto_d.append((float(delay_mat[i, j_d]),  float(lt_mat[i, j_d]),  ts))

    fig, ax = plt.subplots()
    xs_lt = [v[0] for v in pareto_lt if not np.isnan(v[1])]
    ys_lt = [v[1] for v in pareto_lt if not np.isnan(v[1])]
    xs_d  = [v[0] for v in pareto_d  if not np.isnan(v[1])]
    ys_d  = [v[1] for v in pareto_d  if not np.isnan(v[1])]
    ax.plot(xs_lt, ys_lt, "o-", color=COLORS[0], label="Max-lifetime q*")
    ax.plot(xs_d,  ys_d,  "s--", color=COLORS[1], label="Min-delay q*")
    for d, l, ts in pareto_lt:
        if not np.isnan(l):
            ax.annotate(f"ts={ts}", (d, l), textcoords="offset points",
                        xytext=(3, 2), fontsize=7)
    ax.set_xlabel("Mean delay T̄ (slots)"); ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.11 — Pareto Frontier: L̄ vs. T̄")
    ax.legend(); savefig("fig3_11_pareto", fig)

    # Scenarios (fig 3.12)
    scenarios = {
        "Low-Latency":  (2.0/n, 1),
        "Balanced":     (1.0/n, 10),
        "Battery-Life": (0.5/n, 50),
    }
    sc_names, sc_delays, sc_lts = [], [], []
    baseline_d = baseline_l = None
    for sname, (q, ts) in scenarios.items():
        cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
        d, l, _ = mean_results(run_batch(cfg, n_reps))
        sc_names.append(sname); sc_delays.append(d)
        sc_lts.append(l if l != float("inf") else np.nan)
        if sname == "Balanced":
            baseline_d = d; baseline_l = l
        prog.step(f"scenario={sname}", f"T={d:.0f}  L={l:.3f}yr")

    pct_delay = [(d - baseline_d)/baseline_d*100 for d in sc_delays]
    pct_lt    = [(l - baseline_l)/baseline_l*100
                 if baseline_l not in (float("inf"), 0) and not np.isnan(l) else 0
                 for l in sc_lts]
    x = np.arange(len(sc_names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, pct_delay, 0.4, label="Δ T̄ (%)", color=COLORS[0], alpha=0.8)
    ax.bar(x + 0.2, pct_lt,    0.4, label="Δ L̄ (%)", color=COLORS[1], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sc_names)
    ax.set_ylabel("% change vs. Balanced baseline")
    ax.set_title("Fig 3.12 — Scenario Comparison vs. Balanced Baseline")
    ax.legend(); savefig("fig3_12_scenario_bar", fig)

    # On-demand vs duty-cycling (fig 3.13)
    dc_delays, dc_lts, od_delays, od_lts = [], [], [], []
    for ts in ts_compare:
        q   = 1.0 / n
        cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
        d_od, l_od, _ = mean_results(run_batch(cfg, n_reps))
        od_delays.append(d_od)
        od_lts.append(l_od if l_od != float("inf") else np.nan)
        prog.step(f"OD ts={ts}", f"T={d_od:.0f}  L={l_od:.3f}yr")

        dc_reps = [DutyCycleSimulator.run_duty_cycle_simulation(
                       base_config=cfg,
                       cycle_period=ts + tw,
                       awake_fraction=1.0 / max(ts + tw, 2),
                       seed=rep)
                   for rep in range(n_reps)]
        d_dc = float(np.mean([r.mean_delay for r in dc_reps]))
        lt_dc = [r.mean_lifetime_years for r in dc_reps if r.mean_lifetime_years != float("inf")]
        l_dc  = float(np.mean(lt_dc)) if lt_dc else np.nan
        dc_delays.append(d_dc); dc_lts.append(l_dc)
        prog.step(f"DC ts={ts}", f"T={d_dc:.0f}  L={l_dc:.3f}yr")

    x = np.arange(len(ts_compare))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axi in axes:
        axi.tick_params(axis="x", labelsize=8)

    axes[0].bar(x - 0.2, od_delays, 0.4, color=COLORS[0], alpha=0.85)
    axes[0].bar(x + 0.2, dc_delays, 0.4, color=COLORS[2], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"ts={t}" for t in ts_compare])
    axes[0].set_ylabel("T̄ (slots)")
    axes[0].set_title("Mean Delay T̄")

    axes[1].bar(x - 0.2, od_lts, 0.4, color=COLORS[0], alpha=0.85)
    axes[1].bar(x + 0.2, dc_lts, 0.4, color=COLORS[2], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"ts={t}" for t in ts_compare])
    axes[1].set_ylabel("L̄ (years)")
    axes[1].set_title("Expected Lifetime L̄")

    # Single shared legend placed below both subplots — no overlap with bars
    legend_patches = [
        mpatches.Patch(color=COLORS[0], alpha=0.85, label="On-demand sleep"),
        mpatches.Patch(color=COLORS[2], alpha=0.85, label="Duty-cycling"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Fig 3.13 — On-Demand Sleep vs. Duty-Cycling", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    savefig("fig3_13_duty_vs_ondemand", fig)
    prog.finish()


# ============================================================================
# SEC O4 — Figs 3.14–3.17: 3GPP Validation
# ============================================================================
def sec_o4(quick: bool):
    # quick: n=50, 3 reps, 8k slots → 4×25 + 10×25 + 2×5 = 360 steps × ~2s ≈ 12 min
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000
    n, tw = 50 if quick else 100, 2

    ts_settings  = [5, 10, 30, 60]
    t3324_labels = ["T3324=30ms", "T3324=60ms", "T3324=180ms", "T3324=360ms"]

    # 5x more lambda points
    lam_sweep    = list(np.linspace(0.001, 0.04, 25 if quick else 50))
    # 5x more q values for q* vs n search
    q_sweep_16   = list(np.linspace(0.005, 0.3, 25 if quick else 50))
    # 5x more n values for q* vs n
    n_vals_16    = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200] if quick else \
                   [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    # More ts for 3GPP scatter
    ts_scatter   = [5, 10, 15, 20, 30]

    # One prog.step() per (ts,lam) + per (n,q) + per (profile,ts)
    total_sims = (
        len(ts_settings) * len(lam_sweep)
        + len(n_vals_16) * len(q_sweep_16)
        + len(profiles) * len(ts_scatter)
    )
    prog = Progress(total_sims, "Figs 3.14-3.17  3GPP validation (O4)")

    # --- Figs 3.14, 3.15: L̄/T̄ vs λ for four T3324 settings ---
    lt_curves, d_curves = {}, {}
    for ts, label in zip(ts_settings, t3324_labels):
        lts, ds = [], []
        for lm in lam_sweep:
            cfg = base_config(n=n, lam=lm, q=1.0/n, ts=ts, tw=tw, slots=max_slots)
            d, l, _ = mean_results(run_batch(cfg, n_reps))
            lts.append(l if l != float("inf") else np.nan)
            ds.append(d * SLOT_MS / 1000.0)
            prog.step(f"{label} lam={lm:.4f}", f"T={d:.0f} L={l:.2f}yr")
        lt_curves[label] = lts; d_curves[label] = ds

    fig, ax = plt.subplots()
    for i, label in enumerate(t3324_labels):
        ax.plot(lam_sweep, lt_curves[label], "-", color=COLORS[i], label=label, lw=1.6)
    ax.set_xlabel("Arrival rate λ"); ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.14 — L̄ vs. λ for Four T3324 Timer Settings")
    ax.legend(); savefig("fig3_14_lifetime_vs_lambda_t3324", fig)

    fig, ax = plt.subplots()
    for i, label in enumerate(t3324_labels):
        ax.plot(lam_sweep, d_curves[label], "-", color=COLORS[i], label=label, lw=1.6)
    ax.axhline(1.0, color="red", linestyle=":", lw=1.5, label="1 s SLA")
    ax.set_xlabel("Arrival rate λ"); ax.set_ylabel("Mean delay T̄ (seconds)")
    ax.set_title("Fig 3.15 — T̄ vs. λ for Four T3324 Timer Settings")
    ax.legend(); savefig("fig3_15_delay_vs_lambda_t3324", fig)

    # --- Fig 3.16: q* vs n ---
    q_stars_delay, q_stars_lt = [], []
    for nn in n_vals_16:
        best_q_d, best_q_l, best_d, best_l = None, None, float("inf"), 0.0
        for q in q_sweep_16:
            cfg = base_config(n=nn, lam=0.01, q=q, ts=10, tw=tw, slots=max_slots)
            d, l, _ = mean_results(run_batch(cfg, n_reps))
            if d < best_d:
                best_d = d; best_q_d = q
            if l != float("inf") and l > best_l:
                best_l = l; best_q_l = q
            prog.step(f"n={nn} q={q:.4f}", f"T={d:.0f}  L={l:.2f}yr")
        q_stars_delay.append(best_q_d); q_stars_lt.append(best_q_l)

    fig, ax = plt.subplots()
    ax.plot(n_vals_16, [1.0/n for n in n_vals_16], "k--", lw=1.5, label="1/n rule")
    ax.plot(n_vals_16, q_stars_delay, "o-",  color=COLORS[0], label="q* (min delay)")
    ax.plot(n_vals_16, q_stars_lt,    "s--", color=COLORS[1], label="q* (max lifetime)")
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Optimal q*")
    ax.set_title("Fig 3.16 — Optimal q* vs. Number of Nodes")
    ax.legend(); savefig("fig3_16_qstar_vs_n", fig)

    # --- Fig 3.17: NB-IoT vs NR mMTC scatter ---
    profiles = [PowerProfile.NB_IOT, PowerProfile.NR_MMTC]
    labels_17 = ["NB-IoT", "NR mMTC"]
    fig, ax = plt.subplots()
    for i, (prof, label) in enumerate(zip(profiles, labels_17)):
        rates = PowerModel.get_profile(prof)
        xs, ys = [], []
        for ts in ts_scatter:
            cfg = SimulationConfig(
                n_nodes=100, arrival_rate=0.01, transmission_prob=0.01,
                idle_timer=ts, wakeup_time=2, initial_energy=5000.0,
                power_rates=rates, max_slots=max_slots)
            d, l, _ = mean_results(run_batch(cfg, n_reps))
            xs.append(d * SLOT_MS / 1000.0)
            ys.append(l if l != float("inf") else np.nan)
            prog.step(f"{label} ts={ts}", f"T={d:.0f}  L={l:.2f}yr")
        ax.scatter(xs, ys, s=100, color=COLORS[i], label=label, zorder=5)
        for x, y, ts in zip(xs, ys, ts_scatter):
            ax.annotate(f"ts={ts}", (x, y), textcoords="offset points",
                        xytext=(4, 2), fontsize=8)
    ax.set_xlabel("Mean delay T̄ (s)"); ax.set_ylabel("Lifetime L̄ (years)")
    ax.set_title("Fig 3.17 — 3GPP Scenarios: NB-IoT vs. NR mMTC")
    ax.legend(); savefig("fig3_17_3gpp_scatter", fig)
    prog.finish()


# ============================================================================
# SEC O5 — Figs 3.18–3.23: Independence Analysis
# ============================================================================
def sec_o5(quick: bool):
    # quick: n=50, 3 reps, 8k slots → 80 cells × ~2s ≈ 3 min
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000
    n, lam, tw = 50 if quick else 100, 0.01, 2

    # 5x denser factorial grid (10q × 8ts vs old 5q × 5ts)
    q_vals  = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15] if quick else \
              [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
    ts_vals = [1, 2, 5, 10, 15, 20, 30, 50] if quick else \
              [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]

    # One prog.step() per (q,ts) cell
    total_sims = len(q_vals) * len(ts_vals)
    prog = Progress(total_sims, "Figs 3.18-3.23  Independence analysis (O5)")

    # Wrap run_factorial_sweep to update progress per cell
    cfg_base = base_config(n=n, lam=lam, q=q_vals[2], ts=10, tw=tw, slots=max_slots)

    # Manually run the sweep so we can call prog.step() per cell
    analytical = IndependenceAnalyzer.compute_analytical_quantities(q_vals, ts_vals, tw, n)
    kappa_mat  = analytical["kappa_matrix"]
    mu_mat_a   = analytical["mu_matrix"]

    import pandas as pd
    delay_mat = np.zeros((len(ts_vals), len(q_vals)))
    lt_mat    = np.zeros_like(delay_mat)
    lt_std_mat = np.zeros_like(delay_mat)
    rows = []
    for i, ts in enumerate(ts_vals):
        for j, q in enumerate(q_vals):
            cd = _config_dict(cfg_base, transmission_prob=q, idle_timer=ts)
            rep_lt, rep_d = _run_replications(cd, n_reps)
            m_lt  = _mean_finite(rep_lt)
            m_d   = float(np.mean(rep_d))
            sd_d  = float(np.std(rep_d)) if len(rep_d) > 1 else 0.0
            m_lt_v = m_lt if m_lt != float("inf") else np.nan
            delay_mat[i, j] = m_d
            lt_mat[i, j]    = m_lt_v
            kappa = kappa_mat[i, j]
            stable = lam < float(mu_mat_a[i, j])
            rows.append({"q": q, "ts": ts, "mean_delay": m_d, "std_delay": sd_d,
                         "mean_lifetime": m_lt_v, "p_analytical": analytical["p_matrix"][i,j],
                         "mu_analytical": float(mu_mat_a[i, j]), "kappa": kappa, "stable": stable})
            prog.step(f"q={q:.4f} ts={ts:2d}", f"T={m_d:.0f}  kappa={kappa:.3f}")

    df = pd.DataFrame(rows)
    q_arr  = np.array(q_vals)
    ts_arr = np.array(ts_vals)

    # Fig 3.18 — interaction plots (2×2)
    ts_hi = [ts_vals[0], ts_vals[len(ts_vals)//2], ts_vals[-1]]
    q_hi  = [q_vals[0],  q_vals[len(q_vals)//2],  q_vals[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, by, hi_vals, mat, ylabel) in zip(
        axes.ravel(),
        [
            ("T̄ vs q, stratified by ts", "ts", ts_hi, delay_mat, "T̄ (slots)"),
            ("L̄ vs q, stratified by ts", "ts", ts_hi, lt_mat,    "L̄ (years)"),
            ("T̄ vs ts, stratified by q", "q",  q_hi,  delay_mat, "T̄ (slots)"),
            ("L̄ vs ts, stratified by q", "q",  q_hi,  lt_mat,    "L̄ (years)"),
        ]
    ):
        ax.set_title(metric, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
        if by == "ts":
            for j, ts in enumerate(hi_vals):
                idx = ts_vals.index(ts)
                ax.plot(q_vals, mat[idx], "-", color=COLORS[j], label=f"ts={ts}", lw=1.5)
            ax.set_xlabel("q")
        else:
            for j, q in enumerate(hi_vals):
                idx = q_vals.index(q)
                ax.plot(ts_vals, mat[:, idx], "-", color=COLORS[j], label=f"q={q:.3f}", lw=1.5)
            ax.set_xlabel("ts")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 3.18 — Interaction Plots: T̄ and L̄ vs. q and ts", fontsize=12)
    fig.tight_layout(); savefig("fig3_18_interaction_plots", fig)

    # Fig 3.19 — regression residuals
    reg = IndependenceAnalyzer.run_regression_analysis(df)
    delay_reg = reg.get("delay", {})
    if "residuals_additive" in delay_reg:
        fig, ax = plt.subplots()
        kappas = df["kappa"].values
        resid  = np.array(delay_reg["residuals_additive"])
        sc = ax.scatter(kappas, resid, c=df["ts"].values, cmap="viridis",
                        s=40, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="ts")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("κ = p·ts"); ax.set_ylabel("Log residual (additive T̄ model)")
        ax.set_title("Fig 3.19 — Additive Regression Residuals vs. κ")
        savefig("fig3_19_residuals", fig)
    else:
        print("  [skip] Regression residuals not available")

    # Fig 3.20 — κ heatmap
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(q_arr, ts_arr, kappa_mat, cmap="YlOrRd", shading="auto")
    plt.colorbar(im, ax=ax, label="κ = p·ts")
    for level, style, lbl in [(0.1, "--", "κ=0.1"), (1.0, "-", "κ=1.0")]:
        CS = ax.contour(q_arr, ts_arr, kappa_mat, levels=[level],
                        colors="navy", linestyles=style, linewidths=1.8)
        ax.clabel(CS, fmt=lbl, fontsize=9, colors="navy")
    ax.set_xlabel("q"); ax.set_ylabel("ts")
    ax.set_title("Fig 3.20 — Coupling Heatmap: κ = p·ts in (q, ts) Plane")
    savefig("fig3_20_kappa_heatmap", fig)

    # Fig 3.21 — regime map
    regime = np.zeros_like(kappa_mat)
    regime[kappa_mat >= 1.0] = 2
    regime[(kappa_mat >= 0.1) & (kappa_mat < 1.0)] = 1
    cmap_r = mcolors.ListedColormap(["#A8E6CF", "#FFD700", "#FF6B6B"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.pcolormesh(q_arr, ts_arr, regime, cmap=cmap_r, shading="auto", vmin=0, vmax=2)
    patches = [mpatches.Patch(color="#A8E6CF", label="Near-independent (κ < 0.1)"),
               mpatches.Patch(color="#FFD700", label="Moderate (0.1 ≤ κ < 1)"),
               mpatches.Patch(color="#FF6B6B", label="Strongly coupled (κ ≥ 1)")]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_xlabel("q"); ax.set_ylabel("ts")
    ax.set_title("Fig 3.21 — Coupling Regime Map in (q, ts) Plane")
    savefig("fig3_21_regime_map", fig)

    # Fig 3.22 — q*(ts) shift
    q_star_u, q_star_3, q_star_5 = [], [], []
    for ti, ts in enumerate(ts_vals):
        row_d = delay_mat[ti]; row_l = lt_mat[ti]
        q_star_u.append(q_vals[int(np.nanargmin(row_d))])
        mask3 = np.array([l >= 3.0 for l in row_l])
        mask5 = np.array([l >= 5.0 for l in row_l])
        q_star_3.append(q_vals[int(np.argmin(np.where(mask3, row_d, np.inf)))]
                        if mask3.any() else np.nan)
        q_star_5.append(q_vals[int(np.argmin(np.where(mask5, row_d, np.inf)))]
                        if mask5.any() else np.nan)
    fig, ax = plt.subplots()
    ax.plot(ts_vals, q_star_u, "o-",  color=COLORS[0], label="Unconstrained")
    ax.plot(ts_vals, q_star_3, "s--", color=COLORS[1], label="L̄ ≥ 3 yr")
    ax.plot(ts_vals, q_star_5, "^:",  color=COLORS[2], label="L̄ ≥ 5 yr")
    ax.set_xlabel("Idle timer ts"); ax.set_ylabel("q* (delay-minimising)")
    ax.set_title("Fig 3.22 — q*(ts) Shift Under Lifetime Constraints")
    ax.legend(); savefig("fig3_22_qstar_shift", fig)

    # Fig 3.23 — iso-contour
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lt_cl = np.where(np.isnan(lt_mat), 0, lt_mat)
    cf = ax.contourf(q_arr, ts_arr, lt_cl, levels=10, cmap="plasma")
    plt.colorbar(cf, ax=ax, label="L̄ (years)")
    d_cl = np.where(np.isnan(delay_mat), np.nanmax(delay_mat), delay_mat)
    CS = ax.contour(q_arr, ts_arr, d_cl, levels=8, colors="white",
                    linestyles="--", linewidths=1.5, alpha=0.8)
    ax.clabel(CS, fmt="T̄=%.0f", fontsize=8)
    ax.set_xlabel("q"); ax.set_ylabel("ts")
    ax.set_title("Fig 3.23 — L̄ Contourf with T̄ Iso-lines")
    savefig("fig3_23_isocontour", fig)
    prog.finish()


# ============================================================================
# SEC O6 — Figs 3.24–3.25: Finite Retry Limits
# ============================================================================
def sec_o6(quick: bool):
    # K_values are discrete categories, so "5x data points" means
    # more K levels and more replications per level.
    K_values   = [2, 3, 5, 8, 10, 15, None]   # was [3, 5, 10, None]
    n_reps_o6  = 5 if quick else 20
    max_slots  = 10_000 if quick else 80_000

    total_sims = len(K_values) * n_reps_o6
    prog = Progress(total_sims, "Figs 3.24-3.25  Finite retry limits (O6)")

    from src.experiments import run_o6_experiments
    # run_o6_experiments uses its own K list; call it but also track progress manually
    res = run_o6_experiments(
        n_nodes=50 if quick else 100,
        n_replications=n_reps_o6,
        max_slots=max_slots,
        quick_mode=False,  # already parametrised above
        K_values=K_values if "K_values" in run_o6_experiments.__code__.co_varnames else None,
    )
    # Mark all steps as done
    for _ in range(total_sims):
        prog.step()

    K_labels = [str(k) if k is not None else "inf" for k in res["K_values"]]
    xs = list(range(len(K_labels)))

    # Fig 3.24 — T̄ and drop_rate vs K
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(xs, res["mean_delays"], "o-", color=COLORS[0], label="T̄ (slots)", lw=1.8)
    ax2.plot(xs, res["drop_rates"],  "s--", color=COLORS[1], label="Drop rate", lw=1.8)
    ax1.set_xticks(xs); ax1.set_xticklabels(K_labels)
    ax1.set_xlabel("Retry limit K")
    ax1.set_ylabel("Mean delay T̄ (slots)", color=COLORS[0])
    ax2.set_ylabel("Packet drop rate",      color=COLORS[1])
    ax1.set_title("Fig 3.24 — T̄ and Drop Rate vs. Retry Limit K")
    lines = [mpatches.Patch(color=COLORS[0], label="T̄"),
             mpatches.Patch(color=COLORS[1], label="Drop rate")]
    ax1.legend(handles=lines, loc="upper right")
    savefig("fig3_24_retries_delay_drop", fig)

    # Fig 3.25 — Pareto T̄ vs drop_rate
    fig, ax = plt.subplots()
    sc = ax.scatter(res["drop_rates"], res["mean_delays"],
                    c=list(range(len(K_labels))), cmap="coolwarm", s=120, zorder=5)
    plt.colorbar(sc, ax=ax, label="K index (low=small K)")
    for k_lbl, dr, d in zip(K_labels, res["drop_rates"], res["mean_delays"]):
        ax.annotate(f"K={k_lbl}", (dr, d), textcoords="offset points",
                    xytext=(4, 2), fontsize=9)
    ax.set_xlabel("Packet drop rate"); ax.set_ylabel("Mean delay T̄ (slots)")
    ax.set_title("Fig 3.25 — Pareto Scatter: T̄ vs. Drop Rate (varying K)")
    savefig("fig3_25_retries_pareto", fig)
    prog.finish()


# ============================================================================
# SEC O7 — Figs 3.26–3.28: CSMA vs Slotted Aloha
# ============================================================================
def sec_o7(quick: bool):
    # quick: 3 reps, 8k slots → 14 + 50 = 64 steps × ~3s ≈ 3 min
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000
    lam, ts, tw = 0.01, 10, 2

    # 5x more n values
    n_vals    = [10, 15, 20, 30, 50, 75, 100] if quick else \
                [10, 15, 20, 30, 50, 75, 100, 150, 200]
    # 5x more lambda points
    lam_sweep = list(np.linspace(0.001, 0.04, 25 if quick else 50))

    # One prog.step() per (n × 2 schemes) + per (lam × 2 schemes)
    total_sims = len(n_vals) * 2 + len(lam_sweep) * 2
    prog = Progress(total_sims, "Figs 3.26-3.28  CSMA vs Slotted Aloha (O7)")

    aloha_delays, csma_delays, aloha_coll, csma_coll = [], [], [], []

    def coll_rate(reps):
        rates = [r.total_collisions / r.total_transmissions
                 for r in reps if r.total_transmissions > 0]
        return float(np.mean(rates)) if rates else 0.0

    for nn in n_vals:
        q = 1.0 / nn
        pwr = PowerModel.get_profile(PowerProfile.GENERIC_LOW)
        cfg_a = base_config(n=nn, lam=lam, q=q, ts=ts, tw=tw, slots=max_slots)
        cfg_c = SimulationConfig(n_nodes=nn, arrival_rate=lam, transmission_prob=q,
                                 idle_timer=ts, wakeup_time=tw, initial_energy=5000.0,
                                 power_rates=pwr, max_slots=max_slots,
                                 access_scheme="csma_1p")
        reps_a = run_batch(cfg_a, n_reps)
        d_a = float(np.mean([r.mean_delay for r in reps_a]))
        aloha_delays.append(d_a); aloha_coll.append(coll_rate(reps_a))
        prog.step(f"Aloha n={nn}", f"T={d_a:.0f}")

        reps_c = [Simulator(cfg_c).run_simulation() for _ in range(n_reps)]
        d_c = float(np.mean([r.mean_delay for r in reps_c]))
        csma_delays.append(d_c); csma_coll.append(coll_rate(reps_c))
        prog.step(f"CSMA  n={nn}", f"T={d_c:.0f}")

    # Fig 3.26 — T̄ vs n
    fig, ax = plt.subplots()
    ax.plot(n_vals, aloha_delays, "o-",  color=COLORS[0], label="Slotted Aloha", lw=1.8)
    ax.plot(n_vals, csma_delays,  "s--", color=COLORS[1], label="CSMA-1P",       lw=1.8)
    diffs = [c - a for c, a in zip(csma_delays, aloha_delays)]
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i+1] < 0:
            f = abs(diffs[i]) / (abs(diffs[i]) + abs(diffs[i+1]))
            n_star = n_vals[i] + (n_vals[i+1] - n_vals[i]) * f
            ax.axvline(n_star, color="gray", lw=1, ls=":", label=f"n*≈{n_star:.0f}")
            break
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Mean delay T̄ (slots)")
    ax.set_title("Fig 3.26 — T̄ vs. n: CSMA vs. Slotted Aloha")
    ax.legend(); savefig("fig3_26_csma_delay_vs_n", fig)

    # Fig 3.27 — collision rate vs n
    fig, ax = plt.subplots()
    ax.plot(n_vals, aloha_coll, "o-",  color=COLORS[0], label="Slotted Aloha", lw=1.8)
    ax.plot(n_vals, csma_coll,  "s--", color=COLORS[1], label="CSMA-1P",       lw=1.8)
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Collision rate")
    ax.set_title("Fig 3.27 — Collision Rate vs. n: CSMA vs. Aloha")
    ax.legend(); savefig("fig3_27_csma_collision_vs_n", fig)

    # Fig 3.28 — throughput vs λ
    nn = 100
    pwr = PowerModel.get_profile(PowerProfile.GENERIC_LOW)
    aloha_tput, csma_tput = [], []
    for lm in lam_sweep:
        q = 1.0 / nn
        cfg_a = base_config(n=nn, lam=lm, q=q, ts=ts, tw=tw, slots=max_slots)
        cfg_c = SimulationConfig(n_nodes=nn, arrival_rate=lm, transmission_prob=q,
                                 idle_timer=ts, wakeup_time=tw, initial_energy=5000.0,
                                 power_rates=pwr, max_slots=max_slots,
                                 access_scheme="csma_1p")
        ra = run_batch(cfg_a, n_reps)
        tp_a = float(np.mean([r.throughput for r in ra]))
        aloha_tput.append(tp_a)
        prog.step(f"Aloha lam={lm:.4f}", f"tput={tp_a:.5f}")

        rc = [Simulator(cfg_c).run_simulation() for _ in range(n_reps)]
        tp_c = float(np.mean([r.throughput for r in rc]))
        csma_tput.append(tp_c)
        prog.step(f"CSMA  lam={lm:.4f}", f"tput={tp_c:.5f}")

    fig, ax = plt.subplots()
    ax.plot(lam_sweep, aloha_tput, "o-",  color=COLORS[0], label="Slotted Aloha", lw=1.8)
    ax.plot(lam_sweep, csma_tput,  "s--", color=COLORS[1], label="CSMA-1P",       lw=1.8)
    ax.set_xlabel("Arrival rate λ"); ax.set_ylabel("Throughput (pkts/slot)")
    ax.set_title("Fig 3.28 — Throughput vs. λ: CSMA vs. Aloha (n=100)")
    ax.legend(); savefig("fig3_28_csma_throughput_vs_lambda", fig)
    prog.finish()


# ============================================================================
# SEC O8 — Figs 3.29–3.31: Receiver Models
# ============================================================================
def sec_o8(quick: bool):
    # quick: 3 reps, 8k slots → 21 steps × ~3s ≈ 1 min
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000
    lam, ts, tw = 0.01, 10, 2

    # 5x more n values
    n_vals  = [10, 15, 20, 30, 50, 75, 100] if quick else \
              [10, 15, 20, 30, 50, 75, 100, 150, 200]
    models  = [("collision", 10.0, 1.0),
               ("capture",   10.0, 1.0),
               ("sic",       10.0, 1.0)]
    mlabels = ["COLLISION", "CAPTURE (γ=10)", "SIC"]

    # One prog.step() per (n, model) combination
    total_sims = len(n_vals) * len(models)
    prog = Progress(total_sims, "Figs 3.29-3.31  Receiver models (O8)")

    pwr = PowerModel.get_profile(PowerProfile.GENERIC_LOW)
    results: dict[str, tuple[list, list, list]] = {m[0]: ([], [], []) for m in models}

    for nn in n_vals:
        q = 1.0 / nn
        for (mstr, cap, sic_thr), mlabel in zip(models, mlabels):
            cfg = SimulationConfig(n_nodes=nn, arrival_rate=lam, transmission_prob=q,
                                   idle_timer=ts, wakeup_time=tw, initial_energy=5000.0,
                                   power_rates=pwr, max_slots=max_slots,
                                   receiver_model=mstr,
                                   capture_threshold=cap,
                                   sic_sinr_threshold=sic_thr)
            reps = [Simulator(cfg).run_simulation() for _ in range(n_reps)]
            ep   = float(np.mean([r.empirical_success_prob for r in reps]))
            d, l, _ = mean_results(reps)
            ps, ds, ls = results[mstr]
            ps.append(ep); ds.append(d)
            ls.append(l if l != float("inf") else np.nan)
            prog.step(f"{mlabel} n={nn}", f"p={ep:.4f}  T={d:.0f}")

    # Fig 3.29 — p vs n
    fig, ax = plt.subplots()
    for i, (mstr, mlabel) in enumerate(zip([m[0] for m in models], mlabels)):
        ax.plot(n_vals, results[mstr][0], "o-", color=COLORS[i], label=mlabel, lw=1.8)
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Effective success prob. p")
    ax.set_title("Fig 3.29 — Effective p vs. n for Three Receiver Models")
    ax.legend(); savefig("fig3_29_receiver_p_vs_n", fig)

    # Fig 3.30 — T̄ vs n
    fig, ax = plt.subplots()
    for i, (mstr, mlabel) in enumerate(zip([m[0] for m in models], mlabels)):
        ax.plot(n_vals, results[mstr][1], "o-", color=COLORS[i], label=mlabel, lw=1.8)
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Mean delay T̄ (slots)")
    ax.set_title("Fig 3.30 — T̄ vs. n for Three Receiver Models")
    ax.legend(); savefig("fig3_30_receiver_delay_vs_n", fig)

    # Fig 3.31 — L̄ vs n
    fig, ax = plt.subplots()
    for i, (mstr, mlabel) in enumerate(zip([m[0] for m in models], mlabels)):
        ax.plot(n_vals, results[mstr][2], "o-", color=COLORS[i], label=mlabel, lw=1.8)
    ax.set_xlabel("Number of nodes n"); ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.31 — L̄ vs. n for Three Receiver Models")
    ax.legend(); savefig("fig3_31_receiver_lifetime_vs_n", fig)
    prog.finish()


# ============================================================================
# SEC O9 — Figs 3.32–3.34: Age of Information
# ============================================================================
def sec_o9(quick: bool):
    # 5x more q points and ts levels
    # quick: 3 reps, 8k slots → 25×6 = 150 steps × ~1.5s ≈ 4 min
    q_values  = list(np.linspace(0.005, 0.15, 25 if quick else 50))
    ts_values = [1, 3, 5, 10, 20, 30] if quick else [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    n_reps    = 3 if quick else 15
    max_slots = 8_000 if quick else 80_000

    total_sims = len(q_values) * len(ts_values) * n_reps
    prog = Progress(total_sims, "Figs 3.32-3.34  Age of Information (O9)")

    res = run_o9_experiments(
        n_nodes=50 if quick else 100,
        n_replications=n_reps,
        max_slots=max_slots,
        q_values=q_values,
        ts_values=ts_values,
        quick_mode=False,
    )
    for _ in range(total_sims):
        prog.step()

    q_vals = res["q_values"]
    ts_vals = res["ts_values"]
    grid   = res["results_grid"]

    # Fig 3.32 — AoI vs q by ts
    fig, ax = plt.subplots()
    aoi_star_per_ts, d_star_per_ts = [], []
    for i, ts in enumerate(ts_vals):
        row  = grid[ts]
        aois = [r["mean_aoi"] for r in row]
        ax.plot(q_vals, aois, "-", color=COLORS[i % 10], label=f"ts={ts}", lw=1.6)
        aoi_star_per_ts.append(q_vals[int(np.argmin(aois))])
        d_star_per_ts.append(q_vals[int(np.argmin([r["mean_delay"] for r in row]))])
    ax.set_xlabel("Transmission probability q"); ax.set_ylabel("Mean AoI (slots)")
    ax.set_title("Fig 3.32 — Mean AoI vs. q for Multiple ts Values")
    ax.legend(title="ts", ncol=2, fontsize=9); savefig("fig3_32_aoi_vs_q", fig)

    # Fig 3.33 — AoI-optimal vs delay-optimal q*
    fig, ax = plt.subplots()
    ax.plot(ts_vals, aoi_star_per_ts, "o-",  color=COLORS[0], label="AoI-optimal q*", lw=1.8)
    ax.plot(ts_vals, d_star_per_ts,   "s--", color=COLORS[1], label="Delay-optimal q*", lw=1.8)
    ax.set_xlabel("Idle timer ts"); ax.set_ylabel("q*")
    ax.set_title("Fig 3.33 — AoI-Optimal vs. Delay-Optimal q*(ts)")
    ax.legend(); savefig("fig3_33_aoi_vs_delay_qstar", fig)

    # Fig 3.34 — AoI-Lifetime Pareto
    aoi_all, lt_all, d_all = [], [], []
    for ts in ts_vals:
        for r in grid[ts]:
            aoi_all.append(r["mean_aoi"])
            lt_all.append(r["lifetime"] if r["lifetime"] != float("inf") else np.nan)
            d_all.append(r["mean_delay"])
    fig, ax = plt.subplots()
    sc = ax.scatter(aoi_all, lt_all, c=d_all, cmap="viridis", s=50, zorder=5, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Mean delay T̄ (slots)")
    ax.set_xlabel("Mean AoI (slots)"); ax.set_ylabel("Expected lifetime L̄ (years)")
    ax.set_title("Fig 3.34 — AoI–Lifetime Tradeoff (delay color-coded)")
    savefig("fig3_34_aoi_lt_pareto", fig)
    prog.finish()


# ============================================================================
# SEC O10 — Figs 3.35–3.36: MMBP Arrivals
# ============================================================================
def sec_o10(quick: bool):
    # 5x more BI values
    bi_values  = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0] if quick else \
                 [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    n_reps     = 4 if quick else 20
    max_slots  = 10_000 if quick else 80_000

    total_sims = len(bi_values) * n_reps
    prog = Progress(total_sims, "Figs 3.35-3.36  MMBP arrivals (O10)")

    res = run_o10_experiments(
        n_nodes=50 if quick else 100,
        n_replications=n_reps,
        max_slots=max_slots,
        bi_values=bi_values,
        quick_mode=False,
    )
    for _ in range(total_sims):
        prog.step()

    bi    = res["bi_values"]
    m_mu  = res["mu_mmbp"]
    b_mu  = res["mu_bernoulli"]
    e_mu  = res["mu_empirical"]
    m_err = res["mu_error_mmbp"]
    b_err = res["mu_error_bernoulli"]

    # Fig 3.35 — μ_MMBP vs empirical μ
    fig, ax = plt.subplots()
    lims = [0, max(max(e_mu), max(m_mu)) * 1.2]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.fill_between(lims, [v*.9 for v in lims], [v*1.1 for v in lims],
                    alpha=0.12, color="gray", label="±10%")
    for i, bi_val in enumerate(bi):
        c = COLORS[i % 10]
        ax.scatter(e_mu[i], m_mu[i], s=90,  color=c, marker="o", zorder=5)
        ax.scatter(e_mu[i], b_mu[i], s=90,  color=c, marker="^", alpha=0.55, zorder=5)
    custom = [Line2D([0],[0], marker="o", color="gray", lw=0, label="μ_MMBP"),
              Line2D([0],[0], marker="^", color="gray", lw=0, label="μ_Bernoulli")]
    custom += [mpatches.Patch(color=COLORS[i % 10], label=f"BI={v:.2g}")
               for i, v in enumerate(bi)]
    ax.legend(handles=custom, fontsize=8, ncol=2)
    ax.set_xlabel("Empirical μ"); ax.set_ylabel("Analytical μ")
    ax.set_title("Fig 3.35 — Analytical vs. Empirical μ for Varying BI")
    savefig("fig3_35_mmbp_mu_scatter", fig)

    # Fig 3.36 — prediction error vs BI
    fig, ax = plt.subplots()
    ax.plot(bi, m_err, "o-",  color=COLORS[0], label="MMBP formula",    lw=1.8)
    ax.plot(bi, b_err, "s--", color=COLORS[1], label="Bernoulli approx.", lw=1.8)
    ax.axhline(10, color="red", lw=1.2, ls=":", label="10% threshold")
    for i in range(len(bi) - 1):
        if b_err[i] < 10 <= b_err[i+1]:
            f  = (10 - b_err[i]) / (b_err[i+1] - b_err[i])
            bi_star = bi[i] + (bi[i+1] - bi[i]) * f
            ax.axvline(bi_star, color="gray", lw=1, ls="--", label=f"BI*≈{bi_star:.1f}")
            break
    ax.set_xlabel("Burstiness Index BI"); ax.set_ylabel("μ prediction error (%)")
    ax.set_title("Fig 3.36 — μ Prediction Error vs. BI: MMBP vs. Bernoulli")
    ax.legend(); savefig("fig3_36_mmbp_error_vs_bi", fig)
    prog.finish()


# ============================================================================
# Entry point
# ============================================================================
SECTIONS: dict[str, object] = {
    "fig2": sec_fig2,
    "val":  sec_val,
    "o2":   sec_o2,
    "o3":   sec_o3,
    "o4":   sec_o4,
    "o5":   sec_o5,
    "o6":   sec_o6,
    "o7":   sec_o7,
    "o8":   sec_o8,
    "o9":   sec_o9,
    "o10":  sec_o10,
}


def main():
    global _SECTION_TOTAL

    parser = argparse.ArgumentParser(description="Generate all report figures.")
    parser.add_argument("--full", action="store_true",
                        help="Full quality: more replications & slots (~4-6 hr)")
    parser.add_argument("--only", nargs="+", metavar="SECTION",
                        help=f"Run only these sections. Choices: {', '.join(SECTIONS)}")
    args = parser.parse_args()
    quick = not args.full

    to_run = args.only if args.only else list(SECTIONS.keys())
    _SECTION_TOTAL = len(to_run)

    t_global = time.perf_counter()
    print("=" * 72)
    print(f"  Report Figure Generator  —  {'QUICK (5x dense)' if quick else 'FULL QUALITY'} mode")
    print(f"  Output: {OUTDIR}")
    print(f"  Sections: {', '.join(to_run)}")
    print("=" * 72)

    for name in to_run:
        if name not in SECTIONS:
            print(f"\n  [WARNING] Unknown section '{name}', skipping.")
            continue
        fn = SECTIONS[name]
        try:
            if name == "fig2":
                fn()
            else:
                fn(quick)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] Section '{name}' failed: {exc}")
            traceback.print_exc()

    elapsed = time.perf_counter() - t_global
    m, s = divmod(int(elapsed), 60)
    figs = sorted(OUTDIR.glob("*.png"))
    print(f"\n{'=' * 72}")
    print(f"  Finished in {m:02d}:{s:02d}   |   {len(figs)} figures in {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
