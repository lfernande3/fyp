"""
generate_diagrams.py
====================
Runs all simulations and saves every presentation diagram to ppt/diagrams/.

Usage (from repo root):
    python generate_diagrams.py

Output files
------------
Slide 03  slide03_tradeoff_ts_scatter.png
Slide 05  slide05_power_profiles.png
Slide 06  slide06_timeseries.png
          slide06_state_energy_pies.png
Slide 07  slide07_q_sweep.png
          slide07_ts_sweep.png
Slide 08  slide08_scenario_comparison.png
Slide 09  slide09_pareto_frontier.png
          slide09_heatmaps.png
          slide09_duty_cycle_comparison.png
Slide 10  slide10_3gpp_scenarios.png
          slide10_formula_validation.png
          slide10_convergence.png
          slide10_publication_summary.png
"""

import os
import sys
import time
import warnings

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT  = os.path.join(ROOT, 'ppt', 'diagrams')
os.makedirs(OUT, exist_ok=True)

# Non-interactive backend MUST be set before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import numpy as np
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        pass

# ── Project imports ───────────────────────────────────────────────────────────
from src.simulator        import Simulator, BatchSimulator, SimulationConfig
from src.node             import NodeState
from src.power_model      import PowerModel, PowerProfile, BatteryConfig
from src.experiments      import ParameterSweep, ScenarioExperiments
from src.traffic_models   import TrafficGenerator, BurstyTrafficConfig, compare_poisson_vs_bursty
from src.metrics          import MetricsCalculator, analyze_batch_results
from src.visualizations   import (SimulationVisualizer, PlotConfig,
                                   plot_parameter_sweep_summary)
from src.optimization     import (ParameterOptimizer, DutyCycleSimulator,
                                   PrioritizationAnalyzer, OptimizationVisualizer,
                                   _run_replications, _mean_finite, _config_dict)
from src.validation_3gpp  import (ThreeGPPAlignment, AnalyticsValidator,
                                   DesignGuidelines, ValidationVisualizer,
                                   SLOT_DURATION_MS)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]  {name}')

TOTAL_SECTIONS = 7
_section_idx = 0
_run_start = time.time()
_prev_section_start = _run_start

def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s'

def section(title):
    global _section_idx, _prev_section_start
    now = time.time()
    elapsed = now - _run_start
    if _section_idx > 0:
        prev_dur = now - _prev_section_start
        print(f'  (previous section took {_fmt_time(prev_dur)})')
    _prev_section_start = now
    _section_idx += 1
    if _section_idx > 1:
        avg_per_section = elapsed / (_section_idx - 1)
        remaining = avg_per_section * (TOTAL_SECTIONS - _section_idx + 1)
        eta_str = f'  ~{_fmt_time(remaining)} remaining'
    else:
        eta_str = ''
    print(f'\n{"-"*60}')
    print(f'  [{_section_idx}/{TOTAL_SECTIONS}] {title}')
    print(f'  Elapsed: {_fmt_time(elapsed)}{eta_str}')
    print(f'{"-"*60}')

# ── Shared configs (quick mode — increase N_REPS / MAX_SLOTS for final run) ──
QUICK          = False
N_REPS         = 5  if QUICK else 3
MAX_SLOTS      = 30_000 if QUICK else 5_000
N_NODES        = 10
LAMBDA         = 0.01
INITIAL_ENERGY = 200_000
POINT_REDUCTION_FACTOR = 10

Q_POINTS_03  = max(5, 50 // POINT_REDUCTION_FACTOR)
Q_POINTS_07  = max(5, 50 // POINT_REDUCTION_FACTOR)
TS_POINTS_07 = max(3, 15 // POINT_REDUCTION_FACTOR)
Q_POINTS_OPT = max(3, 25 // POINT_REDUCTION_FACTOR)
Q_POINTS_2D  = max(3, 12 // POINT_REDUCTION_FACTOR)
DC_POINTS    = max(3, 8 // POINT_REDUCTION_FACTOR)
LAM_POINTS   = max(3, 12 // POINT_REDUCTION_FACTOR)

base_config = SimulationConfig(
    n_nodes=N_NODES,
    arrival_rate=LAMBDA,
    transmission_prob=0.05,
    idle_timer=10,
    wakeup_time=5,
    initial_energy=INITIAL_ENERGY,
    power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
    max_slots=MAX_SLOTS,
    seed=None,
)

print(f'Output directory : {OUT}')
print(f'Quick mode       : {QUICK}  (set QUICK=False for publication quality)')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 03  –  Lifetime vs delay CURVES  (ts swept per q)
#
# Physical insight: ts (idle timer) is the natural "knob" for the delay-lifetime
# tradeoff in on-demand sleep devices:
#   - Small ts  → node sleeps quickly after delivery → wakeup needed for next
#     packet → higher access delay, but mostly in sleep (low energy) → long life
#   - Large ts  → node stays idle monitoring → next packet starts service
#     immediately (no wakeup penalty) → lower delay, but idle energy is paid
#     continuously → shorter life
#
# For this effect to be visible the queue must empty between arrivals, which
# requires light load: lambda_03 << 1/n so nodes sleep between packets.
# With lambda=0.001 per node, inter-arrival ~1000 slots >> ts=[1..500].
# initial_energy is set lower so lifetimes land in the 1–10 year range.
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 03 – Tradeoff curves (ts swept per q)')

LAMBDA_03         = 0.001    # very light load: queue always clears, ts fires every cycle
INITIAL_ENERGY_03 = 100_000  # 100 Wh battery → lifetimes in 2–11 year range at this load
SLOTS_03          = 50_000   # 50 packets/node/run at lambda=0.001 → stable rate estimate
N_REPS_03         = 5
q_curves_03       = [0.05, 0.15, 0.35]           # 3 fixed q values → 3 distinct curves
ts_sweep_03       = [1, 20, 100, 500, 2000]       # idle timer values swept per curve

palette_03 = plt.cm.plasma(np.linspace(0.15, 0.85, len(q_curves_03)))

print(f'Sweeping ts={ts_sweep_03} for each q={q_curves_03} ...')
print(f'  (lambda={LAMBDA_03}, slots={SLOTS_03:,}, reps={N_REPS_03}, E0={INITIAL_ENERGY_03:,} mWh)')

curves = {}
for q_val in q_curves_03:
    cfg_q = SimulationConfig(
        n_nodes=N_NODES,
        arrival_rate=LAMBDA_03,
        transmission_prob=q_val,
        idle_timer=1,                           # overridden by sweep below
        wakeup_time=50,   # 300 ms → wakeup avoided at high ts gives 20-slot delay spread
        initial_energy=INITIAL_ENERGY_03,
        power_rates=base_config.power_rates,
        max_slots=SLOTS_03,
        seed=None,
    )
    res      = ParameterSweep.sweep_idle_timer(cfg_q, ts_values=ts_sweep_03,
                                               n_replications=N_REPS_03, verbose=False)
    analysis = ParameterSweep.analyze_sweep_results(res, 'ts')
    delays    = [analysis[ts]['mean_delay'][0]     for ts in ts_sweep_03]
    lifetimes = [analysis[ts]['lifetime_years'][0] for ts in ts_sweep_03]
    curves[q_val] = (delays, lifetimes, ts_sweep_03)
    print(f'  q={q_val:.2f}: delay range [{min(delays):.1f}, {max(delays):.1f}] slots  '
          f'lifetime range [{min(lifetimes):.4f}, {max(lifetimes):.4f}] years')

# ── build the plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

for i, (q_val, (delays, lifetimes, ts_vals)) in enumerate(curves.items()):
    colour = palette_03[i]

    # sort by delay ascending so the line flows left→right
    order     = np.argsort(delays)
    d_sorted  = np.array(delays)[order]
    lt_sorted = np.array(lifetimes)[order]
    ts_sorted = np.array(ts_vals)[order]

    # PchipInterpolator requires strictly increasing x; nudge any ties caused by noise
    d_plot = d_sorted.astype(float).copy()
    for k in range(1, len(d_plot)):
        if d_plot[k] <= d_plot[k - 1]:
            d_plot[k] = d_plot[k - 1] + 1e-3

    # monotonic cubic interpolation through sorted data
    if len(d_plot) >= 3:
        pchip     = PchipInterpolator(d_plot, lt_sorted)
        d_smooth  = np.linspace(d_plot.min(), d_plot.max(), 200)
        lt_smooth = pchip(d_smooth)
        ax.plot(d_smooth, lt_smooth, '-', color=colour, linewidth=2, alpha=0.7)

    # actual data points
    ax.scatter(d_sorted, lt_sorted, s=70, color=colour,
               edgecolors='white', linewidths=0.8, zorder=5,
               label=f'q = {q_val:.2f}')

    # annotate ts endpoints
    ax.annotate(f'ts={ts_sorted[-1]}',
                (d_sorted[-1], lt_sorted[-1]),
                xytext=(6, 4), textcoords='offset points',
                fontsize=7.5, color=colour)
    ax.annotate(f'ts={ts_sorted[0]}',
                (d_sorted[0], lt_sorted[0]),
                xytext=(-45, -14), textcoords='offset points',
                fontsize=7.5, color=colour)

ax.set_xlabel('Mean Access Delay (slots)', fontsize=12)
ax.set_ylabel('Mean Battery Lifetime (years)', fontsize=12)
ax.set_title('Battery Lifetime vs Access Delay\n'
             r'Each curve = one $q$ value, swept over $t_s$'
             '  (larger ts → lower delay, shorter life)',
             fontsize=13)
ax.legend(title='Tx probability q', fontsize=10, title_fontsize=10,
          framealpha=0.9, loc='upper right')
ax.grid(True, alpha=0.3)

ax.text(0.04, 0.92, 'High lifetime\n(battery priority)',
        transform=ax.transAxes, fontsize=9, color='#2d6a4f',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d8f3dc', alpha=0.7))
ax.text(0.96, 0.08, 'Low delay\n(latency priority)',
        transform=ax.transAxes, fontsize=9, color='#9d0208',
        va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffccd5', alpha=0.7))

plt.tight_layout()
save(fig, 'slide03_tradeoff_ts_scatter.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 05  –  Power profile comparison bar charts
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 05 – Power profiles')

profiles_to_plot = [PowerProfile.LORA, PowerProfile.NB_IOT,
                    PowerProfile.LTE_M, PowerProfile.NR_MMTC]
states = ['PT', 'PB', 'PI', 'PW', 'PS']
x = np.arange(len(states))
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, profile in enumerate(profiles_to_plot):
    info  = PowerModel.get_profile_info(profile)
    rates = PowerModel.get_profile(profile)
    vals  = [rates[s] for s in states]
    axes[0].bar(x + i * width, vals, width, label=info['name'])
    axes[1].bar(x + i * width, vals, width, label=info['name'])

for ax, yscale, title in zip(axes,
                              ['linear', 'log'],
                              ['Power Consumption by State (mW)',
                               'Power Consumption – Log Scale (mW)']):
    ax.set_xlabel('Power State')
    ax.set_ylabel('Power (mW)')
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(states)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('3GPP-Inspired Power Profiles  –  PT > PB > PI > PW > PS', fontsize=13)
plt.tight_layout()
save(fig, 'slide05_power_profiles.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 06  –  Single-simulation time series + state/energy pies
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 06 – Framework: time-series & pies')

cfg_06 = SimulationConfig(
    n_nodes=10, arrival_rate=0.01, transmission_prob=0.1,
    idle_timer=10, wakeup_time=5, initial_energy=INITIAL_ENERGY,
    power_rates={'PT': 10.0, 'PB': 5.0, 'PI': 1.0, 'PW': 2.0, 'PS': 0.1},
    max_slots=10_000, seed=42,
)
sim_06 = Simulator(cfg_06)
result_06 = sim_06.run_simulation(track_history=True, verbose=False)

# ── 3-panel time series ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(result_06.queue_length_history, linewidth=1, alpha=0.7)
axes[0].set_xlabel('Slot'); axes[0].set_ylabel('Avg Queue Length')
axes[0].set_title('Average Queue Length Over Time'); axes[0].grid(True, alpha=0.3)

axes[1].plot(result_06.energy_history, linewidth=1, color='orange', alpha=0.7)
axes[1].set_xlabel('Slot'); axes[1].set_ylabel('Avg Energy (units)')
axes[1].set_title('Average Energy Remaining Over Time'); axes[1].grid(True, alpha=0.3)

state_names = ['active', 'idle', 'sleep', 'wakeup']
state_data  = {s: [] for s in state_names}
for sd in result_06.state_history:
    for s in state_names:
        state_data[s].append(sd.get(s, 0))
axes[2].stackplot(range(len(result_06.state_history)),
                  [state_data[s] for s in state_names],
                  labels=state_names, alpha=0.7)
axes[2].set_xlabel('Slot'); axes[2].set_ylabel('Number of Nodes')
axes[2].set_title('Node State Distribution Over Time')
axes[2].legend(loc='upper right'); axes[2].grid(True, alpha=0.3)

plt.suptitle('Simulation Framework – Single Run Output  (n=10, ts=10, q=0.1)', fontsize=13)
plt.tight_layout()
save(fig, 'slide06_timeseries.png')

# ── State + energy pies ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

labels = list(result_06.state_fractions.keys())
sizes  = list(result_06.state_fractions.values())
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Time in Each State', fontsize=13)

e_labels = list(result_06.energy_fractions_by_state.keys())
e_sizes  = list(result_06.energy_fractions_by_state.values())
axes[1].pie(e_sizes, labels=e_labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Energy Consumption by State', fontsize=13)

fig.suptitle('Simulation Framework – State & Energy Breakdown', fontsize=13)
plt.tight_layout()
save(fig, 'slide06_state_energy_pies.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 07  –  q sweep (6-panel) and ts sweep (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 07 – Parameter sweeps (q and ts)')

q_values_07 = list(np.linspace(0.005, 0.38, Q_POINTS_07)) if not QUICK else [0.01, 0.05, 0.1, 0.2, 0.3]
print(f'Sweeping q = {q_values_07} ...')
q_results_07 = ParameterSweep.sweep_transmission_prob(
    base_config, q_values=q_values_07,
    n_replications=N_REPS, verbose=False,
)
q_analysis = ParameterSweep.analyze_sweep_results(q_results_07, 'q')

q_arr      = np.array(q_values_07)
q_delays   = [q_analysis[q]['mean_delay'][0]          for q in q_values_07]
q_delay_s  = [q_analysis[q]['mean_delay'][1]          for q in q_values_07]
q_lifetimes= [q_analysis[q]['lifetime_years'][0]*8760 for q in q_values_07]  # years → hours
q_energy   = [q_analysis[q]['energy_per_packet'][0]   for q in q_values_07]
q_throughp = [q_analysis[q]['throughput'][0]          for q in q_values_07]

fig = plt.figure(figsize=(16, 10))
gs  = matplotlib.gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(q_arr, q_delays, yerr=q_delay_s, marker='o', capsize=5, linewidth=2, markersize=7)
ax1.set_xlabel('Transmission Probability (q)'); ax1.set_ylabel('Mean Delay (slots)')
ax1.set_title('Delay vs q'); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(q_arr, q_lifetimes, marker='s', linewidth=2, markersize=7, color='green')
ax2.set_xlabel('Transmission Probability (q)'); ax2.set_ylabel('Mean Lifetime (hours)')
ax2.set_title('Lifetime vs q'); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(q_arr, q_energy, marker='^', linewidth=2, markersize=7, color='red')
ax3.set_xlabel('Transmission Probability (q)'); ax3.set_ylabel('Energy per Packet')
ax3.set_title('Energy Efficiency vs q'); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(q_arr, q_throughp, marker='D', linewidth=2, markersize=7, color='purple')
ax4.set_xlabel('Transmission Probability (q)'); ax4.set_ylabel('Throughput (pkts/slot)')
ax4.set_title('Throughput vs q'); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
sc = ax5.scatter(q_delays, q_lifetimes, s=60, c=q_arr, cmap='viridis', edgecolor='black', linewidth=0.8)
# annotate only key q values: min, q≈1/n, max, and one mid point
key_qs = {q_values_07[0], q_values_07[-1]}
qstar  = min(q_values_07, key=lambda q: abs(q - 1.0/N_NODES))
key_qs.add(qstar)
key_qs.add(q_values_07[len(q_values_07)//2])
for i, q in enumerate(q_values_07):
    if q in key_qs:
        lbl = f'q*=1/n' if abs(q - qstar) < 1e-6 else f'q={q:.2f}'
        ax5.annotate(lbl, (q_delays[i], q_lifetimes[i]),
                     xytext=(7, 7), textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='-', lw=0.6))
ax5.set_xlabel('Mean Delay (slots)'); ax5.set_ylabel('Mean Lifetime (hours)')
ax5.set_title('Lifetime–Delay Trade-off'); ax5.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax5, label='q')

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
qstar_idx = min(range(len(q_values_07)), key=lambda i: abs(q_values_07[i] - 1.0/N_NODES))
tdata = [['Metric', f'q={q_values_07[0]:.3f}', f'q*={q_values_07[qstar_idx]:.3f}', f'q={q_values_07[-1]:.3f}'],
         ['Delay',    f'{q_delays[0]:.0f} sl', f'{q_delays[qstar_idx]:.0f} sl', f'{q_delays[-1]:.0f} sl'],
         ['Lifetime', f'{q_lifetimes[0]:.0f} h', f'{q_lifetimes[qstar_idx]:.0f} h', f'{q_lifetimes[-1]:.0f} h']]
tbl = ax6.table(cellText=tdata, cellLoc='center', loc='center',
                colWidths=[0.3, 0.23, 0.23, 0.23])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.2)
for j in range(4):
    tbl[(0, j)].set_facecolor('#4CAF50')
    tbl[(0, j)].set_text_props(weight='bold', color='white')

fig.suptitle('Parameter Sweep: Transmission Probability q\n(n=20, λ=0.01, ts=10)', fontsize=14)
save(fig, 'slide07_q_sweep.png')

# ts sweep ----
ts_values_07 = list(map(int, np.unique(np.round(np.geomspace(1, 100, TS_POINTS_07))))) if not QUICK else [1, 5, 10, 50]
print(f'Sweeping ts = {ts_values_07} ...')
ts_results_07 = ParameterSweep.sweep_idle_timer(
    base_config, ts_values=ts_values_07,
    n_replications=N_REPS, verbose=False,
)
ts_analysis = ParameterSweep.analyze_sweep_results(ts_results_07, 'ts')

ts_arr      = np.array(ts_values_07)
ts_delays   = [ts_analysis[ts]['mean_delay'][0]          for ts in ts_values_07]
ts_lifetimes= [ts_analysis[ts]['lifetime_years'][0]*8760  for ts in ts_values_07]
ts_energy   = [ts_analysis[ts]['energy_per_packet'][0]   for ts in ts_values_07]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].semilogx(ts_arr, ts_delays, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Idle Timer ts (log scale)'); axes[0, 0].set_ylabel('Mean Delay (slots)')
axes[0, 0].set_title('Delay vs ts'); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogx(ts_arr, ts_lifetimes, marker='s', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Idle Timer ts (log scale)'); axes[0, 1].set_ylabel('Mean Lifetime (hours)')
axes[0, 1].set_title('Lifetime vs ts'); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].semilogx(ts_arr, ts_energy, marker='^', linewidth=2, markersize=8, color='red')
axes[1, 0].set_xlabel('Idle Timer ts (log scale)'); axes[1, 0].set_ylabel('Energy per Packet')
axes[1, 0].set_title('Energy Efficiency vs ts'); axes[1, 0].grid(True, alpha=0.3)

sc2 = axes[1, 1].scatter(ts_delays, ts_lifetimes, s=60, c=np.log10(ts_arr),
                          cmap='plasma', edgecolor='black', linewidth=0.8)
# label only a handful of key ts values (first, last, and ~4 evenly spaced)
key_idx = sorted(set([0, len(ts_values_07)-1] + list(range(0, len(ts_values_07), max(1, len(ts_values_07)//5)))))
for i in key_idx:
    axes[1, 1].annotate(f'ts={ts_values_07[i]}', (ts_delays[i], ts_lifetimes[i]),
                        xytext=(7, 7), textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='-', lw=0.6))
axes[1, 1].set_xlabel('Mean Delay (slots)'); axes[1, 1].set_ylabel('Mean Lifetime (hours)')
axes[1, 1].set_title('Lifetime–Delay Trade-off (ts)'); axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes[1, 1], label='log10(ts)')

fig.suptitle('Parameter Sweep: Idle Timer ts\n(n=20, λ=0.01, q=0.05)', fontsize=14)
plt.tight_layout()
save(fig, 'slide07_ts_sweep.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 08  –  Scenario comparison (Low-Latency / Balanced / Battery-Life)
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 08 – Scenario comparison')

scenarios_08 = [
    ScenarioExperiments.create_low_latency_scenario(n_nodes=N_NODES, arrival_rate=LAMBDA),
    ScenarioExperiments.create_balanced_scenario(n_nodes=N_NODES, arrival_rate=LAMBDA),
    ScenarioExperiments.create_battery_life_scenario(n_nodes=N_NODES, arrival_rate=LAMBDA),
]
print('Running 3 scenario simulations ...')
scenario_results_08 = ScenarioExperiments.compare_scenarios(
    scenarios_08, n_replications=N_REPS, verbose=False,
)
tradeoffs_08 = ScenarioExperiments.analyze_tradeoffs(scenario_results_08)

snames   = list(tradeoffs_08.keys())
delays8  = [tradeoffs_08[s]['delay']['mean']            for s in snames]
lifetimes8 = [tradeoffs_08[s]['lifetime']['mean'] * 8760 for s in snames]
energies8  = [tradeoffs_08[s]['energy_per_packet']['mean'] for s in snames]
throughputs8 = [tradeoffs_08[s]['throughput']['mean']    for s in snames]

colors8 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
fig = plt.figure(figsize=(16, 10))
gs8 = matplotlib.gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

def bar_panel(ax, vals, ylabel, title, add_text=True, fmt='.1f'):
    bars = ax.bar(range(len(snames)), vals, color=colors8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(snames)))
    ax.set_xticklabels(snames, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')
    if add_text:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=9, fontweight='bold')

bar_panel(fig.add_subplot(gs8[0, 0]), delays8,   'Mean Delay (slots)',    'Delay Comparison')
bar_panel(fig.add_subplot(gs8[0, 1]), lifetimes8, 'Mean Lifetime (hours)', 'Lifetime Comparison', fmt='.0f')
bar_panel(fig.add_subplot(gs8[0, 2]), energies8,  'Energy per Packet',    'Energy Efficiency')
bar_panel(fig.add_subplot(gs8[1, 0]), throughputs8, 'Throughput (pkts/slot)', 'Throughput', fmt='.5f')

ax_sc = fig.add_subplot(gs8[1, 1])
ax_sc.scatter(delays8, lifetimes8, s=400, c=colors8, edgecolor='black', linewidth=2)
for i, nm in enumerate(snames):
    ax_sc.annotate(nm, (delays8[i], lifetimes8[i]),
                   xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=colors8[i], alpha=0.7))
ax_sc.set_xlabel('Mean Delay (slots)'); ax_sc.set_ylabel('Mean Lifetime (hours)')
ax_sc.set_title('Lifetime–Delay Trade-off (3 scenarios)'); ax_sc.grid(True, alpha=0.3)

ax_tbl = fig.add_subplot(gs8[1, 2]); ax_tbl.axis('off')
tbl8 = ax_tbl.table(
    cellText=[['Metric', 'Low-Lat', 'Balanced', 'Battery'],
              ['Delay',    f'{delays8[0]:.0f}', f'{delays8[1]:.0f}', f'{delays8[2]:.0f}'],
              ['Life (h)', f'{lifetimes8[0]:.0f}', f'{lifetimes8[1]:.0f}', f'{lifetimes8[2]:.0f}']],
    cellLoc='center', loc='center', colWidths=[0.28]*4)
tbl8.auto_set_font_size(False); tbl8.set_fontsize(11); tbl8.scale(1, 2.5)
for j in range(4):
    tbl8[(0, j)].set_facecolor('#2E7D32')
    tbl8[(0, j)].set_text_props(weight='bold', color='white')

fig.suptitle('Prioritisation Scenarios: Low-Latency vs Balanced vs Battery-Life', fontsize=14)
save(fig, 'slide08_scenario_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 09  –  Pareto frontier, heatmaps, duty-cycle comparison
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 09 – Optimisation: Pareto + heatmaps + duty-cycle')

cfg_o3 = SimulationConfig(
    n_nodes=N_NODES, arrival_rate=LAMBDA,
    transmission_prob=0.05, idle_timer=10, wakeup_time=5,
    initial_energy=INITIAL_ENERGY,
    power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
    max_slots=MAX_SLOTS, seed=None,
)

q_opt   = list(np.linspace(0.005, 0.40, 8 if QUICK else Q_POINTS_OPT))
ts_opt  = [1, 10, 50] if QUICK else [1, 3, 5, 10, 20, 50]

print('Running Pareto tradeoff analysis ...')
tradeoff_pts = ParameterOptimizer.tradeoff_analysis(
    cfg_o3, q_values=q_opt, ts_values=ts_opt,
    n_replications=N_REPS, verbose=False,
)

fig, ax = OptimizationVisualizer.plot_pareto_frontier(
    tradeoff_pts,
    title='Pareto Frontier: Max Lifetime vs Min Delay  (varying ts)',
)
save(fig, 'slide09_pareto_frontier.png')

print('Running 2-D grid search (q × ts) ...')
q_2d  = list(np.linspace(0.005, 0.38, Q_POINTS_2D))
ts_2d = [1, 5, 10, 20, 50]
grid  = ParameterOptimizer.grid_search_q_ts(
    cfg_o3, q_values=q_2d, ts_values=ts_2d,
    n_replications=max(N_REPS - 2, 3), verbose=False,
)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
OptimizationVisualizer.plot_tradeoff_heatmap(grid, metric='lifetime',
    title='Mean Lifetime (years) over (q, ts) Grid', ax=axes[0])
OptimizationVisualizer.plot_tradeoff_heatmap(grid, metric='delay',
    title='Mean Delay (slots) over (q, ts) Grid', ax=axes[1])
fig.suptitle('2-D Parameter Space: (q, ts)', fontsize=14)
plt.tight_layout()
save(fig, 'slide09_heatmaps.png')

print('Running duty-cycle comparison ...')
fracs_dc = [0.1, 0.3, 0.7] if QUICK else list(np.linspace(0.05, 0.80, DC_POINTS))
dc_comparison = DutyCycleSimulator.compare_with_on_demand(
    cfg_o3, ts_values=ts_opt, awake_fractions=fracs_dc,
    cycle_period=20, n_replications=N_REPS, verbose=False,
)
fig, ax = OptimizationVisualizer.plot_duty_cycle_comparison(
    dc_comparison,
    title='On-Demand Sleep vs Duty Cycling: Lifetime–Delay Tradeoff',
)
save(fig, 'slide09_duty_cycle_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10  –  3GPP scenarios, formula validation, convergence, pub summary
# ═══════════════════════════════════════════════════════════════════════════════
section('SLIDE 10 – 3GPP Validation')

print('Running 3GPP standard scenarios ...')
scenarios_3gpp = ThreeGPPAlignment.create_standard_scenarios(
    n_nodes=N_NODES, arrival_rate=LAMBDA,
    initial_energy=INITIAL_ENERGY, max_slots=MAX_SLOTS,
)
scenario_results_3gpp = {}
for sc in scenarios_3gpp:
    cd = _config_dict(sc.config)
    rep_lt, rep_d = _run_replications(cd, N_REPS)
    mean_lt  = _mean_finite(rep_lt)
    mean_d_ms = float(np.mean(rep_d)) * SLOT_DURATION_MS
    std_lt   = float(np.std([lt for lt in rep_lt if lt != float('inf')]))
    std_d_ms = float(np.std(rep_d)) * SLOT_DURATION_MS
    scenario_results_3gpp[sc.name] = (mean_d_ms, mean_lt, std_d_ms, std_lt)
    lt_str = f'{mean_lt:.4f}' if mean_lt != float('inf') else 'inf'
    print(f'  {sc.name}: delay={mean_d_ms:.1f} ms  lifetime={lt_str} y')

fig, ax = ValidationVisualizer.plot_3gpp_scenario_comparison(
    scenario_results_3gpp,
    title='3GPP mMTC Scenarios: Lifetime vs Delay (MICO + RA-SDT)',
)
save(fig, 'slide10_3gpp_scenarios.png')

print('Running formula validation across n ...')
n_vals_v = [5, 10, 20] if QUICK else [5, 8, 10, 15, 20, 30, 50]
val_reports = AnalyticsValidator.validate_across_n(
    n_values=n_vals_v, q_per_n=True,
    lambda_rate=0.005, tw=2, ts=10,
    max_slots=MAX_SLOTS, n_replications=N_REPS, verbose=False,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ValidationVisualizer.plot_analytical_vs_empirical(
    val_reports, metric='p',
    title='Success Probability p: Analytical vs Simulation', ax=axes[0])
ValidationVisualizer.plot_analytical_vs_empirical(
    val_reports, metric='mu',
    title='Service Rate μ: Analytical vs Simulation', ax=axes[1])
plt.tight_layout()
save(fig, 'slide10_formula_validation.png')

print('Running convergence analysis ...')
base_cfg_conv = SimulationConfig(
    n_nodes=10, arrival_rate=0.005,
    transmission_prob=0.1, idle_timer=10, wakeup_time=2,
    initial_energy=INITIAL_ENERGY,
    power_rates=PowerModel.get_profile(PowerProfile.GENERIC_LOW),
    max_slots=50_000, seed=None,
)
slot_counts = [5_000, 20_000, 50_000] if QUICK else [1_000, 3_000, 5_000, 10_000, 20_000, 40_000, 80_000, 150_000]
conv_reports = AnalyticsValidator.validate_convergence(
    base_cfg_conv, slot_counts=slot_counts,
    n_replications=N_REPS, verbose=False,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ValidationVisualizer.plot_convergence(conv_reports, metric='p',  ax=axes[0])
ValidationVisualizer.plot_convergence(conv_reports, metric='mu', ax=axes[1])
plt.tight_layout()
save(fig, 'slide10_convergence.png')

print('Running lifetime vs lambda sweep (for publication summary) ...')
ts_pub   = [1, 5, 10, 20, 50]
lam_pub  = [0.001, 0.01, 0.05] if QUICK else list(np.linspace(0.001, 0.05, LAM_POINTS))
lt_data  = DesignGuidelines.lifetime_vs_lambda(
    ts_values=ts_pub, lambda_values=lam_pub,
    n=N_NODES, tw=2, power_profile=PowerProfile.NR_MMTC,
    max_slots=MAX_SLOTS, n_replications=N_REPS, verbose=False,
)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
ValidationVisualizer.plot_analytical_vs_empirical(val_reports, 'p',  ax=axes[0, 0])
ValidationVisualizer.plot_analytical_vs_empirical(val_reports, 'mu', ax=axes[0, 1])
ValidationVisualizer.plot_q_star_vs_n(list(range(5, 51, 5)),          ax=axes[0, 2])
ValidationVisualizer.plot_lifetime_vs_lambda(lt_data,                  ax=axes[1, 0])
ValidationVisualizer.plot_delay_vs_lambda(lt_data, delay_target_ms=1000.0, ax=axes[1, 1])
ValidationVisualizer.plot_convergence(conv_reports, 'p',               ax=axes[1, 2])
fig.suptitle('O4 Validation & Design Guidelines Summary  –  5G NR mMTC / NB-IoT with MICO',
             fontsize=13, y=1.01)
plt.tight_layout()
save(fig, 'slide10_publication_summary.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - _run_start
saved = sorted(os.listdir(OUT))
print(f'\n{"="*60}')
print(f'  All done!  {len(saved)} files saved to:')
print(f'  {OUT}')
print(f'  Total time: {_fmt_time(total_elapsed)}')
print(f'{"="*60}')
for f in saved:
    print(f'  - {f}')
