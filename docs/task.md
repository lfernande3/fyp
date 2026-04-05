# Tasks.md for Sleep-Based Low-Latency Access for M2M Communications Simulator

**Project Title:** Sleep-Based Low-Latency Access for Machine-to-Machine Communications  
**Version:** 1.0  
**Date:** February 10, 2026  
**Purpose:** This document outlines detailed tasks, subtasks, dependencies, and milestones for implementing the discrete-event simulator as per the PRD.md and FYP Progress Report. Tasks are aligned with objectives O1-O4 from the Progress Report PDF. Focus is on unsaturated regimes, on-demand sleep with slotted Aloha baseline, and demonstrating lifetime-delay tradeoffs. Dates are updated to reflect current progress (as of Feb 2026) and extend from the original 2025 planning chart.

## Key Assumptions
- Tool: Python with SimPy for discrete-event simulation.
- Metrics: Mean delay, tail delay, energy per packet, battery lifetime (years), queue length, state fractions, throughput, success probability p, service rate μ.
- Parameters: n (100-10,000), λ (arrival rate), ts (idle timer), tw (wake-up time), q (transmission prob), traffic models (Poisson/bursty).
- Realism: 3GPP NR power values (e.g., PT, PB, PI, PW, PS), slot duration ~6 ms.
- Environment: Google Colab for fast runs (<30 min per experiment).
- Extensions (optional/later): Duty-cycling comparison, heterogeneous nodes.
- Validation: Compare sim results to paper's analytical expressions (e.g., μ from Eq. 12, ¯L from Sec IV-A, ¯T from Eq. 3).

## Task Breakdown by Objective

### Objective O1: Set up a discrete-event simulation framework for sleep-based random access schemes (slotted Aloha with on-demand sleep as baseline).
**Status:** 100% COMPLETE ✓  
**Milestone:** Fully functional baseline simulator by end Feb 2026 (extended from Dec 2025).  
**Dependencies:** Background study (completed).  
**Progress Update (Feb 10, 2026):** ALL TASKS COMPLETE - Node, Simulator, PowerModel, and Validation modules fully implemented. Baseline simulator is production-ready with comprehensive testing, 3GPP-realistic power models, and validation utilities.  

- **Task 1.1: Define Node Class** ✓ COMPLETED (Feb 10, 2026)
  Description: Implement MTD node with states (active, idle, sleep, wakeup), queue (deque for packets with arrival times), energy tracking, idle_timer (ts), wakeup_counter (tw).  
  Subtasks:  
  - Add methods: arrive_packet (Bernoulli λ), update_state, attempt_transmit (prob q), handle_success (record delay), consume_energy (based on state and power rates). ✓  
  - Implemented full Node class in `src/node.py` with NodeState enum ✓  
  - Added comprehensive statistics tracking (delays, energy by state, state fractions) ✓  
  - Created unit test suite with 8 tests covering all functionality ✓  
  - All tests passing successfully ✓  
  Estimated Effort: 4-6 hours.  
  Actual Effort: ~5 hours.  
  Deadline: Feb 15, 2026.  
  Critical: Yes.  

- **Task 1.2: Define Simulator Class** ✓ COMPLETED (Feb 10, 2026)
  Description: Manage n nodes, slotted time loop, collision detection (success if exactly 1 transmit).  
  Subtasks:  
  - Implement run_simulation (loop slots until energy depletion or max_slots). ✓  
  - Add batch sweeps: Vary parameters (q, ts, n, λ) with multiple seeds/replications (20-50). ✓  
  - Include randomness control (fixed seeds for reproducibility). ✓  
  - Implemented Simulator class with full discrete-event simulation ✓  
  - Created BatchSimulator for parameter sweeps and replications ✓  
  - Added SimulationConfig and SimulationResults dataclasses ✓  
  - Comprehensive test suite with 10 tests covering all functionality ✓  
  - All tests passing successfully ✓  
  - Demo notebook with visualizations and trade-off analysis ✓  
  Estimated Effort: 6-8 hours.  
  Actual Effort: ~6 hours.  
  Deadline: Feb 20, 2026.  
  Critical: Yes.  

- **Task 1.3: Integrate Power Model** ✓ COMPLETED (Feb 10, 2026)
  Description: Configurable power consumption (PT for transmit, PB busy, PI idle, PW wakeup, PS sleep). Use 3GPP-inspired values (e.g., PS=0.1, PT=10 units/slot).  
  Subtasks:  
  - Make rates configurable via params dict. ✓  
  - Track initial energy E (e.g., 5000 units) and estimate lifetime in years (slots * 6ms / seconds in year). ✓  
  - Created PowerModel module with 6 predefined 3GPP-inspired profiles ✓  
  - Implemented LoRa, NB-IoT, LTE-M, 5G NR mMTC, Generic Low/High profiles ✓  
  - Added BatteryConfig class with 5 battery types (AA, AAA, coin cell, LiPo) ✓  
  - Lifetime estimation utilities with realistic calculations ✓  
  - Custom profile creation support ✓  
  - Comprehensive test suite with 11 tests (all passing) ✓  
  - Demo notebook with visualizations ✓  
  Estimated Effort: 3-4 hours.  
  Actual Effort: ~3 hours.  
  Deadline: Feb 25, 2026.  
  Critical: Yes.  

- **Task 1.4: Basic Testing & Debugging** ✓ COMPLETED (Feb 10, 2026) 
  Description: Run small-scale tests (n=5, 1000 slots) to verify state transitions, collisions, energy depletion.  
  Subtasks:  
  - Add trace logging (per-slot states, queues, energy). ✓  
  - Sanity check: No sleep (ts=∞) matches no-sleep Aloha; immediate sleep (ts=0) increases delay. ✓  
  - Created validation module with trace logging ✓  
  - Implemented analytical validation (success probability, service rate) ✓  
  - Added 3 sanity checks (no-sleep, immediate-sleep, high-q) ✓  
  - Small-scale integration test (n=5, 1000 slots) ✓  
  - Comprehensive test suite with 7 tests (all passing) ✓  
  - Standalone validation script ✓  
  Estimated Effort: 4 hours.  
  Actual Effort: ~4 hours.  
  Deadline: Feb 28, 2026.  
  Critical: Yes.  

### Objective O2: Quantify impact of key parameters (ts, tw, q, λ, n, traffic models) via simulation.
**Status:** 100% COMPLETE ✓  
**Milestone:** Trade-off curves and parameter impact plots by mid-Mar 2026.  
**Dependencies:** O1 baseline (Task 1).  
**Progress Update (Feb 10, 2026):** ALL TASKS COMPLETE - Full parameter sweep framework with traffic models, scenario comparisons, and comprehensive visualization suite.

- **Task 2.1: Implement Metrics Calculation** ✓ COMPLETED (Feb 10, 2026)
  Description: Compute all required metrics post-simulation.  
  Subtasks:  
  - Average lifetime, mean/tail queueing delay ¯T, throughput, queue length time-series, state fractions, energy breakdown, empirical p/μ. ✓  
  - Compare to paper: e.g., p = q(1-q)^{n-1}, μ = p / (1 + tw * λ / (1-λ) if sleep). ✓  
  - Implemented MetricsCalculator with analytical formulas (p, μ, ¯T, ¯L) ✓  
  - Created comprehensive empirical metrics (energy, latency, network performance) ✓  
  - Built comparison framework for validation ✓  
  - Added batch analysis utilities with confidence intervals ✓  
  - Comprehensive test suite with 33 tests (all passing) ✓  
  - Demo notebook with visualizations and parameter sweeps ✓  
  Estimated Effort: 5-7 hours.  
  Actual Effort: ~6 hours.  
  Deadline: Mar 5, 2026.  
  Critical: Yes.

- **Task 2.2: Parameter Sweep Experiments** ✓ COMPLETED (Feb 10, 2026)
  Description: Run sweeps to show impacts (e.g., higher q reduces delay but drains battery faster).  
  Subtasks:  
  - Sweep q (0.01-0.5), ts (1-100), n (10-500), λ (0.001-0.1). ✓  
  - Add traffic models: Poisson (default), bursty (e.g., batch arrivals). ✓  
  - Prioritize low-latency vs. battery scenarios (e.g., small ts for latency, large ts for lifetime). ✓  
  - Implemented ParameterSweep class for systematic sweeps ✓  
  - Created ScenarioExperiments for low-latency vs battery-life comparisons ✓  
  - Built TrafficGenerator with Poisson, bursty, periodic, and on-off models ✓  
  - Comprehensive test suite with 29 tests (13 experiments + 16 traffic) ✓  
  - Demo notebook with all sweeps and trade-off visualizations ✓  
  Estimated Effort: 8-10 hours (including run time).  
  Actual Effort: ~9 hours.  
  Deadline: Mar 15, 2026.  
  Critical: Yes.

- **Task 2.3: Visualization Integration** ✓ COMPLETED (Feb 10, 2026)
  Description: Use matplotlib for post-sim plot selection in Jupyter.  
  Subtasks:  
  - Plots: Lifetime vs. delay (scatter for ts values), vs. q (curves), queue over time, energy pies. ✓  
  - Interactive: ipywidgets sliders for q/ts/n. ✓  
  - Implemented SimulationVisualizer with 10+ plot types ✓  
  - Created InteractiveVisualizer for real-time parameter exploration ✓  
  - Added comprehensive 6-panel summary dashboards ✓  
  - Comprehensive test suite with 37 tests (all passing) ✓  
  - Demo notebook with all visualizations ✓  
  Estimated Effort: 6 hours.  
  Actual Effort: ~6 hours.  
  Deadline: Mar 20, 2026.  
  Critical: Yes.  

### Objective O3: Optimize sleep and access parameters for latency-longevity trade-offs using simulation results.
**Status:** 100% COMPLETE ✓  
**Milestone:** Optimization routines and guidelines by end Mar 2026.  
**Dependencies:** O2 experiments (Task 2).  
**Progress Update (Mar 27, 2026):** ALL TASKS COMPLETE – Full optimization module implemented with grid search, Pareto tradeoff analysis, prioritization comparison, duty-cycling extension, and comprehensive visualization suite.  

- **Task 3.1: Implement Optimization Logic** ✓ COMPLETED (Mar 27, 2026)  
  Description: Find optimal q/ts for max lifetime or min delay (monotonic as per paper).  
  Subtasks:  
  - Grid search over q values maximizing ¯L or minimizing ¯T (per paper Sec IV). ✓  
  - 2-D grid search over (q, ts) space producing lifetime/delay heatmaps. ✓  
  - Tradeoff analysis: for each ts, find max ¯L and record Pareto point. ✓  
  - Pareto frontier plot (max lifetime vs. min delay, varying ts). ✓  
  - Implemented `ParameterOptimizer` class in `src/optimization.py`. ✓  
  - `OptimizationResult` and `TradeoffPoint` dataclasses. ✓  
  - Visualization: `OptimizationVisualizer.plot_q_sweep`, `plot_pareto_frontier`, `plot_tradeoff_heatmap`. ✓  
  Estimated Effort: 7-9 hours.  
  Actual Effort: ~8 hours.  
  Deadline: Mar 25, 2026.  
  Critical: Yes.  

- **Task 3.2: Compare Prioritization Scenarios** ✓ COMPLETED (Mar 27, 2026)  
  Description: Simulate and plot differences: Low-latency priority (e.g., ts=1, high q) vs. battery priority (ts=50, low q).  
  Subtasks:  
  - Three canonical scenarios: Low-Latency (ts=1, q=2/n), Balanced (ts=10, q=1/n), Battery-Life (ts=50, q=0.5/n). ✓  
  - Quantify gains/losses (% delay and lifetime change vs. balanced baseline). ✓  
  - `PrioritizationAnalyzer` class with `run_scenario_comparison`, `print_comparison_summary`. ✓  
  - `PrioritizationComparison` dataclass with `gains_vs_baseline` dict. ✓  
  - Duty-cycling comparison: `DutyCycleSimulator` with periodic awake/sleep model. ✓  
  - Visualization: `plot_prioritization_comparison`, `plot_gains_losses_bar`, `plot_duty_cycle_comparison`. ✓  
  - Comprehensive test suite with 45 tests (all passing). ✓  
  - Demo notebook `examples/optimization_demo.ipynb` with all plots and design guidelines. ✓  
  Estimated Effort: 5 hours.  
  Actual Effort: ~5 hours.  
  Deadline: Mar 30, 2026.  
  Critical: Yes.  

### Objective O4: Validate results against 3GPP mMTC parameters (e.g., RA-SDT, MICO mode, T3324 timer); produce design guidelines and plots.
**Status:** 100% COMPLETE ✓  
**Milestone:** Validation report and guidelines by mid-Apr 2026.  
**Dependencies:** O1-O3.  
**Progress Update (Mar 29, 2026):** ALL TASKS COMPLETE – Full 3GPP alignment module with T3324/MICO/RA-SDT mapping, extended analytical formula validation, convergence analysis, on-demand superiority demonstration, design guideline table, and publication-quality plots.  

- **Task 4.1: Align with 3GPP Parameters** ✓ COMPLETED (Mar 29, 2026)  
  Description: Incorporate realistic values (e.g., MICO as on-demand sleep, T3324 as ts, RA-SDT for access).  
  Subtasks:  
  - Mapped 3GPP concepts: MICO→on-demand sleep, T3324→ts, PSM→sleep state, RA-SDT→tw. ✓  
  - `ThreeGPPAlignment` class with `t3324_to_ts()`, `ts_to_t3324()`, `RA_SDT_2STEP_TW=2`, `RA_SDT_4STEP_TW=4`. ✓  
  - `create_mico_nb_iot_scenario()` and `create_mico_nr_mmtc_scenario()` with realistic profiles. ✓  
  - `create_standard_scenarios()` returning 4 canonical 3GPP scenarios. ✓  
  - `ThreeGPPScenario` dataclass capturing T3324, RA-SDT steps, power profile. ✓  
  - Power rates already set from 3GPP NR (NB_IOT: PS=15μW, NR_MMTC: PS=10μW). ✓  
  Estimated Effort: 4-6 hours.  
  Actual Effort: ~5 hours.  
  Deadline: Apr 5, 2026.  
  Critical: Yes.  

- **Task 4.2: Validation & Comparative Study** ✓ COMPLETED (Mar 29, 2026)  
  Description: Compare sim to paper analytics and 3GPP benchmarks.  
  Subtasks:  
  - `AnalyticsValidator` class validating p, μ, ¯T, ¯L against paper formulas. ✓  
  - `validate_one()`: per-config validation with FormulaValidationResult (±5/10/20% bands). ✓  
  - `validate_across_n()`: systematic validation for n = 5, 10, 20, 50. ✓  
  - `validate_convergence()`: error vs. number of slots (convergence analysis). ✓  
  - `demonstrate_on_demand_superiority()`: confirms on-demand outperforms duty-cycling. ✓  
  - `ValidationReport` dataclass with overall pass/fail, stability condition. ✓  
  - Comprehensive test suite with 49 tests (all passing). ✓  
  Estimated Effort: 6 hours.  
  Actual Effort: ~6 hours.  
  Deadline: Apr 10, 2026.  
  Critical: Yes.  

- **Task 4.3: Produce Design Guidelines & Plots** ✓ COMPLETED (Mar 29, 2026)  
  Description: Summarize findings (e.g., recommended ts vs. λ for <1s delay).  
  Subtasks:  
  - `DesignGuidelines.recommended_ts_for_delay_target()`: find largest ts meeting delay SLA. ✓  
  - `DesignGuidelines.lifetime_vs_lambda()`: sweep λ for each ts, lifetime and delay matrices. ✓  
  - `DesignGuidelines.generate_guideline_table()`: comprehensive table (λ, ts, T3324, q*, delay, lifetime, stability). ✓  
  - `ValidationVisualizer` with 6 publication-quality plot types (analytical vs empirical, convergence, lifetime/delay vs λ, q* vs n, 3GPP scenario scatter). ✓  
  - 6-panel summary dashboard saved as PNG. ✓  
  - Design guidelines written for MICO T3324, RA-SDT, q* = 1/n rule, traffic-load guidance. ✓  
  - Demo notebook `examples/o4_validation_demo.ipynb` with all analyses and plots. ✓  
  - `run_o4_experiments()` convenience function for end-to-end execution. ✓  
  Estimated Effort: 5 hours.  
  Actual Effort: ~5 hours.  
  Deadline: Apr 15, 2026.  
  Critical: Yes.  

### Objective O5: Determine whether q and t_s are independent parameters and produce graphs that answer this question definitively.
**Status:** COMPLETE ✓ (Apr 2026)  
**Milestone:** Complete independence analysis and publication-quality figures by late Apr 2026.  
**Dependencies:** O1–O4 (all simulation and validation infrastructure complete).  
**Research Question:** Are `q` (transmission probability) and `t_s` (idle timer) independent, in the sense that the effect of changing one on system performance (¯T, ¯L) does not depend on the current value of the other?  
**Answer:** They are NOT independent. The service rate formula `μ ≈ p / (1 + p·t_s + p·t_w)` contains the cross-term `p·t_s` (where `p = q(1−q)^(n−1)`), meaning `q` and `t_s` interact multiplicatively in the denominator. Tasks 5.1–5.8 systematically prove and visualise this.  
**Implementation:** `src/independence.py` (`IndependenceAnalyzer`, `IndependenceVisualizer`, `run_o5_experiments`), `tests/test_independence.py` (20 tests, all passing), `examples/o5_independence_analysis.ipynb`.

- **Task 5.1: Full Factorial Sweep (Data Generation)** ✓ COMPLETED (Apr 2026)  
  Description: Run a full factorial experiment across a (q, t_s) grid, recording metrics per cell. This is the single data source for all subsequent tasks.  
  Subtasks:  
  - Grid: `q` ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} × `t_s` ∈ {1, 2, 5, 10, 20, 50} = 36 cells; fixed `n=100`, `λ=0.01`, `t_w=2`, 20 reps. ✓  
  - Per cell: mean delay ¯T, std(¯T), lifetime ¯L, std(¯L), analytical μ and p, kappa = p·t_s, stability flag. ✓  
  - Flag configurations violating λ < μ; exclude unstable cells from analysis. ✓  
  - Save to `data/o5_factorial_sweep.csv`. ✓  
  - Implemented in `IndependenceAnalyzer.run_factorial_sweep()`. ✓  
  Estimated Effort: 4 hours. Actual Effort: ~4 hours.  
  Deadline: Apr 10, 2026. Critical: Yes.

- **Task 5.2: Analytical Decomposition of the Coupling Mechanism** ✓ COMPLETED (Apr 2026)  
  Description: Derive analytically why `q` and `t_s` are coupled, providing the theoretical grounding for the independence answer.  
  Subtasks:  
  - Derive E[cycle] = 1/p + t_s + t_w → μ = p / (1 + p·t_s + p·t_w). ✓  
  - Show ∂μ/∂q depends on t_s and ∂μ/∂t_s = −p²/(1 + p·t_s + p·t_w)² depends on q through p. ✓  
  - Define coupling strength κ = p·t_s; κ < 0.1 → near-independent, κ ≥ 1 → strongly coupled. ✓  
  - Implemented in `IndependenceAnalyzer.compute_analytical_quantities()`. ✓  
  - Visualised via `IndependenceVisualizer.plot_coupling_heatmap()` saved as `figures/o5_coupling_heatmap.png`. ✓  
  Estimated Effort: 3 hours. Actual Effort: ~3 hours.  
  Deadline: Apr 11, 2026. Critical: Yes.

- **Task 5.3: Interaction Plots (Primary Independence Test)** ✓ COMPLETED (Apr 2026)  
  Description: Two-way interaction plots — the primary visual answer to the professors' question. Parallel curves = independent; fanning curves = coupled.  
  Subtasks:  
  - Panel A: ¯T vs q stratified by t_s (log-log, with 95% CI shading). ✓  
  - Panel B: ¯L vs q stratified by t_s. ✓  
  - Panel C: ¯T vs t_s stratified by q. ✓  
  - Panel D: ¯L vs t_s stratified by q. ✓  
  - 2×2 figure; curves fan visibly on all panels (non-parallel → interaction confirmed). ✓  
  - Implemented in `IndependenceVisualizer.plot_interaction_plots()`; saved as `figures/o5_interaction_plots.png`. ✓  
  Estimated Effort: 3 hours. Actual Effort: ~3 hours.  
  Deadline: Apr 12, 2026. Critical: Yes.

- **Task 5.4: Quantitative Interaction Test (Regression with Interaction Term)** ✓ COMPLETED (Apr 2026)  
  Description: Formally quantify the interaction using a regression model with a `log q × log t_s` interaction term. The F-test p-value is a rigorous numerical answer to the independence question.  
  Subtasks:  
  - Additive model: `log Y = a·log q + b·log t_s + c`. ✓  
  - Interaction model: `log Y = a·log q + b·log t_s + c·log q·log t_s + d`. ✓  
  - F-test for H₀: c = 0 (independence); report F-statistic, p-value, R² improvement. ✓  
  - Residual plot of additive model showing systematic pattern correlated with the other variable. ✓  
  - Implemented in `IndependenceAnalyzer.run_regression_analysis()`; visualised in `IndependenceVisualizer.plot_regression_summary()`; saved as `figures/o5_residual_interaction.png`. ✓  
  Estimated Effort: 3 hours. Actual Effort: ~3 hours.  
  Deadline: Apr 13, 2026. Critical: Yes.

- **Task 5.5: Regime Map — When Are They Near-Independent?** ✓ COMPLETED (Apr 2026)  
  Description: Identify and visualise parameter regimes where κ = p·t_s is small enough that the interaction is negligible. Gives a nuanced answer: "coupled in general, but approximately independent when κ < 0.1".  
  Subtasks:  
  - (q, t_s) plane colour-coded by κ with κ = 0.1 and κ = 1 regime boundaries. ✓  
  - Stable operating region annotated (λ < μ boundary). ✓  
  - Near-independent regime labelled ("low q, low t_s — sparse traffic, quick sleep"). ✓  
  - Implemented in `IndependenceVisualizer.plot_regime_map()`; saved as `figures/o5_regime_map.png`. ✓  
  Estimated Effort: 3 hours. Actual Effort: ~3 hours.  
  Deadline: Apr 14, 2026. Critical: Yes.

- **Task 5.6: Iso-Contour Plot — What the Coupling Means for Design** ✓ COMPLETED (Apr 2026)  
  Description: Iso-delay and iso-lifetime contours in the (q, t_s) plane. If independent, contours would be axis-aligned straight lines; their curvature makes the coupling visible for a designer.  
  Subtasks:  
  - Filled contourf of ¯L with solid iso-lifetime contour lines (1, 3, 5, 10 yr). ✓  
  - Dashed iso-delay contour lines (2, 5, 10, 20 slots) on same axes. ✓  
  - Annotation: "If independent, contours would be horizontal/vertical lines". ✓  
  - Implemented in `IndependenceVisualizer.plot_iso_contours()`; saved as `figures/o5_iso_contours.png`. ✓  
  Estimated Effort: 2 hours. Actual Effort: ~2 hours.  
  Deadline: Apr 15, 2026. Critical: Yes.

- **Task 5.7: Optimal q* Shifts with t_s (Consequence of Coupling)** ✓ COMPLETED (Apr 2026)  
  Description: A direct corollary of non-independence — the delay-minimising q* under a lifetime constraint changes with t_s. A flat q*(t_s) would mean independence; a trend proves coupling.  
  Subtasks:  
  - For each t_s, find q*(t_s) = argmin ¯T subject to ¯L ≥ 3 yr and ¯L ≥ 5 yr. ✓  
  - Plot q*(t_s) for both constraints; overlay q* = 1/n baseline (no-sleep optimum). ✓  
  - Monotone shift in q*(t_s) confirms coupling; annotated with "If independent, curves would be flat". ✓  
  - Implemented in `IndependenceAnalyzer.find_optimal_q_per_ts()` and `IndependenceVisualizer.plot_optimal_q_shift()`; saved as `figures/o5_optimal_q_vs_ts.png`. ✓  
  Estimated Effort: 2 hours. Actual Effort: ~2 hours.  
  Deadline: Apr 15, 2026. Critical: Yes.

- **Task 5.8: Notebook and Concise Written Answer** ✓ COMPLETED (Apr 2026)  
  Description: Self-contained notebook `examples/o5_independence_analysis.ipynb` with a direct written answer to the panel's question.  
  Subtasks:  
  - 8-section notebook: Setup → Analytical derivation (Section 2) → Factorial sweep (3) → Interaction plots (4) → Regression F-test (5) → Regime map (6) → Iso-contours + q* shift (7) → Written answer (8). ✓  
  - Each section has a one-sentence hypothesis and one-sentence finding. ✓  
  - 250-word written answer: (a) No, not independent; (b) analytical reason (p·t_s cross-term); (c) empirical proof (fanning in interaction plots, significant F-test); (d) regime caveat (κ < 0.1); (e) design implication (q* shifts → must co-optimise). ✓  
  - Runs end-to-end in < 10 min on Colab with `n=100`, `slots=30000` (quick_mode). ✓  
  - `run_o5_experiments()` convenience function for end-to-end execution. ✓  
  - 20 unit tests, all passing (`tests/test_independence.py`). ✓  
  Estimated Effort: 2 hours. Actual Effort: ~2 hours.  
  Deadline: Apr 17, 2026. Critical: Yes.

## Overall Milestones & Timeline
- **End Feb 2026:** Baseline simulator ready (O1 complete).  
- **Mid-Mar 2026:** Parameter impact quantified (O2 complete).  
- **End Mar 2026:** Optimizations done (O3 complete).  
- **Mid-Apr 2026:** Validation & guidelines (O4 complete).  
- **Apr 10–17, 2026:** Independence analysis of q and t_s — analytical derivation, interaction plots, regression F-test, regime map, design consequences (O5 complete).  
- **Late Apr 2026:** Final presentation/demo, report submission (Weeks 11-12 Sem B).  
- **Iteration Buffer:** 3 weeks scattered for refinements (e.g., bursty traffic fixes).  

## Tracking & Tools
- Use Jupyter Notebook for dev/experiments (import from .py modules for modularity).  
- GitHub repo for version control (optional).  
- Weekly hours: Aim for 7 hours/week as per Progress Report.  
- Risks: Long run times – mitigate with small n initially, vectorized code (numpy).  

Update this document as tasks progress. If extensions needed (e.g., CSMA), add as new objective.