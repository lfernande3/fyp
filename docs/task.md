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
**Status:** 0% achieved.  
**Milestone:** Optimization routines and guidelines by end Mar 2026.  
**Dependencies:** O2 experiments (Task 2).  

- **Task 3.1: Implement Optimization Logic**  
  Description: Find optimal q/ts for max lifetime or min delay (monotonic as per paper).  
  Subtasks:  
  - Grid search or binary search for q maximizing ¯L or minimizing ¯T (per paper Sec IV).  
  - Tradeoff analysis: Plot max ¯L vs. min ¯T for varying ts.  
  Estimated Effort: 7-9 hours.  
  Deadline: Mar 25, 2026.  
  Critical: Yes.  

- **Task 3.2: Compare Prioritization Scenarios**  
  Description: Simulate and plot differences: Low-latency priority (e.g., ts=1, high q) vs. battery priority (ts=50, low q).  
  Subtasks:  
  - Quantify gains/losses (e.g., 20% delay reduction costs 30% lifetime).  
  - Include duty-cycling comparison (optional extension: periodic sleep/awake).  
  Estimated Effort: 5 hours.  
  Deadline: Mar 30, 2026.  
  Critical: Yes.  

### Objective O4: Validate results against 3GPP mMTC parameters (e.g., RA-SDT, MICO mode, T3324 timer); produce design guidelines and plots.
**Status:** 20% achieved.  
**Milestone:** Validation report and guidelines by mid-Apr 2026.  
**Dependencies:** O1-O3.  

- **Task 4.1: Align with 3GPP Parameters**  
  Description: Incorporate realistic values (e.g., MICO as on-demand sleep, T3324 as ts, RA-SDT for access).  
  Subtasks:  
  - Set power rates from 3GPP NR (e.g., sleep power ~1μW).  
  - Simulate 2-step RA-SDT with MICO.  
  Estimated Effort: 4-6 hours.  
  Deadline: Apr 5, 2026.  
  Critical: Yes.  

- **Task 4.2: Validation & Comparative Study**  
  Description: Compare sim to paper analytics and 3GPP benchmarks.  
  Subtasks:  
  - Built-in checks: Empirical μ/p vs. formulas for small n.  
  - On-demand vs. duty-cycling: Show superiority as per paper conclusion.  
  Estimated Effort: 6 hours.  
  Deadline: Apr 10, 2026.  
  Critical: Yes.  

- **Task 4.3: Produce Design Guidelines & Plots**  
  Description: Summarize findings (e.g., recommended ts vs. λ for <1s delay).  
  Subtasks:  
  - Generate publication-quality plots (PNG/PDF).  
  - Write guidelines: e.g., "For λ=0.01, set q=1/n for balanced tradeoff."  
  Estimated Effort: 5 hours.  
  Deadline: Apr 15, 2026.  
  Critical: Yes.  

## Overall Milestones & Timeline
- **End Feb 2026:** Baseline simulator ready (O1 complete).  
- **Mid-Mar 2026:** Parameter impact quantified (O2 complete).  
- **End Mar 2026:** Optimizations done (O3 complete).  
- **Mid-Apr 2026:** Validation & guidelines (O4 complete).  
- **Late Apr 2026:** Final presentation/demo, report submission (Weeks 11-12 Sem B).  
- **Iteration Buffer:** 3 weeks scattered for refinements (e.g., bursty traffic fixes).  

## Tracking & Tools
- Use Jupyter Notebook for dev/experiments (import from .py modules for modularity).  
- GitHub repo for version control (optional).  
- Weekly hours: Aim for 7 hours/week as per Progress Report.  
- Risks: Long run times – mitigate with small n initially, vectorized code (numpy).  

Update this document as tasks progress. If extensions needed (e.g., CSMA), add as new objective.