# Product Requirements Document (PRD) for M2M Sleep-Based Simulator

**Project Title:** Sleep-Based Low-Latency Access for Machine-to-Machine Communications  
**Version:** 2.0  
**Date:** February 10, 2026  

## 1. Overview & Purpose

### Primary Goal
To set up a discrete-event simulation framework for sleep-based random access schemes (with slotted Aloha and on-demand sleep as the baseline) to quantify the impact of key parameters, optimize for latency-longevity trade-offs, and validate against 3GPP mMTC parameters (e.g., RA-SDT, MICO mode, T3324 timer). The simulator will demonstrate the tension between battery life and low-latency access in M2M/IoT networks with massive battery-powered Machine-Type Devices (MTDs), focusing on how prioritizing low latency affects battery life and vice versa. This includes visualizing differences in scenarios where low latency is prioritized (e.g., via smaller idle timer ts or higher transmission probability q) versus battery life prioritization (e.g., longer ts or lower q).

### Problem Solved
The simulator addresses the challenge of understanding and optimizing sleep-aware access schemes to meet stringent latency requirements under energy constraints and massive access, which analytical models alone cannot easily show through stochastic variability, visual traces of node states over time, sensitivity to realistic power models, and edge cases.

### Intended Audience
Primarily professors and assessors for evaluating FYP progress and results. Secondary audiences may include other researchers or students in IoT/M2M communications.

### Scope Level
Undecided (prototype for FYP demonstration, with potential for reusable tool).

## 2. Background & Alignment with the Paper/PDF

### Alignment with Project Objectives (from PDF)
The simulator must faithfully reproduce and validate key elements from the provided paper (Wang et al., 2024) and align with the FYP objectives:
- **O1:** Discrete-event simulation framework for slotted Aloha with on-demand sleep.
- **O2:** Quantify impact of parameters (sleep idle timer ts, wake-up time tw, transmission probability q, arrival rate λ, population size n, traffic models like Poisson and bursty arrivals).
- **O3:** Optimize sleep and access parameters for latency-longevity trade-offs.
- **O4:** Validate against 3GPP mMTC parameters (e.g., RA-SDT, MICO mode, T3324 timer); produce design guidelines and plots.
Focus on unsaturated regimes (λ < μ to ensure finite delays), as emphasized in the paper.

### Extensions
Implement options for extensions (e.g., heterogeneous nodes, non-Bernoulli arrivals, capture effect, duty-cycling comparison) later if needed, but not necessary for core scope.

### Direct Comparison to Analytical Results
The simulator should output values (e.g., service rate μ, success probability p, expected lifetime ¯L, mean queueing delay ¯T) that can be directly compared to the paper's analytical curves (e.g., monotonicity of lifetime/delay with q, tradeoff with ts).

## 3. Functional Requirements

### Key Simulation Outputs / Metrics
All metrics from the paper and PDF must be supported:
- Average / per-node lifetime (slots until energy ≤ 0, or estimated in years using realistic slot duration e.g., 6 ms).
- Mean queueing delay ¯T (and access delay ¯D, tail delay).
- Throughput (successful transmissions per slot).
- Average queue length over time.
- Fraction of time in each state (active, idle, sleep, wakeup).
- Energy consumption breakdown by state.
- Success probability p (empirical vs. analytical).
- Service rate μ (empirical).
Additional: Energy per successful packet.

### Logging
- Trace-level logging: Per-slot logs of node states, queue lengths, transmissions, collisions, energy (configurable for debugging / plots).

### Visualizations / Plots
Post-simulation selection of plots (e.g., via Jupyter interface):
- Lifetime vs. delay scatter for different ts.
- Lifetime vs. q curve.
- Delay vs. q curve.
- Queue evolution over time.
- Energy depletion curves.
- State occupation pie charts.
- Trade-off curves (e.g., on-demand vs. duty-cycling).
- Comparison plots for low-latency prioritization vs. battery-life prioritization.

### Experiment Support
- Batch / parameter sweep experiments: Run multiple replications for each (n, q, ts, λ) combination, average over seeds.
- Interactive mode: Sliders in Jupyter for q, ts, n to see real-time effects (good for demos to supervisor/assessor).

## 4. Non-Functional & Technical Requirements

### Target Scale
Align with PDF methodology:
- Number of nodes n: 100–10,000 (typical 10–100 for fast runs, max 500 for validation).
- Number of slots: 10^5–10^7 to get reliable statistics and deplete batteries.
- Replications per config: 20–50 for confidence intervals.

### Performance
Run reasonably fast on Google Colab (e.g., one full experiment in <5–30 minutes on standard laptop/Colab).

### Randomness Control
- Fixed seed per run, different seeds across replications, logging of seed for reproducibility.

### Validation / Sanity Checks
- Built-in comparisons: Simulated μ and p vs. analytical formulas for small cases.

### File Formats for Results
- Options for all: CSV per experiment, JSON summary, pickled objects, plots as PNG/PDF.

## 5. Constraints, Assumptions, Out-of-Scope

### Hard Constraints
- Align with PDF: Pure Python + SimPy library, no external simulators (e.g., ns-3/OMNeT++).
- Dependencies: Minimal (numpy, matplotlib, pandas, scipy; SimPy for discrete-event simulation).
- Traffic models: Poisson and bursty arrivals.
- Realism: Use 3GPP NR power values and slot duration (6 ms) where possible.

### Out of Scope
- Physical layer modeling, mobility, multi-channel, non-slotted Aloha, CSMA instead of Aloha, real hardware emulation.
- Saturated regimes (focus on unsaturated as per paper).

### Assumptions
- Power model: Configurable (PT >> PB > PI > PW > PS, with specific ratios inspired by LoRa/5G MTDs).
- Wake-up cost: Modeled realistically (energy during tw slots).

## 6. Success & Deliverables

### Success Criteria
Undecided, but aligned with PDF progress: Simulator produces publication-quality plots showing clear tradeoff improvements over default parameters (e.g., gains in lifetime/delay via optimal q/ts), matches paper trends (e.g., on-demand outperforms duty-cycling), runs stably, and supports FYP milestones (e.g., trade-off curves by Jan 2026).

### Key Deliverables
- Jupyter notebook with examples (e.g., baseline simulations, parameter sweeps, interactive plots).
- Set of plots for thesis/paper (e.g., delay vs. lifetime, design guidelines like recommended ts vs. traffic load).
- GitHub repo (optional) with README + sample runs.
- Preliminary report on simulation results by end Jan 2026.