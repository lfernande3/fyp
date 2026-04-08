# Product Requirements Document

**Project Title:** Sleep-Based Low-Latency Access for Machine-to-Machine Communications  
**Version:** 3.0  
**Date:** April 8, 2026

## 1. Project Definition

### Purpose
Build a discrete-event simulation framework for sleep-aware random access in M2M/IoT networks, using slotted Aloha with on-demand sleep as the baseline, so the project can quantify the delay-lifetime trade-off, validate the analytical model, and produce report-ready design guidance.

### Core Problem
Battery-powered machine-type devices must balance two competing goals:

- low access delay for fresh and timely delivery
- long battery lifetime under sparse and bursty traffic

Analytical models explain the mean trends, but they do not show transient behavior, stochastic variability, validation confidence, or how the conclusions change once richer traffic, retry, and receiver assumptions are added.

### Primary Research Question
How should the transmission probability `q` and sleep timer `t_s` be configured so that large populations of MTDs can achieve acceptable delay without giving up battery lifetime?

### Secondary Research Question
Can `q` and `t_s` be tuned independently, or does the system require joint optimization?

## 2. Intended Outcome

The project should produce:

- a validated simulator for the baseline model
- a structured set of experiments covering parameter sensitivity, optimization, and 3GPP-inspired interpretation
- report-ready figures, tables, and discussion points
- a clear separation between the core contribution and the extension studies

## 3. Objective Structure

To improve report flow, the objectives are grouped by contribution rather than listed as a flat set of ten unrelated tasks.

### A. Foundation Objectives

- **O1:** Build the discrete-event simulation framework for slotted Aloha with on-demand sleep.

### B. Core Analysis Objectives

- **O2:** Quantify how `t_s`, `q`, `n`, `lambda`, `t_w`, and traffic assumptions affect delay and lifetime.
- **O5:** Determine analytically and empirically whether `q` and `t_s` behave independently.

### C. Design and Validation Objectives

- **O3:** Identify Pareto-efficient operating points for the latency-longevity trade-off.
- **O4:** Validate simulation output against analytical formulas and map the results to 3GPP-inspired settings.

### D. Extension Objectives

- **O6:** Add finite retry limits and analyze the delay-drop trade-off.
- **O7:** Compare slotted Aloha with a CSMA-based alternative.
- **O8:** Add capture-effect and SIC receiver models.
- **O9:** Add Age of Information as a timeliness metric.
- **O10:** Extend the analysis to MMBP arrivals and identify when the Bernoulli approximation breaks down.

## 4. Scope

### In Scope

- discrete-event simulation in Python
- slotted random access with on-demand sleep
- unsaturated operating regime with `lambda < mu`
- battery-lifetime and delay analysis
- parameter sweeps, validation, optimization, and figure generation
- 3GPP-inspired interpretation of `t_s`, `t_w`, and related settings
- extension studies that reuse the validated simulator core

### Out of Scope

- full physical-layer modeling
- mobility and multi-channel access
- hardware emulation
- standard-compliant implementation of complete 3GPP procedures
- saturated-regime performance as the main study target

## 5. Functional Requirements

### Simulation Capabilities

The simulator must support:

- per-node state evolution across active, idle, sleep, and wakeup states
- packet generation under baseline and richer traffic models
- slotted contention and transmission resolution
- configurable energy consumption by state
- repeated runs across parameter grids with seeded randomness

### Required Outputs

The simulator must be able to produce:

- mean queueing delay and related delay statistics
- empirical success probability `p`
- empirical service rate `mu`
- throughput and queue statistics
- lifetime estimates in slots and physical time
- state occupancy fractions
- energy-consumption breakdowns
- report-ready plots and summary tables

### Logging and Export

The framework should support:

- trace-level logging for debugging
- CSV and JSON experiment summaries
- PNG and PDF plot export
- notebook-friendly workflows for demonstrations and figure generation

## 6. Experimental Requirements

### Baseline Experimental Program

The project should cover:

- validation of `p`, `mu`, delay, and lifetime against the analytical baseline
- parameter sweeps over `q`, `t_s`, `n`, and `lambda`
- optimization over the `(q, t_s)` design space
- scenario comparison for latency-priority, balanced, and battery-priority settings
- 3GPP-inspired mappings for practical interpretation
- explicit independence analysis for `q` and `t_s`

### Extension Experimental Program

Each extension objective should have a focused experiment set and a compact report write-up, rather than being allowed to dominate the main narrative.

## 7. Non-Functional Requirements

### Performance

- runs should remain practical on a normal laptop or notebook environment
- standard batch experiments should complete in minutes rather than hours when possible

### Reproducibility

- random seeds must be recorded
- parameter configurations must be logged clearly
- exported figures and tables should be reproducible from saved notebooks or scripts

### Maintainability

- the baseline simulator should remain modular so extensions do not break core validation
- metrics, traffic models, and receiver logic should remain separable where possible

## 8. Validation Requirements

The project is successful only if the simulator is credible before optimization and extensions are discussed.

Validation should therefore include:

- sanity checks on state transitions and contention behavior
- direct comparisons against analytical formulas
- convergence checks over increasing run lengths
- exclusion or clear labeling of unstable configurations

## 9. Success Criteria

The project should be considered successful if it delivers:

- a stable and testable simulator
- strong agreement with the analytical baseline in the valid regime
- clear plots showing the delay-lifetime trade-off
- practical guidance on how `q` and `t_s` should be chosen
- a report structure where the core argument is easy to follow and the extensions support, rather than distract from, that argument

## 10. Deliverables

- simulation codebase
- experiment notebooks and scripts
- thesis/report figures and tables
- a final report built around the grouped objective structure above
- concise extension write-ups that sit after the core results
