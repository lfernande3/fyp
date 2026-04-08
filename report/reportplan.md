# FYP Report Plan

**Project:** Sleep-Based Low-Latency Access for Machine-to-Machine Communications  
**Module:** EE4080  
**Target length:** about 55-65 pages excluding appendices

## 1. Planning Priorities

This report should optimize for clarity, not for maximum detail in the outline. The main narrative should be:

1. define the problem
2. explain the model and simulator
3. validate the baseline
4. derive design guidance from the core results
5. present extensions without breaking the main story

## 2. Report Thesis

The report argues that a validated discrete-event simulator for sleep-aware slotted random access can reproduce the baseline analytical results and show that low-delay, long-lifetime operation depends on jointly tuning sleep and access behavior rather than treating them as independent controls.

## 3. Objective Organization

The objectives should be grouped by contribution instead of being treated as ten equal items throughout the report.

### Foundation

- **O1:** Build the baseline simulator.

### Core Analysis

- **O2:** Quantify the effect of `q`, `t_s`, `n`, `lambda`, `t_w`, and traffic assumptions.
- **O5:** Determine whether `q` and `t_s` can be treated as independent.

### Design and Validation

- **O3:** Find Pareto-efficient operating points.
- **O4:** Validate against analytical and 3GPP-inspired references.

### Extension Studies

- **O6:** Finite retry limits
- **O7:** CSMA comparison
- **O8:** Capture and SIC receivers
- **O9:** Age of Information
- **O10:** MMBP arrivals

## 4. Recommended Report Structure

## Front Matter

Include the standard institutional items plus:

- abstract
- acknowledgements
- generative AI disclosure if required
- auto-generated table of contents
- auto-generated list of figures and tables

## Chapter 1 - Introduction and Context

**Purpose:** define the problem, explain why it matters, and position the contribution.

### 1.1 Background and Motivation

- large populations of battery-powered MTDs
- tension between low delay and long lifetime
- why sleep-aware access matters in M2M/IoT
- practical relevance of 3GPP-inspired sleep and access settings

### 1.2 Problem Statement and Research Questions

- define the delay-lifetime trade-off
- explain why analysis alone is not enough
- state the main and secondary research questions

### 1.3 Objectives and Contribution Structure

- present the grouped objectives
- state clearly that O1-O5 form the core contribution
- state that O6-O10 are extensions built on the validated baseline

### 1.4 Scope and Assumptions

- unsaturated regime
- Python-based simulation
- 3GPP-inspired but not full standards emulation
- what is intentionally out of scope

### 1.5 Report Roadmap

- one short paragraph previewing Chapters 2-6

### 1.6 Literature Context

- Wang et al. as the analytical foundation
- 3GPP documents for design interpretation
- slotted random access and simulation background only as needed

## Chapter 2 - Methodology

**Purpose:** explain exactly what was modeled, implemented, and measured.

### 2.1 System Model

- slotted time
- homogeneous MTD population
- active, idle, sleep, wakeup states
- role of `q`, `t_s`, `t_w`, `n`, and `lambda`

### 2.2 Analytical Baseline

- success probability `p`
- service rate `mu`
- stability condition
- role of `kappa = p*t_s`

### 2.3 Simulator Architecture

- node logic
- simulator loop
- batch execution and reproducibility support
- extension hooks for traffic and receiver models

### 2.4 Metrics

- delay
- lifetime
- throughput
- success probability
- service rate
- state fractions
- energy metrics

### 2.5 Experimental Program

- validation experiments
- O2 sweeps
- O3 optimization
- O4 3GPP-inspired scenarios
- O5 independence experiments
- note that O6-O10 are treated in Chapter 4

### 2.6 Validation Strategy

- sanity checks
- direct formula comparison
- convergence checks
- handling unstable configurations

### 2.7 Implementation Notes

- runtime expectations
- random-seed handling
- test coverage and reproducibility notes

## Chapter 3 - Core Results and Design Guidance

**Purpose:** deliver the main validated contribution in the strongest order.

### 3.1 Simulator Validation

- empirical versus analytical `p`
- empirical versus analytical `mu`
- convergence evidence
- validation summary table

**Key message:** the simulator is credible in the stable regime.

### 3.2 Parameter Impact

- delay versus `q`
- lifetime versus `q`
- delay versus `t_s`
- lifetime versus `t_s`
- throughput versus `lambda`
- representative scenario table

**Key message:** the trade-off is measurable and systematic, not anecdotal.

### 3.3 Optimization Results

- delay heatmap
- lifetime heatmap
- Pareto frontier
- scenario comparison
- on-demand sleep versus duty-cycling

**Key message:** there is no universal optimum, only objective-dependent operating points.

### 3.4 3GPP-Inspired Interpretation

- timer mapping and practical scenarios
- delay and lifetime under representative settings
- recommended `q` versus population size
- guideline table

**Key message:** the simulator can produce design-oriented guidance, not just abstract trade-off plots.

### 3.5 Independence Analysis

- interaction plots
- residual diagnostics
- coupling heatmap
- regime map
- constrained-optimal `q*(t_s)` shifts

**Key message:** `q` and `t_s` are coupled except in a weak-interaction regime.

## Chapter 4 - Extension Studies

**Purpose:** show how the validated framework generalizes to richer scenarios.

Each extension should stay short and use the same mini-structure:

1. motivation
2. method change
3. main result
4. implication

### 4.1 Finite Retry Limits

- delay versus retry limit
- drop-rate trade-off
- optional `mu_K` comparison table

### 4.2 CSMA Comparison

- delay comparison
- collision comparison
- throughput comparison
- crossover interpretation

### 4.3 Capture and SIC Receivers

- effective success probability comparison
- delay comparison
- lifetime comparison

### 4.4 Age of Information

- AoI versus `q`
- AoI-optimal versus delay-optimal settings
- AoI-delay-lifetime trade-off view

### 4.5 MMBP Arrivals

- analytical versus empirical `mu`
- error versus burstiness
- threshold where Bernoulli approximation degrades

## Chapter 5 - Discussion

**Purpose:** interpret the significance and limits of the results.

### 5.1 Why the Validation Matters

- simulation as a complement to analysis
- what the agreement does and does not prove

### 5.2 What the Trade-Off Means

- why delay and lifetime cannot both be optimized fully
- how application priorities change the preferred operating point

### 5.3 Design Implications

- usefulness and limits of `q ~= 1/n`
- why co-optimization matters
- when the weak-coupling approximation is acceptable

### 5.4 Realism and Limitations

- abstraction of 3GPP mappings
- homogeneous-node baseline
- omitted channel and physical-layer effects

### 5.5 What the Extensions Add

- whether the core story remains robust
- which extensions point to the strongest future directions

## Chapter 6 - Conclusion

**Purpose:** close the argument cleanly.

### 6.1 Summary of Contribution

- validated simulator
- quantified trade-off
- optimization and design guidance
- coupling result for `q` and `t_s`

### 6.2 Objective Review

- revisit the grouped objectives rather than repeating ten isolated bullets

### 6.3 Limitations

- short and honest

### 6.4 Future Work

- prioritize the next most credible extensions, not a long wish list

## 5. Figure and Table Planning

Keep the main body focused. Use figures to support claims, not to inflate the chapter.

### Core Figures

- Chapter 2: state diagram, architecture diagram
- Chapter 3.1: validation and convergence figures
- Chapter 3.2: parameter-sweep figures
- Chapter 3.3: optimization heatmaps and Pareto view
- Chapter 3.4: 3GPP-inspired guidance figures
- Chapter 3.5: interaction and coupling figures

### Extension Figures

- one to three figures per extension only if they add clear value
- move extra plots to appendices if body length becomes tight

### Core Tables

- experiment summary table
- validation summary table
- scenario summary table
- design guideline table
- independence-analysis summary table

## 6. Page Allocation Guide

Use the following as a planning range rather than a rigid target:

| Section | Pages |
| --- | ---: |
| Front matter | 6-8 |
| Chapter 1 | 6-8 |
| Chapter 2 | 10-12 |
| Chapter 3 | 14-18 |
| Chapter 4 | 5-7 |
| Chapter 5 | 5-7 |
| Chapter 6 | 2-3 |
| References | 2-3 |

This keeps the body comfortably within the target range while giving the core results the most space.

## 7. Writing Order

Write the report in this order:

1. Chapter 2
2. Chapter 3
3. Chapter 4
4. Chapter 1
5. Chapter 5
6. Chapter 6
7. Abstract and front matter

## 8. Style Rules for the Final Report

- prefer grouped contribution language over constant references to O1-O10
- keep the baseline story separate from the extensions
- move from validation to findings to interpretation
- use short topic sentences at the start of each section
- keep claims proportional to the evidence shown in the figures and tables

## 9. Appendices

Possible appendix content:

- source-code structure
- selected code listings
- additional plots
- full design-guideline tables
- extra extension results that are too detailed for the main body

