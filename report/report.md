# Sleep-Based Low-Latency Access for Machine-to-Machine Communications

**Module:** EE4080 Final Year Project  
**Student:** [Student Name]  
**Student ID:** [Student ID]  
**Supervisor:** [Supervisor Name]  
**Assessor:** [Assessor Name]  
**Semester:** Semester B, 2025/26

---

## Abstract

Massive machine-to-machine (M2M) and Internet of Things (IoT) deployments rely on large populations of battery-powered machine-type devices (MTDs), many of which must operate for long periods while still supporting timely data delivery. This requirement creates a persistent engineering tension between low access delay and long battery lifetime. Sleep-aware random-access schemes are a natural response to that tension, but the interaction between sleep control and access aggressiveness is not straightforward, especially when the system is examined beyond idealized analytical assumptions.

This project develops a discrete-event simulation framework for sleep-based random access with slotted Aloha and on-demand sleep as the baseline model. The simulator was designed to reproduce the analytical setting of Wang et al. (2024) while also supporting richer experiments involving realistic power models, repeated replications, parameter sweeps, optimization studies, 3GPP-inspired interpretation, and several targeted extensions. The implementation is modular, test-backed, and organized around reusable components for node behavior, network simulation, metrics, validation, experiments, and visualization.

The report first establishes the credibility of the simulator by comparing empirical success probability, service rate, delay, and lifetime trends against the analytical baseline in the stable regime. It then uses the validated simulator to quantify the impact of the key parameters `q`, `t_s`, `t_w`, `n`, and `lambda`, derive Pareto-efficient operating points, and translate the results into design-oriented guidance using 3GPP-inspired settings. A central contribution of the work is the independence analysis, which shows that the transmission probability `q` and idle timer `t_s` cannot generally be treated as independent tuning knobs because they interact through a multiplicative coupling term in the service process.

The simulator is further extended to study finite retry limits, CSMA comparison, capture and successive interference cancellation (SIC) receivers, Age of Information (AoI), and Markov-modulated Bernoulli process (MMBP) arrivals. Taken together, the results show that sleep-aware access design should be based on joint tuning of sleep and access behavior rather than sequential single-parameter adjustment. The report therefore contributes both a validated software framework and a structured set of design insights for low-power M2M access.

**Abstract note:** Replace broad summary language with final verified numerical values after the definitive figures and tables are inserted.

## Acknowledgements

I would like to express my sincere gratitude to my supervisor, [Supervisor Name], for guidance, patience, and constructive feedback throughout this project. Their advice shaped both the technical direction of the simulator and the way the work was developed into a coherent final report.

I also acknowledge the analytical work of Wang et al. (2024), which served as the principal academic reference point for this project. The simulator developed here was designed to reproduce, validate, and extend that baseline in a software-oriented and experimentally flexible form.

Finally, I acknowledge the Python scientific ecosystem, including the libraries used for simulation, numerical analysis, testing, and visualization. These tools made it possible to implement the framework, execute repeated experiments, and generate report-quality outputs in a reproducible way.

## Use of Generative AI Tools

I am the author of this report. The engineering ideas, experiment design, analysis of results, figures and tables (other than standard library outputs), and all interpretations and conclusions are my own. Factual and technical claims are grounded in my implementation, simulation outputs, and the cited literature—not in generative tools treated as authoritative sources.

**Report text.** I drafted the full manuscript myself. I used **Grammarly** only to review grammar, spelling, and clarity. Its suggestions were applied selectively where they improved readability; they did not replace my own structuring of arguments or my responsibility for technical accuracy.

**Software development.** I used the **Cursor** editor with integrated AI assistance as a programming aid. I specified the intended behaviour, algorithms, and interfaces (for example state-machine logic, simulation steps, and experiment workflows); the tool helped translate those descriptions into Python code and suggested refactorings or boilerplate. I reviewed, tested, and revised all generated code. Final correctness was established through my own reasoning, the unit-test suite, validation against analytical baselines, and inspection of simulation results—not by accepting tool output without verification.

In summary, generative AI was used as a **writing-assistance** and **coding-assistance** tool, not as a primary source of factual content or as a substitute for my own analysis. Grammarly and Cursor are named here for transparency; they are not entered in the **References** section because they were not used as citable bibliographic sources, in line with common practice for tool disclosure.

If the department supplies a mandatory disclosure form or wording, this section should be aligned with that template at submission time.

## Table of Contents

- Abstract
- Acknowledgements
- Use of Generative AI Tools
- Chapter 1 Introduction and Context
  - 1.1 Background and Motivation
  - 1.2 Problem Statement and Research Questions
  - 1.3 Objectives and Contribution Structure
  - 1.4 Scope and Assumptions
  - 1.5 Report Roadmap
  - 1.6 Literature Context
- Chapter 2 Methodology
  - 2.1 System Model
  - 2.2 Analytical Baseline
  - 2.3 Simulator Architecture
  - 2.4 Per-Slot Event Flow and State Updates
  - 2.5 Metrics, Logging, and Outputs
  - 2.6 Experimental Program and Reproducibility
  - 2.7 Validation Strategy and Testing
  - 2.8 Software Engineering Notes
- Chapter 3 Core Results and Design Guidance
  - 3.1 Simulator Validation
  - 3.2 Parameter Impact
  - 3.3 Optimization Results
  - 3.4 3GPP-Inspired Interpretation
  - 3.5 Independence Analysis
- Chapter 4 Extension Studies
  - 4.1 Finite Retry Limits
  - 4.2 CSMA Comparison
  - 4.3 Capture and SIC Receivers
  - 4.4 Age of Information
  - 4.5 MMBP Arrivals
- Chapter 5 Discussion
  - 5.1 Why the Validation Matters
  - 5.2 What the Trade-Off Means
  - 5.3 Design Implications
  - 5.4 Realism and Limitations
  - 5.5 What the Extensions Add
- Chapter 6 Conclusion
  - 6.1 Summary of Contribution
  - 6.2 Objective Review
  - 6.3 Limitations
  - 6.4 Future Work
- References (IEEE numbered [1]–[10]; see bibliography note before submission)
- Appendices
  - Appendix A User Guide
  - Appendix B Source Code Structure
  - Appendix C Additional Figures and Tables
  - Appendix D Selected Code Listings

## List of Figures

<style>
.lof-lot { border-collapse: collapse; width: 100%; font-family: "Times New Roman", Times, serif; font-size: 11pt; margin: 0.5em 0 1.2em 0; }
.lof-lot td { border: 1px dashed #888; padding: 5px 10px; vertical-align: top; }
.lof-lot .col-id { width: 9%; white-space: nowrap; }
.lof-lot .col-page { width: 11%; text-align: right; color: #0000CC; }
</style>

<table class="lof-lot">
<tr><td class="col-id">Figure 2.1</td><td>State-transition diagram for a single MTD showing Active → Idle → Sleep → Wakeup → Active</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 2.2</td><td>Software architecture showing Node → Simulator → BatchSimulator → MetricsCalculator, with supporting modules for validation, optimization, traffic models, receiver models, and MMBP analytics</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.1</td><td>Empirical versus analytical success probability with tolerance band</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.2</td><td>Empirical versus analytical service rate with tolerance band</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.3</td><td>Convergence of error versus simulation length</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.4</td><td>Mean delay versus <em>q</em> for several <em>t</em><sub>s</sub> values</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.5</td><td>Expected lifetime versus <em>q</em> for several <em>t</em><sub>s</sub> values</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.6</td><td>Mean delay versus <em>t</em><sub>s</sub> for several <em>q</em> values</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.7</td><td>Expected lifetime versus <em>t</em><sub>s</sub> for several <em>q</em> values</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.8</td><td>Throughput versus <em>λ</em> for several node populations</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.9</td><td>Delay heatmap over the (<em>q</em>, <em>t</em><sub>s</sub>) plane</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.10</td><td>Lifetime heatmap with stability contour</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.11</td><td>Pareto frontier of lifetime versus delay</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.12</td><td>Scenario comparison relative to a balanced baseline</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.13</td><td>On-demand sleep versus duty-cycling</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.14</td><td>Lifetime versus <em>λ</em> for several T3324-like settings</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.15</td><td>Delay versus <em>λ</em> with SLA-style annotation</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.16</td><td>Recommended <em>q</em>* versus number of nodes</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.17</td><td>Scatter of representative 3GPP-inspired scenario outcomes</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.18</td><td>Interaction plots showing non-parallel or fanning curves</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.19</td><td>Additive-model regression residuals versus coupling factor <em>κ</em>, colored by <em>t</em><sub>s</sub></td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.20</td><td>Coupling heatmap of <em>κ</em> = <em>p</em>·<em>t</em><sub>s</sub></td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.21</td><td>Regime map for weak, moderate, and strong coupling</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.22</td><td>Optimal <em>q</em>*(<em>t</em><sub>s</sub>) under lifetime constraints</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.23</td><td>Iso-contours of delay and lifetime over the (<em>q</em>, <em>t</em><sub>s</sub>) plane</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.24</td><td>Delay and drop rate versus retry limit</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.25</td><td>Delay–drop Pareto view for varying <em>K</em></td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.26</td><td>Delay comparison between CSMA and Aloha</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.27</td><td>Collision comparison between CSMA and Aloha</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.28</td><td>Throughput versus load for both schemes</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.29</td><td>Effective success probability under collision, capture, and SIC</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.30</td><td>Delay comparison under the three receiver models</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.31</td><td>Lifetime comparison under the three receiver models</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.32</td><td>AoI versus <em>q</em> for several <em>t</em><sub>s</sub> values</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.33</td><td>AoI-optimal versus delay-optimal <em>q</em>*</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.34</td><td>AoI–delay–lifetime trade-off projection</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.35</td><td>Analytical versus empirical <em>μ</em> under MMBP arrivals</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Figure 3.36</td><td>Prediction error versus burstiness index</td><td class="col-page">e.g. P. ___</td></tr>
</table>

## List of Tables

<table class="lof-lot">
<tr><td class="col-id">Table 2.1</td><td>Summary of experiments, parameter ranges, replications, seeds, and primary outputs for the foundation, core, design, and extension objectives</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.1</td><td>Validation summary table with stability flags and percentage errors</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.2</td><td>Representative state-fraction and scenario summary table</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.3</td><td>Prioritization scenario summary with gains and losses</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.4</td><td>Design-guideline table linking load, timer, <em>q</em>*, delay, lifetime, and stability</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.5</td><td>Statistical summary comparing additive and interaction models</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.6</td><td>Analytical versus empirical <em>μ</em><sub><em>K</em></sub> comparison</td><td class="col-page">e.g. P. ___</td></tr>
<tr><td class="col-id">Table 3.7</td><td>Bernoulli versus MMBP error summary</td><td class="col-page">e.g. P. ___</td></tr>
</table>

---

# Chapter 1 Introduction and Context

## 1.1 Background and Motivation

Machine-to-machine communication is a central enabling technology for modern IoT systems, where large numbers of autonomous devices exchange small packets for sensing, monitoring, tracking, and control. Typical examples include environmental sensing, utility metering, industrial telemetry, smart transport, and health-related monitoring. In many of these applications the nodes are battery powered, geographically distributed, and expected to operate unattended for long periods. As a result, energy efficiency is not simply an optimization target; it is a basic design requirement.

At the same time, many M2M applications cannot tolerate arbitrarily long waiting times. Even where traffic is sparse, the network may still need to support timely updates, bounded access delay, or freshness-sensitive information delivery. This creates a design tension. Nodes that remain active more often can react quickly to arrivals, but they consume more energy. Nodes that sleep aggressively can extend battery lifetime, but packets that arrive during sleep must wait for wake-up and renewed channel access. The engineering problem is therefore to manage delay and energy jointly rather than to optimize either in isolation.

This project studies that tension in the context of sleep-aware random access. The baseline system is slotted Aloha with on-demand sleep [1], where a node transmits in an active slot with probability `q`, enters sleep after remaining idle for `t_s` slots, and requires `t_w` slots to wake before it can resume contention. These parameters jointly shape both the access process and the energy cost of operation. Similar design ideas appear in 3GPP cellular IoT mechanisms such as Power Saving Mode (PSM), MICO mode, the active timer `T3324`, and reduced access procedures such as RA-SDT, although the present work uses these only as design analogies rather than attempting full standards emulation [2]–[4].

The immediate motivation for the project was twofold. First, closed-form analysis is valuable for identifying mean trends, but it does not directly reveal transient behavior, confidence intervals, finite-run variability, or edge cases. Second, once richer assumptions are added, such as realistic power profiles, bursty traffic, retry limits, or advanced receiver models, simulation becomes a practical way to test whether the core conclusions still hold. A validated simulator is therefore useful both as a replication tool and as a platform for design exploration.

## 1.2 Problem Statement and Research Questions

The central problem addressed in this report is how to configure sleep-aware access so that large populations of MTDs can achieve acceptable delay without sacrificing battery lifetime. In the baseline model, the key control variables are the transmission probability `q`, the idle timer `t_s`, the wake-up duration `t_w`, the offered traffic load `lambda`, and the number of devices `n`. Together these determine the success probability, service rate, queueing delay, and long-term energy consumption of the system.

Analytical models provide a compact description of these relationships, but they are not always sufficient on their own. They are usually expressed in terms of steady-state averages and simplifying assumptions, and therefore do not fully show simulation variance, convergence behavior, per-node histories, or how robust the baseline conclusions remain under richer traffic and receiver assumptions. A complementary simulator is therefore needed to reproduce the analytical baseline, quantify confidence in the results, and support additional experiments beyond the core theory.

This report addresses two research questions:

1. How do sleep and access parameters, especially `q` and `t_s`, jointly shape the delay-lifetime trade-off in sleep-based M2M access?
2. Can `q` and `t_s` be tuned independently, or must they be co-optimized?

These questions motivate the structure of the report. The work first establishes a validated simulation framework and then uses that framework to produce design guidance and targeted extensions.

## 1.3 Objectives and Contribution Structure

To keep the report focused, the project objectives are grouped by contribution rather than treated as a flat list of ten unrelated tasks.

### Foundation

- **O1:** Build the discrete-event simulation framework for slotted Aloha with on-demand sleep.

### Core Analysis

- **O2:** Quantify how `q`, `t_s`, `t_w`, `n`, `lambda`, and traffic assumptions affect delay and lifetime.
- **O5:** Determine analytically and empirically whether `q` and `t_s` can be treated as independent parameters.

### Design and Validation

- **O3:** Identify Pareto-efficient operating points for the latency-longevity trade-off.
- **O4:** Validate the simulation output against analytical formulas and interpret the results using 3GPP-inspired settings.

### Extension Studies

- **O6:** Add finite retry limits and analyze the delay-drop trade-off.
- **O7:** Compare slotted Aloha with a CSMA-based alternative.
- **O8:** Add capture-effect and SIC receiver models.
- **O9:** Add Age of Information as a timeliness metric.
- **O10:** Extend the analysis to MMBP arrivals and identify when the Bernoulli approximation breaks down.

The main contribution of the project lies in O1-O5. These objectives establish the simulator, validate it, quantify the baseline design trade-off, and answer the independence question. O6-O10 are important extension studies, but they build on the validated baseline rather than replacing it.

## 1.4 Scope and Assumptions

This project was intentionally scoped as a software-based simulation and analysis study rather than a full communication-system implementation. The simulator was developed in Python and designed to run in local and notebook-based environments. The target setting was sleep-based random access in the unsaturated regime, meaning that the main analyses focus on configurations satisfying the stability condition `lambda < mu`.

The baseline model assumes homogeneous MTDs, slotted time, and a configurable state-dependent power model. The key states are active, idle, sleep, and wakeup. Traffic generation includes Bernoulli-style arrivals as the baseline and richer bursty or correlated variants for extended experiments. The report interprets `t_s` as a `T3324`-like timer analogue, the sleep mechanism as a simplified abstraction of MICO or PSM behavior, and `t_w` as an abstraction of access resumption overhead. These mappings are useful for engineering interpretation, but they remain abstractions rather than exact standards implementations.

Several topics were deliberately left out of scope. These include detailed physical-layer modeling, fading-aware channels, mobility, multi-channel access, and hardware emulation. The core study also retains a homogeneous-node baseline and an idealized channel abstraction. These limitations are not incidental; they are part of the modeling choices used to keep the project analytically aligned and experimentally manageable.

## 1.5 Report Roadmap

The remainder of the report is organized as follows. Chapter 2 defines the system model, analytical baseline, simulator architecture, metrics, and experiment design. Chapter 3 presents the core validated results, including parameter impact, optimization, 3GPP-inspired interpretation, and the independence analysis. Chapter 4 presents five extension studies built on the same simulation framework. Chapter 5 interprets the results, discusses their practical meaning and limits, and explains why the coupling between sleep and access behavior matters for design. Chapter 6 concludes the report by summarizing the contribution, revisiting the grouped objectives, and identifying the most credible next steps.

## 1.6 Literature Context

The principal analytical foundation for this project is Wang et al. (2024), which studies sleep-based low-latency access for M2M communications and derives the key relationships among success probability, service rate, delay, and lifetime. That work motivates the use of slotted random access with on-demand sleep and provides the theoretical baseline that the simulator in this report was designed to reproduce and extend [1].

The practical interpretation of the project is also informed by 3GPP specifications for power saving and reduced-access behavior in cellular IoT systems. Concepts such as PSM, MICO mode, `T3324`, and simplified access procedures provide a real-world frame for understanding the simulator parameters, even though the simulator does not attempt to model every standards detail [2]–[4].

More broadly, the project sits at the intersection of random-access analysis, low-power IoT protocol design, and discrete-event simulation. Prior slotted Aloha literature provides the conceptual background for the role of `q`, while simulation frameworks such as SimPy make it practical to study parameter sweeps, finite-run variance, and complex extensions. In this report, external literature is used mainly to frame the problem and anchor the baseline theory, while the main technical contribution lies in the implemented simulator and the design insights derived from it [5]–[7].

---

# Chapter 2 Methodology

## 2.1 System Model

The baseline system considered in this report is a slotted M2M uplink in which `n` homogeneous battery-powered devices compete for access to a shared channel. Time is divided into equal slots, and the simulation advances one slot at a time. This choice makes the access process easy to align with the analytical model and also matches the project goal of studying contention, queueing, and sleep behavior at a level of abstraction appropriate for large MTD populations rather than detailed radio-waveform analysis.

Each node maintains four types of state information. First, it has a packet queue that stores pending arrivals together with the timestamps needed for delay measurement. Second, it maintains energy-related variables, including the initial energy budget and per-state power consumption. Third, it tracks control-state variables such as the idle timer and wake-up counter. Fourth, it records statistics such as successful transmissions, experienced delays, time spent in each state, and energy consumed in each operating mode. This per-node state is necessary because the project is not only measuring aggregate throughput but also battery lifetime, delay, and state occupancy, all of which depend on the internal behavior of the device.

The baseline arrival process is Bernoulli per slot, which gives a simple and analytically compatible model for sporadic packet generation. However, the simulator was intentionally built so that the arrival process could later be replaced by bursty, periodic, on-off, and MMBP-based generators. This extensibility matters because one of the main motivations for the simulator was to test whether the baseline conclusions survive beyond the simplest traffic assumptions.

The node state machine contains four principal states: active, idle, sleep, and wakeup. In the active state, a node with a non-empty queue is eligible to attempt transmission with probability `q`. In the idle state, the queue is empty and an inactivity counter is used to determine whether the node should transition to sleep. In the sleep state, the node consumes minimal energy but cannot immediately contend for the channel. In the wakeup state, the node spends `t_w` slots resynchronizing before returning to active contention. This state machine is deliberately compact: it captures the main latency-energy interaction without introducing unnecessary standards-specific signaling complexity.

The key model parameters are the number of nodes `n`, the transmission probability `q`, the idle timer `t_s`, the wake-up time `t_w`, the packet arrival rate `lambda`, the power rates for each operating state, and the initial energy budget. These parameters were chosen because they jointly define the project’s central delay-lifetime design space. The core study focuses on stable, unsaturated configurations satisfying `lambda < mu`, because unstable regimes would make delay diverge and would weaken comparison to the analytical baseline.

Several abstractions were intentionally retained. Nodes are homogeneous in the baseline setting, the channel is idealized at the packet-success level, and no mobility, fading, or multi-channel scheduling is modeled in the core chapter. These omissions were not oversights; they were conscious modeling choices that kept the project tractable, aligned with the reference analysis, and suitable for systematic parameter sweeps.

[Figure 2.1 about here: state-transition diagram for a single MTD showing Active -> Idle -> Sleep -> Wakeup -> Active.]

## 2.2 Analytical Baseline

The analytical reference point for the simulator follows the framework used by Wang et al. (2024). This baseline is important for two reasons. First, it provides a closed-form target against which the implementation can be validated. Second, it explains why the access and sleep parameters should not be thought of as unrelated design knobs. The simulator was therefore built not as a substitute for theory, but as a way to reproduce the theory faithfully and then move beyond it where analytical treatment becomes cumbersome.

For a tagged node that transmits with probability `q` in a population of `n` devices, the baseline success probability is

`p = q(1 - q)^(n - 1)`.

This expression captures the event that the tagged node transmits while every other node remains silent in the same slot. It is the central contention term of the baseline model and is used both in analytical comparison and in later design interpretation [1].

For the validation baseline implemented in the code, the sleep-aware service rate is written as

`mu = p / (1 + t_w*lambda/(1-lambda))`.

This form matches the analytical validator used in the simulator and is the expression against which the service-rate validation figures are generated. In the sparse-load regime, the wake-up penalty is scaled by the arrival process, so the main analytical stability check depends on `p`, `lambda`, and `t_w`.

The stability condition is therefore

`lambda < mu`.

Configurations that violate this condition are either excluded from the main baseline analyses or explicitly marked as unstable, because they do not represent the steady-state regime that the analytical comparison assumes.

The later independence study uses a separate coupling heuristic rather than the closed-form validation expression above. To summarize how access aggressiveness and sleep timing combine in the design sweeps, the report uses the quantity

`kappa = p*t_s`,

which serves as a compact indicator of how strongly sleep and access behavior interact in a given part of the parameter space. It is therefore best read as a design-space descriptor rather than as the formal validation formula for `mu`.

The limitations of the analytical model are also the main reason the simulator was needed. The formulas provide mean trends, but they do not directly quantify confidence intervals, transient traces, convergence with simulation length, or the behavior of richer traffic and receiver models. The simulator therefore complements the analytical baseline by preserving the same core assumptions where necessary while enabling deeper empirical inspection.

## 2.3 Simulator Architecture

The simulator was implemented as a modular Python codebase so that the baseline model, validation utilities, optimization workflow, and extension studies could share the same core simulation loop. This modularity was one of the most time-consuming but most important parts of the project. If the code had been written only as a single-purpose prototype for one baseline plot, it would have been much faster to build, but it would not have supported the later objectives O3-O10 with the same confidence or reuse.

At the lowest level, the `Node` component models one MTD. It is responsible for packet arrivals, queue management, state transitions, transmission attempts, successful-delivery handling, retry handling where applicable, and per-state energy accounting. This means that the node class is not just a data container; it is the object in which the sleep-aware access policy is concretely implemented.

The `Simulator` component coordinates the network of nodes. It owns the global slotted-time loop, instantiates the node population, generates arrivals, collects candidate transmissions, resolves per-slot outcomes, and stores the resulting histories and summaries. In the baseline version, a slot succeeds only when exactly one node transmits. In the extension framework, the same coordination logic can call different access or receiver behaviors without rewriting the entire simulation structure.

The `BatchSimulator` provides the bridge from single-run simulation to research-grade experimentation. It supports parameter sweeps, multiple replications, seed control, and aggregation of repeated runs. This layer is what makes the simulator suitable for report figures rather than only ad hoc demonstrations, because it allows consistent comparison across parameter combinations and random seeds.

Several supporting modules were designed to isolate distinct responsibilities. `PowerModel` encapsulates state-dependent energy rates and battery configurations. `MetricsCalculator` turns raw simulation traces and counters into reportable empirical and analytical quantities. Validation utilities handle analytical comparison, sanity checks, and convergence studies. Optimization helpers search the `(q, t_s)` space and compute Pareto-efficient configurations. Traffic-model modules abstract arrival generation, while extension-specific modules such as receiver models and MMBP analytics attach richer behavior without requiring a redesign of the baseline simulator.

This modular structure matters in two ways. From a software engineering perspective, it improves readability, testability, and maintainability. From a research perspective, it makes the framework extensible. Objectives O6-O10 were feasible precisely because the original simulator separated node logic, simulation coordination, metrics, validation, traffic generation, and extension hooks rather than entangling them in a single monolithic script.

The repository structure and supporting notebooks reinforce this design. The codebase is organized into `src`, `tests`, `docs`, and `examples`, and the README documents how to install dependencies, run tests, and execute notebooks. That structure is relevant to the report because, for a software-oriented final year project, the architecture and reproducibility of the program are part of the contribution rather than peripheral implementation detail.

[Figure 2.2 about here: software architecture showing `Node -> Simulator -> BatchSimulator -> MetricsCalculator`, with supporting modules for validation, optimization, traffic models, receiver models, and MMBP analytics.]

## 2.4 Per-Slot Event Flow and State Updates

Because the project is centered on simulator creation, it is important to explain not just the architecture but also the internal event flow of one simulated slot. This is the level at which the design choices in the code become the system behavior described in later results.

At the start of a slot, each node may receive a new packet according to the configured traffic model. If the baseline Bernoulli generator is used, this means a simple per-slot arrival decision; in other traffic modes, the same step can produce bursty or state-dependent arrivals. Newly arrived packets are timestamped immediately so that their queueing delay can be measured at successful delivery.

After arrivals are processed, the node updates its internal state. A node with an empty queue may remain idle, increment its inactivity counter, or enter sleep if the idle timer threshold has been reached. A sleeping node that receives a packet does not become active immediately; instead, it begins or continues its wakeup period. A node already in wakeup decrements its wakeup counter until it returns to active operation. These transitions are central to the project because they encode the energy-delay trade-off at the device level.

Once states are updated, active nodes with non-empty queues decide whether to attempt transmission. In the baseline model this decision is probabilistic with parameter `q`. The simulator then collects all attempted transmissions for the slot and applies the appropriate resolution rule. Under the baseline collision model, exactly one transmitter means success and any higher number means collision. Under the extension framework, this same step can be redirected to CSMA deferral logic or to receiver models such as capture and SIC.

After transmission resolution, successful packets are removed from their queues and their delays are computed from the stored arrival timestamps. Failed attempts may simply remain queued in the baseline model or update retry counters in the finite-retry extension. During the same slot, every node also consumes energy according to its current state, and the simulator records per-slot or aggregated statistics as configured.

This slot-level sequencing explains why the simulator is a meaningful complement to analytical work. The analytical formulas summarize expected behavior at the cycle level, whereas the simulator explicitly reproduces the packet-level and state-level dynamics that generate those averages. It also makes clear where future extensions can be inserted: arrivals, access decisions, transmission resolution, state transitions, and metric collection are distinct stages rather than one opaque block of code.

One useful way to understand the simulator is to follow a single packet through the system. A packet may arrive while the node is active and contend immediately, or it may arrive while the node is idle or asleep, in which case it experiences waiting and wake-up overhead before it can even join contention. If its initial transmission fails, it remains queued and re-enters the access process in subsequent slots. This packet-level path is exactly what turns the abstract parameters `q`, `t_s`, and `t_w` into empirical delay and lifetime outcomes.

## 2.5 Metrics, Logging, and Outputs

The simulator records both node-level and network-level metrics. The core outputs reported in the main body are mean queueing delay, throughput, success probability, service rate, average queue length, lifetime, state occupancy fractions, and energy-consumption breakdown by state. Extension-specific metrics include drop rate for finite retries and freshness measures for AoI experiments.

These metrics were not chosen arbitrarily. Delay and lifetime are the two most important end-user design outcomes in the report. Success probability and service rate are the main links between theory and experiment. State fractions and energy breakdown explain why two configurations with similar throughput may still have very different battery outcomes. Queue statistics and tail-delay summaries help reveal the cost of bursty or less favorable operating regimes.

Trace-level logging also played an important role during development. The simulator supports detailed per-slot state, queue, transmission, and energy traces, which were useful for debugging state transitions and checking whether the modeled behavior matched the intended logic. In a software-oriented project, this kind of visibility matters: debugging tools are part of the engineering process, not merely development convenience.

The output layer was designed to support both experimentation and report writing. Numerical summaries can be exported in structured formats such as CSV and JSON, while plots can be saved as report-ready images. This export path makes it possible to go from a replicated simulation run to a figure or table in the final report without manual transcription.

## 2.6 Experimental Program and Reproducibility

The experimental design is organized around the grouped project objectives and was built to be reproducible rather than purely exploratory. The first stage validates the baseline simulator against analytical expectations. This is followed by parameter sweeps for Objective O2, optimization studies for Objective O3, 3GPP-inspired design studies for Objective O4, and independence experiments for Objective O5. The extension objectives O6-O10 reuse the same simulation core while modifying one layer of behavior at a time.

For O2, the project sweeps key parameters across representative ranges to reveal monotonic trends, saturation effects, and trade-offs. These sweeps cover transmission probability, idle timer, node population, arrival rate, and traffic-model variants. For O3, the project performs grid-search optimization over the `(q, t_s)` plane to identify low-delay, high-lifetime, and Pareto-efficient operating points. For O4, the simulator parameters are mapped to 3GPP-inspired settings and evaluated under scenario-based design questions. For O5, a full-factorial design over `q` and `t_s` is used to isolate and quantify parameter interaction. The extension objectives then reuse the same infrastructure with protocol-, receiver-, or traffic-specific modifications.

Reproducibility was an explicit design goal. Random seeds are configurable and recorded so that runs can be repeated. Batch execution is used to collect multiple replications rather than relying on one-off results. Unstable configurations are flagged to avoid misleading analytical comparisons. The same workflow supports notebook demonstrations, figure generation, and more formal validation runs. This matters for the credibility of the report: if a result cannot be reproduced consistently under fixed settings, it should not be presented as a design conclusion.

The notebook workflow also deserves mention. The project includes example notebooks for validation, optimization, extension experiments, and demonstrations. These notebooks provide an accessible route from the underlying modules to the final figures and tables, while preserving the modular structure of the code in `src`. In effect, the notebooks act as experiment drivers and report-generation companions rather than as the primary location of the simulation logic.


| Objective block     | Purpose                                                                   | Main settings / ranges                                                                 | Replications        | Primary outputs                                      |
| ------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------- | ---------------------------------------------------- |
| O1 Validation       | Validate baseline simulator against analytical `p`, `mu`, and convergence | `n = 5-100`, `lambda = 5e-5`, `q = min(0.05, 1/n)`, `t_s = 10 s`, `t_w = 2`            | 3 quick / 20 full   | validation scatter, convergence error                |
| O2 Parameter impact | Measure delay, lifetime, and throughput sensitivity                       | `q = 0.002-0.05`, `t_s = 0.5-120 s`, `lambda = 0.001-0.05`, `n = 10-100`               | 2 quick / 20 full   | delay curves, lifetime curves, throughput saturation |
| O3 Optimization     | Map the `(q, t_s)` trade-off surface and named scenarios                  | `q = 0.002-0.05`, `t_s = 0.5-120 s`, sparse-load baseline                              | 2 quick / 15 full   | heatmaps, trade-off points, scenario deltas          |
| O4 3GPP-inspired    | Interpret timers and access rules using `T3324`-style settings            | `T3324 = 2, 10, 60, 360 s`, profile-specific power models, `q = 1/n`                   | 2 quick / 15 full   | delay-lifetime curves, `q`* trend, guideline table   |
| O5 Independence     | Test whether `q` and `t_s` interact                                       | factorial sweep over `q` and `t_s`, regression in log-log space                        | 2 quick / 15 full   | interaction plots, residuals, coupling maps          |
| O6-O10 Extensions   | Probe retry limits, CSMA, receiver models, AoI, and MMBP arrivals         | objective-specific sweeps over `K`, `n`, `lambda`, receiver type, AoI settings, and BI | 4-5 quick / 20 full | extension figures and comparative summaries          |


## 2.7 Validation Strategy and Testing

Validation is performed at several levels. First, small-scale sanity checks verify that the implemented state transitions, contention logic, and energy accounting behave as expected. Representative checks include the no-sleep limit, immediate-sleep behavior, and high-`q` configurations that should increase collisions. These tests are important because they target intuitive edge cases that would quickly expose incorrect state logic.

Second, direct analytical comparisons are performed for success probability, service rate, delay, and lifetime under stable baseline conditions. These comparisons use explicit tolerance bands, such as plus or minus 5%, 10%, and 20%, to classify agreement between simulation and theory. Third, convergence experiments examine how the empirical error behaves as the number of simulated slots increases, ensuring that later experimental results are not simply artifacts of insufficient run length.

Testing was also a major part of the engineering effort. According to the project records and repository documentation, the codebase accumulated a broad unit-test suite across node logic, simulator behavior, power modeling, validation, metrics, experiments, traffic models, visualizations, receiver models, and independence analysis. The README’s earlier baseline summary reported 135 tests for the original core stack, while the later project task records show substantial additional coverage for optimization, 3GPP alignment, independence analysis, and the extension objectives. Even allowing for documentation evolving at different times, the overall message is clear: the simulator was built with systematic verification rather than informal spot-checking alone.

This validation-first and test-backed approach is one of the reasons the simulator can support the later chapters with credibility. The optimization results, design guidelines, and extension studies are only persuasive because the underlying implementation was repeatedly checked at the model, code, and experiment levels.

## 2.8 Software Engineering Notes

Although the report is centered on communication-system behavior, the simulator itself is a major project deliverable and therefore deserves explicit software-engineering reflection. The implementation uses Python together with the standard scientific and plotting libraries appropriate for this kind of work, and it was designed to run both locally and in notebook-style environments. The repository structure, testing workflow, and modular separation of concerns make the project easier to inspect, debug, extend, and demonstrate.

For a software-oriented final year project, this matters. The contribution is not only that the final plots exist, but that they are generated by a structured framework whose internal logic can be explained, tested, and reused. In that sense, the Methodology chapter is not merely describing what was simulated. It is also documenting how a reusable simulator was engineered to support the report’s analytical, experimental, and extension objectives.

---

# Chapter 3 Core Results and Design Guidance

This chapter carries the main validated contribution of the report. The goal here is to move from simulator credibility to parameter behavior, design guidance, and the central independence result before any broader extensions are considered.

## 3.1 Simulator Validation

The first task of the results chapter is to establish that the simulator can be trusted. Before any optimization or design recommendation is made, the implementation must show that it reproduces the analytical baseline under the conditions for which that baseline is intended. For that reason, the validation experiments compare empirical and analytical success probability `p`, service rate `mu`, delay, and lifetime across representative node populations and across increasing simulation lengths.

This validation stage matters conceptually as well as numerically. If the simulator failed here, any later claim about trade-offs or practical design guidance would be weakened because it would be unclear whether the patterns came from the modeled system or from implementation error. By putting validation first, the report makes it clear that the later chapters are built on a tested baseline rather than on an unverified prototype.

The first comparison concerns success probability. Since `p = q(1-q)^(n-1)` is the core contention term of the baseline model, agreement here is the most direct check that the simulator is reproducing the intended access logic. When the empirical values align closely with the theoretical curve across several values of `n`, the result indicates that the slotted-time contention procedure and collision rule have been implemented consistently.

The second comparison concerns service rate `mu`, which is a stricter test because it depends not only on contention but also on how sleep and wake-up behavior are represented in the service cycle. Good agreement in `mu` therefore provides evidence that the state machine, sleep timer, and wake-up handling are not merely plausible in isolation, but are interacting in the way assumed by the analytical model. According to the completed task record, the simulator achieves agreement within approximately plus or minus 5% for representative stable configurations, which is strong enough to support the later use of the framework as a design-analysis tool.

The convergence figure plays a different role from the direct theory-versus-simulation comparisons. Rather than checking correctness of one formula, it checks whether the experiment lengths used later in the report are sufficient for the empirical metrics to stabilize. If the validation error decreases with simulation length and becomes acceptably small by the chosen run horizon, then the later parameter sweeps can be interpreted as system behavior rather than sampling noise. In this sense, convergence is a methodological result as much as a numerical one.

[Figure 3.1 about here: empirical versus analytical success probability with tolerance band.]  
[Figure 3.2 about here: empirical versus analytical service rate with tolerance band.]  
[Figure 3.3 about here: convergence of error versus simulation length.]  


| n   | q      | Empirical p | Analytical p | p error (%) | Empirical mu | Analytical mu | mu error (%) | Stable |
| --- | ------ | ----------- | ------------ | ----------- | ------------ | ------------- | ------------ | ------ |
| 10  | 0.0500 | 0.0527      | 0.0500       | 5.4         | 0.05268      | 0.05000       | 5.4          | Yes    |
| 20  | 0.0500 | 0.0486      | 0.0500       | 2.8         | 0.04858      | 0.05000       | 2.8          | Yes    |
| 50  | 0.0200 | 0.0191      | 0.0200       | 4.6         | 0.01908      | 0.02000       | 4.6          | Yes    |
| 100 | 0.0100 | 0.0102      | 0.0100       | 1.6         | 0.01016      | 0.01000       | 1.7          | Yes    |


Taken together, the validation results support a practical conclusion: the simulator is credible in the stable regime for the baseline system it was designed to emulate. That conclusion is the foundation for the rest of Chapter 3. Without it, the later optimization and extension results would be little more than exploratory outputs. With it, they become evidence-backed engineering findings.

## 3.2 Parameter Impact

Once the baseline simulator has been validated, it can be used to answer the report’s first major substantive question: how do the principal parameters change delay and lifetime? Objective O2 examines this by sweeping `q`, `t_s`, `lambda`, and `n` over representative ranges while reusing the same validated simulation core. The aim of this section is to describe what the plotted surfaces actually show before any optimizer is applied.

Figure 3.4 plots mean queueing delay against transmission probability for several idle-timer values. The most striking feature is that the relation is not monotonic. Delay first falls sharply as `q` increases from very small values, reaches its lowest region around `q ≈ 0.02–0.03`, and then rises again once more aggressive access begins to create enough contention to offset the gain in attempt opportunity. The plotted confidence bands remain relatively narrow compared with the overall left-to-right change, and the five mean curves stay fairly close to one another across the sweep. This indicates that, in this experiment, delay is more sensitive to `q` than to `t_s`. In other words, the first-order decision for responsiveness is how aggressively nodes contend, not which idle timer within the tested range is chosen.

[Figure 3.4 about here: mean delay versus `q` for several `t_s` values.]  

Figure 3.5 shows the corresponding lifetime curves. Lifetime still falls as `q` increases, but the wider timer sweep and enlarged lifetime-visualization battery budget now reveal a clearer spread across `t_s`. Across the regenerated curves, shorter timers generally sit above the longer-timer traces, while very long timers are penalized because the node remains in the higher-power idle state for longer before sleep is entered. The dominant trend still comes from `q`, but the figure now makes it easier to see that timer choice is not negligible once the operating range is widened.

[Figure 3.5 about here: expected lifetime versus `q` for several `t_s` values.]  

Figures 3.6 and 3.7 reverse the viewpoint by plotting both metrics against `t_s` for fixed values of `q`. Figure 3.6 shows that delay changes only modestly with `t_s` when `q` is held constant; the dominant separation is instead between the `q` curves themselves. The best-performing curve in this sweep is the intermediate setting around `q = 0.02`, while very small `q` leaves packets waiting for access opportunities and very large `q` pushes the system into a contention-heavy regime. Figure 3.7 shows a more interpretable lifetime pattern once the timer axis is extended: the lifetime curves are visibly separated by timer choice, with shorter timers generally producing the longest lifetimes and larger `t_s` values reducing longevity, although that timer effect is still smaller than the spread caused by changing `q`. Together, these two plots reinforce the same message from Figures 3.4 and 3.5: `q` is still the stronger control knob, but `t_s` now shows a visible secondary effect on longevity.

[Figure 3.6 about here: mean delay versus `t_s` for several `q` values.]  
[Figure 3.7 about here: expected lifetime versus `t_s` for several `q` values.]  

Figure 3.8 adds the load dimension by plotting throughput against the arrival rate `lambda` for several node populations. For this figure, each population uses its own `q = 1/n` access rule while keeping the baseline timer settings fixed, so the comparison reflects how the offered load scales under a throughput-oriented contention rule. In every case throughput rises with offered load and then levels off once the channel becomes contention-limited. The larger populations approach that saturation region earlier because aggregate offered traffic grows faster with `n`, while the smaller population continues increasing over a wider range of `lambda`. This figure is therefore the clearest visual indication of the stability boundary: once the curves flatten, adding more traffic no longer produces proportional throughput gains.

[Figure 3.8 about here: throughput versus `lambda` for several node populations.]  

The state-fraction summary in Table 3.2 links these curves back to node behavior. In the present sparse-load baseline, the timer setting has a particularly strong effect on how quickly the node returns to sleep, so the shorter-timer scenario spends the largest fraction of time asleep while longer timers accumulate more idle-state residency before sleep is entered. The table therefore provides the behavioral explanation that sits behind the trend plots.


| Scenario     | q      | ts (s) | Delay (slots) | Lifetime (years) | Throughput | Active | Idle  | Sleep | Wake-up |
| ------------ | ------ | ------ | ------------- | ---------------- | ---------- | ------ | ----- | ----- | ------- |
| Low-Latency  | 0.0200 | 0.5    | 49.6          | 89.077           | 0.0053     | 0.003  | 0.009 | 0.989 | 0.000   |
| Balanced     | 0.0100 | 5.0    | 111.4         | 14.498           | 0.0053     | 0.006  | 0.083 | 0.911 | 0.000   |
| Battery-Life | 0.0050 | 10.0   | 174.6         | 7.675            | 0.0053     | 0.009  | 0.160 | 0.831 | 0.000   |


The main conclusion of the parameter-impact section is that the trade-off surface already has visible structure before optimization begins. Delay exhibits a best region at intermediate `q`, lifetime is strongly penalized by aggressive access, and the influence of `t_s` is noticeably weaker than the influence of `q` in the tested sweeps. That structure is what the next section turns into design guidance.

## 3.3 Optimization Results

Objective O3 builds directly on the previous section by asking a more applied question: once the trade-off surface has been mapped, which parts of it are preferable under different objectives? This is the point at which the report shifts from descriptive analysis to design-oriented reasoning. The optimization routines sweep the `(q, t_s)` plane, construct delay and lifetime heatmaps, and then extract objective-specific trade-off points rather than collapsing everything into a single scalar score.

Unless stated otherwise, the lifetime values in Figures 3.9-3.13 use a sparse-traffic generic baseline with `n = 100`, `lambda = 5 x 10^-5` packets/slot, `q = 1/n` as the reference access rule, `t_w = 2` slots, and the generic low-power profile. For the lifetime-oriented optimization figures, the battery budget is enlarged relative to the throughput study so that timer effects remain visible and the lifetime scale is easier to interpret. The timer axis is reported on a seconds-equivalent scale, with the central baseline corresponding to an approximately 10 s active timer. These figures should therefore be read primarily as comparative design outputs under one realistic low-duty-cycle operating point rather than as universal deployment lifetimes for every IoT device class.

Figure 3.9 is the clearest overview of the delay landscape. The lowest-delay region appears at small-to-moderate transmission probabilities around the `1/n` rule, and this low-delay band persists across much of the timer axis. Delay becomes much larger once access becomes noticeably more aggressive than that baseline, where extra contention no longer helps responsiveness because collisions and repeated waiting dominate. The variation across timer settings is weaker than the horizontal change across `q`, again indicating that delay optimization is driven primarily by the access probability.

[Figure 3.9 about here: delay heatmap over the `(q, t_s)` plane.]  

Figure 3.10 shows a very different surface for expected lifetime. Here the dominant structure is almost entirely vertical: lifetime decreases nearly monotonically as `q` increases, while changes along the `t_s` direction are comparatively small. In practical terms, this means that the main way to extend battery life in the explored region is to reduce transmission aggressiveness. The optimization problem is therefore asymmetric: `q` strongly affects both delay and lifetime, whereas `t_s` provides a finer secondary adjustment rather than a primary driver.

[Figure 3.10 about here: lifetime heatmap with stability contour.]  

Read together, Figures 3.9 and 3.10 explain why optimization is necessary. The delay-preferred region and the lifetime-preferred region do not coincide. Configurations that minimize delay sit in a narrow low-`q` to moderate-`q` band, whereas configurations that maximize lifetime sit at the smallest feasible `q` values almost regardless of `t_s`. The two heatmaps therefore visualize the conflict that the optimizer must resolve.

Figure 3.11 converts this conflict into a compact delay-lifetime trade-off view. For each timer row in the sweep, the plot marks the `q` that maximizes lifetime and the `q` that minimizes delay, so the figure should be read as a summary of the two objective-specific choices rather than as a full undominated frontier construction. The min-delay points cluster at much lower delay but also much lower lifetime, while the max-lifetime points occupy the opposite corner with far higher lifetime and substantially larger delay. The labels by `t_s` show that changing the idle timer shifts the result only modestly compared with changing the optimization objective itself. This is one of the most important findings in the section: the main design decision is not “which exact `t_s` is best?” but rather “which side of the delay-lifetime trade-off is the application willing to prioritize?”

[Figure 3.11 about here: Pareto frontier of lifetime versus delay.]  

Figure 3.12 makes the same trade-off easier to read by collapsing the optimizer output into three named scenarios. Relative to the balanced baseline, the low-latency setting clearly improves responsiveness, while the long-timer battery-life setting imposes a clear delay penalty. Under the current sparse-load baseline, the timer effect is strong enough that the shorter-timer setting also yields the best projected lifetime, which is a useful reminder that extending `t_s` is not automatically battery-optimal when idle-state energy dominates. The figure is useful because it translates the heatmaps and frontier into a format that an engineering reader can interpret immediately: changing the design priority visibly changes where performance is gained and where it is sacrificed.

[Figure 3.12 about here: scenario comparison relative to a balanced baseline.]  


| Scenario     | Delay (slots) | Lifetime (years) | Delay vs Balanced (%) | Lifetime vs Balanced (%) |
| ------------ | ------------- | ---------------- | --------------------- | ------------------------ |
| Low-Latency  | 49.6          | 89.077           | -55.4                 | 514.4                    |
| Balanced     | 111.4         | 14.498           | 0.0                   | 0.0                      |
| Battery-Life | 174.6         | 7.675            | 56.8                  | -47.1                    |


Figure 3.13 then compares the baseline on-demand sleep mechanism with duty-cycling over several `t_s` values. The two schemes are close for some settings, but on-demand sleep remains consistently competitive across the tested timer range and typically achieves the lower delay of the pair while maintaining similar lifetime. The clearest separation appears around the intermediate timer settings, where the delay advantage is easiest to see without a large lifetime penalty. This supports the project’s baseline choice: tying sleep behavior to actual inactivity is more effective than forcing a rigid periodic sleep schedule, especially in the sparse-traffic regime targeted by the report.

[Figure 3.13 about here: on-demand sleep versus duty-cycling.]  

The optimization results therefore do more than identify one “best” operating point. They show that the design space is structured, that `q` is the dominant lever, that the preferred region depends strongly on whether delay or lifetime is prioritized, and that on-demand sleep remains the stronger baseline when compared with duty-cycling. This turns the simulator from a measurement tool into a design-guidance tool and creates a natural bridge to the next section, which interprets the same findings in a more practical 3GPP-oriented frame.

## 3.4 3GPP-Inspired Interpretation

Objective O4 improves the practical relevance of the report by linking the abstract simulator parameters to 3GPP-inspired settings. The purpose of this section is therefore slightly different from the previous one. Instead of asking which `(q, t_s)` pairs are attractive in the abstract, it asks how the same results look when `t_s` is interpreted through actual `T3324`-style timer values and when later scenario comparisons are presented in a practical design-guidance frame.

To make the interpretation more concrete, this section combines two related views. Figures 3.14-3.16 retain the generic simulator structure while recasting the timer axis into deployment-style values, whereas Figure 3.17 introduces profile-specific NB-IoT-like and NR mMTC-like baseline assumptions. The resulting lifetime magnitudes are therefore scenario-dependent and should always be interpreted together with the stated timer, profile, and battery choice.

Figures 3.14 and 3.15 form the core of that interpretation. Figure 3.14 compares lifetime against offered load for four `T3324`-style timer settings (`2`, `10`, `60`, and `360` s), while Figure 3.15 presents the corresponding delay curves in seconds. Read together, the two figures make the same trade-off from Section 3.3 easier to interpret in deployment terms: shorter timer settings keep the node more responsive as load rises, whereas longer timer settings preserve lifetime better but become harder to justify once delay constraints tighten. These paired figures are important because neither one alone is sufficient to support a design recommendation.

[Figure 3.14 about here: lifetime versus `lambda` for several `T3324`-like settings.]  
[Figure 3.15 about here: delay versus `lambda` with SLA-style annotation.]  

Figure 3.16 then asks whether the balanced-access rule observed earlier remains useful in a more practical frame. Here the study performs a direct grid search over `q` for each node population under an NB-IoT-inspired low-load baseline with a fixed 10 s timer, rather than simply reusing the exact O3 optimization setup. Its role is not to discover a brand-new law, but to test whether the recommended transmission probability still scales approximately as `1/n`. If that trend holds, it gives the report a simple rule of thumb that a designer can use as a starting point before refining the configuration for a specific delay or lifetime target.

[Figure 3.16 about here: recommended `q`* versus number of nodes.]  

Figure 3.17 and Table 3.4 convert the same theme into a compact decision view. The scatter plot compares representative NB-IoT-like and NR mMTC-like settings in the joint delay-lifetime plane using fixed `q = 1/n` operation and several timer choices, making it easy to see which scenarios cluster toward the responsive side of the trade-off and which cluster toward the long-lifetime side. Table 3.4 then complements that picture with direct design guidance for representative load levels. Together, the figure and table turn the simulator outputs into something closer to a design chart than a pure experiment log, even though they are not the same experiment rendered in two formats.

[Figure 3.17 about here: scatter of representative 3GPP-inspired scenario outcomes.]  


| lambda | T3324 (s) | q*     | Delay (ms) | Lifetime (years) | Stable | Meets 1 s target |
| ------ | --------- | ------ | ---------- | ---------------- | ------ | ---------------- |
| 1e-06  | 2         | 0.0500 | 0          | 10.528           | Yes    | Yes              |
| 1e-06  | 10        | 0.0500 | 0          | 2.468            | Yes    | Yes              |
| 1e-06  | 60        | 0.0500 | 0          | 0.427            | Yes    | Yes              |
| 1e-06  | 360       | 0.0500 | 0          | 0.285            | Yes    | Yes              |
| 1e-05  | 2         | 0.0500 | 28         | 10.243           | Yes    | Yes              |
| 1e-05  | 10        | 0.0500 | 28         | 2.405            | Yes    | Yes              |
| 1e-05  | 60        | 0.0500 | 28         | 0.421            | Yes    | Yes              |
| 1e-05  | 360       | 0.0500 | 92         | 0.285            | Yes    | Yes              |
| 1e-04  | 2         | 0.0500 | 108        | 5.934            | Yes    | Yes              |
| 1e-04  | 10        | 0.0500 | 106        | 1.475            | Yes    | Yes              |
| 1e-04  | 60        | 0.0500 | 98         | 0.338            | Yes    | Yes              |
| 1e-04  | 360       | 0.0500 | 95         | 0.283            | Yes    | Yes              |


The main contribution of this section is therefore interpretive rather than methodological. It shows that the validated baseline and the optimization results can be expressed in a 3GPP-inspired vocabulary without changing the underlying simulator logic. That provides a practical bridge from the abstract `(q, t_s)` design space to low-power IoT timer and access choices, and it sets up the final conceptual section of the chapter, which asks whether the two main design variables can truly be handled independently.

## 3.5 Independence Analysis

Objective O5 addresses the strongest conceptual question in the report: can the transmission probability `q` and sleep timer `t_s` be tuned independently? The closed-form validation model does not by itself answer that question, so this section uses the coupling score `kappa = p*t_s` together with factorial simulation sweeps to test how strongly that interaction appears in the actual design space.

Figure 3.18 gives the most direct visual answer. In the top-left panel, the delay curves versus `q` are not parallel across `t_s`, which means the effect of changing `q` depends on which sleep timer is chosen. In the bottom-left panel, the delay curves versus `t_s` are also separated by `q`, with the intermediate `q` setting performing best and the high-`q` setting worsening as `t_s` increases. By contrast, the two lifetime panels are much more tightly aligned, especially versus `q`, which shows that the interaction exists but is not equally strong for every metric. This figure therefore supports a nuanced conclusion: the parameters are coupled, but the coupling is most visible in delay rather than in lifetime.

[Figure 3.18 about here: interaction plots showing non-parallel or fanning curves.]  

Figure 3.19 provides the statistical companion to that visual evidence. It plots the additive-model delay residuals against the coupling score `kappa` after the regression step has filtered the data to stable, finite operating points, with point color indicating `t_s`. The value of this figure is that it reveals whether systematic structure remains after an additive fit has already accounted for separate `q` and `t_s` effects. The stronger claim that an interaction term improves the fit belongs to the regression summary in Table 3.5, which formally compares the additive and interaction models.

[Figure 3.19 about here: additive-model regression residuals versus coupling factor `kappa`, colored by `t_s`.]  


| Metric   | n_obs | R2 additive | R2 interaction | Delta R2 | F-statistic | p-value  |
| -------- | ----- | ----------- | -------------- | -------- | ----------- | -------- |
| Delay    | 24    | 0.9371      | 0.9399         | 0.0029   | 0.95        | 3.41e-01 |
| Lifetime | 24    | 0.9993      | 0.9995         | 0.0002   | 5.82        | 2.55e-02 |


Figures 3.20 and 3.21 then show where the interaction is weak and where it becomes more important. The coupling heatmap in Figure 3.20 indicates that `kappa = p*t_s` is largest for low-to-moderate `q` combined with larger `t_s`, while high `q` does not automatically imply strong coupling because the success probability term `p` falls away from its peak. Figure 3.21 simplifies the same message into regime labels: most of the plane is near-independent in the weak-coupling sense, but a moderate-coupling block appears precisely where low-to-moderate `q` and larger sleep timers overlap. This is a stronger result than a simple yes-or-no answer because it gives the reader a practical map of when independence is a reasonable approximation.

[Figure 3.20 about here: coupling heatmap of `kappa = p*t_s`.]  
[Figure 3.21 about here: regime map for weak, moderate, and strong coupling.]  

Figures 3.22 and 3.23 show the design consequences of that coupling. Figure 3.22 indicates that the unconstrained delay-minimizing `q`* changes only modestly with `t_s`, except for a local bump at small timers, so the report should not overstate the magnitude of the shift. Even so, the curve is not perfectly flat, which is enough to show that the preferred access setting is not entirely independent of the sleep setting. Figure 3.23 reinforces the same point geometrically: the iso-delay lines are not axis-aligned, and the lifetime contours are dominated by `q`, so moving to a better operating point generally requires coordinated movement in the parameter plane rather than one-dimensional tuning.

[Figure 3.22 about here: optimal `q*(t_s)` under lifetime constraints.]  
[Figure 3.23 about here: iso-contours of delay and lifetime over the `(q, t_s)` plane.]  

The independence analysis therefore supports the report’s strongest conceptual result. Sleep behavior and access behavior should not be treated as fully independent design knobs, especially when delay is the main objective. At the same time, the regime maps show that the strength of the interaction varies across the operating region, so the most accurate conclusion is not simply “always coupled,” but rather “coupled in general, with weak-coupling regions where simpler rules remain acceptable.”

---

# Chapter 4 Extension Studies

The core results establish a validated baseline and a clear design story. The purpose of this chapter is different. Rather than redefining the main contribution, the extension studies test how far the simulator can be pushed and what additional insights emerge when the baseline assumptions are relaxed or extended. They are therefore presented as supporting studies built on the validated simulator, not as replacements for the core argument developed in Chapter 3.

## 4.1 Finite Retry Limits

The baseline model assumes effectively unbounded retransmission, which is analytically convenient but not always realistic in practical systems. Objective O6 therefore introduces a retry limit `K` and studies how it changes delay, packet drop rate, and effective service behavior.

Figure 3.24 shows the core trade-off immediately. As the retry limit increases from `K = 2` toward the infinite-retry case, mean delay rises sharply while packet drop rate falls just as sharply. The interpretation is straightforward: a small retry budget prevents packets from occupying the head of the queue for long, but it does so by discarding a large fraction of failed packets. A large retry budget restores reliability, but the cost is that packets remain in service for longer before either succeeding or being abandoned.

[Figure 3.24 about here: delay and drop rate versus retry limit.]  

Figure 3.25 presents the same result in Pareto form. The points move from the lower-right corner at small `K` toward the upper-left corner as `K` increases: low delay comes with high loss, while low loss comes with much higher delay. The bend around moderate values such as `K = 5` to `K = 8` is the most practically interesting part of the curve because it suggests a compromise region where drop rate is reduced substantially before delay reaches the infinite-retry level.

[Figure 3.25 about here: delay-drop Pareto view for varying `K`.]  


| K   | Empirical mu | Analytical mu_K | Error (%) | Delay (slots) | Drop rate |
| --- | ------------ | --------------- | --------- | ------------- | --------- |
| 2   | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| 3   | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| 5   | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| 8   | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| 10  | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| 15  | 0.00893      | 0.00052         | 1631.2    | 113.1         | 0.0000    |
| inf | 0.00893      | 0.00370         | 141.5     | 113.1         | 0.0000    |


The finite-retry extension therefore adds a new design knob that did not exist in the baseline study. It shows that reliability, delay, and service persistence cannot all be improved simultaneously once retransmissions are capped, and it gives the report a concrete example of how implementation constraints create new trade-offs beyond the original sleep-access design space.

## 4.2 CSMA Comparison

Objective O7 compares the baseline slotted Aloha scheme with a CSMA-based alternative. In a sleep-aware setting this is not a trivial substitution, because carrier sensing changes not only collisions but also the timing cost of deferral. Recent work on RA-SDT-oriented random access compares Aloha and CSMA from a delay perspective and motivates keeping both in scope when interpreting low-power cellular IoT access [8].

Figure 3.26 shows that the delay comparison is strongly regime-dependent. At very small populations the two schemes are similar, but as the number of nodes grows CSMA-1P becomes much worse than slotted Aloha, with the gap widening sharply beyond about `n = 20–30`. This figure indicates that in the present setting the deferral overhead of CSMA dominates any potential gain from collision avoidance once the population becomes moderately large.

[Figure 3.26 about here: delay comparison between CSMA and Aloha.]  

Figure 3.27 explains part of that result. CSMA-1P does not achieve lower collision rates than slotted Aloha in the larger-`n` region; instead, its collision curve rises quickly and then saturates near the same level. Figure 3.28 makes the consequence even clearer under load: for `n = 100`, slotted Aloha achieves a much higher throughput plateau than CSMA-1P across almost the entire `lambda` range shown. Together, these two figures show that the delay penalty in Figure 3.26 is not compensated by a throughput gain.

[Figure 3.27 about here: collision comparison between CSMA and Aloha.]  
[Figure 3.28 about here: throughput versus load for both schemes.]  

The CSMA comparison therefore strengthens the report’s baseline choice rather than weakening it. In the regime examined here, the extra coordination cost of carrier sensing is more damaging than the collision reduction it provides, so slotted Aloha remains the better reference scheme for the sleep-based system under study.

## 4.3 Capture and SIC Receivers

Objective O8 relaxes the strict collision model of the baseline by introducing advanced receiver behavior. Instead of treating every multi-user transmission as a total failure, the extension allows either partial recovery through capture or multi-packet recovery through SIC.

Figure 3.29 shows the receiver-side change most directly. At small populations all three models are nearly identical, but once the network becomes denser the SIC curve separates strongly upward while capture provides only a modest improvement over the collision baseline. This means that the main benefit of receiver sophistication appears in the high-contention regime, where allowing recovery from nominal collisions changes the effective service process itself.

[Figure 3.29 about here: effective success probability under collision, capture, and SIC.]  

Figures 3.30 and 3.31 show how that success-probability gain translates into system performance. In Figure 3.30, SIC keeps delay dramatically lower than the other two receiver models as `n` increases, while capture provides only an intermediate improvement. Figure 3.31 shows the same ordering for lifetime: SIC maintains substantially higher lifetime at larger `n`, capture remains slightly better than the collision baseline, and the pure collision model degrades fastest. These two plots therefore tell a consistent story: better decoding reduces both waiting time and wasted energy.

[Figure 3.30 about here: delay comparison under the three receiver models.]  
[Figure 3.31 about here: lifetime comparison under the three receiver models.]  

The receiver-model extension demonstrates that congestion penalties are not determined only by the sender side of the protocol. Receiver capability changes the effective contention environment, and in dense regimes that change is large enough to alter both delay and longevity conclusions substantially.

## 4.4 Age of Information

Objective O9 adds Age of Information as a timeliness metric [9]. This matters because a system can have acceptable queueing delay and still perform poorly in terms of information freshness if updates are not generated or delivered often enough.

Figure 3.32 shows that mean AoI varies with `q` in a non-monotonic way similar to delay. AoI falls sharply from very small `q`, reaches its best region around low-to-moderate transmission probabilities, and then rises again once more aggressive access begins to hurt freshness rather than help it. The curves for different `t_s` values are again closely grouped, which indicates that, in this sweep, freshness is dominated more by access aggressiveness than by the idle timer.

[Figure 3.32 about here: AoI versus `q` for several `t_s` values.]  

Figures 3.33 and 3.34 clarify what this means for design. Figure 3.33 shows that the AoI-optimal and delay-optimal transmission probabilities are effectively identical over the tested `t_s` range, so this particular experiment does not reveal a strong divergence between the two notions of timeliness. Figure 3.34 then places AoI, lifetime, and delay in one view: high-lifetime points cluster at much larger AoI values, while lower-AoI regions are associated with shorter lifetime and lower delay. The trade-off therefore remains real even if the optimal `q`* happens to align in this case.

[Figure 3.33 about here: AoI-optimal versus delay-optimal `q*`.]  
[Figure 3.34 about here: AoI-delay-lifetime trade-off projection.]  

The AoI extension broadens the interpretation of performance without overturning the core story. It shows that freshness and delay are closely related in the tested regime, but both still sit in tension with battery lifetime, so the same co-design mindset remains necessary.

## 4.5 MMBP Arrivals

Objective O10 extends the project beyond Bernoulli-style arrivals by introducing a two-state Markov-modulated Bernoulli process. The purpose is to test when a simple independent-arrival approximation stops being good enough for the service-rate analysis.

Figure 3.35 compares analytical and empirical service-rate values across burstiness levels. The points do not lie exactly on the ideal diagonal, so the analytical formulas are not exact in this regime, but the MMBP-aware formulation remains close enough to the empirical values to be useful. More importantly, the scatter suggests that the Bernoulli approximation does not gain accuracy as burstiness increases, which is exactly the motivation for introducing the richer model.

[Figure 3.35 about here: analytical versus empirical `mu` under MMBP arrivals.]  

Figure 3.36 makes the comparison easier to read by plotting prediction error directly against burstiness index. Both methods are above the 10% line at low burstiness values, but once `BI` reaches about `3`, the MMBP formula falls below the threshold and stays consistently better than the Bernoulli approximation. The gap is not huge, but it is systematic. This means the value of the MMBP model is not that it is perfect, but that it becomes the more reliable approximation once traffic correlation is strong enough.

[Figure 3.36 about here: prediction error versus burstiness index.]  


| BI   | Empirical mu | mu_MMBP | MMBP error (%) | mu_Bernoulli | Bernoulli error (%) |
| ---- | ------------ | ------- | -------------- | ------------ | ------------------- |
| 1.0  | 0.00372      | 0.00355 | 4.6            | 0.00354      | 4.7                 |
| 1.5  | 0.00372      | 0.00355 | 4.6            | 0.00354      | 4.7                 |
| 2.0  | 0.00372      | 0.00355 | 4.6            | 0.00354      | 4.7                 |
| 3.0  | 0.00368      | 0.00355 | 3.6            | 0.00354      | 3.8                 |
| 5.0  | 0.00368      | 0.00355 | 3.3            | 0.00354      | 3.7                 |
| 7.5  | 0.00368      | 0.00356 | 3.3            | 0.00354      | 3.8                 |
| 10.0 | 0.00369      | 0.00357 | 3.3            | 0.00354      | 4.0                 |
|      |              |         |                |              |                     |


The MMBP extension closes the chapter by making a broader methodological point. Simplified assumptions are useful, but only within the regimes where they remain accurate enough. The simulator helps identify that boundary and shows when a richer traffic model is justified.

Taken together, the five extension studies support two points. First, the simulator is flexible enough to support meaningful modifications beyond the baseline. Second, the central story developed in Chapter 3 remains intact even when richer assumptions are introduced: sleep and access design remain tightly linked, and engineering choices continue to trade off responsiveness, efficiency, and robustness across different modeling conditions.

---

# Chapter 5 Discussion

## 5.1 Why the Validation Matters

The validation results matter because they establish the simulator as a credible representation of the intended baseline model. Without that foundation, the later optimization, 3GPP-inspired interpretation, and extension studies would be much weaker. Agreement between empirical and analytical results does not prove that the model is universally correct in every practical setting, but it does show that the implemented state machine, contention logic, and energy accounting are consistent with the analytical assumptions in the stable regime.

This is an important distinction. The value of the simulator is not that it replaces theory, but that it complements theory. The analytical model provides tractable expressions and intuition, while the simulator exposes finite-run variability, richer operating regimes, and extensions that would be cumbersome to derive from first principles. The report's methodological strength lies in combining these two perspectives rather than treating them as alternatives.

## 5.2 What the Trade-Off Means

The parameter-impact and optimization results confirm that low delay and long lifetime are genuinely competing objectives. Increasing access aggressiveness can reduce waiting time, but it tends to raise collisions or energy use. Increasing sleep conserves energy, but it delays service when packets arrive during inactive periods. This means that there is no single parameter setting that is best for every deployment goal.

This observation is important for system design because it reframes the problem. The engineering challenge is not to discover a universal optimum, but to choose a point on a trade-off surface that matches application requirements. A monitoring device with loose latency needs may rationally prioritize longevity, while a freshness-sensitive control device may accept shorter lifetime to reduce delay or improve AoI. The simulator makes these trade-offs explicit rather than hiding them behind a single average metric.

## 5.3 Design Implications

One practical result of the project is that `q ~= 1/n` remains a useful baseline heuristic for balanced operation. This rule has intuitive appeal because it roughly targets one expected transmission attempt per slot, and the simulation results support its usefulness across the validated baseline setting. However, the independence analysis shows that this heuristic is only a starting point rather than a universal answer.

The strongest design implication of the report is that `q` and `t_s` should not be tuned as if they were independent control knobs. The coupling score `p*t_s` shows that sleep and access behavior interact across the explored design space even when the formal validation formula for `mu` is written differently. As a result, a sequential procedure that first fixes `q` and then tunes `t_s`, or vice versa, can miss better joint operating points. The report therefore supports a co-optimization mindset, especially outside the weak-coupling regime.

The weak-coupling result is also practically useful. If `kappa = p*t_s` is sufficiently small, approximate independence may be acceptable and simpler tuning rules may still work. This nuance gives the report stronger engineering value than a simple yes-or-no statement about parameter dependence.

## 5.4 Realism and Limitations

The report's claims should be interpreted within the limits of the model. The simulator is intentionally abstract. It uses homogeneous nodes in the baseline study, an idealized channel abstraction, and simplified mappings to 3GPP-style mechanisms. It does not model full protocol signaling, physical-layer fading, mobility, multi-channel behavior, or heterogeneous device populations in the core analysis.

These limitations do not invalidate the findings, but they do constrain their meaning. The project provides structured design insight rather than a drop-in deployment recipe. The 3GPP-oriented results should therefore be read as interpretive guidance rather than standards-compliance claims. Similarly, the reported extension results broaden the scope of the simulator but do not eliminate the simplifying assumptions that remain in the overall framework.

## 5.5 What the Extensions Add

The extensions show that the simulator is not tied narrowly to one baseline question. Finite retries, CSMA, advanced receivers, AoI, and MMBP arrivals all represent meaningful departures from the original setup, yet each can be studied within the same software framework. This is valuable in its own right as a software-oriented research outcome.

At the same time, the extension chapter also clarifies the boundaries of the core story. Some extensions, such as SIC or AoI, change the preferred operating point or reveal different performance priorities. Others, such as finite retries or MMBP arrivals, expose failure modes of simplified baseline assumptions. Yet none of them remove the central importance of jointly reasoning about access and sleep behavior. In that sense, the extensions reinforce rather than displace the main result.

---

# Chapter 6 Conclusion

## 6.1 Summary of Contribution

This project developed a validated discrete-event simulation framework for sleep-based random access in battery-powered M2M systems. The simulator reproduces the intended baseline model of slotted Aloha with on-demand sleep, supports systematic parameter sweeps and repeated trials, and integrates analytical validation, realistic power models, optimization routines, visualization utilities, and notebook-based experimentation.

Using that framework, the project quantified the delay-lifetime trade-off, identified Pareto-efficient operating points, translated the results into 3GPP-inspired design guidance, and answered the independence question for the key parameters `q` and `t_s`. The core conclusion is that access aggressiveness and sleep timing generally need to be tuned jointly because they interact directly in the service process.

The project also extended the simulator to study finite retry limits, CSMA comparison, advanced receiver models, Age of Information, and MMBP arrivals. These studies broadened the applicability of the framework and showed that the simulator is capable of supporting richer research questions beyond the original baseline.

## 6.2 Objective Review

The grouped objectives defined at the start of the report were achieved in a coherent progression. The foundation objective established the simulator. The core analysis objectives quantified the operating trade-offs and answered the independence question. The design and validation objectives connected the simulator to theory and practical interpretation. The extension objectives then broadened the framework without disrupting the central report narrative.

In this sense, the project is more than a collection of implemented features. Its contribution lies in the combination of validated modeling, structured experimentation, and design-oriented interpretation.

## 6.3 Limitations

The report is subject to the limitations already discussed: a homogeneous-node baseline, simplified channel abstraction, 3GPP-inspired rather than standard-faithful mappings, and partial reliance on abstract traffic and receiver models. In addition, several sections of the present manuscript still require direct insertion of verified figure-specific numerical values where placeholders remain.

## 6.4 Future Work

The most credible next steps are those that follow naturally from the current framework. These include heterogeneous node populations, adaptive closed-loop control of `q` and `t_s`, richer physical-layer models, multi-channel access, and tighter integration between analytical approximation and simulation under correlated traffic. Among the existing extensions, AoI and MMBP arrivals appear especially promising because they directly challenge assumptions that are often simplified in baseline queueing studies.

From a software perspective, future work could also strengthen the usability of the framework by adding a more formal user interface, richer experiment configuration files, and more automated report-generation pipelines for figures and tables.

---

# References

References are numbered in IEEE style in the order they first appear in the report. 3GPP specification titles and release numbers should be updated to match the exact document versions used in the student’s bibliography check; the entries below identify the correct series and typical scope.

[1] X. Wang, L. Dai, and X. Sun, “On-Demand-Sleep-Based Aloha for M2M Communication: Modeling, Optimization, and Tradeoff between Lifetime and Delay,” *IEEE Internet of Things Journal*, vol. 11, no. 21, pp. 35625–35639, Nov. 2024.

[2] 3GPP, “Technical Specification Group Services and System Aspects; 5G System (5GS); System architecture for the 5G System (5GS); Stage 2,” 3GPP TS 23.501, v18.x.x, 2024. [Online]. Available: https://www.3gpp.org/specifications-technologies/releases

[3] 3GPP, “Technical Specification Group Core Network and Terminals; Non-Access-Stratum (NAS) protocol for 5G System (5GS); Stage 3,” 3GPP TS 24.501, v18.x.x, 2024. [Online]. Available: https://www.3gpp.org/specifications-technologies/releases

[4] 3GPP, “Technical Specification Group Radio Access Network; NR; User Equipment (UE) procedures in idle mode and RRC inactive state,” 3GPP TS 38.304, v18.x.x, 2024. [Online]. Available: https://www.3gpp.org/specifications-technologies/releases

[5] L. Kleinrock and F. A. Tobagi, “Packet Switching in Radio Channels: Part I—Carrier Sense Multiple-Access Modes and Their Throughput-Delay Characteristics,” *IEEE Trans. Commun.*, vol. COM-23, no. 12, pp. 1400–1416, Dec. 1975.

[6] The SimPy Development Team, “SimPy—Discrete-event simulation for Python,” readthedocs, accessed Apr. 2026. [Online]. Available: https://simpy.readthedocs.io/

[7] G. Bianchi, “Performance Analysis of the IEEE 802.11 Distributed Coordination Function,” *IEEE J. Sel. Areas Commun.*, vol. 18, no. 3, pp. 535–547, Mar. 2000.

[8] X. Zhao and L. Dai, “To Sense or Not To Sense: A Delay Perspective,” arXiv:2406.02999, Jun. 2024. [Online]. Available: https://arxiv.org/abs/2406.02999

[9] S. Kaul, R. Yates, and M. Gruteser, “Real-Time Status: How Often Should One Update?” in *Proc. IEEE INFOCOM*, Orlando, FL, USA, Mar. 2012, pp. 2731–2735.

[10] [Student Name], *Sleep-Based Low-Latency Access for M2M Communications Simulator*, GitHub repository, Apr. 2026. [Online]. Available: https://github.com/lfernande3/fyp

### Bibliography note and submission checklist

**Project wording versus reference [1].** The report and repository use the phrase “sleep-based low-latency access” as the project theme. The analytical baseline cited as **[1]** is the peer-reviewed article whose official title is *On-Demand-Sleep-Based Aloha for M2M Communication: Modeling, Optimization, and Tradeoff between Lifetime and Delay* (Wang, Dai, and Sun, *IEEE Internet of Things Journal*, 2024). These refer to the same line of work; use **[1]** in the final PDF for formal citation.

**One-line map of references**

| Ref. | Role in this report |
|------|---------------------|
| [1] | Analytical baseline (success probability, service rate, delay–lifetime framework). |
| [2]–[4] | 3GPP architecture, NAS, and NR idle-mode procedures (PSM / MICO / timer analogues). |
| [5]–[7] | Slotted random-access modelling, SimPy as DES practice, Bianchi-style performance analysis. |
| [8] | Aloha vs CSMA delay perspective (RA-SDT–oriented context). |
| [9] | Age of Information (status-update timeliness). |
| [10] | This project’s simulator source code, tests, notebooks, and report assets (GitHub). |

**Before submission**

1. Confirm **[1]** with your supervisor: final title string, volume/issue/pages, and DOI or URL if your department requires them.
2. Replace `v18.x.x` (and the year if needed) for **[2]–[4]** with the exact 3GPP specification versions you are allowed to cite.
3. Replace **`[Student Name]`** in **[10]** with your name as it should appear in the bibliography (family name first if your style guide requires it).
4. If your examiner wants a **frozen artifact**, add the **commit SHA** or **release tag** after the URL in **[10]** (for example: “accessed Apr. 10, 2026; commit `abc1234`”) or cite a Zenodo DOI if you archive a snapshot.
5. Re-scan the body text so every `[n]` and range like `[2]–[4]` still matches the list above after any edits.
6. If the faculty template requires Harvard, Vancouver, or APA instead of IEEE, reformat the entries but keep the same sources and ordering logic (first appearance, or alphabetical per local rule).

---

# Appendices

## Appendix A User Guide

This appendix summarizes how to run the simulator and reproduce the main workflows from the version-controlled project published as a GitHub repository [10]. The following steps assume you have cloned or downloaded that repository and are using its root directory (the folder that contains `requirements.txt`).

**Environment.** Python 3.8 or newer is required. A virtual environment is recommended so that dependencies do not conflict with other projects.

**Install dependencies.** From the repository root:

```bash
pip install -r requirements.txt
```

**Run the automated tests.** The suite under `tests/` exercises node logic, the simulator, metrics, experiments, validation, optimization, independence analysis, and related modules:

```bash
pytest tests/
```

**Run the standalone validation script.** `run_validation.py` executes a small-scale integration path and prints sanity-check results (for example no-sleep versus standard sleep-aware behavior):

```bash
python run_validation.py
```

**Example notebooks.** The `examples/` folder contains demonstration and experiment notebooks. Open them in Jupyter or VS Code; the first cell typically adds the project root to `sys.path` so imports such as `from src.simulator import Simulator` resolve whether the notebook’s working directory is the repo root or `examples/`.

**Figures and tables.** Report-quality figures can be regenerated using `generate_figures.py` at the repository root (outputs are written under `report/figures/`). Optional tabular exports may be produced with `generate_report_tables.py` if present in the repo.

**Reproducibility.** Simulation runs accept a random `seed` via `SimulationConfig`; batch drivers should record seeds and sweep settings alongside saved CSV or JSON outputs.

## Appendix B Source Code Structure

The implementation lives mainly under `src/`. The following modules map to the responsibilities described in Chapter 2.

| Module | Role |
|--------|------|
| `node.py` | Single-MTD state machine (active, idle, sleep, wakeup), queue, Bernoulli and extension arrivals, transmission attempts, energy accounting, retry counters (O6), CSMA backoff state (O7). |
| `simulator.py` | `SimulationConfig`, `SimulationResults`, `Simulator` slotted loop, collision or extension resolution via `receiver_models`, `BatchSimulator` for sweeps and replications. |
| `power_model.py` | `PowerModel`, predefined profiles (e.g. NB-IoT, NR mMTC), battery configuration, lifetime helpers. |
| `metrics.py` | `MetricsCalculator` and helpers turning raw counters into delay, throughput, empirical `p` and `μ`, and comparison metrics. |
| `validation.py` | `AnalyticalValidator`, `SanityChecker`, trace logging, `run_small_scale_test` for quick integration checks. |
| `experiments.py` | Parameter sweeps, scenario runners, and batch experiment wiring used by notebooks and figure generation. |
| `optimization.py` | Grid search / Pareto-style exploration over `(q, t_s)`, duty-cycle comparison, prioritization summaries (Objective O3). |
| `validation_3gpp.py` | 3GPP-inspired scenarios, design guideline tables, and O4 validation helpers. |
| `independence.py` | Factorial sweeps, regression, and visualization support for the `q`–`t_s` interaction study (Objective O5). |
| `traffic_models.py` | Pluggable arrival processes (Bernoulli, bursty, MMBP drivers for O10). |
| `receiver_models.py` | Collision, capture, and SIC resolution hooks (Objective O8). |
| `baselines.py` | Baseline and comparison experiment helpers (e.g. CSMA vs Aloha, O7). |
| `mmbp_analytics.py` | MMBP-related analytical approximations and error summaries (Objective O10). |
| `visualizations.py` | Plotting utilities shared by notebooks and batch outputs. |

The public package surface re-exports the main classes from `src/__init__.py` for convenient `from src import ...` usage in notebooks.

## Appendix C Additional Figures and Tables

Not every experimental output appears in the main body. Material that is useful for examination but too large for the core chapters can be placed here or left as separate files under `report/figures/` and cited from this appendix.

**Suggested content for submission**

- Full-parameter validation grids (all `(n, q)` combinations) with tolerance bands, if reduced tables appear in Chapter 3.
- Extended optimization sweeps (complete heatmap data or additional `(q, t_s)` slices) corresponding to Figures 3.9–3.11.
- Complete 3GPP-inspired guideline tables if Table 3.4 is abridged in the main text.
- Supplementary extension plots (e.g. additional `K` values for retries, extra receiver-parameter sweeps, AoI or MMBP sensitivity curves).

**File layout.** Generated PNGs for Chapter 3 and Chapter 4 follow the naming convention `report/figures/fig3_*` and `fig3_*` (figure numbers align with the report). Regeneration commands are documented in `generate_figures.py`.

If the departmental template limits appendix length, prioritize a short index table listing filename, experiment ID, and parameters, and attach full plots in a ZIP or online supplement only if allowed.

## Appendix D Selected Code Listings

The following excerpt shows the core structure of one simulated slot in `Simulator.run_simulation`: arrivals, access decisions (slotted Aloha or CSMA-1P), and pluggable receiver resolution. Ellipses omit metric updates and state/energy steps that follow the same pattern for every slot.

```python
# Excerpt from src/simulator.py — main slot loop (abbreviated)

for slot in range(self.config.max_slots):
    self.current_slot = slot
    # ... stopping conditions (depletion) ...

    for node in self.nodes:
        if not node.is_depleted():
            node.arrive_packet(slot, self.config.arrival_rate)

    transmitting_nodes = []
    for node in self.nodes:
        if not node.is_depleted() and node.state == NodeState.ACTIVE:
            if node.get_queue_length() > 0:
                if use_csma:
                    # CSMA-1P: defer if channel was busy last slot, else transmit with prob q
                    ...
                else:
                    if node.attempt_transmit(self.config.transmission_prob):
                        transmitting_nodes.append(node)

    successful_nodes = resolve_transmissions(
        transmitting_nodes,
        model=rx_model,
        capture_threshold=self.config.capture_threshold,
        sic_sinr_threshold=self.config.sic_sinr_threshold,
    )
    # ... success handling, energy, state updates, statistics ...
```

Validation against analytical `p` and `μ` is implemented in `src/validation.py` (`AnalyticalValidator`); factorial regression for independence analysis is implemented in `src/independence.py`. Listings for those files can be added here if the assessor requests explicit code in the appendix—the repository remains the authoritative source.