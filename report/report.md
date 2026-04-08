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

Generative AI tools were used only as support tools during software development and report preparation. Their use was limited to drafting assistance, wording refinement, debugging support, and structural suggestions for planning documents.

All technical content, implementation decisions, analysis, results, interpretations, and conclusions in the final report are my own work. No figure, formula, design claim, or reported finding was included without manual checking against the codebase, planning documents, or experiment outputs.

This section should be revised to match the exact departmental disclosure format required at submission time.

## Table of Contents

[Generate automatically in the final submission template.]

## List of Figures

[Generate automatically in the final submission template.]

## List of Tables

[Generate automatically in the final submission template.]

---

# Chapter 1 Introduction and Context

## 1.1 Background and Motivation

Machine-to-machine communication is a central enabling technology for modern IoT systems, where large numbers of autonomous devices exchange small packets for sensing, monitoring, tracking, and control. Typical examples include environmental sensing, utility metering, industrial telemetry, smart transport, and health-related monitoring. In many of these applications the nodes are battery powered, geographically distributed, and expected to operate unattended for long periods. As a result, energy efficiency is not simply an optimization target; it is a basic design requirement.

At the same time, many M2M applications cannot tolerate arbitrarily long waiting times. Even where traffic is sparse, the network may still need to support timely updates, bounded access delay, or freshness-sensitive information delivery. This creates a design tension. Nodes that remain active more often can react quickly to arrivals, but they consume more energy. Nodes that sleep aggressively can extend battery lifetime, but packets that arrive during sleep must wait for wake-up and renewed channel access. The engineering problem is therefore to manage delay and energy jointly rather than to optimize either in isolation.

This project studies that tension in the context of sleep-aware random access. The baseline system is slotted Aloha with on-demand sleep, where a node transmits in an active slot with probability `q`, enters sleep after remaining idle for `t_s` slots, and requires `t_w` slots to wake before it can resume contention. These parameters jointly shape both the access process and the energy cost of operation. Similar design ideas appear in 3GPP cellular IoT mechanisms such as Power Saving Mode (PSM), MICO mode, the active timer `T3324`, and reduced access procedures such as RA-SDT, although the present work uses these only as design analogies rather than attempting full standards emulation. [CITATION: Wang2024] [CITATION: 3GPP]

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

The principal analytical foundation for this project is Wang et al. (2024), which studies sleep-based low-latency access for M2M communications and derives the key relationships among success probability, service rate, delay, and lifetime. That work motivates the use of slotted random access with on-demand sleep and provides the theoretical baseline that the simulator in this report was designed to reproduce and extend. [CITATION: Wang2024]

The practical interpretation of the project is also informed by 3GPP specifications for power saving and reduced-access behavior in cellular IoT systems. Concepts such as PSM, MICO mode, `T3324`, and simplified access procedures provide a real-world frame for understanding the simulator parameters, even though the simulator does not attempt to model every standards detail. [CITATION: 3GPP_TS_23_501] [CITATION: 3GPP_TS_36_304_OR_38_304]

More broadly, the project sits at the intersection of random-access analysis, low-power IoT protocol design, and discrete-event simulation. Prior slotted Aloha literature provides the conceptual background for the role of `q`, while simulation frameworks such as SimPy make it practical to study parameter sweeps, finite-run variance, and complex extensions. In this report, external literature is used mainly to frame the problem and anchor the baseline theory, while the main technical contribution lies in the implemented simulator and the design insights derived from it. [CITATION: Bianchi2000] [CITATION: SimPy]

---

# Chapter 2 Methodology

## 2.1 System Model

The baseline system considered in this report is a slotted M2M uplink in which `n` homogeneous battery-powered devices compete for access to a shared channel. Time is divided into discrete slots. Each node maintains a packet queue, an energy budget, and a state-machine representation of its current operating mode. The baseline arrival model is Bernoulli per slot, while richer traffic models are introduced later for sensitivity and extension studies. A slot duration of approximately 6 ms is used when converting simulated slot counts into physical time or battery-lifetime estimates.

Each node evolves through four main states: active, idle, sleep, and wakeup. While active, a node with a non-empty queue attempts transmission with probability `q`. If the node remains without traffic for `t_s` consecutive idle slots, it enters sleep. When a packet arrives during sleep, the node must first spend `t_w` slots in wakeup before returning to active contention. This abstraction is simple enough to support systematic sweeps and analytical comparison, while still capturing the core interaction between energy saving and access delay.

The key parameters used throughout the report are the number of nodes `n`, the transmission probability `q`, the sleep idle timer `t_s`, the wake-up time `t_w`, the packet arrival rate `lambda`, and the initial energy budget. Most core experiments restrict attention to stable configurations in which the offered load remains below the effective service rate so that delays remain finite and the empirical metrics remain comparable to the analytical baseline.

[Figure 2.1 about here: state-transition diagram for a single MTD showing Active -> Idle -> Sleep -> Wakeup -> Active.]

## 2.2 Analytical Baseline

The analytical reference point for the simulator follows the framework used by Wang et al. (2024). For a tagged node that transmits with probability `q` in a population of `n` devices, the baseline success probability is

`p = q(1 - q)^(n - 1)`.

This is the probability that the tagged node transmits while every other node remains silent in the same slot. [CITATION: Wang2024]

Using the mean-cycle interpretation adopted in the project, the service rate of the sleep-aware baseline can be written as

`mu = p / (1 + p*t_s + p*t_w)`.

The denominator captures the combined effect of transmission opportunity, idle waiting before sleep, and wake-up overhead. The stability condition is then `lambda < mu`, and unstable operating points are excluded or clearly marked in the main baseline analyses.

This formulation also motivates Objective O5. The term `p*t_s` implies that the effect of changing `t_s` depends on `p`, and therefore on `q`. Conversely, the effect of changing `q` depends on the current value of `t_s`. The report therefore uses the coupling quantity `kappa = p*t_s` as a compact indicator of the interaction strength between sleep and access behavior.

## 2.3 Simulator Architecture

The simulator was implemented as a modular Python codebase so that the baseline model, validation utilities, and extension studies could share the same core logic. At the lowest level, the `Node` component represents an individual MTD with queue state, energy state, timers, and per-node statistics. The node logic includes packet arrival handling, state updates, transmission attempts, successful delivery processing, retry handling where applicable, and accounting for energy consumption in different states.

The `Simulator` component coordinates all nodes over slotted time. In each slot it generates arrivals, updates node states, collects transmission attempts, resolves collisions or successful receptions, and records the resulting network statistics. A run terminates after a configured number of slots or when the selected stopping condition is reached. In the baseline configuration, a slot is successful only if exactly one node transmits. Later extensions replace this rule with CSMA-style deferral or advanced receiver logic such as capture and SIC.

To support large experiment sets, the project also implements a `BatchSimulator` and associated configuration and result dataclasses. These utilities support parameter sweeps across `q`, `t_s`, `n`, and `lambda`, repeated simulation under multiple random seeds, and aggregation across replications. Supporting modules include `PowerModel` for state-dependent energy rates and battery profiles, `MetricsCalculator` for empirical and analytical quantities, validation utilities for analytical comparison, optimization utilities for design-space search, and dedicated analyzers for independence and extension studies.

This architecture is consistent with the expectations of a software-oriented final year project report. The work is not only a set of numerical experiments but also a reusable and testable software artifact with clear program structure, debugging support, and notebook-based usage pathways.

[Figure 2.2 about here: software architecture showing `Node -> Simulator -> BatchSimulator -> MetricsCalculator`, with supporting modules for validation, optimization, traffic models, receiver models, and MMBP analytics.]

## 2.4 Metrics and Logging

The simulator records both node-level and network-level metrics. The core reported outputs are mean queueing delay, throughput, success probability, service rate, average queue length, lifetime, state occupancy fractions, and energy-consumption breakdown by state. Additional outputs include tail-delay measures, energy per successful packet, trace histories for debugging, and extension-specific metrics such as drop rate and AoI.

These metrics were selected because they align directly with the project objectives. Delay and lifetime are the main end-user design outcomes. Success probability and service rate provide a bridge between analysis and simulation. State fractions and energy breakdown help explain why different parameter settings produce different operating regimes. Trace-level logging, meanwhile, supports debugging and small-scale validation by exposing the detailed evolution of states, queues, collisions, and energy consumption over time.

Results can be summarized numerically and exported in structured formats such as CSV, JSON, and plot images. This export path is important because it allows the same codebase to support fast testing, notebook-based experimentation, and report-grade visualization.

## 2.5 Experimental Program

The experimental design is organized around the grouped project objectives. The first stage validates the baseline simulator against analytical expectations. This is followed by parameter sweeps for Objective O2, optimization studies for Objective O3, 3GPP-inspired design studies for Objective O4, and independence experiments for Objective O5. The extension objectives O6-O10 reuse the same simulator core but modify selected aspects of the access process, receiver model, retry policy, or traffic model.

For O2, the project sweeps key parameters across representative ranges to reveal monotonic trends, saturation effects, and trade-offs. These sweeps cover transmission probability, idle timer, node population, arrival rate, and traffic model variations. For O3, the project uses grid-search optimization over the `(q, t_s)` plane to identify low-delay, high-lifetime, and Pareto-efficient operating points. For O4, the project maps simulator parameters to 3GPP-inspired scenarios and generates guideline-style outputs. For O5, a full-factorial design over `q` and `t_s` is used to test the independence hypothesis and to quantify the resulting interaction.

[Table 2.1 about here: summary of experiments, parameter ranges, replications, and primary outputs for the foundation, core, design, and extension objectives.]

## 2.6 Validation Strategy

Validation is performed at several levels. First, small-scale sanity checks verify that the implemented state transitions, collision logic, and energy accounting behave as expected. Representative checks include the no-sleep limit, immediate-sleep behavior, and high-`q` configurations that should increase collisions. Second, direct analytical comparisons are made for success probability, service rate, delay, and lifetime under stable baseline conditions. Third, convergence experiments examine how the error between empirical and analytical quantities changes as the simulation length increases.

The validation utilities classify agreement using explicit tolerance bands such as plus or minus 5%, 10%, and 20%. Unstable configurations are flagged and excluded from baseline comparison so that disagreements caused by violating the assumptions of the analytical model are not misinterpreted as implementation errors. This validation-first approach is essential because the later optimization and extension studies would be much less credible if the baseline simulator had not already been shown to align with theory.

## 2.7 Implementation Notes

The implementation uses Python together with the standard numerical and plotting libraries expected in a scientific simulation workflow. The project brief also identified SimPy as the intended discrete-event simulation framework, although the practical emphasis of the report is on the implemented simulation architecture rather than on the library itself. The codebase was designed to run in both local and notebook-style environments, including the kind of workflow typically used in Jupyter or Colab.

Testing forms a substantial part of the software engineering contribution. According to the project task record, the codebase accumulated an extensive unit-test suite spanning the baseline simulator, metrics, visualization, optimization, validation, independence analysis, traffic models, receiver models, and other extensions. This broad coverage reduces the risk that later extensions silently invalidate the baseline logic. It also helps justify the report's claim that the simulator is not merely a quick prototype but a structured and test-backed engineering artifact.

The project also provides practical usage pathways through notebooks and scripts. The repository documentation describes dependency installation, notebook execution, and test execution using `pytest`. These details are not central to the theoretical contribution, but they matter for a software-oriented project because they demonstrate that the framework can be inspected, reproduced, and reused.

---

# Chapter 3 Core Results and Design Guidance

## 3.1 Simulator Validation

The first result required by the report is not a design insight but a credibility claim: before using the simulator to argue about trade-offs, it must be shown to reproduce the analytical baseline with acceptable accuracy in the stable regime. The validation framework therefore compares empirical and analytical success probability `p`, service rate `mu`, delay, and lifetime across multiple node populations and run lengths.

According to the completed task record, the simulator reproduces the major analytical trends closely and achieves agreement within approximately plus or minus 5% for representative stable cases. This result is important because it shows that the implemented state machine, contention logic, and energy accounting are internally consistent with the intended model rather than only qualitatively plausible. It also justifies the use of the simulator as an extension platform for scenarios where closed-form analysis is less convenient.

The validation chapter should present at least three pieces of evidence: a comparison of empirical and analytical success probability, a comparison of empirical and analytical service rate, and a convergence plot showing that the error in the empirical estimates decreases as the number of simulated slots increases. Together, these demonstrate both correctness and numerical stability.

[Figure 3.1 about here: empirical versus analytical success probability with tolerance band.]  
[Figure 3.2 about here: empirical versus analytical service rate with tolerance band.]  
[Figure 3.3 about here: convergence of error versus simulation length.]  
[Table 3.1 about here: validation summary table with stability flags and percentage errors.]

## 3.2 Parameter Impact

Once the baseline simulator is validated, it can be used to map how the main design parameters affect performance. Objective O2 examines the influence of transmission probability, idle timer, traffic load, and population size on delay and lifetime. The resulting plots transform the project from a validation exercise into a measurable design study.

The key trends are conceptually intuitive but important to demonstrate empirically. Increasing `q` makes channel access more aggressive and can reduce delay in the stable regime, but it also raises collision pressure and energy expenditure. Increasing `t_s` allows nodes to remain inactive for longer before entering sleep-aware behavior, which can improve lifetime under suitable traffic conditions, but it also increases the waiting time experienced by packets that arrive while the node is inactive or sleeping. Varying `lambda` and `n` then reveals the stability boundary and the operating conditions under which the trade-off becomes more pronounced.

The parameter-impact study also benefits from the project's richer traffic support. While Bernoulli arrivals define the analytical baseline, bursty, periodic, and on-off traffic variants help show that mean trends alone are not the whole story. In particular, richer traffic can increase delay variability and tail behavior even when the mean load remains unchanged. This matters because one of the motivations for simulation was precisely to expose behavior that is hidden by average-case formulas.

[Figure 3.4 about here: mean delay versus `q` for several `t_s` values.]  
[Figure 3.5 about here: expected lifetime versus `q` for several `t_s` values.]  
[Figure 3.6 about here: mean delay versus `t_s` for several `q` values.]  
[Figure 3.7 about here: expected lifetime versus `t_s` for several `q` values.]  
[Figure 3.8 about here: throughput versus `lambda` for several node populations.]  
[Table 3.2 about here: representative state-fraction and scenario summary table.]

The central result of this section is that the delay-lifetime trade-off is systematic and structured rather than anecdotal. That conclusion provides the necessary foundation for the optimization and design guidance sections that follow.

## 3.3 Optimization Results

Objective O3 extends the parameter-impact study by asking not only how the parameters behave, but which combinations are preferable under different design priorities. The project implements dedicated optimization logic to sweep the `(q, t_s)` design space, produce delay and lifetime heatmaps, and extract Pareto-efficient operating points.

The main conclusion of this section is that there is no single universally optimal parameter pair. Instead, the appropriate operating point depends on the application objective. If latency is prioritized, the preferred configuration tends to keep the node more responsive and more aggressive in access. If lifetime is prioritized, the preferred configuration allows longer inactivity and more conservative access. Between these extremes lies a set of Pareto-efficient configurations that make the trade-off explicit rather than hiding it behind a single aggregated metric.

The project also compares canonical operating scenarios representing low-latency, balanced, and battery-priority behavior. This kind of scenario analysis is important because it translates abstract heatmaps into more interpretable deployment choices. In addition, the work compares on-demand sleep against duty-cycling and reports that on-demand sleep performs more favorably in the intended low-traffic setting. That result is practically significant because it shows that activity-aware sleeping can outperform rigid periodic sleep scheduling under sporadic traffic.

The task record further indicates that `q* ~= 1/n` emerges as a strong baseline rule of thumb for balanced performance. This does not eliminate the need for optimization, but it provides a useful scaling heuristic that connects access aggressiveness to population size.

[Figure 3.9 about here: delay heatmap over the `(q, t_s)` plane.]  
[Figure 3.10 about here: lifetime heatmap with stability contour.]  
[Figure 3.11 about here: Pareto frontier of lifetime versus delay.]  
[Figure 3.12 about here: scenario comparison relative to a balanced baseline.]  
[Figure 3.13 about here: on-demand sleep versus duty-cycling.]  
[Table 3.3 about here: prioritization scenario summary with gains and losses.]

## 3.4 3GPP-Inspired Interpretation

Objective O4 improves the practical relevance of the project by linking the abstract simulator parameters to 3GPP-inspired settings. The project implements a dedicated alignment module that maps MICO-like sleep behavior, `T3324`-like timers, and reduced access procedures to the simulator variables `t_s` and `t_w`. The goal is not to claim full standards compliance, but to translate the results into a more interpretable engineering frame.

This section should present the simulator as a design-oriented study rather than a purely analytical replication. Representative outputs include lifetime versus traffic load for several timer settings, delay versus traffic load under the same settings, the recommended transmission probability as a function of node population, and a guideline table relating traffic intensity, timer settings, lifetime, delay, and stability. According to the project record, this part of the work also includes convergence analysis and scenario constructors for NB-IoT and NR mMTC analogues.

The design value of this section is that it expresses the trade-off in terms closer to deployment choices. Instead of discussing `t_s` and `q` only as abstract mathematical parameters, the report can describe how timer settings and access choices affect lifetime and delay under recognizable low-power IoT scenarios. This strengthens the project's practical contribution while still keeping the claims proportionate to the abstraction level of the model.

[Figure 3.14 about here: lifetime versus `lambda` for several `T3324`-like settings.]  
[Figure 3.15 about here: delay versus `lambda` with SLA-style annotation.]  
[Figure 3.16 about here: recommended `q*` versus number of nodes.]  
[Figure 3.17 about here: scatter of representative 3GPP-inspired scenario outcomes.]  
[Table 3.4 about here: design-guideline table linking load, timer, `q*`, delay, lifetime, and stability.]

## 3.5 Independence Analysis

Objective O5 addresses the strongest conceptual question in the report: can the transmission probability `q` and sleep timer `t_s` be tuned independently? The analytical structure of the model suggests that the answer is no, because the service-rate expression contains the multiplicative term `p*t_s`. Since `p` itself depends on `q`, the effect of changing one parameter necessarily depends on the other.

The project treats this question rigorously rather than impressionistically. A full-factorial sweep over `q` and `t_s` is used as the common data source for interaction plots, regression models with and without an interaction term, coupling heatmaps, regime maps, iso-contour plots, and constrained optimization of `q*(t_s)`. These analyses all point in the same direction. The interaction plots show non-parallel or fanning curves, the additive regression model leaves structured residuals that are reduced by an explicit interaction term, and the constrained-optimal `q` shifts as `t_s` changes. Together these results provide both analytical and empirical evidence that `q` and `t_s` are not independent in the baseline system.

The report should also preserve the nuance identified in the project record. The parameters are not independent in general, but they can be approximately independent in a weak-coupling regime where `kappa = p*t_s` is sufficiently small. The project identifies `kappa < 0.1` as a practical threshold for near-independent behavior. This is a more useful conclusion than a simple binary answer because it explains not only that coupling exists, but also when it becomes weak enough to ignore.

[Figure 3.18 about here: interaction plots showing non-parallel or fanning curves.]  
[Figure 3.19 about here: regression residual comparison for additive and interaction models.]  
[Figure 3.20 about here: coupling heatmap of `kappa = p*t_s`.]  
[Figure 3.21 about here: regime map for weak, moderate, and strong coupling.]  
[Figure 3.22 about here: optimal `q*(t_s)` under lifetime constraints.]  
[Figure 3.23 about here: iso-contours of delay and lifetime over the `(q, t_s)` plane.]  
[Table 3.5 about here: statistical summary comparing additive and interaction models.]

This section provides the strongest single conceptual takeaway of the report: sleep behavior and access behavior must generally be designed jointly, not sequentially.

---

# Chapter 4 Extension Studies

The core results establish a validated baseline and a clear design story. The purpose of this chapter is different. Rather than redefining the main contribution, the extension studies test how far the simulator can be pushed and what additional insights emerge when the baseline assumptions are relaxed or extended.

## 4.1 Finite Retry Limits

The baseline model assumes effectively unbounded retransmission, which is analytically convenient but not always realistic. Objective O6 therefore introduces a retry limit `K` and studies how it changes delay, packet drop rate, and effective service behavior. This modification adds a new design trade-off: shorter head-of-line service can reduce delay, but only by accepting non-zero packet loss.

According to the task record, the simulator was extended with a configurable retry counter, drop tracking, and a finite-retry analytical service-rate expression `mu_K`. The resulting study shows that moderate retry limits can reduce delay while keeping drop rates manageable, whereas very small retry budgets create an unfavorable loss penalty. The extension is especially useful because it makes explicit a trade-off that is absent from the infinite-retry baseline.

[Figure 4.1 about here: delay and drop rate versus retry limit.]  
[Figure 4.2 about here: delay-drop Pareto view for varying `K`.]  
[Table 4.1 about here: analytical versus empirical `mu_K` comparison.]

## 4.2 CSMA Comparison

Objective O7 compares the baseline slotted Aloha scheme with a CSMA-based alternative. The motivation is straightforward: carrier sensing can reduce collisions, but it also changes the activity pattern of the nodes and may introduce deferral overhead that interacts with sleep behavior.

The extension adds an access-scheme abstraction to the simulator and implements a CSMA-1-persistent style alternative with optional backoff behavior. The completed experiments compare delay, throughput, and collision behavior across several node populations and traffic loads. The key result reported in the task record is that CSMA can outperform Aloha in lighter or less collision-prone regimes, but that its advantage shrinks and can reverse as population size grows and the cost of repeated deferral becomes more significant.

[Figure 4.3 about here: delay comparison between CSMA and Aloha.]  
[Figure 4.4 about here: collision comparison between CSMA and Aloha.]  
[Figure 4.5 about here: throughput versus load for both schemes.]

## 4.3 Capture and SIC Receivers

Objective O8 relaxes the strict collision model of the baseline by introducing advanced receiver behavior. In the collision baseline, any multi-node transmission in a slot is treated as a complete failure. Capture and SIC challenge that assumption by allowing one or more packets to be decoded under suitable signal conditions.

The project implements three receiver models: the original collision model, a probabilistic capture model, and a SIC model that can iteratively decode multiple transmissions. These extensions alter the effective success probability and therefore the delay and lifetime behavior of the system. The task record reports that SIC provides the strongest gains in dense or collision-heavy regimes, while capture gives a smaller but still useful improvement over the pure collision baseline.

[Figure 4.6 about here: effective success probability under collision, capture, and SIC.]  
[Figure 4.7 about here: delay comparison under the three receiver models.]  
[Figure 4.8 about here: lifetime comparison under the three receiver models.]

## 4.4 Age of Information

Objective O9 adds Age of Information as a timeliness metric. This extension is important because delay and freshness are related but not identical. A system can have acceptable queueing delay while still delivering stale information if updates are infrequent or if the sampling-access interaction is unfavorable.

The project instruments the simulator to track AoI histories and to report mean and peak-style freshness metrics. The resulting experiments compare AoI-optimal parameter choices with delay-optimal choices across several values of `t_s`. The task record indicates that the AoI-optimal transmission probability is generally higher than the delay-optimal one for moderate sleep settings, showing that freshness-oriented operation may favor more aggressive updates than queueing-delay optimization alone would suggest.

[Figure 4.9 about here: AoI versus `q` for several `t_s` values.]  
[Figure 4.10 about here: AoI-optimal versus delay-optimal `q*`.]  
[Figure 4.11 about here: AoI-delay-lifetime trade-off projection.]

## 4.5 MMBP Arrivals

Objective O10 extends the project beyond Bernoulli-style arrivals by introducing a two-state Markov-modulated Bernoulli process. This extension addresses a common limitation of analytical queueing models: real traffic can be correlated or bursty even when its long-run mean is unchanged.

The project formalizes the MMBP traffic model in the simulator and implements a corresponding analytical expression for the service process under correlated arrivals. The central question is not simply whether MMBP traffic behaves differently, but when the simpler Bernoulli approximation becomes unreliable. According to the task record, the resulting analysis identifies a burstiness threshold, reported as approximately `BI ~= 2-3`, beyond which the Bernoulli approximation accumulates enough error that the dedicated MMBP treatment becomes preferable.

[Figure 4.12 about here: analytical versus empirical `mu` under MMBP arrivals.]  
[Figure 4.13 about here: prediction error versus burstiness index.]  
[Table 4.2 about here: Bernoulli versus MMBP error summary.]

Taken together, the five extension studies support two points. First, the simulator is flexible enough to support meaningful modifications beyond the baseline. Second, the central core story of the report remains intact even when richer assumptions are introduced: sleep and access design remain tightly linked, and design choices continue to trade off latency-related performance against energy behavior.

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

The strongest design implication of the report is that `q` and `t_s` should not be tuned as if they were independent control knobs. The coupling term `p*t_s` means that sleep and access behavior interact in the service process. As a result, a sequential procedure that first fixes `q` and then tunes `t_s`, or vice versa, can miss better joint operating points. The report therefore supports a co-optimization mindset, especially outside the weak-coupling regime.

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

The report is subject to the limitations already discussed: a homogeneous-node baseline, simplified channel abstraction, 3GPP-inspired rather than standard-faithful mappings, and partial reliance on abstract traffic and receiver models. In addition, several sections of the present manuscript still require final citation tightening and direct insertion of verified figure-specific numerical values.

## 6.4 Future Work

The most credible next steps are those that follow naturally from the current framework. These include heterogeneous node populations, adaptive closed-loop control of `q` and `t_s`, richer physical-layer models, multi-channel access, and tighter integration between analytical approximation and simulation under correlated traffic. Among the existing extensions, AoI and MMBP arrivals appear especially promising because they directly challenge assumptions that are often simplified in baseline queueing studies.

From a software perspective, future work could also strengthen the usability of the framework by adding a more formal user interface, richer experiment configuration files, and more automated report-generation pipelines for figures and tables.

---

# References

[Insert final IEEE-style references here.]

Minimum expected entries:

1. Wang et al. (2024), the analytical baseline paper.  
2. Relevant 3GPP specifications for PSM, MICO mode, `T3324`, and access procedures.  
3. SimPy documentation or equivalent simulation reference.  
4. Background slotted Aloha literature if retained in the final introduction.

---

# Appendices

## Appendix A User Guide

This appendix should summarize how to install dependencies, run the tests, execute the validation script, and open the example notebooks. The repository README already provides a good starting point for this material and can be adapted into formal report prose.

Suggested content:

- environment setup
- dependency installation
- running `pytest`
- using `run_validation.py`
- opening the example notebooks in `examples/`

## Appendix B Source Code Structure

This appendix should describe the main source modules and their roles, for example:

- `src/node.py`
- `src/simulator.py`
- `src/power_model.py`
- `src/metrics.py`
- `src/experiments.py`
- `src/validation.py`
- `src/traffic_models.py`
- `src/receiver_models.py`
- `src/mmbp_analytics.py`

## Appendix C Additional Figures and Tables

This appendix should contain detailed plots or larger tables that support the main body but are too bulky to include directly, such as full design-guideline tables, additional parameter sweeps, or supplementary extension figures.

## Appendix D Selected Code Listings

If required by the final template or supervisor guidance, this appendix can include selected code excerpts such as the main simulator loop, validation routines, or independence-analysis utilities.
