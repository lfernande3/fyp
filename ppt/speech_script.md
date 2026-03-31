# Presentation Speech Script

## "Sleep-Based Low-Latency Access for M2M Communications"

### Final Year Project — 15-Minute Presentation

> **Audience:** Three chair professors of the department  
> **Tone:** Confident, technically precise, academically grounded  
> **Pace:** ~130 words/min · ~1,950 words total  
>
> **Conventions:**  
> `[ACTION]` — stage direction or slide cue  
> `(~Xs)` — approximate elapsed time at that point  
> Bold = emphasis to stress aloud  
> Italics = optional elaboration if time allows

---

## Slide 01 — Title `⏱ 0:00 – 0:20`

`[Advance to title slide. Pause 3 seconds. Make eye contact with all three professors before speaking.]`

Good afternoon. My project is titled *Sleep-Based Low-Latency Access for M2M Communications*.

In IoT networks, every time a sensor wakes up to send a packet, it costs battery life. My project asks: **how do we tune that sleep behaviour to satisfy both goals at once** — minimising access delay while maximising battery lifetime?

`[Advance to Slide 02 at ~0:20]`

---

## Slide 02 — System Model & Key Parameters `⏱ 0:20 – 1:20`

`[Gesture to the parameter table on the left.]`

Before I get into the problem, let me ground the model. The system we study has *n* nodes — between 100 and 10,000. Packet arrivals follow a **Poisson process** with rate λ — and we restrict ourselves to the unsaturated regime, where λ is strictly less than the service rate μ, ensuring finite delays.

`[Gesture to the power profile chart on the right.]`

Power consumption follows a strict hierarchy: transmit power is the highest, followed by busy-receive, idle, wakeup, and sleep — corresponding to 3GPP NR values. The slot duration is **6 milliseconds**, consistent with 5G NR mMTC.

The four metrics I track are: **mean delay T-bar, battery lifetime L-bar, success probability p, and service rate μ**.

The parameters I sweep are: transmission probability *q* from 0.01 to 0.5, idle timer *ts* from 1 to 100 slots, population *n* from 10 to 500, and arrival rate λ from 0.001 to 0.1.

Everything feeds into those four metrics — and the tension between them is what makes this problem interesting.

`[Advance to Slide 03 at ~1:20]`

---

## Slide 03 — The Problem & Background `⏱ 1:20 – 3:20`

`[Gesture to the left panel — the motivation.]`

Now with the model in mind, the problem becomes sharp. We are dealing with networks of **battery-powered Machine-Type Devices** — IoT sensors, smart meters, environmental monitors — numbering in the billions. These devices send small, infrequent packets. They must sleep aggressively just to survive on a coin cell for years.

But sleeping is precisely what creates the problem. When a node is asleep, it misses its transmission window. The result is **higher access delay** — which is unacceptable in applications like industrial monitoring or emergency alerting.

This is the core tension: **minimising delay requires the node to stay awake; maximising lifetime requires the node to stay asleep.**

`[Gesture to the right panel — the state machine.]`

The baseline scheme we study is **on-demand sleep with slotted Aloha**. Each active node transmits with probability *q* per slot. After its queue drains, it enters an idle timer of *ts* slots; if no new packet arrives by then, it sleeps. On the next arrival, it wakes up — paying a wake-up cost of *tw* slots.

This gives us four states: Active, Idle, Sleep, and Wakeup.

Critically, this maps directly onto **3GPP standards**: MICO mode corresponds to on-demand sleep, the T3324 timer corresponds to *ts*, and RA-SDT corresponds to the wake-up procedure.

The analytical foundation comes from **Wang et al., 2024** — my supervisor's published work — which derives closed-form expressions for delay and lifetime. My project builds on and validates that analytical framework using simulation.

`[Advance to Slide 04 at ~3:20]`

---

## Slide 04 — O1: Simulation Framework `⏱ 3:20 – 5:00`

`[Gesture to the architecture diagram on the left.]`

Objective one was to build the simulation framework itself. I implemented a **pure Python, SimPy-based discrete-event simulator** — three layers: the Node class, the Simulator, and the BatchSimulator.

The **Node class** is a full state machine with a packet queue and an energy tracker. The **Simulator** runs *n* nodes in a synchronised slotted loop and detects collisions — a transmission succeeds only if exactly one node transmits in a slot. The **BatchSimulator** runs 20 independent replications per configuration and computes **95% confidence intervals** on all metrics.

`[Gesture to the time-series plot on the right.]`

This trace shows queue length, energy, and node state evolving over time for a single node. The simulator runs stably at **80,000 slots per replication** — long enough to fully deplete batteries and accurately measure tail delays.

I also implemented six power profiles — NB-IoT, LoRa, LTE-M, 5G NR mMTC, and a generic profile — so the same framework can be reused across different hardware assumptions.

`[Advance to Slide 05 at ~5:00]`

---

## Slide 05 — O2: Parameter Sweep — q and ts `⏱ 5:00 – 7:00`

`[Gesture to the q-sweep plot on the left.]`

Objective two was to quantify the impact of *q* and *ts*. Starting with the transmission probability: as *q* increases, mean delay falls — the node transmits more aggressively, so packets clear the queue faster. But battery lifetime falls sharply at the same time.

Both effects are **monotonic** and confirmed across all 20 replications, with tight confidence intervals.

The key finding here is that the **optimal transmission probability is q-star equals 1 over n**. Beyond that, each incremental reduction in delay costs a disproportionate amount of lifetime. This confirms the analytical prediction from Wang et al.

`[Gesture to the ts-sweep plot on the right.]`

Now for the idle timer. As *ts* increases, the node stays awake longer between packets, which *reduces* delay — because it's already active when the next packet arrives. But it also consumes more idle-state power, **shortening lifetime**.

I also characterised **bursty traffic** — under a Pareto inter-arrival distribution, the 95th and 99th percentile delays grow sharply relative to Poisson traffic. This is important for dimensioning worst-case latency.

The key insight is this: **every point on those scatter plots is a design choice**. *q* controls where you sit on a given curve. *ts* controls which curve you are on. That naturally motivates the next two slides.

`[Advance to Slide 06 at ~7:00]`

---

## Slide 06 — O2: Prioritisation Scenarios `⏱ 7:00 – 8:00`

`[Gesture to the scatter plot.]`

To make the trade-off concrete, I defined three operating scenarios. The **Low-Latency scenario** uses a short idle timer of *ts* = 1 and a higher *q* = 2/n — it achieves the minimum possible delay at the cost of a shorter battery life. The **Battery-Life scenario** uses *ts* = 50 and *q* = 0.5/n — the node extends to years of operation but accepts significantly higher delay. The **Balanced scenario** sits in between at *ts* = 10, *q* = 1/n.

These three labelled points on the scatter are not arbitrary — they represent the **deployable parameter choices** an operator would actually configure in a 3GPP network via the T3324 timer.

This slide bridges what-happens into what-is-optimal, which is the Pareto analysis.

`[Advance to Slide 07 at ~8:00]`

---

## Slide 07 — O3: Pareto Frontier `⏱ 8:00 – 10:30`  ⭐ HERO

`[Gesture broadly to the large Pareto frontier plot. Pause.]`

This is the central result of the project.

I performed a **grid search over the full (q, ts) parameter space**, sweeping *q* from 0.01 to 0.35 and *ts* from 1 to 100. For each combination, the BatchSimulator produced mean delay and mean lifetime. The result is a set of Pareto-optimal points — where no configuration can improve lifetime without worsening delay, or vice versa.

`[Trace along the frontier slowly.]`

Each dot on this frontier corresponds to a different value of *ts*. Moving **left along the frontier** means accepting more delay in exchange for more lifetime. Moving **right** sacrifices lifetime for lower latency.

Three things stand out.

First: **q-star equals 1/n holds near-optimally across the entire frontier** — not just at one operating point. This validates the analytical prediction as a robust design rule.

Second: the frontier is **well-separated from the interior** of the parameter space. Poorly chosen parameters sit far from the frontier — meaning operators who default to, say, *q* = 0.3 and *ts* = 1 are leaving significant lifetime on the table.

Third: `[gesture to the heatmap inset]` the 2-D heatmaps confirm that **ts is the primary knob** — it shifts which curve you are on. *q* fine-tunes your position within a chosen ts. The practical implication is that an operator can set *ts* based on their latency budget and then fix *q* = 1/n with confidence.

For most IoT deployments — where access delays under two seconds are acceptable — the sweet spot is **ts = 10, q = 1/n**.

`[Advance to Slide 08 at ~10:30]`

---

## Slide 08 — O3: On-Demand Sleep vs. Duty-Cycling `⏱ 10:30 – 11:30`

`[Gesture to the comparison plot.]`

A natural question is: why not just use **duty-cycling** — a simpler scheme where the node wakes on a fixed periodic schedule regardless of whether a packet has arrived?

My simulator ran both schemes under **identical conditions** — same *n*, same λ, same power profile. On-demand sleep **saves 20 to 40 percent more energy** at equivalent delay across all tested configurations.

The reason is intuitive: duty-cycling wastes idle-state power waking up when there is nothing to transmit. On-demand sleep only wakes when a packet actually arrives.

This is not just a restatement of the analytical claim from the paper — it is a simulation-confirmed result, including under bursty traffic conditions where the advantage of on-demand sleep is even more pronounced.

`[Advance to Slide 09 at ~11:30]`

---

## Slide 09 — O4: 3GPP Validation & Design Guidelines `⏱ 11:30 – 13:00`

`[Gesture to the validation bar chart on the left.]`

Objective four was to validate the simulator against 3GPP-realistic scenarios and the Wang et al. analytical formulas. I tested four standard configurations: NB-IoT with a 2-second T3324 timer, NB-IoT with a 60-second timer, 5G NR mMTC with 2-step RA-SDT, and 5G NR mMTC with 4-step RA-SDT.

In all four cases, the **simulated values of p, μ, T-bar, and L-bar agree with the analytical formulas to within ±5%** — well within the bounds expected from stochastic simulation.

The convergence analysis confirms that the error drops below 5% at 10⁵ slots and stabilises by 10⁶ slots. This tells us the simulator is **not a black box** — it has a principled, quantifiable relationship to theory.

`[Gesture to the design guideline table on the right.]`

From these validated results, I produced a **design guideline table** — for each traffic load λ, the table gives the recommended *ts* and *q-star* that minimises delay subject to a lifetime constraint. An operator can read directly from this table when configuring T3324 and RA-SDT parameters in a 5G NR deployment.

`[Advance to Slide 10 at ~13:00]`

---

## Slide 10 — Key Findings `⏱ 13:00 – 14:10`

`[Speak at a measured pace. Allow each point to land.]`

Let me summarise what was achieved.

The simulator **reproduces the Wang et al. analytical results within ±5%** — confirming both the implementation and the underlying theory.

**On-demand sleep outperforms duty-cycling by 20 to 40 percent** in energy efficiency at equivalent delay — a result that holds under both Poisson and bursty traffic.

**q-star = 1/n** is a robust, near-optimal design rule — not just asymptotically, but across the entire Pareto frontier.

**ts is the dominant design knob** — small ts for latency-critical applications, large ts for lifetime-critical deployments.

All four project objectives — the simulation framework, parameter quantification, Pareto optimisation, and 3GPP validation — were fully completed.

For future work, the most interesting extension is **heterogeneous node populations**, where different devices have different traffic profiles and energy budgets. The capture effect and multi-channel access are also natural next steps that the simulator architecture already supports.

`[Advance to Slide 11 at ~14:10]`

---

## Slide 11 — Thank You & Q&A `⏱ 14:10 – 15:00`

`[Pause. Make eye contact. Deliver the closing line slowly and clearly.]`

The bottom line is this:

**With a T3324 timer of 10 slots and q = 1/n, IoT devices achieve sub-second access delay while lasting over two years on a single AA battery — and now we have a simulator to prove it.**

Thank you. I am happy to take questions.

`[Leave the Pareto frontier slide visible as background during Q&A.]`

---

## Q&A Preparation — Likely Questions from Chair Professors

These are the questions a panel of three chairs is most likely to ask. Have concise answers ready.

---

### Q1: How do you know the simulation has converged — what is your confidence interval methodology?

> *"Each configuration runs 20 independent replications with distinct random seeds. I compute 95% confidence intervals using the t-distribution with 19 degrees of freedom on each metric. The convergence plot shows that the coefficient of variation on delay and lifetime drops below 2% by 10⁵ slots, which I use as the minimum run length. For the validation results, I verified that the analytical value lies within the 95% CI in all 16 tested scenarios."*

---

### Q2: Wang et al. already have analytical results. What is the scientific contribution of simulation?

> *"The analytical model makes several simplifying assumptions — independence of node states, Poisson arrivals, and steady-state approximations. The simulator relaxes these: it models finite populations with correlated collision behaviour, supports bursty inter-arrival distributions, and captures transient dynamics like battery depletion events. The ±5% agreement validates the analytical model's robustness. The simulation also produces quantities the analytical model does not give directly — tail delay distributions, per-node energy traces, and the full Pareto frontier — which are required for design guidelines."*

---

### Q3: Why slotted Aloha specifically — what about CSMA or more sophisticated MAC protocols?

> *"Slotted Aloha is the access scheme specified in 3GPP NR for mMTC random access — specifically the 2-step and 4-step RA-SDT procedures map directly onto this model. More sophisticated protocols like CSMA assume carrier sensing, which is impractical at the energy budgets of IoT devices. The contribution here is the sleep layer on top of the MAC, which is what Wang et al. — and this project — specifically study."*

---

### Q4: What is the impact of your assumption that λ < μ — how sensitive are your results to this?

> *"The unsaturated regime is necessary for finite mean delay — in saturation, the queue grows without bound and lifetime drops to zero, making the trade-off degenerate. I verified that all tested configurations satisfy λ < μ by monitoring the empirical service rate from the simulation. At the boundary — λ ≈ 0.8μ — I observe the delay distribution developing a heavier tail, which is why the 95th and 99th percentile metrics are included. Entering saturation is a deliberate out-of-scope boundary, consistent with the Wang et al. paper."*

---

### Q5: Could this simulator be extended to heterogeneous nodes — is the architecture ready for that?

> *"Yes — the Node class is parameterised per-instance. Heterogeneous populations only require passing different (λ, ts, q, power profile) tuples to each Node at initialisation. The Simulator already handles n nodes independently. The main research question then becomes how to jointly optimise across node classes, which is a natural extension of the Pareto analysis."*

---

### Q6: Your validation is against the same paper the model is derived from — is that circular?

> *"It is an important distinction: the analytical formulas are derived under assumptions the simulation does not encode. The simulation implements the physical process — individual nodes, independent Bernoulli transmissions, slot-by-slot energy accounting — without referencing the analytical expressions. Agreement between the two is therefore non-trivial. It is analogous to verifying a numerical solver against a closed-form solution: the two computations are independent, and agreement is evidence that both are correct."*

---

## Timing Safety Cues


| Situation                        | Recovery                                                                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ahead by 1 min at Slide 06       | Expand the scenario comparison — walk through quantified gains vs. baseline                                                                                 |
| Behind by 1 min at Slide 07      | Cut the heatmap inset explanation; deliver only the three frontier findings                                                                                 |
| Behind by 2 min at Slide 07      | Skip Slide 08 (duty-cycling); mention it in one sentence: *"On-demand also outperforms duty-cycling by up to 40% — I have a slide on that if it comes up."* |
| Running short with 2 min left    | Expand the Pareto frontier walk-through — trace more points on the frontier                                                                                 |
| Deep methods question from panel | Pull up backup slide `slide10_publication_summary.png`                                                                                                      |


---

*End of Script*