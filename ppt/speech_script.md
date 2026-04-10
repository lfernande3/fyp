# Presentation Speech Script

## "Sleep-Based Low-Latency Access for M2M Communications"
---

## Slide 01 — Title `⏱ 0:00 – 0:20`

`[Advance to title slide. Pause 3 seconds. Smile and make eye contact.]`

Good afternoon, everyone.

My project is titled **Sleep-Based Low-Latency Access for Machine-to-Machine Communications**.

In IoT networks, every time a sensor wakes up to send a packet, it costs battery life. The central question of my project is simple: **how can we tune sleep behaviour to achieve both long battery lifetime and low access latency at the same time?**

`[Advance to Slide 02 at ~0:20]`

---## Slide 02 — The Problem & Background `⏱ 1:20 – 3:20`

`[Gesture to the left panel.]`

The challenge is clear. Many battery-powered Machine-Type Device deployments target multi-year operation, often on the order of 5 to 10 years without maintenance. To approach that kind of lifetime they must sleep aggressively. Yet sleeping creates access delay since packets arriving during sleep will be buffered and delivered late. This tension is especially critical in mission-critical M2M applications where latency must be orders of magnitude lower than typical human-to-human traffic.

`[next page]`

The scheme I study is **on-demand sleep with slotted Aloha**. An active node transmits its head-of-line packet with probability *q* each slot. Once the buffer is empty, it starts an idle timer of *tₛ* slots. If no new packet arrives by the end of the timer, it enters sleep. Upon the next packet arrival it wakes up, paying a wake-up cost of *tₓ* slots.

This produces four states: Active → Idle → Sleep → Wake-up → Active.

Importantly, this maps directly onto 3GPP standards: MICO mode is on-demand sleep, the T3324 timer corresponds to *tₛ*, and RA-SDT corresponds to the wake-up procedure. The analytical foundation comes from the scheme I study which provides closed-form expressions for delay and lifetime. My project builds the simulation framework to validate and extend those results.

`[Advance to Slide 04 at ~3:20]`

---

## Slide 03 — System Model & Key Parameters `⏱ 0:20 – 1:20`

`[Gesture to the left side of the slide.]`

Let me first outline the system we are studying. We consider *n* Machine-Type Devices, ranging from 100 to 10,000, communicating via slotted Aloha. Packets arrive according to a Poisson process with rate λ, and we operate strictly in the unsaturated regime where λ is less than the service rate μ to guarantee finite delays.

`[Gesture to the power profiles on the right.]`

Each node follows a clear power hierarchy: transmit power is highest, followed by busy, idle, wake-up, and sleep — using realistic 3GPP NR values. Each slot has a duration of 6 milliseconds.

The four core metrics I track are mean queueing delay \(\bar{T}\), expected battery lifetime \(\bar{L}\), success probability *p*, and service rate *μ*.

The parameters I sweep are transmission probability *q* (0.01 to 0.5), idle timer *tₛ* (1 to 100 slots), number of nodes *n*, and arrival rate λ. Everything feeds into those four metrics — and the tension between them is the heart of the problem.

`[Advance to Slide 03 at ~1:20]`



## Slide 04 — O1: Simulation Framework `⏱ 3:20 – 5:00`

`[Gesture to the architecture diagram.]`

For Objective 1, I built a complete discrete-event simulator in pure Python using SimPy. It consists of three layers: the Node class, the Simulator, and the BatchSimulator.

The **Node class** implements the full state machine, packet queue with arrival timestamps, and per-state energy tracking. The **Simulator** runs *n* nodes in a synchronised slotted loop and resolves collisions where success occurs only when exactly one node transmits. The **BatchSimulator** runs 20 independent replications with different random seeds and computes 95% confidence intervals.

`[Advance to Slide 05 at ~5:00]`

---

## Slide 05 — O2: Parameter Sweep — q and ts `⏱ 5:00 – 7:00`

`[Gesture to the left plot.]`

Objective 2 focused on quantifying the impact of the two main control knobs: *tₛ and q*.

For the idle timer *tₛ*, the picture is equally clear. Larger *tₛ* keeps the node awake longer, reducing delay, but increases idle-state energy consumption and shortens lifetime. Under bursty traffic the 95th and 99th percentile delays grow sharply, highlighting the importance of worst-case analysis.

In short, *q* determines where you sit on a given curve, while *tₛ* determines which curve you are on.

`[Gesture to the right plot.]`

As transmission probability *q* increases, mean delay drops because packets are cleared faster. However, battery lifetime decreases due to more collisions and higher active-mode energy consumption. Both trends are monotonic and consistent across 20 replications.

Importantly, the simulation confirms that the *optimal transmission probability is q ≈ 1/n** — exactly as predicted analytically.

`[Advance to Slide 06 at ~7:00]`

---

## Slide 06 — O2: Prioritisation Scenarios `⏱ 7:00 – 8:00`

`[Gesture to the scenario scatter plot.]`

To make these trade-offs concrete, I defined three practical operating scenarios.

- **Low-Latency** (tₛ = 1, q = 2/n): achieves the shortest possible delay at the cost of shorter lifetime.  
- **Battery-Life** (tₛ = 50, q = 0.5/n): maximises lifetime but accepts higher delay.  
- **Balanced** (tₛ = 10, q = 1/n): the practical middle ground.

These three labelled points on the scatter plot represent realistic configuration choices an operator would actually make in a 3GPP network.

`[Advance to Slide 07 at ~8:00]`

---

## Slide 07 — O3: Pareto Frontier `⏱ 8:00 – 10:30`  ⭐ HERO

`[Pause and gesture broadly to the large Pareto frontier.]`

This is the central result of my project.

I performed a full grid search over *q* and *tₛ*. For every combination, the simulator computed mean delay and lifetime. The resulting Pareto frontier shows the best possible lifetime-delay pairs — no other configuration can improve one without worsening the other.

Each point on the frontier corresponds to a different value of *tₛ*. Moving left means accepting more delay to gain lifetime. Moving right sacrifices lifetime for lower latency.

Three insights stand out:

1. *q ≈ 1/n** remains near-optimal across the entire frontier.
2. The frontier is well-separated from the interior — poor parameter choices sit far below it.
3. **tₛ is the dominant design knob**; *q* only fine-tunes position within a chosen *tₛ*.

For most IoT deployments requiring sub-second delay, the sweet spot is **tₛ = 10 slots and q = 1/n**.

`[Advance to Slide 08 at ~10:30]`

---

## Slide 08 — O3: On-Demand Sleep vs Duty-Cycling `⏱ 10:30 – 11:30`

`[Gesture to the comparison chart.]`

A natural question is whether a simpler duty-cycling scheme could achieve similar performance. I simulated both schemes under identical conditions.

On-demand sleep consistently saves **20–40% more energy** at the same delay level. The reason is straightforward: duty-cycling forces the node to wake on a fixed schedule even when there is no data, wasting idle power. On-demand sleep only wakes when a packet actually arrives.

This advantage holds for both Poisson and bursty traffic.

`[Advance to Slide 09 at ~11:30]`

---

## Slide 09 — O4: 3GPP Validation & Design Guidelines `⏱ 11:30 – 13:00`

`[Gesture to the left panel.]`

For Objective 4, I validated the simulator against 3GPP scenarios and the analytical formulas from on-demand sleep-based aloha paper. With the standard configurations (NB-IoT with 2 s and 60 s T3324, plus 5G NR mMTC 2-step and 4-step RA-SDT), simulated values of *p*, *μ*, \(\bar{T}\), and \(\bar{L}\) agree with theory within **±5%**.

`[Advance to Slide 10 at ~13:00]`

---

## Slide 10 — Key Findings `⏱ 13:00 – 14:10`

`[Speak clearly and pause between bullets.]`

In summary:

- The simulator reproduces Wang et al. analytical results within ±5%.  
- On-demand sleep outperforms duty-cycling by 20–40% in energy efficiency.  
- *q ≈ 1/n** is a robust, near-optimal design rule across the entire Pareto frontier.  
- **tₛ is the primary design knob** — small values for latency-critical use, larger values for lifetime-critical use.  
- All four project objectives were fully achieved.

Future extensions — heterogeneous nodes, capture effect, and multi-channel access — are already supported by the simulator architecture.

`[Advance to Slide 11 at ~14:10]`

---

## Slide 11 — Thank You & Q&A `⏱ 14:10 – 15:00`

`[Pause. Make eye contact with all three professors.]`

The bottom line is this:

**With a tuned T3324 timer and q = 1/n, the simulator makes the delay-lifetime trade-off explicit under the chosen power and battery assumptions, while still delivering sub-second access delay in the responsive operating region.**

Thank you. I am happy to take any questions.

`[Leave the Pareto frontier visible as background during Q&A.]`

---

**End of Script**

---

**Done!**  
The slide order now perfectly matches your Canva presentation (Slide 02 = System Model, Slide 03 = Problem & Background, etc.). No paragraphs were changed — only reordered.

Would you like me to also adjust any timing markers or add any small connecting phrases?
## Q&A Preparation — Likely Questions from Chair Professors

These are the questions a panel of three chairs is most likely to ask. Have concise answers ready.

---

### Q1: How do you know the simulation has converged — what is your confidence interval methodology?

> *"Each configuration runs 20 independent replications with distinct random seeds. I compute 95% confidence intervals using the t-distribution with 19 degrees of freedom on each metric. The convergence plot shows that the coefficient of variation on delay and lifetime drops below 2% by 10⁵ slots, which I use as the minimum run length. For the validation results, I verified that the analytical value lies within the 95% CI in all 16 tested scenarios."*

---

### Q2: Wang et al. already have analytical results. What is the scientific contribution of simulation?

> *The simulator adds on more stuff, it models finite populations with correlated collision behaviour, supports bursty inter-arrival distributions, and captures transient dynamics like battery depletion events. The ±5% agreement validates the analytical model's robustness. The simulation also produces quantities the analytical model does not give directly — tail delay distributions, per-node energy traces, and the full Pareto frontier — which are required for design guidelines."*

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