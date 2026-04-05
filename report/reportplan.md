# FYP Report Plan — Page-by-Page Breakdown
**Project:** Sleep-Based Low-Latency Access for Machine-to-Machine Communications  
**Module:** EE4080 | Estimated Total Length: ~55–65 pages (excluding appendices)

---

## Front Matter

### Cover Page — p. 1
- Project title: *Sleep-Based Low-Latency Access for Machine-to-Machine Communications*
- Project code, student name & ID, major
- Supervisor name, Assessor name
- Semester B, 2024–25

### Student Declaration Form (Appendix V) — p. 2
- Signed plagiarism and academic honesty declaration
- Statement confirming the work is your own except where cited

### Abstract — p. 3
One page covering:
1. **Background** (2–3 sentences): Massive M2M/IoT deployments require battery-powered MTDs to balance low-latency access with long battery life; existing analytical models cannot capture stochastic variability.
2. **Problem** (1–2 sentences): How do sleep idle timer ts and transmission probability q jointly affect mean delay and battery lifetime in slotted Aloha with on-demand sleep?
3. **Methodology** (2–3 sentences): Discrete-event simulator built in Python/SimPy; full factorial parameter sweeps; analytical validation against Wang et al. (2024) formulas; 3GPP NR/NB-IoT power profiles.
4. **Main Results** (2–3 sentences): Simulator validates paper trends to within ±5%; q and ts are multiplicatively coupled via κ = p·ts; Pareto-optimal design guidelines produced for MICO/T3324 configurations.
5. **Conclusion** (1 sentence): Co-optimising q and ts delivers substantial lifetime gains (X×) at controlled delay cost versus single-parameter tuning.

### Acknowledgements — p. 4
- Thank supervisor for guidance
- Acknowledge Wang et al. (2024) as the foundational reference paper
- Brief note on use of open-source libraries (SimPy, NumPy, Matplotlib)

### Use of Generative AI (GenAI) Tools — p. 5
- State which tools were used (e.g., Cursor AI / Claude) and for what purposes (code scaffolding, debugging, report drafting)
- Confirm all technical content, simulation results, and analysis are your own
- List specific tasks where GenAI assisted vs. where it was not used
- Include relevant prompts or describe extent of use per the EE4080 checklist

### Table of Contents — p. 6
Auto-generated; list all chapters, sections, and appendices with page numbers.

### List of Figures — p. 7
All figures numbered Figure 3.x, with captions and page references.

### List of Tables — p. 8
All tables numbered Table X.x, with captions and page references.

---

## Chapter 1 — Introduction (pp. 9–16, ~8 pages)

### 1.1 Background and Motivation (pp. 9–11, ~3 pages)
- Growth of IoT/M2M: billions of battery-powered MTDs, 3GPP NR mMTC use cases
- Core tension: low-latency access vs. energy conservation (battery life)
- Why existing duty-cycling falls short; motivation for on-demand sleep with slotted Aloha
- Brief overview of 3GPP mechanisms: PSM, MICO mode, T3324 timer, RA-SDT; explain how they map to simulation parameters (ts, tw, q)
- Introduce Wang et al. (2024) as the paper being validated and extended

### 1.2 Problem Statement (p. 12, ~1 page)
- Formal statement of the latency–longevity trade-off problem
- Limitation of purely analytical approaches: no stochastic variability, no transient behavior, no sensitivity to edge cases
- Need for a discrete-event simulator as a complement to closed-form analysis

### 1.3 Project Objectives (p. 13, ~1 page)
List the four (or five) measurable objectives clearly:
- **O1:** Build a discrete-event simulation framework for slotted Aloha with on-demand sleep
- **O2:** Quantify the impact of ts, q, n, λ, tw, and traffic models on mean delay and lifetime
- **O3:** Optimize sleep and access parameters for Pareto-optimal latency–longevity trade-offs
- **O4:** Validate simulation output against Wang et al. analytical formulas and 3GPP mMTC parameters
- **O5:** Determine analytically and empirically whether q and ts are independent parameters

### 1.4 Scope and Constraints (p. 14, ~1 page)
- Pure Python + SimPy; no hardware emulation, no ns-3/OMNeT++
- Focus on unsaturated regime (λ < μ)
- Traffic models: Poisson and bursty arrivals; 3GPP NR power values; slot duration 6 ms
- Out of scope: physical layer, mobility, multi-channel, CSMA

### 1.5 Report Structure (p. 15, ~0.5 page)
- One-paragraph roadmap of the remaining chapters

### 1.6 Literature Review (pp. 15–16, ~1.5 pages)
- Wang et al. (2024): key analytical results for μ, p, T̄, L̄
- 3GPP TS 23.501 / 36.304: PSM and MICO mode specifications
- Prior work on slotted Aloha in IoT (e.g., Bianchi, de Hoog); gap this work fills
- SimPy and discrete-event simulation for network modelling (brief)

---

## Chapter 2 — Methodology (pp. 17–30, ~14 pages)

### 2.1 System Model (pp. 17–19, ~3 pages)
- Slotted time model; n homogeneous MTDs; slot duration 6 ms
- Packet arrival model: Bernoulli with rate λ per slot; bursty batch-arrival variant
- On-demand sleep state machine: Active → Idle (ts countdown) → Sleep → Wakeup (tw slots) → Active
- Figure 2.1: State transition diagram for a single MTD
- Key parameters table: n, q, ts, tw, λ, E_init; typical ranges used

### 2.2 Analytical Background (pp. 19–21, ~2 pages)
- Success probability: p = q(1 − q)^(n−1)
- Service rate: μ = p / (1 + p·ts + p·tw) from Wang et al. Eq. 12
- Mean delay T̄ (M/G/1 approximation, Eq. 3)
- Mean lifetime L̄ (Sec IV-A formula)
- Stability condition: λ < μ
- Coupling term κ = p·ts and its role in the independence analysis (O5 preview)

### 2.3 Simulator Architecture (pp. 21–24, ~3 pages)
- Figure 2.2: Software architecture diagram (Node → Simulator → BatchSimulator → MetricsCalculator)
- **Node class** (`src/node.py`): NodeState enum, queue (deque), energy tracking, idle_timer, arrive_packet, attempt_transmit, consume_energy
- **Simulator class** (`src/simulator.py`): slotted time loop, collision detection (success iff exactly 1 transmitter per slot), run_simulation, randomness control (fixed seeds)
- **BatchSimulator**: parameter sweeps, 20–50 replications per configuration, confidence intervals
- **PowerModel** (`src/power_model.py`): 6 predefined 3GPP-inspired profiles (NB-IoT, NR mMTC, LoRa, LTE-M, Generic Low/High); BatteryConfig (AA, AAA, coin cell, LiPo)

### 2.4 Metrics and Logging (pp. 24–25, ~1 page)
- **MetricsCalculator**: empirical p, μ, T̄, L̄, throughput, state fractions, energy breakdown, energy per successful packet, tail delay (95th percentile)
- Trace-level logging: per-slot node states, queue lengths, collisions, energy (configurable)
- Output formats: CSV per experiment, JSON summary, PNG/PDF plots

### 2.5 Experimental Design (pp. 25–27, ~2 pages)
- **O2 sweeps**: q ∈ [0.01, 0.5], ts ∈ [1, 100], n ∈ [10, 500], λ ∈ [0.001, 0.1]; 10^5–10^6 slots; 20–50 reps
- **O3 optimization**: grid search over (q, ts) space; Pareto frontier; three canonical scenarios (Low-Latency ts=1 q=2/n, Balanced ts=10 q=1/n, Battery-Life ts=50 q=0.5/n)
- **O4 3GPP validation**: MICO→on-demand sleep, T3324→ts, RA-SDT→tw; NB-IoT and NR mMTC scenarios; convergence analysis (error vs. slots)
- **O5 independence**: Full factorial 6×6 grid (q ∈ {0.005…0.2} × ts ∈ {1…50}), n=100, λ=0.01, 20 reps; regression with interaction term; F-test for H₀: c = 0
- Table 2.1: Summary of all experiments and their parameters

### 2.6 Validation Strategy (pp. 27–28, ~1 page)
- Sanity checks: no-sleep (ts=∞) matches pure Aloha delay; ts=0 increases delay; high-q causes collisions
- AnalyticsValidator: per-config comparison to analytical formulas with ±5 / ±10 / ±20% bands
- Stability flag: exclude configurations where λ ≥ μ from analysis

### 2.7 Implementation Notes (pp. 28–30, ~2 pages)
- Python 3.x + SimPy; environment: Google Colab; runtime < 30 min per full experiment
- Reproducibility: fixed seeds, logged per run; batch runner averages across different seeds
- Testing: 180+ unit tests across all modules (Node 8, Simulator 10, PowerModel 11, Validation 7, Metrics 33, Experiments 29, Visualization 37, Optimization 45, 3GPP/Validation 49, Independence 20)
- GitHub repo structure (optional appendix)

---

## Chapter 3 — Results (pp. 31–47, ~17 pages)

### 3.1 Simulator Validation (pp. 31–33, ~3 pages)
- **Figure 3.1**: Empirical vs. analytical p across n = 5, 10, 20, 50 (scatter + ±5% band)
- **Figure 3.2**: Empirical vs. analytical μ across same n values
- **Figure 3.3**: Convergence plot — |T̄_sim − T̄_analytical| vs. number of slots; convergence by ~10^5 slots
- **Table 3.1**: ValidationReport summary — pass/fail per metric, stability condition, % error at n=100
- Key finding: simulator reproduces paper trends to within ±5% for n ≥ 20 in stable regime

### 3.2 Parameter Impact — O2 Results (pp. 33–37, ~4 pages)
- **Figure 3.4**: T̄ vs. q for multiple ts values (fanning curves confirm coupling); include 95% CI shading
- **Figure 3.5**: L̄ vs. q for same ts values
- **Figure 3.6**: T̄ vs. ts for multiple q values
- **Figure 3.7**: L̄ vs. ts for multiple q values
- **Figure 3.8**: Throughput vs. λ for n = 10, 50, 100; saturation visible at λ → μ
- **Table 3.2**: State fraction breakdown (active, idle, sleep, wakeup) for the three canonical scenarios
- Key findings: increasing q monotonically reduces delay but accelerates energy drain; larger ts extends lifetime but increases delay

### 3.3 Optimization Results — O3 (pp. 37–41, ~4 pages)
- **Figure 3.9**: 2D heatmap of T̄ in the (q, ts) plane; annotate optimal region
- **Figure 3.10**: 2D heatmap of L̄ in the (q, ts) plane; annotate stable boundary (λ = μ contour)
- **Figure 3.11**: Pareto frontier — maximum L̄ vs. minimum T̄ for varying ts; Pareto points labeled
- **Figure 3.12**: Bar chart — % change in T̄ and L̄ for Low-Latency and Battery-Life scenarios vs. Balanced baseline
- **Figure 3.13**: Duty-cycling vs. on-demand sleep comparison — T̄ and L̄ side by side; on-demand outperforms
- **Table 3.3**: Prioritization scenario summary (ts, q, T̄, L̄, gains vs. baseline)
- Key findings: on-demand sleep dominates duty-cycling; q* = 1/n is near-optimal for balanced performance

### 3.4 3GPP Validation and Design Guidelines — O4 (pp. 41–44, ~3 pages)
- **Figure 3.14**: L̄ vs. λ for 4 T3324 timer settings (ts = 5, 10, 30, 60 slots / 0.03–0.36 s); matches MICO guidance
- **Figure 3.15**: T̄ vs. λ same settings; recommended T3324 for <1 s delay SLA annotated
- **Figure 3.16**: q* vs. n (optimal q per population size); q* = 1/n rule validated
- **Figure 3.17**: 3GPP scenario scatter — NB-IoT vs. NR mMTC profiles in (T̄, L̄) space
- **Table 3.4**: Design guideline table — λ, recommended ts, T3324 equivalent, q*, T̄, L̄, stability flag
- Key findings: T3324 ≤ 0.18 s (ts ≤ 30 slots) meets <1 s delay for λ ≤ 0.01; q* ≈ 1/n holds for n ≥ 20

### 3.5 Independence Analysis — O5 (pp. 44–47, ~3 pages)
- **Figure 3.18**: Interaction plots (2×2 panel) — T̄ and L̄ vs. q stratified by ts, and vs. ts stratified by q; fanning confirms interaction
- **Figure 3.19**: Regression residual plot — additive model residuals correlate with the other variable; interaction model eliminates pattern
- **Figure 3.20**: Coupling heatmap — κ = p·ts in the (q, ts) plane; κ = 0.1 and κ = 1 boundaries annotated
- **Figure 3.21**: Regime map — near-independent (κ < 0.1), moderately coupled (0.1 ≤ κ < 1), strongly coupled (κ ≥ 1) regions in (q, ts) plane
- **Figure 3.22**: q*(ts) shift — delay-minimising q* under L̄ ≥ 3 yr and ≥ 5 yr constraints; monotone trend vs. flat (independent) baseline
- **Figure 3.23**: Iso-contour plot — L̄ filled contourf with dashed T̄ iso-lines; curvature visible
- **Table 3.5**: F-test results — F-statistic, p-value, R² (additive), R² (interaction) for T̄ and L̄ models
- Key finding: q and ts are NOT independent; coupling is driven by multiplicative cross-term p·ts in the service rate denominator; they are approximately independent only when κ = p·ts < 0.1

---

## Chapter 4 — Discussion (pp. 48–54, ~7 pages)

### 4.1 Interpretation of Validation Results (p. 48, ~1 page)
- Why simulator matches paper formulas for n ≥ 20 but deviates at small n: finite-population effects, stochastic correlations not captured by mean-field analysis
- Convergence by ~10^5 slots justifies experiment lengths chosen

### 4.2 Parameter Impact vs. Objectives (pp. 49–50, ~1.5 pages)
- Discuss monotonicity of T̄ and L̄ with q and ts consistent with paper Sec IV
- Explain why bursty traffic widens delay distribution (higher tail delay) without affecting mean lifetime significantly
- Compare Poisson vs. bursty regime at same mean λ

### 4.3 Optimization and Practical Implications (pp. 50–51, ~1.5 pages)
- Significance of Pareto frontier: there is no single "best" configuration; the trade-off is inherent
- Why q* ≈ 1/n rule works: at this probability, the expected number of simultaneous transmissions ≈ 1
- On-demand vs. duty-cycling: quantify the gain; explain why on-demand adapts better to low-traffic periods

### 4.4 3GPP Alignment and Real-World Relevance (p. 52, ~1 page)
- Map T3324 timer settings to concrete battery life estimates for AA battery NB-IoT device
- Discuss RA-SDT two-step vs. four-step access (tw = 2 vs. 4 slots) trade-off visible in results
- Limitations: simulator assumes homogeneous nodes and ideal channel; real NB-IoT includes coverage enhancement (CE) modes not modeled

### 4.5 Independence Analysis — What It Means for Designers (pp. 53–54, ~2 pages)
- Formal answer: q and ts are NOT independent; the coupling term κ = p·ts appears in the service rate formula
- Practical consequence: optimizing q with fixed ts and then optimizing ts with fixed q (sequential approach) is suboptimal; co-optimization is necessary
- When the approximation of independence is valid (κ < 0.1): sparse traffic, quick sleep — gives a simple rule of thumb
- Comparison with prior work that treats them as independent parameters; why this matters

---

## Chapter 5 — Conclusion (pp. 55–57, ~3 pages)

### 5.1 Summary of Achievements (p. 55, ~1 page)
- O1: Production-ready discrete-event simulator (Node, Simulator, PowerModel, Validation modules; 180+ passing tests)
- O2: Full parameter sensitivity mapped; bursty and Poisson traffic models; interactive visualization
- O3: Pareto optimizer, prioritization analyzer, duty-cycling comparison
- O4: Validated against Wang et al. to ±5%; design guideline table for T3324/RA-SDT/q* produced
- O5: Analytical and empirical proof that q and ts are coupled; regime map and design consequences

### 5.2 Revisiting Objectives (p. 56, ~0.5 page)
- Short table or bullet list mapping each objective (O1–O5) to the specific results that demonstrate its achievement

### 5.3 Limitations (p. 56, ~0.5 page)
- Homogeneous nodes; ideal collision channel; no fading or capture effect
- Unsaturated regime only; extensions to heterogeneous nodes not implemented
- Simulator runtime limits very large n (> 500) experiments

### 5.4 Future Work (p. 57, ~1 page)
- Heterogeneous MTD populations (different ts, q, power profiles)
- Capture effect and imperfect collision detection
- Integration of non-Bernoulli arrivals (e.g., Markov-modulated)
- Extension to multi-channel Aloha and NOMA
- Closed-loop controller that adapts q and ts dynamically based on queue state

---

## References (pp. 58–60, ~2–3 pages)
IEEE-style numbered list. Key references to include:
1. Wang et al. (2024) — the foundational paper
2. 3GPP TS 23.501 (PSM and MICO mode specifications)
3. 3GPP TS 36.304 / 38.304 (NB-IoT and NR RRC idle procedures)
4. SimPy documentation / original paper
5. Bianchi (2000) — slotted Aloha analysis
6. de Hoog et al. (2019 or similar) — IoT energy models
7. Any additional references cited in Chapters 1–5

---

## Appendices

### Appendix A — Source Code Structure (~2–3 pages)
- Directory tree of the project repository
- Brief description of each module (`src/node.py`, `src/simulator.py`, `src/power_model.py`, `src/metrics.py`, `src/experiments.py`, `src/visualization.py`, `src/optimization.py`, `src/validation.py`, `src/independence.py`)
- Note on how to run: `python -m pytest tests/` and key notebook entry points

### Appendix B — Selected Code Listings (optional, ~3–5 pages)
- Core simulation loop from `src/simulator.py`
- AnalyticsValidator formula implementations
- IndependenceAnalyzer regression analysis snippet

### Appendix C — Additional Figures (optional)
- Queue evolution time-series traces
- Energy depletion curves for individual nodes
- State occupation pie charts for each scenario
- Any figures referenced in text but not included in Results chapter

### Appendix D — Raw Design Guideline Table (full version)
- Extended version of Table 3.4 with more λ / ts combinations than fits in the main text

---

## Page Count Summary

| Section | Pages |
|---|---|
| Front matter (cover to LOT) | 8 |
| Chapter 1 Introduction | 8 |
| Chapter 2 Methodology | 14 |
| Chapter 3 Results | 17 |
| Chapter 4 Discussion | 7 |
| Chapter 5 Conclusion | 3 |
| References | 3 |
| **Body total** | **~60** |
| Appendices (variable) | 8–12 |

---

## Writing Order Recommendation
1. **Chapter 2 (Methodology)** — write first while the implementation is freshest
2. **Chapter 3 (Results)** — generate all figures from notebooks, write captions and findings
3. **Chapter 1 (Introduction)** — frame the problem around what you ended up building
4. **Chapter 4 (Discussion)** — interpret Chapter 3 results against Chapter 1 objectives
5. **Chapter 5 (Conclusion)** — summarize and look forward
6. **Abstract** — write last, summarizing the whole report in one page
7. **Front matter and appendices** — fill in once body is stable
