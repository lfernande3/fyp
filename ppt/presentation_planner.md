# FYP Presentation Planner

**Sleep-Based Low-Latency Access for M2M Communications** · 15 min

| Slides | Duration | Objectives | Avg / Slide |
|--------|----------|------------|-------------|
| 12     | 15 min   | 4          | ~75 s       |

---

## Time Distribution

```
0        1        2        3        4        5        6        7        8        9        10       11       12       13       14       15
|  Intro  |         Background          |   Model  |      O1       |         O2          |         O3          |      O4       |   Wrap-up   |
 (0:30)         (3:00)                   (1:30)        (2:00)             (2:30)                (2:30)              (1:30)        (2:00)
```

---

## Slide-by-Slide Breakdown

### Opening

---

#### Slide 01 — Title Slide `⏱ 0:00 – 0:30`

- Project title: "Sleep-Based Low-Latency Access for M2M Communications"
- Your name, supervisor, institution, date
- A single compelling visual — e.g., an IoT node with ZZZ sleep ↔ transmit icons

> **Speaker note:** Let the slide sit for 10 s, then briefly introduce yourself. Don't read the title aloud word-for-word.

---

#### Slide 02 — Agenda `⏱ 0:30 – 1:00`

- Motivation & Problem
- System Model & Methodology
- Simulation Framework (O1)
- Parameter Impact (O2)
- Optimisation & Trade-offs (O3)
- 3GPP Validation (O4)
- Key Findings & Conclusion

> **Speaker note:** Keep it visual — numbered icons or a progress bar. "I'll walk you through these 7 areas in 15 minutes."

---

### Background

---

#### Slide 03 — The Problem: Battery Life vs. Low Latency `⏱ 1:00 – 2:30`

- Massive IoT: billions of MTDs sending small packets sporadically
- Devices must sleep to survive years on a battery
- But sleeping = missed access windows = higher delay
- Core tension: **minimise delay** vs. **maximise lifetime**
- No single analytic formula captures stochastic variability at scale

> **Speaker note:** Use a 2-axis diagram: left axis = battery lifetime (years), bottom axis = delay (slots). Show the trade-off arrow. This is your hook.

---

#### Slide 04 — Background: On-Demand Sleep & Slotted Aloha `⏱ 2:30 – 4:00`

- Slotted Aloha: each active node transmits with probability q per slot
- On-demand sleep: node sleeps after idle timer ts expires, wakes on arrival
- 4 states: ACTIVE → IDLE → SLEEP → WAKEUP → ACTIVE
- 3GPP mapping: MICO mode ≈ on-demand sleep, T3324 ≈ ts, RA-SDT ≈ tw
- Based on Wang et al., 2024 analytical model

> **Speaker note:** Show a state-machine diagram (circle nodes with arrows). Mention you're extending the paper with a simulator.

---

#### Slide 05 — System Model & Key Parameters `⏱ 4:00 – 5:30`

- n nodes (100–10,000), each with energy budget E
- Poisson arrivals (rate λ), unsaturated regime (λ < μ)
- Power hierarchy: PT > PB > PI > PW > PS (3GPP NR values)
- Metrics: mean delay T̄, battery lifetime L̄, throughput, success prob p
- Slot duration 6 ms → lifetime in realistic years

> **Speaker note:** A clean parameter table + power state bar chart works great here. Keep it quick — this is setup not results.

---

### Results by Objective

---

#### Slide 06 — Simulation Framework (O1) `⏱ 5:30 – 7:30`

- Pure Python + SimPy discrete-event simulator
- Node class: state machine, packet queue, energy tracker
- Simulator class: n-node loop, collision detection (success if exactly 1 Tx)
- BatchSimulator: 20–50 replications per config, confidence intervals
- 6 power profiles: NB-IoT, LoRa, LTE-M, 5G NR mMTC, Generic
- Validated: simulated p & μ match analytical formulas ✓

> **Speaker note:** Show a simple architecture diagram: Node → Simulator → BatchSimulator → Results. One code snippet optional but keep it brief (3 lines max).

---

#### Slide 07 — Parameter Impact: Effect of q and ts (O2) `⏱ 7:30 – 9:00`

- Sweep q (0.01–0.5): higher q → lower delay but faster battery drain
- Sweep ts (1–100 slots): larger ts → longer lifetime but higher delay
- Traffic models: Poisson (default) + bursty (batch arrivals)
- Bursty traffic amplifies tail delay — 95th/99th percentile grows sharply
- Optimal q ≈ 1/n rule confirmed by simulation

> **Speaker note:** Use actual simulation plots here — lifetime vs q and delay vs ts curves. Two plots side by side. This is where simulation adds value over analytics.

---

#### Slide 08 — Low-Latency vs. Battery-Life Scenarios `⏱ 9:00 – 9:30`

- Low-Latency priority: ts=1, q=2/n → minimal delay, shorter life
- Balanced: ts=10, q=1/n → moderate trade-off
- Battery-Life priority: ts=50, q=0.5/n → years of life, higher delay
- Scatter plot: each ts value = one Pareto point (delay, lifetime)

> **Speaker note:** The scatter plot of 3 labelled points is very clean for Canva — colour-code them (red=latency, green=balanced, blue=battery).

---

#### Slide 09 — Optimisation & Pareto Trade-off (O3) `⏱ 9:30 – 11:30`

- Grid search over (q, ts) space → lifetime/delay heatmaps
- Pareto frontier: set of (max lifetime, min delay) non-dominated points
- On-demand sleep dominates duty-cycling across all traffic loads
- DutyCycleSimulator confirms: on-demand saves 20–40% more energy
- Design rule: q* = 1/n is near-optimal for most λ values

> **Speaker note:** The Pareto frontier curve is your most impressive result — feature it large. Heatmap as a secondary inset is a nice touch.

---

#### Slide 10 — 3GPP Validation & Design Guidelines (O4) `⏱ 11:30 – 13:00`

- MICO mode → on-demand sleep; T3324 timer → ts; RA-SDT → wake-up tw
- 4 standard 3GPP scenarios: NB-IoT + NR mMTC (2-step & 4-step RA-SDT)
- Simulated p, μ, T̄, L̄ validated within ±5% of analytical formulas
- Convergence: error <5% at 10⁵ slots, stabilises by 10⁶ slots
- Guideline table: recommended ts & q* for each traffic load λ

> **Speaker note:** Show one clean analytical-vs-simulated comparison chart. The guideline table is great as a visual callout box at the bottom.

---

### Wrap-up

---

#### Slide 11 — Key Findings & Conclusion `⏱ 13:00 – 14:30`

- ✅ Simulator faithfully reproduces analytical results from Wang et al.
- ✅ On-demand sleep outperforms duty-cycling by up to 40% in energy
- ✅ q* = 1/n is a robust design rule across traffic loads
- ✅ T3324 (ts) is the dominant knob: small ts → low latency, large ts → long life
- ✅ All 4 objectives (O1–O4) fully complete
- Future: heterogeneous nodes, capture effect, multi-channel extension

> **Speaker note:** Use big checkmarks/icons per point. Keep future work to 1 bullet — don't open new questions. End with your "so what": this gives operators a concrete tuning guide.

---

#### Slide 12 — Thank You & Q&A `⏱ 14:30 – 15:00`

- Project title + your name
- GitHub repo / notebook link (QR code if you have it)
- One memorable takeaway sentence
- Leave the Pareto frontier plot visible in background

> **Speaker note:** "The simulator shows that with a T3324 timer of just 10 slots and q = 1/n, IoT devices can achieve sub-second latency while lasting over 2 years on a single AA battery."

---

## Canva Design Tips

### Colour Palette

| Role | Hex | Used for |
|------|-----|----------|
| Background | `#0f1117` | Dark theme (or swap to white) |
| Accent 1 | `#6c63ff` | O1 / framework slides |
| Accent 2 | `#00d4ff` | System model |
| Accent 3 | `#06d6a0` | Results / O2 |
| Accent 4 | `#ffd166` | Parameters / O2 |
| Accent 5 | `#ff6b6b` | Validation / O4 |

### Slide Layout

- Use 16:9 widescreen (1920×1080)
- Left panel: title + 3–4 bullets (40%)
- Right panel: plot / diagram (60%)
- Every content slide needs a visual — no text-only slides
- Use consistent section colour strips at the top

### Typography

- Title: 36–40 pt, bold (e.g. Montserrat Bold)
- Body bullets: 18–20 pt (no smaller)
- Speaker cue text: 12 pt, light grey, italic
- Max 4–5 bullets per slide
- Avoid full sentences — use fragments

### Key Visuals to Include

| Slide | Visual |
|-------|--------|
| 04 | State machine (4 circles + arrows) |
| 07 | Lifetime vs q & Delay vs ts line charts (side by side) |
| 08 | 3-point scatter — Pareto scenarios |
| 09 | Pareto frontier curve **(hero chart — make it large)** |
| 10 | Analytical vs simulated comparison bar |

### Canva Shortcuts

- Use **Presenter view** to store speaker notes per slide
- Smart mockup: paste notebook plots directly as PNG into frames
- Use grid/frame elements for consistent plot placeholders
- Brand kit: save your 5 accent colours for one-click reuse
- Export as **PDF** for submission + **PPTX** as backup

### Timing Guidance

- Practice aloud — 15 min is tight
- If running long: cut Slide 08 (scenarios detail)
- If running short: expand Slide 09 Pareto results
- Leave 2 min for Q&A buffer if your slot allows
- Have 1 backup slide ready: the guideline table from O4
