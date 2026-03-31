# FYP Presentation Planner — Revised

**Sleep-Based Low-Latency Access for M2M Communications** · 15 min

| Slides | Duration | Objectives | Avg / Slide |
|--------|----------|------------|-------------|
| 11     | 15 min   | 4 (O1–O4)  | ~82 s       |

> **What changed from v1:** Removed the Agenda slide (saves 30s, unnecessary at 15 min).
> Merged Problem + Background into one stronger opening. Split O3 into Pareto (hero) + Duty-Cycling
> (separate win). Slide 08 promoted from 30s cameo to a full 60s standalone.
> Net result: every slide now has enough breathing room to land properly.

---

## Time Distribution

```
0    0:20  2:20  3:20   5:00    7:00  8:00    10:30 11:30   13:00 14:10  15:00
|Title|  Problem &  |System| O1 Sim  | O2 Sweeps |O2 |  O3 Pareto  | O4  |Findings|Q&A|
|     |  Background |Model | Fwk     |  q & ts   |Scn| + DutyCycle |Val  |        |   |
 0:20    2:00    1:00   1:40    2:00     1:00   2:30    1:00   1:30   1:10   0:50
```

---

## Slide-by-Slide Breakdown

### Opening

---

#### Slide 01 — Title `⏱ 0:00 – 0:20`

- "Sleep-Based Low-Latency Access for M2M Communications"
- Your name · Supervisor · Institution · Date
- Visual: IoT node icon with sleep ZZZ ↔ transmit signal arrows

> **Speaker note:** Let it sit. Don't read the title. Open with: *"In IoT networks, every time a sensor wakes up to send a packet, it costs battery life. My project asks: how do we tune that sleep behaviour to hit both goals at once?"*

---

### Background & Motivation

---

#### Slide 02 — The Problem & Background `⏱ 0:20 – 2:20` · 2 min

**Left panel — the tension (motivation):**
- Billions of battery-powered MTDs send small packets sporadically
- Must sleep to survive years on a coin cell
- Sleeping = missed access windows = higher access delay
- Core tension: **minimise delay** ↔ **maximise battery lifetime**

**Right panel — how on-demand sleep works:**
- Slotted Aloha: each active node transmits with probability q per slot
- On-demand sleep: node sleeps after idle timer ts expires, wakes on new arrival
- 4 states: `ACTIVE → IDLE → SLEEP → WAKEUP → ACTIVE`
- 3GPP mapping: MICO mode ≈ on-demand sleep · T3324 ≈ ts · RA-SDT ≈ tw
- Based on Wang et al., 2024 analytical model

> **Speaker note:** Draw the state machine on the right half (4 circles + arrows in Canva).
> On the left, use `slide03_tradeoff_ts_scatter.png` as the motivating image — show it briefly
> to say *"this is what the trade-off actually looks like."*
> Keep moving — this slide sets up everything else. Target: off this slide by 2:20.

---

#### Slide 03 — System Model & Key Parameters `⏱ 2:20 – 3:20` · 1 min

- n nodes (100–10,000), arrival rate λ (Poisson, unsaturated: λ < μ)
- Power states: PT > PB > PI > PW > PS — slot duration 6 ms (3GPP NR)
- Key metrics: mean delay T̄, battery lifetime L̄, success prob p, service rate μ
- Parameters swept: q (0.01–0.5), ts (1–100), n (10–500), λ (0.001–0.1)

> **Speaker note:** Use `slide05_power_profiles.png` as the visual (bar chart on the right).
> A clean parameter table on the left. Say *"everything feeds into four metrics"* — point at them.
> This slide is setup, not results — keep it under 60 seconds.

---

### Results by Objective

---

#### Slide 04 — O1: Simulation Framework `⏱ 3:20 – 5:00` · 1 min 40 s

- Pure Python + SimPy discrete-event simulator
- **Node class:** state machine · packet queue · energy tracker
- **Simulator:** n-node slotted loop · collision detection (success = exactly 1 Tx)
- **BatchSimulator:** 20 replications per config · confidence intervals
- 6 power profiles: NB-IoT, LoRa, LTE-M, 5G NR mMTC, Generic
- Validation: simulated p & μ match analytical formulas within ±5% ✓

> **Speaker note:** Architecture diagram left: `Node → Simulator → BatchSimulator → Results` (draw in Canva, 4 boxes).
> Right panel: use `slide06_timeseries.png` (queue / energy / state stacked area over time).
> Say *"the simulator runs stably at 80,000 slots per replication — enough to deplete batteries
> and measure tail delays."* Don't over-explain the code.

---

#### Slide 05 — O2: Parameter Sweep — q and ts `⏱ 5:00 – 7:00` · 2 min

- **Sweep q (0.01 → 0.35):** higher q → lower delay, but faster battery drain
- **Sweep ts (1 → 100 slots):** larger ts → longer lifetime, higher delay
- Both effects are monotonic and confirmed across 20 replications
- Optimal transmission probability: **q* ≈ 1/n** (confirmed by simulation)
- Bursty traffic amplifies tail delay — 95th/99th percentile grows sharply

> **Speaker note:** Show `slide07_q_sweep.png` and `slide07_ts_sweep.png` side by side.
> Point at the "Lifetime–Delay Trade-off" scatter in each — that's the key sub-plot.
> Say: *"every point on those curves is a design choice — q controls where you sit,
> ts controls which curve you're on."* This motivates the next two slides naturally.

---

#### Slide 06 — O2: Prioritisation Scenarios `⏱ 7:00 – 8:00` · 1 min

| Scenario | ts | q | Result |
|----------|----|---|--------|
| **Low-Latency** | 1 | 2/n | Minimal delay · shorter life |
| **Balanced** | 10 | 1/n | Moderate trade-off |
| **Battery-Life** | 50 | 0.5/n | Years of life · higher delay |

- Quantified gains/losses vs balanced baseline computed for each scenario
- Scatter of 3 labelled points makes the trade-off immediately readable

> **Speaker note:** Use `slide08_scenario_comparison.png` — the bottom-centre scatter
> (Lifetime–Delay Tradeoff panel) is the cleanest sub-chart to crop and enlarge.
> This slide bridges the *"what happens"* of slide 05 to the *"what's optimal"* of slide 07.

---

#### Slide 07 — O3: Pareto Frontier `⏱ 8:00 – 10:30` · 2 min 30 s ⭐ HERO

- Grid search over full (q, ts) parameter space → 2-D lifetime/delay heatmaps
- **Pareto frontier:** each ts value yields one best (max-lifetime, min-delay) point
- Moving along the frontier: increasing ts trades delay for lifetime monotonically
- q* = 1/n is near-optimal across the entire frontier
- Design rule: **ts is the primary knob** — q fine-tunes within a chosen ts

> **Speaker note:** Feature `slide09_pareto_frontier.png` large — full right half of slide.
> Put `slide09_heatmaps.png` as a small inset (bottom-left).
> Walk through the frontier slowly: *"each dot is a different ts value.
> Moving left means you accept more delay to get more battery life.
> The sweet spot for most IoT deployments is right here — ts=10, q=1/n."*
> This is your main result. Give it the full 2.5 minutes.

---

#### Slide 08 — O3: On-Demand Sleep vs Duty-Cycling `⏱ 10:30 – 11:30` · 1 min

- Duty-cycling: node wakes on a fixed periodic schedule regardless of arrivals
- On-demand sleep: only wakes when a packet actually arrives
- Simulation confirms: **on-demand saves 20–40% more energy** at equivalent delay
- On-demand dominates duty-cycling across all tested ts and awake-fraction values

> **Speaker note:** Use `slide09_duty_cycle_comparison.png`.
> Say: *"this isn't just a claim from the paper — we simulated both schemes under
> identical conditions and on-demand wins every time."*
> One concise slide, don't linger — 60 seconds and move on.

---

#### Slide 09 — O4: 3GPP Validation & Design Guidelines `⏱ 11:30 – 13:00` · 1 min 30 s

- 3GPP mapping: MICO mode → on-demand · T3324 → ts · PSM → sleep · RA-SDT → tw
- 4 standard scenarios tested: NB-IoT (2s & 60s T3324) + 5G NR mMTC (2-step & 4-step RA-SDT)
- Simulated p, μ, T̄, L̄ all within **±5%** of Wang et al. analytical formulas
- Convergence: error < 5% at 10⁵ slots, stable by 10⁶ slots
- **Design guideline table:** recommended ts & q* for each traffic load λ

> **Speaker note:** Left panel: use `slide10_formula_validation.png` (analytical vs simulated bars).
> Right panel: paste the design guideline table as a Canva table element
> (generated by `DesignGuidelines.print_guideline_table()` in `o4_validation_demo.ipynb`).
> Say: *"the simulator is not just a black box — it agrees with theory to within 5%,
> which means we can trust the design rules it produces."*

---

### Wrap-up

---

#### Slide 10 — Key Findings `⏱ 13:00 – 14:10` · 1 min 10 s

- ✅ Simulator reproduces Wang et al. analytical results within ±5%
- ✅ On-demand sleep outperforms duty-cycling by 20–40% in energy
- ✅ **q* = 1/n** is a robust, near-optimal design rule
- ✅ **ts is the dominant knob** — small ts for latency, large ts for lifetime
- ✅ All 4 objectives (O1–O4) fully completed
- 🔜 Future: heterogeneous nodes, capture effect, multi-channel

> **Speaker note:** Use big tick icons in Canva for each bullet — make it feel like a checklist.
> End on: *"The bottom line is that operators can now pick a T3324 timer value from
> a table and know exactly what delay and lifetime they'll get."*
> Future work: one bullet only — don't open questions you can't answer.

---

#### Slide 11 — Thank You & Q&A `⏱ 14:10 – 15:00` · 50 s

- Project title + your name
- Leave `slide09_pareto_frontier.png` visible as background
- One memorable sentence (see speaker note)
- GitHub / notebook link · QR code if available

> **Speaker note:** Closing line: *"With a T3324 timer of 10 slots and q = 1/n,
> IoT devices achieve sub-second access delay while lasting over 2 years on a single AA battery —
> and now we have a simulator to prove it."*

---

## Diagram Reference

| Slide | File in `ppt/diagrams/` | Where to place |
|-------|------------------------|----------------|
| 02 | `slide03_tradeoff_ts_scatter.png` | Right panel background / motivating image |
| 03 | `slide05_power_profiles.png` | Right panel |
| 04 | `slide06_timeseries.png` | Right panel |
| 04 | `slide06_state_energy_pies.png` | Optional inset or backup slide |
| 05 | `slide07_q_sweep.png` | Left half |
| 05 | `slide07_ts_sweep.png` | Right half |
| 06 | `slide08_scenario_comparison.png` | Right panel (crop the scatter sub-panel) |
| 07 | `slide09_pareto_frontier.png` | Right panel — **LARGE** |
| 07 | `slide09_heatmaps.png` | Small inset bottom-left |
| 08 | `slide09_duty_cycle_comparison.png` | Right panel |
| 09 | `slide10_formula_validation.png` | Left panel |
| 09 | `slide10_3gpp_scenarios.png` | Right panel |
| 10 | `slide10_publication_summary.png` | Optional backup slide |

> **Backup slide (keep hidden):** `slide10_publication_summary.png` — the 6-panel dashboard.
> Show only if an assessor asks a deep methods question during Q&A.

---

## Timing Safety Rules

| Situation | Action |
|-----------|--------|
| Running 1 min long at slide 06 | Skip the scenario table on slide 06, go straight to slide 07 |
| Running 2 min long at slide 07 | Cut duty-cycling slide (08) — mention it in one sentence |
| Running short with 2 min left | Expand slide 07 — walk through the heatmap in detail |
| Assessor asks about validation | Pull up backup slide (`slide10_publication_summary.png`) |

---

## Canva Design Tips

### Colour Palette

| Role | Hex | Used for |
|------|-----|----------|
| Background | `#0f1117` | Dark theme (or swap to white) |
| Accent 1 | `#6c63ff` | O1 / framework |
| Accent 2 | `#00d4ff` | System model |
| Accent 3 | `#06d6a0` | O2 results |
| Accent 4 | `#ffd166` | O2 parameters |
| Accent 5 | `#ff6b6b` | O4 validation |
| Hero | `#a78bfa` | O3 Pareto slides |

### Slide Layout

- 16:9 widescreen (1920×1080)
- Left panel: title + 3–4 bullets (40%) · Right panel: diagram (60%)
- Section colour strip at top (4 px bar matching accent colour)
- Every content slide needs one diagram — no text-only slides
- Max 4 bullets per slide — use fragments not sentences

### Typography

- Slide title: Montserrat Bold 36–40 pt
- Body bullets: 18–20 pt (never smaller)
- Diagram captions: 11–12 pt, muted grey
- Speaker notes in Canva Presenter view: 12 pt italic

### Canva Shortcuts

- **Presenter view** stores your speaker notes per slide
- Paste PNGs from `ppt/diagrams/` directly into frame elements
- Brand kit: save the 7 accent colours above for one-click reuse
- Export: **PDF** for submission · **PPTX** as backup
