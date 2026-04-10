# 10-Minute Presentation Speech

## Sleep-Based Low-Latency Access for Machine-to-Machine Communications

### Suggested delivery time

About 9 to 10 minutes at a steady pace.

---

## Full Speech

Good afternoon, and thank you for giving me the opportunity to present my final year project.

My project is titled **Sleep-Based Low-Latency Access for Machine-to-Machine Communications**.

The problem I address is a very practical one in IoT and machine-to-machine systems. Many devices are battery-powered and are expected to last for years, so they need to sleep aggressively to save energy. However, once a device sleeps, any packet that arrives during that period must wait before it can be transmitted. That means the system is always balancing two competing goals: **low delay** and **long battery lifetime**.

This trade-off is the central theme of my report. More specifically, I study how two main design parameters, the transmission probability `q` and the idle timer `t_s`, shape that trade-off, and whether these two parameters can be tuned independently or whether they must be tuned jointly.

If I refer to the report structure, the work begins with the system model and methodology in Chapter 2, then presents the main results in Chapter 3, followed by extension studies in Chapter 4, and finally the interpretation and conclusions in Chapters 5 and 6.

I will summarize the presentation in five parts:

1. the motivation and system model,
2. the simulation framework,
3. the main validated results,
4. the key design insight,
5. and the overall contribution of the project.

Starting with the system model, shown in Chapter 2, the baseline system is a slotted M2M uplink with a population of homogeneous machine-type devices competing for channel access.

Each node can be in one of four states: **active, idle, sleep, and wake-up**. This state flow is illustrated in Figure 2.1. A node with a packet attempts transmission with probability `q`. If its queue becomes empty, it waits for an idle timer `t_s`. If no new packet arrives before that timer expires, the node goes to sleep. When a new packet arrives during sleep, the node cannot transmit immediately. It must first pay a wake-up delay of `t_w` slots before it can return to active contention.

This is a simple model, but it captures the engineering tension very clearly. A longer timer can reduce unnecessary sleeping and help responsiveness, but it also keeps the node in a higher-power state for longer. A more aggressive access probability can reduce waiting time, but it can also increase collisions and energy use.

The analytical baseline in Section 2.2 gives the success probability and service-rate expressions used for validation. In particular, it provides a closed-form reference for success probability `p`, service rate `mu`, and the stability condition `lambda < mu`. That analytical reference is important because the simulator is not intended to replace theory. Instead, the simulator is built to reproduce the theory faithfully and then go beyond it where the analytical model becomes too restrictive.

The simulation framework itself is one of the main contributions of the project. As described in Section 2.3 and Figure 2.2, I developed a modular Python-based discrete-event simulation framework. The architecture is organized around a `Node` component, a `Simulator`, and a `BatchSimulator`, with supporting modules for power modeling, metrics, validation, optimization, traffic generation, and later extensions.

This structure matters for two reasons. First, it makes the software easier to test, debug, and extend. Second, it allows the same core simulator to support not only the baseline study but also optimization, 3GPP-inspired interpretation, and additional extension experiments such as retry limits, CSMA comparison, advanced receiver models, Age of Information, and MMBP arrivals.

Once the simulator was built, the first major question was whether it was credible. That is why Chapter 3 begins with validation. The validation figures compare empirical and analytical success probability and service rate, and the results show that the simulator tracks the intended baseline well in the stable regime. This is important because without that validation, the later optimization and interpretation would be much weaker.

After validation, the report moves to parameter impact in Section 3.2. This is where the main design trade-off becomes visually clear.

Figures 3.4 and 3.5 show how delay and lifetime change when the transmission probability `q` is swept for several timer settings. The main takeaway is that delay is not simply monotonic in `q`. At very small `q`, packets wait too long for access opportunities. At very large `q`, contention becomes too aggressive and collisions dominate. So delay has a best region at intermediate `q`.

Lifetime behaves differently. As shown in Figure 3.5, lifetime generally falls as `q` increases, because more aggressive access leads to more active behavior and more energy consumption. The updated lifetime figures also make the timer effect easier to see. Shorter timers generally preserve lifetime better, while very long timers keep the device in higher-power idle operation for too long before sleep is entered.

Figures 3.6 and 3.7 then reverse the viewpoint by plotting the metrics against `t_s` for fixed `q`. These figures reinforce a consistent message: **`q` is the stronger control knob, while `t_s` provides a secondary but still meaningful adjustment**, especially for lifetime.

Figure 3.8 adds the load dimension by sweeping `lambda`. This figure is important because it shows the throughput curves rising with offered load and then saturating once the channel becomes contention-limited. In the current version of the report, this figure uses a restored wide `lambda` sweep and a `q = 1/n` rule for each node population. That makes the figure a clearer illustration of the stable operating region and the onset of channel saturation.

The next major step is optimization, in Section 3.3. Here the project asks not just how the parameters affect performance, but how a designer should choose them.

Figures 3.9 and 3.10 show delay and lifetime over the full `(q, t_s)` plane. These two heatmaps show very clearly that the delay-preferred region and the lifetime-preferred region do not coincide. In other words, there is no single universal best configuration.

Figure 3.11 summarizes this conflict in a delay-lifetime trade-off view. The important message is not that one setting dominates everything else, but that different objectives pull the design in different directions. If the application prioritizes responsiveness, the preferred region is different from the one preferred by a battery-life-oriented application.

Figure 3.12 makes this easier to interpret by comparing three named scenarios: low-latency, balanced, and battery-life. This is useful from an engineering perspective because real deployment choices are usually made in terms of priorities, not just in terms of raw parameter values.

Figure 3.13 then compares on-demand sleep against duty-cycling. The report shows that on-demand sleep remains consistently competitive and usually achieves lower delay while maintaining similar lifetime. This matters because it supports the baseline design choice of triggering sleep behavior from actual inactivity rather than from a rigid periodic schedule.

Section 3.4 then interprets the same ideas in a 3GPP-inspired frame. Rather than claiming a full standards implementation, the report uses practical analogies such as `T3324`-style timers, NB-IoT-like and NR mMTC-like power profiles, and simple access heuristics to translate the abstract parameter study into a more recognizable engineering vocabulary.

Figures 3.14 and 3.15 show how different timer settings behave under varying load. Figure 3.16 tests whether the practical `1/n` rule still works well as the number of nodes changes. Figure 3.17 then gives a compact delay-lifetime comparison for representative NB-IoT-like and NR mMTC-like settings. Together, these figures move the report from pure analysis toward design guidance.

However, I would say the strongest conceptual contribution of the project appears in Section 3.5, the independence analysis.

The key question here is: can `q` and `t_s` be tuned separately?

At first glance, one might think yes, because `q` mainly affects contention while `t_s` mainly affects sleep behavior. But the report shows that this is not generally true.

Figure 3.18 provides the first visual evidence. The delay curves are not parallel when stratified by `q` and `t_s`, which means the effect of one parameter depends on the value of the other. Figure 3.19 then gives the statistical companion to that visual result by plotting additive-model delay residuals against the coupling score `kappa = p * t_s` after filtering to stable operating points.

The broader conclusion, supported by the regression summary and the surrounding analysis, is that `q` and `t_s` should **not** generally be treated as independent design knobs. They interact in the service process, which means sequential one-parameter tuning can miss better joint operating points.

This is, in my view, the most important design result of the project.

The later extension studies in Chapter 4 broaden the framework by adding finite retries, CSMA comparison, receiver models, Age of Information, and MMBP arrivals. I would present these as evidence that the simulator is not a one-purpose prototype. It is a reusable experimental framework. Even when these richer assumptions are introduced, the central message remains consistent: sleep behavior and access behavior must still be reasoned about together.

To conclude, the main contribution of this project is not only that I built a simulator, but that I built a **validated and extensible simulation framework** and used it to derive structured design insight.

The three most important conclusions are:

1. there is a real and measurable trade-off between delay and lifetime in sleep-aware M2M access,
2. the heuristic `q ~= 1/n` is a useful starting point for balanced operation,
3. and most importantly, `q` and `t_s` should generally be tuned jointly rather than independently.

So the project contributes both a software artifact and an engineering conclusion. The software framework is modular, testable, and reusable. The engineering conclusion is that low-power random access design must be approached as a co-optimization problem rather than a sequence of isolated parameter choices.

Thank you for listening. I would be happy to answer any questions.

---

## Quick Delivery Notes

- Spend about 1 minute on motivation and system model.
- Spend about 1.5 minutes on methodology and validation.
- Spend about 4 minutes on the Chapter 3 figures and key findings.
- Spend about 1 minute on extensions.
- Spend about 1 minute on conclusions and questions.

## If You Need To Shorten On The Spot

- Shorten the architecture explanation in Chapter 2.
- Mention the Chapter 4 extensions in one sentence instead of listing all of them.
- Keep the strongest emphasis on Figures `3.5`, `3.8`, `3.11`, `3.13`, and `3.18`-`3.19`.
