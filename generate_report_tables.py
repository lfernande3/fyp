from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

from generate_figures import (
    GENERIC_LITERATURE_BASELINE,
    base_config,
    lifetime_vis_config,
    run_batch,
)
from src.baselines import q_one_over_n, seconds_to_slots, slots_to_seconds
from src.independence import IndependenceAnalyzer
from src.metrics import MetricsCalculator
from src.mmbp_analytics import run_o10_experiments
from src.power_model import PowerModel, PowerProfile
from src.simulator import BatchSimulator, SimulationConfig
from src.validation import AnalyticalValidator
from src.validation_3gpp import DesignGuidelines


def fmt(value: object, decimals: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if value == float("inf"):
        return "inf"
    return f"{float(value):.{decimals}f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def table_2_1() -> str:
    rows = [
        [
            "O1 Validation",
            "Validate baseline simulator against analytical `p`, `mu`, and convergence",
            "`n = 5-100`, `lambda = 5e-5`, `q = min(0.05, 1/n)`, `t_s = 10 s`, `t_w = 2`",
            "3 quick / 20 full",
            "validation scatter, convergence error",
        ],
        [
            "O2 Parameter impact",
            "Measure delay, lifetime, and throughput sensitivity",
            "`q = 0.002-0.05`, `t_s = 0.5-120 s`, `lambda = 0.001-0.05`, `n = 10-100`",
            "2 quick / 20 full",
            "delay curves, lifetime curves, throughput saturation",
        ],
        [
            "O3 Optimization",
            "Map the `(q, t_s)` trade-off surface and named scenarios",
            "`q = 0.002-0.05`, `t_s = 0.5-120 s`, sparse-load baseline",
            "2 quick / 15 full",
            "heatmaps, trade-off points, scenario deltas",
        ],
        [
            "O4 3GPP-inspired",
            "Interpret timers and access rules using `T3324`-style settings",
            "`T3324 = 2, 10, 60, 360 s`, profile-specific power models, `q = 1/n`",
            "2 quick / 15 full",
            "delay-lifetime curves, `q*` trend, guideline table",
        ],
        [
            "O5 Independence",
            "Test whether `q` and `t_s` interact",
            "factorial sweep over `q` and `t_s`, regression in log-log space",
            "2 quick / 15 full",
            "interaction plots, residuals, coupling maps",
        ],
        [
            "O6-O10 Extensions",
            "Probe retry limits, CSMA, receiver models, AoI, and MMBP arrivals",
            "objective-specific sweeps over `K`, `n`, `lambda`, receiver type, AoI settings, and BI",
            "4-5 quick / 20 full",
            "extension figures and comparative summaries",
        ],
    ]
    return md_table(
        ["Objective block", "Purpose", "Main settings / ranges", "Replications", "Primary outputs"],
        rows,
    )


def table_3_1() -> str:
    lam = GENERIC_LITERATURE_BASELINE.arrival_rate
    q_base = 0.05
    ts = GENERIC_LITERATURE_BASELINE.idle_timer_slots
    tw = GENERIC_LITERATURE_BASELINE.wakeup_time
    rows: list[list[object]] = []
    for n in [5, 10, 15, 20, 30, 50, 75, 100]:
        q = min(q_base, 1.0 / n)
        cfg = base_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=20_000)
        reps = run_batch(cfg, 3)
        validations = [AnalyticalValidator.validate_results(cfg, r, tolerance=0.2) for r in reps]
        emp_p = float(np.mean([v["success_probability"]["empirical"] for v in validations]))
        ana_p = float(np.mean([v["success_probability"]["analytical"] for v in validations]))
        emp_mu = float(np.mean([v["service_rate"]["empirical"] for v in validations]))
        ana_mu = float(np.mean([v["service_rate"]["analytical"] for v in validations]))
        p_err = abs(emp_p - ana_p) / max(ana_p, 1e-12) * 100
        mu_err = abs(emp_mu - ana_mu) / max(ana_mu, 1e-12) * 100
        rows.append(
            [
                n,
                fmt(q, 4),
                fmt(emp_p, 4),
                fmt(ana_p, 4),
                fmt(p_err, 1),
                fmt(emp_mu, 5),
                fmt(ana_mu, 5),
                fmt(mu_err, 1),
                "Yes" if lam < ana_mu else "No",
            ]
        )
    return md_table(
        ["n", "q", "Empirical p", "Analytical p", "p error (%)", "Empirical mu", "Analytical mu", "mu error (%)", "Stable"],
        rows,
    )


def scenario_results() -> tuple[str, str]:
    n = GENERIC_LITERATURE_BASELINE.n_nodes
    lam = GENERIC_LITERATURE_BASELINE.arrival_rate
    tw = GENERIC_LITERATURE_BASELINE.wakeup_time
    scenarios = {
        "Low-Latency": (2.0 / n, seconds_to_slots(0.5)),
        "Balanced": (1.0 / n, seconds_to_slots(5.0)),
        "Battery-Life": (0.5 / n, seconds_to_slots(10.0)),
    }

    state_rows: list[list[object]] = []
    summary: dict[str, dict[str, float]] = {}
    for name, (q, ts) in scenarios.items():
        cfg = lifetime_vis_config(n=n, lam=lam, q=q, ts=ts, tw=tw, slots=20_000)
        reps = run_batch(cfg, 3)
        delay = float(np.mean([r.mean_delay for r in reps]))
        finite_lt = [r.mean_lifetime_years for r in reps if r.mean_lifetime_years != float("inf")]
        lifetime = float(np.mean(finite_lt)) if finite_lt else float("inf")
        throughput = float(np.mean([r.throughput for r in reps]))
        fractions = {
            key: float(np.mean([r.state_fractions.get(key, 0.0) for r in reps]))
            for key in ["active", "idle", "sleep", "wakeup"]
        }
        summary[name] = {"delay": delay, "lifetime": lifetime}
        state_rows.append(
            [
                name,
                fmt(q, 4),
                fmt(slots_to_seconds(ts), 1),
                fmt(delay, 1),
                fmt(lifetime, 3),
                fmt(throughput, 4),
                fmt(fractions["active"], 3),
                fmt(fractions["idle"], 3),
                fmt(fractions["sleep"], 3),
                fmt(fractions["wakeup"], 3),
            ]
        )

    base_delay = summary["Balanced"]["delay"]
    base_lifetime = summary["Balanced"]["lifetime"]
    scenario_rows: list[list[object]] = []
    for name in ["Low-Latency", "Balanced", "Battery-Life"]:
        delay = summary[name]["delay"]
        lifetime = summary[name]["lifetime"]
        delay_pct = (delay - base_delay) / base_delay * 100 if base_delay else 0.0
        lifetime_pct = (
            (lifetime - base_lifetime) / base_lifetime * 100
            if base_lifetime not in (0.0, float("inf")) and lifetime != float("inf")
            else 0.0
        )
        scenario_rows.append([name, fmt(delay, 1), fmt(lifetime, 3), fmt(delay_pct, 1), fmt(lifetime_pct, 1)])

    return (
        md_table(
            ["Scenario", "q", "ts (s)", "Delay (slots)", "Lifetime (years)", "Throughput", "Active", "Idle", "Sleep", "Wake-up"],
            state_rows,
        ),
        md_table(
            ["Scenario", "Delay (slots)", "Lifetime (years)", "Delay vs Balanced (%)", "Lifetime vs Balanced (%)"],
            scenario_rows,
        ),
    )


def table_3_4() -> str:
    entries = DesignGuidelines.generate_guideline_table(
        lambda_values=[1e-6, 1e-5, 1e-4],
        ts_values=[seconds_to_slots(v) for v in [2.0, 10.0, 60.0, 360.0]],
        n=20,
        tw=GENERIC_LITERATURE_BASELINE.wakeup_time,
        delay_target_ms=1000.0,
        max_slots=15_000,
        n_replications=2,
        verbose=False,
    )
    rows = []
    for entry in entries:
        rows.append(
            [
                f"{entry.lambda_rate:.0e}",
                fmt(entry.t3324_s, 0),
                fmt(entry.recommended_q, 4),
                fmt(entry.mean_delay_ms, 0),
                fmt(entry.mean_lifetime_years, 3),
                "Yes" if entry.stability_margin > 0 else "No",
                "Yes" if entry.meets_delay_target else "No",
            ]
        )
    return md_table(
        ["lambda", "T3324 (s)", "q*", "Delay (ms)", "Lifetime (years)", "Stable", "Meets 1 s target"],
        rows,
    )


def table_3_5() -> str:
    base = base_config(
        n=20,
        lam=GENERIC_LITERATURE_BASELINE.arrival_rate,
        q=0.01,
        ts=seconds_to_slots(5.0),
        tw=GENERIC_LITERATURE_BASELINE.wakeup_time,
        slots=8_000,
    )
    sweep = IndependenceAnalyzer.run_factorial_sweep(
        base,
        q_values=[0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
        ts_values=[seconds_to_slots(v) for v in [0.5, 2.0, 10.0, 30.0]],
        n_replications=1,
        verbose=False,
    )
    regression = IndependenceAnalyzer.run_regression_analysis(sweep["df"])
    rows = []
    for metric in ["delay", "lifetime"]:
        result = regression[metric]
        rows.append(
            [
                metric.capitalize(),
                result["n_observations"],
                fmt(result["R2_additive"], 4),
                fmt(result["R2_interaction"], 4),
                fmt(result["R2_improvement"], 4),
                fmt(result["F_statistic"], 2),
                f"{result['p_value']:.2e}",
            ]
        )
    return md_table(
        ["Metric", "n_obs", "R2 additive", "R2 interaction", "Delta R2", "F-statistic", "p-value"],
        rows,
    )


def table_3_6() -> str:
    rows = []
    arrival_rate = GENERIC_LITERATURE_BASELINE.arrival_rate
    q = q_one_over_n(100)
    p = q * (1.0 - q) ** 99
    power_rates = PowerModel.get_profile(PowerProfile.GENERIC_LOW)
    for k in [2, 3, 5, 8, 10, 15, None]:
        cfg = SimulationConfig(
            n_nodes=100,
            arrival_rate=arrival_rate,
            transmission_prob=q,
            idle_timer=GENERIC_LITERATURE_BASELINE.idle_timer_slots,
            wakeup_time=GENERIC_LITERATURE_BASELINE.wakeup_time,
            initial_energy=GENERIC_LITERATURE_BASELINE.initial_energy_mwh,
            power_rates=power_rates,
            max_slots=8_000,
            max_retries=k,
        )
        results = BatchSimulator(cfg).run_replications(n_replications=2)
        empirical_mu = float(np.mean([r.empirical_service_rate for r in results]))
        mean_delay = float(np.mean([r.mean_delay for r in results]))
        drop_rate = float(np.mean([r.mean_drop_rate for r in results]))
        if k is None:
            analytical_mu = MetricsCalculator.compute_analytical_service_rate(
                p,
                arrival_rate,
                GENERIC_LITERATURE_BASELINE.wakeup_time,
            )
            label = "inf"
        else:
            analytical_mu = MetricsCalculator.compute_mu_finite_k(
                p,
                GENERIC_LITERATURE_BASELINE.idle_timer_slots,
                GENERIC_LITERATURE_BASELINE.wakeup_time,
                k,
            )
            label = str(k)
        error_pct = abs(empirical_mu - analytical_mu) / max(analytical_mu, 1e-12) * 100
        rows.append([label, fmt(empirical_mu, 5), fmt(analytical_mu, 5), fmt(error_pct, 1), fmt(mean_delay, 1), fmt(drop_rate, 4)])
    return md_table(
        ["K", "Empirical mu", "Analytical mu_K", "Error (%)", "Delay (slots)", "Drop rate"],
        rows,
    )


def table_3_7() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_o10_experiments(
            n_nodes=100,
            ts=10,
            tw=2,
            bi_values=[1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0],
            n_replications=2,
            max_slots=8_000,
            quick_mode=False,
        )
    rows = []
    for bi, empirical, mu_mmbp, err_mmbp, mu_bern, err_bern in zip(
        result["bi_values"],
        result["mu_empirical"],
        result["mu_mmbp"],
        result["mu_error_mmbp"],
        result["mu_bernoulli"],
        result["mu_error_bernoulli"],
    ):
        rows.append(
            [
                fmt(bi, 1),
                fmt(empirical, 5),
                fmt(mu_mmbp, 5),
                fmt(err_mmbp, 1),
                fmt(mu_bern, 5),
                fmt(err_bern, 1),
            ]
        )
    return md_table(
        ["BI", "Empirical mu", "mu_MMBP", "MMBP error (%)", "mu_Bernoulli", "Bernoulli error (%)"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown tables for the report.")
    parser.add_argument("--table", help="Optional table key, e.g. TABLE_3_1")
    args = parser.parse_args()

    if args.table:
        if args.table == "TABLE_2_1":
            print(table_2_1())
        elif args.table == "TABLE_3_1":
            print(table_3_1())
        elif args.table == "TABLE_3_2":
            print(scenario_results()[0])
        elif args.table == "TABLE_3_3":
            print(scenario_results()[1])
        elif args.table == "TABLE_3_4":
            print(table_3_4())
        elif args.table == "TABLE_3_5":
            print(table_3_5())
        elif args.table == "TABLE_3_6":
            print(table_3_6())
        elif args.table == "TABLE_3_7":
            print(table_3_7())
        else:
            raise KeyError(args.table)
        return

    t32, t33 = scenario_results()
    outputs = {
        "TABLE_2_1": table_2_1(),
        "TABLE_3_1": table_3_1(),
        "TABLE_3_2": t32,
        "TABLE_3_3": t33,
        "TABLE_3_4": table_3_4(),
        "TABLE_3_5": table_3_5(),
        "TABLE_3_6": table_3_6(),
        "TABLE_3_7": table_3_7(),
    }

    for key, value in outputs.items():
        print(f"<<<{key}>>>")
        print(value)
        print()


if __name__ == "__main__":
    main()
