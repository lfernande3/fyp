from __future__ import annotations

from dataclasses import dataclass

from .power_model import PowerModel, PowerProfile
from .simulator import SimulationConfig


SLOT_DURATION_MS = 6.0


def seconds_to_slots(seconds: float, slot_duration_ms: float = SLOT_DURATION_MS) -> int:
    """Convert a timer in seconds to simulator slots."""
    return max(1, int(round(seconds * 1000.0 / slot_duration_ms)))


def slots_to_seconds(slots: int, slot_duration_ms: float = SLOT_DURATION_MS) -> float:
    """Convert simulator slots to seconds."""
    return slots * slot_duration_ms / 1000.0


def battery_energy_mwh(battery_type: str) -> float:
    """Return the energy budget of a named battery preset."""
    return PowerModel.create_battery_config(battery_type).get_energy_mwh()


def q_one_over_n(n_nodes: int) -> float:
    """Throughput-friendly Aloha access rule."""
    return 1.0 / max(n_nodes, 1)


@dataclass(frozen=True)
class BaselineScenario:
    """Shared baseline assumptions used by figures and experiment helpers."""

    name: str
    n_nodes: int
    arrival_rate: float
    idle_timer_s: float
    wakeup_time: int
    power_profile: PowerProfile
    battery_type: str
    transmission_prob: float | None = None

    @property
    def idle_timer_slots(self) -> int:
        return seconds_to_slots(self.idle_timer_s)

    @property
    def initial_energy_mwh(self) -> float:
        return battery_energy_mwh(self.battery_type)

    def resolve_q(self, n_nodes: int | None = None) -> float:
        if self.transmission_prob is not None:
            return self.transmission_prob
        return q_one_over_n(n_nodes or self.n_nodes)

    def build_config(
        self,
        *,
        n_nodes: int | None = None,
        arrival_rate: float | None = None,
        transmission_prob: float | None = None,
        idle_timer: int | None = None,
        wakeup_time: int | None = None,
        initial_energy: float | None = None,
        power_profile: PowerProfile | None = None,
        max_slots: int = 50_000,
        seed: int | None = None,
        **extra: object,
    ) -> SimulationConfig:
        final_n = n_nodes or self.n_nodes
        final_profile = power_profile or self.power_profile
        return SimulationConfig(
            n_nodes=final_n,
            arrival_rate=self.arrival_rate if arrival_rate is None else arrival_rate,
            transmission_prob=(
                self.resolve_q(final_n)
                if transmission_prob is None
                else transmission_prob
            ),
            idle_timer=self.idle_timer_slots if idle_timer is None else idle_timer,
            wakeup_time=self.wakeup_time if wakeup_time is None else wakeup_time,
            initial_energy=(
                self.initial_energy_mwh if initial_energy is None else initial_energy
            ),
            power_rates=PowerModel.get_profile(final_profile),
            max_slots=max_slots,
            seed=seed,
            **extra,
        )


GENERIC_LITERATURE_BASELINE = BaselineScenario(
    name="generic_literature",
    n_nodes=100,
    arrival_rate=5e-5,
    idle_timer_s=10.0,
    wakeup_time=2,
    power_profile=PowerProfile.GENERIC_LOW,
    battery_type="AA",
)


NB_IOT_BASELINE = BaselineScenario(
    name="nb_iot_interpretation",
    n_nodes=100,
    arrival_rate=1e-5,
    idle_timer_s=10.0,
    wakeup_time=2,
    power_profile=PowerProfile.NB_IOT,
    battery_type="AA",
)


NR_MMTC_BASELINE = BaselineScenario(
    name="nr_mmtc_interpretation",
    n_nodes=100,
    arrival_rate=1e-5,
    idle_timer_s=10.0,
    wakeup_time=2,
    power_profile=PowerProfile.NR_MMTC,
    battery_type="coin_cell",
)
