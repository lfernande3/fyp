"""
Power Model for M2M Sleep-Based Random Access Simulator

This module provides realistic power consumption models based on 3GPP specifications
for Machine-Type Device (MTD) communications, particularly for NR (5G New Radio) and
legacy LTE-M/NB-IoT systems.

References:
- 3GPP TS 38.213: NR Physical layer procedures for control
- 3GPP TS 36.321: E-UTRA MAC protocol specification
- 3GPP TS 24.501: Non-Access-Stratum (NAS) protocol (MICO mode, T3324 timer)

Date: February 10, 2026
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PowerProfile(Enum):
    """Predefined power consumption profiles."""
    LORA = "lora"                    # LoRa-like low power
    NB_IOT = "nb_iot"               # NB-IoT (3GPP Release 13+)
    LTE_M = "lte_m"                 # LTE-M (eMTC)
    NR_MMTC = "nr_mmtc"            # 5G NR mMTC
    GENERIC_LOW = "generic_low"     # Generic low-power device
    GENERIC_HIGH = "generic_high"   # Generic high-power device


@dataclass
class BatteryConfig:
    """Battery configuration for lifetime estimation."""
    capacity_mah: float              # Battery capacity in mAh
    voltage_v: float                 # Operating voltage in Volts
    initial_charge_fraction: float   # Initial charge (0.0-1.0)
    
    def get_energy_joules(self) -> float:
        """Calculate total energy in Joules."""
        # Energy (J) = Capacity (Ah) * Voltage (V) * 3600 (s/h)
        capacity_ah = self.capacity_mah / 1000.0
        return capacity_ah * self.voltage_v * 3600.0 * self.initial_charge_fraction
    
    def get_energy_mwh(self) -> float:
        """Calculate total energy in mWh (milliwatt-hours)."""
        return self.capacity_mah * self.voltage_v * self.initial_charge_fraction
    
    def get_energy_units(self, slot_duration_ms: float = 6.0) -> float:
        """
        Calculate energy in simulation units.
        
        Simulation unit = energy consumed per slot.
        This converts battery capacity to equivalent simulation units.
        
        Args:
            slot_duration_ms: Slot duration in milliseconds (default 6ms for NR)
            
        Returns:
            Energy in simulation units
        """
        # Energy in mWh
        energy_mwh = self.get_energy_mwh()
        
        # Convert to mWs (milliwatt-seconds)
        energy_mws = energy_mwh * 3600.0
        
        # Slot duration in seconds
        slot_duration_s = slot_duration_ms / 1000.0
        
        # Energy units (assuming 1 unit = 1 mW per slot)
        return energy_mws / slot_duration_s


class PowerModel:
    """
    Power consumption model for MTD nodes.
    
    Defines power consumption rates for different states based on 3GPP specifications
    and empirical measurements from literature.
    """
    
    # Predefined power profiles (in mW - milliwatts)
    PROFILES = {
        PowerProfile.LORA: {
            'PT': 120.0,    # Transmit: ~120mW (typical LoRa TX at +14dBm)
            'PB': 15.0,     # Busy/Listen: ~15mW
            'PI': 1.5,      # Idle: ~1.5mW
            'PW': 10.0,     # Wakeup: ~10mW
            'PS': 0.001,    # Deep sleep: ~1μW
            'name': 'LoRa-like',
            'description': 'Low-power long-range, typical for LoRaWAN devices'
        },
        
        PowerProfile.NB_IOT: {
            'PT': 220.0,    # Transmit: ~220mW (3GPP TS 36.101, +23dBm)
            'PB': 80.0,     # Busy/Listen: ~80mW (receiving)
            'PI': 3.0,      # Idle: ~3mW (RRC_IDLE with paging)
            'PW': 50.0,     # Wakeup: ~50mW (RACH preamble)
            'PS': 0.015,    # PSM (Power Saving Mode): ~15μW
            'name': 'NB-IoT',
            'description': '3GPP NB-IoT (Release 13+) for mMTC'
        },
        
        PowerProfile.LTE_M: {
            'PT': 250.0,    # Transmit: ~250mW (+23dBm)
            'PB': 100.0,    # Busy: ~100mW
            'PI': 5.0,      # Idle: ~5mW
            'PW': 60.0,     # Wakeup: ~60mW
            'PS': 0.020,    # PSM: ~20μW
            'name': 'LTE-M (eMTC)',
            'description': '3GPP LTE-M for enhanced MTC'
        },
        
        PowerProfile.NR_MMTC: {
            'PT': 200.0,    # Transmit: ~200mW (5G NR, lower power class)
            'PB': 70.0,     # Busy: ~70mW
            'PI': 2.0,      # Idle: ~2mW (with MICO mode)
            'PW': 40.0,     # Wakeup: ~40mW (2-step RACH)
            'PS': 0.010,    # Deep sleep (MICO): ~10μW
            'name': '5G NR mMTC',
            'description': '5G NR for massive MTC with MICO mode'
        },
        
        PowerProfile.GENERIC_LOW: {
            'PT': 100.0,    # Transmit: 100mW
            'PB': 50.0,     # Busy: 50mW
            'PI': 2.0,      # Idle: 2mW
            'PW': 20.0,     # Wakeup: 20mW
            'PS': 0.005,    # Sleep: 5μW
            'name': 'Generic Low Power',
            'description': 'Generic low-power IoT device'
        },
        
        PowerProfile.GENERIC_HIGH: {
            'PT': 500.0,    # Transmit: 500mW
            'PB': 200.0,    # Busy: 200mW
            'PI': 10.0,     # Idle: 10mW
            'PW': 100.0,    # Wakeup: 100mW
            'PS': 0.100,    # Sleep: 100μW
            'name': 'Generic High Power',
            'description': 'Generic high-power device'
        }
    }
    
    # Slot duration for different 3GPP technologies (in milliseconds)
    SLOT_DURATIONS = {
        'NR_15kHz': 1.0,     # 5G NR with 15 kHz SCS
        'NR_30kHz': 0.5,     # 5G NR with 30 kHz SCS
        'LTE': 1.0,          # LTE (1ms subframe)
        'NB_IoT': 2.0,       # NB-IoT (2ms)
        'default': 6.0       # Configurable (used in simulator)
    }
    
    @staticmethod
    def get_profile(profile: PowerProfile) -> Dict[str, float]:
        """
        Get predefined power profile.
        
        Args:
            profile: PowerProfile enum value
            
        Returns:
            Dictionary with power rates (PT, PB, PI, PW, PS) in mW
        """
        if profile not in PowerModel.PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        
        profile_data = PowerModel.PROFILES[profile].copy()
        # Remove metadata
        profile_data.pop('name', None)
        profile_data.pop('description', None)
        
        return profile_data
    
    @staticmethod
    def get_profile_info(profile: PowerProfile) -> Dict[str, str]:
        """Get profile metadata (name, description)."""
        if profile not in PowerModel.PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        
        return {
            'name': PowerModel.PROFILES[profile].get('name', ''),
            'description': PowerModel.PROFILES[profile].get('description', '')
        }
    
    @staticmethod
    def normalize_to_simulation_units(
        power_rates_mw: Dict[str, float],
        reference_power_mw: float = 1.0
    ) -> Dict[str, float]:
        """
        Normalize power rates to simulation units.
        
        Simulation units are relative to a reference power.
        For example, if reference = 1mW, then all powers are expressed
        as multiples of 1mW.
        
        Args:
            power_rates_mw: Power rates in mW
            reference_power_mw: Reference power for normalization (default 1mW)
            
        Returns:
            Normalized power rates (simulation units)
        """
        return {
            key: value / reference_power_mw
            for key, value in power_rates_mw.items()
        }
    
    @staticmethod
    def create_custom_profile(
        transmit_mw: float,
        busy_mw: Optional[float] = None,
        idle_mw: Optional[float] = None,
        wakeup_mw: Optional[float] = None,
        sleep_mw: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Create custom power profile.
        
        If optional parameters not provided, uses typical ratios:
        - Busy: 0.4 * Transmit
        - Idle: 0.02 * Transmit
        - Wakeup: 0.2 * Transmit
        - Sleep: 0.00001 * Transmit (10μW if TX=1W)
        
        Args:
            transmit_mw: Transmit power in mW
            busy_mw: Busy power (optional)
            idle_mw: Idle power (optional)
            wakeup_mw: Wakeup power (optional)
            sleep_mw: Sleep power (optional)
            
        Returns:
            Power profile dictionary
        """
        return {
            'PT': transmit_mw,
            'PB': busy_mw if busy_mw is not None else transmit_mw * 0.4,
            'PI': idle_mw if idle_mw is not None else transmit_mw * 0.02,
            'PW': wakeup_mw if wakeup_mw is not None else transmit_mw * 0.2,
            'PS': sleep_mw if sleep_mw is not None else transmit_mw * 0.00001
        }
    
    @staticmethod
    def estimate_lifetime(
        initial_energy_units: float,
        mean_energy_consumed_per_slot: float,
        slot_duration_ms: float = 6.0
    ) -> Dict[str, float]:
        """
        Estimate device lifetime.
        
        Args:
            initial_energy_units: Initial energy in simulation units
            mean_energy_consumed_per_slot: Average energy per slot
            slot_duration_ms: Slot duration in milliseconds
            
        Returns:
            Dictionary with lifetime in various units
        """
        if mean_energy_consumed_per_slot <= 0:
            return {
                'slots': float('inf'),
                'seconds': float('inf'),
                'hours': float('inf'),
                'days': float('inf'),
                'years': float('inf')
            }
        
        # Total slots until depletion
        total_slots = initial_energy_units / mean_energy_consumed_per_slot
        
        # Convert to time units
        slot_duration_s = slot_duration_ms / 1000.0
        total_seconds = total_slots * slot_duration_s
        
        return {
            'slots': total_slots,
            'seconds': total_seconds,
            'minutes': total_seconds / 60.0,
            'hours': total_seconds / 3600.0,
            'days': total_seconds / 86400.0,
            'years': total_seconds / (365.25 * 86400.0)
        }
    
    @staticmethod
    def create_battery_config(
        battery_type: str,
        initial_charge: float = 1.0
    ) -> BatteryConfig:
        """
        Create battery configuration from common types.
        
        Args:
            battery_type: One of 'AA', 'AAA', 'coin_cell', 'lipo_small', 'lipo_large'
            initial_charge: Initial charge fraction (0.0-1.0)
            
        Returns:
            BatteryConfig object
        """
        battery_specs = {
            'AA': {'capacity_mah': 2500.0, 'voltage_v': 1.5},
            'AAA': {'capacity_mah': 1200.0, 'voltage_v': 1.5},
            'coin_cell': {'capacity_mah': 220.0, 'voltage_v': 3.0},
            'lipo_small': {'capacity_mah': 1000.0, 'voltage_v': 3.7},
            'lipo_large': {'capacity_mah': 5000.0, 'voltage_v': 3.7}
        }
        
        if battery_type not in battery_specs:
            raise ValueError(f"Unknown battery type: {battery_type}. "
                           f"Choose from: {list(battery_specs.keys())}")
        
        spec = battery_specs[battery_type]
        return BatteryConfig(
            capacity_mah=spec['capacity_mah'],
            voltage_v=spec['voltage_v'],
            initial_charge_fraction=initial_charge
        )


def print_power_profiles():
    """Print all available power profiles."""
    print("=" * 70)
    print("Available Power Profiles (3GPP-Inspired)")
    print("=" * 70)
    
    for profile in PowerProfile:
        info = PowerModel.get_profile_info(profile)
        rates = PowerModel.get_profile(profile)
        
        print(f"\n{info['name']} ({profile.value})")
        print(f"  Description: {info['description']}")
        print(f"  Power Consumption:")
        print(f"    Transmit (PT): {rates['PT']:.3f} mW")
        print(f"    Busy (PB):     {rates['PB']:.3f} mW")
        print(f"    Idle (PI):     {rates['PI']:.3f} mW")
        print(f"    Wakeup (PW):   {rates['PW']:.3f} mW")
        print(f"    Sleep (PS):    {rates['PS']:.6f} mW ({rates['PS']*1000:.3f} uW)")
        
        # Calculate relative ratios
        pt = rates['PT']
        print(f"  Ratios (relative to transmit):")
        print(f"    PB/PT: {rates['PB']/pt:.3f}")
        print(f"    PI/PT: {rates['PI']/pt:.4f}")
        print(f"    PW/PT: {rates['PW']/pt:.3f}")
        print(f"    PS/PT: {rates['PS']/pt:.6f}")
    
    print("\n" + "=" * 70)


def example_lifetime_calculation():
    """Example: Calculate expected lifetime for different profiles."""
    print("\n" + "=" * 70)
    print("Battery Lifetime Estimation Example")
    print("=" * 70)
    
    # Battery: AA battery
    battery = PowerModel.create_battery_config('AA', initial_charge=1.0)
    print(f"\nBattery: AA ({battery.capacity_mah}mAh @ {battery.voltage_v}V)")
    print(f"Energy: {battery.get_energy_mwh():.2f} mWh")
    
    # Simulation parameters
    slot_duration_ms = 6.0
    arrival_rate = 0.01
    transmission_prob = 0.1
    
    print(f"\nSimulation Parameters:")
    print(f"  Slot duration: {slot_duration_ms} ms")
    print(f"  Arrival rate (lambda): {arrival_rate}")
    print(f"  Transmission prob (q): {transmission_prob}")
    
    print(f"\nEstimated Lifetimes:")
    print("-" * 70)
    
    for profile in [PowerProfile.LORA, PowerProfile.NB_IOT, PowerProfile.NR_MMTC]:
        info = PowerModel.get_profile_info(profile)
        rates = PowerModel.get_profile(profile)
        
        # Rough estimate: assume 50% sleep, 30% idle, 15% active, 5% transmit
        avg_power = (0.50 * rates['PS'] + 
                    0.30 * rates['PI'] + 
                    0.15 * rates['PI'] + 
                    0.05 * rates['PT'])
        
        # Energy per slot (in mWs)
        energy_per_slot = avg_power * (slot_duration_ms / 1000.0)
        
        # Total slots
        total_energy = battery.get_energy_mwh() * 3600.0  # mWs
        total_slots = total_energy / energy_per_slot
        
        # Lifetime in years
        total_seconds = total_slots * (slot_duration_ms / 1000.0)
        lifetime_years = total_seconds / (365.25 * 86400.0)
        
        print(f"{info['name']:20s}: {lifetime_years:.2f} years "
              f"({total_slots/1e6:.1f}M slots)")
    
    print("=" * 70)


if __name__ == "__main__":
    # Print all profiles
    print_power_profiles()
    
    # Example lifetime calculation
    example_lifetime_calculation()
