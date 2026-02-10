"""
Unit tests for the PowerModel module

Tests verify correct implementation of:
- Power profile definitions
- Battery configuration
- Energy calculations
- Lifetime estimations
- Custom profile creation

Date: February 10, 2026
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.power_model import PowerModel, PowerProfile, BatteryConfig


def test_power_profiles():
    """Test that all power profiles are defined correctly."""
    print("Testing power profiles...")
    
    for profile in PowerProfile:
        rates = PowerModel.get_profile(profile)
        
        # Check all required keys present
        assert 'PT' in rates
        assert 'PB' in rates
        assert 'PI' in rates
        assert 'PW' in rates
        assert 'PS' in rates
        
        # Check values are positive
        assert rates['PT'] > 0
        assert rates['PB'] > 0
        assert rates['PI'] > 0
        assert rates['PW'] > 0
        assert rates['PS'] > 0
        
        # Check typical relationships
        assert rates['PT'] > rates['PB']  # Transmit > Busy
        assert rates['PB'] > rates['PI']  # Busy > Idle
        assert rates['PI'] > rates['PS']  # Idle > Sleep
        assert rates['PT'] > rates['PW']  # Transmit > Wakeup (usually)
    
    print(f"[PASS] All {len(PowerProfile)} power profiles are valid")


def test_profile_info():
    """Test profile metadata retrieval."""
    print("\nTesting profile info...")
    
    for profile in PowerProfile:
        info = PowerModel.get_profile_info(profile)
        
        assert 'name' in info
        assert 'description' in info
        assert len(info['name']) > 0
        assert len(info['description']) > 0
    
    print("[PASS] Profile info test passed")


def test_specific_profiles():
    """Test specific profile values."""
    print("\nTesting specific profiles...")
    
    # Test NB-IoT profile
    nb_iot = PowerModel.get_profile(PowerProfile.NB_IOT)
    assert nb_iot['PT'] == 220.0
    assert nb_iot['PS'] == 0.015
    
    # Test LoRa profile
    lora = PowerModel.get_profile(PowerProfile.LORA)
    assert lora['PT'] == 120.0
    assert lora['PS'] == 0.001
    
    # Test NR mMTC profile
    nr_mmtc = PowerModel.get_profile(PowerProfile.NR_MMTC)
    assert nr_mmtc['PT'] == 200.0
    assert nr_mmtc['PS'] == 0.010
    
    print("[PASS] Specific profile values correct")


def test_normalization():
    """Test power rate normalization."""
    print("\nTesting normalization...")
    
    # Get a profile
    lora = PowerModel.get_profile(PowerProfile.LORA)
    
    # Normalize to 1mW reference
    normalized = PowerModel.normalize_to_simulation_units(lora, reference_power_mw=1.0)
    
    # Values should be same (reference is 1mW)
    assert normalized['PT'] == lora['PT']
    assert normalized['PS'] == lora['PS']
    
    # Normalize to different reference
    normalized_10 = PowerModel.normalize_to_simulation_units(lora, reference_power_mw=10.0)
    
    # Values should be scaled
    assert normalized_10['PT'] == lora['PT'] / 10.0
    assert normalized_10['PS'] == lora['PS'] / 10.0
    
    print("[PASS] Normalization test passed")


def test_custom_profile():
    """Test custom profile creation."""
    print("\nTesting custom profile creation...")
    
    # Create custom profile with only transmit power
    custom = PowerModel.create_custom_profile(transmit_mw=100.0)
    
    assert custom['PT'] == 100.0
    assert custom['PB'] == 40.0  # 0.4 * PT
    assert custom['PI'] == 2.0   # 0.02 * PT
    assert custom['PW'] == 20.0  # 0.2 * PT
    assert custom['PS'] == 0.001 # 0.00001 * PT
    
    # Create custom profile with all parameters
    custom_full = PowerModel.create_custom_profile(
        transmit_mw=200.0,
        busy_mw=100.0,
        idle_mw=5.0,
        wakeup_mw=50.0,
        sleep_mw=0.01
    )
    
    assert custom_full['PT'] == 200.0
    assert custom_full['PB'] == 100.0
    assert custom_full['PI'] == 5.0
    assert custom_full['PW'] == 50.0
    assert custom_full['PS'] == 0.01
    
    print("[PASS] Custom profile test passed")


def test_battery_config():
    """Test battery configuration."""
    print("\nTesting battery configuration...")
    
    # Create AA battery config
    battery = BatteryConfig(
        capacity_mah=2500.0,
        voltage_v=1.5,
        initial_charge_fraction=1.0
    )
    
    # Check energy calculations
    energy_j = battery.get_energy_joules()
    assert energy_j > 0
    
    energy_mwh = battery.get_energy_mwh()
    assert energy_mwh == 2500.0 * 1.5  # mAh * V = mWh
    
    energy_units = battery.get_energy_units(slot_duration_ms=6.0)
    assert energy_units > 0
    
    # Test with partial charge
    battery_half = BatteryConfig(
        capacity_mah=2500.0,
        voltage_v=1.5,
        initial_charge_fraction=0.5
    )
    
    assert battery_half.get_energy_mwh() == battery.get_energy_mwh() * 0.5
    
    print("[PASS] Battery config test passed")


def test_battery_types():
    """Test predefined battery types."""
    print("\nTesting predefined battery types...")
    
    battery_types = ['AA', 'AAA', 'coin_cell', 'lipo_small', 'lipo_large']
    
    for btype in battery_types:
        battery = PowerModel.create_battery_config(btype)
        
        assert battery.capacity_mah > 0
        assert battery.voltage_v > 0
        assert battery.initial_charge_fraction == 1.0
        
        # Check energy calculations work
        energy = battery.get_energy_mwh()
        assert energy > 0
    
    print(f"[PASS] All {len(battery_types)} battery types valid")


def test_lifetime_estimation():
    """Test lifetime estimation."""
    print("\nTesting lifetime estimation...")
    
    # Test with finite energy consumption
    lifetime = PowerModel.estimate_lifetime(
        initial_energy_units=5000.0,
        mean_energy_consumed_per_slot=1.0,
        slot_duration_ms=6.0
    )
    
    assert lifetime['slots'] == 5000.0
    assert lifetime['seconds'] == 5000.0 * 0.006  # 30 seconds
    assert lifetime['hours'] > 0
    assert lifetime['days'] > 0
    assert lifetime['years'] > 0
    
    # Test with zero consumption (infinite lifetime)
    lifetime_inf = PowerModel.estimate_lifetime(
        initial_energy_units=5000.0,
        mean_energy_consumed_per_slot=0.0,
        slot_duration_ms=6.0
    )
    
    assert lifetime_inf['slots'] == float('inf')
    assert lifetime_inf['years'] == float('inf')
    
    print("[PASS] Lifetime estimation test passed")


def test_realistic_lifetime():
    """Test realistic lifetime calculation."""
    print("\nTesting realistic lifetime calculation...")
    
    # AA battery with LoRa profile (lower power)
    battery = PowerModel.create_battery_config('AA')
    lora = PowerModel.get_profile(PowerProfile.LORA)
    
    # Assume mostly sleeping (98%), some idle (1.5%), little active (0.5%)
    avg_power_mw = (0.98 * lora['PS'] + 
                    0.015 * lora['PI'] + 
                    0.005 * lora['PT'])
    
    slot_duration_ms = 6.0
    
    # Energy per slot in mWs
    energy_per_slot_mws = avg_power_mw * (slot_duration_ms / 1000.0)
    
    # Total energy in mWs
    total_energy_mws = battery.get_energy_mwh() * 3600.0
    
    # Total slots
    total_slots = total_energy_mws / energy_per_slot_mws
    
    # Lifetime in years
    total_seconds = total_slots * (slot_duration_ms / 1000.0)
    lifetime_years = total_seconds / (365.25 * 86400.0)
    
    # Should be on the order of years for IoT device with mostly sleep
    assert lifetime_years > 0.01  # At least a few days
    assert lifetime_years < 100   # Sanity check
    
    print(f"  AA + LoRa (98% sleep): {lifetime_years:.2f} years")
    print("[PASS] Realistic lifetime test passed")


def test_energy_unit_conversion():
    """Test energy unit conversions."""
    print("\nTesting energy unit conversions...")
    
    battery = BatteryConfig(
        capacity_mah=1000.0,
        voltage_v=3.7,
        initial_charge_fraction=1.0
    )
    
    # mWh
    energy_mwh = battery.get_energy_mwh()
    assert energy_mwh == 1000.0 * 3.7
    
    # Joules (should be mWh * 3.6)
    energy_j = battery.get_energy_joules()
    expected_j = (1000.0 / 1000.0) * 3.7 * 3600.0
    assert abs(energy_j - expected_j) < 0.01
    
    # Simulation units
    energy_units = battery.get_energy_units(slot_duration_ms=6.0)
    assert energy_units > 0
    
    print("[PASS] Energy unit conversion test passed")


def test_power_profile_ratios():
    """Test that power ratios are reasonable."""
    print("\nTesting power profile ratios...")
    
    for profile in PowerProfile:
        rates = PowerModel.get_profile(profile)
        pt = rates['PT']
        
        # Check ratios are reasonable
        assert rates['PS'] / pt < 0.01  # Sleep should be < 1% of transmit
        assert rates['PI'] / pt < 0.5   # Idle should be < 50% of transmit
        assert rates['PB'] / pt < 1.0   # Busy should be < transmit
        assert rates['PW'] / pt < 1.0   # Wakeup should be <= transmit
    
    print("[PASS] Power profile ratios reasonable")


def run_all_tests():
    """Run all power model tests."""
    print("=" * 60)
    print("Running PowerModel Unit Tests")
    print("=" * 60)
    
    test_power_profiles()
    test_profile_info()
    test_specific_profiles()
    test_normalization()
    test_custom_profile()
    test_battery_config()
    test_battery_types()
    test_lifetime_estimation()
    test_realistic_lifetime()
    test_energy_unit_conversion()
    test_power_profile_ratios()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
