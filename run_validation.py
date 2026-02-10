"""
Standalone script to run validation and sanity checks.

Date: February 10, 2026
"""

from src.validation import run_small_scale_test, SanityChecker
from src.simulator import SimulationConfig

# Run small-scale test
print("Running small-scale integration test...\n")
test_results = run_small_scale_test(verbose=True)

# Run sanity checks
print("\n\n")
base_config = test_results['config']
sanity_results = SanityChecker.run_all_checks(base_config)

# Print detailed results
print("\n\n" + "=" * 80)
print("DETAILED SANITY CHECK RESULTS")
print("=" * 80)

# No-sleep check
print("\n1. No-Sleep vs. Standard Aloha:")
print(f"   Sleep fraction: {sanity_results['no_sleep_check']['sleep_fraction']:.6f}")
print(f"   Wakeup fraction: {sanity_results['no_sleep_check']['wakeup_fraction']:.6f}")
print(f"   Empirical p: {sanity_results['no_sleep_check']['empirical_p']:.4f}")
print(f"   Analytical p: {sanity_results['no_sleep_check']['analytical_p']:.4f}")
print(f"   Error: {sanity_results['no_sleep_check']['p_error']*100:.2f}%")
print(f"   Status: {'PASS' if sanity_results['no_sleep_check']['passed'] else 'FAIL'}")

# Immediate sleep check
print("\n2. Immediate Sleep Increases Sleep/Wakeup Activity:")
print(f"   Normal delay: {sanity_results['immediate_sleep_check']['normal_delay']:.2f} slots")
print(f"   Immediate delay: {sanity_results['immediate_sleep_check']['immediate_delay']:.2f} slots")
print(f"   Delay change: {sanity_results['immediate_sleep_check']['delay_change']:.2f} slots")
print(f"   Normal sleep fraction: {sanity_results['immediate_sleep_check']['normal_sleep_fraction']:.4f}")
print(f"   Immediate sleep fraction: {sanity_results['immediate_sleep_check']['immediate_sleep_fraction']:.4f}")
print(f"   Normal wakeup fraction: {sanity_results['immediate_sleep_check']['normal_wakeup_fraction']:.4f}")
print(f"   Immediate wakeup fraction: {sanity_results['immediate_sleep_check']['immediate_wakeup_fraction']:.4f}")
print(f"   Sleep increased: {sanity_results['immediate_sleep_check']['sleep_increased']}")
print(f"   Wakeup increased: {sanity_results['immediate_sleep_check']['wakeup_increased']}")
print(f"   Status: {'PASS' if sanity_results['immediate_sleep_check']['passed'] else 'FAIL'}")

# High q check
print("\n3. Higher q Increases Collisions:")
print(f"   Low q (0.05) collisions: {sanity_results['high_q_check']['low_q_collisions']}")
print(f"   High q (0.30) collisions: {sanity_results['high_q_check']['high_q_collisions']}")
print(f"   Collision increase: {sanity_results['high_q_check']['collision_increase']}")
print(f"   Status: {'PASS' if sanity_results['high_q_check']['passed'] else 'FAIL'}")

print("\n" + "=" * 80)
