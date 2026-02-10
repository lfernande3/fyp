"""
Sleep-Based Low-Latency Access for M2M Communications Simulator

A discrete-event simulation framework for sleep-based random access schemes
using slotted Aloha with on-demand sleep.

Date: February 10, 2026
"""

from .node import Node, NodeState
from .simulator import Simulator, BatchSimulator, SimulationConfig, SimulationResults
from .power_model import PowerModel, PowerProfile, BatteryConfig
from .validation import (
    TraceLogger, 
    AnalyticalValidator, 
    SanityChecker, 
    run_small_scale_test
)
from .metrics import (
    MetricsCalculator,
    AnalyticalMetrics,
    ComparisonMetrics,
    analyze_batch_results
)

__version__ = '1.0.0'
__all__ = [
    'Node', 
    'NodeState',
    'Simulator',
    'BatchSimulator',
    'SimulationConfig',
    'SimulationResults',
    'PowerModel',
    'PowerProfile',
    'BatteryConfig',
    'TraceLogger',
    'AnalyticalValidator',
    'SanityChecker',
    'run_small_scale_test',
    'MetricsCalculator',
    'AnalyticalMetrics',
    'ComparisonMetrics',
    'analyze_batch_results'
]
