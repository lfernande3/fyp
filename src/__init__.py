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
from .experiments import (
    ParameterSweep,
    ScenarioExperiments,
    SweepConfig,
    ScenarioConfig
)
from .traffic_models import (
    TrafficGenerator,
    TrafficModel,
    BurstyTrafficConfig
)
from .visualizations import (
    SimulationVisualizer,
    InteractiveVisualizer,
    PlotConfig,
    plot_parameter_sweep_summary,
    save_figure
)
from .optimization import (
    ParameterOptimizer,
    DutyCycleSimulator,
    PrioritizationAnalyzer,
    OptimizationVisualizer,
    OptimizationResult,
    TradeoffPoint,
    PrioritizationComparison,
    run_optimization_experiments,
)
from .validation_3gpp import (
    ThreeGPPAlignment,
    ThreeGPPScenario,
    AnalyticsValidator,
    FormulaValidationResult,
    ValidationReport,
    DesignGuidelines,
    GuidelineEntry,
    ValidationVisualizer,
    run_o4_experiments,
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
    'analyze_batch_results',
    'ParameterSweep',
    'ScenarioExperiments',
    'SweepConfig',
    'ScenarioConfig',
    'TrafficGenerator',
    'TrafficModel',
    'BurstyTrafficConfig',
    'SimulationVisualizer',
    'InteractiveVisualizer',
    'PlotConfig',
    'plot_parameter_sweep_summary',
    'save_figure',
    # O3 – Optimization
    'ParameterOptimizer',
    'DutyCycleSimulator',
    'PrioritizationAnalyzer',
    'OptimizationVisualizer',
    'OptimizationResult',
    'TradeoffPoint',
    'PrioritizationComparison',
    'run_optimization_experiments',
    # O4 – 3GPP Validation & Guidelines
    'ThreeGPPAlignment',
    'ThreeGPPScenario',
    'AnalyticsValidator',
    'FormulaValidationResult',
    'ValidationReport',
    'DesignGuidelines',
    'GuidelineEntry',
    'ValidationVisualizer',
    'run_o4_experiments',
]
