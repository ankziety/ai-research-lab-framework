"""
Physics Engines Package

This package contains implementations of high-fidelity physics simulation engines
for the AI research laboratory framework. Each engine provides a unified interface
while maintaining uncompromising scientific accuracy and reproducibility.

Author: Scientific Computing Engineer
Date: 2025-01-18
Phase: Phase 2 - Physics Engine Integration (Tracer Bullet Approach)
"""

# Import physics engine implementations
try:
    from .lammps_engine import LAMMPSEngine
except ImportError as e:
    print(f"Warning: LAMMPS engine not available: {e}")

try:
    from .fenics_engine import FEniCSEngine
except ImportError as e:
    print(f"Warning: FEniCS engine not available: {e}")

# Import interface components
from ..physics_engine_interface import (
    PhysicsEngineInterface,
    PhysicsEngineType,
    SimulationParameters,
    SimulationContext,
    SimulationResults,
    SimulationState,
    ValidationReport,
    CheckpointData,
    PhysicsEngineFactory,
)

__all__ = [
    'LAMMPSEngine',
    'FEniCSEngine',
    'PhysicsEngineInterface',
    'PhysicsEngineType',
    'SimulationParameters',
    'SimulationContext',
    'SimulationResults',
    'SimulationState',
    'ValidationReport',
    'CheckpointData',
    'PhysicsEngineFactory',
] 