"""
Computational Physics Engines for AI Research Lab Framework

This module provides sophisticated physics simulation engines for scientific research,
inspired by the Virtual Lab methodology. The engines support various domains of 
computational physics with integration to external physics software packages.

Key Features:
- Advanced mathematical modeling (quantum mechanics, relativity, statistical physics)
- Computational physics simulations (molecular dynamics, quantum chemistry, fluid dynamics)
- Integration with physics software (LAMMPS, Quantum ESPRESSO, OpenFOAM, GROMACS, VASP)
- Parallel computing support
- Numerical stability and error handling
- Performance optimization for physics simulations

Available Engines:
- QuantumSimulationEngine: Quantum mechanics and quantum chemistry
- MolecularDynamicsEngine: Particle simulations and statistical mechanics
- StatisticalPhysicsEngine: Monte Carlo methods and phase transitions
- MultiPhysicsEngine: Multi-scale and multi-domain coupling
- NumericalMethodsEngine: Advanced numerical methods for PDEs
"""

from .base_physics_engine import BasePhysicsEngine
from .quantum_simulation_engine import QuantumSimulationEngine
from .molecular_dynamics_engine import MolecularDynamicsEngine
from .statistical_physics_engine import StatisticalPhysicsEngine
from .multi_physics_engine import MultiPhysicsEngine
from .numerical_methods import NumericalMethodsEngine
from .physics_engine_factory import PhysicsEngineFactory
from .physics_engine_registry import PhysicsEngineRegistry

__all__ = [
    'BasePhysicsEngine',
    'QuantumSimulationEngine',
    'MolecularDynamicsEngine', 
    'StatisticalPhysicsEngine',
    'MultiPhysicsEngine',
    'NumericalMethodsEngine',
    'PhysicsEngineFactory',
    'PhysicsEngineRegistry'
]

__version__ = "1.0.0"
__author__ = "AI Research Lab Framework"