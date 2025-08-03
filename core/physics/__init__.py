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

from .base_physics_engine import (
    BasePhysicsEngine, 
    PhysicsEngineType, 
    SoftwareInterface,
    PhysicsProblemSpec,
    PhysicsResult
)
from .quantum_simulation_engine import QuantumSimulationEngine
from .molecular_dynamics_engine import MolecularDynamicsEngine
from .statistical_physics_engine import StatisticalPhysicsEngine
from .multi_physics_engine import MultiPhysicsEngine
from .numerical_methods import NumericalMethodsEngine
from .physics_engine_factory import PhysicsEngineFactory, PhysicsEngineConfig
from .physics_engine_registry import PhysicsEngineRegistry, EngineStatus

__all__ = [
    # Base classes and types
    'BasePhysicsEngine',
    'PhysicsEngineType',
    'SoftwareInterface', 
    'PhysicsProblemSpec',
    'PhysicsResult',
    
    # Physics engines
    'QuantumSimulationEngine',
    'MolecularDynamicsEngine', 
    'StatisticalPhysicsEngine',
    'MultiPhysicsEngine',
    'NumericalMethodsEngine',
    
    # Management components
    'PhysicsEngineFactory',
    'PhysicsEngineConfig',
    'PhysicsEngineRegistry',
    'EngineStatus'
]

# Convenience function for easy access
def create_physics_engine_suite(config=None, cost_manager=None):
    """
    Create a complete suite of physics engines.
    
    Args:
        config: Optional configuration dictionary
        cost_manager: Optional cost manager
        
    Returns:
        Dictionary with factory, registry, and all engines
    """
    factory = PhysicsEngineFactory(config, cost_manager)
    registry = PhysicsEngineRegistry(factory)
    
    # Create all engine types
    engines = {}
    for engine_type in PhysicsEngineType:
        try:
            engine = factory.create_engine(engine_type)
            engines[engine_type.value] = engine
            
            # Register in registry
            registry.register_engine(
                engine_id=f"default_{engine_type.value}",
                engine_type=engine_type,
                auto_create=False  # Already created
            )
            registry.engines[f"default_{engine_type.value}"].engine_instance = engine
            
        except Exception as e:
            print(f"Warning: Could not create {engine_type.value}: {e}")
    
    return {
        'factory': factory,
        'registry': registry,
        'engines': engines
    }

__version__ = "1.0.0"
__author__ = "AI Research Lab Framework"