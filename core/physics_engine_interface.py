"""
Physics Engine Interface - Unified Abstraction Layer

This module provides a unified interface for high-fidelity physics simulation engines,
enabling AI agents to conduct scientifically rigorous simulations suitable for 
peer-reviewed publication. The interface handles both particle-based (LAMMPS) and 
mesh-based (FEniCS) approaches while maintaining uncompromising scientific accuracy.

Author: Scientific Computing Engineer
Date: 2025-01-18
Phase: Phase 2 - Physics Engine Integration (Tracer Bullet Approach)
"""

import abc
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import vtk
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsEngineType(Enum):
    """Enumeration of supported physics engine types."""
    LAMMPS = "lammps"  # Particle-based molecular dynamics
    FENICS = "fenics"   # Finite element methods for PDEs
    GEANT4 = "geant4"   # Particle physics and radiation transport
    CHRONO = "chrono"   # Mechanical multi-physics simulations
    OPENMM = "openmm"   # Molecular dynamics simulation
    GROMACS = "gromacs" # Biomolecular dynamics package
    DEALII = "dealii"   # Finite element library for scientific computing


class SimulationState(Enum):
    """Enumeration of simulation states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"


@dataclass
class SimulationParameters:
    """Container for simulation parameters with validation."""
    
    # Core parameters
    engine_type: PhysicsEngineType
    simulation_name: str
    time_step: float
    total_time: float
    
    # Physical parameters
    temperature: Optional[float] = None  # Kelvin
    pressure: Optional[float] = None     # Pascal
    density: Optional[float] = None      # kg/m³
    
    # Numerical parameters
    tolerance: float = 1e-6
    max_iterations: int = 1000
    seed: Optional[int] = None
    
    # Output parameters
    output_frequency: int = 100
    checkpoint_frequency: int = 1000
    
    # Validation
    def validate(self) -> List[str]:
        """Validate parameters and return list of errors."""
        errors = []
        
        if self.time_step <= 0:
            errors.append("time_step must be positive")
        
        if self.total_time <= 0:
            errors.append("total_time must be positive")
        
        if self.tolerance <= 0:
            errors.append("tolerance must be positive")
        
        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        
        if self.output_frequency <= 0:
            errors.append("output_frequency must be positive")
        
        if self.checkpoint_frequency <= 0:
            errors.append("checkpoint_frequency must be positive")
        
        return errors


@dataclass
class SimulationContext:
    """Context for simulation execution with full provenance tracking."""
    
    parameters: SimulationParameters
    state: SimulationState = SimulationState.INITIALIZED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Provenance tracking
    input_files: List[Path] = field(default_factory=list)
    output_files: List[Path] = field(default_factory=list)
    checkpoint_files: List[Path] = field(default_factory=list)
    
    # Performance metrics
    wall_clock_time: Optional[float] = None
    cpu_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    # Validation results
    conservation_errors: Dict[str, float] = field(default_factory=dict)
    benchmark_errors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "parameters": {
                "engine_type": self.parameters.engine_type.value,
                "simulation_name": self.parameters.simulation_name,
                "time_step": self.parameters.time_step,
                "total_time": self.parameters.total_time,
                "temperature": self.parameters.temperature,
                "pressure": self.parameters.pressure,
                "density": self.parameters.density,
                "tolerance": self.parameters.tolerance,
                "max_iterations": self.parameters.max_iterations,
                "seed": self.parameters.seed,
                "output_frequency": self.parameters.output_frequency,
                "checkpoint_frequency": self.parameters.checkpoint_frequency,
            },
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input_files": [str(f) for f in self.input_files],
            "output_files": [str(f) for f in self.output_files],
            "checkpoint_files": [str(f) for f in self.checkpoint_files],
            "wall_clock_time": self.wall_clock_time,
            "cpu_time": self.cpu_time,
            "memory_usage": self.memory_usage,
            "conservation_errors": self.conservation_errors,
            "benchmark_errors": self.benchmark_errors,
        }


@dataclass
class SimulationResults:
    """Container for simulation results with validation metadata."""
    
    # Core results
    final_time: float
    final_state: np.ndarray
    time_series: Optional[np.ndarray] = None
    
    # Physical quantities
    energy: Optional[np.ndarray] = None
    momentum: Optional[np.ndarray] = None
    mass: Optional[np.ndarray] = None
    charge: Optional[np.ndarray] = None
    
    # Validation results
    conservation_laws: Dict[str, float] = field(default_factory=dict)
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    units: Dict[str, str] = field(default_factory=dict)
    grid_info: Optional[Dict[str, Any]] = None
    particle_info: Optional[Dict[str, Any]] = None
    
    def validate_conservation(self) -> Dict[str, float]:
        """Validate conservation laws and return relative errors."""
        errors = {}
        
        if self.energy is not None:
            energy_error = np.std(self.energy) / np.mean(self.energy)
            errors["energy"] = energy_error
        
        if self.momentum is not None:
            momentum_array = np.array(self.momentum) if isinstance(self.momentum, list) else self.momentum
            if len(momentum_array.shape) > 1:
                momentum_error = np.std(np.linalg.norm(momentum_array, axis=1)) / np.mean(np.linalg.norm(momentum_array, axis=1))
                errors["momentum"] = momentum_error
        
        if self.mass is not None:
            mass_error = np.std(self.mass) / np.mean(self.mass)
            errors["mass"] = mass_error
        
        if self.charge is not None:
            charge_error = np.std(self.charge) / np.mean(self.charge)
            errors["charge"] = charge_error
        
        return errors
    
    def save_hdf5(self, filename: Path) -> None:
        """Save results to HDF5 format for efficient I/O."""
        with h5py.File(filename, 'w') as f:
            # Core data
            f.create_dataset('final_time', data=self.final_time)
            f.create_dataset('final_state', data=self.final_state)
            
            if self.time_series is not None:
                f.create_dataset('time_series', data=self.time_series)
            
            # Physical quantities
            if self.energy is not None:
                f.create_dataset('energy', data=self.energy)
            if self.momentum is not None:
                f.create_dataset('momentum', data=self.momentum)
            if self.mass is not None:
                f.create_dataset('mass', data=self.mass)
            if self.charge is not None:
                f.create_dataset('charge', data=self.charge)
            
            # Metadata
            f.attrs['units'] = json.dumps(self.units)
            if self.grid_info is not None:
                f.attrs['grid_info'] = json.dumps(self.grid_info)
            if self.particle_info is not None:
                f.attrs['particle_info'] = json.dumps(self.particle_info)
    
    def save_vtk(self, filename: Path) -> None:
        """Save results to VTK format for visualization."""
        # Implementation for VTK output
        pass


@dataclass
class ValidationReport:
    """Comprehensive validation report for simulation results."""
    
    # Conservation law validation
    conservation_passed: bool
    conservation_errors: Dict[str, float]
    benchmark_passed: bool
    benchmark_errors: Dict[str, float]
    numerical_stable: bool
    stability_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    conservation_threshold: float = 1e-15  # Machine precision
    benchmark_threshold: float = 1e-3  # 0.1% tolerance
    
    def is_valid(self) -> bool:
        """Check if all validation criteria are met."""
        return (self.conservation_passed and 
                self.benchmark_passed and 
                self.numerical_stable)


@dataclass
class CheckpointData:
    """Checkpoint data for fault tolerance and restart capabilities."""
    
    context: SimulationContext
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def save(self, filename: Path) -> None:
        """Save checkpoint to file."""
        checkpoint_data = {
            "context": self.context.to_dict(),
            "state_data": self.state_data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
        
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    @classmethod
    def load(cls, filename: Path) -> 'CheckpointData':
        """Load checkpoint from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct context
        context_data = data["context"]
        parameters_data = context_data["parameters"]
        
        parameters = SimulationParameters(
            engine_type=PhysicsEngineType(parameters_data["engine_type"]),
            simulation_name=parameters_data["simulation_name"],
            time_step=parameters_data["time_step"],
            total_time=parameters_data["total_time"],
            temperature=parameters_data.get("temperature"),
            pressure=parameters_data.get("pressure"),
            density=parameters_data.get("density"),
            tolerance=parameters_data["tolerance"],
            max_iterations=parameters_data["max_iterations"],
            seed=parameters_data.get("seed"),
            output_frequency=parameters_data["output_frequency"],
            checkpoint_frequency=parameters_data["checkpoint_frequency"],
        )
        
        context = SimulationContext(
            parameters=parameters,
            state=SimulationState(context_data["state"]),
            start_time=datetime.fromisoformat(context_data["start_time"]) if context_data["start_time"] else None,
            end_time=datetime.fromisoformat(context_data["end_time"]) if context_data["end_time"] else None,
            input_files=[Path(f) for f in context_data["input_files"]],
            output_files=[Path(f) for f in context_data["output_files"]],
            checkpoint_files=[Path(f) for f in context_data["checkpoint_files"]],
            wall_clock_time=context_data.get("wall_clock_time"),
            cpu_time=context_data.get("cpu_time"),
            memory_usage=context_data.get("memory_usage"),
            conservation_errors=context_data.get("conservation_errors", {}),
            benchmark_errors=context_data.get("benchmark_errors", {}),
        )
        
        return cls(
            context=context,
            state_data=data["state_data"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class PhysicsEngineInterface(ABC):
    """
    Abstract base class for all physics engines.
    
    This interface provides a unified API for both particle-based (LAMMPS) and 
    mesh-based (FEniCS) physics engines while preserving scientific accuracy and
    enabling reproducible research.
    """
    
    def __init__(self, engine_type: PhysicsEngineType):
        """Initialize physics engine interface."""
        self.engine_type = engine_type
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set random seed for reproducibility
        if hasattr(self, 'set_seed'):
            self.set_seed(42)  # Default seed for reproducibility
    
    @abstractmethod
    def initialize_simulation(self, parameters: SimulationParameters) -> SimulationContext:
        """
        Initialize simulation with validated parameters.
        
        Args:
            parameters: Validated simulation parameters
            
        Returns:
            SimulationContext with initialized state
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def run_simulation(self, context: SimulationContext) -> SimulationResults:
        """
        Execute simulation with full provenance tracking.
        
        Args:
            context: Initialized simulation context
            
        Returns:
            SimulationResults with validated output
            
        Raises:
            RuntimeError: If simulation fails
            ValueError: If results fail validation
        """
        pass
    
    @abstractmethod
    def validate_results(self, results: SimulationResults) -> ValidationReport:
        """
        Validate results against analytical solutions and benchmarks.
        
        Args:
            results: Simulation results to validate
            
        Returns:
            ValidationReport with comprehensive validation results
        """
        pass
    
    @abstractmethod
    def checkpoint_state(self, context: SimulationContext) -> CheckpointData:
        """
        Create reproducible checkpoint for fault tolerance.
        
        Args:
            context: Current simulation context
            
        Returns:
            CheckpointData for restart capability
        """
        pass
    
    @abstractmethod
    def restore_from_checkpoint(self, checkpoint: CheckpointData) -> SimulationContext:
        """
        Restore simulation from checkpoint with bit-level accuracy.
        
        Args:
            checkpoint: Checkpoint data to restore from
            
        Returns:
            SimulationContext ready for continuation
        """
        pass
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results."""
        np.random.seed(seed)
        # Engine-specific seed setting will be implemented in subclasses
    
    def validate_parameters(self, parameters: SimulationParameters) -> None:
        """Validate simulation parameters."""
        errors = parameters.validate()
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")
    
    def create_output_directory(self, simulation_name: str) -> Path:
        """Create output directory for simulation results."""
        output_dir = Path("simulations") / simulation_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def log_simulation_start(self, context: SimulationContext) -> None:
        """Log simulation start with full context."""
        self.logger.info(f"Starting {context.parameters.engine_type.value} simulation: {context.parameters.simulation_name}")
        self.logger.info(f"Parameters: {context.parameters}")
        context.start_time = datetime.now()
    
    def log_simulation_end(self, context: SimulationContext, results: SimulationResults) -> None:
        """Log simulation completion with performance metrics."""
        context.end_time = datetime.now()
        duration = (context.end_time - context.start_time).total_seconds()
        
        self.logger.info(f"Completed {context.parameters.engine_type.value} simulation: {context.parameters.simulation_name}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Final time: {results.final_time}")
        
        # Log conservation errors
        conservation_errors = results.validate_conservation()
        for quantity, error in conservation_errors.items():
            self.logger.info(f"{quantity} conservation error: {error:.2e}")
    
    def save_results(self, results: SimulationResults, output_dir: Path) -> None:
        """Save results in multiple formats for analysis."""
        # Save HDF5 for efficient I/O
        hdf5_file = output_dir / "results.h5"
        results.save_hdf5(hdf5_file)
        
        # Save VTK for visualization
        vtk_file = output_dir / "results.vtk"
        results.save_vtk(vtk_file)
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        metadata = {
            "engine_type": self.engine_type.value,
            "final_time": results.final_time,
            "units": results.units,
            "grid_info": results.grid_info,
            "particle_info": results.particle_info,
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


class PhysicsEngineFactory:
    """Factory for creating physics engine instances."""
    
    _engines: Dict[PhysicsEngineType, type] = {}
    
    @classmethod
    def register_engine(cls, engine_type: PhysicsEngineType, engine_class: type) -> None:
        """Register a physics engine implementation."""
        if not issubclass(engine_class, PhysicsEngineInterface):
            raise ValueError(f"Engine class must inherit from PhysicsEngineInterface")
        cls._engines[engine_type] = engine_class
    
    @classmethod
    def create_engine(cls, engine_type: PhysicsEngineType) -> PhysicsEngineInterface:
        """Create a physics engine instance."""
        if engine_type not in cls._engines:
            raise ValueError(f"Engine type {engine_type} not registered")
        
        engine_class = cls._engines[engine_type]
        return engine_class(engine_type)
    
    @classmethod
    def list_available_engines(cls) -> List[PhysicsEngineType]:
        """List all available physics engines."""
        return list(cls._engines.keys())


# Example usage and testing
if __name__ == "__main__":
    # Test parameter validation
    params = SimulationParameters(
        engine_type=PhysicsEngineType.LAMMPS,
        simulation_name="test_simulation",
        time_step=0.001,
        total_time=1.0,
        temperature=300.0,
        tolerance=1e-6,
        max_iterations=1000,
        seed=42,
        output_frequency=100,
        checkpoint_frequency=1000,
    )
    
    errors = params.validate()
    print(f"Parameter validation errors: {errors}")
    
    # Test context creation
    context = SimulationContext(parameters=params)
    print(f"Created simulation context: {context}")
    
    # Test factory pattern
    print(f"Available engines: {PhysicsEngineFactory.list_available_engines()}") 