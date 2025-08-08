"""
Physics Engine Adapter

Adapter layer that connects physics tools with physics engines.
Provides a unified interface for physics tools to discover and use
physics engines for actual computational work.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PhysicsEngineAdapter:
    """
    Adapter that enables physics tools to discover and use physics engines.
    
    Acts as a bridge between the agent-facing physics tools and the 
    computational physics engines, handling engine discovery, parameter
    translation, and result formatting.
    """
    
    def __init__(self):
        """Initialize the physics engine adapter."""
        self._engine_registry = None
        self._engine_factory = None
        self._available_engines = {}
        self._engine_mappings = self._create_engine_mappings()
        
        # Try to import and initialize physics engines
        self._initialize_engine_systems()
    
    def _initialize_engine_systems(self):
        """Initialize physics engine registry and factory if available."""
        try:
            # Try to import physics engine components from PR #18
            from core.physics import PhysicsEngineRegistry, PhysicsEngineFactory
            
            self._engine_registry = PhysicsEngineRegistry()
            self._engine_factory = PhysicsEngineFactory(registry=self._engine_registry)
            
            # Discover available engines
            self._discover_available_engines()
            
            logger.info(f"Initialized physics engine adapter with {len(self._available_engines)} engines")
            
        except ImportError as e:
            logger.warning(f"Physics engines not available: {e}")
            logger.info("Physics tools will use fallback implementations")
    
    def _create_engine_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Create mappings between tool capabilities and engine types."""
        return {
            # Quantum chemistry tool mappings
            "quantum_chemistry": {
                "engine_type": "quantum_simulation",
                "calculation_mappings": {
                    "energy_calculation": "energy_calculation",
                    "geometry_optimization": "geometry_optimization",
                    "molecular_properties": "molecular_properties",
                    "orbital_analysis": "electronic_structure",
                    "vibrational_analysis": "vibrational_analysis"
                }
            },
            
            # Materials science tool mappings
            "materials_science": {
                "engine_type": "multi_physics",  # Can use multiple engines
                "calculation_mappings": {
                    "structure_analysis": "structure_analysis",
                    "mechanical_properties": "mechanical_simulation",
                    "electronic_properties": "electronic_structure",
                    "thermal_properties": "thermal_simulation",
                    "phase_stability": "thermodynamic_analysis"
                }
            },
            
            # Astrophysics tool mappings
            "astrophysics": {
                "engine_type": "numerical_methods",
                "calculation_mappings": {
                    "stellar_evolution": "ode_simulation",
                    "orbital_dynamics": "dynamical_system",
                    "cosmological_distance": "numerical_calculation",
                    "galaxy_rotation": "fluid_dynamics",
                    "gravitational_waves": "wave_equation"
                }
            },
            
            # Experimental physics tool mappings
            "experimental_physics": {
                "engine_type": "statistical_physics",
                "calculation_mappings": {
                    "descriptive_statistics": "statistical_analysis",
                    "curve_fitting": "parameter_estimation",
                    "hypothesis_test": "statistical_test",
                    "uncertainty_analysis": "error_analysis",
                    "correlation_analysis": "correlation_analysis"
                }
            }
        }
    
    def _discover_available_engines(self):
        """Discover available physics engines."""
        if not self._engine_registry:
            return
        
        try:
            # Get available engine types from the registry
            engine_info = self._engine_registry.get_available_engines()
            
            for engine_id, engine_data in engine_info.items():
                self._available_engines[engine_id] = {
                    "capabilities": engine_data.get("capabilities", []),
                    "domains": engine_data.get("physics_domains", []),
                    "performance": engine_data.get("performance_stats", {}),
                    "available": True
                }
                
        except Exception as e:
            logger.error(f"Failed to discover physics engines: {e}")
    
    def is_engine_available(self, physics_domain: str, calculation_type: str) -> bool:
        """
        Check if a physics engine is available for the given domain and calculation.
        
        Args:
            physics_domain: Physics domain (e.g., 'quantum_chemistry')
            calculation_type: Type of calculation needed
            
        Returns:
            True if suitable engine is available
        """
        if not self._engine_registry:
            return False
        
        domain_mapping = self._engine_mappings.get(physics_domain)
        if not domain_mapping:
            return False
        
        engine_type = domain_mapping["engine_type"]
        return engine_type in self._available_engines
    
    def execute_with_engine(self, 
                           physics_domain: str,
                           calculation_type: str,
                           parameters: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a physics calculation using an appropriate engine.
        
        Args:
            physics_domain: Physics domain for the calculation
            calculation_type: Type of calculation to perform
            parameters: Calculation parameters
            context: Execution context
            
        Returns:
            Calculation results from the physics engine
        """
        if not self.is_engine_available(physics_domain, calculation_type):
            raise RuntimeError(f"No physics engine available for {physics_domain}.{calculation_type}")
        
        try:
            # Get engine mapping
            domain_mapping = self._engine_mappings[physics_domain]
            engine_type = domain_mapping["engine_type"]
            calc_mapping = domain_mapping["calculation_mappings"]
            
            # Translate calculation type
            engine_calc_type = calc_mapping.get(calculation_type, calculation_type)
            
            # Create or get engine
            engine = self._get_or_create_engine(engine_type, context)
            
            # Translate parameters for engine
            engine_params = self._translate_parameters_for_engine(
                parameters, physics_domain, calculation_type
            )
            
            # Execute calculation
            result = engine.solve_problem(
                problem_type=engine_calc_type,
                parameters=engine_params,
                context=context
            )
            
            # Translate results back for tool
            translated_result = self._translate_results_from_engine(
                result, physics_domain, calculation_type
            )
            
            return translated_result
            
        except Exception as e:
            logger.error(f"Engine execution failed: {e}")
            raise
    
    def _get_or_create_engine(self, engine_type: str, context: Dict[str, Any]):
        """Get or create a physics engine of the specified type."""
        if not self._engine_factory:
            raise RuntimeError("Physics engine factory not available")
        
        try:
            # Map engine type names to factory types
            engine_type_mapping = {
                "quantum_simulation": "QuantumSimulationEngine",
                "molecular_dynamics": "MolecularDynamicsEngine", 
                "statistical_physics": "StatisticalPhysicsEngine",
                "multi_physics": "MultiPhysicsEngine",
                "numerical_methods": "NumericalMethodsEngine"
            }
            
            factory_type = engine_type_mapping.get(engine_type, engine_type)
            
            # Create engine with appropriate configuration
            engine_config = self._create_engine_config(engine_type, context)
            engine = self._engine_factory.create_engine(factory_type, engine_config)
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create engine {engine_type}: {e}")
            raise
    
    def _create_engine_config(self, engine_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for physics engine based on context."""
        base_config = {
            "computational_resources": {
                "cpu_cores": context.get("available_cpu_cores", 1),
                "memory_mb": context.get("available_memory", 2048),
                "use_gpu": context.get("gpu_available", False)
            },
            "numerical_precision": context.get("precision_preference", "standard"),
            "performance_mode": context.get("performance_mode", "balanced")
        }
        
        # Engine-specific configurations
        if engine_type == "quantum_simulation":
            base_config.update({
                "default_method": "hartree_fock",
                "convergence_threshold": 1e-6,
                "max_iterations": 100
            })
        elif engine_type == "molecular_dynamics":
            base_config.update({
                "integrator": "verlet",
                "timestep": 1.0,  # fs
                "temperature_control": "nose_hoover"
            })
        elif engine_type == "statistical_physics":
            base_config.update({
                "sampling_method": "metropolis",
                "equilibration_steps": 1000,
                "production_steps": 10000
            })
        
        return base_config
    
    def _translate_parameters_for_engine(self, 
                                       parameters: Dict[str, Any],
                                       physics_domain: str,
                                       calculation_type: str) -> Dict[str, Any]:
        """Translate tool parameters to engine parameters."""
        # This is a simplified translation - in practice would be more sophisticated
        engine_params = parameters.copy()
        
        # Domain-specific parameter translations
        if physics_domain == "quantum_chemistry":
            # Translate molecular specification
            if "molecule" in parameters:
                molecule = parameters["molecule"]
                engine_params["system"] = {
                    "atoms": molecule.get("atoms", []),
                    "charge": molecule.get("charge", 0),
                    "multiplicity": molecule.get("multiplicity", 1)
                }
                
            # Translate method specifications
            if "method" in parameters:
                engine_params["calculation_method"] = parameters["method"]
            if "basis_set" in parameters:
                engine_params["basis_set"] = parameters["basis_set"]
        
        elif physics_domain == "materials_science":
            # Translate material specification
            if "material" in parameters:
                material = parameters["material"]
                engine_params["material_system"] = {
                    "composition": material.get("composition", {}),
                    "structure": material.get("structure", {}),
                    "unit_cell": material.get("unit_cell", {})
                }
        
        elif physics_domain == "astrophysics":
            # Translate astrophysical parameters
            if "mass" in parameters:
                engine_params["stellar_mass"] = parameters["mass"]
            if "distance" in parameters:
                engine_params["distance_scale"] = parameters["distance"]
        
        return engine_params
    
    def _translate_results_from_engine(self,
                                     engine_result: Dict[str, Any],
                                     physics_domain: str,
                                     calculation_type: str) -> Dict[str, Any]:
        """Translate engine results back to tool format."""
        # Basic result structure
        tool_result = {
            "engine_used": True,
            "raw_engine_result": engine_result,
            "calculation_metadata": engine_result.get("metadata", {})
        }
        
        # Extract main results
        if "results" in engine_result:
            results = engine_result["results"]
            
            # Domain-specific result translations
            if physics_domain == "quantum_chemistry":
                if "energy" in results:
                    tool_result["energy"] = results["energy"]
                if "optimized_geometry" in results:
                    tool_result["optimized_geometry"] = results["optimized_geometry"]
                if "molecular_properties" in results:
                    tool_result.update(results["molecular_properties"])
            
            elif physics_domain == "materials_science":
                if "mechanical_properties" in results:
                    tool_result.update(results["mechanical_properties"])
                if "electronic_structure" in results:
                    tool_result.update(results["electronic_structure"])
                if "crystal_structure" in results:
                    tool_result.update(results["crystal_structure"])
            
            elif physics_domain == "astrophysics":
                if "stellar_properties" in results:
                    tool_result.update(results["stellar_properties"])
                if "orbital_parameters" in results:
                    tool_result.update(results["orbital_parameters"])
                if "cosmological_parameters" in results:
                    tool_result.update(results["cosmological_parameters"])
        
        # Performance information
        if "performance" in engine_result:
            tool_result["engine_performance"] = engine_result["performance"]
        
        return tool_result
    
    def get_engine_capabilities(self, physics_domain: str) -> Dict[str, Any]:
        """Get capabilities of available engines for a physics domain."""
        if not self._available_engines:
            return {"engines_available": False, "capabilities": []}
        
        domain_mapping = self._engine_mappings.get(physics_domain, {})
        engine_type = domain_mapping.get("engine_type")
        
        if engine_type in self._available_engines:
            engine_info = self._available_engines[engine_type]
            return {
                "engines_available": True,
                "engine_type": engine_type,
                "capabilities": engine_info["capabilities"],
                "domains": engine_info["domains"],
                "performance": engine_info["performance"]
            }
        
        return {"engines_available": False, "capabilities": []}
    
    def get_available_engines_summary(self) -> Dict[str, Any]:
        """Get summary of all available physics engines."""
        return {
            "adapter_initialized": self._engine_registry is not None,
            "total_engines": len(self._available_engines),
            "available_engines": list(self._available_engines.keys()),
            "supported_domains": list(self._engine_mappings.keys()),
            "engine_mappings": self._engine_mappings
        }


# Singleton instance for global access
_physics_engine_adapter = None

def get_physics_engine_adapter() -> PhysicsEngineAdapter:
    """Get the global physics engine adapter instance."""
    global _physics_engine_adapter
    if _physics_engine_adapter is None:
        _physics_engine_adapter = PhysicsEngineAdapter()
    return _physics_engine_adapter