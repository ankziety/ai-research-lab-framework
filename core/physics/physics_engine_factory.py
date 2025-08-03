"""
Physics Engine Factory

Factory class for creating and managing physics engines with dependency injection
and configuration management.
"""

import logging
from typing import Dict, Any, Optional, Type, List
from enum import Enum

from .base_physics_engine import BasePhysicsEngine, PhysicsEngineType
from .quantum_simulation_engine import QuantumSimulationEngine
from .molecular_dynamics_engine import MolecularDynamicsEngine
from .statistical_physics_engine import StatisticalPhysicsEngine
from .multi_physics_engine import MultiPhysicsEngine
from .numerical_methods import NumericalMethodsEngine

logger = logging.getLogger(__name__)


class PhysicsEngineConfig:
    """Configuration class for physics engines."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize physics engine configuration."""
        self.config = config_dict or {}
        
        # Default configurations for each engine type
        self.default_configs = {
            PhysicsEngineType.QUANTUM_SIMULATION: {
                'default_basis': 'sto-3g',
                'scf_convergence': 1e-8,
                'max_iterations': 100,
                'correlation_method': 'dft',
                'exchange_functional': 'pbe',
                'temperature': 0.0
            },
            PhysicsEngineType.MOLECULAR_DYNAMICS: {
                'default_timestep': 1e-15,
                'default_temperature': 300.0,
                'default_pressure': 1.0,
                'force_field': 'lennard_jones',
                'cutoff_radius': 2.5,
                'thermostat': 'nose_hoover',
                'barostat': 'parrinello_rahman'
            },
            PhysicsEngineType.STATISTICAL_PHYSICS: {
                'default_temperature': 2.0,
                'mc_steps': 100000,
                'equilibration_steps': 10000,
                'sampling_frequency': 10,
                'finite_size_scaling_enabled': True
            },
            PhysicsEngineType.MULTI_PHYSICS: {
                'default_coupling_type': 'weak_coupling',
                'convergence_tolerance': 1e-6,
                'max_coupling_iterations': 100,
                'relaxation_factor': 1.0,
                'adaptive_coupling': True,
                'load_balancing': True
            },
            PhysicsEngineType.NUMERICAL_METHODS: {
                'default_discretization': 'finite_element',
                'default_element_order': 1,
                'convergence_tolerance': 1e-8,
                'max_iterations': 1000,
                'adaptive_refinement': True,
                'multigrid_levels': 3
            }
        }
    
    def get_engine_config(self, engine_type: PhysicsEngineType) -> Dict[str, Any]:
        """Get configuration for a specific engine type."""
        # Start with default configuration
        engine_config = self.default_configs.get(engine_type, {}).copy()
        
        # Override with user-provided configuration
        if engine_type.value in self.config:
            engine_config.update(self.config[engine_type.value])
        
        # Add global configuration settings
        global_config = self.config.get('global', {})
        for key, value in global_config.items():
            if key not in engine_config:
                engine_config[key] = value
        
        return engine_config
    
    def update_config(self, engine_type: PhysicsEngineType, updates: Dict[str, Any]):
        """Update configuration for a specific engine type."""
        if engine_type.value not in self.config:
            self.config[engine_type.value] = {}
        
        self.config[engine_type.value].update(updates)
    
    def set_global_config(self, global_config: Dict[str, Any]):
        """Set global configuration that applies to all engines."""
        self.config['global'] = global_config


class PhysicsEngineFactory:
    """
    Factory for creating and managing physics engines.
    
    Provides dependency injection, configuration management, and engine lifecycle management
    for the physics simulation framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize the physics engine factory.
        
        Args:
            config: Configuration dictionary for physics engines
            cost_manager: Optional cost manager for tracking computational costs
        """
        self.config = PhysicsEngineConfig(config)
        self.cost_manager = cost_manager
        
        # Engine registry
        self.engine_classes: Dict[PhysicsEngineType, Type[BasePhysicsEngine]] = {
            PhysicsEngineType.QUANTUM_SIMULATION: QuantumSimulationEngine,
            PhysicsEngineType.MOLECULAR_DYNAMICS: MolecularDynamicsEngine,
            PhysicsEngineType.STATISTICAL_PHYSICS: StatisticalPhysicsEngine,
            PhysicsEngineType.MULTI_PHYSICS: MultiPhysicsEngine,
            PhysicsEngineType.NUMERICAL_METHODS: NumericalMethodsEngine
        }
        
        # Active engine instances
        self.active_engines: Dict[str, BasePhysicsEngine] = {}
        
        # Engine dependencies and injection
        self.dependencies: Dict[PhysicsEngineType, List[PhysicsEngineType]] = {
            PhysicsEngineType.MULTI_PHYSICS: [
                PhysicsEngineType.QUANTUM_SIMULATION,
                PhysicsEngineType.MOLECULAR_DYNAMICS,
                PhysicsEngineType.STATISTICAL_PHYSICS,
                PhysicsEngineType.NUMERICAL_METHODS
            ]
        }
        
        logger.info("Physics engine factory initialized")
    
    def create_engine(self, engine_type: PhysicsEngineType, 
                     engine_id: Optional[str] = None,
                     custom_config: Optional[Dict[str, Any]] = None) -> BasePhysicsEngine:
        """
        Create a physics engine instance.
        
        Args:
            engine_type: Type of physics engine to create
            engine_id: Optional unique identifier for the engine
            custom_config: Optional custom configuration for this instance
            
        Returns:
            Configured physics engine instance
        """
        if engine_id is None:
            engine_id = f"{engine_type.value}_{len(self.active_engines)}"
        
        logger.info(f"Creating physics engine: {engine_type.value} with ID: {engine_id}")
        
        # Check if engine already exists
        if engine_id in self.active_engines:
            logger.warning(f"Engine {engine_id} already exists, returning existing instance")
            return self.active_engines[engine_id]
        
        # Get engine configuration
        engine_config = self.config.get_engine_config(engine_type)
        
        # Override with custom configuration if provided
        if custom_config:
            engine_config.update(custom_config)
        
        # Get engine class
        engine_class = self.engine_classes.get(engine_type)
        if not engine_class:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        # Create engine instance
        try:
            engine = engine_class(engine_config, self.cost_manager)
            
            # Inject dependencies for multi-physics engine
            if engine_type == PhysicsEngineType.MULTI_PHYSICS:
                self._inject_dependencies(engine)
            
            # Store active engine
            self.active_engines[engine_id] = engine
            
            logger.info(f"Successfully created physics engine: {engine_id}")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create physics engine {engine_id}: {e}")
            raise
    
    def get_engine(self, engine_id: str) -> Optional[BasePhysicsEngine]:
        """
        Get an existing physics engine by ID.
        
        Args:
            engine_id: Unique identifier of the engine
            
        Returns:
            Physics engine instance or None if not found
        """
        return self.active_engines.get(engine_id)
    
    def list_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        List all active physics engines.
        
        Returns:
            Dictionary of engine information
        """
        engine_info = {}
        
        for engine_id, engine in self.active_engines.items():
            engine_info[engine_id] = {
                'engine_type': engine.engine_type.value,
                'version': engine.version,
                'capabilities': engine.capabilities,
                'performance_metrics': engine.get_performance_metrics(),
                'active': True
            }
        
        return engine_info
    
    def remove_engine(self, engine_id: str) -> bool:
        """
        Remove and cleanup a physics engine.
        
        Args:
            engine_id: Unique identifier of the engine to remove
            
        Returns:
            True if engine was removed successfully
        """
        if engine_id not in self.active_engines:
            logger.warning(f"Engine {engine_id} not found for removal")
            return False
        
        try:
            # Cleanup engine resources
            engine = self.active_engines[engine_id]
            engine.cleanup()
            
            # Remove from active engines
            del self.active_engines[engine_id]
            
            logger.info(f"Successfully removed physics engine: {engine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove physics engine {engine_id}: {e}")
            return False
    
    def create_engine_suite(self, suite_config: Optional[Dict[str, Any]] = None) -> Dict[str, BasePhysicsEngine]:
        """
        Create a complete suite of physics engines for comprehensive simulations.
        
        Args:
            suite_config: Optional configuration for the engine suite
            
        Returns:
            Dictionary of all created physics engines
        """
        logger.info("Creating complete physics engine suite")
        
        suite_config = suite_config or {}
        engine_suite = {}
        
        # Create individual engines
        for engine_type in PhysicsEngineType:
            try:
                engine_config = suite_config.get(engine_type.value, {})
                engine_id = f"suite_{engine_type.value}"
                
                engine = self.create_engine(engine_type, engine_id, engine_config)
                engine_suite[engine_type.value] = engine
                
            except Exception as e:
                logger.error(f"Failed to create {engine_type.value} for suite: {e}")
                # Continue creating other engines
        
        logger.info(f"Created physics engine suite with {len(engine_suite)} engines")
        return engine_suite
    
    def get_available_engine_types(self) -> List[PhysicsEngineType]:
        """Get list of available physics engine types."""
        return list(self.engine_classes.keys())
    
    def get_engine_capabilities(self, engine_type: PhysicsEngineType) -> List[str]:
        """Get capabilities of a specific engine type."""
        if engine_type not in self.engine_classes:
            return []
        
        # Create temporary instance to get capabilities
        try:
            temp_config = self.config.get_engine_config(engine_type)
            temp_engine = self.engine_classes[engine_type](temp_config)
            capabilities = temp_engine.capabilities.copy()
            temp_engine.cleanup()
            return capabilities
        except Exception as e:
            logger.error(f"Failed to get capabilities for {engine_type.value}: {e}")
            return []
    
    def validate_engine_compatibility(self, engine_types: List[PhysicsEngineType]) -> Dict[str, Any]:
        """
        Validate compatibility between multiple physics engines.
        
        Args:
            engine_types: List of engine types to check compatibility
            
        Returns:
            Compatibility report
        """
        compatibility_report = {
            'compatible': True,
            'compatibility_matrix': {},
            'recommendations': [],
            'warnings': []
        }
        
        # Check pairwise compatibility
        for i, engine_type_1 in enumerate(engine_types):
            for j, engine_type_2 in enumerate(engine_types[i+1:], i+1):
                compatibility = self._check_engine_compatibility(engine_type_1, engine_type_2)
                
                key = f"{engine_type_1.value}_{engine_type_2.value}"
                compatibility_report['compatibility_matrix'][key] = compatibility
                
                if not compatibility['compatible']:
                    compatibility_report['compatible'] = False
                    compatibility_report['warnings'].append(
                        f"Incompatibility detected between {engine_type_1.value} and {engine_type_2.value}: "
                        f"{compatibility['reason']}"
                    )
        
        # Add general recommendations
        if PhysicsEngineType.MULTI_PHYSICS in engine_types:
            compatibility_report['recommendations'].append(
                "Multi-physics engine can coordinate between different physics domains"
            )
        
        if len(engine_types) > 3:
            compatibility_report['recommendations'].append(
                "Consider using load balancing for multiple active engines"
            )
        
        return compatibility_report
    
    def optimize_engine_configuration(self, engine_type: PhysicsEngineType,
                                    optimization_objective: str,
                                    constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize configuration for a specific engine type.
        
        Args:
            engine_type: Type of engine to optimize
            optimization_objective: Optimization objective
            constraints: Optional optimization constraints
            
        Returns:
            Optimized configuration
        """
        logger.info(f"Optimizing configuration for {engine_type.value}")
        
        # Get current configuration
        current_config = self.config.get_engine_config(engine_type)
        constraints = constraints or {}
        
        # Optimization logic based on engine type and objective
        if optimization_objective == 'minimize_memory_usage':
            optimized_config = self._optimize_for_memory(engine_type, current_config, constraints)
        elif optimization_objective == 'maximize_performance':
            optimized_config = self._optimize_for_performance(engine_type, current_config, constraints)
        elif optimization_objective == 'balance_accuracy_cost':
            optimized_config = self._optimize_for_accuracy_cost(engine_type, current_config, constraints)
        else:
            optimized_config = current_config.copy()
            logger.warning(f"Unknown optimization objective: {optimization_objective}")
        
        return {
            'original_config': current_config,
            'optimized_config': optimized_config,
            'optimization_objective': optimization_objective,
            'improvements': self._calculate_improvements(current_config, optimized_config, engine_type)
        }
    
    def _inject_dependencies(self, multi_physics_engine: MultiPhysicsEngine):
        """Inject dependencies into multi-physics engine."""
        logger.info("Injecting dependencies into multi-physics engine")
        
        # Create or get required engines
        quantum_engine = None
        md_engine = None
        statistical_engine = None
        numerical_engine = None
        
        for engine_id, engine in self.active_engines.items():
            if engine.engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
                quantum_engine = engine
            elif engine.engine_type == PhysicsEngineType.MOLECULAR_DYNAMICS:
                md_engine = engine
            elif engine.engine_type == PhysicsEngineType.STATISTICAL_PHYSICS:
                statistical_engine = engine
            elif engine.engine_type == PhysicsEngineType.NUMERICAL_METHODS:
                numerical_engine = engine
        
        # Create missing engines if needed
        if quantum_engine is None:
            quantum_engine = self.create_engine(PhysicsEngineType.QUANTUM_SIMULATION, "dependency_quantum")
        
        if md_engine is None:
            md_engine = self.create_engine(PhysicsEngineType.MOLECULAR_DYNAMICS, "dependency_md")
        
        if statistical_engine is None:
            statistical_engine = self.create_engine(PhysicsEngineType.STATISTICAL_PHYSICS, "dependency_statistical")
        
        if numerical_engine is None:
            numerical_engine = self.create_engine(PhysicsEngineType.NUMERICAL_METHODS, "dependency_numerical")
        
        # Inject engines
        multi_physics_engine.inject_physics_engines(
            quantum_engine=quantum_engine,
            md_engine=md_engine,
            statistical_engine=statistical_engine,
            numerical_engine=numerical_engine
        )
        
        logger.info("Dependencies successfully injected into multi-physics engine")
    
    def _check_engine_compatibility(self, engine_type_1: PhysicsEngineType, 
                                   engine_type_2: PhysicsEngineType) -> Dict[str, Any]:
        """Check compatibility between two engine types."""
        # Define compatibility rules
        compatibility_rules = {
            (PhysicsEngineType.QUANTUM_SIMULATION, PhysicsEngineType.MOLECULAR_DYNAMICS): {
                'compatible': True,
                'reason': 'QM/MM coupling supported'
            },
            (PhysicsEngineType.MOLECULAR_DYNAMICS, PhysicsEngineType.STATISTICAL_PHYSICS): {
                'compatible': True,
                'reason': 'MD and statistical mechanics are complementary'
            },
            (PhysicsEngineType.MULTI_PHYSICS, PhysicsEngineType.QUANTUM_SIMULATION): {
                'compatible': True,
                'reason': 'Multi-physics can coordinate quantum simulations'
            },
            (PhysicsEngineType.MULTI_PHYSICS, PhysicsEngineType.MOLECULAR_DYNAMICS): {
                'compatible': True,
                'reason': 'Multi-physics can coordinate MD simulations'
            },
            (PhysicsEngineType.NUMERICAL_METHODS, PhysicsEngineType.MULTI_PHYSICS): {
                'compatible': True,
                'reason': 'Numerical methods support multi-physics coupling'
            }
        }
        
        # Check both directions
        key1 = (engine_type_1, engine_type_2)
        key2 = (engine_type_2, engine_type_1)
        
        if key1 in compatibility_rules:
            return compatibility_rules[key1]
        elif key2 in compatibility_rules:
            return compatibility_rules[key2]
        else:
            # Default compatibility
            return {
                'compatible': True,
                'reason': 'No known incompatibilities'
            }
    
    def _optimize_for_memory(self, engine_type: PhysicsEngineType, 
                           current_config: Dict[str, Any], 
                           constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for memory usage."""
        optimized_config = current_config.copy()
        
        if engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
            # Reduce basis set size, use iterative solvers
            optimized_config['default_basis'] = 'sto-3g'
            optimized_config['use_iterative_solver'] = True
            optimized_config['memory_limit_gb'] = constraints.get('memory_limit_gb', 4.0)
        
        elif engine_type == PhysicsEngineType.MOLECULAR_DYNAMICS:
            # Reduce neighbor list frequency, use smaller buffers
            optimized_config['neighbor_list_frequency'] = 20
            optimized_config['buffer_size_reduction'] = 0.5
        
        elif engine_type == PhysicsEngineType.NUMERICAL_METHODS:
            # Use iterative solvers, reduce mesh size
            optimized_config['use_iterative_solver'] = True
            optimized_config['max_mesh_nodes'] = constraints.get('max_mesh_nodes', 100000)
        
        return optimized_config
    
    def _optimize_for_performance(self, engine_type: PhysicsEngineType, 
                                 current_config: Dict[str, Any], 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for performance."""
        optimized_config = current_config.copy()
        
        # Enable parallel processing
        optimized_config['parallel_enabled'] = True
        optimized_config['num_processes'] = constraints.get('max_processes', 4)
        
        if engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
            # Use efficient algorithms and approximations
            optimized_config['use_density_fitting'] = True
            optimized_config['scf_algorithm'] = 'direct_inversion'
        
        elif engine_type == PhysicsEngineType.MOLECULAR_DYNAMICS:
            # Optimize force calculations
            optimized_config['neighbor_list_enabled'] = True
            optimized_config['force_calculation_optimization'] = True
        
        elif engine_type == PhysicsEngineType.STATISTICAL_PHYSICS:
            # Use efficient sampling methods
            optimized_config['use_cluster_algorithms'] = True
            optimized_config['parallel_tempering'] = True
        
        return optimized_config
    
    def _optimize_for_accuracy_cost(self, engine_type: PhysicsEngineType, 
                                   current_config: Dict[str, Any], 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for accuracy-cost balance."""
        optimized_config = current_config.copy()
        
        accuracy_requirement = constraints.get('accuracy_requirement', 'medium')
        cost_budget = constraints.get('cost_budget', 'medium')
        
        if accuracy_requirement == 'high' and cost_budget == 'high':
            # High accuracy, high cost allowed
            optimized_config = self._high_accuracy_config(engine_type, optimized_config)
        elif accuracy_requirement == 'medium' or cost_budget == 'medium':
            # Balanced configuration
            optimized_config = self._balanced_config(engine_type, optimized_config)
        else:
            # Low cost, accept lower accuracy
            optimized_config = self._low_cost_config(engine_type, optimized_config)
        
        return optimized_config
    
    def _high_accuracy_config(self, engine_type: PhysicsEngineType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure for high accuracy."""
        if engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
            config['default_basis'] = 'cc-pvdz'
            config['scf_convergence'] = 1e-10
            config['correlation_method'] = 'ccsd'
        
        elif engine_type == PhysicsEngineType.NUMERICAL_METHODS:
            config['default_element_order'] = 3
            config['convergence_tolerance'] = 1e-10
            config['adaptive_refinement'] = True
        
        return config
    
    def _balanced_config(self, engine_type: PhysicsEngineType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure for balanced accuracy and cost."""
        if engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
            config['default_basis'] = '6-31g'
            config['scf_convergence'] = 1e-8
            config['correlation_method'] = 'dft'
        
        elif engine_type == PhysicsEngineType.NUMERICAL_METHODS:
            config['default_element_order'] = 2
            config['convergence_tolerance'] = 1e-8
        
        return config
    
    def _low_cost_config(self, engine_type: PhysicsEngineType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure for low computational cost."""
        if engine_type == PhysicsEngineType.QUANTUM_SIMULATION:
            config['default_basis'] = 'sto-3g'
            config['scf_convergence'] = 1e-6
            config['max_iterations'] = 50
        
        elif engine_type == PhysicsEngineType.NUMERICAL_METHODS:
            config['default_element_order'] = 1
            config['convergence_tolerance'] = 1e-6
            config['adaptive_refinement'] = False
        
        return config
    
    def _calculate_improvements(self, original_config: Dict[str, Any], 
                               optimized_config: Dict[str, Any], 
                               engine_type: PhysicsEngineType) -> Dict[str, Any]:
        """Calculate estimated improvements from optimization."""
        improvements = {
            'estimated_speedup': 1.0,
            'estimated_memory_reduction': 0.0,
            'estimated_accuracy_change': 0.0,
            'configuration_changes': []
        }
        
        # Track configuration changes
        for key, value in optimized_config.items():
            if key not in original_config or original_config[key] != value:
                improvements['configuration_changes'].append({
                    'parameter': key,
                    'original_value': original_config.get(key, 'not_set'),
                    'optimized_value': value
                })
        
        # Estimate improvements (simplified)
        if 'parallel_enabled' in optimized_config and optimized_config['parallel_enabled']:
            improvements['estimated_speedup'] *= optimized_config.get('num_processes', 1) * 0.8
        
        if 'memory_limit_gb' in optimized_config:
            improvements['estimated_memory_reduction'] = 0.3  # 30% reduction
        
        return improvements
    
    def cleanup_all_engines(self):
        """Cleanup all active physics engines."""
        logger.info("Cleaning up all physics engines")
        
        for engine_id in list(self.active_engines.keys()):
            self.remove_engine(engine_id)
        
        logger.info("All physics engines cleaned up")
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """Get factory statistics and usage information."""
        statistics = {
            'total_engines_created': len(self.active_engines),
            'active_engines_by_type': {},
            'memory_usage_estimate': 0.0,
            'total_computational_cost': 0.0
        }
        
        # Count engines by type
        for engine in self.active_engines.values():
            engine_type = engine.engine_type.value
            if engine_type not in statistics['active_engines_by_type']:
                statistics['active_engines_by_type'][engine_type] = 0
            statistics['active_engines_by_type'][engine_type] += 1
            
            # Estimate resource usage
            performance_metrics = engine.get_performance_metrics()
            statistics['total_computational_cost'] += performance_metrics.get('efficiency_score', 0.0)
        
        return statistics