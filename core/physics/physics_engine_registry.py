"""
Physics Engine Registry

Registry system for managing physics engines separately from the main tool registry.
Provides registration, discovery, and metadata management for physics engines.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from .base_physics_engine import BasePhysicsEngine, PhysicsEngineType, SoftwareInterface
from .physics_engine_factory import PhysicsEngineFactory

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """Status of physics engines in the registry."""
    AVAILABLE = "available"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class EngineRegistryEntry:
    """Registry entry for a physics engine."""
    engine_id: str
    engine_type: PhysicsEngineType
    engine_instance: Optional[BasePhysicsEngine]
    status: EngineStatus
    registration_time: float
    last_used_time: float
    usage_count: int
    configuration: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry entry to dictionary."""
        return {
            'engine_id': self.engine_id,
            'engine_type': self.engine_type.value,
            'status': self.status.value,
            'registration_time': self.registration_time,
            'last_used_time': self.last_used_time,
            'usage_count': self.usage_count,
            'configuration': self.configuration,
            'metadata': self.metadata,
            'dependencies': self.dependencies,
            'performance_metrics': self.performance_metrics,
            'error_count': len(self.error_log),
            'has_instance': self.engine_instance is not None
        }


class PhysicsEngineRegistry:
    """
    Registry for managing physics engines separately from the main framework.
    
    Provides centralized registration, discovery, and lifecycle management
    for physics engines with metadata tracking and dependency management.
    """
    
    def __init__(self, factory: Optional[PhysicsEngineFactory] = None):
        """
        Initialize the physics engine registry.
        
        Args:
            factory: Optional physics engine factory for creating engines
        """
        self.factory = factory or PhysicsEngineFactory()
        
        # Registry storage
        self.engines: Dict[str, EngineRegistryEntry] = {}
        self.engine_types: Dict[PhysicsEngineType, List[str]] = {
            engine_type: [] for engine_type in PhysicsEngineType
        }
        
        # Registry metadata
        self.registry_id = f"physics_registry_{int(time.time())}"
        self.creation_time = time.time()
        self.total_registrations = 0
        self.total_usage_count = 0
        
        # Event callbacks
        self.registration_callbacks: List[Callable] = []
        self.usage_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        
        logger.info(f"Physics engine registry initialized: {self.registry_id}")
    
    def register_engine(self, engine_id: str, engine_type: PhysicsEngineType,
                       configuration: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       dependencies: Optional[List[str]] = None,
                       auto_create: bool = True) -> bool:
        """
        Register a physics engine in the registry.
        
        Args:
            engine_id: Unique identifier for the engine
            engine_type: Type of physics engine
            configuration: Engine configuration
            metadata: Additional metadata
            dependencies: List of dependent engine IDs
            auto_create: Whether to automatically create the engine instance
            
        Returns:
            True if registration was successful
        """
        logger.info(f"Registering physics engine: {engine_id} ({engine_type.value})")
        
        # Check if engine already registered
        if engine_id in self.engines:
            logger.warning(f"Engine {engine_id} already registered")
            return False
        
        # Validate dependencies
        dependencies = dependencies or []
        for dep_id in dependencies:
            if dep_id not in self.engines:
                logger.error(f"Dependency {dep_id} not found for engine {engine_id}")
                return False
        
        try:
            # Create engine instance if requested
            engine_instance = None
            if auto_create:
                engine_instance = self.factory.create_engine(
                    engine_type, engine_id, configuration
                )
            
            # Create registry entry
            entry = EngineRegistryEntry(
                engine_id=engine_id,
                engine_type=engine_type,
                engine_instance=engine_instance,
                status=EngineStatus.ACTIVE if engine_instance else EngineStatus.AVAILABLE,
                registration_time=time.time(),
                last_used_time=time.time(),
                usage_count=0,
                configuration=configuration or {},
                metadata=metadata or {},
                dependencies=dependencies
            )
            
            # Add to registry
            self.engines[engine_id] = entry
            self.engine_types[engine_type].append(engine_id)
            
            # Update dependency graph
            self._update_dependency_graph(engine_id, dependencies)
            
            # Update statistics
            self.total_registrations += 1
            
            # Trigger callbacks
            self._trigger_registration_callbacks(engine_id, engine_type)
            
            logger.info(f"Successfully registered physics engine: {engine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register physics engine {engine_id}: {e}")
            return False
    
    def unregister_engine(self, engine_id: str) -> bool:
        """
        Unregister a physics engine from the registry.
        
        Args:
            engine_id: Unique identifier of the engine to unregister
            
        Returns:
            True if unregistration was successful
        """
        logger.info(f"Unregistering physics engine: {engine_id}")
        
        if engine_id not in self.engines:
            logger.warning(f"Engine {engine_id} not found for unregistration")
            return False
        
        try:
            entry = self.engines[engine_id]
            
            # Check for dependents
            dependents = self.reverse_dependencies.get(engine_id, set())
            if dependents:
                logger.warning(f"Engine {engine_id} has dependents: {dependents}")
                # Could force unregister or fail - here we'll warn but continue
            
            # Cleanup engine instance
            if entry.engine_instance:
                entry.engine_instance.cleanup()
            
            # Remove from registry
            del self.engines[engine_id]
            self.engine_types[entry.engine_type].remove(engine_id)
            
            # Update dependency graph
            self._remove_from_dependency_graph(engine_id)
            
            # Remove from factory if it exists there
            self.factory.remove_engine(engine_id)
            
            logger.info(f"Successfully unregistered physics engine: {engine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister physics engine {engine_id}: {e}")
            return False
    
    def get_engine(self, engine_id: str) -> Optional[BasePhysicsEngine]:
        """
        Get a physics engine instance by ID.
        
        Args:
            engine_id: Unique identifier of the engine
            
        Returns:
            Physics engine instance or None if not found
        """
        if engine_id not in self.engines:
            return None
        
        entry = self.engines[engine_id]
        
        # Create instance if it doesn't exist
        if entry.engine_instance is None and entry.status == EngineStatus.AVAILABLE:
            try:
                entry.engine_instance = self.factory.create_engine(
                    entry.engine_type, engine_id, entry.configuration
                )
                entry.status = EngineStatus.ACTIVE
                logger.info(f"Created engine instance for {engine_id}")
            except Exception as e:
                logger.error(f"Failed to create engine instance for {engine_id}: {e}")
                entry.status = EngineStatus.ERROR
                entry.error_log.append(f"Instance creation failed: {str(e)}")
                return None
        
        # Update usage tracking
        if entry.engine_instance:
            entry.last_used_time = time.time()
            entry.usage_count += 1
            self.total_usage_count += 1
            
            # Update performance metrics
            self._update_performance_metrics(engine_id)
            
            # Trigger usage callbacks
            self._trigger_usage_callbacks(engine_id, entry.engine_type)
        
        return entry.engine_instance
    
    def list_engines(self, engine_type: Optional[PhysicsEngineType] = None,
                    status: Optional[EngineStatus] = None) -> List[Dict[str, Any]]:
        """
        List registered physics engines with optional filtering.
        
        Args:
            engine_type: Optional filter by engine type
            status: Optional filter by engine status
            
        Returns:
            List of engine information dictionaries
        """
        engines_list = []
        
        for engine_id, entry in self.engines.items():
            # Apply filters
            if engine_type and entry.engine_type != engine_type:
                continue
            
            if status and entry.status != status:
                continue
            
            engines_list.append(entry.to_dict())
        
        return engines_list
    
    def get_engines_by_type(self, engine_type: PhysicsEngineType) -> List[str]:
        """
        Get list of engine IDs for a specific engine type.
        
        Args:
            engine_type: Type of physics engine
            
        Returns:
            List of engine IDs
        """
        return self.engine_types.get(engine_type, []).copy()
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine IDs."""
        available = []
        for engine_id, entry in self.engines.items():
            if entry.status in [EngineStatus.AVAILABLE, EngineStatus.ACTIVE]:
                available.append(engine_id)
        return available
    
    def get_engine_dependencies(self, engine_id: str) -> List[str]:
        """
        Get dependencies for a specific engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            List of dependency engine IDs
        """
        if engine_id not in self.engines:
            return []
        
        return self.engines[engine_id].dependencies.copy()
    
    def get_engine_dependents(self, engine_id: str) -> List[str]:
        """
        Get engines that depend on the specified engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            List of dependent engine IDs
        """
        return list(self.reverse_dependencies.get(engine_id, set()))
    
    def validate_dependencies(self, engine_id: str) -> Dict[str, Any]:
        """
        Validate that all dependencies for an engine are satisfied.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Validation report
        """
        validation_report = {
            'valid': True,
            'missing_dependencies': [],
            'circular_dependencies': [],
            'dependency_chain': []
        }
        
        if engine_id not in self.engines:
            validation_report['valid'] = False
            return validation_report
        
        # Check for missing dependencies
        dependencies = self.engines[engine_id].dependencies
        for dep_id in dependencies:
            if dep_id not in self.engines:
                validation_report['missing_dependencies'].append(dep_id)
                validation_report['valid'] = False
            elif self.engines[dep_id].status == EngineStatus.ERROR:
                validation_report['missing_dependencies'].append(f"{dep_id} (error state)")
                validation_report['valid'] = False
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        if has_cycle(engine_id):
            validation_report['circular_dependencies'] = self._find_circular_dependencies(engine_id)
            validation_report['valid'] = False
        
        # Build dependency chain
        validation_report['dependency_chain'] = self._build_dependency_chain(engine_id)
        
        return validation_report
    
    def get_engine_status(self, engine_id: str) -> Optional[EngineStatus]:
        """
        Get the current status of an engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Engine status or None if not found
        """
        if engine_id not in self.engines:
            return None
        
        return self.engines[engine_id].status
    
    def set_engine_status(self, engine_id: str, status: EngineStatus,
                         reason: Optional[str] = None) -> bool:
        """
        Set the status of an engine.
        
        Args:
            engine_id: Engine identifier
            status: New status
            reason: Optional reason for status change
            
        Returns:
            True if status was updated successfully
        """
        if engine_id not in self.engines:
            return False
        
        old_status = self.engines[engine_id].status
        self.engines[engine_id].status = status
        
        if reason:
            self.engines[engine_id].metadata['last_status_change_reason'] = reason
            self.engines[engine_id].metadata['last_status_change_time'] = time.time()
        
        logger.info(f"Engine {engine_id} status changed from {old_status.value} to {status.value}")
        
        return True
    
    def get_engine_configuration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration of an engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Engine configuration or None if not found
        """
        if engine_id not in self.engines:
            return None
        
        return self.engines[engine_id].configuration.copy()
    
    def update_engine_configuration(self, engine_id: str, 
                                   configuration_updates: Dict[str, Any]) -> bool:
        """
        Update the configuration of an engine.
        
        Args:
            engine_id: Engine identifier
            configuration_updates: Configuration updates to apply
            
        Returns:
            True if configuration was updated successfully
        """
        if engine_id not in self.engines:
            return False
        
        try:
            # Update configuration
            self.engines[engine_id].configuration.update(configuration_updates)
            
            # If engine instance exists, update it
            if self.engines[engine_id].engine_instance:
                # Engine would need to support configuration updates
                # This is a placeholder for that functionality
                logger.info(f"Configuration updated for active engine {engine_id}")
            
            logger.info(f"Configuration updated for engine {engine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration for engine {engine_id}: {e}")
            return False
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            'registry_info': {
                'registry_id': self.registry_id,
                'creation_time': self.creation_time,
                'uptime_seconds': time.time() - self.creation_time
            },
            'engine_counts': {
                'total_registered': len(self.engines),
                'total_registrations': self.total_registrations,
                'by_type': {},
                'by_status': {}
            },
            'usage_statistics': {
                'total_usage_count': self.total_usage_count,
                'average_usage_per_engine': self.total_usage_count / max(1, len(self.engines)),
                'most_used_engines': [],
                'least_used_engines': []
            },
            'performance_metrics': {},
            'dependency_analysis': {
                'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
                'engines_with_dependencies': len([e for e in self.engines.values() if e.dependencies]),
                'circular_dependencies': []
            }
        }
        
        # Count by type
        for engine_type in PhysicsEngineType:
            stats['engine_counts']['by_type'][engine_type.value] = len(self.engine_types[engine_type])
        
        # Count by status
        for entry in self.engines.values():
            status = entry.status.value
            stats['engine_counts']['by_status'][status] = stats['engine_counts']['by_status'].get(status, 0) + 1
        
        # Usage statistics
        engine_usage = [(engine_id, entry.usage_count) for engine_id, entry in self.engines.items()]
        engine_usage.sort(key=lambda x: x[1], reverse=True)
        
        stats['usage_statistics']['most_used_engines'] = engine_usage[:5]
        stats['usage_statistics']['least_used_engines'] = engine_usage[-5:]
        
        # Performance metrics
        for engine_id, entry in self.engines.items():
            if entry.performance_metrics:
                stats['performance_metrics'][engine_id] = entry.performance_metrics.copy()
        
        # Check for circular dependencies
        for engine_id in self.engines:
            circular_deps = self._find_circular_dependencies(engine_id)
            if circular_deps:
                stats['dependency_analysis']['circular_dependencies'].append({
                    'engine_id': engine_id,
                    'cycle': circular_deps
                })
        
        return stats
    
    def export_registry(self, include_instances: bool = False) -> Dict[str, Any]:
        """
        Export registry state to a dictionary.
        
        Args:
            include_instances: Whether to include engine instances (not serializable)
            
        Returns:
            Registry state dictionary
        """
        export_data = {
            'registry_id': self.registry_id,
            'creation_time': self.creation_time,
            'total_registrations': self.total_registrations,
            'total_usage_count': self.total_usage_count,
            'engines': {},
            'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
            'performance_history': self.performance_history
        }
        
        # Export engine entries
        for engine_id, entry in self.engines.items():
            engine_data = entry.to_dict()
            
            if not include_instances:
                engine_data.pop('engine_instance', None)
            
            export_data['engines'][engine_id] = engine_data
        
        return export_data
    
    def import_registry(self, import_data: Dict[str, Any], 
                       recreate_instances: bool = False) -> bool:
        """
        Import registry state from a dictionary.
        
        Args:
            import_data: Registry state dictionary
            recreate_instances: Whether to recreate engine instances
            
        Returns:
            True if import was successful
        """
        try:
            logger.info("Importing physics engine registry state")
            
            # Clear current state
            self.cleanup_all_engines()
            
            # Import basic registry info
            self.registry_id = import_data.get('registry_id', self.registry_id)
            self.total_registrations = import_data.get('total_registrations', 0)
            self.total_usage_count = import_data.get('total_usage_count', 0)
            self.performance_history = import_data.get('performance_history', {})
            
            # Import engines
            engines_data = import_data.get('engines', {})
            for engine_id, engine_data in engines_data.items():
                engine_type = PhysicsEngineType(engine_data['engine_type'])
                configuration = engine_data.get('configuration', {})
                metadata = engine_data.get('metadata', {})
                dependencies = engine_data.get('dependencies', [])
                
                # Register engine
                self.register_engine(
                    engine_id=engine_id,
                    engine_type=engine_type,
                    configuration=configuration,
                    metadata=metadata,
                    dependencies=dependencies,
                    auto_create=recreate_instances
                )
                
                # Restore additional state
                entry = self.engines[engine_id]
                entry.status = EngineStatus(engine_data.get('status', 'available'))
                entry.registration_time = engine_data.get('registration_time', time.time())
                entry.last_used_time = engine_data.get('last_used_time', time.time())
                entry.usage_count = engine_data.get('usage_count', 0)
                entry.performance_metrics = engine_data.get('performance_metrics', {})
                entry.error_log = engine_data.get('error_log', [])
            
            # Import dependency graph
            dependency_graph_data = import_data.get('dependency_graph', {})
            self.dependency_graph = {k: set(v) for k, v in dependency_graph_data.items()}
            self._rebuild_reverse_dependencies()
            
            logger.info(f"Successfully imported registry with {len(self.engines)} engines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry state: {e}")
            return False
    
    def add_registration_callback(self, callback: Callable[[str, PhysicsEngineType], None]):
        """Add a callback for engine registration events."""
        self.registration_callbacks.append(callback)
    
    def add_usage_callback(self, callback: Callable[[str, PhysicsEngineType], None]):
        """Add a callback for engine usage events."""
        self.usage_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, str], None]):
        """Add a callback for engine error events."""
        self.error_callbacks.append(callback)
    
    def cleanup_all_engines(self):
        """Cleanup all registered engines."""
        logger.info("Cleaning up all registered physics engines")
        
        for engine_id in list(self.engines.keys()):
            self.unregister_engine(engine_id)
        
        # Clear all data structures
        self.engines.clear()
        for engine_type in PhysicsEngineType:
            self.engine_types[engine_type].clear()
        self.dependency_graph.clear()
        self.reverse_dependencies.clear()
        self.performance_history.clear()
        
        logger.info("All registered physics engines cleaned up")
    
    # Private helper methods
    
    def _update_dependency_graph(self, engine_id: str, dependencies: List[str]):
        """Update the dependency graph with new engine dependencies."""
        self.dependency_graph[engine_id] = set(dependencies)
        
        # Update reverse dependencies
        for dep_id in dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = set()
            self.reverse_dependencies[dep_id].add(engine_id)
    
    def _remove_from_dependency_graph(self, engine_id: str):
        """Remove an engine from the dependency graph."""
        # Remove from dependency graph
        dependencies = self.dependency_graph.pop(engine_id, set())
        
        # Remove from reverse dependencies
        for dep_id in dependencies:
            if dep_id in self.reverse_dependencies:
                self.reverse_dependencies[dep_id].discard(engine_id)
                if not self.reverse_dependencies[dep_id]:
                    del self.reverse_dependencies[dep_id]
        
        # Remove as dependency of others
        if engine_id in self.reverse_dependencies:
            del self.reverse_dependencies[engine_id]
    
    def _rebuild_reverse_dependencies(self):
        """Rebuild reverse dependency mapping."""
        self.reverse_dependencies.clear()
        
        for engine_id, dependencies in self.dependency_graph.items():
            for dep_id in dependencies:
                if dep_id not in self.reverse_dependencies:
                    self.reverse_dependencies[dep_id] = set()
                self.reverse_dependencies[dep_id].add(engine_id)
    
    def _find_circular_dependencies(self, engine_id: str) -> List[str]:
        """Find circular dependencies starting from an engine."""
        visited = set()
        rec_stack = set()
        cycle = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle.extend(path[cycle_start:])
                    cycle.append(neighbor)
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        dfs(engine_id, [])
        return cycle
    
    def _build_dependency_chain(self, engine_id: str) -> List[str]:
        """Build the full dependency chain for an engine."""
        chain = []
        visited = set()
        
        def build_chain(node):
            if node in visited:
                return
            
            visited.add(node)
            dependencies = self.dependency_graph.get(node, set())
            
            for dep_id in dependencies:
                build_chain(dep_id)
                if dep_id not in chain:
                    chain.append(dep_id)
            
            if node not in chain:
                chain.append(node)
        
        build_chain(engine_id)
        return chain
    
    def _update_performance_metrics(self, engine_id: str):
        """Update performance metrics for an engine."""
        if engine_id not in self.engines:
            return
        
        entry = self.engines[engine_id]
        
        if entry.engine_instance:
            # Get current performance metrics from engine
            current_metrics = entry.engine_instance.get_performance_metrics()
            entry.performance_metrics = current_metrics
            
            # Store in history
            if engine_id not in self.performance_history:
                self.performance_history[engine_id] = []
            
            self.performance_history[engine_id].append({
                'timestamp': time.time(),
                'metrics': current_metrics.copy()
            })
            
            # Keep only recent history (last 100 entries)
            if len(self.performance_history[engine_id]) > 100:
                self.performance_history[engine_id] = self.performance_history[engine_id][-100:]
    
    def _trigger_registration_callbacks(self, engine_id: str, engine_type: PhysicsEngineType):
        """Trigger registration event callbacks."""
        for callback in self.registration_callbacks:
            try:
                callback(engine_id, engine_type)
            except Exception as e:
                logger.error(f"Registration callback failed: {e}")
    
    def _trigger_usage_callbacks(self, engine_id: str, engine_type: PhysicsEngineType):
        """Trigger usage event callbacks."""
        for callback in self.usage_callbacks:
            try:
                callback(engine_id, engine_type)
            except Exception as e:
                logger.error(f"Usage callback failed: {e}")
    
    def _trigger_error_callbacks(self, engine_id: str, error_message: str):
        """Trigger error event callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(engine_id, error_message)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_all_engines()
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return (f"PhysicsEngineRegistry(id='{self.registry_id}', "
                f"engines={len(self.engines)}, "
                f"total_usage={self.total_usage_count})")