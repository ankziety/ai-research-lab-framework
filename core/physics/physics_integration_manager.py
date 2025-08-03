"""
Physics Integration Manager - Integration with existing framework components.

This module manages the seamless integration of physics workflow enhancements
with the existing AI Research Lab Framework. It provides dependency injection,
configuration management, and orchestration of physics components without
modifying existing framework code.

Integration Features:
- Non-invasive framework enhancement via dependency injection
- Configuration management for physics components
- Orchestration of physics workflow engines
- Cross-component communication and data flow
- Plugin-style architecture for easy activation/deactivation
- Backward compatibility with existing framework
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json

# Import physics components
from .physics_workflow_engine import PhysicsWorkflowEngine, PhysicsResearchDomain
from .physics_phase_enhancer import PhysicsPhaseEnhancer, PhysicsEnhancementConfig
from .physics_validation_engine import PhysicsValidationEngine, ValidationLevel
from .physics_discovery_engine import PhysicsDiscoveryEngine, DiscoveryType

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for physics enhancements."""
    PASSIVE = "passive"          # Monitor only, no active enhancement
    SELECTIVE = "selective"      # Enhance specific phases only
    COMPREHENSIVE = "comprehensive"  # Full physics enhancement
    EXPERIMENTAL = "experimental"    # Experimental features enabled


class PhysicsComponentStatus(Enum):
    """Status of physics components."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class PhysicsIntegrationConfig:
    """Configuration for physics integration."""
    integration_mode: IntegrationMode = IntegrationMode.COMPREHENSIVE
    enable_workflow_engine: bool = True
    enable_phase_enhancer: bool = True
    enable_validation_engine: bool = True
    enable_discovery_engine: bool = True
    
    # Component-specific configurations
    workflow_config: Dict[str, Any] = None
    enhancer_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    discovery_config: Dict[str, Any] = None
    
    # Integration settings
    auto_activate: bool = True
    compatibility_mode: bool = True
    logging_level: str = "INFO"
    performance_monitoring: bool = True
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.workflow_config is None:
            self.workflow_config = {}
        if self.enhancer_config is None:
            self.enhancer_config = {}
        if self.validation_config is None:
            self.validation_config = {}
        if self.discovery_config is None:
            self.discovery_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'integration_mode': self.integration_mode.value,
            'enable_workflow_engine': self.enable_workflow_engine,
            'enable_phase_enhancer': self.enable_phase_enhancer,
            'enable_validation_engine': self.enable_validation_engine,
            'enable_discovery_engine': self.enable_discovery_engine,
            'workflow_config': self.workflow_config,
            'enhancer_config': self.enhancer_config,
            'validation_config': self.validation_config,
            'discovery_config': self.discovery_config,
            'auto_activate': self.auto_activate,
            'compatibility_mode': self.compatibility_mode,
            'logging_level': self.logging_level,
            'performance_monitoring': self.performance_monitoring
        }


@dataclass
class IntegrationStatus:
    """Status of physics integration."""
    integration_active: bool
    integration_mode: IntegrationMode
    component_status: Dict[str, PhysicsComponentStatus]
    enhanced_phases: List[str]
    performance_metrics: Dict[str, Any]
    error_log: List[Dict[str, Any]]
    last_update: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'integration_active': self.integration_active,
            'integration_mode': self.integration_mode.value,
            'component_status': {k: v.value for k, v in self.component_status.items()},
            'enhanced_phases': self.enhanced_phases,
            'performance_metrics': self.performance_metrics,
            'error_log': self.error_log,
            'last_update': self.last_update
        }


class PhysicsIntegrationManager:
    """
    Manages integration of physics enhancements with existing framework.
    
    Provides non-invasive enhancement capabilities, configuration management,
    and orchestration of physics components while maintaining backward
    compatibility with the existing AI Research Lab Framework.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], PhysicsIntegrationConfig]] = None):
        """
        Initialize physics integration manager.
        
        Args:
            config: Integration configuration (dict or PhysicsIntegrationConfig)
        """
        # Parse configuration
        if isinstance(config, dict):
            self.config = PhysicsIntegrationConfig(**config)
        elif isinstance(config, PhysicsIntegrationConfig):
            self.config = config
        else:
            self.config = PhysicsIntegrationConfig()
        
        # Initialize component tracking
        self.component_status = {
            'workflow_engine': PhysicsComponentStatus.INACTIVE,
            'phase_enhancer': PhysicsComponentStatus.INACTIVE,
            'validation_engine': PhysicsComponentStatus.INACTIVE,
            'discovery_engine': PhysicsComponentStatus.INACTIVE
        }
        
        # Initialize physics components
        self.physics_components = {}
        self.enhanced_frameworks = {}
        self.integration_history = []
        self.performance_metrics = {}
        self.error_log = []
        
        # Setup logging
        self._setup_logging()
        
        # Auto-initialize if enabled
        if self.config.auto_activate:
            self._initialize_physics_components()
        
        logger.info(f"Physics Integration Manager initialized in {self.config.integration_mode.value} mode")
    
    def _setup_logging(self):
        """Setup logging for physics integration."""
        log_level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        
        # Create physics-specific logger
        physics_logger = logging.getLogger('physics_integration')
        physics_logger.setLevel(log_level)
        
        # Add handler if not already present
        if not physics_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            physics_logger.addHandler(handler)
    
    def _initialize_physics_components(self):
        """Initialize physics components based on configuration."""
        
        logger.info("Initializing physics components")
        
        try:
            # Initialize Physics Workflow Engine
            if self.config.enable_workflow_engine:
                self.component_status['workflow_engine'] = PhysicsComponentStatus.INITIALIZING
                self.physics_components['workflow_engine'] = PhysicsWorkflowEngine(
                    self.config.workflow_config
                )
                self.component_status['workflow_engine'] = PhysicsComponentStatus.ACTIVE
                logger.info("Physics Workflow Engine initialized")
            
            # Initialize Physics Phase Enhancer
            if self.config.enable_phase_enhancer:
                self.component_status['phase_enhancer'] = PhysicsComponentStatus.INITIALIZING
                self.physics_components['phase_enhancer'] = PhysicsPhaseEnhancer(
                    self.config.enhancer_config
                )
                self.component_status['phase_enhancer'] = PhysicsComponentStatus.ACTIVE
                logger.info("Physics Phase Enhancer initialized")
            
            # Initialize Physics Validation Engine
            if self.config.enable_validation_engine:
                self.component_status['validation_engine'] = PhysicsComponentStatus.INITIALIZING
                self.physics_components['validation_engine'] = PhysicsValidationEngine(
                    self.config.validation_config
                )
                self.component_status['validation_engine'] = PhysicsComponentStatus.ACTIVE
                logger.info("Physics Validation Engine initialized")
            
            # Initialize Physics Discovery Engine
            if self.config.enable_discovery_engine:
                self.component_status['discovery_engine'] = PhysicsComponentStatus.INITIALIZING
                self.physics_components['discovery_engine'] = PhysicsDiscoveryEngine(
                    self.config.discovery_config
                )
                self.component_status['discovery_engine'] = PhysicsComponentStatus.ACTIVE
                logger.info("Physics Discovery Engine initialized")
            
            logger.info("All enabled physics components initialized successfully")
            
        except Exception as e:
            self._log_error('component_initialization', str(e))
            logger.error(f"Failed to initialize physics components: {e}")
    
    def enhance_framework(self, base_framework: Any) -> Any:
        """
        Enhance an existing AI Research Lab Framework with physics capabilities.
        
        Args:
            base_framework: Existing framework instance to enhance
            
        Returns:
            Enhanced framework with physics capabilities
        """
        logger.info("Enhancing framework with physics capabilities")
        
        try:
            # Create enhanced framework wrapper
            enhanced_framework = PhysicsEnhancedFramework(
                base_framework, self, self.config.integration_mode
            )
            
            # Store reference for tracking
            framework_id = id(base_framework)
            self.enhanced_frameworks[framework_id] = {
                'original_framework': base_framework,
                'enhanced_framework': enhanced_framework,
                'enhancement_timestamp': time.time(),
                'integration_mode': self.config.integration_mode
            }
            
            # Log integration
            self._log_integration_event('framework_enhanced', {
                'framework_id': framework_id,
                'integration_mode': self.config.integration_mode.value,
                'components_active': [
                    name for name, status in self.component_status.items()
                    if status == PhysicsComponentStatus.ACTIVE
                ]
            })
            
            logger.info(f"Framework enhanced successfully with {len(self.get_active_components())} physics components")
            return enhanced_framework
            
        except Exception as e:
            self._log_error('framework_enhancement', str(e))
            logger.error(f"Failed to enhance framework: {e}")
            return base_framework
    
    def enhance_virtual_lab(self, virtual_lab_system: Any) -> Any:
        """
        Enhance a Virtual Lab system with physics capabilities.
        
        Args:
            virtual_lab_system: Existing VirtualLabMeetingSystem instance
            
        Returns:
            Enhanced virtual lab system
        """
        logger.info("Enhancing Virtual Lab system with physics capabilities")
        
        try:
            if 'phase_enhancer' not in self.physics_components:
                logger.warning("Phase enhancer not available - initializing")
                self._initialize_physics_components()
            
            phase_enhancer = self.physics_components.get('phase_enhancer')
            if not phase_enhancer:
                logger.error("Failed to initialize phase enhancer")
                return virtual_lab_system
            
            # Apply physics enhancements
            enhanced_virtual_lab = phase_enhancer.enhance_virtual_lab(virtual_lab_system)
            
            # Log enhancement
            self._log_integration_event('virtual_lab_enhanced', {
                'virtual_lab_id': id(virtual_lab_system),
                'enhanced_phases': [
                    'team_selection', 'literature_review', 'project_specification',
                    'tools_selection', 'execution', 'synthesis'
                ]
            })
            
            logger.info("Virtual Lab system enhanced successfully")
            return enhanced_virtual_lab
            
        except Exception as e:
            self._log_error('virtual_lab_enhancement', str(e))
            logger.error(f"Failed to enhance virtual lab: {e}")
            return virtual_lab_system
    
    def create_physics_workflow(self, research_question: str,
                              domains: List[PhysicsResearchDomain],
                              constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a physics research workflow.
        
        Args:
            research_question: Research question to investigate
            domains: Physics domains relevant to the research
            constraints: Optional research constraints
            
        Returns:
            Workflow ID for tracking
        """
        logger.info(f"Creating physics workflow for {len(domains)} domains")
        
        try:
            workflow_engine = self.physics_components.get('workflow_engine')
            if not workflow_engine:
                raise ValueError("Physics Workflow Engine not available")
            
            workflow_id = workflow_engine.create_physics_workflow(
                research_question, domains, constraints
            )
            
            # Track workflow creation
            self._log_integration_event('workflow_created', {
                'workflow_id': workflow_id,
                'domains': [domain.value for domain in domains],
                'research_question_length': len(research_question)
            })
            
            logger.info(f"Physics workflow {workflow_id} created successfully")
            return workflow_id
            
        except Exception as e:
            self._log_error('workflow_creation', str(e))
            logger.error(f"Failed to create physics workflow: {e}")
            raise
    
    def execute_physics_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a physics research workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            
        Returns:
            Workflow execution results
        """
        logger.info(f"Executing physics workflow {workflow_id}")
        
        try:
            workflow_engine = self.physics_components.get('workflow_engine')
            if not workflow_engine:
                raise ValueError("Physics Workflow Engine not available")
            
            # Execute workflow
            start_time = time.time()
            results = workflow_engine.execute_physics_workflow(workflow_id)
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics('workflow_execution', {
                'execution_time': execution_time,
                'workflow_id': workflow_id,
                'confidence_score': results.confidence_score
            })
            
            # Log execution
            self._log_integration_event('workflow_executed', {
                'workflow_id': workflow_id,
                'execution_time': execution_time,
                'tasks_completed': len(results.tasks_completed),
                'confidence_score': results.confidence_score
            })
            
            logger.info(f"Physics workflow {workflow_id} executed successfully in {execution_time:.2f}s")
            return results.to_dict()
            
        except Exception as e:
            self._log_error('workflow_execution', str(e))
            logger.error(f"Failed to execute physics workflow {workflow_id}: {e}")
            raise
    
    def validate_physics_research(self, research_results: Dict[str, Any],
                                validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """
        Validate physics research results.
        
        Args:
            research_results: Research results to validate
            validation_level: Level of validation rigor
            
        Returns:
            Validation report
        """
        logger.info(f"Validating physics research at {validation_level.value} level")
        
        try:
            validation_engine = self.physics_components.get('validation_engine')
            if not validation_engine:
                raise ValueError("Physics Validation Engine not available")
            
            # Perform validation
            start_time = time.time()
            validation_report = validation_engine.validate_physics_research(
                research_results, validation_level
            )
            validation_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics('validation', {
                'validation_time': validation_time,
                'validation_level': validation_level.value,
                'overall_score': validation_report.overall_score,
                'passed': validation_report.overall_passed
            })
            
            # Log validation
            self._log_integration_event('research_validated', {
                'validation_id': validation_report.validation_id,
                'validation_time': validation_time,
                'overall_score': validation_report.overall_score,
                'passed': validation_report.overall_passed,
                'categories_validated': len(validation_report.category_results)
            })
            
            logger.info(f"Physics research validated: score {validation_report.overall_score:.2f}, {'PASSED' if validation_report.overall_passed else 'FAILED'}")
            return validation_report.to_dict()
            
        except Exception as e:
            self._log_error('validation', str(e))
            logger.error(f"Failed to validate physics research: {e}")
            raise
    
    def discover_physics_phenomena(self, research_results: Dict[str, Any],
                                 discovery_scope: str = 'comprehensive') -> Dict[str, Any]:
        """
        Discover novel physics phenomena from research results.
        
        Args:
            research_results: Research results to analyze
            discovery_scope: Scope of discovery analysis
            
        Returns:
            Discovery report
        """
        logger.info(f"Analyzing physics discoveries with {discovery_scope} scope")
        
        try:
            discovery_engine = self.physics_components.get('discovery_engine')
            if not discovery_engine:
                raise ValueError("Physics Discovery Engine not available")
            
            # Perform discovery analysis
            start_time = time.time()
            discovery_report = discovery_engine.discover_physics_phenomena(
                research_results, discovery_scope
            )
            discovery_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics('discovery', {
                'discovery_time': discovery_time,
                'discoveries_found': len(discovery_report.discoveries),
                'breakthroughs_found': len(discovery_report.breakthrough_discoveries),
                'discovery_scope': discovery_scope
            })
            
            # Log discovery
            self._log_integration_event('discoveries_analyzed', {
                'report_id': discovery_report.report_id,
                'discovery_time': discovery_time,
                'total_discoveries': len(discovery_report.discoveries),
                'breakthrough_discoveries': len(discovery_report.breakthrough_discoveries),
                'cross_domain_connections': len(discovery_report.cross_domain_connections)
            })
            
            logger.info(f"Physics discovery analysis completed: {len(discovery_report.discoveries)} discoveries, {len(discovery_report.breakthrough_discoveries)} breakthroughs")
            return discovery_report.to_dict()
            
        except Exception as e:
            self._log_error('discovery', str(e))
            logger.error(f"Failed to analyze physics discoveries: {e}")
            raise
    
    def get_integration_status(self) -> IntegrationStatus:
        """Get current integration status."""
        
        # Count enhanced phases
        enhanced_phases = []
        if self.config.enable_phase_enhancer and self.component_status['phase_enhancer'] == PhysicsComponentStatus.ACTIVE:
            enhanced_phases = [
                'team_selection', 'literature_review', 'project_specification',
                'tools_selection', 'execution', 'synthesis'
            ]
        
        return IntegrationStatus(
            integration_active=any(
                status == PhysicsComponentStatus.ACTIVE 
                for status in self.component_status.values()
            ),
            integration_mode=self.config.integration_mode,
            component_status=self.component_status.copy(),
            enhanced_phases=enhanced_phases,
            performance_metrics=self.performance_metrics.copy(),
            error_log=self.error_log[-10:],  # Last 10 errors
            last_update=time.time()
        )
    
    def get_active_components(self) -> List[str]:
        """Get list of active physics components."""
        return [
            name for name, status in self.component_status.items()
            if status == PhysicsComponentStatus.ACTIVE
        ]
    
    def reconfigure_integration(self, new_config: Union[Dict[str, Any], PhysicsIntegrationConfig]):
        """
        Reconfigure physics integration.
        
        Args:
            new_config: New configuration to apply
        """
        logger.info("Reconfiguring physics integration")
        
        try:
            # Parse new configuration
            if isinstance(new_config, dict):
                self.config = PhysicsIntegrationConfig(**new_config)
            elif isinstance(new_config, PhysicsIntegrationConfig):
                self.config = new_config
            else:
                raise ValueError("Invalid configuration type")
            
            # Reinitialize components if needed
            if self.config.auto_activate:
                self._shutdown_components()
                self._initialize_physics_components()
            
            # Log reconfiguration
            self._log_integration_event('reconfigured', {
                'integration_mode': self.config.integration_mode.value,
                'enabled_components': [
                    name for name in ['workflow_engine', 'phase_enhancer', 'validation_engine', 'discovery_engine']
                    if getattr(self.config, f'enable_{name}')
                ]
            })
            
            logger.info("Physics integration reconfigured successfully")
            
        except Exception as e:
            self._log_error('reconfiguration', str(e))
            logger.error(f"Failed to reconfigure integration: {e}")
            raise
    
    def shutdown_integration(self):
        """Shutdown physics integration and cleanup resources."""
        logger.info("Shutting down physics integration")
        
        try:
            # Shutdown all components
            self._shutdown_components()
            
            # Clear references
            self.physics_components.clear()
            self.enhanced_frameworks.clear()
            
            # Log shutdown
            self._log_integration_event('shutdown', {
                'shutdown_time': time.time(),
                'enhanced_frameworks_count': len(self.enhanced_frameworks)
            })
            
            logger.info("Physics integration shutdown completed")
            
        except Exception as e:
            self._log_error('shutdown', str(e))
            logger.error(f"Error during integration shutdown: {e}")
    
    def _shutdown_components(self):
        """Shutdown all physics components."""
        for component_name in self.component_status.keys():
            if self.component_status[component_name] == PhysicsComponentStatus.ACTIVE:
                self.component_status[component_name] = PhysicsComponentStatus.INACTIVE
                # Could add cleanup code for individual components here
        
        self.physics_components.clear()
    
    def _log_integration_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log integration event."""
        event = {
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': time.time(),
            'integration_mode': self.config.integration_mode.value
        }
        
        self.integration_history.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.integration_history) > 1000:
            self.integration_history = self.integration_history[-1000:]
    
    def _log_error(self, error_type: str, error_message: str):
        """Log integration error."""
        error = {
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': time.time(),
            'integration_mode': self.config.integration_mode.value
        }
        
        self.error_log.append(error)
        
        # Keep only recent errors (last 100)
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
    
    def _update_performance_metrics(self, metric_type: str, metric_data: Dict[str, Any]):
        """Update performance metrics."""
        if metric_type not in self.performance_metrics:
            self.performance_metrics[metric_type] = {
                'count': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'last_update': 0.0
            }
        
        metrics = self.performance_metrics[metric_type]
        metrics['count'] += 1
        
        if 'execution_time' in metric_data:
            metrics['total_time'] += metric_data['execution_time']
            metrics['average_time'] = metrics['total_time'] / metrics['count']
        elif 'validation_time' in metric_data:
            metrics['total_time'] += metric_data['validation_time']
            metrics['average_time'] = metrics['total_time'] / metrics['count']
        elif 'discovery_time' in metric_data:
            metrics['total_time'] += metric_data['discovery_time']
            metrics['average_time'] = metrics['total_time'] / metrics['count']
        
        metrics['last_update'] = time.time()
        metrics['last_data'] = metric_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for physics integration."""
        summary = {
            'integration_active': any(
                status == PhysicsComponentStatus.ACTIVE 
                for status in self.component_status.values()
            ),
            'total_events': len(self.integration_history),
            'total_errors': len(self.error_log),
            'enhanced_frameworks': len(self.enhanced_frameworks),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        # Calculate overall performance metrics
        if self.performance_metrics:
            total_operations = sum(metrics['count'] for metrics in self.performance_metrics.values())
            total_time = sum(metrics['total_time'] for metrics in self.performance_metrics.values())
            
            summary['total_operations'] = total_operations
            summary['total_processing_time'] = total_time
            summary['average_operation_time'] = total_time / max(1, total_operations)
        
        return summary
    
    def export_integration_report(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Export comprehensive integration report.
        
        Args:
            include_history: Whether to include full integration history
            
        Returns:
            Comprehensive integration report
        """
        report = {
            'integration_config': self.config.to_dict(),
            'integration_status': self.get_integration_status().to_dict(),
            'performance_summary': self.get_performance_summary(),
            'component_capabilities': self._get_component_capabilities(),
            'export_timestamp': time.time()
        }
        
        if include_history:
            report['integration_history'] = self.integration_history
            report['error_log'] = self.error_log
        
        return report
    
    def _get_component_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of active physics components."""
        capabilities = {}
        
        # Workflow Engine capabilities
        if 'workflow_engine' in self.physics_components:
            workflow_engine = self.physics_components['workflow_engine']
            capabilities['workflow_engine'] = workflow_engine.list_physics_capabilities()
        
        # Phase Enhancer capabilities
        if 'phase_enhancer' in self.physics_components:
            phase_enhancer = self.physics_components['phase_enhancer']
            capabilities['phase_enhancer'] = phase_enhancer.get_enhancement_statistics()
        
        # Validation Engine capabilities
        if 'validation_engine' in self.physics_components:
            validation_engine = self.physics_components['validation_engine']
            capabilities['validation_engine'] = validation_engine.get_validation_statistics()
        
        # Discovery Engine capabilities
        if 'discovery_engine' in self.physics_components:
            discovery_engine = self.physics_components['discovery_engine']
            capabilities['discovery_engine'] = discovery_engine.get_discovery_statistics()
        
        return capabilities


class PhysicsEnhancedFramework:
    """
    Physics-enhanced wrapper for existing framework.
    
    Provides physics capabilities while preserving original functionality.
    """
    
    def __init__(self, original_framework: Any, 
                 integration_manager: PhysicsIntegrationManager,
                 integration_mode: IntegrationMode):
        """
        Initialize physics-enhanced framework wrapper.
        
        Args:
            original_framework: Original framework instance
            integration_manager: Physics integration manager
            integration_mode: Integration mode for enhancements
        """
        self.original_framework = original_framework
        self.integration_manager = integration_manager
        self.integration_mode = integration_mode
        self.enhancement_active = True
        
        logger.info(f"Physics-enhanced framework created in {integration_mode.value} mode")
    
    def create_physics_workflow(self, research_question: str,
                              domains: List[PhysicsResearchDomain],
                              constraints: Optional[Dict[str, Any]] = None) -> str:
        """Create physics research workflow."""
        return self.integration_manager.create_physics_workflow(
            research_question, domains, constraints
        )
    
    def execute_physics_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute physics research workflow."""
        return self.integration_manager.execute_physics_workflow(workflow_id)
    
    def validate_physics_research(self, research_results: Dict[str, Any],
                                validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Validate physics research results."""
        return self.integration_manager.validate_physics_research(
            research_results, validation_level
        )
    
    def discover_physics_phenomena(self, research_results: Dict[str, Any],
                                 discovery_scope: str = 'comprehensive') -> Dict[str, Any]:
        """Discover physics phenomena."""
        return self.integration_manager.discover_physics_phenomena(
            research_results, discovery_scope
        )
    
    def get_physics_capabilities(self) -> Dict[str, Any]:
        """Get available physics capabilities."""
        return self.integration_manager._get_component_capabilities()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get physics integration status."""
        return self.integration_manager.get_integration_status().to_dict()
    
    def disable_physics_enhancements(self):
        """Temporarily disable physics enhancements."""
        self.enhancement_active = False
        logger.info("Physics enhancements disabled")
    
    def enable_physics_enhancements(self):
        """Re-enable physics enhancements."""
        self.enhancement_active = True
        logger.info("Physics enhancements enabled")
    
    def __getattr__(self, name):
        """Delegate all other attributes to original framework."""
        return getattr(self.original_framework, name)


# Convenience functions for easy integration

def create_physics_enhanced_framework(base_framework: Any, 
                                    config: Optional[Dict[str, Any]] = None) -> PhysicsEnhancedFramework:
    """
    Create a physics-enhanced framework from an existing framework.
    
    Args:
        base_framework: Existing framework instance
        config: Optional physics configuration
        
    Returns:
        Physics-enhanced framework
    """
    integration_manager = PhysicsIntegrationManager(config)
    return integration_manager.enhance_framework(base_framework)


def create_physics_integration_manager(config: Optional[Dict[str, Any]] = None) -> PhysicsIntegrationManager:
    """
    Create a physics integration manager.
    
    Args:
        config: Optional configuration
        
    Returns:
        Physics integration manager instance
    """
    return PhysicsIntegrationManager(config)


def apply_physics_enhancements(virtual_lab_system: Any,
                              config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Apply physics enhancements to a Virtual Lab system.
    
    Args:
        virtual_lab_system: Existing Virtual Lab system
        config: Optional physics configuration
        
    Returns:
        Enhanced Virtual Lab system
    """
    integration_manager = PhysicsIntegrationManager(config)
    return integration_manager.enhance_virtual_lab(virtual_lab_system)