"""
Physics Workflow Decorators - Decorator implementations for physics enhancements.

This module provides a comprehensive set of decorators for enhancing existing
research workflow methods with physics-specific capabilities. These decorators
can be applied to any research phase method to add physics functionality
without modifying the original code.

Decorator Categories:
- Phase Enhancement Decorators: Enhance specific research phases
- Validation Decorators: Add physics validation to methods
- Discovery Decorators: Add physics discovery capabilities
- Performance Monitoring Decorators: Track physics enhancement performance
- Configuration Decorators: Apply physics configurations dynamically
"""

import logging
import time
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

# Import physics components for decorator functionality
from .physics_workflow_engine import PhysicsWorkflowEngine, PhysicsResearchDomain
from .physics_phase_enhancer import PhysicsPhaseEnhancer, PhysicsEnhancementConfig
from .physics_validation_engine import PhysicsValidationEngine, ValidationLevel
from .physics_discovery_engine import PhysicsDiscoveryEngine, DiscoveryType

logger = logging.getLogger(__name__)


class DecoratorMode(Enum):
    """Modes for physics decorators."""
    PASSIVE = "passive"      # Monitor and log only
    ENHANCE = "enhance"      # Enhance with physics capabilities
    VALIDATE = "validate"    # Add validation steps
    DISCOVER = "discover"    # Add discovery analysis
    COMPREHENSIVE = "comprehensive"  # Full physics enhancement


@dataclass
class DecoratorConfig:
    """Configuration for physics decorators."""
    mode: DecoratorMode = DecoratorMode.ENHANCE
    enable_validation: bool = True
    enable_discovery: bool = True
    enable_performance_tracking: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    discovery_scope: str = "comprehensive"
    
    # Component configurations
    physics_config: Dict[str, Any] = None
    enhancer_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    discovery_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.physics_config is None:
            self.physics_config = {}
        if self.enhancer_config is None:
            self.enhancer_config = {}
        if self.validation_config is None:
            self.validation_config = {}
        if self.discovery_config is None:
            self.discovery_config = {}


# Global decorator configuration (can be overridden per decorator)
_global_decorator_config = DecoratorConfig()


def configure_physics_decorators(config: Union[Dict[str, Any], DecoratorConfig]):
    """
    Configure global physics decorator settings.
    
    Args:
        config: Decorator configuration (dict or DecoratorConfig)
    """
    global _global_decorator_config
    
    if isinstance(config, dict):
        _global_decorator_config = DecoratorConfig(**config)
    elif isinstance(config, DecoratorConfig):
        _global_decorator_config = config
    else:
        raise ValueError("Invalid configuration type")
    
    logger.info(f"Physics decorators configured in {_global_decorator_config.mode.value} mode")


def physics_enhanced_phase(phase_name: str = None, 
                          config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance any research phase with physics capabilities.
    
    Args:
        phase_name: Name of the research phase being enhanced
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided config or global config
            effective_config = config or _global_decorator_config
            
            # Initialize phase enhancer if needed
            if not hasattr(wrapper, '_phase_enhancer'):
                wrapper._phase_enhancer = PhysicsPhaseEnhancer(effective_config.enhancer_config)
            
            # Determine phase name
            actual_phase_name = phase_name or func.__name__
            
            # Log enhancement start
            logger.info(f"Enhancing {actual_phase_name} phase with physics capabilities")
            
            if effective_config.mode == DecoratorMode.PASSIVE:
                # Passive mode: just call original and log
                result = func(*args, **kwargs)
                logger.info(f"Phase {actual_phase_name} completed (passive monitoring)")
                return result
            
            elif effective_config.mode in [DecoratorMode.ENHANCE, DecoratorMode.COMPREHENSIVE]:
                # Enhancement mode: apply physics enhancements
                
                # Apply phase-specific enhancement
                if actual_phase_name == 'team_selection' or 'team' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_team_selection(func)
                elif actual_phase_name == 'literature_review' or 'literature' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_literature_review(func)
                elif actual_phase_name == 'project_specification' or 'specification' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_project_specification(func)
                elif actual_phase_name == 'tools_selection' or 'tools' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_tools_selection(func)
                elif actual_phase_name == 'execution' or 'execute' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_execution(func)
                elif actual_phase_name == 'synthesis' or 'synthesis' in actual_phase_name:
                    enhanced_func = wrapper._phase_enhancer.enhance_synthesis(func)
                else:
                    # Generic enhancement for unknown phases
                    enhanced_func = func
                
                # Execute enhanced function
                result = enhanced_func(*args, **kwargs)
                
                # Add validation if enabled
                if effective_config.enable_validation and effective_config.mode == DecoratorMode.COMPREHENSIVE:
                    result = _add_validation_to_result(result, effective_config)
                
                # Add discovery analysis if enabled
                if effective_config.enable_discovery and effective_config.mode == DecoratorMode.COMPREHENSIVE:
                    result = _add_discovery_to_result(result, effective_config)
                
                logger.info(f"Phase {actual_phase_name} enhanced with physics capabilities")
                return result
            
            else:
                # Fallback to original function
                return func(*args, **kwargs)
        
        # Add metadata to wrapper
        wrapper._physics_enhanced = True
        wrapper._phase_name = phase_name or func.__name__
        wrapper._enhancement_config = config or _global_decorator_config
        
        return wrapper
    
    return decorator


def physics_agent_selection(domains: Optional[List[PhysicsResearchDomain]] = None,
                           config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance agent selection with physics-specific agents.
    
    Args:
        domains: Specific physics domains to focus on
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info("Enhancing agent selection with physics specialists")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize phase enhancer
                if not hasattr(wrapper, '_phase_enhancer'):
                    wrapper._phase_enhancer = PhysicsPhaseEnhancer(effective_config.enhancer_config)
                
                # Extract research context
                research_question = kwargs.get('research_question', '')
                constraints = kwargs.get('constraints', {})
                session_id = kwargs.get('session_id', 'unknown')
                
                # Analyze physics requirements
                physics_analysis = wrapper._phase_enhancer._analyze_physics_requirements(research_question)
                
                # Add physics agents if physics is involved
                if physics_analysis['requires_physics']:
                    physics_enhancement = wrapper._phase_enhancer._enhance_team_with_physics_agents(
                        physics_analysis, constraints, session_id
                    )
                    
                    # Merge physics agents with existing result
                    if isinstance(result, dict):
                        result['physics_agents'] = physics_enhancement['recommended_agents']
                        result['physics_analysis'] = physics_analysis
                        result['enhanced_with_physics_agents'] = True
                    
                    logger.info(f"Added {len(physics_enhancement['recommended_agents'])} physics agents")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'agent_selection'
        wrapper._target_domains = domains
        
        return wrapper
    
    return decorator


def physics_literature_analysis(focus_areas: Optional[List[str]] = None,
                               config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance literature analysis with physics-specific focus.
    
    Args:
        focus_areas: Specific physics focus areas
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info("Enhancing literature analysis with physics focus")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize phase enhancer
                if not hasattr(wrapper, '_phase_enhancer'):
                    wrapper._phase_enhancer = PhysicsPhaseEnhancer(effective_config.enhancer_config)
                
                # Extract research context
                research_question = kwargs.get('research_question', '')
                
                # Perform physics-specific literature analysis
                physics_literature_analysis = wrapper._phase_enhancer._analyze_physics_literature(
                    research_question, result
                )
                
                # Add physics enhancements to result
                if isinstance(result, dict):
                    result['physics_literature_analysis'] = physics_literature_analysis
                    result['enhanced_with_physics_literature'] = True
                
                logger.info("Literature analysis enhanced with physics-specific insights")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'literature_analysis'
        wrapper._focus_areas = focus_areas
        
        return wrapper
    
    return decorator


def physics_mathematical_modeling(complexity_level: str = "high",
                                 config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance mathematical modeling with physics formalism.
    
    Args:
        complexity_level: Level of mathematical complexity (low, medium, high, extreme)
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info(f"Enhancing mathematical modeling with {complexity_level} physics formalism")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize phase enhancer
                if not hasattr(wrapper, '_phase_enhancer'):
                    wrapper._phase_enhancer = PhysicsPhaseEnhancer(effective_config.enhancer_config)
                
                # Extract research context
                research_question = kwargs.get('research_question', '')
                constraints = kwargs.get('constraints', {})
                
                # Develop mathematical models with specified complexity
                mathematical_models = wrapper._phase_enhancer._develop_mathematical_models(
                    research_question, constraints
                )
                
                # Adjust complexity based on decorator parameter
                mathematical_models['requirements']['mathematical_rigor'] = complexity_level
                
                # Add to result
                if isinstance(result, dict):
                    result['physics_mathematical_models'] = mathematical_models
                    result['enhanced_with_physics_math'] = True
                
                logger.info(f"Mathematical modeling enhanced with {complexity_level} complexity physics")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'mathematical_modeling'
        wrapper._complexity_level = complexity_level
        
        return wrapper
    
    return decorator


def physics_simulation_execution(simulation_types: Optional[List[str]] = None,
                                config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance execution with physics simulations.
    
    Args:
        simulation_types: Specific simulation types to include
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info("Enhancing execution with physics simulations")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize phase enhancer
                if not hasattr(wrapper, '_phase_enhancer'):
                    wrapper._phase_enhancer = PhysicsPhaseEnhancer(effective_config.enhancer_config)
                
                # Extract research context
                research_question = kwargs.get('research_question', '')
                constraints = kwargs.get('constraints', {})
                session_id = kwargs.get('session_id', 'unknown')
                
                # Execute physics simulations
                physics_execution_results = wrapper._phase_enhancer._execute_physics_simulations(
                    research_question, constraints, session_id
                )
                
                # Filter by simulation types if specified
                if simulation_types:
                    filtered_results = {}
                    for sim_type in simulation_types:
                        if sim_type in physics_execution_results.get('simulation_results', {}):
                            filtered_results[sim_type] = physics_execution_results['simulation_results'][sim_type]
                    physics_execution_results['simulation_results'] = filtered_results
                
                # Add to result
                if isinstance(result, dict):
                    result['physics_simulations'] = physics_execution_results
                    result['enhanced_with_physics_simulations'] = True
                
                logger.info("Execution enhanced with physics simulations")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'simulation_execution'
        wrapper._simulation_types = simulation_types
        
        return wrapper
    
    return decorator


def physics_law_discovery(discovery_types: Optional[List[DiscoveryType]] = None,
                         config: Optional[DecoratorConfig] = None):
    """
    Decorator to enhance synthesis with physics law discovery.
    
    Args:
        discovery_types: Specific types of discoveries to focus on
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info("Enhancing synthesis with physics law discovery")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize discovery engine
                if not hasattr(wrapper, '_discovery_engine'):
                    wrapper._discovery_engine = PhysicsDiscoveryEngine(effective_config.discovery_config)
                
                # Perform discovery analysis
                discovery_report = wrapper._discovery_engine.discover_physics_phenomena(
                    result, effective_config.discovery_scope
                )
                
                # Filter by discovery types if specified
                if discovery_types:
                    filtered_discoveries = [
                        d for d in discovery_report.discoveries
                        if d.discovery_type in discovery_types
                    ]
                    discovery_report.discoveries = filtered_discoveries
                
                # Add to result
                if isinstance(result, dict):
                    result['physics_discoveries'] = discovery_report.to_dict()
                    result['enhanced_with_physics_discovery'] = True
                
                logger.info(f"Synthesis enhanced with {len(discovery_report.discoveries)} physics discoveries")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'law_discovery'
        wrapper._discovery_types = discovery_types
        
        return wrapper
    
    return decorator


def physics_validation(validation_level: ValidationLevel = ValidationLevel.STANDARD,
                      config: Optional[DecoratorConfig] = None):
    """
    Decorator to add physics validation to any method.
    
    Args:
        validation_level: Level of validation rigor
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info(f"Adding physics validation at {validation_level.value} level")
            
            # Call original function
            result = func(*args, **kwargs)
            
            if effective_config.mode != DecoratorMode.PASSIVE:
                # Initialize validation engine
                if not hasattr(wrapper, '_validation_engine'):
                    wrapper._validation_engine = PhysicsValidationEngine(effective_config.validation_config)
                
                # Validate result
                validation_report = wrapper._validation_engine.validate_physics_research(
                    result, validation_level
                )
                
                # Add validation to result
                if isinstance(result, dict):
                    result['physics_validation'] = validation_report.to_dict()
                    result['validation_passed'] = validation_report.overall_passed
                    result['validation_score'] = validation_report.overall_score
                
                logger.info(f"Physics validation completed: score {validation_report.overall_score:.2f}, {'PASSED' if validation_report.overall_passed else 'FAILED'}")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'validation'
        wrapper._validation_level = validation_level
        
        return wrapper
    
    return decorator


def physics_performance_monitor(config: Optional[DecoratorConfig] = None):
    """
    Decorator to monitor performance of physics-enhanced methods.
    
    Args:
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            if not effective_config.enable_performance_tracking:
                return func(*args, **kwargs)
            
            # Initialize performance tracking
            if not hasattr(wrapper, '_performance_metrics'):
                wrapper._performance_metrics = {
                    'call_count': 0,
                    'total_time': 0.0,
                    'average_time': 0.0,
                    'last_call_time': 0.0,
                    'physics_enhancement_time': 0.0
                }
            
            # Track execution time
            start_time = time.time()
            
            logger.debug(f"Monitoring performance for {func.__name__}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Update performance metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            metrics = wrapper._performance_metrics
            metrics['call_count'] += 1
            metrics['total_time'] += execution_time
            metrics['average_time'] = metrics['total_time'] / metrics['call_count']
            metrics['last_call_time'] = execution_time
            
            # Track physics enhancement overhead if result contains physics data
            if isinstance(result, dict) and any(key.startswith('physics_') for key in result.keys()):
                # Estimate physics enhancement time (rough approximation)
                physics_keys = [key for key in result.keys() if key.startswith('physics_')]
                physics_overhead = len(physics_keys) * 0.1  # Rough estimate
                metrics['physics_enhancement_time'] += physics_overhead
            
            # Add performance data to result if it's a dictionary
            if isinstance(result, dict) and effective_config.enable_performance_tracking:
                result['_performance_metrics'] = {
                    'execution_time': execution_time,
                    'call_count': metrics['call_count'],
                    'average_time': metrics['average_time']
                }
            
            logger.debug(f"Performance monitoring: {func.__name__} executed in {execution_time:.3f}s")
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'performance_monitor'
        
        return wrapper
    
    return decorator


def physics_workflow_orchestrator(workflow_config: Optional[Dict[str, Any]] = None,
                                config: Optional[DecoratorConfig] = None):
    """
    Decorator to orchestrate complete physics workflows.
    
    Args:
        workflow_config: Configuration for workflow orchestration
        config: Optional configuration for this decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_config = config or _global_decorator_config
            
            logger.info("Orchestrating complete physics workflow")
            
            # Initialize workflow engine
            if not hasattr(wrapper, '_workflow_engine'):
                wrapper._workflow_engine = PhysicsWorkflowEngine(
                    workflow_config or effective_config.physics_config
                )
            
            # Extract research context
            research_question = kwargs.get('research_question', '')
            domains = kwargs.get('domains', [])
            constraints = kwargs.get('constraints', {})
            
            if effective_config.mode == DecoratorMode.COMPREHENSIVE:
                # Create and execute complete physics workflow
                try:
                    workflow_id = wrapper._workflow_engine.create_physics_workflow(
                        research_question, domains, constraints
                    )
                    
                    workflow_results = wrapper._workflow_engine.execute_physics_workflow(workflow_id)
                    
                    # Call original function
                    result = func(*args, **kwargs)
                    
                    # Merge workflow results with original result
                    if isinstance(result, dict):
                        result['physics_workflow'] = workflow_results.to_dict()
                        result['workflow_id'] = workflow_id
                        result['orchestrated_with_physics'] = True
                    
                    logger.info(f"Physics workflow {workflow_id} orchestrated successfully")
                    
                except Exception as e:
                    logger.error(f"Physics workflow orchestration failed: {e}")
                    # Fallback to original function only
                    result = func(*args, **kwargs)
            else:
                # Just call original function
                result = func(*args, **kwargs)
            
            return result
        
        wrapper._physics_enhanced = True
        wrapper._enhancement_type = 'workflow_orchestrator'
        wrapper._workflow_config = workflow_config
        
        return wrapper
    
    return decorator


# Helper functions for decorator implementation

def _add_validation_to_result(result: Any, config: DecoratorConfig) -> Any:
    """Add validation to result if it's a dictionary."""
    if not isinstance(result, dict):
        return result
    
    try:
        validation_engine = PhysicsValidationEngine(config.validation_config)
        validation_report = validation_engine.validate_physics_research(
            result, config.validation_level
        )
        
        result['physics_validation'] = validation_report.to_dict()
        result['validation_passed'] = validation_report.overall_passed
        
    except Exception as e:
        logger.warning(f"Failed to add validation to result: {e}")
    
    return result


def _add_discovery_to_result(result: Any, config: DecoratorConfig) -> Any:
    """Add discovery analysis to result if it's a dictionary."""
    if not isinstance(result, dict):
        return result
    
    try:
        discovery_engine = PhysicsDiscoveryEngine(config.discovery_config)
        discovery_report = discovery_engine.discover_physics_phenomena(
            result, config.discovery_scope
        )
        
        result['physics_discoveries'] = discovery_report.to_dict()
        
    except Exception as e:
        logger.warning(f"Failed to add discovery analysis to result: {e}")
    
    return result


# Utility functions for decorator management

def get_physics_enhanced_methods(obj: Any) -> List[str]:
    """
    Get list of physics-enhanced methods in an object.
    
    Args:
        obj: Object to inspect
        
    Returns:
        List of physics-enhanced method names
    """
    enhanced_methods = []
    
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if callable(attr) and hasattr(attr, '_physics_enhanced'):
            enhanced_methods.append(attr_name)
    
    return enhanced_methods


def get_enhancement_info(method: Callable) -> Dict[str, Any]:
    """
    Get enhancement information for a physics-enhanced method.
    
    Args:
        method: Method to inspect
        
    Returns:
        Enhancement information dictionary
    """
    if not hasattr(method, '_physics_enhanced'):
        return {'enhanced': False}
    
    info = {
        'enhanced': True,
        'enhancement_type': getattr(method, '_enhancement_type', 'unknown'),
        'phase_name': getattr(method, '_phase_name', None),
        'target_domains': getattr(method, '_target_domains', None),
        'focus_areas': getattr(method, '_focus_areas', None),
        'complexity_level': getattr(method, '_complexity_level', None),
        'simulation_types': getattr(method, '_simulation_types', None),
        'discovery_types': getattr(method, '_discovery_types', None),
        'validation_level': getattr(method, '_validation_level', None),
        'workflow_config': getattr(method, '_workflow_config', None)
    }
    
    # Add performance metrics if available
    if hasattr(method, '_performance_metrics'):
        info['performance_metrics'] = method._performance_metrics
    
    return info


def disable_physics_enhancements(obj: Any):
    """
    Temporarily disable physics enhancements for all methods in an object.
    
    Args:
        obj: Object to disable enhancements for
    """
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if callable(attr) and hasattr(attr, '_physics_enhanced'):
            attr._enhancement_disabled = True
    
    logger.info(f"Physics enhancements disabled for {obj.__class__.__name__}")


def enable_physics_enhancements(obj: Any):
    """
    Re-enable physics enhancements for all methods in an object.
    
    Args:
        obj: Object to enable enhancements for
    """
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if callable(attr) and hasattr(attr, '_physics_enhanced'):
            attr._enhancement_disabled = False
    
    logger.info(f"Physics enhancements enabled for {obj.__class__.__name__}")


def get_physics_enhancement_summary(obj: Any) -> Dict[str, Any]:
    """
    Get summary of physics enhancements for an object.
    
    Args:
        obj: Object to summarize
        
    Returns:
        Enhancement summary
    """
    enhanced_methods = get_physics_enhanced_methods(obj)
    
    summary = {
        'object_class': obj.__class__.__name__,
        'total_methods': len([attr for attr in dir(obj) if callable(getattr(obj, attr))]),
        'enhanced_methods': len(enhanced_methods),
        'enhancement_types': [],
        'performance_summary': {},
        'enhanced_method_details': {}
    }
    
    # Analyze each enhanced method
    enhancement_types = set()
    total_calls = 0
    total_time = 0.0
    
    for method_name in enhanced_methods:
        method = getattr(obj, method_name)
        info = get_enhancement_info(method)
        
        enhancement_types.add(info.get('enhancement_type', 'unknown'))
        summary['enhanced_method_details'][method_name] = info
        
        # Aggregate performance metrics
        if 'performance_metrics' in info:
            metrics = info['performance_metrics']
            total_calls += metrics.get('call_count', 0)
            total_time += metrics.get('total_time', 0.0)
    
    summary['enhancement_types'] = list(enhancement_types)
    summary['performance_summary'] = {
        'total_calls': total_calls,
        'total_time': total_time,
        'average_call_time': total_time / max(1, total_calls)
    }
    
    return summary