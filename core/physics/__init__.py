"""
Physics Workflow Enhancement Module

This module provides physics-specific enhancements to the AI Research Lab Framework
without modifying existing components. It uses decorator patterns and dependency
injection to seamlessly integrate advanced physics capabilities.

Key Components:
- PhysicsWorkflowEngine: Main coordination engine for physics research workflows
- PhysicsPhaseEnhancer: Decorator-based enhancements for existing research phases  
- PhysicsValidationEngine: Physics-specific validation and quality control
- PhysicsDiscoveryEngine: Advanced physics discovery workflows
- PhysicsIntegrationManager: Integration with existing framework components

Capabilities:
- Advanced mathematical modeling (quantum mechanics, relativity, statistical physics)
- Computational physics simulations (molecular dynamics, quantum chemistry, fluid dynamics)
- Experimental design and validation for physics research
- Cross-scale phenomena handling (nano to cosmic scales)
- Novel discovery of physical laws and phenomena
"""

__version__ = "1.0.0"
__author__ = "Physics Workflow Enhancement Team"

# Core physics workflow components
from .physics_workflow_engine import PhysicsWorkflowEngine
from .physics_phase_enhancer import PhysicsPhaseEnhancer  
from .physics_validation_engine import PhysicsValidationEngine
from .physics_discovery_engine import PhysicsDiscoveryEngine
from .physics_integration_manager import PhysicsIntegrationManager
from .physics_workflow_decorators import (
    physics_enhanced_phase,
    physics_agent_selection,
    physics_literature_analysis,
    physics_mathematical_modeling,
    physics_simulation_execution,
    physics_law_discovery
)

# Convenience functions for physics workflow integration
def create_physics_enhanced_framework(base_framework, config=None):
    """
    Create a physics-enhanced version of an existing framework.
    
    Args:
        base_framework: Existing AI Research Lab Framework instance
        config: Optional physics configuration
        
    Returns:
        Physics-enhanced framework with decorator-based enhancements
    """
    integration_manager = PhysicsIntegrationManager(config or {})
    return integration_manager.enhance_framework(base_framework)

def create_physics_workflow_engine(config=None):
    """
    Create a standalone physics workflow engine.
    
    Args:
        config: Optional physics configuration
        
    Returns:
        PhysicsWorkflowEngine instance
    """
    return PhysicsWorkflowEngine(config or {})

def apply_physics_enhancements(virtual_lab_system, config=None):
    """
    Apply physics enhancements to an existing Virtual Lab system.
    
    Args:
        virtual_lab_system: Existing VirtualLabMeetingSystem instance
        config: Optional physics configuration
        
    Returns:
        Enhanced virtual lab system with physics capabilities
    """
    enhancer = PhysicsPhaseEnhancer(config or {})
    return enhancer.enhance_virtual_lab(virtual_lab_system)

__all__ = [
    'PhysicsWorkflowEngine',
    'PhysicsPhaseEnhancer',
    'PhysicsValidationEngine', 
    'PhysicsDiscoveryEngine',
    'PhysicsIntegrationManager',
    'physics_enhanced_phase',
    'physics_agent_selection',
    'physics_literature_analysis',
    'physics_mathematical_modeling',
    'physics_simulation_execution',
    'physics_law_discovery',
    'create_physics_enhanced_framework',
    'create_physics_workflow_engine',
    'apply_physics_enhancements'
]