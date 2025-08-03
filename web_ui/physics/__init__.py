"""
Physics-Specific Web Interface Components

This module provides physics-specific web interface components for the AI Research Lab Framework.
It includes specialized tools for physics research, visualization, experiment design, and results analysis.
"""

from .physics_blueprint import physics_blueprint
from .physics_dashboard import PhysicsDashboard
from .physics_visualization import PhysicsVisualization
from .physics_experiment_interface import PhysicsExperimentInterface
from .physics_tool_management import PhysicsToolManagement
from .physics_results_display import PhysicsResultsDisplay

__version__ = "1.0.0"
__author__ = "AI Research Lab Framework"

# Export main components
__all__ = [
    'physics_blueprint',
    'PhysicsDashboard',
    'PhysicsVisualization',
    'PhysicsExperimentInterface',
    'PhysicsToolManagement',
    'PhysicsResultsDisplay'
]