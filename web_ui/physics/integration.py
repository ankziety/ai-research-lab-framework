"""
Physics UI Integration Factory

This module provides integration interfaces for seamlessly integrating
the physics UI components with the existing AI Research Lab Framework
without modifying existing files.
"""

import logging
from typing import Dict, Any, Optional
from flask import Flask
from flask_socketio import SocketIO

from .physics_blueprint import physics_blueprint, physics_namespace
from .physics_dashboard import physics_dashboard
from .physics_visualization import physics_visualization
from .physics_experiment_interface import physics_experiment_interface
from .physics_tool_management import physics_tool_management
from .physics_results_display import physics_results_display

logger = logging.getLogger(__name__)

class PhysicsUIIntegration:
    """
    Physics UI Integration Factory
    
    Provides clean integration of physics UI components with the main application
    without requiring modifications to existing code.
    """
    
    def __init__(self):
        self.physics_components = {
            'dashboard': physics_dashboard,
            'visualization': physics_visualization,
            'experiment_interface': physics_experiment_interface,
            'tool_management': physics_tool_management,
            'results_display': physics_results_display
        }
        self.integration_status = {
            'blueprint_registered': False,
            'socketio_integrated': False,
            'components_initialized': False
        }
    
    def integrate_with_app(self, app: Flask, socketio: Optional[SocketIO] = None) -> bool:
        """
        Integrate physics UI components with the Flask app.
        
        Args:
            app: Flask application instance
            socketio: SocketIO instance for real-time updates
            
        Returns:
            bool: True if integration successful, False otherwise
        """
        try:
            # Register physics blueprint
            app.register_blueprint(physics_blueprint)
            self.integration_status['blueprint_registered'] = True
            logger.info("Physics blueprint registered successfully")
            
            # Integrate with SocketIO if available
            if socketio:
                socketio.on_namespace(physics_namespace)
                self.integration_status['socketio_integrated'] = True
                logger.info("Physics SocketIO namespace integrated")
            
            # Initialize physics components
            self._initialize_components()
            self.integration_status['components_initialized'] = True
            
            logger.info("Physics UI integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating physics UI: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'status': self.integration_status,
            'components': list(self.physics_components.keys()),
            'routes_available': self._get_available_routes(),
            'features': self._get_feature_list()
        }
    
    def get_physics_dashboard_data(self) -> Dict[str, Any]:
        """Get physics dashboard data for display."""
        try:
            return physics_dashboard.get_dashboard_overview()
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def create_physics_visualization(self, viz_type: str, data: Any, 
                                   config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a physics visualization."""
        try:
            if viz_type == 'phase_diagram':
                from .physics_visualization import DataPoint
                data_points = [DataPoint(x=d.get('x', 0), y=d.get('y', 0), 
                                       value=d.get('value', 0)) for d in data]
                return physics_visualization.create_phase_diagram(
                    data_points, (0, 100), (0, 100), config
                )
            elif viz_type == 'energy_landscape':
                def simple_potential(x, y):
                    return x**2 + y**2  # Simple quadratic potential
                return physics_visualization.create_energy_landscape(
                    simple_potential, (-5, 5), (-5, 5), config=config
                )
            elif viz_type == 'molecular_structure':
                atoms = data.get('atoms', [])
                bonds = data.get('bonds', [])
                return physics_visualization.create_molecular_structure(
                    atoms, bonds, config
                )
            # Add more visualization types as needed
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def start_physics_experiment(self, experiment_config: Dict[str, Any]) -> Optional[str]:
        """Start a physics experiment."""
        try:
            experiment_id = physics_experiment_interface.create_experiment(
                experiment_config.get('name', 'Physics Experiment'),
                experiment_config.get('type'),
                experiment_config.get('description', '')
            )
            
            if experiment_id:
                # Add parameters
                for param_data in experiment_config.get('parameters', []):
                    from .physics_experiment_interface import ExperimentParameter
                    parameter = ExperimentParameter(**param_data)
                    physics_experiment_interface.add_parameter(experiment_id, parameter)
                
                # Queue the experiment
                physics_experiment_interface.queue_experiment(experiment_id)
                
                return experiment_id
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return None
    
    def get_physics_tools_status(self) -> Dict[str, Any]:
        """Get physics tools status."""
        try:
            return {
                'tools': physics_tool_management.list_tools(),
                'system_resources': physics_tool_management.get_system_resources()
            }
        except Exception as e:
            logger.error(f"Error getting tools status: {e}")
            return {'error': str(e)}
    
    def analyze_physics_results(self, result_data: Any, analysis_type: str = 'statistical') -> Dict[str, Any]:
        """Analyze physics results."""
        try:
            from .physics_results_display import PhysicsResult, ResultType, DataFormat
            
            # Create a physics result object
            result = PhysicsResult(
                result_id='',
                name='Analysis Result',
                description='Physics data analysis',
                result_type=ResultType.STATISTICAL_ANALYSIS,
                data_format=DataFormat.NUMERICAL,
                data=result_data,
                metadata={}
            )
            
            # Add to results display
            physics_results_display.add_result(result)
            
            # Perform analysis
            return physics_results_display.analyze_result_statistics(result.result_id)
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {'error': str(e)}
    
    def _initialize_components(self):
        """Initialize physics components with default data."""
        try:
            # Initialize dashboard with sample data
            physics_dashboard.update_simulation_metrics({
                'total_simulations': 5,
                'active_simulations': 2,
                'completed_simulations': 3
            })
            
            # Initialize sample tools
            sample_tools = [
                {
                    'tool_id': 'quantum_simulator_demo',
                    'name': 'Quantum Simulator (Demo)',
                    'status': 'online',
                    'version': '2.1.0'
                },
                {
                    'tool_id': 'molecular_dynamics_demo',
                    'name': 'Molecular Dynamics (Demo)',
                    'status': 'online',
                    'version': '1.8.3'
                }
            ]
            
            # Register sample experiments
            sample_experiments = [
                {
                    'name': 'Quantum Gate Fidelity Test',
                    'type': 'quantum_simulation',
                    'description': 'Test quantum gate fidelity under noise'
                },
                {
                    'name': 'Molecular Equilibration',
                    'type': 'molecular_dynamics',
                    'description': 'Equilibrate water system at 300K'
                }
            ]
            
            logger.info("Physics components initialized with sample data")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _get_available_routes(self) -> list:
        """Get list of available physics routes."""
        return [
            '/physics/',
            '/physics/dashboard',
            '/physics/dashboard/metrics',
            '/physics/visualization/create',
            '/physics/visualization/list',
            '/physics/experiment/create',
            '/physics/experiment/list',
            '/physics/tools/status',
            '/physics/results/analyze',
            '/physics/results/export'
        ]
    
    def _get_feature_list(self) -> list:
        """Get list of available physics features."""
        return [
            'Real-time physics research monitoring',
            'Interactive 3D data visualization',
            'Physics experiment design and management',
            'Tool installation and monitoring',
            'Statistical analysis and uncertainty quantification',
            'WebSocket integration for live updates',
            'Export and reporting capabilities',
            'Multi-physics domain support'
        ]

# Global integration instance
physics_ui_integration = PhysicsUIIntegration()

# Integration function for easy import
def integrate_physics_ui(app: Flask, socketio: Optional[SocketIO] = None) -> bool:
    """
    Convenience function to integrate physics UI with Flask app.
    
    Usage:
        from web_ui.physics.integration import integrate_physics_ui
        success = integrate_physics_ui(app, socketio)
    """
    return physics_ui_integration.integrate_with_app(app, socketio)

# Export main components for direct access if needed
__all__ = [
    'PhysicsUIIntegration',
    'physics_ui_integration',
    'integrate_physics_ui',
    'physics_blueprint',
    'physics_namespace'
]