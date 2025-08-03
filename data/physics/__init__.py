"""
Physics data management components for the AI Research Lab.

This package contains physics-specific data processing and management components including:
- PhysicsDataManager: Core physics data operations
- PhysicsDatabaseConnector: External physics database integration
- PhysicsDataValidation: Physics data quality control and validation
- PhysicsDataVisualization: Physics data plotting and visualization
- PhysicsDataExport: Physics results export and reporting
- PhysicsDataAdapter: Adapter for existing data system integration
"""

from .physics_data_manager import PhysicsDataManager, PhysicsDataset
from .physics_database_connector import PhysicsDatabaseConnector, DatabaseConnection
from .physics_data_validation import PhysicsDataValidation, ValidationResult, ValidationLevel
from .physics_data_visualization import PhysicsDataVisualization, PlotConfig
from .physics_data_export import PhysicsDataExport, ExportConfig
from .physics_data_adapter import PhysicsDataAdapter

# Convenience function to create a complete physics data system
def create_physics_data_system(config: dict = None, base_dir: str = None):
    """
    Create a complete physics data management system.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for physics data storage
    
    Returns:
        PhysicsDataAdapter instance with all components initialized
    """
    if config is None:
        config = {}
    
    return PhysicsDataAdapter(config)

__all__ = [
    'PhysicsDataManager',
    'PhysicsDataset',
    'PhysicsDatabaseConnector',
    'DatabaseConnection',
    'PhysicsDataValidation',
    'ValidationResult',
    'ValidationLevel',
    'PhysicsDataVisualization',
    'PlotConfig',
    'PhysicsDataExport',
    'ExportConfig',
    'PhysicsDataAdapter',
    'create_physics_data_system'
]