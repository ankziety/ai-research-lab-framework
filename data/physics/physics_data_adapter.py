"""
Physics Data Adapter for AI Research Lab Framework.

Provides adapter pattern integration between the physics data system
and the existing data management infrastructure.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Import physics data components
from .physics_data_manager import PhysicsDataManager, PhysicsDataset
from .physics_database_connector import PhysicsDatabaseConnector
from .physics_data_validation import PhysicsDataValidation
from .physics_data_visualization import PhysicsDataVisualization
from .physics_data_export import PhysicsDataExport

# Try to import existing data manager
try:
    from ..data_manager import DataManager
    DATA_MANAGER_AVAILABLE = True
except ModuleNotFoundError as e:
    if e.name == 'data_manager':
        try:
            from ...web_ui.data_manager import DataManager
            DATA_MANAGER_AVAILABLE = True
        except ModuleNotFoundError as e2:
            if e2.name == 'data_manager':
                DATA_MANAGER_AVAILABLE = False
                DataManager = None
            else:
                raise
    else:
        raise

logger = logging.getLogger(__name__)

class PhysicsDataAdapter:
    """
    Adapter for integrating physics data system with existing data management.
    
    This class provides a unified interface that combines physics-specific
    data operations with the existing data management infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any], existing_data_manager: Optional[Any] = None):
        """
        Initialize the Physics Data Adapter.
        
        Args:
            config: Configuration dictionary
            existing_data_manager: Optional existing data manager instance
        """
        self.config = config
        self.existing_data_manager = existing_data_manager
        
        # Initialize physics data components
        self._initialize_physics_components()
        
        # Create unified data registry
        self.unified_registry = {}
        
        logger.info("PhysicsDataAdapter initialized successfully")
    
    def _initialize_physics_components(self) -> None:
        """Initialize all physics data components."""
        try:
            # Base directory for physics data
            base_dir = self.config.get('physics_data_dir', Path.home() / ".ai_research_lab" / "physics")
            
            # Initialize components
            self.physics_data_manager = PhysicsDataManager(
                config=self.config,
                data_dir=str(base_dir / "data")
            )
            
            self.physics_database_connector = PhysicsDatabaseConnector(
                config=self.config,
                cache_dir=str(base_dir / "cache")
            )
            
            self.physics_data_validation = PhysicsDataValidation(
                config=self.config
            )
            
            self.physics_data_visualization = PhysicsDataVisualization(
                config=self.config,
                output_dir=str(base_dir / "visualizations")
            )
            
            self.physics_data_export = PhysicsDataExport(
                config=self.config,
                output_dir=str(base_dir / "exports")
            )
            
            logger.info("All physics data components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize physics components: {str(e)}")
            raise
    
    # Unified data operations
    def load_data(self, source: str, data_type: str = "auto", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Load data from various sources (physics or general).
        
        Args:
            source: Data source (file path, URL, database ID)
            data_type: Type of data (physics, general, auto)
            **kwargs: Additional loading parameters
        
        Returns:
            Loaded data or None if failed
        """
        try:
            # Auto-detect data type if needed
            if data_type == "auto":
                data_type = self._detect_data_type(source, **kwargs)
            
            if data_type == "physics":
                # Use physics data manager
                dataset = self.physics_data_manager.load_physics_data(source, **kwargs)
                if dataset:
                    return self._physics_dataset_to_dict(dataset)
            
            elif data_type == "general" and self.existing_data_manager:
                # Use existing data manager
                return self._load_with_existing_manager(source, **kwargs)
            
            else:
                # Try both systems
                result = None
                
                # Try physics first
                try:
                    dataset = self.physics_data_manager.load_physics_data(source, **kwargs)
                    if dataset:
                        result = self._physics_dataset_to_dict(dataset)
                except Exception:
                    pass
                
                # Try existing manager if physics failed
                if not result and self.existing_data_manager:
                    try:
                        result = self._load_with_existing_manager(source, **kwargs)
                    except Exception:
                        pass
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load data from {source}: {str(e)}")
            return None
    
    def save_data(self, data: Dict[str, Any], output_path: str = None, **kwargs) -> bool:
        """
        Save data using appropriate system.
        
        Args:
            data: Data to save
            output_path: Optional output path
            **kwargs: Additional saving parameters
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine if this is physics data
            if self._is_physics_data(data):
                # Convert to physics dataset and save
                dataset = self._dict_to_physics_dataset(data)
                return self.physics_data_manager.save_physics_data(dataset, output_path)
            
            elif self.existing_data_manager:
                # Use existing data manager
                return self._save_with_existing_manager(data, output_path, **kwargs)
            
            else:
                # Default to physics system
                dataset = self._dict_to_physics_dataset(data)
                return self.physics_data_manager.save_physics_data(dataset, output_path)
            
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            return False
    
    def search_database(self, query: str, database: str = "auto", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Search external databases.
        
        Args:
            query: Search query
            database: Database to search (or auto for intelligent selection)
            **kwargs: Additional search parameters
        
        Returns:
            Search results or None if failed
        """
        try:
            if database == "auto":
                # Intelligent database selection based on query
                database = self._select_database_for_query(query, **kwargs)
            
            # Use physics database connector
            return self.physics_database_connector.search_physics_database(query, database, **kwargs)
            
        except Exception as e:
            logger.error(f"Database search failed: {str(e)}")
            return None
    
    def validate_data(self, data: Dict[str, Any], criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Validate data quality and consistency.
        
        Args:
            data: Data to validate
            criteria: Validation criteria
        
        Returns:
            List of validation results
        """
        try:
            # Use physics validation if physics data, otherwise basic validation
            if self._is_physics_data(data):
                criteria = criteria or self._get_default_physics_criteria(data)
                results = self.physics_data_validation.validate_physics_data(data, criteria)
                return [result.to_dict() for result in results]
            
            elif self.existing_data_manager:
                # Use existing validation if available
                return self._validate_with_existing_manager(data, criteria)
            
            else:
                # Basic validation
                return self._basic_validation(data, criteria)
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return []
    
    def visualize_data(self, data: Dict[str, Any], plot_type: str, **kwargs) -> Optional[str]:
        """
        Create data visualizations.
        
        Args:
            data: Data to visualize
            plot_type: Type of plot to create
            **kwargs: Additional plotting parameters
        
        Returns:
            Path to created visualization or None if failed
        """
        try:
            # Use physics visualization
            return self.physics_data_visualization.visualize_physics_data(data, plot_type, **kwargs)
            
        except Exception as e:
            logger.error(f"Data visualization failed: {str(e)}")
            return None
    
    def export_data(self, data: Dict[str, Any], format: str, **kwargs) -> Optional[str]:
        """
        Export data in specified format.
        
        Args:
            data: Data to export
            format: Export format
            **kwargs: Additional export parameters
        
        Returns:
            Path to exported file or None if failed
        """
        try:
            # Use physics export system
            return self.physics_data_export.export_physics_results(data, format, **kwargs)
            
        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            return None
    
    # Data management integration
    def list_all_datasets(self, source: str = "all") -> List[Dict[str, Any]]:
        """
        List datasets from all sources.
        
        Args:
            source: Data source (physics, general, all)
        
        Returns:
            List of dataset metadata
        """
        datasets = []
        
        try:
            if source in ["physics", "all"]:
                # Get physics datasets
                physics_datasets = self.physics_data_manager.list_datasets()
                for dataset in physics_datasets:
                    dataset['source_system'] = 'physics'
                    datasets.append(dataset)
            
            if source in ["general", "all"] and self.existing_data_manager:
                # Get general datasets
                try:
                    general_datasets = self._list_existing_datasets()
                    for dataset in general_datasets:
                        dataset['source_system'] = 'general'
                        datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to list general datasets: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
        
        return datasets
    
    def get_dataset(self, dataset_id: str, source_system: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Get dataset by ID from appropriate system.
        
        Args:
            dataset_id: Dataset identifier
            source_system: System to search (physics, general, auto)
        
        Returns:
            Dataset data or None if not found
        """
        try:
            if source_system == "physics":
                dataset = self.physics_data_manager.get_dataset(dataset_id)
                return self._physics_dataset_to_dict(dataset) if dataset else None
            
            elif source_system == "general" and self.existing_data_manager:
                return self._get_existing_dataset(dataset_id)
            
            else:  # auto
                # Try physics first
                dataset = self.physics_data_manager.get_dataset(dataset_id)
                if dataset:
                    return self._physics_dataset_to_dict(dataset)
                
                # Try existing system
                if self.existing_data_manager:
                    return self._get_existing_dataset(dataset_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_id: str, source_system: str = "auto") -> bool:
        """
        Delete dataset from appropriate system.
        
        Args:
            dataset_id: Dataset identifier
            source_system: System to delete from (physics, general, auto)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if source_system == "physics":
                return self.physics_data_manager.delete_dataset(dataset_id)
            
            elif source_system == "general" and self.existing_data_manager:
                return self._delete_existing_dataset(dataset_id)
            
            else:  # auto
                # Try to determine source and delete
                dataset = self.physics_data_manager.get_dataset(dataset_id)
                if dataset:
                    return self.physics_data_manager.delete_dataset(dataset_id)
                
                if self.existing_data_manager:
                    return self._delete_existing_dataset(dataset_id)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
            return False
    
    # Helper methods for data type detection and conversion
    def _detect_data_type(self, source: str, **kwargs) -> str:
        """Detect if data source contains physics data."""
        # Check file extension and path
        if isinstance(source, str):
            source_path = Path(source)
            
            # Check for physics-related keywords in path
            physics_keywords = ['physics', 'simulation', 'experiment', 'pdb', 'quantum', 'particle']
            path_str = str(source_path).lower()
            
            if any(keyword in path_str for keyword in physics_keywords):
                return "physics"
            
            # Check file format
            physics_formats = ['.h5', '.hdf5', '.pdb', '.xyz', '.cif']
            if source_path.suffix.lower() in physics_formats:
                return "physics"
        
        # Check kwargs for physics indicators
        physics_indicators = ['domain', 'data_type', 'units', 'uncertainty']
        if any(indicator in kwargs for indicator in physics_indicators):
            return "physics"
        
        return "general"
    
    def _is_physics_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains physics-specific information."""
        physics_keys = [
            'units', 'uncertainty', 'energy', 'mass', 'charge', 'spin',
            'wavelength', 'frequency', 'temperature', 'pressure',
            'coordinates', 'atoms', 'molecules', 'particles'
        ]
        
        # Check for physics-specific keys
        for key in physics_keys:
            if key in data or any(key in str(k).lower() for k in data.keys()):
                return True
        
        # Check metadata
        if 'metadata' in data:
            metadata = data['metadata']
            if isinstance(metadata, dict):
                domain = metadata.get('domain', '').lower()
                data_type = metadata.get('data_type', '').lower()
                
                physics_domains = ['physics', 'quantum', 'particle', 'condensed_matter', 'astrophysics']
                physics_types = ['experimental', 'simulation', 'theoretical']
                
                if domain in physics_domains or data_type in physics_types:
                    return True
        
        return False
    
    def _physics_dataset_to_dict(self, dataset: PhysicsDataset) -> Dict[str, Any]:
        """Convert PhysicsDataset to dictionary."""
        if not dataset:
            return {}
        
        result = dataset.to_dict()
        if dataset.data is not None:
            result['data'] = dataset.data
        
        return result
    
    def _dict_to_physics_dataset(self, data: Dict[str, Any]) -> PhysicsDataset:
        """Convert dictionary to PhysicsDataset."""
        # Extract required fields with defaults
        dataset_id = data.get('id', f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        name = data.get('name', 'Unnamed Dataset')
        data_type = data.get('data_type', 'experimental')
        domain = data.get('domain', 'general')
        source = data.get('source', 'unknown')
        format = data.get('format', 'json')
        
        # Handle datetime
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        # Extract metadata and actual data
        metadata = data.get('metadata', {})
        actual_data = data.get('data', data)
        
        return PhysicsDataset(
            id=dataset_id,
            name=name,
            data_type=data_type,
            domain=domain,
            source=source,
            format=format,
            created_at=created_at,
            metadata=metadata,
            file_path=data.get('file_path'),
            data=actual_data
        )
    
    def _select_database_for_query(self, query: str, **kwargs) -> str:
        """Intelligently select database based on query content."""
        query_lower = query.lower()
        
        # Keywords for different databases
        pdb_keywords = ['protein', 'structure', 'pdb', 'amino', 'chain', 'crystal']
        arxiv_keywords = ['paper', 'article', 'author', 'theory', 'model']
        nist_keywords = ['constant', 'reference', 'standard', 'value']
        materials_keywords = ['material', 'compound', 'formula', 'crystal', 'lattice']
        astro_keywords = ['star', 'galaxy', 'object', 'coordinate', 'ra', 'dec']
        
        # Check query content
        if any(keyword in query_lower for keyword in pdb_keywords):
            return 'pdb'
        elif any(keyword in query_lower for keyword in arxiv_keywords):
            return 'arxiv'
        elif any(keyword in query_lower for keyword in nist_keywords):
            return 'nist'
        elif any(keyword in query_lower for keyword in materials_keywords):
            return 'materials_project'
        elif any(keyword in query_lower for keyword in astro_keywords):
            return 'sdss'
        
        # Default to arXiv for general physics queries
        return 'arxiv'
    
    def _get_default_physics_criteria(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default validation criteria for physics data."""
        criteria = {
            'required_fields': [],
            'unit_fields': [],
            'positive_fields': [],
            'uncertainty_fields': []
        }
        
        # Analyze data structure to create appropriate criteria
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'energy' in key.lower() or 'mass' in key.lower():
                    criteria['positive_fields'].append(key)
                    criteria['uncertainty_fields'].append(key)
                
                if any(unit_word in key.lower() for unit_word in ['temperature', 'pressure', 'length']):
                    criteria['unit_fields'].append(key)
        
        # Domain-specific criteria
        domain = data.get('metadata', {}).get('domain', 'general')
        if domain == 'particle_physics':
            criteria['conservation_checks'] = ['energy', 'momentum', 'charge']
        elif domain == 'astrophysics':
            criteria['value_ranges'] = {
                'ra': (0, 360),
                'dec': (-90, 90)
            }
        
        return criteria
    
    # Integration methods with existing data manager
    def _load_with_existing_manager(self, source: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Load data using existing data manager."""
        if not self.existing_data_manager:
            return None
        
        # This would need to be adapted based on the actual existing data manager interface
        # For now, return a placeholder
        logger.info(f"Loading {source} with existing data manager")
        return None
    
    def _save_with_existing_manager(self, data: Dict[str, Any], output_path: str, **kwargs) -> bool:
        """Save data using existing data manager."""
        if not self.existing_data_manager:
            return False
        
        # This would need to be adapted based on the actual existing data manager interface
        logger.info(f"Saving data with existing data manager to {output_path}")
        return True
    
    def _validate_with_existing_manager(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate data using existing data manager."""
        if not self.existing_data_manager:
            return []
        
        # Placeholder for existing validation
        return []
    
    def _list_existing_datasets(self) -> List[Dict[str, Any]]:
        """List datasets from existing data manager."""
        if not self.existing_data_manager:
            return []
        
        # This would integrate with the actual existing data manager
        return []
    
    def _get_existing_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset from existing data manager."""
        if not self.existing_data_manager:
            return None
        
        # This would integrate with the actual existing data manager
        return None
    
    def _delete_existing_dataset(self, dataset_id: str) -> bool:
        """Delete dataset from existing data manager."""
        if not self.existing_data_manager:
            return False
        
        # This would integrate with the actual existing data manager
        return True
    
    def _basic_validation(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform basic data validation."""
        results = []
        
        # Basic structure validation
        if not isinstance(data, dict):
            results.append({
                'level': 'error',
                'message': 'Data must be a dictionary',
                'field': 'root'
            })
        
        # Check for empty data
        if not data:
            results.append({
                'level': 'warning',
                'message': 'Data is empty',
                'field': 'root'
            })
        
        return results
    
    # Utility and status methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all data management systems."""
        status = {
            'physics_system': {
                'available': True,
                'components': {
                    'data_manager': self.physics_data_manager is not None,
                    'database_connector': self.physics_database_connector is not None,
                    'validation': self.physics_data_validation is not None,
                    'visualization': self.physics_data_visualization is not None,
                    'export': self.physics_data_export is not None
                }
            },
            'existing_system': {
                'available': self.existing_data_manager is not None,
                'type': type(self.existing_data_manager).__name__ if self.existing_data_manager else None
            },
            'integration': {
                'unified_interface': True,
                'auto_detection': True,
                'cross_system_operations': True
            }
        }
        
        # Get statistics
        if self.physics_data_manager:
            status['physics_system']['statistics'] = self.physics_data_manager.get_data_statistics()
        
        return status
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get list of system capabilities."""
        return {
            'data_loading': {
                'formats': ['json', 'csv', 'hdf5', 'txt', 'binary'],
                'sources': ['file', 'url', 'database'],
                'auto_detection': True
            },
            'database_search': {
                'databases': list(self.physics_database_connector.databases.keys()),
                'intelligent_selection': True,
                'caching': True
            },
            'validation': {
                'physics_specific': True,
                'units_checking': True,
                'uncertainty_analysis': True,
                'conservation_laws': True
            },
            'visualization': {
                'plot_types': self.physics_data_visualization.get_supported_plot_types(),
                'interactive': True,
                'export_formats': ['png', 'svg', 'pdf', 'html']
            },
            'export': {
                'formats': self.physics_data_export.get_supported_formats(),
                'reports': True,
                'manuscripts': True
            }
        }
    
    def create_unified_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unified data workflow combining multiple operations."""
        workflow_results = {
            'workflow_id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'steps': [],
            'results': {},
            'status': 'success'
        }
        
        try:
            steps = workflow_config.get('steps', [])
            
            for step_config in steps:
                step_type = step_config.get('type')
                step_name = step_config.get('name', step_type)
                
                step_result = {
                    'name': step_name,
                    'type': step_type,
                    'status': 'success',
                    'output': None
                }
                
                try:
                    if step_type == 'load':
                        result = self.load_data(**step_config.get('params', {}))
                        step_result['output'] = result
                        if result:
                            workflow_results['results'][step_name] = result
                    
                    elif step_type == 'search':
                        result = self.search_database(**step_config.get('params', {}))
                        step_result['output'] = result
                        if result:
                            workflow_results['results'][step_name] = result
                    
                    elif step_type == 'validate':
                        data_key = step_config.get('data_key', 'data')
                        data = workflow_results['results'].get(data_key, {})
                        result = self.validate_data(data, **step_config.get('params', {}))
                        step_result['output'] = result
                        workflow_results['results'][f"{step_name}_validation"] = result
                    
                    elif step_type == 'visualize':
                        data_key = step_config.get('data_key', 'data')
                        data = workflow_results['results'].get(data_key, {})
                        result = self.visualize_data(data, **step_config.get('params', {}))
                        step_result['output'] = result
                        if result:
                            workflow_results['results'][f"{step_name}_plot"] = result
                    
                    elif step_type == 'export':
                        data_key = step_config.get('data_key', 'results')
                        data = workflow_results['results'].get(data_key, workflow_results['results'])
                        result = self.export_data(data, **step_config.get('params', {}))
                        step_result['output'] = result
                        if result:
                            workflow_results['results'][f"{step_name}_export"] = result
                    
                    else:
                        step_result['status'] = 'error'
                        step_result['error'] = f"Unknown step type: {step_type}"
                
                except Exception as e:
                    step_result['status'] = 'error'
                    step_result['error'] = str(e)
                    workflow_results['status'] = 'partial_success'
                
                workflow_results['steps'].append(step_result)
            
        except Exception as e:
            workflow_results['status'] = 'error'
            workflow_results['error'] = str(e)
        
        return workflow_results