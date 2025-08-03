"""
Physics Experiment Interface Component

This module provides a comprehensive interface for designing, configuring, and monitoring
physics experiments including parameter sweeps, data collection, and validation.
"""

import time
import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    """Types of physics experiments."""
    QUANTUM_SIMULATION = "quantum_simulation"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    MONTE_CARLO = "monte_carlo"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    PARAMETER_SWEEP = "parameter_sweep"
    OPTIMIZATION = "optimization"
    MEASUREMENT_SIMULATION = "measurement_simulation"
    FIELD_CALCULATION = "field_calculation"

class ExperimentStatus(Enum):
    """Status of physics experiments."""
    DRAFT = "draft"
    CONFIGURED = "configured"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentParameter:
    """Single experiment parameter."""
    name: str
    value: Any
    type: str  # 'float', 'int', 'string', 'boolean', 'array'
    description: str = ""
    unit: str = ""
    valid_range: Optional[tuple] = None
    is_sweep: bool = False
    sweep_values: Optional[List[Any]] = None

@dataclass
class ExperimentResult:
    """Result from an experiment run."""
    experiment_id: str
    run_id: str
    timestamp: float
    parameters: Dict[str, Any]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class ExperimentConfiguration:
    """Complete experiment configuration."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    parameters: List[ExperimentParameter]
    output_variables: List[str]
    validation_criteria: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)

class PhysicsExperimentInterface:
    """
    Physics Experiment Design and Management Interface
    
    Provides comprehensive tools for designing, configuring, running, and analyzing
    physics experiments with parameter sweeps, validation, and result tracking.
    """
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfiguration] = {}
        self.experiment_queue: List[str] = []
        self.running_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        
        # Initialize experiment templates
        self._initialize_templates()
    
    def create_experiment(self, name: str, experiment_type: ExperimentType,
                         description: str = "") -> str:
        """Create a new experiment configuration."""
        try:
            experiment_id = str(uuid.uuid4())
            
            config = ExperimentConfiguration(
                experiment_id=experiment_id,
                name=name,
                description=description,
                experiment_type=experiment_type,
                parameters=[],
                output_variables=[],
                validation_criteria={},
                resource_requirements={}
            )
            
            self.experiments[experiment_id] = config
            logger.info(f"Created experiment: {name} ({experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def add_parameter(self, experiment_id: str, parameter: ExperimentParameter) -> bool:
        """Add a parameter to an experiment."""
        try:
            if experiment_id not in self.experiments:
                return False
            
            # Validate parameter
            if not self._validate_parameter(parameter):
                return False
            
            experiment = self.experiments[experiment_id]
            
            # Check if parameter already exists
            existing_names = [p.name for p in experiment.parameters]
            if parameter.name in existing_names:
                # Update existing parameter
                for i, p in enumerate(experiment.parameters):
                    if p.name == parameter.name:
                        experiment.parameters[i] = parameter
                        break
            else:
                # Add new parameter
                experiment.parameters.append(parameter)
            
            experiment.modified_at = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding parameter: {e}")
            return False
    
    def configure_parameter_sweep(self, experiment_id: str, parameter_name: str,
                                sweep_values: List[Any]) -> bool:
        """Configure parameter sweep for a specific parameter."""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            
            # Find and update parameter
            for param in experiment.parameters:
                if param.name == parameter_name:
                    param.is_sweep = True
                    param.sweep_values = sweep_values
                    experiment.modified_at = time.time()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error configuring parameter sweep: {e}")
            return False
    
    def set_output_variables(self, experiment_id: str, output_variables: List[str]) -> bool:
        """Set the output variables for an experiment."""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment.output_variables = output_variables
            experiment.modified_at = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting output variables: {e}")
            return False
    
    def set_validation_criteria(self, experiment_id: str, criteria: Dict[str, Any]) -> bool:
        """Set validation criteria for experiment results."""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment.validation_criteria = criteria
            experiment.modified_at = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting validation criteria: {e}")
            return False
    
    def estimate_experiment_time(self, experiment_id: str) -> Dict[str, Any]:
        """Estimate experiment execution time and resource requirements."""
        try:
            if experiment_id not in self.experiments:
                return {'error': 'Experiment not found'}
            
            experiment = self.experiments[experiment_id]
            
            # Calculate number of runs needed
            sweep_parameters = [p for p in experiment.parameters if p.is_sweep]
            total_runs = 1
            
            for param in sweep_parameters:
                if param.sweep_values:
                    total_runs *= len(param.sweep_values)
            
            # Estimate time per run based on experiment type
            time_estimates = {
                ExperimentType.QUANTUM_SIMULATION: 5.0,  # seconds per run
                ExperimentType.MOLECULAR_DYNAMICS: 15.0,
                ExperimentType.MONTE_CARLO: 8.0,
                ExperimentType.STATISTICAL_ANALYSIS: 2.0,
                ExperimentType.PARAMETER_SWEEP: 3.0,
                ExperimentType.OPTIMIZATION: 20.0,
                ExperimentType.MEASUREMENT_SIMULATION: 4.0,
                ExperimentType.FIELD_CALCULATION: 10.0
            }
            
            time_per_run = time_estimates.get(experiment.experiment_type, 5.0)
            estimated_total_time = total_runs * time_per_run
            
            # Estimate resource requirements
            resource_estimates = {
                'cpu_cores': min(8, max(1, total_runs // 10)),
                'memory_gb': max(1, total_runs * 0.1),
                'disk_space_gb': max(0.1, total_runs * 0.05),
                'gpu_required': experiment.experiment_type in [
                    ExperimentType.QUANTUM_SIMULATION,
                    ExperimentType.MOLECULAR_DYNAMICS
                ]
            }
            
            return {
                'total_runs': total_runs,
                'estimated_time_seconds': estimated_total_time,
                'estimated_time_formatted': self._format_duration(estimated_total_time),
                'resource_requirements': resource_estimates,
                'parallelizable': total_runs > 1,
                'complexity': 'low' if total_runs < 10 else 'medium' if total_runs < 100 else 'high'
            }
            
        except Exception as e:
            logger.error(f"Error estimating experiment time: {e}")
            return {'error': str(e)}
    
    def validate_experiment_configuration(self, experiment_id: str) -> Dict[str, Any]:
        """Validate experiment configuration before execution."""
        try:
            if experiment_id not in self.experiments:
                return {'valid': False, 'errors': ['Experiment not found']}
            
            experiment = self.experiments[experiment_id]
            errors = []
            warnings = []
            
            # Check required fields
            if not experiment.name:
                errors.append("Experiment name is required")
            
            if not experiment.parameters:
                errors.append("At least one parameter is required")
            
            if not experiment.output_variables:
                warnings.append("No output variables specified")
            
            # Validate parameters
            for param in experiment.parameters:
                param_errors = self._validate_parameter_detailed(param)
                errors.extend(param_errors)
            
            # Check parameter sweep consistency
            sweep_params = [p for p in experiment.parameters if p.is_sweep]
            if len(sweep_params) > 3:
                warnings.append("Large parameter sweeps may take significant time")
            
            # Validate resource requirements
            resource_check = self._validate_resource_requirements(experiment)
            if resource_check:
                warnings.extend(resource_check)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'parameter_count': len(experiment.parameters),
                'sweep_parameter_count': len(sweep_params)
            }
            
        except Exception as e:
            logger.error(f"Error validating experiment: {e}")
            return {'valid': False, 'errors': [str(e)]}
    
    def queue_experiment(self, experiment_id: str, priority: int = 0) -> bool:
        """Add experiment to execution queue."""
        try:
            if experiment_id not in self.experiments:
                return False
            
            # Validate configuration
            validation = self.validate_experiment_configuration(experiment_id)
            if not validation['valid']:
                logger.error(f"Cannot queue invalid experiment: {validation['errors']}")
                return False
            
            # Add to queue if not already there
            if experiment_id not in self.experiment_queue:
                if priority > 0:
                    # High priority - add to front
                    self.experiment_queue.insert(0, experiment_id)
                else:
                    # Normal priority - add to end
                    self.experiment_queue.append(experiment_id)
                
                logger.info(f"Queued experiment: {experiment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error queuing experiment: {e}")
            return False
    
    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Start experiment execution."""
        try:
            if experiment_id not in self.experiments:
                return {'success': False, 'error': 'Experiment not found'}
            
            if experiment_id in self.running_experiments:
                return {'success': False, 'error': 'Experiment already running'}
            
            experiment = self.experiments[experiment_id]
            
            # Generate execution plan
            execution_plan = self._generate_execution_plan(experiment)
            
            # Initialize experiment tracking
            self.running_experiments[experiment_id] = {
                'status': ExperimentStatus.RUNNING,
                'started_at': time.time(),
                'current_run': 0,
                'total_runs': execution_plan['total_runs'],
                'execution_plan': execution_plan,
                'progress': 0.0,
                'results': []
            }
            
            # Initialize results storage
            if experiment_id not in self.experiment_results:
                self.experiment_results[experiment_id] = []
            
            # Remove from queue if present
            if experiment_id in self.experiment_queue:
                self.experiment_queue.remove(experiment_id)
            
            logger.info(f"Started experiment: {experiment_id}")
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'execution_plan': execution_plan,
                'estimated_completion': time.time() + execution_plan.get('estimated_time', 60)
            }
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        try:
            if experiment_id not in self.experiments:
                return {'error': 'Experiment not found'}
            
            experiment = self.experiments[experiment_id]
            status_info = {
                'experiment_id': experiment_id,
                'name': experiment.name,
                'type': experiment.experiment_type.value,
                'created_at': experiment.created_at,
                'modified_at': experiment.modified_at
            }
            
            if experiment_id in self.running_experiments:
                running_info = self.running_experiments[experiment_id]
                status_info.update({
                    'status': running_info['status'].value,
                    'progress': running_info['progress'],
                    'current_run': running_info['current_run'],
                    'total_runs': running_info['total_runs'],
                    'started_at': running_info['started_at'],
                    'elapsed_time': time.time() - running_info['started_at']
                })
            elif experiment_id in self.experiment_queue:
                queue_position = self.experiment_queue.index(experiment_id) + 1
                status_info.update({
                    'status': ExperimentStatus.QUEUED.value,
                    'queue_position': queue_position
                })
            else:
                status_info['status'] = ExperimentStatus.CONFIGURED.value
            
            # Add results summary
            if experiment_id in self.experiment_results:
                results = self.experiment_results[experiment_id]
                status_info['results_count'] = len(results)
                status_info['successful_runs'] = len([r for r in results if r.success])
                status_info['failed_runs'] = len([r for r in results if not r.success])
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return {'error': str(e)}
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause a running experiment."""
        try:
            if experiment_id in self.running_experiments:
                self.running_experiments[experiment_id]['status'] = ExperimentStatus.PAUSED
                logger.info(f"Paused experiment: {experiment_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error pausing experiment: {e}")
            return False
    
    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        try:
            if experiment_id in self.running_experiments:
                if self.running_experiments[experiment_id]['status'] == ExperimentStatus.PAUSED:
                    self.running_experiments[experiment_id]['status'] = ExperimentStatus.RUNNING
                    logger.info(f"Resumed experiment: {experiment_id}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error resuming experiment: {e}")
            return False
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running or queued experiment."""
        try:
            # Remove from queue
            if experiment_id in self.experiment_queue:
                self.experiment_queue.remove(experiment_id)
                logger.info(f"Removed experiment from queue: {experiment_id}")
                return True
            
            # Cancel running experiment
            if experiment_id in self.running_experiments:
                self.running_experiments[experiment_id]['status'] = ExperimentStatus.CANCELLED
                logger.info(f"Cancelled running experiment: {experiment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling experiment: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str, 
                             include_data: bool = True) -> List[Dict[str, Any]]:
        """Get results from an experiment."""
        try:
            if experiment_id not in self.experiment_results:
                return []
            
            results = []
            for result in self.experiment_results[experiment_id]:
                result_dict = {
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'parameters': result.parameters,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'metadata': result.metadata
                }
                
                if include_data:
                    result_dict['data'] = result.data
                
                if not result.success and result.error_message:
                    result_dict['error_message'] = result.error_message
                
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting experiment results: {e}")
            return []
    
    def export_experiment_data(self, experiment_id: str, 
                             format_type: str = "json") -> Dict[str, Any]:
        """Export experiment configuration and results."""
        try:
            if experiment_id not in self.experiments:
                return {'error': 'Experiment not found'}
            
            experiment = self.experiments[experiment_id]
            results = self.get_experiment_results(experiment_id)
            
            export_data = {
                'experiment_id': experiment_id,
                'export_timestamp': time.time(),
                'export_format': format_type,
                'configuration': {
                    'name': experiment.name,
                    'description': experiment.description,
                    'type': experiment.experiment_type.value,
                    'parameters': [self._parameter_to_dict(p) for p in experiment.parameters],
                    'output_variables': experiment.output_variables,
                    'validation_criteria': experiment.validation_criteria,
                    'created_at': experiment.created_at,
                    'modified_at': experiment.modified_at
                },
                'results': results,
                'statistics': self._calculate_experiment_statistics(results),
                'metadata': {
                    'version': '1.0.0',
                    'generated_by': 'physics_experiment_interface'
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting experiment data: {e}")
            return {'error': str(e)}
    
    def create_experiment_from_template(self, template_name: str, 
                                      experiment_name: str) -> Optional[str]:
        """Create a new experiment from a template."""
        try:
            if template_name not in self.templates:
                return None
            
            template = self.templates[template_name]
            
            # Create experiment
            experiment_id = self.create_experiment(
                experiment_name,
                ExperimentType(template['type']),
                template.get('description', '')
            )
            
            # Add parameters from template
            for param_data in template.get('parameters', []):
                parameter = ExperimentParameter(**param_data)
                self.add_parameter(experiment_id, parameter)
            
            # Set other configuration
            if 'output_variables' in template:
                self.set_output_variables(experiment_id, template['output_variables'])
            
            if 'validation_criteria' in template:
                self.set_validation_criteria(experiment_id, template['validation_criteria'])
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment from template: {e}")
            return None
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filtering."""
        try:
            experiments = []
            
            for experiment_id, config in self.experiments.items():
                exp_info = {
                    'experiment_id': experiment_id,
                    'name': config.name,
                    'type': config.experiment_type.value,
                    'created_at': config.created_at,
                    'modified_at': config.modified_at,
                    'parameter_count': len(config.parameters),
                    'output_variable_count': len(config.output_variables)
                }
                
                # Determine status
                if experiment_id in self.running_experiments:
                    exp_info['status'] = self.running_experiments[experiment_id]['status'].value
                elif experiment_id in self.experiment_queue:
                    exp_info['status'] = ExperimentStatus.QUEUED.value
                else:
                    exp_info['status'] = ExperimentStatus.CONFIGURED.value
                
                # Add results summary
                if experiment_id in self.experiment_results:
                    results = self.experiment_results[experiment_id]
                    exp_info['results_count'] = len(results)
                    exp_info['last_run'] = max([r.timestamp for r in results]) if results else None
                
                # Apply status filter
                if status_filter is None or exp_info['status'] == status_filter:
                    experiments.append(exp_info)
            
            # Sort by creation time (newest first)
            experiments.sort(key=lambda x: x['created_at'], reverse=True)
            
            return experiments
            
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    # Helper methods
    
    def _initialize_templates(self):
        """Initialize experiment templates."""
        self.templates = {
            'quantum_gate_fidelity': {
                'type': 'quantum_simulation',
                'description': 'Measure quantum gate fidelity under noise',
                'parameters': [
                    {
                        'name': 'gate_type',
                        'value': 'CNOT',
                        'type': 'string',
                        'description': 'Type of quantum gate',
                        'unit': ''
                    },
                    {
                        'name': 'noise_level',
                        'value': 0.01,
                        'type': 'float',
                        'description': 'Noise level',
                        'unit': '',
                        'valid_range': (0.0, 1.0)
                    }
                ],
                'output_variables': ['fidelity', 'gate_error'],
                'validation_criteria': {
                    'fidelity': {'min': 0.0, 'max': 1.0},
                    'gate_error': {'min': 0.0}
                }
            },
            'molecular_dynamics_equilibration': {
                'type': 'molecular_dynamics',
                'description': 'Equilibrate molecular system',
                'parameters': [
                    {
                        'name': 'temperature',
                        'value': 300.0,
                        'type': 'float',
                        'description': 'System temperature',
                        'unit': 'K',
                        'valid_range': (0.0, 1000.0)
                    },
                    {
                        'name': 'pressure',
                        'value': 1.0,
                        'type': 'float',
                        'description': 'System pressure',
                        'unit': 'atm',
                        'valid_range': (0.0, 100.0)
                    }
                ],
                'output_variables': ['total_energy', 'kinetic_energy', 'potential_energy'],
                'validation_criteria': {
                    'total_energy': {'stability_threshold': 0.01}
                }
            }
        }
    
    def _validate_parameter(self, parameter: ExperimentParameter) -> bool:
        """Basic parameter validation."""
        if not parameter.name:
            return False
        
        if parameter.valid_range and parameter.type in ['float', 'int']:
            try:
                value = float(parameter.value)
                min_val, max_val = parameter.valid_range
                if not (min_val <= value <= max_val):
                    return False
            except:
                return False
        
        return True
    
    def _validate_parameter_detailed(self, parameter: ExperimentParameter) -> List[str]:
        """Detailed parameter validation with specific error messages."""
        errors = []
        
        if not parameter.name:
            errors.append(f"Parameter name is required")
        
        if parameter.type not in ['float', 'int', 'string', 'boolean', 'array']:
            errors.append(f"Invalid parameter type: {parameter.type}")
        
        if parameter.valid_range and parameter.type in ['float', 'int']:
            try:
                value = float(parameter.value)
                min_val, max_val = parameter.valid_range
                if not (min_val <= value <= max_val):
                    errors.append(f"Parameter '{parameter.name}' value {value} outside valid range [{min_val}, {max_val}]")
            except:
                errors.append(f"Parameter '{parameter.name}' value cannot be converted to number")
        
        if parameter.is_sweep and not parameter.sweep_values:
            errors.append(f"Parameter '{parameter.name}' marked for sweep but no sweep values provided")
        
        return errors
    
    def _validate_resource_requirements(self, experiment: ExperimentConfiguration) -> List[str]:
        """Validate resource requirements."""
        warnings = []
        
        # Check for resource-intensive experiment types
        if experiment.experiment_type in [ExperimentType.MOLECULAR_DYNAMICS, ExperimentType.OPTIMIZATION]:
            warnings.append("This experiment type may require significant computational resources")
        
        # Check for large parameter sweeps
        sweep_params = [p for p in experiment.parameters if p.is_sweep]
        total_combinations = 1
        for param in sweep_params:
            if param.sweep_values:
                total_combinations *= len(param.sweep_values)
        
        if total_combinations > 1000:
            warnings.append(f"Large parameter sweep ({total_combinations} combinations) will require significant time")
        
        return warnings
    
    def _generate_execution_plan(self, experiment: ExperimentConfiguration) -> Dict[str, Any]:
        """Generate detailed execution plan for an experiment."""
        # Calculate parameter combinations
        sweep_params = [p for p in experiment.parameters if p.is_sweep]
        fixed_params = [p for p in experiment.parameters if not p.is_sweep]
        
        # Generate all parameter combinations
        combinations = []
        if sweep_params:
            import itertools
            sweep_values = [p.sweep_values for p in sweep_params]
            for combo in itertools.product(*sweep_values):
                param_set = {}
                # Add fixed parameters
                for param in fixed_params:
                    param_set[param.name] = param.value
                # Add sweep parameters
                for i, param in enumerate(sweep_params):
                    param_set[param.name] = combo[i]
                combinations.append(param_set)
        else:
            # No sweep parameters, single run
            param_set = {p.name: p.value for p in experiment.parameters}
            combinations.append(param_set)
        
        # Estimate execution time
        time_per_run = {
            ExperimentType.QUANTUM_SIMULATION: 5.0,
            ExperimentType.MOLECULAR_DYNAMICS: 15.0,
            ExperimentType.MONTE_CARLO: 8.0,
            ExperimentType.STATISTICAL_ANALYSIS: 2.0,
            ExperimentType.PARAMETER_SWEEP: 3.0,
            ExperimentType.OPTIMIZATION: 20.0,
            ExperimentType.MEASUREMENT_SIMULATION: 4.0,
            ExperimentType.FIELD_CALCULATION: 10.0
        }.get(experiment.experiment_type, 5.0)
        
        estimated_time = len(combinations) * time_per_run
        
        return {
            'total_runs': len(combinations),
            'parameter_combinations': combinations,
            'estimated_time': estimated_time,
            'parallelizable': len(combinations) > 1,
            'resource_requirements': experiment.resource_requirements
        }
    
    def _parameter_to_dict(self, parameter: ExperimentParameter) -> Dict[str, Any]:
        """Convert ExperimentParameter to dictionary."""
        return {
            'name': parameter.name,
            'value': parameter.value,
            'type': parameter.type,
            'description': parameter.description,
            'unit': parameter.unit,
            'valid_range': parameter.valid_range,
            'is_sweep': parameter.is_sweep,
            'sweep_values': parameter.sweep_values
        }
    
    def _calculate_experiment_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from experiment results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r['success']]
        
        stats = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'failed_runs': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'average_execution_time': sum([r['execution_time'] for r in results]) / len(results)
        }
        
        if successful_results:
            # Calculate statistics for output variables
            for result in successful_results:
                if 'data' in result:
                    for var_name, value in result['data'].items():
                        if isinstance(value, (int, float)):
                            var_key = f'{var_name}_values'
                            if var_key not in stats:
                                stats[var_key] = []
                            stats[var_key].append(value)
            
            # Calculate mean, std for numerical variables
            for key, values in stats.items():
                if key.endswith('_values') and isinstance(values, list):
                    var_name = key[:-7]  # Remove '_values'
                    stats[f'{var_name}_mean'] = np.mean(values)
                    stats[f'{var_name}_std'] = np.std(values)
                    stats[f'{var_name}_min'] = np.min(values)
                    stats[f'{var_name}_max'] = np.max(values)
        
        return stats
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"

# Global experiment interface instance
physics_experiment_interface = PhysicsExperimentInterface()