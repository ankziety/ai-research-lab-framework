"""
Physics Tool Management Component

This module provides comprehensive management of physics software tools,
resource monitoring, installation tracking, and status monitoring.
"""

import time
import logging
import subprocess
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolStatus(Enum):
    """Status of physics tools."""
    ONLINE = "online"
    OFFLINE = "offline"
    INSTALLING = "installing"
    UPDATING = "updating"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

class ToolCategory(Enum):
    """Categories of physics tools."""
    QUANTUM_SIMULATION = "quantum_simulation"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    STATISTICAL_PHYSICS = "statistical_physics"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    NUMERICAL_COMPUTATION = "numerical_computation"
    MACHINE_LEARNING = "machine_learning"
    EXPERIMENTAL_CONTROL = "experimental_control"

@dataclass
class ToolCapability:
    """Single tool capability."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolConfiguration:
    """Tool configuration and metadata."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    version: str
    status: ToolStatus
    capabilities: List[ToolCapability]
    installation_path: Optional[str] = None
    config_file: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    installation_date: Optional[float] = None
    update_available: bool = False
    error_log: List[str] = field(default_factory=list)

class PhysicsToolManagement:
    """
    Physics Tool Management System
    
    Provides comprehensive management of physics software tools including
    installation, monitoring, resource tracking, and capability management.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolConfiguration] = {}
        self.system_resources = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_space': psutil.disk_usage('/').total,
            'platform': platform.system(),
            'architecture': platform.machine()
        }
        self.resource_monitors: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default physics tools
        self._initialize_default_tools()
    
    def register_tool(self, tool_config: ToolConfiguration) -> bool:
        """Register a new physics tool."""
        try:
            self.tools[tool_config.tool_id] = tool_config
            logger.info(f"Registered physics tool: {tool_config.name}")
            
            # Start resource monitoring for the tool
            self._start_resource_monitoring(tool_config.tool_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool: {e}")
            return False
    
    def install_tool(self, tool_id: str, version: str = "latest", 
                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Install or update a physics tool."""
        try:
            if tool_id not in self.tools:
                return {'success': False, 'error': 'Tool not found'}
            
            tool = self.tools[tool_id]
            
            # Check if already installing
            if tool.status == ToolStatus.INSTALLING:
                return {'success': False, 'error': 'Tool installation already in progress'}
            
            # Update status
            tool.status = ToolStatus.INSTALLING
            tool.last_check = time.time()
            
            # Simulate installation process
            installation_result = self._simulate_tool_installation(tool, version, config)
            
            if installation_result['success']:
                tool.status = ToolStatus.ONLINE
                tool.version = version
                tool.installation_date = time.time()
                tool.error_log.clear()
                
                logger.info(f"Successfully installed {tool.name} v{version}")
            else:
                tool.status = ToolStatus.ERROR
                tool.error_log.append(f"Installation failed: {installation_result['error']}")
                
                logger.error(f"Failed to install {tool.name}: {installation_result['error']}")
            
            return installation_result
            
        except Exception as e:
            logger.error(f"Error installing tool: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_tool_status(self, tool_id: str) -> Dict[str, Any]:
        """Check the current status of a physics tool."""
        try:
            if tool_id not in self.tools:
                return {'status': ToolStatus.UNKNOWN.value, 'error': 'Tool not found'}
            
            tool = self.tools[tool_id]
            
            # Perform health check
            health_check = self._perform_health_check(tool)
            
            # Update status based on health check
            if health_check['healthy']:
                if tool.status == ToolStatus.ERROR:
                    tool.status = ToolStatus.ONLINE
                    tool.error_log.clear()
            else:
                tool.status = ToolStatus.ERROR
                if health_check.get('error'):
                    tool.error_log.append(health_check['error'])
            
            tool.last_check = time.time()
            
            status_info = {
                'tool_id': tool_id,
                'name': tool.name,
                'status': tool.status.value,
                'version': tool.version,
                'last_check': tool.last_check,
                'health_check': health_check,
                'resource_usage': self._get_tool_resource_usage(tool_id),
                'capabilities_available': len(tool.capabilities),
                'update_available': tool.update_available
            }
            
            if tool.error_log:
                status_info['recent_errors'] = tool.error_log[-5:]  # Last 5 errors
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error checking tool status: {e}")
            return {'status': ToolStatus.ERROR.value, 'error': str(e)}
    
    def get_tool_capabilities(self, tool_id: str) -> List[Dict[str, Any]]:
        """Get available capabilities for a tool."""
        try:
            if tool_id not in self.tools:
                return []
            
            tool = self.tools[tool_id]
            
            capabilities = []
            for cap in tool.capabilities:
                cap_info = {
                    'name': cap.name,
                    'description': cap.description,
                    'parameters': cap.parameters,
                    'requirements': cap.requirements,
                    'available': tool.status == ToolStatus.ONLINE
                }
                capabilities.append(cap_info)
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting tool capabilities: {e}")
            return []
    
    def execute_tool_capability(self, tool_id: str, capability_name: str,
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific capability of a physics tool."""
        try:
            if tool_id not in self.tools:
                return {'success': False, 'error': 'Tool not found'}
            
            tool = self.tools[tool_id]
            
            if tool.status != ToolStatus.ONLINE:
                return {'success': False, 'error': f'Tool is {tool.status.value}'}
            
            # Find the capability
            capability = None
            for cap in tool.capabilities:
                if cap.name == capability_name:
                    capability = cap
                    break
            
            if not capability:
                return {'success': False, 'error': 'Capability not found'}
            
            # Validate parameters
            validation_result = self._validate_capability_parameters(capability, parameters)
            if not validation_result['valid']:
                return {'success': False, 'error': f"Invalid parameters: {validation_result['errors']}"}
            
            # Execute capability
            execution_result = self._execute_capability(tool, capability, parameters)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing tool capability: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource information."""
        try:
            # Get current resource usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get GPU information if available
            gpu_info = self._get_gpu_information()
            
            # Get network information
            network = psutil.net_io_counters()
            
            resources = {
                'timestamp': time.time(),
                'cpu': {
                    'count': self.system_resources['cpu_count'],
                    'usage_percent': cpu_usage,
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu': gpu_info,
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'platform': {
                    'system': self.system_resources['platform'],
                    'architecture': self.system_resources['architecture'],
                    'python_version': platform.python_version()
                }
            }
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {'error': str(e)}
    
    def monitor_resource_usage(self, tool_id: str) -> Dict[str, Any]:
        """Monitor resource usage for a specific tool."""
        try:
            if tool_id not in self.resource_monitors:
                return {'error': 'Tool not being monitored'}
            
            monitor_data = self.resource_monitors[tool_id]
            
            # Update monitoring data
            current_time = time.time()
            
            # Simulate resource usage (in real implementation, would track actual processes)
            cpu_usage = 5.0 + (hash(tool_id) % 20)  # 5-25% CPU
            memory_usage = 100 + (hash(tool_id) % 500)  # 100-600 MB
            
            monitor_data['measurements'].append({
                'timestamp': current_time,
                'cpu_percent': cpu_usage,
                'memory_mb': memory_usage,
                'disk_io_mb': (hash(tool_id) % 10),
                'network_io_kb': (hash(tool_id) % 100)
            })
            
            # Keep only last 100 measurements
            if len(monitor_data['measurements']) > 100:
                monitor_data['measurements'] = monitor_data['measurements'][-100:]
            
            # Calculate statistics
            recent_measurements = monitor_data['measurements'][-10:]  # Last 10 measurements
            
            if recent_measurements:
                avg_cpu = sum([m['cpu_percent'] for m in recent_measurements]) / len(recent_measurements)
                avg_memory = sum([m['memory_mb'] for m in recent_measurements]) / len(recent_measurements)
                
                monitor_data['statistics'] = {
                    'average_cpu_percent': avg_cpu,
                    'average_memory_mb': avg_memory,
                    'peak_cpu_percent': max([m['cpu_percent'] for m in recent_measurements]),
                    'peak_memory_mb': max([m['memory_mb'] for m in recent_measurements])
                }
            
            return monitor_data
            
        except Exception as e:
            logger.error(f"Error monitoring resource usage: {e}")
            return {'error': str(e)}
    
    def list_tools(self, category: Optional[ToolCategory] = None,
                  status: Optional[ToolStatus] = None) -> List[Dict[str, Any]]:
        """List all registered physics tools with optional filtering."""
        try:
            tools_list = []
            
            for tool_id, tool in self.tools.items():
                # Apply filters
                if category and tool.category != category:
                    continue
                if status and tool.status != status:
                    continue
                
                tool_info = {
                    'tool_id': tool_id,
                    'name': tool.name,
                    'description': tool.description,
                    'category': tool.category.value,
                    'version': tool.version,
                    'status': tool.status.value,
                    'capabilities_count': len(tool.capabilities),
                    'last_check': tool.last_check,
                    'installation_date': tool.installation_date,
                    'update_available': tool.update_available
                }
                
                # Add resource usage if monitored
                if tool_id in self.resource_monitors:
                    resource_data = self.monitor_resource_usage(tool_id)
                    if 'statistics' in resource_data:
                        tool_info['resource_usage'] = resource_data['statistics']
                
                tools_list.append(tool_info)
            
            # Sort by name
            tools_list.sort(key=lambda x: x['name'])
            
            return tools_list
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    def get_tool_logs(self, tool_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get logs for a specific tool."""
        try:
            if tool_id not in self.tools:
                return []
            
            tool = self.tools[tool_id]
            
            logs = []
            
            # Add error logs
            for i, error in enumerate(tool.error_log[-limit:]):
                logs.append({
                    'timestamp': time.time() - (len(tool.error_log) - i) * 3600,  # Simulate timestamps
                    'level': 'ERROR',
                    'message': error,
                    'source': 'tool_manager'
                })
            
            # Add status change logs (simulated)
            status_changes = [
                {'status': 'ONLINE', 'message': 'Tool came online'},
                {'status': 'INSTALLING', 'message': 'Tool installation started'},
                {'status': 'MAINTENANCE', 'message': 'Tool undergoing maintenance'}
            ]
            
            for change in status_changes[-5:]:  # Last 5 status changes
                logs.append({
                    'timestamp': time.time() - hash(change['status']) % 86400,  # Random time in last day
                    'level': 'INFO',
                    'message': change['message'],
                    'source': 'status_monitor'
                })
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return logs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting tool logs: {e}")
            return []
    
    def check_for_updates(self, tool_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for available updates for tools."""
        try:
            update_info = {}
            
            tools_to_check = [tool_id] if tool_id else list(self.tools.keys())
            
            for tid in tools_to_check:
                if tid not in self.tools:
                    continue
                
                tool = self.tools[tid]
                
                # Simulate update check
                current_version = tool.version
                latest_version = self._get_latest_version(tool)
                
                has_update = self._compare_versions(current_version, latest_version) < 0
                
                update_info[tid] = {
                    'tool_name': tool.name,
                    'current_version': current_version,
                    'latest_version': latest_version,
                    'update_available': has_update,
                    'update_size_mb': hash(tid) % 100 + 10 if has_update else 0,  # Simulated size
                    'release_notes': f"Bug fixes and performance improvements for {tool.name}" if has_update else None
                }
                
                # Update tool status
                tool.update_available = has_update
            
            return update_info
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return {'error': str(e)}
    
    def export_tool_configuration(self, tool_id: str) -> Dict[str, Any]:
        """Export tool configuration for backup or transfer."""
        try:
            if tool_id not in self.tools:
                return {'error': 'Tool not found'}
            
            tool = self.tools[tool_id]
            
            config_export = {
                'tool_id': tool.tool_id,
                'name': tool.name,
                'description': tool.description,
                'category': tool.category.value,
                'version': tool.version,
                'capabilities': [
                    {
                        'name': cap.name,
                        'description': cap.description,
                        'parameters': cap.parameters,
                        'requirements': cap.requirements
                    }
                    for cap in tool.capabilities
                ],
                'dependencies': tool.dependencies,
                'resource_requirements': tool.resource_requirements,
                'installation_date': tool.installation_date,
                'export_timestamp': time.time(),
                'export_version': '1.0.0'
            }
            
            return config_export
            
        except Exception as e:
            logger.error(f"Error exporting tool configuration: {e}")
            return {'error': str(e)}
    
    # Helper methods
    
    def _initialize_default_tools(self):
        """Initialize default physics tools."""
        default_tools = [
            {
                'tool_id': 'quantum_simulator',
                'name': 'Quantum Circuit Simulator',
                'description': 'High-performance quantum circuit simulation',
                'category': ToolCategory.QUANTUM_SIMULATION,
                'version': '2.1.0',
                'capabilities': [
                    ToolCapability(
                        name='simulate_circuit',
                        description='Simulate quantum circuit',
                        parameters={'qubits': 'int', 'gates': 'list', 'shots': 'int'},
                        requirements={'memory_gb': 4, 'cpu_cores': 2}
                    ),
                    ToolCapability(
                        name='noise_modeling',
                        description='Add noise models to simulation',
                        parameters={'noise_type': 'string', 'error_rate': 'float'},
                        requirements={'memory_gb': 2}
                    )
                ]
            },
            {
                'tool_id': 'molecular_dynamics',
                'name': 'Molecular Dynamics Engine',
                'description': 'Classical and quantum molecular dynamics',
                'category': ToolCategory.MOLECULAR_DYNAMICS,
                'version': '1.8.3',
                'capabilities': [
                    ToolCapability(
                        name='run_simulation',
                        description='Run MD simulation',
                        parameters={'steps': 'int', 'temperature': 'float', 'pressure': 'float'},
                        requirements={'memory_gb': 8, 'cpu_cores': 4}
                    ),
                    ToolCapability(
                        name='energy_minimization',
                        description='Minimize system energy',
                        parameters={'method': 'string', 'tolerance': 'float'},
                        requirements={'memory_gb': 4}
                    )
                ]
            },
            {
                'tool_id': 'statistical_mechanics',
                'name': 'Statistical Mechanics Toolkit',
                'description': 'Monte Carlo and statistical analysis',
                'category': ToolCategory.STATISTICAL_PHYSICS,
                'version': '3.2.1',
                'capabilities': [
                    ToolCapability(
                        name='monte_carlo',
                        description='Monte Carlo sampling',
                        parameters={'samples': 'int', 'temperature': 'float'},
                        requirements={'memory_gb': 2, 'cpu_cores': 1}
                    ),
                    ToolCapability(
                        name='thermodynamics',
                        description='Calculate thermodynamic properties',
                        parameters={'ensemble': 'string', 'variables': 'list'},
                        requirements={'memory_gb': 1}
                    )
                ]
            },
            {
                'tool_id': 'data_analysis',
                'name': 'Physics Data Analysis Suite',
                'description': 'Statistical analysis and fitting tools',
                'category': ToolCategory.DATA_ANALYSIS,
                'version': '2.5.0',
                'capabilities': [
                    ToolCapability(
                        name='curve_fitting',
                        description='Fit data to theoretical models',
                        parameters={'data': 'array', 'model': 'string', 'parameters': 'dict'},
                        requirements={'memory_gb': 1}
                    ),
                    ToolCapability(
                        name='uncertainty_analysis',
                        description='Propagate uncertainties',
                        parameters={'measurements': 'array', 'correlations': 'matrix'},
                        requirements={'memory_gb': 2}
                    )
                ]
            }
        ]
        
        for tool_data in default_tools:
            capabilities = tool_data.pop('capabilities')
            
            tool_config = ToolConfiguration(
                status=ToolStatus.ONLINE,
                capabilities=capabilities,
                dependencies=[],
                resource_requirements={},
                **tool_data
            )
            
            self.register_tool(tool_config)
    
    def _start_resource_monitoring(self, tool_id: str):
        """Start resource monitoring for a tool."""
        self.resource_monitors[tool_id] = {
            'tool_id': tool_id,
            'started_at': time.time(),
            'measurements': [],
            'statistics': {}
        }
    
    def _simulate_tool_installation(self, tool: ToolConfiguration, version: str,
                                  config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate tool installation process."""
        # Simulate installation delay
        import random
        installation_time = random.uniform(2, 10)  # 2-10 seconds
        
        # Simulate success/failure
        success_rate = 0.9  # 90% success rate
        success = random.random() < success_rate
        
        if success:
            return {
                'success': True,
                'installation_time': installation_time,
                'installation_path': f'/opt/physics_tools/{tool.tool_id}',
                'version': version,
                'message': f'Successfully installed {tool.name} v{version}'
            }
        else:
            return {
                'success': False,
                'error': 'Dependency conflict detected',
                'installation_time': installation_time
            }
    
    def _perform_health_check(self, tool: ToolConfiguration) -> Dict[str, Any]:
        """Perform health check on a tool."""
        # Simulate health check
        import random
        
        health_score = random.uniform(0.7, 1.0)  # 70-100% health
        healthy = health_score > 0.8
        
        return {
            'healthy': healthy,
            'health_score': health_score,
            'response_time_ms': random.uniform(10, 100),
            'memory_usage_mb': random.uniform(50, 500),
            'error': None if healthy else 'High memory usage detected'
        }
    
    def _get_tool_resource_usage(self, tool_id: str) -> Dict[str, Any]:
        """Get current resource usage for a tool."""
        if tool_id in self.resource_monitors:
            monitor_data = self.monitor_resource_usage(tool_id)
            return monitor_data.get('statistics', {})
        return {}
    
    def _get_gpu_information(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        try:
            # Try to get GPU info (simplified)
            return {
                'available': False,
                'count': 0,
                'memory_total': 0,
                'memory_used': 0
            }
        except:
            return {'available': False, 'error': 'GPU information not available'}
    
    def _validate_capability_parameters(self, capability: ToolCapability,
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for a capability."""
        errors = []
        
        # Check required parameters
        for param_name, param_type in capability.parameters.items():
            if param_name not in parameters:
                errors.append(f"Missing required parameter: {param_name}")
                continue
            
            # Basic type checking
            value = parameters[param_name]
            if param_type == 'int' and not isinstance(value, int):
                errors.append(f"Parameter {param_name} must be an integer")
            elif param_type == 'float' and not isinstance(value, (int, float)):
                errors.append(f"Parameter {param_name} must be a number")
            elif param_type == 'string' and not isinstance(value, str):
                errors.append(f"Parameter {param_name} must be a string")
            elif param_type == 'list' and not isinstance(value, list):
                errors.append(f"Parameter {param_name} must be a list")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _execute_capability(self, tool: ToolConfiguration, capability: ToolCapability,
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool capability."""
        # Simulate capability execution
        import random
        
        execution_time = random.uniform(0.5, 5.0)  # 0.5-5 seconds
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            # Generate mock results based on capability
            if capability.name == 'simulate_circuit':
                results = {
                    'state_vector': [random.random() for _ in range(2**parameters.get('qubits', 2))],
                    'measurement_counts': {f'{i:0{parameters.get("qubits", 2)}b}': random.randint(0, 100) 
                                         for i in range(2**parameters.get('qubits', 2))}
                }
            elif capability.name == 'run_simulation':
                results = {
                    'final_energy': random.uniform(-1000, -500),
                    'trajectory': [[random.uniform(-10, 10) for _ in range(3)] for _ in range(100)]
                }
            else:
                results = {'output': f'Results from {capability.name}', 'value': random.random()}
            
            return {
                'success': True,
                'results': results,
                'execution_time': execution_time,
                'resource_usage': {
                    'cpu_time': execution_time * 0.8,
                    'memory_peak_mb': random.uniform(10, 100)
                }
            }
        else:
            return {
                'success': False,
                'error': 'Capability execution failed',
                'execution_time': execution_time
            }
    
    def _get_latest_version(self, tool: ToolConfiguration) -> str:
        """Get latest version for a tool (simulated)."""
        # Simulate version incrementing
        version_parts = tool.version.split('.')
        if len(version_parts) >= 3:
            # Increment patch version
            patch = int(version_parts[2]) + 1
            return f"{version_parts[0]}.{version_parts[1]}.{patch}"
        return tool.version
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad with zeros to make same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            
            return 0
        except:
            # Fallback to string comparison
            return -1 if version1 < version2 else (1 if version1 > version2 else 0)

# Global tool management instance
physics_tool_management = PhysicsToolManagement()