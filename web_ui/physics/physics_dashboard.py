"""
Physics Dashboard Component

This module provides a comprehensive dashboard for physics research monitoring,
including simulation status, agent activity, and research progress tracking.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SimulationMetrics:
    """Metrics for physics simulations."""
    total_simulations: int = 0
    active_simulations: int = 0
    completed_simulations: int = 0
    failed_simulations: int = 0
    average_runtime: float = 0.0
    success_rate: float = 0.0
    resource_utilization: float = 0.0

@dataclass
class ExperimentMetrics:
    """Metrics for physics experiments."""
    total_experiments: int = 0
    pending_experiments: int = 0
    running_experiments: int = 0
    completed_experiments: int = 0
    failed_experiments: int = 0
    average_completion_time: float = 0.0
    throughput: float = 0.0

@dataclass
class AgentActivity:
    """Physics agent activity tracking."""
    agent_id: str
    agent_name: str
    current_task: str
    status: str  # 'idle', 'working', 'analyzing', 'simulating'
    specialization: str  # 'quantum_physics', 'condensed_matter', 'astrophysics', etc.
    progress: float = 0.0
    last_update: float = 0.0

class PhysicsDashboard:
    """
    Physics Research Dashboard
    
    Provides comprehensive monitoring and control interface for physics research activities,
    including simulations, experiments, agent coordination, and progress tracking.
    """
    
    def __init__(self):
        self.simulation_metrics = SimulationMetrics()
        self.experiment_metrics = ExperimentMetrics()
        self.active_agents: Dict[str, AgentActivity] = {}
        self.research_progress = {
            'current_phase': 'initialization',
            'overall_progress': 0.0,
            'phase_progress': 0.0,
            'estimated_completion': None,
            'milestones': []
        }
        self.real_time_data = {
            'system_load': 0.0,
            'memory_usage': 0.0,
            'gpu_utilization': 0.0,
            'network_io': 0.0,
            'disk_io': 0.0
        }
        
    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview."""
        try:
            overview = {
                'simulations': {
                    'total': self.simulation_metrics.total_simulations,
                    'active': self.simulation_metrics.active_simulations,
                    'completed': self.simulation_metrics.completed_simulations,
                    'failed': self.simulation_metrics.failed_simulations,
                    'success_rate': self.simulation_metrics.success_rate,
                    'average_runtime': self.simulation_metrics.average_runtime
                },
                'experiments': {
                    'total': self.experiment_metrics.total_experiments,
                    'pending': self.experiment_metrics.pending_experiments,
                    'running': self.experiment_metrics.running_experiments,
                    'completed': self.experiment_metrics.completed_experiments,
                    'failed': self.experiment_metrics.failed_experiments,
                    'throughput': self.experiment_metrics.throughput
                },
                'agents': {
                    'total': len(self.active_agents),
                    'active': len([a for a in self.active_agents.values() if a.status != 'idle']),
                    'specializations': list(set([a.specialization for a in self.active_agents.values()])),
                    'current_tasks': len([a for a in self.active_agents.values() if a.current_task])
                },
                'research_progress': self.research_progress,
                'system_status': self.real_time_data,
                'timestamp': time.time()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {'error': str(e)}
    
    def update_simulation_metrics(self, metrics_update: Dict[str, Any]) -> bool:
        """Update simulation metrics."""
        try:
            for key, value in metrics_update.items():
                if hasattr(self.simulation_metrics, key):
                    setattr(self.simulation_metrics, key, value)
            
            # Calculate derived metrics
            total = self.simulation_metrics.total_simulations
            if total > 0:
                self.simulation_metrics.success_rate = (
                    self.simulation_metrics.completed_simulations / total
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating simulation metrics: {e}")
            return False
    
    def update_experiment_metrics(self, metrics_update: Dict[str, Any]) -> bool:
        """Update experiment metrics."""
        try:
            for key, value in metrics_update.items():
                if hasattr(self.experiment_metrics, key):
                    setattr(self.experiment_metrics, key, value)
            
            # Calculate throughput
            completed = self.experiment_metrics.completed_experiments
            if completed > 0 and self.experiment_metrics.average_completion_time > 0:
                self.experiment_metrics.throughput = (
                    completed / self.experiment_metrics.average_completion_time
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating experiment metrics: {e}")
            return False
    
    def register_agent(self, agent_id: str, agent_name: str, specialization: str) -> bool:
        """Register a new physics agent."""
        try:
            self.active_agents[agent_id] = AgentActivity(
                agent_id=agent_id,
                agent_name=agent_name,
                current_task="",
                status="idle",
                specialization=specialization,
                progress=0.0,
                last_update=time.time()
            )
            
            logger.info(f"Registered physics agent: {agent_name} ({specialization})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return False
    
    def update_agent_activity(self, agent_id: str, activity_update: Dict[str, Any]) -> bool:
        """Update agent activity and status."""
        try:
            if agent_id not in self.active_agents:
                logger.warning(f"Unknown agent ID: {agent_id}")
                return False
            
            agent = self.active_agents[agent_id]
            
            # Update agent properties
            for key, value in activity_update.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            agent.last_update = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent activity: {e}")
            return False
    
    def get_agent_activities(self) -> List[Dict[str, Any]]:
        """Get current agent activities."""
        try:
            activities = []
            
            for agent in self.active_agents.values():
                activity = {
                    'agent_id': agent.agent_id,
                    'agent_name': agent.agent_name,
                    'current_task': agent.current_task,
                    'status': agent.status,
                    'specialization': agent.specialization,
                    'progress': agent.progress,
                    'last_update': agent.last_update,
                    'time_since_update': time.time() - agent.last_update
                }
                activities.append(activity)
            
            # Sort by last update (most recent first)
            activities.sort(key=lambda x: x['last_update'], reverse=True)
            
            return activities
            
        except Exception as e:
            logger.error(f"Error getting agent activities: {e}")
            return []
    
    def update_research_progress(self, phase: str, overall_progress: float, 
                               phase_progress: float, milestones: List[str] = None) -> bool:
        """Update research progress tracking."""
        try:
            self.research_progress['current_phase'] = phase
            self.research_progress['overall_progress'] = max(0.0, min(100.0, overall_progress))
            self.research_progress['phase_progress'] = max(0.0, min(100.0, phase_progress))
            
            if milestones:
                self.research_progress['milestones'] = milestones
            
            # Estimate completion time
            if overall_progress > 0:
                remaining_progress = 100.0 - overall_progress
                if remaining_progress > 0:
                    # Simple linear estimation
                    estimated_time = (remaining_progress / overall_progress) * time.time()
                    self.research_progress['estimated_completion'] = time.time() + estimated_time
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating research progress: {e}")
            return False
    
    def update_system_metrics(self, system_data: Dict[str, float]) -> bool:
        """Update real-time system metrics."""
        try:
            for key, value in system_data.items():
                if key in self.real_time_data:
                    self.real_time_data[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        try:
            summary = {
                'simulation_performance': {
                    'success_rate': self.simulation_metrics.success_rate,
                    'average_runtime': self.simulation_metrics.average_runtime,
                    'resource_efficiency': self.simulation_metrics.resource_utilization,
                    'active_count': self.simulation_metrics.active_simulations
                },
                'experiment_performance': {
                    'completion_rate': (
                        self.experiment_metrics.completed_experiments / 
                        max(1, self.experiment_metrics.total_experiments)
                    ),
                    'throughput': self.experiment_metrics.throughput,
                    'queue_length': self.experiment_metrics.pending_experiments,
                    'average_time': self.experiment_metrics.average_completion_time
                },
                'agent_performance': {
                    'utilization_rate': len([a for a in self.active_agents.values() 
                                           if a.status != 'idle']) / max(1, len(self.active_agents)),
                    'active_specializations': len(set([a.specialization for a in self.active_agents.values()])),
                    'average_progress': sum([a.progress for a in self.active_agents.values()]) / 
                                     max(1, len(self.active_agents))
                },
                'system_performance': {
                    'overall_load': (
                        self.real_time_data['system_load'] + 
                        self.real_time_data['memory_usage'] + 
                        self.real_time_data['gpu_utilization']
                    ) / 3.0,
                    'bottlenecks': self._identify_bottlenecks(),
                    'efficiency_score': self._calculate_efficiency_score()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []
        
        if self.real_time_data['system_load'] > 80:
            bottlenecks.append('cpu_overload')
        if self.real_time_data['memory_usage'] > 85:
            bottlenecks.append('memory_pressure')
        if self.real_time_data['gpu_utilization'] > 90:
            bottlenecks.append('gpu_saturation')
        if self.real_time_data['disk_io'] > 75:
            bottlenecks.append('disk_io_limit')
        
        # Check for agent coordination issues
        idle_agents = len([a for a in self.active_agents.values() if a.status == 'idle'])
        if idle_agents > len(self.active_agents) * 0.5:
            bottlenecks.append('agent_underutilization')
        
        return bottlenecks
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score."""
        try:
            # System resource efficiency
            avg_utilization = (
                self.real_time_data['system_load'] + 
                self.real_time_data['memory_usage'] + 
                self.real_time_data['gpu_utilization']
            ) / 3.0
            
            # Optimal utilization is around 70-80%
            resource_efficiency = 1.0 - abs(75.0 - avg_utilization) / 75.0
            
            # Agent efficiency
            active_agents = len([a for a in self.active_agents.values() if a.status != 'idle'])
            agent_efficiency = active_agents / max(1, len(self.active_agents))
            
            # Simulation efficiency
            sim_efficiency = self.simulation_metrics.success_rate
            
            # Weighted combination
            efficiency_score = (
                resource_efficiency * 0.3 + 
                agent_efficiency * 0.4 + 
                sim_efficiency * 0.3
            ) * 100.0
            
            return max(0.0, min(100.0, efficiency_score))
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.0
    
    def get_recent_activities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent physics research activities."""
        try:
            activities = []
            
            # Recent agent activities
            for agent in self.active_agents.values():
                if agent.current_task and agent.last_update > time.time() - 3600:  # Last hour
                    activities.append({
                        'type': 'agent_activity',
                        'timestamp': agent.last_update,
                        'description': f"{agent.agent_name} ({agent.specialization}): {agent.current_task}",
                        'status': agent.status,
                        'progress': agent.progress
                    })
            
            # Sort by timestamp (most recent first)
            activities.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent activities: {e}")
            return []
    
    def export_dashboard_data(self, format_type: str = 'json') -> Dict[str, Any]:
        """Export dashboard data for analysis."""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'export_format': format_type,
                'dashboard_overview': self.get_dashboard_overview(),
                'performance_summary': self.get_performance_summary(),
                'agent_activities': self.get_agent_activities(),
                'recent_activities': self.get_recent_activities(50),
                'system_metrics': self.real_time_data,
                'metadata': {
                    'version': '1.0.0',
                    'generated_by': 'physics_dashboard'
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return {'error': str(e)}

# Global dashboard instance
physics_dashboard = PhysicsDashboard()

# Initialize with default physics agents
default_agents = [
    ('quantum_simulator_agent', 'Quantum Simulator Agent', 'quantum_physics'),
    ('molecular_dynamics_agent', 'Molecular Dynamics Agent', 'condensed_matter'),
    ('statistical_analysis_agent', 'Statistical Analysis Agent', 'statistical_physics'),
    ('data_visualization_agent', 'Data Visualization Agent', 'data_analysis')
]

for agent_id, agent_name, specialization in default_agents:
    physics_dashboard.register_agent(agent_id, agent_name, specialization)