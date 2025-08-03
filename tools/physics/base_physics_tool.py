"""
Base Physics Tool

Abstract base class for physics tools that agents can request and use.
Extends the framework's BaseTool with physics-specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging
import asyncio
from datetime import datetime

from ..base_tool import BaseTool

logger = logging.getLogger(__name__)


class BasePhysicsTool(BaseTool, ABC):
    """
    Abstract base class for physics tools that agents can request.
    
    Provides common physics tool functionality including:
    - Cost estimation for computational resources
    - Physics-specific error handling
    - Result formatting for agents
    - Resource requirement validation
    - Integration with agent marketplace
    """
    
    def __init__(self, 
                 tool_id: str,
                 name: str, 
                 description: str,
                 physics_domain: str,
                 computational_cost_factor: float = 1.0,
                 software_requirements: Optional[List[str]] = None,
                 hardware_requirements: Optional[Dict[str, Any]] = None):
        """
        Initialize physics tool.
        
        Args:
            tool_id: Unique identifier for the tool
            name: Human-readable name
            description: Tool description for agents
            physics_domain: Physics domain (e.g., 'quantum_chemistry', 'materials')
            computational_cost_factor: Relative computational cost multiplier
            software_requirements: Required software packages
            hardware_requirements: Hardware requirements (memory, GPU, etc.)
        """
        # Base capabilities all physics tools provide
        physics_capabilities = [
            "physics_calculation",
            "cost_estimation", 
            "result_formatting",
            "error_handling",
            "agent_integration"
        ]
        
        # Combine physics requirements with tool-specific requirements
        requirements = {
            "required_packages": software_requirements or [],
            "physics_domain": physics_domain,
            "computational_cost_factor": computational_cost_factor
        }
        
        if hardware_requirements:
            requirements.update(hardware_requirements)
        
        super().__init__(
            tool_id=tool_id,
            name=name,
            description=description,
            capabilities=physics_capabilities,
            requirements=requirements
        )
        
        self.physics_domain = physics_domain
        self.computational_cost_factor = computational_cost_factor
        self.software_requirements = software_requirements or []
        self.hardware_requirements = hardware_requirements or {}
        
        # Physics tool specific tracking
        self.calculation_history = []
        self.average_calculation_time = 0.0
        self.total_computational_cost = 0.0
    
    @abstractmethod
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute physics calculation requested by an agent.
        
        Args:
            task: Task specification with physics parameters
            context: Agent context and execution environment
            
        Returns:
            Results dictionary formatted for agents
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate agent input parameters for physics calculations.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        pass
    
    @abstractmethod
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format physics calculation results for agents.
        
        Args:
            output_data: Raw physics calculation results
            
        Returns:
            Agent-friendly formatted results
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for agent decision making.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        pass
    
    @abstractmethod
    def get_physics_requirements(self) -> Dict[str, Any]:
        """
        Get physics-specific requirements that agents need to know.
        
        Returns:
            Physics requirements dictionary
        """
        pass
    
    def handle_errors(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle physics tool specific errors and report to agents.
        
        Args:
            error: Exception that occurred
            context: Error context
            
        Returns:
            Agent-friendly error report
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Categorize physics-specific errors
        if "memory" in error_message.lower():
            error_category = "insufficient_memory"
            suggestion = "Try reducing calculation size or request more memory"
        elif "convergence" in error_message.lower():
            error_category = "convergence_failure"
            suggestion = "Adjust convergence criteria or initial conditions"
        elif "package" in error_message.lower() or "import" in error_message.lower():
            error_category = "missing_software"
            suggestion = f"Install required software: {', '.join(self.software_requirements)}"
        else:
            error_category = "calculation_error"
            suggestion = "Check input parameters and calculation setup"
        
        error_report = {
            "success": False,
            "error_type": error_type,
            "error_category": error_category,
            "error_message": error_message,
            "suggestion": suggestion,
            "physics_domain": self.physics_domain,
            "tool_id": self.tool_id,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        logger.error(f"Physics tool error in {self.tool_id}: {error_message}")
        return error_report
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """
        Assess if this physics tool can handle a specific task.
        
        Args:
            task_type: Type of task being requested
            requirements: Task requirements and constraints
            
        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.0
        
        # Check if task type matches physics domain
        if self.physics_domain.lower() in task_type.lower():
            confidence += 0.5
        
        # Check for physics keywords
        physics_keywords = [
            "calculation", "simulation", "analysis", "compute", 
            "model", "solve", "optimize", "predict"
        ]
        
        for keyword in physics_keywords:
            if keyword in task_type.lower():
                confidence += 0.1
                
        # Check specific domain keywords
        domain_keywords = self._get_domain_keywords()
        for keyword in domain_keywords:
            if keyword in task_type.lower():
                confidence += 0.2
        
        # Penalize if requirements can't be met
        if requirements:
            if not self.validate_requirements(requirements):
                confidence *= 0.5
        
        return min(1.0, confidence)
    
    def get_agent_requirements(self) -> Dict[str, Any]:
        """
        Get requirements that agents need to know about.
        
        Returns:
            Agent-friendly requirements
        """
        return {
            "physics_domain": self.physics_domain,
            "software_requirements": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "computational_cost_factor": self.computational_cost_factor,
            "average_calculation_time": self.average_calculation_time,
            "success_rate": self.success_rate,
            "usage_guidelines": self._get_usage_guidelines()
        }
    
    def update_calculation_stats(self, calculation_time: float, cost: float, success: bool):
        """
        Update calculation statistics for cost estimation.
        
        Args:
            calculation_time: Time taken for calculation
            cost: Computational cost
            success: Whether calculation succeeded
        """
        self.calculation_history.append({
            "time": calculation_time,
            "cost": cost,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update averages
        successful_calculations = [c for c in self.calculation_history if c["success"]]
        if successful_calculations:
            self.average_calculation_time = sum(c["time"] for c in successful_calculations) / len(successful_calculations)
        
        self.total_computational_cost += cost
        self.update_statistics(success)
    
    async def execute_async(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous execution for long-running physics calculations.
        
        Args:
            task: Task specification
            context: Execution context
            
        Returns:
            Results dictionary
        """
        # Default implementation runs synchronous execute in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, task, context)
    
    @abstractmethod
    def _get_domain_keywords(self) -> List[str]:
        """Get physics domain specific keywords for task matching."""
        pass
    
    def _get_usage_guidelines(self) -> str:
        """Get usage guidelines for agents."""
        return f"""
        Physics Tool: {self.name}
        Domain: {self.physics_domain}
        
        Usage Guidelines:
        1. Provide clear physics parameters in task specification
        2. Check cost estimates before running expensive calculations
        3. Validate input parameters using validate_input method
        4. Monitor calculation progress for long-running tasks
        5. Handle errors gracefully using provided error information
        
        Average Calculation Time: {self.average_calculation_time:.2f} seconds
        Success Rate: {self.success_rate:.2%}
        Total Calculations: {self.usage_count}
        """
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get detailed calculation statistics."""
        return {
            "tool_id": self.tool_id,
            "physics_domain": self.physics_domain,
            "total_calculations": len(self.calculation_history),
            "successful_calculations": len([c for c in self.calculation_history if c["success"]]),
            "average_calculation_time": self.average_calculation_time,
            "total_computational_cost": self.total_computational_cost,
            "success_rate": self.success_rate,
            "recent_calculations": self.calculation_history[-10:] if self.calculation_history else []
        }