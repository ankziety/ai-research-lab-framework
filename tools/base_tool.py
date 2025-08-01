"""
Base Tool Class

Defines the interface for all research tools that agents can request and use.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all research tools.
    
    Tools are capabilities that agents can discover and use to conduct research,
    run experiments, analyze data, or collaborate with other agents.
    """
    
    def __init__(self, tool_id: str, name: str, description: str, 
                 capabilities: List[str], requirements: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.
        
        Args:
            tool_id: Unique identifier for the tool
            name: Human-readable name
            description: Description of what the tool does
            capabilities: List of capabilities this tool provides
            requirements: Optional requirements (dependencies, resources, etc.)
        """
        self.tool_id = tool_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.requirements = requirements or {}
        self.usage_count = 0
        self.success_rate = 1.0
        
    @abstractmethod
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given task parameters.
        
        Args:
            task: Task specification with parameters
            context: Execution context (agent info, session data, etc.)
            
        Returns:
            Results dictionary with output data and metadata
        """
        pass
    
    @abstractmethod
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """
        Assess if this tool can handle a specific task.
        
        Args:
            task_type: Type of task being requested
            requirements: Task requirements and constraints
            
        Returns:
            Confidence score (0.0-1.0) indicating tool's ability to handle task
        """
        pass
    
    def get_usage_instructions(self) -> str:
        """Get instructions for using this tool."""
        return f"""
        Tool: {self.name}
        Description: {self.description}
        Capabilities: {', '.join(self.capabilities)}
        
        Usage Statistics:
        - Used {self.usage_count} times
        - Success rate: {self.success_rate:.2%}
        
        Requirements: {self.requirements}
        """
    
    def update_statistics(self, success: bool):
        """Update tool usage statistics."""
        self.usage_count += 1
        if success:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1.0) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count
    
    def validate_requirements(self, context: Dict[str, Any]) -> bool:
        """
        Validate that all tool requirements are met in the given context.
        
        Args:
            context: Execution context to validate against
            
        Returns:
            True if all requirements are satisfied
        """
        for req_type, req_value in self.requirements.items():
            if req_type == 'min_memory' and context.get('available_memory', 0) < req_value:
                return False
            elif req_type == 'required_packages' and not all(
                pkg in context.get('available_packages', []) for pkg in req_value
            ):
                return False
            elif req_type == 'api_keys' and not all(
                key in context.get('api_keys', {}) for key in req_value
            ):
                return False
                
        return True