"""
Physics Tool Factory

Factory for creating and configuring physics tools with specific parameters.
Provides centralized creation, configuration, and integration management
for physics tools that agents can request and use.
"""

from typing import Dict, Any, List, Optional, Type, Union
import logging
from datetime import datetime

from .base_physics_tool import BasePhysicsTool
from .physics_tool_registry import PhysicsToolRegistry

logger = logging.getLogger(__name__)


class PhysicsToolFactory:
    """
    Factory for creating and configuring physics research tools.
    
    Provides centralized management for tool creation, configuration,
    and integration with the existing framework infrastructure.
    """
    
    def __init__(self, registry: Optional[PhysicsToolRegistry] = None):
        """
        Initialize the physics tool factory.
        
        Args:
            registry: Optional physics tool registry for automatic registration
        """
        self.registry = registry
        self.tool_classes: Dict[str, Type[BasePhysicsTool]] = {}
        self.tool_configurations: Dict[str, Dict[str, Any]] = {}
        self.creation_history: List[Dict[str, Any]] = []
        
        # Default tool configurations
        self.default_configurations = {
            "quantum_chemistry_tool": {
                "computational_cost_factor": 3.0,
                "software_requirements": ["numpy", "scipy"],
                "hardware_requirements": {"min_memory": 2048, "cpu_cores": 4}
            },
            "materials_science_tool": {
                "computational_cost_factor": 2.5,
                "software_requirements": ["numpy", "scipy", "matplotlib"],
                "hardware_requirements": {"min_memory": 1024, "cpu_cores": 2}
            },
            "astrophysics_tool": {
                "computational_cost_factor": 2.0,
                "software_requirements": ["numpy", "scipy", "matplotlib"],
                "hardware_requirements": {"min_memory": 512, "cpu_cores": 2}
            },
            "experimental_tool": {
                "computational_cost_factor": 1.5,
                "software_requirements": ["numpy", "scipy", "pandas", "matplotlib"],
                "hardware_requirements": {"min_memory": 256, "cpu_cores": 1}
            },
            "visualization_tool": {
                "computational_cost_factor": 1.0,
                "software_requirements": ["matplotlib", "numpy", "seaborn"],
                "hardware_requirements": {"min_memory": 256, "cpu_cores": 1}
            }
        }
        
        self._register_default_tool_classes()
    
    def _register_default_tool_classes(self):
        """Register default physics tool classes."""
        try:
            from .quantum_chemistry_tool import QuantumChemistryTool
            from .materials_science_tool import MaterialsScienceTool
            from .astrophysics_tool import AstrophysicsTool
            from .experimental_tool import ExperimentalTool
            from .visualization_tool import VisualizationTool
            
            self.tool_classes = {
                "quantum_chemistry_tool": QuantumChemistryTool,
                "materials_science_tool": MaterialsScienceTool,
                "astrophysics_tool": AstrophysicsTool,
                "experimental_tool": ExperimentalTool,
                "visualization_tool": VisualizationTool
            }
            
            logger.info(f"Registered {len(self.tool_classes)} physics tool classes")
            
        except ImportError as e:
            logger.error(f"Failed to import physics tool classes: {e}")
    
    def create_tool(self, 
                   tool_type: str,
                   configuration: Optional[Dict[str, Any]] = None,
                   auto_register: bool = True) -> BasePhysicsTool:
        """
        Create a physics tool with specified configuration.
        
        Args:
            tool_type: Type of physics tool to create
            configuration: Optional tool-specific configuration
            auto_register: Whether to automatically register with registry
            
        Returns:
            Configured physics tool instance
        """
        if tool_type not in self.tool_classes:
            raise ValueError(f"Unknown tool type: {tool_type}. "
                           f"Available types: {list(self.tool_classes.keys())}")
        
        # Merge configurations
        tool_config = self.default_configurations.get(tool_type, {}).copy()
        if configuration:
            tool_config.update(configuration)
        
        # Create tool instance
        tool_class = self.tool_classes[tool_type]
        
        try:
            # Create tool with configuration if constructor supports it
            if hasattr(tool_class, '_configure_from_dict'):
                tool = tool_class()
                tool._configure_from_dict(tool_config)
            else:
                # Standard creation
                tool = tool_class()
                
                # Apply configuration post-creation
                self._apply_configuration(tool, tool_config)
            
            # Record creation
            creation_record = {
                "tool_type": tool_type,
                "tool_id": tool.tool_id,
                "configuration": tool_config,
                "created_at": datetime.now().isoformat(),
                "auto_registered": auto_register
            }
            self.creation_history.append(creation_record)
            
            # Auto-register if requested and registry available
            if auto_register and self.registry:
                self._auto_register_tool(tool, tool_type)
            
            logger.info(f"Created physics tool: {tool.name} ({tool.tool_id})")
            return tool
            
        except Exception as e:
            logger.error(f"Failed to create tool {tool_type}: {e}")
            raise
    
    def create_custom_tool(self,
                          tool_class: Type[BasePhysicsTool],
                          tool_id: str,
                          name: str,
                          description: str,
                          physics_domain: str,
                          configuration: Optional[Dict[str, Any]] = None) -> BasePhysicsTool:
        """
        Create a custom physics tool from a user-provided class.
        
        Args:
            tool_class: Custom tool class inheriting from BasePhysicsTool
            tool_id: Unique identifier for the tool
            name: Human-readable name
            description: Tool description
            physics_domain: Physics domain the tool belongs to
            configuration: Optional configuration parameters
            
        Returns:
            Custom physics tool instance
        """
        if not issubclass(tool_class, BasePhysicsTool):
            raise TypeError("Tool class must inherit from BasePhysicsTool")
        
        # Create instance with custom parameters
        try:
            tool = tool_class()
            
            # Override basic properties
            tool.tool_id = tool_id
            tool.name = name
            tool.description = description
            tool.physics_domain = physics_domain
            
            # Apply configuration
            if configuration:
                self._apply_configuration(tool, configuration)
            
            # Record creation
            creation_record = {
                "tool_type": "custom",
                "tool_class": tool_class.__name__,
                "tool_id": tool_id,
                "configuration": configuration or {},
                "created_at": datetime.now().isoformat(),
                "auto_registered": False
            }
            self.creation_history.append(creation_record)
            
            logger.info(f"Created custom physics tool: {name} ({tool_id})")
            return tool
            
        except Exception as e:
            logger.error(f"Failed to create custom tool: {e}")
            raise
    
    def create_tool_suite(self,
                         suite_name: str,
                         tool_specifications: List[Dict[str, Any]],
                         shared_configuration: Optional[Dict[str, Any]] = None) -> Dict[str, BasePhysicsTool]:
        """
        Create a suite of related physics tools.
        
        Args:
            suite_name: Name for the tool suite
            tool_specifications: List of tool specifications
            shared_configuration: Configuration applied to all tools
            
        Returns:
            Dictionary of created tools {tool_id: tool_instance}
        """
        tools = {}
        shared_config = shared_configuration or {}
        
        for spec in tool_specifications:
            tool_type = spec["type"]
            tool_config = shared_config.copy()
            tool_config.update(spec.get("configuration", {}))
            
            try:
                tool = self.create_tool(
                    tool_type=tool_type,
                    configuration=tool_config,
                    auto_register=spec.get("auto_register", True)
                )
                tools[tool.tool_id] = tool
                
            except Exception as e:
                logger.error(f"Failed to create tool {tool_type} in suite {suite_name}: {e}")
                continue
        
        logger.info(f"Created tool suite '{suite_name}' with {len(tools)} tools")
        return tools
    
    def configure_tool_for_agent(self,
                                tool: BasePhysicsTool,
                                agent_profile: Dict[str, Any]) -> BasePhysicsTool:
        """
        Configure a physics tool for a specific agent's needs.
        
        Args:
            tool: Physics tool to configure
            agent_profile: Agent profile with preferences and constraints
            
        Returns:
            Configured tool instance
        """
        # Extract agent preferences
        cost_preference = agent_profile.get("cost_preference", "balanced")  # "low", "balanced", "high"
        accuracy_preference = agent_profile.get("accuracy_preference", "balanced")
        speed_preference = agent_profile.get("speed_preference", "balanced")
        
        # Adjust tool configuration based on preferences
        if cost_preference == "low":
            # Reduce computational cost factor
            tool.computational_cost_factor *= 0.7
        elif cost_preference == "high":
            # Allow higher computational cost for better results
            tool.computational_cost_factor *= 1.3
        
        # Apply accuracy preferences (tool-specific logic)
        if hasattr(tool, '_configure_accuracy'):
            tool._configure_accuracy(accuracy_preference)
        
        # Apply speed preferences
        if hasattr(tool, '_configure_speed'):
            tool._configure_speed(speed_preference)
        
        logger.info(f"Configured tool {tool.tool_id} for agent preferences")
        return tool
    
    def create_tool_from_template(self,
                                template_name: str,
                                customizations: Optional[Dict[str, Any]] = None) -> BasePhysicsTool:
        """
        Create a tool from a predefined template.
        
        Args:
            template_name: Name of the template
            customizations: Optional customizations to apply
            
        Returns:
            Tool created from template
        """
        templates = self._get_tool_templates()
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available templates: {list(templates.keys())}")
        
        template = templates[template_name]
        
        # Merge template configuration with customizations
        config = template["configuration"].copy()
        if customizations:
            config.update(customizations)
        
        return self.create_tool(
            tool_type=template["tool_type"],
            configuration=config,
            auto_register=template.get("auto_register", True)
        )
    
    def optimize_tool_for_task(self,
                              tool_type: str,
                              task_specification: Dict[str, Any]) -> BasePhysicsTool:
        """
        Create and optimize a tool for a specific task.
        
        Args:
            tool_type: Type of physics tool
            task_specification: Detailed task requirements
            
        Returns:
            Optimized tool instance
        """
        # Analyze task requirements
        task_analysis = self._analyze_task_requirements(task_specification)
        
        # Create optimized configuration
        optimized_config = self._create_optimized_configuration(
            tool_type, task_analysis
        )
        
        # Create tool with optimized configuration
        tool = self.create_tool(
            tool_type=tool_type,
            configuration=optimized_config,
            auto_register=True
        )
        
        logger.info(f"Created optimized tool {tool.tool_id} for task: "
                   f"{task_specification.get('type', 'unknown')}")
        
        return tool
    
    def batch_create_tools(self,
                          creation_requests: List[Dict[str, Any]]) -> List[BasePhysicsTool]:
        """
        Create multiple tools in batch.
        
        Args:
            creation_requests: List of tool creation requests
            
        Returns:
            List of created tools
        """
        created_tools = []
        
        for request in creation_requests:
            try:
                tool = self.create_tool(
                    tool_type=request["tool_type"],
                    configuration=request.get("configuration"),
                    auto_register=request.get("auto_register", True)
                )
                created_tools.append(tool)
                
            except Exception as e:
                logger.error(f"Failed to create tool in batch: {e}")
                continue
        
        logger.info(f"Batch created {len(created_tools)} tools")
        return created_tools
    
    def validate_tool_configuration(self,
                                  tool_type: str,
                                  configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool configuration before creation.
        
        Args:
            tool_type: Type of physics tool
            configuration: Configuration to validate
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check if tool type exists
        if tool_type not in self.tool_classes:
            errors.append(f"Unknown tool type: {tool_type}")
            return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}
        
        # Get default configuration for comparison
        default_config = self.default_configurations.get(tool_type, {})
        
        # Validate configuration parameters
        for key, value in configuration.items():
            if key in default_config:
                default_value = default_config[key]
                
                # Type checking
                if type(value) != type(default_value):
                    warnings.append(f"Parameter '{key}' type mismatch: "
                                  f"expected {type(default_value).__name__}, "
                                  f"got {type(value).__name__}")
                
                # Range checking for numerical values
                if isinstance(value, (int, float)):
                    if key == "computational_cost_factor":
                        if value < 0.1 or value > 10.0:
                            warnings.append(f"Computational cost factor {value} is outside typical range [0.1, 10.0]")
                    elif "memory" in key:
                        if value < 64:  # MB
                            warnings.append(f"Memory requirement {value} MB seems very low")
                        elif value > 32768:  # 32 GB
                            warnings.append(f"Memory requirement {value} MB seems very high")
            else:
                suggestions.append(f"Unknown configuration parameter: {key}")
        
        # Check for missing required parameters
        required_params = ["computational_cost_factor"]
        for param in required_params:
            if param not in configuration and param in default_config:
                suggestions.append(f"Consider specifying '{param}' parameter")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def get_available_tool_types(self) -> List[Dict[str, Any]]:
        """Get information about available tool types."""
        tool_info = []
        
        for tool_type, tool_class in self.tool_classes.items():
            # Create temporary instance to get information
            try:
                temp_tool = tool_class()
                info = {
                    "tool_type": tool_type,
                    "class_name": tool_class.__name__,
                    "physics_domain": temp_tool.physics_domain,
                    "capabilities": temp_tool.capabilities,
                    "default_configuration": self.default_configurations.get(tool_type, {}),
                    "description": temp_tool.description
                }
                tool_info.append(info)
            except Exception as e:
                logger.warning(f"Could not get info for tool type {tool_type}: {e}")
        
        return tool_info
    
    def _apply_configuration(self, tool: BasePhysicsTool, configuration: Dict[str, Any]):
        """Apply configuration to a tool instance."""
        for key, value in configuration.items():
            if hasattr(tool, key):
                setattr(tool, key, value)
            elif key in ["software_requirements", "hardware_requirements"]:
                # Update requirements
                if hasattr(tool, "requirements"):
                    tool.requirements[key] = value
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def _auto_register_tool(self, tool: BasePhysicsTool, tool_type: str):
        """Automatically register tool with registry."""
        try:
            # Determine domains for registration
            domains = [tool.physics_domain]
            
            # Add related domains based on tool type
            domain_mappings = {
                "quantum_chemistry_tool": ["quantum_chemistry", "molecular_physics"],
                "materials_science_tool": ["materials_science", "condensed_matter"],
                "astrophysics_tool": ["astrophysics", "cosmology"],
                "experimental_tool": ["experimental_physics", "data_analysis"],
                "visualization_tool": ["data_visualization", "experimental_physics"]
            }
            
            if tool_type in domain_mappings:
                domains = domain_mappings[tool_type]
            
            self.registry.register_physics_tool(tool, domains=domains)
            
        except Exception as e:
            logger.error(f"Failed to auto-register tool {tool.tool_id}: {e}")
    
    def _get_tool_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined tool templates."""
        return {
            "beginner_quantum": {
                "tool_type": "quantum_chemistry_tool",
                "configuration": {
                    "computational_cost_factor": 1.5,
                    "hardware_requirements": {"min_memory": 1024, "cpu_cores": 2}
                },
                "auto_register": True
            },
            "advanced_materials": {
                "tool_type": "materials_science_tool",
                "configuration": {
                    "computational_cost_factor": 4.0,
                    "hardware_requirements": {"min_memory": 8192, "cpu_cores": 8}
                },
                "auto_register": True
            },
            "quick_visualization": {
                "tool_type": "visualization_tool",
                "configuration": {
                    "computational_cost_factor": 0.5,
                    "software_requirements": ["matplotlib", "numpy"]
                },
                "auto_register": True
            },
            "statistical_analysis": {
                "tool_type": "experimental_tool",
                "configuration": {
                    "computational_cost_factor": 1.0,
                    "software_requirements": ["numpy", "scipy", "pandas"]
                },
                "auto_register": True
            }
        }
    
    def _analyze_task_requirements(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements for optimization."""
        analysis = {
            "complexity": "medium",
            "data_size": "medium",
            "accuracy_requirement": "standard",
            "speed_requirement": "standard"
        }
        
        # Analyze task type
        task_type = task_spec.get("type", "")
        
        # Complexity analysis
        if any(word in task_type for word in ["advanced", "complex", "detailed"]):
            analysis["complexity"] = "high"
        elif any(word in task_type for word in ["simple", "basic", "quick"]):
            analysis["complexity"] = "low"
        
        # Data size analysis
        if "data" in task_spec:
            data = task_spec["data"]
            if isinstance(data, dict):
                total_size = sum(len(v) if hasattr(v, '__len__') else 1 for v in data.values())
                if total_size > 10000:
                    analysis["data_size"] = "large"
                elif total_size < 100:
                    analysis["data_size"] = "small"
        
        # Accuracy and speed requirements from parameters
        params = task_spec.get("parameters", {})
        if params.get("high_accuracy", False):
            analysis["accuracy_requirement"] = "high"
        if params.get("fast_calculation", False):
            analysis["speed_requirement"] = "high"
        
        return analysis
    
    def _create_optimized_configuration(self, 
                                      tool_type: str,
                                      task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized configuration based on task analysis."""
        base_config = self.default_configurations.get(tool_type, {}).copy()
        
        # Adjust based on complexity
        complexity = task_analysis["complexity"]
        if complexity == "high":
            base_config["computational_cost_factor"] = base_config.get("computational_cost_factor", 1.0) * 1.5
        elif complexity == "low":
            base_config["computational_cost_factor"] = base_config.get("computational_cost_factor", 1.0) * 0.7
        
        # Adjust based on data size
        data_size = task_analysis["data_size"]
        if data_size == "large":
            # Increase memory requirements
            hw_req = base_config.get("hardware_requirements", {})
            hw_req["recommended_memory"] = hw_req.get("min_memory", 512) * 4
            base_config["hardware_requirements"] = hw_req
        
        # Adjust based on accuracy requirements
        accuracy_req = task_analysis["accuracy_requirement"]
        if accuracy_req == "high":
            base_config["high_accuracy_mode"] = True
        
        # Adjust based on speed requirements
        speed_req = task_analysis["speed_requirement"]
        if speed_req == "high":
            base_config["fast_mode"] = True
            base_config["computational_cost_factor"] = base_config.get("computational_cost_factor", 1.0) * 0.8
        
        return base_config
    
    def get_creation_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool creation."""
        return {
            "total_tools_created": len(self.creation_history),
            "tools_by_type": self._count_tools_by_type(),
            "tools_with_custom_config": len([r for r in self.creation_history 
                                           if r["configuration"] != self.default_configurations.get(r["tool_type"], {})]),
            "auto_registered_tools": len([r for r in self.creation_history if r["auto_registered"]]),
            "creation_timeline": self._get_creation_timeline(),
            "available_tool_types": list(self.tool_classes.keys()),
            "available_templates": list(self._get_tool_templates().keys())
        }
    
    def _count_tools_by_type(self) -> Dict[str, int]:
        """Count tools created by type."""
        counts = {}
        for record in self.creation_history:
            tool_type = record["tool_type"]
            counts[tool_type] = counts.get(tool_type, 0) + 1
        return counts
    
    def _get_creation_timeline(self) -> List[Dict[str, Any]]:
        """Get creation timeline for analysis."""
        timeline = []
        for record in self.creation_history:
            timeline.append({
                "timestamp": record["created_at"],
                "tool_type": record["tool_type"],
                "tool_id": record["tool_id"]
            })
        return sorted(timeline, key=lambda x: x["timestamp"])