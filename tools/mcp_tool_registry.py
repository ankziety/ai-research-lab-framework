"""
MCP Tool Registry

Manages discovery, loading, and execution of MCP-compatible tools created by the coding specialist.
"""

import logging
import yaml
import importlib.util
import sys
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Callable
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """
    Registry for MCP-compatible tools that can be dynamically discovered and loaded.
    
    This registry works alongside the main tool registry to provide access to
    tools created by the coding specialist agent.
    """
    
    def __init__(self, tools_directory: Optional[str] = None):
        """
        Initialize MCP tool registry.
        
        Args:
            tools_directory: Directory containing MCP-compatible tools
        """
        base_dir = tools_directory or os.environ.get("MCP_TOOLS_DIR", "shared_tools")
        self.tools_directory = Path(base_dir)
        self.tools_directory.mkdir(exist_ok=True)
        self.loaded_tools: Dict[str, BaseTool] = {}
        self.mcp_descriptions: Dict[str, Dict[str, Any]] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"MCP Tool Registry initialized with directory: {self.tools_directory}")
    
    def discover_mcp_tools(self, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover available MCP-compatible tools.
        
        Args:
            domain_filter: Optional domain to filter tools by
            
        Returns:
            List of discovered tools with metadata
        """
        discovered_tools = []
        
        # Scan for MCP description files
        for mcp_file in self.tools_directory.glob("*.mcp.yaml"):
            try:
                # Derive tool name from filename 'name.mcp.yaml' -> 'name'
                filename = mcp_file.name
                tool_name = filename[:-9] if filename.endswith('.mcp.yaml') else mcp_file.stem
                tool_file = self.tools_directory / f"{tool_name}.py"
                
                if not tool_file.exists():
                    logger.warning(f"MCP description found but no tool file: {mcp_file}")
                    continue
                
                # Load MCP description
                with open(mcp_file, 'r') as f:
                    mcp_description = yaml.safe_load(f)
                
                # Check domain filter (robust matching: either contains the other or token overlap)
                if domain_filter:
                    filter_norm = domain_filter.lower().replace('-', '_')
                    tool_domain = mcp_description.get('domain', '').lower().replace('-', '_')
                    # Quick contains either way
                    contains_match = filter_norm in tool_domain or tool_domain in filter_norm
                    if not contains_match:
                        # Token overlap match
                        def tokenize(s: str) -> set:
                            return set([t for t in s.replace('/', '_').replace(' ', '_').split('_') if t])
                        if tokenize(filter_norm).isdisjoint(tokenize(tool_domain)):
                            continue
                
                tool_info = {
                    'name': tool_name,
                    'mcp_description': mcp_description,
                    'file_path': str(tool_file),
                    'mcp_path': str(mcp_file),
                    'domain': mcp_description.get('domain', 'general'),
                    'capabilities': mcp_description.get('capabilities', []),
                    'loaded': tool_name in self.loaded_tools
                }
                
                discovered_tools.append(tool_info)
                
            except Exception as e:
                logger.error(f"Failed to process MCP file {mcp_file}: {e}")
        
        logger.info(f"Discovered {len(discovered_tools)} MCP-compatible tools")
        return discovered_tools
    
    def load_mcp_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Load an MCP-compatible tool into memory.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Loaded tool instance or None if failed
        """
        if tool_name in self.loaded_tools:
            logger.info(f"Tool {tool_name} already loaded")
            return self.loaded_tools[tool_name]
        
        tool_file = self.tools_directory / f"{tool_name}.py"
        mcp_file = self.tools_directory / f"{tool_name}.mcp.yaml"
        
        if not tool_file.exists():
            logger.error(f"Tool file not found: {tool_file}")
            return None
        
        if not mcp_file.exists():
            logger.error(f"MCP description not found: {mcp_file}")
            return None
        
        try:
            # Load MCP description
            with open(mcp_file, 'r') as f:
                mcp_description = yaml.safe_load(f)
            
            # Load tool module
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for {tool_name}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[tool_name] = module
            spec.loader.exec_module(module)
            
            # Find tool class (convention: ToolNameTool)
            tool_class_name = f"{tool_name.title().replace('_', '')}Tool"
            tool_class = getattr(module, tool_class_name, None)
            
            if tool_class is None:
                logger.error(f"Tool class {tool_class_name} not found in {tool_name}")
                return None
            
            # Create tool instance
            tool_instance = tool_class()
            
            # Validate tool interface
            if not isinstance(tool_instance, BaseTool):
                logger.error(f"Tool {tool_name} does not inherit from BaseTool")
                return None
            
            # Store tool and metadata
            self.loaded_tools[tool_name] = tool_instance
            self.mcp_descriptions[tool_name] = mcp_description
            self.tool_metadata[tool_name] = {
                'file_path': str(tool_file),
                'mcp_path': str(mcp_file),
                'domain': mcp_description.get('domain', 'general'),
                'capabilities': mcp_description.get('capabilities', []),
                'load_time': __import__('time').time()
            }
            
            logger.info(f"Successfully loaded MCP tool: {tool_name}")
            return tool_instance
            
        except Exception as e:
            logger.error(f"Failed to load MCP tool {tool_name}: {e}")
            return None
    
    def get_mcp_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get an MCP tool, loading it if necessary.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not available
        """
        if tool_name not in self.loaded_tools:
            return self.load_mcp_tool(tool_name)
        return self.loaded_tools[tool_name]
    
    def execute_mcp_tool(self, tool_name: str, task: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP-compatible tool.
        
        Args:
            tool_name: Name of the tool to execute
            task: Task parameters
            context: Execution context
            
        Returns:
            Execution result
        """
        tool = self.get_mcp_tool(tool_name)
        if tool is None:
            return {
                'success': False,
                'error': f'Tool {tool_name} not available',
                'tool_name': tool_name
            }
        
        try:
            # Validate input against MCP schema
            validation_result = self._validate_input_against_schema(
                tool_name, task
            )
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f'Input validation failed: {validation_result["errors"]}',
                    'tool_name': tool_name
                }
            
            # Execute tool
            result = tool.execute(task, context)
            
            # Validate output against MCP schema
            output_validation = self._validate_output_against_schema(
                tool_name, result
            )
            if not output_validation['valid']:
                logger.warning(f"Output validation failed for {tool_name}: {output_validation['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool_name': tool_name
            }
    
    def get_mcp_description(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get MCP description for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            MCP description or None if not found
        """
        return self.mcp_descriptions.get(tool_name)
    
    def list_available_tools(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available MCP tools.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            List of available tools
        """
        discovered = self.discover_mcp_tools(domain)
        
        available_tools = []
        for tool_info in discovered:
            tool_name = tool_info['name']
            
            # Try to load if not already loaded
            if not tool_info['loaded']:
                self.load_mcp_tool(tool_name)
            
            if tool_name in self.loaded_tools:
                tool_info['available'] = True
                tool_info['loaded'] = True
                available_tools.append(tool_info)
            else:
                tool_info['available'] = False
                tool_info['loaded'] = False
                available_tools.append(tool_info)
        
        return available_tools
    
    def recommend_tools_for_task(self, task_description: str, 
                                domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recommend MCP tools for a specific task.
        
        Args:
            task_description: Description of the task
            domain: Optional domain filter
            
        Returns:
            List of recommended tools with confidence scores
        """
        available_tools = self.list_available_tools(domain)
        
        recommendations = []
        for tool_info in available_tools:
            confidence = self._calculate_tool_relevance(
                tool_info, task_description
            )
            
            if confidence > 0.1:  # Minimum relevance threshold
                recommendation = {
                    **tool_info,
                    'confidence': confidence,
                    'relevance_reason': self._get_relevance_reason(tool_info, task_description)
                }
                recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def _validate_input_against_schema(self, tool_name: str, 
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against MCP input schema.
        
        Args:
            tool_name: Name of the tool
            input_data: Input data to validate
            
        Returns:
            Validation result
        """
        mcp_description = self.mcp_descriptions.get(tool_name)
        if not mcp_description:
            return {'valid': False, 'errors': ['No MCP description available']}
        
        input_schema = mcp_description.get('inputSchema', {})
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        errors = []
        
        # Check required fields
        for field in required:
            if field not in input_data:
                errors.append(f"Required field '{field}' missing")
        
        # Check field types
        for field_name, field_value in input_data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                field_type = field_schema.get('type', 'string')
                
                if not self._validate_field_type(field_value, field_type):
                    errors.append(f"Field '{field_name}' has invalid type. Expected {field_type}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_output_against_schema(self, tool_name: str, 
                                      output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output data against MCP output schema.
        
        Args:
            tool_name: Name of the tool
            output_data: Output data to validate
            
        Returns:
            Validation result
        """
        mcp_description = self.mcp_descriptions.get(tool_name)
        if not mcp_description:
            return {'valid': False, 'errors': ['No MCP description available']}
        
        output_schema = mcp_description.get('outputSchema', {})
        properties = output_schema.get('properties', {})
        
        errors = []
        
        # Check field types
        for field_name, field_value in output_data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                field_type = field_schema.get('type', 'string')
                
                if not self._validate_field_type(field_value, field_type):
                    errors.append(f"Output field '{field_name}' has invalid type. Expected {field_type}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate a field value against expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            
        Returns:
            True if valid, False otherwise
        """
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'object':
            return isinstance(value, dict)
        elif expected_type == 'array':
            return isinstance(value, list)
        else:
            return True  # Unknown type, assume valid
    
    def _calculate_tool_relevance(self, tool_info: Dict[str, Any], 
                                task_description: str) -> float:
        """
        Calculate relevance score for a tool given a task description.
        
        Args:
            tool_info: Tool information
            task_description: Task description
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Simple keyword matching for now
        # In a real implementation, this could use semantic similarity
        
        tool_description = tool_info.get('mcp_description', {}).get('description', '')
        tool_capabilities = tool_info.get('capabilities', [])
        
        task_lower = task_description.lower()
        desc_lower = tool_description.lower()
        
        # Check description match
        desc_score = 0.0
        for word in task_lower.split():
            if word in desc_lower:
                desc_score += 0.1
        
        # Check capabilities match
        cap_score = 0.0
        for capability in tool_capabilities:
            if capability.lower() in task_lower:
                cap_score += 0.2
        
        total_score = min(1.0, desc_score + cap_score)
        return total_score
    
    def _get_relevance_reason(self, tool_info: Dict[str, Any], 
                            task_description: str) -> str:
        """
        Get reason for tool relevance.
        
        Args:
            tool_info: Tool information
            task_description: Task description
            
        Returns:
            Relevance reason
        """
        tool_description = tool_info.get('mcp_description', {}).get('description', '')
        tool_capabilities = tool_info.get('capabilities', [])
        
        reasons = []
        
        # Check description keywords
        task_words = set(task_description.lower().split())
        desc_words = set(tool_description.lower().split())
        common_words = task_words.intersection(desc_words)
        
        if common_words:
            reasons.append(f"Keywords: {', '.join(list(common_words)[:3])}")
        
        # Check capabilities
        task_lower = task_description.lower()
        matching_caps = [cap for cap in tool_capabilities if cap.lower() in task_lower]
        
        if matching_caps:
            reasons.append(f"Capabilities: {', '.join(matching_caps)}")
        
        return '; '.join(reasons) if reasons else "General purpose tool"
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Registry statistics
        """
        discovered = self.discover_mcp_tools()
        
        return {
            'total_tools_discovered': len(discovered),
            'total_tools_loaded': len(self.loaded_tools),
            'tools_by_domain': self._get_tools_by_domain(discovered),
            'tools_by_capability': self._get_tools_by_capability(discovered),
            'registry_health': self._assess_registry_health()
        }
    
    def _get_tools_by_domain(self, discovered: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        """Get count of tools by domain from discovered and loaded metadata."""
        domain_counts: Dict[str, int] = {}
        # Count discovered tools
        for tool_info in (discovered or []):
            domain = tool_info.get('domain', 'general')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        # Include loaded metadata as well (may add domains not in discovered due to runtime loads)
        for tool_info in self.tool_metadata.values():
            domain = tool_info.get('domain', 'general')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts
    
    def _get_tools_by_capability(self, discovered: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        """Get count of tools by capability from discovered and loaded metadata."""
        capability_counts: Dict[str, int] = {}
        # Count discovered tools
        for tool_info in (discovered or []):
            for capability in tool_info.get('capabilities', []):
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        # Include loaded metadata
        for tool_info in self.tool_metadata.values():
            for capability in tool_info.get('capabilities', []):
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        return capability_counts
    
    def _assess_registry_health(self) -> str:
        """Assess overall registry health."""
        discovered = self.discover_mcp_tools()
        loaded = len(self.loaded_tools)
        
        if len(discovered) == 0:
            return "Empty - No tools discovered"
        elif loaded == 0:
            return "Unhealthy - Tools discovered but none loaded"
        elif loaded < len(discovered) * 0.8:
            return "Warning - Some tools failed to load"
        else:
            return "Healthy - Most tools loaded successfully"
