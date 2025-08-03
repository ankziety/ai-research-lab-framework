"""
Tool Registry

Manages discovery, registration, and access to research tools.
"""

from typing import Dict, List, Any, Optional, Type
import logging
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for managing research tools available to agents.
    
    Provides discovery, recommendation, and access management for tools
    that agents can use to conduct research and experiments.
    """
    
    def __init__(self, auto_register_default: bool = True):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self.agent_tool_usage: Dict[str, Dict[str, int]] = {}
        
        if auto_register_default:
            self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools from the tools module."""
        try:
            # Import and register analysis tools
            from .analysis_tools import DataVisualizer, PatternDetector, HypothesisValidator
            
            # Register data visualizer
            data_viz = DataVisualizer()
            self.register_tool(data_viz, categories=['analysis', 'visualization'])
            
            # Register pattern detector if available
            try:
                pattern_detector = PatternDetector()
                self.register_tool(pattern_detector, categories=['analysis', 'pattern_detection'])
            except Exception as e:
                logger.warning(f"Could not register PatternDetector: {e}")
            
            # Register hypothesis validator if available
            try:
                hypothesis_validator = HypothesisValidator()
                self.register_tool(hypothesis_validator, categories=['analysis', 'validation'])
            except Exception as e:
                logger.warning(f"Could not register HypothesisValidator: {e}")
            
            # Import and register experimental tools
            from .experimental_tools import ExperimentRunner, DataCollector, StatisticalAnalyzer
            
            try:
                experiment_runner = ExperimentRunner()
                self.register_tool(experiment_runner, categories=['experimental', 'research'])
            except Exception as e:
                logger.warning(f"Could not register ExperimentRunner: {e}")
            
            try:
                data_collector = DataCollector()
                self.register_tool(data_collector, categories=['experimental', 'data_collection'])
            except Exception as e:
                logger.warning(f"Could not register DataCollector: {e}")
                
            try:
                statistical_analyzer = StatisticalAnalyzer()
                self.register_tool(statistical_analyzer, categories=['analysis', 'statistics'])
            except Exception as e:
                logger.warning(f"Could not register StatisticalAnalyzer: {e}")
            
            # Import and register literature tools
            from .literature_tools import LiteratureSearchTool, CitationAnalyzer
            
            try:
                literature_search = LiteratureSearchTool()
                self.register_tool(literature_search, categories=['literature', 'search'])
            except Exception as e:
                logger.warning(f"Could not register LiteratureSearchTool: {e}")
                
            try:
                citation_analyzer = CitationAnalyzer()
                self.register_tool(citation_analyzer, categories=['literature', 'analysis'])
            except Exception as e:
                logger.warning(f"Could not register CitationAnalyzer: {e}")
            
            # Import and register collaboration tools
            from .collaboration_tools import TeamCommunication, TaskCoordinator
            
            try:
                team_comm = TeamCommunication()
                self.register_tool(team_comm, categories=['collaboration', 'communication'])
            except Exception as e:
                logger.warning(f"Could not register TeamCommunication: {e}")
                
            try:
                task_coordinator = TaskCoordinator()
                self.register_tool(task_coordinator, categories=['collaboration', 'coordination'])
            except Exception as e:
                logger.warning(f"Could not register TaskCoordinator: {e}")
            
            # Import and register generic tools
            from .generic_tools import WebSearchTool, CodeInterpreterTool, DataAnalysisTool, ResearchAssistantTool
            
            try:
                web_search = WebSearchTool()
                self.register_tool(web_search, categories=['research', 'information'])
            except Exception as e:
                logger.warning(f"Could not register WebSearchTool: {e}")
                
            try:
                code_interpreter = CodeInterpreterTool()
                self.register_tool(code_interpreter, categories=['analysis', 'computation'])
            except Exception as e:
                logger.warning(f"Could not register CodeInterpreterTool: {e}")
                
            try:
                data_analysis = DataAnalysisTool()
                self.register_tool(data_analysis, categories=['analysis', 'statistics'])
            except Exception as e:
                logger.warning(f"Could not register DataAnalysisTool: {e}")
                
            try:
                research_assistant = ResearchAssistantTool()
                self.register_tool(research_assistant, categories=['research', 'coordination'])
            except Exception as e:
                logger.warning(f"Could not register ResearchAssistantTool: {e}")
            
            logger.info(f"Registered {len(self.tools)} default tools in ToolRegistry")
            
        except Exception as e:
            logger.error(f"Failed to register default tools: {e}")
        
    def register_tool(self, tool: BaseTool, categories: List[str] = None):
        """
        Register a new tool in the registry.
        
        Args:
            tool: Tool instance to register
            categories: Optional categories this tool belongs to
        """
        self.tools[tool.tool_id] = tool
        
        # Add to categories
        if categories:
            for category in categories:
                if category not in self.tool_categories:
                    self.tool_categories[category] = []
                self.tool_categories[category].append(tool.tool_id)
        
        logger.info(f"Registered tool: {tool.name} ({tool.tool_id})")
    
    def discover_tools(self, agent_id: str, task_description: str, 
                      requirements: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Discover tools that can help with a specific task.
        
        Args:
            agent_id: ID of the requesting agent
            task_description: Description of the task to be performed
            requirements: Optional task requirements and constraints
            
        Returns:
            List of recommended tools with confidence scores
        """
        requirements = requirements or {}
        recommendations = []
        
        for tool_id, tool in self.tools.items():
            # Get confidence score for this task
            confidence = tool.can_handle(task_description, requirements)
            
            # Accept lower confidences if no obvious keywords match
            if confidence > 0.05:
                usage_bonus = self._get_agent_usage_bonus(agent_id, tool_id)
                adjusted_confidence = min(1.0, confidence + usage_bonus)
                recommendations.append({
                    'tool_id': tool_id,
                    'confidence': adjusted_confidence,
                    'name': tool.name,
                    'description': tool.description,
                    'capabilities': tool.capabilities,
                    'success_rate': tool.success_rate,
                    'usage_count': tool.usage_count
                })

        # Fallback: if no tools pass threshold, return all available tools with base confidence 0.05
        if not recommendations:
            for tool_id, tool in self.tools.items():
                recommendations.append({
                    'tool_id': tool_id,
                    'confidence': 0.05,
                    'name': tool.name,
                    'description': tool.description,
                    'capabilities': tool.capabilities,
                    'success_rate': tool.success_rate,
                    'usage_count': tool.usage_count
                })
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations
    
    def request_tool(self, agent_id: str, tool_id: str, 
                    context: Dict[str, Any]) -> Optional[BaseTool]:
        """
        Request access to a specific tool.
        
        Args:
            agent_id: ID of the requesting agent
            tool_id: ID of the requested tool
            context: Execution context for validation
            
        Returns:
            Tool instance if accessible, None if not available
        """
        if tool_id not in self.tools:
            logger.warning(f"Tool {tool_id} not found in registry")
            return None
        
        tool = self.tools[tool_id]
        
        # Validate requirements
        if not tool.validate_requirements(context):
            logger.warning(f"Tool {tool_id} requirements not met for agent {agent_id}")
            return None
        
        # Track usage
        if agent_id not in self.agent_tool_usage:
            self.agent_tool_usage[agent_id] = {}
        
        self.agent_tool_usage[agent_id][tool_id] = (
            self.agent_tool_usage[agent_id].get(tool_id, 0) + 1
        )
        
        logger.info(f"Agent {agent_id} granted access to tool {tool_id}")
        return tool
    
    def get_tool_by_capability(self, capability: str) -> List[BaseTool]:
        """
        Find tools that provide a specific capability.
        
        Args:
            capability: Desired capability
            
        Returns:
            List of tools that provide this capability
        """
        matching_tools = []
        for tool in self.tools.values():
            if capability in tool.capabilities:
                matching_tools.append(tool)
        
        # Sort by success rate and usage
        matching_tools.sort(
            key=lambda t: (t.success_rate, t.usage_count), 
            reverse=True
        )
        return matching_tools
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        if category not in self.tool_categories:
            return []
        
        return [self.tools[tool_id] for tool_id in self.tool_categories[category]]
    
    def suggest_tools_for_research(self, research_question: str, 
                                 domain: str = None) -> List[Dict[str, Any]]:
        """
        Suggest tools that would be useful for a research question.
        
        Args:
            research_question: The research question or problem
            domain: Optional research domain
            
        Returns:
            List of suggested tools with explanations
        """
        suggestions = []
        
        # Analyze research question for key requirements
        keywords = research_question.lower().split()
        
        # Suggest experimental tools for research involving "experiment", "study", "test"
        if any(word in keywords for word in ['experiment', 'study', 'test', 'trial']):
            exp_tools = self.get_tools_by_category('experimental')
            suggestions.extend([{
                'tool': tool,
                'reason': 'Needed for conducting experiments and collecting data',
                'priority': 'high'
            } for tool in exp_tools])
        
        # Suggest analysis tools for "analysis", "correlation", "pattern"
        if any(word in keywords for word in ['analysis', 'analyze', 'pattern', 'correlation']):
            analysis_tools = self.get_tools_by_category('analysis')
            suggestions.extend([{
                'tool': tool,
                'reason': 'Required for data analysis and pattern detection', 
                'priority': 'high'
            } for tool in analysis_tools])
        
        # Suggest literature tools for "review", "survey", "literature"
        if any(word in keywords for word in ['review', 'literature', 'survey', 'meta']):
            lit_tools = self.get_tools_by_category('literature')
            suggestions.extend([{
                'tool': tool,
                'reason': 'Essential for literature review and citation analysis',
                'priority': 'medium'
            } for tool in lit_tools])
        
        return suggestions
    
    def _get_agent_usage_bonus(self, agent_id: str, tool_id: str) -> float:
        """
        Calculate usage bonus for agent-tool pair.
        
        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            
        Returns:
            Bonus score based on past successful usage
        """
        if agent_id not in self.agent_tool_usage:
            return 0.0
        
        usage_count = self.agent_tool_usage[agent_id].get(tool_id, 0)
        # Small bonus for frequently used tools (max 0.1)
        return min(0.1, usage_count * 0.02)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            'total_tools': len(self.tools),
            'categories': list(self.tool_categories.keys()),
            'tools_per_category': {
                cat: len(tools) for cat, tools in self.tool_categories.items()
            },
            'most_used_tools': sorted(
                [(tool.tool_id, tool.usage_count) for tool in self.tools.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'highest_success_rate': sorted(
                [(tool.tool_id, tool.success_rate) for tool in self.tools.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }