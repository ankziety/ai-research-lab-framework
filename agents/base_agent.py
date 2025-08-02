"""
Base Agent class for the multi-agent AI research framework.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the AI-powered research framework.
    
    Each agent has a specific role, expertise, and can participate in
    collaborative research through structured communication.
    """
    
    def __init__(self, agent_id: str, role: str, expertise: List[str], 
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: The agent's role (e.g., "Ophthalmology Expert")
            expertise: List of expertise domains
            model_config: Configuration for the underlying LLM
        """
        self.agent_id = agent_id
        self.role = role
        self.expertise = expertise
        self.model_config = model_config or {}
        self.performance_metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'average_quality_score': 0.0,
            'last_active': None
        }
        self.conversation_history = []
        self.current_task = None
        
        # Initialize LLM client with this agent's configuration
        self.llm_client = get_llm_client(self.model_config)
        
        logger.info(f"Agent {self.agent_id} ({self.role}) initialized")
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate a response to a given prompt with context.
        
        Args:
            prompt: The input prompt or question
            context: Additional context information
            
        Returns:
            Generated response string
        """
        # Default implementation uses role and expertise for specialized responses
        specialized_prompt = f"""
        You are a {self.role} with expertise in: {', '.join(self.expertise)}.
        Please provide a detailed analysis of the following:
        
        {prompt}
        
        Focus on aspects relevant to your expertise and provide evidence-based insights.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt, 
            context, 
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """
        Assess how relevant a task is to this agent's expertise.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Default implementation based on keyword matching with expertise
        task_lower = task_description.lower()
        
        # Check for matches with expertise areas
        total_words = len(task_lower.split())
        matches = 0
        
        for expertise_area in self.expertise:
            expertise_words = expertise_area.lower().split()
            for word in expertise_words:
                if word in task_lower:
                    matches += 1
        
        # Calculate relevance score
        if total_words == 0:
            return 0.0
        
        base_score = min(1.0, matches / max(1, total_words) * 5)  # Scale factor
        
        # Use LLM for more sophisticated relevance assessment if available
        if hasattr(self.llm_client, 'openai_api_key') and self.llm_client.openai_api_key:
            try:
                relevance_prompt = f"""
                Rate the relevance of this task to an expert in {', '.join(self.expertise)} on a scale of 0.0 to 1.0:
                
                Task: {task_description}
                
                Only respond with a number between 0.0 and 1.0.
                """
                
                llm_response = self.llm_client.generate_response(
                    relevance_prompt, 
                    {}, 
                    agent_role="Relevance Assessor"
                )
                
                # Extract numeric score from response
                try:
                    import re
                    score_match = re.search(r'(\d+\.?\d*)', llm_response)
                    if score_match:
                        llm_score = float(score_match.group(1))
                        if llm_score <= 1.0:
                            return llm_score
                except:
                    pass
            except:
                pass
        
        return base_score
    
    def receive_message(self, sender_id: str, message: str, 
                       context: Optional[Dict[str, Any]] = None) -> str:
        """
        Receive and process a message from another agent or the PI.
        
        Args:
            sender_id: ID of the sending agent
            message: The message content
            context: Optional context information
            
        Returns:
            Response message
        """
        logger.info(f"Agent {self.agent_id} received message from {sender_id}")
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': time.time(),
            'sender': sender_id,
            'message': message,
            'context': context or {}
        })
        
        # Generate response
        response = self.generate_response(message, context or {})
        
        # Add response to history
        self.conversation_history.append({
            'timestamp': time.time(),
            'sender': self.agent_id,
            'message': response,
            'context': context or {}
        })
        
        return response
    
    def assign_task(self, task: Dict[str, Any]) -> bool:
        """
        Assign a task to this agent.
        
        Args:
            task: Task description with required parameters
            
        Returns:
            True if task accepted, False otherwise
        """
        relevance_score = self.assess_task_relevance(task.get('description', ''))
        
        if relevance_score >= 0.3:  # Threshold for task acceptance
            self.current_task = task
            logger.info(f"Agent {self.agent_id} accepted task: {task.get('id', 'unknown')}")
            return True
        else:
            logger.info(f"Agent {self.agent_id} declined task due to low relevance: {relevance_score}")
            return False
    
    def complete_task(self, task_result: Any) -> Dict[str, Any]:
        """
        Mark current task as completed and update metrics.
        
        Args:
            task_result: The result of the completed task
            
        Returns:
            Task completion summary
        """
        if not self.current_task:
            raise ValueError("No active task to complete")
        
        completion_data = {
            'agent_id': self.agent_id,
            'task_id': self.current_task.get('id'),
            'result': task_result,
            'completion_time': time.time(),
            'success': True
        }
        
        # Update performance metrics
        self.performance_metrics['tasks_completed'] += 1
        self.performance_metrics['last_active'] = time.time()
        
        # Clear current task
        self.current_task = None
        
        logger.info(f"Agent {self.agent_id} completed task: {completion_data['task_id']}")
        return completion_data
    
    def update_performance_metrics(self, quality_score: float, success: bool = True):
        """
        Update agent performance metrics.
        
        Args:
            quality_score: Quality score for the completed work (0.0-1.0)
            success: Whether the task was successful
        """
        total_tasks = self.performance_metrics['tasks_completed']
        
        if total_tasks > 0:
            # Update success rate
            current_successes = self.performance_metrics['success_rate'] * total_tasks
            if success:
                current_successes += 1
            self.performance_metrics['success_rate'] = current_successes / (total_tasks + 1)
            
            # Update average quality score
            current_avg = self.performance_metrics['average_quality_score']
            self.performance_metrics['average_quality_score'] = (
                (current_avg * total_tasks + quality_score) / (total_tasks + 1)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and metrics.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            'agent_id': self.agent_id,
            'role': self.role,
            'expertise': self.expertise,
            'current_task': self.current_task.get('id') if self.current_task else None,
            'performance_metrics': self.performance_metrics.copy(),
            'conversation_count': len(self.conversation_history)
        }
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for this agent.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the agent's conversation history."""
        self.conversation_history.clear()
        logger.info(f"Cleared conversation history for agent {self.agent_id}")
    
    def discover_available_tools(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Discover tools available for a specific task.
        
        Args:
            task_description: Description of the task to be performed
            
        Returns:
            List of available tools with confidence scores
        """
        try:
            from tools.tool_registry import ToolRegistry
            
            # Get tool registry instance
            tool_registry = ToolRegistry()
            
            # Discover tools for this task
            available_tools = tool_registry.discover_tools(
                agent_id=self.agent_id,
                task_description=task_description,
                requirements={'agent_expertise': self.expertise}
            )
            
            # Enhance with cost and capability information
            enhanced_tools = []
            for tool_info in available_tools:
                tool = tool_info['tool']
                enhanced_tool = {
                    'tool_id': tool.tool_id,
                    'name': tool.name,
                    'description': tool.description,
                    'capabilities': tool.capabilities,
                    'confidence': tool_info['confidence'],
                    'success_rate': tool.success_rate,
                    'usage_count': tool.usage_count,
                    'requirements': tool.requirements
                }
                enhanced_tools.append(enhanced_tool)
            
            logger.info(f"Agent {self.agent_id} discovered {len(enhanced_tools)} tools for task")
            return enhanced_tools
            
        except Exception as e:
            logger.error(f"Tool discovery failed for agent {self.agent_id}: {e}")
            return []
    
    def request_tool(self, tool_id: str, context: Dict[str, Any]) -> Optional[Any]:
        """
        Request access to a specific tool.
        
        Args:
            tool_id: ID of the requested tool
            context: Execution context for validation
            
        Returns:
            Tool instance if accessible, None if not available
        """
        try:
            from tools.tool_registry import ToolRegistry
            
            # Get tool registry instance
            tool_registry = ToolRegistry()
            
            # Request tool with context
            tool = tool_registry.request_tool(
                agent_id=self.agent_id,
                tool_id=tool_id,
                context=context
            )
            
            if tool:
                logger.info(f"Agent {self.agent_id} granted access to tool {tool_id}")
                return tool
            else:
                logger.warning(f"Agent {self.agent_id} denied access to tool {tool_id}")
                return None
                
        except Exception as e:
            logger.error(f"Tool request failed for agent {self.agent_id}: {e}")
            return None
    
    def execute_with_tools(self, task: str, tools: List[Any]) -> Dict[str, Any]:
        """
        Execute a task using multiple tools.
        
        Args:
            task: Task description
            tools: List of tool instances to use
            
        Returns:
            Execution results with output and metadata
        """
        results = {
            'success': False,
            'output': '',
            'tool_results': [],
            'metadata': {
                'agent_id': self.agent_id,
                'tools_used': len(tools),
                'execution_time': 0,
                'total_cost': 0.0
            }
        }
        
        start_time = time.time()
        
        try:
            # Execute each tool
            for i, tool in enumerate(tools):
                try:
                    # Prepare task for tool
                    tool_task = {
                        'description': task,
                        'agent_id': self.agent_id,
                        'tool_index': i
                    }
                    
                    # Execute tool
                    tool_result = tool.execute(tool_task, {
                        'agent_id': self.agent_id,
                        'agent_role': self.role,
                        'agent_expertise': self.expertise
                    })
                    
                    # Track cost if available
                    if 'cost' in tool_result.get('metadata', {}):
                        results['metadata']['total_cost'] += tool_result['metadata']['cost']
                    
                    results['tool_results'].append({
                        'tool_id': tool.tool_id,
                        'tool_name': tool.name,
                        'result': tool_result,
                        'success': tool_result.get('success', False)
                    })
                    
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    results['tool_results'].append({
                        'tool_id': getattr(tool, 'tool_id', 'unknown'),
                        'tool_name': getattr(tool, 'name', 'unknown'),
                        'result': {'success': False, 'error': str(e)},
                        'success': False
                    })
            
            # Determine overall success
            successful_tools = sum(1 for result in results['tool_results'] if result['success'])
            results['success'] = successful_tools > 0
            
            # Generate combined output
            if results['success']:
                outputs = []
                for result in results['tool_results']:
                    if result['success']:
                        output = result['result'].get('output', '')
                        if output:
                            outputs.append(f"Tool {result['tool_name']}: {output}")
                
                results['output'] = '\n\n'.join(outputs)
            
            results['metadata']['execution_time'] = time.time() - start_time
            
            logger.info(f"Agent {self.agent_id} executed task with {len(tools)} tools, {successful_tools} successful")
            return results
            
        except Exception as e:
            logger.error(f"Tool execution failed for agent {self.agent_id}: {e}")
            results['metadata']['execution_time'] = time.time() - start_time
            results['error'] = str(e)
            return results
    
    def build_custom_tool(self, tool_spec: Dict[str, Any]) -> Optional[Any]:
        """
        Build a custom tool based on specification.
        
        Args:
            tool_spec: Tool specification with parameters
            
        Returns:
            Custom tool instance if successful, None otherwise
        """
        try:
            from tools.dynamic_tool_builder import WebSearchTool, CodeExecutionTool, ModelSwitchingTool, CustomToolChainBuilder
            
            tool_type = tool_spec.get('type', '')
            tool_config = tool_spec.get('config', {})
            
            if tool_type == 'web_search':
                # Build web search tool
                api_keys = tool_config.get('api_keys', {})
                cost_manager = tool_config.get('cost_manager')
                return WebSearchTool(api_keys, cost_manager)
                
            elif tool_type == 'code_execution':
                # Build code execution tool
                sandbox_config = tool_config.get('sandbox_config', {})
                cost_manager = tool_config.get('cost_manager')
                return CodeExecutionTool(sandbox_config, cost_manager)
                
            elif tool_type == 'model_switching':
                # Build model switching tool
                cost_manager = tool_config.get('cost_manager')
                llm_client = tool_config.get('llm_client')
                if cost_manager and llm_client:
                    return ModelSwitchingTool(cost_manager, llm_client)
                    
            elif tool_type == 'custom_chain':
                # Build custom tool chain
                tool_registry = tool_config.get('tool_registry')
                cost_manager = tool_config.get('cost_manager')
                if tool_registry:
                    return CustomToolChainBuilder(tool_registry, cost_manager)
            
            logger.warning(f"Unknown tool type: {tool_type}")
            return None
            
        except Exception as e:
            logger.error(f"Custom tool building failed for agent {self.agent_id}: {e}")
            return None
    
    def optimize_tool_usage(self, task_description: str, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize tool selection based on cost and performance.
        
        Args:
            task_description: Description of the task
            available_tools: List of available tools
            
        Returns:
            Optimized list of tools with usage recommendations
        """
        try:
            # Score tools based on multiple factors
            scored_tools = []
            
            for tool in available_tools:
                score = 0.0
                
                # Confidence score (0-1)
                score += tool.get('confidence', 0.0) * 0.4
                
                # Success rate score (0-1)
                score += tool.get('success_rate', 0.0) * 0.3
                
                # Usage experience score (0-1)
                usage_count = tool.get('usage_count', 0)
                experience_score = min(1.0, usage_count / 10.0)  # Cap at 10 uses
                score += experience_score * 0.2
                
                # Capability match score (0-1)
                task_terms = set(task_description.lower().split())
                capability_terms = set()
                for capability in tool.get('capabilities', []):
                    capability_terms.update(capability.lower().split())
                
                if capability_terms:
                    match_score = len(task_terms & capability_terms) / max(1, len(task_terms))
                    score += match_score * 0.1
                
                scored_tools.append({
                    **tool,
                    'optimization_score': score
                })
            
            # Sort by optimization score
            scored_tools.sort(key=lambda x: x['optimization_score'], reverse=True)
            
            # Return top tools with usage recommendations
            optimized_tools = []
            for tool in scored_tools[:3]:  # Top 3 tools
                recommendation = {
                    **tool,
                    'recommended_usage': self._generate_usage_recommendation(tool, task_description)
                }
                optimized_tools.append(recommendation)
            
            logger.info(f"Agent {self.agent_id} optimized {len(optimized_tools)} tools for task")
            return optimized_tools
            
        except Exception as e:
            logger.error(f"Tool optimization failed for agent {self.agent_id}: {e}")
            return available_tools
    
    def _generate_usage_recommendation(self, tool: Dict[str, Any], task_description: str) -> str:
        """Generate usage recommendation for a tool."""
        confidence = tool.get('confidence', 0.0)
        success_rate = tool.get('success_rate', 0.0)
        
        if confidence > 0.8 and success_rate > 0.8:
            return "Primary tool - high confidence and success rate"
        elif confidence > 0.6 and success_rate > 0.6:
            return "Secondary tool - good confidence and success rate"
        elif confidence > 0.4:
            return "Fallback tool - moderate confidence"
        else:
            return "Experimental tool - low confidence, use with caution"
    
    def __str__(self) -> str:
        return f"Agent({self.agent_id}, {self.role})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the agent
        """
        return {
            'agent_id': self.agent_id,
            'role': self.role,
            'expertise': self.expertise,
            'performance_metrics': self.performance_metrics.copy(),
            'current_task': self.current_task,
            'conversation_count': len(self.conversation_history),
            'agent_type': self.__class__.__name__
        }