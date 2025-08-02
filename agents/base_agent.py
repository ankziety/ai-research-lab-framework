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