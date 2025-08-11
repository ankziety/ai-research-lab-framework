"""
Virtual Lab Agent Bridge

Provides a bridge between the current BaseAgent system and the Virtual Lab Agent system.
This enables the use of Virtual Lab's proven meeting methodology while preserving
existing agent functionality.
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from openai import OpenAI

from .base_agent import BaseAgent
from core.virtual_lab_integration.agent import Agent

logger = logging.getLogger(__name__)


class VirtualLabAgentBridge:
    """
    Bridge between BaseAgent and Virtual Lab Agent.
    
    This class provides an interface between the current agent system
    and the Virtual Lab based system, enabling long-running persistent sessions
    using OpenAI Assistants API.
    """
    
    def __init__(self, base_agent: BaseAgent):
        """
        Initialize the Virtual Lab agent bridge.
        
        Args:
            base_agent: The base agent to bridge
        """
        self.base_agent = base_agent
        self.vl_agent = self._create_virtual_lab_agent()
        self.assistant_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.client: Optional[OpenAI] = None
        
        logger.info(f"Created Virtual Lab bridge for agent: {base_agent.agent_id}")
    
    def _create_virtual_lab_agent(self) -> Agent:
        """
        Convert BaseAgent to Virtual Lab Agent.
        
        Returns:
            Virtual Lab Agent instance
        """
        # Extract model from base agent configuration
        model = self.base_agent.model_config.get('model', 'gpt-4o')
        
        # Create expertise string
        expertise = ", ".join(self.base_agent.expertise) if self.base_agent.expertise else "general research"
        
        # Create goal based on agent role
        goal = f"Contribute {expertise} expertise to research collaboration"
        
        return Agent(
            title=self.base_agent.role,
            expertise=expertise,
            goal=goal,
            role=self.base_agent.role,
            model=model
        )
    
    def initialize_assistant(self, client: OpenAI) -> str:
        """
        Initialize OpenAI Assistant for persistent state.
        
        Args:
            client: OpenAI client instance
            
        Returns:
            Assistant ID
        """
        self.client = client
        
        try:
            # Create assistant with agent's prompt
            assistant = client.beta.assistants.create(
                name=self.vl_agent.title,
                instructions=self.vl_agent.prompt,
                model=self.vl_agent.model
            )
            
            self.assistant_id = assistant.id
            logger.info(f"Created assistant for {self.base_agent.agent_id}: {assistant.id}")
            
            return assistant.id
            
        except Exception as e:
            logger.error(f"Failed to create assistant for {self.base_agent.agent_id}: {e}")
            raise
    
    def create_thread(self, client: Optional[OpenAI] = None) -> str:
        """
        Create persistent thread for long-running conversations.
        
        Args:
            client: OpenAI client instance (uses self.client if not provided)
            
        Returns:
            Thread ID
        """
        if client:
            self.client = client
        elif not self.client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Create thread
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            
            logger.info(f"Created thread for {self.base_agent.agent_id}: {thread.id}")
            return thread.id
            
        except Exception as e:
            logger.error(f"Failed to create thread for {self.base_agent.agent_id}: {e}")
            raise
    
    def send_message(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message to the agent's thread.
        
        Args:
            message: Message content
            thread_id: Thread ID (uses self.thread_id if not provided)
            
        Returns:
            Response data
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        if not thread_id:
            thread_id = self.thread_id
        if not thread_id:
            raise ValueError("Thread ID not provided and no thread created")
        
        try:
            # Add message to thread
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            
            # Get response
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id
            )
            
            # Extract the latest response
            latest_message = messages.data[0]
            response_content = latest_message.content[0].text.value
            
            return {
                'agent_id': self.base_agent.agent_id,
                'response': response_content,
                'run_status': run.status,
                'message_id': latest_message.id,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to send message to {self.base_agent.agent_id}: {e}")
            raise
    
    def get_conversation_history(self, thread_id: Optional[str] = None, limit: int = 10) -> list:
        """
        Get conversation history from the thread.
        
        Args:
            thread_id: Thread ID (uses self.thread_id if not provided)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        if not thread_id:
            thread_id = self.thread_id
        if not thread_id:
            raise ValueError("Thread ID not provided and no thread created")
        
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit
            )
            
            history = []
            for message in messages.data:
                history.append({
                    'role': message.role,
                    'content': message.content[0].text.value,
                    'timestamp': message.created_at,
                    'message_id': message.id
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for {self.base_agent.agent_id}: {e}")
            raise
    
    def cleanup(self):
        """Clean up OpenAI resources."""
        if self.client and self.assistant_id:
            try:
                self.client.beta.assistants.delete(self.assistant_id)
                logger.info(f"Deleted assistant: {self.assistant_id}")
            except Exception as e:
                logger.warning(f"Failed to delete assistant {self.assistant_id}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bridge to dictionary representation."""
        return {
            'base_agent_id': self.base_agent.agent_id,
            'vl_agent_title': self.vl_agent.title,
            'assistant_id': self.assistant_id,
            'thread_id': self.thread_id,
            'expertise': self.vl_agent.expertise,
            'role': self.vl_agent.role,
            'model': self.vl_agent.model
        }
    
    def __str__(self) -> str:
        """String representation of the bridge."""
        return f"VirtualLabBridge({self.base_agent.agent_id} -> {self.vl_agent.title})"
    
    def __repr__(self) -> str:
        """String representation of the bridge."""
        return self.__str__()
