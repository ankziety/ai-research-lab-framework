"""
LLM Client for agent interactions.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, etc.).
"""

import os
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with various LLM providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client with configuration.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.provider = config.get('default_llm_provider', 'openai')
        self.model = config.get('default_model', 'gpt-4')
        
        # API keys from config or environment
        self.openai_api_key = (
            config.get('openai_api_key') or 
            os.getenv('OPENAI_API_KEY')
        )
        self.anthropic_api_key = (
            config.get('anthropic_api_key') or 
            os.getenv('ANTHROPIC_API_KEY')
        )
        
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate that required API keys are available."""
        if self.provider == 'openai' and not self.openai_api_key:
            logger.warning("OpenAI API key not found. Using mock responses.")
        elif self.provider == 'anthropic' and not self.anthropic_api_key:
            logger.warning("Anthropic API key not found. Using mock responses.")
    
    def generate_response(self, prompt: str, context: Dict[str, Any], 
                         agent_role: str = "AI Assistant") -> str:
        """
        Generate a response using the configured LLM provider.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            agent_role: The role/persona of the agent
            
        Returns:
            Generated response string
        """
        try:
            if self.provider == 'openai' and self.openai_api_key:
                return self._generate_openai_response(prompt, context, agent_role)
            elif self.provider == 'anthropic' and self.anthropic_api_key:
                return self._generate_anthropic_response(prompt, context, agent_role)
            else:
                return self._generate_mock_response(prompt, context, agent_role)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
    def _generate_openai_response(self, prompt: str, context: Dict[str, Any], 
                                 agent_role: str) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI library not installed. Using mock response.")
            return self._generate_mock_response(prompt, context, agent_role)
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
    def _generate_anthropic_response(self, prompt: str, context: Dict[str, Any], 
                                    agent_role: str) -> str:
        """Generate response using Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.warning("Anthropic library not installed. Using mock response.")
            return self._generate_mock_response(prompt, context, agent_role)
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
    def _generate_mock_response(self, prompt: str, context: Dict[str, Any], 
                               agent_role: str) -> str:
        """Generate a mock response for demonstration purposes."""
        # Simulate API delay
        time.sleep(0.1)
        
        return f"""
        [{agent_role} Response - Mock Mode]
        
        Regarding your inquiry: "{prompt[:100]}..."
        
        As a {agent_role}, I would provide specialized insights based on my expertise.
        However, this is a mock response since no valid API key was configured.
        
        To enable full AI-powered responses, please configure:
        - OpenAI API key for GPT models
        - Anthropic API key for Claude models
        
        Context received: {len(str(context))} characters
        """


# Global client instance
_llm_client = None


def get_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Get or create global LLM client instance."""
    global _llm_client
    
    if _llm_client is None or config:
        _llm_client = LLMClient(config or {})
    
    return _llm_client