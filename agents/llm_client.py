"""
LLM Client for agent interactions.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, etc.).
"""

import os
import logging
import json
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class LLMClient:
    # Model cost table (USD per 1K tokens)
    _MODEL_COSTS = {
        'gpt-4o': (0.005, 0.015),
        'gpt-4o-mini': (0.00015, 0.0006),
        'gpt-3.5-turbo': (0.0005, 0.0015),
        'claude-3-sonnet': (0.003, 0.015),
        'claude-3-haiku': (0.00025, 0.00125),
        'gemini-pro': (0.0005, 0.0015),
        'llama2': (0.0, 0.0)
    }

    def _estimate_cost_local(self, tokens_input: int, tokens_output: int) -> float:
        """Quick cost estimate without external CostManager."""
        input_cost_per_1k, output_cost_per_1k = self._MODEL_COSTS.get(
            self.model,
            (0.0005, 0.0015)  # sensible default
        )
        return (tokens_input / 1000) * input_cost_per_1k + (tokens_output / 1000) * output_cost_per_1k

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
        # Check both top-level and nested locations
        api_keys = config.get('api_keys', {})
        framework = config.get('framework', {})
        
        self.openai_api_key = (
            config.get('openai_api_key') or 
            framework.get('openai_api_key') or
            api_keys.get('openai') or
            os.getenv('OPENAI_API_KEY')
        )
        self.anthropic_api_key = (
            config.get('anthropic_api_key') or 
            framework.get('anthropic_api_key') or
            api_keys.get('anthropic') or
            os.getenv('ANTHROPIC_API_KEY')
        )
        self.gemini_api_key = (
            config.get('gemini_api_key') or
            framework.get('gemini_api_key') or
            api_keys.get('gemini') or
            os.getenv('GEMINI_API_KEY')
        )
        self.huggingface_api_key = (
            config.get('huggingface_api_key') or
            framework.get('huggingface_api_key') or
            api_keys.get('huggingface') or
            os.getenv('HUGGINGFACE_API_KEY')
        )

        # Local model configurations
        self.ollama_endpoint = config.get('ollama_endpoint', 'http://localhost:11434')
        self.local_model_endpoint = config.get('local_model_endpoint', None)

        # Provider pricing for cost optimization
        self.provider_costs = {
            'openai': {'gpt-4o': 0.03, 'gpt-4o-mini': 0.00015},
            'anthropic': {'claude-3-sonnet': 0.003, 'claude-3-haiku': 0.00025},
            'gemini': {'gemini-pro': 0.0005, 'gemini-pro-vision': 0.002},
            'huggingface': {'default': 0.0001},  # Typically cheaper
            'ollama': {'default': 0.0},  # Free local inference
        }

        self._validate_configuration()

    def _validate_configuration(self):
        """Validate configuration and log available providers."""
        available_providers = []
        
        # Check all available providers, not just the configured one
        if self.openai_api_key:
            available_providers.append('openai')
        if self.anthropic_api_key:
            available_providers.append('anthropic')
        if self.gemini_api_key:
            available_providers.append('gemini')
        if self.huggingface_api_key:
            available_providers.append('huggingface')
        if self.ollama_endpoint:
            available_providers.append('ollama')

        if not available_providers:
            raise RuntimeError(f"No valid API keys found. Cannot proceed without real LLM provider.")
        else:
            logger.info(f"LLM client configured with {len(available_providers)} available providers: {available_providers}")
            
            # If the configured provider is not available, use the first available one
            if self.provider not in available_providers:
                logger.warning(f"Configured provider '{self.provider}' not available. Using '{available_providers[0]}' instead.")
                self.provider = available_providers[0]

    def generate_response(self, prompt: str, context: Dict[str, Any], 
                         agent_role: str = "AI Assistant", cost_manager=None) -> str:
        """
        Generate a response using the configured LLM provider with cost tracking.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            agent_role: The role/persona of the agent
            cost_manager: Optional cost manager for tracking
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If no valid LLM provider is available
            Exception: If LLM generation fails
        """
        start_time = time.time()
        tokens_input = len(prompt.split())
        agent_id = context.get('agent_id', 'unknown')
        task_type = context.get('task_type', 'general')
        
        # Estimate cost before generation
        estimated_tokens_output = tokens_input * 2  # Rough estimate
        if cost_manager:
            estimated_cost = cost_manager.estimate_cost(self.model, tokens_input, estimated_tokens_output)
            # Check if we can afford this request
            if not cost_manager.can_afford(estimated_cost):
                raise RuntimeError(f"Insufficient budget for LLM request: ${estimated_cost:.4f}")
        else:
            estimated_cost = self._estimate_cost_local(tokens_input, estimated_tokens_output)
        
        # Generate response
        if self.provider == 'openai' and self.openai_api_key:
            response = self._generate_openai_response(prompt, context, agent_role)
        elif self.provider == 'anthropic' and self.anthropic_api_key:
            response = self._generate_anthropic_response(prompt, context, agent_role)
        elif self.provider == 'gemini' and self.gemini_api_key:
            response = self._generate_gemini_response(prompt, context, agent_role)
        elif self.provider == 'huggingface' and self.huggingface_api_key:
            response = self._generate_huggingface_response(prompt, context, agent_role)
        elif self.provider == 'ollama':
            response = self._generate_ollama_response(prompt, context, agent_role)
        else:
            raise RuntimeError(f"No valid LLM provider available for {self.provider}")
        
        # Track actual usage and cost
        tokens_output = len(response.split())
        if cost_manager:
            actual_cost = cost_manager.estimate_cost(self.model, tokens_input, tokens_output)
            cost_manager.track_usage(
                model=self.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                actual_cost=actual_cost,
                task_type=task_type,
                agent_id=agent_id,
                success=True
            )
        else:
            actual_cost = self._estimate_cost_local(tokens_input, tokens_output)
        
        execution_time = time.time() - start_time
        logger.info(f"LLM response generated: {tokens_input + tokens_output} tokens, ${actual_cost:.4f}, {execution_time:.2f}s")
        
        return response

    def _generate_openai_response(self, prompt: str, context: Dict[str, Any], 
                                 agent_role: str) -> str:
        """Generate response using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)

            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."

            response = client.chat.completions.create(model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7)

            return response.choices[0].message.content

        except ImportError:
            raise RuntimeError("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

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
            raise RuntimeError("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def _generate_gemini_response(self, prompt: str, context: Dict[str, Any], 
                                 agent_role: str) -> str:
        """Generate response using Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')

            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."
            full_prompt = f"{system_message}\n\nUser: {prompt}\nAssistant:"

            response = model.generate_content(full_prompt)
            return response.text

        except ImportError:
            raise RuntimeError("Google Generative AI library not installed. Install with: pip install google-generative-ai")
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def _generate_huggingface_response(self, prompt: str, context: Dict[str, Any], 
                                      agent_role: str) -> str:
        """Generate response using HuggingFace Inference API."""
        try:
            import requests

            api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}

            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."
            payload = {
                "inputs": f"{system_message}\n\nUser: {prompt}\nAssistant:",
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').split('Assistant:')[-1].strip()
            else:
                return str(result)

        except ImportError:
            raise RuntimeError("HuggingFace Inference API library not installed. Install with: pip install requests")
        except Exception as e:
            raise RuntimeError(f"HuggingFace API error: {str(e)}")

    def _generate_ollama_response(self, prompt: str, context: Dict[str, Any], 
                                 agent_role: str) -> str:
        """Generate response using OLLAMA local inference."""
        try:
            import requests

            system_message = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."

            payload = {
                "model": self.model,  # e.g., 'llama2', 'mistral', 'codellama'
                "prompt": f"{system_message}\n\nUser: {prompt}\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }

            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', 'No response generated')

        except ImportError:
            raise RuntimeError("OLLAMA local inference library not installed. Install with: pip install requests")
        except Exception as e:
            raise RuntimeError(f"OLLAMA API error: {str(e)}")

    def select_optimal_provider(self, prompt: str, task_complexity: str = 'medium') -> str:
        """
        Select the most cost-effective provider for the given task.
        
        Args:
            prompt: The input prompt
            task_complexity: 'simple', 'medium', or 'complex'
            
        Returns:
            Optimal provider name
        """
        prompt_length = len(prompt.split())

        # For simple tasks, prefer cheaper options
        if task_complexity == 'simple' or prompt_length < 50:
            if self.huggingface_api_key:
                return 'huggingface'
            elif self.ollama_endpoint:
                return 'ollama'
            elif self.gemini_api_key:
                return 'gemini'

        # For complex tasks, prefer more capable models
        elif task_complexity == 'complex' or prompt_length > 500:
            if self.openai_api_key:
                return 'openai'
            elif self.anthropic_api_key:
                return 'anthropic'

        # Default to configured provider
        return self.provider

    def generate_response_optimized(self, prompt: str, context: Dict[str, Any], 
                                  agent_role: str = "AI Assistant", 
                                  task_complexity: str = 'medium') -> str:
        """
        Generate response using the most cost-effective provider for the task.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            agent_role: The role/persona of the agent
            task_complexity: Complexity level for provider selection
            
        Returns:
            Generated response string
        """
        optimal_provider = self.select_optimal_provider(prompt, task_complexity)

        # Temporarily switch to optimal provider
        original_provider = self.provider
        self.provider = optimal_provider

        try:
            response = self.generate_response(prompt, context, agent_role)
            return response
        finally:
            # Restore original provider
            self.provider = original_provider


# Global client instance
_llm_client = None


def get_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """
    Get or create LLM client instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LLMClient instance
    """
    global _llm_client
    
    # If no global client exists, create one
    if _llm_client is None:
        if config is None:
            # Try to load configuration from config file
            try:
                import json
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        file_config = json.load(f)
                    
                    # Extract API keys from config file
                    api_keys = file_config.get('api_keys', {})
                    framework = file_config.get('framework', {})
                    
                    config = {
                        'default_llm_provider': framework.get('default_llm_provider', 'openai'),
                        'default_model': framework.get('default_model', 'gpt-4'),
                        'openai_api_key': (
                            framework.get('openai_api_key') or 
                            api_keys.get('openai') or
                            os.getenv('OPENAI_API_KEY')
                        ),
                        'anthropic_api_key': (
                            framework.get('anthropic_api_key') or 
                            api_keys.get('anthropic') or
                            os.getenv('ANTHROPIC_API_KEY')
                        ),
                        'gemini_api_key': (
                            framework.get('gemini_api_key') or 
                            api_keys.get('gemini') or
                            os.getenv('GEMINI_API_KEY')
                        ),
                        'huggingface_api_key': (
                            framework.get('huggingface_api_key') or 
                            api_keys.get('huggingface') or
                            os.getenv('HUGGINGFACE_API_KEY')
                        ),
                        'ollama_endpoint': (
                            framework.get('ollama_endpoint') or 
                            api_keys.get('ollama_endpoint') or
                            os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
                        )
                    }
                else:
                    # Fallback to environment variables only
                    config = {
                        'default_llm_provider': 'openai',
                        'default_model': 'gpt-4',
                        'openai_api_key': os.getenv('OPENAI_API_KEY'),
                        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
                        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
                        'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
                        'ollama_endpoint': os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
                    }
            except Exception as e:
                logger.warning(f"Failed to load config file, using environment variables: {e}")
                # Fallback to environment variables only
                config = {
                    'default_llm_provider': 'openai',
                    'default_model': 'gpt-4',
                    'openai_api_key': os.getenv('OPENAI_API_KEY'),
                    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
                    'gemini_api_key': os.getenv('GEMINI_API_KEY'),
                    'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
                    'ollama_endpoint': os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
                }
        
        _llm_client = LLMClient(config)
    
    return _llm_client


def reset_llm_client():
    """Reset the global LLM client instance."""
    global _llm_client
    _llm_client = None