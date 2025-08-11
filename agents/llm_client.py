"""
LLM Client for agent interactions.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, etc.).
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Callable
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

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_manager=None, session_id: Optional[str] = None):
        """Initialize the LLM client with configuration and data manager integration."""
        self.config = config or {}
        
        # API keys - check both direct keys and api_keys dictionary
        api_keys = self.config.get('api_keys', {})
        framework = self.config.get('framework', {})
        
        self.openai_api_key = (
            self.config.get('openai_api_key') or 
            framework.get('openai_api_key') or
            api_keys.get('openai') or
            os.getenv('OPENAI_API_KEY')
        )
        self.anthropic_api_key = (
            self.config.get('anthropic_api_key') or 
            framework.get('anthropic_api_key') or
            api_keys.get('anthropic') or
            os.getenv('ANTHROPIC_API_KEY')
        )
        self.gemini_api_key = (
            self.config.get('gemini_api_key') or
            framework.get('gemini_api_key') or
            api_keys.get('gemini') or
            os.getenv('GEMINI_API_KEY')
        )
        self.huggingface_api_key = (
            self.config.get('huggingface_api_key') or
            framework.get('huggingface_api_key') or
            api_keys.get('huggingface') or
            os.getenv('HUGGINGFACE_API_KEY')
        )
        
        # Local model configurations
        self.ollama_endpoint = self.config.get('ollama_endpoint', 'http://localhost:11434')
        self.local_model_endpoint = self.config.get('local_model_endpoint', None)
        
        # Model configuration
        self.provider = self.config.get('default_llm_provider', 'openai')
        self.model = self.config.get('default_model', 'gpt-5-nano')
        
        # Cost management integration
        self.cost_manager = None
        self.budget_limit = self.config.get('budget_limit', 30.0)  # Default $30 budget
        self._initialize_cost_manager()
        
        # Provider pricing for cost optimization (updated August 2025)
        self.provider_costs = {
            'openai': {
                'gpt-5': 0.00125, 'gpt-5-mini': 0.00025, 'gpt-5-nano': 0.00005,
                'gpt-4.1': 0.003, 'gpt-4.1-mini': 0.0008, 'gpt-4.1-nano': 0.0002,
                'o3': 0.002, 'o4-mini': 0.004,
                'gpt-4o': 0.005, 'gpt-4o-mini': 0.0006,
                'gpt-4': 0.03, 'gpt-4-turbo': 0.01 # Legacy expensive
            },
            'anthropic': {'claude-3-sonnet': 0.003, 'claude-3-haiku': 0.00025},
            'gemini': {'gemini-pro': 0.0005, 'gemini-pro-vision': 0.002},
            'huggingface': {'default': 0.0001},  # Typically cheaper
            'ollama': {'default': 0.0},  # Free local inference
        }
        
        # Debug callback for UI integration
        self.debug_callback = None
        
        # Data manager integration for comprehensive logging
        self.data_manager = data_manager
        self.session_id = session_id
        
        self._validate_configuration()
        logger.info(f"LLM Client initialized with provider: {self.provider}, model: {self.model}, budget: ${self.budget_limit}")
    
    def _initialize_cost_manager(self):
        """Initialize the cost manager for budget protection."""
        try:
            from data.cost_manager import CostManager
            self.cost_manager = CostManager(self.budget_limit, self.config)
            logger.info(f"Cost manager initialized with budget: ${self.budget_limit}")
        except ImportError:
            logger.warning("Cost manager not available - budget protection disabled")
            self.cost_manager = None

    def set_debug_callback(self, callback: Callable[[str, str, Optional[Dict]], None]):
        """Set callback for capturing debug information."""
        self.debug_callback = callback
    
    def _capture_debug_info(self, debug_type: str, content: str, metadata: Optional[Dict] = None):
        """Capture debug information for UI display and data manager."""
        # Call debug callback for UI integration
        if self.debug_callback:
            try:
                self.debug_callback(debug_type, content, metadata)
            except Exception as e:
                logger.error(f"Error in debug callback: {e}")
        
        # Log to data manager if available
        if self.data_manager and self.session_id:
            try:
                # Extract request/response data from metadata
                request_data = None
                response_data = None
                error_info = None
                performance_data = None
                
                if metadata:
                    if debug_type == 'llm_api_call':
                        request_data = json.dumps(metadata, indent=2)
                    elif debug_type == 'llm_response':
                        response_data = content
                        performance_data = {
                            'tokens_input': metadata.get('tokens_input', 0),
                            'tokens_output': metadata.get('tokens_output', 0),
                            'execution_time': metadata.get('execution_time', 0),
                            'actual_cost': metadata.get('actual_cost', 0)
                        }
                    elif debug_type == 'llm_error':
                        error_info = content
                
                self.data_manager.persist_debug_log(
                    session_id=self.session_id,
                    debug_type=debug_type,
                    content=content,
                    metadata=metadata,
                    request_data=request_data,
                    response_data=response_data,
                    error_info=error_info,
                    performance_data=performance_data
                )
            except Exception as e:
                logger.error(f"Error logging to data manager: {e}")

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
        Generate a response using the configured LLM provider with cost tracking and budget protection.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            agent_role: The role/persona of the agent
            cost_manager: Optional cost manager for tracking (uses internal if not provided)
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If no valid LLM provider is available or budget exceeded
            Exception: If LLM generation fails
        """
        start_time = time.time()
        tokens_input = len(prompt.split())
        agent_id = context.get('agent_id', 'unknown')
        task_type = context.get('task_type', 'general')
        
        # Use internal cost manager if none provided
        if cost_manager is None:
            cost_manager = self.cost_manager
        
        # Budget protection: Check if we can afford this request
        estimated_tokens_output = tokens_input * 2  # Rough estimate
        if cost_manager:
            estimated_cost = cost_manager.estimate_cost(self.model, tokens_input, estimated_tokens_output)
            
            # Check if we can afford this request
            if not cost_manager.can_afford(estimated_cost):
                # Try to find a cheaper model
                budget_remaining = cost_manager.get_budget_status().get('budget_remaining', 0)
                cheaper_model = cost_manager.optimize_model_selection(
                    task_complexity='medium', 
                    budget_remaining=budget_remaining,
                    required_capabilities=['reasoning', 'analysis']
                )
                
                if cheaper_model != self.model:
                    logger.warning(f"Budget exceeded for {self.model} (${estimated_cost:.4f}), switching to {cheaper_model}")
                    original_model = self.model
                    self.model = cheaper_model
                    estimated_cost = cost_manager.estimate_cost(self.model, tokens_input, estimated_tokens_output)
                    
                    if not cost_manager.can_afford(estimated_cost):
                        raise RuntimeError(f"Insufficient budget for any model. Required: ${estimated_cost:.4f}, Available: ${budget_remaining:.2f}")
                else:
                    raise RuntimeError(f"Insufficient budget for LLM request: ${estimated_cost:.4f}, Available: ${budget_remaining:.2f}")
        else:
            estimated_cost = self._estimate_cost_local(tokens_input, estimated_tokens_output)
        
        # Capture API call for debug panel and data manager
        api_call_info = {
            'provider': self.provider,
            'model': self.model,
            'agent_role': agent_role,
            'agent_id': agent_id,
            'task_type': task_type,
            'prompt_length': len(prompt),
            'estimated_cost': estimated_cost,
            'timestamp': time.time(),
            'session_id': self.session_id
        }
        
        # Log to data manager if available
        if self.data_manager and self.session_id:
            try:
                self.data_manager.persist_chat_log(
                    session_id=self.session_id,
                    log_type='llm_prompt',
                    author=f"{agent_id} ({self.provider})",
                    message=prompt,
                    metadata=api_call_info,
                    message_category='llm_communication',
                    priority=1
                )
            except Exception as e:
                logger.error(f"Error logging LLM prompt to data manager: {e}")
        
        self._capture_debug_info("llm_api_call", f"API Call to {self.provider}/{self.model} (est. ${estimated_cost:.4f})", api_call_info)
        self._capture_debug_info("llm_prompt", f"Prompt sent to {self.provider}/{self.model}:\n\n{prompt}", api_call_info)
        
        # Generate response
        try:
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
                logger.info(f"LLM response generated: {self.provider}/{self.model}, cost: ${actual_cost:.4f}")
            
            # Log response to data manager
            if self.data_manager and self.session_id:
                try:
                    self.data_manager.persist_chat_log(
                        session_id=self.session_id,
                        log_type='llm_response',
                        author=f"{self.provider} ({self.model})",
                        message=response,
                        metadata={'response_length': len(response), 'actual_cost': actual_cost if cost_manager else estimated_cost},
                        message_category='llm_communication',
                        priority=1
                    )
                except Exception as e:
                    logger.error(f"Error logging LLM response to data manager: {e}")
            
            self._capture_debug_info("llm_response", f"Response from {self.provider}/{self.model}:\n\n{response}", api_call_info)
            
            return response
            
        except Exception as e:
            # Track failed usage
            if cost_manager:
                cost_manager.track_usage(
                    model=self.model,
                    tokens_input=tokens_input,
                    tokens_output=0,
                    actual_cost=0.0,
                    task_type=task_type,
                    agent_id=agent_id,
                    success=False,
                    error_message=str(e)
                )
            raise

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
                model="claude-3-5-sonnet-20240620",
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
        Generate response using the most cost-effective provider for the task with budget protection.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            agent_role: The role/persona of the agent
            task_complexity: Complexity level for provider selection
            
        Returns:
            Generated response string
        """
        # Use cost manager for optimal model selection if available
        if self.cost_manager:
            budget_status = self.cost_manager.get_budget_status()
            budget_remaining = budget_status.get('budget_remaining', 0)
            
            # Get optimal model based on cost and budget
            optimal_model = self.cost_manager.optimize_model_selection(
                task_complexity=task_complexity,
                budget_remaining=budget_remaining,
                required_capabilities=['reasoning', 'analysis']
            )
            
            # Temporarily switch to optimal model
            original_model = self.model
            self.model = optimal_model
            
            try:
                response = self.generate_response(prompt, context, agent_role, self.cost_manager)
                return response
            finally:
                # Restore original model
                self.model = original_model
        else:
            # Fallback to provider-based selection
            optimal_provider = self.select_optimal_provider(prompt, task_complexity)
            original_provider = self.provider
            self.provider = optimal_provider
            
            try:
                response = self.generate_response(prompt, context, agent_role)
                return response
            finally:
                # Restore original provider
                self.provider = original_provider

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending analytics."""
        if self.cost_manager:
            return self.cost_manager.get_budget_status()
        else:
            return {
                'budget_limit': self.budget_limit,
                'current_spending': 0.0,
                'budget_remaining': self.budget_limit,
                'budget_utilization': 0.0,
                'cost_manager_available': False
            }
    
    def set_budget_limit(self, budget_limit: float):
        """Update the budget limit."""
        self.budget_limit = budget_limit
        if self.cost_manager:
            self.cost_manager.reset_budget(budget_limit)
        logger.info(f"Budget limit updated to: ${budget_limit}")
    
    def check_pricing(self) -> Dict[str, Any]:
        """Check current OpenAI pricing and validate our cost estimates."""
        if self.cost_manager:
            return self.cost_manager.check_openai_pricing()
        else:
            return {'error': 'Cost manager not available'}
    
    def validate_cost_estimates(self) -> Dict[str, Any]:
        """Validate our cost estimates against current pricing."""
        if self.cost_manager:
            return self.cost_manager.validate_cost_estimates()
        else:
            return {'error': 'Cost manager not available'}


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
                        'default_model': framework.get('default_model', 'gpt-4.1'),
                        'budget_limit': framework.get('budget_limit', 30.0),  # Default $30 budget
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
                        'default_model': 'gpt-4.1',
                        'budget_limit': 30.0, # Default budget
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
                    'default_model': 'gpt-4.1',
                    'budget_limit': 30.0, # Default budget
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