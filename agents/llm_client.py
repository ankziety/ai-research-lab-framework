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
        self.gemini_api_key = (
            config.get('gemini_api_key') or
            os.getenv('GEMINI_API_KEY')
        )
        self.huggingface_api_key = (
            config.get('huggingface_api_key') or
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
        """Validate that required API keys are available."""
        available_providers = []
        
        if self.provider == 'openai' and self.openai_api_key:
            available_providers.append('openai')
        elif self.provider == 'anthropic' and self.anthropic_api_key:
            available_providers.append('anthropic')
        elif self.provider == 'gemini' and self.gemini_api_key:
            available_providers.append('gemini')
        elif self.provider == 'huggingface' and self.huggingface_api_key:
            available_providers.append('huggingface')
        elif self.provider == 'ollama':
            available_providers.append('ollama')
        
        if not available_providers:
            logger.warning(f"No valid API keys found for {self.provider}. Using mock responses.")
        else:
            logger.info(f"LLM client configured with {len(available_providers)} available providers")
    
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
            elif self.provider == 'gemini' and self.gemini_api_key:
                return self._generate_gemini_response(prompt, context, agent_role)
            elif self.provider == 'huggingface' and self.huggingface_api_key:
                return self._generate_huggingface_response(prompt, context, agent_role)
            elif self.provider == 'ollama':
                return self._generate_ollama_response(prompt, context, agent_role)
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
    
    def _generate_gemini_response(self, prompt: str, context: Dict[str, Any], 
                                 agent_role: str) -> str:
        """Generate response using Google Gemini API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            system_prompt = f"You are a {agent_role}. Provide expert insights based on your domain knowledge."
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            response = model.generate_content(full_prompt)
            return response.text
            
        except ImportError:
            logger.warning("Google Generative AI library not installed. Using mock response.")
            return self._generate_mock_response(prompt, context, agent_role)
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
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
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
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
            
        except Exception as e:
            logger.error(f"OLLAMA API error: {str(e)}")
            return self._generate_mock_response(prompt, context, agent_role)
    
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
    
    def _generate_mock_response(self, prompt: str, context: Dict[str, Any], 
                               agent_role: str) -> str:
        """Generate a mock response for demonstration purposes."""
        import time
        # Simulate API delay
        time.sleep(0.1)
        
        # Extract key terms from prompt for context-aware mock responses
        prompt_lower = prompt.lower()
        
        # Domain-specific mock responses based on agent role
        if "research" in agent_role.lower():
            if "experiment" in prompt_lower or "study" in prompt_lower:
                return f"""Based on my analysis as a {agent_role}, I recommend a controlled experimental design with the following considerations:

1. Sample size calculation based on expected effect size
2. Randomization and blinding protocols  
3. Primary and secondary outcome measures
4. Statistical analysis plan with appropriate controls
5. Ethical considerations and participant safety

The proposed methodology should follow best practices for research integrity and reproducibility."""

            elif "literature" in prompt_lower or "review" in prompt_lower:
                return f"""As a {agent_role}, I suggest a systematic approach to literature analysis:

1. Comprehensive database search across PubMed, Web of Science, and relevant repositories
2. Inclusion/exclusion criteria based on research objectives
3. Quality assessment using established frameworks
4. Data extraction and synthesis methodology
5. Meta-analysis where appropriate

This approach will ensure comprehensive coverage of the existing evidence base."""

        elif "data" in agent_role.lower() or "statistics" in agent_role.lower():
            return f"""From a {agent_role} perspective, I recommend:

1. Exploratory data analysis to understand distributions and patterns
2. Appropriate statistical tests based on data characteristics
3. Effect size calculations and confidence intervals
4. Multiple comparison corrections where necessary
5. Visualization of key findings

The analysis should prioritize both statistical significance and practical significance."""

        elif "critic" in agent_role.lower():
            return f"""As a {agent_role}, I identify several areas for consideration:

Strengths:
- Clear research objectives and methodology
- Appropriate statistical approaches
- Consideration of ethical implications

Areas for improvement:
- Sample size justification could be more detailed
- Potential confounding variables need addressing
- Generalizability limitations should be discussed

Overall assessment: The approach is methodologically sound with minor improvements needed."""

        else:
            # Generic expert response
            return f"""As a {agent_role}, I provide the following expert analysis:

Key considerations:
1. The approach aligns with current best practices in the field
2. Methodology appears appropriate for the research objectives  
3. Potential limitations should be acknowledged
4. Results should be interpreted within the study context
5. Future research directions could explore related questions

This analysis provides a solid foundation for evidence-based decision making."""


# Global client instance
_llm_client = None


def get_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Get or create global LLM client instance."""
    global _llm_client
    
    if _llm_client is None or config:
        _llm_client = LLMClient(config or {})
    
    return _llm_client