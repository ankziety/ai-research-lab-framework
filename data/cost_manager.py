"""
Cost Management System for AI Research Framework

Provides real-time cost tracking, budget enforcement, and cost optimization
for all API calls and tool usage in the Virtual Lab framework.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Record of a single cost transaction."""
    timestamp: float
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    cost: float
    task_type: str
    agent_id: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelCost:
    """Cost configuration for a specific model."""
    model_name: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_tokens: int
    capabilities: List[str]
    reliability_score: float


class CostManager:
    """
    Comprehensive cost management system for AI research framework.
    
    Features:
    - Real-time cost tracking for all API calls
    - Budget limit enforcement with automatic model switching
    - Cost estimation before API calls
    - Detailed spending analytics and reporting
    - Integration with all LLM providers
    """
    
    def __init__(self, budget_limit: float, config: Dict[str, Any]):
        """Initialize the cost manager with budget and configuration."""
        self.budget_limit = budget_limit
        self.config = config
        self.current_spending = 0.0
        self.cost_history = []
        self.usage_stats = {}
        
        # Load model costs
        self.model_costs = self._load_model_costs(config)
        
        # Cost optimization settings
        self.optimization_enabled = config.get('cost_optimization', True)
        self.auto_switch_threshold = config.get('auto_switch_threshold', 0.8)
        
        # Load existing cost data if available
        self._load_cost_data()
        
        logger.info(f"Cost manager initialized with budget: ${budget_limit:.2f}")
    
    def _load_model_costs(self, config: Dict[str, Any]) -> Dict[str, ModelCost]:
        """Load model cost configurations with current OpenAI pricing."""
        default_costs = {
            # ===== OPENAI MODELS =====
            # GPT-5 models (latest flagship - August 2025)
            'gpt-5': ModelCost(
                model_name='gpt-5',
                provider='openai',
                input_cost_per_1k=0.00125,  # $1.25 per million tokens
                output_cost_per_1k=0.01,    # $10.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.98
            ),
            'gpt-5-mini': ModelCost(
                model_name='gpt-5-mini',
                provider='openai',
                input_cost_per_1k=0.00025,  # $0.25 per million tokens
                output_cost_per_1k=0.002,   # $2.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.95
            ),
            'gpt-5-nano': ModelCost(
                model_name='gpt-5-nano',
                provider='openai',
                input_cost_per_1k=0.00005,  # $0.05 per million tokens
                output_cost_per_1k=0.0004,  # $0.40 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.90
            ),
            # GPT-4.1 models (fine-tuning models)
            'gpt-4.1': ModelCost(
                model_name='gpt-4.1',
                provider='openai',
                input_cost_per_1k=0.003,    # $3.00 per million tokens
                output_cost_per_1k=0.012,   # $12.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'gpt-4.1-mini': ModelCost(
                model_name='gpt-4.1-mini',
                provider='openai',
                input_cost_per_1k=0.0008,   # $0.80 per million tokens
                output_cost_per_1k=0.0032,  # $3.20 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'gpt-4.1-nano': ModelCost(
                model_name='gpt-4.1-nano',
                provider='openai',
                input_cost_per_1k=0.0002,   # $0.20 per million tokens
                output_cost_per_1k=0.0008,  # $0.80 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.85
            ),
            # OpenAI o3/o4 models
            'o3': ModelCost(
                model_name='o3',
                provider='openai',
                input_cost_per_1k=0.002,    # $2.00 per million tokens
                output_cost_per_1k=0.008,   # $8.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'o4-mini': ModelCost(
                model_name='o4-mini',
                provider='openai',
                input_cost_per_1k=0.004,    # $4.00 per million tokens (fine-tuning)
                output_cost_per_1k=0.016,   # $16.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            # GPT-4o models (current generation)
            'gpt-4o': ModelCost(
                model_name='gpt-4o',
                provider='openai',
                input_cost_per_1k=0.005,    # $5.00 per million tokens
                output_cost_per_1k=0.02,    # $20.00 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'gpt-4o-mini': ModelCost(
                model_name='gpt-4o-mini',
                provider='openai',
                input_cost_per_1k=0.0006,   # $0.60 per million tokens
                output_cost_per_1k=0.0024,  # $2.40 per million tokens
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            # Legacy GPT-4 models (expensive - use sparingly)
            'gpt-4': ModelCost(
                model_name='gpt-4',
                provider='openai',
                input_cost_per_1k=0.03,
                output_cost_per_1k=0.06,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.95
            ),
            'gpt-4-turbo': ModelCost(
                model_name='gpt-4-turbo',
                provider='openai',
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.03,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.95
            ),
            'gpt-4-turbo-preview': ModelCost(
                model_name='gpt-4-turbo-preview',
                provider='openai',
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.03,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.95
            ),
            # Legacy GPT-3.5 models
            'gpt-3.5-turbo': ModelCost(
                model_name='gpt-3.5-turbo',
                provider='openai',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0015,
                max_tokens=16385,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'gpt-3.5-turbo-16k': ModelCost(
                model_name='gpt-3.5-turbo-16k',
                provider='openai',
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.004,
                max_tokens=16385,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'gpt-3.5-turbo-instruct': ModelCost(
                model_name='gpt-3.5-turbo-instruct',
                provider='openai',
                input_cost_per_1k=0.0015,
                output_cost_per_1k=0.002,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            
            # ===== ANTHROPIC MODELS =====
            'claude-3-5-sonnet': ModelCost(
                model_name='claude-3-5-sonnet',
                provider='anthropic',
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'claude-3-5-haiku': ModelCost(
                model_name='claude-3-5-haiku',
                provider='anthropic',
                input_cost_per_1k=0.00025,
                output_cost_per_1k=0.00125,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'claude-3-sonnet': ModelCost(
                model_name='claude-3-sonnet',
                provider='anthropic',
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'claude-3-haiku': ModelCost(
                model_name='claude-3-haiku',
                provider='anthropic',
                input_cost_per_1k=0.00025,
                output_cost_per_1k=0.00125,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'claude-3-opus': ModelCost(
                model_name='claude-3-opus',
                provider='anthropic',
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'claude-2.1': ModelCost(
                model_name='claude-2.1',
                provider='anthropic',
                input_cost_per_1k=0.008,
                output_cost_per_1k=0.024,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'claude-instant-1.2': ModelCost(
                model_name='claude-instant-1.2',
                provider='anthropic',
                input_cost_per_1k=0.0008,
                output_cost_per_1k=0.0024,
                max_tokens=100000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            ),
            
            # ===== GOOGLE/GEMINI MODELS =====
            'gemini-1.5-pro': ModelCost(
                model_name='gemini-1.5-pro',
                provider='google',
                input_cost_per_1k=0.00375,
                output_cost_per_1k=0.0105,
                max_tokens=1000000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.90
            ),
            'gemini-1.5-flash': ModelCost(
                model_name='gemini-1.5-flash',
                provider='google',
                input_cost_per_1k=0.000075,
                output_cost_per_1k=0.0003,
                max_tokens=1000000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'gemini-pro': ModelCost(
                model_name='gemini-pro',
                provider='google',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0015,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'gemini-pro-vision': ModelCost(
                model_name='gemini-pro-vision',
                provider='google',
                input_cost_per_1k=0.0025,
                output_cost_per_1k=0.0075,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.85
            ),
            'gemini-flash': ModelCost(
                model_name='gemini-flash',
                provider='google',
                input_cost_per_1k=0.000075,
                output_cost_per_1k=0.0003,
                max_tokens=1048576,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.80
            ),
            
            # ===== META/LLAMA MODELS =====
            'llama-3.1-8b-instruct': ModelCost(
                model_name='llama-3.1-8b-instruct',
                provider='meta',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'llama-3.1-70b-instruct': ModelCost(
                model_name='llama-3.1-70b-instruct',
                provider='meta',
                input_cost_per_1k=0.0007,
                output_cost_per_1k=0.0008,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama-3.1-405b-instruct': ModelCost(
                model_name='llama-3.1-405b-instruct',
                provider='meta',
                input_cost_per_1k=0.0024,
                output_cost_per_1k=0.0024,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            
            # ===== MISTRAL MODELS =====
            'mistral-large': ModelCost(
                model_name='mistral-large',
                provider='mistral',
                input_cost_per_1k=0.007,
                output_cost_per_1k=0.024,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'mistral-medium': ModelCost(
                model_name='mistral-medium',
                provider='mistral',
                input_cost_per_1k=0.0027,
                output_cost_per_1k=0.0084,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'mistral-small': ModelCost(
                model_name='mistral-small',
                provider='mistral',
                input_cost_per_1k=0.002,
                output_cost_per_1k=0.006,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'mixtral-8x7b': ModelCost(
                model_name='mixtral-8x7b',
                provider='mistral',
                input_cost_per_1k=0.00014,
                output_cost_per_1k=0.00042,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            
            # ===== COHERE MODELS =====
            'command-r-plus': ModelCost(
                model_name='command-r-plus',
                provider='cohere',
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'command-r': ModelCost(
                model_name='command-r',
                provider='cohere',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0015,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.85
            ),
            'command-light': ModelCost(
                model_name='command-light',
                provider='cohere',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0006,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            
            # ===== PERPLEXITY MODELS =====
            'llama-3.1-sonar-small-128k': ModelCost(
                model_name='llama-3.1-sonar-small-128k',
                provider='perplexity',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'llama-3.1-sonar-small-32k': ModelCost(
                model_name='llama-3.1-sonar-small-32k',
                provider='perplexity',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'llama-3.1-sonar-medium-128k': ModelCost(
                model_name='llama-3.1-sonar-medium-128k',
                provider='perplexity',
                input_cost_per_1k=0.0006,
                output_cost_per_1k=0.0006,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama-3.1-sonar-large-128k': ModelCost(
                model_name='llama-3.1-sonar-large-128k',
                provider='perplexity',
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.001,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            
            # ===== HUGGINGFACE MODELS =====
            'meta-llama-3-8b-instruct': ModelCost(
                model_name='meta-llama-3-8b-instruct',
                provider='huggingface',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'meta-llama-3-70b-instruct': ModelCost(
                model_name='meta-llama-3-70b-instruct',
                provider='huggingface',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0005,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'mistralai-mistral-7b-instruct': ModelCost(
                model_name='mistralai-mistral-7b-instruct',
                provider='huggingface',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'microsoft-phi-3-mini': ModelCost(
                model_name='microsoft-phi-3-mini',
                provider='huggingface',
                input_cost_per_1k=0.00005,
                output_cost_per_1k=0.00005,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            ),
            
            # ===== OLLAMA (LOCAL) MODELS =====
            'llama2': ModelCost(
                model_name='llama2',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            ),
            'llama2:13b': ModelCost(
                model_name='llama2:13b',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.75
            ),
            'llama2:70b': ModelCost(
                model_name='llama2:70b',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.80
            ),
            'codellama': ModelCost(
                model_name='codellama',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.75
            ),
            'mistral': ModelCost(
                model_name='mistral',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'mixtral': ModelCost(
                model_name='mixtral',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.80
            ),
            'phi': ModelCost(
                model_name='phi',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=2048,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.65
            ),
            'neural-chat': ModelCost(
                model_name='neural-chat',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            ),
            
            # ===== GROQ MODELS =====
            'llama-3.1-8b-instruct': ModelCost(
                model_name='llama-3.1-8b-instruct',
                provider='groq',
                input_cost_per_1k=0.00005,
                output_cost_per_1k=0.0001,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'llama-3.1-70b-instruct': ModelCost(
                model_name='llama-3.1-70b-instruct',
                provider='groq',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0002,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama-3.1-405b-instruct': ModelCost(
                model_name='llama-3.1-405b-instruct',
                provider='groq',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0004,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'mixtral-8x7b-32768': ModelCost(
                model_name='mixtral-8x7b-32768',
                provider='groq',
                input_cost_per_1k=0.00005,
                output_cost_per_1k=0.0001,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'gemma-7b-it': ModelCost(
                model_name='gemma-7b-it',
                provider='groq',
                input_cost_per_1k=0.00005,
                output_cost_per_1k=0.0001,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            
            # ===== TOGETHER AI MODELS =====
            'llama-3.1-8b-instruct': ModelCost(
                model_name='llama-3.1-8b-instruct',
                provider='together',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'llama-3.1-70b-instruct': ModelCost(
                model_name='llama-3.1-70b-instruct',
                provider='together',
                input_cost_per_1k=0.0007,
                output_cost_per_1k=0.0008,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama-3.1-405b-instruct': ModelCost(
                model_name='llama-3.1-405b-instruct',
                provider='together',
                input_cost_per_1k=0.0024,
                output_cost_per_1k=0.0024,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'mixtral-8x7b-instruct': ModelCost(
                model_name='mixtral-8x7b-instruct',
                provider='together',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'qwen2.5-7b-instruct': ModelCost(
                model_name='qwen2.5-7b-instruct',
                provider='together',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'qwen2.5-32b-instruct': ModelCost(
                model_name='qwen2.5-32b-instruct',
                provider='together',
                input_cost_per_1k=0.0004,
                output_cost_per_1k=0.0004,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.80
            ),
            'qwen2.5-72b-instruct': ModelCost(
                model_name='qwen2.5-72b-instruct',
                provider='together',
                input_cost_per_1k=0.0008,
                output_cost_per_1k=0.0008,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            
            # ===== FIREWORKS AI MODELS =====
            'llama-3.1-8b-instruct': ModelCost(
                model_name='llama-3.1-8b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0002,
                output_cost_per_1k=0.0002,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'llama-3.1-70b-instruct': ModelCost(
                model_name='llama-3.1-70b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0007,
                output_cost_per_1k=0.0008,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama-3.1-405b-instruct': ModelCost(
                model_name='llama-3.1-405b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0024,
                output_cost_per_1k=0.0024,
                max_tokens=8192,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'qwen2.5-7b-instruct': ModelCost(
                model_name='qwen2.5-7b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'qwen2.5-32b-instruct': ModelCost(
                model_name='qwen2.5-32b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0004,
                output_cost_per_1k=0.0004,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.80
            ),
            'qwen2.5-72b-instruct': ModelCost(
                model_name='qwen2.5-72b-instruct',
                provider='fireworks',
                input_cost_per_1k=0.0008,
                output_cost_per_1k=0.0008,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            
            # ===== DEEPSEEK MODELS =====
            'deepseek-coder': ModelCost(
                model_name='deepseek-coder',
                provider='deepseek',
                input_cost_per_1k=0.00014,
                output_cost_per_1k=0.00028,
                max_tokens=16384,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'deepseek-chat': ModelCost(
                model_name='deepseek-chat',
                provider='deepseek',
                input_cost_per_1k=0.00014,
                output_cost_per_1k=0.00028,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'deepseek-llm-7b-chat': ModelCost(
                model_name='deepseek-llm-7b-chat',
                provider='deepseek',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0002,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            
            # ===== ZEPHYR MODELS =====
            'zephyr-7b-beta': ModelCost(
                model_name='zephyr-7b-beta',
                provider='huggingface',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'zephyr-7b-alpha': ModelCost(
                model_name='zephyr-7b-alpha',
                provider='huggingface',
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0001,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            )
        }
        
        # Override with config values
        if 'model_costs' in config:
            for model_name, cost_config in config['model_costs'].items():
                if model_name in default_costs:
                    default_costs[model_name] = ModelCost(
                        model_name=model_name,
                        provider=cost_config.get('provider', default_costs[model_name].provider),
                        input_cost_per_1k=cost_config.get('input_cost_per_1k', default_costs[model_name].input_cost_per_1k),
                        output_cost_per_1k=cost_config.get('output_cost_per_1k', default_costs[model_name].output_cost_per_1k),
                        max_tokens=cost_config.get('max_tokens', default_costs[model_name].max_tokens),
                        capabilities=cost_config.get('capabilities', default_costs[model_name].capabilities),
                        reliability_score=cost_config.get('reliability_score', default_costs[model_name].reliability_score)
                    )
        
        return default_costs
    
    def estimate_cost(self, model: str, tokens_input: int, tokens_output: int = 0) -> float:
        """
        Estimate cost for a specific model and token usage.
        
        Args:
            model: Model name
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens (estimated)
            
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_costs:
            logger.warning(f"Unknown model: {model}, using default cost")
            return tokens_input * 0.001 + tokens_output * 0.002
        
        model_cost = self.model_costs[model]
        
        input_cost = (tokens_input / 1000) * model_cost.input_cost_per_1k
        output_cost = (tokens_output / 1000) * model_cost.output_cost_per_1k
        
        return input_cost + output_cost
    
    def can_afford(self, estimated_cost: float) -> bool:
        """
        Check if the estimated cost fits within budget.
        
        Args:
            estimated_cost: Estimated cost in USD
            
        Returns:
            True if cost is affordable
        """
        return (self.current_spending + estimated_cost) <= self.budget_limit
    
    def track_usage(self, model: str, tokens_input: int, tokens_output: int, 
                   actual_cost: float, task_type: str, agent_id: str, 
                   success: bool = True, error_message: Optional[str] = None):
        """
        Track actual usage and cost.
        
        Args:
            model: Model used
            tokens_input: Input tokens consumed
            tokens_output: Output tokens generated
            actual_cost: Actual cost incurred
            task_type: Type of task performed
            agent_id: ID of the agent that used the model
            success: Whether the operation was successful
            error_message: Error message if failed
        """
        cost_record = CostRecord(
            timestamp=time.time(),
            model=model,
            provider=self.model_costs.get(model, ModelCost(model, 'unknown', 0, 0, 0, [], 0)).provider,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=actual_cost,
            task_type=task_type,
            agent_id=agent_id,
            success=success,
            error_message=error_message
        )
        
        self.cost_history.append(cost_record)
        self.current_spending += actual_cost
        
        # Update usage statistics
        if model not in self.usage_stats:
            self.usage_stats[model] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'usage_count': 0,
                'success_count': 0,
                'last_used': None
            }
        
        stats = self.usage_stats[model]
        stats['total_tokens'] += tokens_input + tokens_output
        stats['total_cost'] += actual_cost
        stats['usage_count'] += 1
        stats['last_used'] = time.time()
        
        if success:
            stats['success_count'] += 1
        
        # Check budget alerts
        self._check_budget_alerts()
        
        # Save cost data periodically
        if len(self.cost_history) % 10 == 0:
            self._save_cost_data()
        
        logger.info(f"Cost tracked: ${actual_cost:.4f} for {model} ({tokens_input + tokens_output} tokens)")
    
    def optimize_model_selection(self, task_complexity: str, budget_remaining: float, 
                               required_capabilities: List[str] = None) -> str:
        """
        Select optimal model based on cost, capability, and budget.
        Prioritizes cost-effective models to prevent budget overflow.
        
        Args:
            task_complexity: 'simple', 'medium', or 'complex'
            budget_remaining: Remaining budget
            required_capabilities: List of required capabilities
            
        Returns:
            Optimal model name
        """
        if not self.optimization_enabled:
            return 'gpt-5'  # Default to most capable new model
        
        # Filter models by required capabilities
        available_models = []
        for model_name, model_cost in self.model_costs.items():
            if required_capabilities:
                if not all(cap in model_cost.capabilities for cap in required_capabilities):
                    continue
            available_models.append((model_name, model_cost))
        
        if not available_models:
            logger.warning("No models available with required capabilities")
            return 'gpt-5'
        
        # Budget protection: Avoid expensive models when budget is low
        expensive_models = ['gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o']  # Legacy expensive models
        if budget_remaining < 5.0:  # Less than $5 remaining
            available_models = [(name, cost) for name, cost in available_models 
                              if name not in expensive_models]
            logger.warning(f"Budget low (${budget_remaining:.2f}), excluding expensive models")
        
        if budget_remaining < 1.0:  # Less than $1 remaining
            # Only use the cheapest models
            available_models = [(name, cost) for name, cost in available_models 
                              if cost.input_cost_per_1k < 0.001]
            logger.warning(f"Budget very low (${budget_remaining:.2f}), using only cheapest models")
        
        if not available_models:
            logger.error("No affordable models available")
            return 'gpt-5-nano'  # Fallback to cheapest option
        
        # Score models based on cost efficiency and capability
        model_scores = []
        for model_name, model_cost in available_models:
            # Cost efficiency score (lower is better)
            avg_cost_per_1k = (model_cost.input_cost_per_1k + model_cost.output_cost_per_1k) / 2
            cost_score = 1.0 / (avg_cost_per_1k + 0.0001)  # Avoid division by zero
            
            # Capability score based on task complexity
            capability_score = 1.0
            if task_complexity == 'complex':
                capability_score = model_cost.reliability_score
            elif task_complexity == 'simple':
                capability_score = 0.5 + (model_cost.reliability_score * 0.5)
            
            # Budget consideration - heavily weight cost when budget is low
            budget_score = 1.0
            if budget_remaining < 10.0:  # Low budget threshold
                budget_score = cost_score / max(cost_score for _, _ in available_models)
            
            # Penalize expensive models more heavily
            if model_name in expensive_models:
                cost_score *= 0.1  # 90% penalty for expensive models
            
            # Bonus for new GPT-5 models
            if model_name.startswith('gpt-5'):
                cost_score *= 1.3  # 30% bonus for latest models
            elif model_name.startswith('gpt-4.1'):
                cost_score *= 1.1  # 10% bonus for fine-tuning models
            
            # Combined score
            total_score = cost_score * capability_score * budget_score
            model_scores.append((model_name, total_score))
        
        # Sort by score and return best model
        model_scores.sort(key=lambda x: x[1], reverse=True)
        optimal_model = model_scores[0][0]
        
        logger.info(f"Selected optimal model: {optimal_model} for {task_complexity} task (budget: ${budget_remaining:.2f})")
        return optimal_model
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending analytics."""
        total_usage = len(self.cost_history)
        successful_usage = sum(1 for record in self.cost_history if record.success)
        
        # Calculate spending by provider
        provider_spending = {}
        for record in self.cost_history:
            provider = record.provider
            if provider not in provider_spending:
                provider_spending[provider] = 0.0
            provider_spending[provider] += record.cost
        
        # Calculate spending by model
        model_spending = {}
        for record in self.cost_history:
            model = record.model
            if model not in model_spending:
                model_spending[model] = 0.0
            model_spending[model] += record.cost
        
        return {
            'budget_limit': self.budget_limit,
            'current_spending': self.current_spending,
            'budget_remaining': self.budget_limit - self.current_spending,
            'budget_utilization': self.current_spending / self.budget_limit,
            'total_usage_count': total_usage,
            'successful_usage_count': successful_usage,
            'success_rate': successful_usage / max(1, total_usage),
            'provider_spending': provider_spending,
            'model_spending': model_spending,
            'usage_stats': self.usage_stats,
            'budget_alerts': self.budget_alerts
        }
    
    def get_cost_analytics(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Get detailed cost analytics for the specified time period.
        
        Args:
            time_period_hours: Hours to look back for analytics
            
        Returns:
            Cost analytics dictionary
        """
        cutoff_time = time.time() - (time_period_hours * 3600)
        recent_records = [r for r in self.cost_history if r.timestamp >= cutoff_time]
        
        if not recent_records:
            return {'error': 'No cost data available for specified period'}
        
        # Calculate analytics
        total_cost = sum(r.cost for r in recent_records)
        total_tokens = sum(r.tokens_input + r.tokens_output for r in recent_records)
        avg_cost_per_token = total_cost / max(1, total_tokens)
        
        # Cost by task type
        task_costs = {}
        for record in recent_records:
            task_type = record.task_type
            if task_type not in task_costs:
                task_costs[task_type] = 0.0
            task_costs[task_type] += record.cost
        
        # Cost by agent
        agent_costs = {}
        for record in recent_records:
            agent_id = record.agent_id
            if agent_id not in agent_costs:
                agent_costs[agent_id] = 0.0
            agent_costs[agent_id] += record.cost
        
        return {
            'period_hours': time_period_hours,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'avg_cost_per_token': avg_cost_per_token,
            'record_count': len(recent_records),
            'task_costs': task_costs,
            'agent_costs': agent_costs,
            'model_usage': {r.model: sum(1 for rr in recent_records if rr.model == r.model) for r in recent_records}
        }
    
    def _check_budget_alerts(self):
        """Check and trigger budget alerts."""
        utilization = self.current_spending / self.budget_limit
        
        for threshold, alerted in self.budget_alerts.items():
            if utilization >= threshold and not alerted:
                logger.warning(f"Budget alert: {utilization:.1%} of budget used (${self.current_spending:.2f}/{self.budget_limit:.2f})")
                self.budget_alerts[threshold] = True
            elif utilization < threshold:
                self.budget_alerts[threshold] = False
    
    def _save_cost_data(self):
        """Save cost data to persistent storage."""
        try:
            data_dir = Path("data/costs")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            cost_data = {
                'budget_limit': self.budget_limit,
                'current_spending': self.current_spending,
                'cost_history': [asdict(record) for record in self.cost_history],
                'usage_stats': self.usage_stats,
                'budget_alerts': self.budget_alerts
            }
            
            with open(data_dir / "cost_data.json", 'w') as f:
                json.dump(cost_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def _load_cost_data(self):
        """Load cost data from persistent storage."""
        try:
            data_file = Path("data/costs/cost_data.json")
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                self.current_spending = data.get('current_spending', 0.0)
                self.usage_stats = data.get('usage_stats', {})
                self.budget_alerts = data.get('budget_alerts', {0.5: False, 0.8: False, 0.95: False})
                
                # Load cost history
                cost_history_data = data.get('cost_history', [])
                self.cost_history = []
                for record_data in cost_history_data:
                    self.cost_history.append(CostRecord(**record_data))
                
                logger.info(f"Loaded cost data: ${self.current_spending:.2f} spent")
                
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
    
    def reset_budget(self, new_budget: float):
        """Reset budget and clear history."""
        self.budget_limit = new_budget
        self.current_spending = 0.0
        self.cost_history.clear()
        self.usage_stats.clear()
        self.budget_alerts = {0.5: False, 0.8: False, 0.95: False}
        
        logger.info(f"Budget reset to ${new_budget:.2f}")
    
    def export_cost_report(self, filepath: str):
        """
        Export detailed cost report to file.
        
        Args:
            filepath: Path to export report
        """
        try:
            report = {
                'budget_status': self.get_budget_status(),
                'cost_analytics_24h': self.get_cost_analytics(24),
                'cost_analytics_7d': self.get_cost_analytics(168),
                'model_costs': {name: asdict(cost) for name, cost in self.model_costs.items()},
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Cost report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export cost report: {e}") 

    def check_openai_pricing(self, api_key: str = None) -> Dict[str, Any]:
        """
        Check real-time OpenAI pricing using their API.
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            
        Returns:
            Dictionary with current pricing information
        """
        try:
            import requests
            
            # Use provided API key or get from config
            if api_key is None:
                api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("No OpenAI API key available for pricing check")
                return {'error': 'No API key available'}
            
            # OpenAI doesn't have a public pricing API, but we can check model availability
            # and use their published pricing from their website
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Check available models
            models_response = requests.get(
                'https://api.openai.com/v1/models',
                headers=headers,
                timeout=10
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                
                # Current OpenAI pricing (as of August 2025)
                current_pricing = {
                    'gpt-5': {'input': 0.00125, 'output': 0.01},  # $1.25/$10.00 per million
                    'gpt-5-mini': {'input': 0.00025, 'output': 0.002},  # $0.25/$2.00 per million
                    'gpt-5-nano': {'input': 0.00005, 'output': 0.0004},  # $0.05/$0.40 per million
                    'gpt-4.1': {'input': 0.003, 'output': 0.012},  # $3.00/$12.00 per million
                    'gpt-4.1-mini': {'input': 0.0008, 'output': 0.0032},  # $0.80/$3.20 per million
                    'gpt-4.1-nano': {'input': 0.0002, 'output': 0.0008},  # $0.20/$0.80 per million
                    'o3': {'input': 0.002, 'output': 0.008},  # $2.00/$8.00 per million
                    'o4-mini': {'input': 0.004, 'output': 0.016},  # $4.00/$16.00 per million
                    'gpt-4o': {'input': 0.005, 'output': 0.02},  # $5.00/$20.00 per million
                    'gpt-4o-mini': {'input': 0.0006, 'output': 0.0024},  # $0.60/$2.40 per million
                    'gpt-4': {'input': 0.03, 'output': 0.06},  # Legacy expensive
                    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},  # Legacy
                }
                
                # Filter available models
                available_pricing = {
                    model: pricing for model, pricing in current_pricing.items()
                    if any(available_model.startswith(model) for available_model in available_models)
                }
                
                return {
                    'status': 'success',
                    'available_models': available_models,
                    'current_pricing': available_pricing,
                    'pricing_source': 'OpenAI official pricing (August 2025)',
                    'note': 'Pricing is per 1K tokens'
                }
            else:
                return {
                    'error': f'Failed to fetch models: {models_response.status_code}',
                    'response': models_response.text
                }
                
        except Exception as e:
            logger.error(f"Error checking OpenAI pricing: {e}")
            return {'error': str(e)}
    
    def check_anthropic_pricing(self, api_key: str = None) -> Dict[str, Any]:
        """
        Check Anthropic pricing information.
        
        Args:
            api_key: Anthropic API key (uses config if not provided)
            
        Returns:
            Dictionary with current pricing information
        """
        try:
            import requests
            
            # Use provided API key or get from config
            if api_key is None:
                api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                logger.warning("No Anthropic API key available for pricing check")
                return {'error': 'No API key available'}
            
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json'
            }
            
            # Check available models
            models_response = requests.get(
                'https://api.anthropic.com/v1/models',
                headers=headers,
                timeout=10
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                
                # Current Anthropic pricing (as of 2024)
                current_pricing = {
                    'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
                    'claude-3-5-haiku': {'input': 0.00025, 'output': 0.00125},
                    'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
                    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
                    'claude-3-opus': {'input': 0.015, 'output': 0.075},
                    'claude-2.1': {'input': 0.008, 'output': 0.024},
                    'claude-instant-1.2': {'input': 0.0008, 'output': 0.0024},
                }
                
                # Filter available models
                available_pricing = {
                    model: pricing for model, pricing in current_pricing.items()
                    if any(available_model.startswith(model) for available_model in available_models)
                }
                
                return {
                    'status': 'success',
                    'available_models': available_models,
                    'current_pricing': available_pricing,
                    'pricing_source': 'Anthropic official pricing (2024)',
                    'note': 'Pricing is per 1K tokens'
                }
            else:
                return {
                    'error': f'Failed to fetch models: {models_response.status_code}',
                    'response': models_response.text
                }
                
        except Exception as e:
            logger.error(f"Error checking Anthropic pricing: {e}")
            return {'error': str(e)}
    
    def check_google_pricing(self, api_key: str = None) -> Dict[str, Any]:
        """
        Check Google/Gemini pricing information.
        
        Args:
            api_key: Google API key (uses config if not provided)
            
        Returns:
            Dictionary with current pricing information
        """
        try:
            # Google doesn't have a public models API, but we can provide current pricing
            # Current Google/Gemini pricing (as of 2024)
            current_pricing = {
                'gemini-1.5-pro': {'input': 0.00375, 'output': 0.0105},
                'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
                'gemini-pro': {'input': 0.0005, 'output': 0.0015},
                'gemini-pro-vision': {'input': 0.0025, 'output': 0.0075},
                'gemini-flash': {'input': 0.000075, 'output': 0.0003},
            }
            
            return {
                'status': 'success',
                'available_models': list(current_pricing.keys()),
                'current_pricing': current_pricing,
                'pricing_source': 'Google official pricing (2024)',
                'note': 'Pricing is per 1K tokens. Model availability may vary by region.'
            }
                
        except Exception as e:
            logger.error(f"Error checking Google pricing: {e}")
            return {'error': str(e)}
    
    def check_mistral_pricing(self, api_key: str = None) -> Dict[str, Any]:
        """
        Check Mistral pricing information.
        
        Args:
            api_key: Mistral API key (uses config if not provided)
            
        Returns:
            Dictionary with current pricing information
        """
        try:
            import requests
            
            # Use provided API key or get from config
            if api_key is None:
                api_key = self.config.get('mistral_api_key') or os.getenv('MISTRAL_API_KEY')
            
            if not api_key:
                logger.warning("No Mistral API key available for pricing check")
                return {'error': 'No API key available'}
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Check available models
            models_response = requests.get(
                'https://api.mistral.ai/v1/models',
                headers=headers,
                timeout=10
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                
                # Current Mistral pricing (as of 2024)
                current_pricing = {
                    'mistral-large': {'input': 0.007, 'output': 0.024},
                    'mistral-medium': {'input': 0.0027, 'output': 0.0084},
                    'mistral-small': {'input': 0.002, 'output': 0.006},
                    'mixtral-8x7b': {'input': 0.00014, 'output': 0.00042},
                }
                
                # Filter available models
                available_pricing = {
                    model: pricing for model, pricing in current_pricing.items()
                    if any(available_model.startswith(model) for available_model in available_models)
                }
                
                return {
                    'status': 'success',
                    'available_models': available_models,
                    'current_pricing': available_pricing,
                    'pricing_source': 'Mistral official pricing (2024)',
                    'note': 'Pricing is per 1K tokens'
                }
            else:
                return {
                    'error': f'Failed to fetch models: {models_response.status_code}',
                    'response': models_response.text
                }
                
        except Exception as e:
            logger.error(f"Error checking Mistral pricing: {e}")
            return {'error': str(e)}
    
    def check_all_provider_pricing(self) -> Dict[str, Any]:
        """
        Check pricing for all major providers.
        
        Returns:
            Dictionary with pricing information for all providers
        """
        results = {
            'timestamp': time.time(),
            'providers': {}
        }
        
        # Check each provider
        providers = {
            'openai': self.check_openai_pricing,
            'anthropic': self.check_anthropic_pricing,
            'google': self.check_google_pricing,
            'mistral': self.check_mistral_pricing,
        }
        
        for provider_name, check_func in providers.items():
            try:
                results['providers'][provider_name] = check_func()
            except Exception as e:
                results['providers'][provider_name] = {'error': str(e)}
        
        return results
    
    def validate_cost_estimates(self) -> Dict[str, Any]:
        """
        Validate our cost estimates against current pricing for all providers.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'timestamp': time.time(),
            'models_checked': [],
            'discrepancies': [],
            'recommendations': [],
            'providers_checked': []
        }
        
        # Check all provider pricing
        all_pricing = self.check_all_provider_pricing()
        
        for provider_name, pricing_info in all_pricing.get('providers', {}).items():
            if 'current_pricing' in pricing_info:
                validation_results['providers_checked'].append(provider_name)
                
                for model_name, model_cost in self.model_costs.items():
                    if model_cost.provider == provider_name and model_name in pricing_info['current_pricing']:
                        current_pricing = pricing_info['current_pricing'][model_name]
                        
                        # Check for discrepancies
                        input_diff = abs(model_cost.input_cost_per_1k - current_pricing['input'])
                        output_diff = abs(model_cost.output_cost_per_1k - current_pricing['output'])
                        
                        if input_diff > 0.0001 or output_diff > 0.0001:
                            validation_results['discrepancies'].append({
                                'provider': provider_name,
                                'model': model_name,
                                'our_input_cost': model_cost.input_cost_per_1k,
                                'current_input_cost': current_pricing['input'],
                                'our_output_cost': model_cost.output_cost_per_1k,
                                'current_output_cost': current_pricing['output'],
                                'input_diff': input_diff,
                                'output_diff': output_diff
                            })
                        
                        validation_results['models_checked'].append(f"{provider_name}:{model_name}")
        
        # Add recommendations
        if validation_results['discrepancies']:
            validation_results['recommendations'].append(
                "Update cost estimates to match current provider pricing"
            )
        
        # Check for new models across all providers
        for provider_name, pricing_info in all_pricing.get('providers', {}).items():
            if 'available_models' in pricing_info:
                new_models = []
                for model in pricing_info['available_models']:
                    if not any(model.startswith(existing) for existing in self.model_costs.keys()):
                        new_models.append(model)
                
                if new_models:
                    validation_results['recommendations'].append(
                        f"Add pricing for new {provider_name} models: {new_models[:3]}"  # Limit to first 3
                    )
        
        # Add summary statistics
        validation_results['summary'] = {
            'total_models_checked': len(validation_results['models_checked']),
            'total_discrepancies': len(validation_results['discrepancies']),
            'providers_with_discrepancies': list(set(d['provider'] for d in validation_results['discrepancies'])),
            'accuracy_percentage': max(0, 100 - (len(validation_results['discrepancies']) / max(1, len(validation_results['models_checked'])) * 100))
        }
        
        return validation_results 