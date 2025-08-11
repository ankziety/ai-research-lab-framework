import os
import json
import pytest
from typing import Optional, Dict, Any

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from environment or config file."""
    # First check environment variables
    env_key = os.getenv(f"{provider.upper()}_API_KEY")
    if env_key:
        return env_key
    
    # Then check config file
    config_path = "config/config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_keys = config.get('api_keys', {})
                return api_keys.get(provider)
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None

def has_api_key(provider: str) -> bool:
    """Check if API key is available for a provider."""
    key = get_api_key(provider)
    return key is not None and key.strip() != ""

def skip_if_no_api_key(provider: str, reason: str = None) -> pytest.MarkDecorator:
    """Skip test if API key is not available."""
    if reason is None:
        reason = f"API key for {provider} not available. Set {provider.upper()}_API_KEY environment variable or add to config/config.json"
    
    return pytest.mark.skipif(
        not has_api_key(provider),
        reason=reason
    )

def get_test_config() -> Dict[str, Any]:
    """Get test configuration with available API keys."""
    config = {
        'default_llm_provider': 'openai',
        'default_model': 'gpt-4o',
        'api_keys': {}
    }
    
    # Add available API keys
    providers = ['openai', 'anthropic', 'gemini', 'huggingface']
    for provider in providers:
        key = get_api_key(provider)
        if key:
            config['api_keys'][provider] = key
    
    # Set default provider to first available one
    for provider in providers:
        if has_api_key(provider):
            config['default_llm_provider'] = provider
            break
    
    return config
