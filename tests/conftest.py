"""
Pytest configuration and fixtures for ai-research-lab-framework tests.

This file provides shared fixtures and configuration to ensure proper test isolation
and prevent state pollution between tests.
"""

import pytest
import os
import tempfile
import shutil
import json
from typing import Dict, Any, Optional

from agents.llm_client import LLMClient, reset_llm_client


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
    return get_api_key(provider) is not None

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


@pytest.fixture
def llm_client():
    """
    Provide a fresh LLM client instance for each test.
    
    This fixture ensures test isolation by creating a new client instance
    for each test, preventing global state pollution.
    """
    # Reset global state before each test
    reset_llm_client()
    
    # Create a minimal config for testing
    config = {
        'default_llm_provider': 'openai',
        'default_model': 'gpt-4o',
        'api_keys': {
            'openai': os.getenv('OPENAI_API_KEY', 'test-key'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', 'test-key'),
            'gemini': os.getenv('GEMINI_API_KEY', 'test-key'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY', 'test-key')
        }
    }
    
    # Create fresh client instance
    client = LLMClient(config)
    yield client
    
    # Reset global state after each test
    reset_llm_client()


@pytest.fixture
def temp_workspace():
    """
    Provide a temporary workspace directory for tests.
    
    This fixture creates a clean temporary directory for each test
    that needs file system operations.
    """
    temp_dir = tempfile.mkdtemp(prefix="ai_research_lab_test_")
    yield temp_dir
    
    # Cleanup: remove temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """
    Provide a mock configuration for testing.
    
    This fixture provides a consistent test configuration that can be
    used across multiple test modules.
    """
    return {
        'default_llm_provider': 'openai',
        'default_model': 'gpt-4o',
        'api_keys': {
            'openai': 'test-openai-key',
            'anthropic': 'test-anthropic-key',
            'gemini': 'test-gemini-key',
            'huggingface': 'test-huggingface-key'
        },
        'framework': {
            'default_llm_provider': 'openai',
            'default_model': 'gpt-4o'
        },
        'max_agents_per_research': 5,
        'budget_limit': 100.0,
        'agent_timeout': 1800,
        'agent_memory_limit': 1000
    }


@pytest.fixture
def tools_directory():
    """
    Provide a temporary tools directory for MCP tool testing.
    
    This fixture creates a clean directory for testing tool creation
    and discovery functionality.
    """
    temp_dir = tempfile.mkdtemp(prefix="mcp_tools_test_")
    yield temp_dir
    
    # Cleanup: remove temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


# Configure pytest to show local variables on failures
def pytest_configure(config):
    """Configure pytest behavior for better debugging."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "physics: marks tests as physics engine tests"
    )


# Ensure clean environment for each test
@pytest.fixture(autouse=True)
def clean_environment():
    """
    Automatically clean environment variables before each test.
    
    This fixture ensures that tests don't interfere with each other
    through environment variable pollution.
    """
    # Store original environment
    original_env = os.environ.copy()
    
    # Reset global LLM client to ensure test isolation
    reset_llm_client()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    
    # Reset global LLM client after each test
    reset_llm_client()
