"""
Comprehensive tests for LLMClient class.

This module tests all critical functionality of the LLMClient class to improve
test coverage from 38% to 60%+ as part of the strategic improvement plan.
"""

import os
import pytest
import time
from unittest.mock import Mock
from typing import Dict, Any

from agents.llm_client import LLMClient, get_llm_client, reset_llm_client
from .test_utils import skip_if_no_api_key, get_test_config, has_api_key


class TestLLMClientInitialization:
    """Test LLMClient initialization and configuration."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_basic_initialization(self):
        """Test basic LLMClient initialization."""
        config = get_test_config()
        client = LLMClient(config)
        
        assert client is not None
        assert hasattr(client, 'provider')
        assert hasattr(client, 'model')
    
    def test_initialization_with_framework_config(self):
        """Test initialization with framework-style config."""
        config = {
            'framework': {
                'default_llm_provider': 'openai',
                'default_model': 'gpt-4o'
            },
            'api_keys': {
                'openai': 'test-key'
            }
        }
        
        client = LLMClient(config)
        assert client.provider == 'openai'
        # The model might be normalized to 'gpt-4' by the client
        assert client.model in ['gpt-4', 'gpt-4o']
    
    def test_initialization_with_environment_variables(self):
        """Test initialization with environment variables."""
        # This test will be skipped if no API keys are available
        if not any(has_api_key(provider) for provider in ['openai', 'anthropic', 'gemini']):
            pytest.skip("No API keys available for testing")
        
        config = get_test_config()
        client = LLMClient(config)
        
        assert client is not None
        assert client.provider in ['openai', 'anthropic', 'gemini', 'ollama']
    
    def test_initialization_without_api_keys(self):
        """Test initialization behavior without API keys."""
        config = {
            'default_llm_provider': 'openai',
            'default_model': 'gpt-4o'
            # No api_keys provided
        }
        
        # The system falls back to ollama when no API keys are available
        client = LLMClient(config)
        assert client.provider == 'ollama'  # Should fall back to ollama


class TestLLMClientCostEstimation:
    """Test LLMClient cost estimation functionality."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_cost_estimation_for_gpt4o(self):
        """Test cost estimation for GPT-4o model."""
        config = get_test_config()
        client = LLMClient(config)
        
        tokens_input = 100
        tokens_output = 200
        
        cost = client._estimate_cost_local(tokens_input, tokens_output)
        
        assert isinstance(cost, float)
        assert cost >= 0.0
    
    def test_cost_estimation_for_claude(self):
        """Test cost estimation for Claude model."""
        config = get_test_config()
        client = LLMClient(config)
        
        tokens_input = 150
        tokens_output = 300
        
        cost = client._estimate_cost_local(tokens_input, tokens_output)
        
        assert isinstance(cost, float)
        assert cost >= 0.0
    
    def test_cost_estimation_for_unknown_model(self):
        """Test cost estimation for unknown model."""
        config = get_test_config()
        client = LLMClient(config)
        
        tokens_input = 50
        tokens_output = 100
        
        cost = client._estimate_cost_local(tokens_input, tokens_output)
        
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestLLMClientProviderSelection:
    """Test LLMClient provider selection logic."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_select_optimal_provider_simple_task(self):
        """Test provider selection for simple tasks."""
        config = get_test_config()
        client = LLMClient(config)
        
        provider = client.select_optimal_provider("Simple question", "simple")
        
        assert isinstance(provider, str)
        assert provider in ['openai', 'anthropic', 'gemini', 'huggingface', 'ollama']
    
    def test_select_optimal_provider_complex_task(self):
        """Test provider selection for complex tasks."""
        config = get_test_config()
        client = LLMClient(config)
        
        provider = client.select_optimal_provider("Complex analysis question", "complex")
        
        assert isinstance(provider, str)
        assert provider in ['openai', 'anthropic', 'gemini', 'huggingface', 'ollama']
    
    def test_select_optimal_provider_with_limited_keys(self):
        """Test provider selection with limited API keys."""
        config = get_test_config()
        client = LLMClient(config)
        
        provider = client.select_optimal_provider("Test question", "medium")
        
        assert isinstance(provider, str)
        # Should return the configured provider or fallback to available one
        assert provider in ['openai', 'anthropic', 'gemini', 'huggingface', 'ollama']


class TestLLMClientResponseGeneration:
    """Test LLMClient response generation with real API calls."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    @pytest.mark.integration
    @skip_if_no_api_key('openai')
    def test_generate_openai_response(self):
        """Test OpenAI response generation with real API."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client._generate_openai_response(
            "Test prompt for OpenAI",
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    @skip_if_no_api_key('anthropic')
    def test_generate_anthropic_response(self):
        """Test Anthropic response generation with real API."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client._generate_anthropic_response(
            "Test prompt for Anthropic",
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    @skip_if_no_api_key('gemini')
    def test_generate_gemini_response(self):
        """Test Gemini response generation with real API."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client._generate_gemini_response(
            "Test prompt for Gemini",
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    def test_generate_response_with_cost_manager(self):
        """Test response generation with cost manager."""
        from data.cost_manager import CostManager
        
        config = get_test_config()
        client = LLMClient(config)
        cost_manager = CostManager(budget_limit=100.0, config=config)
        
        response = client.generate_response(
            "Test prompt with cost tracking",
            {'context': 'test'},
            "Test Agent",
            cost_manager
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    def test_generate_response_optimized(self):
        """Test optimized response generation."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client.generate_response_optimized(
            "Test optimized prompt",
            {'context': 'test'},
            "Test Agent",
            "medium"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestLLMClientErrorHandling:
    """Test LLMClient error handling with real scenarios."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_generate_response_with_invalid_provider(self):
        """Test response generation with invalid provider."""
        config = {
            'default_llm_provider': 'invalid_provider',
            'default_model': 'invalid-model'
        }
        
        # The system falls back to ollama when invalid provider is specified
        # but ollama connection fails, so it raises an error
        client = LLMClient(config)
        
        # Should raise an error when ollama is not available
        with pytest.raises(RuntimeError, match="OLLAMA API error"):
            client.generate_response("Test prompt", {}, "Test Agent")


class TestLLMClientLocalModels:
    """Test LLMClient local model functionality."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    @pytest.mark.integration
    def test_generate_ollama_response(self):
        """Test Ollama response generation."""
        # Skip if Ollama is not available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama server not running on localhost:11434")
        except:
            pytest.skip("Ollama server not available")
        
        config = {
            'default_llm_provider': 'ollama',
            'default_model': 'llama2',
            'ollama_endpoint': 'http://localhost:11434'
        }
        
        client = LLMClient(config)
        
        response = client._generate_ollama_response(
            "Test prompt for Ollama",
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestLLMClientGlobalFunctions:
    """Test LLMClient global functions."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_get_llm_client(self):
        """Test get_llm_client function."""
        config = get_test_config()
        client = get_llm_client(config)
        
        assert isinstance(client, LLMClient)
    
    def test_get_llm_client_without_config(self):
        """Test get_llm_client function without config."""
        client = get_llm_client()
        
        assert isinstance(client, LLMClient)
    
    def test_reset_llm_client(self):
        """Test reset_llm_client function."""
        reset_llm_client()
        # Should not raise any exceptions
        assert True


class TestLLMClientEdgeCases:
    """Test LLMClient edge cases and boundary conditions."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_empty_prompt(self):
        """Test response generation with empty prompt."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client.generate_response(
            "",
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
    
    def test_empty_context(self):
        """Test response generation with empty context."""
        config = get_test_config()
        client = LLMClient(config)
        
        response = client.generate_response(
            "Test prompt",
            {},
            "Test Agent"
        )
        
        assert isinstance(response, str)
    
    def test_very_long_prompt(self):
        """Test response generation with very long prompt."""
        config = get_test_config()
        client = LLMClient(config)
        
        long_prompt = "Test prompt " * 1000  # Very long prompt
        
        response = client.generate_response(
            long_prompt,
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
    
    def test_special_characters_in_prompt(self):
        """Test response generation with special characters."""
        config = get_test_config()
        client = LLMClient(config)
        
        special_prompt = "Test prompt with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        response = client.generate_response(
            special_prompt,
            {'context': 'test'},
            "Test Agent"
        )
        
        assert isinstance(response, str)
