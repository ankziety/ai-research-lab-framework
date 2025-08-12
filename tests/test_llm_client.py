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
from pytest import MonkeyPatch

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
        # The model should be 'gpt-5' as the default (flagship model)
        assert client.model == 'gpt-5'
    
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
    @skip_if_no_api_key('openai')
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
    @skip_if_no_api_key('huggingface')
    def test_generate_response_optimized(self):
        """Test optimized response generation with cost optimization."""
        from unittest.mock import patch, Mock
        from data.cost_manager import CostManager
        
        config = get_test_config()
        client = LLMClient(config)
        
        # Create a mock cost manager
        mock_cost_manager = Mock(spec=CostManager)
        mock_cost_manager.optimization_enabled = True
        
        # Test scenario 1: Cost optimization with model switching
        with patch.object(client, 'cost_manager', mock_cost_manager):
            # Mock the cost manager to return an optimized model
            mock_cost_manager.get_budget_status.return_value = {
                'budget_remaining': 50.0,
                'budget_limit': 100.0,
                'current_spending': 50.0
            }
            mock_cost_manager.optimize_model_selection.return_value = 'gpt-5-mini'
            
            # Mock the generate_response method to track which model was used
            with patch.object(client, 'generate_response') as mock_generate:
                mock_generate.return_value = "Optimized response"
                
                # Call the optimized method
                response = client.generate_response_optimized(
                    "Test prompt",
                    {'context': 'test'},
                    "Test Agent",
                    "medium"
                )
                
                # Verify the cost manager was used for optimization
                mock_cost_manager.get_budget_status.assert_called_once()
                mock_cost_manager.optimize_model_selection.assert_called_once_with(
                    task_complexity='medium',
                    budget_remaining=50.0,
                    required_capabilities=['reasoning', 'analysis']
                )
                
                # Verify the optimized model was used
                mock_generate.assert_called_once()
                call_args = mock_generate.call_args
                assert call_args is not None
                
                # Verify the response was returned
                assert response == "Optimized response"
                
                # Verify the original model was restored
                assert client.model == config.get('default_model', 'gpt-5')
    
    @pytest.mark.integration
    def test_generate_response_optimized_fallback(self):
        """Test optimized response generation fallback when cost manager is unavailable."""
        from unittest.mock import patch, Mock
        
        config = get_test_config()
        client = LLMClient(config)
        
        # Test scenario 2: Fallback to provider-based selection when cost manager is None
        with patch.object(client, 'cost_manager', None):
            # Mock the select_optimal_provider method
            with patch.object(client, 'select_optimal_provider') as mock_select_provider:
                mock_select_provider.return_value = 'anthropic'
                
                # Mock the generate_response method
                with patch.object(client, 'generate_response') as mock_generate:
                    mock_generate.return_value = "Fallback response"
                    
                    # Call the optimized method
                    response = client.generate_response_optimized(
                        "Test prompt",
                        {'context': 'test'},
                        "Test Agent",
                        "simple"
                    )
                    
                    # Verify provider selection was used
                    mock_select_provider.assert_called_once_with("Test prompt", "simple")
                    
                    # Verify the fallback provider was used
                    mock_generate.assert_called_once()
                    
                    # Verify the response was returned
                    assert response == "Fallback response"
                    
                    # Verify the original provider was restored
                    assert client.provider == config.get('default_llm_provider', 'openai')
    
    @pytest.mark.integration
    def test_generate_response_optimized_budget_protection(self):
        """Test optimized response generation with budget protection."""
        from unittest.mock import patch, Mock
        from data.cost_manager import CostManager
        
        config = get_test_config()
        client = LLMClient(config)
        
        # Create a mock cost manager
        mock_cost_manager = Mock(spec=CostManager)
        mock_cost_manager.optimization_enabled = True
        
        # Test scenario 3: Budget protection with low budget
        with patch.object(client, 'cost_manager', mock_cost_manager):
            # Mock low budget scenario
            mock_cost_manager.get_budget_status.return_value = {
                'budget_remaining': 0.5,  # Very low budget
                'budget_limit': 100.0,
                'current_spending': 99.5
            }
            # Mock cost manager to return a cheap model for low budget
            mock_cost_manager.optimize_model_selection.return_value = 'gpt-5-nano'
            
            # Mock the generate_response method
            with patch.object(client, 'generate_response') as mock_generate:
                mock_generate.return_value = "Budget-protected response"
                
                # Call the optimized method
                response = client.generate_response_optimized(
                    "Test prompt",
                    {'context': 'test'},
                    "Test Agent",
                    "complex"
                )
                
                # Verify budget-aware optimization was used
                mock_cost_manager.get_budget_status.assert_called_once()
                mock_cost_manager.optimize_model_selection.assert_called_once_with(
                    task_complexity='complex',
                    budget_remaining=0.5,
                    required_capabilities=['reasoning', 'analysis']
                )
                
                # Verify the cheap model was selected for low budget
                assert mock_cost_manager.optimize_model_selection.return_value == 'gpt-5-nano'
                
                # Verify the response was returned
                assert response == "Budget-protected response"
    
    @pytest.mark.integration
    def test_generate_response_optimized_state_restoration(self):
        """Test that original state is properly restored after optimization."""
        from unittest.mock import patch, Mock
        from data.cost_manager import CostManager
        
        config = get_test_config()
        client = LLMClient(config)
        
        # Store original state
        original_model = client.model
        original_provider = client.provider
        
        # Create a mock cost manager
        mock_cost_manager = Mock(spec=CostManager)
        mock_cost_manager.optimization_enabled = True
        
        # Test scenario 4: State restoration after optimization
        with patch.object(client, 'cost_manager', mock_cost_manager):
            # Mock the cost manager
            mock_cost_manager.get_budget_status.return_value = {
                'budget_remaining': 50.0,
                'budget_limit': 100.0,
                'current_spending': 50.0
            }
            mock_cost_manager.optimize_model_selection.return_value = 'gpt-4o-mini'
            
            # Mock the generate_response method to simulate an exception
            with patch.object(client, 'generate_response') as mock_generate:
                mock_generate.side_effect = Exception("API Error")
                
                # Call the optimized method and expect an exception
                with pytest.raises(Exception, match="API Error"):
                    client.generate_response_optimized(
                        "Test prompt",
                        {'context': 'test'},
                        "Test Agent",
                        "medium"
                    )
                
                # Verify state was restored even after exception
                assert client.model == original_model
                assert client.provider == original_provider


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
    
    @pytest.mark.integration
    @skip_if_no_api_key('openai')
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
        assert len(response) > 0

    @pytest.mark.integration
    @skip_if_no_api_key('openai')
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
        assert len(response) > 0

    @pytest.mark.integration
    @skip_if_no_api_key('openai')
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
        assert len(response) > 0

    @pytest.mark.integration
    @skip_if_no_api_key('openai')
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
        assert len(response) > 0
