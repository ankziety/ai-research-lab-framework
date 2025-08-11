"""
Comprehensive tests for BaseAgent class.

This module tests all critical functionality of the BaseAgent class to improve
test coverage from 30% to 60%+ as part of the strategic improvement plan.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from agents.llm_client import reset_llm_client


class TestBaseAgentInitialization:
    """Test BaseAgent initialization and basic properties."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_basic_initialization(self):
        """Test basic BaseAgent initialization with required parameters."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing", "validation"],
            model_config={'default_model': 'gpt-4o'}
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.role == "Test Expert"
        assert agent.expertise == ["testing", "validation"]
        assert agent.model_config == {'default_model': 'gpt-4o'}
        assert agent.performance_metrics['tasks_completed'] == 0
        assert agent.performance_metrics['success_rate'] == 0.0
        assert agent.conversation_history == []
        assert agent.current_task is None
    
    def test_initialization_with_cost_manager(self):
        """Test BaseAgent initialization with cost manager."""
        mock_cost_manager = Mock()
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"],
            cost_manager=mock_cost_manager
        )
        
        assert agent.cost_manager == mock_cost_manager
    
    def test_initialization_without_model_config(self):
        """Test BaseAgent initialization without model config."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        assert agent.model_config == {}
    
    def test_initialization_without_cost_manager(self):
        """Test BaseAgent initialization without cost manager."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        assert agent.cost_manager is None


class TestBaseAgentPromptFormatting:
    """Test BaseAgent prompt formatting methods."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_format_role_prompt(self):
        """Test role prompt formatting."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Medical Expert",
            expertise=["cardiology", "neurology"]
        )
        
        context = {'goal': 'analyze patient symptoms'}
        role_prompt = agent._format_role_prompt(context)
        
        assert "Medical Expert" in role_prompt
        assert "cardiology" in role_prompt
        assert "neurology" in role_prompt
        assert "analyze patient symptoms" in role_prompt
    
    def test_format_context_section(self):
        """Test context section formatting."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Test with previous_discussion
        context = {
            'previous_discussion': 'Previous research shows...'
        }
        
        context_section = agent._format_context_section(context)
        assert "Previous research shows" in context_section
        
        # Test with context key
        context = {
            'context': 'Current context information'
        }
        
        context_section = agent._format_context_section(context)
        assert "Current context information" in context_section
        
        # Test with no context keys
        context = {
            'other_key': 'other value'
        }
        
        context_section = agent._format_context_section(context)
        assert context_section == ""
    
    def test_format_agenda_section(self):
        """Test agenda section formatting."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        context = {'agenda': 'Discuss testing strategies'}
        agenda_section = agent._format_agenda_section(context)
        
        assert "agenda" in agenda_section.lower()
        assert "testing strategies" in agenda_section
    
    def test_format_expectations_section(self):
        """Test expectations section formatting."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        context = {'expectations': 'Provide detailed analysis'}
        expectations_section = agent._format_expectations_section(context)
        
        assert "expectations" in expectations_section.lower()
        assert "detailed analysis" in expectations_section
    
    def test_format_structure_section(self):
        """Test structure section formatting."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        context = {'structure': 'Use bullet points'}
        structure_section = agent._format_structure_section(context)
        
        assert "structure" in structure_section.lower()
        assert "bullet points" in structure_section


class TestBaseAgentResponseGeneration:
    """Test BaseAgent response generation methods."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    @patch('agents.base_agent.get_llm_client')
    def test_generate_response(self, mock_get_llm_client):
        """Test basic response generation."""
        mock_client = Mock()
        mock_client.generate_response.return_value = "Test response"
        mock_get_llm_client.return_value = mock_client
        
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        prompt = "What is testing?"
        context = {'task_type': 'analysis'}
        
        response = agent.generate_response(prompt, context)
        
        assert response == "Test response"
        mock_client.generate_response.assert_called_once()
        
        # Verify the call includes agent context
        call_args = mock_client.generate_response.call_args
        # The method is called with: (enhanced_prompt, context_with_agent, agent_role, cost_manager)
        enhanced_prompt = call_args[0][0]
        context_with_agent = call_args[0][1]
        
        assert context_with_agent['agent_id'] == "test_agent"
        assert context_with_agent['task_type'] == "analysis"
        assert "What is testing?" in enhanced_prompt
    
    @patch('agents.base_agent.get_llm_client')
    def test_generate_team_lead_response(self, mock_get_llm_client):
        """Test team lead response generation."""
        mock_client = Mock()
        mock_client.generate_response.return_value = "Team lead response"
        mock_get_llm_client.return_value = mock_client
        
        agent = BaseAgent(
            agent_id="test_agent",
            role="Team Lead",
            expertise=["leadership"]
        )
        
        agenda = "Discuss project progress"
        team_members = ["member1", "member2"]
        context = {'project': 'test_project'}
        
        response = agent.generate_team_lead_response(agenda, team_members, context)
        
        assert response == "Team lead response"
        mock_client.generate_response.assert_called_once()
    
    @patch('agents.base_agent.get_llm_client')
    def test_generate_team_member_response(self, mock_get_llm_client):
        """Test team member response generation."""
        mock_client = Mock()
        mock_client.generate_response.return_value = "Team member response"
        mock_get_llm_client.return_value = mock_client
        
        agent = BaseAgent(
            agent_id="test_agent",
            role="Team Member",
            expertise=["development"]
        )
        
        agenda = "Discuss implementation"
        round_num = 1
        total_rounds = 3
        context = {'task': 'implement feature'}
        
        response = agent.generate_team_member_response(agenda, round_num, total_rounds, context)
        
        assert response == "Team member response"
        mock_client.generate_response.assert_called_once()
    
    @patch('agents.base_agent.get_llm_client')
    def test_generate_synthesis_response(self, mock_get_llm_client):
        """Test synthesis response generation."""
        mock_client = Mock()
        mock_client.generate_response.return_value = "Synthesis response"
        mock_get_llm_client.return_value = mock_client
        
        agent = BaseAgent(
            agent_id="test_agent",
            role="Synthesizer",
            expertise=["analysis"]
        )
        
        agenda = "Synthesize findings"
        team_inputs = ["Input 1", "Input 2", "Input 3"]
        context = {'research_question': 'What are the findings?'}
        
        response = agent.generate_synthesis_response(agenda, team_inputs, context)
        
        assert response == "Synthesis response"
        mock_client.generate_response.assert_called_once()


class TestBaseAgentTaskManagement:
    """Test BaseAgent task management functionality."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_assess_task_relevance(self):
        """Test task relevance assessment."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Medical Expert",
            expertise=["cardiology", "neurology"]
        )
        
        # Test relevant task
        relevant_task = "Analyze patient's heart condition"
        relevance_score = agent.assess_task_relevance(relevant_task)
        assert 0.0 <= relevance_score <= 1.0
        
        # Test irrelevant task
        irrelevant_task = "Design a website"
        relevance_score = agent.assess_task_relevance(irrelevant_task)
        assert 0.0 <= relevance_score <= 1.0
    
    def test_assign_task(self):
        """Test task assignment."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Test with a relevant task (should pass relevance threshold)
        task = {
            'id': 'task_1',
            'description': 'Test the system using testing expertise',
            'priority': 'high',
            'deadline': '2024-01-01'
        }
        
        success = agent.assign_task(task)
        # The task should be accepted if relevance score >= 0.3
        # Since this is a testing task for a testing expert, it should pass
        assert success is True
        assert agent.current_task == task
        
        # Test with an irrelevant task
        irrelevant_task = {
            'id': 'task_2',
            'description': 'Design a website for cooking recipes',
            'priority': 'low'
        }
        
        # Reset current task
        agent.current_task = None
        success = agent.assign_task(irrelevant_task)
        # This should fail due to low relevance
        assert success is False
        assert agent.current_task is None
    
    def test_complete_task(self):
        """Test task completion."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Assign a task first
        task = {'id': 'task_1', 'description': 'Test the system using testing expertise'}
        success = agent.assign_task(task)
        assert success is True  # Ensure task was assigned
        
        # Complete the task
        task_result = {'status': 'completed', 'findings': 'All tests passed'}
        result = agent.complete_task(task_result)
        
        assert result['success'] is True
        assert result['task_id'] == 'task_1'
        assert result['result'] == task_result
        assert agent.current_task is None
        assert agent.performance_metrics['tasks_completed'] == 1
    
    def test_update_performance_metrics(self):
        """Test performance metrics update."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Update metrics
        agent.update_performance_metrics(quality_score=0.8, success=True)
        
        assert agent.performance_metrics['success_rate'] > 0.0
        assert agent.performance_metrics['average_quality_score'] > 0.0
    
    def test_get_status(self):
        """Test status retrieval."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        status = agent.get_status()
        
        assert status['agent_id'] == "test_agent"
        assert status['role'] == "Test Expert"
        assert status['expertise'] == ["testing"]
        assert 'performance_metrics' in status
        assert 'current_task' in status
        assert 'is_active' in status
    
    def test_is_active(self):
        """Test active status check."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Should not be active by default (no task assigned)
        assert agent.is_active() is False
        
        # Assign a task
        task = {'id': 'task_1', 'description': 'Test the system using testing expertise'}
        success = agent.assign_task(task)
        assert success is True
        
        # Should be active when task is assigned
        assert agent.is_active() is True


class TestBaseAgentCommunication:
    """Test BaseAgent communication functionality."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    @patch('agents.base_agent.get_llm_client')
    def test_receive_message(self, mock_get_llm_client):
        """Test message reception and response."""
        mock_client = Mock()
        mock_client.generate_response.return_value = "Response to message"
        mock_get_llm_client.return_value = mock_client
        
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        response = agent.receive_message(
            sender_id="sender_1",
            message="Hello, how are you?",
            context={'conversation_type': 'casual'}
        )
        
        assert response == "Response to message"
        # The receive_message method adds both the incoming message and the response
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]['sender'] == "sender_1"
        assert agent.conversation_history[0]['message'] == "Hello, how are you?"
        assert agent.conversation_history[1]['sender'] == "test_agent"
        assert agent.conversation_history[1]['message'] == "Response to message"
    
    def test_get_conversation_history(self):
        """Test conversation history retrieval."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Add some conversation history
        agent.conversation_history = [
            {'sender_id': 'sender_1', 'message': 'Hello', 'timestamp': '2024-01-01'},
            {'sender_id': 'test_agent', 'message': 'Hi', 'timestamp': '2024-01-01'},
            {'sender_id': 'sender_1', 'message': 'How are you?', 'timestamp': '2024-01-01'}
        ]
        
        # Get all history
        history = agent.get_conversation_history()
        assert len(history) == 3
        
        # Get limited history
        limited_history = agent.get_conversation_history(limit=2)
        assert len(limited_history) == 2
    
    def test_clear_conversation_history(self):
        """Test conversation history clearing."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # Add some history
        agent.conversation_history = [
            {'sender_id': 'sender_1', 'message': 'Hello', 'timestamp': '2024-01-01'}
        ]
        
        # Clear history
        agent.clear_conversation_history()
        assert agent.conversation_history == []


class TestBaseAgentToolIntegration:
    """Test BaseAgent tool integration functionality."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_discover_available_tools(self):
        """Test tool discovery."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        task_description = "I need to analyze data"
        tools = agent.discover_available_tools(task_description)
        
        assert isinstance(tools, list)
        # Should return some basic tools even without external tool registry
    
    def test_request_tool(self):
        """Test tool request."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        context = {'task': 'analyze data'}
        tool = agent.request_tool('data_analyzer', context)
        
        # Should return None if tool not available
        assert tool is None
    
    def test_execute_with_tools(self):
        """Test tool execution."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        task = "Analyze this dataset"
        tools = [Mock(), Mock()]  # Mock tools
        
        result = agent.execute_with_tools(task, tools)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'output' in result  # The actual method returns 'output' not 'result'
        assert 'metadata' in result
    
    def test_build_custom_tool(self):
        """Test custom tool building."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        tool_spec = {
            'name': 'test_tool',
            'description': 'A test tool',
            'parameters': {'input': 'string'},
            'capabilities': ['testing']
        }
        
        tool = agent.build_custom_tool(tool_spec)
        
        # Should return None if not implemented
        assert tool is None
    
    def test_optimize_tool_usage(self):
        """Test tool usage optimization."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        task_description = "Analyze complex data"
        available_tools = [
            {'id': 'tool_1', 'name': 'Data Analyzer', 'capabilities': ['analysis']},
            {'id': 'tool_2', 'name': 'Visualizer', 'capabilities': ['visualization']}
        ]
        
        optimized_tools = agent.optimize_tool_usage(task_description, available_tools)
        
        assert isinstance(optimized_tools, list)
        assert len(optimized_tools) <= len(available_tools)
    
    def test_get_primary_domain(self):
        """Test primary domain identification."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Medical Expert",
            expertise=["cardiology", "neurology", "pediatrics"]
        )
        
        primary_domain = agent._get_primary_domain()
        
        # Should return one of the expertise domains
        assert primary_domain in ["cardiology", "neurology", "pediatrics"]
    
    def test_execute_tool(self):
        """Test tool execution."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        tool_id = "test_tool"
        task = {'input': 'test data'}
        context = {'agent_id': 'test_agent'}
        
        result = agent.execute_tool(tool_id, task, context)
        
        assert isinstance(result, dict)
        assert 'success' in result


class TestBaseAgentUtilityMethods:
    """Test BaseAgent utility methods."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_str_representation(self):
        """Test string representation."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        str_repr = str(agent)
        assert "test_agent" in str_repr
        assert "Test Expert" in str_repr
    
    def test_repr_representation(self):
        """Test repr representation."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        repr_repr = repr(agent)
        assert "test_agent" in repr_repr
        assert "Test Expert" in repr_repr
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        agent_dict = agent.to_dict()
        
        assert agent_dict['agent_id'] == "test_agent"
        assert agent_dict['role'] == "Test Expert"
        assert agent_dict['expertise'] == ["testing"]
        assert 'performance_metrics' in agent_dict
        assert 'conversation_count' in agent_dict  # The actual method returns conversation_count
        assert 'agent_type' in agent_dict


class TestBaseAgentEdgeCases:
    """Test BaseAgent edge cases and error handling."""
    
    def setup_method(self):
        """Reset LLM client state before each test."""
        reset_llm_client()
    
    def teardown_method(self):
        """Reset LLM client state after each test."""
        reset_llm_client()
    
    def test_empty_expertise_list(self):
        """Test initialization with empty expertise list."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=[]
        )
        
        assert agent.expertise == []
        assert agent._get_primary_domain() == "general"
    
    def test_none_context(self):
        """Test methods with None context."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # The methods should handle None context gracefully
        # But the actual implementation doesn't handle None, so we test the expected behavior
        with pytest.raises(TypeError):
            agent._format_context_section(None)
        
        with pytest.raises(AttributeError):
            agent._format_agenda_section(None)
        
        with pytest.raises(AttributeError):
            agent._format_expectations_section(None)
        
        with pytest.raises(AttributeError):
            agent._format_structure_section(None)
    
    def test_empty_task_description(self):
        """Test task relevance assessment with empty description."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        relevance_score = agent.assess_task_relevance("")
        assert 0.0 <= relevance_score <= 1.0
    
    def test_none_task(self):
        """Test task assignment with None task."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # This should raise an AttributeError since None has no 'get' method
        with pytest.raises(AttributeError):
            agent.assign_task(None)
    
    def test_complete_task_without_assignment(self):
        """Test completing a task without first assigning it."""
        agent = BaseAgent(
            agent_id="test_agent",
            role="Test Expert",
            expertise=["testing"]
        )
        
        # This should raise a ValueError since there's no current task
        with pytest.raises(ValueError, match="No active task to complete"):
            agent.complete_task({'status': 'completed'})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
