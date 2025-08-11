"""
Integration Tests for Phase 2 Systems

Tests the integration of:
- LeReT Retrieval Optimization
- Enhanced VirtualLab Meeting Agents
- Persistent Cross-Agent Context Management
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from data.literature_retriever import LiteratureRetriever
from data.leret_retrieval_optimizer import LeReTRetrievalOptimizer
from agents.specialized_meeting_agents import (
    RetrieverAgent, HypothesisGeneratorAgent, ToolsmithAgent, TesterAgent
)
from memory.enhanced_context_manager import EnhancedContextManager
from memory.vector_database import VectorDatabase
from memory.context_manager import ContextManager


class TestLeReTIntegration:
    """Test LeReT retrieval optimization system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.literature_retriever = Mock(spec=LiteratureRetriever)
        self.leret_optimizer = LeReTRetrievalOptimizer(
            self.literature_retriever,
            config={'learning_rate': 0.01, 'exploration_rate': 0.1}
        )
    
    def test_leret_initialization(self):
        """Test LeReT optimizer initialization."""
        assert self.leret_optimizer.literature_retriever == self.literature_retriever
        assert len(self.leret_optimizer.query_strategies) == 5
        assert 'keyword_expansion' in self.leret_optimizer.query_strategies
        assert 'question_reformulation' in self.leret_optimizer.query_strategies
    
    def test_diverse_query_generation(self):
        """Test diverse query generation."""
        base_query = "neuron simulation physics"
        diverse_queries = self.leret_optimizer._generate_diverse_queries(base_query, 0)
        
        assert len(diverse_queries) > 0
        assert base_query in diverse_queries
        assert all(isinstance(q, str) for q in diverse_queries)
    
    def test_multi_hop_retrieval_structure(self):
        """Test multi-hop retrieval structure."""
        # Mock the literature retriever to return test data
        self.literature_retriever.search.return_value = [
            {'title': 'Test Paper', 'abstract': 'Test abstract', 'authors': ['Test Author']}
        ]
        
        result = self.leret_optimizer.multi_hop_retrieval(
            "test query", max_hops=2, max_results_per_hop=5
        )
        
        assert 'session_id' in result
        assert 'query_chain' in result
        assert 'documents' in result
        assert 'total_hops' in result
        assert result['total_hops'] >= 1
    
    def test_preference_learning(self):
        """Test preference learning functionality."""
        feedback = {
            'query': 'test query',
            'documents': [{'relevance_score': 0.8}],
            'user_ratings': {'doc1': 0.9}
        }
        
        self.leret_optimizer.update_preferences(feedback)
        
        assert len(self.leret_optimizer.feedback_history) == 1
        assert self.leret_optimizer.feedback_history[0]['query'] == 'test query'


class TestEnhancedMeetingAgents:
    """Test enhanced VirtualLab meeting agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model_config = {'model': 'gpt-4o', 'temperature': 0.7}
        self.mock_cost_manager = Mock()
        
        self.retriever_agent = RetrieverAgent(
            model_config=self.mock_model_config,
            cost_manager=self.mock_cost_manager
        )
        
        self.hypothesis_agent = HypothesisGeneratorAgent(
            model_config=self.mock_model_config,
            cost_manager=self.mock_cost_manager
        )
        
        self.toolsmith_agent = ToolsmithAgent(
            model_config=self.mock_model_config,
            cost_manager=self.mock_cost_manager
        )
        
        self.tester_agent = TesterAgent(
            model_config=self.mock_model_config,
            cost_manager=self.mock_cost_manager
        )
    
    def test_retriever_agent_initialization(self):
        """Test RetrieverAgent initialization."""
        assert self.retriever_agent.role == "Literature Retriever"
        assert "Information Retrieval" in self.retriever_agent.expertise
        assert "Literature Search" in self.retriever_agent.expertise
        assert len(self.retriever_agent.search_history) == 0
    
    def test_hypothesis_agent_initialization(self):
        """Test HypothesisGeneratorAgent initialization."""
        assert self.hypothesis_agent.role == "Hypothesis Generator"
        assert "Hypothesis Formulation" in self.hypothesis_agent.expertise
        assert "Research Design" in self.hypothesis_agent.expertise
        assert len(self.hypothesis_agent.hypothesis_history) == 0
    
    def test_toolsmith_agent_initialization(self):
        """Test ToolsmithAgent initialization."""
        assert self.toolsmith_agent.role == "Toolsmith"
        assert "Tool Development" in self.toolsmith_agent.expertise
        assert "API Integration" in self.toolsmith_agent.expertise
        assert len(self.toolsmith_agent.tool_registry) == 0
    
    def test_tester_agent_initialization(self):
        """Test TesterAgent initialization."""
        assert self.tester_agent.role == "Research Tester"
        assert "Experimental Design" in self.tester_agent.expertise
        assert "Quality Assurance" in self.tester_agent.expertise
        assert len(self.tester_agent.validation_history) == 0
    
    @patch.object(RetrieverAgent, 'generate_response')
    def test_retriever_literature_search_structure(self, mock_generate):
        """Test RetrieverAgent literature search structure."""
        mock_generate.return_value = "Mock search strategy"
        
        result = self.retriever_agent.conduct_literature_search(
            "test research question",
            {'max_results': 10}
        )
        
        assert 'search_strategy' in result
        assert 'search_results' in result
        assert 'synthesis' in result
        assert 'research_gaps' in result
        assert 'citations' in result
    
    @patch.object(HypothesisGeneratorAgent, 'generate_response')
    def test_hypothesis_generation_structure(self, mock_generate):
        """Test HypothesisGeneratorAgent hypothesis generation structure."""
        mock_generate.return_value = "Mock hypothesis"
        
        result = self.hypothesis_agent.generate_hypotheses(
            "test research question",
            {'literature_summary': 'test summary'},
            {'constraints': 'test constraints'}
        )
        
        assert 'primary_hypothesis' in result
        assert 'alternative_hypotheses' in result
        assert 'feasibility_analysis' in result
        assert 'question_analysis' in result


class TestEnhancedContextManager:
    """Test enhanced context manager with persistent cross-agent context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db = Mock(spec=VectorDatabase)
        self.base_context_manager = Mock(spec=ContextManager)
        
        self.enhanced_context = EnhancedContextManager(
            self.vector_db,
            self.base_context_manager,
            config={'persistence_dir': self.temp_dir}
        )
        
        self.session_id = "test_session_123"
        self.agent_id = "test_agent_456"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_enhanced_context_initialization(self):
        """Test EnhancedContextManager initialization."""
        assert self.enhanced_context.vector_db == self.vector_db
        assert self.enhanced_context.base_context_manager == self.base_context_manager
        assert len(self.enhanced_context.thought_history) == 0
        assert len(self.enhanced_context.vibe_check_history) == 0
        assert len(self.enhanced_context.agent_contexts) == 0
    
    def test_add_thought(self):
        """Test adding thought entries."""
        thought = "This is a test thought about the research process"
        
        result = self.enhanced_context.add_thought(
            self.session_id,
            self.agent_id,
            thought,
            1,
            5,
            "Analysis",
            {'test_metadata': 'test_value'}
        )
        
        assert result is True
        assert self.session_id in self.enhanced_context.thought_history
        assert len(self.enhanced_context.thought_history[self.session_id]) == 1
        
        thought_entry = self.enhanced_context.thought_history[self.session_id][0]
        assert thought_entry.thought == thought
        assert thought_entry.agent_id == self.agent_id
        assert thought_entry.stage == "Analysis"
    
    def test_add_vibe_check(self):
        """Test adding vibe_check entries."""
        goal = "Complete research analysis"
        plan = "Analyze data and generate insights"
        progress = "50% complete"
        uncertainties = ["Data quality", "Model accuracy"]
        
        result = self.enhanced_context.add_vibe_check(
            self.session_id,
            self.agent_id,
            goal,
            plan,
            progress,
            uncertainties,
            "Research context",
            0.8
        )
        
        assert result is True
        assert self.session_id in self.enhanced_context.vibe_check_history
        assert len(self.enhanced_context.vibe_check_history[self.session_id]) == 1
        
        vibe_check = self.enhanced_context.vibe_check_history[self.session_id][0]
        assert vibe_check.goal == goal
        assert vibe_check.plan == plan
        assert vibe_check.uncertainties == uncertainties
        assert vibe_check.assessment_score == 0.8
    
    def test_get_session_state(self):
        """Test getting session state."""
        # Add some test data
        self.enhanced_context.add_thought(
            self.session_id, self.agent_id, "Test thought", 1, 2, "Test", {}
        )
        
        session_state = self.enhanced_context.get_session_state(self.session_id)
        
        assert session_state['session_id'] == self.session_id
        assert 'agents' in session_state
        assert 'thoughts' in session_state
        assert 'vibe_checks' in session_state
        assert 'context_summary' in session_state
    
    def test_get_cross_agent_context(self):
        """Test getting cross-agent context."""
        # Add test data
        self.enhanced_context.add_thought(
            self.session_id, self.agent_id, "Test thought", 1, 2, "Test", {}
        )
        
        cross_context = self.enhanced_context.get_cross_agent_context(
            self.session_id, "test query"
        )
        
        assert cross_context['session_id'] == self.session_id
        assert 'agent_contexts' in cross_context
        assert 'thoughts_summary' in cross_context
        assert 'vibe_checks_summary' in cross_context
        assert 'relevant_context' in cross_context
        assert 'session_duration' in cross_context
        assert 'agent_activity' in cross_context
    
    def test_get_reasoning_trace(self):
        """Test getting reasoning trace."""
        # Add test thoughts
        self.enhanced_context.add_thought(
            self.session_id, self.agent_id, "Thought 1", 1, 3, "Analysis", {}
        )
        self.enhanced_context.add_thought(
            self.session_id, self.agent_id, "Thought 2", 2, 3, "Synthesis", {}
        )
        
        trace = self.enhanced_context.get_reasoning_trace(self.session_id)
        
        assert len(trace) == 2
        assert trace[0]['thought'] == "Thought 1"
        assert trace[0]['stage'] == "Analysis"
        assert trace[1]['thought'] == "Thought 2"
        assert trace[1]['stage'] == "Synthesis"
    
    def test_get_self_assessment_history(self):
        """Test getting self-assessment history."""
        # Add test vibe_checks
        self.enhanced_context.add_vibe_check(
            self.session_id, self.agent_id, "Goal 1", "Plan 1", "Progress 1",
            ["Uncertainty 1"], "Context 1", 0.7
        )
        self.enhanced_context.add_vibe_check(
            self.session_id, self.agent_id, "Goal 2", "Plan 2", "Progress 2",
            ["Uncertainty 2"], "Context 2", 0.8
        )
        
        assessments = self.enhanced_context.get_self_assessment_history(self.session_id)
        
        assert len(assessments) == 2
        assert assessments[0]['goal'] == "Goal 1"
        assert assessments[0]['assessment_score'] == 0.7
        assert assessments[1]['goal'] == "Goal 2"
        assert assessments[1]['assessment_score'] == 0.8


class TestPhase2Integration:
    """Test integration between Phase 2 systems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock components
        self.literature_retriever = Mock(spec=LiteratureRetriever)
        self.vector_db = Mock(spec=VectorDatabase)
        self.base_context_manager = Mock(spec=ContextManager)
        
        # Initialize systems
        self.leret_optimizer = LeReTRetrievalOptimizer(
            self.literature_retriever,
            config={'learning_rate': 0.01}
        )
        
        self.enhanced_context = EnhancedContextManager(
            self.vector_db,
            self.base_context_manager,
            config={'persistence_dir': self.temp_dir}
        )
        
        self.retriever_agent = RetrieverAgent()
        self.hypothesis_agent = HypothesisGeneratorAgent()
        
        self.session_id = "integration_test_session"
        self.agent_id = "integration_test_agent"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_leret_context_integration(self):
        """Test integration between LeReT and context management."""
        # Simulate LeReT retrieval
        self.literature_retriever.search.return_value = [
            {'title': 'Test Paper', 'abstract': 'Test abstract'}
        ]
        
        leret_result = self.leret_optimizer.multi_hop_retrieval(
            "test query", max_hops=1, max_results_per_hop=5
        )
        
        # Add to context
        self.enhanced_context.add_thought(
            self.session_id,
            self.agent_id,
            f"LeReT retrieval completed: {leret_result['total_documents']} documents found",
            1,
            3,
            "Retrieval",
            {'leret_result': leret_result}
        )
        
        # Verify integration
        session_state = self.enhanced_context.get_session_state(self.session_id)
        assert len(session_state['thoughts']) == 1
        
        thought = session_state['thoughts'][0]
        assert "LeReT retrieval completed" in thought.thought
        assert thought.metadata['leret_result']['total_documents'] == 1
    
    def test_agent_context_integration(self):
        """Test integration between meeting agents and context management."""
        # Simulate agent activities
        self.enhanced_context.add_thought(
            self.session_id,
            "retriever_agent",
            "Conducted literature search for neuroscience papers",
            1,
            4,
            "Research",
            {'search_results': 15}
        )
        
        self.enhanced_context.add_thought(
            self.session_id,
            "hypothesis_agent",
            "Generated primary hypothesis about neural dynamics",
            2,
            4,
            "Hypothesis",
            {'hypothesis_type': 'primary'}
        )
        
        # Get cross-agent context
        cross_context = self.enhanced_context.get_cross_agent_context(self.session_id)
        
        assert len(cross_context['agent_contexts']) >= 2
        assert 'retriever_agent' in cross_context['agent_contexts']
        assert 'hypothesis_agent' in cross_context['agent_contexts']
        
        # Verify thoughts summary
        thoughts_summary = cross_context['thoughts_summary']
        assert thoughts_summary['count'] == 2
        assert 'Research' in thoughts_summary['stages']
        assert 'Hypothesis' in thoughts_summary['stages']
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow integration."""
        # Step 1: LeReT retrieval
        self.literature_retriever.search.return_value = [
            {'title': 'Research Paper', 'abstract': 'Important findings'}
        ]
        
        leret_result = self.leret_optimizer.multi_hop_retrieval(
            "neural network simulation", max_hops=2, max_results_per_hop=3
        )
        
        # Step 2: Context logging
        self.enhanced_context.add_thought(
            self.session_id,
            "retriever_agent",
            f"Retrieved {leret_result['total_documents']} documents using LeReT",
            1,
            5,
            "Retrieval",
            {'leret_session': leret_result['session_id']}
        )
        
        # Step 3: Hypothesis generation (simulated)
        self.enhanced_context.add_thought(
            self.session_id,
            "hypothesis_agent",
            "Generated hypothesis: Neural networks exhibit emergent properties",
            2,
            5,
            "Hypothesis",
            {'hypothesis_strength': 0.8}
        )
        
        # Step 4: Self-assessment
        self.enhanced_context.add_vibe_check(
            self.session_id,
            "coordinator_agent",
            "Complete research workflow",
            "Retrieve → Analyze → Hypothesize → Test",
            "75% complete",
            ["Data quality", "Model complexity"],
            "Neuroscience research",
            0.85
        )
        
        # Step 5: Verify integration
        session_state = self.enhanced_context.get_session_state(self.session_id)
        reasoning_trace = self.enhanced_context.get_reasoning_trace(self.session_id)
        assessments = self.enhanced_context.get_self_assessment_history(self.session_id)
        
        assert len(reasoning_trace) == 2
        assert len(assessments) == 1
        assert assessments[0]['assessment_score'] == 0.85
        
        # Verify workflow progression
        stages = [thought['stage'] for thought in reasoning_trace]
        assert 'Retrieval' in stages
        assert 'Hypothesis' in stages
