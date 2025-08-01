"""
Unit tests for the PI Orchestrator module.

Tests the registration of specialists, task decomposition, routing,
and result aggregation functionality.
"""
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the repository root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi_orchestrator import PIOrchestrator
from vector_memory import VectorMemory


class TestPIOrchestrator(unittest.TestCase):
    """Test cases for PIOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.orchestrator = PIOrchestrator()
        self.memory = VectorMemory()
        
        # Create mock specialists
        self.mock_literature_specialist = Mock()
        self.mock_data_specialist = Mock()
        
        # Configure mock return values
        self.mock_literature_specialist.return_value = {
            "summary": "Literature review complete",
            "papers_found": 5,
            "key_findings": ["Finding 1", "Finding 2"]
        }
        
        self.mock_data_specialist.return_value = {
            "analysis": "Data analysis complete", 
            "statistics": {"mean": 2.5, "std": 1.2},
            "insights": ["Insight 1", "Insight 2"]
        }
    
    def test_register_specialist(self):
        """Test specialist registration functionality."""
        # Register specialists
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.register_specialist("data_analyst", self.mock_data_specialist)
        
        # Verify registration
        registered = self.orchestrator.get_registered_specialists()
        self.assertIn("literature_reviewer", registered)
        self.assertIn("data_analyst", registered)
        self.assertEqual(len(registered), 2)
        
        # Check provenance log
        log = self.orchestrator.get_provenance_log()
        registration_actions = [entry for entry in log if entry['action'] == 'register_specialist']
        self.assertEqual(len(registration_actions), 2)
        
        # Verify log entries contain correct role names
        registered_roles = [entry['role_name'] for entry in registration_actions]
        self.assertIn("literature_reviewer", registered_roles)
        self.assertIn("data_analyst", registered_roles)
    
    def test_set_memory(self):
        """Test setting vector memory instance."""
        self.orchestrator.set_memory(self.memory)
        
        # Check that memory was set (indirectly by checking provenance log)
        log = self.orchestrator.get_provenance_log()
        memory_actions = [entry for entry in log if entry['action'] == 'set_memory']
        self.assertEqual(len(memory_actions), 1)
    
    def test_run_research_task_literature_request(self):
        """Test running a research task that requires literature review."""
        # Setup
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.set_memory(self.memory)
        
        # Execute
        request = "Please review literature on machine learning algorithms"
        result = self.orchestrator.run_research_task(request)
        
        # Verify specialist was called
        self.mock_literature_specialist.assert_called_once_with(request)
        
        # Verify result structure
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['request'], request)
        self.assertIn('task_id', result)
        self.assertIn('results', result)
        self.assertIn('literature_reviewer', result['results'])
        
        # Verify the specialist result is included
        lit_result = result['results']['literature_reviewer']
        self.assertEqual(lit_result['summary'], "Literature review complete")
        self.assertEqual(lit_result['papers_found'], 5)
    
    def test_run_research_task_data_analysis_request(self):
        """Test running a research task that requires data analysis."""
        # Setup
        self.orchestrator.register_specialist("data_analyst", self.mock_data_specialist)
        
        # Execute
        request = "Analyze the experimental data and provide statistics"
        result = self.orchestrator.run_research_task(request)
        
        # Verify specialist was called
        self.mock_data_specialist.assert_called_once_with(request)
        
        # Verify result structure
        self.assertEqual(result['status'], 'completed')
        self.assertIn('data_analyst', result['results'])
        
        # Verify the specialist result
        data_result = result['results']['data_analyst']
        self.assertEqual(data_result['analysis'], "Data analysis complete")
        self.assertIn('statistics', data_result)
    
    def test_run_research_task_multiple_specialists(self):
        """Test running a research task that requires multiple specialists."""
        # Setup
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.register_specialist("data_analyst", self.mock_data_specialist)
        
        # Execute - request that should trigger both specialists
        request = "Review literature on data analysis methods and analyze our research data"
        result = self.orchestrator.run_research_task(request)
        
        # Verify both specialists were called
        self.mock_literature_specialist.assert_called_once_with(request)
        self.mock_data_specialist.assert_called_once_with(request)
        
        # Verify result contains both specialist outputs
        self.assertIn('literature_reviewer', result['results'])
        self.assertIn('data_analyst', result['results'])
        self.assertEqual(result['specialist_count'], 2)
    
    def test_run_research_task_no_matching_specialist(self):
        """Test running a research task with no matching specialists."""
        # Execute without registering any specialists
        request = "Please review literature on quantum computing"
        result = self.orchestrator.run_research_task(request)
        
        # Should still complete but with errors for missing specialists
        self.assertEqual(result['status'], 'completed')
        self.assertIsNotNone(result.get('errors'))
    
    def test_run_research_task_specialist_error(self):
        """Test handling of specialist execution errors."""
        # Setup specialist that raises an exception
        error_specialist = Mock(side_effect=Exception("Specialist error"))
        self.orchestrator.register_specialist("literature_reviewer", error_specialist)
        
        # Execute
        request = "Review literature on machine learning"
        result = self.orchestrator.run_research_task(request)
        
        # Should complete but with errors
        self.assertEqual(result['status'], 'completed')
        self.assertIn('errors', result)
        self.assertIn('literature_reviewer', result['errors'])
    
    def test_task_decomposition_logic(self):
        """Test that task decomposition correctly identifies required specialists."""
        # Register all specialist types
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.register_specialist("data_analyst", self.mock_data_specialist)
        
        # Test literature-focused request
        result1 = self.orchestrator.run_research_task("Review papers on neural networks")
        self.assertIn('literature_reviewer', result1['results'])
        self.assertNotIn('data_analyst', result1['results'])
        
        # Clear mocks
        self.mock_literature_specialist.reset_mock()
        self.mock_data_specialist.reset_mock()
        
        # Test data-focused request  
        result2 = self.orchestrator.run_research_task("Analyze experimental results statistically")
        self.assertIn('data_analyst', result2['results'])
        self.assertNotIn('literature_reviewer', result2['results'])
    
    def test_memory_integration(self):
        """Test integration with vector memory system."""
        # Setup
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.set_memory(self.memory)
        
        # Execute
        request = "Review literature on deep learning"
        result = self.orchestrator.run_research_task(request)
        
        # Verify entries were stored in memory
        stored_entries = self.memory.get_all()
        self.assertTrue(len(stored_entries) > 0)
        
        # Check that we can retrieve relevant context
        retrieved = self.memory.retrieve_context("literature", top_k=3)
        self.assertTrue(len(retrieved) > 0)
        
        # Verify some expected content is stored
        all_text = " ".join([entry['text'] for entry in stored_entries])
        self.assertIn("literature", all_text.lower())
    
    def test_provenance_logging(self):
        """Test that all actions are properly logged for provenance."""
        # Setup and execute a complete workflow
        self.orchestrator.register_specialist("literature_reviewer", self.mock_literature_specialist)
        self.orchestrator.set_memory(self.memory)
        result = self.orchestrator.run_research_task("Review machine learning literature")
        
        # Get provenance log
        log = self.orchestrator.get_provenance_log()
        
        # Verify key actions are logged
        actions = [entry['action'] for entry in log]
        self.assertIn('register_specialist', actions)
        self.assertIn('set_memory', actions)
        self.assertIn('start_research_task', actions)
        self.assertIn('decompose_request', actions)
        self.assertIn('execute_subtask', actions)
        self.assertIn('aggregate_results', actions)
        self.assertIn('complete_research_task', actions)
        
        # Verify all log entries have required fields
        for entry in log:
            self.assertIn('action', entry)
            self.assertIn('timestamp', entry)
            self.assertIn('id', entry)
    
    def test_clear_provenance_log(self):
        """Test clearing the provenance log."""
        # Generate some log entries
        self.orchestrator.register_specialist("test_specialist", Mock())
        self.assertGreater(len(self.orchestrator.get_provenance_log()), 0)
        
        # Clear and verify
        self.orchestrator.clear_provenance_log()
        self.assertEqual(len(self.orchestrator.get_provenance_log()), 0)
    
    def test_general_research_fallback(self):
        """Test fallback to general research when no specific specialists match."""
        # Register a general researcher
        general_specialist = Mock(return_value={"result": "General research complete"})
        self.orchestrator.register_specialist("general_researcher", general_specialist)
        
        # Execute with a request that doesn't match specific keywords
        request = "What is the meaning of life?"
        result = self.orchestrator.run_research_task(request)
        
        # Should use general researcher
        general_specialist.assert_called_once_with(request)
        self.assertIn('general_researcher', result['results'])


class TestVectorMemory(unittest.TestCase):
    """Test cases for VectorMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = VectorMemory()
    
    def test_store_and_retrieve_context(self):
        """Test basic storage and retrieval functionality."""
        # Store some context
        content_id = self.memory.store_context("Machine learning algorithms", {"topic": "AI"})
        self.assertIsNotNone(content_id)
        
        # Retrieve it
        results = self.memory.retrieve_context("machine learning", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Machine learning algorithms", results[0]['text'])
    
    def test_retrieve_multiple_results(self):
        """Test retrieving multiple relevant results."""
        # Store multiple related entries
        self.memory.store_context("Deep learning neural networks", {"type": "AI"})
        self.memory.store_context("Machine learning algorithms", {"type": "AI"})
        self.memory.store_context("Natural language processing", {"type": "AI"})
        self.memory.store_context("Weather patterns", {"type": "climate"})
        
        # Retrieve AI-related content
        results = self.memory.retrieve_context("learning", top_k=3)
        self.assertGreaterEqual(len(results), 2)  # Should find at least 2 learning-related entries
    
    def test_clear_memory(self):
        """Test clearing all stored content."""
        # Store some content
        self.memory.store_context("Test content", {})
        self.assertGreater(len(self.memory.get_all()), 0)
        
        # Clear and verify
        self.memory.clear()
        self.assertEqual(len(self.memory.get_all()), 0)


if __name__ == '__main__':
    unittest.main()