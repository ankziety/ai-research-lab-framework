#!/usr/bin/env python3
"""
Integration tests for the AI Research Lab Framework.

Tests the complete framework functionality including component integration,
workflow orchestration, and end-to-end scenarios.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import unittest

from ai_research_lab import AIPoweredResearchFramework, create_framework


class TestAIPoweredResearchFramework(unittest.TestCase):
    """Test cases for the integrated AI-Powered Research Framework."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'experiment_db_path': os.path.join(self.test_dir, 'test_experiments.db'),
            'output_dir': os.path.join(self.test_dir, 'output'),
            'manuscript_dir': os.path.join(self.test_dir, 'manuscripts'),
            'visualization_dir': os.path.join(self.test_dir, 'visualizations'),
            'auto_visualize': False,  # Disable to avoid matplotlib issues in tests
            'auto_critique': True
        }
        self.framework = AIPoweredResearchFramework(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_framework_initialization(self):
        """Test framework initializes correctly."""
        self.assertIsInstance(self.framework, AIPoweredResearchFramework)
        self.assertEqual(self.framework.config['output_dir'], self.config['output_dir'])
        
        # Check directories were created
        self.assertTrue(Path(self.config['output_dir']).exists())
        self.assertTrue(Path(self.config['manuscript_dir']).exists())
        self.assertTrue(Path(self.config['visualization_dir']).exists())
    
    def test_create_framework_factory(self):
        """Test framework factory function."""
        framework = create_framework(self.config)
        self.assertIsInstance(framework, AIPoweredResearchFramework)
    
    def test_experiment_runner_integration(self):
        """Test experiment runner integration."""
        params = {
            'algorithm': 'test_algorithm',
            'learning_rate': 0.001,
            'epochs': 10
        }
        
        results = self.framework.run_experiment(params)
        
        self.assertIn('experiment_id', results)
        self.assertIn('status', results)
        self.assertEqual(results['status'], 'completed')
        self.assertIn('computed_results', results)
        self.assertEqual(results['parameters'], params)
    
    def test_literature_retriever_integration(self):
        """Test literature retriever integration."""
        query = "machine learning neural networks"
        results = self.framework.retrieve_literature(query, max_results=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        # Check structure of results (stubbed data)
        if results:
            for result in results:
                self.assertIn('title', result)
                self.assertIn('authors', result)
    
    def test_manuscript_drafter_integration(self):
        """Test manuscript drafter integration."""
        results = [{
            'description': 'Test experiment',
            'data': {'accuracy': 0.95, 'loss': 0.05},
            'implication': 'High performance achieved'
        }]
        
        context = {
            'objective': 'Test machine learning model',
            'methods': 'Neural network training',
            'conclusion': 'Model performed well'
        }
        
        manuscript = self.framework.draft_manuscript(results, context)
        
        self.assertIsInstance(manuscript, str)
        self.assertIn('# ', manuscript)  # Should have title
        self.assertIn('## Abstract', manuscript)
        self.assertIn('## Introduction', manuscript)
        self.assertIn('## Methods', manuscript)
        self.assertIn('## Results', manuscript)
        self.assertIn('## Discussion', manuscript)
        self.assertIn('## Conclusion', manuscript)
    
    def test_critic_integration(self):
        """Test critic integration."""
        text = "This is a test research output with some content that should be critiqued."
        
        critique = self.framework.critique_output(text)
        
        self.assertIn('strengths', critique)
        self.assertIn('weaknesses', critique)
        self.assertIn('suggestions', critique)
        self.assertIn('overall_score', critique)
        self.assertIsInstance(critique['overall_score'], int)
        self.assertGreaterEqual(critique['overall_score'], 0)
        self.assertLessEqual(critique['overall_score'], 100)
    
    def test_specialist_registry_integration(self):
        """Test specialist registry integration."""
        # Test default specialists are registered
        specialists = self.framework.list_specialists()
        expected_specialists = [
            'experiment_runner', 'literature_retriever', 'manuscript_drafter',
            'critic', 'visualizer'
        ]
        
        for specialist in expected_specialists:
            self.assertIn(specialist, specialists)
        
        # Test getting a specialist
        experiment_specialist = self.framework.get_specialist('experiment_runner')
        self.assertEqual(experiment_specialist, self.framework.run_experiment)
    
    def test_complete_workflow(self):
        """Test complete workflow integration."""
        experiment_params = {
            'treatment': 'antioxidant_supplement',
            'dosage_mg': 500,
            'duration_weeks': 8
        }
        
        manuscript_context = {
            'objective': 'Evaluate antioxidant supplement effects on oxidative stress',
            'methods': 'Randomized controlled trial with biomarker analysis',
            'conclusion': 'Antioxidant supplementation significantly reduced oxidative stress markers'
        }
        
        literature_query = 'antioxidant supplements oxidative stress'
        
        results = self.framework.run_complete_workflow(
            experiment_params=experiment_params,
            manuscript_context=manuscript_context,
            literature_query=literature_query
        )
        
        # Check workflow results structure
        self.assertIn('workflow_id', results)
        self.assertIn('status', results)
        self.assertEqual(results['status'], 'completed')
        
        # Check experiment results
        self.assertIn('experiment', results)
        self.assertIn('experiment_id', results['experiment'])
        
        # Check manuscript results
        self.assertIn('manuscript', results)
        self.assertIn('content', results['manuscript'])
        self.assertIn('path', results['manuscript'])
        
        # Check manuscript file was created
        manuscript_path = results['manuscript']['path']
        self.assertTrue(Path(manuscript_path).exists())
        
        # Check critique results
        self.assertIn('critique', results)
        self.assertIn('overall_score', results['critique'])
        
        # Check literature results
        self.assertIn('literature', results)
        self.assertIsInstance(results['literature'], list)
    
    def test_workflow_config_save_load(self):
        """Test workflow configuration saving and loading."""
        config = {
            'experiment_params': {'test': 'value'},
            'manuscript_context': {'objective': 'test'},
            'literature_query': 'test query'
        }
        
        # Save config
        config_path = self.framework.save_workflow_config(config, 'test_config')
        self.assertTrue(Path(config_path).exists())
        
        # Load config
        loaded_config = self.framework.load_workflow_config('test_config')
        self.assertEqual(loaded_config, config)
    
    def test_workflow_with_minimal_params(self):
        """Test workflow with minimal parameters."""
        experiment_params = {'test_param': 1}
        manuscript_context = {'objective': 'minimal test'}
        
        results = self.framework.run_complete_workflow(
            experiment_params=experiment_params,
            manuscript_context=manuscript_context
        )
        
        self.assertEqual(results['status'], 'completed')
        self.assertIn('experiment', results)
        self.assertIn('manuscript', results)
        self.assertIn('critique', results)
    
    def test_error_handling_in_workflow(self):
        """Test error handling in workflows."""
        # Test with invalid experiment parameters
        with self.assertRaises(TypeError):
            self.framework.run_experiment({'invalid': object()})  # Non-serializable
    
    def test_auto_critique_in_manuscript_drafting(self):
        """Test auto-critique functionality."""
        results = [{'description': 'Test', 'data': {}, 'implication': 'Test result'}]
        context = {'objective': 'Test objective'}
        
        # Auto-critique is enabled in config
        manuscript = self.framework.draft_manuscript(results, context)
        
        # Should complete without error (critique happens in background)
        self.assertIsInstance(manuscript, str)
        self.assertGreater(len(manuscript), 0)
    
    def test_directory_creation(self):
        """Test that framework creates necessary directories."""
        # Directories should be created during initialization
        directories = [
            self.config['output_dir'],
            self.config['manuscript_dir'],
            self.config['visualization_dir']
        ]
        
        for directory in directories:
            self.assertTrue(Path(directory).exists())
            self.assertTrue(Path(directory).is_dir())


class TestFrameworkComponents(unittest.TestCase):
    """Test individual component integrations."""
    
    def setUp(self):
        """Set up framework for component testing."""
        self.test_dir = tempfile.mkdtemp()
        config = {
            'experiment_db_path': os.path.join(self.test_dir, 'test.db'),
            'output_dir': self.test_dir,
            'auto_visualize': False,
            'auto_critique': False
        }
        self.framework = create_framework(config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_experiment_runner_component(self):
        """Test experiment runner component specifically."""
        params = {'test': 'value', 'number': 42}
        results = self.framework.run_experiment(params)
        
        self.assertEqual(results['parameters'], params)
        self.assertIn('computed_results', results)
        
        # Check database persistence
        stored_experiment = self.framework.experiment_runner.get_experiment(
            results['experiment_id']
        )
        self.assertIsNotNone(stored_experiment)
        self.assertEqual(stored_experiment['parameters'], params)
    
    def test_literature_retriever_component(self):
        """Test literature retriever component specifically."""
        # Test with different parameters
        results_5 = self.framework.retrieve_literature("test query", 5)
        self.assertLessEqual(len(results_5), 5)
        
        results_2 = self.framework.retrieve_literature("test query", 2)
        self.assertLessEqual(len(results_2), 2)
    
    def test_critic_component(self):
        """Test critic component specifically."""
        # Test with different types of content
        short_text = "Short text."
        long_text = "This is a much longer text that contains multiple sentences and should provide more material for the critic to analyze. It includes various aspects that the critic can evaluate including structure, content, and overall quality."
        
        short_critique = self.framework.critique_output(short_text)
        long_critique = self.framework.critique_output(long_text)
        
        # Longer text should generally score higher
        self.assertIsInstance(short_critique['overall_score'], int)
        self.assertIsInstance(long_critique['overall_score'], int)


def run_integration_tests():
    """Run all integration tests."""
    print("Running AI Research Lab Framework Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAIPoweredResearchFramework))
    suite.addTest(unittest.makeSuite(TestFrameworkComponents))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return 1


if __name__ == '__main__':
    exit_code = run_integration_tests()
    exit(exit_code)