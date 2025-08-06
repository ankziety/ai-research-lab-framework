"""
Unit tests for the experiment module.

This module contains comprehensive tests for the ExperimentRunner class,
including testing of all required methods and error handling.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime

# Import the ExperimentRunner from the experiments module
from experiments.experiment import ExperimentRunner


class TestExperimentRunner:
    """Test class for ExperimentRunner functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        # Create a temporary directory and database file with proper permissions
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test_experiments.db')
        
        # Ensure the database file is created with proper permissions
        with sqlite3.connect(db_path) as conn:
            conn.execute('SELECT 1')  # Test connection
        
        yield db_path
        
        # Cleanup
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except OSError:
            pass  # Ignore cleanup errors
    
    @pytest.fixture
    def runner(self, temp_db):
        """Create an ExperimentRunner instance with temporary database."""
        return ExperimentRunner(db_path=temp_db)
    
    def test_init_default_db_path(self):
        """Test ExperimentRunner initialization with default database path."""
        runner = ExperimentRunner()
        assert runner.db_path is not None
        assert runner.db_path.endswith('experiments.db')
        # Cleanup
        if os.path.exists(runner.db_path):
            os.unlink(runner.db_path)
    
    def test_init_custom_db_path(self, temp_db):
        """Test ExperimentRunner initialization with custom database path."""
        runner = ExperimentRunner(db_path=temp_db)
        assert runner.db_path == temp_db
    
    def test_database_initialization(self, temp_db):
        """Test that the database is properly initialized with required tables."""
        runner = ExperimentRunner(db_path=temp_db)
        
        # Check that the experiments table exists
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='experiments'
            """)
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 'experiments'
    
    def test_run_experiment_basic(self, runner):
        """Test basic experiment execution."""
        params = {'learning_rate': 0.01, 'epochs': 10, 'model': 'neural_net'}
        result = runner.run_experiment(params)
        
        # Check required fields in result
        assert 'experiment_id' in result
        assert 'status' in result
        assert 'parameters' in result
        assert 'created_at' in result
        assert 'completed_at' in result
        assert 'computed_results' in result
        
        # Check values
        assert result['status'] == 'completed'
        assert result['parameters'] == params
        assert isinstance(result['experiment_id'], str)
        assert len(result['experiment_id']) > 0
    
    def test_run_experiment_with_numerical_params(self, runner):
        """Test experiment with numerical parameters."""
        params = {'param1': 10, 'param2': 20.5, 'param3': -5}
        result = runner.run_experiment(params)
        
        computed = result['computed_results']
        assert 'param_sum' in computed
        assert 'param_mean' in computed
        assert 'param_count' in computed
        
        assert computed['param_sum'] == 25.5
        assert computed['param_mean'] == 25.5 / 3
        assert computed['param_count'] == 3
    
    def test_run_experiment_with_string_params(self, runner):
        """Test experiment with string parameters."""
        params = {'name': 'test', 'model': 'cnn', 'dataset': 'mnist'}
        result = runner.run_experiment(params)
        
        computed = result['computed_results']
        assert 'string_param_count' in computed
        assert computed['string_param_count'] == 3
    
    def test_run_experiment_invalid_params_type(self, runner):
        """Test run_experiment with invalid parameter types."""
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            runner.run_experiment("not a dict")
        
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            runner.run_experiment(None)
        
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            runner.run_experiment(123)
    
    def test_run_experiment_non_serializable_params(self, runner):
        """Test run_experiment with non-JSON-serializable parameters."""
        # Create a non-serializable object
        class NonSerializable:
            pass
        
        params = {'valid_param': 'test', 'invalid_param': NonSerializable()}
        
        with pytest.raises(TypeError, match="Object of type NonSerializable is not JSON serializable"):
            runner.run_experiment(params)
    
    def test_record_result_basic(self, runner):
        """Test basic result recording."""
        # First run an experiment
        params = {'test_param': 'value'}
        experiment_result = runner.run_experiment(params)
        experiment_id = experiment_result['experiment_id']
        
        # Record a new result
        new_result = {
            'experiment_id': experiment_id,
            'custom_metric': 0.95,
            'accuracy': 0.87
        }
        
        # Should not raise any exception
        runner.record_result(new_result)
        
        # Verify the result was recorded
        stored_experiment = runner.get_experiment(experiment_id)
        assert stored_experiment is not None
        stored_result = stored_experiment['results']
        assert stored_result['custom_metric'] == 0.95
        assert stored_result['accuracy'] == 0.87
    
    def test_record_result_invalid_type(self, runner):
        """Test record_result with invalid result types."""
        with pytest.raises(ValueError, match="Result must be a dictionary"):
            runner.record_result("not a dict")
        
        with pytest.raises(ValueError, match="Result must be a dictionary"):
            runner.record_result(None)
        
        with pytest.raises(ValueError, match="Result must be a dictionary"):
            runner.record_result(123)
    
    def test_record_result_missing_experiment_id(self, runner):
        """Test record_result with missing experiment_id."""
        result = {'metric': 0.95}  # Missing experiment_id
        
        with pytest.raises(ValueError, match="Result must contain 'experiment_id' key"):
            runner.record_result(result)
    
    def test_record_result_nonexistent_experiment(self, runner):
        """Test record_result with non-existent experiment ID."""
        result = {'experiment_id': 'nonexistent-id', 'metric': 0.95}
        
        with pytest.raises(ValueError, match="Experiment with ID nonexistent-id not found"):
            runner.record_result(result)
    
    def test_record_result_non_serializable(self, runner):
        """Test record_result with non-JSON-serializable result."""
        # First run an experiment
        params = {'test_param': 'value'}
        experiment_result = runner.run_experiment(params)
        experiment_id = experiment_result['experiment_id']
        
        # Create a non-serializable object
        class NonSerializable:
            pass
        
        result = {
            'experiment_id': experiment_id,
            'valid_metric': 0.95,
            'invalid_metric': NonSerializable()
        }
        
        with pytest.raises(TypeError, match="Object of type NonSerializable is not JSON serializable"):
            runner.record_result(result)
    
    def test_get_experiment_existing(self, runner):
        """Test retrieving an existing experiment."""
        params = {'test_param': 'value'}
        experiment_result = runner.run_experiment(params)
        experiment_id = experiment_result['experiment_id']
        
        retrieved = runner.get_experiment(experiment_id)
        assert retrieved is not None
        assert retrieved['experiment_id'] == experiment_id
        assert retrieved['parameters'] == params
        assert retrieved['status'] == 'completed'
        assert 'created_at' in retrieved
        assert 'completed_at' in retrieved
    
    def test_get_experiment_nonexistent(self, runner):
        """Test retrieving a non-existent experiment."""
        result = runner.get_experiment('nonexistent-id')
        assert result is None
    
    def test_list_experiments_empty(self, runner):
        """Test listing experiments when none exist."""
        # Clear any existing data to ensure clean state
        conn = runner._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM experiments')
        conn.commit()
        
        experiments = runner.list_experiments()
        assert experiments == []
    
    def test_list_experiments_multiple(self, runner):
        """Test listing multiple experiments."""
        # Run several experiments
        params1 = {'param1': 'value1'}
        params2 = {'param2': 'value2'}
        params3 = {'param3': 'value3'}
        
        result1 = runner.run_experiment(params1)
        result2 = runner.run_experiment(params2)
        result3 = runner.run_experiment(params3)
        
        experiments = runner.list_experiments()
        assert len(experiments) == 3
        
        # Check that all experiments are present
        experiment_ids = {exp['experiment_id'] for exp in experiments}
        assert result1['experiment_id'] in experiment_ids
        assert result2['experiment_id'] in experiment_ids
        assert result3['experiment_id'] in experiment_ids
        
        # Check that experiments are ordered by created_at (newest first)
        created_times = [exp['created_at'] for exp in experiments]
        assert created_times == sorted(created_times, reverse=True)
    
    def test_experiment_persistence(self, temp_db):
        """Test that experiments persist across ExperimentRunner instances."""
        # Create first runner and run experiment
        runner1 = ExperimentRunner(db_path=temp_db)
        params = {'persistent_param': 'test_value'}
        result = runner1.run_experiment(params)
        experiment_id = result['experiment_id']
        
        # Create second runner with same database
        runner2 = ExperimentRunner(db_path=temp_db)
        
        # Should be able to retrieve the experiment
        retrieved = runner2.get_experiment(experiment_id)
        assert retrieved is not None
        assert retrieved['parameters'] == params
        
        # Should see the experiment in the list
        experiments = runner2.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]['experiment_id'] == experiment_id
    
    def test_concurrent_experiments(self, runner):
        """Test running multiple experiments concurrently (sequentially in this test)."""
        results = []
        params_list = [
            {'batch_size': 32, 'lr': 0.01},
            {'batch_size': 64, 'lr': 0.001},
            {'batch_size': 128, 'lr': 0.1}
        ]
        
        # Run experiments
        for params in params_list:
            result = runner.run_experiment(params)
            results.append(result)
        
        # Check that all experiments have unique IDs
        experiment_ids = {result['experiment_id'] for result in results}
        assert len(experiment_ids) == len(results)
        
        # Check that all experiments are in the database
        all_experiments = runner.list_experiments()
        assert len(all_experiments) == len(results)
    
    def test_empty_parameters(self, runner):
        """Test running experiment with empty parameters."""
        params = {}
        result = runner.run_experiment(params)
        
        assert result['parameters'] == {}
        assert result['status'] == 'completed'
        assert 'experiment_id' in result
        
        # Check computed results for empty params
        computed = result['computed_results']
        assert computed['param_count'] == 0
        assert computed['string_param_count'] == 0