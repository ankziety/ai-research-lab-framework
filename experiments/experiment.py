"""
Experiment module for running and tracking computational experiments.

This module provides the ExperimentRunner class for executing experiments,
recording results, and persisting data using SQLite.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentRunner:
    """
    A class for running and tracking computational experiments.
    
    This class provides methods to run experiments with given parameters,
    record results, and persist them to a SQLite database.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the ExperimentRunner.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses a default path.
        """
        if db_path is None:
            # Create experiments directory if it doesn't exist
            experiments_dir = Path(__file__).parent
            db_path = experiments_dir / "experiments.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    parameters TEXT NOT NULL,
                    results TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            ''')
            conn.commit()
    
    def run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an experiment with the given parameters.
        
        Args:
            params: Dictionary containing experiment parameters
            
        Returns:
            Dictionary containing experiment results including experiment_id,
            status, and any computed results
            
        Raises:
            ValueError: If params is not a valid dictionary
            TypeError: If params contains non-serializable values
        """
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Generate unique experiment ID
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        try:
            # Validate that params can be serialized to JSON
            params_json = json.dumps(params)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Parameters must be JSON-serializable: {e}")
        
        # Store experiment in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiments (id, parameters, status, created_at)
                VALUES (?, ?, ?, ?)
            ''', (experiment_id, params_json, 'running', timestamp))
            conn.commit()
        
        # Simulate experiment execution
        # In a real implementation, this would contain the actual experiment logic
        # Simulate experiment execution
        # Update status to 'running'
        self._update_experiment_status(experiment_id, 'running')
        
        # Simulate a delay for long-running experiments (e.g., using time.sleep)
        # In a real implementation, this would contain the actual experiment logic
        computed_results = self._compute_example_results(params)
        
        # Update status to 'completed' and record results
        results = {
            'experiment_id': experiment_id,
            'status': 'completed',
            'parameters': params,
            'created_at': timestamp,
            'completed_at': datetime.now().isoformat(),
            'computed_results': computed_results
        }
        self._update_experiment_status(experiment_id, 'completed', results)
        
        return results
    
    def record_result(self, result: Dict[str, Any]) -> None:
        """
        Record the result of an experiment.
        
        Args:
            result: Dictionary containing experiment results.
                   Must include 'experiment_id' key.
                   
        Raises:
            ValueError: If result is not a valid dictionary or missing experiment_id
            TypeError: If result contains non-serializable values
        """
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        
        if 'experiment_id' not in result:
            raise ValueError("Result must contain 'experiment_id' key")
        
        experiment_id = result['experiment_id']
        
        try:
            # Validate that result can be serialized to JSON
            result_json = json.dumps(result)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Result must be JSON-serializable: {e}")
        
        # Update the experiment with the recorded result
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if experiment exists
            cursor.execute('SELECT id FROM experiments WHERE id = ?', (experiment_id,))
            if not cursor.fetchone():
                raise ValueError(f"Experiment with ID {experiment_id} not found")
            
            # Update the experiment results
            cursor.execute('''
                UPDATE experiments 
                SET results = ?, completed_at = ?
                WHERE id = ?
            ''', (result_json, datetime.now().isoformat(), experiment_id))
            conn.commit()
    
    def _compute_example_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute example results based on parameters.
        
        This is a placeholder for actual experiment computation logic.
        """
        results = {}
        
        # Example: if parameters contain numerical values, compute some statistics
        numerical_params = {k: v for k, v in params.items() 
                          if isinstance(v, (int, float))}
        
        if numerical_params:
            values = list(numerical_params.values())
            results['param_sum'] = sum(values)
            results['param_mean'] = sum(values) / len(values)
            results['param_count'] = len(values)
        else:
            results['param_sum'] = 0
            results['param_mean'] = 0
            results['param_count'] = 0
        
        # Example: count string parameters
        string_params = {k: v for k, v in params.items() 
                        if isinstance(v, str)}
        results['string_param_count'] = len(string_params)
        
        return results
    
    def _update_experiment_status(self, experiment_id: str, status: str, 
                                results: Optional[Dict[str, Any]] = None) -> None:
        """Update experiment status and optionally results in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if results:
                results_json = json.dumps(results)
                cursor.execute('''
                    UPDATE experiments 
                    SET status = ?, results = ?, completed_at = ?
                    WHERE id = ?
                ''', (status, results_json, datetime.now().isoformat(), experiment_id))
            else:
                cursor.execute('''
                    UPDATE experiments 
                    SET status = ?
                    WHERE id = ?
                ''', (status, experiment_id))
            
            conn.commit()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an experiment by ID.
        
        Args:
            experiment_id: The unique experiment identifier
            
        Returns:
            Dictionary containing experiment data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, parameters, results, status, created_at, completed_at
                FROM experiments WHERE id = ?
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'experiment_id': row[0],
                'parameters': json.loads(row[1]),
                'results': json.loads(row[2]) if row[2] else None,
                'status': row[3],
                'created_at': row[4],
                'completed_at': row[5]
            }
    
    def list_experiments(self) -> list[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of dictionaries containing experiment data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, parameters, results, status, created_at, completed_at
                FROM experiments ORDER BY created_at DESC
            ''')
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    'experiment_id': row[0],
                    'parameters': json.loads(row[1]),
                    'results': json.loads(row[2]) if row[2] else None,
                    'status': row[3],
                    'created_at': row[4],
                    'completed_at': row[5]
                })
            
            return experiments