"""
Experiment module for running and tracking computational experiments.

This module provides the ExperimentRunner class for executing experiments,
recording results, and persisting data using SQLite.
"""

import sqlite3
import json
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# Thread-local storage for database connections
_thread_local = threading.local()

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
    
    def _get_db_connection(self):
        """Get a fresh database connection."""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with self._get_db_connection() as conn:
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
            params: Dictionary of experiment parameters
            
        Returns:
            Dictionary containing experiment results
        """
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Validate that parameters are JSON-serializable
        try:
            json.dumps(params)
        except TypeError as e:
            # Extract the actual non-serializable object type from the error
            error_msg = str(e)
            if "Object of type" in error_msg:
                raise TypeError(error_msg)
            else:
                raise TypeError("Parameters must be JSON-serializable")
        
        experiment_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Store experiment in database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiments (id, parameters, status, created_at)
                VALUES (?, ?, ?, ?)
            ''', (experiment_id, json.dumps(params), 'running', created_at))
            conn.commit()
            
            try:
                # Execute the experiment (placeholder for actual experiment logic)
                results = self._execute_experiment(params)
                
                # Update database with results
                completed_at = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE experiments 
                    SET results = ?, status = 'completed', completed_at = ?
                    WHERE id = ?
                ''', (json.dumps(results), completed_at, experiment_id))
                conn.commit()
                
                return {
                    'experiment_id': experiment_id,
                    'status': 'completed',
                    'parameters': params,
                    'created_at': created_at,
                    'completed_at': completed_at,
                    'computed_results': results
                }
            except Exception as e:
                # Update database with error status
                cursor.execute('''
                    UPDATE experiments 
                    SET status = 'failed', completed_at = ?
                    WHERE id = ?
                ''', (datetime.now().isoformat(), experiment_id))
                conn.commit()
                raise e
    
    def record_result(self, result: Dict[str, Any]) -> None:
        """
        Record additional results for an existing experiment.
        
        Args:
            result: Dictionary containing experiment results with 'experiment_id' key
            
        Raises:
            ValueError: If result is not a dictionary or missing experiment_id
            ValueError: If experiment with given ID doesn't exist
            TypeError: If result is not JSON-serializable
        """
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        
        if 'experiment_id' not in result:
            raise ValueError("Result must contain 'experiment_id' key")
        
        # Validate that result is JSON-serializable
        try:
            json.dumps(result)
        except TypeError as e:
            # Extract the actual non-serializable object type from the error
            error_msg = str(e)
            if "Object of type" in error_msg:
                raise TypeError(error_msg)
            else:
                raise TypeError("Result must be JSON-serializable")
        
        experiment_id = result['experiment_id']
        
        # Check if experiment exists
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM experiments WHERE id = ?', (experiment_id,))
            if not cursor.fetchone():
                raise ValueError(f"Experiment with ID {experiment_id} not found")
            
            # Update the experiment with new results
            completed_at = datetime.now().isoformat()
            cursor.execute('''
                UPDATE experiments 
                SET results = ?, status = 'completed', completed_at = ?
                WHERE id = ?
            ''', (json.dumps(result), completed_at, experiment_id))
            conn.commit()
    
    def _execute_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual experiment logic.
        
        This is a placeholder method. In a real implementation, this would
        contain the actual computational experiment logic.
        
        Args:
            params: Experiment parameters
            
        Returns:
            Experiment results
        """
        # Placeholder implementation with computed results based on parameters
        results = {
            'message': 'Experiment executed successfully',
            'parameters_used': params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add computed results based on parameter types
        if params:
            # Count numeric parameters
            numeric_params = {k: v for k, v in params.items() 
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if numeric_params:
                results['param_sum'] = sum(numeric_params.values())
                results['param_mean'] = sum(numeric_params.values()) / len(numeric_params)
                results['param_count'] = len(numeric_params)
            
            # Count string parameters
            string_params = {k: v for k, v in params.items() 
                           if isinstance(v, str)}
            if string_params:
                results['string_param_count'] = len(string_params)
            
            # Overall parameter count
            results['param_count'] = len(params)
        else:
            # Empty parameters case
            results['param_count'] = 0
            results['param_sum'] = 0
            results['param_mean'] = 0
            results['string_param_count'] = 0
        
        return results
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment results by ID.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            Experiment data or None if not found
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, parameters, results, status, created_at, completed_at
                FROM experiments
                WHERE id = ?
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'experiment_id': row[0],
                    'parameters': json.loads(row[1]),
                    'results': json.loads(row[2]) if row[2] else None,
                    'status': row[3],
                    'created_at': row[4],
                    'completed_at': row[5]
                }
            return None
    
    def list_experiments(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status ('running', 'completed', 'failed')
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment data
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT id, parameters, results, status, created_at, completed_at
                    FROM experiments
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                cursor.execute('''
                    SELECT id, parameters, results, status, created_at, completed_at
                    FROM experiments
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
            
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
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment from the database.
        
        Args:
            experiment_id: The experiment ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
        conn.commit()
        
        return cursor.rowcount > 0
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored experiments.
        
        Returns:
            Dictionary with experiment statistics
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Total experiments
        cursor.execute('SELECT COUNT(*) FROM experiments')
        total_experiments = cursor.fetchone()[0]
        
        # Experiments by status
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM experiments 
            GROUP BY status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Recent experiments (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute('''
            SELECT COUNT(*) 
            FROM experiments 
            WHERE created_at >= ?
        ''', (thirty_days_ago,))
        recent_experiments = cursor.fetchone()[0]
        
        return {
            'total_experiments': total_experiments,
            'status_counts': status_counts,
            'recent_experiments': recent_experiments
        }
    
    def close(self):
        """Close database connections."""
        # No explicit close needed for fresh connections, but keeping for consistency
        pass