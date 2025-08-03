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
        """Get thread-local database connection."""
        if not hasattr(_thread_local, 'experiment_db'):
            _thread_local.experiment_db = sqlite3.connect(self.db_path)
        return _thread_local.experiment_db
    
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
        experiment_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Store experiment in database
        conn = self._get_db_connection()
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
                'results': results,
                'created_at': created_at,
                'completed_at': completed_at
            }
            
        except Exception as e:
            # Update database with error
            cursor.execute('''
                UPDATE experiments 
                SET status = 'failed', results = ?
                WHERE id = ?
            ''', (json.dumps({'error': str(e)}), experiment_id))
            conn.commit()
            
            raise
    
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
        # Placeholder implementation
        return {
            'message': 'Experiment executed successfully',
            'parameters_used': params,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment results by ID.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            Experiment data or None if not found
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, parameters, results, status, created_at, completed_at
            FROM experiments
            WHERE id = ?
        ''', (experiment_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
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
        conn = self._get_db_connection()
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
                'id': row[0],
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
        if hasattr(_thread_local, 'experiment_db'):
            _thread_local.experiment_db.close()
            delattr(_thread_local, 'experiment_db')