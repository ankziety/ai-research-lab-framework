"""
Distributed Compute Manager

A simple module for scheduling computational tasks to remote nodes.
This is an MVP implementation that logs actions and returns fake responses.
"""

import uuid
import time
from typing import Dict, Optional


class ComputeManager:
    """Manages distributed computational tasks across remote nodes."""
    
    def __init__(self):
        """Initialize the compute manager."""
        self._tasks: Dict[str, Dict] = {}
        self._task_counter = 0
    
    def schedule(self, task: dict, node: str) -> str:
        """
        Schedule a computational task to a remote node.
        
        Args:
            task: Dictionary containing task information
            node: Target node identifier
            
        Returns:
            str: A unique task ID (UUID)
        """
        task_id = str(uuid.uuid4())
        
        # Store task information
        self._tasks[task_id] = {
            'task': task,
            'node': node,
            'status': 'queued',
            'created_at': time.time(),
            'updated_at': time.time()
        }
        
        # Log the scheduling action
        print(f"[SCHEDULE] Task {task_id} scheduled to node '{node}'")
        print(f"[SCHEDULE] Task details: {task}")
        
        return task_id
    
    def status(self, task_id: str) -> str:
        """
        Get the status of a scheduled task.
        
        Args:
            task_id: The unique identifier of the task
            
        Returns:
            str: Current status of the task ('queued', 'running', 'done', 'failed', 'unknown')
        """
        if task_id not in self._tasks:
            print(f"[STATUS] Task {task_id} not found")
            return "unknown"
        
        task_info = self._tasks[task_id]
        status = task_info['status']
        
        # Simulate status progression for demo purposes
        if status == 'queued':
            # Simulate task starting after some time
            if time.time() - task_info['created_at'] > 2.0:
                task_info['status'] = 'running'
                task_info['updated_at'] = time.time()
                status = 'running'
        elif status == 'running':
            # Simulate task completion after some time
            if time.time() - task_info['updated_at'] > 3.0:
                task_info['status'] = 'done'
                task_info['updated_at'] = time.time()
                status = 'done'
        
        print(f"[STATUS] Task {task_id} status: {status}")
        return status
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """
        Get detailed information about a task.
        
        Args:
            task_id: The unique identifier of the task
            
        Returns:
            Optional[Dict]: Task information or None if not found
        """
        return self._tasks.get(task_id)
    
    def list_tasks(self) -> Dict[str, Dict]:
        """
        Get all scheduled tasks.
        
        Returns:
            Dict[str, Dict]: Dictionary of all tasks with their IDs as keys
        """
        return self._tasks.copy()


# Global instance for convenience
_manager = ComputeManager()


def schedule(task: dict, node: str) -> str:
    """
    Schedule a computational task to a remote node.
    
    Args:
        task: Dictionary containing task information
        node: Target node identifier
        
    Returns:
        str: A unique task ID (UUID)
    """
    return _manager.schedule(task, node)


def status(task_id: str) -> str:
    """
    Get the status of a scheduled task.
    
    Args:
        task_id: The unique identifier of the task
        
    Returns:
        str: Current status of the task
    """
    return _manager.status(task_id)


def get_task_info(task_id: str) -> Optional[Dict]:
    """
    Get detailed information about a task.
    
    Args:
        task_id: The unique identifier of the task
        
    Returns:
        Optional[Dict]: Task information or None if not found
    """
    return _manager.get_task_info(task_id)


def list_tasks() -> Dict[str, Dict]:
    """
    Get all scheduled tasks.
    
    Returns:
        Dict[str, Dict]: Dictionary of all tasks with their IDs as keys
    """
    return _manager.list_tasks()