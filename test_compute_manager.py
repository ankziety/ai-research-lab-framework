"""
Test suite for the compute_manager module.

Tests scheduling tasks, checking status, and handling edge cases.
"""

import pytest
import time
from unittest.mock import patch
from io import StringIO
import sys

from compute_manager import (
    schedule, 
    status, 
    get_task_info, 
    list_tasks, 
    ComputeManager
)


class TestComputeManager:
    """Test cases for the ComputeManager class."""
    
    def test_schedule_task(self):
        """Test scheduling a task returns a valid task ID."""
        manager = ComputeManager()
        task = {"command": "python script.py", "args": ["--input", "data.csv"]}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        assert task_id in manager._tasks
        
        # Verify task information is stored correctly
        task_info = manager._tasks[task_id]
        assert task_info['task'] == task
        assert task_info['node'] == node
        assert task_info['status'] == 'queued'
        assert 'created_at' in task_info
        assert 'updated_at' in task_info
    
    def test_schedule_multiple_tasks(self):
        """Test scheduling multiple tasks generates unique IDs."""
        manager = ComputeManager()
        task1 = {"command": "python script1.py"}
        task2 = {"command": "python script2.py"}
        node = "compute-node-01"
        
        task_id1 = manager.schedule(task1, node)
        task_id2 = manager.schedule(task2, node)
        
        assert task_id1 != task_id2
        assert len(manager._tasks) == 2
    
    def test_status_queued_task(self):
        """Test status of a newly queued task."""
        manager = ComputeManager()
        task = {"command": "python script.py"}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        status_result = manager.status(task_id)
        
        assert status_result == "queued"
    
    def test_status_unknown_task(self):
        """Test status of a non-existent task."""
        manager = ComputeManager()
        
        status_result = manager.status("non-existent-task-id")
        
        assert status_result == "unknown"
    
    def test_status_progression(self):
        """Test that task status progresses over time."""
        manager = ComputeManager()
        task = {"command": "python script.py"}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        # Initially should be queued
        assert manager.status(task_id) == "queued"
        
        # Simulate time passing to trigger status changes
        task_info = manager._tasks[task_id]
        task_info['created_at'] = time.time() - 3.0  # Simulate 3 seconds ago
        
        # Should now be running
        assert manager.status(task_id) == "running"
        
        # Simulate more time passing
        task_info['updated_at'] = time.time() - 4.0  # Simulate 4 seconds ago
        
        # Should now be done
        assert manager.status(task_id) == "done"
    
    def test_get_task_info(self):
        """Test getting detailed task information."""
        manager = ComputeManager()
        task = {"command": "python script.py", "args": ["--verbose"]}
        node = "compute-node-02"
        
        task_id = manager.schedule(task, node)
        task_info = manager.get_task_info(task_id)
        
        assert task_info is not None
        assert task_info['task'] == task
        assert task_info['node'] == node
        assert task_info['status'] == 'queued'
    
    def test_get_task_info_nonexistent(self):
        """Test getting task info for non-existent task."""
        manager = ComputeManager()
        
        task_info = manager.get_task_info("non-existent-task-id")
        
        assert task_info is None
    
    def test_list_tasks(self):
        """Test listing all tasks."""
        manager = ComputeManager()
        task1 = {"command": "python script1.py"}
        task2 = {"command": "python script2.py"}
        node = "compute-node-01"
        
        task_id1 = manager.schedule(task1, node)
        task_id2 = manager.schedule(task2, node)
        
        all_tasks = manager.list_tasks()
        
        assert len(all_tasks) == 2
        assert task_id1 in all_tasks
        assert task_id2 in all_tasks
        assert all_tasks[task_id1]['task'] == task1
        assert all_tasks[task_id2]['task'] == task2


class TestModuleFunctions:
    """Test cases for the module-level functions."""
    
    def test_schedule_function(self):
        """Test the module-level schedule function."""
        task = {"command": "python analysis.py", "input": "data.txt"}
        node = "gpu-node-01"
        
        task_id = schedule(task, node)
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
    
    def test_status_function(self):
        """Test the module-level status function."""
        task = {"command": "python script.py"}
        node = "compute-node-01"
        
        task_id = schedule(task, node)
        status_result = status(task_id)
        
        assert status_result == "queued"
    
    def test_status_function_unknown(self):
        """Test the module-level status function with unknown task."""
        status_result = status("unknown-task-id")
        
        assert status_result == "unknown"
    
    def test_get_task_info_function(self):
        """Test the module-level get_task_info function."""
        task = {"command": "python script.py"}
        node = "compute-node-01"
        
        task_id = schedule(task, node)
        task_info = get_task_info(task_id)
        
        assert task_info is not None
        assert task_info['task'] == task
        assert task_info['node'] == node
    
    def test_list_tasks_function(self):
        """Test the module-level list_tasks function."""
        # Clear any existing tasks by creating a new manager
        from compute_manager import _manager
        _manager._tasks.clear()
        
        task1 = {"command": "python script1.py"}
        task2 = {"command": "python script2.py"}
        node = "compute-node-01"
        
        task_id1 = schedule(task1, node)
        task_id2 = schedule(task2, node)
        
        all_tasks = list_tasks()
        
        assert len(all_tasks) == 2
        assert task_id1 in all_tasks
        assert task_id2 in all_tasks


class TestLogging:
    """Test cases for logging functionality."""
    
    def test_schedule_logging(self):
        """Test that scheduling logs the correct information."""
        manager = ComputeManager()
        task = {"command": "python script.py", "args": ["--input", "data.csv"]}
        node = "compute-node-01"
        
        # Capture stdout to check logging
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            task_id = manager.schedule(task, node)
            
            output = captured_output.getvalue()
            assert f"[SCHEDULE] Task {task_id} scheduled to node '{node}'" in output
            assert f"[SCHEDULE] Task details: {task}" in output
        finally:
            sys.stdout = sys.__stdout__
    
    def test_status_logging(self):
        """Test that status checking logs the correct information."""
        manager = ComputeManager()
        task = {"command": "python script.py"}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        # Capture stdout to check logging
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            status_result = manager.status(task_id)
            
            output = captured_output.getvalue()
            assert f"[STATUS] Task {task_id} status: {status_result}" in output
        finally:
            sys.stdout = sys.__stdout__
    
    def test_status_logging_unknown(self):
        """Test that status checking logs unknown task correctly."""
        manager = ComputeManager()
        
        # Capture stdout to check logging
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            status_result = manager.status("unknown-task-id")
            
            output = captured_output.getvalue()
            assert "[STATUS] Task unknown-task-id not found" in output
            assert status_result == "unknown"
        finally:
            sys.stdout = sys.__stdout__


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_empty_task_dict(self):
        """Test scheduling with an empty task dictionary."""
        manager = ComputeManager()
        task = {}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        assert isinstance(task_id, str)
        assert task_id in manager._tasks
        assert manager._tasks[task_id]['task'] == {}
    
    def test_large_task_dict(self):
        """Test scheduling with a large task dictionary."""
        manager = ComputeManager()
        task = {
            "command": "python script.py",
            "args": ["--input", "data.csv", "--output", "results.json"],
            "env": {"PYTHONPATH": "/usr/local/lib/python3.9"},
            "resources": {"cpu": 4, "memory": "8GB", "gpu": 1},
            "timeout": 3600,
            "priority": "high"
        }
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        assert isinstance(task_id, str)
        assert manager._tasks[task_id]['task'] == task
    
    def test_special_characters_in_node_name(self):
        """Test scheduling with special characters in node name."""
        manager = ComputeManager()
        task = {"command": "python script.py"}
        node = "compute-node-01@cluster.local"
        
        task_id = manager.schedule(task, node)
        
        assert isinstance(task_id, str)
        assert manager._tasks[task_id]['node'] == node
    
    def test_unicode_in_task(self):
        """Test scheduling with unicode characters in task."""
        manager = ComputeManager()
        task = {"command": "python script.py", "description": "数据分析任务"}
        node = "compute-node-01"
        
        task_id = manager.schedule(task, node)
        
        assert isinstance(task_id, str)
        assert manager._tasks[task_id]['task'] == task
    
    def test_none_values(self):
        """Test handling of None values."""
        manager = ComputeManager()
        
        # Test with None task (should raise TypeError)
        with pytest.raises(TypeError):
            manager.schedule(None, "compute-node-01")
        
        # Test with None node (should raise TypeError)
        with pytest.raises(TypeError):
            manager.schedule({"command": "python script.py"}, None)
    
    def test_invalid_task_id_types(self):
        """Test status checking with invalid task ID types."""
        manager = ComputeManager()
        
        # Test with None task_id
        assert manager.status(None) == "unknown"
        
        # Test with non-string task_id
        assert manager.status(123) == "unknown"
        assert manager.status(["task-id"]) == "unknown"


if __name__ == "__main__":
    pytest.main([__file__])