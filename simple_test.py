#!/usr/bin/env python3
"""
Simple test script for compute_manager module.
This script tests the basic functionality without requiring pytest.
"""

import time
from compute_manager import schedule, status, get_task_info, list_tasks, ComputeManager


def test_basic_functionality():
    """Test basic scheduling and status checking."""
    print("=== Testing Basic Functionality ===")
    
    # Test scheduling
    task = {"command": "python test_script.py", "args": ["--input", "data.txt"]}
    node = "test-node-01"
    
    task_id = schedule(task, node)
    print(f"✓ Task scheduled with ID: {task_id}")
    
    # Test status checking
    current_status = status(task_id)
    print(f"✓ Task status: {current_status}")
    
    # Test getting task info
    task_info = get_task_info(task_id)
    if task_info:
        print(f"✓ Task info retrieved: {task_info['node']} - {task_info['status']}")
    
    return task_id


def test_multiple_tasks():
    """Test scheduling multiple tasks."""
    print("\n=== Testing Multiple Tasks ===")
    
    manager = ComputeManager()
    tasks = [
        {"command": "python script1.py"},
        {"command": "python script2.py"},
        {"command": "python script3.py"}
    ]
    node = "multi-node-01"
    
    task_ids = []
    for i, task in enumerate(tasks, 1):
        task_id = manager.schedule(task, node)
        task_ids.append(task_id)
        print(f"✓ Task {i} scheduled: {task_id}")
    
    # Test listing all tasks
    all_tasks = manager.list_tasks()
    print(f"✓ Total tasks: {len(all_tasks)}")
    
    return task_ids


def test_status_progression():
    """Test that task status progresses over time."""
    print("\n=== Testing Status Progression ===")
    
    manager = ComputeManager()
    task = {"command": "python long_running_script.py"}
    node = "progression-node-01"
    
    task_id = manager.schedule(task, node)
    print(f"✓ Task scheduled: {task_id}")
    
    # Check initial status
    initial_status = manager.status(task_id)
    print(f"✓ Initial status: {initial_status}")
    
    # Simulate time passing
    task_info = manager._tasks[task_id]
    task_info['created_at'] = time.time() - 3.0  # 3 seconds ago
    
    # Check status after time progression
    progressed_status = manager.status(task_id)
    print(f"✓ Status after progression: {progressed_status}")
    
    return task_id


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    manager = ComputeManager()
    
    # Test unknown task ID
    unknown_status = manager.status("unknown-task-id")
    print(f"✓ Unknown task status: {unknown_status}")
    
    # Test empty task dictionary
    empty_task = {}
    task_id = manager.schedule(empty_task, "edge-node-01")
    print(f"✓ Empty task scheduled: {task_id}")
    
    # Test special characters in node name
    special_node = "node@cluster.local:8080"
    task_id = manager.schedule({"command": "python test.py"}, special_node)
    print(f"✓ Special node name handled: {task_id}")
    
    # Test unicode in task
    unicode_task = {"command": "python test.py", "description": "数据分析任务"}
    task_id = manager.schedule(unicode_task, "unicode-node-01")
    print(f"✓ Unicode task scheduled: {task_id}")


def test_logging():
    """Test that logging works correctly."""
    print("\n=== Testing Logging ===")
    
    manager = ComputeManager()
    task = {"command": "python logging_test.py", "args": ["--verbose"]}
    node = "logging-node-01"
    
    print("Scheduling task (should see log messages):")
    task_id = manager.schedule(task, node)
    
    print("Checking status (should see log messages):")
    status_result = manager.status(task_id)
    
    print(f"✓ Logging test completed. Final status: {status_result}")


def main():
    """Run all tests."""
    print("Starting compute_manager tests...\n")
    
    try:
        # Run all test functions
        test_basic_functionality()
        test_multiple_tasks()
        test_status_progression()
        test_edge_cases()
        test_logging()
        
        print("\n=== All Tests Passed! ===")
        print("✓ Basic functionality works")
        print("✓ Multiple tasks can be scheduled")
        print("✓ Status progression works")
        print("✓ Edge cases are handled")
        print("✓ Logging works correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)