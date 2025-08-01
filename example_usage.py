#!/usr/bin/env python3
"""
Example usage of the compute_manager module.

This script demonstrates how to schedule tasks and check their status.
"""

import time
from compute_manager import schedule, status, get_task_info, list_tasks


def main():
    """Demonstrate the compute manager functionality."""
    print("=== Distributed Compute Manager Demo ===\n")
    
    # Example 1: Schedule a simple task
    print("1. Scheduling a simple task...")
    task1 = {
        "command": "python analysis.py",
        "args": ["--input", "data.csv", "--output", "results.json"],
        "priority": "high"
    }
    task_id1 = schedule(task1, "compute-node-01")
    print(f"   Task scheduled with ID: {task_id1}")
    
    # Example 2: Schedule a GPU task
    print("\n2. Scheduling a GPU task...")
    task2 = {
        "command": "python train_model.py",
        "args": ["--model", "resnet50", "--epochs", "100"],
        "resources": {"gpu": 2, "memory": "16GB"},
        "priority": "medium"
    }
    task_id2 = schedule(task2, "gpu-node-01")
    print(f"   Task scheduled with ID: {task_id2}")
    
    # Example 3: Schedule a batch processing task
    print("\n3. Scheduling a batch processing task...")
    task3 = {
        "command": "python batch_process.py",
        "args": ["--input-dir", "/data/raw", "--output-dir", "/data/processed"],
        "resources": {"cpu": 8, "memory": "32GB"},
        "timeout": 7200
    }
    task_id3 = schedule(task3, "batch-node-01")
    print(f"   Task scheduled with ID: {task_id3}")
    
    # Example 4: Check status of all tasks
    print("\n4. Checking status of all tasks...")
    for i, task_id in enumerate([task_id1, task_id2, task_id3], 1):
        current_status = status(task_id)
        print(f"   Task {i} ({task_id[:8]}...): {current_status}")
    
    # Example 5: Get detailed information about a task
    print("\n5. Getting detailed information about task 1...")
    task_info = get_task_info(task_id1)
    if task_info:
        print(f"   Node: {task_info['node']}")
        print(f"   Status: {task_info['status']}")
        print(f"   Created: {task_info['created_at']}")
        print(f"   Command: {task_info['task']['command']}")
    
    # Example 6: List all tasks
    print("\n6. Listing all scheduled tasks...")
    all_tasks = list_tasks()
    print(f"   Total tasks: {len(all_tasks)}")
    for task_id, info in all_tasks.items():
        print(f"   - {task_id[:8]}... -> {info['node']} ({info['status']})")
    
    # Example 7: Check status of unknown task
    print("\n7. Checking status of unknown task...")
    unknown_status = status("unknown-task-id")
    print(f"   Unknown task status: {unknown_status}")
    
    # Example 8: Simulate status progression
    print("\n8. Simulating status progression...")
    print("   (Waiting for tasks to progress through statuses...)")
    
    for i in range(3):
        print(f"\n   Check {i+1}:")
        for j, task_id in enumerate([task_id1, task_id2, task_id3], 1):
            current_status = status(task_id)
            print(f"   Task {j}: {current_status}")
        time.sleep(1)  # Wait a bit to see status changes
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()