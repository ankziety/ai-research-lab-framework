# Distributed Compute Manager

A Python module for scheduling computational tasks to remote nodes in a research compute cluster. This is an MVP implementation that logs actions and returns fake responses for demonstration purposes.

## Features

- **Task Scheduling**: Schedule computational tasks to remote nodes
- **Status Tracking**: Monitor task status (queued, running, done)
- **Logging**: Comprehensive logging of all actions
- **Fake Task IDs**: Generate unique UUIDs for task identification
- **Status Progression**: Simulate realistic task lifecycle

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from compute_manager import schedule, status

# Schedule a task
task = {
    "command": "python analysis.py",
    "args": ["--input", "data.csv", "--output", "results.json"],
    "priority": "high"
}
task_id = schedule(task, "compute-node-01")

# Check task status
current_status = status(task_id)
print(f"Task status: {current_status}")
```

### Advanced Usage

```python
from compute_manager import schedule, status, get_task_info, list_tasks

# Schedule multiple tasks
task1 = {"command": "python script1.py"}
task2 = {"command": "python script2.py"}

task_id1 = schedule(task1, "node-01")
task_id2 = schedule(task2, "node-02")

# Get detailed task information
task_info = get_task_info(task_id1)
print(f"Task on node: {task_info['node']}")
print(f"Task status: {task_info['status']}")

# List all tasks
all_tasks = list_tasks()
for task_id, info in all_tasks.items():
    print(f"{task_id}: {info['node']} - {info['status']}")
```

## API Reference

### Functions

#### `schedule(task: dict, node: str) -> str`
Schedule a computational task to a remote node.

**Parameters:**
- `task`: Dictionary containing task information
- `node`: Target node identifier

**Returns:**
- `str`: A unique task ID (UUID)

#### `status(task_id: str) -> str`
Get the status of a scheduled task.

**Parameters:**
- `task_id`: The unique identifier of the task

**Returns:**
- `str`: Current status ("queued", "running", "done", "unknown")

#### `get_task_info(task_id: str) -> Optional[Dict]`
Get detailed information about a task.

**Parameters:**
- `task_id`: The unique identifier of the task

**Returns:**
- `Optional[Dict]`: Task information or None if not found

#### `list_tasks() -> Dict[str, Dict]`
Get all scheduled tasks.

**Returns:**
- `Dict[str, Dict]`: Dictionary of all tasks with their IDs as keys

### Class

#### `ComputeManager`
Main class for managing distributed computational tasks.

**Methods:**
- `schedule(task, node)`: Schedule a task
- `status(task_id)`: Get task status
- `get_task_info(task_id)`: Get detailed task info
- `list_tasks()`: List all tasks

## Testing

Run the test suite:

```bash
pytest test_compute_manager.py -v
```

Run with coverage:

```bash
pytest test_compute_manager.py --cov=compute_manager --cov-report=html
```

## Example

Run the example script to see the module in action:

```bash
python example_usage.py
```

## Logging

The module provides comprehensive logging:

- `[SCHEDULE]` messages when tasks are scheduled
- `[STATUS]` messages when task status is checked
- All actions are logged to stdout

## Status Lifecycle

Tasks progress through the following statuses:

1. **queued**: Task is scheduled and waiting to start
2. **running**: Task is currently executing
3. **done**: Task has completed successfully
4. **unknown**: Task ID not found

Status progression is simulated based on time elapsed since task creation.

## License

This project is licensed under the MIT License.
