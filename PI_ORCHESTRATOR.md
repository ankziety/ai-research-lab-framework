# PI Orchestrator

The PI (Principal Investigator) Orchestrator is the main coordination layer for research automation in the AI Research Lab Framework.

## Features

- **Specialist Registration**: Register specialist agents by role/name with callable handlers
- **Task Decomposition**: Automatically decompose research requests into subtasks
- **Intelligent Routing**: Route subtasks to appropriate specialists based on content analysis
- **Result Aggregation**: Combine results from multiple specialists into cohesive outputs
- **Vector Memory Integration**: Store and retrieve context using vector similarity search
- **Provenance Logging**: Complete audit trail of all decisions and specialist calls

## API

### Core Methods

```python
from pi_orchestrator import PIOrchestrator
from vector_memory import VectorMemory

# Initialize
orchestrator = PIOrchestrator()
memory = VectorMemory()

# Setup
orchestrator.set_memory(memory)
orchestrator.register_specialist("literature_reviewer", my_lit_specialist)
orchestrator.register_specialist("data_analyst", my_data_specialist)

# Execute research task
result = orchestrator.run_research_task("Review ML literature and analyze data")
```

### Method Details

- `register_specialist(role_name: str, handler: Callable)` - Register a specialist agent
- `run_research_task(request: str) -> dict` - Execute a research task with automatic specialist coordination
- `set_memory(memory_instance)` - Set vector memory instance for context storage

## Task Decomposition Logic

The orchestrator automatically identifies required specialists based on keywords:

- **Literature Review**: Keywords like "literature", "papers", "research", "review"
- **Data Analysis**: Keywords like "analyze", "data", "statistics", "results"  
- **Manuscript Writing**: Keywords like "write", "draft", "manuscript", "paper"
- **General Research**: Fallback for requests not matching specific patterns

## Usage Examples

See `demo_pi_orchestrator.py` for comprehensive usage examples.

## Testing

Run the test suite:
```bash
python -m unittest test_pi_orchestrator.py -v
```

## Files

- `pi_orchestrator.py` - Main orchestrator implementation
- `vector_memory.py` - Vector memory stub for context storage  
- `test_pi_orchestrator.py` - Comprehensive unit tests
- `demo_pi_orchestrator.py` - Usage demonstration script