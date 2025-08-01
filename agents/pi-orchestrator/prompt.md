# Builder Agent Prompt for PI Orchestrator

You are a builder agent tasked with implementing the PI (Principal Investigator) Orchestrator component of the AI Research Lab Framework. This module will coordinate specialist agents, manage tasks and memory interactions, and serve as the high-level executive of experiments. Use Python.

## Goals
- Design a PI Orchestrator class that can:
  * Maintain a registry of specialist agent roles and their corresponding instructions. For now, stub this registry with a few example roles (e.g., LiteratureRetriever, Critic).
  * Spawn specialist agents on demand and assign them tasks; for now, stub the spawning logic.
  * Provide a method `run_experiment(topic: str, objectives: List[str])` that orchestrates a high-level research experiment by invoking tasks on specialist agents in sequence (e.g., retrieving literature, summarizing, drafting manuscript).
  * Maintain internal state about ongoing experiments, including experiment IDs, objectives, and status. Persist this state to SQLite or a JSON file.
  * Interact with the Vector Memory module (via an interface) to store and retrieve experiment context and agent outputs. For now, stub integration but define method signatures (e.g., `store_context(text, metadata)`, `retrieve_context(query, top_k)`).
  * Expose simple logging for each action taken by the orchestrator.

## Constraints
- Use Python 3.x. Keep dependencies minimal; rely on standard library modules such as `sqlite3`, `uuid`, `logging`, and `typing`. Do not import any external agent frameworks.
- Put the implementation in `agents/pi_orchestrator.py` with class definitions and helper functions.
- Put unit tests in `tests/test_pi_orchestrator.py`. Tests should verify that:
  - The orchestrator can register and spawn stub specialist agents.
  - `run_experiment` creates an experiment entry with proper status transitions.
  - Persistent state is saved and loaded across sessions.
- Document all public methods with docstrings.
- Where functionality cannot be implemented without other components (e.g., actual agent spawning), stub functions with clear TODO comments.

## Deliverables
- Source file `agents/pi_orchestrator.py`.
- `tests/test_pi_orchestrator.py` with pytest-based tests.
- Update any necessary `README` files to explain usage of the orchestrator.

After completing code and tests, commit to `feature/pi-orchestrator` branch and open a pull request into `dev` once tests pass. Provide notes on design decisions and assumptions.
