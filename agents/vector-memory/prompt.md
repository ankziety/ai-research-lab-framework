# Builder Agent Prompt for Vector-Memory Component

You are a builder agent designing the Vector Memory component for the AI Research Lab Framework. This component will provide persistent vector storage and search capabilities using FAISS for similarity search and SQLite for metadata storage. Use Python.

## Goals
- Implement a modular vector memory store with the ability to add vectors with associated metadata, and query for similar vectors.
- Use the `faiss` open-source library to index 1536-dimensional float32 vectors (embedding size; later will vary) and perform nearest neighbor search.
- Use `sqlite3` to persist metadata associated with each vector (e.g., text, id, timestamp). The SQLite database should store at least: id (primary key), vector_id (faiss index row), text, metadata JSON, timestamp.
- Provide methods: `add(text: str, metadata: dict) -> id`, `query(query_text: str, top_k: int) -> List[Tuple[id, score, metadata]]`. For embedding generation, stub out `get_embedding(text: str) -> np.ndarray` with random vector; later replaced.
- Provide ability to persist and reload the FAISS index and SQLite database from disk path specified via config.
- Write unit tests using pytest to verify:
  - Vectors can be added and retrieved;
  - Query returns nearest neighbours;
  - Persistence reload restores index and metadata.

## Constraints
- Only use pure Python standard library except for `faiss`, `numpy`, `pytest`. If `faiss` is not installed, include instructions for installation in README.
- Put the implementation in `memory/vector_memory.py` with an object-oriented design.
- Put tests in `tests/test_vector_memory.py`.
- Document public methods with docstrings.
- Do not include any secrets or API keys. Use stubbed embeddings.

## Deliverables
- Code and tests passing locally.
- README in `memory` explaining usage and limitations.

After implementing and passing all tests, open a pull request from `feature/vector-memory` into the `dev` branch. Include notes on design decisions and limitations.
