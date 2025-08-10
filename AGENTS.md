## AGENTS.md – Development Rules and Handoff Guide

This file defines the non-negotiable standards and the minimum context required for any new coding agent working in this repository. Read this before making changes.

### Core Standards
- **Tests must be 100% passing for every PR**: run `pytest` locally before submitting. Do not skip tests or disable checks.
- **Use the project virtual environment**: this repo assumes a `.venv` per workspace. Activate or call binaries via `.venv/bin/...`.
- **Use MCP tools for file operations and interactive steps** where applicable to preserve context and auditability.
- **No temporary or stub implementations**: production-quality edits only.
- **Cost awareness**: prefer cheaper model calls and keep token use reasonable. The `CostManager` integrates provider costs.

### Environment Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r web_ui/requirements.txt
pip install -e .
```

### Quick Validation
```bash
pytest -q
```
Expected: all tests pass. Current baseline: 32 passed, 1 skipped.

### Key Entry Points
- Import surface for examples and scripts: `from ai_research_lab import create_framework`
- Web UI launcher: `python launch.py --web`
- CLI help: `python -m data.cli --help`

### Coding Rules
- Python target: 3.9+; avoid 3.10-only typing (use `typing.Optional`, `typing.List`, `typing.Tuple`, etc.).
- Keep imports explicit and stable. Avoid creating cycles in aggregator modules.
- Follow repository code style: clear names, guard clauses, handle edge cases first, avoid deep nesting.
- Do not introduce new linter errors.

### Testing Rules
- Add tests for new functionality.
- Keep tests deterministic and self-contained.
- If you modify Virtual Lab integration modules under `core/virtual_lab_integration/`, run the full suite.

### Tools and MCP Usage
- Prefer MCP-enabled actions for file edits, searches, and prompts to ensure traceability.
- Use cost-conscious models or local inference when appropriate; see `data/cost_manager.py`.

### Handoff Notes
- Virtual Lab integration is import-stable; corrected for Python 3.9 typing.
- The package provides a stable alias via `ai_research_lab/__init__.py`.
- `.gitignore` no longer excludes tests; tests are tracked in VCS.

If any standard conflicts with your task requirements, raise it explicitly and propose a compliant alternative.