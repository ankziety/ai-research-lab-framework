# Tests

The `tests` directory includes unit tests and integration tests for the various components of the AI Research Lab framework. Each module has corresponding test files to ensure correctness and maintainability. We use pytest for Python modules.

## Current Test Status

### Overall Results
- **Total Tests**: 44 passed, 4 skipped (91.7% pass rate)
- **Real LLM Integration**: 3/4 integration tests passing with actual API calls
- **Performance**: 22.7s for 4 integration tests (reasonable for API calls)

### Test Coverage
- **Overall Coverage**: 14% (heavy presence of large physics modules not tested)
- **Core Modules**:
  - `agents/coding_specialist.py`: 84% ✅ (well tested)
  - `tools/mcp_tool_registry.py`: 78% ✅ (well tested)
  - `tools/tool_registry.py`: 57% ⚠️ (moderate)
  - `agents/base_agent.py`: 30% ⚠️ (needs improvement)
  - `agents/llm_client.py`: 37% ⚠️ (needs improvement)

## Running Tests

### Basic Test Suite
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest -q

# Expected: 44 passed, 4 skipped (91.7% pass rate)
```

### Real LLM Integration Tests
```bash
# Run real LLM integration tests (requires API keys in config/config.json)
pytest tests/test_coding_specialist_mcp_integration.py -vv

# Expected: 3 passed, 1 skipped (real API calls)
```

### Coverage Analysis
```bash
# Run with coverage analysis
pytest --cov=agents --cov=core --cov=tools --cov=ai_research_lab --cov-report=term-missing

# Core modules coverage:
# - agents/coding_specialist.py: 84%
# - tools/mcp_tool_registry.py: 78%
# - tools/tool_registry.py: 57%
# - agents/base_agent.py: 30%
```

## Test Categories

### Unit Tests
- **Agent Tests**: Individual agent functionality and behavior
- **Tool Tests**: Tool registry and MCP tool system validation
- **Framework Tests**: Core framework components and utilities

### Integration Tests
- **Real LLM Integration**: Tests using actual API calls for end-to-end validation
- **MCP Tool System**: Dynamic tool creation, discovery, and execution
- **Virtual Lab Integration**: Meeting system and agent coordination

### Skipped Tests
- **Domain Agent Integration**: 1 test skipped due to minor integration issue
- **Virtual Lab Enhanced**: 1 test skipped due to complex dependencies
- **Real LLM Error Recovery**: 2 tests skipped due to mock client issues

## Test Requirements

### API Configuration
Real LLM integration tests require API keys in `config/config.json`:
```json
{
    "api_keys": {
        "openai": "your-openai-api-key"
    },
    "framework": {
        "default_llm_provider": "openai",
        "default_model": "gpt-4o"
    }
}
```

### Environment Setup
```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure virtual environment is activated
source .venv/bin/activate
```

## Test Development Guidelines

### Adding New Tests
1. **Test Naming**: Use descriptive test names that explain the expected behavior
2. **Test Isolation**: Each test should be independent and not rely on other tests
3. **Mock Usage**: Use mocks for external dependencies, but prefer real API calls for integration tests
4. **Coverage**: Aim for >80% coverage on core modules

### Real LLM Tests
- **Purpose**: Validate end-to-end functionality with actual API calls
- **Cost Awareness**: Keep test complexity reasonable to manage API costs
- **Error Handling**: Test both success and failure scenarios
- **Performance**: Monitor test execution time and optimize if needed

### Test Maintenance
- **Regular Updates**: Update tests when functionality changes
- **Coverage Monitoring**: Track coverage trends and address gaps
- **Performance Monitoring**: Monitor test execution time and optimize
- **Documentation**: Keep test documentation current with implementation changes

## Known Issues

### Current Limitations
- **Physics Modules**: 0% coverage (not in current scope, Phase 2 target)
- **Base Agent**: 30% coverage (needs more integration tests)
- **LLM Client**: 37% coverage (needs more provider tests)

### Next Steps
1. **Improve Coverage**: Focus on `agents/base_agent.py` and `agents/llm_client.py`
2. **Fix Skipped Tests**: Resolve domain agent tool discovery issue
3. **Physics Integration**: Begin Phase 2 physics engine testing
4. **Performance Optimization**: Profile and optimize test execution time
