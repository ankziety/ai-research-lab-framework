# AI Research Lab Framework

A comprehensive AI-powered research framework that coordinates teams of AI experts to collaborate on research problems across any domain. The framework implements both traditional multi-agent collaboration and the Virtual Lab methodology for structured meeting-based research.

## Current Status (Latest)

### Test Coverage Initiative ✅ **COMPLETED**
- **Strategic Improvement**: BaseAgent 30%→72% (+42%), LLM Client 38%→78% (+40%)
- **Test Results**: 135 passed, 3 skipped (97.8% pass rate) with comprehensive coverage
- **Foundation Stability**: Core modules now have >70% coverage with robust test suites
- **Quality Standards**: Established comprehensive testing patterns and error handling

### Real LLM Integration ✅ **PRODUCTION READY**
- **Real API Integration**: Tests use actual OpenAI GPT-4o API calls for end-to-end validation
- **Core Functionality**: Coding Specialist and MCP Tool Registry fully operational with real LLM
- **Production Coverage**: 17% overall, with core modules well-tested (CodingSpecialist: 84%, MCPToolRegistry: 78%)

### Recent Updates

#### Test Coverage Initiative (Latest)
- **Comprehensive Testing**: Added 70 new tests covering BaseAgent and LLM Client functionality
- **Error Handling**: Comprehensive error handling with fail-fast behavior
- **Edge Cases**: Boundary condition testing for robust error handling
- **Integration Ready**: Foundation established for comprehensive integration testing
- **Documentation**: Created detailed handoff documentation for continued development

#### Real LLM Test Validation
- **Real API Calls**: Integration tests use actual OpenAI GPT-4o API
- **End-to-End Workflows**: Tool creation → MCP registration → Discovery → Execution validated
- **Error Recovery**: LLM retry mechanisms tested with real failures
- **Performance**: 22.7s for 4 integration tests (reasonable for API calls)

#### Serialization Fixes
- **Fixed JSON serialization errors**: Enhanced the `make_json_serializable` function in `multi_agent_framework.py` to properly handle `MeetingRecord` and `MeetingAgenda` objects
- **Improved error handling**: Added specific handling for Enum objects and complex data structures
- **Comprehensive testing**: All serialization tests now pass successfully

#### Literature Search Improvements
- **Free API Integration**: Updated literature retriever to use more free APIs by default:
  - PubMed (no API key required)
  - ArXiv (no API key required) 
  - CrossRef (no API key required)
  - Semantic Scholar (free tier available)
  - Base-search.net (free academic search)
- **Enhanced search capabilities**: Better ranking algorithms and duplicate removal
- **Fail-fast behavior**: System requires real APIs and dependencies, no mock fallbacks

## Key Features

### Multi-Agent Research System
- **Principal Investigator (PI) Agent**: Coordinates research and manages the team
- **Agent Marketplace**: Dynamic hiring of domain experts based on research needs
- **Scientific Critic Agent**: Quality control and validation of research outputs
- **Domain Expert Agents**: Specialized agents for different research domains
- **Coding Specialist Agent**: Implements custom tools with real LLM integration

### Virtual Lab Methodology
- **Structured Meetings**: Research conducted through organized meetings between AI agents
- **Phase-based Research**: Systematic progression through research phases
- **Meeting Records**: Comprehensive tracking of all research interactions
- **Quality Assessment**: Continuous evaluation of research quality and progress

### MCP Tool System
- **Dynamic Tool Creation**: AI agents can create custom tools using real LLM
- **MCP Integration**: Model Context Protocol compatible tool registry
- **Tool Discovery**: Domain agents can discover and execute custom tools
- **Real-time Implementation**: Tools generated and validated with actual API calls

### Memory and Knowledge Management
- **Vector Database**: Semantic storage and retrieval of research context
- **Knowledge Repository**: Validated findings and research insights
- **Context Management**: Session-based memory for ongoing research

### Literature Integration
- **Multi-source Search**: PubMed, ArXiv, CrossRef, Semantic Scholar, and more
- **Citation Analysis**: Automatic extraction and analysis of citations
- **Research Synthesis**: AI-powered literature review and synthesis

## Quick Start

### Installation

```bash
git clone <repository-url>
cd ai-research-lab-framework

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Web Interface (Recommended)

The framework provides a modern Gradio-based web interface that's easier to set up and use than the legacy Flask interface.

**Simple Gradio Interface (Recommended for most users):**
```bash
cd web_ui
python simple_gradio_app.py
```

**Full-featured Gradio Interface (Advanced users):**
```bash
cd web_ui
python gradio_app.py
```

> **Note:** The Flask web interface is deprecated and will be removed in a future version. The Gradio interface provides a more modern, easier-to-use experience with better real-time updates and simplified setup.

### System Requirements

**Important**: This framework requires real APIs and dependencies. Mock fallbacks have been removed to ensure production-quality behavior.

**Required Dependencies:**
- `sentence-transformers` - For vector embeddings
- `faiss-cpu` - For vector similarity search
- `openai` - For OpenAI API integration
- `anthropic` - For Anthropic API integration (optional)
- `google-generative-ai` - For Google Gemini API integration (optional)

**Required API Keys:**
- At least one LLM provider API key (OpenAI, Anthropic, or Google Gemini)
- Literature search API keys (optional but recommended for full functionality)

### API Configuration

Create a `config/config.json` file with your API keys:

```json
{
    "api_keys": {
        "openai": "your-openai-api-key",
        "anthropic": "your-anthropic-api-key",
        "gemini": "your-gemini-api-key",
        "huggingface": "your-huggingface-api-key"
    },
    "framework": {
        "default_llm_provider": "openai",
        "default_model": "gpt-4o"
    }
}
```

**Required for Full Functionality**: At minimum, provide an OpenAI API key. The system will fail fast if required APIs or dependencies are not available.

### Basic Usage

```python
from ai_research_lab import create_framework

# Initialize the framework
framework = create_framework({
    'openai_api_key': 'your-api-key',  # Required for real LLM integration
    'max_agents_per_research': 5,
    'budget_limit': 100.0
})

# Conduct research using Virtual Lab methodology
results = framework.conduct_virtual_lab_research(
    research_question="How can we improve machine learning model interpretability?",
    constraints={'budget': 50.0, 'timeline_weeks': 2},
    context={'domain': 'computer_science', 'priority': 'high'}
)

# Or use traditional multi-agent research
results = framework.conduct_research(
    research_question="What are the latest developments in natural language processing?",
    constraints={'team_size_max': 3},
    context={'domain': 'nlp'}
)

print(f"Research completed: {results['status']}")
print(f"Key findings: {results['key_findings']}")
```

### MCP Tool Creation (Real LLM)

```python
from agents.coding_specialist import CodingSpecialist
from tools.mcp_tool_registry import MCPToolRegistry

# Initialize coding specialist with real LLM
coding_specialist = CodingSpecialist(
    agent_id="coding_specialist",
    model_config={'openai_api_key': 'your-key'},
    tools_directory="shared_tools"
)

# Create a custom tool specification
tool_spec = {
    'name': 'data_analyzer',
    'description': 'Analyze numerical data for statistical insights',
    'domain': 'data_science',
    'parameters': {
        'data': {'type': 'array', 'description': 'Input data array', 'required': True},
        'analysis_type': {'type': 'string', 'description': 'Type of analysis', 'required': False}
    },
    'capabilities': ['statistical_analysis', 'data_processing'],
    'requirements': {}
}

# Implement the tool using real LLM
result = coding_specialist.implement_tool(tool_spec)
print(f"Tool created: {result['success']}")

# Discover and use the tool
mcp_registry = MCPToolRegistry()
discovered_tools = mcp_registry.discover_mcp_tools()
print(f"Discovered {len(discovered_tools)} MCP tools")
```

### Literature Search

```python
from literature_retriever import LiteratureRetriever

# Initialize retriever (works with free APIs)
retriever = LiteratureRetriever()

# Search across multiple free sources
papers = retriever.search(
    query="machine learning interpretability",
    max_results=10,
    sources=['pubmed', 'arxiv', 'crossref', 'semantic_scholar']
)

for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Source: {paper['source']}")
    print(f"URL: {paper['url']}")
    print("---")
```

## Configuration

The framework supports extensive configuration options:

```python
config = {
    # API Configuration (Required for real LLM)
    'openai_api_key': 'your-key',
    'anthropic_api_key': 'your-key',
    'default_llm_provider': 'openai',
    'default_model': 'gpt-4o',
    
    # Agent Configuration
    'max_agents_per_research': 8,
    'agent_timeout': 1800,
    'agent_memory_limit': 1000,
    
    # Memory Configuration
    'vector_db_path': 'memory/vector_memory.db',
    'embedding_model': 'all-MiniLM-L6-v2',
    'max_context_length': 4000,
    
    # Research Configuration
    'budget_limit': 100.0,
    'max_literature_results': 10,
    
    # Literature API Keys (optional)
    'semantic_scholar_api_key': 'your-key',
    'openalex_email': 'your-email',
    'core_api_key': 'your-key'
}
```

## Testing

### Run All Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run comprehensive test suite
pytest -q

# Expected: 44 passed, 4 skipped (91.7% pass rate)
```

### Real LLM Integration Tests

```bash
# Run real LLM integration tests (requires API keys)
pytest tests/test_coding_specialist_mcp_integration.py -vv

# Expected: 3 passed, 1 skipped (real API calls)
```

### Testing

The project has comprehensive test coverage for core modules with established testing standards.

```bash
# Run all tests
pytest

# Run with coverage analysis
pytest --cov=agents --cov=core --cov=tools --cov=ai_research_lab --cov-report=term-missing

# Run specific test files
pytest tests/test_base_agent.py -v
pytest tests/test_llm_client.py -v

# Core modules coverage (updated):
# - agents/base_agent.py: 72% (comprehensive test suite)
# - agents/llm_client.py: 78% (comprehensive test suite)
# - agents/coding_specialist.py: 84%
# - tools/mcp_tool_registry.py: 78%
# - tools/tool_registry.py: 57%
```

**Testing Standards**: Comprehensive coverage with error handling, edge cases, and real-world scenarios. See `HANDOFF.md` for detailed testing patterns and next steps.

## Architecture

### Core Components

1. **MultiAgentResearchFramework**: Main orchestrator
2. **VirtualLabMeetingSystem**: Implements Virtual Lab methodology
3. **AgentMarketplace**: Manages domain expert agents
4. **CodingSpecialist**: Creates custom tools with real LLM
5. **MCPToolRegistry**: Manages MCP-compatible tools
6. **LiteratureRetriever**: Multi-source literature search
7. **Memory Systems**: Vector database and knowledge repository

### Web Interface Options

- **Gradio Interface (Recommended)**: Modern, easy-to-setup web interface with real-time updates
  - Simple version: Streamlined chat interface with research capabilities
  - Full version: Comprehensive dashboard with agent management and advanced features
- **Flask Interface (Deprecated)**: Legacy web interface requiring complex setup and configuration

### Research Phases (Virtual Lab)

1. **Team Selection**: Identify required expertise
2. **Literature Review**: Comprehensive literature search and analysis
3. **Project Specification**: Define research scope and methodology
4. **Tools Selection**: Choose appropriate research tools
5. **Tools Implementation**: Set up and configure tools (Real LLM)
6. **Workflow Design**: Plan research execution
7. **Execution**: Conduct the research
8. **Synthesis**: Compile and validate results

## Production Readiness

### ✅ **Production Ready Components**
- **Real LLM Integration**: OpenAI GPT-4o API integration working
- **MCP Tool System**: Dynamic tool creation and discovery operational
- **Core Agent Framework**: Principal Investigator, Coding Specialist, Domain Experts
- **Virtual Lab Methodology**: Structured meeting system implemented
- **Literature Search**: Multi-source search with free APIs

### ⚠️ **Areas for Improvement**
- **Test Coverage**: Overall 17% (core modules well-tested, physics modules untested)
- **Integration Testing**: Need comprehensive integration tests between modules
- **Coverage Enforcement**: Automated coverage thresholds not yet implemented

### 🔧 **Known Issues**
- 1 skipped test: Domain agent tool discovery (minor integration issue)
- Physics simulation modules: 0% coverage (not in current scope)

## Free APIs Available

The literature retriever works with these free APIs:

- **PubMed**: No API key required
- **ArXiv**: No API key required  
- **CrossRef**: No API key required
- **Semantic Scholar**: Free tier available
- **Base-search.net**: Free academic search
- **OpenAlex**: Free with email registration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest -q`
6. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd ai-research-lab-framework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run tests to validate setup
pytest -q
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
