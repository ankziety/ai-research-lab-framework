# AI Research Lab Framework

A comprehensive AI-powered research framework that coordinates teams of AI experts to collaborate on research problems across any domain. The framework implements both traditional multi-agent collaboration and the Virtual Lab methodology for structured meeting-based research.

## Recent Updates

### Serialization Fixes (Latest)
- **Fixed JSON serialization errors**: Enhanced the `make_json_serializable` function in `multi_agent_framework.py` to properly handle `MeetingRecord` and `MeetingAgenda` objects
- **Improved error handling**: Added specific handling for Enum objects and complex data structures
- **Comprehensive testing**: All serialization tests now pass successfully

### Literature Search Improvements
- **Free API Integration**: Updated literature retriever to use more free APIs by default:
  - PubMed (no API key required)
  - ArXiv (no API key required) 
  - CrossRef (no API key required)
  - Semantic Scholar (free tier available)
  - Base-search.net (free academic search)
- **Enhanced search capabilities**: Better ranking algorithms and duplicate removal
- **Fallback mechanisms**: Robust mock data generation when APIs are unavailable

## Key Features

### Multi-Agent Research System
- **Principal Investigator (PI) Agent**: Coordinates research and manages the team
- **Agent Marketplace**: Dynamic hiring of domain experts based on research needs
- **Scientific Critic Agent**: Quality control and validation of research outputs
- **Domain Expert Agents**: Specialized agents for different research domains

### Virtual Lab Methodology
- **Structured Meetings**: Research conducted through organized meetings between AI agents
- **Phase-based Research**: Systematic progression through research phases
- **Meeting Records**: Comprehensive tracking of all research interactions
- **Quality Assessment**: Continuous evaluation of research quality and progress

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
pip install -r requirements.txt
```

### Basic Usage

```python
from ai_research_lab import create_framework

# Initialize the framework
framework = create_framework({
    'openai_api_key': 'your-api-key',  # Optional for basic functionality
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
    # API Configuration
    'openai_api_key': 'your-key',
    'anthropic_api_key': 'your-key',
    'default_llm_provider': 'openai',
    
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

## Architecture

### Core Components

1. **MultiAgentResearchFramework**: Main orchestrator
2. **VirtualLabMeetingSystem**: Implements Virtual Lab methodology
3. **AgentMarketplace**: Manages domain expert agents
4. **LiteratureRetriever**: Multi-source literature search
5. **Memory Systems**: Vector database and knowledge repository

### Research Phases (Virtual Lab)

1. **Team Selection**: Identify required expertise
2. **Literature Review**: Comprehensive literature search and analysis
3. **Project Specification**: Define research scope and methodology
4. **Tools Selection**: Choose appropriate research tools
5. **Tools Implementation**: Set up and configure tools
6. **Workflow Design**: Plan research execution
7. **Execution**: Conduct the research
8. **Synthesis**: Compile and validate results

## Testing

Run comprehensive tests:

```bash
# Test serialization (should all pass)
python test_comprehensive_serialization.py

# Test literature retriever
python test_literature_retriever.py

# Test Virtual Lab functionality
python test_virtual_lab.py
```

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
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
