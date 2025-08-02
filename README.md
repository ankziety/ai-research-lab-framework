# AI-Powered Research Framework

A comprehensive framework for AI-powered research workflows that integrates experiment execution, literature retrieval, manuscript drafting, result visualization, and research critique capabilities across any research domain. **Now enhanced with Virtual Lab methodology for structured meeting-based research collaboration.**

## Overview

The AI-Powered Research Framework provides a unified platform for conducting end-to-end research workflows in any field. It leverages AI to assist researchers by automating literature reviews, generating manuscript drafts, providing intelligent critique, and orchestrating complete research pipelines - from initial experiments to publication-ready outputs.

**NEW: Virtual Lab Methodology** - Inspired by the paper "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies" by Swanson et al. (2025), the framework now includes a sophisticated meeting-based research coordination system where AI agents collaborate through structured meetings to conduct interdisciplinary research.

## Features

### 🧪 Virtual Lab Research Methodology (NEW)
- **Meeting-Based Coordination**: Structured team and individual meetings between AI agents
- **7-Phase Research Process**: Team selection, project specification, tools selection, implementation, workflow design, execution, and synthesis
- **Cross-Agent Collaboration**: Agents with different expertise collaborate and critique each other's work
- **Iterative Refinement**: Multiple rounds of discussion and improvement
- **Scientific Critique Integration**: Built-in quality control and validation
- **Minimal Human Input**: AI agents handle most of the research coordination autonomously

### 🧪 Experiment Management
- **Automated Execution**: Run computational experiments with parameter tracking
- **Result Persistence**: SQLite-based storage for experiment results and metadata
- **Status Tracking**: Monitor experiment progress and completion status

### 📚 Literature Integration
- **Smart Retrieval**: Search and retrieve relevant scientific literature
- **Context Integration**: Automatically incorporate literature into research workflows
- **Reference Management**: Format and manage literature references

### 📝 Manuscript Generation
- **Structured Output**: Generate complete manuscripts with all standard sections
- **Context-Aware**: Incorporate experiment results and literature seamlessly
- **Markdown Format**: Clean, editable output format
- **Auto-Integration**: Combine multiple experiment results into cohesive narratives

### 🔍 Intelligent Critique
- **Automated Review**: Rule-based critique of research outputs
- **Structured Feedback**: Organized strengths, weaknesses, and suggestions
- **Quality Scoring**: Numerical quality assessment (0-100 scale)
- **Improvement Guidance**: Actionable recommendations for enhancement

### 📊 Result Visualization
- **Multi-Format Support**: Generate publication-ready visualizations
- **Automated Plotting**: Smart chart selection based on data types
- **Trend Analysis**: Time-series and comparative visualizations
- **Export Options**: PNG, PDF, and other standard formats

### 🔄 Workflow Orchestration
- **End-to-End Pipelines**: Complete research workflows from experiment to manuscript
- **Component Integration**: Seamless interaction between all framework components
- **Configuration Management**: Save and reuse workflow configurations
- **Error Handling**: Robust error management and recovery

### 👥 Specialist Registry
- **Modular Architecture**: Register and manage specialized processing components
- **Custom Extensions**: Easy integration of domain-specific tools
- **Role-Based Access**: Organize functionality by research roles and tasks

## Quick Start

### Installation

#### Prerequisites
- Python 3.8+ (recommended: Python 3.10 or newer)
- Git

#### Step 1: Clone the Repository
```bash
# Clone the AI Research Lab Framework
git clone https://github.com/ankziety/ai-research-lab-framework.git
cd ai-research-lab-framework
```

#### Step 2: Create Virtual Environment
We recommend using a virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
# Install all required dependencies
pip install -r requirements.txt
```

The framework requires the following core dependencies:
- matplotlib>=3.5.0
- numpy>=1.21.0  
- pytest>=6.0.0
- sentence-transformers>=2.2.0
- faiss-cpu>=1.7.0
- openai>=1.0.0 (for AI features)
- anthropic>=0.7.0 (for Claude AI models)

#### Step 4: Verify Installation
```bash
# Test that the framework works correctly
python cli.py --help

# Run basic tests
python test_simple.py
```

### Configuration

The framework works without API keys but provides mock responses for demonstration. For full AI-powered functionality, configure your API keys:

#### Required API Keys (for AI features)
- **OpenAI API Key**: For AI agent responses (Principal Investigator, domain experts, scientific critic)

#### Optional API Keys (for Enhanced Literature Search)  
- **Google Search API**: For Google Custom Search academic results
- **SerpAPI Key**: For Google Scholar search access
- **Semantic Scholar API**: Enhanced search (free tier available)
- **OpenAlex Email**: Required for OpenAlex API access (free)
- **CORE API Key**: Optional for CORE repository access

#### Method 1: Environment Variables (Recommended)
Set your API keys as environment variables:

```bash
# Essential for AI features
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional for additional AI models
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GEMINI_API_KEY="your-google-api-key"  
export HUGGINGFACE_API_KEY="your-huggingface-token"

# Optional for enhanced literature search
export GOOGLE_SEARCH_API_KEY="your-google-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-custom-search-engine-id" 
export SERPAPI_KEY="your-serpapi-key"
export SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-key"
export OPENALEX_EMAIL="your-email@domain.com"
export CORE_API_KEY="your-core-key"
```

#### Method 2: Configuration in Code
```python
from ai_research_lab import create_framework

# Configure with API keys
config = {
    'openai_api_key': 'your-openai-api-key-here',
    
    # Literature search APIs (all optional - fallback to mock data)
    'google_search_api_key': 'your-google-api-key',
    'google_search_engine_id': 'your-custom-search-engine-id',
    'serpapi_key': 'your-serpapi-key',  # For Google Scholar
    'semantic_scholar_api_key': 'your-semantic-scholar-key',
    'openalex_email': 'your-email@domain.com',  # Required for OpenAlex
    'core_api_key': 'your-core-key',  # Optional
    
    'default_llm_provider': 'openai',  # 'openai', 'anthropic', 'gemini', 'huggingface', 'ollama'
    'default_model': 'gpt-4o',  # Latest GPT-4o model
    'cost_optimization': True,  # Auto-select cheapest suitable provider
}

framework = create_framework(config)
```

#### Getting API Keys

**OpenAI API Key** (Essential for AI features):
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up for an account if you don't have one
3. Click "Create new secret key"
4. Copy and save your API key securely

**For other API keys**, see the respective provider documentation.

**Note**: Without API keys, the framework will use mock responses for demonstration purposes, but will not provide actual AI-powered research assistance.

### Getting Help

The CLI provides comprehensive help for all commands:

```bash
# Main help menu with all available commands
python cli.py --help

# Get help for specific commands
python cli.py virtual-lab-research --help
python cli.py run-experiment --help
python cli.py draft-manuscript --help

# Quick command reference
python cli.py --help | grep -A 20 "Examples:"
```

The help system shows:
- All available commands and their descriptions
- Required and optional parameters for each command
- Practical examples for common use cases
- Both Virtual Lab (NEW) and traditional research workflows

### Basic Usage

```python
from ai_research_lab import create_framework

# Create framework instance
framework = create_framework()

# NEW: Virtual Lab Research (Recommended)
# Uses structured meeting-based collaboration between AI agents
virtual_lab_results = framework.conduct_virtual_lab_research(
    research_question="Design new computational approaches for drug discovery targeting viral proteins",
    constraints={
        'budget': 50000,
        'timeline_weeks': 12,
        'team_size_max': 6
    },
    context={
        'domain': 'computational_biology',
        'priority': 'high'
    }
)

print(f"Virtual Lab Session: {virtual_lab_results['session_id']}")
print(f"Phases Completed: {virtual_lab_results['final_results']['session_summary']['phases_completed']}/7")
print(f"Key Findings: {virtual_lab_results['final_results']['validated_findings']}")

# Traditional Multi-Agent Research (Still Available)
# Example 1: Biology research workflow
experiment_params = {
    'treatment': 'drug_compound_X',
    'dosage': 50,  # mg/kg
    'duration': 14,  # days
    'subjects': 30
}

manuscript_context = {
    'objective': 'Evaluate efficacy of compound X on cellular regeneration',
    'methods': 'Double-blind placebo-controlled study',
    'conclusion': 'Compound X showed significant improvement in regeneration rates'
}

# Execute complete workflow
results = framework.run_complete_workflow(
    experiment_params=experiment_params,
    manuscript_context=manuscript_context,
    literature_query='cellular regeneration drug compounds'
)

print(f"Workflow completed: {results['workflow_id']}")
print(f"Manuscript saved to: {results['manuscript']['path']}")
```

### Virtual Lab Command Line Interface

The framework includes enhanced CLI support for Virtual Lab research:

```bash
# Run Virtual Lab research session (NEW)
python cli.py virtual-lab-research \
    --question "Design new computational approaches for drug discovery" \
    --budget 50000 \
    --timeline-weeks 12 \
    --max-agents 6 \
    --domain computational_biology \
    --priority high \
    --output vlab_results.json

# View Virtual Lab session results (NEW)
python cli.py show-vlab-session --session-id vlab_session_123456

# List all Virtual Lab sessions (NEW)
python cli.py list-vlab-sessions

# Get Virtual Lab meeting statistics (NEW)
python cli.py vlab-stats --output vlab_statistics.json

# Traditional commands still available
# Biology/Medicine: Run a treatment study
python cli.py run-experiment --params treatment=drug_X dosage=50 duration=14

# Chemistry: Run a synthesis experiment  
python cli.py run-experiment --params catalyst=platinum temperature=350 pressure=2.5

# Physics: Run a materials study
python cli.py run-experiment --params material=graphene voltage=1.2 frequency=1000

# Draft manuscripts from any domain
python cli.py draft-manuscript --results-file results.json --objective "Study drug efficacy"

# Run complete research workflow
python cli.py run-workflow --config-file biology_workflow.json

# Generate domain-specific visualizations
python cli.py visualize --results-file results.json --output research_plot.png

# AI-powered critique of any research text
python cli.py critique --file manuscript.md
```

## Virtual Lab Methodology

The Virtual Lab is an enhanced research approach that coordinates AI agents through structured meetings to conduct sophisticated, interdisciplinary research. It implements the methodology described in "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies" by Swanson et al.

### 7-Phase Research Process

1. **Team Selection**: Individual meeting with PI to analyze research requirements and hire appropriate expert agents
2. **Project Specification**: Team meeting to define objectives, scope, and success criteria
3. **Tools Selection**: Team meeting to brainstorm and select computational/analytical tools
4. **Tools Implementation**: Individual meetings to implement selected tools with scientific critique
5. **Workflow Design**: Individual meeting with PI to design integrated research workflow
6. **Execution**: Execute workflow with cross-agent collaboration and critique
7. **Synthesis**: Final team meeting to synthesize findings and conduct scientific critique

### Meeting Types

- **Team Meetings**: All agents collaborate on broad research questions
- **Individual Meetings**: Focused work between specific agents (often with scientific critic)
- **Aggregation Meetings**: Combine results from parallel work streams

### Key Advantages

- **Structured Coordination**: Systematic approach to complex research problems
- **Cross-Pollination**: Agents critique and build upon each other's work
- **Quality Control**: Integrated scientific critique throughout the process
- **Scalability**: Can handle teams of 2-8 expert agents
- **Minimal Human Input**: Requires only initial research question and constraints

### Example Virtual Lab Usage

```python
from multi_agent_framework import MultiAgentResearchFramework

# Initialize framework
framework = MultiAgentResearchFramework({
    'openai_api_key': 'your-api-key',
    'max_agents_per_research': 6,
    'store_all_interactions': True
})

# Conduct Virtual Lab research
results = framework.conduct_virtual_lab_research(
    research_question="Develop machine learning models for predicting protein-protein interactions in cancer",
    constraints={
        'budget': 75000,
        'timeline_weeks': 16,
        'team_size_max': 5
    },
    context={
        'domain': 'oncology_informatics',
        'priority': 'medium',
        'requires_validation': True
    }
)

# Access results
session_summary = results['final_results']['session_summary']
validated_findings = results['final_results']['validated_findings']
quality_assessment = results['final_results']['quality_assessment']

print(f"Research completed in {session_summary['duration']:.1f} seconds")
print(f"Quality score: {quality_assessment['overall_score']}/100")

# Access detailed meeting records
meeting_history = framework.get_meeting_history()
for meeting in meeting_history:
    print(f"Meeting: {meeting.meeting_id} ({meeting.phase.value})")
    print(f"Participants: {meeting.participants}")
    print(f"Decisions: {meeting.decisions}")
```

## Individual Component Usage

### Virtual Lab Demo

```bash
# Run the Virtual Lab demonstration
python demo_virtual_lab.py

# Run minimal Virtual Lab verification
python minimal_virtual_lab_demo.py
```

## Individual Component Usage

### Experiment Runner

```python
# Biology: Run a drug treatment experiment
experiment_params = {
    'drug_name': 'aspirin',
    'dosage_mg': 325,
    'treatment_duration_days': 7,
    'control_group_size': 50
}

# Chemistry: Run a catalytic reaction study
experiment_params = {
    'catalyst': 'palladium',
    'temperature_C': 250,
    'pressure_atm': 3.0,
    'reaction_time_hours': 4
}

# Physics: Run a conductivity measurement
experiment_params = {
    'material': 'copper_nanowire',
    'voltage_V': 1.5,
    'temperature_K': 300,
    'measurement_points': 100
}

results = framework.run_experiment(experiment_params)
print(f"Experiment ID: {results['experiment_id']}")
```

### Literature Retrieval

```python
# AI-powered literature search across multiple sources and domains
# Supports: PubMed, ArXiv, CrossRef, Google Scholar, Google Search, 
# Semantic Scholar, OpenAlex, and CORE

# Biology/Medicine with multiple sources
literature = framework.retrieve_literature(
    query="CRISPR gene editing therapeutic applications",
    max_results=10,
    sources=['pubmed', 'semantic_scholar', 'google_scholar']
)

# Chemistry with open access focus
literature = framework.retrieve_literature(
    query="green chemistry sustainable catalysis",
    max_results=10,
    sources=['arxiv', 'openalex', 'core']
)

# Physics with comprehensive search
literature = framework.retrieve_literature(
    query="quantum computing error correction",
    max_results=10,
    sources=['pubmed', 'arxiv', 'crossref', 'semantic_scholar', 'google_search']
)

# Configure API keys for enhanced search capabilities
config = {
    'openai_api_key': 'your-openai-key',
    'google_search_api_key': 'your-google-api-key',
    'google_search_engine_id': 'your-search-engine-id',
    'serpapi_key': 'your-serpapi-key',  # For Google Scholar
    'semantic_scholar_api_key': 'your-semantic-scholar-key',
    'openalex_email': 'your-email@domain.com',  # Required for OpenAlex
    'core_api_key': 'your-core-key'  # Optional for CORE
}

framework = create_framework(config)

for paper in literature:
    print(f"{paper['title']} ({paper['publication_year']}) - {paper['source']}")
    print(f"Citations: {paper.get('citation_count', 0)}")
    if paper.get('open_access'):
        print(f"Open Access PDF: {paper.get('pdf_url', 'Available')}")
```
```

### Manuscript Drafting

```python
# Generate manuscript from results
results = [{"description": "Experiment results", "data": {...}}]
context = {"objective": "Research goal", "methods": "Methodology"}

manuscript = framework.draft_manuscript(results, context)
```

### Research Critique

```python
# Critique research content
critique = framework.critique_output(manuscript_text)
print(f"Quality score: {critique['overall_score']}/100")
print(f"Suggestions: {critique['suggestions']}")
```

### Result Visualization

```python
# Generate visualizations
framework.visualize_results(
    results=experiment_results,
    out_path="results_plot.png"
)
```

## Configuration

The framework supports flexible configuration:

```python
config = {
    'experiment_db_path': 'custom/experiments.db',
    'output_dir': 'research_output',
    'manuscript_dir': 'papers',
    'visualization_dir': 'plots',
    'max_literature_results': 15,
    'auto_visualize': True,
    'auto_critique': True
}

framework = create_framework(config)
```

### Configuration Options

- `experiment_db_path`: Path to SQLite database for experiments
- `output_dir`: Base directory for all outputs
- `manuscript_dir`: Directory for generated manuscripts
- `visualization_dir`: Directory for plots and visualizations
- `literature_api_url`: API endpoint for literature retrieval
- `literature_api_key`: API key for literature services
- `max_literature_results`: Maximum papers to retrieve per query
- `auto_visualize`: Automatically generate plots after experiments
- `auto_critique`: Automatically critique generated manuscripts

## Testing

### Run All Tests

```bash
# Run Virtual Lab tests
python test_virtual_lab.py

# Run integration tests
python test_integration.py

# Run individual component tests  
python test_simple.py
python test_manuscript_drafter.py
python test_literature_retriever.py
python test_critic.py
python test_results_visualizer.py
python test_specialist_registry.py

# Using pytest (if available)
python -m pytest test_*.py -v
```

### Demo and Examples

```bash
# Run Virtual Lab demo (NEW)
python demo_virtual_lab.py

# Run minimal Virtual Lab verification
python minimal_virtual_lab_demo.py

# Run comprehensive framework demo
python demo_integrated.py

# Run individual component demos
python demo.py                # Original manuscript drafter demo
python demo_critic.py         # Critic component demo
python example_usage.py       # Results visualizer examples
```

## Architecture

The framework follows a modular architecture with the following components:

### Core Components
- **MultiAgentResearchFramework**: Main orchestration class with Virtual Lab integration
- **VirtualLabMeetingSystem**: Meeting-based research coordination (NEW)
- **ExperimentRunner**: Experiment execution and tracking
- **LiteratureRetriever**: Scientific literature search and retrieval
- **ManuscriptDrafter**: Scientific manuscript generation
- **Critic**: Research output critique and feedback
- **ResultsVisualizer**: Data visualization and plotting
- **SpecialistRegistry**: Component management and extensibility

### Integration Points
- **Virtual Lab Workflow**: Structured 7-phase research methodology (NEW)
- **Meeting-Based Coordination**: Team and individual meetings between AI agents (NEW)
- **Cross-Agent Collaboration**: Agents critique and enhance each other's work (NEW)
- **Workflow Orchestration**: End-to-end research pipelines
- **Configuration Management**: Centralized settings and preferences
- **Error Handling**: Consistent error management across components
- **Logging**: Unified logging and monitoring
- **CLI Interface**: Command-line access to all functionality

## Requirements

- Python 3.6+
- matplotlib>=3.5.0
- numpy>=1.21.0
- pytest>=6.0.0 (for testing)

## File Structure

```
ai-powered-research-framework/
├── virtual_lab.py               # NEW: Virtual Lab meeting system implementation
├── ai_research_lab.py          # Main AI-powered framework integration class
├── multi_agent_framework.py    # Enhanced multi-agent system with Virtual Lab
├── cli.py                      # Command-line interface
├── __init__.py                 # Package initialization
├── demo_virtual_lab.py         # NEW: Virtual Lab comprehensive demo
├── minimal_virtual_lab_demo.py # NEW: Virtual Lab minimal verification
├── test_virtual_lab.py         # NEW: Virtual Lab tests
├── demo_integrated.py          # Comprehensive framework demo
├── test_integration.py         # Integration tests
├── README.md                   # This documentation
├── requirements.txt            # Package dependencies
│
├── AI-Powered Components/
├── manuscript_drafter.py       # AI-assisted manuscript generation
├── literature_retriever.py     # AI-powered literature search and retrieval
├── critic.py                   # AI-driven research output critique
├── results_visualizer.py       # Intelligent data visualization
├── specialist_registry.py      # Component registry and management
├── agents/                     # Multi-agent system components
│   ├── __init__.py
│   ├── base_agent.py           # Base agent class
│   ├── principal_investigator.py # PI agent for coordination
│   ├── scientific_critic.py    # Scientific critic agent
│   ├── domain_experts.py       # Domain expert agents
│   ├── agent_marketplace.py    # Agent hiring and management
│   └── llm_client.py           # LLM integration
├── memory/                     # Memory and context management
│   ├── __init__.py
│   ├── vector_database.py      # Vector database for embeddings
│   ├── context_manager.py      # Context management
│   └── knowledge_repository.py # Validated knowledge storage
├── experiments/
│   ├── __init__.py
│   ├── experiment.py           # Automated experiment execution and tracking
│   └── README.md
├── tools/                      # Research tools and utilities
│   ├── __init__.py
│   ├── base_tool.py            # Base tool interface
│   ├── experimental_tools.py   # Experiment design and execution
│   ├── collaboration_tools.py  # Agent collaboration utilities
│   └── tool_registry.py       # Tool management
│
├── Original Demos & Tests/
├── demo.py                     # Original manuscript drafter demo
├── demo_critic.py              # Critic demo
├── example_usage.py            # Visualizer examples
├── test_*.py                   # Individual component tests
│
└── Generated Outputs/
    ├── manuscripts/            # Generated manuscript files
    ├── visualizations/         # Generated plots and charts
    ├── output/                 # General output directory
    └── sessions/               # Session data and logs
```

## AI-Powered Research Approach

This framework leverages artificial intelligence to enhance research productivity across any domain, now with advanced Virtual Lab methodology:

### 🤖 Virtual Lab Multi-Agent Collaboration (NEW)
- **Meeting-Based Research**: AI agents conduct structured team and individual meetings
- **Cross-Pollination**: Agents with different expertise critique and enhance each other's work
- **Iterative Refinement**: Multiple rounds of discussion and improvement
- **Scientific Quality Control**: Integrated critique and validation throughout the process
- **Systematic Methodology**: 7-phase structured approach to complex research problems

### 🤖 AI-Assisted Literature Review
- **Intelligent Query Processing**: Natural language queries automatically translated to database searches
- **Relevance Ranking**: AI-powered ranking of literature based on research context
- **Citation Networks**: Automated discovery of related papers and research trends

### ✍️ AI-Generated Manuscripts  
- **Context-Aware Writing**: Manuscripts tailored to specific research domains and objectives
- **Structure Optimization**: Automatic section organization based on research type
- **Style Adaptation**: Writing style adjusted for target journals and audiences

### 🔍 Intelligent Research Critique
- **Automated Quality Assessment**: AI-driven evaluation of research methodology and conclusions
- **Bias Detection**: Identification of potential biases and limitations
- **Improvement Suggestions**: Actionable recommendations for strengthening research

### 📊 Smart Data Visualization
- **Automatic Chart Selection**: AI chooses optimal visualization types for data
- **Insight Generation**: Automated identification of trends and patterns
- **Publication-Ready Outputs**: Professional-quality figures for manuscripts

### 🧠 Domain-Agnostic Intelligence
The framework's AI components are designed to work across research disciplines:
- **Biology & Medicine**: Drug discovery, clinical trials, genetic analysis
- **Chemistry**: Reaction optimization, materials characterization, synthesis planning  
- **Physics**: Materials properties, experimental design, data analysis
- **Social Sciences**: Survey analysis, behavioral studies, statistical modeling
- **Environmental Science**: Climate modeling, ecosystem analysis, pollution studies

## Contributing

The framework is designed to be easily extensible. Key areas for enhancement:

### Component Extensions
- **Additional Literature Sources**: Integration with more academic databases
- **Advanced Critique Models**: LLM-based critique and feedback
- **Enhanced Visualizations**: Interactive plots and dashboards
- **Export Formats**: LaTeX, HTML, and other output formats

### Workflow Enhancements  
- **Parallel Processing**: Concurrent experiment execution
- **Real-time Monitoring**: Live experiment tracking and alerts
- **Version Control**: Experiment versioning and reproducibility
- **Collaboration**: Multi-user workflows and shared resources

### Integration Opportunities
- **Cloud Platforms**: AWS, GCP, Azure integration
- **Jupyter Notebooks**: Native notebook support
- **Web Interface**: Browser-based research dashboard
- **API Services**: RESTful API for remote access

### Custom Specialists
```python
# Example: Register custom specialist
def custom_analyzer(data):
    # Custom analysis logic
    return analysis_results

framework.specialist_registry.register('custom_analyzer', custom_analyzer)
```

## License

This AI-Powered Research Framework is provided as-is for educational and research purposes. See LICENSE file for details.
