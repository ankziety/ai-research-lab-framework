# AI-Powered Research Framework

A comprehensive framework for AI-powered research workflows that integrates experiment execution, literature retrieval, manuscript drafting, result visualization, and research critique capabilities across any research domain.

## Overview

The AI-Powered Research Framework provides a unified platform for conducting end-to-end research workflows in any field. It leverages AI to assist researchers by automating literature reviews, generating manuscript drafts, providing intelligent critique, and orchestrating complete research pipelines - from initial experiments to publication-ready outputs.

## Features

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

The framework requires Python 3.6+ and has minimal dependencies:

```bash
pip install matplotlib numpy pytest
```

### Configuration

The framework requires API keys for full AI-powered functionality:

#### Required API Keys
- **OpenAI API Key**: For AI agent responses (Principal Investigator, domain experts, scientific critic)

#### Optional API Keys for Enhanced Literature Search
- **Google Search API**: For Google Custom Search academic results
- **SerpAPI Key**: For Google Scholar search access
- **Semantic Scholar API**: Enhanced search (free tier available)
- **OpenAlex Email**: Required for OpenAlex API access (free)
- **CORE API Key**: Optional for CORE repository access

#### Configuration Options
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

#### Environment Variables
You can also set API keys via environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GEMINI_API_KEY="your-google-api-key"  
export HUGGINGFACE_API_KEY="your-huggingface-token"
export LITERATURE_API_KEY="your-pubmed-api-key"  # optional
```

**Note**: Without API keys, the framework will use mock responses for demonstration purposes, but will not provide actual AI-powered research assistance.

### Basic Usage

```python
from ai_research_lab import create_framework

# Create framework instance
framework = create_framework()

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

### Command Line Interface

The framework includes a comprehensive CLI for all operations across any research domain:

```bash
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

## Individual Component Usage

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
- **AIResearchLabFramework**: Main orchestration class
- **ExperimentRunner**: Experiment execution and tracking
- **LiteratureRetriever**: Scientific literature search and retrieval
- **ManuscriptDrafter**: Scientific manuscript generation
- **Critic**: Research output critique and feedback
- **ResultsVisualizer**: Data visualization and plotting
- **SpecialistRegistry**: Component management and extensibility

### Integration Points
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
├── ai_research_lab.py          # Main AI-powered framework integration class
├── cli.py                      # Command-line interface
├── __init__.py                 # Package initialization
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
├── experiments/
│   ├── __init__.py
│   ├── experiment.py           # Automated experiment execution and tracking
│   └── README.md
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

This framework leverages artificial intelligence to enhance research productivity across any domain:

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
