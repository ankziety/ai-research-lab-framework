# AI Research Lab Framework

A comprehensive framework for AI research workflows that integrates experiment execution, literature retrieval, manuscript drafting, result visualization, and research critique capabilities.

## Overview

The AI Research Lab Framework provides a unified platform for conducting end-to-end AI research workflows. It combines multiple specialized components into a cohesive system that can run experiments, analyze results, retrieve relevant literature, draft manuscripts, and provide automated critique - all through a simple, unified API.

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

### Basic Usage

```python
from ai_research_lab import create_framework

# Create framework instance
framework = create_framework()

# Run a complete research workflow
experiment_params = {
    'algorithm': 'neural_network',
    'learning_rate': 0.001,
    'epochs': 100
}

manuscript_context = {
    'objective': 'Evaluate neural network performance',
    'methods': 'Deep learning with backpropagation',
    'conclusion': 'Neural network achieved high accuracy'
}

# Execute complete workflow
results = framework.run_complete_workflow(
    experiment_params=experiment_params,
    manuscript_context=manuscript_context,
    literature_query='neural networks machine learning'
)

print(f"Workflow completed: {results['workflow_id']}")
print(f"Manuscript saved to: {results['manuscript']['path']}")
```

### Command Line Interface

The framework includes a comprehensive CLI for all operations:

```bash
# Run a single experiment
python cli.py run-experiment --params algorithm=svm kernel=rbf C=1.0

# Draft a manuscript from results
python cli.py draft-manuscript --results-file results.json --objective "Study SVM performance"

# Run complete workflow
python cli.py run-workflow --config-file workflow.json

# Generate visualizations
python cli.py visualize --results-file results.json --output plot.png

# Critique research text
python cli.py critique --file manuscript.md
```

## Individual Component Usage

### Experiment Runner

```python
# Run individual experiments
experiment_params = {
    'algorithm': 'random_forest',
    'n_estimators': 100,
    'max_depth': 10
}

results = framework.run_experiment(experiment_params)
print(f"Experiment ID: {results['experiment_id']}")
```

### Literature Retrieval

```python
# Search for relevant literature
literature = framework.retrieve_literature(
    query="machine learning classification",
    max_results=10
)

for paper in literature:
    print(f"{paper['title']} ({paper['publication_year']})")
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
ai-research-lab-framework/
├── ai_research_lab.py          # Main framework integration class
├── cli.py                      # Command-line interface
├── __init__.py                 # Package initialization
├── demo_integrated.py          # Comprehensive framework demo
├── test_integration.py         # Integration tests
├── README.md                   # This documentation
├── requirements.txt            # Package dependencies
│
├── Individual Components/
├── manuscript_drafter.py       # Scientific manuscript generation
├── literature_retriever.py     # Literature search and retrieval
├── critic.py                   # Research output critique
├── results_visualizer.py       # Data visualization
├── specialist_registry.py      # Component registry
├── experiments/
│   ├── __init__.py
│   ├── experiment.py           # Experiment execution and tracking
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

This framework is provided as-is for educational and research purposes. See LICENSE file for details.
