# Quick Start Guide - AI Research Lab Framework

## 🚀 Quick Launch Options

### Option 1: Web Interface (Recommended for beginners)
```bash
# Install dependencies and start web interface
python launch.py --all

# Or just start web interface (if dependencies already installed)
python launch.py --web
```
The web interface will be available at: **http://localhost:5000**

### Option 2: Command Line Interface
```bash
# Show CLI help
python launch.py --cli

# Run research with a question
python -m data.cli virtual-lab-research --question "What are the latest developments in AI?"

# Run an experiment
python -m data.cli run-experiment --params "model=transformer" "dataset=imagenet"

# Draft a manuscript
python -m data.cli draft-manuscript --results-file results.json --output manuscript.md
```

### Option 3: Python API
```bash
# Show Python API example
python launch.py --example
```

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Git** (for cloning the repository)
3. **Optional**: OpenAI API key for enhanced functionality

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-research-lab-framework

# Install dependencies
python launch.py --install

# Or install manually
pip install -r requirements.txt
pip install -r web_ui/requirements.txt
```

## 🎯 Getting Started

### 1. First Time Setup
```bash
# Install everything and start web interface
python launch.py --all
```

### 2. Web Interface Features
- **Research Dashboard**: Monitor ongoing research sessions
- **Agent Management**: View and interact with AI agents
- **Literature Search**: Search academic papers
- **Experiment Runner**: Run and track experiments
- **Manuscript Drafting**: Generate research manuscripts
- **Real-time Updates**: Live updates via WebSocket

### 3. CLI Commands
```bash
# Virtual Lab Research (recommended)
python -m data.cli virtual-lab-research --question "Your research question here"

# Traditional Multi-Agent Research
python -m data.cli run-workflow --question "Your research question here"

# Literature Search
python -m data.cli literature-search --query "machine learning interpretability"

# Experiment Management
python -m data.cli run-experiment --params "param1=value1" "param2=value2"

# Manuscript Generation
python -m data.cli draft-manuscript --results-file results.json

# Visualization
python -m data.cli visualize --results-file results.json --output plot.png
```

### 4. Python API Usage
```python
from ai_research_lab import create_framework

# Initialize framework
framework = create_framework({
    'openai_api_key': 'your-api-key',  # Optional
    'max_agents_per_research': 3,
    'budget_limit': 50.0
})

# Conduct research
results = framework.conduct_virtual_lab_research(
    research_question="How can we improve machine learning model interpretability?",
    constraints={'budget': 25.0, 'timeline_weeks': 1},
    context={'domain': 'computer_science', 'priority': 'high'}
)

print(f"Research completed: {results['status']}")
print(f"Key findings: {results.get('key_findings', [])}")
```

## 🔑 Configuration

### API Keys (Optional but Recommended)
```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or configure via web interface
# Go to http://localhost:5000 and use the Settings page
```

### Configuration File
Create `config/config.json`:
```json
{
  "openai_api_key": "your-api-key",
  "anthropic_api_key": "your-anthropic-key",
  "max_agents_per_research": 5,
  "budget_limit": 100.0,
  "default_model": "gpt-4",
  "literature_sources": ["pubmed", "arxiv", "crossref"]
}
```

## 📊 Features Overview

### Multi-Agent Research System
- **Principal Investigator (PI) Agent**: Coordinates research
- **Agent Marketplace**: Dynamic hiring of domain experts
- **Scientific Critic Agent**: Quality control and validation
- **Domain Expert Agents**: Specialized for different domains

### Virtual Lab Methodology
- **Structured Meetings**: Organized research meetings
- **Phase-based Research**: Systematic progression
- **Meeting Records**: Comprehensive tracking
- **Quality Assessment**: Continuous evaluation

### Literature Integration
- **Multi-source Search**: PubMed, ArXiv, CrossRef, Semantic Scholar
- **Citation Analysis**: Automatic extraction and analysis
- **Research Synthesis**: AI-powered literature review

### Memory and Knowledge Management
- **Vector Database**: Semantic storage and retrieval
- **Knowledge Repository**: Validated findings
- **Context Management**: Session-based memory

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory
2. **Missing Dependencies**: Run `python launch.py --install`
3. **Port Already in Use**: Change port in `web_ui/app.py` or kill existing process
4. **API Key Issues**: Set environment variables or configure via web interface

### Debug Mode
```bash
# Run with debug logging
FLASK_DEBUG=1 python launch.py --web
```

### Reset Database
```bash
# Remove existing data
rm -rf data/*.db web_ui/*.db
```

## 📚 Next Steps

1. **Explore the Web Interface**: Start with the web UI for the best experience
2. **Try Different Research Questions**: Experiment with various domains
3. **Configure API Keys**: For enhanced functionality
4. **Read Documentation**: Check the `docs/` directory for detailed guides
5. **Join the Community**: Contribute to the project

## 🆘 Support

- **Documentation**: Check `docs/` directory
- **Issues**: Report bugs on GitHub
- **Examples**: See `tests/` directory for usage examples 