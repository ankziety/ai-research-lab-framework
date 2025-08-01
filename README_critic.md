# Critic Module

The Critic module provides rule-based critique functionality for research outputs, designed to analyze text quality and provide constructive feedback.

## Overview

The `Critic` class analyzes research content using heuristic rules and returns structured feedback including strengths, weaknesses, improvement suggestions, and an overall quality score.

## Features

- **Content Analysis**: Word count, paragraph structure, sentence complexity
- **Academic Quality**: Citation detection, methodology identification, academic terminology
- **Structural Analysis**: Section organization, results presentation
- **Quality Scoring**: 0-100 point scoring system
- **Actionable Feedback**: Specific suggestions for improvement

## Usage

### Basic Usage

```python
from critic import Critic

# Initialize the critic
critic = Critic()

# Review research content
research_text = """
Your research content here...
"""

result = critic.review(research_text)

# Access the results
print(f"Score: {result['overall_score']}/100")
print("Strengths:", result['strengths'])
print("Weaknesses:", result['weaknesses'])
print("Suggestions:", result['suggestions'])
```

### Return Format

The `review()` method returns a dictionary with:

- `strengths`: List of identified positive aspects
- `weaknesses`: List of identified problems
- `suggestions`: List of actionable improvement recommendations
- `overall_score`: Integer score from 0-100

### Configuration

The Critic class can be customized by modifying thresholds:

```python
critic = Critic()
critic.min_word_count = 100      # Minimum words for adequate content
critic.max_word_count = 5000     # Maximum words before flagging as verbose
critic.min_paragraph_count = 3   # Minimum paragraphs for good structure
critic.max_repetition_ratio = 0.2 # Maximum allowed repetition ratio
```

## Analysis Criteria

### Strengths Detection
- Adequate content length (50+ words)
- Presence of citations (`[1]`, `(Author 2023)`, `et al.`)
- Methodology discussion
- Results presentation
- Structured academic sections
- Good paragraph organization
- Appropriate sentence length (15-25 words)
- Minimal repetition (<15%)
- Academic terminology usage

### Weaknesses Detection
- Content too short (<50 words) or too long (>10,000 words)
- Missing citations
- No clear methodology
- No results presented
- Poor paragraph structure
- Inappropriate sentence length
- Excessive repetition (>30%)
- Limited academic terminology

### Scoring System
- Base score: 50 points
- Bonuses for: proper length, structure, citations, methodology, results, readability, terminology
- Penalties for: inadequate length, poor structure, excessive repetition, missing elements

## Design Philosophy

The module is designed to be:
- **Modular**: Easy to replace with LLM-based approaches
- **Configurable**: Adjustable thresholds for different contexts
- **Comprehensive**: Covers multiple quality dimensions
- **Actionable**: Provides specific improvement suggestions

## Future Extensions

The rule-based system can be easily extended or replaced with:
- LLM API integration
- Domain-specific heuristics
- Machine learning models
- Custom scoring algorithms

## Testing

Run the test suite:
```bash
python -m unittest test_critic.py -v
```

Run the demo:
```bash
python demo_critic.py
```

## Dependencies

- Python 3.6+
- No external dependencies (uses only standard library)