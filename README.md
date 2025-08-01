# Scientific Manuscript Drafter

A Python module for generating structured Markdown drafts from experimental results and contextual information.

## Overview

The `manuscript_drafter.py` module provides a function to automatically generate scientific manuscript drafts in Markdown format. It takes experimental results and contextual information as input and produces a well-structured document with all standard scientific manuscript sections.

## Features

- **Structured Output**: Generates complete manuscripts with title, abstract, introduction, methods, results, discussion, conclusion, and references
- **Flexible Input**: Accepts various formats of experimental results and contextual information
- **Markdown Format**: Outputs clean, editable Markdown text
- **No Dependencies**: Pure Python implementation with no external package requirements
- **Error Handling**: Robust input validation and error handling
- **Comprehensive Testing**: Full test suite covering all functionality

## Usage

### Basic Usage

```python
from manuscript_drafter import draft

# Define experimental results
results = [
    {
        "description": "Temperature effect on enzyme activity",
        "data": {"temperature_20C": 0.45, "temperature_30C": 0.78},
        "implication": "Optimal activity at 30°C"
    }
]

# Define contextual information
context = {
    "objective": "Study enzyme temperature dependence",
    "methods": "Standard spectrophotometric assays",
    "conclusion": "Temperature significantly affects enzyme activity"
}

# Generate manuscript
manuscript = draft(results, context)
print(manuscript)
```

### Input Format

#### Results (List of Dictionaries)

Each result dictionary can contain:

- `description`: Text description of the result
- `outcome`: Alternative to description
- `finding`: Alternative to description
- `data`: Quantitative data (dict, int, or float)
- `implication`: What the result means
- `significance`: Why the result is important
- `id`: Optional identifier
- `metadata`: Optional metadata

#### Context (Dictionary)

The context dictionary can contain:

- `study_type`: Type of study (e.g., "Longitudinal", "Comparative")
- `subject`: Subject of study
- `objective`: Main objective
- `background`: Background information
- `significance`: Why the study is important
- `objectives`: List of specific objectives
- `methods`: Methods description
- `materials`: List of materials used
- `procedures`: List of procedures
- `conclusion`: Main conclusion
- `interpretation`: Interpretation of results
- `limitations`: List of study limitations
- `future_work`: List of future research directions
- `references`: List of references (dict or string format)

## Output Structure

The generated manuscript includes:

1. **Title**: Generated from context information
2. **Abstract**: Summary with background, methods, results, and conclusion
3. **Introduction**: Background, significance, and objectives
4. **Methods**: Materials, procedures, and experimental approach
5. **Results**: Detailed presentation of experimental findings
6. **Discussion**: Interpretation, implications, and limitations
7. **Conclusion**: Main findings and future directions
8. **References**: Formatted reference list (if provided)

## Examples

### Minimal Input

```python
results = [{"outcome": "Positive result"}]
context = {"objective": "Test hypothesis"}
manuscript = draft(results, context)
```

### Rich Context

```python
results = [
    {
        "description": "Temperature measurement",
        "data": {"temperature": 25.5, "humidity": 60},
        "implication": "Optimal conditions identified"
    }
]

context = {
    "study_type": "Experimental",
    "subject": "Environmental Control",
    "objective": "Optimize environmental parameters",
    "background": "Previous studies show temperature affects growth",
    "methods": "Controlled environment experiments",
    "materials": ["Sensors", "Growth chambers"],
    "procedures": ["Setup chambers", "Monitor conditions"],
    "conclusion": "Optimal temperature identified",
    "references": [
        {"authors": "Smith et al.", "year": "2020", "title": "Environmental Studies"}
    ]
}

manuscript = draft(results, context)
```

## Error Handling

The module validates input types and provides clear error messages:

```python
# This will raise ValueError
draft("not a list", {})  # Error: results must be a list
draft([], "not a dict")  # Error: context must be a dictionary
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_simple.py
```

Or run the pytest tests (if pytest is available):

```bash
python3 -m pytest test_manuscript_drafter.py -v
```

## Demonstration

Run the demonstration script to see a complete example:

```bash
python3 demo.py
```

## Requirements

- Python 3.6+
- No external dependencies

## File Structure

```
.
├── manuscript_drafter.py      # Main module
├── test_manuscript_drafter.py # Comprehensive pytest tests
├── test_simple.py            # Simple test script
├── demo.py                   # Demonstration script
└── README.md                 # This documentation
```

## Contributing

The module is designed to be easily extensible. Key areas for enhancement:

- Additional output formats (LaTeX, HTML)
- More sophisticated title generation
- Enhanced reference formatting
- Custom section templates
- Integration with data analysis libraries

## License

This module is provided as-is for educational and research purposes.
