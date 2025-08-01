# Scientific Results Visualizer

A Python module for generating matplotlib plots from experiment result dictionaries. Designed for headless environments and scientific data visualization.

## Features

- **Multi-plot Dashboard**: Creates a comprehensive 2x2 grid of visualizations
- **Automatic Field Detection**: Intelligently identifies numeric fields for plotting
- **Multiple Plot Types**: Bar charts, line plots, summary statistics, and pie charts
- **Headless Environment Support**: Configured for server/CI environments
- **Robust Error Handling**: Graceful handling of missing data and edge cases
- **Flexible Input**: Works with various data structures and field types

## Installation

```bash
# Install dependencies
pip install matplotlib numpy pytest

# Or using the requirements file
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from results_visualizer import visualize

# Your experiment results
results = [
    {
        'accuracy': 0.85,
        'loss': 0.15,
        'precision': 0.82,
        'recall': 0.88,
        'status': 'completed',
        'model': 'resnet50',
        'dataset': 'cifar10'
    },
    {
        'accuracy': 0.87,
        'loss': 0.13,
        'precision': 0.84,
        'recall': 0.86,
        'status': 'completed',
        'model': 'resnet50',
        'dataset': 'cifar10'
    }
]

# Generate visualization
visualize(results, 'experiment_results.png')
```

### Supported Data Types

The visualizer handles various data types:

- **Numeric fields**: Automatically detected and plotted (accuracy, loss, precision, etc.)
- **Categorical fields**: Used for grouping and status analysis (model, dataset, status)
- **Boolean fields**: Ignored for numeric plots but can be used in summaries
- **Missing fields**: Gracefully handled with appropriate fallbacks

### Output Format

The visualization creates a 2x2 grid with:

1. **Bar Chart** (top-left): Average values of numeric metrics
2. **Line Plot** (top-right): Trends across experiments
3. **Summary Statistics** (bottom-left): Text summary with experiment counts
4. **Pie Chart** (bottom-right): Distribution by status/type

## API Reference

### `visualize(results: List[Dict], out_path: str) -> None`

Generate and save visualization plots from experiment results.

**Parameters:**
- `results`: List of dictionaries containing experiment results
- `out_path`: Path where the visualization will be saved (should include file extension)

**Raises:**
- `ValueError`: If results list is empty or invalid
- `IOError`: If unable to save the file

**Supported file formats:**
- PNG (`.png`)
- JPG (`.jpg`)
- PDF (`.pdf`)
- SVG (`.svg`)

## Examples

### Machine Learning Results

```python
ml_results = [
    {
        'accuracy': 0.85,
        'loss': 0.15,
        'precision': 0.82,
        'recall': 0.88,
        'status': 'completed',
        'model': 'resnet50'
    }
]
visualize(ml_results, 'ml_results.png')
```

### Scientific Experiment Results

```python
science_results = [
    {
        'temperature': 25.5,
        'pressure': 101.3,
        'yield_percentage': 78.5,
        'status': 'completed',
        'experiment_type': 'catalysis'
    }
]
visualize(science_results, 'science_results.png')
```

## Testing

Run the test suite:

```bash
python -m pytest test_results_visualizer.py -v
```

The test suite covers:
- Numeric data visualization
- Mixed data types
- String-only data
- Empty results handling
- Single result handling
- Large datasets
- Missing fields
- Boolean fields
- File extension handling
- Directory creation

## Requirements

- Python 3.6+
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- pytest >= 6.0.0 (for testing)

## Design Decisions

1. **Headless Environment**: Uses 'Agg' backend for compatibility with servers and CI systems
2. **Automatic Field Detection**: Intelligently identifies numeric vs categorical fields
3. **Robust Error Handling**: Graceful degradation when data is missing or malformed
4. **Multiple Plot Types**: Provides comprehensive overview in a single figure
5. **High DPI Output**: Saves at 300 DPI for publication-quality images

## Contributing

To extend the visualizer:

1. Add new plot types in separate functions
2. Update the main `visualize()` function to include new plots
3. Add corresponding tests
4. Update documentation

## License

This module is designed for scientific research and educational use.