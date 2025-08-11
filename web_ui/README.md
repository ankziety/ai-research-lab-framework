# Web UI

This directory contains the Gradio-based web interface for the AI Research Lab framework.

## Components

- `gradio_app.py` - Main Gradio application with comprehensive research interface
- `data_manager.py` - Data persistence and session management
- `gradio_patch.py` - Monkey patch to fix Gradio client utils compatibility issues

## Gradio Compatibility Patch

### Issue
The application uses Gradio 4.44.1 which has a compatibility issue with `gradio_client.utils.py`. The `get_type` function doesn't handle boolean values in JSON Schema (like `additionalProperties: true`), causing a `TypeError: argument of type 'bool' is not iterable`.

### Solution
We've implemented a monkey patch in `gradio_patch.py` that:

1. **Preserves pip module integrity** - No modification of installed packages
2. **Handles boolean values** - Properly processes `additionalProperties: true` in JSON Schema
3. **Maintains compatibility** - Works with all existing Gradio functionality
4. **Easy to apply** - Simply import and call `apply_gradio_patch()` before importing gradio

### Usage
```python
# Apply patch before importing gradio
from gradio_patch import apply_gradio_patch
apply_gradio_patch()

import gradio as gr
# ... rest of your code
```

### Technical Details
The patch replaces the `get_type` function in `gradio_client.utils` with a version that:
- Checks for boolean values first: `if isinstance(schema, bool): return "boolean"`
- Maintains all original functionality for dictionary schemas
- Provides proper error handling for unexpected types

## Running the Application

```bash
# From project root
source .venv/bin/activate
cd web_ui
python gradio_app.py
```

## Testing

```bash
# Run web UI tests
pytest web_ui/test_gradio_ui.py -q -W ignore::DeprecationWarning
```

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- gradio>=4.44.0
- fastapi>=0.116.0
- starlette>=0.47.0