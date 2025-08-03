"""
Configuration management for the AI Research Lab.

This package contains configuration files and settings for the framework.
"""

import os
import json
from pathlib import Path

def load_config(config_name="config.json"):
    """Load configuration from the config directory."""
    config_path = Path(__file__).parent / config_name
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config, config_name="config.json"):
    """Save configuration to the config directory."""
    config_path = Path(__file__).parent / config_name
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

__all__ = ['load_config', 'save_config'] 