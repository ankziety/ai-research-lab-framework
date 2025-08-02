#!/usr/bin/env python3

import os
import sys
import json

# Add parent directory to path
sys.path.append('..')

import app

print("Current working directory:", os.getcwd())
config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
print("Config file path:", config_file_path)
print("Config file exists:", os.path.exists(config_file_path))

if os.path.exists(config_file_path):
    print("Reading config file...")
    with open(config_file_path, 'r') as f:
        loaded_config = json.load(f)
    print("Loaded config:", loaded_config)

default_config = {
    'api_keys': {
        'openai': '',
        'anthropic': '',
        'gemini': '',
        'huggingface': '',
        'ollama_endpoint': 'http://localhost:11434'
    },
    'search_api_keys': {
        'google_search': '',
        'google_search_engine_id': '',
        'serpapi': '',
        'semantic_scholar': '',
        'openalex_email': '',
        'core': ''
    },
    'system': {
        'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'),
        'max_concurrent_agents': 8,
        'auto_save_results': True,
        'enable_notifications': True
    },
    'framework': {
        'experiment_db_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'experiments.db'),
        'manuscript_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'manuscripts'),
        'visualization_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations'),
        'max_literature_results': 10,
        'default_llm_provider': 'openai',
        'default_model': 'gpt-4',
        'enable_free_search': True,
        'enable_mock_responses': True
    }
}

print("Default config:", default_config)

merged_config = app._deep_merge(default_config, loaded_config)
print("Merged config:", merged_config)

app.load_system_config()

print("Final API Keys:", app.system_config.get('api_keys', {}))
print("Final Framework:", app.system_config.get('framework', {}))
print("Final Search API Keys:", app.system_config.get('search_api_keys', {})) 