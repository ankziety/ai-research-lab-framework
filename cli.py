#!/usr/bin/env python3
"""
AI Research Lab Framework CLI

Command-line interface for the AI research lab framework, providing easy access
to all framework functionality.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from ai_research_lab import create_framework


def run_experiment_command(args):
    """Handle the run-experiment command."""
    framework = create_framework()
    
    # Parse experiment parameters
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {}
        if args.params:
            for param in args.params:
                key, value = param.split('=', 1)
                # Try to parse as number, fallback to string
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                params[key] = value
    
    if not params:
        print("Error: No experiment parameters provided. Use --params or --params-file")
        return 1
    
    print(f"Running experiment with parameters: {params}")
    results = framework.run_experiment(params)
    
    print(f"Experiment completed successfully!")
    print(f"Experiment ID: {results['experiment_id']}")
    print(f"Status: {results['status']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return 0


def draft_manuscript_command(args):
    """Handle the draft-manuscript command."""
    framework = create_framework()
    
    # Load experiment results
    if args.results_file:
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
            # Handle both single result and list of results
            results = [results_data] if isinstance(results_data, dict) else results_data
    else:
        print("Error: --results-file is required for manuscript drafting")
        return 1
    
    # Load or create context
    if args.context_file:
        with open(args.context_file, 'r') as f:
            context = json.load(f)
    else:
        context = {
            'objective': args.objective or 'Research study',
            'methods': args.methods or 'Standard experimental methods',
            'conclusion': args.conclusion or 'Results demonstrate significant findings'
        }
    
    print("Drafting manuscript...")
    manuscript = framework.draft_manuscript(results, context)
    
    # Save manuscript
    output_path = args.output or f"manuscript_{framework._generate_id()}.md"
    with open(output_path, 'w') as f:
        f.write(manuscript)
    
    print(f"Manuscript saved to: {output_path}")
    
    # Auto-critique if requested
    if args.critique:
        print("\nCritiquing manuscript...")
        critique = framework.critique_output(manuscript)
        print(f"Overall score: {critique['overall_score']}/100")
        print(f"Strengths: {len(critique['strengths'])}")
        print(f"Weaknesses: {len(critique['weaknesses'])}")
        print(f"Suggestions: {len(critique['suggestions'])}")
    
    return 0


def run_workflow_command(args):
    """Handle the run-workflow command."""
    framework = create_framework()
    
    # Load workflow configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        
        experiment_params = config.get('experiment_params', {})
        manuscript_context = config.get('manuscript_context', {})
        literature_query = config.get('literature_query')
    else:
        # Create basic configuration from command line args
        experiment_params = {}
        if args.params:
            for param in args.params:
                key, value = param.split('=', 1)
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                experiment_params[key] = value
        
        manuscript_context = {
            'objective': args.objective or 'Research study',
            'methods': args.methods or 'Standard experimental methods',
            'conclusion': args.conclusion or 'Results demonstrate significant findings'
        }
        
        literature_query = args.literature_query
    
    if not experiment_params:
        print("Error: No experiment parameters provided")
        return 1
    
    print("Running complete research workflow...")
    print(f"Experiment parameters: {experiment_params}")
    if literature_query:
        print(f"Literature query: {literature_query}")
    
    results = framework.run_complete_workflow(
        experiment_params=experiment_params,
        manuscript_context=manuscript_context,
        literature_query=literature_query
    )
    
    print(f"\nWorkflow completed: {results['workflow_id']}")
    print(f"Status: {results['status']}")
    
    if results['status'] == 'completed':
        print(f"Experiment ID: {results['experiment']['experiment_id']}")
        print(f"Manuscript saved to: {results['manuscript']['path']}")
        print(f"Critique score: {results['critique']['overall_score']}/100")
        
        if 'literature' in results:
            print(f"Literature results: {len(results['literature'])} papers found")
    
    # Save workflow results
    if args.output:
        with open(args.output, 'w') as f:
            # Create a serializable version of results
            serializable_results = {
                'workflow_id': results['workflow_id'],
                'status': results['status'],
                'experiment': results.get('experiment'),
                'manuscript_path': results.get('manuscript', {}).get('path'),
                'critique': results.get('critique'),
                'literature_count': len(results.get('literature', []))
            }
            json.dump(serializable_results, f, indent=2)
        print(f"Workflow summary saved to: {args.output}")
    
    return 0


def visualize_command(args):
    """Handle the visualize command."""
    framework = create_framework()
    
    # Load results data
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)
        # Handle both single result and list of results
        results = [results_data] if isinstance(results_data, dict) else results_data
    
    output_path = args.output or f"visualization_{framework._generate_id()}.png"
    
    print(f"Generating visualization from {len(results)} results...")
    framework.visualize_results(results, output_path)
    print(f"Visualization saved to: {output_path}")
    
    return 0


def critique_command(args):
    """Handle the critique command."""
    framework = create_framework()
    
    # Read input text
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Either --file or --text must be provided")
        return 1
    
    print("Critiquing text...")
    critique = framework.critique_output(text)
    
    print(f"\nCritique Results:")
    print(f"Overall Score: {critique['overall_score']}/100")
    print(f"\nStrengths ({len(critique['strengths'])}):")
    for i, strength in enumerate(critique['strengths'], 1):
        print(f"  {i}. {strength}")
    
    print(f"\nWeaknesses ({len(critique['weaknesses'])}):")
    for i, weakness in enumerate(critique['weaknesses'], 1):
        print(f"  {i}. {weakness}")
    
    print(f"\nSuggestions ({len(critique['suggestions'])}):")
    for i, suggestion in enumerate(critique['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(critique, f, indent=2)
        print(f"\nCritique saved to: {args.output}")
    
    return 0


def list_specialists_command(args):
    """Handle the list-specialists command."""
    framework = create_framework()
    specialists = framework.list_specialists()
    
    print("Registered Specialists:")
    for specialist in specialists:
        print(f"  - {specialist}")
    
    return 0


def create_config_command(args):
    """Handle the create-config command."""
    config = {
        'experiment_params': {},
        'manuscript_context': {
            'objective': 'Research study objective',
            'methods': 'Experimental methods description',
            'conclusion': 'Study conclusions'
        },
        'literature_query': None
    }
    
    output_path = args.output or 'workflow_config.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Template configuration created: {output_path}")
    print("Edit this file with your specific parameters and use with --config-file")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI Research Lab Framework CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run an experiment
  %(prog)s run-experiment --params algorithm=neural_network learning_rate=0.001
  
  # Draft a manuscript
  %(prog)s draft-manuscript --results-file results.json --objective "Study AI performance"
  
  # Run complete workflow
  %(prog)s run-workflow --config-file workflow.json
  
  # Visualize results
  %(prog)s visualize --results-file results.json --output plot.png
  
  # Critique text
  %(prog)s critique --file manuscript.md
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiment command
    exp_parser = subparsers.add_parser('run-experiment', help='Run a single experiment')
    exp_parser.add_argument('--params', nargs='+', help='Experiment parameters as key=value pairs')
    exp_parser.add_argument('--params-file', help='JSON file containing experiment parameters')
    exp_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Draft manuscript command
    manuscript_parser = subparsers.add_parser('draft-manuscript', help='Draft a manuscript from results')
    manuscript_parser.add_argument('--results-file', required=True, help='JSON file containing experiment results')
    manuscript_parser.add_argument('--context-file', help='JSON file containing manuscript context')
    manuscript_parser.add_argument('--objective', help='Research objective')
    manuscript_parser.add_argument('--methods', help='Methods description')
    manuscript_parser.add_argument('--conclusion', help='Study conclusion')
    manuscript_parser.add_argument('--output', help='Output file for manuscript (Markdown)')
    manuscript_parser.add_argument('--critique', action='store_true', help='Auto-critique the manuscript')
    
    # Run workflow command
    workflow_parser = subparsers.add_parser('run-workflow', help='Run complete research workflow')
    workflow_parser.add_argument('--config-file', help='JSON configuration file for workflow')
    workflow_parser.add_argument('--params', nargs='+', help='Experiment parameters as key=value pairs')
    workflow_parser.add_argument('--objective', help='Research objective')
    workflow_parser.add_argument('--methods', help='Methods description')
    workflow_parser.add_argument('--conclusion', help='Study conclusion')
    workflow_parser.add_argument('--literature-query', help='Query for literature retrieval')
    workflow_parser.add_argument('--output', help='Output file for workflow summary (JSON)')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations from results')
    viz_parser.add_argument('--results-file', required=True, help='JSON file containing results to visualize')
    viz_parser.add_argument('--output', help='Output file for visualization (PNG)')
    
    # Critique command
    critique_parser = subparsers.add_parser('critique', help='Critique research text')
    critique_parser.add_argument('--file', help='File containing text to critique')
    critique_parser.add_argument('--text', help='Text string to critique')
    critique_parser.add_argument('--output', help='Output file for critique results (JSON)')
    
    # List specialists command
    subparsers.add_parser('list-specialists', help='List all registered specialists')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create a template workflow configuration')
    config_parser.add_argument('--output', help='Output file for configuration template')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    command_handlers = {
        'run-experiment': run_experiment_command,
        'draft-manuscript': draft_manuscript_command,
        'run-workflow': run_workflow_command,
        'visualize': visualize_command,
        'critique': critique_command,
        'list-specialists': list_specialists_command,
        'create-config': create_config_command
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())