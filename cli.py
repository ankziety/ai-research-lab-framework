#!/usr/bin/env python3
"""
AI-Powered Research Framework CLI

Command-line interface for the AI-powered research framework, providing easy access
to all framework functionality across any research domain.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from core.ai_research_lab import create_framework
    from core.virtual_lab import MeetingRecord, MeetingAgenda
except ImportError:
    # Fallback for when running as module
    from ..core.ai_research_lab import create_framework
    from ..core.virtual_lab import MeetingRecord, MeetingAgenda


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


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


def virtual_lab_research_command(args):
    """Handle the virtual-lab-research command."""
    framework = create_framework()
    
    # Prepare constraints
    constraints = {}
    if args.budget:
        constraints['budget'] = args.budget
    if args.timeline_weeks:
        constraints['timeline_weeks'] = args.timeline_weeks
    if args.max_agents:
        constraints['team_size_max'] = args.max_agents
    
    # Prepare context
    context = {}
    if args.domain:
        context['domain'] = args.domain
    if args.priority:
        context['priority'] = args.priority
    
    print(f"Starting Virtual Lab research session...")
    print(f"Research question: {args.question}")
    if constraints:
        print(f"Constraints: {constraints}")
    if context:
        print(f"Context: {context}")
    
    results = framework.conduct_virtual_lab_research(
        research_question=args.question,
        constraints=constraints,
        context=context
    )
    
    if results.get('success', True):
        print(f"\nVirtual Lab Session: {results.get('session_id', 'N/A')}")
        if 'final_results' in results:
            final_results = results['final_results']
            if 'session_summary' in final_results:
                summary = final_results['session_summary']
                print(f"Phases Completed: {summary.get('phases_completed', 'N/A')}/7")
                print(f"Duration: {summary.get('duration', 'N/A')} seconds")
            
            if 'validated_findings' in final_results:
                findings = final_results['validated_findings']
                print(f"\nKey Findings: {len(findings)} validated findings")
                for i, finding in enumerate(findings[:3], 1):  # Show first 3
                    print(f"  {i}. {finding}")
                if len(findings) > 3:
                    print(f"  ... and {len(findings) - 3} more findings")
            
            if 'quality_assessment' in final_results:
                quality = final_results['quality_assessment']
                print(f"\nQuality Score: {quality.get('overall_score', 'N/A')}/100")
    else:
        print(f"Virtual Lab research failed: {results.get('error', 'Unknown error')}")
        return 1
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            serializable_results = make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def show_vlab_session_command(args):
    """Handle the show-vlab-session command."""
    framework = create_framework()
    
    session_data = framework.get_virtual_lab_session(args.session_id)
    
    if not session_data:
        print(f"Virtual Lab session '{args.session_id}' not found")
        return 1
    
    print(f"Virtual Lab Session: {args.session_id}")
    print("=" * 50)
    
    if 'research_question' in session_data:
        print(f"Research Question: {session_data['research_question']}")
    
    if 'constraints' in session_data:
        print(f"Constraints: {session_data['constraints']}")
    
    if 'session_summary' in session_data:
        summary = session_data['session_summary']
        print(f"Phases Completed: {summary.get('phases_completed', 'N/A')}")
        print(f"Duration: {summary.get('duration', 'N/A')} seconds")
        print(f"Status: {summary.get('status', 'N/A')}")
    
    if 'validated_findings' in session_data:
        findings = session_data['validated_findings']
        print(f"\nValidated Findings ({len(findings)}):")
        for i, finding in enumerate(findings, 1):
            print(f"  {i}. {finding}")
    
    if 'quality_assessment' in session_data:
        quality = session_data['quality_assessment']
        print(f"\nQuality Assessment:")
        print(f"  Overall Score: {quality.get('overall_score', 'N/A')}/100")
        if 'detailed_scores' in quality:
            for aspect, score in quality['detailed_scores'].items():
                print(f"  {aspect}: {score}")
    
    # Save session data if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"\nSession data saved to: {args.output}")
    
    return 0


def list_vlab_sessions_command(args):
    """Handle the list-vlab-sessions command."""
    framework = create_framework()
    
    sessions = framework.list_virtual_lab_sessions()
    
    if not sessions:
        print("No Virtual Lab sessions found")
        return 0
    
    print(f"Virtual Lab Sessions ({len(sessions)}):")
    print("=" * 40)
    
    for session_id in sessions:
        session_data = framework.get_virtual_lab_session(session_id)
        if session_data:
            question = session_data.get('research_question', 'Unknown question')
            # Truncate long questions
            if len(question) > 60:
                question = question[:57] + "..."
            
            status = session_data.get('session_summary', {}).get('status', 'Unknown')
            phases = session_data.get('session_summary', {}).get('phases_completed', 'N/A')
            
            print(f"  {session_id}")
            print(f"    Question: {question}")
            print(f"    Status: {status} | Phases: {phases}/7")
            print()
    
    return 0


def vlab_stats_command(args):
    """Handle the vlab-stats command."""
    framework = create_framework()
    
    stats = framework.get_virtual_lab_statistics()
    
    if not stats:
        print("No Virtual Lab statistics available")
        return 0
    
    print("Virtual Lab Statistics")
    print("=" * 30)
    
    if 'total_sessions' in stats:
        print(f"Total Sessions: {stats['total_sessions']}")
    
    if 'total_meetings' in stats:
        print(f"Total Meetings: {stats['total_meetings']}")
    
    if 'meeting_types' in stats:
        print("\nMeeting Types:")
        for meeting_type, count in stats['meeting_types'].items():
            print(f"  {meeting_type}: {count}")
    
    if 'phase_statistics' in stats:
        print("\nPhase Statistics:")
        for phase, phase_stats in stats['phase_statistics'].items():
            print(f"  {phase}: {phase_stats}")
    
    if 'average_session_duration' in stats:
        print(f"\nAverage Session Duration: {stats['average_session_duration']:.2f} seconds")
    
    if 'success_rate' in stats:
        print(f"Success Rate: {stats['success_rate']:.1%}")
    
    # Save stats if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.output}")
    
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
  # Run Virtual Lab research (NEW)
  %(prog)s virtual-lab-research --question "Design new computational approaches for drug discovery" --budget 50000 --timeline-weeks 12 --max-agents 6
  
  # Show Virtual Lab session details
  %(prog)s show-vlab-session --session-id vlab_session_123456
  
  # List all Virtual Lab sessions
  %(prog)s list-vlab-sessions
  
  # Get Virtual Lab statistics
  %(prog)s vlab-stats
  
  # Traditional Commands:
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
    
    # Virtual Lab research command
    vlab_parser = subparsers.add_parser('virtual-lab-research', help='Run Virtual Lab research session')
    vlab_parser.add_argument('--question', required=True, help='Research question to investigate')
    vlab_parser.add_argument('--budget', type=int, help='Budget constraint')
    vlab_parser.add_argument('--timeline-weeks', type=int, help='Timeline in weeks')
    vlab_parser.add_argument('--max-agents', type=int, help='Maximum number of agents')
    vlab_parser.add_argument('--domain', help='Research domain (e.g., computational_biology)')
    vlab_parser.add_argument('--priority', choices=['high', 'medium', 'low'], help='Research priority')
    vlab_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Show Virtual Lab session command
    show_vlab_parser = subparsers.add_parser('show-vlab-session', help='Show Virtual Lab session details')
    show_vlab_parser.add_argument('--session-id', required=True, help='Virtual Lab session ID')
    show_vlab_parser.add_argument('--output', help='Output file for session data (JSON)')
    
    # List Virtual Lab sessions command
    subparsers.add_parser('list-vlab-sessions', help='List all Virtual Lab sessions')
    
    # Virtual Lab statistics command
    vlab_stats_parser = subparsers.add_parser('vlab-stats', help='Show Virtual Lab statistics')
    vlab_stats_parser.add_argument('--output', help='Output file for statistics (JSON)')
    
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
        'create-config': create_config_command,
        'virtual-lab-research': virtual_lab_research_command,
        'show-vlab-session': show_vlab_session_command,
        'list-vlab-sessions': list_vlab_sessions_command,
        'vlab-stats': vlab_stats_command
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())