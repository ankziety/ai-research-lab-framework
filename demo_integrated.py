#!/usr/bin/env python3
"""
Comprehensive Demo for AI-Powered Research Framework

This script demonstrates the complete functionality of the integrated AI-powered  
research framework, showing how AI assists research workflows across any domain.
"""

import os
import json
import tempfile
from pathlib import Path

from ai_research_lab import create_framework


def demo_individual_components():
    """Demonstrate individual component functionality."""
    print("🔬 AI-Powered Research Framework - Component Demo")
    print("=" * 60)
    
    # Create framework with demo configuration
    config = {
        'output_dir': 'demo_output',
        'manuscript_dir': 'demo_manuscripts',
        'visualization_dir': 'demo_visualizations',
        'auto_visualize': True,
        'auto_critique': True
    }
    
    framework = create_framework(config)
    
    print("\n1. 🧪 Testing Experiment Runner")
    print("-" * 30)
    
    # Run a sample biology experiment
    experiment_params = {
        'treatment': 'enzyme_inhibitor_A',
        'concentration_uM': 50,
        'incubation_time_hours': 24,
        'cell_line': 'HeLa',
        'sample_size': 100
    }
    
    print(f"Running experiment with parameters: {experiment_params}")
    experiment_results = framework.run_experiment(experiment_params)
    print(f"✅ Experiment completed: {experiment_results['experiment_id']}")
    print(f"   Status: {experiment_results['status']}")
    print(f"   Computed results: {experiment_results['computed_results']}")
    
    print("\n2. 📚 Testing Literature Retriever")
    print("-" * 30)
    
    literature_query = "machine learning random forest ensemble methods"
    print(f"Searching literature for: '{literature_query}'")
    literature_results = framework.retrieve_literature(literature_query, max_results=5)
    print(f"✅ Found {len(literature_results)} literature results")
    
    for i, paper in enumerate(literature_results[:3], 1):
        print(f"   {i}. {paper.get('title', 'Unknown Title')} ({paper.get('publication_year', 'Unknown Year')})")
    
    print("\n3. 📝 Testing Manuscript Drafter")
    print("-" * 30)
    
    # Create manuscript context
    manuscript_context = {
        'study_type': 'Comparative Analysis',
        'subject': 'Machine Learning Algorithms',
        'objective': 'Compare random forest performance across datasets',
        'background': 'Random forests are ensemble methods that combine multiple decision trees',
        'significance': 'Understanding ensemble performance is crucial for model selection',
        'methods': 'Cross-validation with multiple performance metrics',
        'materials': ['scikit-learn', 'pandas', 'numpy'],
        'procedures': [
            'Load and preprocess datasets',
            'Train random forest models',
            'Evaluate using cross-validation',
            'Analyze performance metrics'
        ],
        'conclusion': 'Random forest demonstrates robust performance across datasets',
        'limitations': ['Limited to tabular data', 'Hyperparameter sensitivity not fully explored'],
        'future_work': ['Explore neural network comparisons', 'Test on larger datasets']
    }
    
    print("Drafting manuscript from experiment results...")
    manuscript = framework.draft_manuscript([experiment_results], manuscript_context)
    print("✅ Manuscript drafted successfully")
    print(f"   Length: {len(manuscript)} characters")
    print(f"   Sections: {'✓' if '## Abstract' in manuscript else '✗'} Abstract, "
          f"{'✓' if '## Introduction' in manuscript else '✗'} Introduction, "
          f"{'✓' if '## Methods' in manuscript else '✗'} Methods, "
          f"{'✓' if '## Results' in manuscript else '✗'} Results")
    
    print("\n4. 🔍 Testing Critic")
    print("-" * 30)
    
    print("Critiquing the generated manuscript...")
    critique = framework.critique_output(manuscript)
    print(f"✅ Critique completed")
    print(f"   Overall Score: {critique['overall_score']}/100")
    print(f"   Strengths: {len(critique['strengths'])}")
    print(f"   Weaknesses: {len(critique['weaknesses'])}")
    print(f"   Suggestions: {len(critique['suggestions'])}")
    
    if critique['strengths']:
        print(f"   Top strength: {critique['strengths'][0]}")
    if critique['suggestions']:
        print(f"   Top suggestion: {critique['suggestions'][0]}")
    
    print("\n5. 👥 Testing Specialist Registry")
    print("-" * 30)
    
    specialists = framework.list_specialists()
    print(f"✅ Available specialists: {', '.join(specialists)}")
    
    # Test getting a specialist
    critic_specialist = framework.get_specialist('critic')
    print(f"   Critic specialist callable: {callable(critic_specialist)}")
    
    return framework, experiment_results, manuscript_context


def demo_complete_workflow():
    """Demonstrate complete research workflow."""
    print("\n\n🚀 Complete Research Workflow Demo")
    print("=" * 60)
    
    # Create framework
    framework = create_framework({
        'output_dir': 'workflow_output',
        'auto_visualize': False,  # Skip visualization for demo
        'auto_critique': True
    })
    
    # Define workflow parameters for a chemistry research study
    experiment_params = {
        'reaction_type': 'catalytic_hydrogenation',
        'catalyst': 'palladium_on_carbon',
        'substrate': 'benzene_derivative',
        'temperature_C': 25,
        'pressure_atm': 1.5,
        'reaction_time_hours': 4,
        'solvent': 'ethanol',
        'yield_target': 0.85,
        'selectivity_target': 0.95
    }
    
    manuscript_context = {
        'study_type': 'Experimental Study',
        'subject': 'Catalytic Hydrogenation Optimization',
        'objective': 'Optimize palladium-catalyzed hydrogenation for improved yield and selectivity',
        'background': 'Catalytic hydrogenation is essential for pharmaceutical synthesis but requires optimization',
        'significance': 'Improved catalytic processes reduce waste and enhance sustainability',
        'methods': 'Systematic catalyst screening with reaction condition optimization',
        'conclusion': 'Optimized conditions achieved high yield and excellent selectivity',
        'materials': ['Palladium on carbon', 'Ethanol', 'Hydrogen gas', 'Benzene derivative'],
        'procedures': [
            'Prepare catalyst suspension in solvent',
            'Add substrate and purge with hydrogen',
            'Monitor reaction progress by GC-MS',
            'Analyze product yield and selectivity',
            'Characterize reaction kinetics'
        ],
        'limitations': [
            'Limited to single substrate class',
            'Ambient pressure conditions only',
            'Short reaction time optimization'
        ],
        'future_work': [
            'Expand substrate scope',
            'Implement automated architecture search',
            'Compare with state-of-the-art models'
        ]
    }
    
    literature_query = 'palladium catalyzed hydrogenation reaction optimization'
    
    print("Starting complete workflow...")
    print(f"📊 Experiment: {experiment_params['reaction_type']} using {experiment_params['catalyst']}")
    print(f"📚 Literature query: '{literature_query}'")
    print(f"📝 Manuscript topic: {manuscript_context['subject']}")
    
    print("\nExecuting workflow...")
    workflow_results = framework.run_complete_workflow(
        experiment_params=experiment_params,
        manuscript_context=manuscript_context,
        literature_query=literature_query
    )
    
    print(f"\n✅ Workflow completed: {workflow_results['workflow_id']}")
    print(f"   Status: {workflow_results['status']}")
    
    if workflow_results['status'] == 'completed':
        print(f"\n📊 Experiment Results:")
        exp_results = workflow_results['experiment']
        print(f"   Experiment ID: {exp_results['experiment_id']}")
        print(f"   Computed Results: {exp_results['computed_results']}")
        
        print(f"\n📚 Literature Results:")
        literature = workflow_results.get('literature', [])
        print(f"   Papers found: {len(literature)}")
        
        print(f"\n📝 Manuscript:")
        manuscript_info = workflow_results['manuscript']
        print(f"   Saved to: {manuscript_info['path']}")
        print(f"   Length: {len(manuscript_info['content'])} characters")
        
        print(f"\n🔍 Critique:")
        critique = workflow_results['critique']
        print(f"   Overall Score: {critique['overall_score']}/100")
        print(f"   Strengths: {len(critique['strengths'])}")
        print(f"   Weaknesses: {len(critique['weaknesses'])}")
        print(f"   Suggestions: {len(critique['suggestions'])}")
        
        # Show a sample of the manuscript
        manuscript_content = manuscript_info['content']
        lines = manuscript_content.split('\n')
        print(f"\n📄 Manuscript Preview (first 10 lines):")
        for i, line in enumerate(lines[:10], 1):
            print(f"   {i:2d}: {line}")
        if len(lines) > 10:
            print(f"   ... ({len(lines) - 10} more lines)")
    
    return workflow_results


def demo_configuration_management():
    """Demonstrate configuration management features."""
    print("\n\n⚙️  Configuration Management Demo")
    print("=" * 60)
    
    framework = create_framework()
    
    # Create a sample workflow configuration
    workflow_config = {
        'experiment_params': {
            'algorithm': 'support_vector_machine',
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'dataset': 'iris'
        },
        'manuscript_context': {
            'objective': 'Evaluate SVM performance on iris dataset',
            'methods': 'Support Vector Machine with RBF kernel',
            'conclusion': 'SVM achieves high accuracy on iris classification'
        },
        'literature_query': 'support vector machine classification iris dataset'
    }
    
    print("Saving workflow configuration...")
    config_path = framework.save_workflow_config(workflow_config, 'svm_iris_study')
    print(f"✅ Configuration saved to: {config_path}")
    
    print("\nLoading workflow configuration...")
    loaded_config = framework.load_workflow_config('svm_iris_study')
    print(f"✅ Configuration loaded successfully")
    print(f"   Algorithm: {loaded_config['experiment_params']['algorithm']}")
    print(f"   Objective: {loaded_config['manuscript_context']['objective']}")
    
    return config_path


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n\n🛡️  Error Handling Demo")
    print("=" * 60)
    
    framework = create_framework()
    
    print("1. Testing invalid experiment parameters...")
    try:
        # This should fail due to non-serializable object
        framework.run_experiment({'invalid_param': object()})
        print("❌ Expected error not raised")
    except TypeError as e:
        print(f"✅ Correctly caught TypeError: {e}")
    
    print("\n2. Testing invalid manuscript inputs...")
    try:
        framework.draft_manuscript("not a list", {})
        print("❌ Expected error not raised")
    except ValueError as e:
        print(f"✅ Correctly caught ValueError: {e}")
    
    print("\n3. Testing invalid critic input...")
    try:
        framework.critique_output(None)
        print("❌ Expected error not raised")
    except ValueError as e:
        print(f"✅ Correctly caught ValueError: {e}")
    
    print("\n4. Testing missing specialist...")
    try:
        framework.get_specialist('nonexistent_specialist')
        print("❌ Expected error not raised")
    except KeyError as e:
        print(f"✅ Correctly caught KeyError: {e}")


def main():
    """Run complete framework demonstration."""
    print("🤖 AI Research Lab Framework - Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the integrated AI research lab framework")
    print("with all components working together in research workflows.")
    print("=" * 80)
    
    try:
        # Demo individual components
        framework, exp_results, context = demo_individual_components()
        
        # Demo complete workflow
        workflow_results = demo_complete_workflow()
        
        # Demo configuration management
        config_path = demo_configuration_management()
        
        # Demo error handling
        demo_error_handling()
        
        print("\n\n🎉 Demo Completed Successfully!")
        print("=" * 60)
        print("Summary of what was demonstrated:")
        print("✅ Individual component functionality")
        print("✅ Complete research workflow orchestration")
        print("✅ Configuration management and persistence")
        print("✅ Error handling and validation")
        print("✅ Integration between all framework components")
        
        print(f"\nGenerated files:")
        print(f"📊 Experiment database: experiments/experiments.db")
        print(f"📝 Workflow manuscript: {workflow_results['manuscript']['path']}")
        print(f"⚙️  Sample configuration: {config_path}")
        
        print(f"\nFramework features demonstrated:")
        print(f"🧪 Experiment execution and tracking")
        print(f"📚 Literature retrieval (stubbed)")
        print(f"📝 Manuscript drafting from results")
        print(f"🔍 Automated critique and feedback")
        print(f"👥 Specialist registry and management")
        print(f"🔄 End-to-end workflow orchestration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)