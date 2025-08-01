"""
Comprehensive AI Research Lab Framework Demo

This unified demo showcases the complete multi-agent research framework
with tool integration, dynamic agent creation, and collaborative research.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the framework to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_agent_framework import MultiAgentResearchFramework


def run_comprehensive_demo():
    """Run a comprehensive demonstration of the AI Research Lab Framework."""
    
    print("🧬 AI Research Lab Framework - Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases the complete multi-agent research system")
    print("with dynamic agent creation, tool integration, and collaboration.")
    print()
    
    # Configuration with optional API keys
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'), 
        'literature_api_key': os.getenv('LITERATURE_API_KEY'),
        'default_llm_provider': 'openai',
        'default_model': 'gpt-4o',  # Updated to latest model
        
        # Framework configuration
        'max_agents_per_research': 6,
        'agent_timeout': 1800,  # 30 minutes instead of 5
        'enable_tool_integration': True,
        'enable_vector_database': True,
        'enable_collaboration': True,
        
        # Output configuration
        'output_dir': 'output',
        'visualization_dir': 'visualizations',
        'manuscript_dir': 'manuscripts',
        'session_dir': 'sessions'
    }
    
    # Create the framework
    print("🚀 Initializing Multi-Agent Research Framework...")
    framework = MultiAgentResearchFramework(config)
    
    # Demo scenarios
    scenarios = [
        {
            'name': 'Interdisciplinary Climate Research',
            'research_question': 'Investigate the impact of microplastics on marine ecosystems and human health',
            'description': 'Cross-domain research requiring environmental science, biology, and health expertise'
        },
        {
            'name': 'AI-Powered Drug Discovery',
            'research_question': 'Develop machine learning models for predicting drug-target interactions in cancer therapy',
            'description': 'Computational research combining AI, chemistry, and oncology'
        },
        {
            'name': 'Quantum Computing Applications',
            'research_question': 'Explore quantum algorithms for optimization problems in supply chain management',
            'description': 'Theoretical research bridging quantum physics and operations research'
        }
    ]
    
    print("\n📋 Available Research Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Research Question: {scenario['research_question']}")
        print(f"   Description: {scenario['description']}")
        print()
    
    # Interactive scenario selection
    try:
        choice = input("Select a scenario (1-3) or press Enter for all scenarios: ").strip()
        
        if choice == "":
            selected_scenarios = scenarios
        elif choice in ['1', '2', '3']:
            selected_scenarios = [scenarios[int(choice) - 1]]
        else:
            print("Invalid choice. Running all scenarios.")
            selected_scenarios = scenarios
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        return
    
    # Run selected scenarios
    for i, scenario in enumerate(selected_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"🔬 SCENARIO {i}: {scenario['name']}")
        print(f"{'='*80}")
        
        run_research_scenario(framework, scenario)
        
        if i < len(selected_scenarios):
            input("\nPress Enter to continue to next scenario...")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 DEMO SUMMARY")
    print("="*80)
    
    # Get framework statistics
    stats = framework.get_framework_statistics()
    print_framework_statistics(stats)
    
    print("\n✅ Demo completed successfully!")
    print("Check the output directories for generated files:")
    print(f"  - Manuscripts: {config['manuscript_dir']}/")
    print(f"  - Visualizations: {config['visualization_dir']}/")
    print(f"  - Session logs: {config['session_dir']}/")


def run_research_scenario(framework: MultiAgentResearchFramework, scenario: Dict[str, Any]):
    """Run a complete research scenario demonstration."""
    
    research_question = scenario['research_question']
    
    print(f"\n🎯 Research Question: {research_question}")
    print(f"📝 Description: {scenario['description']}")
    print()
    
    # Step 1: Conduct Multi-Agent Research
    print("🤖 Step 1: Multi-Agent Research Coordination")
    print("-" * 50)
    
    try:
        research_results = framework.conduct_research(
            research_question=research_question,
            constraints={
                'max_agents': 5,
                'research_depth': 'comprehensive',
                'include_literature_review': True,
                'generate_visualizations': True,
                'require_statistical_analysis': True
            }
        )
        
        print("✅ Multi-agent research completed!")
        print(f"   - Principal Investigator: {research_results.get('pi_agent', 'AI Research Coordinator')}")
        print(f"   - Expert agents hired: {len(research_results.get('hired_agents', []))}")
        print(f"   - Research tasks completed: {len(research_results.get('completed_tasks', []))}")
        print(f"   - Knowledge artifacts created: {len(research_results.get('knowledge_artifacts', []))}")
        
        # Display hired agents
        if research_results.get('hired_agents'):
            print("   - Expert agents involved:")
            for agent in research_results['hired_agents']:
                print(f"     • {agent.get('name', 'Unknown Agent')} ({agent.get('expertise', 'General')})")
        
    except Exception as e:
        print(f"❌ Multi-agent research failed: {str(e)}")
        research_results = {}
    
    # Step 2: Tool Integration Demonstration
    print(f"\n🛠️ Step 2: Research Tool Integration")
    print("-" * 50)
    
    try:
        # Demonstrate tool discovery and usage
        available_tools = framework.discover_research_tools(research_question)
        print(f"✅ Discovered {len(available_tools)} relevant research tools:")
        
        for tool in available_tools[:5]:  # Show top 5
            print(f"   - {tool.get('name', 'Unknown Tool')}: {tool.get('description', 'No description')[:60]}...")
        
        # Request and use specific tools
        if available_tools:
            tool_results = framework.use_research_tool(
                tool_id=available_tools[0]['tool_id'],
                task={
                    'type': 'comprehensive_analysis',
                    'research_question': research_question,
                    'data_requirements': ['experimental', 'literature', 'statistical']
                }
            )
            
            if tool_results.get('success'):
                print(f"✅ Tool execution successful: {tool_results.get('summary', 'Analysis completed')}")
            else:
                print(f"⚠️ Tool execution had issues: {tool_results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Tool integration failed: {str(e)}")
    
    # Step 3: Literature Analysis
    print(f"\n📚 Step 3: Comprehensive Literature Analysis")
    print("-" * 50)
    
    try:
        literature_results = framework.retrieve_literature(
            query=research_question,
            max_results=15,
            include_analysis=True
        )
        
        print(f"✅ Literature analysis completed!")
        print(f"   - Papers found: {len(literature_results.get('papers', []))}")
        print(f"   - Sources searched: {', '.join(literature_results.get('sources_searched', []))}")
        print(f"   - Average relevance score: {literature_results.get('avg_relevance', 0):.2f}")
        
        # Show top papers
        top_papers = literature_results.get('papers', [])[:3]
        if top_papers:
            print("   - Top relevant papers:")
            for i, paper in enumerate(top_papers, 1):
                print(f"     {i}. {paper.get('title', 'Unknown Title')[:60]}...")
                print(f"        Authors: {', '.join(paper.get('authors', [])[:2])}...")
                print(f"        Relevance: {paper.get('relevance_score', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Literature analysis failed: {str(e)}")
        literature_results = {}
    
    # Step 4: Experimental Design and Execution
    print(f"\n🧪 Step 4: Experimental Design and Data Analysis")
    print("-" * 50)
    
    try:
        # Design experiment
        experiment_params = {
            'research_question': research_question,
            'study_type': 'observational' if 'impact' in research_question.lower() else 'experimental',
            'sample_size': 200,
            'duration_weeks': 12,
            'variables': {
                'independent': ['treatment_type', 'exposure_level'],
                'dependent': ['outcome_measure', 'response_rate'],
                'control': ['age', 'gender', 'baseline_measure']
            }
        }
        
        experiment_results = framework.run_experiment(experiment_params)
        
        print(f"✅ Experimental analysis completed!")
        print(f"   - Experiment ID: {experiment_results.get('experiment_id', 'N/A')}")
        print(f"   - Status: {experiment_results.get('status', 'Unknown')}")
        print(f"   - Sample size: {experiment_results.get('metadata', {}).get('sample_size', 0)}")
        print(f"   - Statistical power: {experiment_results.get('metadata', {}).get('statistical_power', 0):.2f}")
        
        # Show key results
        if experiment_results.get('computed_results'):
            results = experiment_results['computed_results']
            print(f"   - Primary outcome: {results.get('primary_outcome', 'Not available')}")
            print(f"   - Effect size: {results.get('effect_size', 0):.3f}")
            print(f"   - P-value: {results.get('p_value', 1.0):.3f}")
        
    except Exception as e:
        print(f"❌ Experimental analysis failed: {str(e)}")
        experiment_results = {}
    
    # Step 5: Manuscript Generation
    print(f"\n📄 Step 5: Scientific Manuscript Generation")
    print("-" * 50)
    
    try:
        manuscript_context = {
            'research_question': research_question,
            'study_type': 'interdisciplinary_research',
            'key_findings': research_results.get('key_findings', []),
            'methodology': 'multi-agent_research_framework',
            'statistical_results': experiment_results.get('computed_results', {}),
            'literature_context': literature_results.get('summary', {})
        }
        
        manuscript_results = framework.run_complete_workflow(
            experiment_params=experiment_params,
            manuscript_context=manuscript_context
        )
        
        print(f"✅ Manuscript generation completed!")
        print(f"   - Manuscript ID: {manuscript_results.get('manuscript_id', 'N/A')}")
        print(f"   - Status: {manuscript_results.get('manuscript_status', 'Unknown')}")
        print(f"   - Word count: ~{manuscript_results.get('manuscript_stats', {}).get('word_count', 0)}")
        print(f"   - Sections generated: {len(manuscript_results.get('manuscript_sections', []))}")
        
        # Show file location
        if manuscript_results.get('manuscript_file'):
            print(f"   - File saved: {manuscript_results['manuscript_file']}")
        
    except Exception as e:
        print(f"❌ Manuscript generation failed: {str(e)}")
    
    # Step 6: Quality Assessment and Critique
    print(f"\n🔍 Step 6: Scientific Quality Assessment")
    print("-" * 50)
    
    try:
        # Combine all results for critique
        research_output = {
            'research_question': research_question,
            'methodology': experiment_params,
            'results': experiment_results.get('computed_results', {}),
            'literature_support': literature_results.get('papers', []),
            'agent_analysis': research_results.get('analysis_summary', {})
        }
        
        critique_results = framework.critique_research(research_output)
        
        print(f"✅ Quality assessment completed!")
        print(f"   - Overall quality score: {critique_results.get('overall_score', 0):.1f}/10")
        print(f"   - Methodology score: {critique_results.get('methodology_score', 0):.1f}/10")
        print(f"   - Evidence strength: {critique_results.get('evidence_score', 0):.1f}/10")
        print(f"   - Logical consistency: {critique_results.get('logic_score', 0):.1f}/10")
        
        # Show key recommendations
        recommendations = critique_results.get('recommendations', [])
        if recommendations:
            print("   - Key recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec}")
        
        # Show identified issues
        issues = critique_results.get('identified_issues', [])
        if issues:
            print("   - Issues identified:")
            for i, issue in enumerate(issues[:2], 1):
                print(f"     {i}. {issue}")
        
    except Exception as e:
        print(f"❌ Quality assessment failed: {str(e)}")
    
    print(f"\n🎉 Research scenario '{scenario['name']}' completed successfully!")


def print_framework_statistics(stats: Dict[str, Any]):
    """Print comprehensive framework statistics."""
    
    print("Multi-Agent System Performance:")
    agent_stats = stats.get('agent_statistics', {})
    print(f"  - Total agents created: {agent_stats.get('total_agents', 0)}")
    print(f"  - Active agents: {agent_stats.get('active_agents', 0)}")
    print(f"  - Agent success rate: {agent_stats.get('success_rate', 0):.1%}")
    print(f"  - Most used agent type: {agent_stats.get('most_used_type', 'N/A')}")
    
    print("\nTool Integration Performance:")
    tool_stats = stats.get('tool_statistics', {})
    print(f"  - Tools available: {tool_stats.get('total_tools', 0)}")
    print(f"  - Tools used in demo: {tool_stats.get('tools_used', 0)}")
    print(f"  - Tool success rate: {tool_stats.get('tool_success_rate', 0):.1%}")
    print(f"  - Most popular tool: {tool_stats.get('most_used_tool', 'N/A')}")
    
    print("\nResearch Output Quality:")
    quality_stats = stats.get('quality_metrics', {})
    print(f"  - Average research quality: {quality_stats.get('avg_quality_score', 0):.1f}/10")
    print(f"  - Papers analyzed: {quality_stats.get('papers_processed', 0)}")
    print(f"  - Experiments designed: {quality_stats.get('experiments_designed', 0)}")
    print(f"  - Manuscripts generated: {quality_stats.get('manuscripts_created', 0)}")
    
    print("\nSystem Performance:")
    performance_stats = stats.get('performance_metrics', {})
    print(f"  - Total execution time: {performance_stats.get('total_time', 0):.1f} seconds")
    print(f"  - Memory usage peak: {performance_stats.get('peak_memory', 0):.1f} MB")
    print(f"  - API calls made: {performance_stats.get('api_calls', 0)}")
    print(f"  - Success rate: {performance_stats.get('overall_success_rate', 0):.1%}")


def main():
    """Main entry point for the demo."""
    try:
        run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()