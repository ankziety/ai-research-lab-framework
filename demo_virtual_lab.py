"""
Virtual Lab Demo - Research Framework Demonstration

This demo demonstrates the Virtual Lab methodology implemented in the AI Research
Lab Framework, inspired by the paper "The Virtual Lab of AI agents designs new 
SARS-CoV-2 nanobodies" by Swanson et al.

The demo demonstrates:
- Meeting-based research coordination
- Structured research phases
- Cross-agent collaboration and critique
- Iterative refinement workflows
- Scientific critique integration
"""

import logging
import time
from multi_agent_framework import MultiAgentResearchFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def demo_virtual_lab_research():
    """Demonstrate Virtual Lab research methodology."""
    
    print("🧪 Virtual Lab Research Framework Demo")
    print("=" * 50)
    print()
    
    # Create framework with mock configuration (no API keys needed for demo)
    config = {
        'experiment_db_path': 'experiments/demo_experiments.db',
        'output_dir': 'output',
        'manuscript_dir': 'manuscripts', 
        'visualization_dir': 'visualizations',
        'auto_critique': True,
        'auto_visualize': False,
        'store_all_interactions': True,
        'enable_memory_management': True,
        
        # Virtual Lab specific configuration
        'max_agents_per_research': 6,
        'agent_timeout': 1800,
        'agent_memory_limit': 1000,
        'max_context_length': 10000
    }
    
    try:
        # Initialize framework
        print("🔧 Initializing Multi-Agent Research Framework with Virtual Lab...")
        framework = MultiAgentResearchFramework(config)
        print("✅ Framework initialized successfully!")
        print()
        
        # Example research questions for different domains
        research_examples = [
            {
                'question': "Design new computational approaches for drug discovery targeting viral proteins",
                'constraints': {'budget': 50000, 'timeline_weeks': 12, 'team_size_max': 6},
                'context': {'domain': 'computational_biology', 'priority': 'high'}
            },
            {
                'question': "Develop machine learning models for predicting protein-protein interactions in cancer",
                'constraints': {'budget': 75000, 'timeline_weeks': 16, 'team_size_max': 5},
                'context': {'domain': 'oncology_informatics', 'priority': 'medium'}
            },
            {
                'question': "Create sustainable catalytic processes for green chemistry applications",
                'constraints': {'budget': 60000, 'timeline_weeks': 20, 'team_size_max': 4},
                'context': {'domain': 'green_chemistry', 'priority': 'high'}
            }
        ]
        
        # Select research example for demo
        selected_example = research_examples[0]  # Drug discovery example
        
        print(f"🔬 Research Question: {selected_example['question']}")
        print(f"💰 Budget: ${selected_example['constraints']['budget']:,}")
        print(f"⏰ Timeline: {selected_example['constraints']['timeline_weeks']} weeks")
        print(f"👥 Max Team Size: {selected_example['constraints']['team_size_max']} agents")
        print()
        
        # Conduct Virtual Lab research
        print("🚀 Starting Virtual Lab Research Session...")
        print("   This will demonstrate the 7-phase Virtual Lab methodology:")
        print("   1. Team Selection")
        print("   2. Project Specification") 
        print("   3. Tools Selection")
        print("   4. Tools Implementation")
        print("   5. Workflow Design")
        print("   6. Execution")
        print("   7. Synthesis")
        print()
        
        start_time = time.time()
        
        vlab_results = framework.conduct_virtual_lab_research(
            research_question=selected_example['question'],
            constraints=selected_example['constraints'],
            context=selected_example['context']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Research session completed in {duration:.1f} seconds")
        print()
        
        # Display results summary
        if vlab_results.get('success', True) and vlab_results.get('status') == 'completed':
            print("✅ Virtual Lab Research Session Completed Successfully!")
            print("=" * 50)
            
            # Session summary
            session_summary = vlab_results.get('final_results', {}).get('session_summary', {})
            print(f"📊 Session ID: {session_summary.get('session_id', 'Unknown')}")
            print(f"📈 Phases Completed: {session_summary.get('phases_completed', 0)}/7")
            print(f"🏢 Total Meetings: {session_summary.get('total_meetings', 0)}")
            print(f"⏰ Total Duration: {session_summary.get('duration', 0):.1f} seconds")
            print()
            
            # Phase results summary
            phases = vlab_results.get('phases', {})
            print("📋 Phase Results Summary:")
            for phase_name, phase_result in phases.items():
                status = "✅" if phase_result.get('success', False) else "❌"
                print(f"   {status} {phase_name.replace('_', ' ').title()}")
                
                # Show key decisions for each phase
                decisions = phase_result.get('decisions', [])
                if decisions:
                    for decision in decisions[:2]:  # Show first 2 decisions
                        print(f"      • {decision}")
            print()
            
            # Key outcomes
            key_outcomes = vlab_results.get('final_results', {}).get('key_outcomes', {})
            if key_outcomes:
                print("🎯 Key Outcomes by Phase:")
                for phase, outcomes in key_outcomes.items():
                    if outcomes:
                        print(f"   • {phase.replace('_', ' ').title()}:")
                        for outcome in outcomes[:2]:  # Show first 2 outcomes
                            print(f"     - {outcome}")
                print()
            
            # Validated findings
            validated_findings = vlab_results.get('final_results', {}).get('validated_findings', [])
            if validated_findings:
                print("🔬 Validated Research Findings:")
                for i, finding in enumerate(validated_findings[:3], 1):
                    print(f"   {i}. {finding}")
                print()
            
            # Research recommendations
            recommendations = vlab_results.get('final_results', {}).get('research_recommendations', [])
            if recommendations:
                print("💡 Research Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec}")
                print()
            
            # Quality assessment
            quality_assessment = vlab_results.get('final_results', {}).get('quality_assessment', {})
            if quality_assessment:
                print("📊 Quality Assessment:")
                print(f"   Overall Score: {quality_assessment.get('overall_score', 0)}/100")
                print(f"   Summary: {quality_assessment.get('summary', 'No summary available')}")
                
                strengths = quality_assessment.get('strengths', [])
                if strengths:
                    print("   Strengths:")
                    for strength in strengths[:2]:
                        print(f"     + {strength}")
                
                weaknesses = quality_assessment.get('weaknesses', [])
                if weaknesses:
                    print("   Areas for Improvement:")
                    for weakness in weaknesses[:2]:
                        print(f"     - {weakness}")
                print()
        
        else:
            print("❌ Virtual Lab Research Session Failed")
            print(f"Error: {vlab_results.get('error', 'Unknown error')}")
            print()
        
        # Show Virtual Lab statistics
        print("📈 Virtual Lab Meeting Statistics:")
        vlab_stats = framework.get_virtual_lab_statistics()
        if vlab_stats:
            print(f"   Total Meetings Conducted: {vlab_stats.get('total_meetings', 0)}")
            print(f"   Successful Meetings: {vlab_stats.get('successful_meetings', 0)}")
            print(f"   Average Meeting Duration: {vlab_stats.get('average_duration', 0):.1f} seconds")
            
            meeting_types = vlab_stats.get('meeting_types', {})
            if meeting_types:
                print("   Meeting Types:")
                for meeting_type, count in meeting_types.items():
                    print(f"     • {meeting_type.replace('_', ' ').title()}: {count}")
            
            phases_covered = vlab_stats.get('phases_covered', {})
            if phases_covered:
                print("   Research Phases:")
                for phase, count in phases_covered.items():
                    print(f"     • {phase.replace('_', ' ').title()}: {count}")
        print()
        
        # Framework statistics
        print("🏗️  Framework Statistics:")
        framework_stats = framework.get_framework_statistics()
        
        # Agent marketplace stats
        marketplace_stats = framework_stats.get('agent_marketplace', {})
        if marketplace_stats:
            print(f"   Available Agents: {marketplace_stats.get('total_agents', 0)}")
            print(f"   Active Agents: {marketplace_stats.get('active_agents', 0)}")
            print(f"   Expertise Domains: {marketplace_stats.get('unique_expertise_areas', 0)}")
        
        # Knowledge repository stats
        knowledge_stats = framework_stats.get('knowledge_repository', {})
        if knowledge_stats:
            print(f"   Validated Findings: {knowledge_stats.get('total_findings', 0)}")
            print(f"   Research Domains: {knowledge_stats.get('unique_domains', 0)}")
        
        print()
        print("🎉 Virtual Lab Demo Completed!")
        print()
        print("Features Demonstrated:")
        print("• Meeting-based research coordination")
        print("• Structured 7-phase research methodology")
        print("• Multi-agent collaboration and expertise")
        print("• Cross-agent interaction and critique")
        print("• Scientific quality assessment")
        print("• Automated workflow design and execution")
        print("• Comprehensive results synthesis")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 50)
    print("Virtual Lab Demo Complete")

def demo_comparison():
    """Demonstrate the difference between traditional and Virtual Lab approaches."""
    
    print("🔄 Comparing Traditional vs Virtual Lab Approaches")
    print("=" * 50)
    
    config = {
        'store_all_interactions': False,  # Simplified for comparison
        'enable_memory_management': False
    }
    
    try:
        framework = MultiAgentResearchFramework(config)
        
        research_question = "Analyze the potential of CRISPR gene editing for treating genetic diseases"
        constraints = {'budget': 40000, 'timeline_weeks': 8}
        
        print("1️⃣  Traditional Multi-Agent Approach:")
        print("   • Simple task decomposition")
        print("   • Individual agent responses")
        print("   • Basic synthesis")
        print()
        
        start_time = time.time()
        traditional_results = framework.conduct_research(
            research_question=research_question,
            constraints=constraints
        )
        traditional_duration = time.time() - start_time
        
        print(f"   ⏱️  Completed in {traditional_duration:.1f} seconds")
        print(f"   📊 Status: {traditional_results.get('status', 'unknown')}")
        print()
        
        print("2️⃣  Virtual Lab Approach:")
        print("   • Structured meeting-based collaboration")
        print("   • 7-phase research methodology")
        print("   • Cross-agent interaction and critique")
        print("   • Iterative refinement")
        print()
        
        start_time = time.time()
        vlab_results = framework.conduct_virtual_lab_research(
            research_question=research_question,
            constraints=constraints
        )
        vlab_duration = time.time() - start_time
        
        print(f"   ⏱️  Completed in {vlab_duration:.1f} seconds")
        print(f"   📊 Status: {vlab_results.get('status', 'unknown')}")
        print()
        
        # Comparison summary
        print("📊 Comparison Summary:")
        print(f"   Traditional: {traditional_duration:.1f}s | Virtual Lab: {vlab_duration:.1f}s")
        
        traditional_agents = len(traditional_results.get('coordination_result', {}).get('hired_agents', {}).get('hired_agents', {}))
        vlab_phases = len(vlab_results.get('phases', {}))
        
        print(f"   Traditional agents used: {traditional_agents}")
        print(f"   Virtual Lab phases completed: {vlab_phases}")
        print()
        
        print("📋 Virtual Lab Method Summary:")
        print("   • Structured 7-phase research process")
        print("   • Meeting-based agent coordination") 
        print("   • Integrated critique and quality control")
        print("   • Cross-agent collaboration features")
        
    except Exception as e:
        print(f"❌ Comparison demo failed: {e}")

if __name__ == "__main__":
    # Run the main Virtual Lab demo
    demo_virtual_lab_research()
    
    print("\n" + "=" * 50)
    print()
    
    # Run the comparison demo
    demo_comparison()