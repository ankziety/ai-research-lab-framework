#!/usr/bin/env python3
"""
Demo script for Multi-Agent AI-Powered Research Framework

This script demonstrates the new multi-agent capabilities including:
- Principal Investigator coordination
- Agent marketplace and hiring
- Multi-agent collaboration
- Vector database memory management
- Scientific critique and quality control
"""

import sys
import json
from pathlib import Path

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_framework import create_framework


def demo_basic_multi_agent_research():
    """Demonstrate basic multi-agent research workflow."""
    print("=== Multi-Agent Research Framework Demo ===\n")
    
    # Create framework instance
    framework = create_framework({
        'store_all_interactions': True,
        'enable_memory_management': True,
        'auto_critique': True
    })
    
    print("1. Framework initialized with multi-agent capabilities")
    print(f"   - Agent marketplace: {len(framework.agent_marketplace.available_agents)} agents available")
    print(f"   - Vector database ready for memory management")
    print(f"   - PI agent ready to coordinate research\n")
    
    # Research question about binocular vision and mental health
    research_question = """
    Investigate the relationship between binocular vision dysfunction and anxiety disorders.
    
    Research Focus:
    - What are the mechanisms linking binocular vision problems to anxiety?
    - How prevalent is this association in clinical populations?
    - What are the implications for treatment approaches?
    
    Consider ophthalmological, psychological, neurological, and data science perspectives.
    """
    
    print("2. Research Question:")
    print(research_question)
    print()
    
    # Conduct multi-agent research
    print("3. Initiating multi-agent research coordination...")
    research_results = framework.conduct_research(
        research_question=research_question,
        constraints={'max_agents': 4, 'focus_domains': ['ophthalmology', 'psychology']}
    )
    
    # Display results
    print(f"4. Research Session Completed!")
    print(f"   Session ID: {research_results['session_id']}")
    print(f"   Status: {research_results['status']}\n")
    
    if research_results['status'] == 'completed':
        # Show PI analysis
        analysis = research_results['coordination_result']['analysis']
        print("5. PI Analysis:")
        print(f"   - Required expertise: {analysis['required_expertise']}")
        print(f"   - Complexity score: {analysis['complexity_score']}")
        print(f"   - Agents needed: {analysis['estimated_agents_needed']}\n")
        
        # Show hired agents
        hired_agents = research_results['coordination_result']['hired_agents']
        print("6. Hired Agents:")
        for expertise, agent in hired_agents['hired_agents'].items():
            print(f"   - {expertise}: {agent.agent_id} ({agent.role})")
        print()
        
        # Show agent outputs summary
        agent_outputs = research_results['collaboration_results']['agent_outputs']
        print("7. Agent Contributions (Summary):")
        for agent_id, output in agent_outputs.items():
            if 'response' in output:
                response_preview = output['response'][:150] + "..." if len(output['response']) > 150 else output['response']
                print(f"   - {agent_id}: {response_preview}")
        print()
        
        # Show critique results
        critique = research_results['critique_result']
        print("8. Scientific Critique:")
        print(f"   - Overall score: {critique['overall_score']}/100")
        print(f"   - Key recommendations: {len(critique['recommendations'])}")
        if critique['critical_issues']:
            print(f"   - Critical issues flagged: {len(critique['critical_issues'])}")
        print()
        
        # Show synthesis
        synthesis = research_results['synthesis']
        print("9. Research Synthesis:")
        synthesis_preview = synthesis['synthesis_text'][:300] + "..." if len(synthesis['synthesis_text']) > 300 else synthesis['synthesis_text']
        print(f"   {synthesis_preview}")
        print(f"   - Confidence score: {synthesis['confidence_score']}")
        print(f"   - Key findings: {len(synthesis['key_findings'])}")
        print()
        
        # Show validated findings
        if research_results['validated_findings']:
            print("10. Validated Findings:")
            for finding_id in research_results['validated_findings']:
                print(f"    - Finding ID: {finding_id}")
        print()
    
    else:
        print(f"Research failed: {research_results.get('error', 'Unknown error')}")
    
    # Show framework statistics
    stats = framework.get_framework_statistics()
    print("11. Framework Statistics:")
    print(f"    - Total agents: {stats['agent_marketplace']['total_agents']}")
    print(f"    - Agent utilization: {stats['agent_marketplace']['utilization_rate']:.2%}")
    print(f"    - Vector DB entries: {stats['vector_database']['total_content']}")
    print(f"    - Knowledge findings: {stats['knowledge_repository']['total_findings']}")
    print()
    
    return framework, research_results


def demo_agent_marketplace():
    """Demonstrate agent marketplace features."""
    print("\n=== Agent Marketplace Demo ===\n")
    
    framework = create_framework()
    marketplace = framework.agent_marketplace
    
    # Show available agents
    print("1. Available Agents:")
    available_agents = marketplace.list_available_agents()
    for agent in available_agents:
        print(f"   - {agent['agent_id']}: {agent['role']}")
        print(f"     Expertise: {', '.join(agent['expertise'])}")
        print(f"     Performance: {agent['performance_metrics']['average_quality_score']:.2f}")
        print()
    
    # Show agent recommendations for a research problem
    research_problem = "Study the neurological basis of visual processing disorders"
    recommendations = marketplace.recommend_agents_for_research(research_problem)
    
    print("2. Agent Recommendations for Research Problem:")
    print(f"   Problem: {research_problem}")
    print("   Recommended agents:")
    for rec in recommendations:
        print(f"   - {rec['agent_id']}: {rec['role']}")
        print(f"     Relevance: {rec['relevance_score']:.2f}")
        print(f"     Recommendation score: {rec['recommendation_score']:.2f}")
        print()
    
    # Show marketplace statistics
    stats = marketplace.get_marketplace_statistics()
    print("3. Marketplace Statistics:")
    print(f"   - Total agents: {stats['total_agents']}")
    print(f"   - Available: {stats['available_agents']}")
    print(f"   - Utilization: {stats['utilization_rate']:.2%}")
    print(f"   - Health status: {stats['marketplace_health']}")
    print()


def demo_memory_management():
    """Demonstrate vector database and memory management."""
    print("\n=== Memory Management Demo ===\n")
    
    framework = create_framework()
    
    # Store some research content
    session_id = "demo_session"
    
    print("1. Storing research content in vector database...")
    
    # Store various types of content
    research_contents = [
        {
            'content': 'Binocular vision dysfunction can lead to increased anxiety and stress responses',
            'type': 'research_finding',
            'agent': 'ophthalmology_expert'
        },
        {
            'content': 'Anxiety disorders show high comorbidity with visual processing difficulties',
            'type': 'literature_insight', 
            'agent': 'psychology_expert'
        },
        {
            'content': 'Neural pathways for vision and emotion show significant overlap in brain imaging',
            'type': 'neuroscience_finding',
            'agent': 'neuroscience_expert'
        }
    ]
    
    for content_data in research_contents:
        framework.context_manager.add_to_context(
            session_id=session_id,
            content=content_data['content'],
            content_type=content_data['type'],
            agent_id=content_data['agent'],
            importance_score=0.8
        )
    
    print(f"   Stored {len(research_contents)} content items\n")
    
    # Demonstrate context retrieval
    print("2. Retrieving relevant context...")
    query = "What is the relationship between vision problems and mental health?"
    
    relevant_context = framework.context_manager.retrieve_relevant_context(
        session_id=session_id,
        query=query,
        max_items=3
    )
    
    print(f"   Query: {query}")
    print("   Relevant context found:")
    for i, context in enumerate(relevant_context, 1):
        print(f"   {i}. {context['content'][:100]}...")
        print(f"      Similarity: {context.get('similarity_score', 0):.3f}")
    print()
    
    # Show memory statistics
    context_stats = framework.context_manager.get_context_stats(session_id)
    vector_stats = framework.vector_db.get_stats()
    
    print("3. Memory Statistics:")
    print(f"   - Session items: {context_stats['total_items']}")
    print(f"   - Total vector entries: {vector_stats['total_content']}")
    print(f"   - Content by type: {vector_stats['content_by_type']}")
    print()


def main():
    """Run all demo functions."""
    print("Multi-Agent AI-Powered Research Framework Demo")
    print("=" * 50)
    
    try:
        # Run demos
        framework, research_results = demo_basic_multi_agent_research()
        demo_agent_marketplace()
        demo_memory_management()
        
        print("\n=== Demo Completed Successfully ===")
        print("\nThe multi-agent framework demonstrates:")
        print("✓ PI agent coordination and task decomposition")
        print("✓ Dynamic agent hiring based on expertise")
        print("✓ Multi-agent collaboration and cross-pollination")
        print("✓ Vector database memory management")
        print("✓ Scientific critique and quality control")
        print("✓ Knowledge repository for validated findings")
        print("✓ Backward compatibility with legacy workflows")
        
        # Cleanup
        framework.close()
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()