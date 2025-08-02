#!/usr/bin/env python3
"""
Comprehensive test to verify JSON serialization fixes work together.
"""

import json
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_framework import MultiAgentResearchFramework, make_json_serializable
from agents.agent_marketplace import AgentMarketplace
from agents.principal_investigator import PrincipalInvestigatorAgent
from agents.scientific_critic import ScientificCriticAgent


def test_virtual_lab_serialization():
    """Test Virtual Lab research session serialization."""
    print("Testing Virtual Lab research session serialization...")
    
    try:
        # Create framework
        framework = MultiAgentResearchFramework({
            'openai_api_key': None,  # Use mock responses
            'store_all_interactions': True
        })
        
        # Conduct Virtual Lab research
        research_question = "How can we improve machine learning model interpretability?"
        constraints = {'time_limit': 300, 'max_agents': 5}
        
        results = framework.conduct_virtual_lab_research(
            research_question=research_question,
            constraints=constraints
        )
        
        # Test JSON serialization of results
        serializable_results = make_json_serializable(results)
        json.dumps(serializable_results)
        print("✓ Virtual Lab research results serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Virtual Lab research serialization failed: {e}")
        return False


def test_traditional_research_serialization():
    """Test traditional research session serialization."""
    print("\nTesting traditional research session serialization...")
    
    try:
        # Create framework
        framework = MultiAgentResearchFramework({
            'openai_api_key': None,  # Use mock responses
            'store_all_interactions': True
        })
        
        # Conduct traditional research
        research_question = "What are the latest developments in natural language processing?"
        constraints = {'time_limit': 300, 'max_agents': 5}
        
        results = framework.conduct_research(
            research_question=research_question,
            constraints=constraints
        )
        
        # Test JSON serialization of results
        serializable_results = make_json_serializable(results)
        json.dumps(serializable_results)
        print("✓ Traditional research results serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Traditional research serialization failed: {e}")
        return False


def test_agent_marketplace_serialization():
    """Test AgentMarketplace serialization."""
    print("\nTesting AgentMarketplace serialization...")
    
    try:
        marketplace = AgentMarketplace()
        
        # Test hiring agents
        pi_agent = PrincipalInvestigatorAgent("test_pi")
        hiring_result = pi_agent.hire_agents(
            marketplace, 
            ['data_science', 'literature_research'], 
            {'max_agents': 3}
        )
        
        # Test JSON serialization of hiring result
        serializable_hiring = make_json_serializable(hiring_result)
        json.dumps(serializable_hiring)
        print("✓ AgentMarketplace hiring results serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ AgentMarketplace serialization failed: {e}")
        return False


def test_session_data_serialization():
    """Test session data serialization."""
    print("\nTesting session data serialization...")
    
    try:
        # Create framework
        framework = MultiAgentResearchFramework({
            'openai_api_key': None,
            'store_all_interactions': True
        })
        
        # Get session data
        sessions = framework.list_virtual_lab_sessions()
        if sessions:
            session_id = sessions[0]
            session_data = framework.get_virtual_lab_session(session_id)
            
            if session_data:
                # Test JSON serialization of session data
                serializable_session = make_json_serializable(session_data)
                json.dumps(serializable_session)
                print("✓ Session data serialization works")
                return True
        
        print("✓ No sessions to test (this is normal for fresh framework)")
        return True
        
    except Exception as e:
        print(f"✗ Session data serialization failed: {e}")
        return False


def test_meeting_history_serialization():
    """Test meeting history serialization."""
    print("\nTesting meeting history serialization...")
    
    try:
        # Create framework
        framework = MultiAgentResearchFramework({
            'openai_api_key': None,
            'store_all_interactions': True
        })
        
        # Get meeting history
        meeting_history = framework.get_meeting_history()
        
        # Test JSON serialization of meeting history
        serializable_history = make_json_serializable(meeting_history)
        json.dumps(serializable_history)
        print("✓ Meeting history serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Meeting history serialization failed: {e}")
        return False


def test_framework_statistics_serialization():
    """Test framework statistics serialization."""
    print("\nTesting framework statistics serialization...")
    
    try:
        # Create framework
        framework = MultiAgentResearchFramework({
            'openai_api_key': None,
            'store_all_interactions': True
        })
        
        # Get framework statistics
        stats = framework.get_framework_statistics()
        
        # Test JSON serialization of statistics
        serializable_stats = make_json_serializable(stats)
        json.dumps(serializable_stats)
        print("✓ Framework statistics serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Framework statistics serialization failed: {e}")
        return False


def main():
    """Run all comprehensive tests."""
    print("Running comprehensive JSON serialization tests...\n")
    
    tests = [
        test_virtual_lab_serialization,
        test_traditional_research_serialization,
        test_agent_marketplace_serialization,
        test_session_data_serialization,
        test_meeting_history_serialization,
        test_framework_statistics_serialization
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All comprehensive JSON serialization tests passed!")
        print("✅ No more JSON serialization errors should occur!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 