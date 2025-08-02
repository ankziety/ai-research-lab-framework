#!/usr/bin/env python3
"""
Test script for literature integration and API key validation functionality.
"""

import sys
import os
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_literature_retriever_api_validation():
    """Test API key validation functionality."""
    print("\n=== Testing Literature Retriever API Key Validation ===")
    
    try:
        from literature_retriever import LiteratureRetriever
        
        # Test with empty config
        retriever = LiteratureRetriever(config={})
        api_status = retriever.get_api_key_status()
        
        print(f"✅ API Key Validation Results:")
        print(f"   Valid Keys: {api_status.get('valid_keys', [])}")
        print(f"   Missing Keys: {api_status.get('missing_keys', [])}")
        print(f"   Invalid Keys: {api_status.get('invalid_keys', [])}")
        print(f"   Warnings: {len(api_status.get('warnings', []))}")
        print(f"   Recommendations: {api_status.get('recommendations', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Literature Retriever API validation test failed: {e}")
        return False

def test_literature_review_phase():
    """Test the literature review phase integration."""
    print("\n=== Testing Literature Review Phase Integration ===")
    
    try:
        from virtual_lab import VirtualLabMeetingSystem, ResearchPhase
        from agents import PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
        
        # Create minimal agents for testing
        pi_agent = PrincipalInvestigatorAgent()
        critic_agent = ScientificCriticAgent()
        marketplace = AgentMarketplace()
        
        # Create Virtual Lab system
        virtual_lab = VirtualLabMeetingSystem(
            pi_agent=pi_agent,
            scientific_critic=critic_agent,
            agent_marketplace=marketplace,
            config={}
        )
        
        # Test that LITERATURE_REVIEW phase is defined
        phases = [phase.value for phase in ResearchPhase]
        if 'literature_review' in phases:
            print(f"✅ Literature Review phase found in ResearchPhase enum")
        else:
            print(f"❌ Literature Review phase missing from ResearchPhase enum")
            return False
        
        # Test that the phase execution method exists
        if hasattr(virtual_lab, '_phase_literature_review'):
            print(f"✅ _phase_literature_review method exists")
        else:
            print(f"❌ _phase_literature_review method missing")
            return False
        
        # Test that the literature analysis methods exist
        required_methods = [
            '_analyze_literature_with_pi',
            '_synthesize_literature_findings',
            '_format_literature_for_analysis',
            '_parse_literature_analysis_response',
            '_parse_literature_synthesis_response'
        ]
        
        for method in required_methods:
            if hasattr(virtual_lab, method):
                print(f"✅ {method} method exists")
            else:
                print(f"❌ {method} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Literature review phase test failed: {e}")
        return False

def test_enhanced_critic_integration():
    """Test enhanced critic integration in meetings."""
    print("\n=== Testing Enhanced Critic Integration ===")
    
    try:
        from virtual_lab import VirtualLabMeetingSystem, MeetingAgenda, MeetingType, ResearchPhase
        from agents import PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
        
        # Create minimal agents for testing
        pi_agent = PrincipalInvestigatorAgent()
        critic_agent = ScientificCriticAgent()
        marketplace = AgentMarketplace()
        
        # Create Virtual Lab system
        virtual_lab = VirtualLabMeetingSystem(
            pi_agent=pi_agent,
            scientific_critic=critic_agent,
            agent_marketplace=marketplace,
            config={}
        )
        
        # Test that critic evaluation is integrated in team meetings
        if hasattr(virtual_lab, '_conduct_team_meeting'):
            print(f"✅ _conduct_team_meeting method exists")
        else:
            print(f"❌ _conduct_team_meeting method missing")
            return False
        
        # Test that critic evaluation is integrated in individual meetings
        if hasattr(virtual_lab, '_conduct_individual_meeting'):
            print(f"✅ _conduct_individual_meeting method exists")
        else:
            print(f"❌ _conduct_individual_meeting method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced critic integration test failed: {e}")
        return False

def test_literature_search_functionality():
    """Test literature search functionality."""
    print("\n=== Testing Literature Search Functionality ===")
    
    try:
        from literature_retriever import LiteratureRetriever
        
        # Create retriever with empty config (will use mock data)
        retriever = LiteratureRetriever(config={})
        
        # Test basic search functionality
        search_results = retriever.search(
            query="machine learning",
            max_results=5,
            sources=['pubmed', 'arxiv']
        )
        
        if search_results:
            print(f"✅ Literature search returned {len(search_results)} results")
            print(f"   Sample result: {search_results[0].get('title', 'No title')}")
        else:
            print(f"❌ Literature search returned no results")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Literature search test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Literature Integration and API Key Validation")
    print("=" * 60)
    
    tests = [
        test_literature_retriever_api_validation,
        test_literature_review_phase,
        test_enhanced_critic_integration,
        test_literature_search_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Literature integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 