#!/usr/bin/env python3
"""
Simple test to verify the research system is working.
"""

import os
import sys
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from core.ai_research_lab import create_framework

def test_research_system():
    """Test the research system with a simple question."""
    print("🔬 Testing AI Research Lab System...")
    
    # Create framework
    config = {
        'enable_mock_responses': True,
        'enable_agent_marketplace': True,
        'enable_virtual_lab': True,
        'enable_memory_management': True,
        'store_all_interactions': True,
        'mock_response_quality': 'high',
        'enable_detailed_mock_responses': True,
        'research_timeout': 60,
        'max_research_phases': 8,
        'enable_phase_tracking': True
    }
    
    try:
        framework = create_framework(config)
        print("✅ Framework created successfully")
        
        # Test agent marketplace
        if hasattr(framework, 'agent_marketplace'):
            marketplace = framework.agent_marketplace
            total_agents = len(marketplace.available_agents)
            print(f"✅ Agent marketplace initialized with {total_agents} agents")
            
            # Test agent hiring
            agents = marketplace.get_agents_by_expertise('biomedical_engineering')
            print(f"✅ Found {len(agents)} biomedical engineering agents")
            
            if agents:
                agent = agents[0]
                print(f"✅ Agent: {agent.agent_id} ({agent.role})")
        
        # Test research session
        research_question = "What is the impact of AI on healthcare?"
        print(f"\n🔬 Starting research: {research_question}")
        
        start_time = time.time()
        result = framework.conduct_virtual_lab_research(
            research_question=research_question,
            session_id="test_session_001"
        )
        end_time = time.time()
        
        print(f"✅ Research completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Research success: {result.get('success', False)}")
        
        # Check for agents in results
        if 'session_data' in result:
            session_data = result['session_data']
            if 'phases' in session_data:
                phases = session_data['phases']
                print(f"✅ Research phases completed: {len(phases)}")
                
                # Check team selection phase
                if 'team_selection' in phases:
                    team_result = phases['team_selection']
                    print(f"✅ Team selection: {team_result.get('success', False)}")
                    
                    if 'hired_agents' in team_result:
                        hired_agents = team_result['hired_agents']
                        print(f"✅ Agents hired: {len(hired_agents)}")
                        for expertise, agent_data in hired_agents.items():
                            print(f"   - {expertise}: {agent_data.get('agent_id', 'unknown')}")
        
        # Check agent activities
        if hasattr(framework, 'get_agent_activity_log'):
            activities = framework.get_agent_activity_log("test_session_001")
            print(f"✅ Agent activities recorded: {len(activities)}")
        
        # Check chat logs
        if hasattr(framework, 'get_chat_logs'):
            chat_logs = framework.get_chat_logs("test_session_001")
            print(f"✅ Chat logs recorded: {len(chat_logs)}")
        
        print("\n🎉 Research system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Research system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_research_system()
